#!/usr/bin/env python3
"""Retrofit ``is_tenant=True`` payload indexes onto existing Qdrant collections.

Qdrant 1.11+ understands ``KeywordIndexParams(type='keyword', is_tenant=True)``
as a signal that a payload field's unique values partition the collection into
independent tenants. The engine uses this to skip cross-tenant sub-graphs
during filtered vector search (filtered-HNSW optimization). The win scales
with collection count × tenant count, and costs nothing to apply — it's a
pure hint.

This script walks every collection and (re)creates payload indexes for:

* tenant fields (``is_tenant=True``): ``kb_id``, ``chat_id``, ``owner_user_id``
* filter fields (plain KEYWORD):       ``subtag_id``, ``doc_id``, ``deleted``

Re-registering an index that already exists is a no-op at the Qdrant side —
it raises, we catch and continue. The script is therefore idempotent and
safe to re-run.

It does NOT re-index data, rename, or delete anything. Purely additive.

Usage:
    # Dry-run (default): print the plan, make no writes.
    python scripts/apply_tenant_indexes.py --qdrant-url http://localhost:6333

    # Apply.
    python scripts/apply_tenant_indexes.py --qdrant-url http://localhost:6333 --apply

    # Skip specific collections (exact name, may be passed multiple times).
    python scripts/apply_tenant_indexes.py --exclude kb_eval --apply

Exit codes:
    0  success (or dry-run finished cleanly)
    1  Qdrant error
    2  qdrant-client not installed
    4  invalid arguments
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Make ``ext.services.*`` importable regardless of how the script is invoked
# (future-proof — no current use, but mirrors the sibling scripts).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Tenant vs filter partitioning. Must stay in sync with
# ``ext.services.vector_store.ensure_collection``.
_TENANT_FIELDS = ("kb_id", "chat_id", "owner_user_id")
_FILTER_FIELDS = ("subtag_id", "doc_id", "deleted")

# Collections we never touch — these are Open WebUI's own data, not RAG.
_DEFAULT_EXCLUDES = ("open-webui_files",)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant REST URL (default: http://localhost:6333).",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Collection name to skip (exact match). May be passed multiple "
            f"times. {', '.join(_DEFAULT_EXCLUDES)!r} is always skipped."
        ),
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for Qdrant calls (default: 60s).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the plan, make no writes (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Actually create the payload indexes. Opt-in.",
    )
    return p.parse_args(argv)


def _should_skip(name: str, exclude: tuple[str, ...]) -> bool:
    if name in _DEFAULT_EXCLUDES:
        return True
    return name in exclude


async def _apply(args: argparse.Namespace) -> int:
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http import models as qm
    except ImportError as e:
        print(f"error: qdrant-client not installed: {e}", file=sys.stderr)
        return 2

    apply = bool(args.apply)
    banner = "APPLY" if apply else "DRY-RUN"
    exclude = tuple(args.exclude or ())
    print(f"[{banner}] apply tenant indexes @ {args.qdrant_url}")
    if exclude:
        print(f"  excluded (in addition to defaults): {list(exclude)}")

    qdrant = AsyncQdrantClient(url=args.qdrant_url, timeout=args.timeout)
    try:
        try:
            cols = [c.name for c in (await qdrant.get_collections()).collections]
        except Exception as e:
            print(f"error: failed to list collections: {e}", file=sys.stderr)
            return 1

        targets = [c for c in cols if not _should_skip(c, exclude)]
        skipped = [c for c in cols if _should_skip(c, exclude)]
        print(f"  found {len(cols)} collections ({len(targets)} targets, {len(skipped)} skipped)")
        for name in skipped:
            print(f"    - skip {name}")
        for name in targets:
            print(f"    - target {name}")

        if not apply:
            print(
                f"  [dry-run] would create tenant-marked keyword index on "
                f"{{{', '.join(_TENANT_FIELDS)}}} and plain keyword index on "
                f"{{{', '.join(_FILTER_FIELDS)}}} per target"
            )
            print("  [dry-run] pass --apply to execute")
            return 0

        tenant_ok = 0
        tenant_noop = 0
        filter_ok = 0
        filter_noop = 0
        errors = 0

        for name in targets:
            for field in _TENANT_FIELDS:
                try:
                    await qdrant.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=qm.KeywordIndexParams(
                            type="keyword",
                            is_tenant=True,
                        ),
                    )
                    tenant_ok += 1
                    print(f"    {name}.{field} tenant-index OK")
                except Exception as e:
                    # Already exists (same params) is the common case — not an error.
                    msg = str(e).lower()
                    if "already exists" in msg or "already indexed" in msg:
                        tenant_noop += 1
                    else:
                        errors += 1
                        print(
                            f"    {name}.{field} tenant-index ERROR: {e}",
                            file=sys.stderr,
                        )
            for field in _FILTER_FIELDS:
                try:
                    await qdrant.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                    filter_ok += 1
                    print(f"    {name}.{field} filter-index OK")
                except Exception as e:
                    msg = str(e).lower()
                    if "already exists" in msg or "already indexed" in msg:
                        filter_noop += 1
                    else:
                        errors += 1
                        print(
                            f"    {name}.{field} filter-index ERROR: {e}",
                            file=sys.stderr,
                        )

        print(
            f"\n[done] tenant: {tenant_ok} created, {tenant_noop} noop; "
            f"filter: {filter_ok} created, {filter_noop} noop; "
            f"{errors} errors"
        )
        return 0 if errors == 0 else 1
    finally:
        await qdrant.close()


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    return asyncio.run(_apply(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
