#!/usr/bin/env python3
"""GC blobs + Qdrant points + DB rows for soft-deleted ``kb_documents``.

Rows are eligible once ``deleted_at < now() - retention_days`` (default
30 days, override via ``RAG_BLOB_RETENTION_DAYS`` or ``--retention-days``).

Usage::

    # Preview (default — no writes).
    python scripts/blob_gc.py \\
        --database-url "$DATABASE_URL" \\
        --blob-root /var/ingest \\
        --qdrant-url http://localhost:6333 \\
        --retention-days 30

    # Apply.
    python scripts/blob_gc.py --apply ...

The script is idempotent — re-running is safe and reports a no-op once the
eligible rows have been cleared. Failure to reach Qdrant does not block
blob / row cleanup: a stale point in Qdrant is vastly cheaper than an
unbounded blob directory.

Exit codes:
    0  success (or dry-run finished cleanly)
    1  database / store error
    2  a required dependency is not installed
    4  invalid arguments
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy async DB URL (required; defaults to $DATABASE_URL).",
    )
    p.add_argument(
        "--blob-root",
        default=os.environ.get("INGEST_BLOB_ROOT", "/var/ingest"),
        help="BlobStore root (default: $INGEST_BLOB_ROOT or /var/ingest).",
    )
    p.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant REST URL (default: $QDRANT_URL or http://localhost:6333).",
    )
    p.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Override retention in days (default: $RAG_BLOB_RETENTION_DAYS or 30).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max rows per pass (default: 1000). Re-run to clear backlog.",
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
        help="Actually delete blobs, Qdrant points, and DB rows.",
    )
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    if not args.database_url:
        print("error: --database-url or $DATABASE_URL is required", file=sys.stderr)
        return 4

    try:
        from ext.db.session import make_engine, make_sessionmaker  # noqa: WPS433
        from ext.services.blob_gc import retention_days as env_retention, run_gc
        from ext.services.blob_store import BlobStore
    except ImportError as e:
        print(f"error: missing dependency: {e}", file=sys.stderr)
        return 2

    try:
        from ext.services.vector_store import VectorStore
    except ImportError as e:
        print(f"error: missing dependency (vector_store): {e}", file=sys.stderr)
        return 2

    retention = args.retention_days if args.retention_days is not None else env_retention()
    apply = bool(args.apply)
    dry_run = not apply
    banner = "DRY-RUN" if dry_run else "APPLY"

    print(
        f"[{banner}] blob GC: retention={retention}d limit={args.limit} "
        f"blob_root={args.blob_root} qdrant={args.qdrant_url}"
    )

    engine = make_engine(args.database_url)
    sm = make_sessionmaker(engine)
    blob_store = BlobStore(args.blob_root)
    vector_size = int(os.environ.get("RAG_VECTOR_SIZE", "1024"))
    vs = VectorStore(url=args.qdrant_url, vector_size=vector_size)

    rc = 0
    try:
        async with sm() as session:
            summary = await run_gc(
                session=session,
                blob_store=blob_store,
                vector_store=vs,
                retention_days=retention,
                dry_run=dry_run,
                limit=args.limit,
            )
        print(json.dumps(summary, indent=2))
    except Exception as e:  # noqa: BLE001
        print(f"error: GC pass failed: {e}", file=sys.stderr)
        rc = 1
    finally:
        try:
            await vs.close()
        except Exception:
            pass
        await engine.dispose()
    return rc


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
