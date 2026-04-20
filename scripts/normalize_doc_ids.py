#!/usr/bin/env python3
"""Normalize `doc_id` payload field in every kb_* Qdrant collection to int.

Walks each collection via scroll, detects string doc_ids, rewrites them
using Qdrant's payload update API (no re-embed needed). Idempotent.

Why: historically ``ingest.py`` stamped ``doc_id`` as ``str(doc.id)`` (see the
P0 plan circa 2026-04-18), so older collections like ``kb_4`` / ``kb_3`` carry
strings while earlier ones (``kb_1``) carry ints. Doc-level eval metrics
(doc_recall@k, mrr@k) silently mis-score unless every caller coerces. This
script fixes the data once; ``ingest.py`` defensively coerces going forward.

Usage:
    python scripts/normalize_doc_ids.py --qdrant-url http://localhost:6333 --dry-run
    python scripts/normalize_doc_ids.py --qdrant-url http://localhost:6333 --apply

Exit codes:
    0  every collection scanned cleanly (no skipped rows)
    1  HTTP / transport failure
    2  at least one non-numeric doc_id encountered (caller decides)
    4  invalid arguments
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable

import httpx


def _list_collections(client: httpx.Client, prefix: str) -> list[str]:
    r = client.get("/collections")
    r.raise_for_status()
    names = [c["name"] for c in r.json()["result"]["collections"]]
    return sorted(n for n in names if n.startswith(prefix))


def _scroll(
    client: httpx.Client, collection: str, batch_size: int
) -> Iterable[list[dict]]:
    """Yield batches of points (id + payload) from ``collection``."""
    offset = None
    while True:
        body: dict = {
            "limit": batch_size,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        r = client.post(f"/collections/{collection}/points/scroll", json=body)
        r.raise_for_status()
        result = r.json()["result"]
        points = result.get("points", [])
        if not points:
            return
        yield points
        offset = result.get("next_page_offset")
        if offset is None:
            return


def _plan_batch(points: list[dict]) -> tuple[int, int, int, int, dict[int, list]]:
    """Classify points by doc_id status.

    Returns (clean, convertible, skipped_nonnumeric, no_id, plan) where
    ``plan`` maps new int doc_id -> list of point ids to update.
    """
    clean = 0
    convertible = 0
    skipped = 0
    no_id = 0
    plan: dict[int, list] = {}
    for p in points:
        payload = p.get("payload") or {}
        if "doc_id" not in payload:
            no_id += 1
            continue
        v = payload["doc_id"]
        if v is None:
            no_id += 1
            continue
        if isinstance(v, int):
            clean += 1
            continue
        if isinstance(v, str):
            try:
                as_int = int(v)
            except ValueError:
                skipped += 1
                print(
                    f"    skip point id={p['id']!r} doc_id={v!r} (non-numeric)",
                    file=sys.stderr,
                )
                continue
            plan.setdefault(as_int, []).append(p["id"])
            convertible += 1
        else:
            # unexpected type (float, list, dict) — surface, don't touch
            skipped += 1
            print(
                f"    skip point id={p['id']!r} doc_id={v!r} (unexpected type {type(v).__name__})",
                file=sys.stderr,
            )
    return clean, convertible, skipped, no_id, plan


def _apply_plan(
    client: httpx.Client,
    collection: str,
    plan: dict[int, list],
    batch_size: int,
) -> None:
    """POST payload updates to Qdrant, grouped by new doc_id, batched by size."""
    for new_id, point_ids in plan.items():
        # Batch large update lists to avoid oversized request bodies.
        for i in range(0, len(point_ids), batch_size):
            chunk = point_ids[i : i + batch_size]
            body = {"payload": {"doc_id": new_id}, "points": chunk}
            r = client.post(
                f"/collections/{collection}/points/payload",
                params={"wait": "true"},
                json=body,
            )
            r.raise_for_status()


def _process_collection(
    client: httpx.Client, collection: str, *, batch_size: int, apply: bool
) -> tuple[int, int, int, int, int]:
    """Walk one collection. Returns (scanned, clean, converted, skipped, no_id)."""
    scanned = 0
    clean_total = 0
    converted_total = 0
    skipped_total = 0
    no_id_total = 0
    for batch in _scroll(client, collection, batch_size):
        scanned += len(batch)
        clean, convertible, skipped, no_id, plan = _plan_batch(batch)
        clean_total += clean
        converted_total += convertible
        skipped_total += skipped
        no_id_total += no_id
        if apply and plan:
            _apply_plan(client, collection, plan, batch_size)
        elif plan:
            # dry-run: print summary so caller sees the plan
            total = sum(len(v) for v in plan.values())
            print(
                f"    [dry-run] would update {total} point(s) across "
                f"{len(plan)} doc_id value(s)",
            )
    return scanned, clean_total, converted_total, skipped_total, no_id_total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument(
        "--collection-prefix",
        default="kb_",
        help="Only process collections whose name starts with this prefix "
        "(default 'kb_', so chat_* and open-webui_files are skipped).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Scroll + payload-update batch size (default 256).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the plan; make no writes (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Actually POST payload updates. Opt-in flip; overrides --dry-run.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default 30).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    apply = bool(args.apply)  # default stays dry-run unless --apply is set
    mode_banner = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode_banner}] normalize doc_ids at {args.qdrant_url}")

    try:
        with httpx.Client(base_url=args.qdrant_url, timeout=args.timeout) as client:
            try:
                collections = _list_collections(client, args.collection_prefix)
            except httpx.HTTPError as e:
                print(f"failed to list collections: {e}", file=sys.stderr)
                return 1
            if not collections:
                print(
                    f"no collections match prefix {args.collection_prefix!r}",
                    file=sys.stderr,
                )
                return 0

            any_skipped = False
            for c in collections:
                print(f"  {c}")
                try:
                    scanned, clean, converted, skipped, no_id = _process_collection(
                        client,
                        c,
                        batch_size=args.batch_size,
                        apply=apply,
                    )
                except httpx.HTTPError as e:
                    print(f"    HTTP error: {e}", file=sys.stderr)
                    return 1
                print(
                    f"    {c}: scanned={scanned}, clean={clean}, "
                    f"converted={converted}, skipped={skipped}, no_id={no_id}"
                )
                if skipped:
                    any_skipped = True
    except httpx.HTTPError as e:
        print(f"transport error: {e}", file=sys.stderr)
        return 1

    return 2 if any_skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
