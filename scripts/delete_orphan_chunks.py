#!/usr/bin/env python3
"""Delete orphan chunks from Qdrant KB collections.

An "orphan" is a point that survived a re-ingest but no longer has a
valid parent in Postgres. Two classes:

1. **Stale pipeline**: point's ``model_version`` payload starts with
   ``chunker=v2`` (the pre-Phase-1a chunker). After a successful
   re-ingest, every new v3 point has replaced its v2 counterpart at the
   same ``point_id`` (deterministic UUID5 keeps them aligned), so any
   remaining v2 point is for a doc whose re-ingest produced strictly
   fewer chunks — those trailing indices are orphans.

2. **High chunk_index**: point's ``chunk_index >= kb_documents.chunk_count``
   for that ``doc_id``. v3 produces fewer chunks (coalescence packs short
   blocks), so a doc that had 40 v2 chunks may end up with 15 v3 chunks.
   Chunks 15..39 now dangle and must go.

Dry-run by default. Lists candidates, groups by doc_id, prints the
delete plan. Pass ``--apply`` to actually issue the deletes.

CLI:

    python scripts/delete_orphan_chunks.py (--kb N | --all) [--apply]
                                           [--database-url URL] [--qdrant-url URL]

Uses the existing ``vector_store.delete_by_doc`` for wholesale
doc-level deletions (when every surviving point for a doc_id is
orphaned) and a payload filter on ``(doc_id, chunk_index)`` for the
common case (some chunks stay, some go).

Exit codes:
    0  success (no-op or cleanup complete)
    1  operational failure (DB / Qdrant error)
    2  missing dependency
    4  invalid arguments

The supervisor runs this after ``scripts/reingest_all.py`` completes.
See ``docs/rag-phase0-1a-4-execution-plan.md`` §4.3.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# ``scripts/delete_orphan_chunks.py`` → repo root is one level up.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--kb",
        type=int,
        help="Only clean up this KB id (collection ``kb_{id}``).",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Clean up every kb_{id} collection (not chat_private).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete orphan points. Default is dry-run.",
    )
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy async DB URL (required; defaults to $DATABASE_URL).",
    )
    p.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant REST URL (default: $QDRANT_URL or http://localhost:6333).",
    )
    p.add_argument(
        "--vector-size",
        type=int,
        default=int(os.environ.get("RAG_VECTOR_SIZE", "1024")),
        help="Dense vector dim (default: $RAG_VECTOR_SIZE or 1024).",
    )
    p.add_argument(
        "--scroll-batch",
        type=int,
        default=256,
        help="Points per scroll request when surveying a collection (default: 256).",
    )
    return p.parse_args(argv)


async def _list_kb_ids_from_db(session, *, only_kb: Optional[int]) -> list[int]:
    """Return sorted KB ids to process based on ``--kb`` / ``--all``."""
    from sqlalchemy import select

    from ext.db.models import KnowledgeBase

    q = select(KnowledgeBase.id).where(KnowledgeBase.deleted_at.is_(None))
    if only_kb is not None:
        q = q.where(KnowledgeBase.id == only_kb)
    rows = (await session.execute(q)).scalars().all()
    return sorted(int(r) for r in rows)


async def _chunk_counts_for_kb(session, *, kb_id: int) -> dict[int, int]:
    """Return ``{doc_id: chunk_count}`` for all live docs in a KB.

    Uses the canonical ``chunk_count`` column stamped at ingest time.
    Soft-deleted rows are excluded — their chunks should have been
    deleted at soft-delete time anyway.
    """
    from sqlalchemy import select

    from ext.db.models import KBDocument

    q = select(KBDocument.id, KBDocument.chunk_count).where(
        KBDocument.kb_id == kb_id,
        KBDocument.deleted_at.is_(None),
    )
    rows = (await session.execute(q)).all()
    return {int(doc_id): int(count or 0) for doc_id, count in rows}


async def _scan_collection(
    qdrant, collection: str, *, batch: int
) -> dict[int, list[tuple[int, str, object]]]:
    """Return ``{doc_id: [(chunk_index, model_version, point_id), ...]}``.

    Scrolls every point in the collection, pulling only payload fields
    we need to classify orphans. Idempotent (read-only).
    """
    from qdrant_client.http import models as qm  # noqa: F401

    buckets: dict[int, list[tuple[int, str, object]]] = {}
    offset = None
    while True:
        points, offset = await qdrant.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=[
                "doc_id", "chunk_index", "model_version", "deleted",
            ],
            with_vectors=False,
        )
        if not points:
            break
        for pt in points:
            p = pt.payload or {}
            if p.get("deleted") is True:
                continue
            doc_id_raw = p.get("doc_id")
            if doc_id_raw is None:
                # No doc_id on a KB collection is weird — could be a legacy
                # chat_* row that was cross-upserted. Leave it alone.
                continue
            try:
                doc_id = int(doc_id_raw)
            except (ValueError, TypeError):
                continue
            try:
                chunk_index = int(p.get("chunk_index", -1))
            except (ValueError, TypeError):
                chunk_index = -1
            model_version = str(p.get("model_version") or "")
            buckets.setdefault(doc_id, []).append(
                (chunk_index, model_version, pt.id)
            )
        if offset is None:
            break
    return buckets


def _classify_orphans(
    points_for_doc: list[tuple[int, str, object]],
    *,
    current_chunk_count: Optional[int],
) -> tuple[list[object], list[str]]:
    """Return (orphan_point_ids, reason_tags) for a single doc's points.

    An orphan is:
    * a v2 point (``chunker=v2`` prefix) — the re-ingest overwrites v3
      at the same ``chunk_index`` so any v2 point left over is for an
      index the v3 run didn't produce.
    * a point whose ``chunk_index >= current_chunk_count`` — above the
      new high-water mark.

    If the doc is not in Postgres at all (``current_chunk_count`` is
    None) we treat every point as orphan — the doc was soft-deleted
    but its chunks weren't cleaned up.
    """
    orphan_ids: list[object] = []
    reasons: list[str] = []
    for chunk_index, model_version, point_id in points_for_doc:
        if current_chunk_count is None:
            orphan_ids.append(point_id)
            reasons.append("doc_deleted")
            continue
        if model_version.startswith("chunker=v2"):
            orphan_ids.append(point_id)
            reasons.append("v2_pipeline")
            continue
        if chunk_index >= current_chunk_count:
            orphan_ids.append(point_id)
            reasons.append(f"index_ge_{current_chunk_count}")
            continue
    return orphan_ids, reasons


async def _run(args: argparse.Namespace) -> int:
    if not args.database_url:
        print("error: --database-url or $DATABASE_URL is required", file=sys.stderr)
        return 4

    try:
        from qdrant_client import AsyncQdrantClient

        from ext.db.session import make_engine, make_sessionmaker
    except ImportError as exc:
        print(f"error: missing dependency: {exc}", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(
        f"[{mode}] delete_orphan_chunks "
        f"scope={'all' if args.all else args.kb} "
        f"qdrant={args.qdrant_url}"
    )

    engine = make_engine(args.database_url)
    sm = make_sessionmaker(engine)
    qdrant = AsyncQdrantClient(url=args.qdrant_url, timeout=60.0)

    rc = 0
    totals = {
        "collections_scanned": 0,
        "docs_with_orphans": 0,
        "orphan_points_total": 0,
        "orphan_points_deleted": 0,
    }
    try:
        async with sm() as session:
            kb_ids = await _list_kb_ids_from_db(
                session, only_kb=(None if args.all else args.kb)
            )

            # Only survey collections that actually exist.
            existing = {
                c.name for c in (await qdrant.get_collections()).collections
            }

            for kb_id in kb_ids:
                collection = f"kb_{kb_id}"
                if collection not in existing:
                    print(
                        f"  [skip] {collection} does not exist — nothing to clean"
                    )
                    continue
                totals["collections_scanned"] += 1
                print(f"  scanning {collection}")
                pg_counts = await _chunk_counts_for_kb(session, kb_id=kb_id)
                buckets = await _scan_collection(
                    qdrant, collection, batch=args.scroll_batch
                )

                per_doc_plan: list[tuple[int, int, list[str], list[object]]] = []
                for doc_id, points in sorted(buckets.items()):
                    current_chunks = pg_counts.get(doc_id)
                    orphan_ids, reasons = _classify_orphans(
                        points, current_chunk_count=current_chunks
                    )
                    if not orphan_ids:
                        continue
                    per_doc_plan.append(
                        (doc_id, current_chunks or 0, reasons, orphan_ids)
                    )

                for doc_id, new_count, reasons, orphan_ids in per_doc_plan:
                    totals["docs_with_orphans"] += 1
                    totals["orphan_points_total"] += len(orphan_ids)
                    # Summarize reasons compactly.
                    reason_summary: dict[str, int] = {}
                    for r in reasons:
                        reason_summary[r] = reason_summary.get(r, 0) + 1
                    print(
                        f"    doc_id={doc_id} pg_chunk_count={new_count} "
                        f"orphans={len(orphan_ids)} reasons={reason_summary}"
                    )

                if not per_doc_plan:
                    print("    (no orphans)")
                    continue

                if not args.apply:
                    continue

                # Apply path — delete orphans per doc.
                from qdrant_client.http import models as qm

                for doc_id, _nc, _reasons, orphan_ids in per_doc_plan:
                    try:
                        await qdrant.delete(
                            collection_name=collection,
                            points_selector=qm.PointIdsList(points=list(orphan_ids)),
                            wait=True,
                        )
                        totals["orphan_points_deleted"] += len(orphan_ids)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"    WARN delete failed collection={collection} "
                            f"doc_id={doc_id}: {exc}",
                            file=sys.stderr,
                        )

        print("\n=== summary ===")
        for k, v in totals.items():
            print(f"  {k}: {v}")
        print(f"  mode: {mode}")
    except Exception as exc:  # noqa: BLE001
        print(f"error: run failed: {exc}", file=sys.stderr)
        rc = 1
    finally:
        try:
            await qdrant.close()
        except Exception:
            pass
        await engine.dispose()

    return rc


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
