#!/usr/bin/env python3
"""Backfill per-document summaries for the Tier 1 summary index.

For every document in the given KB that doesn't already have a
``kb_documents.doc_summary`` row, this script:

  1. Scrolls Qdrant for the doc's chunk points (level != "doc") to get
     the raw chunk text — avoids re-extracting the source bytes.
  2. Calls ``ext.services.doc_summarizer.summarize_document`` via the
     chat model (bounded concurrency, default 4 in-flight).
  3. Embeds the summary through the TEI client.
  4. Upserts a single summary point into the same collection with
     ``level="doc"``, ``kind="doc_summary"``, ``chunk_index=-1``.
  5. UPDATEs ``kb_documents.doc_summary`` so the text is queryable from
     Postgres.

Idempotent: docs with a non-NULL ``doc_summary`` are skipped. Safe to
re-run after partial progress or an interrupted session.

Usage:
  # Dry-run (default) — prints the list of docs that would be
  # summarized without touching Qdrant or Postgres.
  docker exec orgchat-open-webui python scripts/backfill_doc_summaries.py \\
      --kb-id 1

  # Apply.
  docker exec orgchat-open-webui python scripts/backfill_doc_summaries.py \\
      --kb-id 1 --apply

  # Just the first 20 docs (useful for smoke-testing).
  docker exec orgchat-open-webui python scripts/backfill_doc_summaries.py \\
      --kb-id 1 --limit 20 --apply

Exit codes:
    0  success (or dry-run finished cleanly)
    1  any error — DB unavailable, Qdrant error, etc.
    4  invalid arguments
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import uuid
from pathlib import Path
from typing import Optional

# Make ``ext.*`` importable when invoked from the repo root or from inside
# the container (``/app`` has ext/ on sys.path already but belt-and-brace).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# UUIDv5 namespace must match ingest.py._POINT_NS so the id collides with
# any future ingest-time summary upsert (idempotent).
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill per-document summaries into the doc-summary index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--kb-id", type=int, required=True,
                   help="KB id to backfill (collection kb_<id>).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N documents (useful for smoke tests).")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max summary calls in flight (default 4).")
    p.add_argument("--apply", action="store_true",
                   help="Actually write. Without this flag the script is a dry run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Explicit dry-run (default behaviour).")
    return p.parse_args(argv)


async def _load_docs_to_process(session, kb_id: int, limit: Optional[int]) -> list[tuple[int, str]]:
    """Return [(doc_id, filename), ...] for docs in kb_id that lack a summary."""
    from sqlalchemy import text as _sql_text
    sql = (
        "SELECT id, filename FROM kb_documents "
        "WHERE kb_id = :kb AND deleted_at IS NULL AND doc_summary IS NULL "
        "ORDER BY uploaded_at DESC, id"
    )
    if limit:
        sql += " LIMIT :lim"
        rows = (await session.execute(_sql_text(sql), {"kb": kb_id, "lim": limit})).all()
    else:
        rows = (await session.execute(_sql_text(sql), {"kb": kb_id})).all()
    return [(int(r[0]), str(r[1] or "")) for r in rows]


async def _fetch_chunk_texts(vs, collection: str, doc_id: int) -> list[str]:
    """Scroll Qdrant for the chunk texts of ``doc_id``, sorted by chunk_index.

    Filters out any existing ``level=="doc"`` point (so a re-run won't
    feed the summary back into itself).
    """
    from qdrant_client.http import models as qm
    client = vs._client
    flt = qm.Filter(
        must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))],
        must_not=[qm.FieldCondition(key="deleted", match=qm.MatchValue(value=True))],
    )
    collected: list[tuple[int, str]] = []
    next_page = None
    while True:
        res, next_page = await client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=next_page,
        )
        for r in res:
            pl = r.payload or {}
            # Skip any existing doc-summary point.
            if (pl.get("level") or "chunk") == "doc":
                continue
            idx = pl.get("chunk_index")
            txt = pl.get("text") or ""
            if not txt:
                continue
            try:
                idx_i = int(idx) if idx is not None else 10**9
            except (TypeError, ValueError):
                idx_i = 10**9
            collected.append((idx_i, txt))
        if next_page is None:
            break
    collected.sort(key=lambda t: t[0])
    return [t for _, t in collected]


async def _summarize_one(
    *,
    sem: asyncio.Semaphore,
    doc_id: int,
    filename: str,
    vs,
    embedder,
    sessionmaker,
    kb_id: int,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str],
    apply: bool,
) -> tuple[str, int]:
    """Summarize one doc. Returns (status, summary_len).

    status ∈ {"ok", "empty", "error"}.
    """
    from ext.services.doc_summarizer import summarize_document
    from sqlalchemy import text as _sql_text

    collection = f"kb_{kb_id}"
    async with sem:
        try:
            chunks = await _fetch_chunk_texts(vs, collection, doc_id)
        except Exception as e:
            print(f"  [doc_id={doc_id}] scroll failed: {e}", file=sys.stderr)
            return "error", 0
        if not chunks:
            return "empty", 0

        summary = await summarize_document(
            chunks=chunks,
            filename=filename,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            timeout=float(os.environ.get("RAG_DOC_SUMMARY_TIMEOUT", "30.0")),
        )
        if not summary:
            return "empty", 0
        if not apply:
            return "ok", len(summary)

        # Fetch the payload template (owner_user_id + subtag_id) from one
        # chunk so the summary point carries identical tenant tags.
        try:
            payload_base = await _payload_base_from_chunk(vs, collection, doc_id, kb_id)
        except Exception as e:
            print(f"  [doc_id={doc_id}] payload-base fetch failed: {e}", file=sys.stderr)
            return "error", 0

        try:
            [vec] = await embedder.embed([summary])
        except Exception as e:
            print(f"  [doc_id={doc_id}] embed failed: {e}", file=sys.stderr)
            return "error", 0

        # Compute BM25 sparse vector too so the upsert writes to the hybrid
        # collection's named-vector form. Without a sparse companion,
        # VectorStore.upsert falls through to the legacy single-unnamed-vector
        # path which Qdrant rejects on hybrid-shaped collections
        # ("Not existing vector name").
        sparse_vec: Optional[tuple[list[int], list[float]]] = None
        try:
            from ext.services.sparse_embedder import embed_sparse
            [sparse_vec] = list(embed_sparse([summary]))
        except Exception as e:
            print(f"  [doc_id={doc_id}] sparse-embed skipped: {e}", file=sys.stderr)

        import time as _time
        summary_payload = dict(payload_base)
        summary_payload.update({
            "chunk_index": -1,
            "text": summary,
            "filename": filename,
            "uploaded_at": _time.time_ns(),
            "deleted": False,
            "level": "doc",
            "kind": "doc_summary",
            "page": None,
            "heading_path": [],
            "sheet": None,
            "block_type": "prose",
        })
        # Stamp the temporal shard_key so upsert into a custom-sharded
        # collection works (the chunk-tier upsert at ingest time already
        # stamps this; backfill must mirror that behavior).
        if os.environ.get("RAG_SHARDING_ENABLED", "0") == "1":
            try:
                from ext.services.temporal_shard import extract_shard_key as _extract_sk
                body_sample = chunks[0] if chunks else ""
                _sk, _sk_origin = _extract_sk(filename=filename, body=body_sample)
                summary_payload["shard_key"] = _sk
                summary_payload["shard_key_origin"] = _sk_origin.value
            except Exception as _e:
                print(f"  [doc_id={doc_id}] shard_key derivation failed: {_e}", file=sys.stderr)
                return "error", 0
        point_id = str(uuid.uuid5(_POINT_NS, f"doc:{doc_id}:doc_summary"))
        point: dict = {"id": point_id, "vector": vec, "payload": summary_payload}
        if sparse_vec is not None:
            point["sparse_vector"] = sparse_vec
        try:
            await vs.upsert(collection, [point])
        except Exception as e:
            print(f"  [doc_id={doc_id}] upsert failed: {e}", file=sys.stderr)
            return "error", 0

        try:
            async with sessionmaker() as s:
                await s.execute(
                    _sql_text(
                        "UPDATE kb_documents SET doc_summary = :s WHERE id = :d"
                    ),
                    {"s": summary, "d": doc_id},
                )
                await s.commit()
        except Exception as e:
            print(f"  [doc_id={doc_id}] kb_documents update failed: {e}",
                  file=sys.stderr)
            return "error", 0

        return "ok", len(summary)


async def _payload_base_from_chunk(vs, collection: str, doc_id: int, kb_id: int) -> dict:
    """Read one chunk point's payload to inherit tenant tags.

    Returns a dict with the minimum tenancy fields. Falls back to
    ``{kb_id, doc_id}`` when a probe can't find any chunk (shouldn't
    happen — we only got here because _fetch_chunk_texts returned text).
    """
    from qdrant_client.http import models as qm
    client = vs._client
    flt = qm.Filter(
        must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))],
    )
    res, _ = await client.scroll(
        collection_name=collection,
        scroll_filter=flt,
        with_payload=True,
        with_vectors=False,
        limit=1,
    )
    base: dict = {"kb_id": kb_id, "doc_id": doc_id}
    if res:
        pl = res[0].payload or {}
        for k in ("kb_id", "subtag_id", "doc_id", "owner_user_id"):
            if k in pl and pl[k] is not None:
                base[k] = pl[k]
        # Force doc_id to match argument (guards against corrupt points).
        base["doc_id"] = doc_id
        base["kb_id"] = kb_id
    return base


async def _amain(args: argparse.Namespace) -> int:
    apply = bool(args.apply) and not args.dry_run

    # Set up DB + Qdrant + TEI clients the same way the app does, but
    # without pulling in the full FastAPI boot.
    from ext.services.vector_store import VectorStore
    from ext.services.embedder import TEIEmbedder
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL unset — can't backfill", file=sys.stderr)
        return 1
    # Convert plain postgres:// to asyncpg driver URL if needed.
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    vector_size = int(os.environ.get("RAG_VECTOR_SIZE", "1024"))
    vs = VectorStore(url=qdrant_url, vector_size=vector_size)

    tei_url = os.environ.get("TEI_URL") or os.environ.get(
        "RAG_EMBEDDING_OPENAI_API_BASE_URL", "http://tei:80"
    )
    embedder = TEIEmbedder(base_url=tei_url)

    chat_url = os.environ.get("OPENAI_API_BASE_URL", "http://vllm-chat:8000/v1")
    chat_model = os.environ.get("SUMMARY_MODEL",
                                os.environ.get("CHAT_MODEL", "orgchat-chat"))
    api_key = os.environ.get("OPENAI_API_KEY")

    async with Session() as s:
        docs = await _load_docs_to_process(s, args.kb_id, args.limit)

    total = len(docs)
    print(f"kb_id={args.kb_id}: {total} document(s) to summarize "
          f"(apply={apply}, concurrency={args.concurrency})")
    if not docs:
        return 0

    if not apply:
        for doc_id, filename in docs[:20]:
            print(f"  DRY doc_id={doc_id} filename={filename}")
        if total > 20:
            print(f"  ... ({total - 20} more)")
        print("Dry run — no writes. Re-run with --apply to proceed.")
        return 0

    # Bounded concurrency. Use a semaphore rather than a hard-sized gather
    # so an interruption between batches flushes cleanly.
    sem = asyncio.Semaphore(max(1, args.concurrency))

    # Graceful CTRL+C — flip a flag, let in-flight tasks finish, then
    # report. Avoids partial upserts from being silently lost.
    stop_flag = {"stop": False}

    def _handle_sigint(signum, frame):  # pragma: no cover — interactive only
        if stop_flag["stop"]:
            print("\n(forcing exit)", file=sys.stderr)
            sys.exit(130)
        stop_flag["stop"] = True
        print("\n(CTRL+C received — finishing in-flight tasks, press again to force)",
              file=sys.stderr)

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    ok = empty = err = 0
    tasks: list[asyncio.Task] = []

    async def _wrap(doc_id: int, filename: str) -> tuple[int, str, str, int]:
        status, slen = await _summarize_one(
            sem=sem,
            doc_id=doc_id,
            filename=filename,
            vs=vs,
            embedder=embedder,
            sessionmaker=Session,
            kb_id=args.kb_id,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            apply=apply,
        )
        return doc_id, filename, status, slen

    for i, (doc_id, filename) in enumerate(docs, start=1):
        if stop_flag["stop"]:
            break
        tasks.append(asyncio.create_task(_wrap(doc_id, filename)))
        # Flush every 10 for progress visibility.
        if i % 10 == 0 or i == total:
            done_chunk = await asyncio.gather(*tasks, return_exceptions=True)
            tasks = []
            for r in done_chunk:
                if isinstance(r, Exception):
                    err += 1
                    print(f"  ERR unexpected: {r}", file=sys.stderr)
                    continue
                d_id, fn, status, slen = r
                if status == "ok":
                    ok += 1
                elif status == "empty":
                    empty += 1
                else:
                    err += 1
            print(f"[{min(i, total)}/{total}] running_tally: "
                  f"ok={ok} empty={empty} err={err}")

    # Drain anything still in flight (shouldn't happen with the %10 flush,
    # but be safe).
    if tasks:
        remaining = await asyncio.gather(*tasks, return_exceptions=True)
        for r in remaining:
            if isinstance(r, Exception):
                err += 1
            else:
                _, _, status, _ = r
                if status == "ok":
                    ok += 1
                elif status == "empty":
                    empty += 1
                else:
                    err += 1

    print(f"Done. summarized={ok} empty={empty} errors={err} total={total}")
    return 0 if err == 0 else 1


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 4
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
