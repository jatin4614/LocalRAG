#!/usr/bin/env python3
"""In-place recontextualize of an existing KB collection.

For every chunk (level != "doc") in the target KB:
  1. Fetches the document's ``kb_documents.doc_summary`` (falls back to filename).
  2. Calls ``contextualizer.contextualize_batch`` with (chunk_text, doc_context)
     pairs for that document. Concurrency is bounded globally.
  3. Re-embeds the augmented texts via TEI (dense) + fastembed (sparse BM25).
  4. Upserts with the SAME UUIDv5 point IDs so existing chunks are overwritten
     in place — no collection rebuild, no alias swap.
  5. Updates ``kb_documents.pipeline_version`` to stamp ``ctx=contextual-v1``.

Summary points (``level="doc"``) are left untouched — they're already whole-
document summaries and need no situating context.

Failures are per-chunk fail-open (contextualizer's existing behaviour) or
per-doc fail-noisy (logged, next doc continues).

Usage:
  docker exec orgchat-open-webui bash -c \\
    "cd /app && PYTHONPATH=/app python scripts/recontextualize_kb.py --kb-id 1 [--limit N] [--apply]"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kb-id", type=int, required=True)
    p.add_argument("--limit", type=int, default=None, help="Process at most N docs")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max in-flight chat calls (default 8)")
    p.add_argument("--apply", action="store_true",
                   help="Actually write. Dry-run otherwise.")
    p.add_argument("--doc-id", type=int, default=None,
                   help="Only process this one doc_id (useful for smoke tests).")
    return p.parse_args(argv)


async def _load_docs(session, kb_id: int, limit: Optional[int], one_doc_id: Optional[int]) -> list[tuple[int, str, Optional[str]]]:
    from sqlalchemy import text as _sql
    if one_doc_id is not None:
        rows = (await session.execute(
            _sql("SELECT id, filename, doc_summary FROM kb_documents "
                 "WHERE kb_id = :k AND id = :did AND deleted_at IS NULL"),
            {"k": kb_id, "did": one_doc_id},
        )).all()
    else:
        sql = ("SELECT id, filename, doc_summary FROM kb_documents "
               "WHERE kb_id = :k AND deleted_at IS NULL ORDER BY id")
        if limit:
            sql += " LIMIT :lim"
            rows = (await session.execute(_sql(sql), {"k": kb_id, "lim": limit})).all()
        else:
            rows = (await session.execute(_sql(sql), {"k": kb_id})).all()
    return [(int(r[0]), str(r[1] or ""), r[2]) for r in rows]


async def _scroll_chunks(client, collection: str, doc_id: int) -> list[dict]:
    """Fetch all non-summary points for a doc_id, preserving chunk_index order."""
    from qdrant_client.http import models as qm
    flt = qm.Filter(
        must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))],
        must_not=[
            qm.FieldCondition(key="level", match=qm.MatchValue(value="doc")),
            qm.FieldCondition(key="deleted", match=qm.MatchValue(value=True)),
        ],
    )
    collected: list[dict] = []
    next_page = None
    while True:
        res, next_page = await client.scroll(
            collection_name=collection, scroll_filter=flt,
            with_payload=True, with_vectors=False,
            limit=512, offset=next_page,
        )
        for p in res:
            collected.append({"id": p.id, "payload": p.payload})
        if next_page is None:
            break
    # Sort by chunk_index for deterministic order; missing chunk_index sorts last.
    collected.sort(key=lambda x: (x["payload"].get("chunk_index") is None,
                                  x["payload"].get("chunk_index") or 0))
    return collected


async def _recontextualize_doc(
    *, doc_id: int, filename: str, doc_summary: Optional[str],
    qdrant_client, collection: str, embedder, chat_url: str, chat_model: str,
    api_key: Optional[str], concurrency: int, apply: bool,
) -> tuple[int, int]:
    """Returns (n_chunks_processed, n_chunks_updated)."""
    from ext.services.contextualizer import contextualize_batch
    from ext.services.sparse_embedder import embed_sparse
    from qdrant_client.http import models as qm

    points = await _scroll_chunks(qdrant_client, collection, doc_id)
    if not points:
        return 0, 0

    # Build the situating anchor. Prefer the existing doc_summary (richer
    # context); fall back to filename if summary is missing/empty.
    anchor = (doc_summary or "").strip() or filename

    chunk_texts = [p["payload"].get("text", "") for p in points]
    if not any(chunk_texts):
        return len(points), 0

    pairs = [(t, anchor) for t in chunk_texts]
    augmented = await contextualize_batch(
        pairs, chat_url=chat_url, chat_model=chat_model,
        api_key=api_key, concurrency=concurrency,
    )

    # Count how many chunks were actually augmented (contextualizer's
    # fail-open returns the raw chunk unchanged).
    changed = sum(1 for orig, aug in zip(chunk_texts, augmented) if aug is not orig and aug != orig)
    if not changed:
        return len(points), 0

    if not apply:
        return len(points), changed

    # Re-embed dense via TEI (batched internally to RAG_TEI_MAX_BATCH=32).
    new_vecs = await embedder.embed(augmented)

    # Re-compute BM25 sparse vectors locally.
    try:
        new_sparse = list(embed_sparse(augmented))
    except Exception:
        new_sparse = [None] * len(augmented)  # fall open — dense-only upsert is still valid on hybrid collection if paired properly

    # Build point structs in the hybrid-collection shape.
    pts = []
    for i, p in enumerate(points):
        payload = dict(p["payload"])  # copy
        payload["text"] = augmented[i]
        payload["ctx_version"] = "contextual-v1"
        vec_map: dict = {"dense": new_vecs[i]}
        if new_sparse[i] is not None:
            indices, values = new_sparse[i]
            vec_map["bm25"] = qm.SparseVector(indices=list(indices), values=list(values))
        pts.append(qm.PointStruct(id=p["id"], vector=vec_map, payload=payload))

    await qdrant_client.upsert(collection_name=collection, points=pts, wait=True)
    return len(points), changed


async def _amain(args: argparse.Namespace) -> int:
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from qdrant_client import AsyncQdrantClient
    from ext.services.embedder import TEIEmbedder

    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if not db_url:
        print("DATABASE_URL unset — abort", file=sys.stderr)
        return 1

    engine = create_async_engine(db_url, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    qc = AsyncQdrantClient(url=qdrant_url)

    tei_url = os.environ.get("TEI_URL") or os.environ.get(
        "RAG_EMBEDDING_OPENAI_API_BASE_URL", "http://tei:80"
    )
    if "/v1" in tei_url:
        tei_url = tei_url.split("/v1")[0]
    embedder = TEIEmbedder(base_url=tei_url)

    chat_url = os.environ.get("OPENAI_API_BASE_URL", "http://vllm-chat:8000/v1")
    chat_model = os.environ.get("CHAT_MODEL", "orgchat-chat")
    api_key = os.environ.get("OPENAI_API_KEY")

    collection = f"kb_{args.kb_id}"
    print(f"Recontextualize kb_id={args.kb_id} collection={collection} apply={args.apply} concurrency={args.concurrency}")

    async with Session() as s:
        docs = await _load_docs(s, args.kb_id, args.limit, args.doc_id)
    print(f"  {len(docs)} document(s) to process")
    if not docs:
        return 0

    stopping = {"sig": False}

    def _handler(sig, frame):
        print(f"\n  signal {sig} received — finishing current doc then exiting")
        stopping["sig"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    t0 = time.perf_counter()
    total_chunks = 0
    total_changed = 0
    total_docs_ok = 0
    total_docs_err = 0
    stamp_ids: list[int] = []
    for idx, (doc_id, filename, doc_summary) in enumerate(docs):
        if stopping["sig"]:
            break
        try:
            n, c = await _recontextualize_doc(
                doc_id=doc_id, filename=filename, doc_summary=doc_summary,
                qdrant_client=qc, collection=collection,
                embedder=embedder, chat_url=chat_url, chat_model=chat_model,
                api_key=api_key, concurrency=args.concurrency,
                apply=args.apply,
            )
            total_chunks += n
            total_changed += c
            total_docs_ok += 1
            if c > 0 and args.apply:
                stamp_ids.append(doc_id)
        except Exception as e:
            total_docs_err += 1
            print(f"  [doc_id={doc_id}] ERROR: {e}", file=sys.stderr)
            continue
        if (idx + 1) % 10 == 0 or idx + 1 == len(docs):
            elapsed = time.perf_counter() - t0
            print(f"  [{idx+1}/{len(docs)}] chunks={total_chunks} changed={total_changed} "
                  f"ok={total_docs_ok} err={total_docs_err} elapsed={elapsed:.1f}s")

    # Stamp pipeline_version on kb_documents for docs we actually updated.
    if args.apply and stamp_ids:
        from sqlalchemy import text as _sql
        async with Session() as s:
            await s.execute(
                _sql("UPDATE kb_documents SET pipeline_version = "
                     "'chunker=v2|extractor=v2|embedder=bge-m3|ctx=contextual-v1' "
                     "WHERE id = ANY(:ids)"),
                {"ids": stamp_ids},
            )
            await s.commit()
        print(f"  stamped pipeline_version on {len(stamp_ids)} doc(s)")

    elapsed = time.perf_counter() - t0
    print(f"Done. docs_ok={total_docs_ok} docs_err={total_docs_err} "
          f"chunks={total_chunks} changed={total_changed} elapsed={elapsed:.1f}s")
    await qc.close()
    await embedder.aclose()
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
