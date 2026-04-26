"""Build the temporal-semantic RAPTOR tree on top of an existing temporally
sharded collection (e.g. ``kb_1_v4``).

Wraps ``ext.services.temporal_raptor.build_temporal_tree`` with concrete
``summarize`` (vllm-chat) and ``embed`` (TEI) backends, then upserts the
returned L1-L4 nodes into the same collection with the appropriate
``shard_key_selector`` for each node.

Plan B Phase 5 operator follow-up. Mirrors the pattern of
``scripts/reingest_kb_dual.py`` (Plan A 3.7) for client wiring + idempotent
upserts.

Run:
  python scripts/build_temporal_tree.py --collection kb_1_v4
  python scripts/build_temporal_tree.py --collection kb_1_v4 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from typing import Iterable

# Make ``ext`` importable when run as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import httpx  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.http.models import (  # noqa: E402
    Filter, FieldCondition, MatchValue, MatchAny, PointStruct,
)

from ext.services.embedder import TEIEmbedder  # noqa: E402
from ext.services.temporal_raptor import build_temporal_tree  # noqa: E402

log = logging.getLogger("build_temporal_tree")

# Deterministic UUID5 namespace for tree nodes — same collection + same
# (level, shard_key, time_range) always produces the same point id, so
# re-running the script overwrites L1-L4 nodes in place rather than
# duplicating them. Mirrors the Plan A 3.7 idempotency contract.
_NS = uuid.UUID("9e8ad4c7-0a18-4f45-9e39-7b54fe6a6d71")


def _node_id(collection: str, payload: dict) -> str:
    tr = payload.get("time_range", {})
    key = (
        f"{collection}|L{payload.get('level')}|"
        f"sk={payload.get('shard_key')}|"
        f"start={tr.get('start')}|end={tr.get('end')}"
    )
    return str(uuid.uuid5(_NS, key))


async def _scroll_l0_chunks(
    qc: AsyncQdrantClient, collection: str, *, batch: int = 256,
) -> list[dict]:
    """Read every L0 leaf (no ``level`` payload) from the collection.

    L1-L4 nodes carry ``level`` in their payload; L0 chunks do not. We
    filter them out so the tree builder only sees leaves on a re-run.
    """
    chunks: list[dict] = []
    offset = None
    while True:
        records, offset = await qc.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            payload = r.payload or {}
            if "level" in payload and payload["level"] != 0:
                continue
            sk = payload.get("shard_key")
            text = payload.get("text") or payload.get("content")
            if not sk or not text:
                continue
            chunks.append({
                "text": text,
                "shard_key": sk,
                "chunk_index": payload.get("chunk_index", 0),
            })
        if offset is None:
            break
    return chunks


def _make_summarize_callable(
    chat_url: str, chat_model: str, *, timeout_s: float = 120.0,
):
    """Return an async ``summarize(prompt) -> str`` backed by vllm-chat."""
    client = httpx.AsyncClient(base_url=chat_url, timeout=timeout_s)

    async def _summarize(prompt: str) -> str:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 512,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    _summarize.aclose = client.aclose  # type: ignore[attr-defined]
    return _summarize


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    qc = AsyncQdrantClient(url=args.qdrant_url, timeout=120.0)
    if not await qc.collection_exists(collection_name=args.collection):
        log.error("collection %r does not exist", args.collection)
        await qc.close()
        return 1

    tei_url = (
        os.environ.get("TEI_URL")
        or os.environ.get("RAG_EMBEDDING_OPENAI_API_BASE_URL", "")
        or args.tei_url
    )
    if "/v1" in tei_url:
        tei_url = tei_url.split("/v1")[0]
    embedder = TEIEmbedder(base_url=tei_url)
    summarize = _make_summarize_callable(args.chat_url, args.chat_model)

    log.info("scrolling L0 chunks from %r", args.collection)
    chunks = await _scroll_l0_chunks(qc, args.collection)
    if not chunks:
        log.warning("no L0 chunks found — nothing to build")
        await qc.close(); await embedder.aclose(); await summarize.aclose()
        return 0
    log.info("found %d L0 chunks across %d shard_keys",
             len(chunks), len({c["shard_key"] for c in chunks}))

    log.info("building temporal tree (concurrency=%d)", args.concurrency)
    nodes = await build_temporal_tree(
        chunks=chunks,
        summarize=summarize,
        embed=embedder.embed,
        chat_model=args.chat_model,
        concurrency=args.concurrency,
    )
    log.info("built %d tree nodes", len(nodes))
    if args.dry_run:
        from collections import Counter
        levels = Counter(n["payload"]["level"] for n in nodes)
        for lvl in sorted(levels):
            log.info("  L%d: %d nodes", lvl, levels[lvl])
        await qc.close(); await embedder.aclose(); await summarize.aclose()
        return 0

    by_shard: dict[str, list[PointStruct]] = {}
    for n in nodes:
        sk = n["payload"]["shard_key"]
        by_shard.setdefault(sk, []).append(PointStruct(
            id=_node_id(args.collection, n["payload"]),
            vector={"dense": n["embedding"]},
            payload={**n["payload"], "text": n["text"]},
        ))

    for sk, points in sorted(by_shard.items()):
        log.info("upserting %d nodes to shard_key=%s", len(points), sk)
        await qc.upsert(
            collection_name=args.collection,
            points=points,
            shard_key_selector=sk,
        )

    log.info("done")
    await qc.close(); await embedder.aclose(); await summarize.aclose()
    return 0


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", required=True,
                   help="Target temporally sharded collection (e.g. kb_1_v4)")
    p.add_argument("--qdrant-url", default=os.environ.get(
        "QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--tei-url", default="http://localhost:8080")
    p.add_argument("--chat-url", default=os.environ.get(
        "OPENAI_API_BASE_URL", "http://localhost:8000"),
        help="vllm-chat base URL (no /v1 suffix)")
    p.add_argument("--chat-model", default=os.environ.get(
        "RAG_SUMMARIZER_MODEL", "gemma-3-27b-it"))
    p.add_argument("--concurrency", type=int, default=4,
                   help="Per-level summarize concurrency")
    p.add_argument("--dry-run", action="store_true",
                   help="Build tree + report counts; skip upsert")
    return p.parse_args(argv)


def main() -> int:
    return asyncio.run(_amain(_parse()))


if __name__ == "__main__":
    raise SystemExit(main())
