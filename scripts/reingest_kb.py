#!/usr/bin/env python3
"""Re-ingest all documents from a source Qdrant collection into a target
collection, using the current pipeline (which may have contextualize /
colbert enabled per per-KB rag_config).

Reads docs by scrolling the source collection, grouping chunks back into
per-document sets, then resubmitting to the ingest pipeline for the KB.

Usage:
    python scripts/reingest_kb.py \\
        --source-collection kb_1_rebuild \\
        --target-collection kb_1_v3 \\
        --kb-id 1 \\
        --api-base-url http://localhost:6100 \\
        --admin-token $RAG_ADMIN_TOKEN \\
        --throttle-ceiling-ms 3000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import httpx
from qdrant_client import AsyncQdrantClient


log = logging.getLogger("reingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def collect_docs_from_source(client, collection):
    """Scroll source, group chunks by doc_id. Returns {doc_id: doc_record}."""
    docs: dict[int, dict] = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = dict(p.payload or {})
            did = payload.get("doc_id")
            if did is None:
                continue
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "filename": payload.get("filename", f"doc_{did}.txt"),
                    "subtag_id": payload.get("subtag_id"),
                    "chunks": [],
                }
            docs[did]["chunks"].append(payload)
        if offset is None:
            break
    # Sort each doc's chunks by chunk_index to reconstruct order
    for d in docs.values():
        d["chunks"].sort(key=lambda c: c.get("chunk_index", 0))
    return docs


async def chat_p95_ms(client, prom_url) -> float:
    q = 'histogram_quantile(0.95, rate(llm_latency_seconds_bucket{stage="chat"}[5m])) * 1000'
    r = await client.get(f"{prom_url}/api/v1/query", params={"query": q})
    r.raise_for_status()
    data = r.json()
    if data["data"]["result"]:
        return float(data["data"]["result"][0]["value"][1])
    return 0.0


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-collection", required=True)
    p.add_argument("--target-collection", required=True)
    p.add_argument("--kb-id", type=int, required=True)
    p.add_argument("--api-base-url", default="http://localhost:6100")
    p.add_argument("--admin-token", default=os.environ.get("RAG_ADMIN_TOKEN", ""))
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--prom-url", default="http://localhost:9090")
    p.add_argument("--throttle-ceiling-ms", type=float, default=3000.0)
    args = p.parse_args()

    if not args.admin_token:
        print("ERROR: --admin-token or RAG_ADMIN_TOKEN required")
        return 2

    qc = AsyncQdrantClient(url=args.qdrant_url, timeout=60.0)
    log.info("scrolling source %s", args.source_collection)
    docs = await collect_docs_from_source(qc, args.source_collection)
    log.info("found %d documents across %d chunks",
             len(docs),
             sum(len(d["chunks"]) for d in docs.values()))

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {args.admin_token}"}, timeout=300.0,
    ) as c:
        count = 0
        for did, doc in docs.items():
            # Throttle
            while True:
                p95 = await chat_p95_ms(c, args.prom_url)
                if p95 <= args.throttle_ceiling_ms:
                    break
                log.warning("chat p95 %.0fms > ceiling %.0fms; sleeping 30s",
                            p95, args.throttle_ceiling_ms)
                await asyncio.sleep(30)

            # Reconstruct document text from chunks (removing prefixes if present)
            body = "\n\n".join(
                ch.get("text", "").split("\n\n", 1)[-1]  # strip old context_prefix if present
                if ch.get("context_prefix")
                else ch.get("text", "")
                for ch in doc["chunks"]
            )
            subtag_id = doc.get("subtag_id")
            if subtag_id is None:
                log.warning("doc_id=%s has no subtag_id, skipping", did)
                continue

            # POST to upload endpoint — the ingest pipeline will read
            # per-KB rag_config and apply contextualize + colbert as configured.
            files = {"file": (doc["filename"], body, "text/plain")}
            data = {"doc_id_hint": str(did)}
            r = await c.post(
                f"{args.api_base_url}/api/kb/{args.kb_id}/subtag/{subtag_id}/upload",
                files=files, data=data,
            )
            if r.status_code == 409:
                log.info("doc_id=%s already ingested (idempotent)", did)
            else:
                r.raise_for_status()
            count += 1
            if count % 50 == 0:
                log.info("re-ingested %d/%d docs", count, len(docs))

    log.info("re-ingest complete: %d docs", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
