#!/usr/bin/env python3
"""Re-ingest existing docs that have empty / short extracted text.

Plan B Phase 6.8. Use after enabling OCR on a KB to recover docs that
were originally ingested without OCR.

Detection: a doc is "needs OCR" if its average chunk text length is
below ``--short-text-threshold`` (default 100 chars).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import os
import sys

import httpx
from qdrant_client import AsyncQdrantClient


log = logging.getLogger("reingest_ocr")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


async def find_short_docs(qc, collection: str, threshold: int) -> dict:
    """Scroll collection. Return {doc_id: {filename, avg_chars}}."""
    chunks_by_doc = collections.defaultdict(list)
    filenames = {}
    offset = None
    while True:
        points, offset = await qc.scroll(
            collection_name=collection, limit=512, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            did = payload.get("doc_id")
            if did is None:
                continue
            chunks_by_doc[did].append(len(payload.get("text", "")))
            filenames[did] = payload.get("filename", f"doc_{did}")
        if offset is None:
            break

    short = {}
    for did, lens in chunks_by_doc.items():
        avg = sum(lens) / max(1, len(lens))
        if avg < threshold:
            short[did] = {"filename": filenames[did], "avg_chars": avg}
    return short


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kb-id", type=int, required=True)
    p.add_argument("--short-text-threshold", type=int, default=100)
    p.add_argument("--api-base", default="http://localhost:6100")
    p.add_argument("--admin-token",
                   default=os.environ.get("RAG_ADMIN_TOKEN", ""))
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    args = p.parse_args()

    if not args.admin_token:
        log.error("RAG_ADMIN_TOKEN required")
        return 2

    qc = AsyncQdrantClient(url=args.qdrant_url, timeout=60.0)
    short = await find_short_docs(
        qc, f"kb_{args.kb_id}", args.short_text_threshold,
    )
    log.info("found %d docs with avg chunk text < %d chars",
             len(short), args.short_text_threshold)

    if not short:
        return 0

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {args.admin_token}"},
        timeout=300.0,
    ) as client:
        for did, info in short.items():
            log.info("re-ingest doc_id=%d filename=%s avg=%d",
                     did, info["filename"], int(info["avg_chars"]))
            # Trigger re-ingest via the admin endpoint that fetches the
            # original blob and runs the full pipeline (which now includes
            # OCR fallback)
            r = await client.post(
                f"{args.api_base}/api/kb/{args.kb_id}/doc/{did}/reingest",
            )
            if r.status_code not in (200, 202, 409):
                log.warning("re-ingest failed for doc_id=%d: %d %s",
                            did, r.status_code, r.text[:200])
                continue
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
