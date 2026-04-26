#!/usr/bin/env python3
"""Celery soak test — 1000-doc concurrent upload.

Plan B Phase 6.1. Validates that switching RAG_SYNC_INGEST=0 (Celery
async path) handles bursty uploads without losing documents.

Generates synthetic 5KB markdown docs with filename-encoded dates so
shard_key derivation works as it would on real production data.

Usage:
    # Standard 1000-doc soak
    python scripts/celery_soak_test.py \\
        --target-kb 1 --target-subtag 1 \\
        --doc-count 1000 --concurrency 8 \\
        --api-base http://localhost:6100

    # Verify after run completes
    python scripts/celery_soak_test.py \\
        --verify --target-kb 1 \\
        --expected-doc-count 1000
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import os
import random
import string
import sys
import time
from typing import Any

import httpx


def _gen_doc(idx: int) -> tuple[str, str]:
    """Return (filename, body) for synthetic doc."""
    # Date spans the last 24 months for realistic shard distribution
    today = _dt.date.today()
    days_ago = random.randint(0, 24 * 30)
    doc_date = today - _dt.timedelta(days=days_ago)
    fn = doc_date.strftime("%d %b %Y") + f"-soak-{idx:04d}.md"
    body = f"""# Soak document {idx}

Date: {doc_date.isoformat()}
Topic: synthetic-{idx % 20}

This is a synthetic document used by the Celery soak test. It contains
roughly 1000 words of generated text so the chunker has real chunks to
emit.

""" + "\n\n".join(
        " ".join(random.choices(string.ascii_lowercase + " ", k=80))
        for _ in range(20)
    )
    return fn, body


async def upload_one(
    client: httpx.AsyncClient,
    api_base: str, token: str, kb_id: int, subtag_id: int,
    filename: str, body: str,
) -> tuple[bool, float]:
    start = time.monotonic()
    files = {"file": (filename, body, "text/markdown")}
    try:
        r = await client.post(
            f"{api_base}/api/kb/{kb_id}/subtag/{subtag_id}/upload",
            headers={"Authorization": f"Bearer {token}"},
            files=files, timeout=30.0,
        )
        return (r.status_code in (200, 202, 409), time.monotonic() - start)
    except Exception as e:
        print(f"upload err {filename}: {e}", file=sys.stderr)
        return (False, time.monotonic() - start)


async def soak(args) -> int:
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(idx: int):
        async with sem:
            fn, body = _gen_doc(idx)
            return await upload_one(
                client, args.api_base, args.token,
                args.target_kb, args.target_subtag, fn, body,
            )

    async with httpx.AsyncClient() as client:
        start = time.monotonic()
        tasks = [worker(i) for i in range(args.doc_count)]
        results = []
        for completed in asyncio.as_completed(tasks):
            results.append(await completed)
            if len(results) % 50 == 0:
                print(f"progress: {len(results)}/{args.doc_count}",
                      file=sys.stderr)
        elapsed = time.monotonic() - start

    success = sum(1 for ok, _ in results if ok)
    failures = len(results) - success
    durations = sorted(d for _, d in results)
    p50 = durations[len(durations) // 2]
    p95 = durations[int(0.95 * len(durations))]
    p99 = durations[int(0.99 * len(durations))]
    print(f"\nSoak complete in {elapsed:.1f}s")
    print(f"  uploads: {len(results)} success={success} failures={failures}")
    print(f"  per-upload latency: p50={p50:.2f}s p95={p95:.2f}s p99={p99:.2f}s")
    print(f"  rate: {len(results)/elapsed:.1f} uploads/s")
    return 0 if failures == 0 else 1


async def verify(args) -> int:
    """Count chunks belonging to the 'soak-' filename prefix in Qdrant."""
    from qdrant_client import AsyncQdrantClient
    qc = AsyncQdrantClient(url=args.qdrant_url, timeout=60.0)
    # Count distinct doc_ids whose filename starts with the soak pattern
    seen_doc_ids: set = set()
    offset = None
    while True:
        points, offset = await qc.scroll(
            collection_name=f"kb_{args.target_kb}",
            limit=512, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            if "-soak-" in (payload.get("filename") or ""):
                seen_doc_ids.add(payload.get("doc_id"))
        if offset is None:
            break
    print(f"Found {len(seen_doc_ids)} unique soak doc_ids in Qdrant")
    if args.expected_doc_count and len(seen_doc_ids) < args.expected_doc_count:
        print(f"  MISSING: {args.expected_doc_count - len(seen_doc_ids)}",
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-kb", type=int, required=True)
    p.add_argument("--target-subtag", type=int, default=1)
    p.add_argument("--doc-count", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--api-base", default="http://localhost:6100")
    p.add_argument("--token", default=os.environ.get("RAG_ADMIN_TOKEN", ""))
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--expected-doc-count", type=int, default=0)
    args = p.parse_args()

    if not args.token and not args.verify:
        print("ERROR: --token or RAG_ADMIN_TOKEN required for upload",
              file=sys.stderr)
        return 2

    if args.verify:
        return asyncio.run(verify(args))
    return asyncio.run(soak(args))


if __name__ == "__main__":
    raise SystemExit(main())
