#!/usr/bin/env python3
"""Chunk-size histogram for a Qdrant KB collection (Phase 0 diagnostic).

Scrolls every point in the target collection, tokenizes each ``payload.text``
with cl100k_base (matches the chunker at HEAD), and reports the distribution
of token counts per chunk. Purpose: confirm or refute the fragmentation
hypothesis in ``docs/rag-phase0-1a-4-execution-plan.md`` — specifically
whether kb_1's average-23.5-chunks-per-doc shape is driven by many sub-100
token chunks (which would support the coalescence fix in Phase 1a).

Read-only — touches only Qdrant HTTP API. No writes.

Usage::

    python tests/eval/chunk_size_histogram.py \\
        --qdrant-url http://localhost:6333 \\
        --collection kb_1 \\
        --out tests/eval/results/chunk-size-histogram-pre-1a.json

The output JSON is intentionally human-readable; Phase 1a will re-run this
against the re-ingested collection and diff the two.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import tiktoken

# Bucket edges (upper-exclusive, upper-inclusive for last): [0-100), [100-300),
# [300-500), [500-800), [800+]. Matches the CHUNK_SIZE=800 target the current
# chunker aims at and the 100/300 "fragment" thresholds the plan cares about.
BUCKET_EDGES = [0, 100, 300, 500, 800]
BUCKET_LABELS = ["0-100", "100-300", "300-500", "500-800", ">800"]


def bucket_of(n_tokens: int) -> str:
    for i, upper in enumerate(BUCKET_EDGES[1:], start=1):
        if n_tokens < upper:
            return BUCKET_LABELS[i - 1]
    return BUCKET_LABELS[-1]


async def scroll_all(client: httpx.AsyncClient, collection: str, page_size: int = 512) -> list[dict]:
    """Scroll every point in ``collection`` and return a list of payloads.

    Uses the raw HTTP API (not qdrant-client) so a missing python SDK can't
    block the diagnostic. Returns the full list — expected ~2590 points for
    kb_1, which fits in memory without issue.
    """
    points: list[dict] = []
    offset = None
    while True:
        body: dict = {
            "limit": page_size,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        r = await client.post(f"/collections/{collection}/points/scroll", json=body)
        r.raise_for_status()
        data = r.json()["result"]
        batch = data.get("points") or []
        points.extend(batch)
        offset = data.get("next_page_offset")
        if not offset or not batch:
            break
    return points


def summarize(token_counts: list[int], per_doc: dict[int, list[int]]) -> dict:
    if not token_counts:
        return {
            "total_chunks": 0,
            "buckets": {lab: 0 for lab in BUCKET_LABELS},
            "percentiles": {},
            "mean_tokens": 0.0,
            "median_tokens": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
            "empty_chunks": 0,
            "per_doc": {},
        }
    buckets = Counter(bucket_of(n) for n in token_counts)
    # Ensure all buckets present even if zero.
    buckets_out = {lab: int(buckets.get(lab, 0)) for lab in BUCKET_LABELS}
    sorted_counts = sorted(token_counts)
    n = len(sorted_counts)

    def pct(p: float) -> int:
        if n == 0:
            return 0
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return int(sorted_counts[idx])

    per_doc_sizes = {
        did: len(counts)
        for did, counts in per_doc.items()
    }
    if per_doc_sizes:
        per_doc_size_vals = sorted(per_doc_sizes.values())
        per_doc_summary = {
            "n_docs": len(per_doc_sizes),
            "chunks_per_doc_min": per_doc_size_vals[0],
            "chunks_per_doc_max": per_doc_size_vals[-1],
            "chunks_per_doc_mean": round(statistics.mean(per_doc_size_vals), 2),
            "chunks_per_doc_median": int(statistics.median(per_doc_size_vals)),
        }
    else:
        per_doc_summary = {
            "n_docs": 0,
            "chunks_per_doc_min": 0,
            "chunks_per_doc_max": 0,
            "chunks_per_doc_mean": 0.0,
            "chunks_per_doc_median": 0,
        }

    return {
        "total_chunks": int(n),
        "empty_chunks": int(sum(1 for c in token_counts if c == 0)),
        "buckets": buckets_out,
        "bucket_pct": {
            lab: round(buckets_out[lab] / n * 100, 2) for lab in BUCKET_LABELS
        },
        "mean_tokens": round(statistics.mean(token_counts), 2),
        "median_tokens": int(statistics.median(token_counts)),
        "min_tokens": int(min(token_counts)),
        "max_tokens": int(max(token_counts)),
        "percentiles": {
            "p10": pct(0.10),
            "p25": pct(0.25),
            "p50": pct(0.50),
            "p75": pct(0.75),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
        },
        "per_doc": per_doc_summary,
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", default="kb_1")
    p.add_argument(
        "--out",
        default="tests/eval/results/chunk-size-histogram-pre-1a.json",
    )
    p.add_argument("--page-size", type=int, default=512)
    return p.parse_args(argv)


async def main(argv=None) -> int:
    ns = parse_args(argv)
    enc = tiktoken.get_encoding("cl100k_base")
    async with httpx.AsyncClient(base_url=ns.qdrant_url, timeout=60.0) as client:
        print(f"scrolling {ns.collection} at {ns.qdrant_url} ...", file=sys.stderr)
        points = await scroll_all(client, ns.collection, page_size=ns.page_size)
    print(f"  got {len(points)} points", file=sys.stderr)

    token_counts: list[int] = []
    per_doc: dict[int, list[int]] = defaultdict(list)
    pipeline_versions: Counter = Counter()
    block_types: Counter = Counter()

    for p in points:
        payload = p.get("payload") or {}
        text = payload.get("text") or ""
        n_tok = len(enc.encode(text)) if text else 0
        token_counts.append(n_tok)
        did = payload.get("doc_id")
        try:
            did_int = int(did)
        except (TypeError, ValueError):
            did_int = -1
        per_doc[did_int].append(n_tok)
        pv = payload.get("model_version") or payload.get("pipeline_version") or "unknown"
        pipeline_versions[pv] += 1
        bt = payload.get("block_type") or "absent"
        block_types[bt] += 1

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "qdrant_url": ns.qdrant_url,
        "collection": ns.collection,
        "tokenizer": "cl100k_base",
        "histogram": summarize(token_counts, per_doc),
        "pipeline_version_distribution": dict(pipeline_versions),
        "block_type_distribution": dict(block_types),
    }

    out_path = Path(ns.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    h = result["histogram"]
    print(f"\nchunk-size histogram for {ns.collection}", file=sys.stderr)
    print(f"  total_chunks   = {h['total_chunks']}", file=sys.stderr)
    print(f"  n_docs         = {h['per_doc']['n_docs']}", file=sys.stderr)
    print(f"  mean_tokens    = {h['mean_tokens']}", file=sys.stderr)
    print(f"  median_tokens  = {h['median_tokens']}", file=sys.stderr)
    print(f"  p25/p50/p75/p95 = {h['percentiles']['p25']}/{h['percentiles']['p50']}/{h['percentiles']['p75']}/{h['percentiles']['p95']}", file=sys.stderr)
    for lab in BUCKET_LABELS:
        print(
            f"    {lab:<8} : {h['buckets'][lab]:>6}  ({h['bucket_pct'][lab]}%)",
            file=sys.stderr,
        )
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
