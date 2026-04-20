"""Run the current retriever against a golden set and write a JSON results file.

Reads a JSONL golden set produced by ``generate_golden.py`` and calls the live
retriever (``ext.services.retriever.retrieve``) against the live Qdrant + TEI
stack. For each row we score top-k with ``scorer.py`` and aggregate means
globally + per-kb.

Usage:
  python -m tests.eval.run_eval \\
      --golden tests/eval/golden.jsonl \\
      --qdrant-url http://localhost:6333 \\
      --tei-url http://localhost:8080 \\
      --k 10 \\
      --out tests/eval/results-2026-04-19.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Local import — must work from repo root (python -m tests.eval.run_eval).
from tests.eval.scorer import (
    chunk_recall_at_k,
    doc_recall_at_k,
    mrr_at_k,
    unique_docs_at_k,
)


def _coerce_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


async def _run_one(row: dict, *, retrieve_fn, vector_store, embedder, k: int) -> dict:
    query = row["query"]
    kb_id = row.get("kb_id")
    t0 = time.perf_counter()
    try:
        hits = await retrieve_fn(
            query=query,
            selected_kbs=[{"kb_id": kb_id, "subtag_ids": []}] if kb_id is not None else [],
            chat_id=None,
            vector_store=vector_store,
            embedder=embedder,
            per_kb_limit=max(k, 10),
            total_limit=max(k * 3, 30),
        )
    except Exception as exc:
        return {
            "query": query,
            "kb_id": kb_id,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
    retrieved_chunk_ids = [str(h.id) for h in hits]
    retrieved_doc_ids: list[int] = []
    for h in hits:
        did = _coerce_int(h.payload.get("doc_id"))
        if did is not None:
            retrieved_doc_ids.append(did)

    gold_chunk = {str(row["gold_chunk_id"])}
    gold_doc = {row["gold_doc_id"]} if row.get("gold_doc_id") is not None else set()

    return {
        "query": query,
        "kb_id": kb_id,
        "n_hits": len(hits),
        "chunk_recall_at_k": chunk_recall_at_k(retrieved_chunk_ids, gold_chunk, k),
        "doc_recall_at_k": doc_recall_at_k(retrieved_doc_ids, gold_doc, k),
        "mrr_at_k": mrr_at_k(retrieved_doc_ids, gold_doc, k),
        "unique_docs_at_k": unique_docs_at_k(retrieved_doc_ids, k),
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }


async def run(args) -> int:
    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"golden file not found: {golden_path}", file=sys.stderr)
        return 2

    rows = [
        json.loads(ln) for ln in golden_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not rows:
        print("golden file is empty", file=sys.stderr)
        return 2

    # Lazy imports — keep the import-time failure surface small so scorer tests
    # don't need the live stack.
    from ext.services.embedder import TEIEmbedder
    from ext.services.retriever import retrieve
    from ext.services.vector_store import VectorStore

    vector_store = VectorStore(url=args.qdrant_url, vector_size=args.vector_size)
    embedder = TEIEmbedder(base_url=args.tei_url, timeout=args.timeout)

    per_row: list[dict] = []
    try:
        for i, row in enumerate(rows, 1):
            res = await _run_one(
                row,
                retrieve_fn=retrieve,
                vector_store=vector_store,
                embedder=embedder,
                k=args.k,
            )
            per_row.append(res)
            if i % 10 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] done", file=sys.stderr)
    finally:
        await vector_store.close()
        await embedder.aclose()

    clean = [r for r in per_row if "error" not in r]
    errored = [r for r in per_row if "error" in r]

    def _agg(subset: list[dict]) -> dict:
        if not subset:
            return {
                "n": 0,
                "chunk_recall_at_k": 0.0,
                "doc_recall_at_k": 0.0,
                "mrr_at_k": 0.0,
                "unique_docs_at_k": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            }
            # NB: keep the zero-path shape identical to the populated one so
            # downstream diffs don't have to guess.
        lats = sorted(r["latency_ms"] for r in subset)
        p50 = lats[len(lats) // 2]
        p95 = lats[min(len(lats) - 1, max(0, int(len(lats) * 0.95) - 1))]
        return {
            "n": len(subset),
            "chunk_recall_at_k": round(_mean([r["chunk_recall_at_k"] for r in subset]), 4),
            "doc_recall_at_k": round(_mean([r["doc_recall_at_k"] for r in subset]), 4),
            "mrr_at_k": round(_mean([r["mrr_at_k"] for r in subset]), 4),
            "unique_docs_at_k": round(_mean([r["unique_docs_at_k"] for r in subset]), 2),
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
        }

    per_kb: dict[str, dict] = {}
    by_kb: dict[Optional[int], list[dict]] = defaultdict(list)
    for r in clean:
        by_kb[r.get("kb_id")].append(r)
    for kb_id, subset in by_kb.items():
        per_kb[str(kb_id)] = _agg(subset)

    summary = {
        "date": datetime.now(timezone.utc).isoformat(),
        "k": args.k,
        "golden_file": str(golden_path),
        "n_queries": len(rows),
        "n_successful": len(clean),
        "n_errored": len(errored),
        "metrics": _agg(clean),
        "per_kb": per_kb,
        "errors": [{"query": r["query"], "error": r["error"]} for r in errored[:10]],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    m = summary["metrics"]
    print(
        f"eval k={args.k}  n={m['n']}  "
        f"chunk_recall@{args.k}={m['chunk_recall_at_k']:.3f}  "
        f"doc_recall@{args.k}={m['doc_recall_at_k']:.3f}  "
        f"mrr@{args.k}={m['mrr_at_k']:.3f}  "
        f"unique_docs@{args.k}={m['unique_docs_at_k']:.2f}  "
        f"p95_ms={m['p95_latency_ms']:.0f}",
    )
    print(f"wrote {out_path}")
    return 0 if summary["n_errored"] == 0 else 1


def _default_out() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"tests/eval/results-{today}.json"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", default="tests/eval/golden.jsonl")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--tei-url", default="http://localhost:8080",
                   help="TEI base URL (it exposes /embed directly)")
    p.add_argument("--vector-size", type=int, default=1024,
                   help="embedding dim; must match the live collections (bge-m3 = 1024)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--out", default=None)
    ns = p.parse_args(argv)
    if ns.out is None:
        ns.out = _default_out()
    return ns


if __name__ == "__main__":
    ns = _parse_args()
    raise SystemExit(asyncio.run(run(ns)))
