#!/usr/bin/env python3
"""Run eval matrix against kb_eval (which uses string kb_id="eval", not numeric).

Bypasses retriever's numeric kb_{id} mapping and targets kb_eval directly,
exercising the same VectorStore.search / hybrid_search path that production
retrieval uses. Flags (RAG_HYBRID, RAG_RERANK, RAG_MMR, RAG_CONTEXT_EXPAND)
are read at call time so one process can toggle them per row.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ext.services.vector_store import VectorStore, Hit  # noqa: E402
from ext.services.embedder import TEIEmbedder  # noqa: E402
from ext.services.reranker import rerank, rerank_with_flag  # noqa: E402
from tests.eval.scorer import (  # noqa: E402
    chunk_recall_at_k,
    doc_recall_at_k,
    mrr_at_k,
    unique_docs_at_k,
)

COLLECTION = "kb_eval"


async def _retrieve_kb_eval(
    *, query: str, vs: VectorStore, emb: TEIEmbedder, limit: int = 30,
) -> list[Hit]:
    [qvec] = await emb.embed([query])
    use_hybrid = False
    if os.environ.get("RAG_HYBRID", "0") == "1":
        try:
            use_hybrid = await vs._refresh_sparse_cache(COLLECTION)
        except Exception:
            use_hybrid = False
    if use_hybrid:
        return await vs.hybrid_search(COLLECTION, qvec, query, limit=limit)
    return await vs.search(COLLECTION, qvec, limit=limit)


async def _maybe_mmr(query: str, hits: list[Hit], emb: TEIEmbedder, top_k: int) -> list[Hit]:
    if os.environ.get("RAG_MMR", "0") != "1" or not hits:
        return hits
    try:
        from ext.services.mmr import mmr_rerank_from_hits
        lam = float(os.environ.get("RAG_MMR_LAMBDA", "0.7"))
        return await mmr_rerank_from_hits(query, hits, emb, top_k=top_k, lambda_=lam)
    except Exception:
        return hits


async def _maybe_expand(hits: list[Hit], vs: VectorStore) -> list[Hit]:
    if os.environ.get("RAG_CONTEXT_EXPAND", "0") != "1" or not hits:
        return hits
    try:
        from ext.services.context_expand import expand_context
        window = int(os.environ.get("RAG_CONTEXT_EXPAND_WINDOW", "1"))
        return await expand_context(hits, collection=COLLECTION, vs=vs, window=window)
    except Exception:
        return hits


async def _run_one(row: dict, *, vs: VectorStore, emb: TEIEmbedder, k: int) -> dict:
    query = row["query"]
    gold_chunk_id = str(row.get("gold_chunk_id", ""))
    gold_doc_id = row.get("gold_doc_id")
    t0 = time.perf_counter()
    try:
        hits = await _retrieve_kb_eval(query=query, vs=vs, emb=emb, limit=max(k * 3, 30))
        # Rerank stage (flag-gated — falls back to legacy max-normalize).
        # P2 — widen the candidate pool when MMR is on so MMR actually has
        # surplus to diversify over (otherwise rerank(top=k) -> MMR(top=k)
        # is a pass-through).
        rerank_top_k_env = os.environ.get("RAG_RERANK_TOP_K")
        mmr_on = os.environ.get("RAG_MMR", "0") == "1"
        if rerank_top_k_env is not None:
            rerank_k = max(int(rerank_top_k_env), k)
        elif mmr_on:
            rerank_k = max(k * 2, 20)
        else:
            rerank_k = k
        hits = rerank_with_flag(query, hits, top_k=min(len(hits), rerank_k), fallback_fn=rerank)
        # MMR stage — always trims down to k.
        hits = await _maybe_mmr(query, hits, emb, top_k=min(len(hits), k))
        # Context expansion stage (after rerank/MMR, before budget; budget is a no-op for eval metrics)
        hits = await _maybe_expand(hits, vs)
    except Exception as exc:
        return {
            "query": query,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
    latency_ms = (time.perf_counter() - t0) * 1000

    retrieved_chunk_ids = [str(h.id) for h in hits[:k]]
    retrieved_doc_ids: list[int] = []
    for h in hits[:k]:
        v = (getattr(h, "payload", None) or {}).get("doc_id")
        if isinstance(v, int):
            retrieved_doc_ids.append(v)
        else:
            try:
                retrieved_doc_ids.append(int(v))
            except (TypeError, ValueError):
                pass

    return {
        "query": query,
        "gold_doc_id": gold_doc_id,
        "gold_chunk_id": gold_chunk_id,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "chunk_recall": chunk_recall_at_k(retrieved_chunk_ids, {gold_chunk_id}, k),
        "doc_recall": doc_recall_at_k(retrieved_doc_ids, {int(gold_doc_id)} if gold_doc_id is not None else set(), k),
        "mrr": mrr_at_k(retrieved_doc_ids, {int(gold_doc_id)} if gold_doc_id is not None else set(), k),
        "unique_docs": unique_docs_at_k(retrieved_doc_ids, k),
        "latency_ms": latency_ms,
    }


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="tests/eval/golden.jsonl")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--tei-url", default="http://172.19.0.6:80")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--label", default="baseline", help="name for this run (recorded in output)")
    p.add_argument("--out", default="")
    ns = p.parse_args()

    rows = [json.loads(l) for l in Path(ns.golden).read_text().splitlines() if l.strip()]
    vs = VectorStore(url=ns.qdrant_url, vector_size=1024)
    emb = TEIEmbedder(base_url=ns.tei_url)

    try:
        results = []
        for row in rows:
            results.append(await _run_one(row, vs=vs, emb=emb, k=ns.k))

        metrics = {
            "n_queries": len(results),
            "n_errors": sum(1 for r in results if "error" in r),
            "chunk_recall_at_k": statistics.mean([r["chunk_recall"] for r in results if "error" not in r]) if results else 0.0,
            "doc_recall_at_k": statistics.mean([r["doc_recall"] for r in results if "error" not in r]) if results else 0.0,
            "mrr_at_k": statistics.mean([r["mrr"] for r in results if "error" not in r]) if results else 0.0,
            "unique_docs_at_k": statistics.mean([r["unique_docs"] for r in results if "error" not in r]) if results else 0.0,
            "p50_latency_ms": statistics.median([r["latency_ms"] for r in results if "error" not in r]) if results else 0.0,
            "p95_latency_ms": sorted([r["latency_ms"] for r in results if "error" not in r])[int(len(results) * 0.95)] if results else 0.0,
        }
        flags = {k: os.environ.get(k, "") for k in
                 ("RAG_HYBRID", "RAG_RERANK", "RAG_MMR", "RAG_CONTEXT_EXPAND", "RAG_BUDGET_TOKENIZER", "RAG_MMR_LAMBDA", "RAG_CONTEXT_EXPAND_WINDOW")}
        out = {"label": ns.label, "k": ns.k, "metrics": metrics, "flags": flags}

        if ns.out:
            Path(ns.out).write_text(json.dumps(out, indent=2))
        print(f"[{ns.label}] chunk_recall@{ns.k}={metrics['chunk_recall_at_k']:.3f} "
              f"doc_recall@{ns.k}={metrics['doc_recall_at_k']:.3f} "
              f"MRR@{ns.k}={metrics['mrr_at_k']:.3f} "
              f"unique_docs@{ns.k}={metrics['unique_docs_at_k']:.2f} "
              f"p50={metrics['p50_latency_ms']:.0f}ms p95={metrics['p95_latency_ms']:.0f}ms "
              f"errors={metrics['n_errors']}/{metrics['n_queries']}")
    finally:
        await vs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
