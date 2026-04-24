"""Unified eval harness — runs golden set through retrieval, emits per-stratum metrics.

Usage:
    python -m tests.eval.harness \
        --golden tests/eval/golden_starter.jsonl \
        --kb-id 1 \
        --qdrant-url http://localhost:6333 \
        --tei-url http://localhost:80 \
        --out tests/eval/results/phase-0-baseline.json

Output: a JSON document keyed by {global, by_intent, by_year, by_difficulty,
by_language, by_intent_year, per_row}. See docs/runbook/slo.md for gating thresholds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from tests.eval.scorer import chunk_recall_at_k, mrr_at_k, ndcg_at_k
from tests.eval.stratify import stratify, intent_year_strata


def _load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _retrieve_http(
    client: httpx.AsyncClient,
    base_url: str,
    chat_id: str | None,
    query: str,
    kb_id: int,
    top_k: int = 20,
) -> list[dict]:
    """Call /api/rag/retrieve and return list of hit dicts."""
    body = {
        "chat_id": chat_id,
        "query": query,
        "selected_kb_config": [{"kb_id": kb_id}],
        "top_k": top_k,
        "max_tokens": 5000,
    }
    r = await client.post(f"{base_url}/api/rag/retrieve", json=body, timeout=30.0)
    r.raise_for_status()
    return r.json().get("hits", [])


def _doc_id_from_hit(hit: dict) -> int | None:
    return hit.get("doc_id")


def _chunk_id_from_hit(hit: dict) -> tuple[int, int] | None:
    did = hit.get("doc_id")
    cidx = hit.get("chunk_index")
    if did is None or cidx is None:
        return None
    return (did, cidx)


def _score_row(row: dict, hits: list[dict], k: int) -> dict:
    expected_docs = set(row.get("expected_doc_ids") or [])
    expected_chunks = {
        (did, cidx)
        for did in row.get("expected_doc_ids") or []
        for cidx in row.get("expected_chunk_indices") or []
    }
    retrieved_doc_ids = [_doc_id_from_hit(h) for h in hits[:k] if _doc_id_from_hit(h) is not None]
    retrieved_chunk_ids = [_chunk_id_from_hit(h) for h in hits[:k] if _chunk_id_from_hit(h) is not None]
    return {
        "chunk_recall@k": chunk_recall_at_k(retrieved_chunk_ids, expected_chunks, k),
        "mrr@k": mrr_at_k(retrieved_doc_ids, expected_docs, k),
        "ndcg@k": ndcg_at_k(retrieved_doc_ids, expected_docs, k),
    }


def _aggregate(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    def _mean(key: str) -> float:
        vals = [r[key] for r in rows if key in r]
        return float(statistics.mean(vals)) if vals else 0.0
    def _p(key: str, p: float) -> float:
        vals = sorted(r[key] for r in rows if key in r)
        if not vals:
            return 0.0
        idx = min(int(round(p / 100.0 * (len(vals) - 1))), len(vals) - 1)
        return float(vals[idx])
    return {
        "n": n,
        "chunk_recall@10": _mean("chunk_recall@k"),
        "mrr@10": _mean("mrr@k"),
        "ndcg@10": _mean("ndcg@k"),
        "p50_latency_ms": _p("latency_ms", 50),
        "p95_latency_ms": _p("latency_ms", 95),
        "p99_latency_ms": _p("latency_ms", 99),
    }


async def run_eval(
    golden_path: Path,
    kb_id: int,
    api_base_url: str,
    k: int = 10,
) -> dict[str, Any]:
    rows = _load_golden(golden_path)
    per_row: list[dict] = []
    async with httpx.AsyncClient() as client:
        for row in rows:
            t0 = time.perf_counter()
            try:
                hits = await _retrieve_http(
                    client, api_base_url, chat_id=None,
                    query=row["query"], kb_id=kb_id, top_k=max(k, 20),
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                err = None
            except Exception as exc:  # noqa: BLE001 — harness catches all, logs, continues
                hits = []
                latency_ms = (time.perf_counter() - t0) * 1000
                err = f"{type(exc).__name__}: {exc}"
            scores = _score_row(row, hits, k)
            per_row.append({
                **{
                    "query": row["query"],
                    "intent_label": row.get("intent_label"),
                    "year_bucket": row.get("year_bucket"),
                    "difficulty": row.get("difficulty"),
                    "language": row.get("language"),
                    "adversarial_category": row.get("adversarial_category"),
                    "latency_ms": latency_ms,
                    "error": err,
                    "n_hits": len(hits),
                },
                **scores,
            })

    strata = stratify(per_row)
    by_intent = {k: _aggregate(v) for k, v in strata["intent"].items()}
    by_year = {k: _aggregate(v) for k, v in strata["year"].items()}
    by_difficulty = {k: _aggregate(v) for k, v in strata["difficulty"].items()}
    by_language = {k: _aggregate(v) for k, v in strata["language"].items()}
    by_intent_year = {k: _aggregate(v) for k, v in intent_year_strata(per_row).items()}

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "golden_path": str(golden_path),
        "kb_id": kb_id,
        "k": k,
        "n_total": len(rows),
        "global": _aggregate(per_row),
        "by_intent": by_intent,
        "by_year": by_year,
        "by_difficulty": by_difficulty,
        "by_language": by_language,
        "by_intent_year": by_intent_year,
        "per_row": per_row,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", required=True, type=Path)
    p.add_argument("--kb-id", required=True, type=int)
    p.add_argument("--api-base-url", default="http://localhost:6100")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    result = asyncio.run(run_eval(args.golden, args.kb_id, args.api_base_url, args.k))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    print(f"global chunk_recall@{args.k}: {result['global']['chunk_recall@10']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
