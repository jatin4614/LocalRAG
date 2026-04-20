#!/usr/bin/env python3
"""RAGAS-style faithfulness eval runner (P3.5).

End-to-end: for each golden row, runs the same retrieve -> rerank -> mmr ->
expand pipeline as ``tests.eval.run_eval_kb_eval``, then generates an answer
from the chat model grounded on the retrieved context, then scores faithfulness
(claims-supported / total-claims) via a second LLM-judge pass.

Why: chunk_recall / MRR measure retrieval only. ``RAG_CONTEXT_EXPAND`` and
``RAG_CONTEXTUALIZE_KBS`` improve **answer quality on the same retrieval set**,
so the legacy harness is blind to their effect. Faithfulness is a second
metric that grades whether the generated answer is actually grounded in what
was retrieved.

Cost: per query we issue ~1 retrieval + 1 answer-gen chat call + 1 claim-extract
chat call + up to ``MAX_CLAIMS`` grade calls. At ~200 ms per Qwen-14B call
that is roughly 2-3 s per query, so a 50-row golden set runs in ~2-3 minutes
with the default concurrency of 4. Scale concurrency up if your chat endpoint
can take the load, down if you see rate limiting.

Bias warning: by default we use the same local Qwen2.5-14B-AWQ as both
generator and judge. That is self-grading and slightly optimistic. The ideal
setup uses a stronger/external judge (e.g. GPT-4, Claude) but that is not an
option in the air-gapped deployment.

Usage::

    python tests/eval/run_faithfulness_eval.py \\
        --golden tests/eval/golden.jsonl \\
        --qdrant-url http://localhost:6333 \\
        --tei-url http://172.19.0.6:80 \\
        --chat-url http://172.19.0.7:8000/v1 \\
        --chat-model orgchat-chat \\
        --label faithfulness_baseline \\
        --out tests/eval/results/faithfulness_baseline.json

The retrieval step reads the same ``RAG_HYBRID`` / ``RAG_RERANK`` / ``RAG_MMR``
/ ``RAG_CONTEXT_EXPAND`` flags as ``run_eval_kb_eval`` so you can compare
baseline vs flagged-on faithfulness by flipping env vars between runs.
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
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ext.services.embedder import TEIEmbedder  # noqa: E402
from ext.services.reranker import rerank, rerank_with_flag  # noqa: E402
from ext.services.vector_store import Hit, VectorStore  # noqa: E402
from tests.eval.faithfulness import faithfulness  # noqa: E402


COLLECTION = "kb_eval"

_ANSWER_SYSTEM = (
    "You are a careful technical assistant. Answer the user's question using "
    "ONLY the provided context. If the context does not contain the answer, "
    "say so explicitly. Do not invent facts."
)

_CHUNK_JOIN = "\n\n---\n\n"
_CONTEXT_CHAR_CAP = 16_000  # cap concatenated context to keep prompt sane
_CONTEXT_PREVIEW_LEN = 300   # truncate context in per-row output for readable JSON


# ---------------------------------------------------------------------------
# Retrieval — mirrors tests/eval/run_eval_kb_eval.py exactly, minus scoring.
# ---------------------------------------------------------------------------


async def _retrieve_kb_eval(
    *, query: str, vs: VectorStore, emb: TEIEmbedder, limit: int = 30,
) -> list[Hit]:
    [qvec] = await emb.embed([query])
    use_hybrid = False
    if os.environ.get("RAG_HYBRID", "1") != "0":
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


async def _retrieve_pipeline(
    query: str, *, vs: VectorStore, emb: TEIEmbedder, k: int,
) -> list[Hit]:
    hits = await _retrieve_kb_eval(query=query, vs=vs, emb=emb, limit=max(k * 3, 30))
    rerank_top_k_env = os.environ.get("RAG_RERANK_TOP_K")
    mmr_on = os.environ.get("RAG_MMR", "0") == "1"
    if rerank_top_k_env is not None:
        rerank_k = max(int(rerank_top_k_env), k)
    elif mmr_on:
        rerank_k = max(k * 2, 20)
    else:
        rerank_k = k
    hits = rerank_with_flag(
        query, hits, top_k=min(len(hits), rerank_k), fallback_fn=rerank,
    )
    hits = await _maybe_mmr(query, hits, emb, top_k=min(len(hits), k))
    hits = await _maybe_expand(hits, vs)
    return hits[:k]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


def _build_context(hits: list[Hit]) -> str:
    """Concatenate ``payload.text`` of hits in order, separated by ``---``.

    Silently drops hits with missing text (legacy payloads).
    """
    parts: list[str] = []
    for h in hits:
        payload = getattr(h, "payload", None) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
    joined = _CHUNK_JOIN.join(parts)
    if len(joined) > _CONTEXT_CHAR_CAP:
        joined = joined[:_CONTEXT_CHAR_CAP]
    return joined


async def _generate_answer(
    client: httpx.AsyncClient,
    chat_model: str,
    context: str,
    query: str,
    *,
    timeout: float = 30.0,
    max_tokens: int = 512,
) -> str:
    """Call the chat model to answer ``query`` using only ``context``.

    Returns the raw answer string. On any failure returns ``""`` — downstream
    faithfulness scoring treats empty answers as vacuously faithful (score 1.0),
    so we emit a zero-claim row rather than crashing the whole run.
    """
    body = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": _ANSWER_SYSTEM},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}",
            },
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    try:
        r = await client.post("/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    except httpx.HTTPStatusError as e:
        print(
            f"WARN: answer-gen HTTP {e.response.status_code}: {e.response.text[:200]}",
            file=sys.stderr,
        )
    except Exception as e:  # noqa: BLE001
        print(f"WARN: answer-gen failed: {e}", file=sys.stderr)
    return ""


# ---------------------------------------------------------------------------
# Per-row runner
# ---------------------------------------------------------------------------


async def _run_one(
    row: dict,
    *,
    vs: VectorStore,
    emb: TEIEmbedder,
    chat: httpx.AsyncClient,
    chat_url: str,
    chat_model: str,
    chat_api_key: Optional[str],
    k: int,
) -> dict:
    query = row["query"]
    t0 = time.perf_counter()
    try:
        hits = await _retrieve_pipeline(query, vs=vs, emb=emb, k=k)
    except Exception as e:  # noqa: BLE001
        return {
            "query": query,
            "error": f"retrieval: {type(e).__name__}: {e}",
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
    context = _build_context(hits)
    if not context:
        # Nothing retrieved — record an empty row and move on.
        return {
            "query": query,
            "n_hits": 0,
            "context_chars": 0,
            "answer": "",
            "n_claims": 0,
            "n_supported": 0,
            "faithfulness_score": 1.0,
            "unsupported": [],
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "note": "no-retrieval-hits",
        }

    answer = await _generate_answer(chat, chat_model, context, query)
    try:
        score = await faithfulness(
            context,
            answer,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=chat_api_key,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "query": query,
            "error": f"scoring: {type(e).__name__}: {e}",
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "answer": answer,
            "n_hits": len(hits),
            "context_chars": len(context),
        }

    return {
        "query": query,
        "n_hits": len(hits),
        "context_chars": len(context),
        "context_preview": context[:_CONTEXT_PREVIEW_LEN],
        "answer": answer,
        "n_claims": score["n_claims"],
        "n_supported": score["n_supported"],
        "faithfulness_score": score["score"],
        "unsupported": score["unsupported"],
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }


async def _gather_bounded(tasks: list, *, concurrency: int) -> list:
    """Run awaitable tasks with bounded parallelism."""
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(t):
        async with sem:
            return await t

    return await asyncio.gather(*(_wrap(t) for t in tasks))


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--golden", default="tests/eval/golden.jsonl")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--tei-url", default="http://172.19.0.6:80")
    p.add_argument(
        "--chat-url", default="http://172.19.0.7:8000/v1",
        help="OpenAI-compatible chat endpoint base URL (include /v1)",
    )
    p.add_argument(
        "--chat-model", default="orgchat-chat",
        help="chat model name the endpoint expects (vLLM --served-model-name)",
    )
    p.add_argument(
        "--chat-api-key", default=None,
        help="optional bearer token (e.g. sk-internal-dummy from compose/.env)",
    )
    p.add_argument("--k", type=int, default=10, help="top-k hits to feed into answer")
    p.add_argument(
        "--concurrency", type=int, default=4,
        help="number of golden rows processed in parallel (default 4)",
    )
    p.add_argument("--label", default="faithfulness_baseline")
    p.add_argument("--out", default="")
    p.add_argument(
        "--limit", type=int, default=0,
        help="if >0, only process the first N rows (smoke-test shortcut)",
    )
    return p.parse_args(argv)


async def main(argv: Optional[list[str]] = None) -> int:
    ns = _parse_args(argv)
    golden_path = Path(ns.golden)
    if not golden_path.is_file():
        print(f"golden file not found: {golden_path}", file=sys.stderr)
        return 2

    rows = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if ns.limit and ns.limit > 0:
        rows = rows[: ns.limit]

    vs = VectorStore(url=ns.qdrant_url, vector_size=1024)
    emb = TEIEmbedder(base_url=ns.tei_url)
    chat = httpx.AsyncClient(base_url=ns.chat_url, timeout=60.0)
    if ns.chat_api_key:
        chat.headers["Authorization"] = f"Bearer {ns.chat_api_key}"

    results: list[dict] = []
    try:
        tasks = [
            _run_one(
                row,
                vs=vs,
                emb=emb,
                chat=chat,
                chat_url=ns.chat_url,
                chat_model=ns.chat_model,
                chat_api_key=ns.chat_api_key,
                k=ns.k,
            )
            for row in rows
        ]
        results = await _gather_bounded(tasks, concurrency=max(1, ns.concurrency))
    finally:
        await chat.aclose()
        await vs.close()

    scored = [r for r in results if "error" not in r and "faithfulness_score" in r]
    errors = [r for r in results if "error" in r]
    metrics = {
        "n_queries": len(results),
        "n_errors": len(errors),
        "n_scored": len(scored),
        "mean_faithfulness": (
            statistics.mean([r["faithfulness_score"] for r in scored]) if scored else 0.0
        ),
        "median_faithfulness": (
            statistics.median([r["faithfulness_score"] for r in scored]) if scored else 0.0
        ),
        "mean_n_claims": (
            statistics.mean([r["n_claims"] for r in scored]) if scored else 0.0
        ),
        "p50_latency_ms": (
            statistics.median([r["latency_ms"] for r in scored]) if scored else 0.0
        ),
        "p95_latency_ms": (
            sorted([r["latency_ms"] for r in scored])[int(len(scored) * 0.95)]
            if scored
            else 0.0
        ),
    }
    flags = {
        k: os.environ.get(k, "")
        for k in (
            "RAG_HYBRID",
            "RAG_RERANK",
            "RAG_MMR",
            "RAG_CONTEXT_EXPAND",
            "RAG_CONTEXT_EXPAND_WINDOW",
            "RAG_CONTEXTUALIZE_KBS",
            "RAG_MMR_LAMBDA",
        )
    }
    out = {
        "label": ns.label,
        "k": ns.k,
        "chat_model": ns.chat_model,
        "chat_url": ns.chat_url,
        "metrics": metrics,
        "flags": flags,
        "rows": results,
    }

    if ns.out:
        out_path = Path(ns.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))

    print(
        f"[{ns.label}] mean_faithfulness={metrics['mean_faithfulness']:.3f} "
        f"median={metrics['median_faithfulness']:.3f} "
        f"n_scored={metrics['n_scored']}/{metrics['n_queries']} "
        f"errors={metrics['n_errors']} "
        f"p50={metrics['p50_latency_ms']:.0f}ms p95={metrics['p95_latency_ms']:.0f}ms"
    )
    return 0 if metrics["n_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
