#!/usr/bin/env python3
"""Unified Phase 0 runner: retrieval + faithfulness + per-intent breakdown.

Drives the live open-webui pipeline against the hand-labeled golden JSONL
(``tests/eval/golden_human.jsonl``) and writes a single JSON with the schema
defined in ``docs/rag-phase0-1a-4-execution-plan.md §3.3``.

Reuses existing modules:
  * ``ext.services.retriever.retrieve`` for retrieval
  * ``ext.services.reranker.rerank_with_flag`` for optional rerank
  * ``tests.eval.scorer`` for chunk_recall / MRR
  * ``tests.eval.faithfulness.faithfulness`` for RAGAS-style faithfulness
  * ``tests.eval.query_mix_classifier.classify`` to sanity-check golden labels

What this adds on top of ``run_eval.py`` + ``run_faithfulness_eval.py``:
  * targets the live numeric KB (kb_id=1), not the synthetic ``kb_eval``
  * joins retrieval, rerank, MMR, context expand with answer generation and
    faithfulness scoring in one pass — no duplicate Qdrant reads
  * per-intent aggregation (``specific``, ``global``, ``metadata``, ``multihop``,
    ``adversarial``)
  * p10 aggregate (bottom decile by chunk_recall / faithfulness) so we catch
    tail quality, not just the mean
  * handles missing ``expected_chunk_indices`` rows (global / metadata queries
    score doc-level recall only; chunk_recall returns 0.0 and is excluded from
    the chunk_recall aggregate to avoid dragging the mean)
  * context_precision, context_recall, answer_relevance — all grounded in the
    same LLM judge as faithfulness (two extra grading calls per row, see
    ``_context_precision``/``_context_recall``/``_answer_relevance`` below)
  * fail-open: if the chat endpoint is unreachable the retrieval metrics still
    land in the JSON with faithfulness / answer_relevance / context_* set to
    null, and a blocker is recorded in ``errors``.

Usage::

    python -m tests.eval.run_all \\
        --golden tests/eval/golden_human.jsonl \\
        --kb-id 1 \\
        --qdrant-url http://localhost:6333 \\
        --tei-url http://172.19.0.6:80 \\
        --chat-url http://172.19.0.8:8000/v1 \\
        --chat-model orgchat-chat \\
        --chat-api-key sk-internal-dummy \\
        --out tests/eval/results/baseline-pre-phase1a.json

The run time is dominated by answer-gen + claim-extract + per-claim grading;
~2-3 seconds per row on Qwen-14B-class GPUs. Faithfulness is OPT-IN via
``--with-faithfulness`` (default on). Pass ``--no-faithfulness`` to capture
retrieval-only metrics in roughly 30 s.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ext.services.embedder import TEIEmbedder  # noqa: E402
from ext.services.reranker import rerank, rerank_with_flag  # noqa: E402
from ext.services.retriever import retrieve  # noqa: E402
from ext.services.vector_store import Hit, VectorStore  # noqa: E402
from tests.eval.faithfulness import faithfulness, grade_claim  # noqa: E402
from tests.eval.scorer import (  # noqa: E402
    chunk_recall_at_k,
    doc_recall_at_k,
    mrr_at_k,
)
from tests.eval.query_mix_classifier import classify  # noqa: E402


INTENT_LABELS = ("specific", "global", "metadata", "multihop", "adversarial")

# UUID namespace used by ingest.py to derive point IDs from (doc_id, chunk_index).
# Matches ``_POINT_NS`` in ``ext/services/ingest.py``. Kept in-tree so the
# mapping from expected_doc_ids/expected_chunk_indices to point-IDs doesn't
# require an additional Qdrant scroll.
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(doc_id: int, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NS, f"doc:{doc_id}:chunk:{chunk_index}"))


def _expected_chunk_ids(row: dict) -> list[str]:
    """Derive expected Qdrant point IDs from ``expected_doc_ids`` + ``expected_chunk_indices``.

    If the two lists have equal length we pair them element-wise. Otherwise
    (typically global/metadata rows where ``expected_chunk_indices`` is an
    empty list) we return an empty list — chunk-level recall isn't meaningful
    for those intents.
    """
    docs = row.get("expected_doc_ids") or []
    chunks = row.get("expected_chunk_indices") or []
    if not docs or not chunks:
        return []
    if len(docs) != len(chunks):
        return []
    return [_point_id(int(d), int(c)) for d, c in zip(docs, chunks)]


# ---------------------------------------------------------------------------
# Retrieval pipeline — reuses ext/services/retriever.retrieve so hybrid/sparse
# behaviour matches production.
# ---------------------------------------------------------------------------


async def _retrieve_pipeline(
    query: str,
    *,
    kb_id: int,
    vs: VectorStore,
    emb: TEIEmbedder,
    per_kb_limit: int,
    total_limit: int,
    rerank_k: int,
) -> list[Hit]:
    hits = await retrieve(
        query=query,
        selected_kbs=[{"kb_id": kb_id, "subtag_ids": []}],
        chat_id=None,
        vector_store=vs,
        embedder=emb,
        per_kb_limit=per_kb_limit,
        total_limit=total_limit,
    )
    # Apply the same optional rerank pathway used by chat_rag_bridge when
    # RAG_RERANK=1. Leaves existing env flag plumbing intact.
    hits = rerank_with_flag(
        query,
        hits,
        top_k=min(len(hits), rerank_k) if hits else 0,
        fallback_fn=rerank,
    )
    return hits


# ---------------------------------------------------------------------------
# Answer generation + quality grading
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = (
    "You are a careful technical assistant. Answer the user's question using "
    "ONLY the provided context. If the context does not contain the answer, "
    "say so explicitly. Do not invent facts."
)
_CONTEXT_CAP_CHARS = 16_000


def _build_context(hits: list[Hit], k: int) -> str:
    parts: list[str] = []
    for h in hits[:k]:
        payload = getattr(h, "payload", None) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
    joined = "\n\n---\n\n".join(parts)
    return joined[:_CONTEXT_CAP_CHARS]


async def _generate_answer(
    client: httpx.AsyncClient,
    chat_model: str,
    context: str,
    query: str,
    *,
    api_key: Optional[str],
    timeout: float = 30.0,
    max_tokens: int = 512,
) -> str:
    headers: dict = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    r = await client.post(
        "/chat/completions", json=body, headers=headers, timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


async def _context_precision(
    context_chunks: list[str],
    query: str,
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str],
) -> Optional[float]:
    """Fraction of retrieved chunks that were judged RELEVANT to the query.

    Reuses faithfulness.grade_claim as an LLM-judge: asks "is this chunk
    relevant to that question?" for each retained chunk. Fail-open → returns
    None if any grade call errors out before we accumulate a meaningful
    count.
    """
    if not context_chunks:
        return None
    hits = 0
    total = 0
    for chunk in context_chunks:
        prompt_claim = f"This chunk is relevant to the question: {query!r}"
        try:
            ok = await grade_claim(
                context=chunk,
                claim=prompt_claim,
                chat_url=chat_url,
                chat_model=chat_model,
                api_key=api_key,
            )
        except Exception:
            return None
        total += 1
        hits += int(bool(ok))
    return hits / total if total else None


async def _context_recall(
    context: str,
    expected_snippet: str,
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str],
) -> Optional[float]:
    """Did the retrieved context actually include the snippet we expected?

    Uses a faithfulness-style single-claim grade: "is ``expected_snippet``
    supported by the context?". 1.0 if yes, 0.0 if no. Returns None on
    absent expected snippets (adversarial rows with empty snippet → recall
    isn't meaningful; caller handles that by omitting the row from the
    recall aggregate).
    """
    if not expected_snippet or not expected_snippet.strip():
        return None
    try:
        ok = await grade_claim(
            context=context,
            claim=expected_snippet.strip(),
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
        )
    except Exception:
        return None
    return 1.0 if ok else 0.0


async def _answer_relevance(
    query: str,
    answer: str,
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str],
) -> Optional[float]:
    """Did the generated answer address the query?

    Simple single-claim grade: "the ANSWER addresses the QUESTION". 1.0 if
    yes, 0.0 if no. Cheap and directional — the gold standard is
    embedding-cosine to back-generated questions (RAGAS), but that adds an
    extra embed call per row and muddies the log for very little signal on
    a 50-row set.
    """
    if not answer or not answer.strip():
        return 0.0
    try:
        ok = await grade_claim(
            context=f"QUESTION: {query}\n\nANSWER: {answer}",
            claim="The ANSWER correctly addresses the QUESTION.",
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
        )
    except Exception:
        return None
    return 1.0 if ok else 0.0


# ---------------------------------------------------------------------------
# Per-row runner
# ---------------------------------------------------------------------------


async def _run_one(
    row: dict,
    *,
    kb_id: int,
    vs: VectorStore,
    emb: TEIEmbedder,
    chat: Optional[httpx.AsyncClient],
    chat_url: str,
    chat_model: str,
    chat_api_key: Optional[str],
    k: int,
    per_kb_limit: int,
    rerank_k: int,
    with_faithfulness: bool,
) -> dict:
    query = row["query"]
    gold_label = row.get("intent_label", "specific")
    expected_chunk_ids = _expected_chunk_ids(row)
    expected_doc_ids = {int(d) for d in (row.get("expected_doc_ids") or [])}
    expected_snippet = row.get("expected_answer_snippet") or ""

    result: dict = {
        "query": query,
        "intent_label": gold_label,
        "classifier_label": classify(query),
        "expected_doc_ids": sorted(expected_doc_ids),
        "expected_chunk_ids": expected_chunk_ids,
    }

    t0 = time.perf_counter()
    try:
        hits = await _retrieve_pipeline(
            query,
            kb_id=kb_id,
            vs=vs,
            emb=emb,
            per_kb_limit=per_kb_limit,
            total_limit=max(per_kb_limit, 30),
            rerank_k=rerank_k,
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"retrieval: {type(exc).__name__}: {exc}"
        result["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result

    retrieved_chunk_ids = [str(h.id) for h in hits[:k]]
    retrieved_doc_ids: list[int] = []
    for h in hits[:k]:
        payload = getattr(h, "payload", None) or {}
        did = payload.get("doc_id")
        try:
            retrieved_doc_ids.append(int(did))
        except (TypeError, ValueError):
            continue

    result["n_hits"] = len(hits)
    result["retrieved_chunk_ids"] = retrieved_chunk_ids
    result["retrieved_doc_ids"] = retrieved_doc_ids

    # Retrieval metrics — chunk_recall is only meaningful when the golden row
    # specified expected_chunk_indices. Doc-level recall+MRR is applied
    # across every intent (including adversarial, where we expect 0.0).
    if expected_chunk_ids:
        result["chunk_recall_at_k"] = chunk_recall_at_k(
            retrieved_chunk_ids, set(expected_chunk_ids), k,
        )
    else:
        result["chunk_recall_at_k"] = None
    if expected_doc_ids:
        result["doc_recall_at_k"] = doc_recall_at_k(
            retrieved_doc_ids, expected_doc_ids, k,
        )
        result["mrr_at_k"] = mrr_at_k(retrieved_doc_ids, expected_doc_ids, k)
    else:
        result["doc_recall_at_k"] = None
        result["mrr_at_k"] = None

    # Build context for downstream grading + store preview
    context_chunks = [(getattr(h, "payload", {}) or {}).get("text", "") for h in hits[:k]]
    context_chunks = [c for c in context_chunks if c]
    context = "\n\n---\n\n".join(context_chunks)[:_CONTEXT_CAP_CHARS]
    result["context_chars"] = len(context)
    result["context_preview"] = context[:400]

    faithfulness_score: Optional[float] = None
    context_precision_score: Optional[float] = None
    context_recall_score: Optional[float] = None
    answer_relevance_score: Optional[float] = None
    answer = ""

    if with_faithfulness and chat is not None and context:
        try:
            answer = await _generate_answer(
                chat, chat_model, context, query, api_key=chat_api_key,
            )
        except Exception as exc:  # noqa: BLE001
            result["answer_error"] = f"{type(exc).__name__}: {exc}"
        if answer:
            try:
                fscore = await faithfulness(
                    context,
                    answer,
                    chat_url=chat_url,
                    chat_model=chat_model,
                    api_key=chat_api_key,
                )
                faithfulness_score = fscore["score"]
                result["n_claims"] = fscore["n_claims"]
                result["n_supported"] = fscore["n_supported"]
                result["unsupported_claims"] = fscore["unsupported"]
            except Exception as exc:  # noqa: BLE001
                result["faithfulness_error"] = f"{type(exc).__name__}: {exc}"
        # Context precision: did retained chunks address the query at all?
        # Skip for adversarial — the answer is "nothing matches" by design.
        if gold_label != "adversarial":
            try:
                context_precision_score = await _context_precision(
                    context_chunks,
                    query,
                    chat_url=chat_url,
                    chat_model=chat_model,
                    api_key=chat_api_key,
                )
            except Exception:
                context_precision_score = None
            try:
                context_recall_score = await _context_recall(
                    context,
                    expected_snippet,
                    chat_url=chat_url,
                    chat_model=chat_model,
                    api_key=chat_api_key,
                )
            except Exception:
                context_recall_score = None
        # Answer relevance — always graded; captures adversarial refusals too
        # (a refusal SHOULD still "address" the question by saying so).
        try:
            answer_relevance_score = await _answer_relevance(
                query,
                answer,
                chat_url=chat_url,
                chat_model=chat_model,
                api_key=chat_api_key,
            )
        except Exception:
            answer_relevance_score = None

    result["answer"] = answer
    result["faithfulness"] = faithfulness_score
    result["context_precision"] = context_precision_score
    result["context_recall"] = context_recall_score
    result["answer_relevance"] = answer_relevance_score
    result["latency_ms"] = (time.perf_counter() - t0) * 1000

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> Optional[float]:
    clean = [x for x in xs if x is not None]
    if not clean:
        return None
    return round(statistics.mean(clean), 4)


def _p10(xs: list[float]) -> Optional[float]:
    clean = [x for x in xs if x is not None]
    if not clean:
        return None
    srt = sorted(clean)
    idx = max(0, int(round(0.10 * (len(srt) - 1))))
    return round(srt[idx], 4)


def _agg_set(rows: list[dict]) -> dict:
    """Compute aggregate metrics for an arbitrary subset of rows."""
    return {
        "n": len(rows),
        "chunk_recall@10": _mean([r.get("chunk_recall_at_k") for r in rows]),
        "doc_recall@10": _mean([r.get("doc_recall_at_k") for r in rows]),
        "mrr@10": _mean([r.get("mrr_at_k") for r in rows]),
        "faithfulness": _mean([r.get("faithfulness") for r in rows]),
        "context_precision": _mean([r.get("context_precision") for r in rows]),
        "context_recall": _mean([r.get("context_recall") for r in rows]),
        "answer_relevance": _mean([r.get("answer_relevance") for r in rows]),
    }


def _latency_aggregate(rows: list[dict]) -> dict:
    lats = sorted((r.get("latency_ms") or 0.0) for r in rows)
    if not lats:
        return {"p50": 0.0, "p95": 0.0}
    p50 = lats[len(lats) // 2]
    idx = min(len(lats) - 1, max(0, int(round(0.95 * (len(lats) - 1)))))
    return {"p50": round(lats[len(lats) // 2], 1), "p95": round(lats[idx], 1)}


def _pipeline_version_string() -> str:
    """Compose the pipeline version string the run was taken against.

    Reads the env flags the retrieval pipeline cares about; lands in the JSON
    so later comparisons know what configuration produced the numbers.
    """
    try:
        from ext.services.pipeline_version import current_version  # type: ignore
        return current_version()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", default="tests/eval/golden_human.jsonl")
    p.add_argument("--kb-id", type=int, default=1)
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--tei-url", default="http://172.19.0.6:80")
    p.add_argument("--chat-url", default="http://172.19.0.8:8000/v1")
    p.add_argument("--chat-model", default="orgchat-chat")
    p.add_argument("--chat-api-key", default="sk-internal-dummy")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--per-kb-limit", type=int, default=10,
                   help="passed to retriever.retrieve; matches production default")
    p.add_argument("--rerank-k", type=int, default=10,
                   help="pool size for rerank_with_flag (no-op if RAG_RERANK=0)")
    p.add_argument("--with-faithfulness", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Toggle answer-gen + faithfulness grading (default on)")
    p.add_argument("--limit", type=int, default=0, help="if >0, process only first N rows")
    p.add_argument("--out", default="tests/eval/results/baseline-pre-phase1a.json")
    p.add_argument("--label", default="baseline-pre-phase1a")
    return p.parse_args(argv)


async def main(argv=None) -> int:
    ns = parse_args(argv)
    golden_path = Path(ns.golden)
    if not golden_path.exists():
        print(f"golden file not found: {golden_path}", file=sys.stderr)
        return 2
    rows = [
        json.loads(ln)
        for ln in golden_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if ns.limit and ns.limit > 0:
        rows = rows[: ns.limit]
    if not rows:
        print("golden file is empty", file=sys.stderr)
        return 2

    vs = VectorStore(url=ns.qdrant_url, vector_size=1024)
    emb = TEIEmbedder(base_url=ns.tei_url)
    chat = None
    blocker: Optional[str] = None
    if ns.with_faithfulness:
        chat = httpx.AsyncClient(base_url=ns.chat_url, timeout=60.0)
        # Probe chat endpoint so we fail fast with a readable blocker, rather
        # than hundreds of line-item errors from per-row grading.
        try:
            probe = await chat.get("/models")
            if probe.status_code >= 400:
                blocker = f"chat endpoint /models returned {probe.status_code}"
        except Exception as e:  # noqa: BLE001
            blocker = f"chat endpoint unreachable: {type(e).__name__}: {e}"
        if blocker:
            print(f"WARN: {blocker} — proceeding with retrieval-only metrics",
                  file=sys.stderr)
            await chat.aclose()
            chat = None
            ns.with_faithfulness = False

    print(f"running {len(rows)} rows against kb_{ns.kb_id} "
          f"(faithfulness={'on' if ns.with_faithfulness else 'off'})",
          file=sys.stderr)

    results: list[dict] = []
    try:
        for i, row in enumerate(rows, 1):
            res = await _run_one(
                row,
                kb_id=ns.kb_id,
                vs=vs,
                emb=emb,
                chat=chat,
                chat_url=ns.chat_url,
                chat_model=ns.chat_model,
                chat_api_key=ns.chat_api_key,
                k=ns.k,
                per_kb_limit=ns.per_kb_limit,
                rerank_k=ns.rerank_k,
                with_faithfulness=ns.with_faithfulness,
            )
            results.append(res)
            if i % 5 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] done", file=sys.stderr)
    finally:
        await vs.close()
        await emb.aclose()
        if chat is not None:
            await chat.aclose()

    clean = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    # Aggregate (all intents)
    aggregate = _agg_set(clean)
    # Per-intent breakdown
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in clean:
        by_intent[r.get("intent_label", "specific")].append(r)
    per_intent = {
        lbl: _agg_set(by_intent.get(lbl, []))
        for lbl in INTENT_LABELS
    }
    # p10 tail aggregates over chunk_recall + faithfulness
    p10_block = {
        "chunk_recall@10": _p10([r.get("chunk_recall_at_k") for r in clean]),
        "faithfulness": _p10([r.get("faithfulness") for r in clean]),
    }
    latency = _latency_aggregate(clean)

    flags = {
        k: os.environ.get(k, "")
        for k in (
            "RAG_HYBRID", "RAG_RERANK", "RAG_MMR", "RAG_CONTEXT_EXPAND",
            "RAG_CONTEXTUALIZE_KBS", "RAG_MMR_LAMBDA",
            "RAG_CONTEXT_EXPAND_WINDOW", "RAG_SPOTLIGHT", "RAG_HYDE",
            "RAG_SEMCACHE", "RAG_BUDGET_TOKENIZER",
        )
    }
    output = {
        "label": ns.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": _pipeline_version_string(),
        "golden_file": str(golden_path),
        "kb_id": ns.kb_id,
        "k": ns.k,
        "n_queries": len(results),
        "n_errored": len(errored),
        "blockers": [blocker] if blocker else [],
        "flags": flags,
        "aggregate": aggregate,
        "per_intent": per_intent,
        "p10": p10_block,
        "latency_ms": latency,
        "errors": [
            {"query": r["query"], "error": r["error"]}
            for r in errored[:25]
        ],
        "rows": results,
    }

    out_path = Path(ns.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    a = aggregate
    print(
        f"[{ns.label}] "
        f"chunk_recall@{ns.k}={a['chunk_recall@10']}  "
        f"doc_recall@{ns.k}={a['doc_recall@10']}  "
        f"mrr@{ns.k}={a['mrr@10']}  "
        f"faithfulness={a['faithfulness']}  "
        f"ctx_p={a['context_precision']}  ctx_r={a['context_recall']}  "
        f"ans_rel={a['answer_relevance']}  "
        f"p50={latency['p50']}ms p95={latency['p95']}ms  "
        f"errors={len(errored)}/{len(results)}"
    )
    print(f"wrote {out_path}")
    return 0 if len(errored) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
