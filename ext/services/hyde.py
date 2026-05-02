"""HyDE — Hypothetical Document Embeddings (P3.3).

Generate synthetic "excerpt"-style answers via the chat model, embed them,
and average the embeddings into a single query vector. Retrieval on that
averaged vector snaps closer to real document chunks than the raw-question
embedding would — because document chunks are written as declarative
statements, not questions.

Reference: Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (arXiv:2212.10496).

Flag-gated by ``RAG_HYDE``. Default OFF. When the flag is off, nothing in
this module is imported (the retriever does the import lazily behind the
flag check), so the default retrieval path is byte-identical to pre-P3.3.

Fail-open policy: any chat-completion failure (timeout, 5xx, malformed
JSON, empty text) short-circuits to ``None`` and the caller falls back
to embedding the raw query. HyDE is best-effort — a retrieval slowdown
from chat latency is acceptable, a retrieval OUTAGE from chat latency
is not.

Per-KB override: ``rag_config.hyde = true`` on a KB (consumed by P3.0's
``config_to_env_overrides``) sets ``RAG_HYDE=1`` in the request overlay
so that KB alone gets HyDE without flipping the process-global flag.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Optional

import httpx

from .llm_telemetry import record_llm_call
from .obs import inject_context_into_headers, span
from .retry_policy import with_transient_retry

log = logging.getLogger("orgchat.hyde")


_PROMPT = """You are an expert writing a concise excerpt from a document that would answer the following question. Write 2-3 sentences in the style of a reference document — declarative statements with plausible specifics. Do not ask clarifying questions, do not hedge.

Question: {query}

Excerpt:"""


@with_transient_retry(attempts=2, base_sec=0.5)
async def _hyde_call(
    url: str,
    body: dict,
    headers: dict,
    timeout_s: float,
    transport: Optional[httpx.AsyncBaseTransport],
) -> dict:
    """POST to the chat-completion endpoint and return parsed JSON.

    Wrapped with the shared transient-error retry policy. ``attempts=2`` is
    lower than the embedder (3) because HyDE adds an extra chat call to
    every retrieval — three retries of a 500 ms timeout would put a 1.5 s
    extra latency tax on every miss, which is worse than just falling
    open to the raw-query embedding baseline.

    Wrapped in ``record_llm_call`` so prompt/completion token counters
    are emitted for each hypothetical-doc request — HyDE is the most
    chat-call-heavy retrieval feature we have and operators need to see
    its token spend in the same dashboard as contextualizer / rewriter.
    """
    model_name = body.get("model") or os.environ.get("CHAT_MODEL", "unknown")
    async with record_llm_call(stage="hyde", model=model_name) as rec:
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            r = await client.post(url, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        usage = data.get("usage") or {}
        rec.set_tokens(
            prompt=usage.get("prompt_tokens", 0),
            completion=usage.get("completion_tokens", 0),
        )
        return data


async def _generate_one(
    query: str,
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout_s: float = 15.0,
    temperature: float = 0.3,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Optional[str]:
    """Generate one hypothetical excerpt. Returns None on any failure.

    Fail-open is load-bearing: a single 500 or a network blip on the
    hypothetical-doc call must not break retrieval, it just degrades
    quality back to the raw-query baseline for that request.

    The 2000-char guard catches the "model ignored the length constraint
    and wrote an essay" case — embedding a 10k-char blob is both slow
    and a quality regression (excerpt-style short statements match
    chunks better than multi-paragraph exposition).
    """
    body = {
        "model": chat_model,
        "messages": [{"role": "user", "content": _PROMPT.format(query=query)}],
        "temperature": temperature,
        "max_tokens": 180,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers = inject_context_into_headers(headers)

    url = f"{chat_url.rstrip('/')}/chat/completions"
    try:
        data = await _hyde_call(url, body, headers, timeout_s, transport)
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:  # noqa: BLE001 — fail-open by design
        log.debug("hyde generation failed after retries: %s", e)
        return None

    if not text or len(text) > 2000:
        return None
    return text


async def generate_hypothetical_docs(
    query: str,
    *,
    n: int = 1,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout_s: float = 15.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> list[str]:
    """Generate ``n`` synthetic excerpts in parallel.

    Returns the list of successful generations (length in [0, n]). Failures
    are silently dropped — callers use the length to decide whether to
    proceed or fall back to the raw query embedding.

    ``asyncio.gather`` is used so the N chat calls overlap on the wire;
    at N=3 with Qwen/Gemma the wall time is still ~400 ms (one network
    round-trip), not 3x that.
    """
    if n <= 0:
        return []
    tasks = [
        _generate_one(
            query,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            timeout_s=timeout_s,
            transport=transport,
        )
        for _ in range(n)
    ]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


async def hyde_embed(
    query: str,
    embedder,
    *,
    n: int = 1,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout_s: float = 15.0,
    include_raw_query: bool = True,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Optional[list[float]]:
    """Return the averaged embedding of N synthetic docs (and optionally the raw query).

    Pipeline:
      1. Generate N hypothetical-doc excerpts (parallel chat calls).
      2. Batch-embed all successful excerpts (+ the raw query if requested)
         in a single ``embedder.embed`` call — one network round-trip.
      3. Component-wise mean across the returned vectors.

    Returns ``None`` when:
      * Every hypothetical-doc generation failed (N of N), and
      * ``include_raw_query`` is False — i.e. there's nothing to embed.

    When at least one excerpt succeeded we average what we have and return
    it. When ``include_raw_query=True`` and all generations fail, we would
    end up with just the raw-query embedding (no benefit over the baseline),
    so we return ``None`` in that case too — the caller is expected to
    embed the raw query via the usual path, yielding an identical result
    with one fewer network round-trip.

    ``include_raw_query`` defaults to True because the HyDE paper's own
    ablations show mean(raw_query, hypothetical_docs) beats
    mean(hypothetical_docs_only) on most benchmarks — the raw query adds
    a lexical anchor that keeps the average centered on-topic even when
    a hypothetical doc drifts.
    """
    with span("hyde.generate", n=n, model=chat_model):
        docs = await generate_hypothetical_docs(
            query,
            n=n,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            timeout_s=timeout_s,
            transport=transport,
        )
    if not docs:
        # All generations failed. Caller falls back to raw-query embedding
        # via the normal path — no point in duplicating that work here.
        return None

    texts = docs + ([query] if include_raw_query else [])
    vecs = await embedder.embed(texts)
    if not vecs:
        return None

    # Component-wise mean. All embedders in this project return vectors of
    # identical dimension (TEI is 1024, StubEmbedder defaults to 1024), so
    # we can trust ``len(vecs[0])`` as the canonical dim.
    dim = len(vecs[0])
    n_vecs = len(vecs)
    avg = [sum(v[i] for v in vecs) / n_vecs for i in range(dim)]
    # Review §3.7: TEI returns unit vectors and Qdrant cosine assumes
    # unit-norm queries. The mean of N unit vectors has magnitude ≤ 1 and
    # equals 1 only when all inputs point in the same direction — so the
    # raw average silently miscalibrates score thresholds. Renormalize so
    # downstream distance scoring stays consistent with the rest of the
    # query path. Guard against the degenerate zero-norm case (opposing
    # vectors that exactly cancel) — div-by-zero would crash retrieval.
    norm = math.sqrt(sum(x * x for x in avg))
    if norm > 0.0:
        avg = [x / norm for x in avg]
    return avg


def is_enabled() -> bool:
    """Read ``RAG_HYDE`` via ``flags.get`` so per-KB overrides take effect.

    Returns True iff the flag is exactly ``"1"``. Any other value (unset,
    ``"0"``, ``""``, ``"true"``) is treated as disabled. We are strict
    here because HyDE has a real cost (+chat call) and should not be
    accidentally enabled by a typo in a config.
    """
    from ext.services import flags
    return flags.get("RAG_HYDE", "0") == "1"


__all__ = [
    "generate_hypothetical_docs",
    "hyde_embed",
    "is_enabled",
]
