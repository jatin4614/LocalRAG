"""Per-document summarizer for the Tier 1 doc-summary index.

Generates a short, high-signal summary for a whole document so retrieval
can route ``global``-intent queries (e.g. "list every report", "summarize
everything about X") against summary points instead of raw chunks. The
summary embedding becomes a single Qdrant point with ``level="doc"``
which bypasses top-k chunk noise.

Fails open: any exception (timeout, 5xx, malformed JSON, missing chat
endpoint) returns ``""`` (empty string). Callers treat an empty summary
as "no summary point to emit" and continue with the normal chunk
ingest — summarization is best-effort, never blocking.

Bounded concurrency is the caller's responsibility (see ``ingest.py``
and ``scripts/backfill_doc_summaries.py``); this module exposes a single
per-document coroutine.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import httpx

from .llm_telemetry import record_llm_call
from .obs import inject_context_into_headers, span

log = logging.getLogger("orgchat.doc_summarizer")


# Approx token budget for the body we feed the summarizer. ~4 chars/token
# → ~32000 chars ≈ 8000 tokens. We truncate by character count to avoid
# pulling a tokenizer dep into the hot path; the model can handle slight
# over/under. Bumped 16000 -> 32000 on 2026-05-04 (Phase 2 of the
# multi-entity-elaborate-answers spec) so the entity-coverage prompt
# sees enough body to enumerate every named formation.
_MAX_BODY_CHARS = 32000

_SUMMARY_PROMPT = """Summarize this document for retrieval. Output two sections, no preamble:

ENTITIES: A comma-separated list of every named formation, brigade, battalion, regiment, or unit that appears at least twice in the document. Use the canonical name as it first appears. If fewer than two distinct named formations are present, list whatever named formations exist (or write "none" if there are none).

SUMMARY: A 5-7 sentence paragraph covering: document name, reporting period, every named entity from the ENTITIES list (one clause per entity naming what activities they were involved in), and any cross-cutting themes (training, construction, intel, etc.) that appear across multiple entities. Do NOT favour the most-mentioned entity over others — every ENTITIES-list member must be named in the SUMMARY.

Document: {filename}

{body}
"""


async def summarize_document(
    chunks: list[str],
    filename: str,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> str:
    """Summarize a document from its chunk texts.

    Args:
        chunks: ordered list of chunk-body strings for the document.
        filename: display name of the document (included in the prompt so
            the summary mentions it by name — useful when the summary is
            later retrieved as context).
        chat_url: base URL of an OpenAI-compatible endpoint
            (e.g. ``http://vllm-chat:8000/v1``).
        chat_model: model name the endpoint expects (e.g. ``orgchat-chat``).
        api_key: optional bearer token. If None, no Authorization header is sent.
        timeout: request timeout (seconds).
        transport: optional httpx transport, primarily for tests.

    Returns:
        The summary text (stripped). On ANY failure returns ``""``.
    """
    if not chunks:
        return ""

    with span("doc.summarize", model=chat_model, n_chunks=len(chunks)):
        return await _summarize_impl(
            chunks=chunks,
            filename=filename,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )


async def _summarize_impl(
    *,
    chunks: list[str],
    filename: str,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str],
    timeout: float,
    transport: Optional[httpx.AsyncBaseTransport],
) -> str:
    # Join first N chunks until we hit the char budget. Preserves the
    # document's natural order — summaries are biased toward the opening
    # content, which is usually the most summary-worthy (abstracts,
    # intros, title pages).
    body_parts: list[str] = []
    total = 0
    for c in chunks:
        if not c:
            continue
        remaining = _MAX_BODY_CHARS - total
        if remaining <= 0:
            break
        if len(c) > remaining:
            body_parts.append(c[:remaining])
            total = _MAX_BODY_CHARS
            break
        body_parts.append(c)
        total += len(c)
    body = "\n\n".join(body_parts)
    if not body:
        return ""

    prompt = _SUMMARY_PROMPT.format(filename=filename or "(untitled)", body=body)
    payload = {
        "model": chat_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 300,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers = inject_context_into_headers(headers)

    url = f"{chat_url.rstrip('/')}/chat/completions"
    # Wave 2 (review §6.12): LLM circuit breaker. Mirrors the TEI breaker
    # at embedder.py:91 — when RAG_CB_LLM_ENABLED=1 + N failures in window,
    # raises CircuitOpenError BEFORE the network call so the chat-LLM
    # outage doesn't cascade into N concurrent retries. Default OFF for
    # safe deploy; flip on after observing baseline failure rate.
    cb_enabled = os.environ.get("RAG_CB_LLM_ENABLED", "0") == "1"
    breaker = None
    if cb_enabled:
        from .circuit_breaker import breaker_for
        breaker = breaker_for("llm")
        try:
            breaker.raise_if_open()
        except Exception as exc:  # noqa: BLE001
            log.warning("doc summary skipped (LLM breaker open): %s", exc)
            return ""
    try:
        # Wrap in ``record_llm_call`` so the doc-summarizer's prompt /
        # completion token spend lands in ``rag_tokens_prompt_total`` /
        # ``rag_tokens_completion_total`` like the contextualizer / hyde /
        # rewriter call sites. This is the heaviest LLM cost during ingest
        # — invisibility here was a real dashboard gap (review §6.3).
        async with record_llm_call(stage="doc_summarizer", model=chat_model) as rec:
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
            usage = data.get("usage") or {}
            rec.set_tokens(
                prompt=usage.get("prompt_tokens", 0),
                completion=usage.get("completion_tokens", 0),
            )
        summary = (data["choices"][0]["message"]["content"] or "").strip()
        if breaker is not None:
            breaker.record_success()
    except Exception as e:  # noqa: BLE001 — fail-open by design
        if breaker is not None:
            breaker.record_failure()
        log.warning("doc summary failed for %s: %s", filename, e)
        return ""

    # Strip common echo prefixes some models emit.
    for prefix in ("Summary:", "SUMMARY:"):
        if summary.startswith(prefix):
            summary = summary[len(prefix):].strip()
            break
    return summary


def parse_structured_summary(raw: str) -> dict[str, list[str] | str]:
    """Parse the chat-LLM ENTITIES + SUMMARY response.

    Returns ``{"entities": [...], "summary": "..."}``. Fail-soft: if a
    section is missing, returns an empty list / empty string for that
    section but keeps whatever was parseable. Empty input returns the
    fully-empty shape.

    Entity dedup is case-insensitive, preserving the first surface form
    (matches ``entity_extractor._dedupe_preserve_first``).
    """
    if not raw:
        return {"entities": [], "summary": ""}

    raw = raw.strip()
    entities_part = ""
    summary_part = ""

    # Find ENTITIES: marker (case-insensitive)
    ent_match = re.search(r"\bENTITIES\s*:\s*", raw, re.IGNORECASE)
    sum_match = re.search(r"\bSUMMARY\s*:\s*", raw, re.IGNORECASE)

    if ent_match and sum_match:
        # Both present — split on the SUMMARY marker
        entities_part = raw[ent_match.end():sum_match.start()].strip()
        summary_part = raw[sum_match.end():].strip()
    elif ent_match:
        # Only ENTITIES — take everything after the marker
        entities_part = raw[ent_match.end():].strip()
    elif sum_match:
        # Only SUMMARY
        summary_part = raw[sum_match.end():].strip()
    else:
        # Neither marker — treat the whole thing as a summary (legacy shape)
        summary_part = raw

    # Parse entities list — comma-separated, dedupe case-insensitively
    if entities_part and entities_part.lower() != "none":
        seen_lower: set[str] = set()
        entities: list[str] = []
        for e in entities_part.split(","):
            e_clean = e.strip()
            if not e_clean:
                continue
            if e_clean.lower() in seen_lower:
                continue
            seen_lower.add(e_clean.lower())
            entities.append(e_clean)
    else:
        entities = []

    return {"entities": entities, "summary": summary_part}


__all__ = ["summarize_document", "parse_structured_summary"]
