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
from typing import Optional

import httpx

from .llm_telemetry import record_llm_call
from .obs import inject_context_into_headers, span

log = logging.getLogger("orgchat.doc_summarizer")


# Approx token budget for the body we feed the summarizer. ~4 chars/token
# → ~4000 tokens ≈ 16000 chars. We truncate by character count to avoid
# pulling a tokenizer dep into the hot path; the model can handle slight
# over/under.
_MAX_BODY_CHARS = 16000

_SUMMARY_PROMPT = """Summarize the following document in 3 sentences. Include the document name, top-line content, and dates/entities/identifiers a reader would need to know. Write as a single paragraph of plain prose — no bullets, no preamble.

Document: {filename}

{body}

Summary:"""


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
    except Exception as e:  # noqa: BLE001 — fail-open by design
        log.warning("doc summary failed for %s: %s", filename, e)
        return ""

    # Strip common echo prefixes some models emit.
    for prefix in ("Summary:", "SUMMARY:"):
        if summary.startswith(prefix):
            summary = summary[len(prefix):].strip()
            break
    return summary


__all__ = ["summarize_document"]
