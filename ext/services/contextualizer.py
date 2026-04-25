"""Contextual chunk augmentation via chat model (P2.7 + Plan A 3.1).

For each chunk, call the chat endpoint with a "situate this chunk in the
document" prompt. The short LLM-generated context is prepended to the
chunk text before embedding, following Anthropic's Contextual Retrieval
recipe (https://www.anthropic.com/news/contextual-retrieval).

Design choices:
- **Fail-open.** Any error (timeout, 5xx, malformed JSON, oversized reply)
  returns the original chunk unchanged. A bad context pass is never worse
  than no context pass, so we never crash ingest over it.
- **Per-chunk, not batched.** Anthropic's pattern is one call per chunk.
  vLLM's prefix cache will reuse the shared system + doc-header messages
  across calls for the same document (see ``build_contextualize_prompt``
  below — the FINAL message is the only chunk-specific payload), so
  batching them into a single request would only obscure the per-chunk
  failure mode without buying KV-cache reuse we don't already get.
- **Bounded concurrency.** ``contextualize_batch`` fans out via
  ``asyncio.gather`` under a semaphore (default 8) so we don't hammer
  the chat endpoint with one request per chunk on a 1000-chunk doc.
- **OFF by default.** ``RAG_CONTEXTUALIZE_KBS=1`` to opt in. Cost on
  Qwen2.5-14B is ~2-3s/chunk; contextual retrieval's +35% win reported
  by Anthropic is measured with Claude — local-model effectiveness is
  unproven. Eval before enabling.

No imports of this module from the default-off ingest path — that keeps
the default behaviour byte-identical (no chat calls, no httpx churn).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

import httpx

from .llm_telemetry import record_llm_call
from .retry_policy import with_transient_retry


log = logging.getLogger("orgchat.contextualize")


# Common echo prefixes Qwen2.5 sometimes emits despite the "no preamble"
# instruction. Stripped case-insensitively from the start of the reply.
_ECHO_PREFIXES = (
    "Context:",
    "Situated context:",
    "Here is the context:",
    "Here's the context:",
    "Succinct context:",
)

# Hard caps. These are safety valves, not tuning knobs.
_MAX_CHUNK_CHARS = 4000     # clip chunk in the prompt to keep request size bounded
_MAX_CONTEXT_CHARS = 800    # longer than this = model rambled → fall open


def is_enabled() -> bool:
    """Whether contextual retrieval is enabled for this process.

    Read at call time (not import) so env var toggles take effect in tests
    without reimporting the module.
    """
    return os.environ.get("RAG_CONTEXTUALIZE_KBS", "0") == "1"


def build_contextualize_prompt(
    *,
    document_text: str,
    chunk_text: str,
    document_metadata: dict[str, Any],
) -> list[dict[str, str]]:
    """Build a cache-friendly chat-completions message list for contextualization.

    The returned list has TWO parts:

    1. The system message and the doc-header user message together carry the
       DOCUMENT-level context — both are byte-identical for all chunks of the
       same document, so vllm-chat's automatic-prefix-caching detects the
       shared prefix and reuses the KV cache across chunks. On a 1000-chunk
       document this turns 1000 cold-prompt evals into one cold + 999 cached.
    2. The FINAL user message is the chunk-specific payload — only this one
       differs between sibling chunks of the same doc.

    Output target: a 50-100 token context prefix that carries the
    *temporal* (document date) and *cross-document* (related titles) anchors
    Anthropic's Contextual Retrieval blog post identifies as the load-bearing
    signal — that's what moves the numbers from generic to the advertised
    49% retrieval-failure reduction on inter-related corpora.

    Optional metadata fields (``subtag_name``, ``document_date``,
    ``related_doc_titles``) collapse to empty strings when missing rather
    than leaking ``"None"`` / ``"[]"`` Python repr noise into the prompt
    body — that noise would burn cache tokens without informing the model.

    Args:
        document_text: full document text. Caller is responsible for any
            length clipping appropriate to the model's context window.
        chunk_text: the specific chunk this prompt should produce a prefix
            for. Caller is responsible for any per-chunk length clipping.
        document_metadata: dict carrying ``filename``, ``kb_name``,
            ``subtag_name``, ``document_date``, ``related_doc_titles``.
            All fields are optional; missing or falsy values degrade
            gracefully with no Python-repr leakage.

    Returns:
        A 3-message list ``[{role: "system"}, {role: "user", doc-header},
        {role: "user", chunk}]`` ready to pass as ``messages`` in an
        OpenAI-compatible chat-completions request.
    """
    filename = document_metadata.get("filename") or "unknown"
    kb_name = document_metadata.get("kb_name") or "unknown"
    subtag_name = document_metadata.get("subtag_name") or ""
    document_date = document_metadata.get("document_date") or ""
    related = document_metadata.get("related_doc_titles") or []

    # Build the document-level header (stable across chunks → cacheable).
    # Empty-string conditional substitution avoids "None" / "[]" leaks per
    # the optional-fields contract — a missing date line is no line at all.
    subtag_line = f" > {subtag_name}" if subtag_name else ""
    date_line = f"Document date: {document_date}\n" if document_date else ""
    related_line = (
        f"Related documents: {', '.join(related)}\n" if related else ""
    )

    system_prompt = (
        "You are a retrieval context generator. Given a full document and one "
        "chunk of that document, write a 50-100 token context prefix that will "
        "be prepended to the chunk before it is embedded and indexed for search.\n\n"
        "The prefix MUST include:\n"
        "- The document's filename or title\n"
        "- The document's date or time period (if known)\n"
        "- The knowledge-base section (KB name and subtag)\n"
        "- Any relationships to prior documents (if listed)\n"
        "- A one-clause summary of what THIS chunk is about within the document\n\n"
        "Output ONLY the prefix text — no explanations, no JSON, no meta-commentary. "
        "Keep it under 100 tokens. Write in the document's language."
    )

    doc_header_msg = (
        f"Document: {filename}\n"
        f"Knowledge base: {kb_name}{subtag_line}\n"
        f"{date_line}"
        f"{related_line}"
        f"\nFull document text:\n"
        f"---\n{document_text}\n---"
    )

    chunk_msg = (
        f"Chunk text:\n---\n{chunk_text}\n---\n\n"
        f"Write the context prefix now."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": doc_header_msg},
        {"role": "user", "content": chunk_msg},
    ]


@with_transient_retry(attempts=2, base_sec=0.5)
async def _contextualize_call(
    url: str,
    body: dict,
    headers: dict,
    timeout_s: float,
    transport: Optional[httpx.AsyncBaseTransport],
) -> dict:
    """POST to the chat-completion endpoint and return parsed JSON.

    Wrapped with the shared transient-error retry policy. ``attempts=2``
    keeps the per-chunk worst-case bounded — a 1000-chunk doc with three
    retries of a 500 ms timeout each would add 1500 ms × 1000 = 25 min
    of pure wait time, which would dwarf the actual ingest. Fail-open in
    the caller ensures a per-chunk failure just degrades to raw chunk
    text, never crashes ingest.

    Wrapped in ``record_llm_call`` so prompt/completion token counters
    are emitted for each request — even on retry-exhausted failure the
    in-flight call still burned tokens upstream and operators need to
    see that spend.
    """
    model_name = body.get("model") or os.environ.get("CHAT_MODEL", "unknown")
    async with record_llm_call(stage="contextualizer", model=model_name) as rec:
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


async def contextualize_chunk(
    chunk_text: str,
    doc_title: str,
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout_s: float = 10.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> str:
    """Prepend an LLM-generated situating context to ``chunk_text``.

    Returns ``"<context>\\n\\n<chunk_text>"`` on success. On ANY error
    (timeout, network, 5xx, empty reply, oversized reply, schema mismatch)
    returns the raw ``chunk_text`` unchanged — augmentation is best-effort.

    Args:
        chunk_text: the chunk to contextualize.
        doc_title: document title (or filename) used as the situating anchor.
        chat_url: base URL of an OpenAI-compatible chat endpoint.
        chat_model: model name the endpoint expects.
        api_key: optional bearer token; no Authorization header when None.
        timeout_s: per-request deadline (seconds).
        transport: optional httpx transport for tests.
    """
    if not chunk_text:
        return chunk_text

    # Use the cache-friendly builder. The legacy entry point only carries
    # ``doc_title`` (no full doc text, no rich metadata) — Phase 3.2 will
    # introduce ``contextualize_chunks_with_prefix`` which threads through
    # the full document text plus the KB / subtag / date / related-titles
    # bundle. For now the builder still gives us byte-identical doc-level
    # messages across sibling chunks — vllm-chat's prefix cache reuses
    # them — and the prompt is the same shape the new caller will use.
    clipped = chunk_text[:_MAX_CHUNK_CHARS]
    messages = build_contextualize_prompt(
        document_text=clipped,
        chunk_text=clipped,
        document_metadata={"filename": doc_title or "Untitled"},
    )
    body = {
        "model": chat_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 96,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{chat_url.rstrip('/')}/chat/completions"
    try:
        data = await _contextualize_call(url, body, headers, timeout_s, transport)
        context = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:  # noqa: BLE001 — fail-open by design
        log.debug("contextualize failed after retries, keeping raw chunk: %s", e)
        return chunk_text

    # Empty / oversized → fall open. An empty reply means the model
    # refused; an oversized one means it rambled past the instruction.
    # Either way, augmentation would be net-negative vs. a clean chunk.
    if not context or len(context) > _MAX_CONTEXT_CHARS:
        return chunk_text

    # Strip common echo prefixes the model emits despite the instruction.
    for prefix in _ECHO_PREFIXES:
        if context.lower().startswith(prefix.lower()):
            context = context[len(prefix):].strip()
            break

    # If stripping nuked the response, fall open rather than prepend ""
    # (which would just add a blank line before the chunk).
    if not context:
        return chunk_text

    return f"{context}\n\n{chunk_text}"


async def contextualize_batch(
    chunks_with_titles: Iterable[tuple[str, str]],
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    concurrency: int = 8,
    timeout_s: float = 10.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> list[str]:
    """Contextualize many chunks in parallel under a concurrency semaphore.

    Preserves input order. Each chunk's augmentation is independent; a
    single failure does not affect siblings (they each fall open on their
    own to their raw text).

    Args:
        chunks_with_titles: iterable of ``(chunk_text, doc_title)`` pairs.
        concurrency: max in-flight chat calls. Cap so we don't saturate
            vLLM's batch queue on large docs.
        Other args: forwarded to ``contextualize_chunk``.

    Returns:
        List of augmented texts in the same order as the input iterable.
    """
    import asyncio  # local import — only needed when contextualization is enabled

    pairs = list(chunks_with_titles)
    if not pairs:
        return []

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(ct: str, dt: str) -> str:
        async with sem:
            return await contextualize_chunk(
                ct,
                dt,
                chat_url=chat_url,
                chat_model=chat_model,
                api_key=api_key,
                timeout_s=timeout_s,
                transport=transport,
            )

    return await asyncio.gather(*(_one(ct, dt) for ct, dt in pairs))


async def _chat_call(
    messages: list[dict[str, str]],
    *,
    chat_url: Optional[str] = None,
    chat_model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 10.0,
    max_tokens: int = 96,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> str:
    """Send ``messages`` to the chat endpoint and return the assistant content.

    Thin wrapper over ``_contextualize_call`` that returns just the content
    string instead of the full chat-completions JSON. Preserves the existing
    OTel/retry/telemetry on ``_contextualize_call`` (we delegate to it) so
    operators still see prompt/completion token counts per request.

    Designed as a module-level seam for ``contextualize_chunks_with_prefix``
    — the test suite patches ``ext.services.contextualizer._chat_call`` to
    inject canned LLM responses without standing up an httpx transport.

    Returns the raw assistant content (untrimmed). Caller is responsible
    for empty / oversized handling and prefix-strip semantics.
    """
    resolved_url = chat_url or os.environ.get("OPENAI_API_BASE_URL") or ""
    resolved_model = chat_model or os.environ.get("CHAT_MODEL", "orgchat-chat")
    body = {
        "model": resolved_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{resolved_url.rstrip('/')}/chat/completions"
    data = await _contextualize_call(url, body, headers, timeout_s, transport)
    return data["choices"][0]["message"]["content"] or ""


async def contextualize_chunks_with_prefix(
    chunks: list[dict],
    *,
    document_text: str,
    document_metadata: dict[str, Any],
    chat_url: Optional[str] = None,
    chat_model: Optional[str] = None,
    api_key: Optional[str] = None,
    concurrency: int = 8,
    timeout_s: float = 10.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> list[dict]:
    """Augment each chunk dict with a separate ``context_prefix`` field.

    Phase-3.2 evolution of ``contextualize_batch``: instead of returning a
    list of concatenated strings (which throws away the prefix as its own
    datum), this mutates each chunk dict to carry BOTH the LLM-generated
    prefix (``chunk["context_prefix"]``) AND the concatenated text
    (``chunk["text"] = f"{prefix}\\n\\n{original}"``). This lets us
    regenerate prefixes later — e.g. after a prompt tweak or model upgrade
    — without re-embedding the originals, and lets retrieval / debugging
    tooling reason about the prefix independently of the chunk body.

    Fail-open per chunk: on any LLM error (timeout, 5xx, oversized reply,
    schema mismatch) we leave the chunk's ``text`` unchanged and set
    ``context_prefix=None``. A bad context pass is never worse than no
    context pass — augmentation is best-effort, ingest never crashes
    because the chat endpoint hiccupped on one chunk.

    Args:
        chunks: list of chunk dicts. Each must have a ``text`` field;
            the caller may pre-seed ``context_prefix=None`` (the
            test contract) but the function ignores any prior value.
        document_text: full document text passed into the cache-friendly
            prompt builder so vllm-chat's automatic prefix caching can
            reuse the doc-level KV cache across sibling chunks.
        document_metadata: dict carrying ``filename``, ``kb_name``,
            ``subtag_name``, ``document_date``, ``related_doc_titles``.
            Forwarded to ``build_contextualize_prompt`` — see that
            function for the optional-fields contract.
        Other args: forwarded to the chat call.

    Returns:
        The same list of chunk dicts, mutated in place. Each chunk has
        ``context_prefix`` set (str on success, None on failure) and its
        ``text`` updated to ``f"{prefix}\\n\\n{original_text}"`` on
        success (unchanged on failure).
    """
    import asyncio  # local import — only needed when contextualization is enabled

    if not chunks:
        return chunks

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(chunk: dict) -> None:
        original_text = chunk.get("text") or ""
        if not original_text:
            chunk["context_prefix"] = None
            return
        clipped = original_text[:_MAX_CHUNK_CHARS]
        messages = build_contextualize_prompt(
            document_text=document_text,
            chunk_text=clipped,
            document_metadata=document_metadata,
        )
        async with sem:
            try:
                content = await _chat_call(
                    messages,
                    chat_url=chat_url,
                    chat_model=chat_model,
                    api_key=api_key,
                    timeout_s=timeout_s,
                    transport=transport,
                )
            except Exception as e:  # noqa: BLE001 — fail-open per chunk
                log.debug("contextualize_with_prefix failed for chunk, keeping raw: %s", e)
                chunk["context_prefix"] = None
                return

        prefix = (content or "").strip()
        # Empty / oversized → fall open. Same guards as the legacy path.
        if not prefix or len(prefix) > _MAX_CONTEXT_CHARS:
            chunk["context_prefix"] = None
            return
        # Strip canonical echo prefixes the model emits despite instructions.
        for echo in _ECHO_PREFIXES:
            if prefix.lower().startswith(echo.lower()):
                prefix = prefix[len(echo):].strip()
                break
        if not prefix:
            chunk["context_prefix"] = None
            return

        chunk["context_prefix"] = prefix
        chunk["text"] = f"{prefix}\n\n{original_text}"

    await asyncio.gather(*(_one(c) for c in chunks))
    return chunks


__all__ = [
    "build_contextualize_prompt",
    "contextualize_chunk",
    "contextualize_batch",
    "contextualize_chunks_with_prefix",
    "is_enabled",
]
