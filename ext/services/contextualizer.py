"""Contextual chunk augmentation via chat model (P2.7).

For each chunk, call the chat endpoint with a "situate this chunk in the
document" prompt. The short LLM-generated context is prepended to the
chunk text before embedding, following Anthropic's Contextual Retrieval
recipe (https://www.anthropic.com/news/contextual-retrieval).

Design choices:
- **Fail-open.** Any error (timeout, 5xx, malformed JSON, oversized reply)
  returns the original chunk unchanged. A bad context pass is never worse
  than no context pass, so we never crash ingest over it.
- **Per-chunk, not batched.** Anthropic's pattern is one call per chunk.
  vLLM's prefix cache will reuse the shared ``{doc_title}``/preamble
  tokens across calls for the same document, so batching them into a
  single request would only obscure the per-chunk failure mode.
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
from typing import Iterable, Optional

import httpx


log = logging.getLogger("orgchat.contextualize")


# Adapted from Anthropic's Contextual Retrieval blog post. The trailing
# instruction ("Answer ONLY ...") is load-bearing: without it Qwen2.5
# sometimes prepends "Sure! Here's the context: ..." which we'd rather
# avoid even though we strip common echo prefixes below.
_PROMPT = """<document>
{doc_title}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short, succinct context to situate this chunk within the overall document, so that retrieval of this chunk works better. Answer ONLY with the succinct context - no preamble, no quotes, no apologies. Under 50 words."""


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

    body = {
        "model": chat_model,
        "messages": [
            {
                "role": "user",
                "content": _PROMPT.format(
                    doc_title=(doc_title or "Untitled"),
                    chunk_text=chunk_text[:_MAX_CHUNK_CHARS],
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 96,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{chat_url.rstrip('/')}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            r = await client.post(url, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        context = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:  # noqa: BLE001 — fail-open by design
        log.debug("contextualize failed, keeping raw chunk: %s", e)
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


__all__ = [
    "contextualize_chunk",
    "contextualize_batch",
    "is_enabled",
]
