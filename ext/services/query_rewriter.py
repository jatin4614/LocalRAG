"""History-aware query rewriter (P0.5).

Rewrites the latest user turn into a self-contained search query using the
chat model. Resolves pronouns and domain references so that downstream
retrieval can match against chunks even when the raw turn ("what about the
pricing?") depends on prior context.

Fails open: on ANY failure (timeout, network, 5xx, empty response, schema
mismatch) returns the original latest_turn unchanged. Rewriting is a best-
effort optimization, never a hard dependency.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from .llm_telemetry import record_llm_call
from .obs import inject_context_into_headers, span

log = logging.getLogger("orgchat.rewrite")


_REWRITE_PROMPT = """You are a search-query rewriter. Given a conversation history and the user's latest turn, rewrite the latest turn into a SELF-CONTAINED search query that a search engine could answer without seeing the history.

Rules:
- Resolve pronouns and references ("it", "that", "the thing we discussed")
- Include domain/entity context from earlier turns if needed
- Output ONLY the rewritten query on a single line, no preamble, no quotes, no explanation
- If the latest turn is already self-contained, output it verbatim
- Maximum 40 words

Conversation history:
{history}

Latest user turn:
{latest}

Rewritten standalone query:"""


async def rewrite_query(
    latest_turn: str,
    history: list[dict],
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout_s: float = 8.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> str:
    """Rewrite ``latest_turn`` into a standalone search query using the chat model.

    Args:
        latest_turn: the user's most recent message (raw text).
        history: prior turns as a list of dicts with ``role`` and ``content``.
            Only the last ~6 are used; older turns are trimmed to cap context.
        chat_url: base URL of an OpenAI-compatible endpoint
            (e.g. ``http://orgchat-vllm-chat:8000/v1``).
        chat_model: model name the endpoint expects (e.g. ``orgchat-chat``).
        api_key: optional bearer token. If None, no Authorization header is sent.
        timeout_s: request timeout (seconds).
        transport: optional httpx transport, primarily for tests.

    Returns:
        The rewritten standalone query. On any error, returns ``latest_turn``
        unchanged — rewriting is best-effort.
    """
    if not latest_turn:
        return latest_turn

    with span("query.rewrite", model=chat_model, history_turns=len(history or [])):
        return await _rewrite_query_impl(
            latest_turn=latest_turn,
            history=history,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            timeout_s=timeout_s,
            transport=transport,
        )


async def _rewrite_query_impl(
    latest_turn: str,
    history: list[dict],
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout_s: float = 8.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> str:
    # Trim history to last 6 turns to cap the prompt size. Each content is
    # clipped to 500 chars so a single giant turn can't blow up the request.
    trimmed = history[-6:] if history else []
    if trimmed:
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {(m.get('content') or '').strip()[:500]}"
            for m in trimmed
        )
    else:
        history_text = "(no prior turns)"

    body = {
        "model": chat_model,
        "messages": [
            {
                "role": "user",
                "content": _REWRITE_PROMPT.format(
                    history=history_text, latest=latest_turn
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 80,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers = inject_context_into_headers(headers)

    url = f"{chat_url.rstrip('/')}/chat/completions"
    model_name = body.get("model") or os.environ.get("CHAT_MODEL", "unknown")
    try:
        # ``record_llm_call`` wraps the HTTP call so prompt/completion token
        # counters fire even on retry-exhausted failure — operators get the
        # rewriter's token spend in the same dashboard as contextualizer /
        # hyde. Fail-open semantics are preserved by the outer try/except.
        async with record_llm_call(stage="rewriter", model=model_name) as rec:
            async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
                r = await client.post(url, json=body, headers=headers)
                r.raise_for_status()
                data = r.json()
            usage = data.get("usage") or {}
            rec.set_tokens(
                prompt=usage.get("prompt_tokens", 0),
                completion=usage.get("completion_tokens", 0),
            )
        rewrite = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:  # noqa: BLE001 — fail-open by design
        log.debug("query rewrite failed, falling back to raw turn: %s", e)
        return latest_turn

    # Sanity guards: empty → fallback, oversized → fallback (model rambled).
    if not rewrite or len(rewrite) > 500:
        return latest_turn

    # Strip common echo prefixes ("Rewritten: ...", "Query: ...") and quotes.
    for prefix in ("Rewritten query:", "Rewritten:", "Query:", "Standalone query:"):
        if rewrite.lower().startswith(prefix.lower()):
            rewrite = rewrite[len(prefix):].strip()
            break
    rewrite = rewrite.strip('"').strip("'").strip()

    return rewrite or latest_turn
