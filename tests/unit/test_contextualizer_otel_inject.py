"""Tests for §7.6 — OTel context propagation in `contextualizer.py`.

Pre-fix: the chat-LLM POSTs at lines 245 and 357 of `contextualizer.py`
built `headers = {...}` then called `client.post(url, headers=headers)`
WITHOUT injecting the W3C `traceparent` header. Distributed traces
broke at the contextualizer → vllm-chat boundary.

Mirror the pattern already in `query_rewriter.py:123`,
`doc_summarizer.py:128`, and `hyde.py:115`:

    headers = inject_context_into_headers(headers)

Test approach: monkeypatch `obs.inject_context_into_headers` to stamp a
sentinel header. The sentinel must appear on every chat-LLM POST.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from ext.services import contextualizer as ctx_mod


def _chat_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


@pytest.fixture
def stamped_inject(monkeypatch):
    """Replace `inject_context_into_headers` with a sentinel-stamping shim.

    Patches both the obs module and the contextualizer module's reference
    (because `from .obs import inject_context_into_headers` binds at import).
    """
    sentinel_value = "test-trace-id-1234567890abcdef"

    def _shim(headers: dict[str, Any] | None = None) -> dict[str, Any]:
        out = dict(headers or {})
        out["traceparent"] = f"00-{sentinel_value}-deadbeefcafebabe-01"
        return out

    from ext.services import obs as _obs
    monkeypatch.setattr(_obs, "inject_context_into_headers", _shim)
    # contextualizer.py uses `from .obs import inject_context_into_headers`
    # which binds the name into its own module — patch that binding too,
    # if it exists. (Test will fail-with-clear-message if the import is
    # missing, which IS the bug we're fixing.)
    if hasattr(ctx_mod, "inject_context_into_headers"):
        monkeypatch.setattr(ctx_mod, "inject_context_into_headers", _shim)
    return sentinel_value


@pytest.mark.asyncio
async def test_contextualize_chunk_injects_otel_traceparent(stamped_inject):
    """The legacy `contextualize_chunk` path (line 245) must inject traceparent."""
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["traceparent"] = req.headers.get("traceparent")
        return _chat_response("situated context")

    transport = httpx.MockTransport(handler)
    out = await ctx_mod.contextualize_chunk(
        "chunk text",
        doc_title="Doc Title",
        chat_url="http://chat.test",
        chat_model="m",
        api_key="key",
        timeout_s=5.0,
        transport=transport,
    )
    # Sanity: the call went through.
    assert isinstance(out, str)
    # The headline assertion: traceparent reached the chat endpoint.
    assert seen.get("traceparent") is not None, (
        "traceparent header was NOT injected — OTel context not propagated to chat-LLM"
    )
    assert stamped_inject in seen["traceparent"]


@pytest.mark.asyncio
async def test_chat_call_seam_injects_otel_traceparent(stamped_inject):
    """The newer `_chat_call` seam (line 357) — used by
    `contextualize_chunks_with_prefix` — must also inject traceparent.
    """
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["traceparent"] = req.headers.get("traceparent")
        return _chat_response("ctx")

    transport = httpx.MockTransport(handler)

    # `_chat_call` is internal but module-public-ish (test seam).
    out = await ctx_mod._chat_call(
        messages=[{"role": "user", "content": "hi"}],
        chat_url="http://chat.test",
        chat_model="m",
        api_key="key",
        max_tokens=64,
        timeout_s=5.0,
        transport=transport,
    )
    assert isinstance(out, str)
    assert seen.get("traceparent") is not None, (
        "_chat_call seam did NOT inject traceparent — OTel context broken for "
        "contextualize_chunks_with_prefix path"
    )
    assert stamped_inject in seen["traceparent"]


# ---------------------------------------------------------------------------
# Regression guard: hyde.py also injects (was added in commit 3b10c39 but
# the §7.6 review re-listed it; ensure no future refactor strips it).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hyde_generate_one_injects_otel_traceparent(monkeypatch):
    """`hyde._generate_one` must inject traceparent before POSTing.

    Pinned here because the §7.6 review listed `hyde.py:114` alongside
    contextualizer; if a future cleanup removes the inject from hyde,
    this test catches it.
    """
    sentinel_value = "test-trace-id-hyde0"

    def _shim(headers=None):
        out = dict(headers or {})
        out["traceparent"] = f"00-{sentinel_value}-deadbeefcafebabe-01"
        return out

    from ext.services import hyde as _hyde_mod
    from ext.services import obs as _obs_mod
    monkeypatch.setattr(_obs_mod, "inject_context_into_headers", _shim)
    if hasattr(_hyde_mod, "inject_context_into_headers"):
        monkeypatch.setattr(_hyde_mod, "inject_context_into_headers", _shim)

    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["traceparent"] = req.headers.get("traceparent")
        return _chat_response("excerpt")

    transport = httpx.MockTransport(handler)
    out = await _hyde_mod._generate_one(
        "what is X?",
        chat_url="http://chat.test",
        chat_model="m",
        api_key="key",
        timeout_s=5.0,
        transport=transport,
    )
    assert isinstance(out, str)
    assert seen.get("traceparent") is not None, (
        "hyde._generate_one stopped injecting traceparent — regression on §7.6"
    )
    assert sentinel_value in seen["traceparent"]
