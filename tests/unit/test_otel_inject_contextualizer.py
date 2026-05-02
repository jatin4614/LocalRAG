"""Tests for §7.6 — OTel context inject for the contextualizer outbound POST.

Both ``contextualize_chunk`` (legacy entry) and ``_chat_call`` (the
``contextualize_chunks_with_prefix`` path) build their own ``headers`` dict
and then POST to the chat endpoint via ``_contextualize_call``. Without the
``inject_context_into_headers`` call, the W3C ``traceparent`` is missing on
the outbound request and Jaeger can't correlate the contextualizer span with
the chat-server span — observability dead-end.

This test monkey-patches ``inject_context_into_headers`` to count calls,
then drives both contextualizer entry points and asserts injection happened
before the POST.
"""
from __future__ import annotations

import importlib
import sys

import pytest


def _fresh_ctx():
    """Force a clean re-import so we get the live module object that test
    code can monkey-patch. ``test_ingest_contextualize_flag.py`` does
    ``sys.modules.pop`` between tests, so a stale module reference here
    would silently miss our patches.
    """
    sys.modules.pop("ext.services.contextualizer", None)
    return importlib.import_module("ext.services.contextualizer")


@pytest.mark.asyncio
async def test_contextualize_chunk_injects_traceparent(monkeypatch):
    """``contextualize_chunk`` calls ``inject_context_into_headers`` before
    handing the dict to ``_contextualize_call``."""
    ctx = _fresh_ctx()
    inject_calls: list[dict] = []

    def _fake_inject(headers):
        inject_calls.append(dict(headers))
        out = dict(headers)
        out["traceparent"] = "00-fake-fake-01"
        return out

    monkeypatch.setattr(ctx, "inject_context_into_headers", _fake_inject, raising=True)

    captured_headers: list[dict] = []

    async def _fake_call(url, body, headers, timeout_s, transport):
        captured_headers.append(dict(headers))
        return {"choices": [{"message": {"content": "context line"}}]}

    monkeypatch.setattr(ctx, "_contextualize_call", _fake_call, raising=True)

    out = await ctx.contextualize_chunk(
        "the chunk text",
        "some-doc",
        chat_url="http://test/v1",
        chat_model="m",
        api_key="abc",
    )
    assert isinstance(out, str)
    assert len(inject_calls) == 1
    # The injected traceparent must reach the POST.
    assert captured_headers[0].get("traceparent") == "00-fake-fake-01"


@pytest.mark.asyncio
async def test_chat_call_injects_traceparent(monkeypatch):
    """The lower-level ``_chat_call`` (used by the per-chunk-with-prefix path)
    also injects."""
    ctx = _fresh_ctx()
    inject_calls: list[dict] = []

    def _fake_inject(headers):
        inject_calls.append(dict(headers))
        out = dict(headers)
        out["traceparent"] = "00-second-fake-01"
        return out

    monkeypatch.setattr(ctx, "inject_context_into_headers", _fake_inject, raising=True)

    captured_headers: list[dict] = []

    async def _fake_call(url, body, headers, timeout_s, transport):
        captured_headers.append(dict(headers))
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(ctx, "_contextualize_call", _fake_call, raising=True)

    out = await ctx._chat_call(
        messages=[{"role": "user", "content": "hi"}],
        chat_url="http://test/v1",
        chat_model="m",
        api_key="abc",
    )
    assert isinstance(out, str)
    assert len(inject_calls) == 1
    assert captured_headers[0].get("traceparent") == "00-second-fake-01"


def test_contextualizer_imports_inject_helper():
    """Sanity: the helper is importable into the module so monkeypatching it
    via the module path works in production wiring too."""
    ctx = _fresh_ctx()
    assert hasattr(ctx, "inject_context_into_headers")
