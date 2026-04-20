"""Unit tests for ``ext.services.query_rewriter.rewrite_query``.

Uses ``httpx.MockTransport`` to stub the chat-completions endpoint so no
network is required. Covers the happy path plus every fail-open branch.
"""
from __future__ import annotations

import httpx
import pytest

from ext.services.query_rewriter import rewrite_query


CHAT_URL = "http://fake-vllm:8000/v1"
CHAT_MODEL = "orgchat-chat"


def _make_transport(handler):
    """Wrap a request handler into an ``httpx.MockTransport``."""
    return httpx.MockTransport(handler)


def _chat_response(content: str, status: int = 200) -> httpx.Response:
    """Build an OpenAI-compatible chat-completions response with ``content``."""
    if status != 200:
        return httpx.Response(status, json={"error": "server error"})
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": content}}
            ]
        },
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_rewrite_resolves_pronoun_with_history():
    """The canonical case: a vague follow-up becomes a self-contained query."""

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = req.content
        return _chat_response("Acme CRM pricing")

    history = [
        {"role": "user", "content": "I'm evaluating Acme CRM"},
        {"role": "assistant", "content": "Got it."},
    ]
    out = await rewrite_query(
        "what about pricing?",
        history,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key="sk-test",
        transport=_make_transport(handler),
    )
    assert out == "Acme CRM pricing"
    assert seen["url"].endswith("/chat/completions")


# ---------------------------------------------------------------------------
# Fail-open paths — each should return the raw latest turn unchanged
# ---------------------------------------------------------------------------

async def test_rewrite_empty_latest_returns_empty():
    """Empty input short-circuits before any network call."""

    def should_not_be_called(req: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be hit for empty input")

    out = await rewrite_query(
        "",
        [{"role": "user", "content": "prior"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(should_not_be_called),
    )
    assert out == ""


async def test_rewrite_network_5xx_falls_back_to_latest():
    """On HTTP 5xx the rewriter returns the raw turn."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("", status=503)

    latest = "what about pricing?"
    out = await rewrite_query(
        latest,
        [{"role": "user", "content": "prior"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == latest


async def test_rewrite_timeout_falls_back_to_latest():
    """A transport that raises a timeout is caught and falls open."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=req)

    latest = "what about pricing?"
    out = await rewrite_query(
        latest,
        [{"role": "user", "content": "prior"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == latest


async def test_rewrite_empty_model_response_falls_back():
    """Model returns empty string → we keep the raw turn."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("")

    latest = "what about pricing?"
    out = await rewrite_query(
        latest,
        [{"role": "user", "content": "prior"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == latest


async def test_rewrite_whitespace_only_response_falls_back():
    """Model returns only whitespace → raw turn, not a blank string."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("   \n  ")

    latest = "what is it?"
    out = await rewrite_query(
        latest,
        [{"role": "user", "content": "prior"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == latest


async def test_rewrite_oversized_response_falls_back():
    """A response >500 chars is treated as the model rambling → fall back."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("x" * 600)

    latest = "what is it?"
    out = await rewrite_query(
        latest,
        [{"role": "user", "content": "prior"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == latest


async def test_rewrite_malformed_json_falls_back():
    """Broken JSON schema (missing choices) is caught and fails open."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    latest = "what about pricing?"
    out = await rewrite_query(
        latest,
        [{"role": "user", "content": "prior"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == latest


# ---------------------------------------------------------------------------
# Echo-prefix / quote normalization
# ---------------------------------------------------------------------------

async def test_rewrite_strips_rewritten_query_prefix():
    """'Rewritten query: foo' echo-prefix is stripped."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("Rewritten query: Acme CRM pricing")

    out = await rewrite_query(
        "what about pricing?",
        [{"role": "user", "content": "Acme CRM"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == "Acme CRM pricing"


async def test_rewrite_strips_bare_rewritten_prefix():
    """'Rewritten: foo' prefix is also stripped."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("Rewritten: cost of March contract")

    out = await rewrite_query(
        "what is it?",
        [{"role": "user", "content": "March contract"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == "cost of March contract"


async def test_rewrite_strips_surrounding_double_quotes():
    """Quoted responses are unquoted."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response('"Acme CRM pricing"')

    out = await rewrite_query(
        "what about pricing?",
        [{"role": "user", "content": "Acme CRM"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == "Acme CRM pricing"


async def test_rewrite_strips_surrounding_single_quotes():
    """Single-quoted responses are unquoted."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("'Acme CRM pricing'")

    out = await rewrite_query(
        "what about pricing?",
        [{"role": "user", "content": "Acme CRM"}],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == "Acme CRM pricing"


# ---------------------------------------------------------------------------
# Edge cases around history
# ---------------------------------------------------------------------------

async def test_rewrite_with_empty_history_still_runs():
    """Caller may pass ``[]`` — rewrite still executes and uses what the model gives."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("what is my order status?")

    out = await rewrite_query(
        "what is my order status?",
        [],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == "what is my order status?"


async def test_rewrite_history_trimmed_to_last_six():
    """Only the last 6 history turns are sent — older ones are dropped."""

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.content.decode()
        return _chat_response("result")

    history = [
        {"role": "user", "content": f"turn-{i}"} for i in range(20)
    ]
    await rewrite_query(
        "latest",
        history,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    # Only the last 6 turns should appear in the prompt body.
    body = seen["body"]
    assert "turn-19" in body
    assert "turn-14" in body
    assert "turn-13" not in body  # 20 turns, last 6 = indices 14..19
    assert "turn-0" not in body


async def test_rewrite_sends_bearer_token_when_api_key_provided():
    """API key is forwarded as Authorization header."""

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        return _chat_response("ok")

    await rewrite_query(
        "hi",
        [],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key="sk-secret",
        transport=_make_transport(handler),
    )
    assert seen["auth"] == "Bearer sk-secret"


async def test_rewrite_omits_auth_header_when_no_api_key():
    """Without an API key, no Authorization header is sent."""

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        return _chat_response("ok")

    await rewrite_query(
        "hi",
        [],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key=None,
        transport=_make_transport(handler),
    )
    assert seen["auth"] is None
