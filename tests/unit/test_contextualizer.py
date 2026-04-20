"""Unit tests for ``ext.services.contextualizer`` (P2.7).

Stubs the chat-completions endpoint via ``httpx.MockTransport`` so no
network is required. Covers the happy path, every fail-open branch, and
the prefix-strip / oversize guards.
"""
from __future__ import annotations

import httpx
import pytest

from ext.services.contextualizer import (
    contextualize_batch,
    contextualize_chunk,
    is_enabled,
)


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
# is_enabled — env var gate
# ---------------------------------------------------------------------------

def test_is_enabled_default_off(monkeypatch):
    """Default state: flag absent → disabled."""
    monkeypatch.delenv("RAG_CONTEXTUALIZE_KBS", raising=False)
    assert is_enabled() is False


def test_is_enabled_when_flag_set_to_1(monkeypatch):
    monkeypatch.setenv("RAG_CONTEXTUALIZE_KBS", "1")
    assert is_enabled() is True


def test_is_enabled_ignores_other_truthy_values(monkeypatch):
    """Only the literal ``"1"`` flips the flag — be conservative."""
    monkeypatch.setenv("RAG_CONTEXTUALIZE_KBS", "true")
    assert is_enabled() is False
    monkeypatch.setenv("RAG_CONTEXTUALIZE_KBS", "yes")
    assert is_enabled() is False
    monkeypatch.setenv("RAG_CONTEXTUALIZE_KBS", "0")
    assert is_enabled() is False


# ---------------------------------------------------------------------------
# contextualize_chunk — happy path
# ---------------------------------------------------------------------------

async def test_contextualize_happy_path_prepends_context():
    """Standard case: chat returns a situating sentence, prepended to the chunk."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = req.content.decode()
        return _chat_response("This chunk is about pricing in the intro section.")

    out = await contextualize_chunk(
        "The monthly plan costs $99.",
        doc_title="pricing.md",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key="sk-test",
        transport=_make_transport(handler),
    )
    assert out == (
        "This chunk is about pricing in the intro section.\n\n"
        "The monthly plan costs $99."
    )
    assert seen["url"].endswith("/chat/completions")
    # Doc title and chunk text both flow into the prompt body.
    assert "pricing.md" in seen["body"]
    assert "The monthly plan costs $99." in seen["body"]


async def test_contextualize_forwards_bearer_token():
    """API key is forwarded as Authorization header."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        return _chat_response("context")

    await contextualize_chunk(
        "chunk",
        doc_title="doc",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key="sk-secret",
        transport=_make_transport(handler),
    )
    assert seen["auth"] == "Bearer sk-secret"


async def test_contextualize_omits_auth_when_no_api_key():
    """Without an API key, no Authorization header is sent."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        return _chat_response("context")

    await contextualize_chunk(
        "chunk",
        doc_title="doc",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key=None,
        transport=_make_transport(handler),
    )
    assert seen["auth"] is None


# ---------------------------------------------------------------------------
# Fail-open branches — each returns the raw chunk unchanged
# ---------------------------------------------------------------------------

async def test_contextualize_empty_chunk_short_circuits():
    """Empty input never hits the network."""

    def should_not_be_called(req: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be hit for empty input")

    out = await contextualize_chunk(
        "",
        doc_title="doc",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(should_not_be_called),
    )
    assert out == ""


async def test_contextualize_timeout_returns_original_chunk():
    """A transport that raises a timeout falls back to the raw chunk."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=req)

    raw = "The monthly plan costs $99."
    out = await contextualize_chunk(
        raw,
        doc_title="pricing.md",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == raw


async def test_contextualize_5xx_returns_original_chunk():
    """HTTP 5xx → raw chunk."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("", status=503)

    raw = "some chunk text"
    out = await contextualize_chunk(
        raw,
        doc_title="d",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == raw


async def test_contextualize_empty_response_returns_original():
    """Model returns empty string → raw chunk (no stray ``\\n\\n`` prefix)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("")

    raw = "some chunk text"
    out = await contextualize_chunk(
        raw,
        doc_title="d",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == raw


async def test_contextualize_whitespace_only_response_returns_original():
    """Whitespace-only reply is treated as empty."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("   \n\t  ")

    raw = "some chunk text"
    out = await contextualize_chunk(
        raw,
        doc_title="d",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == raw


async def test_contextualize_malformed_json_returns_original():
    """Broken JSON schema (missing ``choices``) falls open."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    raw = "some chunk text"
    out = await contextualize_chunk(
        raw,
        doc_title="d",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == raw


async def test_contextualize_oversized_response_returns_original():
    """A response > 800 chars means the model rambled → raw chunk."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("x" * 801)

    raw = "some chunk text"
    out = await contextualize_chunk(
        raw,
        doc_title="d",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == raw


# ---------------------------------------------------------------------------
# Prefix strip — canonical echo patterns
# ---------------------------------------------------------------------------

async def test_contextualize_strips_context_prefix():
    """'Context: ...' echo prefix is stripped."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("Context: About the intro section's pricing tier.")

    out = await contextualize_chunk(
        "The monthly plan costs $99.",
        doc_title="pricing.md",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out.startswith("About the intro section's pricing tier.")
    assert "Context:" not in out.split("\n\n")[0]


async def test_contextualize_strips_situated_context_prefix():
    """'Situated context: ...' is also stripped."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("Situated context: A pricing paragraph.")

    out = await contextualize_chunk(
        "The monthly plan costs $99.",
        doc_title="pricing.md",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out.startswith("A pricing paragraph.")


async def test_contextualize_strips_prefix_case_insensitive():
    """Prefix match is case-insensitive."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("context: shouted prefix.")

    out = await contextualize_chunk(
        "chunk",
        doc_title="d",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out.startswith("shouted prefix.")


async def test_contextualize_prefix_only_response_returns_original():
    """If stripping a prefix leaves only whitespace, fall open."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("Context:    ")

    raw = "chunk"
    out = await contextualize_chunk(
        raw,
        doc_title="d",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == raw


# ---------------------------------------------------------------------------
# contextualize_batch — parallel / ordering / partial failure
# ---------------------------------------------------------------------------

async def test_contextualize_batch_preserves_order():
    """Output order matches input order even under concurrent execution."""
    # Return a deterministic, index-linked context per call so we can
    # verify ordering without relying on call arrival order.
    def handler(req: httpx.Request) -> httpx.Response:
        body = req.content.decode()
        # Encode chunk-identity in the response: which chunk are we asked about?
        for i in range(5):
            if f"chunk-{i}" in body:
                return _chat_response(f"context-{i}")
        return _chat_response("context-unknown")

    pairs = [(f"chunk-{i}", "doc") for i in range(5)]
    out = await contextualize_batch(
        pairs,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        concurrency=3,
        transport=_make_transport(handler),
    )
    assert len(out) == 5
    for i in range(5):
        assert out[i] == f"context-{i}\n\nchunk-{i}"


async def test_contextualize_batch_partial_failure_falls_open_per_chunk():
    """One bad chunk does not poison siblings."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = req.content.decode()
        if "chunk-2" in body:
            return _chat_response("", status=500)  # kill chunk 2 only
        return _chat_response("ctx")

    pairs = [(f"chunk-{i}", "doc") for i in range(4)]
    out = await contextualize_batch(
        pairs,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        concurrency=2,
        transport=_make_transport(handler),
    )
    assert out[0].startswith("ctx\n\n")
    assert out[1].startswith("ctx\n\n")
    assert out[2] == "chunk-2"  # fell open
    assert out[3].startswith("ctx\n\n")


async def test_contextualize_batch_empty_input_returns_empty_list():
    """Empty iterable → empty list, no transport hit."""

    def should_not_be_called(req: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be hit for empty batch")

    out = await contextualize_batch(
        [],
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(should_not_be_called),
    )
    assert out == []
