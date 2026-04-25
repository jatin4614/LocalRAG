"""Unit tests for ``ext.services.doc_summarizer.summarize_document``.

Uses ``httpx.MockTransport`` to stub the chat-completions endpoint so no
network is required. Covers the happy path + every fail-open branch.
"""
from __future__ import annotations

import httpx
import pytest

from ext.services.doc_summarizer import summarize_document


CHAT_URL = "http://fake-vllm:8000/v1"
CHAT_MODEL = "orgchat-chat"


def _chat_response(content: str, status: int = 200) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, json={"error": "server error"})
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------

async def test_summarizes_happy_path() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = req.content.decode()
        return _chat_response(
            "Q1-report.pdf covers Acme's Q1 2026 revenue (up 12% YoY), "
            "key risks in supply chain, and hiring plans for EMEA. "
            "Signed by CFO on 2026-04-02."
        )

    summary = await summarize_document(
        chunks=["First chunk of the Q1 report.", "Second chunk talks revenue."],
        filename="Q1-report.pdf",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
    )
    assert "Q1-report.pdf" in summary
    assert seen["url"].endswith("/chat/completions")
    # Filename is embedded in the prompt.
    assert "Q1-report.pdf" in seen["body"]


async def test_strips_summary_echo_prefix() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("Summary: a concise wrap-up sentence.")

    out = await summarize_document(
        chunks=["text"],
        filename="doc.pdf",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(handler),
    )
    assert out == "a concise wrap-up sentence."


async def test_forwards_bearer_token_when_api_key_set() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        return _chat_response("ok")

    await summarize_document(
        chunks=["text"],
        filename="doc.pdf",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key="sk-secret",
        transport=httpx.MockTransport(handler),
    )
    assert seen["auth"] == "Bearer sk-secret"


# --------------------------------------------------------------------------
# Fail-open paths — each must return "" (empty string), never raise
# --------------------------------------------------------------------------

async def test_empty_chunks_returns_empty() -> None:
    # No transport hit — short-circuits before building a body.
    def should_not_be_called(req: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be called for empty chunks")

    out = await summarize_document(
        chunks=[],
        filename="x",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(should_not_be_called),
    )
    assert out == ""


async def test_http_500_returns_empty() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("", status=500)

    out = await summarize_document(
        chunks=["text"],
        filename="doc.pdf",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(handler),
    )
    assert out == ""


async def test_timeout_returns_empty() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=req)

    out = await summarize_document(
        chunks=["text"],
        filename="doc.pdf",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(handler),
    )
    assert out == ""


async def test_malformed_json_returns_empty() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    out = await summarize_document(
        chunks=["text"],
        filename="doc.pdf",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(handler),
    )
    assert out == ""


async def test_empty_model_response_returns_empty() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("")

    out = await summarize_document(
        chunks=["text"],
        filename="doc.pdf",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(handler),
    )
    assert out == ""
