"""Unit tests for ``ext.services.doc_summarizer`` LLM telemetry wiring.

The summarizer is the heaviest LLM ingest cost and previously did NOT
record token spend through ``record_llm_call``. Bug-fix campaign §6.3
wires it in. These tests assert that the wrapper fires.
"""
from __future__ import annotations

import httpx
import pytest

from ext.services.doc_summarizer import summarize_document


CHAT_URL = "http://fake-vllm:8000/v1"
CHAT_MODEL = "orgchat-chat"


def _chat_response(content: str, *, status: int = 200, usage: dict | None = None) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, json={"error": "server error"})
    body: dict = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


@pytest.mark.asyncio
async def test_doc_summarizer_records_llm_call(monkeypatch):
    """``summarize_document`` must wrap its chat POST in ``record_llm_call``.

    We monkeypatch the telemetry recorder to a list collector and assert
    exactly one call was recorded with stage="doc_summarizer" and the
    expected token counts from the response body.
    """
    calls: list[dict] = []

    from contextlib import asynccontextmanager
    from ext.services import doc_summarizer as ds_mod

    @asynccontextmanager
    async def _fake_record_llm_call(*, stage: str, model: str, kb=None):
        rec = type(
            "FakeRec",
            (),
            {
                "set_tokens": lambda self, *, prompt, completion: calls.append(
                    {
                        "stage": stage,
                        "model": model,
                        "kb": kb,
                        "prompt": prompt,
                        "completion": completion,
                    }
                ),
            },
        )()
        yield rec

    monkeypatch.setattr(ds_mod, "record_llm_call", _fake_record_llm_call)

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response(
            "summary text",
            usage={"prompt_tokens": 1234, "completion_tokens": 56},
        )

    out = await summarize_document(
        chunks=["body"],
        filename="x.pdf",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(handler),
    )
    assert out
    assert len(calls) == 1
    assert calls[0]["stage"] == "doc_summarizer"
    assert calls[0]["model"] == CHAT_MODEL
    assert calls[0]["prompt"] == 1234
    assert calls[0]["completion"] == 56


@pytest.mark.asyncio
async def test_doc_summarizer_records_llm_call_even_on_failure(monkeypatch):
    """The recorder context still wraps the call when the upstream errors.

    Fail-open semantics in ``summarize_document`` (returns "" on error)
    must NOT bypass the recorder context manager — the prompt-token
    spend still happened on the LLM side and operators need to see it.
    """
    entries: list[str] = []

    from contextlib import asynccontextmanager
    from ext.services import doc_summarizer as ds_mod

    @asynccontextmanager
    async def _fake_record_llm_call(*, stage: str, model: str, kb=None):
        entries.append("entered")
        try:
            yield type("FakeRec", (), {"set_tokens": lambda self, *, prompt, completion: None})()
        finally:
            entries.append("exited")

    monkeypatch.setattr(ds_mod, "record_llm_call", _fake_record_llm_call)

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("", status=500)

    out = await summarize_document(
        chunks=["body"],
        filename="x.pdf",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(handler),
    )
    assert out == ""
    # Recorder context entered + exited around the failing call
    assert entries == ["entered", "exited"]
