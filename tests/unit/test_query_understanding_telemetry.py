"""Unit tests for ``ext.services.query_understanding`` LLM telemetry wiring.

The QU LLM fires on every chat turn when ``RAG_QU_ENABLED=1``. Bug-fix
campaign §6.3 wires it through ``record_llm_call`` so the prompt-token
spend lands in ``rag_tokens_prompt_total`` instead of being invisible.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
import pytest

from ext.services import query_understanding as qu_mod


_VALID_RESPONSE = {
    "intent": "specific",
    "resolved_query": "what is the policy",
    "temporal_constraint": None,
    "entities": [],
    "confidence": 0.9,
}


@pytest.mark.asyncio
async def test_analyze_query_records_llm_call(monkeypatch):
    """A successful QU LLM call must invoke ``record_llm_call`` exactly once."""
    calls: list[dict] = []

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

    monkeypatch.setattr(qu_mod, "record_llm_call", _fake_record_llm_call)

    def _handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(_VALID_RESPONSE)}}
                ],
                "usage": {"prompt_tokens": 333, "completion_tokens": 22},
            },
        )

    # Patch httpx.AsyncClient to use a MockTransport with our handler.
    # analyze_query constructs its own client without a transport kwarg, so
    # we need a tighter monkeypatch than passing transport=...
    real_async_client = httpx.AsyncClient

    class _PatchedAsyncClient(real_async_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(_handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(qu_mod.httpx, "AsyncClient", _PatchedAsyncClient)

    out = await qu_mod.analyze_query(
        "what is the policy",
        history=[],
        qu_url="http://fake-qu:8000/v1",
        model="qwen3-4b-qu",
        timeout_ms=2000,
    )
    assert out is not None
    assert out.intent == "specific"
    assert len(calls) == 1
    assert calls[0]["stage"] == "query_understanding"
    assert calls[0]["model"] == "qwen3-4b-qu"
    assert calls[0]["prompt"] == 333
    assert calls[0]["completion"] == 22
