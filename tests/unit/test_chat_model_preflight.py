"""Unit tests for the CHAT_MODEL preflight (review §6.7).

The preflight asserts that ``CHAT_MODEL`` env var is in the ``/v1/models``
response from ``OPENAI_API_BASE_URL`` (or ``RAG_VISION_URL`` if explicit).
On mismatch: log WARNING + bump ``chat_model_mismatch_total`` counter.
Don't crash — operators may use aliases the endpoint resolves transparently.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import httpx
import pytest


def test_chat_model_preflight_warns_on_mismatch(monkeypatch, caplog):
    """If ``/v1/models`` response doesn't list ``CHAT_MODEL``, log WARNING."""
    import logging

    from ext.services import chat_model_preflight

    monkeypatch.setenv("CHAT_MODEL", "gemma-4-not-served")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://fake-vllm:8000/v1")

    def _handler(req: httpx.Request) -> httpx.Response:
        # Fake /v1/models payload that does NOT contain CHAT_MODEL.
        return httpx.Response(
            200,
            json={"data": [{"id": "qwen3-4b-qu"}, {"id": "different-model"}]},
        )

    transport = httpx.MockTransport(_handler)
    caplog.set_level(logging.WARNING)
    counter = MagicMock()
    monkeypatch.setattr(
        chat_model_preflight, "chat_model_mismatch_total", counter
    )
    chat_model_preflight.preflight_chat_model(transport=transport)

    # Counter bumped exactly once
    counter.inc.assert_called_once()
    # WARNING logged with model + endpoint
    assert any(
        "gemma-4-not-served" in rec.message and "fake-vllm" in rec.message
        for rec in caplog.records
    ), [r.message for r in caplog.records]


def test_chat_model_preflight_silent_on_match(monkeypatch, caplog):
    """When ``/v1/models`` lists ``CHAT_MODEL``, no warning, no counter bump."""
    import logging

    from ext.services import chat_model_preflight

    monkeypatch.setenv("CHAT_MODEL", "gemma-4")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://fake-vllm:8000/v1")

    def _handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "gemma-4"}, {"id": "qwen3-4b-qu"}]},
        )

    transport = httpx.MockTransport(_handler)
    caplog.set_level(logging.WARNING)
    counter = MagicMock()
    monkeypatch.setattr(
        chat_model_preflight, "chat_model_mismatch_total", counter
    )
    chat_model_preflight.preflight_chat_model(transport=transport)

    counter.inc.assert_not_called()
    mismatch_msgs = [
        rec.message for rec in caplog.records
        if "mismatch" in rec.message.lower() or "not in" in rec.message.lower()
    ]
    assert not mismatch_msgs


def test_chat_model_preflight_does_not_crash_on_endpoint_failure(monkeypatch, caplog):
    """If ``/v1/models`` is unreachable (vLLM still booting), log + return.

    Operators commonly run preflight while vllm-chat is still warming up;
    a hard crash here would prevent the open-webui container from coming
    up, which is worse than a noisy warning.
    """
    import logging

    from ext.services import chat_model_preflight

    monkeypatch.setenv("CHAT_MODEL", "gemma-4")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://fake-vllm:8000/v1")

    def _handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=req)

    transport = httpx.MockTransport(_handler)
    caplog.set_level(logging.WARNING)
    # Must not raise
    chat_model_preflight.preflight_chat_model(transport=transport)


def test_chat_model_preflight_skips_when_chat_model_unset(monkeypatch):
    """No CHAT_MODEL env → nothing to validate, return cleanly."""
    from ext.services import chat_model_preflight

    monkeypatch.delenv("CHAT_MODEL", raising=False)

    def _handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be called")

    transport = httpx.MockTransport(_handler)
    chat_model_preflight.preflight_chat_model(transport=transport)


def test_chat_model_mismatch_counter_exists():
    from ext.services import metrics
    assert hasattr(metrics, "chat_model_mismatch_total")
