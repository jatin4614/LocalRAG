"""Tests for §5.15 — Total pipeline timeout.

When ``_run_pipeline`` exceeds ``RAG_TOTAL_BUDGET_SEC`` (default 30s),
``retrieve_kb_sources`` catches ``TimeoutError``, increments
``rag_pipeline_timeout_total{intent}``, and returns an early-degraded
source list (datetime preamble only — catalog needs DB which may be the
cause of the hang).
"""
from __future__ import annotations

import asyncio

import pytest

from ext.services import chat_rag_bridge as bridge


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def execute(self, *a, **kw):  # pragma: no cover
        class _R:
            def first(self):
                return None

            def all(self):
                return []

        return _R()


def _fake_sessionmaker():
    return _FakeSession()


def _wire_min_bridge(monkeypatch):
    bridge.configure(
        vector_store=object(),
        embedder=object(),
        sessionmaker=_fake_sessionmaker,
    )

    async def _fake_allowed(session, *, user_id):  # noqa: ARG001
        return [1]

    import ext.services.rbac as _rbac
    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)


@pytest.mark.asyncio
async def test_pipeline_timeout_returns_degraded_sources(monkeypatch):
    """When _run_pipeline hangs longer than RAG_TOTAL_BUDGET_SEC, the bridge
    returns a degraded source list rather than blocking the chat indefinitely.
    """
    _wire_min_bridge(monkeypatch)
    # 1-second budget so the test runs fast.
    monkeypatch.setenv("RAG_TOTAL_BUDGET_SEC", "1")
    # Production sets RAG_INJECT_DATETIME=1; conftest sets it to 0 to avoid
    # noise in unrelated tests. Restore the production default here so the
    # degraded fallback can actually emit the preamble it promises.
    monkeypatch.setenv("RAG_INJECT_DATETIME", "1")

    async def _slow_pipeline(*args, **kwargs):
        await asyncio.sleep(60)  # would-be deadlocked pipeline
        return []  # never reached

    monkeypatch.setattr(bridge, "_run_pipeline", _slow_pipeline, raising=True)

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="anything",
        user_id="user-1",
    )

    # Degraded but non-empty — at minimum the datetime preamble is present.
    assert isinstance(out, list)
    assert len(out) >= 1
    # The datetime preamble carries source name "current-datetime".
    assert any(
        src.get("source", {}).get("name") == "current-datetime"
        for src in out
    )


@pytest.mark.asyncio
async def test_pipeline_timeout_returns_empty_when_datetime_disabled(monkeypatch):
    """When RAG_INJECT_DATETIME=0, the timeout fallback returns [] (no
    degraded source). This honors the operator's explicit "no preamble" choice.
    """
    _wire_min_bridge(monkeypatch)
    monkeypatch.setenv("RAG_TOTAL_BUDGET_SEC", "1")
    monkeypatch.setenv("RAG_INJECT_DATETIME", "0")

    async def _slow_pipeline(*args, **kwargs):
        await asyncio.sleep(60)
        return []

    monkeypatch.setattr(bridge, "_run_pipeline", _slow_pipeline, raising=True)

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="q",
        user_id="user-1",
    )
    assert out == []


@pytest.mark.asyncio
async def test_pipeline_timeout_increments_counter(monkeypatch):
    """The ``rag_pipeline_timeout_total{intent}`` counter ticks on timeout."""
    _wire_min_bridge(monkeypatch)
    monkeypatch.setenv("RAG_TOTAL_BUDGET_SEC", "1")

    async def _slow_pipeline(*args, **kwargs):
        await asyncio.sleep(60)
        return []

    monkeypatch.setattr(bridge, "_run_pipeline", _slow_pipeline, raising=True)

    from ext.services.metrics import rag_pipeline_timeout_total

    # Snapshot the existing counter value (may be non-zero across tests).
    def _get_value(label_value):
        try:
            return rag_pipeline_timeout_total.labels(intent=label_value)._value.get()
        except Exception:
            return None

    before = _get_value("specific") or 0

    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="q",
        user_id="user-1",
    )

    after = _get_value("specific") or 0
    assert after > before


@pytest.mark.asyncio
async def test_pipeline_completes_within_budget_unchanged(monkeypatch):
    """Default behaviour: when the pipeline completes well under budget,
    the bridge passes through the result unchanged."""
    _wire_min_bridge(monkeypatch)
    # Default 30s — irrelevant here because pipeline returns immediately.
    monkeypatch.delenv("RAG_TOTAL_BUDGET_SEC", raising=False)

    async def _fast_pipeline(*args, **kwargs):
        return [
            {
                "source": {"id": "x", "name": "x", "url": "x"},
                "document": ["hello"],
                "metadata": [{"source": "x"}],
            }
        ]

    monkeypatch.setattr(bridge, "_run_pipeline", _fast_pipeline, raising=True)

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="q",
        user_id="user-1",
    )

    assert out == [
        {
            "source": {"id": "x", "name": "x", "url": "x"},
            "document": ["hello"],
            "metadata": [{"source": "x"}],
        }
    ]


def test_rag_pipeline_timeout_counter_exists():
    from ext.services import metrics

    assert hasattr(metrics, "rag_pipeline_timeout_total")
    metrics.rag_pipeline_timeout_total.labels(intent="specific")  # should not raise
