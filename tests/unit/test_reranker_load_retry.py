"""Phase 1.2 — Reranker retryable singleton.

Replaces the original ``@lru_cache`` model loader with a thread-safe
singleton that retries transient failures (HF download blip, OOM on
first CUDA init, lazy-import errors) with exponential backoff and does
NOT cache exceptions — the next call re-attempts from scratch.

Regression guard: ``@lru_cache`` poisoned the singleton on the first
exception, so every subsequent rerank call re-raised the same error
forever, even after the underlying issue resolved.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from ext.services import cross_encoder_reranker as ccr


@pytest.fixture(autouse=True)
def _clear_singleton():
    ccr._reset_model_for_test()
    yield
    ccr._reset_model_for_test()


def test_load_model_retries_on_transient_failure(monkeypatch):
    calls = {"n": 0}
    real_ce = MagicMock(name="real CrossEncoder")

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return real_ce

    monkeypatch.setattr("sentence_transformers.CrossEncoder", flaky)
    # Tighten retry sleeps so the test runs in <1s
    monkeypatch.setenv("RAG_RERANK_LOAD_RETRIES", "3")
    monkeypatch.setenv("RAG_RERANK_LOAD_RETRY_BASE_SEC", "0.01")
    got = ccr.get_model()
    assert got is real_ce
    assert calls["n"] == 3


def test_load_model_does_not_cache_failure_forever(monkeypatch):
    """After exhausting retries, the NEXT call retries from scratch."""
    attempt = {"n": 0}
    real_ce = MagicMock(name="real CrossEncoder")

    def flaky(*args, **kwargs):
        attempt["n"] += 1
        if attempt["n"] <= 5:
            raise RuntimeError("still failing")
        return real_ce

    monkeypatch.setattr("sentence_transformers.CrossEncoder", flaky)
    monkeypatch.setenv("RAG_RERANK_LOAD_RETRIES", "3")
    monkeypatch.setenv("RAG_RERANK_LOAD_RETRY_BASE_SEC", "0.01")
    # First call: 3 attempts, all fail → raises
    with pytest.raises(RuntimeError):
        ccr.get_model()
    # Subsequent call: must retry from scratch (proving singleton doesn't cache exceptions)
    got = ccr.get_model()
    assert got is real_ce


def test_get_model_is_singleton_on_success(monkeypatch):
    real_ce = MagicMock(name="real CrossEncoder")
    monkeypatch.setattr("sentence_transformers.CrossEncoder", lambda *a, **k: real_ce)
    monkeypatch.setenv("RAG_RERANK_LOAD_RETRIES", "1")
    a = ccr.get_model()
    b = ccr.get_model()
    assert a is b
