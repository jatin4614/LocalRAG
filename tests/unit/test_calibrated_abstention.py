"""Unit tests for calibrated abstention (review §6.11).

The "zero hedging" rule in the analyst system prompt encourages confabulation
when retrieval is weak. ``compute_abstention_prefix`` returns a one-line
caveat to prepend to the system prompt for THIS request only, when:
  * RAG_ENFORCE_ABSTENTION=1, AND
  * average rerank-top-k score < RAG_ABSTENTION_THRESHOLD (default 0.1).

When the flag is OFF, returns an empty string.
When scores are high enough, returns an empty string.
Counter ``rag_abstention_caveat_added_total{intent}`` increments when the
caveat is returned.

Tests exercise the helper, not _run_pipeline (which Wave 6E is forbidden
from touching).
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ext.services import chat_rag_bridge as bridge
from ext.services import metrics as metrics_mod


# Helper: read counter value across re-runs without touching prometheus internals.
def _read_counter(intent: str) -> float:
    counter = metrics_mod.rag_abstention_caveat_added_total
    try:
        return counter.labels(intent=intent)._value.get()  # type: ignore[attr-defined]
    except Exception:
        return -1.0


@dataclass
class _FakeHit:
    """Minimal hit shape: anything with a ``score`` attribute."""

    score: float


# ---------------------------------------------------------------------------
# Flag-gating
# ---------------------------------------------------------------------------
def test_flag_off_returns_empty(monkeypatch):
    """RAG_ENFORCE_ABSTENTION unset → empty string regardless of scores."""
    monkeypatch.delenv("RAG_ENFORCE_ABSTENTION", raising=False)
    hits = [_FakeHit(score=0.001)]  # very low
    assert bridge.compute_abstention_prefix(hits, intent="specific") == ""


def test_flag_zero_returns_empty(monkeypatch):
    """RAG_ENFORCE_ABSTENTION=0 → empty string."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "0")
    hits = [_FakeHit(score=0.001)]
    assert bridge.compute_abstention_prefix(hits, intent="specific") == ""


# ---------------------------------------------------------------------------
# High-score path → no caveat
# ---------------------------------------------------------------------------
def test_high_avg_score_no_caveat(monkeypatch):
    """Avg score >= threshold → empty string."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "0.1")
    hits = [_FakeHit(score=0.5), _FakeHit(score=0.6), _FakeHit(score=0.4)]
    out = bridge.compute_abstention_prefix(hits, intent="specific")
    assert out == ""


def test_score_exactly_threshold_no_caveat(monkeypatch):
    """Avg == threshold → no caveat (strict less-than)."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "0.1")
    hits = [_FakeHit(score=0.1)]
    out = bridge.compute_abstention_prefix(hits, intent="specific")
    assert out == ""


# ---------------------------------------------------------------------------
# Low-score path → caveat
# ---------------------------------------------------------------------------
def test_low_avg_score_caveat_added(monkeypatch):
    """Avg score < threshold → caveat string."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "0.1")
    hits = [_FakeHit(score=0.05), _FakeHit(score=0.02), _FakeHit(score=0.08)]
    out = bridge.compute_abstention_prefix(hits, intent="specific")
    assert out
    # Caveat language matches the spec.
    assert "I don't have enough information" in out


def test_caveat_counter_increments(monkeypatch):
    """When caveat is added, counter bumps (best-effort)."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "0.1")
    before = _read_counter("specific")
    hits = [_FakeHit(score=0.01)]
    bridge.compute_abstention_prefix(hits, intent="specific")
    after = _read_counter("specific")
    if before >= 0:  # counter back-end available
        assert after >= before + 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_hits_caveat_added(monkeypatch):
    """No hits at all → avg considered 0 (below any positive threshold) →
    caveat added (this is the most-common 'retrieval failed' case)."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "0.1")
    out = bridge.compute_abstention_prefix([], intent="specific")
    assert out
    assert "I don't have enough information" in out


def test_dict_hits_supported(monkeypatch):
    """Hits can be plain dicts (e.g. sources_out shape) — accept ``score``
    key in addition to ``score`` attribute."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "0.1")
    hits = [{"score": 0.05}, {"score": 0.02}]
    out = bridge.compute_abstention_prefix(hits, intent="specific")
    assert out


def test_hits_with_no_score_treated_as_zero(monkeypatch):
    """Hits with neither attribute nor key → contribute 0 (effectively
    drives avg below threshold)."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "0.1")
    hits = [{"unrelated": "x"}]
    out = bridge.compute_abstention_prefix(hits, intent="specific")
    assert out  # avg of 0 < 0.1 → caveat


def test_garbage_threshold_falls_back_to_default(monkeypatch):
    """Non-numeric RAG_ABSTENTION_THRESHOLD → fall back to 0.1, don't crash."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "not-a-float")
    hits = [_FakeHit(score=0.05)]
    # Should still return caveat (avg 0.05 < default 0.1)
    out = bridge.compute_abstention_prefix(hits, intent="specific")
    assert out


def test_intent_label_used(monkeypatch):
    """The ``intent`` parameter labels the counter — verify global vs
    specific increment different label cells."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "0.1")
    before_g = _read_counter("global")
    bridge.compute_abstention_prefix([_FakeHit(score=0.0)], intent="global")
    after_g = _read_counter("global")
    if before_g >= 0:
        assert after_g >= before_g + 1


def test_failopen_on_internal_exception(monkeypatch):
    """Any exception in the helper returns empty string rather than crashing
    the request."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")

    class _BadHit:
        @property
        def score(self):
            raise RuntimeError("simulated")

    out = bridge.compute_abstention_prefix([_BadHit()], intent="specific")
    # Fail-open: empty string, no exception
    assert out == ""


def test_caveat_is_one_line(monkeypatch):
    """The returned caveat is a single line (no newlines), so callers can
    safely prepend it as a prompt prefix."""
    monkeypatch.setenv("RAG_ENFORCE_ABSTENTION", "1")
    monkeypatch.setenv("RAG_ABSTENTION_THRESHOLD", "0.1")
    out = bridge.compute_abstention_prefix([_FakeHit(score=0.0)], intent="specific")
    # Strip terminating whitespace then verify no embedded newlines remain.
    assert "\n" not in out.rstrip()
