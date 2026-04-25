"""Verify the LLM tiebreaker stub fails loudly when enabled.

Phase 1.6 follow-up. ``_llm_classify`` was a silent stub returning
``("specific", "llm:stub_unimplemented")`` — flipping ``RAG_INTENT_LLM=1``
in production would make intent classification look "LLM-tier" while
actually behaving exactly like the regex tier, with no diagnostic.

The fix: raise ``NotImplementedError`` until Plan B Phase 4 wires the
real Qwen3-4B classifier. This test pins that contract so a future
silent-stub regression breaks CI.
"""
from __future__ import annotations

import pytest

from ext.services import query_intent


def test_llm_classify_raises_not_implemented_directly():
    """Calling ``_llm_classify`` raises NotImplementedError unconditionally."""
    with pytest.raises(NotImplementedError) as exc_info:
        query_intent._llm_classify("anything")
    msg = str(exc_info.value)
    # Diagnostic must point the operator at the right unset/wait action.
    assert "RAG_INTENT_LLM" in msg
    assert "Plan B" in msg or "Phase 4" in msg
    assert "Unset" in msg or "unset" in msg


def test_classify_with_reason_propagates_not_implemented_when_flag_on(monkeypatch):
    """``classify_with_reason`` invokes the LLM tier when
    ``RAG_INTENT_LLM=1`` and the fast path returned ``specific``.

    Today that path raises. Verifying the bubble-up means an operator
    who flips the flag will see the exception in logs / SSE error
    events, not a silent degradation.
    """
    monkeypatch.setenv("RAG_INTENT_LLM", "1")
    # A query that the regex classifier labels ``specific`` (no metadata
    # / global pattern matches, no parseable date).
    query = "what is the third paragraph about"
    with pytest.raises(NotImplementedError):
        query_intent.classify_with_reason(query)


def test_classify_with_reason_does_not_call_llm_when_flag_off(monkeypatch):
    """Default path (flag unset) must NOT hit the LLM stub — operators
    on the default config never see the NotImplementedError.
    """
    monkeypatch.delenv("RAG_INTENT_LLM", raising=False)
    label, reason = query_intent.classify_with_reason(
        "what is the third paragraph about"
    )
    assert label == "specific"
    assert reason  # any non-empty string is fine
