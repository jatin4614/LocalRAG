"""Unit tests for ext.services.citation_checker (review §6.10).

Tests the inline citation enforcement behavior:
  * Default OFF (RAG_ENFORCE_CITATIONS unset / "0") → byte-identical pass-through.
  * Sentences with overlapping content from sources → kept clean.
  * Sentences making factual claims with NO source overlap → tagged [unverified].
  * Sentences with no factual claim (e.g. fillers, transitions) → ignored.
  * Counter ``rag_unverified_sentences_total{intent}`` increments on tagged sentences.
  * Fail-open: any internal exception logs + returns input unchanged.
"""
from __future__ import annotations

import pytest

from ext.services import citation_checker


# Helper: read counter value across re-runs without touching prometheus internals.
def _read_counter(intent: str) -> float:
    counter = citation_checker.rag_unverified_sentences_total
    try:
        return counter.labels(intent=intent)._value.get()  # type: ignore[attr-defined]
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Flag-gating
# ---------------------------------------------------------------------------
def test_default_off_passthrough(monkeypatch):
    """Flag unset → return input unchanged, even with hallucinated content."""
    monkeypatch.delenv("RAG_ENFORCE_CITATIONS", raising=False)
    response = "The CEO is Jane Doe. The company was founded in 1999."
    sources = [{"text": "Some unrelated content."}]
    out = citation_checker.enforce_citations(response, sources, intent="specific")
    assert out == response


def test_flag_zero_passthrough(monkeypatch):
    """Flag=0 → byte-identical pass-through."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "0")
    response = "Random factual claim about 2023 numbers."
    out = citation_checker.enforce_citations(response, [], intent="specific")
    assert out == response


# ---------------------------------------------------------------------------
# Cited sentence stays clean (positive case)
# ---------------------------------------------------------------------------
def test_cited_sentence_stays_clean(monkeypatch):
    """A factual sentence that overlaps with a source's text is NOT tagged."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = "The quarterly revenue was 12 million dollars."
    sources = [
        {
            "text": "In Q3 the quarterly revenue was 12 million dollars per filing.",
        }
    ]
    out = citation_checker.enforce_citations(response, sources, intent="specific")
    assert "[unverified]" not in out
    assert out == response


def test_cited_sentence_string_sources(monkeypatch):
    """``sources`` may be a list of plain strings, too."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = "Acme Corporation employs five hundred staff."
    sources = ["Acme Corporation employs five hundred staff worldwide."]
    out = citation_checker.enforce_citations(response, sources, intent="specific")
    assert "[unverified]" not in out


# ---------------------------------------------------------------------------
# Hallucinated sentence gets tagged
# ---------------------------------------------------------------------------
def test_hallucinated_sentence_tagged(monkeypatch):
    """A factual claim with NO source overlap is prepended with [unverified]."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = "The CEO is Jane Smith and was hired in 2024."
    sources = [{"text": "Completely unrelated content about widgets."}]
    out = citation_checker.enforce_citations(response, sources, intent="specific")
    assert "[unverified]" in out
    # Original text content preserved (just tagged)
    assert "Jane Smith" in out


def test_unverified_counter_increments(monkeypatch):
    """Tagged sentences bump rag_unverified_sentences_total{intent}."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    before = _read_counter("specific")
    response = "Bob Robertson became CEO in 2025."
    citation_checker.enforce_citations(response, [{"text": "irrelevant"}], intent="specific")
    after = _read_counter("specific")
    if before >= 0:  # counter available
        assert after >= before + 1


# ---------------------------------------------------------------------------
# Non-factual sentences ignored
# ---------------------------------------------------------------------------
def test_non_factual_sentence_ignored(monkeypatch):
    """A filler sentence (no NP+verb, no date, no number, no proper noun) is
    skipped entirely — never tagged, not counted as unverified."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = "Yes."
    sources = []
    out = citation_checker.enforce_citations(response, sources, intent="specific")
    assert "[unverified]" not in out
    assert out == response


def test_non_factual_question_ignored(monkeypatch):
    """A question with no factual claim doesn't trip the checker."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = "what?"
    out = citation_checker.enforce_citations(response, [], intent="specific")
    assert out == response


# ---------------------------------------------------------------------------
# Mixed: some sentences cited, some not
# ---------------------------------------------------------------------------
def test_mixed_response_only_uncited_tagged(monkeypatch):
    """A response with one cited and one hallucinated sentence: only the
    hallucinated one is tagged."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = (
        "The factory location is Detroit Michigan. "
        "The factory was acquired in 1888 from Acme Holdings."
    )
    sources = [
        {"text": "The factory location is Detroit Michigan with 200 employees."}
    ]
    out = citation_checker.enforce_citations(response, sources, intent="specific")
    # First sentence should NOT be tagged
    assert "Detroit Michigan" in out
    # Second sentence (no overlap) SHOULD be tagged
    assert "[unverified]" in out
    # Tag count: exactly 1 of the two factual claims
    assert out.count("[unverified]") == 1


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------
def test_empty_response_returns_empty(monkeypatch):
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    assert citation_checker.enforce_citations("", [{"text": "x"}], intent="specific") == ""


def test_empty_sources_tags_all_factual(monkeypatch):
    """No sources → every factual sentence tagged."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = "Acme Corp earned 5 million in 2020."
    out = citation_checker.enforce_citations(response, [], intent="specific")
    assert "[unverified]" in out


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------
def test_failopen_on_internal_exception(monkeypatch):
    """If the checker raises (e.g. a malformed source), it returns input
    unchanged rather than killing the response."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")

    # Force the sentence-walker to raise.
    def _boom(*a, **kw):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(citation_checker, "_iter_sentences", _boom, raising=True)
    response = "Some claim about 2024."
    out = citation_checker.enforce_citations(response, [], intent="specific")
    # Pass-through on exception
    assert out == response


# ---------------------------------------------------------------------------
# Source <source id="X">…</source> XML wrapping is supported
# ---------------------------------------------------------------------------
def test_source_xml_wrapped_text_supported(monkeypatch):
    """Sources can carry pre-wrapped text like
    ``<source id="kb_1_doc-42">payload</source>`` — checker extracts the
    payload and matches against it."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = "Project Alpha launched in March 2024."
    sources = [
        {"text": '<source id="kb_1_doc-42">Project Alpha launched in March 2024 per memo.</source>'}
    ]
    out = citation_checker.enforce_citations(response, sources, intent="specific")
    assert "[unverified]" not in out


# ---------------------------------------------------------------------------
# Sentence with proper noun only (heuristic)
# ---------------------------------------------------------------------------
def test_proper_noun_only_treated_as_factual(monkeypatch):
    """A sentence with a proper noun + verb is a factual claim."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = "Acme Holdings provides services."
    out = citation_checker.enforce_citations(response, [{"text": "Acme Holdings provides services nationally."}], intent="global")
    # Should match → not tagged
    assert "[unverified]" not in out


# ---------------------------------------------------------------------------
# Date detection
# ---------------------------------------------------------------------------
def test_iso_date_treated_as_factual(monkeypatch):
    """ISO-8601 dates trigger the factual-claim heuristic."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    response = "The event happened 2024-03-15 as scheduled."
    sources = []  # No matching source
    out = citation_checker.enforce_citations(response, sources, intent="specific")
    # Factual (date) + no source → tagged
    assert "[unverified]" in out


# ---------------------------------------------------------------------------
# /api/rag/check_citations endpoint integration
# ---------------------------------------------------------------------------
def test_check_citations_endpoint_default_off_passthrough(monkeypatch):
    """POST /api/rag/check_citations with flag off → identical pass-through."""
    monkeypatch.delenv("RAG_ENFORCE_CITATIONS", raising=False)
    monkeypatch.setenv("AUTH_MODE", "stub")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ext.routers import rag_stream

    app = FastAPI()
    app.include_router(rag_stream.router)
    client = TestClient(app)

    payload = {
        "response": "The CEO is Jane Doe.",
        "sources": [{"text": "unrelated"}],
        "intent": "specific",
    }
    r = client.post(
        "/api/rag/check_citations", json=payload,
        headers={"X-User-Id": "u-1", "X-User-Role": "user"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "The CEO is Jane Doe."
    assert "[unverified]" not in body["response"]


def test_check_citations_endpoint_flag_on_tags(monkeypatch):
    """POST /api/rag/check_citations with flag on → tags hallucinated."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    monkeypatch.setenv("AUTH_MODE", "stub")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ext.routers import rag_stream

    app = FastAPI()
    app.include_router(rag_stream.router)
    client = TestClient(app)

    payload = {
        "response": "Bob Smith joined Acme in 2024.",
        "sources": [{"text": "Completely irrelevant content."}],
        "intent": "specific",
    }
    r = client.post(
        "/api/rag/check_citations", json=payload,
        headers={"X-User-Id": "u-1", "X-User-Role": "user"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "[unverified]" in body["response"]


def test_check_citations_endpoint_failopen(monkeypatch):
    """If the checker raises, the endpoint returns the input unchanged
    rather than HTTP 500."""
    monkeypatch.setenv("RAG_ENFORCE_CITATIONS", "1")
    monkeypatch.setenv("AUTH_MODE", "stub")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ext.routers import rag_stream

    # Force enforce_citations to blow up — endpoint must catch and return
    # the input string anyway.
    def _boom(*a, **kw):
        raise RuntimeError("simulated")

    monkeypatch.setattr(rag_stream.citation_checker, "enforce_citations", _boom, raising=True)

    app = FastAPI()
    app.include_router(rag_stream.router)
    client = TestClient(app)

    payload = {"response": "Original text.", "sources": []}
    r = client.post(
        "/api/rag/check_citations", json=payload,
        headers={"X-User-Id": "u-1", "X-User-Role": "user"},
    )
    assert r.status_code == 200
    assert r.json()["response"] == "Original text."
