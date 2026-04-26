"""Unit tests for ``ext.services.query_intent``.

Pure regex classifier — no mocks, no I/O. Coverage spans each of the
three labels plus a few tricky edge cases where the boundary between
``metadata`` and ``global`` is narrow.
"""
from __future__ import annotations

import pathlib

import pytest

from ext.services.query_intent import classify, classify_with_reason


# --------------------------------------------------------------------------
# metadata — the user wants the catalog itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "what files do I have",
        "which reports do we have",
        "list reports",              # bare "list <docs>" → catalog question
        "show documents",
        "how many reports are there",
        "give me the list",
        "what is in the catalog",    # 'catalog' keyword
        "do you have a Q4 report",
    ],
)
def test_metadata_queries(query: str) -> None:
    assert classify(query) == "metadata", query


# --------------------------------------------------------------------------
# global — aggregation / coverage across the corpus
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "list all the dates mentioned",
        "list every report from January",
        "every report from January",
        "give me all the dates",
        "enumerate the entries",
        "across all documents, what are the top risks",
        "summarize the entire knowledge base",
        "I need the full list of months covered",
    ],
)
def test_global_queries(query: str) -> None:
    assert classify(query) == "global", query


# --------------------------------------------------------------------------
# specific — single-doc / content-anchored
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "what did the Jan 5 report say about revenue?",
        "explain the recommendation in Section 3",
        "who signed off on the April deployment",
        "asdf",
        "",
    ],
)
def test_specific_or_fallback(query: str) -> None:
    assert classify(query) == "specific", query


# --------------------------------------------------------------------------
# Edge / tie-breaker cases
# --------------------------------------------------------------------------

def test_bare_list_reports_is_metadata() -> None:
    """Bare 'list reports' (no all/every) is a catalog question →
    metadata. The catalog preamble will answer it without chunk retrieval.
    """
    assert classify("list reports") == "metadata"
    assert classify("show documents") == "metadata"


def test_list_all_reports_is_global_not_metadata() -> None:
    """'list all <docs>' / 'list every <docs>' is an aggregation — route
    to the doc-summary index so every doc contributes one summary point,
    rather than answering from the catalog preamble alone."""
    assert classify("list all reports about Q4") == "global"
    assert classify("list every report from January") == "global"


def test_case_insensitive() -> None:
    assert classify("LIST ALL DATES") == "global"
    assert classify("WHAT FILES DO I HAVE") == "metadata"


def test_leading_and_trailing_whitespace_ok() -> None:
    assert classify("   list all dates   ") == "global"


def test_none_like_input_safe() -> None:
    assert classify("") == "specific"
    assert classify("   ") == "specific"


# --------------------------------------------------------------------------
# classify_with_reason observability contract
# --------------------------------------------------------------------------

def test_reason_for_metadata_match_names_rule() -> None:
    label, reason = classify_with_reason("what files do I have")
    assert label == "metadata"
    assert reason.startswith("metadata:")


def test_reason_for_global_match_names_rule() -> None:
    label, reason = classify_with_reason("list all the dates")
    assert label == "global"
    assert reason.startswith("global:")


def test_reason_for_specific_fallback() -> None:
    label, reason = classify_with_reason("what did the Jan report say")
    assert label == "specific"
    assert "default" in reason or "no_pattern_matched" in reason


# --------------------------------------------------------------------------
# Plan B Phase 4.10 — RAG_INTENT_LLM is retired. The async hybrid
# classifier (classify_with_qu) is the only supported LLM path; the
# legacy stub _llm_classify and its env-flag are gone from the source.
# --------------------------------------------------------------------------
def test_legacy_intent_llm_flag_purged_from_query_intent_module() -> None:
    """The retired flag and stub must not appear in query_intent.py.

    Source-text check (not just attribute introspection) so future
    contributors can't reintroduce a docstring reference that would let
    an operator search the codebase and assume the flag still exists.
    """
    from ext.services import query_intent

    src = pathlib.Path(query_intent.__file__).read_text()
    assert "RAG_INTENT_LLM" not in src, (
        "RAG_INTENT_LLM is retired in Plan B Phase 4.10; "
        "remove all references from ext/services/query_intent.py."
    )
    assert "_llm_classify" not in src, (
        "_llm_classify stub must be removed (replaced by classify_with_qu)."
    )


def test_legacy_intent_llm_flag_purged_from_kb_config() -> None:
    """The kb_config rag_config schema must no longer accept intent_llm."""
    from ext.services import kb_config

    assert "intent_llm" not in kb_config._KEY_TO_ENV, (
        "intent_llm key must be removed from kb_config._KEY_TO_ENV "
        "(Plan B Phase 4.10 retires RAG_INTENT_LLM)."
    )
