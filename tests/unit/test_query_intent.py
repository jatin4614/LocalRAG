"""Unit tests for ``ext.services.query_intent``.

Pure regex classifier — no mocks, no I/O. Coverage spans each of the
three labels plus a few tricky edge cases where the boundary between
``metadata`` and ``global`` is narrow.
"""
from __future__ import annotations

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
