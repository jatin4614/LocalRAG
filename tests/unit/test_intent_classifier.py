"""Unit tests for the Phase 4 intent classifier in ``ext.services.chat_rag_bridge``.

The classifier is a pure string-matching function with no I/O, so we
can exercise every branch without spinning up the bridge. Tests are
organized by the target label.
"""
from __future__ import annotations

import pytest

from ext.services.chat_rag_bridge import classify_intent


@pytest.mark.parametrize(
    "query",
    [
        "List all the April reports.",
        "What files do I have?",
        "Which files mention supply chain?",
        "How many reports are from Q1?",
        "how much data is in the knowledge base",
        "Give me an inventory of docs.",
        "What reports exist?",
        "What documents are available?",
    ],
)
def test_metadata_queries(query: str) -> None:
    assert classify_intent(query) == "metadata", query


@pytest.mark.parametrize(
    "query",
    [
        "Compare March and April risk trends.",
        "What is the trend across all reports?",
        "Summarize the content.",
        "Give me an overview of the reports.",
        "What are the recurring risks?",
        "Describe overall sentiment.",
        "aggregate Q1 risks please",
    ],
)
def test_global_queries(query: str) -> None:
    assert classify_intent(query) == "global", query


@pytest.mark.parametrize(
    "query",
    [
        "What did the 15 Mar report say about supply chain?",
        "Explain the recommendation in Section 3.",
        "Who signed off on the April deployment?",
        "asdf",
        "",
    ],
)
def test_specific_or_fallback(query: str) -> None:
    assert classify_intent(query) == "specific", query


def test_metadata_wins_over_global() -> None:
    """'list' beats 'compare' — enumeration is more correct than aggregation."""
    q = "List all docs comparing March and April."
    assert classify_intent(q) == "metadata"


def test_none_input_is_safe() -> None:
    """An empty/None string should not raise; returns specific by default."""
    assert classify_intent("") == "specific"
    # mypy-wise we promise str in the signature, but runtime defence matters.
    assert classify_intent(None) == "specific"  # type: ignore[arg-type]


def test_case_insensitive() -> None:
    assert classify_intent("LIST ALL FILES") == "metadata"
    assert classify_intent("COMPARE march and april") == "global"
