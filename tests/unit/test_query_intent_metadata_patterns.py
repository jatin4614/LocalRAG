"""B5 — natural metadata phrasings the original regex registry missed.

Soak found that "What are total files available with you complete from
when to when" hit ``default:no_pattern_matched`` and got labelled
``specific``. These tests pin the new families of metadata patterns:
total-counts, date-range questions, complete-from/to scope phrasings,
knowledge-source questions, and bare "show me everything".
"""
from __future__ import annotations

import pytest

from ext.services.query_intent import classify, classify_with_reason


# --- Total files / reports / documents ----------------------------------


def test_what_are_total_files_available():
    label, reason = classify_with_reason("What are total files available?")
    assert label == "metadata", f"got {label} ({reason})"
    assert reason.startswith("metadata:"), reason


def test_total_reports_in_kb():
    label, reason = classify_with_reason("Total reports in the KB?")
    assert label == "metadata", f"got {label} ({reason})"
    assert reason.startswith("metadata:"), reason


def test_total_documents_uploaded():
    assert classify("How about total documents uploaded so far?") == "metadata"


# --- Date range phrasings ----------------------------------------------


def test_from_when_to_when():
    label, reason = classify_with_reason("From when to when do you have data?")
    assert label == "metadata", f"got {label} ({reason})"
    assert reason.startswith("metadata:"), reason


def test_what_is_the_date_range():
    assert classify("What's the date range of your reports?") == "metadata"


def test_earliest_to_latest():
    assert classify("Earliest to latest date you have?") == "metadata"


# --- Complete-from-to scope --------------------------------------------


def test_complete_from_january_to_april():
    label, reason = classify_with_reason(
        "Complete from January to April reports"
    )
    assert label == "metadata", f"got {label} ({reason})"
    assert reason.startswith("metadata:"), reason


def test_combined_total_files_complete_from_to():
    """The exact failing query from production soak."""
    label, reason = classify_with_reason(
        "What are total files available with you complete from when to when"
    )
    assert label == "metadata", f"got {label} ({reason})"
    assert reason.startswith("metadata:"), reason


# --- Knowledge sources / corpus ----------------------------------------


def test_what_is_in_your_knowledge_sources():
    label, reason = classify_with_reason("What's in your knowledge sources?")
    assert label == "metadata", f"got {label} ({reason})"
    assert reason.startswith("metadata:"), reason


def test_what_do_you_have_on_file():
    assert classify("What do you have on file?") == "metadata"


def test_whats_in_your_corpus():
    assert classify("What's in your corpus?") == "metadata"


# --- Show / list everything ---------------------------------------------


def test_show_me_everything_you_know():
    label, reason = classify_with_reason("Show me everything you know")
    assert label == "metadata", f"got {label} ({reason})"
    assert reason.startswith("metadata:"), reason


def test_list_everything_you_have():
    assert classify("List everything you have") == "metadata"


def test_show_all_you_have():
    assert classify("Show all you have") == "metadata"


# --- Regression sanity: existing behaviours still hold ------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        # Classic metadata phrasings — must still classify as metadata
        ("What files do I have?", "metadata"),
        ("List the documents", "metadata"),
        ("How many reports?", "metadata"),
        # Specific-content question — must NOT get pulled into metadata
        # by the new patterns. "Total revenue" is a content number, not
        # a catalog count.
        ("What is the total revenue in Q1?", "specific"),
        # Date-anchored content — must NOT get pulled into metadata.
        ("What did the 5 Jan 2026 report say?", "specific_date"),
    ],
)
def test_no_regression_on_existing_intents(query, expected):
    assert classify(query) == expected, query
