"""B11 + B12 — regex precedence fixes for cases caught in the
2026-04-27 E2E report (intent accuracy 92.5% → ~97.5%).

B11: ``metadata:enumerate_docs`` now accepts an optional quantifier
("all"/"every"/"each") between the verb and the noun, so "List all
documents in the KB" routes as metadata instead of global.

B12: a strong specific_date anchor short-circuit fires BEFORE global
rules when the query has BOTH a parseable date AND a "summary/recap/
report of <date>" anchor phrase. Soak example: "Summary of 4 February
2026 events" was misrouted to global:summary_of.
"""
import pytest

from ext.services.query_intent import classify_with_reason


# ----- B11: "list all <docs> IN THE KB" — explicit catalog qualifier ----
#
# Bare "list all reports" stays global per the project's aggregation
# contract (see test_list_all_reports_is_global_not_metadata). The
# B11 fix only fires when the user adds an explicit container qualifier
# ("in the KB", "in the knowledge base", "in the corpus", "in our database").
# That phrasing unambiguously asks the catalog inventory, not an aggregation.

@pytest.mark.parametrize("q", [
    "List all documents in the KB",
    "list documents in the knowledge base",
    "List all reports in the corpus",
    "show all docs in the KB",
    "What files in the database",
    "list all the documents in our knowledge base",
    "List every report in the corpus",
    "list each document in the KB",
])
def test_b11_list_in_kb_routes_metadata(q):
    intent, reason = classify_with_reason(q)
    assert intent == "metadata", \
        f"{q!r} should be metadata, got {intent}/{reason}"
    assert reason == "metadata:list_in_kb", reason


# Existing global aggregation contract must NOT regress.
@pytest.mark.parametrize("q", [
    "list all reports",                   # bare → global
    "list every report from January",     # date qualifier → global
    "list all reports about Q4",          # topic qualifier → global
    "List all dates of reports",          # "dates" not noun-anchor
])
def test_b11_does_not_overreach_into_global(q):
    intent, _ = classify_with_reason(q)
    assert intent == "global", f"{q!r} should stay global"


# ----- B12: summary of <date> ----------------------------------------

@pytest.mark.parametrize("q,expected_day,expected_month", [
    ("Summary of 4 February 2026 events", 4, "Feb"),
    ("Recap of 5 March 2026", 5, "Mar"),
    ("Report from 17 Feb 2026", 17, "Feb"),
    ("Update for 9 March 2026 outages", 9, "Mar"),
    ("What happened on 12 March 2026?", 12, "Mar"),
])
def test_b12_summary_of_date_routes_specific_date(q, expected_day, expected_month):
    intent, reason = classify_with_reason(q)
    assert intent == "specific_date", \
        f"{q!r} should be specific_date, got {intent}/{reason}"
    # Either anchored path or extracted path is acceptable — both are
    # specific_date and both surface day+month in the reason.
    assert f"{expected_day} {expected_month}" in reason


# Month-only "summary of" without a parseable day must stay global —
# the date short-circuit only fires on full DD MMM YYYY (or ISO) hits,
# so these queries fall through to global:summary_of as before.
@pytest.mark.parametrize("q", [
    "Summary of January",
    "Summarize all of 2026",
    "Recap of last quarter",
    "Summary of the entire knowledge base",
])
def test_b12_does_not_hijack_month_only(q):
    intent, _ = classify_with_reason(q)
    assert intent in ("global", "specific"), \
        f"{q!r} should stay global/specific, got {intent}"
