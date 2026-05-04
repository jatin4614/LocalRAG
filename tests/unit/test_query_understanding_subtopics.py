"""Phase 3 / item 5 — QU LLM extracts subtopics into HybridClassification.

NOTE: HybridClassification lives in ext.services.query_intent (not
query_understanding). The import reflects the actual module layout.
"""
from __future__ import annotations

from ext.services.query_intent import HybridClassification


def test_hybrid_classification_has_subtopics_field() -> None:
    h = HybridClassification(
        intent="specific",
        resolved_query="compare visits and operations",
        temporal_constraint=None,
        entities=[],
        subtopics=["visits", "operations"],
    )
    assert h.subtopics == ["visits", "operations"]


def test_hybrid_classification_subtopics_default_empty() -> None:
    """Subtopics defaults to empty list when not provided (back-compat)."""
    h = HybridClassification(
        intent="specific",
        resolved_query="compare visits and operations",
        temporal_constraint=None,
        entities=[],
    )
    assert h.subtopics == []
