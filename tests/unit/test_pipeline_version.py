"""Unit tests for the composite pipeline version stamp.

``current_version()`` is the single source of truth for what counts as a
"pipeline generation": any constant change must flip the string so reindex
logic can tell stale chunks apart.
"""
from __future__ import annotations

from ext.services import pipeline_version


def test_current_version_non_empty_and_stable():
    v1 = pipeline_version.current_version()
    v2 = pipeline_version.current_version()
    assert v1
    assert v1 == v2  # stable across calls


def test_current_version_contains_all_four_components():
    v = pipeline_version.current_version()
    # Each component is encoded as "key=value" joined by "|". The keys are
    # documented in pipeline_version.py; don't let them drift without a
    # conscious bump.
    assert "chunker=" in v
    assert "extractor=" in v
    assert "embedder=" in v
    assert "ctx=" in v
    assert v.count("|") == 3


def test_constants_roll_into_version_string():
    v = pipeline_version.current_version()
    assert pipeline_version.CHUNKER_VERSION in v
    assert pipeline_version.EXTRACTOR_VERSION in v
    assert pipeline_version.EMBEDDER_MODEL in v
    assert pipeline_version.CONTEXT_AUGMENTATION in v


def test_bumping_chunker_changes_version(monkeypatch):
    before = pipeline_version.current_version()
    monkeypatch.setattr(pipeline_version, "CHUNKER_VERSION", "v99-test")
    after = pipeline_version.current_version()
    assert before != after
    assert "v99-test" in after


def test_bumping_extractor_changes_version(monkeypatch):
    before = pipeline_version.current_version()
    monkeypatch.setattr(pipeline_version, "EXTRACTOR_VERSION", "v99-ext")
    after = pipeline_version.current_version()
    assert before != after
    assert "v99-ext" in after


def test_bumping_embedder_changes_version(monkeypatch):
    before = pipeline_version.current_version()
    monkeypatch.setattr(pipeline_version, "EMBEDDER_MODEL", "test-embed-xl")
    after = pipeline_version.current_version()
    assert before != after
    assert "test-embed-xl" in after


def test_bumping_context_changes_version(monkeypatch):
    before = pipeline_version.current_version()
    monkeypatch.setattr(pipeline_version, "CONTEXT_AUGMENTATION", "anthropic-ctx")
    after = pipeline_version.current_version()
    assert before != after
    assert "anthropic-ctx" in after
