"""Plan B Phase 6.6 — per-KB chunking strategy tests.

Uses ``monkeypatch`` only.
"""
import importlib

import pytest


def test_chunking_strategy_default_window():
    from ext.services.kb_config import get_chunking_strategy
    # No rag_config -> default "window"
    assert get_chunking_strategy(None) == "window"
    assert get_chunking_strategy({}) == "window"


def test_chunking_strategy_explicit_window():
    from ext.services.kb_config import get_chunking_strategy
    assert get_chunking_strategy({"chunking_strategy": "window"}) == "window"


def test_chunking_strategy_structured():
    from ext.services.kb_config import get_chunking_strategy
    assert get_chunking_strategy(
        {"chunking_strategy": "structured"}
    ) == "structured"


def test_chunking_strategy_unknown_falls_back_to_window():
    from ext.services.kb_config import get_chunking_strategy
    assert get_chunking_strategy({"chunking_strategy": "lol"}) == "window"


# Phase 6.X — chunking_strategy is now a first-class VALID_STRING_KEY.
# Before, the PATCH validator silently dropped it; now it's enum-validated
# and ingest-only (excluded from the env flag overlay).


def test_validate_accepts_structured() -> None:
    from ext.services.kb_config import validate_config
    assert validate_config(
        {"chunking_strategy": "structured"}
    ) == {"chunking_strategy": "structured"}


def test_validate_accepts_window() -> None:
    from ext.services.kb_config import validate_config
    assert validate_config(
        {"chunking_strategy": "window"}
    ) == {"chunking_strategy": "window"}


def test_validate_lowercases_and_trims() -> None:
    from ext.services.kb_config import validate_config
    assert validate_config(
        {"chunking_strategy": "  STRUCTURED  "}
    ) == {"chunking_strategy": "structured"}


def test_validate_drops_unknown_string() -> None:
    from ext.services.kb_config import validate_config
    assert validate_config({"chunking_strategy": "raptor"}) == {}


def test_validate_drops_non_string() -> None:
    from ext.services.kb_config import validate_config
    assert validate_config({"chunking_strategy": 5}) == {}
    assert validate_config({"chunking_strategy": True}) == {}
    assert validate_config({"chunking_strategy": None}) == {}


def test_merge_structured_beats_window() -> None:
    from ext.services.kb_config import merge_configs
    merged = merge_configs([
        {"chunking_strategy": "window"},
        {"chunking_strategy": "structured"},
    ])
    assert merged == {"chunking_strategy": "structured"}


def test_merge_first_window_held_when_only_window() -> None:
    from ext.services.kb_config import merge_configs
    merged = merge_configs([
        {"chunking_strategy": "window"},
        {"chunking_strategy": "window"},
    ])
    assert merged == {"chunking_strategy": "window"}


def test_overlay_does_not_emit_chunking_strategy() -> None:
    """chunking_strategy is INGEST_ONLY — must not leak into the
    request-time env flag overlay."""
    from ext.services.kb_config import config_to_env_overrides
    assert config_to_env_overrides(
        {"chunking_strategy": "structured"}
    ) == {}


def test_ingest_calls_structured_chunker_when_kb_opts_in(monkeypatch):
    """Ingest pipeline picks the chunker per KB."""
    monkeypatch.setenv("RAG_STRUCTURED_CHUNKER", "1")
    from ext.services import ingest
    from ext.services import chunker_structured
    importlib.reload(ingest)

    structured_calls = {"n": 0}

    def fake_structured(text, *, chunk_size_tokens, overlap_tokens):
        structured_calls["n"] += 1
        return [{"text": "x", "chunk_type": "prose"}]

    window_calls = {"n": 0}

    def fake_window(text, *, chunk_size_tokens=800, overlap_tokens=100):
        window_calls["n"] += 1
        return ["x"]

    monkeypatch.setattr(chunker_structured, "chunk_structured", fake_structured)
    monkeypatch.setattr(ingest, "_chunk_window", fake_window)

    chunks = ingest.chunk_text_for_kb(
        text="prose", rag_config={"chunking_strategy": "structured"},
    )

    assert structured_calls["n"] == 1
    assert window_calls["n"] == 0


def test_ingest_uses_window_chunker_by_default(monkeypatch):
    monkeypatch.setenv("RAG_STRUCTURED_CHUNKER", "1")
    from ext.services import ingest
    from ext.services import chunker_structured
    importlib.reload(ingest)

    structured_calls = {"n": 0}

    def fake_structured(text, *, chunk_size_tokens, overlap_tokens):
        structured_calls["n"] += 1
        return [{"text": "x", "chunk_type": "prose"}]

    window_calls = {"n": 0}

    def fake_window(text, *, chunk_size_tokens=800, overlap_tokens=100):
        window_calls["n"] += 1
        return ["x"]

    monkeypatch.setattr(chunker_structured, "chunk_structured", fake_structured)
    monkeypatch.setattr(ingest, "_chunk_window", fake_window)

    ingest.chunk_text_for_kb(text="prose", rag_config={})

    assert window_calls["n"] == 1
    assert structured_calls["n"] == 0
