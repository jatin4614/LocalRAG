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
