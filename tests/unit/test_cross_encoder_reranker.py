"""Tests for the cross-encoder reranker wrapper.

Skipped cleanly if ``sentence-transformers`` isn't installed (flag-gated
optional dependency). When installed, runs shape + smoke ordering tests
but does NOT download the full 560 MB model by default — the module-level
``_load_model`` is stubbed via monkeypatch.
"""
from __future__ import annotations

import sys
import types

import pytest

from ext.services import cross_encoder_reranker as cer


# ---------------------------------------------------------------------------
# Fake CrossEncoder — keyword-based "relevance" so smoke tests are deterministic
# without downloading any real model.
# ---------------------------------------------------------------------------
class _FakeCrossEncoder:
    def __init__(self, model_name: str, max_length: int = 512) -> None:  # noqa: D401
        self.model_name = model_name
        self.max_length = max_length

    def predict(self, pairs, batch_size: int = 8, show_progress_bar: bool = False):
        # Very small fake "relevance" signal: check whether any prefix of each
        # query token (length >= 3) appears as a substring in the passage.
        # This handles simple English inflections ("dogs" -> "dog", "barks"
        # -> "bark") sufficiently for unit-test smoke ordering.
        scores = []
        for q, p in pairs:
            q_tokens = [t for t in (q or "").lower().split() if t]
            if not q_tokens:
                scores.append(0.0)
                continue
            p_lc = (p or "").lower()
            hits = 0
            for tok in q_tokens:
                # Try the token itself, then prefixes down to length 3
                for L in range(len(tok), 2, -1):
                    if tok[:L] in p_lc:
                        hits += 1
                        break
            scores.append(float(hits) / float(len(q_tokens)))
        return scores


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    """Stub sentence_transformers.CrossEncoder + clear lru_cache so each test is clean."""
    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.CrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    cer._load_model.cache_clear()
    yield
    cer._load_model.cache_clear()


class _StubHit:
    def __init__(self, id, text, score=0.0):
        self.id = id
        self.score = score
        self.payload = {"text": text}


def test_score_pairs_empty_returns_empty(fake_cross_encoder):
    assert cer.score_pairs("hello", []) == []


def test_score_pairs_returns_floats_same_length(fake_cross_encoder):
    scores = cer.score_pairs("dogs bark", ["a dog barks", "the sun is yellow", "puppies bark too"])
    assert isinstance(scores, list)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_score_pairs_smoke_ordering(fake_cross_encoder):
    # "dogs" clearly appears in the first passage, not the second.
    scores = cer.score_pairs("dogs", ["A dog barks loudly.", "The sun is yellow."])
    assert scores[0] > scores[1]


def test_rerank_cross_encoder_empty_returns_empty(fake_cross_encoder):
    assert cer.rerank_cross_encoder("q", []) == []


def test_rerank_cross_encoder_preserves_hit_objects(fake_cross_encoder):
    hits = [
        _StubHit(1, "The sun is yellow."),
        _StubHit(2, "A dog barks loudly."),
        _StubHit(3, "Cats purr softly."),
    ]
    out = cer.rerank_cross_encoder("dogs", hits, top_k=3)
    assert len(out) == 3
    assert all(isinstance(h, _StubHit) for h in out)
    # hit #2 should be ranked first (contains "dog")
    assert out[0].id == 2


def test_rerank_cross_encoder_top_k_truncation(fake_cross_encoder):
    hits = [
        _StubHit(1, "The sun is yellow."),
        _StubHit(2, "A dog barks loudly."),
        _StubHit(3, "Cats purr softly."),
        _StubHit(4, "Puppies love to play."),
    ]
    out = cer.rerank_cross_encoder("dog", hits, top_k=2)
    assert len(out) == 2


def test_rerank_cross_encoder_missing_text_falls_back(fake_cross_encoder):
    """Hits lacking payload['text'] must still be processed (empty passage)."""

    class _BareHit:
        def __init__(self, id):
            self.id = id
            self.payload = {}  # no text

    hits = [_BareHit(1), _BareHit(2)]
    out = cer.rerank_cross_encoder("anything", hits, top_k=2)
    assert len(out) == 2


def test_missing_sentence_transformers_raises_cross_encoder_unavailable(monkeypatch):
    """If sentence_transformers is NOT installed, score_pairs must raise CrossEncoderUnavailable."""
    # Ensure import fails
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    cer._load_model.cache_clear()

    # Guard: also remove from the import machinery by deleting the key entirely.
    # Setting to None triggers ImportError on `from sentence_transformers import ...`
    # in Python's import system.
    with pytest.raises(cer.CrossEncoderUnavailable):
        cer.score_pairs("q", ["p"])
    cer._load_model.cache_clear()


# Optional: integration smoke test that actually loads the real model.
# Only runs when sentence_transformers is installed AND RAG_RERANK_RUN_REAL=1
# is set — keeps the default unit-suite fast and offline.
@pytest.mark.skipif(
    __import__("os").environ.get("RAG_RERANK_RUN_REAL", "0") != "1",
    reason="Real model test opt-in: set RAG_RERANK_RUN_REAL=1 to exercise BAAI/bge-reranker-v2-m3",
)
def test_real_model_smoke():
    pytest.importorskip("sentence_transformers")
    cer._load_model.cache_clear()
    try:
        scores = cer.score_pairs("dogs", ["A dog barks loudly.", "The sun is yellow."])
    finally:
        cer._load_model.cache_clear()
    assert len(scores) == 2
    assert scores[0] > scores[1]
