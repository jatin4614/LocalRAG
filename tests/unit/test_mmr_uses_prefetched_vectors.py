"""Tests for §5.9 — MMR uses pre-fetched ``Hit.vector`` instead of re-embedding.

When ``Hit.vector`` is populated (the upstream retriever asked Qdrant for
``with_vectors=True``) MMR skips its TEI re-embed of the passage texts and
uses the cached vectors directly. This saves a TEI round-trip per request
when MMR is enabled.

Properties verified:

1. ``Hit.vector`` is an optional field; legacy callers that don't set it
   produce identical hits as before.
2. When all hits carry ``.vector``, ``mmr_rerank_from_hits`` calls the
   embedder ONLY for the query (not for the passages).
3. The ranking output is identical (within float tolerance) to the
   non-prefetched path that re-embeds the passages.
4. Mixed input (some hits with vectors, some without) still works — the
   helper falls back to the embedder when a vector is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _Hit:
    id: int
    score: float
    payload: dict
    vector: Optional[list[float]] = None


class _CountingEmbedder:
    """Records call count and the input texts."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping
        self.call_count = 0
        self.last_inputs: list[str] | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.last_inputs = list(texts)
        return [self._mapping[t] for t in texts]


def test_hit_class_has_optional_vector_field():
    """Sanity: the production ``Hit`` dataclass exposes ``vector`` (default None)."""
    from ext.services.vector_store import Hit

    h = Hit(id=1, score=0.9, payload={})
    assert hasattr(h, "vector")
    assert h.vector is None  # default


@pytest.mark.asyncio
async def test_prefetched_vectors_skip_passage_embed():
    """When every hit carries ``.vector``, embed is called only for the query."""
    from ext.services.mmr import mmr_rerank_from_hits

    hits = [
        _Hit(id=0, score=0.9, payload={"text": "alpha"}, vector=[0.95, 0.312, 0.0]),
        _Hit(id=1, score=0.85, payload={"text": "alpha_dup"}, vector=[0.95, 0.312, 0.0]),
        _Hit(id=2, score=0.7, payload={"text": "diverse"}, vector=[0.5, 0.0, 0.866]),
    ]
    mapping = {"q": [1.0, 0.0, 0.0]}  # only the query is embedded
    embedder = _CountingEmbedder(mapping)

    out = await mmr_rerank_from_hits("q", hits, embedder, top_k=2, lambda_=0.5)

    assert embedder.call_count == 1
    # The single embed call gets ONLY the query — passages were not sent.
    assert embedder.last_inputs == ["q"]

    # Same MMR ranking as the legacy non-prefetched path.
    assert [h.id for h in out] == [0, 2]


@pytest.mark.asyncio
async def test_no_prefetched_vectors_falls_back_to_legacy_embed():
    """Vector field is None → helper still embeds passages (legacy behaviour)."""
    from ext.services.mmr import mmr_rerank_from_hits

    hits = [
        _Hit(id=0, score=0.9, payload={"text": "alpha"}, vector=None),
        _Hit(id=1, score=0.85, payload={"text": "alpha_dup"}, vector=None),
        _Hit(id=2, score=0.7, payload={"text": "diverse"}, vector=None),
    ]
    mapping = {
        "q":          [1.0, 0.0, 0.0],
        "alpha":      [0.95, 0.312, 0.0],
        "alpha_dup":  [0.95, 0.312, 0.0],
        "diverse":    [0.5, 0.0, 0.866],
    }
    embedder = _CountingEmbedder(mapping)

    out = await mmr_rerank_from_hits("q", hits, embedder, top_k=2, lambda_=0.5)

    # Single batched call — passages first, then the query (legacy path).
    assert embedder.call_count == 1
    assert embedder.last_inputs == ["alpha", "alpha_dup", "diverse", "q"]
    assert [h.id for h in out] == [0, 2]


@pytest.mark.asyncio
async def test_prefetched_vs_legacy_returns_same_order():
    """Property-style check: identical ranking with vs without prefetched vectors."""
    from ext.services.mmr import mmr_rerank_from_hits

    vec_table = {
        "alpha":     [0.95, 0.312, 0.0],
        "alpha_dup": [0.95, 0.312, 0.0],
        "diverse":   [0.5, 0.0, 0.866],
        "wildcard":  [0.7, 0.7, 0.14],
    }
    texts = ["alpha", "alpha_dup", "diverse", "wildcard"]

    # Path A — vectors prefetched on the Hit
    hits_with = [
        _Hit(id=i, score=1.0 - i * 0.05, payload={"text": t}, vector=vec_table[t])
        for i, t in enumerate(texts)
    ]
    embedder_with = _CountingEmbedder({"q": [1.0, 0.0, 0.0]})
    out_with = await mmr_rerank_from_hits("q", hits_with, embedder_with, top_k=3, lambda_=0.7)

    # Path B — no prefetched vectors; helper re-embeds passages
    hits_without = [
        _Hit(id=i, score=1.0 - i * 0.05, payload={"text": t}, vector=None)
        for i, t in enumerate(texts)
    ]
    embedder_without = _CountingEmbedder({"q": [1.0, 0.0, 0.0], **vec_table})
    out_without = await mmr_rerank_from_hits(
        "q", hits_without, embedder_without, top_k=3, lambda_=0.7
    )

    # Order must match exactly — same vectors, same algorithm.
    assert [h.id for h in out_with] == [h.id for h in out_without]


@pytest.mark.asyncio
async def test_partial_prefetched_falls_back_to_embed():
    """If any hit lacks ``.vector``, the helper batches passage texts (legacy
    path). Mixed inputs are unsafe to interleave — easier and more robust to
    re-embed the whole batch.
    """
    from ext.services.mmr import mmr_rerank_from_hits

    hits = [
        _Hit(id=0, score=0.9, payload={"text": "alpha"}, vector=[0.95, 0.312, 0.0]),
        _Hit(id=1, score=0.85, payload={"text": "alpha_dup"}, vector=None),  # missing
        _Hit(id=2, score=0.7, payload={"text": "diverse"}, vector=[0.5, 0.0, 0.866]),
    ]
    mapping = {
        "q":          [1.0, 0.0, 0.0],
        "alpha":      [0.95, 0.312, 0.0],
        "alpha_dup":  [0.95, 0.312, 0.0],
        "diverse":    [0.5, 0.0, 0.866],
    }
    embedder = _CountingEmbedder(mapping)

    out = await mmr_rerank_from_hits("q", hits, embedder, top_k=2, lambda_=0.5)
    # Mixed → fall back to legacy batch embed
    assert embedder.call_count == 1
    assert embedder.last_inputs == ["alpha", "alpha_dup", "diverse", "q"]
    assert [h.id for h in out] == [0, 2]


@pytest.mark.asyncio
async def test_search_with_vectors_flag_populates_hit_vector(monkeypatch):
    """``vector_store.search(..., with_vectors=True)`` populates ``Hit.vector``.

    The qdrant client's ``query_points`` is monkey-patched to return points
    that carry a ``.vector`` attribute — we just verify the wrapper passes
    it through onto the dataclass field.
    """
    from ext.services.vector_store import Hit, VectorStore

    # Build a minimal VectorStore-shaped object — bypass __init__ (which tries
    # to construct an AsyncQdrantClient against a real URL).
    vs = VectorStore.__new__(VectorStore)
    vs._url = "http://test"
    vs._vector_size = 3
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {"k": False}
    vs._colbert_cache = {}
    vs._sharding_cache = {}

    class _FakePoint:
        def __init__(self, id, score, payload, vector):
            self.id = id
            self.score = score
            self.payload = payload
            self.vector = vector

    class _Resp:
        def __init__(self, points):
            self.points = points

    async def _fake_query_points(**kwargs):
        # Confirm with_vectors made it into the wire kwargs.
        assert kwargs.get("with_vectors") is True
        return _Resp([
            _FakePoint(id=1, score=0.9, payload={"text": "x"}, vector=[1.0, 0.0, 0.0]),
            _FakePoint(id=2, score=0.8, payload={"text": "y"}, vector=[0.0, 1.0, 0.0]),
        ])

    class _FakeClient:
        async def query_points(self, **kwargs):
            return await _fake_query_points(**kwargs)

    vs._client = _FakeClient()  # type: ignore[assignment]

    out = await vs.search("k", [1.0, 0.0, 0.0], limit=2, with_vectors=True)
    assert all(isinstance(h, Hit) for h in out)
    assert out[0].vector == [1.0, 0.0, 0.0]
    assert out[1].vector == [0.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_search_without_with_vectors_keeps_vector_none(monkeypatch):
    """Default (with_vectors=False) → Hit.vector stays None."""
    from ext.services.vector_store import VectorStore

    vs = VectorStore.__new__(VectorStore)
    vs._url = "http://test"
    vs._vector_size = 3
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {"k": False}
    vs._colbert_cache = {}
    vs._sharding_cache = {}

    class _FakePoint:
        def __init__(self, id, score, payload):
            self.id = id
            self.score = score
            self.payload = payload

    class _Resp:
        def __init__(self, points):
            self.points = points

    async def _fake_query_points(**kwargs):
        # Default path must NOT request vectors.
        assert kwargs.get("with_vectors") in (None, False)
        return _Resp([_FakePoint(id=1, score=0.9, payload={"text": "x"})])

    class _FakeClient:
        async def query_points(self, **kwargs):
            return await _fake_query_points(**kwargs)

    vs._client = _FakeClient()  # type: ignore[assignment]

    out = await vs.search("k", [1.0, 0.0, 0.0], limit=2)
    assert out[0].vector is None
