"""Tests for ``mmr_rerank_from_hits`` and the bridge integration default path.

``mmr_rerank_from_hits`` wraps the pure ``mmr_rerank`` with embedder I/O.
We verify:

* The batch path (``embed(list[str])``) is used once when the embedder
  exposes it (our production TEIEmbedder does).
* The non-batch fallback (``embed_batch(list[str])``) is used when only
  that method is exposed.
* Empty hits / top_k >= len(hits) short-circuit before touching the embedder.

And one integration-shape test against ``chat_rag_bridge`` verifying that
when ``RAG_MMR`` is unset, the ``ext.services.mmr`` module is NOT imported.
"""
from __future__ import annotations

import sys

import pytest


# ---------------------------------------------------------------------------
# Stub hit + embedders
# ---------------------------------------------------------------------------
class _StubHit:
    def __init__(self, id: int, text: str) -> None:
        self.id = id
        self.payload = {"text": text}


class _BatchEmbedder:
    """Exposes ``embed(list[str]) -> list[list[float]]`` — production shape."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping
        self.call_count = 0
        self.last_inputs: list[str] | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.last_inputs = list(texts)
        return [self._mapping[t] for t in texts]


class _LegacyBatchEmbedder:
    """Only exposes ``embed_batch(list[str]) -> list[list[float]]``.

    The ``mmr_rerank_from_hits`` helper checks for ``embed`` first, then
    ``embed_batch``. This class lets us exercise the fallback branch.
    """

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping
        self.call_count = 0

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self._mapping[t] for t in texts]


# ---------------------------------------------------------------------------
# mmr_rerank_from_hits
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_hits_returns_empty_without_embedder_call():
    from ext.services.mmr import mmr_rerank_from_hits

    embedder = _BatchEmbedder({})
    out = await mmr_rerank_from_hits("any query", [], embedder, top_k=5)
    assert out == []
    assert embedder.call_count == 0


@pytest.mark.asyncio
async def test_top_k_ge_len_short_circuits_without_embedder_call():
    from ext.services.mmr import mmr_rerank_from_hits

    hits = [_StubHit(0, "alpha"), _StubHit(1, "beta")]
    embedder = _BatchEmbedder({})  # never consulted
    out = await mmr_rerank_from_hits("q", hits, embedder, top_k=5)
    assert [h.id for h in out] == [0, 1]
    assert embedder.call_count == 0


@pytest.mark.asyncio
async def test_uses_batch_embed_once_when_available():
    from ext.services.mmr import mmr_rerank_from_hits

    hits = [
        _StubHit(0, "alpha"),
        _StubHit(1, "alpha_dup"),     # near-duplicate text content
        _StubHit(2, "diverse"),
    ]
    mapping = {
        "q":          [1.0, 0.0, 0.0],
        "alpha":      [0.95, 0.312, 0.0],
        "alpha_dup":  [0.95, 0.312, 0.0],  # identical vector -> duplicate
        "diverse":    [0.5, 0.0, 0.866],
    }
    embedder = _BatchEmbedder(mapping)
    out = await mmr_rerank_from_hits("q", hits, embedder, top_k=2, lambda_=0.5)

    # Exactly ONE embedder call (batched passages + query together)
    assert embedder.call_count == 1
    # Inputs should be passages first, then the query — as documented
    assert embedder.last_inputs == ["alpha", "alpha_dup", "diverse", "q"]

    # Near-dup of hit 0 should NOT be the second pick at lambda=0.5
    assert out[0].id == 0
    assert out[1].id == 2


@pytest.mark.asyncio
async def test_falls_back_to_embed_batch_when_embed_missing():
    """Helper should use ``embed_batch`` if ``embed`` is not exposed.

    We stage the vectors so hit 0 seeds (relevance 0.95 vs 0.3 and 0.0) and
    then hit 2 (orthogonal to hit 0) clearly beats hit 1 (near-dup of hit 0)
    on MMR diversity at lambda=0.5.
    """
    from ext.services.mmr import mmr_rerank_from_hits

    hits = [_StubHit(0, "a"), _StubHit(1, "b"), _StubHit(2, "c")]
    mapping = {
        "q": [1.0, 0.0, 0.0],
        "a": [0.95, 0.312, 0.0],   # rel=0.95
        "b": [0.95, 0.312, 0.0],   # rel=0.95, exact dup of a
        "c": [0.3, 0.0, 0.954],    # rel=0.3, orthogonal direction
    }
    embedder = _LegacyBatchEmbedder(mapping)
    # sanity: ensure the stub really lacks `embed`
    assert not hasattr(embedder, "embed")

    out = await mmr_rerank_from_hits("q", hits, embedder, top_k=2, lambda_=0.5)

    assert embedder.call_count == 1
    # Seed by relevance: hit 0 (a) wins over hit 1 (b) due to lower index on tie.
    # Second pick at lambda=0.5:
    #   hit 1 (exact dup of 0): 0.5*0.95 - 0.5*1.0   = -0.025
    #   hit 2 (diverse):        0.5*0.3  - 0.5*0.285 =  0.0075   <-- winner
    assert out[0].id == 0
    assert out[1].id == 2


@pytest.mark.asyncio
async def test_hit_without_payload_text_falls_back_to_empty_string():
    """Hits with missing ``payload['text']`` and no ``text`` attr still work."""
    from ext.services.mmr import mmr_rerank_from_hits

    class _BareHit:
        def __init__(self, id):
            self.id = id
            self.payload = {}

    hits = [_BareHit(0), _BareHit(1), _BareHit(2)]
    mapping = {
        "q":  [1.0, 0.0],
        "":   [0.5, 0.5],  # all three bare hits get "" text
    }
    embedder = _BatchEmbedder(mapping)
    out = await mmr_rerank_from_hits("q", hits, embedder, top_k=2)
    assert len(out) == 2
    assert embedder.call_count == 1


# ---------------------------------------------------------------------------
# Integration shape: default flag path must NOT import the mmr module
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_default_flag_path_does_not_import_mmr_module(monkeypatch):
    """With ``RAG_MMR`` unset, chat_rag_bridge.retrieve_kb_sources must not
    pull in ``ext.services.mmr`` at all (byte-identical to pre-P1.3).
    """
    from ext.services import chat_rag_bridge as bridge

    # Flush any previous import of the mmr module so we can detect a fresh one.
    sys.modules.pop("ext.services.mmr", None)
    monkeypatch.delenv("RAG_MMR", raising=False)

    # Wire up bridge with minimal stubs — see test_chat_rag_bridge_spotlight.py
    class _FakeSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def execute(self, *a, **kw):
            class _R:
                def first(self):
                    return None
            return _R()

    def _fake_sessionmaker():
        return _FakeSession()

    bridge.configure(vector_store=object(), embedder=object(), sessionmaker=_fake_sessionmaker)

    async def _fake_allowed(session, *, user_id):  # noqa: ARG001
        return [1]

    import ext.services.rbac as _rbac
    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)

    from dataclasses import dataclass

    @dataclass
    class _FakeHit:
        id: int
        score: float
        payload: dict

    async def _fake_retrieve(*, query, selected_kbs, chat_id, vector_store, embedder, per_kb_limit=10, total_limit=30):  # noqa: ARG001
        return [
            _FakeHit(
                id=1, score=0.9,
                payload={
                    "text": "hello",
                    "kb_id": 1,
                    "subtag_id": None,
                    "doc_id": "d",
                    "filename": "f",
                    "chunk_index": 0,
                    "chat_id": None,
                },
            )
        ]

    import ext.services.retriever as _retriever
    import ext.services.reranker as _reranker
    import ext.services.budget as _budget

    monkeypatch.setattr(_retriever, "retrieve", _fake_retrieve, raising=True)
    monkeypatch.setattr(_reranker, "rerank", lambda hits, *, top_k=10: hits, raising=True)
    monkeypatch.setattr(_budget, "budget_chunks", lambda hits, *, max_tokens=4000: hits, raising=True)

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    assert len(out) == 1
    # The key assertion: with the flag off, mmr module was NEVER imported
    assert "ext.services.mmr" not in sys.modules


@pytest.mark.asyncio
async def test_flag_on_imports_and_reorders_via_mmr(monkeypatch):
    """With ``RAG_MMR=1``, the bridge imports ``mmr`` and routes through it."""
    from ext.services import chat_rag_bridge as bridge

    # Reset cached import so we can see it show up again after the flag flip
    sys.modules.pop("ext.services.mmr", None)
    monkeypatch.setenv("RAG_MMR", "1")

    # Minimal session stub
    class _FakeSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def execute(self, *a, **kw):
            class _R:
                def first(self):
                    return None
            return _R()

    def _fake_sessionmaker():
        return _FakeSession()

    # Embedder needs to be real enough for the helper to call .embed()
    class _DeterministicEmbedder:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            # All-ones vectors so MMR is a no-op reorder; we only care the
            # flag-on path runs cleanly and returns a source list.
            return [[1.0, 0.0, 0.0] for _ in texts]

    bridge.configure(
        vector_store=object(),
        embedder=_DeterministicEmbedder(),
        sessionmaker=_fake_sessionmaker,
    )

    async def _fake_allowed(session, *, user_id):  # noqa: ARG001
        return [1]

    import ext.services.rbac as _rbac
    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)

    from dataclasses import dataclass

    @dataclass
    class _FakeHit:
        id: int
        score: float
        payload: dict

    async def _fake_retrieve(*, query, selected_kbs, chat_id, vector_store, embedder, per_kb_limit=10, total_limit=30):  # noqa: ARG001
        return [
            _FakeHit(id=i, score=0.9 - i * 0.1, payload={
                "text": f"t{i}", "kb_id": 1, "subtag_id": None,
                "doc_id": f"d{i}", "filename": f"f{i}",
                "chunk_index": 0, "chat_id": None,
            }) for i in range(3)
        ]

    import ext.services.retriever as _retriever
    import ext.services.reranker as _reranker
    import ext.services.budget as _budget

    monkeypatch.setattr(_retriever, "retrieve", _fake_retrieve, raising=True)
    monkeypatch.setattr(_reranker, "rerank", lambda hits, *, top_k=10: hits, raising=True)
    monkeypatch.setattr(_budget, "budget_chunks", lambda hits, *, max_tokens=4000: hits, raising=True)

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    # Shape preserved + mmr module was imported
    assert isinstance(out, list)
    assert len(out) >= 1
    assert "ext.services.mmr" in sys.modules
