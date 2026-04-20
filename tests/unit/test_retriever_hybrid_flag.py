"""Unit tests for the ``RAG_HYBRID`` flag in ``ext.services.retriever``.

Validates that the default path (flag off) is unchanged — retriever calls
``vs.search(...)`` exclusively — and that turning the flag on routes through
``vs.hybrid_search(...)`` only for collections with sparse support.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ext.services.retriever import retrieve
from ext.services.vector_store import Hit


class _FakeEmbedder:
    async def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


def _make_vs_stub(*, sparse_support: bool):
    """Build a VectorStore-like stub. If ``sparse_support`` is True,
    ``_refresh_sparse_cache`` returns True; otherwise False.
    """
    vs = MagicMock()
    vs.search = AsyncMock(return_value=[Hit(id="d1", score=0.9, payload={"text": "dense"})])
    vs.hybrid_search = AsyncMock(return_value=[Hit(id="h1", score=0.8, payload={"text": "hybrid"})])
    vs._refresh_sparse_cache = AsyncMock(return_value=sparse_support)
    vs._collection_has_sparse = MagicMock(return_value=sparse_support)
    return vs


@pytest.mark.asyncio
async def test_hybrid_flag_off_uses_dense_search(monkeypatch) -> None:
    """Flag explicitly off ("0"): retriever must call vs.search only.

    As of 2026-04-19 the default flipped to on; this test now sets the flag
    to "0" explicitly to cover the force-dense-only path.
    """
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs_stub(sparse_support=True)  # Even if collection supports sparse.
    hits = await retrieve(
        query="hello",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    assert vs.search.await_count >= 1
    assert vs.hybrid_search.await_count == 0
    assert hits[0].id == "d1"


@pytest.mark.asyncio
async def test_hybrid_flag_explicitly_off(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs_stub(sparse_support=True)
    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    assert vs.hybrid_search.await_count == 0


@pytest.mark.asyncio
async def test_hybrid_flag_on_with_sparse_support_uses_hybrid(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "1")
    vs = _make_vs_stub(sparse_support=True)
    hits = await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    assert vs.hybrid_search.await_count >= 1
    # One call per KB (here 1) + one for chat (None → returns [] w/o vs call).
    # Chat is None, so only the KB arm hits a vector_store method.
    assert hits[0].id == "h1"
    # dense-only search not used when hybrid wins.
    assert vs.search.await_count == 0


@pytest.mark.asyncio
async def test_hybrid_flag_on_but_legacy_collection_falls_back(monkeypatch) -> None:
    """RAG_HYBRID=1 but collection has no sparse → fallback to dense."""
    monkeypatch.setenv("RAG_HYBRID", "1")
    vs = _make_vs_stub(sparse_support=False)
    hits = await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 42}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    assert vs.hybrid_search.await_count == 0
    assert vs.search.await_count >= 1
    assert hits[0].id == "d1"


@pytest.mark.asyncio
async def test_hybrid_flag_mixed_collections(monkeypatch) -> None:
    """Some KBs have sparse, others don't — each routes independently."""
    monkeypatch.setenv("RAG_HYBRID", "1")
    vs = MagicMock()

    async def refresh(name):
        return name == "kb_1"  # only kb_1 is hybrid

    vs._refresh_sparse_cache = AsyncMock(side_effect=refresh)
    vs._collection_has_sparse = MagicMock(side_effect=lambda n: n == "kb_1")
    vs.search = AsyncMock(return_value=[Hit(id="d", score=0.5, payload={})])
    vs.hybrid_search = AsyncMock(return_value=[Hit(id="h", score=0.9, payload={})])

    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}, {"kb_id": 2}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    # kb_1 → hybrid; kb_2 → dense; chat None → no call.
    assert vs.hybrid_search.await_count == 1
    assert vs.search.await_count == 1
