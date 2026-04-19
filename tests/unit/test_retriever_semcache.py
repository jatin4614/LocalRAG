"""Integration tests for retriever ↔ retrieval_cache wiring (P2.6).

Verifies three behaviors:
  * RAG_SEMCACHE=0 (default): cache module is NEVER consulted. Retriever
    behaves byte-identical to pre-P2.6.
  * RAG_SEMCACHE=1 + miss: Qdrant is called AND cache.put receives the results.
  * RAG_SEMCACHE=1 + hit: Qdrant is NOT called, cached Hits returned directly.

We monkeypatch the functions on the ``retrieval_cache`` module itself (not
``retriever``) because ``retriever`` imports them lazily inside ``retrieve()``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ext.services import retrieval_cache as rc
from ext.services.retriever import retrieve
from ext.services.vector_store import Hit


class _FakeEmbedder:
    async def embed(self, texts):
        return [[0.1] * 16 for _ in texts]


def _make_vs_stub():
    """Minimal async VectorStore stub returning a single dense hit."""
    vs = MagicMock()
    vs.search = AsyncMock(return_value=[Hit(id="d1", score=0.9, payload={"text": "dense"})])
    vs.hybrid_search = AsyncMock(return_value=[Hit(id="h1", score=0.8, payload={"text": "hybrid"})])
    vs._refresh_sparse_cache = AsyncMock(return_value=False)  # force dense path
    return vs


@pytest.fixture(autouse=True)
def _reset_cache_state(monkeypatch):
    """Clear flags/state between tests."""
    rc._reset_client_for_tests()
    monkeypatch.delenv("RAG_SEMCACHE", raising=False)
    monkeypatch.delenv("RAG_SEMCACHE_TTL", raising=False)
    monkeypatch.setenv("RAG_HYBRID", "0")  # keep test focused on cache, not hybrid
    yield
    rc._reset_client_for_tests()


# ---------------------------------------------------------------------------
# Default OFF: cache NEVER consulted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_off_does_not_touch_cache(monkeypatch):
    """With RAG_SEMCACHE unset, retriever must not call get/put at all."""
    get_spy = MagicMock(return_value=None)
    put_spy = MagicMock()
    enabled_spy = MagicMock(return_value=True)  # even if "enabled" via module, not called
    monkeypatch.setattr(rc, "get", get_spy)
    monkeypatch.setattr(rc, "put", put_spy)
    monkeypatch.setattr(rc, "is_enabled", enabled_spy)

    vs = _make_vs_stub()
    hits = await retrieve(
        query="hello",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    # Retriever took the Qdrant path normally.
    assert vs.search.await_count >= 1
    assert hits[0].id == "d1"
    # And the cache module was never consulted.
    assert get_spy.call_count == 0
    assert put_spy.call_count == 0
    assert enabled_spy.call_count == 0


@pytest.mark.asyncio
async def test_flag_zero_does_not_touch_cache(monkeypatch):
    """Explicit RAG_SEMCACHE=0 also takes the no-cache path."""
    monkeypatch.setenv("RAG_SEMCACHE", "0")
    get_spy = MagicMock(return_value=None)
    put_spy = MagicMock()
    monkeypatch.setattr(rc, "get", get_spy)
    monkeypatch.setattr(rc, "put", put_spy)

    vs = _make_vs_stub()
    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    assert get_spy.call_count == 0
    assert put_spy.call_count == 0


# ---------------------------------------------------------------------------
# RAG_SEMCACHE=1 + cache miss: Qdrant runs, results are stored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_miss_calls_qdrant_and_puts(monkeypatch):
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    get_spy = MagicMock(return_value=None)  # miss
    put_spy = MagicMock()
    monkeypatch.setattr(rc, "get", get_spy)
    monkeypatch.setattr(rc, "put", put_spy)
    monkeypatch.setattr(rc, "is_enabled", lambda: True)

    vs = _make_vs_stub()
    hits = await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    # Qdrant ran.
    assert vs.search.await_count >= 1
    assert hits[0].id == "d1"
    # get was called once (the miss).
    assert get_spy.call_count == 1
    # put was called once with the retrieved results.
    assert put_spy.call_count == 1
    _args, _kwargs = put_spy.call_args
    # Positional: (query_vec, selected_kbs, chat_id, hits)
    stored_hits = _args[3] if len(_args) >= 4 else _kwargs.get("hits")
    assert stored_hits[0].id == "d1"


# ---------------------------------------------------------------------------
# RAG_SEMCACHE=1 + cache hit: Qdrant is NOT called, cached Hits returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_skips_qdrant(monkeypatch):
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    cached = [
        {"id": "cached-1", "score": 0.99, "payload": {"text": "hot"}},
        {"id": "cached-2", "score": 0.88, "payload": {"text": "warm"}},
    ]
    get_spy = MagicMock(return_value=cached)  # HIT
    put_spy = MagicMock()
    monkeypatch.setattr(rc, "get", get_spy)
    monkeypatch.setattr(rc, "put", put_spy)
    monkeypatch.setattr(rc, "is_enabled", lambda: True)

    vs = _make_vs_stub()
    hits = await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    # Qdrant was NOT called.
    assert vs.search.await_count == 0
    assert vs.hybrid_search.await_count == 0
    # Cache was consulted.
    assert get_spy.call_count == 1
    # No put on hit (that would be wasteful work).
    assert put_spy.call_count == 0
    # Result came from cache, rehydrated into Hit instances.
    assert isinstance(hits[0], Hit)
    assert hits[0].id == "cached-1"
    assert hits[0].score == 0.99
    assert hits[0].payload == {"text": "hot"}
    assert hits[1].id == "cached-2"


@pytest.mark.asyncio
async def test_cache_hit_respects_total_limit(monkeypatch):
    """Caller's total_limit trims even cached results (safety)."""
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    cached = [
        {"id": f"c{i}", "score": 1.0 - i * 0.01, "payload": {}} for i in range(50)
    ]
    monkeypatch.setattr(rc, "get", lambda *a, **k: cached)
    monkeypatch.setattr(rc, "is_enabled", lambda: True)

    vs = _make_vs_stub()
    hits = await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
        total_limit=5,
    )
    assert len(hits) == 5
    assert vs.search.await_count == 0


@pytest.mark.asyncio
async def test_is_enabled_false_takes_normal_path(monkeypatch):
    """RAG_SEMCACHE=1 but is_enabled() False (e.g. Redis down) → normal path."""
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    get_spy = MagicMock()
    put_spy = MagicMock()
    monkeypatch.setattr(rc, "is_enabled", lambda: False)
    monkeypatch.setattr(rc, "get", get_spy)
    monkeypatch.setattr(rc, "put", put_spy)

    vs = _make_vs_stub()
    hits = await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    # Qdrant ran; cache skipped entirely.
    assert vs.search.await_count >= 1
    assert hits[0].id == "d1"
    assert get_spy.call_count == 0
    assert put_spy.call_count == 0
