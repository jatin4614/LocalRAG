"""Unit tests for ``VectorStore.hybrid_search`` + sparse-support detection.

These tests never hit a real Qdrant — they mock ``AsyncQdrantClient`` so we can
assert the exact shape of the ``query_points`` / ``create_collection`` calls.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ext.services.vector_store import VectorStore


def _make_vs() -> VectorStore:
    """Build a VectorStore without actually connecting to Qdrant."""
    vs = VectorStore.__new__(VectorStore)
    vs._client = MagicMock()
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    return vs


def _collection_info(has_sparse: bool):
    """Build a fake CollectionInfo with (or without) a bm25 sparse vector."""
    sparse = {"bm25": object()} if has_sparse else None
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(sparse_vectors=sparse),
        ),
    )


# ---------- _collection_has_sparse / _refresh_sparse_cache --------------------


@pytest.mark.asyncio
async def test_refresh_sparse_cache_detects_hybrid_collection() -> None:
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(return_value=_collection_info(has_sparse=True))
    assert await vs._refresh_sparse_cache("kb_new") is True
    # Cached now — no second Qdrant call.
    vs._client.get_collection.reset_mock()
    assert vs._collection_has_sparse("kb_new") is True
    vs._client.get_collection.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_sparse_cache_detects_legacy_collection() -> None:
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(return_value=_collection_info(has_sparse=False))
    assert await vs._refresh_sparse_cache("kb_legacy") is False
    assert vs._collection_has_sparse("kb_legacy") is False


@pytest.mark.asyncio
async def test_refresh_sparse_cache_handles_nonexistent_collection() -> None:
    """Missing collection → False (fail closed → fallback to dense-only)."""
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=RuntimeError("not found"))
    assert await vs._refresh_sparse_cache("kb_missing") is False
    assert vs._collection_has_sparse("kb_missing") is False


def test_collection_has_sparse_returns_false_for_unknown() -> None:
    """Before any refresh, unknown collection must fail closed."""
    vs = _make_vs()
    assert vs._collection_has_sparse("kb_never_seen") is False


# ---------- ensure_collection(with_sparse=True) -------------------------------


@pytest.mark.asyncio
async def test_ensure_collection_with_sparse_builds_hybrid_config() -> None:
    vs = _make_vs()
    create = AsyncMock()
    vs._client.create_collection = create
    vs._client.create_payload_index = AsyncMock()
    await vs.ensure_collection("kb_new", with_sparse=True)
    create.assert_awaited_once()
    _, kwargs = create.call_args
    # Must pass both vectors_config (dense named) and sparse_vectors_config (bm25).
    assert "vectors_config" in kwargs
    assert "sparse_vectors_config" in kwargs
    assert "dense" in kwargs["vectors_config"]
    assert "bm25" in kwargs["sparse_vectors_config"]
    # And populate the cache so a later retrieval knows sparse is available.
    assert vs._collection_has_sparse("kb_new") is True


@pytest.mark.asyncio
async def test_ensure_collection_default_preserves_legacy_shape() -> None:
    """Default call (with_sparse=False) must be byte-identical to pre-hybrid."""
    vs = _make_vs()
    create = AsyncMock()
    vs._client.create_collection = create
    vs._client.create_payload_index = AsyncMock()
    await vs.ensure_collection("kb_legacy")
    _, kwargs = create.call_args
    assert "sparse_vectors_config" not in kwargs
    # Single unnamed dense VectorParams (not a dict).
    vc = kwargs["vectors_config"]
    assert not isinstance(vc, dict), f"legacy shape must be unnamed, got {type(vc)}"


# ---------- hybrid_search -----------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_search_builds_rrf_fusion_with_two_prefetch_arms(monkeypatch) -> None:
    """Verify the query_points call has exactly dense + sparse prefetch + RRF fusion."""
    vs = _make_vs()
    # Stub out sparse embed so we don't need fastembed installed in this test.
    import ext.services.vector_store as vsm
    monkeypatch.setattr(
        "ext.services.sparse_embedder.embed_sparse_query",
        lambda t: ([1, 2, 3], [1.0, 1.0, 1.0]),
    )
    response = SimpleNamespace(points=[
        SimpleNamespace(id="p1", score=0.9, payload={"text": "hello"}),
    ])
    vs._client.query_points = AsyncMock(return_value=response)

    hits = await vs.hybrid_search(
        "kb_new", [0.1] * 1024, "pricing", limit=5, subtag_ids=[42],
    )

    assert len(hits) == 1 and hits[0].id == "p1"
    _, kwargs = vs._client.query_points.call_args
    prefetch = kwargs["prefetch"]
    assert len(prefetch) == 2, "hybrid must issue dense + sparse prefetch"
    # First arm: dense vector, using='dense', limit=limit*2.
    assert prefetch[0].using == "dense"
    assert prefetch[0].limit == 10  # limit (5) * 2
    # Second arm: sparse vector, using='bm25'.
    assert prefetch[1].using == "bm25"
    assert prefetch[1].limit == 10
    # Fusion = RRF.
    from qdrant_client.http import models as qm
    assert isinstance(kwargs["query"], qm.FusionQuery)
    assert kwargs["query"].fusion == qm.Fusion.RRF
    # Subtag filter propagates to both arms AND the outer filter.
    for arm in prefetch:
        # Look for FieldCondition(key='subtag_id') in must.
        musts = arm.filter.must or []
        assert any(
            getattr(m, "key", None) == "subtag_id" for m in musts
        ), "subtag_ids filter must propagate to prefetch arms"
    outer = kwargs["query_filter"]
    assert any(
        getattr(m, "key", None) == "subtag_id" for m in (outer.must or [])
    )


@pytest.mark.asyncio
async def test_hybrid_search_without_subtag_filter_still_excludes_deleted(monkeypatch) -> None:
    vs = _make_vs()
    monkeypatch.setattr(
        "ext.services.sparse_embedder.embed_sparse_query",
        lambda t: ([1], [1.0]),
    )
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    await vs.hybrid_search("kb_new", [0.0] * 1024, "q", limit=3)
    _, kwargs = vs._client.query_points.call_args
    # must_not must include deleted=True
    outer = kwargs["query_filter"]
    assert any(
        getattr(m, "key", None) == "deleted" for m in (outer.must_not or [])
    )


# ---------- upsert with sparse ----------------------------------------------


@pytest.mark.asyncio
async def test_upsert_without_sparse_uses_legacy_shape() -> None:
    """Default (no sparse_vector on points) keeps the unnamed-vector shape."""
    vs = _make_vs()
    vs._client.upsert = AsyncMock()
    await vs.upsert("kb_legacy", [
        {"id": "p1", "vector": [0.1, 0.2], "payload": {"text": "hi"}},
    ])
    _, kwargs = vs._client.upsert.call_args
    pt = kwargs["points"][0]
    # Legacy shape: vector is a raw list, not a named dict.
    assert pt.vector == [0.1, 0.2]


@pytest.mark.asyncio
async def test_upsert_with_sparse_on_hybrid_collection_writes_named_vectors() -> None:
    vs = _make_vs()
    vs._sparse_cache["kb_new"] = True  # pretend we already warmed the cache
    vs._client.upsert = AsyncMock()
    await vs.upsert("kb_new", [
        {
            "id": "p1",
            "vector": [0.1, 0.2],
            "payload": {"text": "hi"},
            "sparse_vector": ([10, 20], [0.5, 0.8]),
        },
    ])
    _, kwargs = vs._client.upsert.call_args
    pt = kwargs["points"][0]
    # Hybrid shape: vector is a dict with both dense and bm25 entries.
    assert isinstance(pt.vector, dict)
    assert "dense" in pt.vector and "bm25" in pt.vector
    assert pt.vector["dense"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_upsert_with_sparse_on_legacy_collection_falls_back() -> None:
    """If points carry sparse but collection lacks sparse support → legacy shape
    (sparse vector silently dropped — no dimension mismatch errors)."""
    vs = _make_vs()
    vs._sparse_cache["kb_legacy"] = False
    vs._client.upsert = AsyncMock()
    await vs.upsert("kb_legacy", [
        {
            "id": "p1",
            "vector": [0.1, 0.2],
            "payload": {"text": "hi"},
            "sparse_vector": ([1, 2], [0.5, 0.5]),
        },
    ])
    _, kwargs = vs._client.upsert.call_args
    pt = kwargs["points"][0]
    assert pt.vector == [0.1, 0.2]
