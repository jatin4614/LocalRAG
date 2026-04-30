"""Unit tests for P2.4 HNSW tuning + async client pool config.

All env-tunable knobs are lazy (read at call time), so each test can just
monkeypatch ``os.environ`` and assert the resulting Qdrant call shape. No
real network I/O — ``AsyncQdrantClient`` is patched / mocked throughout.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models as qm

from ext.services.vector_store import VectorStore


def _make_vs() -> VectorStore:
    """Build a VectorStore bypassing the real AsyncQdrantClient constructor."""
    vs = VectorStore.__new__(VectorStore)
    vs._client = MagicMock()
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    return vs


# ---------- ensure_collection: HNSW config on create ------------------------


@pytest.mark.asyncio
async def test_ensure_collection_passes_default_hnsw_config(monkeypatch) -> None:
    """Default call → HnswConfigDiff(m=16, ef_construct=200, full_scan_threshold=10000).

    ef_construct=200 is the *only* Qdrant default we deliberately change
    (up from 100 → +2-3pp recall at modest index-time cost).
    """
    # Clear any prior env that could shadow the baseline.
    for v in (
        "RAG_QDRANT_M",
        "RAG_QDRANT_EF_CONSTRUCT",
        "RAG_QDRANT_FULL_SCAN_THRESHOLD",
        "RAG_QDRANT_ON_DISK_PAYLOAD",
    ):
        monkeypatch.delenv(v, raising=False)

    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_default")

    _, kwargs = vs._client.create_collection.call_args
    hnsw = kwargs["hnsw_config"]
    assert isinstance(hnsw, qm.HnswConfigDiff)
    assert hnsw.m == 16
    assert hnsw.ef_construct == 200  # bumped from Qdrant's 100 default
    assert hnsw.full_scan_threshold == 10000
    # on_disk_payload defaults to False (keep payload in RAM → lower latency).
    assert kwargs["on_disk_payload"] is False


@pytest.mark.asyncio
async def test_ensure_collection_honors_m_env_override(monkeypatch) -> None:
    """RAG_QDRANT_M=32 must flow into HnswConfigDiff.m."""
    monkeypatch.setenv("RAG_QDRANT_M", "32")
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_high_recall")

    _, kwargs = vs._client.create_collection.call_args
    assert kwargs["hnsw_config"].m == 32


@pytest.mark.asyncio
async def test_ensure_collection_honors_ef_construct_env_override(monkeypatch) -> None:
    monkeypatch.setenv("RAG_QDRANT_EF_CONSTRUCT", "400")
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_x")

    _, kwargs = vs._client.create_collection.call_args
    assert kwargs["hnsw_config"].ef_construct == 400


@pytest.mark.asyncio
async def test_ensure_collection_honors_on_disk_payload(monkeypatch) -> None:
    monkeypatch.setenv("RAG_QDRANT_ON_DISK_PAYLOAD", "true")
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_big")

    _, kwargs = vs._client.create_collection.call_args
    assert kwargs["on_disk_payload"] is True


@pytest.mark.asyncio
async def test_ensure_collection_with_sparse_also_passes_hnsw_config(monkeypatch) -> None:
    """Hybrid collections (dense + sparse) must receive the same tuning."""
    monkeypatch.delenv("RAG_QDRANT_M", raising=False)
    monkeypatch.delenv("RAG_QDRANT_EF_CONSTRUCT", raising=False)
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_hybrid", with_sparse=True)

    _, kwargs = vs._client.create_collection.call_args
    assert "sparse_vectors_config" in kwargs
    hnsw = kwargs["hnsw_config"]
    assert hnsw.m == 16
    assert hnsw.ef_construct == 200


# ---------- search: per-query SearchParams(hnsw_ef) -------------------------


@pytest.mark.asyncio
async def test_search_passes_default_hnsw_ef(monkeypatch) -> None:
    """Default search → SearchParams(hnsw_ef=128)."""
    monkeypatch.delenv("RAG_QDRANT_EF", raising=False)
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.search("kb_legacy", [0.0] * 1024, limit=5)

    _, kwargs = vs._client.query_points.call_args
    sp = kwargs["search_params"]
    assert isinstance(sp, qm.SearchParams)
    assert sp.hnsw_ef == 128


@pytest.mark.asyncio
async def test_search_honors_ef_env_override(monkeypatch) -> None:
    monkeypatch.setenv("RAG_QDRANT_EF", "256")
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.search("kb_x", [0.0] * 1024)

    _, kwargs = vs._client.query_points.call_args
    assert kwargs["search_params"].hnsw_ef == 256


@pytest.mark.asyncio
async def test_hybrid_search_dense_prefetch_carries_search_params(monkeypatch) -> None:
    """Dense prefetch arm must get SearchParams(hnsw_ef). Sparse arm stays unset
    (BM25 has no HNSW to tune)."""
    monkeypatch.setenv("RAG_QDRANT_EF", "200")
    monkeypatch.setattr(
        "ext.services.sparse_embedder.embed_sparse_query",
        lambda t: ([1], [1.0]),
    )
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.hybrid_search("kb_hybrid", [0.0] * 1024, "q", limit=5)

    _, kwargs = vs._client.query_points.call_args
    prefetch = kwargs["prefetch"]
    # Dense = index 0, sparse = index 1 (per hybrid_search construction).
    dense = prefetch[0]
    sparse = prefetch[1]
    assert dense.using == "dense"
    assert isinstance(dense.params, qm.SearchParams)
    assert dense.params.hnsw_ef == 200
    # Sparse prefetch does not need/use SearchParams — leave it None.
    assert sparse.params is None


# ---------- __init__: client timeout + pool size ----------------------------


def test_init_configures_timeout_and_pool_size() -> None:
    """VectorStore must pass timeout=120.0 (default) and pool_size to AsyncQdrantClient."""
    with patch("ext.services.vector_store.AsyncQdrantClient") as mock_cls:
        VectorStore(url="http://qdrant:6333", vector_size=1024)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        assert kwargs["url"] == "http://qdrant:6333"
        # 2026-04-29 — bumped 30s → 120s. The 30s default was empirically
        # too tight for cluster-mode sharded writes + ColBERT multi-vector
        # payloads; soak fixed via batched upsert (RAG_UPSERT_BATCH=16) but
        # the wider ceiling stays as a safety margin. Override via
        # RAG_QDRANT_TIMEOUT.
        assert kwargs["timeout"] == 120.0
        # Default pool_size = 32.
        assert kwargs["pool_size"] == 32


def test_init_honors_max_conns_env_override(monkeypatch) -> None:
    monkeypatch.setenv("RAG_QDRANT_MAX_CONNS", "128")
    with patch("ext.services.vector_store.AsyncQdrantClient") as mock_cls:
        VectorStore(url="http://qdrant:6333", vector_size=1024)
        _, kwargs = mock_cls.call_args
        assert kwargs["pool_size"] == 128


# ---------- optimize_collection ---------------------------------------------


@pytest.mark.asyncio
async def test_optimize_collection_sends_indexing_threshold_zero() -> None:
    vs = _make_vs()
    vs._client.update_collection = AsyncMock()

    await vs.optimize_collection("kb_rebuild")

    _, kwargs = vs._client.update_collection.call_args
    assert kwargs["collection_name"] == "kb_rebuild"
    oc = kwargs["optimizer_config"]
    assert isinstance(oc, qm.OptimizersConfigDiff)
    assert oc.indexing_threshold == 0


@pytest.mark.asyncio
async def test_optimize_collection_swallows_errors() -> None:
    """Best-effort admin helper must not raise on failure."""
    vs = _make_vs()
    vs._client.update_collection = AsyncMock(side_effect=RuntimeError("boom"))

    # Must return cleanly.
    await vs.optimize_collection("kb_missing")
