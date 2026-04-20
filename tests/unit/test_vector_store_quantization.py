"""Unit tests for P3.2 scalar INT8 quantization + per-query rescoring.

Covers both sides of the quantization feature:

1. ``ensure_collection`` — creates collections with
   ``ScalarQuantization(INT8, quantile=0.99, always_ram=True)`` when the
   caller opts in (kwarg) or the ``RAG_QDRANT_QUANTIZE`` env is set.
2. ``search`` / ``hybrid_search`` — attach
   ``QuantizationSearchParams(rescore=True, oversampling=2.0)`` to the
   SearchParams by default. Togglable per-call and via env.

As with the P2.4 tests, AsyncQdrantClient is mocked throughout — no real
Qdrant, no network I/O.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


def _clear_quant_env(monkeypatch) -> None:
    """Start every test from an unset-env baseline."""
    for v in (
        "RAG_QDRANT_QUANTIZE",
        "RAG_QDRANT_RESCORE",
        "RAG_QDRANT_QUANTILE",
        "RAG_QDRANT_OVERSAMPLING",
    ):
        monkeypatch.delenv(v, raising=False)


# ---------- ensure_collection: quantization_config injection -----------------


@pytest.mark.asyncio
async def test_ensure_collection_default_passes_none_quantization(monkeypatch) -> None:
    """With no kwarg and no env, quantization_config must be None.

    Critical for byte-identical backward compat — legacy deployments that
    don't know about quantization must keep creating plain collections.
    """
    _clear_quant_env(monkeypatch)
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_plain")

    _, kwargs = vs._client.create_collection.call_args
    assert kwargs["quantization_config"] is None
    # And the dense VectorParams must NOT have on_disk=True (that pairing is
    # exclusive to quantized collections).
    vp = kwargs["vectors_config"]
    assert isinstance(vp, qm.VectorParams)
    assert vp.on_disk is None


@pytest.mark.asyncio
async def test_ensure_collection_with_quantization_kwarg(monkeypatch) -> None:
    """with_quantization=True → ScalarQuantization(INT8, 0.99, always_ram=True)."""
    _clear_quant_env(monkeypatch)
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_q", with_quantization=True)

    _, kwargs = vs._client.create_collection.call_args
    qc = kwargs["quantization_config"]
    assert isinstance(qc, qm.ScalarQuantization)
    assert isinstance(qc.scalar, qm.ScalarQuantizationConfig)
    assert qc.scalar.type == qm.ScalarType.INT8
    assert qc.scalar.quantile == 0.99
    assert qc.scalar.always_ram is True

    # Dense vectors spill to disk — originals on disk, quantized index in RAM.
    vp = kwargs["vectors_config"]
    assert isinstance(vp, qm.VectorParams)
    assert vp.on_disk is True


@pytest.mark.asyncio
async def test_ensure_collection_env_triggers_quantization(monkeypatch) -> None:
    """RAG_QDRANT_QUANTIZE=1 without any kwarg → quantization applied."""
    _clear_quant_env(monkeypatch)
    monkeypatch.setenv("RAG_QDRANT_QUANTIZE", "1")
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_env")

    _, kwargs = vs._client.create_collection.call_args
    qc = kwargs["quantization_config"]
    assert isinstance(qc, qm.ScalarQuantization)
    assert qc.scalar.type == qm.ScalarType.INT8


@pytest.mark.asyncio
async def test_ensure_collection_kwarg_overrides_env(monkeypatch) -> None:
    """with_quantization=False must force-off even if env says '1'."""
    _clear_quant_env(monkeypatch)
    monkeypatch.setenv("RAG_QDRANT_QUANTIZE", "1")
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_forced_off", with_quantization=False)

    _, kwargs = vs._client.create_collection.call_args
    assert kwargs["quantization_config"] is None


@pytest.mark.asyncio
async def test_ensure_collection_honors_quantile_env(monkeypatch) -> None:
    """RAG_QDRANT_QUANTILE tweaks the outlier cutoff."""
    _clear_quant_env(monkeypatch)
    monkeypatch.setenv("RAG_QDRANT_QUANTILE", "0.95")
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_q95", with_quantization=True)

    _, kwargs = vs._client.create_collection.call_args
    qc = kwargs["quantization_config"]
    assert qc.scalar.quantile == 0.95


@pytest.mark.asyncio
async def test_ensure_collection_hybrid_plus_quantization(monkeypatch) -> None:
    """with_sparse=True + with_quantization=True must stack cleanly."""
    _clear_quant_env(monkeypatch)
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection(
        "kb_hybrid_q", with_sparse=True, with_quantization=True
    )

    _, kwargs = vs._client.create_collection.call_args
    # Sparse config intact.
    assert "sparse_vectors_config" in kwargs
    # Quantization attached to the outer call (applies to the named dense vec).
    qc = kwargs["quantization_config"]
    assert isinstance(qc, qm.ScalarQuantization)
    # Dense under the named ``dense`` key must have on_disk=True.
    vcfg = kwargs["vectors_config"]
    assert isinstance(vcfg, dict) and "dense" in vcfg
    assert vcfg["dense"].on_disk is True


# ---------- search: QuantizationSearchParams in SearchParams -----------------


@pytest.mark.asyncio
async def test_search_attaches_quantization_search_params_by_default(monkeypatch) -> None:
    """Default: rescore=True, oversampling=2.0 on every search."""
    _clear_quant_env(monkeypatch)
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.search("kb_q", [0.0] * 1024, limit=5)

    _, kwargs = vs._client.query_points.call_args
    sp = kwargs["search_params"]
    assert isinstance(sp, qm.SearchParams)
    assert sp.hnsw_ef == 128
    assert isinstance(sp.quantization, qm.QuantizationSearchParams)
    assert sp.quantization.rescore is True
    assert sp.quantization.oversampling == 2.0


@pytest.mark.asyncio
async def test_search_rescore_false_disables_quantization_hint(monkeypatch) -> None:
    """rescore=False → no QuantizationSearchParams attached."""
    _clear_quant_env(monkeypatch)
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.search("kb_q", [0.0] * 1024, limit=5, rescore=False)

    _, kwargs = vs._client.query_points.call_args
    sp = kwargs["search_params"]
    assert sp.quantization is None
    # hnsw_ef still attached — the two knobs are orthogonal.
    assert sp.hnsw_ef == 128


@pytest.mark.asyncio
async def test_search_rescore_env_override_disables_globally(monkeypatch) -> None:
    """RAG_QDRANT_RESCORE=0 turns off rescoring for all callers by default."""
    _clear_quant_env(monkeypatch)
    monkeypatch.setenv("RAG_QDRANT_RESCORE", "0")
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.search("kb_q", [0.0] * 1024)

    _, kwargs = vs._client.query_points.call_args
    assert kwargs["search_params"].quantization is None


@pytest.mark.asyncio
async def test_search_rescore_true_wins_over_env_off(monkeypatch) -> None:
    """Explicit rescore=True beats RAG_QDRANT_RESCORE=0."""
    _clear_quant_env(monkeypatch)
    monkeypatch.setenv("RAG_QDRANT_RESCORE", "0")
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.search("kb_q", [0.0] * 1024, rescore=True)

    _, kwargs = vs._client.query_points.call_args
    sp = kwargs["search_params"]
    assert sp.quantization is not None
    assert sp.quantization.rescore is True


@pytest.mark.asyncio
async def test_search_honors_oversampling_env_override(monkeypatch) -> None:
    """RAG_QDRANT_OVERSAMPLING=3.0 must flow into QuantizationSearchParams."""
    _clear_quant_env(monkeypatch)
    monkeypatch.setenv("RAG_QDRANT_OVERSAMPLING", "3.0")
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.search("kb_q", [0.0] * 1024)

    _, kwargs = vs._client.query_points.call_args
    sp = kwargs["search_params"]
    assert sp.quantization.oversampling == 3.0


# ---------- hybrid_search: dense prefetch arm gets quantization --------------


@pytest.mark.asyncio
async def test_hybrid_search_dense_arm_gets_quantization_by_default(monkeypatch) -> None:
    """Dense prefetch arm must carry QuantizationSearchParams; sparse stays None."""
    _clear_quant_env(monkeypatch)
    monkeypatch.setattr(
        "ext.services.sparse_embedder.embed_sparse_query",
        lambda t: ([1], [1.0]),
    )
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.hybrid_search("kb_hybrid_q", [0.0] * 1024, "some query", limit=5)

    _, kwargs = vs._client.query_points.call_args
    prefetch = kwargs["prefetch"]
    dense, sparse = prefetch[0], prefetch[1]
    assert dense.using == "dense"
    assert isinstance(dense.params, qm.SearchParams)
    assert dense.params.quantization is not None
    assert dense.params.quantization.rescore is True
    assert dense.params.quantization.oversampling == 2.0
    # Sparse prefetch never gets SearchParams (no HNSW, no quantization).
    assert sparse.params is None


@pytest.mark.asyncio
async def test_hybrid_search_rescore_false_disables_dense_quantization(monkeypatch) -> None:
    _clear_quant_env(monkeypatch)
    monkeypatch.setattr(
        "ext.services.sparse_embedder.embed_sparse_query",
        lambda t: ([1], [1.0]),
    )
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await vs.hybrid_search(
        "kb_hybrid_q", [0.0] * 1024, "some query", limit=5, rescore=False
    )

    _, kwargs = vs._client.query_points.call_args
    dense = kwargs["prefetch"][0]
    assert dense.params.quantization is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
