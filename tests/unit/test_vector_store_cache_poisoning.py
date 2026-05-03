"""Unit tests for transient-error cache-poisoning fix in VectorStore.

Background — 2026-05-03 incident:
    During a brief window when Qdrant had auth-on but the deployed
    ``open-webui`` image was a pre-Wave-1a build (didn't pass ``api_key`` to
    ``AsyncQdrantClient``), every ``get_collection`` call returned 401. The
    three lazy "does this collection have feature X?" caches in
    ``vector_store.py`` swallowed the 401 with a bare ``except Exception``
    and cached ``False`` for the lifetime of the process. After auth was
    disabled, search continued to use the wrong vector shape until
    ``open-webui`` was restarted.

Contract under test:
    * If the underlying error is an ``UnexpectedResponse`` with
      ``status_code == 404`` (collection truly missing), cache ``False`` —
      preserving the legitimate-missing fallback.
    * For ANY OTHER exception (401, 403, 5xx, network errors, timeouts),
      return ``False`` for *this* call but DO NOT write to the cache, so
      the next call re-probes and recovers automatically once Qdrant is
      healthy again.

The three caches:
    * ``_refresh_sparse_cache`` → ``_sparse_cache``
    * ``_refresh_colbert_cache`` → ``_colbert_cache``
    * ``_is_custom_sharded`` → ``_sharding_cache``
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from ext.services.vector_store import VectorStore


# --------- helpers -----------------------------------------------------------


def _make_vs() -> VectorStore:
    """VectorStore instance with empty caches (mirrors test_vector_store_hybrid)."""
    vs = VectorStore.__new__(VectorStore)
    vs._client = MagicMock()
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    vs._colbert_cache = {}
    vs._sharding_cache = {}
    return vs


def _unexpected_response(status_code: int) -> UnexpectedResponse:
    """Build a real qdrant-client UnexpectedResponse with the given status."""
    return UnexpectedResponse(
        status_code=status_code,
        reason_phrase="HTTP error",
        content=b"",
        headers=None,
    )


def _info_with_sparse(has_sparse: bool):
    sparse = {"bm25": object()} if has_sparse else None
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(sparse_vectors=sparse),
        ),
    )


def _info_with_colbert(has_colbert: bool):
    """CollectionInfo whose ``params.vectors`` includes a colbert slot iff ``has_colbert``."""
    if has_colbert:
        slot = SimpleNamespace(multivector_config=object())
        vectors = {"dense": SimpleNamespace(multivector_config=None), "colbert": slot}
    else:
        vectors = {"dense": SimpleNamespace(multivector_config=None)}
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=vectors),
        ),
    )


def _info_with_sharding(method: str | None):
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(sharding_method=method),
        ),
    )


# --------- _refresh_sparse_cache --------------------------------------------


@pytest.mark.asyncio
async def test_refresh_sparse_cache_does_not_poison_on_401() -> None:
    """401 (auth fail) returns False BUT the cache stays empty so a later
    successful call re-probes and gets the right answer.

    This is the regression for the 2026-05-03 incident. Pre-fix, a 401 would
    write ``False`` permanently, so even after auth was reattached the
    collection looked "no sparse vector" forever.
    """
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(401))

    # First call fails — must return False, must NOT cache.
    assert await vs._refresh_sparse_cache("kb_42") is False
    assert "kb_42" not in vs._sparse_cache

    # Second call succeeds — must re-probe, return True, AND cache it.
    vs._client.get_collection = AsyncMock(return_value=_info_with_sparse(True))
    assert await vs._refresh_sparse_cache("kb_42") is True
    assert vs._sparse_cache["kb_42"] is True


@pytest.mark.asyncio
async def test_refresh_sparse_cache_does_not_poison_on_403() -> None:
    """403 (forbidden) is also transient w.r.t. cache — no poisoning."""
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(403))
    assert await vs._refresh_sparse_cache("kb_secret") is False
    assert "kb_secret" not in vs._sparse_cache


@pytest.mark.asyncio
async def test_refresh_sparse_cache_does_not_poison_on_500() -> None:
    """5xx server errors are transient — no poisoning."""
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(503))
    assert await vs._refresh_sparse_cache("kb_x") is False
    assert "kb_x" not in vs._sparse_cache


@pytest.mark.asyncio
async def test_refresh_sparse_cache_does_not_poison_on_network_error() -> None:
    """Plain ConnectionError / TimeoutError → no poisoning."""
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=ConnectionError("dns lookup failed"))
    assert await vs._refresh_sparse_cache("kb_n") is False
    assert "kb_n" not in vs._sparse_cache


@pytest.mark.asyncio
async def test_refresh_sparse_cache_caches_false_on_404() -> None:
    """404 (collection truly does not exist) IS a permanent answer — cache it.

    This is the legitimate-missing path; it must keep working so the read
    side falls back to dense-only without re-probing every request.
    """
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(404))
    assert await vs._refresh_sparse_cache("kb_gone") is False
    assert vs._sparse_cache["kb_gone"] is False


# --------- _refresh_colbert_cache --------------------------------------------


@pytest.mark.asyncio
async def test_refresh_colbert_cache_does_not_poison_on_401() -> None:
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(401))
    assert await vs._refresh_colbert_cache("kb_42") is False
    assert "kb_42" not in vs._colbert_cache

    vs._client.get_collection = AsyncMock(return_value=_info_with_colbert(True))
    assert await vs._refresh_colbert_cache("kb_42") is True
    assert vs._colbert_cache["kb_42"] is True


@pytest.mark.asyncio
async def test_refresh_colbert_cache_does_not_poison_on_403() -> None:
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(403))
    assert await vs._refresh_colbert_cache("kb_x") is False
    assert "kb_x" not in vs._colbert_cache


@pytest.mark.asyncio
async def test_refresh_colbert_cache_caches_false_on_404() -> None:
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(404))
    assert await vs._refresh_colbert_cache("kb_gone") is False
    assert vs._colbert_cache["kb_gone"] is False


# --------- _is_custom_sharded -----------------------------------------------


@pytest.mark.asyncio
async def test_is_custom_sharded_does_not_poison_on_401() -> None:
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(401))
    assert await vs._is_custom_sharded("kb_1_v4") is False
    assert "kb_1_v4" not in vs._sharding_cache

    # Recovery: now the call succeeds and returns a custom-sharded collection.
    vs._client.get_collection = AsyncMock(return_value=_info_with_sharding("custom"))
    assert await vs._is_custom_sharded("kb_1_v4") is True
    assert vs._sharding_cache["kb_1_v4"] is True


@pytest.mark.asyncio
async def test_is_custom_sharded_does_not_poison_on_403() -> None:
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(403))
    assert await vs._is_custom_sharded("kb_x") is False
    assert "kb_x" not in vs._sharding_cache


@pytest.mark.asyncio
async def test_is_custom_sharded_does_not_poison_on_network_error() -> None:
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=TimeoutError("qdrant down"))
    assert await vs._is_custom_sharded("kb_x") is False
    assert "kb_x" not in vs._sharding_cache


@pytest.mark.asyncio
async def test_is_custom_sharded_caches_false_on_404() -> None:
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(404))
    assert await vs._is_custom_sharded("kb_gone") is False
    assert vs._sharding_cache["kb_gone"] is False


# --------- WARNING-level log emitted on transient failure -------------------


@pytest.mark.asyncio
async def test_refresh_sparse_cache_logs_warning_on_transient(caplog) -> None:
    """Operator needs to see the failure to correlate with the auth incident."""
    import logging
    vs = _make_vs()
    vs._client.get_collection = AsyncMock(side_effect=_unexpected_response(401))
    with caplog.at_level(logging.WARNING, logger="ext.services.vector_store"):
        await vs._refresh_sparse_cache("kb_log")
    # At least one WARNING with the collection name in the message.
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("kb_log" in m for m in msgs), f"no kb_log mention in {msgs!r}"
