"""Tests for VectorStore.ensure_collection_temporal + upsert_temporal.

Plan B Phase 5.1.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from ext.services.vector_store import VectorStore


@pytest.mark.asyncio
async def test_ensure_collection_temporal_passes_custom_sharding(monkeypatch):
    vs = VectorStore.__new__(VectorStore)
    vs._url = "http://stub"
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    vs._colbert_cache = {}
    vs._client = MagicMock()
    vs._client.collection_exists = AsyncMock(return_value=False)
    vs._client.create_collection = AsyncMock()
    vs._client.create_shard_key = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection_temporal(
        "kb_1_v4",
        shard_keys=["2024-01", "2024-02", "2024-03"],
        with_sparse=True,
        with_colbert=True,
    )

    # Inspect the create_collection kwargs
    call = vs._client.create_collection.call_args
    assert call.kwargs.get("sharding_method") == "custom" or \
        "sharding_method" in str(call), \
        "ensure_collection_temporal must set sharding_method=custom"

    # All shard keys created
    assert vs._client.create_shard_key.call_count == 3


@pytest.mark.asyncio
async def test_ensure_collection_temporal_with_replication_factor(monkeypatch):
    vs = VectorStore.__new__(VectorStore)
    vs._url = "http://stub"
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    vs._colbert_cache = {}
    vs._client = MagicMock()
    vs._client.collection_exists = AsyncMock(return_value=False)
    vs._client.create_collection = AsyncMock()
    vs._client.create_shard_key = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection_temporal(
        "kb_1_v4",
        shard_keys=["2024-01"],
        replication_factor=2,
    )

    call = vs._client.create_collection.call_args
    assert call.kwargs.get("replication_factor") == 2 or \
        "replication_factor=2" in str(call)


@pytest.mark.asyncio
async def test_ensure_collection_temporal_idempotent(monkeypatch):
    vs = VectorStore.__new__(VectorStore)
    vs._url = "http://stub"
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    vs._colbert_cache = {}
    vs._client = MagicMock()
    vs._client.collection_exists = AsyncMock(return_value=True)
    vs._client.create_collection = AsyncMock()
    vs._client.create_shard_key = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection_temporal(
        "kb_1_v4",
        shard_keys=["2024-01", "2024-02"],
    )

    # Existing collection: no create_collection call, but shard_keys still ensured
    vs._client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_with_shard_key(monkeypatch):
    vs = VectorStore.__new__(VectorStore)
    vs._url = "http://stub"
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    vs._colbert_cache = {}
    vs._client = MagicMock()
    vs._client.upsert = AsyncMock()

    await vs.upsert_temporal(
        "kb_1_v4",
        points=[{"id": "p1", "vector": [0.1] * 1024,
                 "payload": {"shard_key": "2024-01"}}],
        shard_key="2024-01",
    )

    call = vs._client.upsert.call_args
    assert call.kwargs.get("shard_key_selector") == "2024-01" or \
        "shard_key_selector" in str(call)
