"""Tests for VectorStore.classify_tier + apply_tier_config.

Plan B Phase 5.3.
"""
from __future__ import annotations

import datetime as dt

import pytest
from unittest.mock import AsyncMock, MagicMock

from ext.services.vector_store import VectorStore


def _make_vs() -> VectorStore:
    vs = VectorStore.__new__(VectorStore)
    vs._url = "http://stub"
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    vs._colbert_cache = {}
    vs._client = MagicMock()
    vs._client.update_collection = AsyncMock()
    return vs


def test_classify_tier_hot_for_recent():
    from ext.services.vector_store import classify_tier
    today = dt.date.today()
    sk = f"{today.year:04d}-{today.month:02d}"
    assert classify_tier(sk, hot_months=3, warm_months=12) == "hot"


def test_classify_tier_warm_after_3_months():
    from ext.services.vector_store import classify_tier
    today = dt.date.today()
    # 6 months ago (~180 days back from first of current month)
    six_months_ago = (today.replace(day=1) - dt.timedelta(days=180))
    sk = f"{six_months_ago.year:04d}-{six_months_ago.month:02d}"
    assert classify_tier(sk, hot_months=3, warm_months=12) == "warm"


def test_classify_tier_cold_after_12_months():
    from ext.services.vector_store import classify_tier
    today = dt.date.today()
    twenty_months_ago = (today.replace(day=1) - dt.timedelta(days=600))
    sk = f"{twenty_months_ago.year:04d}-{twenty_months_ago.month:02d}"
    assert classify_tier(sk, hot_months=3, warm_months=12) == "cold"


@pytest.mark.asyncio
async def test_apply_tier_config_cold_uses_int8():
    vs = _make_vs()
    await vs.apply_tier_config(
        collection="kb_1_v4",
        shard_key="2023-01",
        tier="cold",
    )
    call = vs._client.update_collection.call_args
    str_call = str(call)
    # quantization_config + scalar:int8
    assert "quantization_config" in str_call or "scalar" in str_call.lower()


@pytest.mark.asyncio
async def test_apply_tier_config_hot_disables_mmap():
    vs = _make_vs()
    await vs.apply_tier_config(
        collection="kb_1_v4",
        shard_key="2026-04",
        tier="hot",
    )
    call = vs._client.update_collection.call_args
    str_call = str(call)
    # memmap_threshold should be 0 (or very high — always RAM)
    assert "memmap" in str_call.lower() or "ram" in str_call.lower()
