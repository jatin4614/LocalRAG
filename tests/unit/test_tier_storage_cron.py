"""Daily tier movement cron tests.

Plan B Phase 5.8 — verify the cron promotes shards to the correct tier
and is a no-op when the cached tier matches the desired tier.
"""
from __future__ import annotations

import datetime as dt

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_cron_promotes_shard_to_correct_tier(monkeypatch):
    from scripts.tier_storage_cron import process_collection

    vs = MagicMock()
    vs._client = MagicMock()
    vs._client.scroll = AsyncMock(return_value=([], None))

    apply_calls = []

    async def fake_apply(collection, shard_key, tier):
        apply_calls.append((collection, shard_key, tier))

    vs.apply_tier_config = fake_apply

    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)  # never moved before
    redis_client.set = AsyncMock()

    today = dt.date.today()
    # 2 shard_keys: one recent (hot), one 18 months old (cold)
    recent = f"{today.year:04d}-{today.month:02d}"
    eighteen_ago = today.replace(day=1) - dt.timedelta(days=540)
    old = f"{eighteen_ago.year:04d}-{eighteen_ago.month:02d}"

    await process_collection(
        vs=vs,
        redis_client=redis_client,
        collection="kb_1_v4",
        shard_keys=[recent, old],
    )

    tiers = {sk: tier for _, sk, tier in apply_calls}
    assert tiers[recent] == "hot"
    assert tiers[old] == "cold"


@pytest.mark.asyncio
async def test_cron_skips_already_correct_tier(monkeypatch):
    from scripts.tier_storage_cron import process_collection

    vs = MagicMock()
    apply_calls = []

    async def fake_apply(collection, shard_key, tier):
        apply_calls.append((collection, shard_key, tier))

    vs.apply_tier_config = fake_apply

    redis_client = MagicMock()
    today = dt.date.today()
    recent_sk = f"{today.year:04d}-{today.month:02d}"
    redis_client.get = AsyncMock(return_value="hot")  # already hot
    redis_client.set = AsyncMock()

    await process_collection(
        vs=vs,
        redis_client=redis_client,
        collection="kb_1_v4",
        shard_keys=[recent_sk],
    )

    assert apply_calls == []  # no-op
