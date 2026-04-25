"""Unit tests for the redis-backed RBAC cache.

Sacred invariant: zero cross-user data leakage (CLAUDE.md §2). The cache
key MUST include user_id so two users can never share an entry.

These tests use a FakeRedis stand-in (no real network) and prove the
contract:

* set/get round-trip
* miss returns None
* TTL expiry is honoured
* invalidate drops the key AND publishes on the pub/sub channel
* TTL=0 disables the cache (operator escape hatch)
"""
from __future__ import annotations

import asyncio
import pytest

from ext.services.rbac_cache import RbacCache, CACHE_NAMESPACE


class FakeRedis:
    """In-memory stand-in with TTL + pub/sub semantics matching redis-py.

    Just enough for RbacCache: ``get``, ``setex``, ``delete``, ``publish``.
    TTL uses the real wall clock (no monkeypatch needed for sub-second
    expiry tests).
    """

    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, float]] = {}
        self.published: list[tuple[str, bytes]] = []

    async def get(self, key: str) -> bytes | None:
        import time
        v = self.store.get(key)
        if not v:
            return None
        val, exp = v
        if time.monotonic() > exp:
            del self.store[key]
            return None
        return val

    async def setex(self, key: str, ttl: int, value: bytes) -> None:
        import time
        self.store[key] = (value, time.monotonic() + ttl)

    async def delete(self, *keys: str) -> None:
        for k in keys:
            self.store.pop(k, None)

    async def publish(self, channel: str, msg: bytes) -> None:
        self.published.append((channel, msg))


@pytest.mark.asyncio
async def test_cache_stores_and_returns_allowed_ids():
    redis = FakeRedis()
    cache = RbacCache(redis=redis, ttl_sec=30)
    await cache.set(user_id="u1", allowed_kb_ids={1, 2, 3})
    got = await cache.get(user_id="u1")
    assert got == {1, 2, 3}


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    redis = FakeRedis()
    cache = RbacCache(redis=redis, ttl_sec=30)
    assert await cache.get(user_id="unknown") is None


@pytest.mark.asyncio
async def test_cache_ttl_expiry():
    """A 1s-TTL entry must be invisible to ``get`` after the TTL window.

    NOTE: the Plan A spec wrapped this in ``monkeypatch.setattr(time.monotonic, ...)``
    which froze the clock and broke both FakeRedis's expiry check AND
    ``asyncio.sleep`` (the event loop reads ``time.monotonic`` too). The
    correct test simply uses a sub-second TTL plus a real ``asyncio.sleep``;
    no clock manipulation needed.
    """
    redis = FakeRedis()
    cache = RbacCache(redis=redis, ttl_sec=1)
    await cache.set(user_id="u1", allowed_kb_ids={1})
    await asyncio.sleep(1.2)
    got = await cache.get(user_id="u1")
    assert got is None


@pytest.mark.asyncio
async def test_invalidate_drops_key_and_publishes_event():
    redis = FakeRedis()
    cache = RbacCache(redis=redis, ttl_sec=30)
    await cache.set(user_id="u1", allowed_kb_ids={1, 2})
    await cache.invalidate(user_ids=["u1"])
    assert await cache.get(user_id="u1") is None
    assert redis.published, "invalidate should publish a pubsub event"
    channel, msg = redis.published[0]
    assert channel.startswith("rbac:")
    assert b"u1" in msg


@pytest.mark.asyncio
async def test_cache_disabled_when_ttl_is_zero():
    """RAG_RBAC_CACHE_TTL_SECS=0 -> cache set/get are no-ops."""
    redis = FakeRedis()
    cache = RbacCache(redis=redis, ttl_sec=0)
    await cache.set(user_id="u1", allowed_kb_ids={1})
    assert await cache.get(user_id="u1") is None
