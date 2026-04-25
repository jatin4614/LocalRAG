"""Unit tests for the ``rbac.resolved_allowed_kb_ids`` cache-first helper.

Phase 1.5 follow-up. This helper is the single source of truth for
"what kb_ids can this user read?" — both ``chat_rag_bridge`` and
``/api/rag/retrieve`` call it. The contract:

  1. Cache hit -> return the cached value, no DB call.
  2. Cache miss -> DB call, then cache write.
  3. Cache outage -> DB call (sacred CLAUDE.md §2 invariant — isolation
     can never weaken under cache failure).
  4. ``redis=None`` -> bypass cache entirely (DB-only path for tests
     that don't have a redis handle).
"""
from __future__ import annotations

import pytest


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.setexes: list[str] = []

    async def get(self, key: str) -> bytes | None:
        self.gets.append(key)
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: bytes) -> None:
        self.setexes.append(key)
        self.store[key] = value

    async def delete(self, *keys: str) -> None:
        for k in keys:
            self.store.pop(k, None)


class _BrokenRedis:
    """Raises on every call. Verifies fail-open behaviour."""

    async def get(self, *_a, **_kw):
        raise RuntimeError("simulated redis outage")

    async def setex(self, *_a, **_kw):
        raise RuntimeError("simulated redis outage")


class _FakeSession:
    """Minimal stand-in: supports the protocol that
    ``get_allowed_kb_ids`` would call. We patch the underlying
    ``get_allowed_kb_ids`` instead, so this only needs to be a
    placeholder for the type hint.
    """


@pytest.mark.asyncio
async def test_cache_hit_skips_db_call(monkeypatch):
    """On cache hit, we return the cached set without invoking the DB."""
    from ext.services import rbac, rbac_cache

    rbac_cache._reset_shared_cache_for_tests()
    redis = _FakeRedis()

    # Pre-populate the cache with [42, 99].
    cache = rbac_cache.get_shared_cache(redis=redis)
    await cache.set(user_id="u1", allowed_kb_ids={42, 99})

    db_called = []

    async def _stub_get_allowed_kb_ids(*_a, **_kw):
        db_called.append(True)
        return [1, 2, 3]

    monkeypatch.setattr(rbac, "get_allowed_kb_ids", _stub_get_allowed_kb_ids)

    out = await rbac.resolved_allowed_kb_ids(
        _FakeSession(), user_id="u1", redis=redis,
    )
    assert out == {42, 99}
    assert db_called == [], "cache hit must not call the DB"

    rbac_cache._reset_shared_cache_for_tests()


@pytest.mark.asyncio
async def test_cache_miss_falls_back_to_db_and_populates(monkeypatch):
    """On cache miss, we hit the DB and then write the cache."""
    from ext.services import rbac, rbac_cache

    rbac_cache._reset_shared_cache_for_tests()
    redis = _FakeRedis()

    async def _stub_get_allowed_kb_ids(*_a, **_kw):
        return [7, 8]

    monkeypatch.setattr(rbac, "get_allowed_kb_ids", _stub_get_allowed_kb_ids)

    out = await rbac.resolved_allowed_kb_ids(
        _FakeSession(), user_id="u9", redis=redis,
    )
    assert out == {7, 8}
    assert redis.gets == ["rbac:user:u9"], "should attempt cache get first"
    assert redis.setexes == ["rbac:user:u9"], "should populate cache on miss"

    rbac_cache._reset_shared_cache_for_tests()


@pytest.mark.asyncio
async def test_cache_outage_still_serves_correct_result(monkeypatch):
    """Sacred invariant: a broken redis must not weaken isolation —
    we still hit the DB and return the correct allowed set.
    """
    from ext.services import rbac, rbac_cache

    rbac_cache._reset_shared_cache_for_tests()

    async def _stub_get_allowed_kb_ids(*_a, **_kw):
        return [55]

    monkeypatch.setattr(rbac, "get_allowed_kb_ids", _stub_get_allowed_kb_ids)

    out = await rbac.resolved_allowed_kb_ids(
        _FakeSession(), user_id="u1", redis=_BrokenRedis(),
    )
    assert out == {55}, "cache outage must fall through to DB-only path"

    rbac_cache._reset_shared_cache_for_tests()


@pytest.mark.asyncio
async def test_redis_none_bypasses_cache(monkeypatch):
    """``redis=None`` -> DB-only, no cache calls anywhere."""
    from ext.services import rbac, rbac_cache

    rbac_cache._reset_shared_cache_for_tests()

    db_calls = 0

    async def _stub_get_allowed_kb_ids(*_a, **_kw):
        nonlocal db_calls
        db_calls += 1
        return [1]

    monkeypatch.setattr(rbac, "get_allowed_kb_ids", _stub_get_allowed_kb_ids)

    out = await rbac.resolved_allowed_kb_ids(
        _FakeSession(), user_id="u1", redis=None,
    )
    assert out == {1}
    assert db_calls == 1

    rbac_cache._reset_shared_cache_for_tests()
