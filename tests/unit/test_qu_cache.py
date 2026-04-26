"""Plan B Phase 4.5 — Redis DB 4 cache for QueryUnderstanding results."""
import pytest

from ext.services.qu_cache import (
    QUCache,
    _make_key,
    _normalize_for_cache,
)
from ext.services.query_understanding import QueryUnderstanding


def test_normalize_lowercases():
    assert _normalize_for_cache("OFC Roadmap") == "ofc roadmap"


def test_normalize_collapses_whitespace():
    assert _normalize_for_cache("hello   world  ") == "hello world"


def test_normalize_strips_trailing_punct():
    assert _normalize_for_cache("what is it??") == "what is it"


def test_normalize_safe_on_empty():
    assert _normalize_for_cache("") == ""
    assert _normalize_for_cache(None) == ""  # type: ignore[arg-type]


def test_make_key_deterministic():
    k1 = _make_key("compare budgets", "turn-42")
    k2 = _make_key("compare budgets", "turn-42")
    assert k1 == k2


def test_make_key_history_sensitive():
    k1 = _make_key("what about Q2", "turn-1")
    k2 = _make_key("what about Q2", "turn-2")
    assert k1 != k2


def test_make_key_normalizes_query():
    """Same content despite case + whitespace = same key."""
    k1 = _make_key("Compare  Budgets", "turn-42")
    k2 = _make_key("compare budgets", "turn-42")
    assert k1 == k2


def test_make_key_starts_with_namespace():
    assert _make_key("x", "y").startswith("qu:")


class TestQUCache:
    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=300)
        result = await cache.get("any query", "turn-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=300)
        qu = QueryUnderstanding(
            intent="global",
            resolved_query="compare budgets across years",
            temporal_constraint=None,
            entities=["budgets"],
            confidence=0.9,
            source="llm",
        )
        await cache.set("compare budgets", "turn-1", qu)
        result = await cache.get("compare budgets", "turn-1")
        assert result is not None
        assert result.intent == "global"
        assert result.cached is True  # set on retrieval

    @pytest.mark.asyncio
    async def test_disabled_cache_is_noop(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=300, enabled=False)
        qu = QueryUnderstanding(
            intent="global",
            resolved_query="x",
            temporal_constraint=None,
            entities=[],
            confidence=0.9,
        )
        await cache.set("q", "t", qu)
        assert await cache.get("q", "t") is None  # never returns

    @pytest.mark.asyncio
    async def test_ttl_applied_on_set(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=42)
        qu = QueryUnderstanding(
            intent="specific",
            resolved_query="x",
            temporal_constraint=None,
            entities=[],
            confidence=0.5,
        )
        await cache.set("q", "t", qu)
        ttl = await fake_redis.ttl(_make_key("q", "t"))
        assert 0 < ttl <= 42

    @pytest.mark.asyncio
    async def test_corrupt_cached_value_returns_none(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=300)
        # Manually inject garbage at the key
        await fake_redis.set(_make_key("q", "t"), "not json", ex=300)
        assert await cache.get("q", "t") is None

    @pytest.mark.asyncio
    async def test_redis_error_on_get_returns_none(self):
        """Connection refused / network glitch must soft-fail to None."""

        class _Broken:
            async def get(self, key):
                raise ConnectionError("broken pipe")

            async def set(self, key, value, ex=None):
                raise ConnectionError("broken pipe")

            async def ttl(self, key):
                return -1

        cache = QUCache(redis_client=_Broken(), ttl_secs=300)
        # Both directions soft-fail
        assert await cache.get("q", "t") is None
        # set must not raise even when the redis connection is broken
        qu = QueryUnderstanding(
            intent="specific",
            resolved_query="x",
            temporal_constraint=None,
            entities=[],
            confidence=0.5,
        )
        await cache.set("q", "t", qu)  # silently swallows

    @pytest.mark.asyncio
    async def test_cached_value_does_not_persist_cached_flag(self, fake_redis):
        """The cached=True marker must be stamped on retrieval, not at write."""
        cache = QUCache(redis_client=fake_redis, ttl_secs=300)
        qu = QueryUnderstanding(
            intent="specific",
            resolved_query="x",
            temporal_constraint=None,
            entities=[],
            confidence=0.5,
            cached=True,  # caller mistakenly set it
        )
        await cache.set("q", "t", qu)
        # Inspect the raw value — it should NOT contain "cached": true
        raw = await fake_redis.get(_make_key("q", "t"))
        assert '"cached": false' in raw or '"cached":false' in raw
