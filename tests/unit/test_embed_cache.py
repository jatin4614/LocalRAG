"""Unit tests for ``ext.services.embed_cache``.

Bug-fix campaign §3.2 — Redis-backed embedding cache. SHA256 key, TTL 7d,
behind ``RAG_EMBED_CACHE_ENABLED=0`` flag (default OFF for first deploy).

These tests run against the in-memory ``fake_redis`` fixture from
``tests/conftest.py`` so we don't need a live Redis server.
"""
from __future__ import annotations

import hashlib

import pytest


@pytest.mark.asyncio
async def test_get_or_set_cache_hit_on_second_call(fake_redis, monkeypatch):
    """First call invokes embedder and stores result; second call reads
    the cached vector and skips embedder entirely.

    Uses a counter on a stub embedder so we can assert it ran exactly
    once across two ``get_or_set`` invocations with the same text.
    """
    from ext.services import embed_cache

    monkeypatch.setenv("RAG_EMBED_CACHE_ENABLED", "1")
    monkeypatch.setattr(embed_cache, "_redis_singleton", fake_redis)

    calls = {"n": 0}

    class _StubEmbedder:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            calls["n"] += 1
            return [[0.1, 0.2, 0.3]] * len(texts)

    emb = _StubEmbedder()
    text = "hello world"
    model = "bge-m3-v1"

    v1 = await embed_cache.get_or_set(text, model, emb)
    v2 = await embed_cache.get_or_set(text, model, emb)

    assert v1 == [0.1, 0.2, 0.3]
    assert v2 == [0.1, 0.2, 0.3]
    # Embedder ran exactly once even though we called get_or_set twice.
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_get_or_set_disabled_when_flag_off(fake_redis, monkeypatch):
    """Default behaviour: flag off means every call runs the embedder."""
    from ext.services import embed_cache

    monkeypatch.delenv("RAG_EMBED_CACHE_ENABLED", raising=False)
    monkeypatch.setattr(embed_cache, "_redis_singleton", fake_redis)

    calls = {"n": 0}

    class _StubEmbedder:
        async def embed(self, texts):
            calls["n"] += 1
            return [[1.0, 2.0]] * len(texts)

    emb = _StubEmbedder()
    await embed_cache.get_or_set("foo", "v1", emb)
    await embed_cache.get_or_set("foo", "v1", emb)

    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_cache_key_isolates_text_and_model(fake_redis, monkeypatch):
    """Same text under two model versions must NOT collide.

    Same text under same model MUST hit cache.
    """
    from ext.services import embed_cache

    monkeypatch.setenv("RAG_EMBED_CACHE_ENABLED", "1")
    monkeypatch.setattr(embed_cache, "_redis_singleton", fake_redis)

    calls = {"n": 0}

    class _StubEmbedder:
        async def embed(self, texts):
            calls["n"] += 1
            return [[0.5]] * len(texts)

    emb = _StubEmbedder()
    await embed_cache.get_or_set("query", "model-a", emb)
    await embed_cache.get_or_set("query", "model-b", emb)
    await embed_cache.get_or_set("query", "model-a", emb)  # cache hit

    assert calls["n"] == 2  # one per distinct model version


def test_cache_key_format():
    """Key is ``embed:{model_version}:{sha256(text)[:16]}``.

    Lock the format because operators may grep Redis directly when
    debugging cache hits/misses.
    """
    from ext.services.embed_cache import _make_key

    key = _make_key("hello", "bge-m3")
    expected_digest = hashlib.sha256(b"hello").hexdigest()[:16]
    assert key == f"embed:bge-m3:{expected_digest}"


@pytest.mark.asyncio
async def test_redis_failure_falls_through_to_embedder(monkeypatch):
    """If Redis is unreachable / raises, ``get_or_set`` must fall through
    to the embedder rather than raise — fail-open per CLAUDE.md §1.2.
    """
    from ext.services import embed_cache

    monkeypatch.setenv("RAG_EMBED_CACHE_ENABLED", "1")

    class _BrokenRedis:
        async def get(self, key):
            raise ConnectionError("redis down")

        async def set(self, key, value, ex=None):
            raise ConnectionError("redis down")

    monkeypatch.setattr(embed_cache, "_redis_singleton", _BrokenRedis())

    class _StubEmbedder:
        async def embed(self, texts):
            return [[7.7]]

    out = await embed_cache.get_or_set("x", "m", _StubEmbedder())
    assert out == [7.7]


@pytest.mark.asyncio
async def test_ttl_set_to_seven_days(fake_redis, monkeypatch):
    """Expiry must be 7 days (604800s) so cache regenerates after a week
    even if the operator never bumps model_version. Caches longer can
    silently mask drift."""
    from ext.services import embed_cache

    monkeypatch.setenv("RAG_EMBED_CACHE_ENABLED", "1")
    monkeypatch.setattr(embed_cache, "_redis_singleton", fake_redis)

    class _StubEmbedder:
        async def embed(self, texts):
            return [[0.9]]

    await embed_cache.get_or_set("k", "v", _StubEmbedder())
    key = embed_cache._make_key("k", "v")
    assert await fake_redis.ttl(key) == 604800
