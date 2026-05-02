"""Redis-backed embedding cache.

Bug-fix campaign §3.2 — wraps single-text embedding calls so identical
``(text, model_version)`` pairs reuse the prior dense vector instead of
re-running TEI. Useful for repeat queries and ingest deduplication
(e.g. per-chunk image-caption embeddings that recur across documents).

Behind ``RAG_EMBED_CACHE_ENABLED`` (default ``0`` for safe first deploy).

Key shape:
    ``embed:{model_version}:{sha256(text)[:16]}``

Hash truncation: 16 hex chars = 64 bits of namespace, ample collision
resistance for embedding-cache scale (~10^7 chunks). The ``model_version``
prefix isolates entries when operators flip ``EMBED_MODEL`` (or stamp a
new ``pipeline_version``).

TTL: 7 days. Long enough to absorb repeat ingest of the same blob and
hot-query cache reuse, short enough that an operator silently flipping
``EMBED_MODEL`` without bumping the version string still sees the cache
regenerate within a week (defence-in-depth — versioning is the primary
guard).

Fail-open: any redis error (timeout, ConnectionError, broken protocol)
falls through to a direct embedder call without raising. Mirrors
``ext.services.qu_cache.QUCache`` semantics — never let the cache layer
take retrieval down.

Wiring: call sites import :func:`get_or_set` and substitute it for a
direct ``embedder.embed([text])`` call. Plumbed into
``ext/services/embedder.py:TEIEmbedder.embed`` for the single-text
(query) path; batch ingest paths can opt in by wrapping per-text.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Optional

log = logging.getLogger("orgchat.embed_cache")


# 7 days in seconds — see module docstring rationale.
_TTL_SECONDS = 7 * 24 * 60 * 60  # 604800


# Lazily-initialized singleton — first call opens the redis connection,
# subsequent calls return the same handle. Reset to None in tests via
# ``monkeypatch.setattr(embed_cache, "_redis_singleton", fake_redis)``.
#
# Mirrors ``chat_rag_bridge._qu_cache_singleton`` double-checked-locking.
_redis_singleton = None
_REDIS_LOCK = threading.Lock()


def _get_redis():
    """Return the process-wide async redis handle, or ``None`` if the
    cache is disabled or redis cannot be reached. Soft-fails so cache
    infra issues never raise from :func:`get_or_set`.
    """
    global _redis_singleton
    if _redis_singleton is not None:
        return _redis_singleton
    if os.environ.get("RAG_EMBED_CACHE_ENABLED", "0") != "1":
        return None
    with _REDIS_LOCK:
        # Re-check inside the lock — another caller may have raced past
        # the first None-check and finished the init while we were blocked.
        if _redis_singleton is not None:
            return _redis_singleton
        try:
            import redis.asyncio as _redis

            url = os.environ.get("REDIS_URL", "redis://redis:6379")
            # Strip any trailing /<db> suffix and apply the dedicated DB.
            # Re-uses the chat_rag_bridge URL-parsing convention.
            if "/" in url.rsplit("@", 1)[-1].split("//", 1)[-1]:
                base = url.rsplit("/", 1)[0]
            else:
                base = url
            db = int(os.environ.get("RAG_EMBED_CACHE_REDIS_DB", "5"))
            _redis_singleton = _redis.from_url(
                f"{base}/{db}", decode_responses=True,
            )
            return _redis_singleton
        except Exception as e:  # pragma: no cover — defensive
            log.warning("embed_cache init failed (%s) — running without cache", e)
            return None


def _make_key(text: str, model_version: str) -> str:
    """Build the Redis key for ``(text, model_version)``.

    Operators may grep keys directly when debugging — keep the format
    locked (asserted in the unit suite).
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"embed:{model_version}:{digest}"


async def get_or_set(text: str, model_version: str, embedder) -> list[float]:
    """Return the cached dense vector for ``text`` under ``model_version``,
    falling through to ``embedder.embed([text])[0]`` on miss / disabled.

    ``embedder`` must implement the :class:`Embedder` protocol from
    ``ext.services.embedder`` (i.e. ``async embed(texts: list[str]) ->
    list[list[float]]``).

    Behaviour:
        * Cache hit: returns the stored vector — embedder is NOT called.
        * Cache miss: calls embedder, stores result with TTL 7d, returns it.
        * Cache disabled / redis unreachable / corrupt cached value:
          falls through to embedder; never raises from this layer.
    """
    # Re-check the flag at call time so an operator flipping
    # ``RAG_EMBED_CACHE_ENABLED=0`` after process start (or test setup
    # injecting a fake redis without flipping the flag on) gets the
    # passthrough behaviour they expect.
    if os.environ.get("RAG_EMBED_CACHE_ENABLED", "0") != "1":
        out = await embedder.embed([text])
        return out[0] if out else []

    redis = _get_redis()
    if redis is None:
        # Cache disabled or init failed — straight passthrough.
        out = await embedder.embed([text])
        return out[0] if out else []

    key = _make_key(text, model_version)
    try:
        raw = await redis.get(key)
    except Exception as e:
        # Redis read failed — log + fall through. Don't pollute the
        # passthrough with retries; the breaker (when enabled, §3.5)
        # handles repeated TEI failure separately.
        log.warning("embed_cache.get failed (%s) — falling through", e)
        raw = None

    if raw is not None:
        try:
            import json

            vec = json.loads(raw)
            if isinstance(vec, list):
                return [float(x) for x in vec]
        except (ValueError, TypeError) as e:
            # Corrupt cached value — log + fall through. Overwrite below.
            log.warning("embed_cache: corrupt value at %s (%s)", key, e)

    out = await embedder.embed([text])
    vec = out[0] if out else []

    try:
        import json

        await redis.set(key, json.dumps(vec), ex=_TTL_SECONDS)
    except Exception as e:
        log.warning("embed_cache.set failed (%s) — vector not cached", e)

    return vec


__all__ = ["get_or_set", "_make_key"]
