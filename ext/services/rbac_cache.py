"""Redis-backed cache for RBAC `allowed_kb_ids`.

Keys: ``rbac:user:{user_id}`` -> JSON array of kb_ids.
TTL: ``RAG_RBAC_CACHE_TTL_SECS`` (default 30).
TTL=0 disables the cache (get/set become no-ops) -- operator escape hatch.

Invalidation: ``kb_admin`` router publishes affected user ids on the
``rbac:invalidate`` channel after any `kb_access` mutation. A background
task in ``chat_rag_bridge`` subscribes and drops the matching keys.

Sacred invariant (CLAUDE.md §2): zero cross-user data leakage. The cache
key namespace ``rbac:user:{user_id}`` includes the user_id so two users
can never share a cache entry.

TTL trade-off (M7)
------------------
The 30-second default TTL is a deliberate balance between two failure
modes:

* **Too long** (e.g. 5 minutes): a revoked grant could remain visible
  to the affected user for up to TTL seconds if the pub/sub
  invalidation is dropped (Redis restart, replica disconnect, network
  blip). For a security-sensitive surface this is the bigger risk.

* **Too short** (e.g. 5 seconds): every chat request hits the DB on
  cache miss, removing the point of the cache for active users —
  exactly the chat-heavy users who would benefit most. Latency
  regresses by the cost of one Postgres lookup per request.

30s is the sweet spot for our deployment shape (20-200 users, single
replica). The pub/sub channel is the fast path; the TTL is the safety
net. Operators in HIPAA/SOC2 environments may want to drop it to 5s
or 0 (disable) and accept the latency hit. ``RAG_RBAC_CACHE_TTL_SECS``
is the knob; ``rag_rbac_cache_inval_failed_total`` counter (incremented
when pub/sub publish fails) is the monitoring signal — a non-zero rate
means the TTL is the only thing keeping users from seeing stale grants,
and operators should investigate the Redis health.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Iterable

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "rbac:user"
PUBSUB_CHANNEL = "rbac:invalidate"


class RbacCache:
    def __init__(self, *, redis, ttl_sec: int | None = None) -> None:
        self._redis = redis
        self._ttl = int(
            ttl_sec
            if ttl_sec is not None
            else os.environ.get("RAG_RBAC_CACHE_TTL_SECS", "30")
        )

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    async def get(self, *, user_id: str) -> set[int] | None:
        if not self.enabled:
            return None
        raw = await self._redis.get(f"{CACHE_NAMESPACE}:{user_id}")
        if raw is None:
            return None
        try:
            return set(json.loads(raw))
        except (ValueError, TypeError):
            logger.warning(
                "rbac cache: corrupt value for user %s, ignoring", user_id
            )
            return None

    async def set(
        self, *, user_id: str, allowed_kb_ids: Iterable[int]
    ) -> None:
        if not self.enabled:
            return
        payload = json.dumps(
            sorted(int(x) for x in allowed_kb_ids)
        ).encode("utf-8")
        await self._redis.setex(
            f"{CACHE_NAMESPACE}:{user_id}", self._ttl, payload
        )

    async def invalidate(self, *, user_ids: Iterable[str]) -> None:
        uids = [str(u) for u in user_ids]
        if not uids:
            return
        keys = [f"{CACHE_NAMESPACE}:{u}" for u in uids]
        await self._redis.delete(*keys)
        msg = json.dumps({"user_ids": uids}).encode("utf-8")
        await self._redis.publish(PUBSUB_CHANNEL, msg)


_SHARED: RbacCache | None = None


def get_shared_cache(*, redis) -> RbacCache:
    """Return process-wide RbacCache, creating if needed.

    The first caller fixes the redis handle for the lifetime of the
    process. Subsequent calls return the same instance regardless of the
    handle they pass -- this is intentional, callers should pass the same
    handle they obtained from ``_redis_client()``.
    """
    global _SHARED
    if _SHARED is None:
        _SHARED = RbacCache(redis=redis)
    return _SHARED


def _reset_shared_cache_for_tests() -> None:
    """Reset the process-wide singleton. Tests only."""
    global _SHARED
    _SHARED = None


async def subscribe_invalidations(redis) -> None:
    """Long-running task: subscribe to rbac:invalidate, drop local cache entries.

    In a multi-replica deployment each replica runs this. Redis pub/sub
    broadcasts, so every replica's local cache sees every invalidation.
    Single-replica today, but future-proof.

    Fail-open: any exception inside a single message handler is logged
    and the loop keeps running. A broken payload (bad JSON, missing
    user_ids) is dropped silently rather than tearing down the listener.
    """
    pubsub = redis.pubsub()
    await pubsub.subscribe(PUBSUB_CHANNEL)
    async for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        try:
            data = message["data"]
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            payload = json.loads(data)
            uids = payload.get("user_ids") or []
            if uids:
                # Note: we invalidate on both the current replica's cache
                # (via direct DELETE) AND any other replicas (they get the
                # same pubsub event and re-issue their own DELETE). Idempotent.
                keys = [f"{CACHE_NAMESPACE}:{u}" for u in uids]
                await redis.delete(*keys)
                logger.info(
                    "rbac cache: invalidated %d keys from pubsub", len(keys)
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("rbac cache: pubsub handler error: %s", exc)
