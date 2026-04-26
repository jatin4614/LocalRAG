"""Redis-backed cache for QueryUnderstanding results.

Plan B Phase 4.5. Uses **Redis DB 4** by convention (Plan A took DB 3 for
the RBAC cache; DB 0 = app, DB 1 = celery broker, DB 2 = celery results).

Key namespace: ``qu:<sha256>`` where the digest covers the normalized
query plus the last assistant turn ID. TTL defaults to 300 s
(``RAG_QU_CACHE_TTL_SECS``).

The cached payload is a JSON-serialized :class:`QueryUnderstanding`. On
retrieval we override ``cached=True`` so the bridge can distinguish hot
cache hits from cold LLM calls in metric labels.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict
from typing import Optional

from .query_understanding import QueryUnderstanding


log = logging.getLogger("orgchat.qu_cache")


_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[?!.,;:]+$")


def _normalize_for_cache(query: str | None) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation.

    Goal: cache-hit ``"What is OFC?"`` and ``"what is ofc"`` as the same
    query while keeping ``"what is OFC roadmap"`` distinct. Safe on empty
    or ``None`` input — returns ``""``.
    """
    if not query:
        return ""
    q = query.strip().lower()
    q = _WHITESPACE_RE.sub(" ", q)
    q = _TRAILING_PUNCT_RE.sub("", q)
    return q


def _make_key(query: str, last_turn_id: str | None) -> str:
    """Build the Redis key.

    ``last_turn_id`` is the assistant's last turn ID (string). Empty
    string for new chats. The hash covers both so different conversation
    contexts don't share entries — important because the QU prompt's
    pronoun resolution depends on the most recent assistant turn.
    """
    norm = _normalize_for_cache(query)
    payload = f"{norm}\x00{last_turn_id or ''}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"qu:{digest}"


class QUCache:
    """Async cache facade.

    Constructor takes a redis client (``redis.asyncio.Redis`` or
    compatible fake — see ``tests/conftest.py:fake_redis``). Tests inject
    the fake; production wiring lives in :mod:`ext.services.chat_rag_bridge`.
    """

    def __init__(
        self,
        redis_client,
        *,
        ttl_secs: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._r = redis_client
        self._ttl = (
            ttl_secs
            if ttl_secs is not None
            else int(os.environ.get("RAG_QU_CACHE_TTL_SECS", "300"))
        )
        self._enabled = (
            enabled
            if enabled is not None
            else os.environ.get("RAG_QU_CACHE_ENABLED", "1") == "1"
        )

    async def get(
        self, query: str, last_turn_id: str | None
    ) -> Optional[QueryUnderstanding]:
        """Look up a cached :class:`QueryUnderstanding`. Returns ``None``
        on cache miss, redis error, or corrupt cached value (soft-fail)."""
        if not self._enabled:
            return None
        key = _make_key(query, last_turn_id)
        try:
            raw = await self._r.get(key)
        except Exception as e:  # connection refused, network glitch, etc.
            log.warning("qu_cache.get failed: %s", e)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
            qu = QueryUnderstanding(**data)
            qu.cached = True
            return qu
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("qu_cache: corrupt cached value at %s: %s", key, e)
            return None

    async def set(
        self,
        query: str,
        last_turn_id: str | None,
        qu: QueryUnderstanding,
    ) -> None:
        """Cache the QU result. Silently no-ops if the cache is disabled
        or if redis is unreachable."""
        if not self._enabled:
            return
        key = _make_key(query, last_turn_id)
        # Don't persist the cached flag — every read sets it to True.
        payload = {**asdict(qu), "cached": False}
        try:
            await self._r.set(key, json.dumps(payload), ex=self._ttl)
        except Exception as e:
            log.warning("qu_cache.set failed: %s", e)


__all__ = ["QUCache", "_normalize_for_cache", "_make_key"]
