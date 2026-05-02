"""Redis-backed cache for cross-encoder rerank scores.

Key format: ``rerank:{model_name}:{pipeline_version}:{sha1(query)[:16]}:{sha1(passage)[:16]}``
Value:      float score, serialized as ASCII bytes.
TTL:        ``RAG_RERANK_CACHE_TTL`` seconds (default 300).

The ``pipeline_version`` component invalidates cached scores whenever the
chunker / extractor / embedder / context-augmentation tag bumps. Without it,
a re-ingest with normalized text would let stale rerank scores from the
old chunking survive for the full TTL window. Mirrors the pattern in
``ext/services/retrieval_cache.py``.

All operations fail-open: if Redis is unreachable or misconfigured, the
cache silently returns misses and the reranker falls through to model
inference. Nothing in the retrieval path should raise because of this
module.
"""
from __future__ import annotations

import hashlib
import os
import threading
from typing import List, Optional, Sequence, Tuple

_LOCK = threading.Lock()
# Sentinel states:
#   None  — not yet tried to connect
#   False — tried and failed (do not retry this process)
#   <obj> — a live redis.Redis client
_CLIENT: object | None = None


def _redis_url() -> str:
    """Return the Redis URL to use.

    Inside ``orgchat-net`` Redis is at ``redis://redis:6379/0``. From a
    host-based test runner the container has no published port by default,
    so set ``RAG_REDIS_URL`` explicitly (e.g. to the container IP) or rely
    on the fail-open path.
    """
    return os.environ.get("RAG_REDIS_URL", "redis://redis:6379/0")


def _get_client():
    """Return a cached redis client, or ``False`` if connection failed."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            return _CLIENT
        try:
            import redis

            client = redis.from_url(
                _redis_url(),
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
            # Ping to fail fast if unreachable.
            client.ping()
            _CLIENT = client
        except Exception:
            _CLIENT = False
        return _CLIENT


def is_enabled() -> bool:
    """Return True if the cache can be used right now.

    Respects ``RAG_RERANK_CACHE_DISABLED=1`` as a hard off-switch.
    """
    if os.environ.get("RAG_RERANK_CACHE_DISABLED", "0") == "1":
        return False
    c = _get_client()
    return bool(c) and c is not False


def _ttl() -> int:
    try:
        return int(os.environ.get("RAG_RERANK_CACHE_TTL", "300"))
    except ValueError:
        return 300


def _current_pipeline_version() -> str:
    """Fetch ``pipeline_version`` lazily so import cost is deferred.

    Falls back to ``"unknown"`` on any failure — the cache still works,
    it just won't invalidate on pipeline bumps in this edge case. Mirrors
    ``retrieval_cache._current_model_version``.
    """
    try:
        from ext.services.pipeline_version import current_version
        return current_version()
    except Exception:
        return "unknown"


def _key(model: str, query: str, passage: str, pipeline_version: str | None = None) -> str:
    """Build a cache key.

    ``pipeline_version`` is included so a re-ingest with a bumped
    chunker/extractor invalidates stale scores. When omitted (callers that
    haven't been updated), the current pipeline version is fetched lazily.
    """
    q = hashlib.sha1(query.encode("utf-8", errors="replace")).hexdigest()[:16]
    p = hashlib.sha1((passage or "").encode("utf-8", errors="replace")).hexdigest()[:16]
    pv = pipeline_version if pipeline_version is not None else _current_pipeline_version()
    return f"rerank:{model}:{pv}:{q}:{p}"


def get_many(
    model: str, queries_passages: Sequence[Tuple[str, str]]
) -> List[Optional[float]]:
    """Look up scores. Returns ``None`` for misses. Fail-open on Redis errors."""
    if not queries_passages:
        return []
    client = _get_client()
    if not client:
        return [None] * len(queries_passages)
    try:
        keys = [_key(model, q, p) for q, p in queries_passages]
        raw = client.mget(keys)
        out: list[Optional[float]] = []
        for v in raw:
            if v is None:
                out.append(None)
                continue
            try:
                out.append(float(v))
            except (ValueError, TypeError):
                out.append(None)
        return out
    except Exception:
        return [None] * len(queries_passages)


def put_many(model: str, triples: Sequence[Tuple[str, str, float]]) -> None:
    """Set scores with TTL. Fail-silent on Redis errors."""
    if not triples:
        return
    client = _get_client()
    if not client:
        return
    try:
        ttl = _ttl()
        pipe = client.pipeline()
        for q, p, s in triples:
            pipe.setex(_key(model, q, p), ttl, str(float(s)))
        pipe.execute()
    except Exception:
        pass


def clear_all(model: Optional[str] = None) -> int:
    """Delete all cache entries (optionally scoped to ``model``).

    Returns the count deleted. Intended for tests / ops. Fail-open on errors.
    """
    client = _get_client()
    if not client:
        return 0
    try:
        pattern = f"rerank:{model}:*" if model else "rerank:*"
        total = 0
        for k in client.scan_iter(match=pattern, count=1000):
            client.delete(k)
            total += 1
        return total
    except Exception:
        return 0


def _reset_client_for_tests() -> None:
    """Clear the cached client. ONLY for unit tests that monkeypatch ``_get_client``."""
    global _CLIENT
    _CLIENT = None
