"""Semantic retrieval cache — memoize top-k Qdrant results by
quantized query embedding + KB selection.

Key format: ``semcache:{model_version}:{kbs_hash}:{vec_hash}``
Value:      JSON-encoded list of ``{"id", "score", "payload"}`` dicts.
TTL:        ``RAG_SEMCACHE_TTL`` seconds (default 300).

All operations fail-open: if Redis is unreachable or misconfigured, the
cache silently returns misses and retrieval proceeds normally. Nothing
in the retrieval path should raise because of this module.

Default OFF via ``RAG_SEMCACHE=0``. Flip ``RAG_SEMCACHE=1`` to engage.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Optional, Sequence

from . import flags

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
    so set ``RAG_REDIS_URL`` explicitly or rely on the fail-open path.
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
            client.ping()
            _CLIENT = client
        except Exception:
            _CLIENT = False
        return _CLIENT


def is_enabled() -> bool:
    """Return True if the cache can be used right now.

    Gated behind ``RAG_SEMCACHE=1``; default off for zero behavior change.
    Reads through ``flags.get`` so per-request KB-config overrides (P3.0)
    can enable the cache for specific KBs without global env mutation.
    """
    if flags.get("RAG_SEMCACHE", "0") != "1":
        return False
    c = _get_client()
    return bool(c) and c is not False


def _ttl() -> int:
    try:
        return int(os.environ.get("RAG_SEMCACHE_TTL", "300"))
    except ValueError:
        return 300


def _quantize(vec: Sequence[float], *, decimals: int = 6) -> str:
    """Round floats to ``decimals`` and hash so near-identical queries collide.

    This is what makes the cache *semantic*: two embeddings that differ only
    in the 7th decimal place hash to the same key.
    """
    return hashlib.sha1(
        ",".join(f"{v:.{decimals}f}" for v in vec).encode("utf-8")
    ).hexdigest()[:16]


def _kbs_hash(selected_kbs: list[dict], chat_id: Optional[int]) -> str:
    """Stable hash over the KB selection + chat_id.

    Normalizes to sorted ``(kb_id, tuple(sorted(subtag_ids)))`` so arg order
    doesn't matter. Includes ``chat_id`` because chat-private namespaces are
    part of the retrieval scope.
    """
    normalized = sorted(
        (
            int(k.get("kb_id")) if k.get("kb_id") is not None else -1,
            tuple(sorted(k.get("subtag_ids") or [])),
        )
        for k in selected_kbs
    )
    payload = (("chat", chat_id), ("kbs", tuple(normalized)))
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _key(model_version: str, kbs_sig: str, vec_sig: str) -> str:
    return f"semcache:{model_version}:{kbs_sig}:{vec_sig}"


def _current_model_version() -> str:
    """Fetch pipeline_version lazily so import cost is deferred.

    Falls back to ``"unknown"`` on any failure — the cache still works,
    it just won't invalidate on pipeline bumps in this edge case.
    """
    try:
        from ext.services.pipeline_version import current_version
        return current_version()
    except Exception:
        return "unknown"


def get(
    query_vec: Sequence[float],
    selected_kbs: list[dict],
    chat_id: Optional[int],
) -> Optional[list[dict]]:
    """Look up a cached top-k result. Returns ``None`` on miss or any error."""
    if not is_enabled():
        return None
    client = _get_client()
    if not client:
        return None
    try:
        mv = _current_model_version()
        raw = client.get(_key(mv, _kbs_hash(selected_kbs, chat_id), _quantize(query_vec)))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        return None


def put(
    query_vec: Sequence[float],
    selected_kbs: list[dict],
    chat_id: Optional[int],
    hits: list[Any],
) -> None:
    """Serialize ``hits`` and store under the cache key with TTL. Fail-silent."""
    if not is_enabled():
        return
    client = _get_client()
    if not client:
        return
    try:
        mv = _current_model_version()
        # Serialize Hit objects. Each Hit has id/score/payload. If a given hit
        # is already a dict (test convenience), normalize via isinstance.
        serialized = []
        for h in hits:
            if hasattr(h, "id") and hasattr(h, "score"):
                serialized.append(
                    {
                        "id": str(h.id),
                        "score": float(h.score),
                        "payload": dict(h.payload or {}),
                    }
                )
            elif isinstance(h, dict):
                serialized.append(
                    {
                        "id": str(h.get("id")),
                        "score": float(h.get("score", 0.0)),
                        "payload": dict(h.get("payload") or {}),
                    }
                )
        client.setex(
            _key(mv, _kbs_hash(selected_kbs, chat_id), _quantize(query_vec)),
            _ttl(),
            json.dumps(serialized),
        )
    except Exception:
        pass


def _reset_client_for_tests() -> None:
    """Clear the cached client. ONLY for unit tests that monkeypatch ``_get_client``."""
    global _CLIENT
    _CLIENT = None


__all__ = [
    "is_enabled",
    "get",
    "put",
]
