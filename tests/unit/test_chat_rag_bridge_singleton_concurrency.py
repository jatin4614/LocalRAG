"""Tests for §7.5 — concurrent first-call protection on
``_qu_cache_singleton`` and `_rbac_redis` lazy-init paths.

Pre-fix: both `chat_rag_bridge._get_qu_cache()` and
`chat_rag_bridge._redis_client()` did unguarded ``if x is None: x = create()``,
so two concurrent first-callers each created + leaked a Redis client.
The fix mirrors the double-checked-locking pattern in
``ext/services/cross_encoder_reranker.py:80-82``: a module-level
``threading.Lock`` guards the read+create.

Race-window strategy
--------------------
Python's GIL serializes most pure-Python work, so a vanilla race test
finishes inside the GIL slice and the bug never manifests. We deliberately
sleep inside the constructor stubs so the threads stack up between the
first None-check and the assignment — the exact gap the lock must cover.
"""
from __future__ import annotations

import threading
import time

import pytest

from ext.services import chat_rag_bridge as bridge


# ---------------------------------------------------------------------------
# _get_qu_cache: concurrent first-callers get exactly one client
# ---------------------------------------------------------------------------


def test_get_qu_cache_concurrent_first_callers_share_one_client(monkeypatch):
    """N concurrent first-time threads must converge on a single QUCache
    instance. Pre-fix each would have built its own and leaked the loser.
    """
    monkeypatch.setattr(bridge, "_qu_cache_singleton", None)
    monkeypatch.setenv("RAG_QU_CACHE_ENABLED", "1")

    construction_count = 0
    construction_lock = threading.Lock()

    # Stub out redis.from_url so we don't try to actually connect.
    import redis.asyncio as _redis_async
    monkeypatch.setattr(_redis_async, "from_url", lambda *_a, **_kw: object())

    class _FakeQUCache:
        def __init__(self, *, redis_client):  # noqa: ARG002
            nonlocal construction_count
            # Sleep first to widen the race window: with the GIL,
            # `time.sleep` releases the GIL so peers can run. Without the
            # fix, every peer enters the construction path before the
            # first one finishes assigning the singleton, so we count 8.
            time.sleep(0.05)
            with construction_lock:
                construction_count += 1

    # The function does `from .qu_cache import QUCache` at call time, so
    # patching the attribute on the module suffices.
    from ext.services import qu_cache as _qu_cache_mod
    monkeypatch.setattr(_qu_cache_mod, "QUCache", _FakeQUCache)

    barrier = threading.Barrier(8)
    results: list = []
    results_lock = threading.Lock()

    def _race():
        barrier.wait()
        c = bridge._get_qu_cache()
        with results_lock:
            results.append(c)

    threads = [threading.Thread(target=_race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one QUCache should have been constructed.
    assert construction_count == 1, (
        f"expected 1 QUCache construction under concurrent first-call, "
        f"got {construction_count}"
    )
    # All threads see the same singleton.
    assert all(r is results[0] for r in results)
    assert results[0] is bridge._qu_cache_singleton


# ---------------------------------------------------------------------------
# _redis_client: concurrent first-callers get exactly one client
# ---------------------------------------------------------------------------


def test_redis_client_concurrent_first_callers_share_one_client(monkeypatch):
    """N concurrent first-time threads must converge on a single
    ``_rbac_redis`` instance.
    """
    monkeypatch.setattr(bridge, "_rbac_redis", None)

    construction_count = 0
    construction_lock = threading.Lock()

    sentinel = object()

    def _fake_from_url(*_a, **_kw):
        nonlocal construction_count
        # Same race-window widening trick as the QU cache test: sleep
        # before the bookkeeping so peers stack up in the unprotected gap.
        time.sleep(0.05)
        with construction_lock:
            construction_count += 1
        return sentinel

    # Patch the from_url attribute directly on the imported module — the
    # function does `import redis.asyncio as _redis` so we need the real
    # module's `from_url` to be the stub.
    import redis.asyncio as _redis_async
    monkeypatch.setattr(_redis_async, "from_url", _fake_from_url)

    barrier = threading.Barrier(8)
    results: list = []
    results_lock = threading.Lock()

    def _race():
        barrier.wait()
        c = bridge._redis_client()
        with results_lock:
            results.append(c)

    threads = [threading.Thread(target=_race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert construction_count == 1, (
        f"expected 1 redis client construction under concurrent first-call, "
        f"got {construction_count}"
    )
    assert all(r is results[0] for r in results)
    assert results[0] is bridge._rbac_redis


# ---------------------------------------------------------------------------
# Sanity: subsequent calls (after init) skip the lock entirely
# ---------------------------------------------------------------------------


def test_redis_client_returns_existing_handle(monkeypatch):
    """Once initialized, calls return the existing handle without rebuilding."""
    sentinel = object()
    monkeypatch.setattr(bridge, "_rbac_redis", sentinel)
    assert bridge._redis_client() is sentinel


def test_get_qu_cache_returns_existing_handle(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(bridge, "_qu_cache_singleton", sentinel)
    assert bridge._get_qu_cache() is sentinel


def test_get_qu_cache_returns_none_when_disabled(monkeypatch):
    """RAG_QU_CACHE_ENABLED=0 must short-circuit before touching the lock."""
    monkeypatch.setattr(bridge, "_qu_cache_singleton", None)
    monkeypatch.setenv("RAG_QU_CACHE_ENABLED", "0")
    assert bridge._get_qu_cache() is None
    # Singleton stayed None so we did not leak a client.
    assert bridge._qu_cache_singleton is None
