"""Tests for the Redis-backed rerank score cache.

These tests use ``fakeredis.FakeRedis`` as a drop-in replacement for the
real Redis client. If ``fakeredis`` isn't installed the suite is skipped
cleanly — the feature works in prod without it.
"""
from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")

from ext.services import rerank_cache as rc


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch):
    """Reset the module-level cached client between tests."""
    rc._reset_client_for_tests()
    # Also clear any env override left by a prior test.
    monkeypatch.delenv("RAG_RERANK_CACHE_DISABLED", raising=False)
    yield
    rc._reset_client_for_tests()


def _patch_fake_client(monkeypatch):
    """Force ``_get_client`` to return a new ``fakeredis.FakeRedis`` instance."""
    client = fakeredis.FakeRedis()
    monkeypatch.setattr(rc, "_get_client", lambda: client)
    return client


# ---------------------------------------------------------------------------
# _key() determinism
# ---------------------------------------------------------------------------


def test_key_is_deterministic():
    k1 = rc._key("m", "q", "p")
    k2 = rc._key("m", "q", "p")
    assert k1 == k2
    assert k1.startswith("rerank:m:")


def test_key_differs_by_query():
    k1 = rc._key("m", "q1", "p")
    k2 = rc._key("m", "q2", "p")
    assert k1 != k2


def test_key_differs_by_passage():
    k1 = rc._key("m", "q", "p1")
    k2 = rc._key("m", "q", "p2")
    assert k1 != k2


def test_key_differs_by_model():
    k1 = rc._key("m1", "q", "p")
    k2 = rc._key("m2", "q", "p")
    assert k1 != k2


# ---------------------------------------------------------------------------
# get_many / put_many roundtrip via fakeredis
# ---------------------------------------------------------------------------


def test_get_many_empty_returns_empty(monkeypatch):
    _patch_fake_client(monkeypatch)
    assert rc.get_many("m", []) == []


def test_put_many_empty_is_noop(monkeypatch):
    _patch_fake_client(monkeypatch)
    rc.put_many("m", [])  # must not raise


def test_roundtrip_basic(monkeypatch):
    _patch_fake_client(monkeypatch)
    rc.put_many("m", [("q", "p1", 0.5), ("q", "p2", 0.75)])
    out = rc.get_many("m", [("q", "p1"), ("q", "p2"), ("q", "missing")])
    assert out == [0.5, 0.75, None]


def test_get_many_all_misses(monkeypatch):
    _patch_fake_client(monkeypatch)
    out = rc.get_many("m", [("q", "p1"), ("q", "p2")])
    assert out == [None, None]


def test_model_scope_is_respected(monkeypatch):
    _patch_fake_client(monkeypatch)
    rc.put_many("model_a", [("q", "p", 1.0)])
    rc.put_many("model_b", [("q", "p", 2.0)])
    assert rc.get_many("model_a", [("q", "p")]) == [1.0]
    assert rc.get_many("model_b", [("q", "p")]) == [2.0]


def test_ttl_is_set(monkeypatch):
    """setex writes with TTL; fakeredis respects this via ttl() API."""
    monkeypatch.setenv("RAG_RERANK_CACHE_TTL", "120")
    client = _patch_fake_client(monkeypatch)
    rc.put_many("m", [("q", "p", 0.42)])
    key = rc._key("m", "q", "p")
    ttl = client.ttl(key)
    # Must be positive and <= configured TTL.
    assert 0 < ttl <= 120


def test_ttl_default_is_300(monkeypatch):
    monkeypatch.delenv("RAG_RERANK_CACHE_TTL", raising=False)
    client = _patch_fake_client(monkeypatch)
    rc.put_many("m", [("q", "p", 0.42)])
    ttl = client.ttl(rc._key("m", "q", "p"))
    assert 0 < ttl <= 300


# ---------------------------------------------------------------------------
# is_enabled semantics
# ---------------------------------------------------------------------------


def test_is_enabled_true_with_fake_client(monkeypatch):
    _patch_fake_client(monkeypatch)
    assert rc.is_enabled() is True


def test_is_enabled_false_when_disabled_env(monkeypatch):
    _patch_fake_client(monkeypatch)
    monkeypatch.setenv("RAG_RERANK_CACHE_DISABLED", "1")
    assert rc.is_enabled() is False


def test_is_enabled_false_when_client_false(monkeypatch):
    monkeypatch.setattr(rc, "_get_client", lambda: False)
    assert rc.is_enabled() is False


# ---------------------------------------------------------------------------
# Fail-open: Redis errors return None / noop
# ---------------------------------------------------------------------------


class _BoomClient:
    """Fakes a client whose mget/pipeline always raise."""

    def mget(self, keys):
        raise ConnectionError("redis down")

    def pipeline(self):
        raise ConnectionError("redis down")

    def scan_iter(self, match=None, count=None):  # noqa: ARG002
        raise ConnectionError("redis down")

    def delete(self, *_a):  # noqa: ARG002
        raise ConnectionError("redis down")


def test_get_many_fail_open_on_redis_exception(monkeypatch):
    monkeypatch.setattr(rc, "_get_client", lambda: _BoomClient())
    out = rc.get_many("m", [("q", "p1"), ("q", "p2")])
    assert out == [None, None]


def test_put_many_fail_silent_on_redis_exception(monkeypatch):
    monkeypatch.setattr(rc, "_get_client", lambda: _BoomClient())
    # Must not raise
    rc.put_many("m", [("q", "p", 0.1)])


def test_clear_all_fail_open(monkeypatch):
    monkeypatch.setattr(rc, "_get_client", lambda: _BoomClient())
    assert rc.clear_all() == 0


def test_get_many_when_client_is_false(monkeypatch):
    """If Redis was unreachable at startup, _get_client returns False."""
    monkeypatch.setattr(rc, "_get_client", lambda: False)
    out = rc.get_many("m", [("q", "p")])
    assert out == [None]


def test_get_many_handles_corrupt_value(monkeypatch):
    """Non-float values in Redis should be treated as misses, not errors."""
    client = _patch_fake_client(monkeypatch)
    # Seed a corrupt value directly
    client.set(rc._key("m", "q", "p"), b"not-a-float")
    out = rc.get_many("m", [("q", "p")])
    assert out == [None]


# ---------------------------------------------------------------------------
# clear_all scoped + unscoped
# ---------------------------------------------------------------------------


def test_clear_all_by_model(monkeypatch):
    _patch_fake_client(monkeypatch)
    rc.put_many("a", [("q", "p", 1.0)])
    rc.put_many("b", [("q", "p", 2.0)])
    deleted = rc.clear_all(model="a")
    assert deleted == 1
    assert rc.get_many("a", [("q", "p")]) == [None]
    assert rc.get_many("b", [("q", "p")]) == [2.0]


def test_clear_all_unscoped(monkeypatch):
    _patch_fake_client(monkeypatch)
    rc.put_many("a", [("q", "p", 1.0)])
    rc.put_many("b", [("q", "p", 2.0)])
    deleted = rc.clear_all()
    assert deleted == 2
    assert rc.get_many("a", [("q", "p")]) == [None]
    assert rc.get_many("b", [("q", "p")]) == [None]
