"""Tests for the semantic retrieval cache (P2.6).

Uses ``fakeredis.FakeRedis`` as a drop-in for real Redis. If fakeredis is
missing the suite is skipped — the feature itself works in prod regardless.
"""
from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")

from ext.services import retrieval_cache as rc


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch):
    """Reset the module-level cached client between tests + clear flag leaks."""
    rc._reset_client_for_tests()
    monkeypatch.delenv("RAG_SEMCACHE", raising=False)
    monkeypatch.delenv("RAG_SEMCACHE_TTL", raising=False)
    yield
    rc._reset_client_for_tests()


def _patch_fake_client(monkeypatch):
    client = fakeredis.FakeRedis()
    monkeypatch.setattr(rc, "_get_client", lambda: client)
    return client


# ---------------------------------------------------------------------------
# _quantize(): semantic-near queries collide
# ---------------------------------------------------------------------------


def test_quantize_is_deterministic():
    q1 = rc._quantize([0.1, 0.2, 0.3])
    q2 = rc._quantize([0.1, 0.2, 0.3])
    assert q1 == q2


def test_quantize_collides_on_7th_decimal():
    """Core "semantic" property: tiny float diffs (below 6 decimals) collide."""
    a = rc._quantize([0.1234567, 0.2345678, 0.3456789])
    b = rc._quantize([0.1234568, 0.2345679, 0.3456788])  # differ at 7th digit
    assert a == b


def test_quantize_differs_on_6th_decimal():
    """Differences at the quantize-resolution boundary should NOT collide."""
    a = rc._quantize([0.100000])
    b = rc._quantize([0.100001])
    assert a != b


def test_quantize_handles_negatives_and_zero():
    """Must not crash on edge-case floats; determinism is the only contract."""
    q = rc._quantize([0.0, -1.0, 1e-12, -1e-12])
    assert isinstance(q, str) and len(q) == 16


# ---------------------------------------------------------------------------
# _kbs_hash(): order-insensitive + includes chat_id
# ---------------------------------------------------------------------------


def test_kbs_hash_is_deterministic():
    a = rc._kbs_hash([{"kb_id": 1, "subtag_ids": [10, 11]}], chat_id=42)
    b = rc._kbs_hash([{"kb_id": 1, "subtag_ids": [10, 11]}], chat_id=42)
    assert a == b


def test_kbs_hash_is_order_insensitive():
    """Caller may pass KBs in any order — cache key must be stable."""
    a = rc._kbs_hash(
        [{"kb_id": 2, "subtag_ids": [5, 6]}, {"kb_id": 1, "subtag_ids": [10]}],
        chat_id=None,
    )
    b = rc._kbs_hash(
        [{"kb_id": 1, "subtag_ids": [10]}, {"kb_id": 2, "subtag_ids": [5, 6]}],
        chat_id=None,
    )
    assert a == b


def test_kbs_hash_subtag_order_insensitive():
    a = rc._kbs_hash([{"kb_id": 1, "subtag_ids": [5, 10]}], chat_id=None)
    b = rc._kbs_hash([{"kb_id": 1, "subtag_ids": [10, 5]}], chat_id=None)
    assert a == b


def test_kbs_hash_differs_on_chat_id():
    a = rc._kbs_hash([{"kb_id": 1}], chat_id=1)
    b = rc._kbs_hash([{"kb_id": 1}], chat_id=2)
    assert a != b


def test_kbs_hash_differs_when_kbs_differ():
    a = rc._kbs_hash([{"kb_id": 1}], chat_id=None)
    b = rc._kbs_hash([{"kb_id": 2}], chat_id=None)
    assert a != b


def test_kbs_hash_handles_missing_subtag_ids():
    """None vs [] should hash identically — both mean "all subtags"."""
    a = rc._kbs_hash([{"kb_id": 1}], chat_id=None)
    b = rc._kbs_hash([{"kb_id": 1, "subtag_ids": []}], chat_id=None)
    c = rc._kbs_hash([{"kb_id": 1, "subtag_ids": None}], chat_id=None)
    assert a == b == c


# ---------------------------------------------------------------------------
# is_enabled() respects RAG_SEMCACHE
# ---------------------------------------------------------------------------


def test_is_enabled_false_by_default(monkeypatch):
    """Default OFF — even with a live client, flag must be explicitly set."""
    _patch_fake_client(monkeypatch)
    # RAG_SEMCACHE unset → disabled.
    assert rc.is_enabled() is False


def test_is_enabled_false_when_flag_is_zero(monkeypatch):
    _patch_fake_client(monkeypatch)
    monkeypatch.setenv("RAG_SEMCACHE", "0")
    assert rc.is_enabled() is False


def test_is_enabled_true_when_flag_set_and_client_alive(monkeypatch):
    _patch_fake_client(monkeypatch)
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    assert rc.is_enabled() is True


def test_is_enabled_false_when_client_missing(monkeypatch):
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    monkeypatch.setattr(rc, "_get_client", lambda: False)
    assert rc.is_enabled() is False


# ---------------------------------------------------------------------------
# get()/put() roundtrip
# ---------------------------------------------------------------------------


def _hits_from_dicts(dicts):
    """Build Hit objects matching the vector_store dataclass shape."""
    from ext.services.vector_store import Hit
    return [Hit(id=d["id"], score=d["score"], payload=d["payload"]) for d in dicts]


def test_get_returns_none_when_disabled(monkeypatch):
    _patch_fake_client(monkeypatch)
    # RAG_SEMCACHE unset → must short-circuit to None even if key exists.
    out = rc.get([0.1, 0.2], [{"kb_id": 1}], chat_id=None)
    assert out is None


def test_put_is_noop_when_disabled(monkeypatch):
    """With flag off, put() must not touch Redis at all."""
    client = _patch_fake_client(monkeypatch)
    rc.put(
        [0.1, 0.2],
        [{"kb_id": 1}],
        None,
        _hits_from_dicts([{"id": "a", "score": 0.9, "payload": {"text": "x"}}]),
    )
    # No flag → no writes.
    assert len(client.keys("*")) == 0


def test_get_put_roundtrip(monkeypatch):
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    _patch_fake_client(monkeypatch)
    vec = [0.1, 0.2, 0.3]
    kbs = [{"kb_id": 1, "subtag_ids": [5]}]
    hits_in = _hits_from_dicts(
        [
            {"id": "a", "score": 0.9, "payload": {"text": "alpha"}},
            {"id": "b", "score": 0.5, "payload": {"text": "beta"}},
        ]
    )
    rc.put(vec, kbs, None, hits_in)
    out = rc.get(vec, kbs, None)
    assert out is not None
    assert len(out) == 2
    assert out[0]["id"] == "a"
    assert out[0]["score"] == 0.9
    assert out[0]["payload"] == {"text": "alpha"}
    assert out[1]["id"] == "b"


def test_get_shape_matches_retriever_expectation(monkeypatch):
    """Retriever reconstructs Hit(id=c["id"], score=c["score"], payload=c["payload"])."""
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    _patch_fake_client(monkeypatch)
    hits_in = _hits_from_dicts(
        [{"id": "x1", "score": 0.8, "payload": {"text": "t", "kb_id": 1}}]
    )
    rc.put([0.5], [{"kb_id": 1}], None, hits_in)
    out = rc.get([0.5], [{"kb_id": 1}], None)
    # Caller must be able to do Hit(id=..., score=..., payload=...).
    from ext.services.vector_store import Hit
    h = Hit(id=out[0]["id"], score=out[0]["score"], payload=out[0]["payload"])
    assert h.id == "x1" and h.score == 0.8
    assert h.payload["kb_id"] == 1


def test_near_identical_queries_hit_same_entry(monkeypatch):
    """Two vectors differing below the quantize threshold must share a cache key."""
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    _patch_fake_client(monkeypatch)
    vec_a = [0.1234567, 0.7654321]
    vec_b = [0.1234568, 0.7654320]  # differ below 6 decimals
    hits = _hits_from_dicts([{"id": "shared", "score": 0.7, "payload": {}}])
    rc.put(vec_a, [{"kb_id": 1}], None, hits)
    out = rc.get(vec_b, [{"kb_id": 1}], None)
    assert out is not None and out[0]["id"] == "shared"


def test_different_kbs_separate_entries(monkeypatch):
    """Same query, different KB selection → different cache entries."""
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    _patch_fake_client(monkeypatch)
    vec = [0.1, 0.2]
    rc.put(vec, [{"kb_id": 1}], None, _hits_from_dicts([{"id": "a", "score": 1.0, "payload": {}}]))
    rc.put(vec, [{"kb_id": 2}], None, _hits_from_dicts([{"id": "b", "score": 1.0, "payload": {}}]))
    out_a = rc.get(vec, [{"kb_id": 1}], None)
    out_b = rc.get(vec, [{"kb_id": 2}], None)
    assert out_a[0]["id"] == "a"
    assert out_b[0]["id"] == "b"


def test_pipeline_version_bump_invalidates(monkeypatch):
    """Cache key includes model_version — bumping it means old entries vanish."""
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    _patch_fake_client(monkeypatch)
    vec = [0.1]

    monkeypatch.setattr(rc, "_current_model_version", lambda: "v1")
    rc.put(vec, [{"kb_id": 1}], None, _hits_from_dicts([{"id": "old", "score": 0.9, "payload": {}}]))
    assert rc.get(vec, [{"kb_id": 1}], None)[0]["id"] == "old"

    # Bump model version → reads under new prefix now miss.
    monkeypatch.setattr(rc, "_current_model_version", lambda: "v2")
    assert rc.get(vec, [{"kb_id": 1}], None) is None


def test_ttl_default_is_300(monkeypatch):
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    client = _patch_fake_client(monkeypatch)
    vec = [0.1]
    kbs = [{"kb_id": 1}]
    rc.put(vec, kbs, None, _hits_from_dicts([{"id": "a", "score": 0.1, "payload": {}}]))
    # Find the key that was set.
    keys = list(client.keys("semcache:*"))
    assert len(keys) == 1
    ttl = client.ttl(keys[0])
    assert 0 < ttl <= 300


def test_ttl_honors_env(monkeypatch):
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    monkeypatch.setenv("RAG_SEMCACHE_TTL", "60")
    client = _patch_fake_client(monkeypatch)
    rc.put([0.1], [{"kb_id": 1}], None, _hits_from_dicts([{"id": "a", "score": 0.1, "payload": {}}]))
    keys = list(client.keys("semcache:*"))
    ttl = client.ttl(keys[0])
    assert 0 < ttl <= 60


# ---------------------------------------------------------------------------
# Fail-open on Redis errors
# ---------------------------------------------------------------------------


class _BoomClient:
    """Simulates Redis failures on every call."""

    def get(self, *_a, **_k):
        raise ConnectionError("redis down")

    def setex(self, *_a, **_k):
        raise ConnectionError("redis down")

    def ping(self):
        raise ConnectionError("redis down")


def test_get_fail_open_on_redis_exception(monkeypatch):
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    monkeypatch.setattr(rc, "_get_client", lambda: _BoomClient())
    out = rc.get([0.1], [{"kb_id": 1}], None)
    assert out is None  # miss, not raise


def test_put_fail_silent_on_redis_exception(monkeypatch):
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    monkeypatch.setattr(rc, "_get_client", lambda: _BoomClient())
    # Must not raise.
    rc.put([0.1], [{"kb_id": 1}], None, _hits_from_dicts([{"id": "a", "score": 0.1, "payload": {}}]))


def test_get_when_client_is_false(monkeypatch):
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    monkeypatch.setattr(rc, "_get_client", lambda: False)
    assert rc.get([0.1], [{"kb_id": 1}], None) is None


def test_get_handles_corrupt_value(monkeypatch):
    """Non-JSON bytes in Redis → treat as miss, not raise."""
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    client = _patch_fake_client(monkeypatch)
    # Seed a corrupt value under the exact key that get() will look for.
    vec = [0.1, 0.2]
    kbs = [{"kb_id": 1}]
    key = rc._key(
        rc._current_model_version(),
        rc._kbs_hash(kbs, None),
        rc._quantize(vec),
    )
    client.set(key, b"not-json-data-{{")
    out = rc.get(vec, kbs, None)
    assert out is None


def test_put_serializes_hit_dataclass_instances(monkeypatch):
    """Retriever hands us Hit objects; put() must extract id/score/payload."""
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    _patch_fake_client(monkeypatch)
    from ext.services.vector_store import Hit
    hits = [
        Hit(id="p1", score=0.9, payload={"text": "hello", "kb_id": 1}),
        Hit(id=42, score=0.5, payload={"text": "world"}),  # int id
    ]
    rc.put([0.1], [{"kb_id": 1}], None, hits)
    out = rc.get([0.1], [{"kb_id": 1}], None)
    assert out is not None and len(out) == 2
    assert out[0]["id"] == "p1"
    assert out[1]["id"] == "42"  # int coerced to str via str()


def test_put_accepts_dict_hits_too(monkeypatch):
    """Tests / non-Hit callers may pass plain dicts."""
    monkeypatch.setenv("RAG_SEMCACHE", "1")
    _patch_fake_client(monkeypatch)
    rc.put(
        [0.1],
        [{"kb_id": 1}],
        None,
        [{"id": "d", "score": 0.7, "payload": {}}],
    )
    out = rc.get([0.1], [{"kb_id": 1}], None)
    assert out[0]["id"] == "d"
