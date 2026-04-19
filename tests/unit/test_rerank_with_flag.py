"""Tests for the ``rerank_with_flag`` dispatcher.

Verifies that:
* ``RAG_RERANK=0`` (or unset) calls the legacy ``fallback_fn`` ONLY and
  never imports the cross-encoder module (byte-identical default path).
* ``RAG_RERANK=1`` routes to the cross-encoder path.
* Cross-encoder failure falls back to the legacy path (fail-open).

These are pure shape tests — the real model is never loaded.
"""
from __future__ import annotations

import pytest

import ext.services.reranker as rr
from ext.services.vector_store import Hit


def _h(id, score, kb_id=1, text="t"):
    return Hit(id=id, score=score, payload={"kb_id": kb_id, "text": text})


def test_flag_off_uses_fallback_fn(monkeypatch):
    monkeypatch.delenv("RAG_RERANK", raising=False)

    calls: list[tuple] = []

    def fake_fallback(hits, *, top_k):
        calls.append(("fallback", len(hits), top_k))
        return list(hits[:top_k])

    hits = [_h(1, 0.9), _h(2, 0.8), _h(3, 0.7)]
    out = rr.rerank_with_flag("q", hits, top_k=2, fallback_fn=fake_fallback)
    assert [h.id for h in out] == [1, 2]
    assert calls == [("fallback", 3, 2)]


def test_flag_zero_uses_fallback_fn(monkeypatch):
    monkeypatch.setenv("RAG_RERANK", "0")

    def fake_fallback(hits, *, top_k):
        return list(hits[:top_k])

    hits = [_h(1, 0.9), _h(2, 0.8)]
    out = rr.rerank_with_flag("q", hits, top_k=1, fallback_fn=fake_fallback)
    assert [h.id for h in out] == [1]


def test_flag_off_never_imports_cross_encoder(monkeypatch):
    """When the flag is off, the cross_encoder_reranker module must not be touched."""
    monkeypatch.setenv("RAG_RERANK", "0")

    import sys
    # Remove any cached import so we can detect a new one.
    sys.modules.pop("ext.services.cross_encoder_reranker", None)

    def fallback(hits, *, top_k):
        return list(hits[:top_k])

    rr.rerank_with_flag("q", [_h(1, 0.9)], top_k=1, fallback_fn=fallback)
    assert "ext.services.cross_encoder_reranker" not in sys.modules


def test_flag_on_routes_to_cross_encoder(monkeypatch):
    monkeypatch.setenv("RAG_RERANK", "1")

    import sys
    import types

    sentinel: list[str] = []

    fake_mod = types.ModuleType("ext.services.cross_encoder_reranker")

    def fake_rerank_cross_encoder(query, hits, *, top_k=10, **_):
        sentinel.append(f"ce:{query}:{top_k}:{len(hits)}")
        return list(hits[:top_k])

    fake_mod.rerank_cross_encoder = fake_rerank_cross_encoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ext.services.cross_encoder_reranker", fake_mod)

    def fallback(hits, *, top_k):
        sentinel.append("fallback-called")
        return list(hits[:top_k])

    hits = [_h(1, 0.9), _h(2, 0.8)]
    out = rr.rerank_with_flag("hello", hits, top_k=2, fallback_fn=fallback)
    assert len(out) == 2
    assert sentinel == ["ce:hello:2:2"]


def test_flag_on_fail_open_falls_back_to_legacy(monkeypatch):
    """If the cross-encoder raises, rerank_with_flag should quietly use fallback_fn."""
    monkeypatch.setenv("RAG_RERANK", "1")

    import sys
    import types

    calls: list[str] = []

    fake_mod = types.ModuleType("ext.services.cross_encoder_reranker")

    def boom(query, hits, *, top_k=10, **_):
        calls.append("ce-raised")
        raise RuntimeError("simulated model failure")

    fake_mod.rerank_cross_encoder = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ext.services.cross_encoder_reranker", fake_mod)

    def fallback(hits, *, top_k):
        calls.append("fallback-called")
        return list(hits[:top_k])

    hits = [_h(1, 0.9), _h(2, 0.8)]
    out = rr.rerank_with_flag("q", hits, top_k=2, fallback_fn=fallback)
    assert [h.id for h in out] == [1, 2]
    assert calls == ["ce-raised", "fallback-called"]


def test_flag_on_import_error_fail_open(monkeypatch):
    """If the cross-encoder module cannot be imported (ImportError), use fallback_fn."""
    monkeypatch.setenv("RAG_RERANK", "1")

    import sys
    # Force ImportError by setting the module to None (Python's import machinery raises).
    monkeypatch.setitem(sys.modules, "ext.services.cross_encoder_reranker", None)

    called = []

    def fallback(hits, *, top_k):
        called.append(True)
        return list(hits[:top_k])

    hits = [_h(1, 0.9), _h(2, 0.8)]
    out = rr.rerank_with_flag("q", hits, top_k=1, fallback_fn=fallback)
    assert len(out) == 1
    assert called == [True]


def test_default_fallback_is_legacy_rerank(monkeypatch):
    """When fallback_fn is not provided, rerank_with_flag defaults to ext.services.reranker.rerank."""
    monkeypatch.setenv("RAG_RERANK", "0")

    # Legacy rerank fast-paths on top1/top2 ratio > 2.0, returning top_k unchanged.
    hits = [_h(1, 0.9), _h(2, 0.2), _h(3, 0.1)]
    out = rr.rerank_with_flag("q", hits, top_k=3)  # no fallback_fn → uses rerank
    assert [h.id for h in out] == [1, 2, 3]


def test_flag_read_at_call_time(monkeypatch):
    """The flag must be read each call so tests can toggle env vars without reload."""
    # Start with flag off.
    monkeypatch.setenv("RAG_RERANK", "0")

    calls: list[str] = []

    def fallback(hits, *, top_k):
        calls.append("fallback")
        return list(hits[:top_k])

    rr.rerank_with_flag("q", [_h(1, 0.9)], top_k=1, fallback_fn=fallback)
    assert calls == ["fallback"]

    # Flip flag mid-session. Use a fake CE to confirm we switch paths.
    monkeypatch.setenv("RAG_RERANK", "1")
    import sys
    import types

    fake_mod = types.ModuleType("ext.services.cross_encoder_reranker")

    def fake_rerank_cross_encoder(query, hits, *, top_k=10, **_):
        calls.append("ce")
        return list(hits[:top_k])

    fake_mod.rerank_cross_encoder = fake_rerank_cross_encoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ext.services.cross_encoder_reranker", fake_mod)

    rr.rerank_with_flag("q", [_h(1, 0.9)], top_k=1, fallback_fn=fallback)
    assert calls == ["fallback", "ce"]
