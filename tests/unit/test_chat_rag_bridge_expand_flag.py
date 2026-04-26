"""Integration-y tests: verify chat_rag_bridge.retrieve_kb_sources honors the
RAG_CONTEXT_EXPAND flag at call time.

Strategy: stub retrieve / rerank / budget. Flag is read at call time inside
the bridge (not module-load), so toggling it via monkeypatch.setenv works
without importlib.reload.

Key invariants:
* Flag unset (default): ``ext.services.context_expand`` is NOT imported.
  Byte-identical to pre-P1.4.
* Flag=1: module is imported, ``expand_context`` is called exactly once
  per retrieval cycle.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

from ext.services import chat_rag_bridge as bridge


@dataclass
class _FakeHit:
    id: int
    score: float
    payload: dict


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def execute(self, *a, **kw):  # pragma: no cover - unused here
        class _R:
            def first(self):
                return None
        return _R()


def _fake_sessionmaker():
    return _FakeSession()


async def _fake_allowed(session, *, user_id):  # noqa: ARG001
    return [1]


async def _fake_retrieve(*, query, selected_kbs, chat_id, vector_store, embedder, per_kb_limit=10, total_limit=30, **kwargs):  # noqa: ARG001
    return [
        _FakeHit(
            id=1,
            score=0.9,
            payload={
                "text": "center chunk",
                "kb_id": 1,
                "subtag_id": None,
                "doc_id": 10,
                "filename": "readme.md",
                "chunk_index": 5,
                "chat_id": None,
            },
        )
    ]


@pytest.fixture
def configured_bridge(monkeypatch):
    """Wire bridge with minimal stubs. Resets context_expand module state.

    Yields to let the test body run, then on teardown drops any stub we
    installed in ``sys.modules``. Without this cleanup a later test that
    runs in the same pytest session and imports ``from
    ext.services.context_expand import expand_context`` would get our
    passthrough stub instead of the real function.
    """
    sys.modules.pop("ext.services.context_expand", None)
    bridge.configure(
        vector_store=object(),
        embedder=object(),
        sessionmaker=_fake_sessionmaker,
    )
    import ext.services.rbac as _rbac
    import ext.services.retriever as _retriever
    import ext.services.reranker as _reranker
    import ext.services.budget as _budget

    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)
    monkeypatch.setattr(_retriever, "retrieve", _fake_retrieve, raising=True)
    monkeypatch.setattr(_reranker, "rerank", lambda hits, *, top_k=10: hits, raising=True)
    monkeypatch.setattr(_budget, "budget_chunks", lambda hits, *, max_tokens=4000: hits, raising=True)

    yield

    # Teardown: scrub any stub module installed by the test body so the real
    # ``expand_context`` is importable in later tests.
    sys.modules.pop("ext.services.context_expand", None)


@pytest.mark.asyncio
async def test_flag_unset_does_not_import_context_expand(configured_bridge, monkeypatch):
    """With RAG_CONTEXT_EXPAND off, context_expand module must not be imported.

    Uses RAG_INTENT_OVERLAY_MODE=env (B3 design call) so the per-intent
    policy overlay defers to the operator-set env value. Combined with an
    explicit RAG_CONTEXT_EXPAND=0, the overlay drops the key and
    flags.get returns the env value, preserving the pre-Phase-2.2
    "env is the single source of truth" contract this test asserts.
    """
    # B3 escape hatch: env wins over the per-intent overlay. Must be set
    # BEFORE the bridge resolves intent flags (read at call time inside
    # retrieve_kb_sources), and the relevant flag must be EXPLICITLY set
    # so env mode actually has a value to surface — pure delete still
    # leaves the intent default in the overlay.
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "0")

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    assert isinstance(out, list)
    assert len(out) == 1
    # Invariant: flag off -> module absent from sys.modules.
    assert "ext.services.context_expand" not in sys.modules


@pytest.mark.asyncio
async def test_flag_zero_does_not_import_context_expand(configured_bridge, monkeypatch):
    """RAG_CONTEXT_EXPAND=0 (explicit) → same as unset.

    Uses RAG_INTENT_OVERLAY_MODE=env (B3 design call) so the operator's
    explicit RAG_CONTEXT_EXPAND=0 wins over the per-intent overlay's
    default of RAG_CONTEXT_EXPAND=1 for 'specific' queries.
    """
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "0")

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    assert len(out) == 1
    assert "ext.services.context_expand" not in sys.modules


@pytest.mark.asyncio
async def test_flag_on_imports_and_calls_expand_once(configured_bridge, monkeypatch):
    """RAG_CONTEXT_EXPAND=1 → module is imported, expand_context called once."""
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "1")

    # Install a stub expand_context *before* the bridge imports the module,
    # by pre-populating the target module with a fake ``expand_context``.
    import types

    call_count = {"n": 0}

    async def _stub_expand(hits, *, vs, window=1):  # noqa: ARG001
        call_count["n"] += 1
        return list(hits)

    # Ensure we actually trigger the import inside the bridge; install a real
    # module object but replace its expand_context symbol.
    fake_mod = types.ModuleType("ext.services.context_expand")
    fake_mod.expand_context = _stub_expand  # type: ignore[attr-defined]
    sys.modules["ext.services.context_expand"] = fake_mod

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    assert len(out) == 1
    assert "ext.services.context_expand" in sys.modules
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_flag_on_failure_falls_back_silently(configured_bridge, monkeypatch):
    """If expand_context raises, bridge falls back to reranker output."""
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "1")

    import types

    async def _raising_expand(hits, *, vs, window=1):  # noqa: ARG001
        raise RuntimeError("boom")

    fake_mod = types.ModuleType("ext.services.context_expand")
    fake_mod.expand_context = _raising_expand  # type: ignore[attr-defined]
    sys.modules["ext.services.context_expand"] = fake_mod

    # Must not propagate; must return the fake_retrieve result.
    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    assert isinstance(out, list)
    assert len(out) == 1


@pytest.mark.asyncio
async def test_flag_on_passes_window_from_env(configured_bridge, monkeypatch):
    """RAG_CONTEXT_EXPAND_WINDOW is read at call time and forwarded."""
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "1")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND_WINDOW", "3")

    import types

    captured: dict = {}

    async def _stub_expand(hits, *, vs, window=1):  # noqa: ARG001
        captured["window"] = window
        return list(hits)

    fake_mod = types.ModuleType("ext.services.context_expand")
    fake_mod.expand_context = _stub_expand  # type: ignore[attr-defined]
    sys.modules["ext.services.context_expand"] = fake_mod

    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    assert captured["window"] == 3
