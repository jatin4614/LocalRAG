"""P2 — chat_rag_bridge widens rerank_top_k when MMR is on.

Baseline pipeline was ``rerank(top_k=10) -> MMR(top_k=10)`` which made MMR
a pass-through. Bridge now picks ``_rerank_k = max(2 * _final_k, 20)`` when
``RAG_MMR=1`` (operator-overridable via ``RAG_RERANK_TOP_K``), then MMR trims
to ``_final_k``. When MMR is off, ``_rerank_k == _final_k`` and the
pre-P2 behaviour is byte-identical.

These tests capture the ``top_k`` that ``rerank_with_flag`` is called with,
and assert the final output length stays at ``_final_k == 10`` in all cases.
They reuse the stub pattern from ``test_chat_rag_bridge_expand_flag``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, List

import pytest

from ext.services import chat_rag_bridge as bridge


_FINAL_K = 10  # mirrors the hard-coded constant in chat_rag_bridge.py


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


def _mk_hits(n: int) -> list[_FakeHit]:
    return [
        _FakeHit(
            id=i,
            score=1.0 - i * 0.01,
            payload={
                "text": f"chunk-{i}",
                "kb_id": 1,
                "subtag_id": None,
                "doc_id": 10 + i,
                "filename": f"doc-{i}.md",
                "chunk_index": i,
                "chat_id": None,
            },
        )
        for i in range(n)
    ]


async def _fake_retrieve_30(*, query, selected_kbs, chat_id, vector_store, embedder,  # noqa: ARG001
                            per_kb_limit=10, total_limit=30):
    # Return 30 so rerank can pull 20 when MMR is on.
    return _mk_hits(30)


@pytest.fixture
def configured_bridge(monkeypatch):
    """Wire bridge with stubs that capture rerank ``top_k``.

    ``captured`` gathers every top_k seen by the stubbed ``rerank_with_flag``
    so tests can assert the bridge called the reranker with the expected
    widened count. MMR is stubbed to a simple head-slice so we can assert
    final length without depending on the real embedder.
    """
    # Ensure the mmr module isn't cached with a stale state for our stub.
    sys.modules.pop("ext.services.mmr", None)

    bridge.configure(
        vector_store=object(),
        embedder=object(),
        sessionmaker=_fake_sessionmaker,
    )
    import ext.services.rbac as _rbac
    import ext.services.retriever as _retriever
    import ext.services.reranker as _reranker
    import ext.services.budget as _budget

    captured: dict[str, list[int]] = {"rerank_top_k": []}

    def _stub_rerank_with_flag(query, hits, *, top_k=10, fallback_fn=None):  # noqa: ARG001
        captured["rerank_top_k"].append(top_k)
        return list(hits)[:top_k]

    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)
    monkeypatch.setattr(_retriever, "retrieve", _fake_retrieve_30, raising=True)
    monkeypatch.setattr(
        _reranker, "rerank_with_flag", _stub_rerank_with_flag, raising=True,
    )
    monkeypatch.setattr(
        _reranker, "rerank", lambda hits, *, top_k=10: list(hits)[:top_k], raising=True,
    )
    # Budget is identity so final list length reflects MMR / trim stage.
    monkeypatch.setattr(_budget, "budget_chunks", lambda hits, *, max_tokens=4000: hits, raising=True)

    yield captured

    sys.modules.pop("ext.services.mmr", None)


def _install_mmr_stub(captured: dict[str, Any]) -> None:
    """Install an ``mmr_rerank_from_hits`` stub that records its ``top_k``."""
    import types

    async def _stub_mmr(query, hits, embedder, *, top_k=10, lambda_=0.7):  # noqa: ARG001
        captured.setdefault("mmr_top_k", []).append(top_k)
        return list(hits)[:top_k]

    fake_mod = types.ModuleType("ext.services.mmr")
    fake_mod.mmr_rerank_from_hits = _stub_mmr  # type: ignore[attr-defined]
    sys.modules["ext.services.mmr"] = fake_mod


async def _invoke(bridge_mod) -> List[dict]:
    return await bridge_mod.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="what is mmr",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_mmr_off_rerank_topk_is_final_k(configured_bridge, monkeypatch):
    """RAG_MMR unset/0 -> rerank called with top_k=_final_k (10). Legacy path."""
    monkeypatch.delenv("RAG_MMR", raising=False)
    monkeypatch.delenv("RAG_RERANK_TOP_K", raising=False)

    out = await _invoke(bridge)

    assert configured_bridge["rerank_top_k"] == [_FINAL_K]
    # Final group-by-doc output: 10 hits -> 10 unique docs -> 10 source groups.
    assert len(out) == _FINAL_K


@pytest.mark.asyncio
async def test_mmr_off_explicit_zero_same_as_unset(configured_bridge, monkeypatch):
    """RAG_MMR=0 explicitly -> same behaviour as unset."""
    monkeypatch.setenv("RAG_MMR", "0")
    monkeypatch.delenv("RAG_RERANK_TOP_K", raising=False)

    await _invoke(bridge)
    assert configured_bridge["rerank_top_k"] == [_FINAL_K]


@pytest.mark.asyncio
async def test_mmr_on_widens_rerank_topk_to_20(configured_bridge, monkeypatch):
    """RAG_MMR=1 -> rerank called with 2*_final_k (=20) by default."""
    monkeypatch.setenv("RAG_MMR", "1")
    monkeypatch.delenv("RAG_RERANK_TOP_K", raising=False)

    _install_mmr_stub(configured_bridge)

    out = await _invoke(bridge)

    # Rerank was called once, with widened top_k = 20.
    assert configured_bridge["rerank_top_k"] == [max(_FINAL_K * 2, 20)]
    # MMR trimmed back to _final_k.
    assert configured_bridge["mmr_top_k"] == [_FINAL_K]
    # Final output size is _final_k.
    assert len(out) == _FINAL_K


@pytest.mark.asyncio
async def test_rerank_top_k_override_with_mmr(configured_bridge, monkeypatch):
    """RAG_RERANK_TOP_K overrides the default widened value when MMR is on."""
    monkeypatch.setenv("RAG_MMR", "1")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "15")

    _install_mmr_stub(configured_bridge)

    out = await _invoke(bridge)

    # Override 15 is >= _final_k so it's honoured verbatim.
    assert configured_bridge["rerank_top_k"] == [15]
    assert configured_bridge["mmr_top_k"] == [_FINAL_K]
    assert len(out) == _FINAL_K


@pytest.mark.asyncio
async def test_rerank_top_k_override_below_final_k_clamped(configured_bridge, monkeypatch):
    """RAG_RERANK_TOP_K below _final_k is clamped UP to _final_k."""
    monkeypatch.setenv("RAG_MMR", "1")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "3")  # deliberately too-small

    _install_mmr_stub(configured_bridge)

    await _invoke(bridge)

    # Clamped to _final_k.
    assert configured_bridge["rerank_top_k"] == [_FINAL_K]


@pytest.mark.asyncio
async def test_rerank_top_k_override_without_mmr_trims_tail(configured_bridge, monkeypatch):
    """RAG_RERANK_TOP_K set with MMR off -> rerank returns override, bridge trims tail to _final_k."""
    monkeypatch.delenv("RAG_MMR", raising=False)
    monkeypatch.setenv("RAG_RERANK_TOP_K", "25")

    out = await _invoke(bridge)

    # Bridge asked rerank for 25 even with MMR off (operator override honoured).
    assert configured_bridge["rerank_top_k"] == [25]
    # No MMR, so the elif branch trims from 25 back to _final_k for the budget stage.
    assert len(out) == _FINAL_K


@pytest.mark.asyncio
async def test_mmr_on_without_override_preserves_final_output_size(configured_bridge, monkeypatch):
    """End-to-end invariant: MMR on, no override -> final output is _final_k items."""
    monkeypatch.setenv("RAG_MMR", "1")
    monkeypatch.delenv("RAG_RERANK_TOP_K", raising=False)

    _install_mmr_stub(configured_bridge)

    out = await _invoke(bridge)

    assert len(out) == _FINAL_K
