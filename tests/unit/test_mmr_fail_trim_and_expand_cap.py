"""Phase 6.X Options B + C — handle the "MMR fails, budget evicts to 3"
cliff under multi-entity decompose.

Option B: ``RAG_MMR_FAIL_TRIM=1`` trims rerank survivors to ``_final_k``
when MMR fails, so context_expand only multiplies a small set.

Option C: ``RAG_CONTEXT_EXPAND_MAX_HITS=N`` caps how many hits the
sibling expansion runs on; the tail passes through unchanged.

Both flags default 0 (pre-fix behaviour). Both are independent —
each can be tested without the other.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import pytest

from ext.services import chat_rag_bridge as bridge


@dataclass
class _FakeHit:
    id: int | str
    score: float
    payload: dict


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def execute(self, *a, **kw):
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
            id=f"h{i}",
            score=0.99 - i * 0.001,
            payload={
                "text": f"chunk {i}", "kb_id": 1, "subtag_id": None,
                "doc_id": 100 + i, "filename": "f.md",
                "chunk_index": i, "chat_id": None,
            },
        )
        for i in range(n)
    ]


@pytest.fixture
def harness(monkeypatch):
    """Stub retrieve / rerank / mmr / expand / budget; record the chain."""
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
    import ext.services.context_expand as _expand
    import types

    captured: dict[str, Any] = {
        "rerank_in": [], "rerank_out": [],
        "mmr_in": [], "mmr_called": False,
        "expand_in": [], "expand_out": [],
        "budget_in": [],
    }

    async def _stub_retrieve(*, query, **kwargs):  # noqa: ARG001
        # Return a wide candidate set so rerank_top_k=50 trims something.
        return _mk_hits(80)

    def _stub_rerank_with_flag(query, hits, *, top_k=10, fallback_fn=None):  # noqa: ARG001
        captured["rerank_in"].append(len(hits))
        captured["rerank_out"].append(top_k)
        return list(hits)[:top_k]

    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)
    monkeypatch.setattr(_retriever, "retrieve", _stub_retrieve, raising=True)
    monkeypatch.setattr(
        _reranker, "rerank_with_flag", _stub_rerank_with_flag, raising=True,
    )
    monkeypatch.setattr(
        _reranker, "rerank",
        lambda h, *, top_k=10: list(h)[:top_k], raising=True,
    )

    # MMR stub that ALWAYS RAISES — simulate TEI OOM.
    fake_mmr_mod = types.ModuleType("ext.services.mmr")

    async def _stub_mmr_raise(*args, **kwargs):  # noqa: ARG001
        captured["mmr_called"] = True
        captured["mmr_in"].append(len(args[1]) if len(args) > 1 else 0)
        raise RuntimeError("simulated TEI OOM")

    fake_mmr_mod.mmr_rerank_from_hits = _stub_mmr_raise  # type: ignore[attr-defined]
    sys.modules["ext.services.mmr"] = fake_mmr_mod

    async def _stub_expand(hits, *, vs, window):  # noqa: ARG001
        captured["expand_in"].append(len(hits))
        # Simulate context_expand growing each hit by appending a sibling
        # marker. Real expand_context returns more chunks; for the test
        # we just record the input size.
        captured["expand_out"].append(len(hits) * 3)
        return list(hits) * 3  # 3× growth simulating ±1 siblings

    monkeypatch.setattr(_expand, "expand_context", _stub_expand, raising=True)

    def _stub_budget(hits, *, max_tokens=4000):  # noqa: ARG001
        captured["budget_in"].append(len(hits))
        return list(hits)

    monkeypatch.setattr(
        _budget, "budget_chunks", _stub_budget, raising=True,
    )
    yield captured
    sys.modules.pop("ext.services.mmr", None)


async def _invoke():
    return await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="give updates on 75 Inf Bde, 5 PoK Bde, 32 Inf Bde, 80 Inf Bde",
        user_id="user-1",
    )


# --- Option B: RAG_MMR_FAIL_TRIM ---


@pytest.mark.asyncio
async def test_mmr_fail_default_passes_all_to_expand(harness, monkeypatch):
    """Default (flag off): MMR raises → all rerank survivors flow through
    unchanged. context_expand sees the full 50."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "50")
    monkeypatch.setenv("RAG_MMR", "1")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "1")
    monkeypatch.delenv("RAG_MMR_FAIL_TRIM", raising=False)

    await _invoke()
    # MMR was called and raised.
    assert harness["mmr_called"] is True
    # context_expand saw all 50 (rerank_top_k).
    assert harness["expand_in"][0] == 50


@pytest.mark.asyncio
async def test_mmr_fail_trim_caps_to_final_k(harness, monkeypatch):
    """Flag on: MMR raises → bridge trims to _final_k before expand."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "50")
    monkeypatch.setenv("RAG_MMR", "1")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "1")
    monkeypatch.setenv("RAG_MMR_FAIL_TRIM", "1")
    # _final_k defaults to RAG_TOP_K=12.
    monkeypatch.setenv("RAG_TOP_K", "12")

    await _invoke()
    assert harness["mmr_called"] is True
    # context_expand should now see _final_k=12, not 50.
    assert harness["expand_in"][0] == 12


# --- Option C: RAG_CONTEXT_EXPAND_MAX_HITS ---


@pytest.mark.asyncio
async def test_expand_cap_zero_unlimited(harness, monkeypatch):
    """Flag 0 (default): all rerank survivors expanded."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "50")
    monkeypatch.setenv("RAG_MMR", "0")  # no mmr to keep this isolated
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "1")
    monkeypatch.delenv("RAG_CONTEXT_EXPAND_MAX_HITS", raising=False)
    monkeypatch.setenv("RAG_TOP_K", "12")

    await _invoke()
    # MMR off → rerank trims to _final_k=12 → expand sees 12.
    assert harness["expand_in"][0] == 12


@pytest.mark.asyncio
async def test_expand_cap_active_appends_tail(harness, monkeypatch):
    """Flag>0 + MMR fail: top-N expanded, tail passes through unchanged."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "50")
    monkeypatch.setenv("RAG_MMR", "1")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "1")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND_MAX_HITS", "10")
    # MMR fails → without RAG_MMR_FAIL_TRIM, all 50 reach expand.
    # Cap=10 → first 10 get expanded, last 40 unchanged.
    monkeypatch.delenv("RAG_MMR_FAIL_TRIM", raising=False)
    monkeypatch.setenv("RAG_TOP_K", "12")

    await _invoke()
    # expand_context was called with only 10 hits (the head).
    assert harness["expand_in"][0] == 10
    # Output: head 10 × 3 (3× growth from stub) + tail 40 unchanged = 70.
    assert harness["budget_in"][0] == 70


@pytest.mark.asyncio
async def test_options_b_and_c_combined(harness, monkeypatch):
    """B + C together: MMR fail trims to _final_k, then expand caps."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "50")
    monkeypatch.setenv("RAG_MMR", "1")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "1")
    monkeypatch.setenv("RAG_MMR_FAIL_TRIM", "1")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND_MAX_HITS", "12")
    monkeypatch.setenv("RAG_TOP_K", "12")

    await _invoke()
    # B trims to 12, then C cap >= 12 means no further trimming.
    assert harness["expand_in"][0] == 12
