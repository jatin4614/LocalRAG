"""Per-KB top_k override widens the pre-rerank pull cap.

Backstop for option B (32 Inf Bde eviction case 2026-05-01). When an
admin stamps ``rag_config.top_k`` on a KB, the bridge MUST raise the
intent-driven ``_per_kb`` default so low-frequency entities aren't
starved out before rerank ever sees them.

Mirrors the stub pattern in test_rerank_topk_widening.py.
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


def _mk_hits(n):
    return [
        _FakeHit(
            id=i, score=1.0 - i * 0.01,
            payload={
                "text": f"chunk-{i}", "kb_id": 1, "subtag_id": None,
                "doc_id": 10 + i, "filename": f"d-{i}.md",
                "chunk_index": i, "chat_id": None,
            },
        )
        for i in range(n)
    ]


@pytest.fixture
def captured_limits(monkeypatch):
    """Wire bridge with stubs that capture per_kb_limit/total_limit."""
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

    captured = {"per_kb_limit": [], "total_limit": []}

    async def _stub_retrieve(*, query, selected_kbs, chat_id, vector_store, embedder,  # noqa: ARG001
                             per_kb_limit=10, total_limit=30, **kwargs):
        captured["per_kb_limit"].append(per_kb_limit)
        captured["total_limit"].append(total_limit)
        return _mk_hits(per_kb_limit)

    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)
    monkeypatch.setattr(_retriever, "retrieve", _stub_retrieve, raising=True)
    monkeypatch.setattr(
        _reranker, "rerank_with_flag",
        lambda q, h, *, top_k=10, fallback_fn=None: list(h)[:top_k],
        raising=True,
    )
    monkeypatch.setattr(
        _reranker, "rerank",
        lambda h, *, top_k=10: list(h)[:top_k], raising=True,
    )
    monkeypatch.setattr(
        _budget, "budget_chunks",
        lambda h, *, max_tokens=4000: h, raising=True,
    )

    yield captured

    sys.modules.pop("ext.services.mmr", None)


async def _invoke():
    return await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="give updates on 75 Inf Bde, 5 PoK Bde, 32 Inf Bde, 80 Inf Bde",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_no_override_uses_intent_default(captured_limits, monkeypatch):
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    await _invoke()
    # Default intent path: per_kb=10, total=30 (no enumerated date)
    assert captured_limits["per_kb_limit"][0] == 10


@pytest.mark.asyncio
async def test_top_k_override_raises_per_kb_above_default(captured_limits, monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "24")
    await _invoke()
    # 24 > intent default (10) → per_kb widens to 24
    assert captured_limits["per_kb_limit"][0] == 24
    # total = 24 * 1 KB = 24 (max with intent default 30 → stays 30)
    assert captured_limits["total_limit"][0] >= 24


@pytest.mark.asyncio
async def test_top_k_below_intent_default_does_not_shrink(captured_limits, monkeypatch):
    """A small override must NOT narrow a heavier intent default."""
    monkeypatch.setenv("RAG_TOP_K", "5")
    await _invoke()
    # Default intent default (10) > override (5) → keep 10
    assert captured_limits["per_kb_limit"][0] == 10
