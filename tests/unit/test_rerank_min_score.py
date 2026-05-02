"""Tests for §5.1 — RAG_RERANK_MIN_SCORE flag.

When set, after ``rerank_with_flag`` returns, the bridge drops hits whose
score is strictly below the configured threshold. Default unset = OFF
(byte-identical to pre-§5.1 behaviour).

Strategy: mock retrieve / rerank / budget so we control the post-rerank
score distribution, then assert which hits flow through to the budget
stage. The flag is read at call time inside the bridge.
"""
from __future__ import annotations

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

    async def execute(self, *a, **kw):  # pragma: no cover - not used by these tests
        class _R:
            def first(self):
                return None

            def all(self):
                return []

        return _R()


def _fake_sessionmaker():
    return _FakeSession()


def _make_hit(idx: int, score: float, text: str = "x") -> _FakeHit:
    return _FakeHit(
        id=idx,
        score=score,
        payload={
            "text": text,
            "kb_id": 1,
            "subtag_id": None,
            "doc_id": f"doc-{idx}",
            "filename": f"{idx}.md",
            "chunk_index": 0,
            "chat_id": None,
            "score": score,
        },
    )


@pytest.fixture
def configure_bridge_with_mixed_scores(monkeypatch):
    """Configure bridge with a retrieve that returns hits of mixed scores
    and a passthrough rerank — so the threshold filter sees the same scores
    we set on the input.
    """
    bridge.configure(
        vector_store=object(),
        embedder=object(),
        sessionmaker=_fake_sessionmaker,
    )

    async def _fake_allowed(session, *, user_id):  # noqa: ARG001
        return [1]

    import ext.services.rbac as _rbac
    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)

    # 5 hits with scores spanning above/below typical thresholds.
    hits = [
        _make_hit(1, 0.95),
        _make_hit(2, 0.50),
        _make_hit(3, 0.10),
        _make_hit(4, 0.04),
        _make_hit(5, 0.01),
    ]

    async def _fake_retrieve(*, query, selected_kbs, chat_id, vector_store, embedder, per_kb_limit=10, total_limit=30, **kwargs):  # noqa: ARG001
        return list(hits)

    import ext.services.retriever as _retriever
    import ext.services.reranker as _reranker
    import ext.services.budget as _budget

    monkeypatch.setattr(_retriever, "retrieve", _fake_retrieve, raising=True)
    # rerank passthrough — preserves the input scores so the threshold filter
    # sees them downstream. Both the legacy and flag-aware paths are stubbed.
    monkeypatch.setattr(_reranker, "rerank", lambda hits, *, top_k=10: list(hits[:top_k]), raising=True)
    monkeypatch.setattr(
        _reranker,
        "rerank_with_flag",
        lambda query, hits, *, top_k=10, fallback_fn=None: list(hits[:top_k]),
        raising=True,
    )
    monkeypatch.setattr(_budget, "budget_chunks", lambda hits, *, max_tokens=4000: list(hits), raising=True)

    return {"hits": hits}


@pytest.mark.asyncio
async def test_threshold_unset_keeps_all_hits(configure_bridge_with_mixed_scores, monkeypatch):
    """Default (env unset) → no threshold filter → all 5 hits flow through."""
    monkeypatch.delenv("RAG_RERANK_MIN_SCORE", raising=False)

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="q",
        user_id="user-1",
    )

    # 5 unique doc_ids → 5 sources groups.
    assert len(out) == 5


@pytest.mark.asyncio
async def test_threshold_zero_keeps_all_hits(configure_bridge_with_mixed_scores, monkeypatch):
    """RAG_RERANK_MIN_SCORE=0 (explicit) → keeps everything (>= 0)."""
    monkeypatch.setenv("RAG_RERANK_MIN_SCORE", "0")

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="q",
        user_id="user-1",
    )
    assert len(out) == 5


@pytest.mark.asyncio
async def test_threshold_filters_low_scoring_hits(configure_bridge_with_mixed_scores, monkeypatch):
    """RAG_RERANK_MIN_SCORE=0.05 → keep ids 1, 2, 3 (scores 0.95, 0.50, 0.10)."""
    monkeypatch.setenv("RAG_RERANK_MIN_SCORE", "0.05")

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="q",
        user_id="user-1",
    )

    # 3 above-threshold survivors → 3 unique-doc source groups.
    assert len(out) == 3
    surviving_doc_ids = {src["metadata"][0].get("doc_id") for src in out}
    assert surviving_doc_ids == {"doc-1", "doc-2", "doc-3"}


@pytest.mark.asyncio
async def test_threshold_emits_progress_event(configure_bridge_with_mixed_scores, monkeypatch):
    """When the filter drops hits, an SSE event is emitted with dropped/kept counts."""
    monkeypatch.setenv("RAG_RERANK_MIN_SCORE", "0.05")
    monkeypatch.setenv("RAG_RERANK", "1")  # so the rerank stage is active

    events: list[dict] = []

    async def cb(ev):
        events.append(ev)

    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="q",
        user_id="user-1",
        progress_cb=cb,
    )

    threshold_events = [e for e in events if e.get("stage") == "rerank.threshold"]
    assert len(threshold_events) == 1
    ev = threshold_events[0]
    assert ev.get("dropped") == 2  # ids 4 (0.04) and 5 (0.01) dropped
    assert ev.get("kept") == 3


@pytest.mark.asyncio
async def test_threshold_above_all_scores_keeps_zero(configure_bridge_with_mixed_scores, monkeypatch):
    """Threshold above every hit's score → all hits dropped (downstream sees empty)."""
    monkeypatch.setenv("RAG_RERANK_MIN_SCORE", "1.5")

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="q",
        user_id="user-1",
    )
    # Empty post-filter; the bridge's empty-short-circuit returns [] for
    # non-metadata, non-global intents.
    assert out == []


@pytest.mark.asyncio
async def test_invalid_threshold_value_falls_open(configure_bridge_with_mixed_scores, monkeypatch):
    """Garbage env value → filter is skipped, byte-identical to unset."""
    monkeypatch.setenv("RAG_RERANK_MIN_SCORE", "not-a-float")

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="q",
        user_id="user-1",
    )
    assert len(out) == 5


def test_rag_rerank_threshold_dropped_counter_exists():
    """Sanity check — the new metric is registered and labelled by intent."""
    from ext.services import metrics

    assert hasattr(metrics, "rag_rerank_threshold_dropped_total")
    # Label values are ``intent`` only; calling .labels(intent=...) should work.
    counter = metrics.rag_rerank_threshold_dropped_total
    # The counter accepts an intent label; this should not raise.
    counter.labels(intent="specific")
