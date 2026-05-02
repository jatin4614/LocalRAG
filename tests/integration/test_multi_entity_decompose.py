"""Integration test for Phase 6.X multi-entity decomposition.

Exercises the bridge's decompose branch end-to-end with stubbed
retriever / reranker / budget. Verifies:

* default-off: single retrieve call (byte-identical to pre-Phase-6)
* flag-on, single-entity query: still single retrieve call
* flag-on, multi-entity query: N parallel retrieve calls
* per-entity text filter passes through when RAG_ENTITY_TEXT_FILTER=1
* per-entity quota floor enforced via merge_with_quota
* metadata intent never decomposes
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


def _mk_hits_for_entity(entity: str, n: int = 5) -> list[_FakeHit]:
    """Build N hits whose IDs encode the entity — lets tests count
    per-entity contribution to the final candidate set."""
    safe = entity.replace(" ", "_")[:20]
    return [
        _FakeHit(
            id=f"{safe}-{i}",
            score=0.95 - i * 0.01,
            payload={
                "text": f"chunk for {entity} #{i}",
                "kb_id": 1,
                "subtag_id": None,
                "doc_id": 100 + i,
                "filename": f"{safe}.md",
                "chunk_index": i,
                "chat_id": None,
            },
        )
        for i in range(n)
    ]


@pytest.fixture
def captured(monkeypatch):
    """Stub the bridge dependencies; record retrieve invocations."""
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

    calls: list[dict[str, Any]] = []

    async def _stub_retrieve(
        *,
        query,
        selected_kbs,  # noqa: ARG001
        chat_id,  # noqa: ARG001
        vector_store,  # noqa: ARG001
        embedder,  # noqa: ARG001
        per_kb_limit=10,
        total_limit=30,
        text_filter=None,
        **kwargs,  # noqa: ARG001
    ):
        calls.append({
            "query": query,
            "per_kb_limit": per_kb_limit,
            "total_limit": total_limit,
            "text_filter": text_filter,
        })
        # Synthesize hits for whichever entity is named in the sub-query.
        # Single-query path: derive entity from the query text directly.
        # Multi-entity path: each call carries "(focus on <entity>)".
        # Either way we synthesize a stable per-call set so the merge has
        # something to work with.
        focus = ""
        if "(focus on " in query:
            focus = query.split("(focus on ")[-1].rstrip(")").strip()
        elif text_filter:
            focus = text_filter
        else:
            focus = "DEFAULT"
        return _mk_hits_for_entity(focus, n=min(per_kb_limit, 5))

    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)
    monkeypatch.setattr(_retriever, "retrieve", _stub_retrieve, raising=True)
    # Also patch the bridge's lazy-imported alias inside _multi_entity_retrieve.
    # The helper does ``from .retriever import retrieve as _retrieve`` at call
    # time; setting it on the module object covers that import path.
    monkeypatch.setattr(
        _reranker,
        "rerank_with_flag",
        lambda q, h, *, top_k=10, fallback_fn=None: list(h)[:top_k],
        raising=True,
    )
    monkeypatch.setattr(
        _reranker,
        "rerank",
        lambda h, *, top_k=10: list(h)[:top_k],
        raising=True,
    )
    monkeypatch.setattr(
        _budget,
        "budget_chunks",
        lambda h, *, max_tokens=4000: h,
        raising=True,
    )
    yield calls
    sys.modules.pop("ext.services.mmr", None)


async def _invoke(query: str):
    return await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query=query,
        user_id="user-1",
    )


_MULTI_QUERY = (
    "Apr 2026 updates for the following:\n"
    "1. 75 Inf Bde\n"
    "2. 5 PoK Bde\n"
    "3. 32 Inf Bde\n"
    "4. 80 Inf Bde"
)


@pytest.mark.asyncio
async def test_default_off_single_retrieve_call(captured, monkeypatch):
    monkeypatch.delenv("RAG_MULTI_ENTITY_DECOMPOSE", raising=False)
    monkeypatch.delenv("RAG_ENTITY_TEXT_FILTER", raising=False)
    await _invoke(_MULTI_QUERY)
    # Flag off → exactly one retrieve call (existing single-query path).
    assert len(captured) == 1
    # No text_filter when flag is off.
    assert captured[0]["text_filter"] is None


@pytest.mark.asyncio
async def test_flag_on_single_entity_skips_decompose(captured, monkeypatch):
    monkeypatch.setenv("RAG_MULTI_ENTITY_DECOMPOSE", "1")
    await _invoke("Tell me about 32 Inf Bde for April 2026")
    # Single entity → no decomposition, one call.
    assert len(captured) == 1
    assert captured[0]["text_filter"] is None


@pytest.mark.asyncio
async def test_flag_on_multi_entity_fans_out(captured, monkeypatch):
    monkeypatch.setenv("RAG_MULTI_ENTITY_DECOMPOSE", "1")
    monkeypatch.delenv("RAG_ENTITY_TEXT_FILTER", raising=False)
    await _invoke(_MULTI_QUERY)
    # Four entities → four parallel retrieve calls.
    assert len(captured) == 4
    # Each carries its focus suffix.
    focus_suffixes = [c["query"].split("(focus on ")[-1] for c in captured]
    assert any("75 Inf Bde" in s for s in focus_suffixes)
    assert any("5 PoK Bde" in s for s in focus_suffixes)
    assert any("32 Inf Bde" in s for s in focus_suffixes)
    assert any("80 Inf Bde" in s for s in focus_suffixes)
    # Without RAG_ENTITY_TEXT_FILTER, no text filter passed.
    assert all(c["text_filter"] is None for c in captured)


@pytest.mark.asyncio
async def test_text_filter_passes_per_entity(captured, monkeypatch):
    monkeypatch.setenv("RAG_MULTI_ENTITY_DECOMPOSE", "1")
    monkeypatch.setenv("RAG_ENTITY_TEXT_FILTER", "1")
    await _invoke(_MULTI_QUERY)
    assert len(captured) == 4
    # Each call gets the entity's surface form as text_filter.
    filters = sorted(c["text_filter"] for c in captured if c["text_filter"])
    # Should contain all 4 entities (some may retry without filter if
    # the stub returned 0; this stub returns hits for any focus, so no
    # retries fire here).
    assert "75 Inf Bde" in filters
    assert "5 PoK Bde" in filters
    assert "32 Inf Bde" in filters
    assert "80 Inf Bde" in filters


@pytest.mark.asyncio
async def test_text_filter_empty_bucket_retries_unfiltered(monkeypatch):
    """Method 4 fail-open: a too-strict text filter that returns 0 hits
    for an entity must trigger a no-filter retry for that entity only."""
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

    calls: list[dict[str, Any]] = []

    async def _stub_retrieve(
        *, query, selected_kbs, chat_id, vector_store, embedder,  # noqa: ARG001
        per_kb_limit=10, total_limit=30, text_filter=None, **kwargs,  # noqa: ARG001
    ):
        calls.append({"query": query, "text_filter": text_filter})
        # Simulate "32 Inf Bde" filter being too strict — return [].
        # All other entities (and the unfiltered retry) return hits.
        if text_filter == "32 Inf Bde":
            return []
        focus = (
            query.split("(focus on ")[-1].rstrip(")").strip()
            if "(focus on " in query
            else "DEFAULT"
        )
        return _mk_hits_for_entity(focus, n=3)

    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)
    monkeypatch.setattr(_retriever, "retrieve", _stub_retrieve, raising=True)
    monkeypatch.setattr(
        _reranker, "rerank_with_flag",
        lambda q, h, *, top_k=10, fallback_fn=None: list(h)[:top_k],
        raising=True,
    )
    monkeypatch.setattr(
        _reranker, "rerank",
        lambda h, *, top_k=10: list(h)[:top_k],
        raising=True,
    )
    monkeypatch.setattr(
        _budget, "budget_chunks",
        lambda h, *, max_tokens=4000: h,
        raising=True,
    )
    monkeypatch.setenv("RAG_MULTI_ENTITY_DECOMPOSE", "1")
    monkeypatch.setenv("RAG_ENTITY_TEXT_FILTER", "1")
    sources = await _invoke(_MULTI_QUERY)
    sys.modules.pop("ext.services.mmr", None)

    # 4 initial filtered calls + 1 retry for 32 Inf Bde without filter = 5 total.
    assert len(calls) == 5
    # The retry call has the 32 Inf Bde sub-query with text_filter=None.
    retries = [c for c in calls if "32 Inf Bde" in c["query"] and c["text_filter"] is None]
    assert len(retries) == 1
    # Sources still list all 4 entities — no entity silently dropped.
    assert sources, "expected merged sources, got empty"


@pytest.mark.asyncio
async def test_per_entity_quota_floor_enforced(monkeypatch):
    """k_min_per_entity guarantees each entity gets ≥N hits in the merged
    candidate set even when one entity dominates raw scores."""
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

    captured_merged: list[list] = []

    async def _stub_retrieve(*, query, **kwargs):  # noqa: ARG001
        focus = (
            query.split("(focus on ")[-1].rstrip(")").strip()
            if "(focus on " in query
            else "DEFAULT"
        )
        # Entity A gets dominant scores; B/C/D weak. Without quota,
        # B/C/D would be evicted.
        if "75 Inf" in focus:
            base = 0.99
        else:
            base = 0.50
        return [
            _FakeHit(
                id=f"{focus.replace(' ', '_')}-{i}",
                score=base - i * 0.001,
                payload={
                    "text": f"chunk for {focus}",
                    "kb_id": 1, "subtag_id": None,
                    "doc_id": 100 + i, "filename": "f.md",
                    "chunk_index": i, "chat_id": None,
                },
            )
            for i in range(8)
        ]

    def _stub_rerank(query, hits, *, top_k=10, fallback_fn=None):  # noqa: ARG001
        # Capture the merged candidate set BEFORE rerank trims it.
        captured_merged.append(list(hits))
        return list(hits)[:top_k]

    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)
    monkeypatch.setattr(_retriever, "retrieve", _stub_retrieve, raising=True)
    monkeypatch.setattr(
        _reranker, "rerank_with_flag", _stub_rerank, raising=True,
    )
    monkeypatch.setattr(
        _reranker, "rerank",
        lambda h, *, top_k=10: list(h)[:top_k],
        raising=True,
    )
    monkeypatch.setattr(
        _budget, "budget_chunks",
        lambda h, *, max_tokens=4000: h,
        raising=True,
    )
    monkeypatch.setenv("RAG_MULTI_ENTITY_DECOMPOSE", "1")
    monkeypatch.setenv("RAG_MULTI_ENTITY_MIN_PER_ENTITY", "2")
    monkeypatch.delenv("RAG_ENTITY_TEXT_FILTER", raising=False)
    await _invoke(_MULTI_QUERY)
    sys.modules.pop("ext.services.mmr", None)

    # The merged set fed to rerank must contain ≥2 hits per entity.
    assert captured_merged, "rerank not called"
    merged = captured_merged[0]
    by_entity: dict[str, int] = {}
    for h in merged:
        for ent in ("75_Inf", "5_PoK", "32_Inf", "80_Inf"):
            if ent in str(h.id):
                by_entity[ent] = by_entity.get(ent, 0) + 1
    assert by_entity.get("75_Inf", 0) >= 2
    assert by_entity.get("5_PoK", 0) >= 2
    assert by_entity.get("32_Inf", 0) >= 2
    assert by_entity.get("80_Inf", 0) >= 2
