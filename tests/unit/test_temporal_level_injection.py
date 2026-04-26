"""Tests for retriever temporal level injection.

Plan B Phase 5.6.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_global_intent_injects_l3_l4(monkeypatch):
    from ext.services import retriever
    monkeypatch.setenv("RAG_TEMPORAL_LEVELS", "1")

    async def fake_search(*a, **kw):
        return [
            {"id": "p1", "score": 0.9, "payload": {"level": 0, "text": "x"}},
            {"id": "p2", "score": 0.8, "payload": {"level": 0, "text": "y"}},
        ]
    monkeypatch.setattr(retriever, "_dense_search", fake_search)

    async def fake_levels(collection, levels, top_k):
        return {
            3: [
                {"id": "y1", "score": 1.0,
                 "payload": {"level": 3, "year": 2025}},
            ],
            4: [
                {"id": "m1", "score": 1.0,
                 "payload": {"level": 4, "is_meta": True}},
            ],
        }
    monkeypatch.setattr(retriever, "_fetch_temporal_levels", fake_levels)

    hits = await retriever.retrieve_for_kb(
        collection="kb_1_v4",
        query="summarize all years",
        query_vec=[0.1] * 4,
        top_k=10,
        intent_hint="global",
    )

    levels_present = {h["payload"].get("level") for h in hits}
    assert 3 in levels_present, "global intent must include L3"
    assert 4 in levels_present, "global intent must include L4"


@pytest.mark.asyncio
async def test_evolution_intent_injects_l2_l3(monkeypatch):
    from ext.services import retriever
    monkeypatch.setenv("RAG_TEMPORAL_LEVELS", "1")

    async def fake_search(*a, **kw):
        return [
            {"id": "p1", "score": 0.9, "payload": {"level": 0, "text": "x"}},
        ]
    monkeypatch.setattr(retriever, "_dense_search", fake_search)

    async def fake_levels(collection, levels, top_k):
        return {
            2: [{"id": "q1", "score": 1.0, "payload": {"level": 2}}],
            3: [{"id": "y1", "score": 1.0, "payload": {"level": 3}}],
        }
    monkeypatch.setattr(retriever, "_fetch_temporal_levels", fake_levels)

    hits = await retriever.retrieve_for_kb(
        collection="kb_1_v4",
        query="how have budgets evolved",
        query_vec=[0.1] * 4,
        top_k=10,
        intent_hint="evolution",
    )

    levels_present = {h["payload"].get("level") for h in hits}
    assert 2 in levels_present and 3 in levels_present


@pytest.mark.asyncio
async def test_specific_date_filters_by_shard_key(monkeypatch):
    from ext.services import retriever
    monkeypatch.setenv("RAG_TEMPORAL_LEVELS", "1")

    captured: dict = {}

    async def fake_search(collection, query_vec, top_k,
                           qdrant_filter=None, **kw):
        captured["filter"] = qdrant_filter
        return [{"id": "p1", "score": 0.9, "payload": {"level": 0}}]

    monkeypatch.setattr(retriever, "_dense_search", fake_search)

    await retriever.retrieve_for_kb(
        collection="kb_1_v4",
        query="outages on 5 Jan 2026",
        query_vec=[0.1] * 4,
        top_k=10,
        intent_hint="specific_date",
        temporal_constraint={"year": 2026, "quarter": None, "month": 1},
    )

    f = captured["filter"]
    assert f is not None
    assert "2026-01" in str(f)


@pytest.mark.asyncio
async def test_temporal_levels_disabled_by_flag(monkeypatch):
    from ext.services import retriever
    monkeypatch.setenv("RAG_TEMPORAL_LEVELS", "0")

    async def fake_search(*a, **kw):
        return [{"id": "p1", "score": 0.9, "payload": {"level": 0}}]

    monkeypatch.setattr(retriever, "_dense_search", fake_search)

    called = {"count": 0}

    async def fake_levels(*a, **kw):
        called["count"] += 1
        return {}

    monkeypatch.setattr(retriever, "_fetch_temporal_levels", fake_levels)

    await retriever.retrieve_for_kb(
        collection="kb_1_v4",
        query="summarize",
        query_vec=[0.1] * 4,
        top_k=10,
        intent_hint="global",
    )
    assert called["count"] == 0  # disabled — no level fetch
