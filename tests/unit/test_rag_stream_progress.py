"""P3.0 — progress_cb events emitted by ``retrieve_kb_sources``.

Verifies that every pipeline stage emits a running/done (or skipped)
event and that the SSE endpoint returns the correct MIME type with
correctly-encoded frames.

Strategy: stub retriever/reranker/mmr/expand/budget so no Qdrant or
embedder is touched, then drive ``retrieve_kb_sources`` with a callback
that records events. For the MIME/shape test, call the SSE route via
FastAPI's TestClient — the pipeline is still stubbed so the stream
terminates quickly.
"""
from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ext.services import chat_rag_bridge as bridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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

    async def execute(self, *a, **kw):  # noqa: ARG002
        class _R:
            def first(self_inner):  # noqa: N805
                return None

            def all(self_inner):  # noqa: N805
                return []
        return _R()


def _fake_sessionmaker():
    return _FakeSession()


async def _fake_allowed(session, *, user_id):  # noqa: ARG001
    return [1]


async def _fake_retrieve(*, query, selected_kbs, chat_id, vector_store, embedder,
                         per_kb_limit=10, total_limit=30, **kwargs):  # noqa: ARG001
    return [
        _FakeHit(
            id=1,
            score=0.9,
            payload={
                "text": "hit one",
                "kb_id": 1,
                "subtag_id": None,
                "doc_id": 10,
                "filename": "doc.md",
                "chunk_index": 5,
                "chat_id": None,
            },
        ),
        _FakeHit(
            id=2,
            score=0.8,
            payload={
                "text": "hit two",
                "kb_id": 1,
                "subtag_id": None,
                "doc_id": 11,
                "filename": "doc2.md",
                "chunk_index": 5,
                "chat_id": None,
            },
        ),
    ]


@pytest.fixture
def configured_bridge(monkeypatch):
    """Wire minimal stubs so retrieve_kb_sources runs end-to-end without
    network calls. Cleans up any stub modules on teardown."""
    sys.modules.pop("ext.services.context_expand", None)
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

    monkeypatch.setattr(_rbac, "get_allowed_kb_ids", _fake_allowed, raising=True)
    monkeypatch.setattr(_retriever, "retrieve", _fake_retrieve, raising=True)
    monkeypatch.setattr(_reranker, "rerank", lambda hits, *, top_k=10: list(hits)[:top_k], raising=True)
    monkeypatch.setattr(_budget, "budget_chunks",
                        lambda hits, *, max_tokens=4000: list(hits), raising=True)

    yield

    sys.modules.pop("ext.services.context_expand", None)
    sys.modules.pop("ext.services.mmr", None)


# ---------------------------------------------------------------------------
# progress_cb event shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_progress_cb_default_is_noop(configured_bridge):
    """retrieve_kb_sources without a progress_cb runs without error and
    returns sources normally."""
    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    assert isinstance(out, list)


@pytest.mark.asyncio
async def test_progress_cb_receives_every_stage(configured_bridge, monkeypatch):
    """With flags off, the pipeline still emits running/done for embed+
    retrieve+rerank+expand+budget+done; mmr/rerank get 'skipped' when
    off."""
    monkeypatch.delenv("RAG_RERANK", raising=False)
    monkeypatch.delenv("RAG_MMR", raising=False)
    monkeypatch.delenv("RAG_CONTEXT_EXPAND", raising=False)

    events: list[dict] = []

    async def cb(event):
        events.append(event)

    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
        progress_cb=cb,
    )

    stages = [(e.get("stage"), e.get("status")) for e in events]
    # Every pipeline stage must appear at least once.
    stage_names = {s for s, _ in stages}
    assert {"embed", "retrieve", "rerank", "mmr", "expand", "budget", "done"} <= stage_names


@pytest.mark.asyncio
async def test_rerank_skipped_when_flag_off(configured_bridge, monkeypatch):
    monkeypatch.delenv("RAG_RERANK", raising=False)

    events: list[dict] = []

    async def cb(e):
        events.append(e)

    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hi",
        user_id="u",
        progress_cb=cb,
    )

    rerank_events = [e for e in events if e.get("stage") == "rerank"]
    assert any(e.get("status") == "skipped" and e.get("reason") == "flag_off"
               for e in rerank_events)


@pytest.mark.asyncio
async def test_mmr_skipped_when_flag_off(configured_bridge, monkeypatch):
    monkeypatch.delenv("RAG_MMR", raising=False)

    events: list[dict] = []

    async def cb(e):
        events.append(e)

    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hi",
        user_id="u",
        progress_cb=cb,
    )

    mmr_events = [e for e in events if e.get("stage") == "mmr"]
    assert mmr_events
    assert any(e.get("status") == "skipped" for e in mmr_events)


@pytest.mark.asyncio
async def test_expand_skipped_when_flag_off(configured_bridge, monkeypatch):
    monkeypatch.delenv("RAG_CONTEXT_EXPAND", raising=False)

    events: list[dict] = []

    async def cb(e):
        events.append(e)

    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hi",
        user_id="u",
        progress_cb=cb,
    )

    expand_events = [e for e in events if e.get("stage") == "expand"]
    assert any(e.get("status") == "skipped" for e in expand_events)


@pytest.mark.asyncio
async def test_hits_event_contains_filenames(configured_bridge):
    events: list[dict] = []

    async def cb(e):
        events.append(e)

    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hi",
        user_id="u",
        progress_cb=cb,
    )

    hits_events = [e for e in events if e.get("stage") == "hits"]
    assert len(hits_events) == 1
    assert hits_events[0]["hits"]  # at least one entry
    assert any("filename" in entry for entry in hits_events[0]["hits"])


@pytest.mark.asyncio
async def test_done_event_has_total_ms(configured_bridge):
    events: list[dict] = []

    async def cb(e):
        events.append(e)

    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hi",
        user_id="u",
        progress_cb=cb,
    )

    done = [e for e in events if e.get("stage") == "done"]
    assert len(done) == 1
    assert "total_ms" in done[0]
    assert done[0]["total_ms"] >= 0


@pytest.mark.asyncio
async def test_progress_cb_error_does_not_break_pipeline(configured_bridge):
    """A broken callback (e.g. client disconnected mid-stream) must not
    propagate and break retrieval. The pipeline completes; the return
    value is still the sources list."""

    async def broken_cb(event):
        raise RuntimeError("client disconnected")

    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hi",
        user_id="u",
        progress_cb=broken_cb,
    )
    assert isinstance(out, list)
    # We stubbed retrieve to return 2 hits; they group into 2 sources
    # (different doc_ids).
    assert len(out) == 2


# ---------------------------------------------------------------------------
# SSE endpoint shape
# ---------------------------------------------------------------------------

def _build_sse_app(configured_bridge, monkeypatch):
    """Helper: mount only kb_retrieval + rag_stream + auth stub so we
    can hit /api/rag/stream without the full app wiring."""
    from fastapi import FastAPI

    from ext.routers import rag_stream

    monkeypatch.setenv("AUTH_MODE", "stub")

    app = FastAPI()
    app.include_router(rag_stream.router)
    return app


def test_sse_endpoint_returns_event_stream_mime(configured_bridge, monkeypatch):
    app = _build_sse_app(configured_bridge, monkeypatch)

    # Pre-stub the bridge's kb_config lookup — the SSE route calls it.
    async def fake_get_kb_config(chat_id, user_id):  # noqa: ARG001
        return [{"kb_id": 1, "subtag_ids": []}]

    monkeypatch.setattr(bridge, "get_kb_config_for_chat", fake_get_kb_config, raising=True)

    client = TestClient(app)
    response = client.get(
        "/api/rag/stream/chat-xyz?q=hello",
        headers={"X-User-Id": "user-1", "X-User-Role": "user"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_sse_endpoint_emits_stage_frames(configured_bridge, monkeypatch):
    app = _build_sse_app(configured_bridge, monkeypatch)

    async def fake_get_kb_config(chat_id, user_id):  # noqa: ARG001
        return [{"kb_id": 1, "subtag_ids": []}]

    monkeypatch.setattr(bridge, "get_kb_config_for_chat", fake_get_kb_config, raising=True)

    client = TestClient(app)
    response = client.get(
        "/api/rag/stream/chat-xyz?q=hello",
        headers={"X-User-Id": "user-1", "X-User-Role": "user"},
    )
    body = response.text
    # TestClient buffers the full stream — we can grep for well-known
    # event names without stream parsing.
    assert "event: stage" in body
    assert "event: done" in body
    # Every pipeline stage should appear as a "stage" event or inline JSON.
    for stage in ("embed", "retrieve", "rerank", "expand", "budget"):
        assert f'"stage": "{stage}"' in body


def test_sse_endpoint_400_on_empty_query(configured_bridge, monkeypatch):
    app = _build_sse_app(configured_bridge, monkeypatch)
    client = TestClient(app)
    # Missing q
    r1 = client.get(
        "/api/rag/stream/chat-xyz",
        headers={"X-User-Id": "user-1", "X-User-Role": "user"},
    )
    assert r1.status_code == 400
    # Empty q
    r2 = client.get(
        "/api/rag/stream/chat-xyz?q=",
        headers={"X-User-Id": "user-1", "X-User-Role": "user"},
    )
    assert r2.status_code == 400
