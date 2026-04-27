"""B4 — verify ``chat_rag_bridge._run_pipeline`` emits OTel spans for every
major RAG stage so Jaeger sees per-stage timing without auto-instrumentation
(which only covers HTTP boundaries).

Strategy: monkey-patch ``ext.services.chat_rag_bridge.span`` with a recording
context manager that captures (name, attrs) for each call. Stub the rest of
the pipeline (rbac/retrieve/rerank/budget) so the bridge runs end-to-end
without network calls.

Asserted invariants:
  * ``rag.intent_classify`` runs and tags intent / source / confidence
  * ``rag.rbac_check``     runs and tags user_id + allowed_kb_count
  * ``rag.retrieve``       runs and tags hits + sharding_mode
  * ``rag.budget``         runs and tags chunks_in + chunks_kept
  * ``rag.context_inject`` runs and tags chunks_in_prompt + prompt_tokens
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

from ext.services import chat_rag_bridge as bridge


@dataclass
class _RecordedSpan:
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    exception: BaseException | None = None

    def set_attribute(self, k: str, v: Any) -> None:
        self.attrs[str(k)] = v

    def set_attributes(self, mapping: dict) -> None:
        for k, v in mapping.items():
            self.attrs[str(k)] = v

    def record_exception(self, exc: BaseException) -> None:
        self.exception = exc

    def set_status(self, *_a, **_kw) -> None:  # noqa: D401
        return None

    def add_event(self, *_a, **_kw) -> None:
        return None


class _SpanRecorder:
    def __init__(self) -> None:
        self.spans: list[_RecordedSpan] = []

    @contextmanager
    def __call__(self, name: str, **attrs: Any):
        sp = _RecordedSpan(name=name, attrs=dict(attrs))
        self.spans.append(sp)
        yield sp


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
            def first(_self):
                return None

            def all(_self):
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
                "text": "hit one " * 5,
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
                "text": "hit two " * 5,
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
    """Stub the pipeline's downstream deps + capture every span."""
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

    recorder = _SpanRecorder()
    monkeypatch.setattr(bridge, "span", recorder, raising=True)

    yield recorder

    sys.modules.pop("ext.services.context_expand", None)
    sys.modules.pop("ext.services.mmr", None)


@pytest.mark.asyncio
async def test_all_rag_stage_spans_are_emitted(configured_bridge):
    """The five canonical RAG-stage spans must all appear during a normal
    retrieve_kb_sources call."""
    out = await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    assert isinstance(out, list)

    span_names = [sp.name for sp in configured_bridge.spans]
    expected = {
        "rag.intent_classify",
        "rag.rbac_check",
        "rag.retrieve",
        "rag.budget",
        "rag.context_inject",
    }
    missing = expected - set(span_names)
    assert not missing, f"missing RAG stage spans: {missing} (got {span_names})"


@pytest.mark.asyncio
async def test_intent_classify_span_tags_intent_and_source(configured_bridge):
    """``rag.intent_classify`` must record intent label + classifier source."""
    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="what files do you have",
        user_id="user-1",
    )
    sp = next(s for s in configured_bridge.spans if s.name == "rag.intent_classify")
    # Pre-call attrs (query_len, history_turns) plus post-call attrs.
    assert "query_len" in sp.attrs
    assert "intent" in sp.attrs
    assert "source" in sp.attrs
    assert "confidence" in sp.attrs


@pytest.mark.asyncio
async def test_rbac_check_span_tags_user_and_kb_count(configured_bridge):
    """``rag.rbac_check`` must record user_id + post-resolution allowed count."""
    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-42",
    )
    sp = next(s for s in configured_bridge.spans if s.name == "rag.rbac_check")
    assert sp.attrs.get("user_id") == "user-42"
    assert sp.attrs.get("requested_kb_count") == 1
    assert sp.attrs.get("allowed_kb_count") == 1


@pytest.mark.asyncio
async def test_retrieve_span_tags_hits_and_sharding(configured_bridge):
    """``rag.retrieve`` records hit count, latency, and sharding mode."""
    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    sp = next(s for s in configured_bridge.spans if s.name == "rag.retrieve")
    assert sp.attrs.get("hits") == 2  # _fake_retrieve returns 2
    assert "latency_ms" in sp.attrs
    # sharding_mode is "all" when no level filter is forced.
    assert sp.attrs.get("sharding_mode") == "all"


@pytest.mark.asyncio
async def test_budget_span_tags_chunks(configured_bridge):
    """``rag.budget`` records chunks_in / chunks_kept."""
    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    sp = next(s for s in configured_bridge.spans if s.name == "rag.budget")
    assert sp.attrs.get("max_tokens") == 5000
    assert sp.attrs.get("chunks_in") == 2
    assert sp.attrs.get("chunks_kept") == 2  # passthrough budget stub


@pytest.mark.asyncio
async def test_context_inject_span_tags_prompt(configured_bridge):
    """``rag.context_inject`` records chunks_in_prompt + prompt_tokens."""
    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    sp = next(s for s in configured_bridge.spans if s.name == "rag.context_inject")
    assert "chunks_in_prompt" in sp.attrs
    assert "prompt_tokens" in sp.attrs
    # prompt_tokens estimate is ints >= 0.
    assert sp.attrs["prompt_tokens"] >= 0


@pytest.mark.asyncio
async def test_no_extra_unknown_rag_stage_spans(configured_bridge):
    """All ``rag.*`` spans should be one of the known set — drift detector."""
    await bridge.retrieve_kb_sources(
        kb_config=[{"kb_id": 1, "subtag_ids": []}],
        query="hello",
        user_id="user-1",
    )
    rag_spans = [sp.name for sp in configured_bridge.spans if sp.name.startswith("rag.")]
    allowed = {
        "rag.intent_classify",
        "rag.rbac_check",
        "rag.retrieve",
        "rag.budget",
        "rag.context_inject",
    }
    extras = set(rag_spans) - allowed
    assert not extras, f"unexpected rag.* spans: {extras}"
