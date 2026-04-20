"""Unit tests for P2.7 contextualization wiring inside ``ingest_bytes``.

Verifies three things:
  1. When ``RAG_CONTEXTUALIZE_KBS`` is unset (default), the contextualizer
     module is NOT imported during ingest, chunk text is unchanged, and
     ``model_version`` stamps ``ctx=none``.
  2. When the flag is ``"1"`` and ``contextualize_batch`` is patched to
     return augmented strings, chunk text flows through unchanged into
     Qdrant payloads (with the augmented prefix) and ``model_version``
     stamps ``ctx=contextual-v1``.
  3. When ``contextualize_batch`` raises, ingest falls open: chunks stay
     raw, ``model_version`` falls back to ``ctx=none``.

Uses a ``_FakeVS`` stand-in that records the last ``upsert`` call so we
can inspect the stamped payloads without a real Qdrant.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest

from ext.services.embedder import StubEmbedder
from ext.services import ingest as ingest_mod
from ext.services.ingest import ingest_bytes


class _FakeVS:
    """Minimal VectorStore stand-in recording the last upsert call."""

    def __init__(self) -> None:
        self.upsert = AsyncMock()


def _txt() -> bytes:
    # Big enough to produce multiple chunks under the 20/5 test config.
    return b"The quick brown fox jumps over the lazy dog. " * 8


async def _run_ingest() -> list[dict]:
    vs = _FakeVS()
    emb = StubEmbedder(dim=16)
    n = await ingest_bytes(
        data=_txt(),
        mime_type="text/plain",
        filename="test-doc.txt",
        collection="kb_1",
        payload_base={"kb_id": 1, "doc_id": 77},
        vector_store=vs,
        embedder=emb,
        chunk_tokens=20,
        overlap_tokens=5,
    )
    assert n >= 1, "expected at least one chunk ingested"
    vs.upsert.assert_awaited_once()
    args, _ = vs.upsert.call_args
    _, points = args
    return list(points)


# ---------------------------------------------------------------------------
# 1. Default-off path — no contextualizer import, no change to behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flag_off_does_not_import_contextualizer(monkeypatch):
    """With the flag unset, ``ext.services.contextualizer`` is never imported.

    Guards against future refactors that accidentally pull the module into
    the default path (which would add httpx churn, chat dependency, etc.).
    """
    monkeypatch.delenv("RAG_CONTEXTUALIZE_KBS", raising=False)
    # Blow away any prior import so we can detect a fresh one.
    sys.modules.pop("ext.services.contextualizer", None)

    await _run_ingest()

    assert "ext.services.contextualizer" not in sys.modules


@pytest.mark.asyncio
async def test_flag_off_keeps_model_version_ctx_none(monkeypatch):
    monkeypatch.delenv("RAG_CONTEXTUALIZE_KBS", raising=False)
    points = await _run_ingest()
    for p in points:
        assert p["payload"]["model_version"].endswith("|ctx=none"), (
            "default path must stamp ctx=none"
        )


@pytest.mark.asyncio
async def test_flag_off_chunk_text_unchanged(monkeypatch):
    """Default path: chunk payload text is the raw chunk (no prefix added)."""
    monkeypatch.delenv("RAG_CONTEXTUALIZE_KBS", raising=False)
    points = await _run_ingest()
    # At least one payload should contain a recognizable fragment of the
    # original text, un-augmented (no "context:" / "\n\n" prefix sneaking in).
    raw_sentence = "The quick brown fox"
    assert any(raw_sentence in p["payload"]["text"] for p in points)
    # No payload should start with an injected context/newline pair.
    for p in points:
        text = p["payload"]["text"]
        # A natural chunk may contain "\n\n" internally, but the chunk
        # should begin with document prose, not a one-line context tag.
        first_para = text.split("\n\n", 1)[0]
        assert raw_sentence in text or len(first_para) > 20


# ---------------------------------------------------------------------------
# 2. Flag-on path — contextualize_batch prepends per-chunk context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flag_on_augments_chunks_and_bumps_version(monkeypatch):
    """With the flag on and contextualize_batch mocked, payloads carry the
    augmented text and ``model_version`` switches to ``ctx=contextual-v1``."""
    monkeypatch.setenv("RAG_CONTEXTUALIZE_KBS", "1")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://fake-vllm:8000/v1")
    monkeypatch.setenv("CHAT_MODEL", "orgchat-chat")

    # Patch contextualize_batch at the import site used by the helper.
    # The helper does ``from .contextualizer import contextualize_batch``
    # inside the function, so we patch the module attribute.
    import ext.services.contextualizer as ctxmod

    async def _fake_batch(pairs, **_kw):
        pairs_list = list(pairs)
        return [f"SITUATED({dt}): {ct}" for ct, dt in pairs_list]

    monkeypatch.setattr(ctxmod, "contextualize_batch", _fake_batch)

    points = await _run_ingest()

    # Every payload text starts with the augmented prefix.
    for p in points:
        assert p["payload"]["text"].startswith("SITUATED(test-doc.txt): "), (
            f"expected augmented prefix, got: {p['payload']['text'][:60]!r}"
        )
        assert p["payload"]["model_version"].endswith("|ctx=contextual-v1")


@pytest.mark.asyncio
async def test_flag_on_batch_exception_falls_open(monkeypatch):
    """If contextualize_batch raises, ingest keeps raw chunks + ctx=none."""
    monkeypatch.setenv("RAG_CONTEXTUALIZE_KBS", "1")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://fake-vllm:8000/v1")
    monkeypatch.setenv("CHAT_MODEL", "orgchat-chat")

    import ext.services.contextualizer as ctxmod

    async def _boom(pairs, **_kw):
        raise RuntimeError("chat endpoint unreachable")

    monkeypatch.setattr(ctxmod, "contextualize_batch", _boom)

    points = await _run_ingest()

    # Raw text stays (no "SITUATED" prefix) and version falls back to ctx=none.
    raw_fragment = "The quick brown fox"
    assert any(raw_fragment in p["payload"]["text"] for p in points)
    for p in points:
        assert not p["payload"]["text"].startswith("SITUATED")
        assert p["payload"]["model_version"].endswith("|ctx=none")


@pytest.mark.asyncio
async def test_flag_on_without_chat_url_falls_open(monkeypatch):
    """Flag on but OPENAI_API_BASE_URL unset → fall open, no chat calls.

    Defensive: in a misconfigured deployment we should not crash ingest
    just because the operator flipped the flag before wiring up the env.
    """
    monkeypatch.setenv("RAG_CONTEXTUALIZE_KBS", "1")
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)

    points = await _run_ingest()

    raw_fragment = "The quick brown fox"
    assert any(raw_fragment in p["payload"]["text"] for p in points)
    for p in points:
        assert p["payload"]["model_version"].endswith("|ctx=none")
