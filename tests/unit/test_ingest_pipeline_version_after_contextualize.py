"""Bug-fix campaign §1.12 — ``kb_documents.pipeline_version`` is stamped
at upload time but never refreshed when contextualization actually runs.

Per CLAUDE.md the column tracks "which pipeline version produced the
embeddings". Upload-time stamps ``ctx=none`` (because contextualize is a
runtime decision based on per-KB ``rag_config``); when ``ingest_bytes``
later flips ``context_augmented=True`` and stamps every Qdrant point's
payload with ``ctx=contextual-v1``, the Postgres column is left lying.
Drift between the column and the points means downstream tooling
(``reembed_all.py``, the kb_admin drift dashboard) thinks the doc was
*not* contextualized.

The fix: after contextualize fires, ``ingest_bytes`` invokes the worker's
``_update_doc_pipeline_version`` to refresh the column. Tests pin the
contract via a stub callback.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_pipeline_version_stamped_when_contextualize_runs(monkeypatch):
    """When contextualize fires successfully, the persistence callback
    must receive the contextualized pipeline_version string."""
    from ext.services import ingest as ing

    captured: dict = {}

    async def _fake_update(doc_id, pv):
        captured["doc_id"] = doc_id
        captured["pv"] = pv

    # Fake out contextualize: pretend it ran and returned new text.
    async def _fake_maybe_ctx(chunks_and_blocks, *, doc_title):
        return True  # context_augmented = True

    monkeypatch.setattr(ing, "_maybe_contextualize_chunks", _fake_maybe_ctx)
    monkeypatch.setattr(
        ing, "_persist_doc_pipeline_version", _fake_update, raising=False,
    )
    # Force contextualize to be requested (env-flag gate).
    monkeypatch.setenv("RAG_CONTEXTUALIZE_KBS", "1")

    # Build a minimal fake extract → 1 block.
    from ext.services.extractor import ExtractedBlock

    def _fake_extract(*a, **kw):
        return [ExtractedBlock(
            text="hello world", page=1, sheet=None, heading_path=["Intro"],
        )]

    monkeypatch.setattr(ing, "extract", _fake_extract)

    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.0] * 4 for _ in texts]

    class _FakeVectorStore:
        async def upsert(self, *a, **kw):
            return None

    await ing.ingest_bytes(
        data=b"x", mime_type="text/plain", filename="x.txt",
        collection="kb_999",
        payload_base={
            "kb_id": 999, "doc_id": 42, "subtag_id": None,
            "owner_user_id": "u", "filename": "x.txt",
        },
        vector_store=_FakeVectorStore(),
        embedder=_FakeEmbedder(),
    )
    assert captured.get("doc_id") == 42
    pv = captured.get("pv", "")
    assert "ctx=contextual-v1" in pv, f"expected contextual-v1 in pv, got {pv!r}"


@pytest.mark.asyncio
async def test_pipeline_version_NOT_stamped_when_contextualize_skipped(monkeypatch):
    """When the env flag is off, the persistence callback must NOT be
    called — keeps the legacy default-off path byte-identical."""
    from ext.services import ingest as ing

    captured: dict = {}

    async def _fake_update(doc_id, pv):
        captured["called"] = True

    monkeypatch.setattr(
        ing, "_persist_doc_pipeline_version", _fake_update, raising=False,
    )
    monkeypatch.delenv("RAG_CONTEXTUALIZE_KBS", raising=False)

    from ext.services.extractor import ExtractedBlock

    def _fake_extract(*a, **kw):
        return [ExtractedBlock(
            text="hello world", page=1, sheet=None, heading_path=["Intro"],
        )]

    monkeypatch.setattr(ing, "extract", _fake_extract)

    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.0] * 4 for _ in texts]

    class _FakeVectorStore:
        async def upsert(self, *a, **kw):
            return None

    await ing.ingest_bytes(
        data=b"x", mime_type="text/plain", filename="x.txt",
        collection="kb_999",
        payload_base={
            "kb_id": 999, "doc_id": 42, "subtag_id": None,
            "owner_user_id": "u", "filename": "x.txt",
        },
        vector_store=_FakeVectorStore(),
        embedder=_FakeEmbedder(),
    )
    assert "called" not in captured


@pytest.mark.asyncio
async def test_persist_pipeline_version_helper_exists():
    """The helper symbol must exist so tests / hooks can monkeypatch it."""
    from ext.services import ingest as ing
    assert hasattr(ing, "_persist_doc_pipeline_version")
