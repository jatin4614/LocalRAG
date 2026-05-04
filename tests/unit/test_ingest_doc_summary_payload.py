"""Test that ingest.py wires summarize_document's structured return into
the level=doc Qdrant payload (entities field).

Phase 2 / item 4 of the 2026-05-04 multi-entity-elaborate-answers spec.

The plan-spec test was authored assuming module-level ``embedder`` /
``vector_store`` singletons in ``ext.services.ingest``. This codebase
actually wires those via DI — they are passed as parameters into
``ingest_bytes`` and forwarded to ``_emit_doc_summary_point``. So the
test below pattern-matches the codebase reality: it constructs mock
``VectorStore`` / ``Embedder`` doubles, and only patches the one
genuinely module-imported symbol — ``summarize_document`` — to return
the new structured ``{entities, summary}`` shape.

Invariant covered (identical to the plan-spec text): the level=doc
upsert payload MUST stamp an ``entities`` field carrying the
summariser-returned list, and the ``text`` field MUST be the combined
``"ENTITIES: ..\n\nSUMMARY: .."`` shape so the dense retriever sees
both signals.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch


async def test_doc_summary_upsert_includes_entities_payload() -> None:
    """When summarize_document returns {entities: [...], summary: "..."},
    the level=doc Qdrant point payload MUST include an 'entities' field
    AND a combined ENTITIES + SUMMARY text field."""

    upserted: list[tuple[str, list[dict]]] = []

    class _FakeVectorStore:
        async def upsert(self, collection, points, **kw):
            upserted.append((collection, list(points)))

    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.0] * 1024 for _ in texts]

    with patch(
        "ext.services.ingest.summarize_document",
        new_callable=AsyncMock,
    ) as mock_sum:
        mock_sum.return_value = {
            "entities": ["75 Inf Bde", "5 PoK Bde", "32 Inf Bde"],
            "summary": "Test summary mentioning every entity.",
        }

        from ext.services import ingest

        summary_dict = await ingest._emit_doc_summary_point(
            kb_id=2,
            doc_id=999,
            subtag_id=11,
            filename="Apr 26.docx",
            chunk_texts=["a", "b"],
            chat_url="http://fake/v1",
            chat_model="fake",
            vector_store=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
        )

    # Helper returned the same dict the summariser produced (so the
    # caller can mirror summary_dict["summary"] into Postgres).
    assert summary_dict == {
        "entities": ["75 Inf Bde", "5 PoK Bde", "32 Inf Bde"],
        "summary": "Test summary mentioning every entity.",
    }

    # Find the doc-summary point in the upserts.
    doc_points = [
        p
        for _coll, pts in upserted
        for p in pts
        if (p.get("payload") or {}).get("level") == "doc"
    ]
    assert len(doc_points) == 1, f"expected 1 doc point, got {len(doc_points)}"
    payload = doc_points[0]["payload"]
    assert payload["entities"] == ["75 Inf Bde", "5 PoK Bde", "32 Inf Bde"]
    assert "Test summary" in payload["text"]
    assert "ENTITIES: 75 Inf Bde, 5 PoK Bde, 32 Inf Bde" in payload["text"]
    # Also assert SUMMARY: marker present so the parsed shape mirrors
    # the doc_summarizer ENTITIES + SUMMARY contract.
    assert "SUMMARY: Test summary mentioning every entity." in payload["text"]


async def test_doc_summary_skipped_when_summary_empty() -> None:
    """If summarize_document returns an empty summary string, no
    upsert happens and the helper returns the empty-shape dict."""

    upserted: list[tuple[str, list[dict]]] = []

    class _FakeVectorStore:
        async def upsert(self, collection, points, **kw):
            upserted.append((collection, list(points)))

    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.0] * 1024 for _ in texts]

    with patch(
        "ext.services.ingest.summarize_document",
        new_callable=AsyncMock,
    ) as mock_sum:
        mock_sum.return_value = {"entities": [], "summary": ""}

        from ext.services import ingest

        out = await ingest._emit_doc_summary_point(
            kb_id=2,
            doc_id=42,
            subtag_id=None,
            filename="empty.docx",
            chunk_texts=["a"],
            chat_url="http://fake/v1",
            chat_model="fake",
            vector_store=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
        )

    assert out == {"entities": [], "summary": ""}
    assert upserted == []


async def test_doc_summary_text_field_is_summary_only_when_no_entities() -> None:
    """When entities list is empty but summary is present, text_field
    is just the bare summary (no "ENTITIES:" prefix)."""

    upserted: list[tuple[str, list[dict]]] = []

    class _FakeVectorStore:
        async def upsert(self, collection, points, **kw):
            upserted.append((collection, list(points)))

    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.0] * 1024 for _ in texts]

    with patch(
        "ext.services.ingest.summarize_document",
        new_callable=AsyncMock,
    ) as mock_sum:
        mock_sum.return_value = {
            "entities": [],
            "summary": "Plain summary without entities.",
        }

        from ext.services import ingest

        await ingest._emit_doc_summary_point(
            kb_id=2,
            doc_id=43,
            subtag_id=None,
            filename="plain.docx",
            chunk_texts=["a"],
            chat_url="http://fake/v1",
            chat_model="fake",
            vector_store=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
        )

    doc_points = [
        p
        for _coll, pts in upserted
        for p in pts
        if (p.get("payload") or {}).get("level") == "doc"
    ]
    assert len(doc_points) == 1
    payload = doc_points[0]["payload"]
    assert payload["entities"] == []
    assert payload["text"] == "Plain summary without entities."
