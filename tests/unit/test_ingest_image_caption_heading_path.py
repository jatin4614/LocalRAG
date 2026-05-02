"""Bug-fix campaign §1.13 — image-caption chunks should inherit the
heading_path of the ExtractedBlock whose page matches the image's page,
not always blocks[0].heading_path.

Before the fix: every image_caption chunk was paired with ``blocks[0]``
as the "host" block, so an image on page 5 of a 100-page PDF inherited
the page-1 chapter heading. Citations and faceted retrieval were
mis-attributed; long PDFs with chapter-level headings were the worst
hit.

After the fix: ``ingest_bytes`` looks up the block whose ``page``
matches the image's ``page`` and uses that block's ``heading_path``.
When no block matches (e.g. images on a page that yielded zero text
blocks), heading_path falls back to ``[]`` so the payload is honest
about its provenance.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_image_caption_inherits_page_aware_heading_path(monkeypatch):
    """Two blocks on different pages with different heading_paths;
    image on page 2 must get page-2's heading_path.
    """
    from ext.services import ingest as ing
    from ext.services.extractor import ExtractedBlock

    captured: list = []

    # Override extract → 2 blocks at different pages w/ different headings.
    def _fake_extract(*a, **kw):
        return [
            ExtractedBlock(
                text="page 1 body",
                page=1,
                sheet=None,
                heading_path=["Chapter 1", "Intro"],
            ),
            ExtractedBlock(
                text="page 2 body",
                page=2,
                sheet=None,
                heading_path=["Chapter 2", "Methods"],
            ),
        ]

    monkeypatch.setattr(ing, "extract", _fake_extract)

    # Force image-caption gate ON.
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")

    async def _fake_extract_images(*, pdf_bytes, filename):
        return [
            {
                "text": "[image caption page 2] a chart",
                "page": 2,
            }
        ]

    monkeypatch.setattr(ing, "extract_images_as_chunks", _fake_extract_images)

    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.0] * 4 for _ in texts]

    class _FakeVectorStore:
        async def upsert(self, collection, points):
            captured.extend(points)

    await ing.ingest_bytes(
        data=b"x", mime_type="application/pdf", filename="x.pdf",
        collection="kb_999",
        payload_base={
            "kb_id": 999, "doc_id": 42, "subtag_id": None,
            "owner_user_id": "u", "filename": "x.pdf",
        },
        vector_store=_FakeVectorStore(),
        embedder=_FakeEmbedder(),
    )

    # Find the image_caption point in the upserted batch.
    image_pt = next(
        p for p in captured
        if p["payload"].get("chunk_type") == "image_caption"
    )
    assert image_pt["payload"]["page"] == 2
    # The bug: image inherited Chapter 1 / Intro from blocks[0].
    # The fix: image inherits Chapter 2 / Methods from page-2's block.
    assert list(image_pt["payload"]["heading_path"]) == ["Chapter 2", "Methods"]


@pytest.mark.asyncio
async def test_image_caption_falls_back_to_empty_when_no_page_match(monkeypatch):
    """Image on a page with no extracted text → heading_path = []."""
    from ext.services import ingest as ing
    from ext.services.extractor import ExtractedBlock

    captured: list = []

    def _fake_extract(*a, **kw):
        return [
            ExtractedBlock(
                text="page 1 only",
                page=1,
                sheet=None,
                heading_path=["Chapter 1"],
            ),
        ]

    monkeypatch.setattr(ing, "extract", _fake_extract)
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")

    async def _fake_extract_images(*, pdf_bytes, filename):
        return [
            {
                "text": "[image caption page 7] a figure",
                "page": 7,  # no block on page 7
            }
        ]

    monkeypatch.setattr(ing, "extract_images_as_chunks", _fake_extract_images)

    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.0] * 4 for _ in texts]

    class _FakeVectorStore:
        async def upsert(self, collection, points):
            captured.extend(points)

    await ing.ingest_bytes(
        data=b"x", mime_type="application/pdf", filename="x.pdf",
        collection="kb_999",
        payload_base={
            "kb_id": 999, "doc_id": 42, "subtag_id": None,
            "owner_user_id": "u", "filename": "x.pdf",
        },
        vector_store=_FakeVectorStore(),
        embedder=_FakeEmbedder(),
    )

    image_pt = next(
        p for p in captured
        if p["payload"].get("chunk_type") == "image_caption"
    )
    assert image_pt["payload"]["page"] == 7
    # No matching block on page 7 → empty heading_path (honest about
    # provenance, no false attribution to Chapter 1).
    assert list(image_pt["payload"]["heading_path"]) == []


@pytest.mark.asyncio
async def test_image_caption_no_page_in_image_dict_falls_back_to_empty(monkeypatch):
    """Image with no ``page`` key (extractor couldn't determine which
    page the image lives on) → heading_path = []."""
    from ext.services import ingest as ing
    from ext.services.extractor import ExtractedBlock

    captured: list = []

    def _fake_extract(*a, **kw):
        return [
            ExtractedBlock(
                text="page 1 only",
                page=1,
                sheet=None,
                heading_path=["Chapter 1"],
            ),
        ]

    monkeypatch.setattr(ing, "extract", _fake_extract)
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")

    async def _fake_extract_images(*, pdf_bytes, filename):
        return [
            {
                "text": "[image caption unknown page] a figure",
                # no "page" key
            }
        ]

    monkeypatch.setattr(ing, "extract_images_as_chunks", _fake_extract_images)

    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.0] * 4 for _ in texts]

    class _FakeVectorStore:
        async def upsert(self, collection, points):
            captured.extend(points)

    await ing.ingest_bytes(
        data=b"x", mime_type="application/pdf", filename="x.pdf",
        collection="kb_999",
        payload_base={
            "kb_id": 999, "doc_id": 42, "subtag_id": None,
            "owner_user_id": "u", "filename": "x.pdf",
        },
        vector_store=_FakeVectorStore(),
        embedder=_FakeEmbedder(),
    )

    image_pt = next(
        p for p in captured
        if p["payload"].get("chunk_type") == "image_caption"
    )
    # No page → empty heading_path (no spurious inherit from blocks[0]).
    assert list(image_pt["payload"]["heading_path"]) == []
