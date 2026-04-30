"""Unit tests for the image-caption fallback chain in ingest.py.

When a PDF has no useful embedded raster images (e.g., vector-drawn
diagrams like NFS.pdf), the pipeline must fall back to rendering each
page to PNG and captioning that — otherwise vector-drawn content is
silently dropped from retrieval.
"""
from __future__ import annotations

import io

import pytest


def _make_blank_pdf(n_pages: int = 1) -> bytes:
    """Return a tiny PDF with ``n_pages`` blank pages (no embedded rasters)."""
    from pypdf import PdfWriter

    w = PdfWriter()
    for _ in range(n_pages):
        w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_image_captions_off_returns_empty(monkeypatch):
    """RAG_IMAGE_CAPTIONS=0 (default) → returns [] without touching vision."""
    monkeypatch.delenv("RAG_IMAGE_CAPTIONS", raising=False)
    from ext.services.ingest import extract_images_as_chunks

    pdf = _make_blank_pdf()
    result = await extract_images_as_chunks(pdf_bytes=pdf, filename="x.pdf")
    assert result == []


@pytest.mark.asyncio
async def test_page_render_fallback_when_no_embedded_rasters(monkeypatch):
    """No embedded rasters → page-render fallback emits one chunk per page."""
    pytest.importorskip("pymupdf")
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")

    captured_sizes: list[int] = []

    async def _stub_caption(image_bytes):
        captured_sizes.append(len(image_bytes))
        return f"stub caption ({len(image_bytes)} bytes)"

    from ext.services import ingest

    monkeypatch.setattr(ingest, "_caption_image", _stub_caption)

    pdf = _make_blank_pdf(n_pages=2)
    result = await ingest.extract_images_as_chunks(
        pdf_bytes=pdf, filename="x.pdf",
    )

    assert len(result) == 2
    assert all(r["chunk_type"] == "image_caption" for r in result)
    assert [r["payload"]["page"] for r in result] == [1, 2]
    # Page-render position marker distinguishes from embedded extraction.
    assert all(r["payload"]["position"] == ("page-render",) for r in result)
    assert len(captured_sizes) == 2
    assert all(b > 0 for b in captured_sizes), "expected non-empty PNG bytes"


@pytest.mark.asyncio
async def test_page_render_fallback_can_be_disabled(monkeypatch):
    """RAG_RENDER_PDF_PAGES_FOR_VISION=0 → no fallback; no captions emitted."""
    pytest.importorskip("pymupdf")
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")
    monkeypatch.setenv("RAG_RENDER_PDF_PAGES_FOR_VISION", "0")

    from ext.services import ingest

    pdf = _make_blank_pdf()
    result = await ingest.extract_images_as_chunks(
        pdf_bytes=pdf, filename="x.pdf",
    )
    assert result == []


@pytest.mark.asyncio
async def test_max_pages_caps_render_count(monkeypatch):
    """RAG_RENDER_PDF_MAX_PAGES caps how many pages are rendered."""
    pytest.importorskip("pymupdf")
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")
    monkeypatch.setenv("RAG_RENDER_PDF_MAX_PAGES", "2")

    async def _stub_caption(image_bytes):
        return "stub"

    from ext.services import ingest

    monkeypatch.setattr(ingest, "_caption_image", _stub_caption)

    pdf = _make_blank_pdf(n_pages=5)
    result = await ingest.extract_images_as_chunks(
        pdf_bytes=pdf, filename="x.pdf",
    )
    assert len(result) == 2
    assert [r["payload"]["page"] for r in result] == [1, 2]


@pytest.mark.asyncio
async def test_min_bytes_filter_skips_decorative_embedded(monkeypatch):
    """Tiny embedded shapes (decorative) → filtered out → fall back to page-render."""
    pytest.importorskip("pymupdf")
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")
    monkeypatch.setenv("RAG_VISION_RASTER_MIN_BYTES", "10000")  # 10 KB

    async def _stub_extract_pdf_images(pdf_bytes):
        # Simulate the 263-byte decorative glyph from NFS.pdf.
        return [{"page": 1, "image_bytes": b"x" * 500, "position": (0,)}]

    async def _stub_caption(image_bytes):
        return "stub"

    from ext.services import ingest

    monkeypatch.setattr(ingest, "_extract_pdf_images", _stub_extract_pdf_images)
    monkeypatch.setattr(ingest, "_caption_image", _stub_caption)

    pdf = _make_blank_pdf(n_pages=1)
    result = await ingest.extract_images_as_chunks(
        pdf_bytes=pdf, filename="x.pdf",
    )

    # The 500-byte embedded shape was below threshold → filtered out.
    # Fallback rendered the 1 page → 1 caption with page-render position.
    assert len(result) == 1
    assert result[0]["payload"]["position"] == ("page-render",)
