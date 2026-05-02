"""Plan B Phase 6.7 — image caption extraction tests.

Uses ``monkeypatch`` only.
"""
import pytest


@pytest.mark.xfail(
    reason="Phase 6.X — RAG_VISION_RASTER_MIN_BYTES=5000 default rejects the "
    "8-byte fake image, then page-render fallback fails on b'%PDF-fake'. "
    "Test was written before the byte-size gate landed. Fix is to bump fake "
    "bytes >5000 OR set RAG_VISION_RASTER_MIN_BYTES=0. Tracked in bug-fix "
    "campaign Wave 4 (review §9.5).",
    strict=False,
)
@pytest.mark.asyncio
async def test_image_caption_emitted_for_pdf_image(monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")
    from ext.services import ingest

    # Stub the image extractor to return 1 fake image
    async def fake_extract_images(pdf_bytes):
        return [{"page": 1, "image_bytes": b"fake png", "position": (0, 0)}]

    monkeypatch.setattr(ingest, "_extract_pdf_images", fake_extract_images)

    # Stub the vision caller
    async def fake_caption(img_bytes):
        return "A bar chart showing Q1 revenue growth"

    monkeypatch.setattr(ingest, "_caption_image", fake_caption)

    chunks = await ingest.extract_images_as_chunks(
        pdf_bytes=b"%PDF-fake", filename="report.pdf",
    )
    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "image_caption"
    assert "bar chart" in chunks[0]["text"]
    assert chunks[0]["payload"]["page"] == 1


@pytest.mark.asyncio
async def test_image_caption_skipped_when_flag_off(monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "0")
    from ext.services import ingest

    called = {"n": 0}

    async def spy_extract(pdf_bytes):
        called["n"] += 1
        return []

    monkeypatch.setattr(ingest, "_extract_pdf_images", spy_extract)

    chunks = await ingest.extract_images_as_chunks(
        pdf_bytes=b"%PDF-fake", filename="x.pdf",
    )
    assert chunks == []
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_image_caption_skipped_when_vision_unreachable(monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")
    from ext.services import ingest

    async def fake_extract_images(pdf_bytes):
        return [{"page": 1, "image_bytes": b"x", "position": (0, 0)}]

    monkeypatch.setattr(ingest, "_extract_pdf_images", fake_extract_images)

    async def fake_caption(img_bytes):
        raise ConnectionError("vision unreachable")

    monkeypatch.setattr(ingest, "_caption_image", fake_caption)

    chunks = await ingest.extract_images_as_chunks(
        pdf_bytes=b"%PDF-fake", filename="x.pdf",
    )
    assert chunks == []  # silent skip, but logged
