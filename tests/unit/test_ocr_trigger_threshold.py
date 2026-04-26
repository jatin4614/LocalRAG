"""Plan B Phase 6.4 — OCR trigger threshold tests.

Uses ``monkeypatch`` only (no pytest-mock).
"""
import pytest


@pytest.mark.asyncio
async def test_ocr_triggers_when_low_text_pages(monkeypatch):
    monkeypatch.setenv("RAG_OCR_ENABLED", "1")
    monkeypatch.setenv("RAG_OCR_TRIGGER_CHARS", "50")

    from ext.services import ingest

    # Stub pdfplumber per-page extraction to return [low_text_page, normal_page]
    fake_pdf_text = ["", "this is a normal page with plenty of extracted text"]
    monkeypatch.setattr(
        ingest, "_extract_pdf_text_per_page", lambda b: fake_pdf_text,
    )

    # Stub OCR to return synthetic
    async def fake_ocr(pdf_bytes, *, backend, language):
        return "OCR-RECOVERED-TEXT"

    monkeypatch.setattr(ingest, "_ocr_pdf_pages", fake_ocr)

    text = await ingest.extract_pdf_with_ocr_fallback(
        pdf_bytes=b"%PDF-fake", filename="x.pdf", ocr_policy=None,
    )
    assert "OCR-RECOVERED-TEXT" in text
    assert "normal page" in text  # non-OCR pages included


@pytest.mark.asyncio
async def test_ocr_does_not_trigger_when_all_pages_have_text(monkeypatch):
    monkeypatch.setenv("RAG_OCR_ENABLED", "1")
    monkeypatch.setenv("RAG_OCR_TRIGGER_CHARS", "50")

    from ext.services import ingest

    fake_pdf_text = [
        "page one has plenty of text far above the threshold for OCR",
        "page two also has plenty of text well over fifty characters",
    ]
    monkeypatch.setattr(
        ingest, "_extract_pdf_text_per_page", lambda b: fake_pdf_text,
    )

    called = {"n": 0}

    async def spy_ocr(*a, **kw):
        called["n"] += 1
        return "should-not-be-called"

    monkeypatch.setattr(ingest, "_ocr_pdf_pages", spy_ocr)

    text = await ingest.extract_pdf_with_ocr_fallback(
        pdf_bytes=b"%PDF-fake", filename="x.pdf", ocr_policy=None,
    )
    assert called["n"] == 0
    assert "page one" in text


@pytest.mark.asyncio
async def test_ocr_disabled_globally(monkeypatch):
    monkeypatch.setenv("RAG_OCR_ENABLED", "0")
    from ext.services import ingest

    monkeypatch.setattr(
        ingest, "_extract_pdf_text_per_page", lambda b: ["", ""],
    )

    called = {"n": 0}

    async def spy_ocr(*a, **kw):
        called["n"] += 1
        return ""

    monkeypatch.setattr(ingest, "_ocr_pdf_pages", spy_ocr)

    await ingest.extract_pdf_with_ocr_fallback(
        pdf_bytes=b"%PDF-fake", filename="x.pdf", ocr_policy=None,
    )
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_ocr_per_kb_policy_overrides_global(monkeypatch):
    monkeypatch.setenv("RAG_OCR_ENABLED", "1")
    from ext.services import ingest

    monkeypatch.setattr(ingest, "_extract_pdf_text_per_page", lambda b: [""])

    called = {"n": 0}

    async def spy_ocr(*a, **kw):
        called["n"] += 1
        return ""

    monkeypatch.setattr(ingest, "_ocr_pdf_pages", spy_ocr)

    # Per-KB explicitly disables
    await ingest.extract_pdf_with_ocr_fallback(
        pdf_bytes=b"%PDF-fake",
        filename="x.pdf",
        ocr_policy={"enabled": False},
    )
    assert called["n"] == 0
