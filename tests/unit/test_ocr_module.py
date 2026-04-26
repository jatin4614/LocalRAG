"""Plan B Phase 6.3 — OCR module unit tests.

Uses ``monkeypatch`` only (no pytest-mock) so the suite stays installable
on the existing .venv. ``pymupdf`` / ``pytesseract`` / ``PIL`` may not be
installed in the test environment — we inject lightweight stub modules
into ``sys.modules`` before the OCR helper imports them.
"""
import io
import sys
import types
import pytest

from ext.services.ocr import (
    ocr_pdf,
    OCRBackend,
    select_ocr_backend,
)


def test_select_backend_default_tesseract():
    assert select_ocr_backend(None) is OCRBackend.TESSERACT
    assert select_ocr_backend({}) is OCRBackend.TESSERACT


def test_select_backend_explicit_tesseract():
    assert select_ocr_backend({"backend": "tesseract"}) is OCRBackend.TESSERACT


def test_select_backend_textract():
    assert select_ocr_backend({"backend": "cloud:textract"}) \
        is OCRBackend.CLOUD_TEXTRACT


def test_select_backend_document_ai():
    assert select_ocr_backend({"backend": "cloud:document_ai"}) \
        is OCRBackend.CLOUD_DOCUMENT_AI


def test_select_backend_unknown_falls_back_to_tesseract():
    assert select_ocr_backend({"backend": "unknown"}) is OCRBackend.TESSERACT


@pytest.mark.asyncio
async def test_ocr_tesseract_calls_pytesseract(monkeypatch):
    """OCR module shells out to pytesseract via async wrapper."""
    fake_text = "extracted text from page"

    # Stub pymupdf — module isn't installed in the test venv.
    class _FakePixmap:
        def tobytes(self, fmt):
            return b"pretend png"

    class _FakePage:
        def get_pixmap(self, dpi=None):
            return _FakePixmap()

    class _FakeDoc:
        def __iter__(self):
            return iter([_FakePage(), _FakePage()])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def close(self):
            pass

    fake_pymupdf = types.ModuleType("pymupdf")
    fake_pymupdf.open = lambda stream=None: _FakeDoc()
    monkeypatch.setitem(sys.modules, "pymupdf", fake_pymupdf)

    # Stub pytesseract.
    fake_pytesseract = types.ModuleType("pytesseract")
    fake_pytesseract.image_to_string = lambda image, lang=None: fake_text
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    # Stub PIL.Image.open — return any sentinel; pytesseract is stubbed too.
    class _FakeImage:
        pass

    monkeypatch.setattr(
        "PIL.Image.open",
        lambda *a, **kw: _FakeImage(),
    )

    out = await ocr_pdf(
        b"%PDF-fake",
        backend=OCRBackend.TESSERACT,
        language="eng",
    )
    assert fake_text in out


@pytest.mark.asyncio
async def test_ocr_cloud_textract_unavailable_raises_clear_error(monkeypatch):
    """Cloud backends raise an actionable error if creds missing."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("TEXTRACT_REGION", raising=False)
    with pytest.raises(RuntimeError, match="TEXTRACT|AWS"):
        await ocr_pdf(
            b"%PDF-fake",
            backend=OCRBackend.CLOUD_TEXTRACT,
            language="eng",
        )
