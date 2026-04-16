import io
import pytest
from ext.services.extractor import extract_text, UnsupportedMimeType


def test_txt_passthrough():
    assert extract_text(b"hello", "text/plain", "a.txt") == "hello"


def test_markdown_passthrough():
    assert extract_text(b"# Title\n\npara", "text/markdown", "a.md") == "# Title\n\npara"


def test_unsupported_mime_raises():
    with pytest.raises(UnsupportedMimeType):
        extract_text(b"...", "application/vnd.ms-excel", "a.xls")


def test_extractor_dispatches_on_extension_if_mime_missing():
    assert extract_text(b"hi", "application/octet-stream", "note.txt") == "hi"


def test_pdf_extraction_tiny():
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)
    assert extract_text(buf.getvalue(), "application/pdf", "a.pdf") == ""
