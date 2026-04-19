"""Structural extractor tests (P0.4).

Verifies that :func:`ext.services.extractor.extract` returns
:class:`ExtractedBlock` objects with format-appropriate metadata:

* PDF  → ``page`` set (1-based)
* DOCX → ``heading_path`` tracks the Heading N style stack
* MD   → ``heading_path`` tracks ATX heading stack
* TXT  → no structural metadata
* XLSX → ``sheet`` set per worksheet

``python-docx`` and ``openpyxl`` are not in the test venv (they're only
installed in the Celery worker image). Tests that need them
``pytest.importorskip`` so they skip cleanly in the dev loop instead of
failing spuriously.
"""
from __future__ import annotations

import io

import pytest

from ext.services.extractor import ExtractedBlock, extract, extract_flat


# ---------------------------------------------------------------------------
# TXT / MD
# ---------------------------------------------------------------------------

def test_txt_produces_single_block_no_structure():
    blocks = extract(b"hello world", "text/plain", "a.txt")
    assert len(blocks) == 1
    b = blocks[0]
    assert isinstance(b, ExtractedBlock)
    assert b.text == "hello world"
    assert b.page is None
    assert b.heading_path == []
    assert b.sheet is None


def test_txt_empty_returns_no_blocks():
    assert extract(b"", "text/plain", "a.txt") == []


def test_markdown_tracks_heading_path():
    md = b"# Title\n\nbody\n\n## Sub\n\nbody2"
    blocks = extract(md, "text/markdown", "a.md")
    assert len(blocks) == 2
    assert blocks[0].heading_path == ["Title"]
    assert blocks[0].text == "body"
    assert blocks[1].heading_path == ["Title", "Sub"]
    assert blocks[1].text == "body2"


def test_markdown_pops_stack_on_shallower_heading():
    md = b"# A\n\nalpha\n\n## B\n\nbeta\n\n# C\n\ngamma"
    blocks = extract(md, "text/markdown", "a.md")
    paths = [b.heading_path for b in blocks]
    assert paths == [["A"], ["A", "B"], ["C"]]


def test_markdown_paragraph_above_first_heading_has_empty_path():
    md = b"intro line\n\n# Title\n\nbody"
    blocks = extract(md, "text/markdown", "a.md")
    assert blocks[0].heading_path == []
    assert blocks[0].text == "intro line"
    assert blocks[1].heading_path == ["Title"]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def test_pdf_empty_page_produces_no_blocks():
    """Blank pages are skipped — extract_flat still keeps "\n\n" separators
    (legacy compat), but blocks only surface pages with actual text."""
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)
    blocks = extract(buf.getvalue(), "application/pdf", "a.pdf")
    assert blocks == []


# A fixture-backed multi-page PDF with real text would need reportlab or
# a checked-in binary. Skipped for now — reportlab is not in the test venv
# and the plan calls out that the concrete PDF case will land with its own
# fixture file later in P0.
@pytest.mark.skip(reason="requires fixture PDF with real text — to be added with reportlab fixture")
def test_pdf_text_pages_have_page_numbers():
    pass


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def _build_docx_with_headings():
    docx = pytest.importorskip("docx", reason="python-docx not installed in test venv")
    Document = docx.Document
    doc = Document()
    doc.add_paragraph("Intro paragraph")  # before any heading
    doc.add_heading("H1", level=1)
    doc.add_paragraph("paragraph")
    doc.add_heading("H2", level=2)
    doc.add_paragraph("paragraph2")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_docx_tracks_heading_path():
    data = _build_docx_with_headings()
    blocks = extract(
        data,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "h.docx",
    )
    # Three prose paragraphs: intro (no heading yet), under H1, under H1>H2.
    prose = [b for b in blocks if b.heading_path is not None and b.text.strip()]
    assert len(prose) >= 3
    assert prose[0].heading_path == []
    assert prose[0].text.strip() == "Intro paragraph"
    assert prose[1].heading_path == ["H1"]
    assert prose[1].text.strip() == "paragraph"
    assert prose[2].heading_path == ["H1", "H2"]
    assert prose[2].text.strip() == "paragraph2"


def test_docx_without_headings_still_extracts_paragraphs():
    docx = pytest.importorskip("docx", reason="python-docx not installed in test venv")
    Document = docx.Document
    doc = Document()
    doc.add_paragraph("alpha")
    doc.add_paragraph("beta")
    buf = io.BytesIO()
    doc.save(buf)
    blocks = extract(
        buf.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "h.docx",
    )
    # Each paragraph emits a block with empty heading_path.
    assert [b.text.strip() for b in blocks[:2]] == ["alpha", "beta"]
    assert all(b.heading_path == [] for b in blocks[:2])


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------

def _build_multi_sheet_xlsx():
    openpyxl = pytest.importorskip("openpyxl", reason="openpyxl not installed in test venv")
    wb = openpyxl.Workbook()
    # openpyxl creates one default sheet named "Sheet" — rename it then add a second.
    ws1 = wb.active
    ws1.title = "Alpha"
    ws1.append(["a", "b", "c"])
    ws1.append([1, 2, 3])
    ws2 = wb.create_sheet("Beta")
    ws2.append(["x", "y"])
    ws2.append([10, 20])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_xlsx_one_block_per_sheet_with_sheet_name():
    data = _build_multi_sheet_xlsx()
    blocks = extract(
        data,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "b.xlsx",
    )
    assert len(blocks) == 2
    assert {b.sheet for b in blocks} == {"Alpha", "Beta"}
    # Every block must have non-empty text (we skip empty sheets) and
    # carry no paginated / heading metadata.
    for b in blocks:
        assert b.text
        assert b.page is None
        assert b.heading_path == []


# ---------------------------------------------------------------------------
# Legacy flat-string byte identity
# ---------------------------------------------------------------------------

def test_extract_flat_txt_identical_to_legacy():
    assert extract_flat(b"hello", "text/plain", "a.txt") == "hello"


def test_extract_flat_markdown_preserves_raw_text():
    """Legacy flat extractor was ``data.decode()`` — verbatim passthrough."""
    md = b"# Title\n\nbody\n\n## Sub\n\nbody2"
    assert extract_flat(md, "text/markdown", "a.md") == md.decode("utf-8")
