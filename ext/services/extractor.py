"""Text extraction from uploaded documents. Plug-in via EXTRACTORS dict."""
from __future__ import annotations

import io
from typing import Callable


class UnsupportedMimeType(RuntimeError):
    pass


def _extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


def _extract_xlsx(data: bytes) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts: list[str] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                parts.append("\t".join(cells))
    return "\n".join(parts)


def _extract_csv(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


EXTRACTORS: dict[str, Callable[[bytes], str]] = {
    "text/plain": _extract_txt,
    "text/markdown": _extract_txt,
    "text/csv": _extract_csv,
    "application/pdf": _extract_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": _extract_docx,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": _extract_xlsx,
    "application/msword": _extract_docx,  # .doc (best effort via python-docx)
}


_EXT_FALLBACK = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
}


def extract_text(data: bytes, mime_type: str, filename: str) -> str:
    fn = EXTRACTORS.get(mime_type)
    if fn is None:
        for ext, m in _EXT_FALLBACK.items():
            if filename.lower().endswith(ext):
                fn = EXTRACTORS[m]
                break
    if fn is None:
        raise UnsupportedMimeType(f"{mime_type} (filename={filename})")
    return fn(data)
