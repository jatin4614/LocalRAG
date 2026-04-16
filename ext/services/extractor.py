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


EXTRACTORS: dict[str, Callable[[bytes], str]] = {
    "text/plain": _extract_txt,
    "text/markdown": _extract_txt,
    "application/pdf": _extract_pdf,
}


_EXT_FALLBACK = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
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
