"""OCR fallback for scanned PDFs.

Plan B Phase 6.3. Default backend: Tesseract (air-gap safe).
Optional cloud backends (per-KB opt-in): AWS Textract, GCP Document AI.

Trigger: ingest detects pages with < N text characters via pdfplumber
(see Phase 6.4), rasterizes those pages, runs OCR, returns extracted
text. The result is concatenated with any pdfplumber text and re-fed
into the chunker.

The cloud backends are the only code path in this codebase that can
make outbound network calls. They are gated by per-KB policy
(``kb_config.ocr_policy.backend``) and disabled globally unless an
operator explicitly opts in. They WILL FAIL CLOSED if their credentials
are missing — never silently fall back to Tesseract.
"""
from __future__ import annotations

import asyncio
import enum
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor


log = logging.getLogger("orgchat.ocr")
_executor = ThreadPoolExecutor(max_workers=2)


class OCRBackend(enum.Enum):
    TESSERACT = "tesseract"
    CLOUD_TEXTRACT = "cloud:textract"
    CLOUD_DOCUMENT_AI = "cloud:document_ai"


def select_ocr_backend(policy: dict | None) -> OCRBackend:
    """Map a kb_config.ocr_policy.backend string to OCRBackend.

    Unknown values fall back to TESSERACT (safer default — never accidentally
    upload to a cloud endpoint).
    """
    if not policy:
        return OCRBackend.TESSERACT
    raw = (policy.get("backend") or "tesseract").lower().strip()
    try:
        return OCRBackend(raw)
    except ValueError:
        log.warning("Unknown OCR backend %r; defaulting to tesseract", raw)
        return OCRBackend.TESSERACT


async def ocr_pdf(
    pdf_bytes: bytes,
    *,
    backend: OCRBackend = OCRBackend.TESSERACT,
    language: str = "eng",
) -> str:
    """Extract text from a PDF via the named backend.

    Returns the concatenated text of all pages. Raises RuntimeError if
    cloud backend credentials are missing.
    """
    if backend is OCRBackend.TESSERACT:
        return await _ocr_tesseract(pdf_bytes, language=language)
    if backend is OCRBackend.CLOUD_TEXTRACT:
        return await _ocr_textract(pdf_bytes)
    if backend is OCRBackend.CLOUD_DOCUMENT_AI:
        return await _ocr_document_ai(pdf_bytes)
    raise ValueError(f"unsupported OCR backend {backend!r}")


async def _ocr_tesseract(pdf_bytes: bytes, *, language: str) -> str:
    import pymupdf
    import pytesseract
    from PIL import Image

    def _run() -> str:
        out_pages = []
        with pymupdf.open(stream=pdf_bytes) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang=language)
                out_pages.append(text)
        return "\n\n".join(out_pages)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run)


async def _ocr_textract(pdf_bytes: bytes) -> str:
    region = os.environ.get("TEXTRACT_REGION")
    if not region or not os.environ.get("AWS_ACCESS_KEY_ID"):
        raise RuntimeError(
            "AWS Textract requires TEXTRACT_REGION + AWS_ACCESS_KEY_ID"
        )
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 not installed; cannot use Textract") from e

    def _run() -> str:
        client = boto3.client("textract", region_name=region)
        resp = client.detect_document_text(Document={"Bytes": pdf_bytes})
        lines = [
            b["Text"] for b in resp.get("Blocks", [])
            if b.get("BlockType") == "LINE"
        ]
        return "\n".join(lines)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run)


async def _ocr_document_ai(pdf_bytes: bytes) -> str:
    project = os.environ.get("DOCUMENT_AI_PROJECT")
    location = os.environ.get("DOCUMENT_AI_LOCATION", "us")
    processor = os.environ.get("DOCUMENT_AI_PROCESSOR")
    if not project or not processor:
        raise RuntimeError(
            "Document AI requires DOCUMENT_AI_PROJECT + DOCUMENT_AI_PROCESSOR"
        )
    try:
        from google.cloud import documentai_v1
    except ImportError as e:
        raise RuntimeError("google-cloud-documentai not installed") from e

    def _run() -> str:
        client = documentai_v1.DocumentProcessorServiceClient()
        name = client.processor_path(project, location, processor)
        raw_doc = documentai_v1.RawDocument(
            content=pdf_bytes, mime_type="application/pdf",
        )
        req = documentai_v1.ProcessRequest(name=name, raw_document=raw_doc)
        result = client.process_document(request=req)
        return result.document.text

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run)


__all__ = ["OCRBackend", "select_ocr_backend", "ocr_pdf"]
