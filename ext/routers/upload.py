"""Upload routes — KB documents (admin) and private chat docs (chat owner)."""
from __future__ import annotations

import os
import pathlib
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..db.models import KBDocument, KBSubtag
from ..services import kb_service
from ..services.auth import CurrentUser, get_current_user, require_admin
from ..services.embedder import Embedder
from ..services.ingest import ingest_bytes
from ..services.vector_store import VectorStore


router = APIRouter(tags=["upload"])


MAX_UPLOAD_BYTES = int(os.environ.get("RAG_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))


def _safe_truncate(msg: str, max_len: int = 1000) -> str:
    """Truncate a string at max_len without splitting multi-byte chars."""
    if len(msg) <= max_len:
        return msg
    # Truncate to max_len bytes, decode with error handling to recover from mid-codepoint split
    encoded = msg.encode("utf-8")[:max_len]
    return encoded.decode("utf-8", errors="ignore")

_SM: async_sessionmaker[AsyncSession] | None = None
_VS: VectorStore | None = None
_EMB: Embedder | None = None


def configure(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    vector_store: VectorStore,
    embedder: Embedder,
) -> None:
    global _SM, _VS, _EMB
    _SM = sessionmaker
    _VS = vector_store
    _EMB = embedder


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    if _SM is None:
        raise RuntimeError("upload router not configured")
    async with _SM() as s:
        yield s


class UploadResult(BaseModel):
    status: str
    chunks: int
    doc_id: Optional[int] = None


async def _read_bounded(file: UploadFile) -> bytes:
    """Stream-read up to MAX_UPLOAD_BYTES; reject larger files early."""
    chunks: list[bytes] = []
    total = 0
    CHUNK_SIZE = 1024 * 1024  # 1 MB
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_filename(name: str | None) -> str:
    if not name:
        return "upload"
    # Strip directory components and control chars
    clean = pathlib.Path(name).name
    clean = "".join(c for c in clean if c.isprintable() and c not in "<>&\"'")
    return clean[:256] or "upload"


@router.post(
    "/api/kb/{kb_id}/subtag/{subtag_id}/upload",
    response_model=UploadResult,
    status_code=status.HTTP_201_CREATED,
)
async def upload_kb_doc(
    kb_id: int,
    subtag_id: int,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
) -> UploadResult:
    if _VS is None or _EMB is None:
        raise RuntimeError("upload router not fully configured")
    if await kb_service.get_kb(session, kb_id=kb_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    sub = (
        await session.execute(
            select(KBSubtag).where(KBSubtag.id == subtag_id, KBSubtag.kb_id == kb_id)
        )
    ).scalar_one_or_none()
    if sub is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="subtag not found")

    data = await _read_bounded(file)

    safe_name = _safe_filename(file.filename)
    doc = KBDocument(
        kb_id=kb_id,
        subtag_id=subtag_id,
        filename=safe_name,
        mime_type=file.content_type,
        bytes=len(data),
        uploaded_by=user.id,
        ingest_status="chunking",
    )
    session.add(doc)
    await session.flush()

    try:
        await _VS.ensure_collection(f"kb_{kb_id}")
        n = await ingest_bytes(
            data=data,
            mime_type=file.content_type or "application/octet-stream",
            filename=safe_name,
            collection=f"kb_{kb_id}",
            payload_base={"kb_id": kb_id, "subtag_id": subtag_id, "doc_id": doc.id},
            vector_store=_VS,
            embedder=_EMB,
        )
    except Exception as e:
        doc.ingest_status = "failed"
        doc.error_message = _safe_truncate(str(e))
        await session.commit()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    doc.ingest_status = "done"
    doc.chunk_count = n
    await session.commit()
    return UploadResult(status="done", chunks=n, doc_id=doc.id)


@router.post(
    "/api/chats/{chat_id}/private_docs/upload",
    response_model=UploadResult,
    status_code=status.HTTP_201_CREATED,
)
async def upload_private_doc(
    chat_id: str,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
) -> UploadResult:
    if _VS is None or _EMB is None:
        raise RuntimeError("upload router not fully configured")
    row = (await session.execute(
        text('SELECT user_id FROM chat WHERE id = :cid'),
        {"cid": chat_id},
    )).first()
    if row is None or str(row[0]) != str(user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")

    data = await _read_bounded(file)
    await _VS.ensure_collection(f"chat_{chat_id}")
    n = await ingest_bytes(
        data=data,
        mime_type=file.content_type or "application/octet-stream",
        filename=_safe_filename(file.filename),
        collection=f"chat_{chat_id}",
        payload_base={"chat_id": chat_id, "owner_user_id": user.id},
        vector_store=_VS,
        embedder=_EMB,
    )
    return UploadResult(status="done", chunks=n)
