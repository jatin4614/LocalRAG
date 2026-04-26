"""Upload routes — KB documents (admin) and private chat docs (chat owner)."""
from __future__ import annotations

import os
import pathlib
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..db.models import KBDocument, KBSubtag, KnowledgeBase
from ..services import kb_service
from ..services.auth import CurrentUser, get_current_user, require_admin
from ..services.embedder import Embedder
from ..services.ingest import ingest_bytes
from ..services.kb_config import resolve_chunk_params
from ..services.metrics import upload_bytes_total
from ..services.obs import inject_context_into_headers, span
from ..services.pipeline_version import current_version
from ..services.vector_store import CHAT_PRIVATE_COLLECTION, VectorStore


router = APIRouter(tags=["upload"])


MAX_UPLOAD_BYTES = int(os.environ.get("RAG_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

# Plan B Phase 6.2 — default flipped to async after Phase 6.1 soak validation.
# Set RAG_SYNC_INGEST=1 to revert to the in-process synchronous ingest path.
# Flipped at module import; change env + restart the FastAPI process to toggle.
RAG_SYNC_INGEST = os.environ.get("RAG_SYNC_INGEST", "0") == "1"


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
    task_id: Optional[str] = None
    sha: Optional[str] = None


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
    safe_name = _safe_filename(file.filename)
    with span(
        "upload.request",
        user_id=str(getattr(user, "id", "") or ""),
        kb_id=int(kb_id),
        subtag_id=int(subtag_id),
        filename=safe_name,
        content_type=file.content_type or "",
    ):
        if await kb_service.get_kb(session, kb_id=kb_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
        sub = (
            await session.execute(
                select(KBSubtag).where(KBSubtag.id == subtag_id, KBSubtag.kb_id == kb_id)
            )
        ).scalar_one_or_none()
        if sub is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="subtag not found")

        with span("upload.read_bounded"):
            data = await _read_bounded(file)
        try:
            upload_bytes_total.labels(kb=str(kb_id)).inc(len(data))
        except Exception:
            pass

        doc = KBDocument(
            kb_id=kb_id,
            subtag_id=subtag_id,
            filename=safe_name,
            mime_type=file.content_type,
            bytes=len(data),
            uploaded_by=user.id,
            ingest_status="chunking",
            # Stamp the pipeline version that will process this upload. Rows
            # inserted before migration 004 ran will carry NULL.
            pipeline_version=current_version(),
        )
        session.add(doc)
        await session.flush()

        # P2.2: stamp the uploading admin's id on every chunk as an audit trail.
        # Retrieval does NOT filter KB collections by owner (KBs stay shared across
        # all users with kb_access grants), but the field is recorded for future
        # audit queries, data-export scrubbing, and consistency with the private
        # chat path. ``filename`` is stamped so citations work for legacy payloads
        # that never got the DB back-fill path.
        kb_payload_base = {
            "kb_id": kb_id,
            "subtag_id": subtag_id,
            "doc_id": doc.id,
            "owner_user_id": user.id,
            "filename": safe_name,
        }

        # Per-KB chunk-size override (JSONB ``rag_config``). Absent or
        # out-of-range values fall back to the process env defaults
        # (``CHUNK_SIZE``/``CHUNK_OVERLAP``) which in turn default to the
        # ``ingest_bytes`` signature defaults (800/100). kb_id is known to
        # exist — the row was validated at line 127.
        kb_row = (await session.execute(
            select(KnowledgeBase.rag_config).where(KnowledgeBase.id == kb_id)
        )).first()
        _chunk_tokens, _overlap_tokens = resolve_chunk_params(
            kb_row[0] if kb_row and kb_row[0] else None
        )

        if RAG_SYNC_INGEST:
            try:
                with span("upload.ensure_collection", collection=f"kb_{kb_id}"):
                    await _VS.ensure_collection(f"kb_{kb_id}", with_sparse=True)
                with span("upload.ingest_sync", collection=f"kb_{kb_id}", size_bytes=len(data)):
                    n = await ingest_bytes(
                        data=data,
                        mime_type=file.content_type or "application/octet-stream",
                        filename=safe_name,
                        collection=f"kb_{kb_id}",
                        payload_base=kb_payload_base,
                        vector_store=_VS,
                        embedder=_EMB,
                        chunk_tokens=_chunk_tokens,
                        overlap_tokens=_overlap_tokens,
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

        # Async path: write blob to shared store, enqueue Celery task, return task_id.
        from ..services.blob_store import BlobStore
        from ..workers.ingest_worker import ingest_blob

        with span("upload.blob_write", size_bytes=len(data)):
            store = BlobStore(os.environ.get("INGEST_BLOB_ROOT", "/var/ingest"))
            sha = store.write(data)
        await _VS.ensure_collection(f"kb_{kb_id}", with_sparse=True)
        with span("upload.enqueue_celery", collection=f"kb_{kb_id}"):
            # Propagate trace context to the Celery worker via task headers.
            task_headers = inject_context_into_headers({})
            # Stash the per-KB chunk params in payload_base so the worker
            # (which has no DB session by default) can forward them to
            # ingest_bytes without a round-trip. Worker pops these keys
            # before passing payload_base downstream since they're not
            # part of the Qdrant payload schema.
            _task_payload = dict(kb_payload_base)
            _task_payload["_chunk_tokens"] = _chunk_tokens
            _task_payload["_overlap_tokens"] = _overlap_tokens
            task = ingest_blob.apply_async(
                args=(
                    sha,
                    file.content_type or "application/octet-stream",
                    safe_name,
                    f"kb_{kb_id}",
                    _task_payload,
                ),
                headers=task_headers,
            )
        doc.ingest_status = "queued"
        await session.commit()
        return UploadResult(status="queued", chunks=0, doc_id=doc.id, task_id=task.id, sha=sha)


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
    safe_filename = _safe_filename(file.filename)
    with span(
        "upload.request",
        user_id=str(getattr(user, "id", "") or ""),
        chat_id=str(chat_id),
        filename=safe_filename,
        content_type=file.content_type or "",
    ):
        row = (await session.execute(
            text('SELECT user_id FROM chat WHERE id = :cid'),
            {"cid": chat_id},
        )).first()
        if row is None or str(row[0]) != str(user.id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")

        with span("upload.read_bounded"):
            data = await _read_bounded(file)
        try:
            upload_bytes_total.labels(kb="chat_private").inc(len(data))
        except Exception:
            pass
        # P2.3: all new private chat uploads land in the consolidated
        # ``chat_private`` collection. We ensure the collection exists with
        # sparse support so hybrid retrieval (P1.1) can use it from day one.
        # ensure_collection is idempotent — safe on every request.
        with span("upload.ensure_collection", collection=CHAT_PRIVATE_COLLECTION):
            await _VS.ensure_collection(CHAT_PRIVATE_COLLECTION, with_sparse=True)

        # payload_base stamps chat_id + owner_user_id + filename. These double as:
        #   - tenant filter keys for Qdrant's is_tenant=True indexes (P2.1),
        #   - read-path filters in retrieve() so neither cross-chat nor
        #     cross-user leaks are possible even though many chats share the
        #     same collection,
        #   - citation label (filename) visible in the UI without a DB lookup.
        chat_payload_base = {
            "chat_id": chat_id,
            "owner_user_id": user.id,
            "filename": safe_filename,
        }

        if RAG_SYNC_INGEST:
            with span(
                "upload.ingest_sync",
                collection=CHAT_PRIVATE_COLLECTION,
                size_bytes=len(data),
            ):
                n = await ingest_bytes(
                    data=data,
                    mime_type=file.content_type or "application/octet-stream",
                    filename=safe_filename,
                    collection=CHAT_PRIVATE_COLLECTION,
                    payload_base=chat_payload_base,
                    vector_store=_VS,
                    embedder=_EMB,
                )
            return UploadResult(status="done", chunks=n)

        # Async path.
        from ..services.blob_store import BlobStore
        from ..workers.ingest_worker import ingest_blob

        with span("upload.blob_write", size_bytes=len(data)):
            store = BlobStore(os.environ.get("INGEST_BLOB_ROOT", "/var/ingest"))
            sha = store.write(data)
        with span("upload.enqueue_celery", collection=CHAT_PRIVATE_COLLECTION):
            task_headers = inject_context_into_headers({})
            task = ingest_blob.apply_async(
                args=(
                    sha,
                    file.content_type or "application/octet-stream",
                    safe_filename,
                    CHAT_PRIVATE_COLLECTION,
                    chat_payload_base,
                ),
                headers=task_headers,
            )
        return UploadResult(status="queued", chunks=0, task_id=task.id, sha=sha)
