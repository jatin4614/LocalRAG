"""Celery task for asynchronous ingest.

Producer (FastAPI) writes the raw bytes to the shared blob store, then enqueues
``ingest_blob`` with the content sha. This worker reads the bytes, extracts →
chunks → embeds → upserts, then deletes the blob.

Failure policy:
  * Blob missing when task picks up → ``Reject(requeue=False)`` → DLQ.
  * Any other exception → retry with exponential backoff, up to 3 retries.
  * After retries exhausted → ``Reject(requeue=False)`` → DLQ.

acks_late + reject_on_worker_lost (configured on the Celery app) ensure a
crashed worker redelivers the task rather than losing it.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Mapping

from celery import Task
from celery.exceptions import Reject

from ..services.blob_store import BlobStore
from ..services.embedder import TEIEmbedder
from ..services.ingest import ingest_bytes
from ..services.obs import extract_context_from_headers, get_tracer
from ..services.vector_store import VectorStore
from .celery_app import app

log = logging.getLogger("orgchat.ingest_worker")


async def _update_doc_status(
    doc_id: int | str | None,
    status: str,
    *,
    chunk_count: int | None = None,
) -> None:
    """Update ``kb_documents.ingest_status`` (and optionally chunk_count).

    Best-effort, fire-and-forget: any DB error is logged + swallowed so a
    transient Postgres blip never causes the task itself to fail. Skips when
    doc_id is missing (private chat-scoped uploads have chat_id, not doc_id).
    """
    if doc_id is None:
        return
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    try:
        from sqlalchemy import text as _sql
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(db_url, pool_pre_ping=True)
        try:
            async with engine.begin() as conn:
                if chunk_count is not None:
                    await conn.execute(
                        _sql(
                            "UPDATE kb_documents "
                            "SET ingest_status = :s, chunk_count = :c "
                            "WHERE id = :i"
                        ),
                        {"s": status, "c": int(chunk_count), "i": int(doc_id)},
                    )
                else:
                    await conn.execute(
                        _sql(
                            "UPDATE kb_documents SET ingest_status = :s "
                            "WHERE id = :i"
                        ),
                        {"s": status, "i": int(doc_id)},
                    )
        finally:
            await engine.dispose()
    except Exception as e:  # noqa: BLE001 — best-effort
        log.warning(
            "ingest: failed to update kb_documents.ingest_status "
            "doc_id=%s status=%s err=%s",
            doc_id, status, e,
        )


def _blob_root() -> str:
    return os.environ.get("INGEST_BLOB_ROOT", "/var/ingest")


def _store() -> BlobStore:
    return BlobStore(_blob_root())


async def _do_ingest(
    data: bytes,
    mime_type: str,
    filename: str,
    collection: str,
    payload_base: Mapping[str, Any],
) -> int:
    """Construct per-call VectorStore + Embedder (both own async httpx clients)
    and run the full pipeline. Returns chunk count.

    ``payload_base`` may carry ``_chunk_tokens`` / ``_overlap_tokens`` keys
    stashed by the HTTP producer (``upload.py``) so this worker honours a
    per-KB override without re-reading the DB. Those keys are popped
    before being passed downstream — the chunk params are call args to
    ``ingest_bytes``, not part of the Qdrant payload.
    """
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    tei_url = os.environ.get("TEI_URL", "http://tei:80")
    vector_size = int(os.environ.get("RAG_VECTOR_SIZE", "1024"))

    # Extract per-KB chunk params if the producer stashed them. Default to
    # env/ingest-signature behaviour when absent.
    pb = dict(payload_base)
    chunk_tokens = pb.pop("_chunk_tokens", None)
    overlap_tokens = pb.pop("_overlap_tokens", None)

    vs = VectorStore(url=qdrant_url, vector_size=vector_size)
    emb = TEIEmbedder(base_url=tei_url)
    try:
        await vs.ensure_collection(collection)
        kwargs = {
            "data": data,
            "mime_type": mime_type,
            "filename": filename,
            "collection": collection,
            "payload_base": pb,
            "vector_store": vs,
            "embedder": emb,
        }
        if chunk_tokens is not None:
            kwargs["chunk_tokens"] = int(chunk_tokens)
        if overlap_tokens is not None:
            kwargs["overlap_tokens"] = int(overlap_tokens)
        return await ingest_bytes(**kwargs)
    finally:
        try:
            await emb.aclose()
        except Exception:
            pass
        try:
            await vs.close()
        except Exception:
            pass


@app.task(bind=True, max_retries=3, queue="ingest", name="ext.workers.ingest_worker.ingest_blob")
def ingest_blob(
    self: Task,
    sha: str,
    mime_type: str,
    filename: str,
    collection: str,
    payload_base: dict,
) -> dict:
    """Read blob by sha, ingest it, delete the blob. Return status dict."""
    # Extract W3C trace context from task headers (populated by the HTTP
    # producer via ``inject_context_into_headers``) so this task's root
    # span is a child of the originating HTTP upload span. Fail-open — a
    # missing/invalid traceparent yields a fresh root span.
    _hdrs = {}
    try:
        _hdrs = dict(getattr(self.request, "headers", None) or {})
    except Exception:
        _hdrs = {}
    _parent_ctx = extract_context_from_headers(_hdrs)
    _doc_id = None
    try:
        _doc_id = payload_base.get("doc_id") if isinstance(payload_base, dict) else None
    except Exception:
        _doc_id = None
    _kb_id = None
    try:
        _kb_id = payload_base.get("kb_id") if isinstance(payload_base, dict) else None
    except Exception:
        _kb_id = None

    tracer = get_tracer("orgchat")
    # Explicit root span independent of CeleryInstrumentor — guarantees a
    # trace even if the auto-instrumentor failed to initialize.
    _span_cm = tracer.start_as_current_span(
        "ingest.celery_task",
        context=_parent_ctx if _parent_ctx is not None else None,
    )
    with _span_cm as _sp:
        try:
            if _sp is not None:
                _sp.set_attribute("celery.task_name", "ingest_blob")
                _sp.set_attribute("sha", sha or "")
                _sp.set_attribute("collection", collection)
                _sp.set_attribute("mime_type", mime_type or "")
                _sp.set_attribute("filename", filename or "")
                if _doc_id is not None:
                    _sp.set_attribute("doc_id", str(_doc_id))
                if _kb_id is not None:
                    _sp.set_attribute("kb_id", str(_kb_id))
        except Exception:
            pass

        # Pull KB id out of the collection name once for progress events.
        # Format is always ``kb_{int}`` for KB ingests; private chat docs
        # use ``chat_{id}`` and skip progress emit (no admin UI subscribes).
        _kb_id_for_progress: int | None = None
        if collection.startswith("kb_"):
            try:
                _kb_id_for_progress = int(collection.split("_", 1)[1])
            except (ValueError, IndexError):
                _kb_id_for_progress = None

        from ext.services.ingest_progress import emit_sync as _emit_progress

        if _kb_id_for_progress is not None:
            _emit_progress(_kb_id_for_progress, {
                "doc_id": _doc_id,
                "filename": filename,
                "stage": "processing",
            })

        store = _store()
        if not store.exists(sha):
            # Permanent failure — blob the producer promised us was never written
            # (or was GC'd). Reject without requeue → DLQ.
            log.error("ingest: blob missing sha=%s collection=%s", sha, collection)
            if _kb_id_for_progress is not None:
                _emit_progress(_kb_id_for_progress, {
                    "doc_id": _doc_id, "filename": filename,
                    "stage": "failed", "error": "blob missing",
                })
            raise Reject(reason=f"blob missing: {sha}", requeue=False)

        try:
            data = store.read(sha)
            n = asyncio.run(_do_ingest(data, mime_type, filename, collection, payload_base))
        except Exception as exc:  # noqa: BLE001 — retry policy lives here
            if self.request.retries < 3:
                log.warning(
                    "ingest: transient failure sha=%s retries=%d err=%s",
                    sha,
                    self.request.retries,
                    exc,
                )
                # Exponential backoff: 1s, 2s, 4s
                raise self.retry(exc=exc, countdown=2 ** self.request.retries)
            log.error("ingest: exhausted retries sha=%s err=%s", sha, exc)
            try:
                asyncio.run(_update_doc_status(_doc_id, "failed"))
            except Exception:  # noqa: BLE001 — best-effort
                pass
            if _kb_id_for_progress is not None:
                _emit_progress(_kb_id_for_progress, {
                    "doc_id": _doc_id, "filename": filename,
                    "stage": "failed", "error": str(exc)[:200],
                })
            raise Reject(reason=str(exc), requeue=False)

        # Success: best-effort blob cleanup (idempotent).
        try:
            store.delete(sha)
        except Exception:  # noqa: BLE001 — cleanup is non-critical
            log.warning("ingest: blob cleanup failed sha=%s", sha)

        # Plan B Phase 6.2 followup: transition kb_documents.ingest_status
        # from 'queued' (set by upload route on async dispatch) to 'done'
        # so the admin UI + cron sweepers see the correct lifecycle state.
        try:
            asyncio.run(_update_doc_status(_doc_id, "done", chunk_count=int(n)))
        except Exception:  # noqa: BLE001 — best-effort
            pass

        if _kb_id_for_progress is not None:
            _emit_progress(_kb_id_for_progress, {
                "doc_id": _doc_id, "filename": filename,
                "stage": "done", "chunks": int(n),
            })

        try:
            if _sp is not None:
                _sp.set_attribute("chunks", int(n))
        except Exception:
            pass
        return {"status": "ok", "chunks": n, "sha": sha}
