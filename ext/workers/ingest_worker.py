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


def _safe_truncate(msg: str, max_len: int = 500) -> str:
    """Truncate a string to a UTF-8 byte budget without splitting codepoints.

    Bug-fix campaign §1.7: the failure-path previously stamped
    ``error_message`` via ``f"...{exc}"[:500]``. Python's ``str[:N]``
    slices by codepoints, not bytes, so a 500-codepoint string can hold
    up to 2000 bytes of CJK / Devanagari / emoji — bypassing the
    intended size cap. Worse, the sync-upload sibling helper in
    ``ext/routers/upload.py`` (also named ``_safe_truncate``) treats
    ``max_len`` as bytes; the two paths produced different
    ``error_message`` strings on identical errors.

    Mirrors the upload-router helper byte-for-byte: encode → byte slice
    → decode with ``errors='ignore'`` so a mid-codepoint cut silently
    drops the dangling partial bytes instead of producing mojibake or
    a U+FFFD replacement char.
    """
    if len(msg.encode("utf-8")) <= max_len:
        return msg
    encoded = msg.encode("utf-8")[:max_len]
    return encoded.decode("utf-8", errors="ignore")


# Lazy module-level engine cache. Bug-fix campaign §1.6: prior to this fix,
# every ``_update_doc_status`` / ``_fetch_kb_rag_config`` call created and
# disposed a SQLAlchemy engine, costing ~1000 inits on a 1000-doc batch.
# The first call instantiates a single engine bound to the resolved
# ``DATABASE_URL``; subsequent calls in the same worker process reuse it.
# Indirected via ``_create_async_engine`` so tests can monkeypatch the
# factory without importing sqlalchemy.
_engine_singleton: object | None = None


def _create_async_engine(db_url: str):
    """Thin wrapper around ``sqlalchemy.ext.asyncio.create_async_engine``.

    Exists as a separate module attribute so tests can monkeypatch the
    engine factory directly (see ``tests/unit/test_ingest_worker_engine_singleton.py``).
    """
    from sqlalchemy.ext.asyncio import create_async_engine as _ce
    return _ce(db_url, pool_pre_ping=True)


def _normalize_db_url() -> str:
    """Return the DATABASE_URL coerced to the asyncpg dialect, or ``""``."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return ""
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return db_url


def _get_engine():
    """Return the cached async engine, creating it on first call.

    Returns ``None`` when ``DATABASE_URL`` is unset (test or stand-alone mode).
    """
    global _engine_singleton
    if _engine_singleton is not None:
        return _engine_singleton
    db_url = _normalize_db_url()
    if not db_url:
        return None
    _engine_singleton = _create_async_engine(db_url)
    return _engine_singleton


async def _update_doc_status(
    doc_id: int | str | None,
    status: str,
    *,
    chunk_count: int | None = None,
    error_message: str | None = None,
) -> None:
    """Update ``kb_documents.ingest_status`` (and optionally chunk_count
    or error_message).

    Best-effort, fire-and-forget: any DB error is logged + swallowed so a
    transient Postgres blip never causes the task itself to fail. Skips when
    doc_id is missing (private chat-scoped uploads have chat_id, not doc_id).

    ``error_message`` is written when provided (typically alongside
    ``status='failed'``) so admins can see WHY a doc failed without
    grepping celery logs. Mirrors the sync upload path in
    ext/routers/upload.py which also stamps error_message on failure.
    """
    if doc_id is None:
        return
    engine = _get_engine()
    if engine is None:
        return
    try:
        from sqlalchemy import text as _sql
        sets = ["ingest_status = :s"]
        params: dict[str, object] = {"s": status, "i": int(doc_id)}
        if chunk_count is not None:
            sets.append("chunk_count = :c")
            params["c"] = int(chunk_count)
        if error_message is not None:
            sets.append("error_message = :e")
            params["e"] = error_message
        sql = f"UPDATE kb_documents SET {', '.join(sets)} WHERE id = :i"
        async with engine.begin() as conn:
            await conn.execute(_sql(sql), params)
    except Exception as e:  # noqa: BLE001 — best-effort
        log.warning(
            "ingest: failed to update kb_documents.ingest_status "
            "doc_id=%s status=%s err=%s",
            doc_id, status, e,
        )


async def _fetch_kb_rag_config(kb_id: int) -> dict | None:
    """Read ``knowledge_bases.rag_config`` for a KB.

    Used at task start so the worker can honour per-KB overrides
    (image_captions, chunking_strategy, contextualize) without
    requiring the HTTP producer to stash the entire JSONB blob in
    payload_base. Returns ``None`` on DB error or when the row is
    absent — caller treats that as "no per-KB override".
    """
    engine = _get_engine()
    if engine is None:
        return None
    from sqlalchemy import text as _sql
    async with engine.begin() as conn:
        row = (await conn.execute(
            _sql("SELECT rag_config FROM knowledge_bases WHERE id = :i"),
            {"i": int(kb_id)},
        )).first()
    if row is None:
        return None
    cfg = row[0]
    return dict(cfg) if cfg else None


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

    # Fetch per-KB rag_config so ingest can honour per-KB overrides
    # (chunking_strategy, image_captions, contextualize, etc.). The
    # producer doesn't stash the full rag_config in payload_base because
    # it's a JSONB blob that may be large; cheaper to look it up here at
    # task start time. Falls back to None on DB error → ingest sees
    # env-only behaviour (pre-Phase-6 path, byte-identical).
    kb_rag_config: dict | None = None
    kb_id = pb.get("kb_id")
    if kb_id is not None:
        try:
            kb_rag_config = await _fetch_kb_rag_config(int(kb_id))
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.warning(
                "ingest_worker: failed to load rag_config for kb_id=%s: %s",
                kb_id, exc,
            )

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
            "kb_rag_config": kb_rag_config,
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
                asyncio.run(_update_doc_status(
                    _doc_id, "failed",
                    error_message=_safe_truncate(
                        f"{type(exc).__name__}: {exc}", 500,
                    ),
                ))
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
