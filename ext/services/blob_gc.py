"""Blob + vector GC for soft-deleted ``kb_documents`` rows.

Queries ``kb_documents`` for rows where ``deleted_at < now() - retention_days``
and, for each match:

1. Deletes the on-disk blob via :meth:`BlobStore.delete(sha)` (idempotent:
   silently tolerates a missing file — counted as ``blobs_missing``).
2. Hard-deletes the Qdrant points for the document via
   :meth:`VectorStore.delete_by_doc(collection, doc_id)`. Note that in the
   current pipeline, vectors are already purged at soft-delete time
   (see ``ext/routers/kb_admin.py``), so this call is usually a no-op — but
   we issue it anyway because it's cheap and defensive against any prior
   deletion that silently failed.
3. Hard-deletes the ``kb_documents`` row.

All three steps run per document. A failure in step 2 (Qdrant unreachable
etc.) is logged and counted but does not block steps 1 or 3 — the blob and
the row are the local state we own and must eventually free regardless.
Idempotent: re-running against the same data set is safe and a no-op once
the row has been hard-deleted.

Callers: :mod:`scripts.blob_gc` (CLI), :mod:`ext.workers.blob_gc_task`
(Celery periodic task). Both share this implementation so behaviour is
identical whether run by cron, beat, or hand.

Summary dict returned::

    {
      "rows_processed": int,
      "blobs_deleted": int,        # blob file was present and we unlinked it
      "blobs_missing": int,        # row had blob_sha but file was already gone
      "blobs_none": int,           # row had no blob_sha (legacy / sync ingest)
      "qdrant_points_deleted": int, # delete_by_doc returned success
      "qdrant_errors": int,         # delete_by_doc raised or returned 0
      "rows_deleted": int,
      "dry_run": bool,
      "retention_days": int,
      "cutoff": str,                # ISO8601 of the cutoff timestamp
    }
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete as sql_delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models.kb import KBDocument

log = logging.getLogger("orgchat.blob_gc")


DEFAULT_RETENTION_DAYS = 30
DEFAULT_LIMIT = 1000


def retention_days() -> int:
    """Return ``RAG_BLOB_RETENTION_DAYS`` env var, defaulting to 30."""
    raw = os.environ.get("RAG_BLOB_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))
    try:
        n = int(raw)
    except ValueError:
        log.warning("invalid RAG_BLOB_RETENTION_DAYS=%r, using default %d", raw, DEFAULT_RETENTION_DAYS)
        return DEFAULT_RETENTION_DAYS
    if n < 0:
        log.warning("negative RAG_BLOB_RETENTION_DAYS=%d, using default %d", n, DEFAULT_RETENTION_DAYS)
        return DEFAULT_RETENTION_DAYS
    return n


async def run_gc(
    *,
    session: AsyncSession,
    blob_store: Any,
    vector_store: Any,
    retention_days: int,
    dry_run: bool = True,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """Run one GC pass. See module docstring for semantics."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    q = (
        select(KBDocument)
        .where(
            KBDocument.deleted_at.is_not(None),
            KBDocument.deleted_at < cutoff,
        )
        .order_by(KBDocument.deleted_at.asc())
        .limit(limit)
    )
    res = await session.execute(q)
    candidates = list(res.scalars())

    summary: dict = {
        "rows_processed": 0,
        "blobs_deleted": 0,
        "blobs_missing": 0,
        "blobs_none": 0,
        "qdrant_points_deleted": 0,
        "qdrant_errors": 0,
        "rows_deleted": 0,
        "dry_run": dry_run,
        "retention_days": retention_days,
        "cutoff": cutoff.isoformat(),
    }

    for doc in candidates:
        summary["rows_processed"] += 1

        # --- step 1: blob
        if doc.blob_sha:
            try:
                present = blob_store.exists(doc.blob_sha)
            except Exception as exc:  # noqa: BLE001
                log.warning("blob_gc: exists() failed sha=%s doc_id=%s err=%s", doc.blob_sha, doc.id, exc)
                present = False
            if present:
                if not dry_run:
                    try:
                        blob_store.delete(doc.blob_sha)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("blob_gc: delete() failed sha=%s doc_id=%s err=%s", doc.blob_sha, doc.id, exc)
                summary["blobs_deleted"] += 1
            else:
                summary["blobs_missing"] += 1
        else:
            summary["blobs_none"] += 1

        # --- step 2: Qdrant points (defensive — usually already deleted at
        # soft-delete time; idempotent, so re-issuing is safe).
        collection = f"kb_{doc.kb_id}"
        if not dry_run:
            try:
                rc = await vector_store.delete_by_doc(collection, doc.id)
                if rc:
                    summary["qdrant_points_deleted"] += 1
                else:
                    summary["qdrant_errors"] += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("blob_gc: delete_by_doc failed collection=%s doc_id=%s err=%s", collection, doc.id, exc)
                summary["qdrant_errors"] += 1
        else:
            # Dry run: count as if it would succeed (informational).
            summary["qdrant_points_deleted"] += 1

        # --- step 3: DB row
        if not dry_run:
            try:
                r = await session.execute(sql_delete(KBDocument).where(KBDocument.id == doc.id))
                summary["rows_deleted"] += int(r.rowcount or 0)
            except Exception as exc:  # noqa: BLE001
                log.error("blob_gc: row delete failed doc_id=%s err=%s", doc.id, exc)
                # Don't commit a partial batch; let the caller decide.
                raise
        else:
            summary["rows_deleted"] += 1

    if not dry_run and candidates:
        await session.commit()

    log.info(
        "blob_gc pass complete: processed=%d blobs_deleted=%d blobs_missing=%d blobs_none=%d "
        "qdrant_ok=%d qdrant_err=%d rows=%d dry_run=%s",
        summary["rows_processed"],
        summary["blobs_deleted"],
        summary["blobs_missing"],
        summary["blobs_none"],
        summary["qdrant_points_deleted"],
        summary["qdrant_errors"],
        summary["rows_deleted"],
        dry_run,
    )
    return summary
