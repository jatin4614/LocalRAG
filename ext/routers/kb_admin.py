"""HTTP admin routes for KB CRUD. Admin-only."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from sqlalchemy import select

from ..services import kb_service
from ..db.models import KBDocument, KnowledgeBase
from ..services.auth import CurrentUser, require_admin
from ..services.kb_config import VALID_KEYS, validate_config
from ..services.vector_store import VectorStore

log = logging.getLogger("orgchat.kb_admin")

router = APIRouter(prefix="/api/kb", tags=["kb-admin"])

_SESSIONMAKER: async_sessionmaker[AsyncSession] | None = None
_VS: VectorStore | None = None


def set_sessionmaker(sm: async_sessionmaker[AsyncSession]) -> None:
    global _SESSIONMAKER
    _SESSIONMAKER = sm


def configure(
    sessionmaker: async_sessionmaker[AsyncSession],
    vector_store: VectorStore,
) -> None:
    global _SESSIONMAKER, _VS
    _SESSIONMAKER = sessionmaker
    _VS = vector_store


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    if _SESSIONMAKER is None:
        raise RuntimeError("sessionmaker not configured; call set_sessionmaker at app startup")
    async with _SESSIONMAKER() as s:
        yield s


class KBIn(BaseModel):
    name: str
    description: Optional[str] = None


class KBPatch(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class KBOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    admin_id: str


def _to_out(kb) -> KBOut:
    return KBOut(id=kb.id, name=kb.name, description=kb.description, admin_id=kb.admin_id)


@router.post("", response_model=KBOut, status_code=status.HTTP_201_CREATED)
async def create_kb(
    body: KBIn,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    try:
        kb = await kb_service.create_kb(session, name=body.name, description=body.description, admin_id=user.id)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(e)) from e
    return _to_out(kb)


@router.get("", response_model=list[KBOut])
async def list_kbs(
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    kbs = await kb_service.list_kbs(session)
    return [_to_out(k) for k in kbs]


@router.get("/{kb_id}", response_model=KBOut)
async def get_kb(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    kb = await kb_service.get_kb(session, kb_id=kb_id)
    if kb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    return _to_out(kb)


@router.patch("/{kb_id}", response_model=KBOut)
async def update_kb(
    kb_id: int,
    body: KBPatch,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    kb = await kb_service.update_kb(session, kb_id=kb_id, name=body.name, description=body.description)
    if kb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    await session.commit()
    return _to_out(kb)


class RAGConfigOut(BaseModel):
    kb_id: int
    rag_config: dict[str, Any]


@router.get("/{kb_id}/config", response_model=RAGConfigOut)
async def get_rag_config(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    """P3.0 — fetch per-KB retrieval quality overrides.

    Returns ``{}`` when the KB inherits process-level defaults (never null
    so admin UIs can round-trip without a null-check).
    """
    kb = await kb_service.get_kb(session, kb_id=kb_id)
    if kb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    return RAGConfigOut(kb_id=kb.id, rag_config=dict(kb.rag_config or {}))


@router.patch("/{kb_id}/config", response_model=RAGConfigOut)
async def patch_rag_config(
    kb_id: int,
    body: dict[str, Any],
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    """P3.0 — merge ``body`` into the KB's ``rag_config``.

    Partial update: only keys present in ``body`` are touched; everything
    else stays at its prior value. Unknown keys are rejected with 400 so
    admin UIs fail loudly when the frontend and backend drift out of sync.
    Values are coerced to their whitelisted types (bool / int / float)
    before persistence — the JSONB column is append-only-safe.

    Example::

        PATCH /api/kb/42/config
        {"rerank": true, "context_expand_window": 3}

    Response::

        {"kb_id": 42, "rag_config": {"rerank": true, "context_expand_window": 3}}
    """
    # Reject unknown keys up-front so the admin UI gets actionable errors.
    unknown = set(body.keys()) - set(VALID_KEYS)
    if unknown:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"unknown config keys: {sorted(unknown)}",
        )

    # Coerce to the expected Python types; any key whose value fails
    # coercion is silently dropped by validate_config, so we compare back
    # to detect bad types and raise 400.
    cleaned = validate_config(body)
    bad_types = [k for k in body if k in VALID_KEYS and k not in cleaned]
    if bad_types:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"bad value type for: {sorted(bad_types)}",
        )

    kb = await kb_service.get_kb(session, kb_id=kb_id)
    if kb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")

    # Partial merge: existing values win only for keys NOT in cleaned.
    merged: dict[str, Any] = dict(kb.rag_config or {})
    merged.update(cleaned)
    kb.rag_config = merged
    # Force SQLAlchemy to detect the JSON mutation on SQLite (where the
    # mutable dict tracker isn't wired up).
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(kb, "rag_config")
    await session.flush()
    await session.commit()
    return RAGConfigOut(kb_id=kb.id, rag_config=merged)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
) -> Response:
    ok = await kb_service.soft_delete_kb(session, kb_id=kb_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Subtag endpoints
# ---------------------------------------------------------------------------

class SubtagIn(BaseModel):
    name: str
    description: Optional[str] = None


class SubtagOut(BaseModel):
    id: int
    kb_id: int
    name: str
    description: Optional[str]


@router.post("/{kb_id}/subtags", response_model=SubtagOut, status_code=status.HTTP_201_CREATED)
async def create_subtag(
    kb_id: int, body: SubtagIn,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    if await kb_service.get_kb(session, kb_id=kb_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    try:
        sub = await kb_service.create_subtag(session, kb_id=kb_id, name=body.name, description=body.description)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(e)) from e
    return SubtagOut(id=sub.id, kb_id=sub.kb_id, name=sub.name, description=sub.description)


@router.get("/{kb_id}/subtags", response_model=list[SubtagOut])
async def list_subtags(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    subs = await kb_service.list_subtags(session, kb_id=kb_id)
    return [SubtagOut(id=s.id, kb_id=s.kb_id, name=s.name, description=s.description) for s in subs]


@router.delete("/{kb_id}/subtags/{subtag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_subtag(
    kb_id: int, subtag_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
) -> Response:
    ok = await kb_service.delete_subtag(session, kb_id=kb_id, subtag_id=subtag_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="subtag not found")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Access (RBAC) endpoints
# ---------------------------------------------------------------------------

class AccessIn(BaseModel):
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    access_type: str = "read"


class AccessOut(BaseModel):
    id: int
    kb_id: int
    user_id: Optional[str]
    group_id: Optional[str]
    access_type: str


@router.post("/{kb_id}/access", response_model=AccessOut, status_code=status.HTTP_201_CREATED)
async def grant_access(
    kb_id: int, body: AccessIn,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    if await kb_service.get_kb(session, kb_id=kb_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    try:
        g = await kb_service.grant_access(session, kb_id=kb_id, user_id=body.user_id,
                                          group_id=body.group_id, access_type=body.access_type)
        await session.commit()
    except ValueError as e:
        await session.rollback()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(e)) from e
    return AccessOut(id=g.id, kb_id=g.kb_id, user_id=g.user_id, group_id=g.group_id, access_type=g.access_type)


@router.get("/{kb_id}/access", response_model=list[AccessOut])
async def list_access(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    grants = await kb_service.list_access(session, kb_id=kb_id)
    return [AccessOut(id=g.id, kb_id=g.kb_id, user_id=g.user_id, group_id=g.group_id, access_type=g.access_type) for g in grants]


@router.delete("/{kb_id}/access/{grant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_access(
    kb_id: int, grant_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
) -> Response:
    ok = await kb_service.revoke_access(session, grant_id=grant_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="grant not found")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class DocOut(BaseModel):
    id: int
    kb_id: int
    subtag_id: int
    filename: str
    mime_type: Optional[str]
    ingest_status: str
    chunk_count: int


@router.get("/{kb_id}/documents", response_model=list[DocOut])
async def list_documents(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    rows = (await session.execute(
        select(KBDocument).where(KBDocument.kb_id == kb_id, KBDocument.deleted_at.is_(None))
        .order_by(KBDocument.id.desc())
    )).scalars().all()
    return [DocOut(id=d.id, kb_id=d.kb_id, subtag_id=d.subtag_id, filename=d.filename,
                   mime_type=d.mime_type, ingest_status=d.ingest_status, chunk_count=d.chunk_count)
            for d in rows]


@router.delete("/{kb_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    kb_id: int, doc_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    # Soft-delete the DB row
    from datetime import datetime, timezone
    from sqlalchemy import update
    r = await session.execute(
        update(KBDocument).where(KBDocument.id == doc_id, KBDocument.kb_id == kb_id)
        .values(deleted_at=datetime.now(timezone.utc))
    )
    if r.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="doc not found")

    # Delete vectors from Qdrant for this doc_id (prevents retrieval of deleted doc's chunks).
    if _VS is not None:
        try:
            await _VS.delete_by_doc(f"kb_{kb_id}", doc_id)
        except Exception as e:
            log.warning("failed to delete vectors for doc %s from kb_%s: %s", doc_id, kb_id, e)

    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{kb_id}/documents/{doc_id}/reembed")
async def reembed_document(
    kb_id: int, doc_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    doc = (await session.execute(
        select(KBDocument).where(KBDocument.id == doc_id, KBDocument.kb_id == kb_id,
                                 KBDocument.deleted_at.is_(None))
    )).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="doc not found")
    # Re-embed requires the original file — which we don't store on disk in Phase 4.
    # For now: return an error directing the admin to re-upload.
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                        detail="Re-embed requires re-upload (original files not stored on disk in this version)")


# ---------------------------------------------------------------------------
# Phase 4 — KB health endpoint
# ---------------------------------------------------------------------------

def compute_drift_pct(expected: int, observed: int) -> float:
    """Return absolute-value drift between expected and observed counts.

    ``drift_pct = |observed - expected| / expected * 100``. When ``expected``
    is 0 (empty KB) the drift is defined as 0 regardless of observed —
    we would never want alarms on newly-created KBs whose chunks are still
    ingesting, and a freshly-emptied KB (expected==0, observed==0) is also
    clean.

    Pure function. Must be import-safe (no DB / Qdrant access).
    """
    if expected <= 0:
        return 0.0
    diff = abs(int(observed) - int(expected))
    return (diff / float(expected)) * 100.0


class FailedDoc(BaseModel):
    doc_id: int
    error_message: Optional[str]


class KBHealthOut(BaseModel):
    kb_id: int
    postgres_doc_count: int
    qdrant_point_count: int
    expected_chunks_from_rows: int
    drift_pct: float
    pipeline_version_distribution: dict[str, int]
    oldest_chunk_uploaded_at: Optional[str]
    newest_chunk_uploaded_at: Optional[str]
    failed_docs: list[FailedDoc]


@router.get("/{kb_id}/health", response_model=KBHealthOut)
async def kb_health(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
) -> KBHealthOut:
    """Phase 4 — operational health snapshot for one KB.

    Joins Postgres truth (``kb_documents`` live rows) against Qdrant
    observation (``collection.count``) so the operator can spot orphan
    chunks (reingest left stale v2 points) or missing chunks (ingest failed
    silently). Also emits ``rag_kb_drift_pct{kb_id=...}`` as a side-effect
    so Prometheus scrapes record the latest value — this avoids a background
    task whose lifecycle would be awkward to manage inside FastAPI's event
    loop, and keeps the metric lazy (only refreshed when someone checks).

    Admin-only. Fail-open on Qdrant — if the collection is missing the
    endpoint still returns Postgres truth plus ``qdrant_point_count=0``.
    """
    # 1) Postgres truth
    kb = await kb_service.get_kb(session, kb_id=kb_id)
    if kb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")

    rows = (await session.execute(
        select(KBDocument).where(
            KBDocument.kb_id == kb_id,
            KBDocument.deleted_at.is_(None),
        )
    )).scalars().all()

    postgres_doc_count = len(rows)
    expected_chunks = sum(int(r.chunk_count or 0) for r in rows)

    failed = [
        FailedDoc(doc_id=r.id, error_message=r.error_message)
        for r in rows if r.ingest_status == "failed"
    ]

    # 2) Qdrant observation. Fail-open — a missing / unreachable Qdrant
    #    returns zeros, not a 500. Surfacing the zero via drift % lets
    #    operators see the problem on their dashboard instead of just an
    #    opaque API failure.
    collection = f"kb_{kb_id}"
    qdrant_point_count = 0
    pipeline_version_distribution: dict[str, int] = {}
    oldest_uploaded_at: Optional[str] = None
    newest_uploaded_at: Optional[str] = None

    if _VS is not None:
        try:
            info = await _VS._client.count(collection_name=collection, exact=True)
            qdrant_point_count = int(getattr(info, "count", 0) or 0)
        except Exception as e:
            log.info("kb_health: qdrant count failed for %s: %s", collection, e)

        # Scroll a sample for pipeline_version distribution and uploaded_at
        # bounds. Hard-capped at 2000 points per call — enough to make the
        # distribution representative on KBs up to a few tens of thousands
        # of points without hammering Qdrant. Health is an operator tool,
        # not a per-request hot path.
        try:
            seen_versions: Counter[str] = Counter()
            oldest: Optional[str] = None
            newest: Optional[str] = None
            offset = None
            pulled = 0
            LIMIT_PER_PAGE = 1024
            MAX_PULL = 2048  # two pages — reasonable upper bound for a health check
            while pulled < MAX_PULL:
                page, offset = await _VS._client.scroll(
                    collection_name=collection,
                    limit=LIMIT_PER_PAGE,
                    offset=offset,
                    with_payload=[
                        "pipeline_version",
                        "uploaded_at",
                    ],
                    with_vectors=False,
                )
                if not page:
                    break
                for pt in page:
                    payload = pt.payload or {}
                    pv = payload.get("pipeline_version") or "unknown"
                    seen_versions[pv] += 1
                    ts = payload.get("uploaded_at")
                    if ts:
                        ts_str = str(ts)
                        if oldest is None or ts_str < oldest:
                            oldest = ts_str
                        if newest is None or ts_str > newest:
                            newest = ts_str
                pulled += len(page)
                if offset is None:
                    break
            pipeline_version_distribution = dict(seen_versions)
            oldest_uploaded_at = oldest
            newest_uploaded_at = newest
        except Exception as e:
            log.info("kb_health: qdrant scroll failed for %s: %s", collection, e)

    # Fall back to Postgres' uploaded_at timestamps if Qdrant didn't
    # surface any — keeps the endpoint informative for pre-1a data that
    # never stamped uploaded_at into Qdrant payloads.
    if oldest_uploaded_at is None and rows:
        try:
            timestamps = [r.uploaded_at for r in rows if r.uploaded_at is not None]
            if timestamps:
                oldest_uploaded_at = min(timestamps).isoformat()
                newest_uploaded_at = max(timestamps).isoformat()
        except Exception:
            pass

    drift = compute_drift_pct(expected_chunks, qdrant_point_count)

    # 3) Emit the drift gauge as a side-effect. Fail-open: any metric
    #    issue must not break the endpoint.
    try:
        from ..services.metrics import rag_kb_drift_pct
        rag_kb_drift_pct.labels(kb_id=str(kb_id)).set(drift)
    except Exception:
        pass

    return KBHealthOut(
        kb_id=kb_id,
        postgres_doc_count=postgres_doc_count,
        qdrant_point_count=qdrant_point_count,
        expected_chunks_from_rows=expected_chunks,
        drift_pct=drift,
        pipeline_version_distribution=pipeline_version_distribution,
        oldest_chunk_uploaded_at=oldest_uploaded_at,
        newest_chunk_uploaded_at=newest_uploaded_at,
        failed_docs=failed,
    )
