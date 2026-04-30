"""HTTP admin routes for KB CRUD. Admin-only."""
from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, constr
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..services import kb_service
from ..db.models import KBAccess, KBDocument, KnowledgeBase
from ..services.auth import CurrentUser, require_admin
from ..services.kb_config import VALID_KEYS, validate_config
from ..services.rbac import users_affected_by_grant
from ..services.rbac_cache import get_shared_cache
from ..services.vector_store import VectorStore

log = logging.getLogger("orgchat.kb_admin")

# Phase 1.5 — process-wide redis handle for the RBAC cache invalidation
# pub/sub. Lazy-init on first grant mutation so unit tests that never
# touch /access don't open a redis connection.
_rbac_redis = None


def _redis_client():
    """Return the shared async redis handle for RBAC cache invalidation.

    Uses ``RAG_RBAC_CACHE_REDIS_URL`` (default ``redis://localhost:6379/3``)
    so it's isolated from the application redis on DB 0.
    """
    global _rbac_redis
    if _rbac_redis is None:
        import redis.asyncio as _redis
        url = os.environ.get(
            "RAG_RBAC_CACHE_REDIS_URL", "redis://localhost:6379/3"
        )
        _rbac_redis = _redis.from_url(url)
    return _rbac_redis


async def _invalidate_rbac_cache_for_grant(
    session: AsyncSession, grant: KBAccess
) -> None:
    """Drop cached ``allowed_kb_ids`` for every user this grant changes.

    Phase 1.5 — invariant: any ``kb_access`` mutation MUST trigger a
    cache invalidation so users see the new permission within
    ``RAG_RBAC_CACHE_TTL_SECS`` (default 30s) at worst, and immediately
    on the publishing replica. The TTL is the safety net for dropped
    pub/sub messages; this call is the fast path.

    Fail-open: any redis error is logged at WARNING level. The grant
    mutation itself MUST NOT roll back -- DB truth must always win, and
    the worst case (cache write failure) is a stale entry for one TTL
    window.
    """
    try:
        affected = await users_affected_by_grant(session, grant)
    except Exception as exc:  # noqa: BLE001
        log.warning("rbac cache: users_affected_by_grant failed: %s", exc)
        return
    if not affected:
        return
    try:
        cache = get_shared_cache(redis=_redis_client())
        await cache.invalidate(user_ids=affected)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "rbac cache: invalidate failed for users %s: %s", affected, exc
        )

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
    # H3: enforce name bounds at the API edge so a 256-char string can
    # never slip through to the DB (knowledge_bases.name is VARCHAR(255)).
    # ``strip_whitespace`` ensures ``"   "`` collapses to ``""`` and is
    # rejected by the min_length=1 floor.
    name: constr(min_length=1, max_length=255, strip_whitespace=True)
    description: Optional[constr(max_length=2000)] = None


class KBPatch(BaseModel):
    # H4: same constraints as KBIn but Optional so PATCH bodies can
    # update only ``description`` without touching ``name``.
    name: Optional[constr(min_length=1, max_length=255, strip_whitespace=True)] = None
    description: Optional[constr(max_length=2000)] = None


class KBOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    admin_id: str


class KBListOut(BaseModel):
    """Paginated KB list response (H2)."""
    items: list[KBOut]
    total_count: int


def _to_out(kb) -> KBOut:
    return KBOut(id=kb.id, name=kb.name, description=kb.description, admin_id=kb.admin_id)


@router.post("", response_model=KBOut, status_code=status.HTTP_201_CREATED)
async def create_kb(
    body: KBIn,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    """H8: catch IntegrityError specifically (concurrent same-name insert
    races past kb_service's pre-check) and ValueError from the service-
    layer pre-check; both surface as 409 with sanitized messages. Other
    exceptions propagate to the standard 500 with full server-side
    logging — they indicate a real bug, not a benign user error.
    """
    try:
        kb = await kb_service.create_kb(
            session, name=body.name, description=body.description, admin_id=user.id,
        )
        await session.commit()
    except ValueError as e:
        # Service-layer pre-check ("kb name already in use: ...")
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(e)) from e
    except IntegrityError as e:
        await session.rollback()
        log.warning("create_kb: integrity error name=%r: %s", body.name, e)
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="name already in use",
        ) from e
    except Exception as e:
        await session.rollback()
        log.exception("create_kb: unexpected error")
        raise
    return _to_out(kb)


@router.get("", response_model=KBListOut)
async def list_kbs(
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """H2: paginated KB listing.

    Response shape: ``{items: [KBOut, ...], total_count: int}``.
    ``limit`` is bounded [1, 1000]; ``offset`` is non-negative. The
    total_count is a separate COUNT(*) query — it reflects all live KBs
    regardless of the current page so the UI can render "showing 1-100 of
    347" indicators.
    """
    kbs = await kb_service.list_kbs(session, limit=limit, offset=offset)
    total = await kb_service.count_kbs(session)
    return KBListOut(items=[_to_out(k) for k in kbs], total_count=total)


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
    # coercion OR violates H5 bounds is silently dropped by
    # validate_config, so we compare back to detect bad input and
    # raise 400. The error message intentionally says "bad value (type
    # or out of range)" since both reasons surface here — admins
    # debugging a rejected key should check both.
    cleaned = validate_config(body)
    bad_keys = [k for k in body if k in VALID_KEYS and k not in cleaned]
    if bad_keys:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=(
                f"bad value (type or out of range) for: {sorted(bad_keys)}"
            ),
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
    except IntegrityError as e:
        # H8: subtags have UNIQUE(kb_id, name) — duplicate name within
        # the same KB violates the index.
        await session.rollback()
        log.warning(
            "create_subtag: integrity error kb=%s name=%r: %s",
            kb_id, body.name, e,
        )
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="subtag name already in use within this KB",
        ) from e
    except Exception:
        await session.rollback()
        log.exception("create_subtag: unexpected error kb=%s", kb_id)
        raise
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
    """H7: cascade Qdrant chunk delete + soft-delete docs + soft-delete subtag.

    Order of operations:
      1. List live (deleted_at IS NULL) docs in this subtag.
      2. For each doc, best-effort ``VectorStore.delete_by_doc(...)``;
         log failures but don't abort — the docs / subtag get soft-
         deleted regardless so they stop appearing in admin lists.
      3. Soft-delete the docs (UPDATE deleted_at = NOW()).
      4. Soft-delete the subtag itself.

    Cross-agent dep: this assumes ``kb_subtags.deleted_at`` exists
    (Agent B's migration). The route uses
    :func:`kb_service.soft_delete_subtag` which falls back gracefully
    when the column isn't there yet (returns False), surfacing as a
    404 — operators will know the migration hasn't been applied.
    """
    # 1) Find live docs in this subtag.
    doc_rows = (await session.execute(
        select(KBDocument).where(
            KBDocument.kb_id == kb_id,
            KBDocument.subtag_id == subtag_id,
            KBDocument.deleted_at.is_(None),
        )
    )).scalars().all()

    # 2) Best-effort Qdrant cascade. Failures don't abort the soft-delete.
    if _VS is not None and doc_rows:
        collection = f"kb_{kb_id}"
        for doc in doc_rows:
            try:
                await _VS.delete_by_doc(collection, doc.id)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "delete_subtag: vector cascade failed kb=%s sub=%s "
                    "doc=%s err=%s",
                    kb_id, subtag_id, doc.id, e,
                )

    # 3) Soft-delete the docs in one UPDATE. Idempotent: rows with
    #    deleted_at already set are excluded by the WHERE clause.
    from sqlalchemy import text as _text
    await session.execute(
        _text(
            "UPDATE kb_documents SET deleted_at = NOW() "
            "WHERE kb_id = :kb AND subtag_id = :sid "
            "AND deleted_at IS NULL"
        ),
        {"kb": kb_id, "sid": subtag_id},
    )

    # 4) Soft-delete the subtag itself.
    ok = await kb_service.soft_delete_subtag(
        session, kb_id=kb_id, subtag_id=subtag_id,
    )
    if not ok:
        await session.rollback()
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
    except IntegrityError as e:
        # H8: kb_access has CHECK constraint (exactly-one user XOR group)
        # plus FK/duplicate constraints. Map to 409 with sanitized text.
        await session.rollback()
        log.warning(
            "grant_access: integrity error kb=%s user=%s group=%s: %s",
            kb_id, body.user_id, body.group_id, e,
        )
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="constraint violation",
        ) from e
    except Exception:
        await session.rollback()
        log.exception("grant_access: unexpected error kb=%s", kb_id)
        raise
    # Phase 1.5 — invalidate cached allowed_kb_ids for every user this
    # grant affects (direct user grant -> 1 user; group grant -> all
    # current group members). Fail-open: a cache miss falls through to
    # the DB on the next request, and the TTL safety net catches any
    # dropped pub/sub message.
    await _invalidate_rbac_cache_for_grant(session, g)
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
    # Phase 1.5 — fetch the grant row BEFORE deletion so we know which
    # users to invalidate. If we read after the delete, the row is gone
    # and we can't reconstruct user/group_id.
    grant = (await session.execute(
        select(KBAccess).where(KBAccess.id == grant_id, KBAccess.kb_id == kb_id)
    )).scalar_one_or_none()
    ok = await kb_service.revoke_access(session, grant_id=grant_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="grant not found")
    await session.commit()
    # Invalidate cached allowed_kb_ids for users who lost access. Done
    # AFTER commit so a rolled-back delete doesn't leave stale-evicted
    # cache entries.
    if grant is not None:
        await _invalidate_rbac_cache_for_grant(session, grant)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class DocOut(BaseModel):
    id: int
    kb_id: int
    subtag_id: int
    filename: str
    mime_type: Optional[str]
    ingest_status: str
    chunk_count: int
    # H10 — surface ingest failure cause to the admin UI so operators can
    # see "OCR timeout" / "tokenizer not loaded" without grepping logs.
    # Always None for live (status != failed) docs.
    error_message: Optional[str] = None


class DocListOut(BaseModel):
    """Paginated documents list response (H2)."""
    items: list[DocOut]
    total_count: int


@router.get("/{kb_id}/documents", response_model=DocListOut)
async def list_documents(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """H2: paginated document listing.

    Returns ``{items, total_count}`` so the UI can render
    "showing 1-100 of N" and trigger fetch-next-page when the user
    scrolls past the visible list.
    """
    from sqlalchemy import func as _f
    base_where = (KBDocument.kb_id == kb_id, KBDocument.deleted_at.is_(None))
    rows = (await session.execute(
        select(KBDocument).where(*base_where)
        .order_by(KBDocument.id.desc())
        .limit(limit)
        .offset(offset)
    )).scalars().all()
    total = int((await session.execute(
        select(_f.count(KBDocument.id)).where(*base_where)
    )).scalar() or 0)
    items = [
        DocOut(
            id=d.id, kb_id=d.kb_id, subtag_id=d.subtag_id, filename=d.filename,
            mime_type=d.mime_type, ingest_status=d.ingest_status,
            chunk_count=d.chunk_count, error_message=d.error_message,
        )
        for d in rows
    ]
    return DocListOut(items=items, total_count=total)


@router.delete("/{kb_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    kb_id: int, doc_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    """Hard-delete a doc end-to-end: Qdrant points first, then DB row.

    Order of operations matters. Previously this endpoint soft-deleted the DB
    row, then best-effort attempted the Qdrant delete and swallowed any
    failure — which left Qdrant points orphaned on custom-sharded collections
    (where the bare ``delete(filter=...)`` call returns "Shard key not
    specified"). Retrieval still surfaced the orphaned chunks because the
    ``deleted=true`` payload bit is only stamped at ingest time, not on a
    soft-delete.

    New behaviour:
      1. Look up the doc row first; 404 if missing.
      2. Issue the Qdrant per-shard delete via :meth:`VectorStore.delete_by_doc`,
         which scrolls to discover ``shard_key`` then deletes per shard.
      3. If Qdrant returns ``0`` (no shards touched and no fallback delete
         succeeded) AND the doc previously had chunks, return HTTP 500 so the
         admin retries instead of seeing a green tick on a half-finished
         delete.
      4. Hard-delete the ``kb_documents`` row only after the Qdrant side is
         clean. Cascade cleans subtag links via FK ``ON DELETE CASCADE``;
         the blob (if any) is reclaimed by ``blob_gc`` on its next sweep.

    Falls open on the "doc had 0 chunks" edge case: we don't fail when the
    Qdrant delete touches 0 shards if the row's ``chunk_count`` is also 0
    (incomplete ingest, or a queued doc that never embedded).
    """
    from sqlalchemy import delete as sql_delete

    doc = (await session.execute(
        select(KBDocument).where(
            KBDocument.id == doc_id, KBDocument.kb_id == kb_id,
        )
    )).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="doc not found")

    expected_chunks = int(doc.chunk_count or 0)

    if _VS is not None:
        try:
            ops = await _VS.delete_by_doc(f"kb_{kb_id}", doc_id)
        except Exception as e:
            log.error(
                "delete_document: vector delete raised kb=%s doc=%s err=%s",
                kb_id, doc_id, e,
            )
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"qdrant delete raised: {e!s:.200}",
            ) from e
        if ops == 0 and expected_chunks > 0:
            log.error(
                "delete_document: vector delete touched 0 shards for "
                "doc=%s kb=%s but expected %d chunks — orphan risk; "
                "DB row left intact for retry",
                doc_id, kb_id, expected_chunks,
            )
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="qdrant delete failed (0 ops, expected chunks > 0); "
                       "DB row left intact for retry",
            )

    # Optional blob cleanup: only relevant when the doc was queued but
    # never finished ingesting (status != done). The blob_store delete is
    # best-effort and idempotent — a missing blob is not an error.
    blob_sha = getattr(doc, "blob_sha", None)
    if blob_sha and (doc.ingest_status or "") in ("queued", "embedding", "chunking"):
        try:
            from ext.services.blob_store import BlobStore
            import os as _os
            BlobStore(
                _os.environ.get("INGEST_BLOB_ROOT", "/var/ingest")
            ).delete(blob_sha)
        except Exception as e:
            log.warning(
                "delete_document: blob cleanup failed sha=%s doc=%s: %s",
                blob_sha, doc_id, e,
            )

    # Hard-delete the row (cascade FKs do the rest).
    r = await session.execute(
        sql_delete(KBDocument).where(
            KBDocument.id == doc_id, KBDocument.kb_id == kb_id,
        )
    )
    if r.rowcount == 0:  # racy concurrent delete; treat as already-gone
        log.info("delete_document: row already gone kb=%s doc=%s", kb_id, doc_id)
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

#: Sentinel returned by :func:`compute_drift_pct` when Postgres expects
#: zero chunks but Qdrant still has points (orphans). 999.0 was chosen
#: so it sorts at the top of any "drift > N" alert and is unmistakable
#: in dashboards — no real KB will ever drift 999% in normal operation.
ORPHAN_DRIFT_SENTINEL = 999.0


def compute_drift_pct(expected: int, observed: int) -> float:
    """Return absolute-value drift between expected and observed counts.

    ``drift_pct = |observed - expected| / expected * 100``. When ``expected``
    is 0 AND ``observed`` is also 0 the drift is 0.0 (clean empty KB). When
    ``expected`` is 0 but ``observed`` is positive the function returns
    :data:`ORPHAN_DRIFT_SENTINEL` (``999.0``) and logs a WARNING — Qdrant
    holds chunks for a KB whose Postgres truth says it's empty, which is
    the orphan-cleanup signal operators want surfaced loudly. ``expected``
    < 0 (corrupt rows) is treated as 0 expected.

    Pure function. Must be import-safe (no DB / Qdrant access). Logging
    is intentional — the warning is only emitted when the sentinel
    fires, so steady-state KBs make no log noise.
    """
    if expected <= 0:
        if int(observed) > 0:
            log.warning(
                "compute_drift_pct: orphan chunks detected — expected=%s "
                "observed=%s; returning sentinel %.0f",
                expected, observed, ORPHAN_DRIFT_SENTINEL,
            )
            return ORPHAN_DRIFT_SENTINEL
        return 0.0
    diff = abs(int(observed) - int(expected))
    return (diff / float(expected)) * 100.0


class FailedDoc(BaseModel):
    doc_id: int
    error_message: Optional[str]


class KBHealthOut(BaseModel):
    """Operational health snapshot for one KB.

    drift_pct semantics
    -------------------
    Normally ``|observed - expected| / expected * 100``. Two special
    values:

    * ``0.0`` — the KB is clean OR empty-and-clean (expected==0,
      observed==0).
    * ``999.0`` — orphan sentinel (M9). expected==0 but Qdrant still
      has points for this KB. Means either a previous KB deletion left
      Qdrant chunks behind, or an ingest ran but the Postgres row was
      never committed. Operators should investigate; the sentinel was
      chosen so it sorts at the top of any "drift > N" alert and is
      unmistakable in dashboards.
    """
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
