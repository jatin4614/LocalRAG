"""HTTP admin routes for KB CRUD. Admin-only."""
from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from sqlalchemy import select

from ..services import kb_service
from ..db.models import KBDocument
from ..services.auth import CurrentUser, require_admin
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
