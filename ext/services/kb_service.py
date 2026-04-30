"""Knowledge Base service layer — CRUD operations with RBAC-agnostic queries.

Callers must filter by `get_allowed_kb_ids(user_id)` BEFORE calling these methods.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional

from sqlalchemy import delete, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import KnowledgeBase, KBSubtag, KBAccess


async def create_kb(
    session: AsyncSession, *, name: str, description: Optional[str], admin_id: int
) -> KnowledgeBase:
    """Create a knowledge base.

    Pre-checks for an existing live KB with the same name and raises
    :class:`ValueError` so the router can map it to a 409 with a clean,
    sanitized message ("kb name already in use: ..."). This avoids
    relying solely on the DB unique-constraint violation, which surfaces
    as a generic IntegrityError that's harder to render to admins.

    The pre-check + flush together is racy under concurrent inserts —
    the DB unique index is the ultimate truth, and the router catches
    :class:`sqlalchemy.exc.IntegrityError` and maps it to 409 too.
    """
    existing = (await session.execute(
        select(KnowledgeBase.id).where(
            KnowledgeBase.name == name,
            KnowledgeBase.deleted_at.is_(None),
        )
    )).scalar_one_or_none()
    if existing is not None:
        raise ValueError(f"kb name already in use: {name}")
    kb = KnowledgeBase(name=name, description=description, admin_id=admin_id)
    session.add(kb)
    await session.flush()
    return kb


async def get_kb(session: AsyncSession, *, kb_id: int) -> Optional[KnowledgeBase]:
    return (await session.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.deleted_at.is_(None),
        )
    )).scalar_one_or_none()


async def list_kbs(
    session: AsyncSession, *, kb_ids: Optional[Iterable[int]] = None
) -> List[KnowledgeBase]:
    stmt = select(KnowledgeBase).where(KnowledgeBase.deleted_at.is_(None))
    if kb_ids is not None:
        ids = list(kb_ids)
        if not ids:
            return []
        stmt = stmt.where(KnowledgeBase.id.in_(ids))
    return list((await session.execute(stmt)).scalars().all())


async def update_kb(
    session: AsyncSession, *, kb_id: int, name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[KnowledgeBase]:
    kb = await get_kb(session, kb_id=kb_id)
    if kb is None:
        return None
    if name is not None:
        kb.name = name
    if description is not None:
        kb.description = description
    await session.flush()
    return kb


async def soft_delete_kb(session: AsyncSession, *, kb_id: int) -> bool:
    r: CursorResult = await session.execute(  # type: ignore[assignment]
        update(KnowledgeBase)
        .where(KnowledgeBase.id == kb_id, KnowledgeBase.deleted_at.is_(None))
        .values(deleted_at=datetime.now(timezone.utc))
    )
    return r.rowcount > 0


async def create_subtag(
    session: AsyncSession, *, kb_id: int, name: str, description: Optional[str] = None,
) -> KBSubtag:
    sub = KBSubtag(kb_id=kb_id, name=name, description=description)
    session.add(sub)
    await session.flush()
    return sub


async def list_subtags(session: AsyncSession, *, kb_id: int) -> List[KBSubtag]:
    return list((await session.execute(
        select(KBSubtag).where(KBSubtag.kb_id == kb_id).order_by(KBSubtag.id)
    )).scalars().all())


async def delete_subtag(session: AsyncSession, *, kb_id: int, subtag_id: int) -> bool:
    r: CursorResult = await session.execute(  # type: ignore[assignment]
        delete(KBSubtag).where(KBSubtag.id == subtag_id, KBSubtag.kb_id == kb_id)
    )
    return r.rowcount > 0


async def grant_access(
    session: AsyncSession, *, kb_id: int, user_id: Optional[str],
    group_id: Optional[str], access_type: str = "read",
) -> KBAccess:
    """Grant read/write on a KB to either a user or a group (never both, never neither).

    Note: ``user_id`` is a string (upstream Open WebUI uses UUID-string user
    ids, mirrored to ``kb_access.user_id`` VARCHAR(255)); ``group_id`` is also
    a string (TEXT column). The previous ``Optional[int]`` typing was a
    historical mistake from before the upstream-UUID migration.
    """
    if (user_id is None) == (group_id is None):
        raise ValueError("grant_access requires exactly one of user_id or group_id")
    grant = KBAccess(kb_id=kb_id, user_id=user_id, group_id=group_id, access_type=access_type)
    session.add(grant)
    await session.flush()
    return grant


async def list_access(session: AsyncSession, *, kb_id: int) -> List[KBAccess]:
    return list((await session.execute(
        select(KBAccess).where(KBAccess.kb_id == kb_id).order_by(KBAccess.id)
    )).scalars().all())


async def revoke_access(session: AsyncSession, *, grant_id: int) -> bool:
    r: CursorResult = await session.execute(  # type: ignore[assignment]
        delete(KBAccess).where(KBAccess.id == grant_id)
    )
    return r.rowcount > 0
