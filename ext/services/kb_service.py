"""Knowledge Base service layer — CRUD operations with RBAC-agnostic queries.

Callers must filter by `get_allowed_kb_ids(user_id)` BEFORE calling these methods.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import KnowledgeBase


async def create_kb(
    session: AsyncSession, *, name: str, description: Optional[str], admin_id: int
) -> KnowledgeBase:
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
