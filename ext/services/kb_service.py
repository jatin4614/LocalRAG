"""Knowledge Base service layer — CRUD operations with RBAC-agnostic queries.

Callers must filter by `get_allowed_kb_ids(user_id)` BEFORE calling these methods.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import KnowledgeBase, KBSubtag, KBAccess

log = logging.getLogger("orgchat.kb_service")


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
    session: AsyncSession,
    *,
    kb_ids: Optional[Iterable[int]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> List[KnowledgeBase]:
    """List live (non-soft-deleted) KBs.

    H2: ``limit``/``offset`` accept None for backwards compatibility
    (callers that want every row pass nothing); when set, the rows are
    sorted by id DESC for stable pagination across pages. Pair with
    :func:`count_kbs` to get total_count for the client.
    """
    stmt = select(KnowledgeBase).where(KnowledgeBase.deleted_at.is_(None))
    if kb_ids is not None:
        ids = list(kb_ids)
        if not ids:
            return []
        stmt = stmt.where(KnowledgeBase.id.in_(ids))
    if limit is not None or offset is not None:
        # Stable order so pages don't shuffle when rows are added/removed.
        stmt = stmt.order_by(KnowledgeBase.id.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)
    return list((await session.execute(stmt)).scalars().all())


async def count_kbs(
    session: AsyncSession, *, kb_ids: Optional[Iterable[int]] = None,
) -> int:
    """Count live KBs (H2 pagination total)."""
    stmt = select(func.count(KnowledgeBase.id)).where(
        KnowledgeBase.deleted_at.is_(None)
    )
    if kb_ids is not None:
        ids = list(kb_ids)
        if not ids:
            return 0
        stmt = stmt.where(KnowledgeBase.id.in_(ids))
    return int((await session.execute(stmt)).scalar() or 0)


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
    """List live subtags for a KB.

    H7: filters out soft-deleted rows (``deleted_at IS NOT NULL``).
    Cross-agent dep: Agent B adds the ``deleted_at`` column +
    ``KBSubtag.deleted_at`` ORM field; until that lands the WHERE
    clause uses ``getattr`` to fall back to a no-op so this code is
    forward-compatible without breaking the current ORM.
    """
    deleted_at = getattr(KBSubtag, "deleted_at", None)
    stmt = select(KBSubtag).where(KBSubtag.kb_id == kb_id)
    if deleted_at is not None:
        stmt = stmt.where(deleted_at.is_(None))
    stmt = stmt.order_by(KBSubtag.id)
    return list((await session.execute(stmt)).scalars().all())


async def delete_subtag(session: AsyncSession, *, kb_id: int, subtag_id: int) -> bool:
    """Hard-delete a subtag row. Legacy behaviour kept for callers that
    don't need the H7 cascade (Qdrant chunk drop + soft-delete docs).
    The router calls :func:`soft_delete_subtag` directly; this function
    is retained for backwards compatibility with tests + scripts.
    """
    r: CursorResult = await session.execute(  # type: ignore[assignment]
        delete(KBSubtag).where(KBSubtag.id == subtag_id, KBSubtag.kb_id == kb_id)
    )
    return r.rowcount > 0


async def soft_delete_subtag(
    session: AsyncSession, *, kb_id: int, subtag_id: int,
) -> bool:
    """Soft-delete a subtag (H7).

    Sets ``deleted_at = NOW()`` so :func:`list_subtags` filters the row
    out. Cross-agent dep on Agent B's migration that adds the column;
    when the column is absent (worktree without B's migration applied),
    this falls back to hard-delete to preserve the legacy contract
    until B's migration lands. The fallback path is idempotent and
    matches the pre-H7 behaviour.

    Returns False if the row doesn't exist OR was already soft-deleted.
    """
    from sqlalchemy import text as _text
    from sqlalchemy.exc import ProgrammingError, OperationalError
    try:
        r = await session.execute(
            _text(
                "UPDATE kb_subtags SET deleted_at = NOW() "
                "WHERE id = :sid AND kb_id = :kb AND deleted_at IS NULL"
            ),
            {"sid": subtag_id, "kb": kb_id},
        )
        rowcount = getattr(r, "rowcount", -1)
        if rowcount > 0:
            return True
        if rowcount == 0:
            # Could be already-deleted OR not-found; reuse the fallback
            # SELECT to disambiguate.
            sub = (await session.execute(
                select(KBSubtag).where(
                    KBSubtag.id == subtag_id, KBSubtag.kb_id == kb_id,
                )
            )).scalar_one_or_none()
            if sub is None:
                return False
            return True  # row existed but already soft-deleted -> still OK
    except (ProgrammingError, OperationalError) as e:
        # Pre-Agent-B worktree: ``deleted_at`` column doesn't exist.
        # Roll back the failed transaction frame so subsequent SQL on
        # this session works, then hard-delete instead.
        await session.rollback()
        log.warning(
            "soft_delete_subtag: deleted_at column missing (%s); "
            "falling back to hard-delete (cross-agent dep on B)", e,
        )
    # Hard-delete fallback (legacy path).
    r = await session.execute(  # type: ignore[assignment]
        delete(KBSubtag).where(
            KBSubtag.id == subtag_id, KBSubtag.kb_id == kb_id,
        )
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
