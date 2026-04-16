"""RBAC: resolve which KB ids a user is allowed to read."""
from __future__ import annotations

from typing import List

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import KBAccess, KnowledgeBase, User, UserGroup


async def get_allowed_kb_ids(session: AsyncSession, *, user_id: int) -> List[int]:
    """Return list of kb_ids the given user can read.

    Admins see every non-deleted KB. Regular users see KBs matched by:
      - direct user grant in kb_access, OR
      - group grant for a group they belong to.
    """
    user = (await session.execute(
        select(User).where(User.id == user_id)
    )).scalar_one_or_none()
    if user is None:
        return []

    if user.role == "admin":
        rows = (await session.execute(
            select(KnowledgeBase.id).where(KnowledgeBase.deleted_at.is_(None))
        )).scalars().all()
        return list(rows)

    group_ids_stmt = select(UserGroup.group_id).where(UserGroup.user_id == user_id)
    rows = (await session.execute(
        select(KBAccess.kb_id).where(
            or_(
                KBAccess.user_id == user_id,
                KBAccess.group_id.in_(group_ids_stmt),
            )
        )
    )).scalars().all()
    return sorted(set(rows))
