"""RBAC: resolve which KB ids a user is allowed to read."""
from __future__ import annotations

from typing import List

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import KBAccess, KnowledgeBase
from .obs import span


async def get_allowed_kb_ids(session: AsyncSession, *, user_id: str) -> List[int]:
    """Return list of kb_ids the given user can read.

    Admins see every non-deleted KB. Regular users see KBs matched by:
      - direct user grant in kb_access, OR
      - group grant for a group they belong to.
    """
    with span("rbac.check", user_id=str(user_id)) as _sp:
        # Use raw SQL for upstream's "user" table (singular name, UUID id)
        row = (await session.execute(
            text('SELECT role FROM "user" WHERE id = :uid'), {"uid": user_id}
        )).first()
        if row is None:
            try:
                _sp.set_attribute("decision", "no_user")
                _sp.set_attribute("kb_ids", "")
            except Exception:
                pass
            return []
        role = row[0]

        if role == "admin":
            rows = (await session.execute(
                select(KnowledgeBase.id).where(KnowledgeBase.deleted_at.is_(None))
            )).scalars().all()
            allowed = list(rows)
            try:
                _sp.set_attribute("decision", "admin_all")
                _sp.set_attribute("kb_ids", ",".join(str(x) for x in allowed))
            except Exception:
                pass
            return allowed

        # Get user's group IDs from upstream's "group_member" table
        group_rows = (await session.execute(
            text('SELECT group_id FROM group_member WHERE user_id = :uid'), {"uid": user_id}
        )).scalars().all()
        group_ids = list(group_rows)

        conditions = [KBAccess.user_id == user_id]
        if group_ids:
            conditions.append(KBAccess.group_id.in_(group_ids))

        rows = (await session.execute(
            select(KBAccess.kb_id).where(or_(*conditions))
        )).scalars().all()
        allowed = sorted(set(rows))
        try:
            _sp.set_attribute("decision", "granted" if allowed else "no_grants")
            _sp.set_attribute("kb_ids", ",".join(str(x) for x in allowed))
        except Exception:
            pass
        return allowed
