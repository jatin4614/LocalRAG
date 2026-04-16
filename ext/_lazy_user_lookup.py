"""Isolated to avoid a circular import between auth and db.models."""
from __future__ import annotations

from typing import Optional


async def lookup_role_async(sessionmaker, user_id: int) -> Optional[str]:
    from sqlalchemy import select
    from .db.models import User
    async with sessionmaker() as s:
        user = (await s.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        return user.role if user is not None else None
