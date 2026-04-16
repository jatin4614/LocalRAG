"""STUB auth layer. Reads X-User-Id + X-User-Role headers.

Phase 5 replaces this with Open WebUI session cookie verification.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status


@dataclass(frozen=True)
class CurrentUser:
    id: int
    role: str


def get_current_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> CurrentUser:
    if x_user_id is None or x_user_role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing auth headers")
    try:
        uid = int(x_user_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Id") from e
    if x_user_role not in {"admin", "user", "pending"}:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Role")
    return CurrentUser(id=uid, role=x_user_role)


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin only")
    return user
