"""Auth layer — two modes selectable via AUTH_MODE env.

- stub (default for tests / local dev): reads X-User-Id + X-User-Role headers.
- jwt  (production): verifies upstream Open WebUI JWT (token cookie or Bearer),
                     looks up role from the `users` table.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

from .jwt_verifier import JWTError, verify_upstream_jwt


@dataclass(frozen=True)
class CurrentUser:
    id: int
    role: str


VALID_ROLES = {"admin", "user", "pending"}


# ----- Stub mode -----
def _stub_user(
    x_user_id: Optional[str],
    x_user_role: Optional[str],
) -> CurrentUser:
    if x_user_id is None or x_user_role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing auth headers")
    try:
        uid = int(x_user_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Id") from e
    if x_user_role not in VALID_ROLES:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Role")
    return CurrentUser(id=uid, role=x_user_role)


# ----- JWT mode -----
_sessionmaker = None


def configure_jwt(*, sessionmaker) -> None:
    """Wire the async sessionmaker for role lookups in jwt mode."""
    global _sessionmaker
    _sessionmaker = sessionmaker


def _lookup_role_by_id(user_id: int) -> Optional[str]:
    """Synchronous wrapper over an async DB read. Monkeypatchable in unit tests."""
    if _sessionmaker is None:
        raise RuntimeError("AUTH_MODE=jwt but sessionmaker not configured; "
                           "call configure_jwt(sessionmaker=...) at app startup")
    from .._lazy_user_lookup import lookup_role_async
    return asyncio.run(lookup_role_async(_sessionmaker, user_id))


def _jwt_user(request: Request) -> CurrentUser:
    secret = os.environ.get("WEBUI_SECRET_KEY", "t0p-s3cr3t")
    token = None
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = request.cookies.get("token")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="no auth token")
    try:
        claims = verify_upstream_jwt(token, secret=secret)
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(e)) from e
    try:
        uid = int(claims["id"])
    except (TypeError, ValueError) as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad id in claims") from e
    role = _lookup_role_by_id(uid)
    if role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="unknown user")
    return CurrentUser(id=uid, role=role)


# ----- Public dependency -----
def get_current_user(
    request: Request,
    x_user_id:   Optional[str] = Header(default=None, alias="X-User-Id"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
) -> CurrentUser:
    mode = os.environ.get("AUTH_MODE", "stub").lower()
    if mode == "stub":
        return _stub_user(x_user_id, x_user_role)
    if mode == "jwt":
        return _jwt_user(request)
    raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"unknown AUTH_MODE: {mode}")


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin only")
    return user
