"""Verifies Open WebUI upstream JWT tokens (HS256 via WEBUI_SECRET_KEY)."""
from __future__ import annotations

from typing import Any, Dict

import jwt as pyjwt


class JWTError(RuntimeError):
    """Invalid/expired/malformed JWT, or missing required claim."""


ALGORITHM = "HS256"


def verify_upstream_jwt(token: str, *, secret: str) -> Dict[str, Any]:
    """Decode + verify a token signed by Open WebUI. Raises JWTError on any failure.

    Requires claim `id` to be present. `exp` is optional but enforced when present.
    """
    try:
        claims = pyjwt.decode(token, secret, algorithms=[ALGORITHM])
    except pyjwt.ExpiredSignatureError as e:
        raise JWTError("token expired") from e
    except pyjwt.InvalidTokenError as e:
        raise JWTError(f"invalid token: {e}") from e

    if "id" not in claims:
        raise JWTError("missing 'id' claim")
    return claims
