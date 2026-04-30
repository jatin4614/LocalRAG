"""Unit tests for the M8 ingest_stream JWT handling refactor.

Verifies that ``_resolve_user_from_token`` no longer mutates
``request.headers.__dict__["_list"]`` — instead it calls
``verify_upstream_jwt`` directly and constructs a CurrentUser.
"""
from __future__ import annotations

import os

import jwt as pyjwt
import pytest

from ext.services.auth import CurrentUser
from ext.routers.ingest_stream import _resolve_user_from_token, _resolve_user


@pytest.mark.asyncio
async def test_resolve_user_from_token_valid_jwt(monkeypatch):
    """Valid token + role lookup -> CurrentUser."""
    secret = "test-secret-key"
    monkeypatch.setenv("WEBUI_SECRET_KEY", secret)
    monkeypatch.setenv("AUTH_MODE", "jwt")

    async def _fake_role(uid):
        assert uid == "user-uuid-123"
        return "admin"

    # _lookup_role_by_id is imported inside _resolve_user_from_token, so
    # patch the auth module's function.
    from ext.services import auth
    monkeypatch.setattr(auth, "_lookup_role_by_id", _fake_role)

    token = pyjwt.encode({"id": "user-uuid-123"}, secret, algorithm="HS256")
    user = await _resolve_user_from_token(token)
    assert user.id == "user-uuid-123"
    assert user.role == "admin"


@pytest.mark.asyncio
async def test_resolve_user_from_token_invalid_jwt(monkeypatch):
    """Invalid token raises HTTPException 401."""
    from fastapi import HTTPException

    monkeypatch.setenv("WEBUI_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("AUTH_MODE", "jwt")

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_user_from_token("not-a-jwt")
    assert exc_info.value.status_code == 401


def test_ingest_stream_module_has_no_header_dict_mutation():
    """M8 contract: no ``request.headers.__dict__["_list"].append(...)``
    pattern anywhere in ingest_stream.py. The previous code grafted the
    query token onto a fake Authorization header by poking at
    MutableHeaders' private state — that's gone.
    """
    from pathlib import Path
    src = Path(
        __file__,
    ).resolve().parents[2] / "ext/routers/ingest_stream.py"
    body = src.read_text()
    # The exact mutation pattern; the docstring may mention "_list" so
    # we look for the offending append call signature explicitly.
    assert 'headers.__dict__["_list"].append' not in body
    assert 'headers.__dict__[\'_list\'].append' not in body
