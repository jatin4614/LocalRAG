"""P3.0 — PATCH /api/kb/{kb_id}/config admin endpoint.

Unit-level shape tests using SQLite in-memory. The integration-level
variants (real Postgres JSONB round-trip) live in
``tests/integration/test_kb_admin_routes.py`` — this file only checks:

* Admin-only (non-admin 403, missing-auth 401)
* Unknown keys rejected with 400
* Bad value types rejected with 400
* Valid partial update merges into the existing config
* GET returns ``{}`` when unset, returns the stamped config when set
* PATCHing a non-existent KB returns 404
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, create_async_engine

from ext.db.base import Base
# Importing compat models registers users/groups/chats on the shared
# Base.metadata so create_all produces the tables the router expects.
from ext.db.models.compat import User as _User, Group as _Group  # noqa: F401
from ext.routers.kb_admin import router as kb_admin_router, set_sessionmaker


ADMIN = {"X-User-Id": "1", "X-User-Role": "admin"}
USER = {"X-User-Id": "2", "X-User-Role": "user"}


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("INSERT INTO users (id, email, password_hash, role) "
                                "VALUES (1, 'admin@x', 'h', 'admin'), "
                                "(2, 'user@x', 'h', 'user')"))
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    set_sessionmaker(SessionLocal)

    app = FastAPI()
    app.include_router(kb_admin_router)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    await engine.dispose()


_COUNTER = {"n": 0}


async def _mk_kb(client: AsyncClient, engine=None, name: str = "Docs") -> int:
    """Insert a KB directly via SQL (SQLite doesn't auto-increment
    BigInteger PKs, so POST /api/kb would 500). Unique-name bump per
    call to sidestep the name UNIQUE index across tests that reuse
    the default."""
    _COUNTER["n"] += 1
    unique_name = f"{name}-{_COUNTER['n']}"
    kb_id = 1000 + _COUNTER["n"]
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    # The fixture also called set_sessionmaker; reuse the same sessionmaker
    # by poking at the module's global. Cleanest interface we have.
    from ext.routers import kb_admin
    SessionLocal = kb_admin._SESSIONMAKER
    assert SessionLocal is not None
    async with SessionLocal() as s:
        await s.execute(text(
            "INSERT INTO knowledge_bases (id, name, admin_id, rag_config) "
            "VALUES (:id, :name, :admin, :cfg)"
        ), {"id": kb_id, "name": unique_name, "admin": "1", "cfg": "{}"})
        await s.commit()
    return kb_id


@pytest.mark.asyncio
async def test_get_config_returns_empty_by_default(client):
    kb_id = await _mk_kb(client)
    r = await client.get(f"/api/kb/{kb_id}/config", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["kb_id"] == kb_id
    assert body["rag_config"] == {}


@pytest.mark.asyncio
async def test_patch_valid_body_merges(client):
    kb_id = await _mk_kb(client)
    r = await client.patch(
        f"/api/kb/{kb_id}/config",
        headers=ADMIN,
        json={"rerank": True, "context_expand_window": 2},
    )
    assert r.status_code == 200
    assert r.json()["rag_config"]["rerank"] is True
    assert r.json()["rag_config"]["context_expand_window"] == 2


@pytest.mark.asyncio
async def test_patch_is_partial_not_replace(client):
    kb_id = await _mk_kb(client)
    # First set two keys.
    await client.patch(
        f"/api/kb/{kb_id}/config",
        headers=ADMIN,
        json={"rerank": True, "mmr": True},
    )
    # Now patch only one — the other should survive.
    r = await client.patch(
        f"/api/kb/{kb_id}/config",
        headers=ADMIN,
        json={"rerank": False},
    )
    assert r.status_code == 200
    cfg = r.json()["rag_config"]
    assert cfg["rerank"] is False
    assert cfg["mmr"] is True  # preserved


@pytest.mark.asyncio
async def test_patch_rejects_unknown_key(client):
    kb_id = await _mk_kb(client)
    r = await client.patch(
        f"/api/kb/{kb_id}/config",
        headers=ADMIN,
        json={"rerank": True, "malicious_key": "drop_tables"},
    )
    assert r.status_code == 400
    assert "malicious_key" in r.json()["detail"]


@pytest.mark.asyncio
async def test_patch_rejects_bad_value_type(client):
    kb_id = await _mk_kb(client)
    r = await client.patch(
        f"/api/kb/{kb_id}/config",
        headers=ADMIN,
        json={"context_expand_window": "not-a-number"},
    )
    assert r.status_code == 400
    assert "context_expand_window" in r.json()["detail"]


@pytest.mark.asyncio
async def test_patch_accepts_string_boolean_coerced(client):
    """``"true"`` is a legitimate JSON value some admin UIs send — we
    accept it and coerce to bool via validate_config."""
    kb_id = await _mk_kb(client)
    r = await client.patch(
        f"/api/kb/{kb_id}/config",
        headers=ADMIN,
        json={"rerank": "true"},
    )
    assert r.status_code == 200
    assert r.json()["rag_config"]["rerank"] is True


@pytest.mark.asyncio
async def test_patch_non_admin_returns_403(client):
    kb_id = await _mk_kb(client)
    r = await client.patch(
        f"/api/kb/{kb_id}/config",
        headers=USER,
        json={"rerank": True},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_patch_no_auth_returns_401(client):
    kb_id = await _mk_kb(client)
    r = await client.patch(
        f"/api/kb/{kb_id}/config",
        json={"rerank": True},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_patch_unknown_kb_returns_404(client):
    r = await client.patch(
        "/api/kb/99999/config",
        headers=ADMIN,
        json={"rerank": True},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_config_returns_stamped_value(client):
    kb_id = await _mk_kb(client)
    await client.patch(
        f"/api/kb/{kb_id}/config",
        headers=ADMIN,
        json={"rerank": True, "mmr_lambda": 0.7},
    )
    r = await client.get(f"/api/kb/{kb_id}/config", headers=ADMIN)
    assert r.status_code == 200
    cfg = r.json()["rag_config"]
    assert cfg["rerank"] is True
    assert cfg["mmr_lambda"] == 0.7


@pytest.mark.asyncio
async def test_patch_all_valid_keys_round_trip(client):
    kb_id = await _mk_kb(client)
    body = {
        "rerank": True,
        "rerank_top_k": 30,
        "mmr": True,
        "mmr_lambda": 0.7,
        "context_expand": True,
        "context_expand_window": 2,
        "spotlight": True,
        "semcache": True,
        "contextualize_on_ingest": True,
    }
    r = await client.patch(f"/api/kb/{kb_id}/config", headers=ADMIN, json=body)
    assert r.status_code == 200
    cfg = r.json()["rag_config"]
    for k, v in body.items():
        assert cfg[k] == v, f"{k}: expected {v}, got {cfg.get(k)}"
