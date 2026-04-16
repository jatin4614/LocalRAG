"""Cross-user isolation: user A's KB content is never visible to user B."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.kb_admin import router as admin_router, set_sessionmaker as set_admin_sm
from ext.routers.kb_retrieval import router as retr_router, set_sessionmaker as set_retr_sm


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    set_admin_sm(SessionLocal)
    set_retr_sm(SessionLocal)
    app = FastAPI()
    # retr_router must be registered first: its static path /api/kb/available
    # would otherwise be shadowed by admin_router's parametric /api/kb/{kb_id}.
    app.include_router(retr_router)
    app.include_router(admin_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


ADMIN = {"X-User-Id": "9", "X-User-Role": "admin"}
ALICE = {"X-User-Id": "1", "X-User-Role": "user"}
BOB   = {"X-User-Id": "2", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (9, 'admin@x', 'h', 'admin'), (1, 'alice@x', 'h', 'user'), (2, 'bob@x', 'h', 'user')"))
        await s.execute(text("INSERT INTO groups (id, name) VALUES (100, 'alice-group'), (200, 'bob-group')"))
        await s.execute(text("INSERT INTO user_groups (user_id, group_id) VALUES (1, 100), (2, 200)"))
        await s.commit()


@pytest.mark.asyncio
async def test_user_cannot_see_other_users_kb(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "AliceKB"})
    alice_kb_id = r.json()["id"]
    await client.post(f"/api/kb/{alice_kb_id}/access", headers=ADMIN,
                      json={"group_id": 100, "access_type": "read"})

    r = await client.post("/api/kb", headers=ADMIN, json={"name": "BobKB"})
    bob_kb_id = r.json()["id"]
    await client.post(f"/api/kb/{bob_kb_id}/access", headers=ADMIN,
                      json={"group_id": 200, "access_type": "read"})

    r = await client.get("/api/kb/available", headers=ALICE)
    names = {kb["name"] for kb in r.json()}
    assert names == {"AliceKB"}, f"leak: alice sees {names}"

    r = await client.get("/api/kb/available", headers=BOB)
    names = {kb["name"] for kb in r.json()}
    assert names == {"BobKB"}, f"leak: bob sees {names}"


@pytest.mark.asyncio
async def test_user_cannot_admin_kb(client):
    r = await client.post("/api/kb", headers=ALICE, json={"name": "Sneak"})
    assert r.status_code == 403

    r = await client.get("/api/kb", headers=ALICE)
    assert r.status_code == 403

    r = await client.delete("/api/kb/1", headers=ALICE)
    assert r.status_code == 403
