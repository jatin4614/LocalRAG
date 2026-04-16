"""Edge cases for RBAC enforcement at the route layer."""
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
    # retrieval router first so /api/kb/available isn't shadowed by /api/kb/{kb_id}
    app.include_router(retr_router)
    app.include_router(admin_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'u@x', 'h', 'user'), (9, 'admin@x', 'h', 'admin')"))
        await s.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (50, 'Private', 9)"))
        await s.execute(text("INSERT INTO chats (id, user_id) VALUES (500, 1)"))
        await s.commit()


@pytest.mark.asyncio
async def test_user_without_access_403_on_chat_config(client):
    r = await client.put("/api/chats/500/kb_config",
                         headers={"X-User-Id": "1", "X-User-Role": "user"},
                         json={"config": [{"kb_id": 50, "subtag_ids": []}]})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_request_401(client):
    r = await client.get("/api/kb/available")
    assert r.status_code == 401

    r = await client.get("/api/kb", headers={"X-User-Role": "admin"})  # missing X-User-Id
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_pending_role_rejected_from_admin_paths(client):
    r = await client.post("/api/kb",
                          headers={"X-User-Id": "1", "X-User-Role": "pending"},
                          json={"name": "X"})
    assert r.status_code == 403
