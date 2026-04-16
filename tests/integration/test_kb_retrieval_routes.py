import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.kb_retrieval import router as kb_retr_router, set_sessionmaker


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    set_sessionmaker(SessionLocal)
    app = FastAPI()
    app.include_router(kb_retr_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


USER_1 = {"X-User-Id": "1", "X-User-Role": "user"}
USER_2 = {"X-User-Id": "2", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'user'), (2, 'b@x', 'h', 'user'), (9, 'admin@x', 'h', 'admin')"))
        await s.execute(text("INSERT INTO groups (id, name) VALUES (1, 'eng'), (2, 'hr')"))
        await s.execute(text("INSERT INTO user_groups (user_id, group_id) VALUES (1, 1), (2, 2)"))
        await s.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (10, 'Eng', 9), (11, 'HR', 9), (12, 'Secret', 9)"))
        await s.execute(text("INSERT INTO kb_access (kb_id, group_id, access_type) VALUES (10, 1, 'read'), (11, 2, 'read')"))
        await s.execute(text("INSERT INTO chats (id, user_id) VALUES (100, 1), (200, 2)"))
        await s.commit()


@pytest.mark.asyncio
async def test_available_returns_only_allowed_kbs(client):
    r = await client.get("/api/kb/available", headers=USER_1)
    assert r.status_code == 200
    names = {kb["name"] for kb in r.json()}
    assert names == {"Eng"}

    r = await client.get("/api/kb/available", headers=USER_2)
    assert {kb["name"] for kb in r.json()} == {"HR"}


@pytest.mark.asyncio
async def test_chat_kb_config_set_and_get(client):
    r = await client.put("/api/chats/100/kb_config", headers=USER_1,
                         json={"config": [{"kb_id": 10, "subtag_ids": []}]})
    assert r.status_code == 200, r.text
    assert r.json()["config"] == [{"kb_id": 10, "subtag_ids": []}]

    r = await client.get("/api/chats/100/kb_config", headers=USER_1)
    assert r.status_code == 200
    assert r.json()["config"] == [{"kb_id": 10, "subtag_ids": []}]


@pytest.mark.asyncio
async def test_chat_kb_config_rejects_unauthorized_kb(client):
    r = await client.put("/api/chats/100/kb_config", headers=USER_1,
                         json={"config": [{"kb_id": 11, "subtag_ids": []}]})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_chat_kb_config_rejects_other_users_chat(client):
    r = await client.put("/api/chats/200/kb_config", headers=USER_1,
                         json={"config": [{"kb_id": 10}]})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_chat_kb_config_rejects_bad_shape(client):
    r = await client.put("/api/chats/100/kb_config", headers=USER_1,
                         json={"config": [{"kb_id": "not-int"}]})
    assert r.status_code == 400
