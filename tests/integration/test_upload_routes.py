import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.upload import router as upload_router, configure as configure_upload
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder


ADMIN = {"X-User-Id": "9", "X-User-Role": "admin"}
ALICE = {"X-User-Id": "1", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin'),(1,'u@x','h','user')"))
        await s.execute(text("INSERT INTO knowledge_bases (id,name,admin_id) VALUES (10,'KB',9)"))
        await s.execute(text("INSERT INTO kb_subtags (id,kb_id,name) VALUES (100,10,'Docs')"))
        await s.execute(text("INSERT INTO chats (id,user_id) VALUES (500,1)"))
        await s.commit()


@pytest_asyncio.fixture
async def client(engine, clean_qdrant):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    configure_upload(sessionmaker=SessionLocal, vector_store=vs, embedder=StubEmbedder(dim=32))
    app = FastAPI()
    app.include_router(upload_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await vs.close()


@pytest.mark.asyncio
async def test_kb_upload_admin(client):
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("a.txt", b"hello world this is a test", "text/plain")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "done"
    assert body["chunks"] >= 1


@pytest.mark.asyncio
async def test_kb_upload_non_admin_forbidden(client):
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ALICE,
        files={"file": ("a.txt", b"hi", "text/plain")},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_private_upload_chat_owner(client):
    r = await client.post(
        "/api/chats/500/private_docs/upload", headers=ALICE,
        files={"file": ("q.txt", b"my private note with enough text", "text/plain")},
    )
    assert r.status_code == 201, r.text
    assert r.json()["chunks"] >= 1


@pytest.mark.asyncio
async def test_private_upload_other_users_chat_404(client):
    r = await client.post(
        "/api/chats/500/private_docs/upload", headers={"X-User-Id": "2", "X-User-Role": "user"},
        files={"file": ("q.txt", b"sneaky", "text/plain")},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_kb_upload_too_large_rejected(client, monkeypatch):
    import ext.routers.upload as up
    monkeypatch.setattr(up, "MAX_UPLOAD_BYTES", 10)
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("big.txt", b"x" * 100, "text/plain")},
    )
    assert r.status_code == 413
