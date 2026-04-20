import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.upload import router as upload_router, configure as configure_upload
from ext.routers.rag import router as rag_router, configure as configure_rag
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder

pytestmark = pytest.mark.integration


ADMIN = {"X-User-Id": "9", "X-User-Role": "admin"}
ALICE = {"X-User-Id": "1", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin'),(1,'u@x','h','user')"))
        await s.execute(text("INSERT INTO groups (id,name) VALUES (1,'eng')"))
        await s.execute(text("INSERT INTO user_groups (user_id, group_id) VALUES (1,1)"))
        await s.execute(text("INSERT INTO knowledge_bases (id,name,admin_id) VALUES (10,'Eng',9)"))
        await s.execute(text("INSERT INTO kb_subtags (id,kb_id,name) VALUES (100,10,'Docs')"))
        await s.execute(text("INSERT INTO kb_access (kb_id, group_id, access_type) VALUES (10,1,'read')"))
        await s.execute(text("INSERT INTO chats (id,user_id) VALUES (500,1)"))
        await s.commit()


@pytest_asyncio.fixture
async def client(engine, clean_qdrant):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    emb = StubEmbedder(dim=32)
    configure_upload(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    configure_rag(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    app = FastAPI()
    app.include_router(upload_router)
    app.include_router(rag_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await vs.close()


@pytest.mark.asyncio
async def test_upload_and_retrieve_roundtrip(client):
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("doc.txt", b"the quick brown fox jumps over the lazy dog", "text/plain")},
    )
    assert r.status_code == 201
    n = r.json()["chunks"]
    assert n >= 1

    r = await client.post("/api/rag/retrieve", headers=ALICE, json={
        "chat_id": 500,
        "query": "quick brown fox",
        "selected_kb_config": [{"kb_id": 10, "subtag_ids": [100]}],
    })
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) >= 1
    assert hits[0]["kb_id"] == 10
    assert hits[0]["subtag_id"] == 100
    assert "fox" in hits[0]["text"].lower()
