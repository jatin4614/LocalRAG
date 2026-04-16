import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.rag import router as rag_router, configure as configure_rag
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder
from ext.services.ingest import ingest_bytes


ALICE = {"X-User-Id": "1", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin'),(1,'u@x','h','user')"))
        await s.execute(text("INSERT INTO groups (id,name) VALUES (1,'eng')"))
        await s.execute(text("INSERT INTO user_groups (user_id, group_id) VALUES (1,1)"))
        await s.execute(text("INSERT INTO knowledge_bases (id,name,admin_id) VALUES (10,'Eng',9),(11,'Secret',9)"))
        await s.execute(text("INSERT INTO kb_access (kb_id, group_id, access_type) VALUES (10,1,'read')"))
        await s.execute(text("INSERT INTO chats (id,user_id) VALUES (500,1)"))
        await s.commit()


@pytest_asyncio.fixture
async def client(engine, clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    emb = StubEmbedder(dim=32)
    await vs.ensure_collection("kb_10")
    await ingest_bytes(
        data=b"the quick brown fox jumps over the lazy dog in the forest",
        mime_type="text/plain", filename="a.txt", collection="kb_10",
        payload_base={"kb_id": 10, "subtag_id": 1, "doc_id": 1},
        vector_store=vs, embedder=emb, chunk_tokens=20, overlap_tokens=5,
    )

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    configure_rag(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    app = FastAPI()
    app.include_router(rag_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await vs.close()


@pytest.mark.asyncio
async def test_retrieve_returns_hits_for_authorized_kb(client):
    r = await client.post("/api/rag/retrieve", headers=ALICE, json={
        "chat_id": 500,
        "query": "quick brown fox",
        "selected_kb_config": [{"kb_id": 10, "subtag_ids": []}],
    })
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert len(hits) >= 1
    assert hits[0]["kb_id"] == 10


@pytest.mark.asyncio
async def test_retrieve_rejects_unauthorized_kb(client):
    r = await client.post("/api/rag/retrieve", headers=ALICE, json={
        "chat_id": 500,
        "query": "x",
        "selected_kb_config": [{"kb_id": 11, "subtag_ids": []}],
    })
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_retrieve_rejects_other_users_chat(client):
    r = await client.post("/api/rag/retrieve",
                          headers={"X-User-Id": "2", "X-User-Role": "user"},
                          json={"chat_id": 500, "query": "x", "selected_kb_config": []})
    assert r.status_code == 404
