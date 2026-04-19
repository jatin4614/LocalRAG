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


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin'),(1,'alice@x','h','user'),(2,'bob@x','h','user')"))
        await s.execute(text("INSERT INTO chats (id,user_id) VALUES (100,1),(200,2)"))
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
async def test_alice_private_doc_invisible_to_bob(client):
    ALICE = {"X-User-Id": "1", "X-User-Role": "user"}
    BOB   = {"X-User-Id": "2", "X-User-Role": "user"}

    r = await client.post(
        "/api/chats/100/private_docs/upload", headers=ALICE,
        files={"file": ("secret.txt", b"alice's private account number is 42", "text/plain")},
    )
    assert r.status_code == 201

    r = await client.post("/api/rag/retrieve", headers=BOB, json={
        "chat_id": 200,
        "query": "account number",
        "selected_kb_config": [],
    })
    assert r.status_code == 200
    hits = r.json()["hits"]
    for h in hits:
        assert "alice" not in h["text"].lower(), f"leak: {h}"
        assert h.get("chat_id") != 100


@pytest.mark.asyncio
async def test_bob_cannot_query_alices_chat(client):
    BOB = {"X-User-Id": "2", "X-User-Role": "user"}
    r = await client.post("/api/rag/retrieve", headers=BOB, json={
        "chat_id": 100,
        "query": "anything",
        "selected_kb_config": [],
    })
    assert r.status_code == 404
