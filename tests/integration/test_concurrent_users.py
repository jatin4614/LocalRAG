"""Concurrent multi-user uploads + queries — isolation holds under load."""
import asyncio
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
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (1,'u1@x','h','user'),(2,'u2@x','h','user'),(3,'u3@x','h','user')"))
        for i, u in enumerate((1, 2, 3), start=1):
            await s.execute(text(f"INSERT INTO chats (id,user_id) VALUES ({i * 100},{u})"))
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


@pytest.mark.concurrency
@pytest.mark.asyncio
async def test_concurrent_private_uploads_stay_isolated(client):
    users_chats = [(1, 100), (2, 200), (3, 300)]

    async def _upload(uid: int, cid: int):
        h = {"X-User-Id": str(uid), "X-User-Role": "user"}
        return await client.post(
            f"/api/chats/{cid}/private_docs/upload", headers=h,
            files={"file": (f"u{uid}.txt",
                            f"user {uid} secret token is {uid*1000}".encode(),
                            "text/plain")},
        )

    results = await asyncio.gather(*[_upload(u, c) for u, c in users_chats])
    for r in results:
        assert r.status_code == 201, r.text

    for uid, cid in users_chats:
        h = {"X-User-Id": str(uid), "X-User-Role": "user"}
        r = await client.post("/api/rag/retrieve", headers=h, json={
            "chat_id": cid, "query": "secret token",
            "selected_kb_config": [],
        })
        assert r.status_code == 200
        for hit in r.json()["hits"]:
            assert f"user {uid}" in hit["text"], f"leak: user {uid} saw {hit}"
            for other_uid in (1, 2, 3):
                if other_uid != uid:
                    assert f"user {other_uid}" not in hit["text"]
