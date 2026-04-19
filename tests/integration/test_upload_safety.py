"""Upload route safety: unsupported MIME, not-found cases."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.upload import router as upload_router, configure as configure_upload
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder

pytestmark = pytest.mark.integration


ADMIN = {"X-User-Id": "9", "X-User-Role": "admin"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin')"))
        await s.execute(text("INSERT INTO knowledge_bases (id,name,admin_id) VALUES (10,'KB',9)"))
        await s.execute(text("INSERT INTO kb_subtags (id,kb_id,name) VALUES (100,10,'Docs')"))
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


@pytest.mark.security
@pytest.mark.asyncio
async def test_unsupported_mime_rejected(client):
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("data.xls", b"binary garbage", "application/vnd.ms-excel")},
    )
    assert r.status_code == 422


@pytest.mark.security
@pytest.mark.asyncio
async def test_nonexistent_subtag_404(client):
    r = await client.post(
        "/api/kb/10/subtag/999999/upload", headers=ADMIN,
        files={"file": ("a.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 404


@pytest.mark.security
@pytest.mark.asyncio
async def test_nonexistent_kb_404(client):
    r = await client.post(
        "/api/kb/999999/subtag/1/upload", headers=ADMIN,
        files={"file": ("a.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 404


@pytest.mark.security
@pytest.mark.asyncio
async def test_octet_stream_with_unknown_extension_rejected(client):
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("file.unknown", b"raw bytes", "application/octet-stream")},
    )
    assert r.status_code == 422
