import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.kb_admin import router as kb_admin_router, set_sessionmaker

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    set_sessionmaker(SessionLocal)
    app = FastAPI()
    app.include_router(kb_admin_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


ADMIN = {"X-User-Id": "1", "X-User-Role": "admin"}
USER  = {"X-User-Id": "2", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed_admin(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'admin@x', 'h', 'admin'), (2, 'user@x', 'h', 'user')"))
        await s.commit()


@pytest.mark.asyncio
async def test_non_admin_cannot_create_kb(client):
    r = await client.post("/api/kb", headers=USER, json={"name": "X", "description": "no"})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_create_and_list_kb(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Eng", "description": "d"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "Eng"
    assert body["id"] > 0

    r2 = await client.get("/api/kb", headers=ADMIN)
    assert r2.status_code == 200
    assert any(kb["name"] == "Eng" for kb in r2.json())


@pytest.mark.asyncio
async def test_get_kb_404(client):
    r = await client.get("/api/kb/99999", headers=ADMIN)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_patch_kb(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Old"})
    kb_id = r.json()["id"]
    r = await client.patch(f"/api/kb/{kb_id}", headers=ADMIN, json={"name": "New"})
    assert r.status_code == 200
    assert r.json()["name"] == "New"


@pytest.mark.asyncio
async def test_delete_kb(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Doomed"})
    kb_id = r.json()["id"]
    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 204
    r = await client.get(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_subtag_crud(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "K"})
    kb_id = r.json()["id"]

    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "OFC"})
    assert r.status_code == 201, r.text
    sub_id = r.json()["id"]

    r = await client.get(f"/api/kb/{kb_id}/subtags", headers=ADMIN)
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = await client.delete(f"/api/kb/{kb_id}/subtags/{sub_id}", headers=ADMIN)
    assert r.status_code == 204


@pytest.mark.asyncio
async def test_access_grant_and_list(client, engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO groups (id, name) VALUES (1, 'eng')"))
        await s.commit()

    r = await client.post("/api/kb", headers=ADMIN, json={"name": "K"})
    kb_id = r.json()["id"]

    r = await client.post(f"/api/kb/{kb_id}/access", headers=ADMIN,
                          json={"group_id": "1", "access_type": "read"})
    assert r.status_code == 201, r.text
    grant_id = r.json()["id"]

    r = await client.get(f"/api/kb/{kb_id}/access", headers=ADMIN)
    assert r.status_code == 200
    assert r.json()[0]["group_id"] == "1"

    r = await client.delete(f"/api/kb/{kb_id}/access/{grant_id}", headers=ADMIN)
    assert r.status_code == 204


@pytest.mark.asyncio
async def test_access_grant_requires_exactly_one(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "K"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/access", headers=ADMIN,
                          json={"user_id": None, "group_id": None, "access_type": "read"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# H3 / H4 — KB name validation at the API edge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_kb_rejects_empty_name(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": ""})
    # pydantic ValidationError → 422 from FastAPI body validation
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_kb_rejects_whitespace_name(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "   "})
    # strip_whitespace collapses to "" -> min_length violation
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_kb_name_max_length_256_rejected_at_api(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "a" * 256})
    # Should never reach the DB; rejected at the pydantic edge.
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_kb_name_max_length_255_accepted(client):
    """Boundary: exactly 255 chars (column max) MUST be accepted."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "a" * 255})
    assert r.status_code == 201, r.text


@pytest.mark.asyncio
async def test_patch_kb_rejects_empty_name(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Has a Name"})
    kb_id = r.json()["id"]
    r = await client.patch(f"/api/kb/{kb_id}", headers=ADMIN, json={"name": ""})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_kb_rejects_overlong_name(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Short"})
    kb_id = r.json()["id"]
    r = await client.patch(f"/api/kb/{kb_id}", headers=ADMIN, json={"name": "x" * 256})
    assert r.status_code == 422
