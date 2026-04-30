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


@pytest.mark.asyncio
async def test_kb_create_duplicate_name_returns_409(client):
    """L7: kb_service pre-check raises ValueError -> router maps to 409."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Unique-1"})
    assert r.status_code == 201, r.text
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Unique-1"})
    assert r.status_code == 409
    assert "already in use" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_kb_create_integrity_error_returns_409(client, engine, monkeypatch):
    """H8: when an IntegrityError reaches the router (e.g. race past
    pre-check), it MUST surface as 409 with a sanitized message —
    not a 500 with raw IntegrityError text exposing constraint names.
    """
    from ext.services import kb_service as svc

    # Bypass the pre-check by monkeypatching it to a no-op for this call.
    # The DB unique constraint will then fire.
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(
            text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (501, 'Race-Me', '1')")
        )
        await s.commit()

    original = svc.create_kb

    async def _bypass_precheck(session, *, name, description, admin_id):
        # Skip the pre-check, go straight to insert + flush.
        from ext.db.models import KnowledgeBase
        kb = KnowledgeBase(name=name, description=description, admin_id=admin_id)
        session.add(kb)
        await session.flush()
        return kb

    monkeypatch.setattr(svc, "create_kb", _bypass_precheck)
    try:
        r = await client.post("/api/kb", headers=ADMIN, json={"name": "Race-Me"})
        assert r.status_code == 409, r.text
        # Sanitized — no raw constraint name leaked
        assert "name already in use" in r.json()["detail"].lower()
    finally:
        monkeypatch.setattr(svc, "create_kb", original)


@pytest.mark.asyncio
async def test_subtag_create_duplicate_name_returns_409(client):
    """H8: duplicate subtag name within same KB → 409 sanitized."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "S-KB"})
    kb_id = r.json()["id"]
    r1 = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Dup"})
    assert r1.status_code == 201, r1.text
    r2 = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Dup"})
    assert r2.status_code == 409, r2.text
    assert "already in use" in r2.json()["detail"].lower()
