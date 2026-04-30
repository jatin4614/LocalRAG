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
    payload = r2.json()
    # H2: paginated response shape
    assert "items" in payload
    assert "total_count" in payload
    assert any(kb["name"] == "Eng" for kb in payload["items"])


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


# ---------------------------------------------------------------------------
# H2 — pagination on list endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_kbs_paginated_returns_total_count(client):
    # Seed 5 KBs.
    for i in range(5):
        r = await client.post("/api/kb", headers=ADMIN, json={"name": f"Page-{i}"})
        assert r.status_code == 201, r.text

    # Default pagination (limit=100): all 5 fit.
    r = await client.get("/api/kb", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["total_count"] >= 5
    assert len(body["items"]) >= 5

    # Limit=2 returns 2 items but total_count counts all.
    r = await client.get("/api/kb?limit=2", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 2
    assert body["total_count"] >= 5

    # Offset=2 + limit=2 returns the 3rd-and-4th newest.
    r = await client.get("/api/kb?limit=2&offset=2", headers=ADMIN)
    assert r.status_code == 200
    assert len(r.json()["items"]) == 2


@pytest.mark.asyncio
async def test_list_kbs_pagination_bounds_enforced(client):
    # limit=0 violates ge=1
    r = await client.get("/api/kb?limit=0", headers=ADMIN)
    assert r.status_code == 422
    # limit=2000 violates le=1000
    r = await client.get("/api/kb?limit=2000", headers=ADMIN)
    assert r.status_code == 422
    # offset=-1 violates ge=0
    r = await client.get("/api/kb?offset=-1", headers=ADMIN)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_doc_out_includes_error_message_for_failed_docs(client, engine):
    """H10: failed ingest exposes error_message in DocOut so the admin UI
    can render the cause inline (instead of forcing operators to grep
    celery logs).
    """
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Err-KB"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "S"})
    sub_id = r.json()["id"]

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(
            text(
                "INSERT INTO kb_documents (kb_id, subtag_id, filename, mime_type, "
                "ingest_status, error_message, uploaded_by, chunk_count) "
                "VALUES (:kb, :sub, 'broken.pdf', 'application/pdf', 'failed', "
                "'OCR timeout after 90s', '1', 0)"
            ),
            {"kb": kb_id, "sub": sub_id},
        )
        await s.execute(
            text(
                "INSERT INTO kb_documents (kb_id, subtag_id, filename, mime_type, "
                "ingest_status, uploaded_by, chunk_count) "
                "VALUES (:kb, :sub, 'fine.pdf', 'application/pdf', 'done', '1', 12)"
            ),
            {"kb": kb_id, "sub": sub_id},
        )
        await s.commit()

    r = await client.get(f"/api/kb/{kb_id}/documents", headers=ADMIN)
    items = r.json()["items"]
    failed = [d for d in items if d["filename"] == "broken.pdf"][0]
    assert failed["ingest_status"] == "failed"
    assert failed["error_message"] == "OCR timeout after 90s"
    fine = [d for d in items if d["filename"] == "fine.pdf"][0]
    # Live docs report error_message=None — schema-required.
    assert fine["error_message"] is None


@pytest.mark.asyncio
async def test_list_documents_paginated(client, engine):
    """H2: documents listing supports limit/offset and returns total_count."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Doc-Page-KB"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "S"})
    sub_id = r.json()["id"]

    # Seed 6 documents directly so we don't depend on the upload pipeline.
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        for i in range(6):
            await s.execute(
                text(
                    "INSERT INTO kb_documents (kb_id, subtag_id, filename, mime_type, "
                    "ingest_status, uploaded_by, chunk_count) "
                    "VALUES (:kb, :sub, :fn, 'text/plain', 'done', '1', 5)"
                ),
                {"kb": kb_id, "sub": sub_id, "fn": f"f{i}.txt"},
            )
        await s.commit()

    r = await client.get(f"/api/kb/{kb_id}/documents", headers=ADMIN)
    body = r.json()
    assert body["total_count"] == 6
    assert len(body["items"]) == 6

    r = await client.get(
        f"/api/kb/{kb_id}/documents?limit=3&offset=0", headers=ADMIN,
    )
    body = r.json()
    assert body["total_count"] == 6
    assert len(body["items"]) == 3
