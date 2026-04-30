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
# KB lifecycle audit (Task 1 — fixes delete-then-recreate-same-name bug
# via migration 013 partial unique index, plus cascade cleanup of grants,
# documents, and Qdrant collections in the delete_kb route).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_then_recreate_same_name(client):
    """Bug fix: delete a KB, recreate with same name — must succeed.

    Pre-fix: bare UNIQUE(name) constraint kept the soft-deleted row's name
    locked, so create returned 409. Migration 013 swaps to a partial unique
    index that ignores soft-deleted rows, which unblocks this flow.
    """
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Recyclable"})
    assert r.status_code == 201, r.text
    kb_id_1 = r.json()["id"]

    r = await client.delete(f"/api/kb/{kb_id_1}", headers=ADMIN)
    assert r.status_code == 204

    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Recyclable"})
    assert r.status_code == 201, r.text
    kb_id_2 = r.json()["id"]
    assert kb_id_2 != kb_id_1, "expected a fresh id for the recreated KB"


@pytest.mark.asyncio
async def test_create_duplicate_live_name_still_409(client):
    """Uniqueness still enforced for live (non-deleted) KBs."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Unique"})
    assert r.status_code == 201
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Unique"})
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_delete_kb_clears_kb_access_grants(client, engine):
    """Cascade cleanup: kb_access rows for the deleted KB are hard-deleted."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "AccessKB"})
    kb_id = r.json()["id"]

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text(
            "INSERT INTO kb_access (kb_id, user_id, group_id, access_type) "
            "VALUES (:kb, '1', NULL, 'read')"
        ), {"kb": kb_id})
        await s.commit()

    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 204

    async with SessionLocal() as s:
        rows = (await s.execute(
            text("SELECT id FROM kb_access WHERE kb_id = :kb"), {"kb": kb_id}
        )).all()
        assert rows == [], "kb_access grants should be hard-deleted on KB delete"


@pytest.mark.asyncio
async def test_delete_kb_soft_deletes_documents(client, engine):
    """Cascade cleanup: kb_documents.deleted_at is set when KB is deleted."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "DocCascade"})
    kb_id = r.json()["id"]

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        sub_row = await s.execute(text(
            "INSERT INTO kb_subtags (kb_id, name) VALUES (:kb, 'sub') RETURNING id"
        ), {"kb": kb_id})
        sub_id = sub_row.scalar()
        await s.execute(text(
            "INSERT INTO kb_documents (kb_id, subtag_id, filename, ingest_status, uploaded_by) "
            "VALUES (:kb, :sid, 'a.pdf', 'done', '1')"
        ), {"kb": kb_id, "sid": sub_id})
        await s.commit()
        row = (await s.execute(
            text("SELECT deleted_at FROM kb_documents WHERE kb_id = :kb"), {"kb": kb_id}
        )).first()
        assert row is not None and row[0] is None

    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 204

    async with SessionLocal() as s:
        row = (await s.execute(
            text("SELECT deleted_at FROM kb_documents WHERE kb_id = :kb"), {"kb": kb_id}
        )).first()
        assert row is not None and row[0] is not None, (
            "kb_documents.deleted_at must be set on KB delete"
        )


@pytest.mark.asyncio
async def test_kb_soft_delete_preserves_audit_row(client, engine):
    """Soft-delete sets deleted_at; row stays for audit, hidden from list."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Auditable"})
    kb_id = r.json()["id"]

    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 204

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        row = (await s.execute(
            text("SELECT id, name, deleted_at FROM knowledge_bases WHERE id = :kb"),
            {"kb": kb_id},
        )).first()
        assert row is not None, "row must persist for audit history"
        assert row[2] is not None, "deleted_at must be populated"

    r = await client.get("/api/kb", headers=ADMIN)
    names = [kb["name"] for kb in r.json()]
    assert "Auditable" not in names


@pytest.mark.asyncio
async def test_rapid_delete_recreate_cycle(client):
    """Stress: 5 cycles of create + delete + recreate-same-name. All 201/204."""
    name = "Cyclic"
    seen_ids: set[int] = set()
    for cycle in range(5):
        r = await client.post("/api/kb", headers=ADMIN, json={"name": name})
        assert r.status_code == 201, f"cycle {cycle}: create failed: {r.text}"
        kb_id = r.json()["id"]
        assert kb_id not in seen_ids, "fresh row id expected per cycle"
        seen_ids.add(kb_id)
        r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
        assert r.status_code == 204, f"cycle {cycle}: delete failed: {r.text}"


@pytest.mark.asyncio
async def test_delete_kb_404_for_missing_id(client):
    r = await client.delete("/api/kb/99999", headers=ADMIN)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_kb_idempotent_double_delete(client):
    """Deleting an already-deleted KB returns 404, not 5xx."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "DoubleDel"})
    kb_id = r.json()["id"]
    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 204
    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_kb_name_unicode_and_special_chars(client):
    """Non-ASCII and special-char names must be accepted (no over-strict regex)."""
    for name in [
        "简体中文",
        "naïve résumé",
        "name-with-dashes",
        "name.with.dots",
        "name with spaces",
        "name/with/slashes",
        "тестовое имя",
    ]:
        r = await client.post("/api/kb", headers=ADMIN, json={"name": name})
        assert r.status_code == 201, f"{name!r}: {r.text}"


@pytest.mark.asyncio
async def test_kb_name_max_length_255(client):
    """KB name column is VARCHAR(255). Up to 255 works; >255 must reject."""
    long_ok = "a" * 255
    r = await client.post("/api/kb", headers=ADMIN, json={"name": long_ok})
    assert r.status_code == 201

    too_long = "a" * 256
    r = await client.post("/api/kb", headers=ADMIN, json={"name": too_long})
    assert r.status_code in (409, 422, 500), r.status_code


@pytest.mark.asyncio
async def test_create_then_get_returns_kb(client):
    """Happy-path read-after-write."""
    r = await client.post("/api/kb", headers=ADMIN,
                          json={"name": "Readable", "description": "d"})
    kb_id = r.json()["id"]
    r = await client.get(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["name"] == "Readable"
    assert r.json()["description"] == "d"
