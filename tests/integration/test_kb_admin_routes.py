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
    """Deleting an already-deleted KB returns 410 Gone (M10), not 5xx."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "DoubleDel"})
    kb_id = r.json()["id"]
    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 204
    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 410


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
async def test_delete_kb_never_existed_returns_404(client):
    """M10: deleting a never-existed KB → 404."""
    r = await client.delete("/api/kb/99999", headers=ADMIN)
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_kb_already_deleted_returns_410(client):
    """M10: second delete on the same KB → 410 Gone (not 404)."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Twice-Killed"})
    kb_id = r.json()["id"]
    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 204, r.text
    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 410, r.text
    assert "already deleted" in r.json()["detail"].lower()


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


# ---------------------------------------------------------------------------
# M6 — subtag rename + move-docs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subtag_rename_endpoint(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "M6-Rename-KB"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Old-Name"})
    sub_id = r.json()["id"]
    r = await client.patch(
        f"/api/kb/{kb_id}/subtags/{sub_id}",
        headers=ADMIN, json={"name": "New-Name"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "New-Name"
    assert body["id"] == sub_id

    # Verify via list
    r = await client.get(f"/api/kb/{kb_id}/subtags", headers=ADMIN)
    assert any(s["name"] == "New-Name" for s in r.json())


@pytest.mark.asyncio
async def test_subtag_rename_404_when_wrong_kb(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "KB-A"})
    kb_a = r.json()["id"]
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "KB-B"})
    kb_b = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_a}/subtags", headers=ADMIN, json={"name": "Sub"})
    sub_a = r.json()["id"]

    # Try renaming via KB-B's URL
    r = await client.patch(
        f"/api/kb/{kb_b}/subtags/{sub_a}",
        headers=ADMIN, json={"name": "Hijack"},
    )
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


@pytest.mark.asyncio
async def test_subtag_rename_409_on_name_collision(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "M6-Collide-KB"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Alpha"})
    a_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Beta"})
    b_id = r.json()["id"]

    # Rename Beta -> Alpha (already taken)
    r = await client.patch(
        f"/api/kb/{kb_id}/subtags/{b_id}",
        headers=ADMIN, json={"name": "Alpha"},
    )
    assert r.status_code == 409, r.text
    assert "already in use" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_subtag_rename_rejects_empty_name(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "M6-Empty-KB"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Sub"})
    sub_id = r.json()["id"]

    r = await client.patch(
        f"/api/kb/{kb_id}/subtags/{sub_id}",
        headers=ADMIN, json={"name": ""},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_subtag_move_docs_endpoint(client, engine):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "M6-Move-KB"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Source"})
    src_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Target"})
    tgt_id = r.json()["id"]

    # Seed 3 docs into source.
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    doc_ids: list[int] = []
    async with SessionLocal() as s:
        for i in range(3):
            r = await s.execute(
                text(
                    "INSERT INTO kb_documents (kb_id, subtag_id, filename, mime_type, "
                    "ingest_status, uploaded_by, chunk_count) "
                    "VALUES (:kb, :sub, :fn, 'text/plain', 'done', '1', 5) "
                    "RETURNING id"
                ),
                {"kb": kb_id, "sub": src_id, "fn": f"f{i}.txt"},
            )
            doc_ids.append(r.scalar_one())
        await s.commit()

    r = await client.post(
        f"/api/kb/{kb_id}/subtags/{src_id}/move-docs",
        headers=ADMIN,
        json={"doc_ids": doc_ids, "target_subtag_id": tgt_id},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"moved": 3}

    # Verify all 3 now point at target.
    async with SessionLocal() as s:
        rows = (await s.execute(
            text("SELECT subtag_id FROM kb_documents WHERE id = ANY(:ids)"),
            {"ids": doc_ids},
        )).scalars().all()
        assert all(r == tgt_id for r in rows)


@pytest.mark.asyncio
async def test_subtag_move_docs_404_target_not_in_kb(client, engine):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "M6-MoveX-KB"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Src"})
    src = r.json()["id"]
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "M6-MoveX-OtherKB"})
    other_kb = r.json()["id"]
    r = await client.post(f"/api/kb/{other_kb}/subtags", headers=ADMIN, json={"name": "Target"})
    other_sub = r.json()["id"]

    r = await client.post(
        f"/api/kb/{kb_id}/subtags/{src}/move-docs",
        headers=ADMIN,
        json={"doc_ids": [], "target_subtag_id": other_sub},
    )
    assert r.status_code == 404


@pytest.mark.skip(reason="cross-agent dep on B's deleted_at column; see integration step")
@pytest.mark.asyncio
async def test_subtag_delete_drops_qdrant_chunks(client, engine, monkeypatch):
    """H7: deleting a subtag MUST cascade Qdrant chunk deletes for each
    live doc in that subtag, then soft-delete docs + the subtag itself.

    TODO(integration): un-skip once Agent B's migration adds
    ``kb_subtags.deleted_at`` and the integration conftest re-runs
    migrations against the testcontainer. Until then the route's
    fallback path hard-deletes the subtag, which makes the assertion on
    ``deleted_at`` meaningless.
    """
    from ext.routers import kb_admin as ka
    calls: list[tuple[str, int]] = []

    class _StubVS:
        async def delete_by_doc(self, collection: str, doc_id: int) -> int:
            calls.append((collection, int(doc_id)))
            return 1

    monkeypatch.setattr(ka, "_VS", _StubVS())

    r = await client.post("/api/kb", headers=ADMIN, json={"name": "H7-KB"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "Subtag-X"})
    sub_id = r.json()["id"]

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        for i in range(3):
            await s.execute(
                text(
                    "INSERT INTO kb_documents (kb_id, subtag_id, filename, mime_type, "
                    "ingest_status, uploaded_by, chunk_count) "
                    "VALUES (:kb, :sub, :fn, 'text/plain', 'done', '1', 7)"
                ),
                {"kb": kb_id, "sub": sub_id, "fn": f"d{i}.txt"},
            )
        await s.commit()
        doc_ids = [r[0] for r in (await s.execute(
            text("SELECT id FROM kb_documents WHERE kb_id = :kb AND subtag_id = :sub"),
            {"kb": kb_id, "sub": sub_id},
        )).all()]

    r = await client.delete(f"/api/kb/{kb_id}/subtags/{sub_id}", headers=ADMIN)
    assert r.status_code == 204, r.text

    # 1) VectorStore.delete_by_doc invoked once per live doc, on kb_{id}
    assert len(calls) == 3
    assert all(c[0] == f"kb_{kb_id}" for c in calls)
    assert sorted(c[1] for c in calls) == sorted(doc_ids)

    # 2) Docs are soft-deleted (deleted_at set, not row-removed)
    async with SessionLocal() as s:
        rows = (await s.execute(
            text(
                "SELECT id, deleted_at FROM kb_documents "
                "WHERE kb_id = :kb AND subtag_id = :sub"
            ),
            {"kb": kb_id, "sub": sub_id},
        )).all()
        assert all(r[1] is not None for r in rows)

    # 3) Subtag itself is soft-deleted (deleted_at set on kb_subtags row)
    async with SessionLocal() as s:
        row = (await s.execute(
            text("SELECT deleted_at FROM kb_subtags WHERE id = :sid"),
            {"sid": sub_id},
        )).first()
        assert row is not None
        assert row[0] is not None


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
