"""Phase 2 / Item 4 — admin endpoint to PATCH per-KB synonyms.

NOTE: Auth uses the stub mode (X-User-Id / X-User-Role headers) — the same
pattern as test_kb_admin_routes.py. The plan description mentions
admin_token/user_token/async_client JWT fixtures which do not exist in this
codebase; stub-header auth is the canonical integration test pattern here.
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.kb_admin import router as kb_admin_router, set_sessionmaker

pytestmark = pytest.mark.integration

ADMIN = {"X-User-Id": "1", "X-User-Role": "admin"}
USER  = {"X-User-Id": "2", "X-User-Role": "user"}


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    set_sessionmaker(SessionLocal)
    app = FastAPI()
    app.include_router(kb_admin_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture(autouse=True)
async def seed_users(engine):
    """Seed admin + regular user rows for stub auth checks."""
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text(
            "INSERT INTO users (id, email, password_hash, role) VALUES "
            "(1, 'admin@x', 'h', 'admin'), "
            "(2, 'user@x', 'h', 'user') "
            "ON CONFLICT DO NOTHING"
        ))
        await s.commit()


@pytest_asyncio.fixture
async def kb_id(client):
    """Create a KB and return its id."""
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "SynKB"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.mark.asyncio
async def test_patch_synonyms_admin_only(client, kb_id):
    """Non-admin gets 403; admin gets 200."""
    body = {"synonyms": [["X", "Y"]]}

    # Non-admin
    r = await client.patch(
        f"/api/kb/{kb_id}/synonyms", headers=USER, json=body,
    )
    assert r.status_code == 403

    # Admin
    r = await client.patch(
        f"/api/kb/{kb_id}/synonyms", headers=ADMIN, json=body,
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_patch_synonyms_persists_then_returns_via_get(client, kb_id, engine):
    """PATCH writes to DB; verify by re-reading the row via direct query."""
    synonyms = [["__test_a__", "__test_b__"]]
    r = await client.patch(
        f"/api/kb/{kb_id}/synonyms",
        headers=ADMIN,
        json={"synonyms": synonyms},
    )
    assert r.status_code == 200, r.text
    assert r.json()["synonyms"] == synonyms
    assert r.json()["kb_id"] == kb_id

    # Verify persistence by direct DB read (not just trusting the echo)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        row = (await s.execute(
            text("SELECT synonyms FROM knowledge_bases WHERE id = :i AND deleted_at IS NULL"),
            {"i": kb_id},
        )).first()
    assert row is not None, "KB row not found after PATCH"
    raw = row[0]
    if isinstance(raw, str):
        import json
        raw = json.loads(raw)
    assert raw == synonyms, f"DB synonyms mismatch: {raw!r}"

    # Cleanup
    await client.patch(
        f"/api/kb/{kb_id}/synonyms",
        headers=ADMIN,
        json={"synonyms": []},
    )


@pytest.mark.asyncio
async def test_patch_synonyms_validates_shape(client, kb_id):
    """Malformed body — synonyms is a plain string, not list-of-lists → 422 from Pydantic."""
    r = await client.patch(
        f"/api/kb/{kb_id}/synonyms",
        headers=ADMIN,
        json={"synonyms": "not a list"},
    )
    # Pydantic rejects non-list at the schema level → 422 Unprocessable Entity.
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_synonyms_validates_inner_shape(client, kb_id):
    """Inner element is not a list of strings → 422 from Pydantic."""
    r = await client.patch(
        f"/api/kb/{kb_id}/synonyms",
        headers=ADMIN,
        json={"synonyms": [["ok", "also-ok"], 42]},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_synonyms_empty_list_accepted(client, kb_id):
    """An empty equivalence-class list is valid (no-op / reset)."""
    r = await client.patch(
        f"/api/kb/{kb_id}/synonyms",
        headers=ADMIN,
        json={"synonyms": []},
    )
    assert r.status_code == 200
    assert r.json()["synonyms"] == []


@pytest.mark.asyncio
async def test_patch_synonyms_missing_kb_returns_404(client):
    """A KB that does not exist returns 404."""
    r = await client.patch(
        "/api/kb/99999/synonyms",
        headers=ADMIN,
        json={"synonyms": [["A", "B"]]},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_patch_synonyms_deleted_kb_returns_404(client, kb_id):
    """A soft-deleted KB returns 404."""
    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 204

    r = await client.patch(
        f"/api/kb/{kb_id}/synonyms",
        headers=ADMIN,
        json={"synonyms": [["A", "B"]]},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_patch_synonyms_multiple_classes(client, kb_id):
    """Multiple equivalence classes are stored and echoed correctly."""
    synonyms = [
        ["5 PoK", "5 POK", "5 PoK Bde"],
        ["75 Inf", "75 INF", "75 Infantry Brigade"],
    ]
    r = await client.patch(
        f"/api/kb/{kb_id}/synonyms",
        headers=ADMIN,
        json={"synonyms": synonyms},
    )
    assert r.status_code == 200
    assert r.json()["synonyms"] == synonyms
