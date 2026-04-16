import pytest
from sqlalchemy import text
from ext.services import kb_service


@pytest.mark.asyncio
async def test_create_and_get_kb(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.commit()
    kb = await kb_service.create_kb(session, name="Engineering", description="eng docs", admin_id=1)
    await session.commit()
    got = await kb_service.get_kb(session, kb_id=kb.id)
    assert got.name == "Engineering"
    assert got.description == "eng docs"
    assert got.deleted_at is None


@pytest.mark.asyncio
async def test_list_kbs_excludes_soft_deleted(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'Alive', 1), (2, 'Dead', 1)"))
    await session.execute(text("UPDATE knowledge_bases SET deleted_at = now() WHERE id = 2"))
    await session.commit()
    kbs = await kb_service.list_kbs(session, kb_ids=[1, 2])
    assert {k.name for k in kbs} == {"Alive"}


@pytest.mark.asyncio
async def test_soft_delete_kb(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (5, 'K', 1)"))
    await session.commit()
    ok = await kb_service.soft_delete_kb(session, kb_id=5)
    await session.commit()
    assert ok is True
    got = await kb_service.get_kb(session, kb_id=5)
    assert got is None


@pytest.mark.asyncio
async def test_duplicate_kb_name_rejected(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'Dup', 1)"))
    await session.commit()
    with pytest.raises(Exception):
        await kb_service.create_kb(session, name="Dup", description=None, admin_id=1)
        await session.commit()
