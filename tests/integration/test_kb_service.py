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


@pytest.mark.asyncio
async def test_create_and_list_subtags(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1)"))
    await session.commit()
    await kb_service.create_subtag(session, kb_id=1, name="OFC", description=None)
    await kb_service.create_subtag(session, kb_id=1, name="Roadmap", description="q2")
    await session.commit()
    subs = await kb_service.list_subtags(session, kb_id=1)
    assert {s.name for s in subs} == {"OFC", "Roadmap"}


@pytest.mark.asyncio
async def test_duplicate_subtag_within_kb_rejected(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1)"))
    await session.execute(text("INSERT INTO kb_subtags (kb_id, name) VALUES (1, 'X')"))
    await session.commit()
    with pytest.raises(Exception):
        await kb_service.create_subtag(session, kb_id=1, name="X", description=None)
        await session.commit()


@pytest.mark.asyncio
async def test_delete_subtag(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1)"))
    await session.execute(text("INSERT INTO kb_subtags (id, kb_id, name) VALUES (10, 1, 'Del')"))
    await session.commit()
    ok = await kb_service.delete_subtag(session, kb_id=1, subtag_id=10)
    await session.commit()
    assert ok is True
    subs = await kb_service.list_subtags(session, kb_id=1)
    assert subs == []


@pytest.mark.asyncio
async def test_grant_user_and_group_access(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin'), (2, 'b@x', 'h', 'user')"))
    await session.execute(text("INSERT INTO groups (id, name) VALUES (1, 'eng')"))
    await session.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1)"))
    await session.commit()

    await kb_service.grant_access(session, kb_id=1, user_id=2, group_id=None)
    await kb_service.grant_access(session, kb_id=1, user_id=None, group_id=1)
    await session.commit()
    grants = await kb_service.list_access(session, kb_id=1)
    assert len(grants) == 2
    assert {g.user_id for g in grants if g.user_id is not None} == {2}
    assert {g.group_id for g in grants if g.group_id is not None} == {1}


@pytest.mark.asyncio
async def test_grant_requires_exactly_one_of_user_or_group(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1)"))
    await session.commit()
    with pytest.raises(ValueError):
        await kb_service.grant_access(session, kb_id=1, user_id=None, group_id=None)
    with pytest.raises(ValueError):
        await kb_service.grant_access(session, kb_id=1, user_id=1, group_id=1)


@pytest.mark.asyncio
async def test_revoke_access(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.execute(text("INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1)"))
    await session.execute(text("INSERT INTO kb_access (id, kb_id, user_id, access_type) VALUES (100, 1, 1, 'read')"))
    await session.commit()
    ok = await kb_service.revoke_access(session, grant_id=100)
    await session.commit()
    assert ok is True
    grants = await kb_service.list_access(session, kb_id=1)
    assert grants == []
