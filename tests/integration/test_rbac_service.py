import pytest
from sqlalchemy import text
from ext.services.rbac import get_allowed_kb_ids


@pytest.mark.asyncio
async def test_rbac_direct_and_group_grants(session):
    await session.execute(text(
        "INSERT INTO users (id, email, password_hash, role) VALUES"
        " (1, 'a@x', 'h', 'user'), (2, 'b@x', 'h', 'user')"
    ))
    await session.execute(text(
        "INSERT INTO groups (id, name) VALUES (1, 'eng'), (2, 'hr')"
    ))
    await session.execute(text(
        "INSERT INTO user_groups (user_id, group_id) VALUES (1, 1), (2, 2)"
    ))
    await session.execute(text(
        "INSERT INTO knowledge_bases (id, name, admin_id) VALUES"
        " (10, 'EngKB', 1), (11, 'HrKB', 1), (12, 'DirectKB', 1)"
    ))
    await session.execute(text(
        "INSERT INTO kb_access (kb_id, user_id, group_id, access_type) VALUES"
        " (10, NULL, 1, 'read'), (11, NULL, 2, 'read'), (12, 1, NULL, 'read')"
    ))
    await session.commit()

    assert set(await get_allowed_kb_ids(session, user_id=1)) == {10, 12}
    assert set(await get_allowed_kb_ids(session, user_id=2)) == {11}
    assert await get_allowed_kb_ids(session, user_id=999) == []


@pytest.mark.asyncio
async def test_rbac_admin_sees_everything(session):
    await session.execute(text(
        "INSERT INTO users (id, email, password_hash, role) VALUES"
        " (1, 'root@x', 'h', 'admin')"
    ))
    await session.execute(text(
        "INSERT INTO knowledge_bases (id, name, admin_id) VALUES"
        " (10, 'A', 1), (11, 'B', 1), (12, 'C', 1)"
    ))
    await session.commit()
    assert set(await get_allowed_kb_ids(session, user_id=1)) == {10, 11, 12}
