import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from ext.db.base import Base
from ext.db.models.compat import User, Group, UserGroup, Chat


@pytest.mark.asyncio
async def test_compat_models_round_trip():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        u = User(id=1, email="a@x", password_hash="h", role="admin")
        g = Group(id=1, name="eng")
        s.add_all([u, g])
        await s.flush()
        s.add(UserGroup(user_id=1, group_id=1))
        await s.flush()
        s.add(Chat(id=1, user_id=1, title="hello"))
        await s.flush()
        await s.commit()
        rows = (await s.execute(select(User))).scalars().all()
        assert len(rows) == 1
        assert rows[0].role == "admin"
    await eng.dispose()
