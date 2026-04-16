import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, Column, Integer, String as SAString, Table

from ext.db.base import Base
from ext.db.models.kb import KnowledgeBase, KBSubtag, KBDocument, KBAccess

# Minimal stub tables so Base.metadata can resolve FKs (users, groups are Open WebUI tables).
_stub_meta = Base.metadata
Table("users",  _stub_meta, Column("id", Integer, primary_key=True), Column("email", SAString), extend_existing=True)
Table("groups", _stub_meta, Column("id", Integer, primary_key=True), Column("name",  SAString), extend_existing=True)


@pytest.mark.asyncio
async def test_kb_models_create_and_query():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        from sqlalchemy import text

        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("INSERT INTO users(id,email)  VALUES (1,'a@x')"))
        await conn.execute(text("INSERT INTO groups(id,name)  VALUES (1,'eng')"))

    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        # SQLite does not auto-increment BigInteger PKs; supply explicit IDs.
        # (PostgreSQL uses sequences; this is a test-only accommodation.)
        kb = KnowledgeBase(id=1, name="Engineering", description="Eng docs", admin_id=1)
        s.add(kb)
        await s.flush()

        sub = KBSubtag(id=1, kb_id=kb.id, name="OFC")
        s.add(sub)
        await s.flush()

        doc = KBDocument(id=1, kb_id=kb.id, subtag_id=sub.id, filename="a.pdf", uploaded_by=1)
        s.add(doc)

        acc = KBAccess(id=1, kb_id=kb.id, group_id=1, access_type="read")
        s.add(acc)

        await s.commit()

        kbs = (await s.execute(select(KnowledgeBase))).scalars().all()
        assert len(kbs) == 1
        assert kbs[0].name == "Engineering"

    await engine.dispose()


def test_kb_access_check_enforced_in_model():
    with pytest.raises(ValueError):
        KBAccess(kb_id=1, user_id=None, group_id=None)
    with pytest.raises(ValueError):
        KBAccess(kb_id=1, user_id=1, group_id=1)
