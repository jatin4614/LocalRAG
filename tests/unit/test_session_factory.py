import pytest
from ext.db.session import make_engine, make_sessionmaker


@pytest.mark.asyncio
async def test_session_round_trip():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    SessionLocal = make_sessionmaker(engine)
    async with SessionLocal() as s:
        from sqlalchemy import text
        r = (await s.execute(text("SELECT 1"))).scalar()
        assert r == 1
    await engine.dispose()
