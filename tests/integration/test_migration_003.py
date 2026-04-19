import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_kb_documents_has_chunk_count(session):
    rows = (await session.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name='kb_documents'"
    ))).scalars().all()
    assert "chunk_count" in rows
