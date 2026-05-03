"""Migration 017 adds the `synonyms` JSONB column to knowledge_bases.

The column defaults to an empty JSON array so existing rows are
unaffected.
"""
import json
import os
import pytest
import asyncpg


pytestmark = pytest.mark.asyncio


async def _conn():
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://orgchat@localhost:5432/orgchat",
    )
    # asyncpg expects postgresql:// not postgresql+asyncpg://
    url = url.replace("+asyncpg", "")
    return await asyncpg.connect(url)


async def test_synonyms_column_exists_after_migration():
    """The `synonyms` column on knowledge_bases is JSONB NOT NULL DEFAULT '[]'."""
    conn = await _conn()
    try:
        rows = await conn.fetch("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'knowledge_bases' AND column_name = 'synonyms'
        """)
        assert rows, "synonyms column missing — migration 017 not applied"
        col = rows[0]
        assert col["data_type"] == "jsonb"
        assert col["is_nullable"] == "NO"
        # Default should evaluate to '[]'::jsonb
        default = (col["column_default"] or "").lower()
        assert "'[]'" in default and "jsonb" in default
    finally:
        await conn.close()


async def test_existing_rows_have_empty_array_default():
    """Pre-existing KBs (1, 2, 3, 8) keep working — synonyms = [] each."""
    conn = await _conn()
    try:
        rows = await conn.fetch(
            "SELECT id, synonyms FROM knowledge_bases WHERE id IN (2, 3, 8)"
        )
        for r in rows:
            # asyncpg returns JSONB as a string; decode before comparing.
            raw = r["synonyms"]
            val = json.loads(raw) if isinstance(raw, str) else raw
            assert val == [], (
                f"KB {r['id']} expected synonyms=[], got {raw!r}"
            )
    finally:
        await conn.close()
