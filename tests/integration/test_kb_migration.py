import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer
from pathlib import Path

MIGRATION = Path(__file__).resolve().parents[2] / "ext/db/migrations/001_create_kb_schema.sql"


@pytest.fixture(scope="module")
def pg():
    with PostgresContainer("postgres:15-alpine") as pg:
        yield pg


@pytest.mark.asyncio
async def test_migration_creates_kb_tables(pg):
    async_url = pg.get_connection_url().replace("psycopg2", "asyncpg")
    engine = create_async_engine(async_url)

    # Simulate upstream schema: minimal users, groups, knowledge, chats tables.
    # asyncpg does not support multi-statement prepared queries; use raw connection.
    async with engine.connect() as conn:
        raw = await conn.get_raw_connection()
        raw_conn = raw.driver_connection
        await raw_conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
              id BIGSERIAL PRIMARY KEY, email TEXT UNIQUE, role TEXT
            );
            CREATE TABLE IF NOT EXISTS groups (
              id BIGSERIAL PRIMARY KEY, name TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS knowledge (
              id TEXT PRIMARY KEY, name TEXT
            );
            CREATE TABLE IF NOT EXISTS chats (
              id BIGSERIAL PRIMARY KEY, user_id BIGINT REFERENCES users(id)
            );
        """)

    sql = MIGRATION.read_text()
    async with engine.connect() as conn:
        raw = await conn.get_raw_connection()
        raw_conn = raw.driver_connection
        await raw_conn.execute(sql)

    async with engine.connect() as conn:
        tables = (await conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ))).scalars().all()
    assert "knowledge_bases" in tables
    assert "kb_subtags" in tables
    assert "kb_documents" in tables
    assert "kb_access" in tables
    assert "knowledge" not in tables  # upstream table dropped

    async with engine.connect() as conn:
        cols = (await conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='chats'"
        ))).scalars().all()
    assert "selected_kb_config" in cols

    await engine.dispose()
