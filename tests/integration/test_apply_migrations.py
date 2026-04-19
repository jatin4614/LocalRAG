import os
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.integration


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "apply_migrations.py"


@pytest_asyncio.fixture
async def pg_seeded():
    """Postgres with upstream-style seed tables already created (but no kb_* tables)."""
    with PostgresContainer("postgres:15-alpine") as pg:
        async_url = pg.get_connection_url().replace("psycopg2", "asyncpg")
        sync_url  = pg.get_connection_url()
        eng = create_async_engine(async_url)
        async with eng.begin() as conn:
            raw = await conn.get_raw_connection()
            await raw.driver_connection.execute("""
                CREATE TABLE users (
                  id BIGSERIAL PRIMARY KEY,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  role TEXT NOT NULL
                );
                CREATE TABLE groups (
                  id BIGSERIAL PRIMARY KEY, name TEXT UNIQUE
                );
                CREATE TABLE chats (
                  id BIGSERIAL PRIMARY KEY, user_id BIGINT
                );
            """)
        await eng.dispose()
        yield sync_url, async_url


@pytest.mark.asyncio
async def test_applier_creates_kb_tables(pg_seeded):
    sync_url, async_url = pg_seeded
    env = os.environ.copy()
    env["DATABASE_URL"] = async_url
    r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    eng = create_async_engine(async_url)
    async with eng.connect() as conn:
        tables = (await conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ))).scalars().all()
    assert "knowledge_bases" in tables
    assert "kb_access" in tables
    async with eng.connect() as conn:
        cols = (await conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='kb_documents'"
        ))).scalars().all()
    assert "chunk_count" in cols
    await eng.dispose()


@pytest.mark.asyncio
async def test_applier_is_idempotent(pg_seeded):
    sync_url, async_url = pg_seeded
    env = os.environ.copy()
    env["DATABASE_URL"] = async_url
    for _ in range(3):
        r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
