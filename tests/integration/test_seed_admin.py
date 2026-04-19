import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "scripts" / "seed_admin.py"
MIGRATION = ROOT / "ext/db/migrations/001_create_kb_schema.sql"


@pytest.mark.asyncio
async def test_seed_admin_idempotent():
    with PostgresContainer("postgres:15-alpine") as pg:
        async_url = pg.get_connection_url().replace("psycopg2", "asyncpg")
        engine = create_async_engine(async_url)

        # Minimal upstream-style tables for seed_admin to operate on.
        # Use the raw asyncpg driver connection so we can submit multi-statement DDL
        # (asyncpg rejects multi-statement prepared queries).
        async with engine.begin() as conn:
            raw = await conn.get_raw_connection()
            await raw.driver_connection.execute("""
                CREATE TABLE IF NOT EXISTS users (
                  id BIGSERIAL PRIMARY KEY,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  role TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS groups (
                  id BIGSERIAL PRIMARY KEY,
                  name TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS chats (
                  id BIGSERIAL PRIMARY KEY,
                  user_id BIGINT
                );
            """)
            sql = MIGRATION.read_text()
            raw2 = await conn.get_raw_connection()
            await raw2.driver_connection.execute(sql)

        env = os.environ.copy()
        env["DATABASE_URL"] = async_url
        env["ADMIN_EMAIL"]  = "admin@test.local"
        env["ADMIN_PASSWORD"] = "hunter2-hunter2"  # 15 chars, passes >=12 check

        r1 = subprocess.run([sys.executable, str(SEED)], env=env, capture_output=True, text=True)
        assert r1.returncode == 0, r1.stderr
        r2 = subprocess.run([sys.executable, str(SEED)], env=env, capture_output=True, text=True)
        assert r2.returncode == 0, r2.stderr  # idempotent

        async with engine.connect() as conn:
            count = (await conn.execute(text(
                "SELECT COUNT(*) FROM users WHERE email=:e"), {"e": "admin@test.local"})).scalar()
        assert count == 1
        await engine.dispose()
