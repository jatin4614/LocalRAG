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
MIGRATION = ROOT / "ext/db/migrations/001_create_kb_schema.sql"
APPLIER = ROOT / "scripts" / "apply_migrations.py"


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


# ---------------------------------------------------------------------------
# Cold-start: migrate from an empty Postgres and verify the resulting
# schema matches what production has (verified via docker exec \d).
# ---------------------------------------------------------------------------
async def _column_type(conn, table: str, col: str) -> tuple[str, int | None]:
    """Return (data_type, character_maximum_length) for a (table, column)."""
    row = (
        await conn.execute(
            text(
                "SELECT data_type, character_maximum_length "
                "FROM information_schema.columns "
                "WHERE table_name = :t AND column_name = :c"
            ),
            {"t": table, "c": col},
        )
    ).one_or_none()
    assert row is not None, f"missing column {table}.{col}"
    return row[0], row[1]


@pytest.mark.asyncio
async def test_cold_start_schema_matches_production():
    """A fresh ``apply_migrations`` against an empty Postgres reproduces production.

    Specifically:
      * knowledge_bases.admin_id     varchar(255) NOT NULL, no FK
      * kb_documents.uploaded_by      varchar(255) NOT NULL, no FK
      * kb_access.user_id             varchar(255), no FK
      * kb_access.group_id            text, no FK
      * kb_subtags.deleted_at         timestamptz (migration 014)
      * kb_access.deleted_at          timestamptz (migration 015)
      * kb_documents.doc_summary      text         (migration 008)
      * schema_migrations table exists with one row per migration file
    """
    with PostgresContainer("postgres:15-alpine") as pg:
        async_url = pg.get_connection_url().replace("psycopg2", "asyncpg")

        # Seed the upstream tables that migration 001 expects to be
        # present (it ALTERs ``chats``).
        eng = create_async_engine(async_url)
        async with eng.begin() as conn:
            raw = await conn.get_raw_connection()
            await raw.driver_connection.execute(
                """
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
                """
            )
        await eng.dispose()

        # Run the applier from scratch.
        env = os.environ.copy()
        env["DATABASE_URL"] = async_url
        r = subprocess.run(
            [sys.executable, str(APPLIER)],
            env=env, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr

        # Now inspect the schema.
        eng = create_async_engine(async_url)
        async with eng.connect() as conn:
            assert await _column_type(conn, "knowledge_bases", "admin_id") == (
                "character varying", 255,
            )
            assert await _column_type(conn, "kb_documents", "uploaded_by") == (
                "character varying", 255,
            )
            assert await _column_type(conn, "kb_access", "user_id") == (
                "character varying", 255,
            )
            assert await _column_type(conn, "kb_access", "group_id") == ("text", None)

            # No FKs from kb_access.user_id / group_id to upstream
            # users / groups (a fresh apply would reject the create
            # if users had been BIGINT-typed otherwise).
            fks = (
                await conn.execute(
                    text(
                        "SELECT conname FROM pg_constraint "
                        "WHERE contype = 'f' "
                        "AND conrelid = 'kb_access'::regclass"
                    )
                )
            ).scalars().all()
            assert fks == ["kb_access_kb_id_fkey"], (
                f"kb_access should only have its kb_id FK; got {fks}"
            )
            fks = (
                await conn.execute(
                    text(
                        "SELECT conname FROM pg_constraint "
                        "WHERE contype = 'f' "
                        "AND conrelid = 'knowledge_bases'::regclass"
                    )
                )
            ).scalars().all()
            assert fks == [], (
                f"knowledge_bases should have no FKs; got {fks}"
            )
            fks = (
                await conn.execute(
                    text(
                        "SELECT conname FROM pg_constraint "
                        "WHERE contype = 'f' AND conname LIKE '%uploaded_by%' "
                        "AND conrelid = 'kb_documents'::regclass"
                    )
                )
            ).scalars().all()
            assert fks == [], (
                f"kb_documents.uploaded_by should have no FK; got {fks}"
            )

            # Migration 008 column.
            row = (
                await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'kb_documents' AND column_name = 'doc_summary'"
                    )
                )
            ).one_or_none()
            assert row is not None, "kb_documents.doc_summary missing"

            # Migration 014 column.
            row = (
                await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'kb_subtags' AND column_name = 'deleted_at'"
                    )
                )
            ).one_or_none()
            assert row is not None, "kb_subtags.deleted_at missing"

            # Migration 015 column.
            row = (
                await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'kb_access' AND column_name = 'deleted_at'"
                    )
                )
            ).one_or_none()
            assert row is not None, "kb_access.deleted_at missing"

            # schema_migrations history.
            cnt = (
                await conn.execute(text("SELECT COUNT(*) FROM schema_migrations"))
            ).scalar_one()
            n_files = len(list((ROOT / "ext" / "db" / "migrations").glob("*.sql")))
            assert cnt == n_files, f"history rows {cnt} != file count {n_files}"
        await eng.dispose()
