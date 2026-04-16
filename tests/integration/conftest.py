"""Shared fixtures for integration tests needing a real Postgres + migrated schema."""
from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer

ROOT = Path(__file__).resolve().parents[2]
MIGRATION_001 = ROOT / "ext/db/migrations/001_create_kb_schema.sql"
MIGRATION_002 = ROOT / "ext/db/migrations/002_soft_delete_kb.sql"
MIGRATION_003 = ROOT / "ext/db/migrations/003_add_chunk_count.sql"


async def _raw_exec(conn, sql: str) -> None:
    """Run multi-statement DDL via asyncpg's raw connection."""
    raw = await conn.get_raw_connection()
    await raw.driver_connection.execute(sql)


@pytest.fixture(scope="session")
def pg():
    with PostgresContainer("postgres:15-alpine") as container:
        yield container


@pytest_asyncio.fixture(scope="function")
async def engine(pg):
    url = pg.get_connection_url().replace("psycopg2", "asyncpg")
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await _raw_exec(conn, """
            DROP TABLE IF EXISTS kb_access, kb_documents, kb_subtags, knowledge_bases CASCADE;
            DROP TABLE IF EXISTS chats, user_groups, users, groups CASCADE;
            CREATE TABLE users (
              id BIGSERIAL PRIMARY KEY,
              email TEXT UNIQUE NOT NULL,
              password_hash TEXT NOT NULL,
              role TEXT NOT NULL DEFAULT 'user'
            );
            CREATE TABLE groups (
              id BIGSERIAL PRIMARY KEY,
              name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE user_groups (
              user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              group_id BIGINT NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
              PRIMARY KEY (user_id, group_id)
            );
            CREATE TABLE chats (
              id BIGSERIAL PRIMARY KEY,
              user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              title TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        await _raw_exec(conn, MIGRATION_001.read_text())
        if MIGRATION_002.exists():
            await _raw_exec(conn, MIGRATION_002.read_text())
        if MIGRATION_003.exists():
            await _raw_exec(conn, MIGRATION_003.read_text())
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(scope="function")
async def session(engine) -> AsyncGenerator[AsyncSession, None]:
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        yield s


@pytest.fixture(scope="session")
def qdrant():
    """Session-scoped Qdrant container (reused across tests for speed)."""
    container = (
        DockerContainer("qdrant/qdrant:latest")
        .with_exposed_ports(6333)
    )
    container.start()
    host = container.get_container_host_ip()
    port = container.get_exposed_port(6333)
    import time
    import httpx
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://{host}:{port}/readyz", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        container.stop()
        raise RuntimeError("qdrant not ready")
    try:
        yield f"http://{host}:{port}"
    finally:
        container.stop()


@pytest_asyncio.fixture(scope="function")
async def clean_qdrant(qdrant):
    """Per-test wiper: deletes all collections at teardown."""
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(url=qdrant)
    yield qdrant
    cols = (await client.get_collections()).collections
    for c in cols:
        await client.delete_collection(c.name)
    await client.close()
