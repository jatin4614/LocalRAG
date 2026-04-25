"""Shared fixtures for integration tests needing a real Postgres + migrated schema."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = ROOT / "ext/db/migrations"


async def _raw_exec(conn, sql: str) -> None:
    """Run multi-statement DDL via asyncpg's raw connection."""
    raw = await conn.get_raw_connection()
    await raw.driver_connection.execute(sql)


def _all_migration_paths() -> list[Path]:
    """Return every NNN_*.sql migration in numeric order.

    Picking up new migrations without conftest churn — when Plan A /
    Plan B add a new migration the test schema stays in lockstep with
    production. Each migration is expected to be idempotent enough to
    run against a fresh DB (CREATE TABLE IF NOT EXISTS / ALTER TABLE
    ADD COLUMN IF NOT EXISTS).
    """
    return sorted(
        MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"),
        key=lambda p: p.name,
    )


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
            DROP TABLE IF EXISTS chat, chats, user_groups, users, groups CASCADE;
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
        for mig in _all_migration_paths():
            await _raw_exec(conn, mig.read_text())
        # Production has drifted from migration 001: the SQLAlchemy
        # model declares ``admin_id``, ``uploaded_by``, ``user_id``
        # as VARCHAR(255) (upstream's user.id is a UUID string) and
        # ``group_id`` as TEXT, but migration 001 still says BIGINT.
        # Live production was bootstrapped via SQLAlchemy
        # ``create_all`` which respects the model types — production
        # schema is the source of truth.
        #
        # ALTER the columns post-migration so SQLAlchemy can round-
        # trip model-typed values. asyncpg silently coerces ``int`` ->
        # ``str`` for VARCHAR, so legacy test fixtures that pass
        # ``admin_id=1`` keep working.
        await _raw_exec(conn, """
            ALTER TABLE knowledge_bases
              DROP CONSTRAINT IF EXISTS knowledge_bases_admin_id_fkey;
            ALTER TABLE knowledge_bases
              ALTER COLUMN admin_id TYPE VARCHAR(255) USING admin_id::text;

            ALTER TABLE kb_documents
              DROP CONSTRAINT IF EXISTS kb_documents_uploaded_by_fkey;
            ALTER TABLE kb_documents
              ALTER COLUMN uploaded_by TYPE VARCHAR(255) USING uploaded_by::text;

            ALTER TABLE kb_access
              DROP CONSTRAINT IF EXISTS kb_access_user_id_fkey;
            ALTER TABLE kb_access
              ALTER COLUMN user_id TYPE VARCHAR(255) USING user_id::text;

            ALTER TABLE kb_access
              DROP CONSTRAINT IF EXISTS kb_access_group_id_fkey;
            ALTER TABLE kb_access
              ALTER COLUMN group_id TYPE TEXT USING group_id::text;
        """)
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


# ---------------------------------------------------------------------------
# Phase 1.5 — RBAC cache fixtures (CLAUDE.md §2 isolation tests)
# ---------------------------------------------------------------------------
#
# The cache itself is module-under-test (`ext.services.rbac_cache`); these
# fixtures spin up a real Redis (testcontainer), seed Postgres with the
# minimum users / groups / KBs, and provide grant-mutation helpers that
# (a) update Postgres state and (b) optionally publish the invalidation
# message that the subscriber would normally drop. ``query_allowed_ids``
# is a thin cache-first wrapper that mirrors the production flow in
# ``chat_rag_bridge`` so the tests exercise the actual cache contract,
# not a mock.
#
# Why a custom resolver instead of calling ``get_allowed_kb_ids``?
# ``ext.services.rbac.get_allowed_kb_ids`` queries upstream's "user" /
# "group_member" tables (singular names, UUID ids) which the integration
# conftest does not provision. The fixtures here read from the test's
# ``users`` / ``user_groups`` tables. This is intentional scope-limiting:
# Phase 1.5's contract is the cache layer, not the underlying SQL.

@pytest.fixture(scope="session")
def rbac_redis_container():
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest_asyncio.fixture(scope="function")
async def redis_client(rbac_redis_container):
    """Async Redis client connected to the testcontainer.

    Per-test scope so each test gets a clean DB. We FLUSHDB before
    yielding to clear any leftover keys from a previous test, and reset
    the process-wide ``RbacCache`` singleton so it picks up the fresh
    handle / TTL config.
    """
    import redis.asyncio as aioredis
    from ext.services import rbac_cache as _rc

    host = rbac_redis_container.get_container_host_ip()
    port = int(rbac_redis_container.get_exposed_port(6379))
    url = f"redis://{host}:{port}/0"
    client = aioredis.from_url(url)
    await client.flushdb()

    # Reset the RbacCache singleton AND make it use this client.
    _rc._reset_shared_cache_for_tests()

    yield client

    try:
        await client.flushdb()
    finally:
        await client.aclose()
    _rc._reset_shared_cache_for_tests()


@pytest_asyncio.fixture(scope="function")
async def rbac_db_session(engine):
    """Per-test session for RBAC fixtures, with the standard tables seeded.

    Re-uses the integration engine fixture (Postgres testcontainer) and
    creates a fresh session bound to ``async_sessionmaker``.
    """
    SessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with SessionLocal() as s:
        yield s


@pytest.fixture
def test_user_a() -> int:
    return 1001


@pytest.fixture
def test_user_b() -> int:
    return 1002


@pytest.fixture
def test_kb_x() -> int:
    return 9001


@pytest.fixture
def test_kb_y() -> int:
    return 9002


@pytest.fixture
def test_group_x() -> int:
    return 8001


@pytest_asyncio.fixture(scope="function")
async def _seed_rbac_fixtures(
    rbac_db_session,
    test_user_a,
    test_user_b,
    test_kb_x,
    test_kb_y,
    test_group_x,
):
    """Insert the small fixed RBAC universe expected by the 6 tests.

    Idempotent within a test (engine fixture is per-function so the
    Postgres state is fresh). Commits so the rows are visible to the
    other helpers' implicit transactions.
    """
    s = rbac_db_session
    await s.execute(
        text(
            "INSERT INTO users (id, email, password_hash, role) VALUES "
            "(:ua, :ea, 'h', 'user'), (:ub, :eb, 'h', 'user')"
        ),
        {
            "ua": test_user_a,
            "ub": test_user_b,
            "ea": f"a-{test_user_a}@x",
            "eb": f"b-{test_user_b}@x",
        },
    )
    await s.execute(
        text("INSERT INTO groups (id, name) VALUES (:gx, 'gx')"),
        {"gx": test_group_x},
    )
    # admin_id is VARCHAR(255) post-migration drift; cast int -> str
    # explicitly so asyncpg's type binding doesn't reject the param.
    await s.execute(
        text(
            "INSERT INTO knowledge_bases (id, name, admin_id) VALUES "
            "(:kx, 'KBX', :ua), (:ky, 'KBY', :ua)"
        ),
        {
            "kx": test_kb_x,
            "ky": test_kb_y,
            "ua": str(test_user_a),
        },
    )
    await s.commit()


@pytest_asyncio.fixture(scope="function")
async def assign_user_to_group(rbac_db_session, _seed_rbac_fixtures):
    """Insert a user_groups row.

    Returns an async callable so tests can invoke per-row.
    """
    s = rbac_db_session

    async def _assign(user_id: int, group_id: int) -> None:
        await s.execute(
            text(
                "INSERT INTO user_groups (user_id, group_id) "
                "VALUES (:u, :g) ON CONFLICT DO NOTHING"
            ),
            {"u": user_id, "g": group_id},
        )
        await s.commit()

    return _assign


@pytest_asyncio.fixture(scope="function")
async def revoke_user_from_group(
    rbac_db_session,
    redis_client,
    _seed_rbac_fixtures,
):
    """Delete a user_groups row, then publish the RBAC invalidation event.

    ``skip_pubsub=True`` simulates a dropped pub/sub message so the TTL
    safety net can be exercised. The DELETE always commits regardless.
    """
    from ext.services.rbac_cache import PUBSUB_CHANNEL
    import json as _json

    s = rbac_db_session

    async def _revoke(
        user_id: int, group_id: int, *, skip_pubsub: bool = False
    ) -> None:
        await s.execute(
            text(
                "DELETE FROM user_groups WHERE user_id = :u AND group_id = :g"
            ),
            {"u": user_id, "g": group_id},
        )
        await s.commit()
        if not skip_pubsub:
            # Mirror what the kb_admin router does after a kb_access
            # mutation: drop the cache key AND broadcast.
            from ext.services.rbac_cache import CACHE_NAMESPACE
            await redis_client.delete(f"{CACHE_NAMESPACE}:{user_id}")
            await redis_client.publish(
                PUBSUB_CHANNEL,
                _json.dumps({"user_ids": [str(user_id)]}).encode("utf-8"),
            )

    return _revoke


@pytest_asyncio.fixture(scope="function")
async def grant_kb_to_group(rbac_db_session, _seed_rbac_fixtures):
    s = rbac_db_session

    async def _grant(kb_id: int, group_id: int) -> None:
        # group_id column is TEXT post-migration drift.
        await s.execute(
            text(
                "INSERT INTO kb_access (kb_id, user_id, group_id, access_type) "
                "VALUES (:kb, NULL, :g, 'read')"
            ),
            {"kb": kb_id, "g": str(group_id)},
        )
        await s.commit()

    return _grant


@pytest_asyncio.fixture(scope="function")
async def grant_kb_to_user(rbac_db_session, _seed_rbac_fixtures):
    s = rbac_db_session

    async def _grant(kb_id: int, user_id: int) -> None:
        # user_id column is VARCHAR(255) post-migration drift.
        await s.execute(
            text(
                "INSERT INTO kb_access (kb_id, user_id, group_id, access_type) "
                "VALUES (:kb, :u, NULL, 'read')"
            ),
            {"kb": kb_id, "u": str(user_id)},
        )
        await s.commit()

    return _grant


async def _resolve_allowed_kb_ids_from_db(
    session: AsyncSession, *, user_id: int
) -> set[int]:
    """Compute ``allowed_kb_ids`` against the integration test schema.

    Mirrors :func:`ext.services.rbac.get_allowed_kb_ids` but reads from
    the ``users`` / ``user_groups`` tables that the integration conftest
    actually provisions (upstream uses ``"user"`` / ``group_member``
    which are not present here).
    """
    role_row = (
        await session.execute(
            text("SELECT role FROM users WHERE id = :uid"), {"uid": user_id}
        )
    ).first()
    if role_row is None:
        return set()
    role = role_row[0]

    if role == "admin":
        rows = (
            await session.execute(
                text(
                    "SELECT id FROM knowledge_bases WHERE deleted_at IS NULL"
                )
            )
        ).scalars().all()
        return set(int(r) for r in rows)

    group_rows = (
        await session.execute(
            text(
                "SELECT group_id FROM user_groups WHERE user_id = :uid"
            ),
            {"uid": user_id},
        )
    ).scalars().all()
    group_ids = [int(g) for g in group_rows]

    # kb_access.user_id is VARCHAR(255); group_id is TEXT. Cast inputs
    # to str so the parameter bind matches the column type.
    uid_str = str(user_id)
    if group_ids:
        gids_str = [str(g) for g in group_ids]
        rows = (
            await session.execute(
                text(
                    "SELECT kb_id FROM kb_access "
                    "WHERE user_id = :uid OR group_id = ANY(:gids)"
                ),
                {"uid": uid_str, "gids": gids_str},
            )
        ).scalars().all()
    else:
        rows = (
            await session.execute(
                text("SELECT kb_id FROM kb_access WHERE user_id = :uid"),
                {"uid": uid_str},
            )
        ).scalars().all()
    return set(int(r) for r in rows)


@pytest_asyncio.fixture(scope="function")
async def query_allowed_ids(rbac_db_session, redis_client):
    """Cache-first wrapper around :func:`_resolve_allowed_kb_ids_from_db`.

    This mirrors the production flow in
    :mod:`ext.services.chat_rag_bridge` so the 6 tests exercise the
    actual cache contract: cache hit -> return; cache miss -> DB +
    populate cache.
    """
    from ext.services.rbac_cache import RbacCache

    s = rbac_db_session

    async def _query(user_id: int) -> set[int]:
        # Use a bound RbacCache instance per-test so the TTL fixture
        # can swap it (see monkeypatch_ttl_to_1sec). We re-resolve the
        # singleton on every call so test fixtures that reset / replace
        # it take effect immediately.
        from ext.services.rbac_cache import get_shared_cache
        cache = get_shared_cache(redis=redis_client)
        cached = await cache.get(user_id=str(user_id))
        if cached is not None:
            return cached
        allowed = await _resolve_allowed_kb_ids_from_db(s, user_id=user_id)
        await cache.set(user_id=str(user_id), allowed_kb_ids=allowed)
        return allowed

    return _query


@pytest_asyncio.fixture(scope="function")
async def monkeypatch_ttl_to_1sec(redis_client):
    """Force the process-wide RbacCache to use TTL=1s.

    Resets the singleton so the next ``get_shared_cache`` returns a new
    instance with the shorter TTL. Used by tests 3 and 4 to exercise
    the safety-net behaviour without sleeping for 30s.
    """
    from ext.services import rbac_cache as _rc

    _rc._reset_shared_cache_for_tests()
    # Override the env so subsequent ``RbacCache()`` instantiations pick
    # up the new TTL.
    import os
    prev = os.environ.get("RAG_RBAC_CACHE_TTL_SECS")
    os.environ["RAG_RBAC_CACHE_TTL_SECS"] = "1"
    yield
    if prev is None:
        os.environ.pop("RAG_RBAC_CACHE_TTL_SECS", None)
    else:
        os.environ["RAG_RBAC_CACHE_TTL_SECS"] = prev
    _rc._reset_shared_cache_for_tests()


@pytest_asyncio.fixture(scope="function")
async def disable_pubsub_subscribe():
    """No-op placeholder.

    Test 4 simulates a replica whose pub/sub subscriber is broken (the
    network dropped the message, the listener task crashed, etc.). In
    this conftest the test never starts a subscriber to begin with --
    invalidations are issued by the ``revoke_user_from_group`` fixture
    inline. So the way to simulate a "dropped" message is simply not
    to call the cache-eviction half of revoke. To prove TTL works we
    therefore route test 4 through the same code path with the regular
    revoke (which DOES try to publish), but the only way the cached key
    could disappear is via the in-process delete in revoke. We verify
    the TTL alone catches a stale entry by combining this no-op fixture
    with the 1-second TTL fixture and asserting after a 1.4s sleep.

    Future-proofing: if a long-running subscriber task is added (for
    multi-replica deployments), this fixture is the right place to
    cancel it for the test's duration.
    """
    yield
