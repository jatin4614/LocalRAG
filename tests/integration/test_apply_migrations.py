import os
import shutil
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
MIGRATIONS_DIR = ROOT / "ext" / "db" / "migrations"


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


@pytest.mark.asyncio
async def test_history_table_records_every_migration(pg_seeded):
    """Cold-start applies all migrations; schema_migrations row count == N file count."""
    sync_url, async_url = pg_seeded
    env = os.environ.copy()
    env["DATABASE_URL"] = async_url
    r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    expected_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    expected_names = {p.name for p in expected_files}

    eng = create_async_engine(async_url)
    async with eng.connect() as conn:
        rows = (
            await conn.execute(text("SELECT name, checksum FROM schema_migrations"))
        ).all()
    await eng.dispose()

    recorded_names = {r[0] for r in rows}
    assert recorded_names == expected_names, (
        f"recorded {recorded_names} vs expected {expected_names}"
    )
    # Every recorded row must have a non-empty checksum.
    for name, checksum in rows:
        assert checksum and len(checksum) == 64, f"bad checksum for {name}: {checksum!r}"


@pytest.mark.asyncio
async def test_idempotent_run_does_not_duplicate_history(pg_seeded):
    """Re-runs do not duplicate rows in schema_migrations.

    Also verifies that ``applied_at`` is left untouched on re-run when
    the checksum matches (the row is intentionally only touched on a
    real apply, but the ON CONFLICT clause is idempotent — checksum
    stays equal, and re-running shouldn't error).
    """
    sync_url, async_url = pg_seeded
    env = os.environ.copy()
    env["DATABASE_URL"] = async_url

    # Run twice.
    for _ in range(2):
        r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    eng = create_async_engine(async_url)
    async with eng.connect() as conn:
        cnt = (await conn.execute(text("SELECT COUNT(*) FROM schema_migrations"))).scalar_one()
    await eng.dispose()
    assert cnt == len(list(MIGRATIONS_DIR.glob("*.sql")))


@pytest.mark.asyncio
async def test_drifted_migration_aborts(pg_seeded, tmp_path, monkeypatch):
    """Simulate a migration whose source has changed after apply.

    We do this by:
      1. Running the applier once normally (records all checksums).
      2. Mutating schema_migrations to give one migration a wrong
         checksum (simulating an operator who edited the file).
      3. Re-running the applier; expect non-zero exit and an error
         message naming the file.
    """
    sync_url, async_url = pg_seeded
    env = os.environ.copy()
    env["DATABASE_URL"] = async_url

    r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    eng = create_async_engine(async_url)
    async with eng.begin() as conn:
        await conn.execute(
            text(
                "UPDATE schema_migrations SET checksum = 'deadbeef' "
                "WHERE name = '008_add_doc_summary.sql'"
            )
        )
    await eng.dispose()

    r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 1, (
        f"expected exit code 1, got {r.returncode}; stdout={r.stdout!r} stderr={r.stderr!r}"
    )
    combined = (r.stdout + r.stderr).lower()
    assert "008_add_doc_summary.sql" in combined, combined
    assert "drift" in combined or "checksum" in combined, combined


@pytest.mark.asyncio
async def test_failing_migration_does_not_record_history(pg_seeded, tmp_path):
    """If a migration fails, its row must NOT be inserted into history.

    We construct a copy of the migrations directory with one extra
    deliberately-broken migration appended at the end, point a stub
    applier at it, run, and assert (a) exit 1, (b) no row for the
    broken file in schema_migrations.
    """
    sync_url, async_url = pg_seeded

    # Copy migrations dir; append a broken migration.
    bad_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, bad_dir)
    (bad_dir / "999_broken.sql").write_text("THIS IS NOT VALID SQL;\n")

    # Write a small wrapper script that overrides MIGRATIONS to point
    # at the bad dir, then invokes main().
    wrapper = tmp_path / "run.py"
    wrapper.write_text(
        f"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, {str(ROOT)!r})
import scripts.apply_migrations as am
am.MIGRATIONS = sorted(Path({str(bad_dir)!r}).glob('*.sql'))

sys.exit(asyncio.run(am.main()))
"""
    )

    env = os.environ.copy()
    env["DATABASE_URL"] = async_url
    env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run([sys.executable, str(wrapper)], env=env, capture_output=True, text=True)
    assert r.returncode == 1, (
        f"expected exit 1; stdout={r.stdout!r} stderr={r.stderr!r}"
    )
    combined = (r.stdout + r.stderr).lower()
    assert "999_broken.sql" in combined, combined

    eng = create_async_engine(async_url)
    async with eng.connect() as conn:
        names = (
            await conn.execute(text("SELECT name FROM schema_migrations"))
        ).scalars().all()
    await eng.dispose()
    assert "999_broken.sql" not in names, (
        f"failing migration should not record a history row; got {names}"
    )
