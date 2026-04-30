#!/usr/bin/env python3
"""Apply every ext/db/migrations/*.sql file against DATABASE_URL in order.

Safe to re-run — migrations use IF EXISTS / IF NOT EXISTS throughout, and
the ``schema_migrations`` history table records every applied file with a
SHA-256 checksum.

Behavior:
    * The first migration (``000_schema_migrations.sql``) creates the
      history table itself; from then on every applied migration is
      tracked in it.
    * On each run we read the existing ``(name, checksum)`` rows. For
      every migration file we compute the SHA-256 of its current
      contents.
        - If name already recorded AND checksum matches: skip.
        - If name recorded but checksum DIFFERS: abort with a clear
          error naming the file and both checksums (drift detection).
        - Otherwise execute, then record the row.
    * Each migration runs in its own transaction (try / except). On
      failure we print the migration name + raw error and exit with
      code 1 — no further migrations run; the history row is NOT
      inserted, so the run is resumable after the operator fixes the
      underlying issue.

Env:
    DATABASE_URL   postgresql+asyncpg://... or postgresql://... (required)
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = sorted((ROOT / "ext" / "db" / "migrations").glob("*.sql"))


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _load_history(pg_conn) -> dict[str, str]:
    """Return ``{name: checksum}`` from schema_migrations.

    If the table does not exist yet (very first run, before
    ``000_schema_migrations.sql`` has been applied) returns ``{}``.
    """
    rows = await pg_conn.fetch(
        "SELECT to_regclass('public.schema_migrations') AS reg"
    )
    if not rows or rows[0]["reg"] is None:
        return {}
    rows = await pg_conn.fetch("SELECT name, checksum FROM schema_migrations")
    return {r["name"]: r["checksum"] for r in rows}


async def main() -> int:
    url = os.environ["DATABASE_URL"]
    if "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    print(f"applying {len(MIGRATIONS)} migration(s) to {url.split('@')[-1]}")
    eng = create_async_engine(url)
    try:
        # Get a single raw asyncpg connection for the whole session.
        # We deliberately do NOT wrap everything in one big tx — each
        # migration is its own atomic unit so a partial failure is
        # visible and resumable.
        async with eng.connect() as conn:
            raw = await conn.get_raw_connection()
            pg_conn = raw.driver_connection
            assert pg_conn is not None, "expected asyncpg driver connection"

            history = await _load_history(pg_conn)

            for m in MIGRATIONS:
                name = m.name
                cur_sum = _checksum(m)
                prev_sum = history.get(name)

                if prev_sum is not None and prev_sum == cur_sum:
                    print(f"  -> {name}  (already applied, checksum match — skipping)")
                    continue

                if prev_sum is not None and prev_sum != cur_sum:
                    print(
                        f"ERROR: migration {name} has drifted!\n"
                        f"  recorded checksum: {prev_sum}\n"
                        f"  current  checksum: {cur_sum}\n"
                        f"  Migrations are append-only. To change a past migration,"
                        f" write a new follow-up migration instead.",
                        file=sys.stderr,
                    )
                    return 1

                print(f"  -> {name}")
                # Per-migration transaction. asyncpg auto-wraps multi-
                # statement queries in a transaction when called via
                # ``execute()``, but to be explicit (and to make sure
                # the history INSERT only fires on success) we manage
                # one ourselves.
                try:
                    async with pg_conn.transaction():
                        await pg_conn.execute(m.read_text())
                        # Refresh schema_migrations existence after the
                        # bootstrap migration ran. From this point on,
                        # the history table always exists.
                        await pg_conn.execute(
                            "INSERT INTO schema_migrations (name, checksum) "
                            "VALUES ($1, $2) "
                            "ON CONFLICT (name) DO UPDATE SET "
                            "checksum=excluded.checksum, applied_at=now()",
                            name,
                            cur_sum,
                        )
                except Exception as e:  # pragma: no cover (covered by tests)
                    print(
                        f"ERROR applying {name}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    return 1
    finally:
        await eng.dispose()
    print("migrations OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
