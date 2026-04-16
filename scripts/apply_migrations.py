#!/usr/bin/env python3
"""Apply every ext/db/migrations/*.sql file against DATABASE_URL in order.

Safe to re-run — migrations use IF EXISTS / IF NOT EXISTS throughout.
Intended to run at upstream's FastAPI startup (after upstream's own schema init).

Env:
    DATABASE_URL   postgresql+asyncpg://... or postgresql://... (required)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = sorted((ROOT / "ext" / "db" / "migrations").glob("*.sql"))


async def main() -> int:
    url = os.environ["DATABASE_URL"]
    if "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    print(f"applying {len(MIGRATIONS)} migration(s) to {url.split('@')[-1]}")
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        raw = await conn.get_raw_connection()
        pg_conn = raw.driver_connection
        assert pg_conn is not None, "expected asyncpg driver connection"
        for m in MIGRATIONS:
            print(f"  -> {m.name}")
            await pg_conn.execute(m.read_text())
    await eng.dispose()
    print("migrations OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
