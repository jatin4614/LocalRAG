#!/usr/bin/env python3
"""Create the bootstrap admin user if one does not already exist.

Env vars:
    DATABASE_URL     postgresql+asyncpg://user:pass@host/db
    ADMIN_EMAIL      admin email (required)
    ADMIN_PASSWORD   plaintext; hashed with Argon2id (required, >=12 chars)
"""
from __future__ import annotations

import asyncio
import os
import sys

from argon2 import PasswordHasher
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def main() -> int:
    url   = os.environ["DATABASE_URL"]
    email = os.environ["ADMIN_EMAIL"]
    pw    = os.environ["ADMIN_PASSWORD"]

    if len(pw) < 12:
        print("!! ADMIN_PASSWORD must be at least 12 chars", file=sys.stderr)
        return 2

    hasher = PasswordHasher()
    hashed = hasher.hash(pw)

    engine = create_async_engine(url)
    async with engine.begin() as conn:
        existing = (await conn.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": email},
        )).scalar()
        if existing is not None:
            print(f"admin {email} already present (id={existing}); leaving in place")
            await engine.dispose()
            return 0

        await conn.execute(
            text("""INSERT INTO users (email, password_hash, role)
                    VALUES (:email, :hash, 'admin')"""),
            {"email": email, "hash": hashed},
        )
        print(f"created admin {email}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
