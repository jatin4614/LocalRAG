# Phase 2 — KB Management + RBAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute task-by-task. Steps use `- [ ]` checkboxes.

**Goal:** Ship the backend HTTP surface for hierarchical Knowledge Base management (KBs + subtags + documents placeholder + access grants), with strict per-user RBAC enforcement and a chat endpoint for per-session KB selection.

**Architecture:** A standalone FastAPI app in `ext/app.py` that mounts two routers (`kb_admin`, `kb_retrieval`), backed by async SQLAlchemy 2.0 sessions against the Phase 1 KB schema. Auth is a stub (`X-User-Id` + `X-User-Role` headers) until Phase 5 wires it into Open WebUI's real session layer. RBAC filtering happens at the service layer: every read passes through `get_allowed_kb_ids(user_id)` which joins `kb_access` across both direct user grants and group membership. The bootstrap admin seeded in Phase 1 is the only user with `role=admin` until groups/users are created in later phases.

**Tech Stack:** FastAPI 0.115, SQLAlchemy 2.0 async, asyncpg, Pydantic 2, Pydantic Settings, httpx (test client), pytest + pytest-asyncio + testcontainers + aiosqlite.

**Working directory:** `/home/vogic/LocalRAG/` (on branch `main`, tagged `phase-1-foundation`).

---

## Decisions (Phase 2 scope)

| # | Decision | Revise-by |
|---|----------|-----------|
| D15 | Auth stub reads `X-User-Id` (int) and `X-User-Role` (`admin`/`user`) headers. Temporary until Phase 5 hooks into Open WebUI's session cookie. | Phase 5 |
| D16 | Our FastAPI app mounts under `/api/kb/*` (admin) and `/api/chats/{chat_id}/kb_config` (retrieval side). Port 9100 internally. Caddy/Open WebUI routing comes in Phase 5. | Phase 5 |
| D17 | KB soft-delete: `knowledge_bases` gets a `deleted_at` column added in Phase 2's migration bump (002); soft-deleted KBs are hidden from all list/get paths but row preserved for audit. | Phase 6 |
| D18 | `kb_access` `access_type='write'` is defined but unused in Phase 2 — only `read` is enforced in retrieval. Write grants matter in Phase 4 (uploads). | Phase 4 |
| D19 | Phase 2 does NOT ingest or upload documents. `kb_documents` rows are created only by Phase 4 upload pipeline. Phase 2 tests create rows directly via the DB. | Phase 4 |

---

## File structure delivered by this phase

```
ext/
├── config.py                       T1
├── app.py                          T12
├── db/
│   ├── base.py                     (existing)
│   ├── session.py                  T2
│   ├── migrations/
│   │   ├── 001_create_kb_schema.sql (existing)
│   │   └── 002_soft_delete_kb.sql   T6 (new)
│   └── models/
│       ├── __init__.py             (existing, extended)
│       ├── kb.py                   (existing, extended with deleted_at)
│       ├── chat_ext.py             (existing)
│       └── compat.py               T3 — minimal User/Group/UserGroup/Chat
├── services/
│   ├── __init__.py                 (existing)
│   ├── auth.py                     T5
│   ├── rbac.py                     T4
│   └── kb_service.py               T6 / T7 / T8
└── routers/
    ├── __init__.py                 (existing)
    ├── kb_admin.py                 T9 / T10
    └── kb_retrieval.py             T11

tests/
├── conftest.py                     (existing)
├── integration/
│   ├── conftest.py                 T2 — shared fixtures (pg + engine)
│   ├── test_kb_migration.py        (existing)
│   ├── test_seed_admin.py          (existing)
│   ├── test_compose_up.py          (existing)
│   ├── test_kb_service.py          T6 / T7 / T8
│   ├── test_kb_admin_routes.py     T9 / T10
│   ├── test_kb_retrieval_routes.py T11
│   ├── test_kb_isolation.py        T13
│   ├── test_rbac.py                T14
│   └── test_chat_kb_config.py      T15
└── unit/
    ├── test_config.py              T1
    ├── test_rbac_logic.py          T4
    └── test_auth_dep.py            T5
```

---

## Task 1: Config loader (Pydantic Settings)

**Files:** Create `ext/config.py`, `tests/unit/test_config.py`.

- [ ] **Step 1: Write failing test**

`tests/unit/test_config.py`:

```python
import os
from ext.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setenv("REDIS_URL", "redis://r:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://q:6333")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    s = Settings()
    assert s.database_url == "postgresql+asyncpg://u:p@h/d"
    assert s.redis_url == "redis://r:6379/0"
    assert s.qdrant_url == "http://q:6333"
    assert s.session_secret == "x" * 32


def test_session_secret_min_length(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setenv("REDIS_URL", "redis://r:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://q:6333")
    monkeypatch.setenv("SESSION_SECRET", "short")  # <32 chars
    import pytest
    with pytest.raises(Exception):
        Settings()
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_config.py -v
```

- [ ] **Step 3: Write `ext/config.py`**

```python
"""Application settings loaded from environment."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, extra="ignore")

    database_url:   str = Field(..., alias="DATABASE_URL")
    redis_url:      str = Field(..., alias="REDIS_URL")
    qdrant_url:     str = Field(..., alias="QDRANT_URL")
    session_secret: str = Field(..., alias="SESSION_SECRET", min_length=32)


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_config.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add ext/config.py tests/unit/test_config.py
git commit -m "feat: pydantic settings loader with SESSION_SECRET length guard"
```

---

## Task 2: DB session factory + shared integration conftest

**Files:** Create `ext/db/session.py`, `tests/integration/conftest.py`.

- [ ] **Step 1: Write failing test**

`tests/unit/test_session_factory.py`:

```python
import pytest
from ext.db.session import make_engine, make_sessionmaker

@pytest.mark.asyncio
async def test_session_round_trip(monkeypatch):
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    SessionLocal = make_sessionmaker(engine)
    async with SessionLocal() as s:
        from sqlalchemy import text
        r = (await s.execute(text("SELECT 1"))).scalar()
        assert r == 1
    await engine.dispose()
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_session_factory.py -v
```

- [ ] **Step 3: Write `ext/db/session.py`**

```python
from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True, future=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session(sm: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with sm() as session:
        yield session
```

- [ ] **Step 4: Write shared integration conftest**

`tests/integration/conftest.py`:

```python
"""Shared fixtures for integration tests that need a real Postgres + migrated schema."""
from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

ROOT = Path(__file__).resolve().parents[2]
MIGRATION_001 = ROOT / "ext/db/migrations/001_create_kb_schema.sql"
MIGRATION_002 = ROOT / "ext/db/migrations/002_soft_delete_kb.sql"


async def _raw_exec(conn, sql: str) -> None:
    """Run multi-statement DDL using asyncpg's raw connection (bypasses prepared-statement limits)."""
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
        # Minimal upstream-style base tables
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
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(scope="function")
async def session(engine) -> AsyncSession:
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        yield s
```

- [ ] **Step 5: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_session_factory.py -v
```

Expected: 1 PASSED.

- [ ] **Step 6: Commit**

```bash
git add ext/db/session.py tests/integration/conftest.py tests/unit/test_session_factory.py
git commit -m "feat: async session factory + shared integration conftest"
```

---

## Task 3: Compat models — User, Group, UserGroup, Chat

**Files:** Create `ext/db/models/compat.py`; update `ext/db/models/__init__.py`.

- [ ] **Step 1: Write failing test**

`tests/unit/test_compat_models.py`:

```python
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from ext.db.base import Base
from ext.db.models.compat import User, Group, UserGroup, Chat


@pytest.mark.asyncio
async def test_compat_models_round_trip():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        u = User(id=1, email="a@x", password_hash="h", role="admin")
        g = Group(id=1, name="eng")
        s.add_all([u, g]); await s.flush()
        s.add(UserGroup(user_id=1, group_id=1)); await s.flush()
        s.add(Chat(id=1, user_id=1, title="hello")); await s.flush()
        await s.commit()
        rows = (await s.execute(select(User))).scalars().all()
        assert len(rows) == 1
        assert rows[0].role == "admin"
    await eng.dispose()
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_compat_models.py -v
```

- [ ] **Step 3: Write `ext/db/models/compat.py`**

```python
"""Minimal ORM models mirroring the upstream-compatible tables our KB FKs point at.

These exist so our code and tests can read/write users/groups/chats without
booting upstream Open WebUI. In production, the same tables are populated by
upstream's auth layer; we only read them here (plus chat.selected_kb_config).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class User(Base):
    __tablename__ = "users"

    id:            Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email:         Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role:          Mapped[str] = mapped_column(String, nullable=False, default="user")


class Group(Base):
    __tablename__ = "groups"

    id:   Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)


class UserGroup(Base):
    __tablename__ = "user_groups"

    user_id:  Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id",  ondelete="CASCADE"), primary_key=True)
    group_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)


class Chat(Base):
    __tablename__ = "chats"

    id:                 Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id:            Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title:              Mapped[Optional[str]] = mapped_column(String)
    created_at:         Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    selected_kb_config: Mapped[Optional[list]] = mapped_column(JSON)  # JSONB on PG, JSON on SQLite
```

- [ ] **Step 4: Update `ext/db/models/__init__.py`**

Replace contents with:

```python
from .kb import KnowledgeBase, KBSubtag, KBDocument, KBAccess
from .chat_ext import SelectedKBConfig, validate_selected_kb_config
from .compat import User, Group, UserGroup, Chat

__all__ = [
    "KnowledgeBase", "KBSubtag", "KBDocument", "KBAccess",
    "SelectedKBConfig", "validate_selected_kb_config",
    "User", "Group", "UserGroup", "Chat",
]
```

- [ ] **Step 5: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_compat_models.py tests/unit/test_kb_models.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add ext/db/models/compat.py ext/db/models/__init__.py tests/unit/test_compat_models.py
git commit -m "feat: compat ORM models (User, Group, UserGroup, Chat)"
```

---

## Task 4: RBAC helper — `get_allowed_kb_ids(user_id)`

**Files:** Create `ext/services/rbac.py`, `tests/integration/test_rbac_service.py`.

- [ ] **Step 1: Write failing test**

`tests/integration/test_rbac_service.py`:

```python
import pytest
from sqlalchemy import text
from ext.services.rbac import get_allowed_kb_ids


@pytest.mark.asyncio
async def test_rbac_direct_and_group_grants(session):
    # Fixture `session` is from tests/integration/conftest.py — connected to a migrated PG.
    # Seed: 2 users, 2 groups, 3 KBs, mixed grants.
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES
          (1, 'a@x', 'h', 'user'), (2, 'b@x', 'h', 'user');
        INSERT INTO groups (id, name) VALUES (1, 'eng'), (2, 'hr');
        INSERT INTO user_groups (user_id, group_id) VALUES (1, 1), (2, 2);
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES
          (10, 'EngKB', 1), (11, 'HrKB', 1), (12, 'DirectKB', 1);
        INSERT INTO kb_access (kb_id, user_id, group_id, access_type) VALUES
          (10, NULL, 1, 'read'),       -- eng group sees EngKB
          (11, NULL, 2, 'read'),       -- hr group sees HrKB
          (12, 1, NULL, 'read');       -- user 1 direct
    """))
    await session.commit()

    assert set(await get_allowed_kb_ids(session, user_id=1)) == {10, 12}
    assert set(await get_allowed_kb_ids(session, user_id=2)) == {11}
    # Unknown user → no access
    assert await get_allowed_kb_ids(session, user_id=999) == []


@pytest.mark.asyncio
async def test_rbac_admin_sees_everything(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES
          (1, 'root@x', 'h', 'admin');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES
          (10, 'A', 1), (11, 'B', 1), (12, 'C', 1);
    """))
    await session.commit()
    assert set(await get_allowed_kb_ids(session, user_id=1)) == {10, 11, 12}
```

- [ ] **Step 2: Run — FAIL (module missing)**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_rbac_service.py -v
```

- [ ] **Step 3: Write `ext/services/rbac.py`**

```python
"""RBAC: resolve which KB ids a user is allowed to read."""
from __future__ import annotations

from typing import List

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import KBAccess, KnowledgeBase, User, UserGroup


async def get_allowed_kb_ids(session: AsyncSession, *, user_id: int) -> List[int]:
    """Return list of kb_ids the given user can read.

    Admins see every non-deleted KB. Regular users see KBs matched by:
      - direct user grant in kb_access, OR
      - group grant in kb_access for a group they belong to.
    """
    user = (await session.execute(
        select(User).where(User.id == user_id)
    )).scalar_one_or_none()
    if user is None:
        return []

    if user.role == "admin":
        rows = (await session.execute(
            select(KnowledgeBase.id).where(_not_deleted())
        )).scalars().all()
        return list(rows)

    group_ids_stmt = select(UserGroup.group_id).where(UserGroup.user_id == user_id)

    rows = (await session.execute(
        select(KBAccess.kb_id).where(
            or_(
                KBAccess.user_id == user_id,
                KBAccess.group_id.in_(group_ids_stmt),
            )
        )
    )).scalars().all()
    return sorted(set(rows))


def _not_deleted():
    """Filter for non-soft-deleted KBs.

    Works whether the `deleted_at` column exists (Phase 2 migration 002) or not
    (fresh Phase-1 DBs). SQLAlchemy renders `deleted_at IS NULL` unconditionally;
    on Phase 1 DBs the migration test fixture applies 002 before running, so this
    is always safe in our test flow.
    """
    return KnowledgeBase.deleted_at.is_(None)
```

- [ ] **Step 4: Extend `ext/db/models/kb.py` — add `deleted_at` to `KnowledgeBase`**

Open `ext/db/models/kb.py`; in `class KnowledgeBase`, add after `created_at`:

```python
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
```

- [ ] **Step 5: Write migration 002**

Create `ext/db/migrations/002_soft_delete_kb.sql`:

```sql
-- 002_soft_delete_kb.sql — add soft-delete to knowledge_bases.
BEGIN;
ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_kb_not_deleted ON knowledge_bases(id) WHERE deleted_at IS NULL;
COMMIT;
```

- [ ] **Step 6: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_rbac_service.py -v
```

Expected: 2 PASSED. Also re-run full suite to confirm no regressions:

```bash
python -m pytest tests/unit tests/integration -v 2>&1 | tail -10
```

- [ ] **Step 7: Commit**

```bash
git add ext/services/rbac.py ext/db/models/kb.py ext/db/migrations/002_soft_delete_kb.sql tests/integration/test_rbac_service.py
git commit -m "feat: rbac helper + soft-delete migration (002)"
```

---

## Task 5: Auth dependency stub

**Files:** Create `ext/services/auth.py`, `tests/unit/test_auth_dep.py`.

- [ ] **Step 1: Write failing test**

`tests/unit/test_auth_dep.py`:

```python
import pytest
from fastapi import FastAPI, Depends, HTTPException
from fastapi.testclient import TestClient

from ext.services.auth import CurrentUser, get_current_user, require_admin


def test_missing_headers_returns_401():
    app = FastAPI()

    @app.get("/me")
    def me(u: CurrentUser = Depends(get_current_user)):
        return {"id": u.id}

    r = TestClient(app).get("/me")
    assert r.status_code == 401


def test_valid_headers_parsed():
    app = FastAPI()

    @app.get("/me")
    def me(u: CurrentUser = Depends(get_current_user)):
        return {"id": u.id, "role": u.role}

    r = TestClient(app).get("/me", headers={"X-User-Id": "42", "X-User-Role": "user"})
    assert r.status_code == 200
    assert r.json() == {"id": 42, "role": "user"}


def test_non_admin_cannot_access_admin_route():
    app = FastAPI()

    @app.get("/admin")
    def admin(u: CurrentUser = Depends(require_admin)):
        return {"ok": True}

    r = TestClient(app).get("/admin", headers={"X-User-Id": "1", "X-User-Role": "user"})
    assert r.status_code == 403


def test_admin_can_access_admin_route():
    app = FastAPI()

    @app.get("/admin")
    def admin(u: CurrentUser = Depends(require_admin)):
        return {"ok": True, "role": u.role}

    r = TestClient(app).get("/admin", headers={"X-User-Id": "1", "X-User-Role": "admin"})
    assert r.status_code == 200
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_auth_dep.py -v
```

- [ ] **Step 3: Write `ext/services/auth.py`**

```python
"""STUB auth layer.

Reads `X-User-Id` and `X-User-Role` headers and builds a CurrentUser.
Phase 5 replaces this with Open WebUI session cookie verification.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class CurrentUser:
    id: int
    role: str


def get_current_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> CurrentUser:
    if x_user_id is None or x_user_role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing auth headers")
    try:
        uid = int(x_user_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Id") from e
    if x_user_role not in {"admin", "user", "pending"}:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Role")
    return CurrentUser(id=uid, role=x_user_role)


def require_admin(user: CurrentUser = None) -> CurrentUser:
    # Dependency-chain form: FastAPI re-resolves via get_current_user.
    from fastapi import Depends
    return _require_admin_inner(Depends(get_current_user))


def _require_admin_inner(user: CurrentUser) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin only")
    return user


# Simpler form used directly as a dependency: FastAPI will resolve get_current_user.
async def require_admin_dep(user: CurrentUser = None) -> CurrentUser:
    raise NotImplementedError  # we use the functional form below
```

Wait — the above has a subtle FastAPI dependency-graph bug. Replace the file with this corrected version:

```python
"""STUB auth layer. Reads X-User-Id + X-User-Role headers.

Phase 5 will replace this with Open WebUI session cookie verification.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status


@dataclass(frozen=True)
class CurrentUser:
    id: int
    role: str


def get_current_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> CurrentUser:
    if x_user_id is None or x_user_role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing auth headers")
    try:
        uid = int(x_user_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Id") from e
    if x_user_role not in {"admin", "user", "pending"}:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Role")
    return CurrentUser(id=uid, role=x_user_role)


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin only")
    return user
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_auth_dep.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add ext/services/auth.py tests/unit/test_auth_dep.py
git commit -m "feat: stub auth dependency (X-User-Id / X-User-Role headers)"
```

---

## Task 6: KB service — create + list + get + soft-delete

**Files:** Create `ext/services/kb_service.py`, `tests/integration/test_kb_service.py`.

- [ ] **Step 1: Write failing test**

`tests/integration/test_kb_service.py`:

```python
import pytest
from sqlalchemy import text
from ext.services import kb_service


@pytest.mark.asyncio
async def test_create_and_get_kb(session):
    await session.execute(text("INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin')"))
    await session.commit()
    kb = await kb_service.create_kb(session, name="Engineering", description="eng docs", admin_id=1)
    await session.commit()
    got = await kb_service.get_kb(session, kb_id=kb.id)
    assert got.name == "Engineering"
    assert got.description == "eng docs"
    assert got.deleted_at is None


@pytest.mark.asyncio
async def test_list_kbs_excludes_soft_deleted(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'Alive', 1), (2, 'Dead', 1);
        UPDATE knowledge_bases SET deleted_at = now() WHERE id = 2;
    """))
    await session.commit()
    kbs = await kb_service.list_kbs(session, kb_ids=[1, 2])
    assert {k.name for k in kbs} == {"Alive"}


@pytest.mark.asyncio
async def test_soft_delete_kb(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES (5, 'K', 1);
    """))
    await session.commit()
    await kb_service.soft_delete_kb(session, kb_id=5)
    await session.commit()
    got = await kb_service.get_kb(session, kb_id=5)
    assert got is None  # hidden from get


@pytest.mark.asyncio
async def test_duplicate_kb_name_rejected(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'Dup', 1);
    """))
    await session.commit()
    with pytest.raises(Exception):
        await kb_service.create_kb(session, name="Dup", description=None, admin_id=1)
        await session.commit()
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_service.py -v
```

- [ ] **Step 3: Write `ext/services/kb_service.py`**

```python
"""Knowledge Base service layer — CRUD operations with RBAC-agnostic queries.

Callers must filter by `get_allowed_kb_ids(user_id)` BEFORE calling these methods.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import KnowledgeBase


async def create_kb(
    session: AsyncSession, *, name: str, description: Optional[str], admin_id: int
) -> KnowledgeBase:
    kb = KnowledgeBase(name=name, description=description, admin_id=admin_id)
    session.add(kb)
    await session.flush()
    return kb


async def get_kb(session: AsyncSession, *, kb_id: int) -> Optional[KnowledgeBase]:
    return (await session.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.deleted_at.is_(None),
        )
    )).scalar_one_or_none()


async def list_kbs(
    session: AsyncSession, *, kb_ids: Optional[Iterable[int]] = None
) -> List[KnowledgeBase]:
    stmt = select(KnowledgeBase).where(KnowledgeBase.deleted_at.is_(None))
    if kb_ids is not None:
        ids = list(kb_ids)
        if not ids:
            return []
        stmt = stmt.where(KnowledgeBase.id.in_(ids))
    return list((await session.execute(stmt)).scalars().all())


async def update_kb(
    session: AsyncSession, *, kb_id: int, name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[KnowledgeBase]:
    kb = await get_kb(session, kb_id=kb_id)
    if kb is None:
        return None
    if name is not None:
        kb.name = name
    if description is not None:
        kb.description = description
    await session.flush()
    return kb


async def soft_delete_kb(session: AsyncSession, *, kb_id: int) -> bool:
    r = await session.execute(
        update(KnowledgeBase)
        .where(KnowledgeBase.id == kb_id, KnowledgeBase.deleted_at.is_(None))
        .values(deleted_at=datetime.now(timezone.utc))
    )
    return r.rowcount > 0
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_service.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add ext/services/kb_service.py tests/integration/test_kb_service.py
git commit -m "feat: kb_service CRUD (create, get, list, update, soft-delete)"
```

---

## Task 7: KB service — subtag CRUD

**Files:** Extend `ext/services/kb_service.py`; extend `tests/integration/test_kb_service.py`.

- [ ] **Step 1: Write failing test** — append to `tests/integration/test_kb_service.py`:

```python
@pytest.mark.asyncio
async def test_create_and_list_subtags(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1);
    """))
    await session.commit()
    s1 = await kb_service.create_subtag(session, kb_id=1, name="OFC", description=None)
    s2 = await kb_service.create_subtag(session, kb_id=1, name="Roadmap", description="q2")
    await session.commit()
    subs = await kb_service.list_subtags(session, kb_id=1)
    assert {s.name for s in subs} == {"OFC", "Roadmap"}


@pytest.mark.asyncio
async def test_duplicate_subtag_within_kb_rejected(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1);
        INSERT INTO kb_subtags (kb_id, name) VALUES (1, 'X');
    """))
    await session.commit()
    with pytest.raises(Exception):
        await kb_service.create_subtag(session, kb_id=1, name="X", description=None)
        await session.commit()


@pytest.mark.asyncio
async def test_delete_subtag(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1);
        INSERT INTO kb_subtags (id, kb_id, name) VALUES (10, 1, 'Del');
    """))
    await session.commit()
    ok = await kb_service.delete_subtag(session, kb_id=1, subtag_id=10)
    await session.commit()
    assert ok is True
    subs = await kb_service.list_subtags(session, kb_id=1)
    assert subs == []
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_service.py -v
```

- [ ] **Step 3: Extend `ext/services/kb_service.py`** — append:

```python
from ..db.models import KBSubtag
from sqlalchemy import delete


async def create_subtag(
    session: AsyncSession, *, kb_id: int, name: str, description: Optional[str] = None,
) -> KBSubtag:
    sub = KBSubtag(kb_id=kb_id, name=name, description=description)
    session.add(sub)
    await session.flush()
    return sub


async def list_subtags(session: AsyncSession, *, kb_id: int) -> List[KBSubtag]:
    return list((await session.execute(
        select(KBSubtag).where(KBSubtag.kb_id == kb_id).order_by(KBSubtag.id)
    )).scalars().all())


async def delete_subtag(session: AsyncSession, *, kb_id: int, subtag_id: int) -> bool:
    r = await session.execute(
        delete(KBSubtag).where(KBSubtag.id == subtag_id, KBSubtag.kb_id == kb_id)
    )
    return r.rowcount > 0
```

- [ ] **Step 4: Run — PASS (7 total now in this file)**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_service.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ext/services/kb_service.py tests/integration/test_kb_service.py
git commit -m "feat: kb_service subtag CRUD"
```

---

## Task 8: KB service — access grants

**Files:** Extend `ext/services/kb_service.py`; extend test file.

- [ ] **Step 1: Write failing test** — append to `tests/integration/test_kb_service.py`:

```python
@pytest.mark.asyncio
async def test_grant_user_and_group_access(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin'), (2, 'b@x', 'h', 'user');
        INSERT INTO groups (id, name) VALUES (1, 'eng');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1);
    """))
    await session.commit()

    g1 = await kb_service.grant_access(session, kb_id=1, user_id=2, group_id=None)
    g2 = await kb_service.grant_access(session, kb_id=1, user_id=None, group_id=1)
    await session.commit()
    grants = await kb_service.list_access(session, kb_id=1)
    assert len(grants) == 2
    assert {g.user_id for g in grants if g.user_id is not None} == {2}
    assert {g.group_id for g in grants if g.group_id is not None} == {1}


@pytest.mark.asyncio
async def test_grant_requires_exactly_one_of_user_or_group(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1);
    """))
    await session.commit()
    with pytest.raises(ValueError):
        await kb_service.grant_access(session, kb_id=1, user_id=None, group_id=None)
    with pytest.raises(ValueError):
        await kb_service.grant_access(session, kb_id=1, user_id=1, group_id=1)


@pytest.mark.asyncio
async def test_revoke_access(session):
    await session.execute(text("""
        INSERT INTO users (id, email, password_hash, role) VALUES (1, 'a@x', 'h', 'admin');
        INSERT INTO knowledge_bases (id, name, admin_id) VALUES (1, 'K', 1);
        INSERT INTO kb_access (id, kb_id, user_id, access_type) VALUES (100, 1, 1, 'read');
    """))
    await session.commit()
    ok = await kb_service.revoke_access(session, grant_id=100)
    await session.commit()
    assert ok is True
    grants = await kb_service.list_access(session, kb_id=1)
    assert grants == []
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_service.py -v
```

- [ ] **Step 3: Extend `ext/services/kb_service.py`** — append:

```python
from ..db.models import KBAccess


async def grant_access(
    session: AsyncSession, *, kb_id: int, user_id: Optional[int],
    group_id: Optional[int], access_type: str = "read",
) -> KBAccess:
    """Grant read/write on a KB to either a user or a group (never both, never neither)."""
    # KBAccess.__init__ enforces the XOR too; we pre-check for a cleaner ValueError path.
    if (user_id is None) == (group_id is None):
        raise ValueError("grant_access requires exactly one of user_id or group_id")
    grant = KBAccess(kb_id=kb_id, user_id=user_id, group_id=group_id, access_type=access_type)
    session.add(grant)
    await session.flush()
    return grant


async def list_access(session: AsyncSession, *, kb_id: int) -> List[KBAccess]:
    return list((await session.execute(
        select(KBAccess).where(KBAccess.kb_id == kb_id).order_by(KBAccess.id)
    )).scalars().all())


async def revoke_access(session: AsyncSession, *, grant_id: int) -> bool:
    r = await session.execute(delete(KBAccess).where(KBAccess.id == grant_id))
    return r.rowcount > 0
```

- [ ] **Step 4: Run — PASS (10 tests now)**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_service.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ext/services/kb_service.py tests/integration/test_kb_service.py
git commit -m "feat: kb_service access-grant API (grant, list, revoke)"
```

---

## Task 9: KB admin router — KB endpoints

**Files:** Create `ext/routers/kb_admin.py`, `tests/integration/test_kb_admin_routes.py`.

This task wires the service layer to HTTP. We won't include subtag/access routes yet — those go in Task 10.

- [ ] **Step 1: Write failing test**

`tests/integration/test_kb_admin_routes.py`:

```python
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.kb_admin import router as kb_admin_router, set_sessionmaker


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    set_sessionmaker(SessionLocal)
    app = FastAPI()
    app.include_router(kb_admin_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


ADMIN = {"X-User-Id": "1", "X-User-Role": "admin"}
USER  = {"X-User-Id": "2", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed_admin(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("""
            INSERT INTO users (id, email, password_hash, role) VALUES
              (1, 'admin@x', 'h', 'admin'), (2, 'user@x', 'h', 'user');
        """))
        await s.commit()


@pytest.mark.asyncio
async def test_non_admin_cannot_create_kb(client):
    r = await client.post("/api/kb", headers=USER, json={"name": "X", "description": "no"})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_create_and_list_kb(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Eng", "description": "d"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "Eng"
    assert body["id"] > 0

    r2 = await client.get("/api/kb", headers=ADMIN)
    assert r2.status_code == 200
    assert any(kb["name"] == "Eng" for kb in r2.json())


@pytest.mark.asyncio
async def test_get_kb_404(client):
    r = await client.get("/api/kb/99999", headers=ADMIN)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_patch_kb(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Old"})
    kb_id = r.json()["id"]
    r = await client.patch(f"/api/kb/{kb_id}", headers=ADMIN, json={"name": "New"})
    assert r.status_code == 200
    assert r.json()["name"] == "New"


@pytest.mark.asyncio
async def test_delete_kb(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "Doomed"})
    kb_id = r.json()["id"]
    r = await client.delete(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 204
    r = await client.get(f"/api/kb/{kb_id}", headers=ADMIN)
    assert r.status_code == 404
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_admin_routes.py -v
```

- [ ] **Step 3: Write `ext/routers/kb_admin.py`**

```python
"""HTTP admin routes for KB CRUD. Admin-only."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..services import kb_service
from ..services.auth import CurrentUser, require_admin


router = APIRouter(prefix="/api/kb", tags=["kb-admin"])

# Sessionmaker injected by application at startup.
_SESSIONMAKER: async_sessionmaker[AsyncSession] | None = None


def set_sessionmaker(sm: async_sessionmaker[AsyncSession]) -> None:
    """Dependency-injection hook — call once in app startup."""
    global _SESSIONMAKER
    _SESSIONMAKER = sm


async def _get_session() -> AsyncSession:
    if _SESSIONMAKER is None:
        raise RuntimeError("sessionmaker not configured; call set_sessionmaker at app startup")
    async with _SESSIONMAKER() as s:
        yield s


class KBIn(BaseModel):
    name: str
    description: Optional[str] = None


class KBPatch(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class KBOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    admin_id: int


def _to_out(kb) -> KBOut:
    return KBOut(id=kb.id, name=kb.name, description=kb.description, admin_id=kb.admin_id)


@router.post("", response_model=KBOut, status_code=status.HTTP_201_CREATED)
async def create_kb(
    body: KBIn,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    try:
        kb = await kb_service.create_kb(session, name=body.name, description=body.description, admin_id=user.id)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(e)) from e
    return _to_out(kb)


@router.get("", response_model=list[KBOut])
async def list_kbs(
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    kbs = await kb_service.list_kbs(session)
    return [_to_out(k) for k in kbs]


@router.get("/{kb_id}", response_model=KBOut)
async def get_kb(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    kb = await kb_service.get_kb(session, kb_id=kb_id)
    if kb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    return _to_out(kb)


@router.patch("/{kb_id}", response_model=KBOut)
async def update_kb(
    kb_id: int,
    body: KBPatch,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    kb = await kb_service.update_kb(session, kb_id=kb_id, name=body.name, description=body.description)
    if kb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    await session.commit()
    return _to_out(kb)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
) -> Response:
    ok = await kb_service.soft_delete_kb(session, kb_id=kb_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

- [ ] **Step 4: Run — PASS (5 tests)**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_admin_routes.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ext/routers/kb_admin.py tests/integration/test_kb_admin_routes.py
git commit -m "feat: kb_admin router (KB CRUD endpoints)"
```

---

## Task 10: KB admin router — subtag + access endpoints

**Files:** Extend `ext/routers/kb_admin.py`; extend test file.

- [ ] **Step 1: Write failing test** — append:

```python
@pytest.mark.asyncio
async def test_subtag_crud(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "K"})
    kb_id = r.json()["id"]

    r = await client.post(f"/api/kb/{kb_id}/subtags", headers=ADMIN, json={"name": "OFC"})
    assert r.status_code == 201, r.text
    sub_id = r.json()["id"]

    r = await client.get(f"/api/kb/{kb_id}/subtags", headers=ADMIN)
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = await client.delete(f"/api/kb/{kb_id}/subtags/{sub_id}", headers=ADMIN)
    assert r.status_code == 204


@pytest.mark.asyncio
async def test_access_grant_and_list(client, engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO groups (id, name) VALUES (1, 'eng')"))
        await s.commit()

    r = await client.post("/api/kb", headers=ADMIN, json={"name": "K"})
    kb_id = r.json()["id"]

    r = await client.post(f"/api/kb/{kb_id}/access", headers=ADMIN,
                          json={"group_id": 1, "access_type": "read"})
    assert r.status_code == 201, r.text
    grant_id = r.json()["id"]

    r = await client.get(f"/api/kb/{kb_id}/access", headers=ADMIN)
    assert r.status_code == 200
    assert r.json()[0]["group_id"] == 1

    r = await client.delete(f"/api/kb/{kb_id}/access/{grant_id}", headers=ADMIN)
    assert r.status_code == 204


@pytest.mark.asyncio
async def test_access_grant_requires_exactly_one(client):
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "K"})
    kb_id = r.json()["id"]
    r = await client.post(f"/api/kb/{kb_id}/access", headers=ADMIN,
                          json={"user_id": None, "group_id": None, "access_type": "read"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_admin_routes.py -v
```

- [ ] **Step 3: Extend `ext/routers/kb_admin.py`** — append:

```python
class SubtagIn(BaseModel):
    name: str
    description: Optional[str] = None


class SubtagOut(BaseModel):
    id: int
    kb_id: int
    name: str
    description: Optional[str]


@router.post("/{kb_id}/subtags", response_model=SubtagOut, status_code=status.HTTP_201_CREATED)
async def create_subtag(
    kb_id: int, body: SubtagIn,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    if await kb_service.get_kb(session, kb_id=kb_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    try:
        sub = await kb_service.create_subtag(session, kb_id=kb_id, name=body.name, description=body.description)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(e)) from e
    return SubtagOut(id=sub.id, kb_id=sub.kb_id, name=sub.name, description=sub.description)


@router.get("/{kb_id}/subtags", response_model=list[SubtagOut])
async def list_subtags(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    subs = await kb_service.list_subtags(session, kb_id=kb_id)
    return [SubtagOut(id=s.id, kb_id=s.kb_id, name=s.name, description=s.description) for s in subs]


@router.delete("/{kb_id}/subtags/{subtag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_subtag(
    kb_id: int, subtag_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    ok = await kb_service.delete_subtag(session, kb_id=kb_id, subtag_id=subtag_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="subtag not found")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class AccessIn(BaseModel):
    user_id: Optional[int] = None
    group_id: Optional[int] = None
    access_type: str = "read"


class AccessOut(BaseModel):
    id: int
    kb_id: int
    user_id: Optional[int]
    group_id: Optional[int]
    access_type: str


@router.post("/{kb_id}/access", response_model=AccessOut, status_code=status.HTTP_201_CREATED)
async def grant_access(
    kb_id: int, body: AccessIn,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    if await kb_service.get_kb(session, kb_id=kb_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    try:
        g = await kb_service.grant_access(session, kb_id=kb_id, user_id=body.user_id,
                                          group_id=body.group_id, access_type=body.access_type)
        await session.commit()
    except ValueError as e:
        await session.rollback()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(e)) from e
    return AccessOut(id=g.id, kb_id=g.kb_id, user_id=g.user_id, group_id=g.group_id, access_type=g.access_type)


@router.get("/{kb_id}/access", response_model=list[AccessOut])
async def list_access(
    kb_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    grants = await kb_service.list_access(session, kb_id=kb_id)
    return [AccessOut(id=g.id, kb_id=g.kb_id, user_id=g.user_id, group_id=g.group_id, access_type=g.access_type) for g in grants]


@router.delete("/{kb_id}/access/{grant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_access(
    kb_id: int, grant_id: int,
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    ok = await kb_service.revoke_access(session, grant_id=grant_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="grant not found")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

- [ ] **Step 4: Run — PASS (8 tests in this file)**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_admin_routes.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ext/routers/kb_admin.py tests/integration/test_kb_admin_routes.py
git commit -m "feat: kb_admin router — subtag + access endpoints"
```

---

## Task 11: KB retrieval router — available KBs + chat config

**Files:** Create `ext/routers/kb_retrieval.py`, `tests/integration/test_kb_retrieval_routes.py`.

- [ ] **Step 1: Write failing test**

`tests/integration/test_kb_retrieval_routes.py`:

```python
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.kb_retrieval import router as kb_retr_router, set_sessionmaker


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    set_sessionmaker(SessionLocal)
    app = FastAPI()
    app.include_router(kb_retr_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


USER_1 = {"X-User-Id": "1", "X-User-Role": "user"}
USER_2 = {"X-User-Id": "2", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("""
            INSERT INTO users (id, email, password_hash, role) VALUES
              (1, 'a@x', 'h', 'user'), (2, 'b@x', 'h', 'user'), (9, 'admin@x', 'h', 'admin');
            INSERT INTO groups (id, name) VALUES (1, 'eng'), (2, 'hr');
            INSERT INTO user_groups (user_id, group_id) VALUES (1, 1), (2, 2);
            INSERT INTO knowledge_bases (id, name, admin_id) VALUES
              (10, 'Eng', 9), (11, 'HR', 9), (12, 'Secret', 9);
            INSERT INTO kb_access (kb_id, group_id, access_type) VALUES
              (10, 1, 'read'), (11, 2, 'read');
            INSERT INTO chats (id, user_id) VALUES (100, 1), (200, 2);
        """))
        await s.commit()


@pytest.mark.asyncio
async def test_available_returns_only_allowed_kbs(client):
    r = await client.get("/api/kb/available", headers=USER_1)
    assert r.status_code == 200
    names = {kb["name"] for kb in r.json()}
    assert names == {"Eng"}

    r = await client.get("/api/kb/available", headers=USER_2)
    assert {kb["name"] for kb in r.json()} == {"HR"}


@pytest.mark.asyncio
async def test_chat_kb_config_set_and_get(client):
    r = await client.put("/api/chats/100/kb_config", headers=USER_1,
                         json={"config": [{"kb_id": 10, "subtag_ids": []}]})
    assert r.status_code == 200, r.text
    assert r.json()["config"] == [{"kb_id": 10, "subtag_ids": []}]

    r = await client.get("/api/chats/100/kb_config", headers=USER_1)
    assert r.status_code == 200
    assert r.json()["config"] == [{"kb_id": 10, "subtag_ids": []}]


@pytest.mark.asyncio
async def test_chat_kb_config_rejects_unauthorized_kb(client):
    r = await client.put("/api/chats/100/kb_config", headers=USER_1,
                         json={"config": [{"kb_id": 11, "subtag_ids": []}]})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_chat_kb_config_rejects_other_users_chat(client):
    r = await client.put("/api/chats/200/kb_config", headers=USER_1,
                         json={"config": [{"kb_id": 10}]})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_chat_kb_config_rejects_bad_shape(client):
    r = await client.put("/api/chats/100/kb_config", headers=USER_1,
                         json={"config": [{"kb_id": "not-int"}]})
    assert r.status_code == 400
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_retrieval_routes.py -v
```

- [ ] **Step 3: Write `ext/routers/kb_retrieval.py`**

```python
"""User-facing read routes: list available KBs, set per-chat KB selection."""
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..db.models import Chat, validate_selected_kb_config
from ..services import kb_service
from ..services.auth import CurrentUser, get_current_user
from ..services.rbac import get_allowed_kb_ids


router = APIRouter(tags=["kb-retrieval"])

_SESSIONMAKER: async_sessionmaker[AsyncSession] | None = None


def set_sessionmaker(sm: async_sessionmaker[AsyncSession]) -> None:
    global _SESSIONMAKER
    _SESSIONMAKER = sm


async def _get_session() -> AsyncSession:
    if _SESSIONMAKER is None:
        raise RuntimeError("sessionmaker not configured")
    async with _SESSIONMAKER() as s:
        yield s


class KBAvailable(BaseModel):
    id: int
    name: str
    description: Optional[str]


@router.get("/api/kb/available", response_model=list[KBAvailable])
async def available_kbs(
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
):
    allowed = await get_allowed_kb_ids(session, user_id=user.id)
    kbs = await kb_service.list_kbs(session, kb_ids=allowed)
    return [KBAvailable(id=k.id, name=k.name, description=k.description) for k in kbs]


class ChatKBConfig(BaseModel):
    config: Optional[List[Any]] = None  # validated per selected_kb_config shape


@router.put("/api/chats/{chat_id}/kb_config", response_model=ChatKBConfig)
async def set_chat_kb_config(
    chat_id: int, body: ChatKBConfig,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
):
    # Only the chat owner can mutate config.
    chat = (await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id)
    )).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")

    try:
        parsed = validate_selected_kb_config(body.config)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    if parsed:
        allowed = set(await get_allowed_kb_ids(session, user_id=user.id))
        for entry in parsed:
            if entry.kb_id not in allowed:
                raise HTTPException(status.HTTP_403_FORBIDDEN,
                                    detail=f"no access to kb_id={entry.kb_id}")

    # Store in the JSONB column.
    chat.selected_kb_config = body.config
    await session.commit()
    return ChatKBConfig(config=body.config)


@router.get("/api/chats/{chat_id}/kb_config", response_model=ChatKBConfig)
async def get_chat_kb_config(
    chat_id: int,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
):
    chat = (await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id)
    )).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")
    return ChatKBConfig(config=chat.selected_kb_config)
```

- [ ] **Step 4: Run — PASS (5 tests)**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_retrieval_routes.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ext/routers/kb_retrieval.py tests/integration/test_kb_retrieval_routes.py
git commit -m "feat: kb_retrieval router (available KBs + chat kb_config)"
```

---

## Task 12: FastAPI app wiring

**Files:** Create `ext/app.py`, `tests/integration/test_app_wiring.py`.

- [ ] **Step 1: Write failing test**

`tests/integration/test_app_wiring.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_healthz(engine, monkeypatch):
    url = str(engine.url).replace("+asyncpg", "+asyncpg")  # normalize
    monkeypatch.setenv("DATABASE_URL", str(engine.url))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)

    from ext.app import build_app
    app = build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_app_mounts_kb_admin_and_retrieval(engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", str(engine.url))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from ext.app import build_app
    app = build_app()
    paths = {r.path for r in app.routes}
    assert "/api/kb" in paths
    assert "/api/kb/available" in paths
    assert "/api/chats/{chat_id}/kb_config" in paths
    assert "/healthz" in paths
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_app_wiring.py -v
```

- [ ] **Step 3: Write `ext/app.py`**

```python
"""FastAPI application entry point for the KB management + retrieval API."""
from __future__ import annotations

from fastapi import FastAPI

from .config import get_settings
from .db.session import make_engine, make_sessionmaker
from .routers import kb_admin, kb_retrieval


def build_app() -> FastAPI:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    SessionLocal = make_sessionmaker(engine)

    kb_admin.set_sessionmaker(SessionLocal)
    kb_retrieval.set_sessionmaker(SessionLocal)

    app = FastAPI(title="orgchat-kb", version="0.2.0")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    app.include_router(kb_admin.router)
    app.include_router(kb_retrieval.router)
    return app


app = None  # populated by uvicorn entrypoint; tests call build_app() directly
```

- [ ] **Step 4: Clear settings cache between tests**

`get_settings` uses `@lru_cache`. Tests monkeypatch env vars; cached Settings would be stale. Patch `ext/config.py` to allow cache reset, OR clear it at build time.

Edit `ext/config.py` — change the bottom to:

```python
@lru_cache
def _settings_cached() -> Settings:
    return Settings()


def get_settings() -> Settings:
    return _settings_cached()


def clear_settings_cache() -> None:
    _settings_cached.cache_clear()
```

And in `ext/app.py` `build_app` — call `clear_settings_cache()` before loading:

```python
from .config import clear_settings_cache, get_settings

def build_app() -> FastAPI:
    clear_settings_cache()
    settings = get_settings()
    # ...
```

- [ ] **Step 5: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_app_wiring.py -v
```

Expected: 2 PASSED. Also re-run the config unit test to confirm no regression.

- [ ] **Step 6: Commit**

```bash
git add ext/app.py ext/config.py tests/integration/test_app_wiring.py
git commit -m "feat: FastAPI app wiring (kb_admin + kb_retrieval + /healthz)"
```

---

## Task 13: Integration test — KB isolation (Phase-6 gate precursor)

**Files:** Create `tests/integration/test_kb_isolation.py`.

- [ ] **Step 1: Write the test**

`tests/integration/test_kb_isolation.py`:

```python
"""Cross-user isolation: user A's KB content is never visible to user B."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.kb_admin import router as admin_router, set_sessionmaker as set_admin_sm
from ext.routers.kb_retrieval import router as retr_router, set_sessionmaker as set_retr_sm


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    set_admin_sm(SessionLocal)
    set_retr_sm(SessionLocal)
    app = FastAPI()
    app.include_router(admin_router)
    app.include_router(retr_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


ADMIN = {"X-User-Id": "9", "X-User-Role": "admin"}
ALICE = {"X-User-Id": "1", "X-User-Role": "user"}
BOB   = {"X-User-Id": "2", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("""
            INSERT INTO users (id, email, password_hash, role) VALUES
              (9, 'admin@x', 'h', 'admin'),
              (1, 'alice@x', 'h', 'user'),
              (2, 'bob@x',   'h', 'user');
            INSERT INTO groups (id, name) VALUES (100, 'alice-group'), (200, 'bob-group');
            INSERT INTO user_groups (user_id, group_id) VALUES (1, 100), (2, 200);
        """))
        await s.commit()


@pytest.mark.asyncio
async def test_user_cannot_see_other_users_kb(client):
    # Admin creates "AliceKB" for alice-group.
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "AliceKB"})
    alice_kb_id = r.json()["id"]
    await client.post(f"/api/kb/{alice_kb_id}/access", headers=ADMIN,
                      json={"group_id": 100, "access_type": "read"})

    # Admin creates "BobKB" for bob-group.
    r = await client.post("/api/kb", headers=ADMIN, json={"name": "BobKB"})
    bob_kb_id = r.json()["id"]
    await client.post(f"/api/kb/{bob_kb_id}/access", headers=ADMIN,
                      json={"group_id": 200, "access_type": "read"})

    # Alice's /available only shows AliceKB.
    r = await client.get("/api/kb/available", headers=ALICE)
    names = {kb["name"] for kb in r.json()}
    assert names == {"AliceKB"}, f"leak: alice sees {names}"

    # Bob's /available only shows BobKB.
    r = await client.get("/api/kb/available", headers=BOB)
    names = {kb["name"] for kb in r.json()}
    assert names == {"BobKB"}, f"leak: bob sees {names}"


@pytest.mark.asyncio
async def test_user_cannot_admin_kb(client):
    r = await client.post("/api/kb", headers=ALICE, json={"name": "Sneak"})
    assert r.status_code == 403

    r = await client.get("/api/kb", headers=ALICE)
    assert r.status_code == 403

    r = await client.delete("/api/kb/1", headers=ALICE)
    assert r.status_code == 403
```

- [ ] **Step 2: Run — expect PASS (tests the existing behavior — this is a gate test)**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_isolation.py -v
```

Expected: 2 PASSED. If any fails, a security regression exists — halt and investigate.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_kb_isolation.py
git commit -m "test: cross-user KB isolation gate (Phase 6 precursor)"
```

---

## Task 14: Integration test — RBAC 403 on unauthorized chat KB config

**Files:** Already covered in `test_kb_retrieval_routes.py::test_chat_kb_config_rejects_unauthorized_kb` (Task 11). This task is a no-op confirmation + commit of any extra combinations.

- [ ] **Step 1: Write extended RBAC test**

`tests/integration/test_rbac_routes.py`:

```python
"""Edge cases for RBAC enforcement at the route layer."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.kb_admin import router as admin_router, set_sessionmaker as set_admin_sm
from ext.routers.kb_retrieval import router as retr_router, set_sessionmaker as set_retr_sm


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    set_admin_sm(SessionLocal)
    set_retr_sm(SessionLocal)
    app = FastAPI()
    app.include_router(admin_router)
    app.include_router(retr_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("""
            INSERT INTO users (id, email, password_hash, role) VALUES
              (1, 'u@x', 'h', 'user'), (9, 'admin@x', 'h', 'admin');
            INSERT INTO knowledge_bases (id, name, admin_id) VALUES (50, 'Private', 9);
            INSERT INTO chats (id, user_id) VALUES (500, 1);
        """))
        await s.commit()


@pytest.mark.asyncio
async def test_user_without_access_403_on_chat_config(client):
    r = await client.put("/api/chats/500/kb_config",
                         headers={"X-User-Id": "1", "X-User-Role": "user"},
                         json={"config": [{"kb_id": 50, "subtag_ids": []}]})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_request_401(client):
    r = await client.get("/api/kb/available")
    assert r.status_code == 401

    r = await client.get("/api/kb", headers={"X-User-Role": "admin"})  # missing X-User-Id
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_pending_role_rejected_from_admin_paths(client):
    r = await client.post("/api/kb",
                          headers={"X-User-Id": "1", "X-User-Role": "pending"},
                          json={"name": "X"})
    assert r.status_code == 403
```

- [ ] **Step 2: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_rbac_routes.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_rbac_routes.py
git commit -m "test: RBAC route-level enforcement (401/403/pending)"
```

---

## Task 15: Full-suite regression + Phase 2 tag

- [ ] **Step 1: Run all tests**

```bash
source .venv/bin/activate
python -m pytest tests/unit -v 2>&1 | tail -5
SKIP_GPU_SMOKE=1 python -m pytest tests/integration -v 2>&1 | tail -20
```

Expected:
- Unit: all previous + config (2) + session (1) + compat (1) + auth_dep (4) = ~41 passed.
- Integration: kb_migration (1) + seed_admin (1) + compose_up (3 pass + 1 skip) + rbac_service (2) + kb_service (10) + kb_admin_routes (8) + kb_retrieval_routes (5) + app_wiring (2) + kb_isolation (2) + rbac_routes (3) = ~37 passed + 1 skipped.

- [ ] **Step 2: Lint**

```bash
ruff check . && mypy .
```

Expected: clean.

- [ ] **Step 3: Tag**

```bash
git tag -a phase-2-kb-management -m "Phase 2 complete: KB CRUD + subtags + access grants + per-chat KB selection + RBAC"
```

- [ ] **Step 4: Commission Phase 3 plan**

Request controller: "Write Phase 3 plan at `docs/superpowers/plans/2026-04-16-phase-3-model-manager.md`: the model-manager sidecar service (FastAPI), idle tracker, HTTP wake/sleep forwarding to vllm-vision + whisper, `/api/models/status` endpoint, docker-compose wiring, integration test against a stubbed vLLM."

---

## Phase 2 acceptance checklist

- [ ] `git submodule status` unchanged (upstream v0.8.12).
- [ ] All previous Phase 1 tests still green (regression-free).
- [ ] Cross-user isolation test (`test_kb_isolation.py`) green — zero leak.
- [ ] Unauthorized KB selection returns 403 in `PUT /api/chats/{id}/kb_config`.
- [ ] Admin-only routes return 403 for `user` and `pending` roles.
- [ ] Missing auth headers return 401.
- [ ] Soft-deleted KBs invisible to all list/get endpoints.
- [ ] `ruff check .` + `mypy .` green.
- [ ] Tag `phase-2-kb-management` exists.

---

## Execution handoff

Plan saved. Execute via superpowers:subagent-driven-development, Sonnet-by-default. Dispatch Task 1 first.
