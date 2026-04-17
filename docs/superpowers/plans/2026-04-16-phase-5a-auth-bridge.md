# Phase 5a — Auth Bridge + Upstream Patch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Replace our header-based auth stub with real Open WebUI JWT verification (HS256, `WEBUI_SECRET_KEY`, `token` cookie / `Authorization: Bearer`), patch upstream's FastAPI `main.py` to mount our four routers, and add a startup-time migration applier so our SQL runs after upstream's schema init.

**Architecture:** `get_current_user` becomes mode-aware — `AUTH_MODE=jwt` (prod, verifies upstream JWT + looks up role from `users` table) or `AUTH_MODE=stub` (dev/tests, reads `X-User-Id`/`X-User-Role` headers). A single patch file (`patches/0001-mount-ext-routers.patch`) adds `app.include_router(...)` calls and a startup event to upstream's `main.py`. Migration applier runs our `ext/db/migrations/*.sql` against `DATABASE_URL` on startup, idempotently (wrapped in `IF NOT EXISTS` / `IF EXISTS` already).

**Tech Stack:** PyJWT (already a transitive dep via upstream requirements; also usable directly), SQLAlchemy async (existing), pytest + testcontainers.

**Working directory:** `/home/vogic/LocalRAG/` (main, tagged `phase-4-rag-pipeline`).

---

## Decisions (Phase 5a)

| # | Decision | Revise-by |
|---|----------|-----------|
| D35 | Two auth modes via `AUTH_MODE` env: `jwt` (prod) and `stub` (tests). Stub keeps `X-User-Id`+`X-User-Role`. JWT reads the same cookie/header upstream does — `token` cookie or `Authorization: Bearer`. | — |
| D36 | JWT payload we require: `id` (int, upstream uses str — we int-cast). We DO NOT require `role` in the JWT; role is looked up from `users.role` by `id` on each request. | — |
| D37 | Our patch to `upstream/main.py` is ONE file: `patches/0001-mount-ext-routers.patch`. Inserts imports, startup hook, and `include_router` calls. Must be minimal so rebases are cheap. | Phase 7 |
| D38 | Migration applier is a thin Python script (`scripts/apply_migrations.py`) invoked by the patch's startup hook. Uses sync asyncpg via sync wrapper to avoid importing our async engine inside upstream's sync context. | Phase 6 |
| D39 | No end-to-end "boot real upstream" test in 5a — the transitive deps are huge (langchain, openai, etc.). We verify: (a) JWT verifier round-trips with a known secret, (b) patch applies cleanly via `git apply --check`, (c) migration applier is idempotent. Full cross-service smoke deferred to Phase 6. | Phase 6 |

---

## File structure

```
ext/
├── services/
│   └── auth.py                (MODIFIED — add jwt_verifier + mode selector)
scripts/
├── apply_migrations.py        NEW — applies ext/db/migrations/*.sql to DATABASE_URL
└── apply_patches.sh           NEW — applies patches/*.patch to upstream
patches/
└── 0001-mount-ext-routers.patch  NEW — upstream/main.py hook
tests/
├── unit/
│   └── test_jwt_auth.py       NEW
└── integration/
    ├── test_apply_migrations.py  NEW
    └── test_patch_applies.py     NEW
compose/.env.example           (MODIFIED — add AUTH_MODE + WEBUI_SECRET_KEY)
```

---

## Task 1: PyJWT dep + JWT verifier helper

**Files:** `ext/services/jwt_verifier.py` (new), `tests/unit/test_jwt_auth.py`.

- [ ] **Step 1: Add pyjwt to runtime deps**

Edit `/home/vogic/LocalRAG/pyproject.toml` `[project] dependencies` — append `"pyjwt>=2.8,<3"`. Then `pip install -e ".[dev]"`.

- [ ] **Step 2: Write failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_jwt_auth.py`:

```python
import time
import jwt as pyjwt
import pytest
from ext.services.jwt_verifier import verify_upstream_jwt, JWTError


SECRET = "t0p-s3cr3t"
ALGO   = "HS256"


def _mint(payload: dict, secret: str = SECRET) -> str:
    return pyjwt.encode(payload, secret, algorithm=ALGO)


def test_valid_token_returns_id():
    tok = _mint({"id": "abc-123", "jti": "xyz"})
    claims = verify_upstream_jwt(tok, secret=SECRET)
    assert claims["id"] == "abc-123"


def test_expired_token_raises():
    tok = _mint({"id": "abc", "exp": int(time.time()) - 60})
    with pytest.raises(JWTError):
        verify_upstream_jwt(tok, secret=SECRET)


def test_bad_signature_raises():
    tok = _mint({"id": "abc"}, secret="wrong-secret")
    with pytest.raises(JWTError):
        verify_upstream_jwt(tok, secret=SECRET)


def test_missing_id_raises():
    tok = _mint({"jti": "x"})
    with pytest.raises(JWTError):
        verify_upstream_jwt(tok, secret=SECRET)


def test_malformed_token_raises():
    with pytest.raises(JWTError):
        verify_upstream_jwt("not-a-jwt", secret=SECRET)
```

- [ ] **Step 3: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_jwt_auth.py -v
```

- [ ] **Step 4: Write `ext/services/jwt_verifier.py`**

```python
"""Verifies Open WebUI upstream JWT tokens (HS256 via WEBUI_SECRET_KEY)."""
from __future__ import annotations

from typing import Any, Dict

import jwt as pyjwt


class JWTError(RuntimeError):
    """Invalid/expired/malformed JWT, or missing required claim."""


ALGORITHM = "HS256"


def verify_upstream_jwt(token: str, *, secret: str) -> Dict[str, Any]:
    """Decode + verify a token signed by Open WebUI. Raises JWTError on any failure.

    Requires claim `id` to be present. `exp` is optional but enforced when present.
    """
    try:
        claims = pyjwt.decode(token, secret, algorithms=[ALGORITHM])
    except pyjwt.ExpiredSignatureError as e:
        raise JWTError("token expired") from e
    except pyjwt.InvalidTokenError as e:
        raise JWTError(f"invalid token: {e}") from e

    if "id" not in claims:
        raise JWTError("missing 'id' claim")
    return claims
```

- [ ] **Step 5: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_jwt_auth.py -v
ruff check . && mypy .
```

- [ ] **Step 6: Commit**

```bash
git add ext/services/jwt_verifier.py tests/unit/test_jwt_auth.py pyproject.toml
git commit -m "feat(auth): JWT verifier (HS256, matches Open WebUI upstream)"
```

---

## Task 2: Mode-aware `get_current_user` (stub or JWT)

**Files:** `ext/services/auth.py` (modified), `tests/unit/test_auth_modes.py` (new).

- [ ] **Step 1: Write failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_auth_modes.py`:

```python
import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from ext.services.auth import CurrentUser, get_current_user, require_admin


SECRET = "test-secret-32-chars-or-whatever-x"


def _mk_app():
    app = FastAPI()

    @app.get("/me")
    def me(u: CurrentUser = Depends(get_current_user)):
        return {"id": u.id, "role": u.role}

    @app.get("/admin")
    def admin(u: CurrentUser = Depends(require_admin)):
        return {"ok": True}

    return app


def test_stub_mode_still_works(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "stub")
    client = TestClient(_mk_app())
    r = client.get("/me", headers={"X-User-Id": "42", "X-User-Role": "user"})
    assert r.status_code == 200
    assert r.json() == {"id": 42, "role": "user"}


def test_jwt_mode_accepts_cookie(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)

    # Patch the user-lookup to avoid needing a real DB in the unit test.
    from ext.services import auth as auth_mod
    monkeypatch.setattr(auth_mod, "_lookup_role_by_id", lambda uid: "user")

    tok = pyjwt.encode({"id": "100"}, SECRET, algorithm="HS256")
    client = TestClient(_mk_app())
    r = client.get("/me", cookies={"token": tok})
    assert r.status_code == 200
    assert r.json() == {"id": 100, "role": "user"}


def test_jwt_mode_accepts_bearer(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    from ext.services import auth as auth_mod
    monkeypatch.setattr(auth_mod, "_lookup_role_by_id", lambda uid: "admin")

    tok = pyjwt.encode({"id": "9"}, SECRET, algorithm="HS256")
    client = TestClient(_mk_app())
    r = client.get("/admin", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200


def test_jwt_mode_rejects_bad_signature(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    tok = pyjwt.encode({"id": "1"}, "wrong-secret", algorithm="HS256")
    client = TestClient(_mk_app())
    r = client.get("/me", cookies={"token": tok})
    assert r.status_code == 401


def test_jwt_mode_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    client = TestClient(_mk_app())
    r = client.get("/me")
    assert r.status_code == 401


def test_jwt_mode_rejects_unknown_user(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    from ext.services import auth as auth_mod
    monkeypatch.setattr(auth_mod, "_lookup_role_by_id", lambda uid: None)  # not found
    tok = pyjwt.encode({"id": "999"}, SECRET, algorithm="HS256")
    client = TestClient(_mk_app())
    r = client.get("/me", cookies={"token": tok})
    assert r.status_code == 401
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_auth_modes.py -v
```

- [ ] **Step 3: Rewrite `ext/services/auth.py`**

Replace the file entirely:

```python
"""Auth layer — two modes selectable via AUTH_MODE env:

- stub (default for tests / local dev): reads X-User-Id + X-User-Role headers.
- jwt  (production): verifies upstream Open WebUI JWT (token cookie or Bearer),
                     looks up role from the `users` table.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select

from .jwt_verifier import JWTError, verify_upstream_jwt


@dataclass(frozen=True)
class CurrentUser:
    id: int
    role: str


VALID_ROLES = {"admin", "user", "pending"}


# ----- Stub mode -----
def _stub_user(
    x_user_id: Optional[str],
    x_user_role: Optional[str],
) -> CurrentUser:
    if x_user_id is None or x_user_role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing auth headers")
    try:
        uid = int(x_user_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Id") from e
    if x_user_role not in VALID_ROLES:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad X-User-Role")
    return CurrentUser(id=uid, role=x_user_role)


# ----- JWT mode -----
_sessionmaker = None


def configure_jwt(*, sessionmaker) -> None:
    """Wire the async sessionmaker for role lookups in jwt mode."""
    global _sessionmaker
    _sessionmaker = sessionmaker


def _lookup_role_by_id(user_id: int) -> Optional[str]:
    """Synchronous wrapper over an async DB read. Monkeypatchable in unit tests."""
    if _sessionmaker is None:
        raise RuntimeError("AUTH_MODE=jwt but sessionmaker not configured; "
                           "call configure_jwt(sessionmaker=...) at app startup")
    from .._lazy_user_lookup import lookup_role_async
    return asyncio.run(lookup_role_async(_sessionmaker, user_id))


def _jwt_user(request: Request) -> CurrentUser:
    secret = os.environ.get("WEBUI_SECRET_KEY", "t0p-s3cr3t")
    token = None
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = request.cookies.get("token")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="no auth token")
    try:
        claims = verify_upstream_jwt(token, secret=secret)
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(e)) from e
    try:
        uid = int(claims["id"])
    except (TypeError, ValueError) as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="bad id in claims") from e
    role = _lookup_role_by_id(uid)
    if role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="unknown user")
    return CurrentUser(id=uid, role=role)


# ----- Public dependency -----
def get_current_user(
    request: Request,
    x_user_id:   Optional[str] = Header(default=None, alias="X-User-Id"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
) -> CurrentUser:
    mode = os.environ.get("AUTH_MODE", "stub").lower()
    if mode == "stub":
        return _stub_user(x_user_id, x_user_role)
    if mode == "jwt":
        return _jwt_user(request)
    raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"unknown AUTH_MODE: {mode}")


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin only")
    return user
```

- [ ] **Step 4: Create the lazy user-lookup module**

Create `/home/vogic/LocalRAG/ext/_lazy_user_lookup.py`:

```python
"""Isolated to avoid a circular import between auth and db.models."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select


async def lookup_role_async(sessionmaker, user_id: int) -> Optional[str]:
    from .db.models import User
    async with sessionmaker() as s:
        user = (await s.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        return user.role if user is not None else None
```

- [ ] **Step 5: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_auth_modes.py tests/unit/test_auth_dep.py -v
ruff check . && mypy .
```

Expected: 6 new PASSED + 4 existing PASSED (the stub tests still work because AUTH_MODE defaults to "stub").

- [ ] **Step 6: Commit**

```bash
git add ext/services/auth.py ext/_lazy_user_lookup.py tests/unit/test_auth_modes.py
git commit -m "feat(auth): mode-aware get_current_user (stub | jwt)"
```

---

## Task 3: Migration applier script

**Files:** `scripts/apply_migrations.py` (new), `tests/integration/test_apply_migrations.py` (new).

- [ ] **Step 1: Write failing test**

Create `/home/vogic/LocalRAG/tests/integration/test_apply_migrations.py`:

```python
import os
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "apply_migrations.py"


@pytest_asyncio.fixture
async def pg_seeded():
    """Postgres with upstream-style seed tables already created."""
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
    # chat_count column added by migration 003
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
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_apply_migrations.py -v
```

- [ ] **Step 3: Write `scripts/apply_migrations.py`**

```python
#!/usr/bin/env python3
"""Apply every ext/db/migrations/*.sql file against DATABASE_URL in order.

Safe to re-run — migrations use IF EXISTS / IF NOT EXISTS throughout.
Intended to run at upstream's FastAPI startup (after upstream's own schema init).

Env:
    DATABASE_URL   postgresql+asyncpg://... (required)
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
        for m in MIGRATIONS:
            print(f"  -> {m.name}")
            await raw.driver_connection.execute(m.read_text())
    await eng.dispose()
    print("migrations OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_apply_migrations.py -v
ruff check . && mypy .
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/apply_migrations.py tests/integration/test_apply_migrations.py
git commit -m "feat(auth): startup migration applier"
```

---

## Task 4: Upstream patch — mount our routers in `main.py`

**Files:** `patches/0001-mount-ext-routers.patch` (new), `scripts/apply_patches.sh` (new).

- [ ] **Step 1: Inspect upstream main.py for the exact insertion line**

Read `/home/vogic/LocalRAG/upstream/backend/open_webui/main.py` around line 1519. Find the last existing `app.include_router(...)` call. We insert after it. Also find the existing `@app.on_event("startup")` or `@asynccontextmanager def lifespan(...)` — we append our block to the end of that. If upstream uses `lifespan`, add to the body. If `on_event`, use `@app.on_event("startup")`.

(The existing agent that researched this reported: routers registered lines 1487–1519.)

- [ ] **Step 2: Produce the patch**

Working approach — generate the patch by modifying `main.py` directly, then `git format-patch`:

```bash
cd /home/vogic/LocalRAG/upstream
# Save upstream commit we're patching against:
UPSTREAM_SHA=$(git rev-parse HEAD)
cd ..

# Edit upstream/backend/open_webui/main.py:
#   After the last `app.include_router(...)` call near line 1519, append:
#
#   # --- orgchat extension wiring (Phase 5a) ---
#   import os, sys
#   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
#   from ext.app import build_ext_routers  # noqa: E402
#   for _r in build_ext_routers():
#       app.include_router(_r)
#   # --- end orgchat extension ---
#
# AND at startup (inside existing lifespan or a new startup handler), add:
#
#   import subprocess, pathlib
#   _mig = pathlib.Path(__file__).parent.parent.parent.parent / "scripts" / "apply_migrations.py"
#   subprocess.run([sys.executable, str(_mig)], check=True)
```

Because fully correct diff generation requires looking at actual lines, the implementer will:
1. `cd /home/vogic/LocalRAG/upstream && git status`  (should be clean)
2. Open `upstream/backend/open_webui/main.py` and insert the two blocks.
3. `git -C upstream diff > /tmp/ext.patch`
4. `git -C upstream checkout main.py`  (revert upstream, we only want the diff)
5. `mv /tmp/ext.patch /home/vogic/LocalRAG/patches/0001-mount-ext-routers.patch`

- [ ] **Step 3: Write `ext/app.py` helper for router listing**

Our current `ext/app.py::build_app()` constructs a full FastAPI app. We also need a helper that returns just the routers for upstream to include. Append to `/home/vogic/LocalRAG/ext/app.py`:

```python
def build_ext_routers():
    """Return the list of APIRouters to mount on an external FastAPI app (upstream).

    Caller is responsible for setting env so our settings + services bootstrap cleanly:
      DATABASE_URL, QDRANT_URL, TEI_URL, RAG_VECTOR_SIZE, WEBUI_SECRET_KEY, AUTH_MODE=jwt.
    """
    from .config import clear_settings_cache, get_settings
    from .db.session import make_engine, make_sessionmaker
    from .routers import kb_admin, kb_retrieval, rag, upload
    from .services import auth as auth_svc
    from .services.embedder import TEIEmbedder
    from .services.vector_store import VectorStore

    clear_settings_cache()
    settings = get_settings()
    engine = make_engine(settings.database_url)
    SessionLocal = make_sessionmaker(engine)

    vs = VectorStore(url=settings.qdrant_url, vector_size=settings.vector_size)
    emb = TEIEmbedder(base_url=settings.tei_url)

    auth_svc.configure_jwt(sessionmaker=SessionLocal)
    kb_admin.set_sessionmaker(SessionLocal)
    kb_retrieval.set_sessionmaker(SessionLocal)
    upload.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    rag.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)

    # Retrieval first — see app.py comment.
    return [kb_retrieval.router, kb_admin.router, upload.router, rag.router]
```

- [ ] **Step 4: Write `scripts/apply_patches.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UP="$ROOT/upstream"

if [[ ! -d "$UP/.git" ]]; then
  echo "!! upstream submodule not initialized — run: git submodule update --init upstream" >&2
  exit 1
fi

shopt -s nullglob
for p in "$ROOT"/patches/*.patch; do
  echo "applying $(basename "$p")"
  if git -C "$UP" apply --check "$p" 2>/dev/null; then
    git -C "$UP" apply "$p"
  else
    # already applied?
    if git -C "$UP" apply --check --reverse "$p" 2>/dev/null; then
      echo "  already applied (skipping)"
    else
      echo "!! patch $p does not apply cleanly; upstream changed — regenerate" >&2
      exit 2
    fi
  fi
done
echo "all patches applied."
```

```bash
chmod +x /home/vogic/LocalRAG/scripts/apply_patches.sh
```

- [ ] **Step 5: Write failing test**

Create `/home/vogic/LocalRAG/tests/integration/test_patch_applies.py`:

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_apply_patches_dry_run():
    """Patches apply cleanly to upstream at its pinned SHA."""
    # Reset upstream to its pinned commit, then apply.
    tag = (ROOT / "UPSTREAM_VERSION").read_text().strip()
    r = subprocess.run(
        ["git", "-C", str(ROOT / "upstream"), "checkout", tag],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "apply_patches.sh")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"apply_patches.sh failed:\n{r.stderr}\n{r.stdout}"

    # Verify our marker appears in main.py
    main_py = (ROOT / "upstream" / "backend" / "open_webui" / "main.py").read_text()
    assert "orgchat extension wiring" in main_py

    # Clean up — revert upstream so the test leaves no state.
    subprocess.run(
        ["git", "-C", str(ROOT / "upstream"), "checkout", tag, "--", "."],
        check=False,
    )


def test_apply_patches_is_idempotent():
    """Running the apply script twice should succeed (second time = skip)."""
    tag = (ROOT / "UPSTREAM_VERSION").read_text().strip()
    subprocess.run(["git", "-C", str(ROOT / "upstream"), "checkout", tag], check=True,
                   capture_output=True)
    # First run applies, second run should detect already-applied and skip.
    r1 = subprocess.run(["bash", str(ROOT / "scripts" / "apply_patches.sh")],
                        capture_output=True, text=True)
    assert r1.returncode == 0
    r2 = subprocess.run(["bash", str(ROOT / "scripts" / "apply_patches.sh")],
                        capture_output=True, text=True)
    assert r2.returncode == 0
    assert "already applied" in r2.stdout

    # Clean up.
    subprocess.run(["git", "-C", str(ROOT / "upstream"), "checkout", tag, "--", "."],
                   check=False)
```

- [ ] **Step 6: Run + fix until PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_patch_applies.py -v
```

If the patch doesn't apply cleanly, the implementer should re-generate the patch against the current upstream HEAD by editing `main.py` directly and running `git -C upstream diff > patches/0001-mount-ext-routers.patch`.

- [ ] **Step 7: Commit**

```bash
git add patches/0001-mount-ext-routers.patch scripts/apply_patches.sh \
        ext/app.py tests/integration/test_patch_applies.py
git commit -m "feat(auth): upstream patch mounts our routers + apply_patches.sh"
```

---

## Task 5: Env config update + Phase 5a regression + tag

**Files:** `compose/.env.example`, final regression.

- [ ] **Step 1: Extend `.env.example`**

Append these keys to `/home/vogic/LocalRAG/compose/.env.example`:

```
# --- Auth bridge (Phase 5a) ---
AUTH_MODE=jwt                               # "jwt" for production, "stub" for tests
WEBUI_SECRET_KEY=change-me-to-random-bytes  # must match upstream Open WebUI's setting
```

- [ ] **Step 2: Regression sweep**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate
python -m pytest tests/unit -v 2>&1 | tail -5
SKIP_GPU_SMOKE=1 python -m pytest tests/integration -v 2>&1 | tail -15
ruff check . && mypy .
```

- [ ] **Step 3: Tag + report**

```bash
cd /home/vogic/LocalRAG
git add compose/.env.example
git commit -m "chore: document AUTH_MODE + WEBUI_SECRET_KEY"
git tag -a phase-5a-auth-bridge -m "Phase 5a complete: JWT auth + upstream patch + migration applier"
```

---

## Phase 5a acceptance checklist

- [ ] `ext/services/jwt_verifier.py` decodes Open WebUI HS256 JWTs.
- [ ] `ext/services/auth.py` supports AUTH_MODE=stub (tests) and AUTH_MODE=jwt (prod).
- [ ] JWT mode reads either `Authorization: Bearer` OR `token` cookie.
- [ ] Unknown users return 401 in JWT mode.
- [ ] `scripts/apply_migrations.py` is idempotent and applies our SQL files in order.
- [ ] `patches/0001-mount-ext-routers.patch` applies cleanly to upstream at the pinned tag.
- [ ] `scripts/apply_patches.sh` is idempotent.
- [ ] Tag `phase-5a-auth-bridge` exists.
- [ ] All existing tests still pass (regression-free).

---

## Execution handoff

Phase 5b (Svelte frontend) — per user decision, handled collaboratively with live browser; NOT dispatched as blind subagent work.
