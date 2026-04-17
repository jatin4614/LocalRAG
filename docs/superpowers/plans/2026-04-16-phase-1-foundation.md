# Phase 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the hybrid thin-fork skeleton, always-on Docker services, KB database schema (replacing Open WebUI's upstream `knowledge` tables), self-signed TLS, admin seed, and a preflight model-weight check — so that `make up && make smoke` is green on a fresh host.

**Architecture:** Git repo with `upstream/` as an Open WebUI submodule pinned to latest stable; our modules in `ext/`; Docker Compose brings up postgres + redis + qdrant + caddy + vllm-chat + tei + vllm-vision + whisper + model-manager (stubs for Phase-3 services land here but stay behind a health gate). Migrations apply on postgres startup. All model weights pre-downloaded into a persistent volume so every subsequent boot is offline.

**Tech Stack:** Python ≥ 3.10 (the host this is being built on has 3.10.12), SQLAlchemy 2.0 async, asyncpg, pytest + pytest-asyncio + testcontainers + httpx, Docker Compose 3.8, Caddy 2, vLLM ≥ 0.7 (with `--enable-sleep-mode`), TEI 1.5, faster-whisper, Argon2-cffi for password hashing.

**Working directory:** `/home/vogic/LocalRAG/`

---

## File structure delivered by this phase

See the master plan (`2026-04-16-org-chat-assistant-master-plan.md` §3). All files listed there that are annotated with Phase 1 or a Task N number in 1–18 are produced here.

---

## Pre-task: directory sanity

Run once, before Task 1:

```bash
cd /home/vogic/LocalRAG
ls -la
# Expect: CLAUDE.md, docs/ present. No .git, no code yet.
```

---

### Task 1: Initialize git repo + .gitignore + README skeleton

**Files:**
- Create: `/home/vogic/LocalRAG/.gitignore`
- Create: `/home/vogic/LocalRAG/README.md`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_repo_skeleton.py`:

```python
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_git_repo_initialized():
    assert (ROOT / ".git").is_dir(), ".git directory missing — run `git init`"

def test_gitignore_ignores_volumes_and_venv():
    content = (ROOT / ".gitignore").read_text()
    for entry in ["volumes/", ".venv/", "__pycache__/", ".pytest_cache/", "*.pyc"]:
        assert entry in content, f".gitignore missing entry: {entry}"
    # upstream/ is a submodule path — must NOT be gitignored (git refuses to add a
    # submodule at a path that is under a gitignore rule).
    assert "upstream/" not in content, (
        "upstream/ must NOT be gitignored — it's a git submodule path"
    )

def test_readme_exists():
    readme = ROOT / "README.md"
    assert readme.exists()
    assert "Org Chat Assistant" in readme.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && python -m pytest tests/unit/test_repo_skeleton.py -v
```

Expected: FAIL — pytest itself missing OR `.git` missing. Either is acceptable; we set up pytest in Task 3 and may need to defer running this assertion until then. If pytest import fails, note the failure and move on; re-run after Task 3.

- [ ] **Step 3: Initialize git + create .gitignore + README**

```bash
cd /home/vogic/LocalRAG
git init -b main
```

Create `/home/vogic/LocalRAG/.gitignore`:

```
# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.venv/
.mypy_cache/

# Editor
.idea/
.vscode/
*.swp

# Secrets / local
.env
.env.local
compose/.env

# NOTE: upstream/ is a git submodule path — do NOT list it here.
# git submodule add refuses to add a submodule at a gitignored path.

# Runtime / volumes
volumes/
*.log

# Build
dist/
build/
*.egg-info/
```

Create `/home/vogic/LocalRAG/README.md`:

```markdown
# Org Chat Assistant

Self-hosted, air-gapped, multi-user ChatGPT-like web assistant with hierarchical Knowledge Bases. Hybrid thin-fork of Open WebUI.

See `CLAUDE.md` for the project summary and `docs/superpowers/plans/2026-04-16-org-chat-assistant-master-plan.md` for the implementation plan.

## Quick start (after Phase 1 complete)

```bash
cp compose/.env.example compose/.env
# edit compose/.env: set WEBUI_NAME, ADMIN_EMAIL, ADMIN_PASSWORD, SESSION_SECRET
make preflight   # downloads model weights once (needs internet)
make up          # brings up the stack
make smoke       # runs smoke test
```

## Repo layout

See the master plan's §3 Repo layout.
```

- [ ] **Step 4: Commit**

```bash
cd /home/vogic/LocalRAG
git add .gitignore README.md
git commit -m "chore: init repo skeleton (gitignore, README)"
```

---

### Task 2: Pin Open WebUI as a submodule

**Files:**
- Create: `/home/vogic/LocalRAG/UPSTREAM_VERSION`
- Create: `/home/vogic/LocalRAG/scripts/rebase_upstream.sh`
- Modify: `.gitmodules` (auto-created by `git submodule add`)

- [ ] **Step 1: Discover latest stable Open WebUI tag**

```bash
# Needs one-time internet access:
curl -sSL "https://api.github.com/repos/open-webui/open-webui/releases/latest" \
  | python -c "import json,sys; print(json.load(sys.stdin)['tag_name'])"
```

Record the printed tag (e.g., `v0.6.x`) for use below. Let `TAG` be that value. Then:

```bash
echo "$TAG" > /home/vogic/LocalRAG/UPSTREAM_VERSION
```

- [ ] **Step 2: Add submodule pinned to that tag**

```bash
cd /home/vogic/LocalRAG
git submodule add https://github.com/open-webui/open-webui.git upstream
cd upstream
git fetch --tags origin
TAG=$(cat ../UPSTREAM_VERSION)
git checkout "$TAG"
cd ..
git add .gitmodules upstream UPSTREAM_VERSION
git commit -m "chore: pin upstream open-webui submodule to $TAG"
```

- [ ] **Step 3: Add rebase helper script**

Create `/home/vogic/LocalRAG/scripts/rebase_upstream.sh`:

```bash
#!/usr/bin/env bash
# Rebase upstream to a new Open WebUI tag, reapply patches.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <new-tag>    e.g. $0 v0.7.0" >&2
  exit 2
fi
NEW_TAG="$1"
cd "$(dirname "$0")/.."

echo ">> fetching upstream"
(cd upstream && git fetch --tags origin && git checkout "$NEW_TAG")

echo ">> staging new gitlink in parent repo"
git add upstream

echo ">> updating UPSTREAM_VERSION"
echo "$NEW_TAG" > UPSTREAM_VERSION

echo ">> reapplying patches"
for p in patches/*.patch; do
  [[ -e "$p" ]] || continue
  echo ">>   applying $p"
  git -C upstream apply --3way "$(realpath "$p")" || {
    echo "!! patch $p failed to apply cleanly; resolve manually then rerun"
    exit 1
  }
done

echo ">> done. Review, test, commit."
```

```bash
chmod +x /home/vogic/LocalRAG/scripts/rebase_upstream.sh
```

- [ ] **Step 4: Verify submodule pinned correctly**

```bash
cd /home/vogic/LocalRAG
cat UPSTREAM_VERSION
git -C upstream describe --tags
# Both should print the same tag.
```

Expected: identical tag strings.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add scripts/rebase_upstream.sh
git commit -m "chore: add upstream rebase helper"
```

---

### Task 3: Python tooling — venv, pyproject, pytest, Makefile

**Files:**
- Create: `/home/vogic/LocalRAG/pyproject.toml`
- Create: `/home/vogic/LocalRAG/Makefile`
- Create: `/home/vogic/LocalRAG/tests/conftest.py`
- Create: `/home/vogic/LocalRAG/tests/__init__.py`
- Create: `/home/vogic/LocalRAG/tests/unit/__init__.py`
- Create: `/home/vogic/LocalRAG/tests/integration/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_tooling.py`:

```python
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_pyproject_has_project_name():
    content = (ROOT / "pyproject.toml").read_text()
    assert 'name = "org-chat-assistant"' in content

def test_pytest_asyncio_installed():
    import pytest_asyncio  # noqa: F401

def test_sqlalchemy_async_installed():
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401

def test_testcontainers_installed():
    import testcontainers  # noqa: F401

def test_httpx_installed():
    import httpx  # noqa: F401

def test_argon2_installed():
    import argon2  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG
python3 -m venv .venv
source .venv/bin/activate
pip install pytest
python -m pytest tests/unit/test_tooling.py -v
```

Expected: FAIL — `pyproject.toml` missing and optional deps (pytest_asyncio, sqlalchemy, testcontainers, httpx, argon2) not installed.

- [ ] **Step 3: Write pyproject.toml**

Create `/home/vogic/LocalRAG/pyproject.toml`:

```toml
[project]
name = "org-chat-assistant"
version = "0.1.0"
description = "Hybrid thin-fork of Open WebUI with KB-based RAG pipeline"
requires-python = ">=3.10,<3.13"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "sqlalchemy[asyncio]>=2.0.30",
  "asyncpg>=0.29",
  "alembic>=1.13",
  "argon2-cffi>=23.1",
  "pydantic>=2.8",
  "pydantic-settings>=2.4",
  "httpx>=0.27",
  "redis>=5.0",
  "qdrant-client>=1.9",
  "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-asyncio>=0.23",
  "testcontainers[postgres,redis]>=4.0",
  "docker>=7.0",
  "ruff>=0.6",
  "mypy>=1.11",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 100
target-version = "py310"
```

- [ ] **Step 4: Install dev dependencies**

```bash
cd /home/vogic/LocalRAG
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 5: Write Makefile**

Create `/home/vogic/LocalRAG/Makefile`:

```makefile
.PHONY: help venv install test test-unit test-integration lint preflight up down smoke logs clean

PYTHON ?= python3
VENV   ?= .venv
ACTIVATE = . $(VENV)/bin/activate

help:
	@echo "venv              create .venv"
	@echo "install           install dev deps into venv"
	@echo "test              run all pytest"
	@echo "test-unit         run unit tests only"
	@echo "test-integration  run integration tests only"
	@echo "lint              ruff + mypy"
	@echo "preflight         download + verify model weights (needs internet once)"
	@echo "up                docker compose up -d"
	@echo "down              docker compose down"
	@echo "smoke             curl healthchecks"
	@echo "logs              docker compose logs -f"
	@echo "clean             remove .venv, caches"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(ACTIVATE) && pip install -e ".[dev]"

test:
	$(ACTIVATE) && pytest -v

test-unit:
	$(ACTIVATE) && pytest tests/unit -v

test-integration:
	$(ACTIVATE) && pytest tests/integration -v

lint:
	$(ACTIVATE) && ruff check . && mypy ext/ model_manager/ whisper_service/ scripts/

preflight:
	$(ACTIVATE) && python scripts/preflight_models.py

up:
	docker compose -f compose/docker-compose.yml --env-file compose/.env up -d

down:
	docker compose -f compose/docker-compose.yml --env-file compose/.env down

smoke:
	$(ACTIVATE) && pytest tests/integration/test_compose_up.py -v

logs:
	docker compose -f compose/docker-compose.yml --env-file compose/.env logs -f

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
```

- [ ] **Step 6: Write tests/conftest.py + package __init__ files**

Create `/home/vogic/LocalRAG/tests/__init__.py` (empty). Create `/home/vogic/LocalRAG/tests/unit/__init__.py` (empty). Create `/home/vogic/LocalRAG/tests/integration/__init__.py` (empty).

Create `/home/vogic/LocalRAG/tests/conftest.py`:

```python
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT
```

- [ ] **Step 7: Run tests — expect pass**

```bash
cd /home/vogic/LocalRAG
source .venv/bin/activate
python -m pytest tests/unit/test_tooling.py tests/unit/test_repo_skeleton.py -v
```

Expected: 8 PASSED (3 from Task 1, 6 from Task 3).

- [ ] **Step 8: Commit**

```bash
cd /home/vogic/LocalRAG
git add pyproject.toml Makefile tests/
git commit -m "chore: add Python tooling (pyproject, Makefile, pytest scaffolding)"
```

---

### Task 4: Scaffolding directories — ext/, model_manager/, whisper_service/, scripts/, compose/

**Files:** all paths below are created empty or with `__init__.py` stubs.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_scaffolding.py`:

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DIRS = [
    "ext",
    "ext/db",
    "ext/db/migrations",
    "ext/db/models",
    "ext/services",
    "ext/routers",
    "model_manager",
    "whisper_service",
    "scripts",
    "compose",
    "compose/caddy",
    "branding",
    "volumes",  # gitignored, but present for bind mounts
]

EXPECTED_INITS = [
    "ext/__init__.py",
    "ext/db/__init__.py",
    "ext/db/models/__init__.py",
    "ext/services/__init__.py",
    "ext/routers/__init__.py",
]

def test_directories_exist():
    for d in EXPECTED_DIRS:
        assert (ROOT / d).is_dir(), f"missing dir: {d}"

def test_init_files_exist():
    for f in EXPECTED_INITS:
        assert (ROOT / f).is_file(), f"missing __init__: {f}"

def test_ext_importable():
    import sys
    sys.path.insert(0, str(ROOT))
    import ext  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_scaffolding.py -v
```

Expected: FAIL on first missing directory.

- [ ] **Step 3: Create directories and `__init__.py` files**

```bash
cd /home/vogic/LocalRAG
mkdir -p ext/db/migrations ext/db/models ext/services ext/routers
mkdir -p model_manager whisper_service scripts compose/caddy branding
mkdir -p volumes/postgres volumes/qdrant volumes/redis volumes/models volumes/certs
touch ext/__init__.py ext/db/__init__.py ext/db/models/__init__.py ext/services/__init__.py ext/routers/__init__.py
touch branding/README.md
```

Put a short note in `branding/README.md`:

```markdown
# Branding assets

User supplies the following (all TODOs noted in the design spec §9):

- `logo.svg`, `logo-dark.svg`
- `favicon.ico`
- `splash.png`
- `theme.css` — primary/accent color tokens

Copies land under `upstream/static/` via `patches/0003-branding-assets.patch` during rebases.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_scaffolding.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add ext/ model_manager/ whisper_service/ scripts/ compose/ branding/
# volumes/ is gitignored, won't be added
git commit -m "chore: scaffold hybrid thin-fork directories"
```

---

### Task 5: Docker Compose — stateful base (postgres + redis + qdrant) + `.env.example`

**Files:**
- Create: `/home/vogic/LocalRAG/compose/docker-compose.yml`
- Create: `/home/vogic/LocalRAG/compose/.env.example`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_compose_config.py`:

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_compose_config_valid():
    result = subprocess.run(
        ["docker", "compose", "-f", str(ROOT / "compose/docker-compose.yml"),
         "--env-file", str(ROOT / "compose/.env.example"), "config"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"compose config failed:\n{result.stderr}"

def test_stateful_services_present():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    for svc in ["postgres:", "redis:", "qdrant:"]:
        assert svc in content, f"missing service block: {svc}"

def test_env_example_keys():
    content = (ROOT / "compose/.env.example").read_text()
    for key in ["POSTGRES_PASSWORD=", "ADMIN_EMAIL=", "ADMIN_PASSWORD=",
                "SESSION_SECRET=", "WEBUI_NAME=", "DOMAIN="]:
        assert key in content, f"missing env key: {key}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_compose_config.py -v
```

Expected: FAIL — files don't exist.

- [ ] **Step 3: Write `compose/.env.example`**

Create `/home/vogic/LocalRAG/compose/.env.example`:

```bash
# --- Core ---
WEBUI_NAME=OrgChat
DOMAIN=orgchat.lan
SESSION_SECRET=change-me-to-32-random-bytes

# --- Admin bootstrap ---
ADMIN_EMAIL=admin@orgchat.lan
ADMIN_PASSWORD=change-me-please

# --- Postgres ---
POSTGRES_USER=orgchat
POSTGRES_PASSWORD=change-me-postgres
POSTGRES_DB=orgchat
DATABASE_URL=postgresql+asyncpg://orgchat:change-me-postgres@postgres:5432/orgchat

# --- Redis ---
REDIS_URL=redis://redis:6379/0

# --- Qdrant ---
QDRANT_URL=http://qdrant:6333

# --- Models ---
CHAT_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
VISION_MODEL=Qwen/Qwen2-VL-7B-Instruct
EMBED_MODEL=BAAI/bge-m3
WHISPER_MODEL=medium

# --- Open WebUI integration ---
ENABLE_SIGNUP=false
DEFAULT_USER_ROLE=pending
ENABLE_OPENAI_API=true
OPENAI_API_BASE_URL=http://vllm-chat:8000/v1
OPENAI_API_KEY=sk-internal-dummy
ENABLE_OLLAMA_API=false
ENABLE_WEB_SEARCH=false
ENABLE_IMAGE_GENERATION=false
ENABLE_RAG_WEB_LOADER=false
AUDIO_STT_ENGINE=whisper-local
RAG_EMBEDDING_ENGINE=openai
RAG_EMBEDDING_OPENAI_API_BASE_URL=http://tei:80/v1
RAG_EMBEDDING_MODEL=BAAI/bge-m3
VECTOR_DB=qdrant

# --- Model manager ---
MODEL_MANAGER_URL=http://model-manager:8080
MODEL_UNLOAD_IDLE_SECS=300
```

- [ ] **Step 4: Write `compose/docker-compose.yml` (stateful base only — remaining services added in later tasks)**

Create `/home/vogic/LocalRAG/compose/docker-compose.yml`:

```yaml
version: "3.8"

name: orgchat

networks:
  default:
    name: orgchat-net
    driver: bridge
    internal: false  # temporarily false; switched to true after Caddy is wired (Task 7)

volumes:
  postgres_data: { driver: local }
  redis_data:    { driver: local }
  qdrant_data:   { driver: local }

services:
  postgres:
    image: postgres:15-alpine
    container_name: orgchat-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER:     ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB:       ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ../ext/db/migrations:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: orgchat-redis
    restart: unless-stopped
    command: ["redis-server", "--appendonly", "yes"]
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    container_name: orgchat-qdrant
    restart: unless-stopped
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "wget -q -O- http://localhost:6333/readyz || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_compose_config.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Bring the stack up and verify**

```bash
cd /home/vogic/LocalRAG
cp compose/.env.example compose/.env
docker compose -f compose/docker-compose.yml --env-file compose/.env up -d
sleep 15
docker compose -f compose/docker-compose.yml --env-file compose/.env ps
```

Expected: all three services `healthy`.

Tear down before committing:

```bash
docker compose -f compose/docker-compose.yml --env-file compose/.env down
```

- [ ] **Step 7: Commit**

```bash
cd /home/vogic/LocalRAG
git add compose/docker-compose.yml compose/.env.example tests/unit/test_compose_config.py
git commit -m "feat: compose stateful base (postgres + redis + qdrant)"
```

---

### Task 6: Self-signed TLS cert generator

**Files:**
- Create: `/home/vogic/LocalRAG/scripts/gen_self_signed_cert.sh`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_cert_gen.py`:

```python
import os
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gen_self_signed_cert.sh"

def test_script_exists_and_executable():
    assert SCRIPT.is_file()
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "script not executable"

def test_script_produces_cert_and_key(tmp_path):
    env = os.environ.copy()
    env["CERT_DIR"]     = str(tmp_path)
    env["CERT_CN"]      = "test.orgchat.lan"
    env["CERT_DAYS"]    = "30"
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "orgchat.crt").is_file()
    assert (tmp_path / "orgchat.key").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_cert_gen.py -v
```

Expected: FAIL — script does not exist.

- [ ] **Step 3: Write the script**

Create `/home/vogic/LocalRAG/scripts/gen_self_signed_cert.sh`:

```bash
#!/usr/bin/env bash
# Generate a self-signed TLS cert for LAN-internal use.
set -euo pipefail

CERT_DIR="${CERT_DIR:-$(cd "$(dirname "$0")/.." && pwd)/volumes/certs}"
CERT_CN="${CERT_CN:-orgchat.lan}"
CERT_DAYS="${CERT_DAYS:-3650}"

mkdir -p "$CERT_DIR"
KEY="$CERT_DIR/orgchat.key"
CRT="$CERT_DIR/orgchat.crt"

if [[ -f "$KEY" && -f "$CRT" ]]; then
  echo "cert already present at $CERT_DIR — leaving in place"
  exit 0
fi

openssl req -x509 -nodes -newkey rsa:4096 \
  -keyout "$KEY" \
  -out "$CRT" \
  -days "$CERT_DAYS" \
  -subj "/CN=$CERT_CN" \
  -addext "subjectAltName=DNS:$CERT_CN,DNS:localhost,IP:127.0.0.1"

chmod 600 "$KEY"
chmod 644 "$CRT"
echo "wrote $CRT and $KEY (CN=$CERT_CN, days=$CERT_DAYS)"
```

```bash
chmod +x /home/vogic/LocalRAG/scripts/gen_self_signed_cert.sh
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_cert_gen.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add scripts/gen_self_signed_cert.sh tests/unit/test_cert_gen.py
git commit -m "feat: self-signed TLS cert generator"
```

---

### Task 7: Caddy service in compose + Caddyfile

**Files:**
- Create: `/home/vogic/LocalRAG/compose/caddy/Caddyfile`
- Modify: `/home/vogic/LocalRAG/compose/docker-compose.yml` (append `caddy` service)

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_caddy_config.py`:

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_caddyfile_routes_root_to_openwebui():
    content = (ROOT / "compose/caddy/Caddyfile").read_text()
    assert "reverse_proxy open-webui:8080" in content

def test_caddy_service_in_compose():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    assert "caddy:" in content
    assert "./caddy/Caddyfile:/etc/caddy/Caddyfile:ro" in content
    assert "../volumes/certs:/certs:ro" in content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_caddy_config.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write the Caddyfile**

Create `/home/vogic/LocalRAG/compose/caddy/Caddyfile`:

```caddy
{
    # No external ACME — self-signed only.
    auto_https off
    admin off
}

{$DOMAIN}:443 {
    tls /certs/orgchat.crt /certs/orgchat.key

    encode zstd gzip

    # Open WebUI is the single entry point for the user-facing app.
    # Upstream listens on :8080 inside the container.
    reverse_proxy open-webui:8080 {
        # Preserve streaming responses for chat.
        flush_interval -1
    }
}

:80 {
    redir https://{host}{uri} permanent
}
```

- [ ] **Step 4: Append Caddy service to compose**

Modify `/home/vogic/LocalRAG/compose/docker-compose.yml` by appending under `services:` (keep existing services above):

```yaml
  caddy:
    image: caddy:2-alpine
    container_name: orgchat-caddy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    environment:
      DOMAIN: ${DOMAIN}
    volumes:
      - ./caddy/Caddyfile:/etc/caddy/Caddyfile:ro
      - ../volumes/certs:/certs:ro
    depends_on:
      - postgres   # real dependency added when open-webui lands
```

- [ ] **Step 5: Run tests**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_caddy_config.py tests/unit/test_compose_config.py -v
```

Expected: 5 PASSED.

- [ ] **Step 6: Commit**

```bash
cd /home/vogic/LocalRAG
git add compose/caddy/Caddyfile compose/docker-compose.yml tests/unit/test_caddy_config.py
git commit -m "feat: add Caddy reverse proxy with self-signed TLS"
```

---

### Task 8: vllm-chat service in compose

**Files:**
- Modify: `/home/vogic/LocalRAG/compose/docker-compose.yml`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_vllm_chat_config.py`:

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_vllm_chat_block_present():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    assert "vllm-chat:" in content
    assert "vllm/vllm-openai" in content
    assert "${CHAT_MODEL}" in content
    assert "--gpu-memory-utilization" in content
    assert "--enable-prefix-caching" in content
    # GPU reservation
    assert "driver: nvidia" in content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_vllm_chat_config.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append vllm-chat service to compose**

Append under `services:` in `/home/vogic/LocalRAG/compose/docker-compose.yml`:

```yaml
  vllm-chat:
    image: vllm/vllm-openai:latest
    container_name: orgchat-vllm-chat
    restart: unless-stopped
    ipc: host
    volumes:
      - ../volumes/models:/root/.cache/huggingface
    environment:
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
    command:
      - "--model"
      - "${CHAT_MODEL}"
      - "--served-model-name"
      - "orgchat-chat"
      - "--max-model-len"
      - "8192"
      - "--gpu-memory-utilization"
      - "0.40"
      - "--enable-prefix-caching"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/v1/models || exit 1"]
      interval: 20s
      timeout: 5s
      retries: 15
      start_period: 180s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Rationale for `--gpu-memory-utilization 0.40`: on 32 GB, 0.40 ≈ 12.8 GB — fits Qwen2.5-14B-AWQ (~12 GB) with KV headroom. Vision and whisper get their own slices (below). Total: chat 0.40 + vision 0.25 (sleeping ≈ 0.03) + tei 0.10 + whisper 0.12 = 0.87; leaves ~10 % safety margin.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_vllm_chat_config.py tests/unit/test_compose_config.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add compose/docker-compose.yml tests/unit/test_vllm_chat_config.py
git commit -m "feat: add vllm-chat service to compose"
```

---

### Task 9: TEI embedding service in compose

**Files:**
- Modify: `/home/vogic/LocalRAG/compose/docker-compose.yml`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_tei_config.py`:

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_tei_block_present():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    assert "tei:" in content
    assert "ghcr.io/huggingface/text-embeddings-inference" in content
    assert "${EMBED_MODEL}" in content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_tei_config.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append TEI service to compose**

Append under `services:` in `/home/vogic/LocalRAG/compose/docker-compose.yml`:

```yaml
  tei:
    image: ghcr.io/huggingface/text-embeddings-inference:1.5
    container_name: orgchat-tei
    restart: unless-stopped
    command:
      - "--model-id"
      - "${EMBED_MODEL}"
      - "--port"
      - "80"
    volumes:
      - ../volumes/models:/data
    environment:
      HF_HUB_OFFLINE: "1"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:80/health || exit 1"]
      interval: 20s
      timeout: 5s
      retries: 10
      start_period: 60s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

- [ ] **Step 4: Run test**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_tei_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add compose/docker-compose.yml tests/unit/test_tei_config.py
git commit -m "feat: add TEI embedding service"
```

---

### Task 10: vllm-vision service with Sleep Mode

**Files:**
- Modify: `/home/vogic/LocalRAG/compose/docker-compose.yml`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_vllm_vision_config.py`:

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_vllm_vision_block_present():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    assert "vllm-vision:" in content
    assert "${VISION_MODEL}" in content
    # Sleep Mode flag is what makes on-demand loading cheap.
    assert "--enable-sleep-mode" in content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_vllm_vision_config.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append vllm-vision service**

Append under `services:` in `/home/vogic/LocalRAG/compose/docker-compose.yml`:

```yaml
  vllm-vision:
    image: vllm/vllm-openai:latest
    container_name: orgchat-vllm-vision
    restart: unless-stopped
    ipc: host
    volumes:
      - ../volumes/models:/root/.cache/huggingface
    environment:
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
    command:
      - "--model"
      - "${VISION_MODEL}"
      - "--served-model-name"
      - "orgchat-vision"
      - "--max-model-len"
      - "4096"
      - "--gpu-memory-utilization"
      - "0.25"
      - "--enable-sleep-mode"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/v1/models || exit 1"]
      interval: 20s
      timeout: 5s
      retries: 15
      start_period: 240s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

- [ ] **Step 4: Run test**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_vllm_vision_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add compose/docker-compose.yml tests/unit/test_vllm_vision_config.py
git commit -m "feat: add vllm-vision with sleep mode"
```

---

### Task 11: Whisper service scaffold (wrapper with sleep/wake endpoints)

**Files:**
- Create: `/home/vogic/LocalRAG/whisper_service/Dockerfile`
- Create: `/home/vogic/LocalRAG/whisper_service/requirements.txt`
- Create: `/home/vogic/LocalRAG/whisper_service/app.py`
- Modify: `/home/vogic/LocalRAG/compose/docker-compose.yml`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_whisper_wrapper.py`:

```python
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "whisper_service"))

def test_app_module_has_expected_endpoints():
    mod = importlib.import_module("app")
    routes = {r.path for r in mod.app.routes}
    for path in ["/health", "/sleep", "/wake_up", "/v1/audio/transcriptions"]:
        assert path in routes, f"endpoint missing: {path}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_whisper_wrapper.py -v
```

Expected: FAIL — `app` module missing.

- [ ] **Step 3: Write requirements + Dockerfile**

Create `/home/vogic/LocalRAG/whisper_service/requirements.txt`:

```
fastapi>=0.115
uvicorn[standard]>=0.30
faster-whisper>=1.0
python-multipart>=0.0.9
pydantic>=2.8
```

Create `/home/vogic/LocalRAG/whisper_service/Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV WHISPER_MODEL=medium
ENV WHISPER_CACHE=/models
ENV PORT=9000

HEALTHCHECK --interval=20s --timeout=5s --retries=10 --start-period=60s \
  CMD curl -sf http://localhost:${PORT}/health || exit 1

EXPOSE 9000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
```

- [ ] **Step 4: Write app.py**

Create `/home/vogic/LocalRAG/whisper_service/app.py`:

```python
"""Thin wrapper over faster-whisper with sleep/wake endpoints matching vLLM's contract."""
from __future__ import annotations

import asyncio
import gc
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger("whisper_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

MODEL_NAME  = os.environ.get("WHISPER_MODEL", "medium")
MODEL_CACHE = os.environ.get("WHISPER_CACHE", "/models")

_model = None          # faster_whisper.WhisperModel | None
_model_lock = asyncio.Lock()
_state: str = "asleep"  # "awake" | "asleep"


async def _load_to_gpu() -> None:
    global _model, _state
    from faster_whisper import WhisperModel
    logger.info("loading whisper %s to cuda", MODEL_NAME)
    _model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16",
                          download_root=MODEL_CACHE)
    _state = "awake"


async def _unload_from_gpu() -> None:
    global _model, _state
    logger.info("unloading whisper from cuda")
    _model = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    _state = "asleep"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start asleep; wake on demand via /wake_up.
    yield
    async with _model_lock:
        await _unload_from_gpu()


app = FastAPI(title="whisper_service", lifespan=lifespan)


class StateResponse(BaseModel):
    state: str
    model: str


@app.get("/health", response_model=StateResponse)
async def health() -> StateResponse:
    return StateResponse(state=_state, model=MODEL_NAME)


@app.post("/wake_up", response_model=StateResponse)
async def wake_up() -> StateResponse:
    async with _model_lock:
        if _state == "asleep":
            await _load_to_gpu()
    return StateResponse(state=_state, model=MODEL_NAME)


@app.post("/sleep", response_model=StateResponse)
async def sleep() -> StateResponse:
    async with _model_lock:
        if _state == "awake":
            await _unload_from_gpu()
    return StateResponse(state=_state, model=MODEL_NAME)


class TranscriptionResponse(BaseModel):
    text: str


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    # Auto-wake if sleeping (the model-manager normally calls /wake_up first; this is a safety net).
    async with _model_lock:
        if _state == "asleep":
            await _load_to_gpu()
        current_model = _model
    if current_model is None:
        raise HTTPException(status_code=503, detail="model failed to load")

    content = await file.read()
    tmp_path = f"/tmp/{file.filename or 'upload.wav'}"
    with open(tmp_path, "wb") as fh:
        fh.write(content)

    segments, _info = current_model.transcribe(tmp_path, beam_size=5)
    text = "".join(seg.text for seg in segments).strip()
    os.remove(tmp_path)
    return TranscriptionResponse(text=text)
```

- [ ] **Step 5: Append whisper service to compose**

Append under `services:` in `/home/vogic/LocalRAG/compose/docker-compose.yml`:

```yaml
  whisper:
    build:
      context: ../whisper_service
    container_name: orgchat-whisper
    restart: unless-stopped
    environment:
      WHISPER_MODEL: ${WHISPER_MODEL}
      WHISPER_CACHE: /models
      PORT: "9000"
    volumes:
      - ../volumes/models:/models
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9000/health || exit 1"]
      interval: 20s
      timeout: 5s
      retries: 10
      start_period: 30s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

- [ ] **Step 6: Run test**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate
pip install faster-whisper  # installs dep so the test can import the module
pytest tests/unit/test_whisper_wrapper.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/vogic/LocalRAG
git add whisper_service/ compose/docker-compose.yml tests/unit/test_whisper_wrapper.py
git commit -m "feat: whisper wrapper with sleep/wake endpoints"
```

---

### Task 12: KB schema migration SQL (drops upstream knowledge, creates kb_*)

**Files:**
- Create: `/home/vogic/LocalRAG/ext/db/migrations/001_create_kb_schema.sql`

- [ ] **Step 1: Write the failing integration test**

Create `/home/vogic/LocalRAG/tests/integration/test_kb_migration.py`:

```python
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer
from pathlib import Path

MIGRATION = Path(__file__).resolve().parents[2] / "ext/db/migrations/001_create_kb_schema.sql"


@pytest.fixture(scope="module")
def pg():
    with PostgresContainer("postgres:15-alpine") as pg:
        yield pg


@pytest.mark.asyncio
async def test_migration_creates_kb_tables(pg):
    async_url = pg.get_connection_url().replace("psycopg2", "asyncpg")
    engine = create_async_engine(async_url)

    # Simulate upstream schema: create minimal users, groups, knowledge tables we are replacing.
    async with engine.begin() as conn:
        await conn.execute(text("""
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
        """))

    sql = MIGRATION.read_text()
    async with engine.begin() as conn:
        await conn.execute(text(sql))

    async with engine.connect() as conn:
        tables = (await conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ))).scalars().all()
    assert "knowledge_bases" in tables
    assert "kb_subtags" in tables
    assert "kb_documents" in tables
    assert "kb_access" in tables
    # Upstream knowledge was dropped.
    assert "knowledge" not in tables
    # Chats got the new JSONB field.
    async with engine.connect() as conn:
        cols = (await conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='chats'"
        ))).scalars().all()
    assert "selected_kb_config" in cols

    await engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate
pytest tests/integration/test_kb_migration.py -v
```

Expected: FAIL — migration file missing.

- [ ] **Step 3: Write the migration**

Create `/home/vogic/LocalRAG/ext/db/migrations/001_create_kb_schema.sql`:

```sql
-- 001_create_kb_schema.sql
-- Replaces Open WebUI's upstream `knowledge` schema with hierarchical KB tables.
-- Idempotent: safe to re-run.

BEGIN;

-- Drop upstream knowledge tables if present (D2).
DROP TABLE IF EXISTS knowledge_file CASCADE;
DROP TABLE IF EXISTS knowledge CASCADE;

CREATE TABLE IF NOT EXISTS knowledge_bases (
  id          BIGSERIAL PRIMARY KEY,
  name        VARCHAR(255) NOT NULL,
  description TEXT,
  admin_id    BIGINT       NOT NULL REFERENCES users(id),
  created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
  UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS kb_subtags (
  id          BIGSERIAL PRIMARY KEY,
  kb_id       BIGINT       NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  name        VARCHAR(255) NOT NULL,
  description TEXT,
  created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
  UNIQUE(kb_id, name)
);

CREATE TABLE IF NOT EXISTS kb_documents (
  id             BIGSERIAL PRIMARY KEY,
  kb_id          BIGINT       NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  subtag_id      BIGINT       NOT NULL REFERENCES kb_subtags(id)       ON DELETE CASCADE,
  filename       VARCHAR(512) NOT NULL,
  mime_type      VARCHAR(100),
  bytes          BIGINT,
  ingest_status  VARCHAR(20)  NOT NULL DEFAULT 'pending'
                 CHECK (ingest_status IN ('pending','chunking','embedding','done','failed')),
  error_message  TEXT,
  uploaded_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
  uploaded_by    BIGINT       NOT NULL REFERENCES users(id),
  deleted_at     TIMESTAMPTZ  -- soft delete
);
CREATE INDEX IF NOT EXISTS idx_kb_documents_kb      ON kb_documents(kb_id)      WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_kb_documents_subtag  ON kb_documents(subtag_id)  WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_kb_documents_status  ON kb_documents(ingest_status) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS kb_access (
  id          BIGSERIAL PRIMARY KEY,
  kb_id       BIGINT      NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  user_id     BIGINT      REFERENCES users(id)  ON DELETE CASCADE,
  group_id    BIGINT      REFERENCES groups(id) ON DELETE CASCADE,
  access_type VARCHAR(20) NOT NULL DEFAULT 'read'
              CHECK (access_type IN ('read','write')),
  granted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK ( (user_id IS NOT NULL)::int + (group_id IS NOT NULL)::int = 1 )
);
CREATE INDEX IF NOT EXISTS idx_kb_access_user  ON kb_access(user_id)  WHERE user_id  IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kb_access_group ON kb_access(group_id) WHERE group_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kb_access_kb    ON kb_access(kb_id);

-- Extend chats with per-session KB config (§3.2 of workflow spec).
ALTER TABLE chats ADD COLUMN IF NOT EXISTS selected_kb_config JSONB;
CREATE INDEX IF NOT EXISTS idx_chats_kb_config ON chats USING GIN (selected_kb_config)
  WHERE selected_kb_config IS NOT NULL;

COMMIT;
```

- [ ] **Step 4: Run test — expect pass**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate
pytest tests/integration/test_kb_migration.py -v
```

Expected: PASS (pulls `postgres:15-alpine` via testcontainers).

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add ext/db/migrations/001_create_kb_schema.sql tests/integration/test_kb_migration.py
git commit -m "feat: KB schema migration — replaces upstream knowledge tables"
```

---

### Task 13: SQLAlchemy KB models

**Files:**
- Create: `/home/vogic/LocalRAG/ext/db/models/kb.py`
- Modify: `/home/vogic/LocalRAG/ext/db/models/__init__.py`
- Create: `/home/vogic/LocalRAG/ext/db/base.py`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_kb_models.py`:

```python
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from ext.db.base import Base
from ext.db.models.kb import KnowledgeBase, KBSubtag, KBDocument, KBAccess


@pytest.mark.asyncio
async def test_kb_models_create_and_query():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        # Stub minimal users/groups for FKs (KB models ref them).
        from sqlalchemy import text
        await conn.execute(text(
            "CREATE TABLE users  (id INTEGER PRIMARY KEY, email TEXT)"))
        await conn.execute(text(
            "CREATE TABLE groups (id INTEGER PRIMARY KEY, name  TEXT)"))
        await conn.execute(text("INSERT INTO users(id,email)  VALUES (1,'a@x')"))
        await conn.execute(text("INSERT INTO groups(id,name)  VALUES (1,'eng')"))
        await conn.run_sync(Base.metadata.create_all)

    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        kb = KnowledgeBase(name="Engineering", description="Eng docs", admin_id=1)
        s.add(kb); await s.flush()
        sub = KBSubtag(kb_id=kb.id, name="OFC")
        s.add(sub); await s.flush()
        doc = KBDocument(kb_id=kb.id, subtag_id=sub.id, filename="a.pdf", uploaded_by=1)
        s.add(doc)
        acc = KBAccess(kb_id=kb.id, group_id=1, access_type="read")
        s.add(acc)
        await s.commit()

        kbs = (await s.execute(select(KnowledgeBase))).scalars().all()
        assert len(kbs) == 1
        assert kbs[0].name == "Engineering"

    await engine.dispose()


def test_kb_access_check_enforced_in_model():
    # Exactly one of user_id / group_id — model-level check.
    with pytest.raises(ValueError):
        KBAccess(kb_id=1, user_id=None, group_id=None)
    with pytest.raises(ValueError):
        KBAccess(kb_id=1, user_id=1,    group_id=1)
```

Install `aiosqlite` for the in-memory test:

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pip install aiosqlite
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_kb_models.py -v
```

Expected: FAIL — models/base missing.

- [ ] **Step 3: Create base + models**

Create `/home/vogic/LocalRAG/ext/db/base.py`:

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

Create `/home/vogic/LocalRAG/ext/db/models/kb.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, String, Text, ForeignKey, CheckConstraint, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from ..base import Base


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id:          Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name:        Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    admin_id:    Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    subtags:   Mapped[list["KBSubtag"]]   = relationship(back_populates="kb", cascade="all, delete-orphan")
    documents: Mapped[list["KBDocument"]] = relationship(back_populates="kb", cascade="all, delete-orphan")
    access:    Mapped[list["KBAccess"]]   = relationship(back_populates="kb", cascade="all, delete-orphan")


class KBSubtag(Base):
    __tablename__ = "kb_subtags"
    __table_args__ = (CheckConstraint("length(name) > 0", name="subtag_name_nonempty"),)

    id:          Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kb_id:       Mapped[int] = mapped_column(BigInteger, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    name:        Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    kb: Mapped[KnowledgeBase] = relationship(back_populates="subtags")


class KBDocument(Base):
    __tablename__ = "kb_documents"

    id:             Mapped[int]    = mapped_column(BigInteger, primary_key=True)
    kb_id:          Mapped[int]    = mapped_column(BigInteger, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    subtag_id:      Mapped[int]    = mapped_column(BigInteger, ForeignKey("kb_subtags.id",       ondelete="CASCADE"), nullable=False)
    filename:       Mapped[str]    = mapped_column(String(512), nullable=False)
    mime_type:      Mapped[Optional[str]] = mapped_column(String(100))
    bytes:          Mapped[Optional[int]] = mapped_column(BigInteger)
    ingest_status:  Mapped[str]    = mapped_column(String(20), default="pending", nullable=False)
    error_message:  Mapped[Optional[str]] = mapped_column(Text)
    uploaded_at:    Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    uploaded_by:    Mapped[int]    = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    deleted_at:     Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    kb: Mapped[KnowledgeBase] = relationship(back_populates="documents")


class KBAccess(Base):
    __tablename__ = "kb_access"

    id:          Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kb_id:       Mapped[int] = mapped_column(BigInteger, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    user_id:     Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id",  ondelete="CASCADE"))
    group_id:    Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("groups.id", ondelete="CASCADE"))
    access_type: Mapped[str] = mapped_column(String(20), default="read", nullable=False)
    granted_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    kb: Mapped[KnowledgeBase] = relationship(back_populates="access")

    def __init__(self, **kwargs):
        # Enforce "exactly one of user_id / group_id" at model level too (defense in depth).
        u = kwargs.get("user_id"); g = kwargs.get("group_id")
        if (u is None) == (g is None):
            raise ValueError("KBAccess requires exactly one of user_id or group_id")
        super().__init__(**kwargs)
```

Update `/home/vogic/LocalRAG/ext/db/models/__init__.py`:

```python
from .kb import KnowledgeBase, KBSubtag, KBDocument, KBAccess

__all__ = ["KnowledgeBase", "KBSubtag", "KBDocument", "KBAccess"]
```

- [ ] **Step 4: Run test — expect pass**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_kb_models.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add ext/db/ tests/unit/test_kb_models.py
git commit -m "feat: SQLAlchemy KB models (KnowledgeBase, KBSubtag, KBDocument, KBAccess)"
```

---

### Task 14: Chat extension — `selected_kb_config` helper + validator

**Files:**
- Create: `/home/vogic/LocalRAG/ext/db/models/chat_ext.py`
- Modify: `/home/vogic/LocalRAG/ext/db/models/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_chat_kb_config.py`:

```python
import pytest
from ext.db.models.chat_ext import validate_selected_kb_config, SelectedKBConfig

def test_none_is_valid():
    assert validate_selected_kb_config(None) is None

def test_valid_shape():
    cfg = [{"kb_id": 5, "subtag_ids": [12, 13]}, {"kb_id": 7, "subtag_ids": []}]
    got = validate_selected_kb_config(cfg)
    assert isinstance(got, list)
    assert got[0].kb_id == 5 and got[0].subtag_ids == [12, 13]
    assert got[1].kb_id == 7 and got[1].subtag_ids == []

@pytest.mark.parametrize("bad", [
    [{"kb_id": "five"}],                         # kb_id not int
    [{"subtag_ids": [1]}],                        # missing kb_id
    [{"kb_id": 1, "subtag_ids": [1, "two"]}],     # non-int subtag
    "not a list",                                  # top-level not list
    [{"kb_id": 1, "subtag_ids": [1, 1]}],         # duplicate subtag
])
def test_invalid_shapes_rejected(bad):
    with pytest.raises(ValueError):
        validate_selected_kb_config(bad)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_chat_kb_config.py -v
```

Expected: FAIL — module missing.

- [ ] **Step 3: Write chat_ext.py**

Create `/home/vogic/LocalRAG/ext/db/models/chat_ext.py`:

```python
"""Helpers for the `chats.selected_kb_config` JSONB column.

Shape (per workflow spec §3.2):
    [ {"kb_id": int, "subtag_ids": [int, ...]}, ... ]
    empty subtag_ids = "all subtags in this KB"
    None/missing      = "no KB selected; private docs only"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class SelectedKBConfig:
    kb_id: int
    subtag_ids: List[int] = field(default_factory=list)


def validate_selected_kb_config(raw: Any) -> Optional[List[SelectedKBConfig]]:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("selected_kb_config must be a list or null")

    result: List[SelectedKBConfig] = []
    seen_kb_ids: set[int] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {i} is not an object")
        if "kb_id" not in entry:
            raise ValueError(f"entry {i} missing kb_id")
        kb_id = entry["kb_id"]
        if not isinstance(kb_id, int) or isinstance(kb_id, bool):
            raise ValueError(f"entry {i} kb_id must be an int")
        if kb_id in seen_kb_ids:
            raise ValueError(f"entry {i} duplicate kb_id={kb_id}")
        seen_kb_ids.add(kb_id)

        subtag_ids = entry.get("subtag_ids", [])
        if not isinstance(subtag_ids, list):
            raise ValueError(f"entry {i} subtag_ids must be a list")
        for j, sid in enumerate(subtag_ids):
            if not isinstance(sid, int) or isinstance(sid, bool):
                raise ValueError(f"entry {i} subtag_ids[{j}] must be an int")
        if len(set(subtag_ids)) != len(subtag_ids):
            raise ValueError(f"entry {i} subtag_ids contains duplicates")

        result.append(SelectedKBConfig(kb_id=kb_id, subtag_ids=list(subtag_ids)))
    return result
```

Update `/home/vogic/LocalRAG/ext/db/models/__init__.py`:

```python
from .kb import KnowledgeBase, KBSubtag, KBDocument, KBAccess
from .chat_ext import SelectedKBConfig, validate_selected_kb_config

__all__ = [
    "KnowledgeBase", "KBSubtag", "KBDocument", "KBAccess",
    "SelectedKBConfig", "validate_selected_kb_config",
]
```

- [ ] **Step 4: Run test — expect pass**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_chat_kb_config.py -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add ext/db/models/chat_ext.py ext/db/models/__init__.py tests/unit/test_chat_kb_config.py
git commit -m "feat: selected_kb_config validator for chats"
```

---

### Task 15: Preflight model-weight check

**Files:**
- Create: `/home/vogic/LocalRAG/scripts/preflight_models.py`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_preflight.py`:

```python
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "preflight_models.py"

def test_script_dryrun_prints_plan():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True, text=True,
        env={"EMBED_MODEL": "BAAI/bge-m3",
             "CHAT_MODEL":  "Qwen/Qwen2.5-14B-Instruct-AWQ",
             "VISION_MODEL":"Qwen/Qwen2-VL-7B-Instruct",
             "WHISPER_MODEL":"medium",
             "MODEL_CACHE": "/tmp/orgchat-models",
             "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0, r.stderr
    for m in ["Qwen/Qwen2.5-14B-Instruct-AWQ", "Qwen/Qwen2-VL-7B-Instruct",
              "BAAI/bge-m3", "medium"]:
        assert m in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_preflight.py -v
```

Expected: FAIL — script missing.

- [ ] **Step 3: Write the script**

Create `/home/vogic/LocalRAG/scripts/preflight_models.py`:

```python
#!/usr/bin/env python3
"""Download + verify the four model weights used by the stack.

Runs ONCE on a machine with internet access; results are cached into
$MODEL_CACHE (default: ./volumes/models) and reused by all services
with HF_HUB_OFFLINE=1 thereafter.

Usage:
    python scripts/preflight_models.py              # download everything
    python scripts/preflight_models.py --dry-run    # print plan, download nothing
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _plan() -> list[tuple[str, str]]:
    """Returns (kind, identifier) pairs."""
    return [
        ("hf", _env("CHAT_MODEL",   "Qwen/Qwen2.5-14B-Instruct-AWQ")),
        ("hf", _env("VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")),
        ("hf", _env("EMBED_MODEL",  "BAAI/bge-m3")),
        ("whisper", _env("WHISPER_MODEL", "medium")),
    ]


def _free_disk_gb(path: Path) -> float:
    total, used, free = shutil.disk_usage(str(path) if path.exists() else str(path.parent))
    return free / (1024 ** 3)


def _download_hf(repo_id: str, cache_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir), resume_download=True)


def _download_whisper(size: str, cache_dir: Path) -> None:
    from faster_whisper import WhisperModel
    WhisperModel(size, download_root=str(cache_dir))  # triggers download


def run(plan: Iterable[tuple[str, str]], cache_dir: Path, dry_run: bool) -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    free = _free_disk_gb(cache_dir)
    print(f"cache: {cache_dir}   free disk: {free:.1f} GB")
    if free < 40 and not dry_run:
        print("!! less than 40 GB free; aborting", file=sys.stderr)
        return 2

    for kind, ident in plan:
        print(f"[{kind}] {ident}")
        if dry_run:
            continue
        if kind == "hf":
            _download_hf(ident, cache_dir)
        elif kind == "whisper":
            _download_whisper(ident, cache_dir)
        else:
            print(f"!! unknown kind {kind}", file=sys.stderr)
            return 3
    print("preflight OK")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache-dir", default=_env("MODEL_CACHE", "./volumes/models"))
    args = ap.parse_args(argv)
    return run(_plan(), Path(args.cache_dir), args.dry_run)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

Install `huggingface_hub` in dev env:

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pip install huggingface_hub
```

- [ ] **Step 4: Run test**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/unit/test_preflight.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add scripts/preflight_models.py tests/unit/test_preflight.py
git commit -m "feat: preflight script — verifies model weights before first compose up"
```

---

### Task 16: Admin seed bootstrap

**Files:**
- Create: `/home/vogic/LocalRAG/scripts/seed_admin.py`

- [ ] **Step 1: Write the failing integration test**

Create `/home/vogic/LocalRAG/tests/integration/test_seed_admin.py`:

```python
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "scripts" / "seed_admin.py"
MIGRATION = ROOT / "ext/db/migrations/001_create_kb_schema.sql"


@pytest.mark.asyncio
async def test_seed_admin_idempotent():
    with PostgresContainer("postgres:15-alpine") as pg:
        async_url = pg.get_connection_url().replace("psycopg2", "asyncpg")
        sync_url  = pg.get_connection_url()
        engine = create_async_engine(async_url)

        async with engine.begin() as conn:
            await conn.execute(text("""
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
            """))
            await conn.execute(text(MIGRATION.read_text()))

        env = os.environ.copy()
        env["DATABASE_URL"] = async_url
        env["ADMIN_EMAIL"]  = "admin@test.local"
        env["ADMIN_PASSWORD"] = "hunter2-hunter2"

        r1 = subprocess.run([sys.executable, str(SEED)], env=env, capture_output=True, text=True)
        assert r1.returncode == 0, r1.stderr
        r2 = subprocess.run([sys.executable, str(SEED)], env=env, capture_output=True, text=True)
        assert r2.returncode == 0, r2.stderr  # idempotent

        async with engine.connect() as conn:
            count = (await conn.execute(text(
                "SELECT COUNT(*) FROM users WHERE email=:e"), {"e": "admin@test.local"})).scalar()
        assert count == 1
        await engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/integration/test_seed_admin.py -v
```

Expected: FAIL — `scripts/seed_admin.py` missing.

- [ ] **Step 3: Write the script**

Create `/home/vogic/LocalRAG/scripts/seed_admin.py`:

```python
#!/usr/bin/env python3
"""Create the bootstrap admin user if one does not already exist.

Env vars:
    DATABASE_URL     postgresql+asyncpg://user:pass@host/db
    ADMIN_EMAIL      admin email (required)
    ADMIN_PASSWORD   plaintext; hashed with Argon2id (required)
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
```

- [ ] **Step 4: Run test**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest tests/integration/test_seed_admin.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/vogic/LocalRAG
git add scripts/seed_admin.py tests/integration/test_seed_admin.py
git commit -m "feat: admin seed bootstrap (Argon2id, idempotent)"
```

---

### Task 17: Compose-level smoke test + health gate

**Files:**
- Create: `/home/vogic/LocalRAG/tests/integration/test_compose_up.py`

> This task verifies the end-to-end Phase-1 deliverable: `make up && make smoke`. It requires the GPU host with Docker + NVIDIA runtime. If you're on a non-GPU dev box, set `SKIP_GPU_SMOKE=1` to restrict the check to stateful services only.

- [ ] **Step 1: Write the smoke test**

Create `/home/vogic/LocalRAG/tests/integration/test_compose_up.py`:

```python
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "compose/docker-compose.yml"),
           "--env-file", str(ROOT / "compose/.env")]

SKIP_GPU = os.environ.get("SKIP_GPU_SMOKE") == "1"

STATEFUL_HEALTHCHECKS = [
    ("postgres", ["docker", "exec", "orgchat-postgres", "pg_isready", "-U", "orgchat"]),
    ("redis",    ["docker", "exec", "orgchat-redis", "redis-cli", "ping"]),
]

HTTP_HEALTHCHECKS = [
    ("qdrant",  "http://localhost:6333/readyz"),
]

GPU_HTTP_HEALTHCHECKS = [
    ("tei",          "http://localhost:80/health"),       # only reachable via network; see note
    ("vllm-chat",    "http://localhost:8000/v1/models"),
    ("whisper",      "http://localhost:9000/health"),
]


def _wait(pred, timeout_s=300, interval_s=5):
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


@pytest.fixture(scope="module", autouse=True)
def compose_stack():
    if not (ROOT / "compose/.env").exists():
        (ROOT / "compose/.env").write_text((ROOT / "compose/.env.example").read_text())
    subprocess.run(COMPOSE + ["up", "-d"], check=True)
    yield
    subprocess.run(COMPOSE + ["down"], check=True)


def test_stateful_services_healthy():
    for name, cmd in STATEFUL_HEALTHCHECKS:
        ok = _wait(lambda: subprocess.run(cmd, capture_output=True).returncode == 0)
        assert ok, f"{name} never became healthy"


def test_qdrant_ready():
    ok = _wait(lambda: httpx.get("http://localhost:6333/readyz", timeout=2).status_code == 200)
    assert ok


@pytest.mark.skipif(SKIP_GPU, reason="GPU services skipped (SKIP_GPU_SMOKE=1)")
def test_gpu_services_healthy():
    # Stack needs a long start_period for vLLM; allow 10 min total.
    ok = _wait(
        lambda: httpx.get("http://localhost:8000/v1/models", timeout=2).status_code == 200,
        timeout_s=600,
    )
    assert ok, "vllm-chat never became healthy"
```

- [ ] **Step 2: Run smoke with GPU (on the target host)**

```bash
cd /home/vogic/LocalRAG
make preflight          # one-time, needs internet
make up
make smoke
```

Expected: all tests PASS on GPU host.

On a non-GPU dev box:

```bash
SKIP_GPU_SMOKE=1 make smoke
```

Expected: 3 PASSED, 1 SKIPPED.

- [ ] **Step 3: Commit**

```bash
cd /home/vogic/LocalRAG
git add tests/integration/test_compose_up.py
git commit -m "test: compose-up smoke test (Phase-1 acceptance gate)"
```

---

### Task 18: Phase-1 tag + handoff

**Files:** none (repo-level operation)

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && pytest -v
```

Expected: all tests PASS (GPU smoke skipped unless run on the host).

- [ ] **Step 2: Run linters**

```bash
cd /home/vogic/LocalRAG && source .venv/bin/activate && make lint
```

Expected: clean exit. Fix any reported issues and re-run before continuing.

- [ ] **Step 3: Tag Phase 1 complete**

```bash
cd /home/vogic/LocalRAG
git tag -a phase-1-foundation -m "Phase 1 complete: foundation, KB schema, admin seed, preflight"
```

- [ ] **Step 4: Commission Phase 2 plan**

Request Claude: *"Write the Phase 2 plan at `docs/superpowers/plans/2026-04-16-phase-2-kb-management-rbac.md` following the same template — bite-sized TDD tasks covering KB CRUD, subtag CRUD, kb_access grants, KB selection at chat start, RBAC enforcement middleware, and the acceptance gate (admin creates KB, grants group, member retrieves / non-member gets 403)."*

---

## Phase 1 acceptance checklist

Phase 1 is done when **all** of the following are true. Reviewer verifies before tagging `phase-1-foundation`.

- [ ] `git submodule status` shows `upstream/` at the tag recorded in `UPSTREAM_VERSION`.
- [ ] `make install && make test` passes (GPU smoke skipped on non-GPU dev box).
- [ ] `make preflight` on the GPU host completes without network errors.
- [ ] `make up` on the GPU host brings all nine services to `healthy` status within 10 min.
- [ ] `make smoke` on the GPU host passes with GPU checks enabled.
- [ ] `scripts/seed_admin.py` creates the admin user; second run is a no-op.
- [ ] `ext/db/migrations/001_create_kb_schema.sql` is idempotent (testcontainer re-run test green).
- [ ] Self-signed cert at `volumes/certs/orgchat.crt` served by Caddy on `https://$DOMAIN/`.
- [ ] Commit history shows one commit per task (18 commits) — useful for rebasing patches later.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-16-phase-1-foundation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (Tasks 1–18), review between tasks, fast iteration.
2. **Inline Execution** — We execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**
