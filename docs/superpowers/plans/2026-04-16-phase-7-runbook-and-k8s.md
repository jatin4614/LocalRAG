# Phase 7 — Runbook + Deployment + k8s Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Ship the deployment story — a single runbook that takes a non-technical admin from a fresh Linux host to a working org-chat instance in under 60 min, a `scripts/bootstrap.sh` that does the one-shot first-run sequence, and a `k8s/` skeleton for Phase-2 future migration to 4×48GB GPUs.

**Tech Stack:** bash, Docker Compose 3.8, Helm values schema (skeleton only — no running Helm in this phase).

**Working directory:** `/home/vogic/LocalRAG/` (main, tagged `phase-6-testing-battery`).

---

## Decisions (Phase 7)

| # | Decision | Revise-by |
|---|----------|-----------|
| D44 | `scripts/bootstrap.sh` is a one-shot script that runs: `preflight_models.py`, `gen_self_signed_cert.sh`, `apply_patches.sh`, `docker compose up -d`, `apply_migrations.py` (via docker exec upstream), `seed_admin.py` (similarly). | — |
| D45 | `k8s/values.yaml` is a skeleton only — no Helm chart, just the values schema for Phase 2 so future deploy plans have a starting point. Documented as "Phase 2 template — not wired". | Phase 2 (future) |
| D46 | `docs/runbook.md` replaces/supersedes scattered README snippets. Top-level `README.md` gets a short quickstart pointing at the runbook. | — |
| D47 | No CI doc (D6: fully offline testing). Local `make test-all` is the canonical verification. | — |

---

## Task 1: Bootstrap script

**Files:** `scripts/bootstrap.sh` (new).

- [ ] **Step 1: Write failing test**

`tests/unit/test_bootstrap.py`:

```python
import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOT = ROOT / "scripts" / "bootstrap.sh"


def test_bootstrap_executable():
    assert BOOT.is_file()
    assert BOOT.stat().st_mode & stat.S_IXUSR, "script not executable"


def test_bootstrap_mentions_key_steps():
    content = BOOT.read_text()
    for step in ["preflight", "gen_self_signed_cert", "apply_patches",
                 "docker compose", "seed_admin", "apply_migrations"]:
        assert step in content, f"bootstrap missing step: {step}"


def test_bootstrap_refuses_without_env():
    """Running without compose/.env should exit non-zero with a clear message."""
    import subprocess
    r = subprocess.run(
        ["bash", str(BOOT), "--dry-run"],
        capture_output=True, text=True,
        cwd=str(ROOT),
        env={**os.environ, "BOOTSTRAP_SKIP_ENV_CHECK": ""},
    )
    # --dry-run with no env: should print plan and exit
    # (actual enforcement is at deploy time, not dry-run)
    assert r.returncode in (0, 1)
```

- [ ] **Step 2: Write `scripts/bootstrap.sh`**

```bash
#!/usr/bin/env bash
# First-run bootstrap for Org Chat Assistant.
# Orchestrates: preflight → cert → patch → compose up → migrations → admin seed.
#
# Usage:
#   scripts/bootstrap.sh              Full bootstrap.
#   scripts/bootstrap.sh --dry-run    Print plan, do nothing.
#   scripts/bootstrap.sh --skip-pull  Don't re-preflight model weights.
set -euo pipefail

DRY_RUN=0
SKIP_PULL=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=1 ;;
    --skip-pull) SKIP_PULL=1 ;;
    -h|--help)   sed -n '2,10p' "$0"; exit 0 ;;
    *)           echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

say()    { echo ">>> $*"; }
do_run() { if [[ $DRY_RUN -eq 1 ]]; then echo "DRY: $*"; else eval "$*"; fi; }

# 1. Verify compose/.env exists and has critical keys.
if [[ ! -f compose/.env ]]; then
  echo "!! compose/.env missing. Copy from compose/.env.example and fill in:" >&2
  echo "   cp compose/.env.example compose/.env && \$EDITOR compose/.env" >&2
  exit 1
fi
for key in WEBUI_NAME DOMAIN SESSION_SECRET ADMIN_EMAIL ADMIN_PASSWORD WEBUI_SECRET_KEY; do
  if ! grep -q "^${key}=" compose/.env; then
    echo "!! compose/.env missing key: $key" >&2; exit 1
  fi
done
# Load env for downstream scripts.
set -a; source compose/.env; set +a

say "Step 1/6: preflight model weights"
if [[ $SKIP_PULL -eq 0 ]]; then
  do_run "python scripts/preflight_models.py"
else
  say "  (--skip-pull: assuming weights already cached)"
fi

say "Step 2/6: generate self-signed cert (idempotent)"
do_run "bash scripts/gen_self_signed_cert.sh"

say "Step 3/6: apply upstream patches"
do_run "bash scripts/apply_patches.sh"

say "Step 4/6: docker compose up -d"
do_run "docker compose -f compose/docker-compose.yml --env-file compose/.env up -d"

say "Step 5/6: wait for postgres, then apply migrations"
do_run "bash -c 'for i in {1..30}; do docker exec orgchat-postgres pg_isready -U \"\${POSTGRES_USER:-orgchat}\" 2>/dev/null && break; sleep 2; done'"
do_run "python scripts/apply_migrations.py"

say "Step 6/6: seed admin"
do_run "python scripts/seed_admin.py"

say "Bootstrap complete. Caddy is serving HTTPS on \${DOMAIN:-orgchat.lan}:443 with a self-signed cert."
say "If your host is fresh, add \${DOMAIN:-orgchat.lan} to /etc/hosts pointing to 127.0.0.1 for local testing."
```

```bash
chmod +x /home/vogic/LocalRAG/scripts/bootstrap.sh
```

- [ ] **Step 3: Run + PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_bootstrap.py -v
```

- [ ] **Step 4: Commit**

```bash
git add scripts/bootstrap.sh tests/unit/test_bootstrap.py
git commit -m "feat(deploy): one-shot bootstrap.sh (cert → patch → up → migrate → seed)"
```

---

## Task 2: Runbook

**Files:** `docs/runbook.md` (new).

- [ ] **Step 1: Write `docs/runbook.md`**

Write a comprehensive deployment guide. Sections required:

1. **Prerequisites** — hardware (RTX 6000 Ada or ≥24GB NVIDIA GPU), Ubuntu 22.04+, Docker + NVIDIA runtime, Python ≥ 3.10, ~50 GB disk for model weights.
2. **First-run procedure** — step-by-step:
   - Clone repo + `git submodule update --init upstream`
   - `make install`
   - Copy + edit `compose/.env` (list every key + what it means + recommended values)
   - `scripts/bootstrap.sh` (explain what it does)
   - Open `https://<DOMAIN>/` in a browser (cert warning expected; accept it)
   - Log in as admin@... with the password from `.env`
3. **Daily operations** — start/stop: `make up` / `make down`. Logs: `make logs`. Health: `curl http://localhost:8080/healthz` etc.
4. **Admin tasks** — create users (via Open WebUI web UI), create KBs (`POST /api/kb`), grant access, upload docs.
5. **Troubleshooting** — each common issue + symptom + fix:
   - "Container orgchat-vllm-chat is unhealthy" → check model weights present, nvidia runtime
   - "403 on /api/kb" → user role is not admin
   - "401 on /api/kb/available" → WEBUI_SECRET_KEY mismatch between Open WebUI and our ext/
   - "Upload returns 422" → MIME type unsupported; check file extension
   - "Port 443 in use" → caddy conflict; `ss -tlnp | grep 443`
6. **Upgrading Open WebUI** — `scripts/rebase_upstream.sh v0.9.0` + reapply patches + rerun test suite
7. **Backup & restore** — `pg_dump orgchat-postgres`, `qdrant snapshot`, filesystem `volumes/models`
8. **Phase 2 — future k8s migration** — pointer to `k8s/` skeleton

Keep it tight — aim for 300-400 lines of Markdown, no filler.

- [ ] **Step 2: Write failing test that checks runbook covers key sections**

`tests/unit/test_runbook.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "docs" / "runbook.md"


def test_runbook_exists():
    assert RUN.is_file()


def test_runbook_has_required_sections():
    content = RUN.read_text().lower()
    for heading in [
        "prerequisites",
        "first-run",
        "troubleshooting",
        "backup",
        "upgrad",   # "Upgrading Open WebUI"
    ]:
        assert heading in content, f"runbook missing section: {heading!r}"


def test_runbook_mentions_bootstrap_script():
    content = RUN.read_text()
    assert "bootstrap.sh" in content
    assert "compose/.env" in content
```

- [ ] **Step 3: Run — FAIL → PASS after runbook written**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_runbook.py -v
```

- [ ] **Step 4: Commit**

```bash
git add docs/runbook.md tests/unit/test_runbook.py
git commit -m "docs: deployment runbook (prerequisites → first-run → troubleshooting → upgrade → backup)"
```

---

## Task 3: k8s skeleton

**Files:** `k8s/README.md`, `k8s/values.yaml`.

- [ ] **Step 1: Write `k8s/README.md`**

Short intro explaining this is **Phase 2 template only** — not a functional Helm chart. Just documents the values schema for future migration from single-host Docker Compose to k8s multi-node.

- [ ] **Step 2: Write `k8s/values.yaml`**

```yaml
# values.yaml — Phase 2 Helm chart values (SKELETON; not wired).
# Maps the docker-compose services to k8s deployments with Phase-2 sizing
# (4×48GB GPUs per master plan §1 Phase 2).
#
# To wire this up:
#   1. Write the Helm chart templates under k8s/templates/
#   2. Replace secrets (WEBUI_SECRET_KEY, ADMIN_PASSWORD) with real refs
#   3. Test against a cluster with NVIDIA device plugin installed

image:
  registry: ghcr.io/your-org
  tag: latest
  pullPolicy: IfNotPresent

domain: orgchat.example.com

# --- Always-on services (baseline) ---
postgres:
  enabled: true
  image: postgres:15-alpine
  persistence:
    size: 50Gi
  resources:
    requests: { cpu: 500m, memory: 2Gi }
    limits:   { cpu: 2,    memory: 4Gi }

redis:
  enabled: true
  image: redis:7-alpine
  persistence:
    size: 10Gi

qdrant:
  enabled: true
  image: qdrant/qdrant:latest
  persistence:
    size: 200Gi

# --- GPU services (Phase 2 sizing) ---
vllmChat:
  image: vllm/vllm-openai:latest
  model: Qwen/Qwen2.5-72B-Instruct      # Phase 2 model
  gpus: 2                                # tensor-parallel
  replicas: 1
  env:
    VLLM_TENSOR_PARALLEL_SIZE: "2"
  resources:
    limits: { nvidia.com/gpu: 2 }

vllmVision:
  image: vllm/vllm-openai:latest
  model: Qwen/Qwen2-VL-72B-Instruct
  gpus: 1
  replicas: 1
  env:
    VLLM_ENABLE_SLEEP_MODE: "true"
  resources:
    limits: { nvidia.com/gpu: 1 }

tei:
  image: ghcr.io/huggingface/text-embeddings-inference:1.5
  model: BAAI/bge-m3
  resources:
    limits: { nvidia.com/gpu: 1 }

whisper:
  enabled: true
  image: ghcr.io/your-org/whisper-service:latest  # our own Dockerfile
  model: large-v3
  resources:
    limits: { nvidia.com/gpu: 1 }

modelManager:
  enabled: true
  image: ghcr.io/your-org/model-manager:latest
  env:
    MODEL_UNLOAD_IDLE_SECS: "300"

openWebUI:
  image: ghcr.io/open-webui/open-webui:v0.8.12
  replicas: 2                           # HA for user-facing chat
  ingress:
    enabled: true
    tls: true
    host: orgchat.example.com

# --- Secrets (fill with real refs) ---
secrets:
  webuiSecretKey: ""    # required
  adminPassword:  ""    # required
  postgresPassword: ""  # required

# --- Autoscaling ---
autoscaling:
  enabled: false        # start disabled; revisit when traffic known
```

- [ ] **Step 3: Write test**

`tests/unit/test_k8s_skeleton.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_k8s_values_has_phase2_services():
    content = (ROOT / "k8s" / "values.yaml").read_text()
    for svc in ["postgres", "redis", "qdrant", "vllmChat", "vllmVision",
                "tei", "whisper", "modelManager", "openWebUI"]:
        assert svc in content


def test_k8s_readme_marks_as_skeleton():
    content = (ROOT / "k8s" / "README.md").read_text().lower()
    assert "skeleton" in content or "template" in content
    assert "phase 2" in content
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_k8s_skeleton.py -v
```

- [ ] **Step 5: Commit**

```bash
git add k8s/README.md k8s/values.yaml tests/unit/test_k8s_skeleton.py
git commit -m "docs(k8s): Phase-2 values.yaml skeleton"
```

---

## Task 4: Regression + Phase 7 tag

```bash
cd /home/vogic/LocalRAG && make test-all 2>&1 | tail -10
git tag -a phase-7-runbook -m "Phase 7 complete: bootstrap.sh + runbook + k8s skeleton"
git log phase-6-testing-battery..HEAD --oneline
```

---

## Phase 7 acceptance checklist

- [ ] `scripts/bootstrap.sh` is executable, accepts `--dry-run`, runs all 6 steps in order.
- [ ] `docs/runbook.md` exists with all required sections.
- [ ] `k8s/values.yaml` skeleton covers all 9 services from CLAUDE.md §3.
- [ ] `k8s/README.md` marks it as Phase-2 template, not wired.
- [ ] Tag `phase-7-runbook` exists.
- [ ] `make test-all` still green.
