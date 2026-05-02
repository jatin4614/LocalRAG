#!/usr/bin/env bash
# First-run bootstrap for Org Chat Assistant.
# Orchestrates: preflight -> cert -> patch -> compose up -> migrations -> admin seed.
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
for key in WEBUI_NAME DOMAIN SESSION_SECRET ADMIN_EMAIL ADMIN_PASSWORD WEBUI_SECRET_KEY POSTGRES_PASSWORD; do
  if ! grep -q "^${key}=" compose/.env; then
    echo "!! compose/.env missing key: $key" >&2; exit 1
  fi
done
set -a; source compose/.env; set +a

# Wave 1a (review §11.5 / §9): refuse to bootstrap when the placeholder
# `change-me-…` literals from .env.example survive. Previously bootstrap only
# verified the keys EXISTED — a blank-fill operator run would happily ship the
# defaults to production where Qdrant is reachable on the LAN with no auth and
# WEBUI_SECRET_KEY signs JWTs with a string everyone has the source for.
REJECTED=0
check_secret() {
  local key="$1" val="$2"
  if [[ "$val" == change-me-* ]] || [[ "$val" == "" ]]; then
    echo "!! compose/.env: $key is unset or still the placeholder ('$val')" >&2
    REJECTED=1
  fi
}
check_secret SESSION_SECRET    "${SESSION_SECRET:-}"
check_secret WEBUI_SECRET_KEY  "${WEBUI_SECRET_KEY:-}"
check_secret ADMIN_PASSWORD    "${ADMIN_PASSWORD:-}"
check_secret POSTGRES_PASSWORD "${POSTGRES_PASSWORD:-}"
# QDRANT_API_KEY is OPTIONAL — empty / unset means Qdrant runs without auth
# (current default). If set, it must NOT be the placeholder. If you want to
# enable Qdrant auth, generate one with `openssl rand -base64 32`.
if [[ -n "${QDRANT_API_KEY:-}" ]]; then
  check_secret QDRANT_API_KEY "$QDRANT_API_KEY"
fi
# DATABASE_URL embeds POSTGRES_PASSWORD; spot-check the literal too.
if [[ "${DATABASE_URL:-}" == *change-me-* ]]; then
  echo "!! compose/.env: DATABASE_URL still contains 'change-me-…' (rotate POSTGRES_PASSWORD AND DATABASE_URL together)" >&2
  REJECTED=1
fi
if [[ $REJECTED -eq 1 ]]; then
  echo "" >&2
  echo "Generate strong secrets:" >&2
  echo "  openssl rand -base64 32        # for SESSION_SECRET / WEBUI_SECRET_KEY / QDRANT_API_KEY" >&2
  echo "  openssl rand -base64 24        # for ADMIN_PASSWORD / POSTGRES_PASSWORD" >&2
  echo "Then edit compose/.env (and rebuild DATABASE_URL with the new pg password)." >&2
  exit 1
fi

say "Step 1/6: preflight model weights"
if [[ $SKIP_PULL -eq 0 ]]; then
  # Gate: if every expected model already exists in the cache, skip the (slow)
  # download. Otherwise run the full preflight. --verify-only exits 0 on hit,
  # 2 on miss; any other code indicates a real error we should not swallow.
  if [[ $DRY_RUN -eq 1 ]]; then
    do_run "python scripts/preflight_models.py --dry-run"
  else
    if python scripts/preflight_models.py --verify-only; then
      say "  preflight: all models present, skipping download"
    else
      rc=$?
      if [[ $rc -eq 2 ]]; then
        say "  preflight: missing models, downloading (this may take a while)"
        python scripts/preflight_models.py
      else
        echo "!! preflight --verify-only exited with $rc (not a missing-cache signal); aborting" >&2
        exit $rc
      fi
    fi
  fi
else
  say "  (--skip-pull: assuming weights already cached)"
fi

say "Step 2/6: generate self-signed cert (idempotent)"
do_run "bash scripts/gen_self_signed_cert.sh"

say "Step 3/6: upstream patches (vendored — no-op since 2026-04-30)"
say "  upstream/ is now a vendored directory; patches are pre-applied in the source."
say "  scripts/apply_patches.sh is kept only for re-deriving patches if/when"
say "  upstream Open WebUI is upgraded; it is not part of the bootstrap path."

say "Step 4/6: docker compose up -d"
do_run "docker compose -f compose/docker-compose.yml --env-file compose/.env up -d"

say "Step 5/6: wait for postgres, then apply migrations"
do_run "bash -c 'for i in {1..30}; do docker exec orgchat-postgres pg_isready -U \"\${POSTGRES_USER:-orgchat}\" 2>/dev/null && break; sleep 2; done'"
do_run "python scripts/apply_migrations.py"

say "Step 6/6: seed admin"
do_run "python scripts/seed_admin.py"

say "Bootstrap complete. Caddy is serving HTTPS on \${DOMAIN:-orgchat.lan}:443 with a self-signed cert."
say "If your host is fresh, add \${DOMAIN:-orgchat.lan} to /etc/hosts pointing to 127.0.0.1 for local testing."
