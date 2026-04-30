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
for key in WEBUI_NAME DOMAIN SESSION_SECRET ADMIN_EMAIL ADMIN_PASSWORD WEBUI_SECRET_KEY; do
  if ! grep -q "^${key}=" compose/.env; then
    echo "!! compose/.env missing key: $key" >&2; exit 1
  fi
done
set -a; source compose/.env; set +a

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
