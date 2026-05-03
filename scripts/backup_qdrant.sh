#!/usr/bin/env bash
#
# Wave 3a (review §11.6) — Qdrant snapshot script (host-side, ad-hoc).
#
# Operators run this BEFORE risky changes (pre-wave-N protocol from the
# bug-fix campaign tracker). The Celery beat in
# `ext/workers/snapshot_task.py` SCHEDULES inside-container snapshots on
# a daily cadence; THIS script is for the operator-on-the-host case
# (e.g. the rollback registry says "snapshot now before applying
# migration 013").
#
# Both pathways must work — the celery task can fail (qdrant down,
# beat-state corrupt, ...) and the host-side script gives the operator
# an unambiguous, no-Celery-required way to capture state.
#
# Usage:
#   scripts/backup_qdrant.sh                                  # ./backups/qdrant, retain 14d
#   scripts/backup_qdrant.sh --dest /mnt/nas/qdrant
#   scripts/backup_qdrant.sh --retention-days 30
#   scripts/backup_qdrant.sh --dry-run
#
# Reads QDRANT_URL + QDRANT_API_KEY from compose/.env; defaults to
# http://localhost:6333 (the loopback-bound port from Wave 1a §11.2).
# If QDRANT_API_KEY is set the `api-key` header is sent on every
# request (Wave 1a per-request auth).
#
# Per-collection snapshot is best-effort: a failure on kb_5 must not
# stop kb_1 + kb_2 from snapshotting. Final exit code is 1 if ANY
# collection failed (cron wrapper alerts on non-zero).
#
set -euo pipefail

# ---- defaults ------------------------------------------------------------
DEST="./backups/qdrant"
RETENTION_DAYS=14
DRY_RUN=0

# ---- arg parse -----------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --retention-days)
      RETENTION_DAYS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    -h|--help)
      sed -n '3,28p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "Run with --help for usage." >&2
      exit 1
      ;;
  esac
done

# ---- helpers -------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---- load env ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../compose/.env"

QDRANT_URL_DEFAULT="http://localhost:6333"
# Env-var precedence: caller's exported QDRANT_URL / QDRANT_API_KEY wins
# over compose/.env (so operator can point at staging without editing
# the deploy file). Empty / unset → fall through to compose/.env.
QDRANT_URL="${QDRANT_URL:-}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"

if [ -f "$ENV_FILE" ]; then
  # Pull single keys without sourcing the whole file (keeps secret hygiene).
  if [ -z "$QDRANT_URL" ]; then
    _qu="$(grep -E '^QDRANT_URL=' "$ENV_FILE" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
    [ -n "$_qu" ] && QDRANT_URL="$_qu"
  fi
  if [ -z "$QDRANT_API_KEY" ]; then
    _qk="$(grep -E '^QDRANT_API_KEY=' "$ENV_FILE" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
    [ -n "$_qk" ] && QDRANT_API_KEY="$_qk"
  fi
fi

[ -n "$QDRANT_URL" ] || QDRANT_URL="$QDRANT_URL_DEFAULT"

# Inside-compose hostname is not resolvable from the host. Rewrite to
# loopback (the qdrant port is bound to 127.0.0.1:6333 since Wave 1a §11.2).
case "$QDRANT_URL" in
  *://qdrant:*|*://orgchat-qdrant:*)
    REWRITTEN="${QDRANT_URL/qdrant/127.0.0.1}"
    REWRITTEN="${REWRITTEN/orgchat-127.0.0.1/127.0.0.1}"
    log "rewriting QDRANT_URL '$QDRANT_URL' → '$REWRITTEN' (compose service not resolvable from host)"
    QDRANT_URL="$REWRITTEN"
    ;;
esac

# ---- dependency checks ---------------------------------------------------
command -v curl >/dev/null 2>&1 || die "curl not on PATH"
command -v jq   >/dev/null 2>&1 || die "jq not on PATH (apt install jq)"

# Build a curl auth-header array we can splat into every call.
CURL_AUTH=()
if [ -n "$QDRANT_API_KEY" ]; then
  CURL_AUTH=(-H "api-key: $QDRANT_API_KEY")
fi

# ---- prepare destination -------------------------------------------------
TS="$(date -u +%F-%H%M)"
OUT="$DEST/$TS"

log "qdrant url:  $QDRANT_URL"
log "destination: $OUT"
log "retention:   ${RETENTION_DAYS}d"
log "auth:        $([ -n "$QDRANT_API_KEY" ] && echo enabled || echo disabled)"

if [ "$DRY_RUN" = "1" ]; then
  log "DRY-RUN: would list collections at $QDRANT_URL/collections"
  log "DRY-RUN: per collection: POST $QDRANT_URL/collections/<name>/snapshots"
  log "DRY-RUN: per collection: download to $OUT/<name>/<snapshot>"
  log "DRY-RUN: would prune snapshot dirs older than ${RETENTION_DAYS}d under $DEST"
  exit 0
fi

mkdir -p "$OUT"

# ---- list collections ----------------------------------------------------
log "listing collections..."
COLLECTIONS_JSON="$(curl -sf "${CURL_AUTH[@]}" "$QDRANT_URL/collections")" || \
  die "could not reach $QDRANT_URL/collections (is qdrant up + the api-key correct?)"

mapfile -t COLLECTIONS < <(echo "$COLLECTIONS_JSON" | jq -r '.result.collections[].name')

if [ ${#COLLECTIONS[@]} -eq 0 ]; then
  log "no collections found — nothing to snapshot"
  exit 0
fi

log "found ${#COLLECTIONS[@]} collection(s): ${COLLECTIONS[*]}"

# ---- per-collection snapshot --------------------------------------------
FAILED=0
SNAPPED=0

for COL in "${COLLECTIONS[@]}"; do
  log "snapshot $COL ..."

  if ! SNAP_JSON=$(curl -sf -X POST "${CURL_AUTH[@]}" "$QDRANT_URL/collections/$COL/snapshots"); then
    log "  $COL: POST snapshot FAILED"
    FAILED=$((FAILED + 1))
    continue
  fi

  SNAP_NAME=$(echo "$SNAP_JSON" | jq -r '.result.name // empty')
  if [ -z "$SNAP_NAME" ]; then
    log "  $COL: snapshot returned no name (response: $SNAP_JSON)"
    FAILED=$((FAILED + 1))
    continue
  fi

  mkdir -p "$OUT/$COL"
  if ! curl -sf "${CURL_AUTH[@]}" -o "$OUT/$COL/$SNAP_NAME" \
        "$QDRANT_URL/collections/$COL/snapshots/$SNAP_NAME"; then
    log "  $COL: download FAILED for $SNAP_NAME"
    FAILED=$((FAILED + 1))
    continue
  fi

  SIZE=$(du -h "$OUT/$COL/$SNAP_NAME" | cut -f1)
  log "  $COL: $SNAP_NAME ($SIZE)"
  SNAPPED=$((SNAPPED + 1))
done

# ---- prune --------------------------------------------------------------
if [ "$RETENTION_DAYS" -gt 0 ]; then
  PRUNED=$(find "$DEST" -maxdepth 1 -type d -mtime "+$RETENTION_DAYS" -name '????-??-??-????' | wc -l | tr -d ' ')
  if [ "$PRUNED" -gt 0 ]; then
    log "pruning $PRUNED snapshot dir(s) older than ${RETENTION_DAYS}d"
    find "$DEST" -maxdepth 1 -type d -mtime "+$RETENTION_DAYS" -name '????-??-??-????' -exec rm -rf {} +
  fi
fi

log "DONE — $SNAPPED snapshotted, $FAILED failed"

# Cron wrapper alerts on non-zero.
[ "$FAILED" -eq 0 ] || exit 1
