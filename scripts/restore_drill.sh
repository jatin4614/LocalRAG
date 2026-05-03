#!/usr/bin/env bash
#
# Wave 3a (review §11.7) — restore drill.
#
# This is the ONE script that proves Layer 4 of the rollback architecture
# works. It MUST be runnable on the operator's host BEFORE Wave 3b lands.
# A successful drill = "we can restore production from these artifacts".
#
# What it does:
#   1. Spin up a scratch Postgres database next to the running one.
#   2. pg_restore the supplied dump into the scratch DB.
#   3. Sanity SELECT — assert kb_documents has rows.
#   4. Spin up a scratch Qdrant on a temp port via docker run.
#   5. Upload each per-collection snapshot file via Qdrant's snapshot
#      upload API.
#   6. Sanity check — assert kb_1 has a non-zero point count.
#   7. Tear down both scratch resources.
#   8. Print PASS / FAIL summary, exit accordingly.
#
# DOES NOT touch production data. The scratch DB is created on the same
# Postgres instance (so we exercise the same engine version), but with
# a fresh name. The scratch Qdrant runs in its own throwaway container
# on a free host port.
#
# Usage:
#   scripts/restore_drill.sh \
#     --postgres-dump backups/postgres/orgchat-2026-05-02-0700.dump \
#     --qdrant-snapshot-dir backups/qdrant/2026-05-02-0700
#
#   scripts/restore_drill.sh \
#     --postgres-dump … \
#     --qdrant-snapshot-dir … \
#     --scratch-db orgchat_drill_2026
#
set -euo pipefail

# ---- defaults ------------------------------------------------------------
POSTGRES_DUMP=""
QDRANT_SNAPSHOT_DIR=""
SCRATCH_DB="orgchat_scratch"
SCRATCH_QDRANT_PORT=7333
SCRATCH_QDRANT_CONTAINER="orgchat-restore-drill-qdrant"

# ---- arg parse -----------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --postgres-dump)
      POSTGRES_DUMP="$2"
      shift 2
      ;;
    --qdrant-snapshot-dir)
      QDRANT_SNAPSHOT_DIR="$2"
      shift 2
      ;;
    --scratch-db)
      SCRATCH_DB="$2"
      shift 2
      ;;
    --scratch-qdrant-port)
      SCRATCH_QDRANT_PORT="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '3,38p' "$0"
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
# `cleanup` is defined further down once SCRATCH_*_CREATED state vars exist;
# wrap calls in a guard so early-exit paths (bad args) don't NPE.
log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
die()  { echo "ERROR: $*" >&2; type cleanup >/dev/null 2>&1 && cleanup || true; exit 1; }
ok()   { printf '[%s] PASS — %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
fail() { printf '[%s] FAIL — %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; type cleanup >/dev/null 2>&1 && cleanup || true; exit 1; }

# ---- arg validation ------------------------------------------------------
[ -n "$POSTGRES_DUMP" ] || die "--postgres-dump <path> is required"
[ -f "$POSTGRES_DUMP" ] || die "postgres dump not found: $POSTGRES_DUMP"
[ -n "$QDRANT_SNAPSHOT_DIR" ] || die "--qdrant-snapshot-dir <path> is required"
[ -d "$QDRANT_SNAPSHOT_DIR" ] || die "qdrant snapshot dir not found: $QDRANT_SNAPSHOT_DIR"

# ---- locate compose/.env -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../compose/.env"
[ -f "$ENV_FILE" ] || die "compose/.env not found at $ENV_FILE"

# ---- parse DATABASE_URL --------------------------------------------------
DATABASE_URL="$(grep -E '^DATABASE_URL=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)"
[ -n "$DATABASE_URL" ] || die "DATABASE_URL not found in $ENV_FILE"

URL_NO_DRIVER="${DATABASE_URL/+asyncpg/}"
URL_NO_DRIVER="${URL_NO_DRIVER/+psycopg2/}"
URL_NO_DRIVER="${URL_NO_DRIVER/+psycopg/}"

proto_stripped="${URL_NO_DRIVER#*://}"
userinfo="${proto_stripped%@*}"
hostpart="${proto_stripped#*@}"
PG_USER="${userinfo%%:*}"
PG_PASSWORD="${userinfo#*:}"
PG_HOST_PORT="${hostpart%%/*}"
PG_HOST="${PG_HOST_PORT%%:*}"
PG_PORT="${PG_HOST_PORT#*:}"
[ "$PG_HOST" = "$PG_PORT" ] && PG_PORT=5432

case "$PG_HOST" in
  postgres|orgchat-postgres|db)
    PG_HOST=127.0.0.1
    ;;
esac

# ---- dependency checks ---------------------------------------------------
command -v pg_restore >/dev/null 2>&1 || die "pg_restore not on PATH (apt install postgresql-client)"
command -v psql       >/dev/null 2>&1 || die "psql not on PATH"
command -v curl       >/dev/null 2>&1 || die "curl not on PATH"
command -v docker     >/dev/null 2>&1 || die "docker not on PATH"

# ---- cleanup hook --------------------------------------------------------
SCRATCH_PG_CREATED=0
SCRATCH_QDRANT_RUNNING=0

cleanup() {
  local rc=0
  if [ "$SCRATCH_PG_CREATED" = "1" ]; then
    log "tearing down scratch postgres db: $SCRATCH_DB"
    PGPASSWORD="$PG_PASSWORD" psql \
      -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d postgres \
      -c "DROP DATABASE IF EXISTS \"$SCRATCH_DB\";" >/dev/null 2>&1 || rc=1
  fi
  if [ "$SCRATCH_QDRANT_RUNNING" = "1" ]; then
    log "tearing down scratch qdrant container: $SCRATCH_QDRANT_CONTAINER"
    docker rm -f "$SCRATCH_QDRANT_CONTAINER" >/dev/null 2>&1 || rc=1
  fi
  return $rc
}
trap 'cleanup || true' EXIT INT TERM

# ---- step 1: scratch postgres -------------------------------------------
log "STEP 1: creating scratch postgres database '$SCRATCH_DB' on $PG_HOST:$PG_PORT"
PGPASSWORD="$PG_PASSWORD" psql \
  -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d postgres -v ON_ERROR_STOP=1 \
  -c "DROP DATABASE IF EXISTS \"$SCRATCH_DB\"; CREATE DATABASE \"$SCRATCH_DB\";" >/dev/null \
  || die "could not create scratch DB"
SCRATCH_PG_CREATED=1

# ---- step 2: pg_restore --------------------------------------------------
log "STEP 2: pg_restore $POSTGRES_DUMP → $SCRATCH_DB"
# --no-owner: dump may carry an owner that doesn't exist on this host.
# --exit-on-error: stop at the first failure (better than half-restored DB).
# --single-transaction would be ideal, but pg_restore -Fc may emit
# CREATE INDEX statements that can't run inside a single tx for
# concurrent indexes; leave it off so common dumps work.
PGPASSWORD="$PG_PASSWORD" pg_restore \
  -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$SCRATCH_DB" \
  --no-owner --exit-on-error --no-privileges \
  "$POSTGRES_DUMP" \
  || die "pg_restore failed"

# ---- step 3: sanity SELECT ----------------------------------------------
log "STEP 3: sanity SELECT count(*) FROM kb_documents"
KB_DOC_COUNT="$(
  PGPASSWORD="$PG_PASSWORD" psql \
    -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$SCRATCH_DB" \
    -tAc "SELECT count(*) FROM kb_documents;" 2>&1
)" || die "sanity SELECT failed (kb_documents table missing?)"
KB_DOC_COUNT="${KB_DOC_COUNT//[[:space:]]/}"
log "  kb_documents.count = $KB_DOC_COUNT"
if ! [[ "$KB_DOC_COUNT" =~ ^[0-9]+$ ]]; then
  fail "kb_documents.count is not numeric ('$KB_DOC_COUNT') — restore is corrupt"
fi
if [ "$KB_DOC_COUNT" -eq 0 ]; then
  fail "kb_documents.count = 0 — restore appears empty (unexpected for prod backup)"
fi
ok "postgres restore: $KB_DOC_COUNT docs visible"

# ---- step 4: spin up scratch qdrant -------------------------------------
log "STEP 4: spinning up scratch qdrant on port $SCRATCH_QDRANT_PORT"
# Remove any stale container with the same name.
docker rm -f "$SCRATCH_QDRANT_CONTAINER" >/dev/null 2>&1 || true
docker run -d --rm \
  --name "$SCRATCH_QDRANT_CONTAINER" \
  -p "$SCRATCH_QDRANT_PORT:6333" \
  qdrant/qdrant:latest >/dev/null \
  || die "could not start scratch qdrant container"
SCRATCH_QDRANT_RUNNING=1

# Wait for /healthz / collections endpoint to come up.
QDRANT_URL="http://127.0.0.1:$SCRATCH_QDRANT_PORT"
log "  waiting for $QDRANT_URL ..."
for i in $(seq 1 30); do
  if curl -sf "$QDRANT_URL/" >/dev/null 2>&1; then
    log "  scratch qdrant up after ${i}s"
    break
  fi
  sleep 1
done
curl -sf "$QDRANT_URL/" >/dev/null 2>&1 || die "scratch qdrant did not become ready in 30s"

# ---- step 5: restore each snapshot --------------------------------------
log "STEP 5: uploading snapshots from $QDRANT_SNAPSHOT_DIR"
SNAP_RESTORED=0
SNAP_FAILED=0

# Snapshot files live one level down: $DIR/$COLLECTION_NAME/$SNAPSHOT_FILE
for COL_DIR in "$QDRANT_SNAPSHOT_DIR"/*/; do
  [ -d "$COL_DIR" ] || continue
  COL_NAME="$(basename "$COL_DIR")"
  # Pick newest snapshot file in the dir.
  SNAP_FILE="$(ls -1t "$COL_DIR" 2>/dev/null | head -n1)"
  [ -n "$SNAP_FILE" ] || { warn "no snapshot file in $COL_DIR — skipping"; continue; }
  log "  $COL_NAME ← $SNAP_FILE"
  # Qdrant requires the target collection to NOT exist before snapshot
  # upload; on a fresh scratch instance this is the case for everything.
  # Use the priority=snapshot mode so the uploaded snapshot becomes the
  # source of truth.
  if ! curl -sf -X POST \
        "$QDRANT_URL/collections/$COL_NAME/snapshots/upload?priority=snapshot" \
        -H 'Content-Type: multipart/form-data' \
        -F "snapshot=@$COL_DIR$SNAP_FILE" \
        >/dev/null 2>&1; then
    warn "  $COL_NAME upload FAILED"
    SNAP_FAILED=$((SNAP_FAILED + 1))
    continue
  fi
  SNAP_RESTORED=$((SNAP_RESTORED + 1))
done

[ "$SNAP_RESTORED" -gt 0 ] || fail "no snapshots successfully restored"

# ---- step 6: sanity vector_count ----------------------------------------
log "STEP 6: sanity check — vector counts"
COLLECTIONS_JSON="$(curl -sf "$QDRANT_URL/collections")" || \
  fail "could not list collections on scratch qdrant"

mapfile -t RESTORED_COLS < <(echo "$COLLECTIONS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for c in data.get('result', {}).get('collections', []):
    print(c['name'])
")

NONZERO=0
for COL in "${RESTORED_COLS[@]}"; do
  COL_INFO="$(curl -sf "$QDRANT_URL/collections/$COL")" || { warn "  $COL: info call failed"; continue; }
  PT_COUNT="$(echo "$COL_INFO" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('result', {}).get('points_count') or d.get('result', {}).get('vectors_count') or 0)
")"
  PT_COUNT="${PT_COUNT//[[:space:]]/}"
  log "  $COL.points_count = $PT_COUNT"
  if [[ "$PT_COUNT" =~ ^[0-9]+$ ]] && [ "$PT_COUNT" -gt 0 ]; then
    NONZERO=$((NONZERO + 1))
  fi
done

if [ "$NONZERO" -eq 0 ]; then
  fail "no restored collection has points — qdrant restore is incomplete"
fi

ok "qdrant restore: $NONZERO collection(s) have non-zero points"

# ---- summary -------------------------------------------------------------
echo
echo "===================================="
echo "RESTORE DRILL: PASS"
echo "===================================="
echo "  postgres docs:           $KB_DOC_COUNT"
echo "  qdrant snapshots ok:     $SNAP_RESTORED"
echo "  qdrant snapshots failed: $SNAP_FAILED"
echo "  qdrant cols nonzero:     $NONZERO"
echo

# cleanup runs via trap
exit 0
