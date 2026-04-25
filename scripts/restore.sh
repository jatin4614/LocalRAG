#!/usr/bin/env bash
#
# Kairos — restore from a backup directory produced by scripts/backup.sh.
#
# Usage:
#   scripts/restore.sh backups/20260418-094500
#
# DANGER: this will DROP and replace the current postgres database.
# Qdrant collections will be restored via snapshot upload.
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
SRC="${1:-}"
[ -n "$SRC" ] && [ -d "$SRC" ] || { echo "usage: $0 <backup-dir>"; exit 1; }

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "restoring from $SRC"
log "verifying checksums..."
(cd "$SRC" && sha256sum -c CHECKSUMS.sha256 --status) || { echo "CHECKSUM MISMATCH"; exit 2; }

# Require explicit confirmation
printf '\nThis will OVERWRITE the current Kairos data. Type YES to continue: '
read -r CONFIRM
[ "$CONFIRM" = "YES" ] || { echo "aborted"; exit 3; }

# ---- Postgres ----
if [ -f "$SRC/postgres.sql.gz" ]; then
  log "restoring postgres..."
  docker exec -i orgchat-postgres psql -U orgchat -d orgchat -v ON_ERROR_STOP=1 \
    < <(gunzip -c "$SRC/postgres.sql.gz")
  log "  postgres restored"
fi

# ---- Qdrant ----
if [ -d "$SRC/qdrant" ]; then
  log "restoring qdrant snapshots..."
  for F in "$SRC"/qdrant/*; do
    [ -f "$F" ] || continue
    NAME=$(basename "$F")
    COL="${NAME%%_*}"
    # Upload snapshot and recover
    curl -sf -X POST "http://localhost:6333/collections/$COL/snapshots/upload?priority=snapshot" \
      -H 'Content-Type: multipart/form-data' \
      -F "snapshot=@$F" > /dev/null && log "  $COL ← $NAME" \
      || log "  $COL FAILED (may need manual recovery)"
  done
fi

# ---- Uploads ----
if [ -f "$SRC/uploads.tar.gz" ]; then
  log "restoring uploads..."
  mkdir -p "$ROOT/volumes"
  tar -xzf "$SRC/uploads.tar.gz" -C "$ROOT/volumes"
fi

# ---- TLS ----
if [ -d "$SRC/config/certs" ]; then
  log "restoring TLS certs..."
  mkdir -p "$ROOT/volumes/certs"
  cp -a "$SRC/config/certs/." "$ROOT/volumes/certs/"
fi

# ---- Models (opt-in) ----
if [ -f "$SRC/models.tar.gz" ] && [ "${RESTORE_MODELS:-0}" = "1" ]; then
  log "restoring models..."
  tar -xzf "$SRC/models.tar.gz" -C "$ROOT/volumes"
fi

log "restart services to pick up restored data:"
log "  docker compose -f compose/docker-compose.yml --env-file compose/.env restart"
log "DONE."
