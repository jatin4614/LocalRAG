#!/usr/bin/env bash
#
# Kairos — production backup.
# Captures: Postgres dump, Qdrant snapshots, uploaded files, env + TLS certs.
# Models volume is NOT included by default (large, re-downloadable).
# Override with BACKUP_INCLUDE_MODELS=1 to include it.
#
# Usage:
#   scripts/backup.sh                         # writes to ./backups/YYYYMMDD-HHMMSS/
#   BACKUP_DIR=/mnt/nas scripts/backup.sh
#   BACKUP_INCLUDE_MODELS=1 scripts/backup.sh
#   RETAIN=7 scripts/backup.sh                # prune all but latest 7 backups
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

TS="$(date -u +%Y%m%d-%H%M%S)"
BASE="${BACKUP_DIR:-$ROOT/backups}"
OUT="$BASE/$TS"
mkdir -p "$OUT"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
need docker
need tar

log "backup target: $OUT"

# ---- 1. Postgres ----
log "dumping postgres..."
if docker ps --format '{{.Names}}' | grep -q '^orgchat-postgres$'; then
  docker exec orgchat-postgres pg_dump -U orgchat -d orgchat \
    --no-owner --no-privileges --clean --if-exists \
    | gzip > "$OUT/postgres.sql.gz"
  PG_SIZE=$(du -h "$OUT/postgres.sql.gz" | cut -f1)
  log "  postgres dump: $PG_SIZE"
else
  log "  postgres container not running — SKIPPING"
fi

# ---- 2. Qdrant snapshots ----
log "snapshotting qdrant..."
mkdir -p "$OUT/qdrant"
if docker ps --format '{{.Names}}' | grep -q '^orgchat-qdrant$'; then
  COLLECTIONS=$(curl -s http://localhost:6333/collections | python3 -c "
import sys,json
d=json.load(sys.stdin)
for c in d.get('result',{}).get('collections',[]):
    print(c['name'])
" 2>/dev/null || echo "")
  for COL in $COLLECTIONS; do
    log "  snapshot $COL"
    SNAP=$(curl -s -X POST "http://localhost:6333/collections/$COL/snapshots" \
      | python3 -c "import sys,json;print(json.load(sys.stdin).get('result',{}).get('name',''))" 2>/dev/null || true)
    if [ -n "$SNAP" ]; then
      curl -sSL -o "$OUT/qdrant/${COL}_${SNAP}" "http://localhost:6333/collections/$COL/snapshots/$SNAP"
    fi
  done
else
  log "  qdrant not running — SKIPPING"
fi

# ---- 3. Upload volume (KB document bodies aren't stored by default but
#        may contain temp files during ingest) ----
if [ -d "$ROOT/volumes/uploads" ]; then
  log "archiving uploads..."
  tar -czf "$OUT/uploads.tar.gz" -C "$ROOT/volumes" uploads 2>/dev/null || log "  no uploads to back up"
fi

# ---- 4. TLS certs + env ----
log "capturing env + TLS..."
mkdir -p "$OUT/config"
cp -a "$ROOT/volumes/certs" "$OUT/config/certs" 2>/dev/null || true
cp -a "$ROOT/compose/.env" "$OUT/config/.env" 2>/dev/null || true
cp -a "$ROOT/compose/caddy/Caddyfile" "$OUT/config/Caddyfile" 2>/dev/null || true
# Mask secrets in a separate env.sanitized if you need to ship this off-host
if [ -f "$OUT/config/.env" ]; then
  sed -E 's/(PASSWORD|SECRET|KEY|TOKEN)=.*/\1=<redacted>/i' "$OUT/config/.env" \
    > "$OUT/config/.env.sanitized"
fi

# ---- 5. Models (opt-in; large) ----
if [ "${BACKUP_INCLUDE_MODELS:-0}" = "1" ]; then
  log "archiving models (large — may take a while)..."
  tar -czf "$OUT/models.tar.gz" -C "$ROOT/volumes" models
fi

# ---- 6. Manifest + checksum ----
cat > "$OUT/MANIFEST" <<EOF
Kairos backup manifest
timestamp_utc: $TS
source_host:   $(hostname)
include_models: ${BACKUP_INCLUDE_MODELS:-0}
files:
$(ls -la "$OUT" | tail -n +2)
EOF
(cd "$OUT" && find . -type f ! -name CHECKSUMS.sha256 -exec sha256sum {} + > CHECKSUMS.sha256)

# ---- 7. Retention ----
if [ -n "${RETAIN:-}" ] && [ "$RETAIN" -gt 0 ] 2>/dev/null; then
  log "pruning older than $RETAIN backups..."
  ls -1dt "$BASE"/*/ 2>/dev/null | tail -n +$((RETAIN+1)) | xargs -r rm -rf
fi

SIZE=$(du -sh "$OUT" | cut -f1)
log "DONE — $OUT ($SIZE)"
echo
echo "To restore: scripts/restore.sh $OUT"
