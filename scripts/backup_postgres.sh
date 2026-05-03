#!/usr/bin/env bash
#
# Wave 3a (review §11.6) — Postgres backup script (host-side, ad-hoc).
#
# Operators run this BEFORE risky changes (pre-wave-N protocol from the
# bug-fix campaign tracker, "Pre-wave-3 protocol" snippet). The Celery
# beat in `ext/workers/snapshot_task.py` handles automatic daily Qdrant
# snapshots; this script handles host-side ad-hoc Postgres dumps.
#
# Usage:
#   scripts/backup_postgres.sh                                  # ./backups/postgres, retain 14d
#   scripts/backup_postgres.sh --dest /mnt/nas/pg
#   scripts/backup_postgres.sh --retention-days 30
#   scripts/backup_postgres.sh --dry-run
#
# Reads connection from compose/.env (DATABASE_URL parse). Exits 1 on any
# error so a cron wrapper can alert on non-zero.
#
# pg_dump options:
#   -Fc          custom compressed format (smaller, parallel restore via pg_restore)
#   --no-owner   strip GRANTs / OWNER directives (portable across hosts)
#
set -euo pipefail

# ---- defaults ------------------------------------------------------------
DEST="./backups/postgres"
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
      sed -n '3,20p' "$0"
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

# ---- locate compose/.env -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../compose/.env"
[ -f "$ENV_FILE" ] || die "compose/.env not found at $ENV_FILE — run from repo root"

# ---- parse DATABASE_URL --------------------------------------------------
# Format: postgresql+asyncpg://USER:PASSWORD@HOST:PORT/DBNAME
#         (or plain postgresql:// — accept both)
# We need: USER, PASSWORD, HOST, PORT, DBNAME for pg_dump.
DATABASE_URL="$(grep -E '^DATABASE_URL=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)"
[ -n "$DATABASE_URL" ] || die "DATABASE_URL not found in $ENV_FILE"

# Strip optional driver suffix (postgresql+asyncpg → postgresql).
URL_NO_DRIVER="${DATABASE_URL/+asyncpg/}"
URL_NO_DRIVER="${URL_NO_DRIVER/+psycopg2/}"
URL_NO_DRIVER="${URL_NO_DRIVER/+psycopg/}"

# Pure-bash URL parser. Cheaper and more robust than Python here.
proto_stripped="${URL_NO_DRIVER#*://}"
userinfo="${proto_stripped%@*}"
hostpart="${proto_stripped#*@}"
PG_USER="${userinfo%%:*}"
PG_PASSWORD="${userinfo#*:}"
PG_HOST_PORT="${hostpart%%/*}"
PG_DB="${hostpart#*/}"
# Strip query string from db name if any (?sslmode=…)
PG_DB="${PG_DB%%\?*}"
PG_HOST="${PG_HOST_PORT%%:*}"
PG_PORT="${PG_HOST_PORT#*:}"
[ "$PG_HOST" = "$PG_PORT" ] && PG_PORT=5432

# Inside compose, host is often "postgres" — that's a docker-network name
# the host can't resolve. Map to localhost (the postgres container binds
# its port to the host network). Operators with a remote DB pass --dest
# pointing at the right URL or override DATABASE_URL in their env.
case "$PG_HOST" in
  postgres|orgchat-postgres|db)
    log "rewriting host '$PG_HOST' → 127.0.0.1 (compose service name not resolvable from host)"
    PG_HOST=127.0.0.1
    ;;
esac

[ -n "$PG_USER" ] && [ -n "$PG_DB" ] && [ -n "$PG_HOST" ] || \
  die "failed to parse DATABASE_URL — got user=$PG_USER db=$PG_DB host=$PG_HOST"

# ---- prepare destination -------------------------------------------------
TS="$(date -u +%F-%H%M)"
DUMP_PATH="$DEST/orgchat-$TS.dump"

log "host=$PG_HOST port=$PG_PORT user=$PG_USER db=$PG_DB"
log "destination: $DUMP_PATH"
log "retention:   ${RETENTION_DAYS}d"

if [ "$DRY_RUN" = "1" ]; then
  log "DRY-RUN: would invoke:"
  log "  PGPASSWORD=*** pg_dump -Fc --no-owner -h $PG_HOST -p $PG_PORT -U $PG_USER $PG_DB > $DUMP_PATH"
  log "DRY-RUN: would prune dumps older than ${RETENTION_DAYS}d under $DEST"
  exit 0
fi

# ---- pg_dump dependency check (real run only) -----------------------------
command -v pg_dump >/dev/null 2>&1 || \
  die "pg_dump not on PATH — install postgresql-client"

mkdir -p "$DEST"

# ---- run dump ------------------------------------------------------------
log "starting pg_dump..."
PGPASSWORD="$PG_PASSWORD" pg_dump \
  -Fc \
  --no-owner \
  -h "$PG_HOST" \
  -p "$PG_PORT" \
  -U "$PG_USER" \
  "$PG_DB" \
  > "$DUMP_PATH"

DUMP_SIZE=$(du -h "$DUMP_PATH" | cut -f1)
log "wrote dump: $DUMP_PATH ($DUMP_SIZE)"

# ---- prune --------------------------------------------------------------
if [ "$RETENTION_DAYS" -gt 0 ]; then
  PRUNED=$(find "$DEST" -name 'orgchat-*.dump' -type f -mtime "+$RETENTION_DAYS" | wc -l | tr -d ' ')
  if [ "$PRUNED" -gt 0 ]; then
    log "pruning $PRUNED dump(s) older than ${RETENTION_DAYS}d"
    find "$DEST" -name 'orgchat-*.dump' -type f -mtime "+$RETENTION_DAYS" -delete
  fi
fi

log "DONE"
