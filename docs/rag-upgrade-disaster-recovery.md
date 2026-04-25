# RAG Upgrade — Backup & Disaster Recovery Plan

**Date:** 2026-04-19
**Purpose:** Before we start the RAG upgrade, capture a complete, restorable snapshot of the working app so we can fall back to the current state from any catastrophe — partial migration, corrupted vector store, broken code, stuck queue, or total stack failure.
**Companion doc:** `docs/rag-upgrade-execution-plan.md` (phased workflow).

---

## 1. What "Fall Back" Means Here

A successful fallback restores **all five** layers to the exact state captured before P0.0 starts:

| Layer | Source of truth | Backup artifact |
|-------|-----------------|-----------------|
| Code + config (tracked + untracked) | `/home/vogic/LocalRAG` working tree | `localrag-tree.tgz` |
| Git state (SHA, branches, worktrees) | `.git` | `git-state.txt` + `.git.tgz` |
| Postgres (users, KBs, chats, docs metadata) | `orgchat-postgres` container | `postgres.sql.gz` |
| Qdrant (vectors + payloads per collection) | `orgchat-qdrant` container + `qdrant_data` volume | `qdrant-<col>.snapshot` files + `qdrant_data.tgz` |
| Redis (sessions, rate-limit counters, queued jobs) | `orgchat-redis` container | `redis-dump.rdb` |
| Docker Compose declaration | `compose/docker-compose.yml` | Included in code tarball + separate copy |
| Environment variables | `compose/.env` | Separate encrypted copy |

Anything not in this list (model weights, logs, Caddy cert cache) is either regeneratable or inconsequential.

---

## 2. One-Time Backup Procedure (run before P0.0)

All commands are copy-paste safe. Run from any shell; no code is modified.

### 2.1 Create backup directory
```bash
export BACKUP_ROOT=~/rag-upgrade-backups
export BACKUP_DIR="$BACKUP_ROOT/$(date +%Y%m%d-%H%M%S)-pre-p0"
mkdir -p "$BACKUP_DIR"
echo "Backup dir: $BACKUP_DIR"
```

### 2.2 Record current git + container state (manifest)
```bash
cd /home/vogic/LocalRAG

{
  echo "=== baseline ==="
  echo "timestamp=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo
  echo "=== git ==="
  git rev-parse HEAD
  git status --short
  git worktree list
  git branch -a
  echo
  echo "=== docker ==="
  docker compose -f compose/docker-compose.yml ps
  docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}}' | sort
  echo
  echo "=== disk ==="
  df -h /home
  docker system df
} > "$BACKUP_DIR/MANIFEST.txt"
```

### 2.3 Back up code + config (working tree, tracked + untracked)
```bash
tar --exclude='.venv' \
    --exclude='volumes/models' \
    --exclude='__pycache__' \
    --exclude='node_modules' \
    --exclude='*.pyc' \
    -czf "$BACKUP_DIR/localrag-tree.tgz" \
    -C /home/vogic LocalRAG

# Separate copy of .env and compose file (quick access during restore)
cp /home/vogic/LocalRAG/compose/.env "$BACKUP_DIR/env.backup" 2>/dev/null || echo "no .env"
cp /home/vogic/LocalRAG/compose/docker-compose.yml "$BACKUP_DIR/docker-compose.yml.backup"

# Full .git so branches/worktree refs are preserved
tar -czf "$BACKUP_DIR/dot-git.tgz" -C /home/vogic/LocalRAG .git
```

### 2.4 Back up Postgres (logical dump + physical volume)
```bash
# Logical dump — portable, restorable to any Postgres 15
docker exec orgchat-postgres pg_dumpall -U postgres \
  | gzip > "$BACKUP_DIR/postgres.sql.gz"

# Verify
gunzip -t "$BACKUP_DIR/postgres.sql.gz" && echo "postgres dump OK"
ls -lh "$BACKUP_DIR/postgres.sql.gz"
```

### 2.5 Back up Qdrant (snapshots + raw volume)
```bash
# Per-collection snapshots via Qdrant API
mkdir -p "$BACKUP_DIR/qdrant-snapshots"
for col in $(curl -s http://localhost:6333/collections | jq -r '.result.collections[].name'); do
  echo "snapshotting $col..."
  resp=$(curl -s -X POST "http://localhost:6333/collections/$col/snapshots")
  snap_name=$(echo "$resp" | jq -r '.result.name')
  # Download the snapshot file out of the container
  docker cp "orgchat-qdrant:/qdrant/snapshots/$col/$snap_name" \
    "$BACKUP_DIR/qdrant-snapshots/${col}__${snap_name}"
done

# Raw volume copy as belt-and-suspenders (stop Qdrant for consistency)
docker compose -f /home/vogic/LocalRAG/compose/docker-compose.yml stop qdrant
docker run --rm \
  -v orgchat_qdrant_data:/data:ro \
  -v "$BACKUP_DIR":/backup \
  alpine tar -czf /backup/qdrant_data.tgz -C /data .
docker compose -f /home/vogic/LocalRAG/compose/docker-compose.yml start qdrant
```

### 2.6 Back up Redis
```bash
# Trigger RDB dump
docker exec orgchat-redis redis-cli BGSAVE
sleep 3
# Copy the RDB file out
docker cp orgchat-redis:/data/dump.rdb "$BACKUP_DIR/redis-dump.rdb"
ls -lh "$BACKUP_DIR/redis-dump.rdb"
```

### 2.7 Seal the backup (integrity + read-only)
```bash
cd "$BACKUP_DIR"
sha256sum * qdrant-snapshots/* > SHA256SUMS
chmod -R a-w "$BACKUP_DIR"     # make read-only
du -sh "$BACKUP_DIR"
echo "Backup sealed at $BACKUP_DIR"
```

### 2.8 Verify before proceeding
- [ ] `MANIFEST.txt` contains current SHA (should be `ba28dd5` on `main`)
- [ ] `gunzip -t postgres.sql.gz` returns zero
- [ ] At least one file per existing Qdrant collection in `qdrant-snapshots/`
- [ ] `qdrant_data.tgz` is > 0 bytes
- [ ] `redis-dump.rdb` exists
- [ ] `sha256sum -c SHA256SUMS` all OK

**Do not start P0.0 until §2.8 passes.**

---

## 3. Restore Procedures

### 3.1 Scenario A — Code rollback only (no data corruption)
Symptom: new worktree code breaks, but Postgres/Qdrant/Redis are healthy.

```bash
# Nothing to restore — just stop any upgrade-branch processes and switch back
docker compose -f /home/vogic/LocalRAG/compose/docker-compose.yml restart open-webui
# Verify main working tree untouched
cd /home/vogic/LocalRAG && git rev-parse HEAD   # should equal baseline SHA
```
Recovery time: ~30 s.

### 3.2 Scenario B — Bad Postgres migration
Symptom: app crashes on startup, SQL errors in logs, or schema visibly wrong.

```bash
cd /home/vogic/LocalRAG
docker compose stop open-webui model-manager   # stop writers first
docker exec -i orgchat-postgres psql -U postgres -c "DROP DATABASE IF EXISTS webui;"
docker exec -i orgchat-postgres psql -U postgres -c "CREATE DATABASE webui;"
gunzip -c "$BACKUP_DIR/postgres.sql.gz" | docker exec -i orgchat-postgres psql -U postgres
docker compose start open-webui model-manager
```
Recovery time: ~2–5 min.

### 3.3 Scenario C — Corrupted Qdrant collection
Symptom: retrieval returns garbage, 0 recall on eval, or Qdrant errors.

```bash
# Fast path: restore single collection from snapshot
COL=kb_10
SNAP_FILE=$(ls "$BACKUP_DIR/qdrant-snapshots/" | grep "^${COL}__" | head -1)

# Upload snapshot back into Qdrant
docker cp "$BACKUP_DIR/qdrant-snapshots/$SNAP_FILE" \
  "orgchat-qdrant:/qdrant/snapshots/$COL/${SNAP_FILE#*__}"

curl -X PUT "http://localhost:6333/collections/$COL/snapshots/recover" \
  -H 'Content-Type: application/json' \
  -d "{\"location\":\"file:///qdrant/snapshots/$COL/${SNAP_FILE#*__}\"}"
```

Nuclear path: restore whole volume.
```bash
docker compose stop qdrant
docker run --rm -v orgchat_qdrant_data:/data -v "$BACKUP_DIR":/backup alpine \
  sh -c "rm -rf /data/* && tar -xzf /backup/qdrant_data.tgz -C /data"
docker compose start qdrant
```
Recovery time: 1–10 min.

### 3.4 Scenario D — Stuck Celery queue / Redis bloat
```bash
docker compose stop redis
docker cp "$BACKUP_DIR/redis-dump.rdb" orgchat-redis:/data/dump.rdb
docker compose start redis
```
Recovery time: ~30 s.

### 3.5 Scenario E — Total stack failure / uncertain state
Full rebuild from backup.

```bash
# 1. Stop everything
cd /home/vogic/LocalRAG
docker compose down

# 2. (Optional) move current tree aside rather than delete
mv /home/vogic/LocalRAG /home/vogic/LocalRAG.broken.$(date +%s)

# 3. Restore code
mkdir -p /home/vogic/LocalRAG
tar -xzf "$BACKUP_DIR/localrag-tree.tgz" -C /home/vogic
tar -xzf "$BACKUP_DIR/dot-git.tgz" -C /home/vogic/LocalRAG

# 4. Bring containers up with empty volumes, then restore data
cd /home/vogic/LocalRAG
docker compose up -d postgres qdrant redis
sleep 15
gunzip -c "$BACKUP_DIR/postgres.sql.gz" | docker exec -i orgchat-postgres psql -U postgres
docker compose stop qdrant redis
docker run --rm -v orgchat_qdrant_data:/data -v "$BACKUP_DIR":/backup alpine \
  sh -c "rm -rf /data/* && tar -xzf /backup/qdrant_data.tgz -C /data"
docker cp "$BACKUP_DIR/redis-dump.rdb" orgchat-redis:/data/dump.rdb

# 5. Start the rest
docker compose up -d

# 6. Verify
curl -f http://localhost:3000/health
cd /home/vogic/LocalRAG && git rev-parse HEAD   # should match baseline
```
Recovery time: 15–30 min.

### 3.6 Scenario F — "I need main exactly as it was, but the upgrade branch is half-done"
```bash
# Main's working tree was never touched, so:
cd /home/vogic/LocalRAG
git status                                      # should match baseline-status.txt
# If the worktree is the problem, remove it:
git worktree remove --force ~/.config/superpowers/worktrees/LocalRAG/rag-upgrade-p0
git branch -D rag-upgrade-p0
```
Recovery time: <1 min. **No data restore needed** — main's code was never modified.

---

## 4. Pre-P0.0 Safety Checklist

- [ ] Backup directory created under `~/rag-upgrade-backups/`
- [ ] `MANIFEST.txt` shows baseline SHA matches `git rev-parse HEAD` on main
- [ ] `postgres.sql.gz` verified with `gunzip -t`
- [ ] Qdrant snapshots exist for every live collection listed by `GET /collections`
- [ ] `qdrant_data.tgz` exists and > 0 bytes
- [ ] `redis-dump.rdb` exists
- [ ] `SHA256SUMS` verified
- [ ] Backup directory made read-only (`chmod -R a-w`)
- [ ] Free disk: at least 2× backup size available on `/home` (for restore workspace)
- [ ] `.env` copied separately and readable
- [ ] Backup path recorded in `/tmp/rag-upgrade-active-backup.txt` for easy reference

```bash
echo "$BACKUP_DIR" > /tmp/rag-upgrade-active-backup.txt
```

---

## 5. Fallback Drill (recommended before real work)

Prove the backup restores. On a test schedule (takes ~5 min):

1. Take backup per §2.
2. Add a throwaway row to Postgres: `INSERT INTO users (email, password_hash, role) VALUES ('drill@x','x','user');`
3. Run §3.2 (Postgres restore).
4. Verify the drill row is gone — backup is valid.
5. If restore failed: **do not proceed with P0.0.** Diagnose backup first.

---

## 6. Retention & Cleanup

- Keep pre-P0 backup **until full RAG upgrade is merged and stable for 7 days** post-flag-flip.
- Create incremental backups before each phase boundary (P0→P1, P1→P2) — small, fast, same procedure.
- Delete only after explicit confirmation; `chmod -R a-w` prevents accidental `rm -rf`.

---

## 7. What This Plan Does NOT Cover

- **GPU/driver catastrophes** — unrelated to RAG data; rebuild from OS-level backup or reinstall driver.
- **Disk failure on `/home`** — backups live on the same disk by default. For true DR, copy `$BACKUP_DIR` to a second disk or NAS after §2.7:
  ```bash
  rsync -a "$BACKUP_DIR" /mnt/second-disk/rag-backups/
  ```
- **Live-traffic rollback** — this plan assumes the work window has no users. If real users are active, switching to Option C (parallel stack) is required instead.

---

## 8. Summary: Two Commands Before Starting

```bash
# 1. Run the full backup (§2)
bash -c '
  export BACKUP_ROOT=~/rag-upgrade-backups
  export BACKUP_DIR="$BACKUP_ROOT/$(date +%Y%m%d-%H%M%S)-pre-p0"
  # ... (paste §2.1–2.7)
'

# 2. Verify checklist in §4. If all green, proceed to P0.0.
```

If anything in §4 fails: **stop. Fix the backup. Do not start the upgrade without a verified fallback.**
