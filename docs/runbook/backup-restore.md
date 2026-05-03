# Backup & Restore Runbook

**Wave 3a foundational.** Layer 4 of the campaign rollback architecture
(data restore from snapshot). All Wave 3 schema migrations (3b/3c/3e/3f)
gate on this runbook being usable on the operator's host.

Two pathways exist; both must work:

| Pathway | What | When |
|---|---|---|
| **Daily Celery beat** | `ext/workers/snapshot_task.py` fires daily at 02:30 UTC and snapshots every Qdrant collection inside the qdrant container. | Automatic safety net — fires even if the operator forgets. |
| **Host-side scripts** | `scripts/backup_postgres.sh` + `scripts/backup_qdrant.sh` | **BEFORE risky changes** (pre-wave-N protocol). |

The Celery task is "we always have something from last night". The
host-side scripts are "I'm about to apply migration 013, snapshot now".
Don't conflate the two — the rollback registry mandates host-side
captures because the beat itself can fail (qdrant down, beat-state
corrupt, scheduler clock skew, ...).

---

## Pre-wave-N protocol

Before applying ANY schema migration, Qdrant collection change, or
multi-day soak, capture the rollback artifacts:

```bash
# 1. Postgres dump (custom format, parallel-restorable)
scripts/backup_postgres.sh --dest backups/pre-wave-N

# 2. Qdrant per-collection snapshots
scripts/backup_qdrant.sh --dest backups/pre-wave-N/qdrant

# 3. Uploads (ingested-doc bodies, may not all be in blob store)
cp -a volumes/uploads backups/pre-wave-N-uploads-$(date +%F)

# 4. Async-ingest blob store (sha-addressable bodies in flight)
cp -a /var/lib/docker/volumes/orgchat_ingest_blobs/_data \
      backups/pre-wave-N-blobs-$(date +%F)

# 5. Config snapshot
tar -czf backups/pre-wave-N-config-$(date +%F).tgz compose/.env compose/caddy/

# 6. VERIFY restore on scratch DB BEFORE proceeding (drill).
scripts/restore_drill.sh \
  --postgres-dump backups/pre-wave-N/orgchat-$(date +%F)*.dump \
  --qdrant-snapshot-dir backups/pre-wave-N/qdrant/$(ls -1 backups/pre-wave-N/qdrant | tail -1)
```

If step 6 fails, **STOP**. Do not proceed with the wave until the drill
passes. The rollback registry is only as good as the artifacts it points
at — un-restorable artifacts are worse than no artifacts because they
create false confidence.

After a successful drill, tag the rollback handle in git:

```bash
git tag rollback-pre-wave-N
git push origin rollback-pre-wave-N
```

---

## Daily backup verification

The Celery beat task fires daily at 02:30 UTC. Verify it ran:

```bash
# 1. Did beat schedule fire today?
docker compose logs celery-beat | grep -i 'qdrant-snapshot' | tail -5

# 2. Did the worker pick up + complete the task?
docker compose logs celery-worker | grep -E 'qdrant_snapshot|snapshotting' | tail -10

# 3. Are there fresh snapshots inside the qdrant container?
docker exec orgchat-qdrant ls -lt /qdrant/snapshots/ | head -20
# Each collection has its own subdir; snapshots are named
# <collection>-<timestamp>.snapshot

# 4. Prometheus check — any per-collection failures?
curl -s http://localhost:9091/api/v1/query?query=rag_snapshot_failure_total | jq
# Expect: empty result. Non-empty means at least one collection failed
# its automatic snapshot — investigate qdrant logs for the failing
# collection.
```

If the beat task is silent (no log entries), check that `celery-beat` is
running:

```bash
docker compose ps celery-beat
# State should be "Up (healthy)". If unhealthy / restarting:
docker compose logs celery-beat --tail=100
```

---

## Restore procedure

For the full DR scenario — the host died, the operator is restoring on a
fresh node from a backup tarball:

### 1. Bring the stack up empty

```bash
cd compose
docker compose -p orgchat up -d postgres qdrant redis
# Wait for postgres + qdrant to report healthy.
docker compose -p orgchat ps
```

### 2. Restore Postgres

```bash
# Locate the dump file from your backup set.
DUMP=backups/postgres/orgchat-2026-05-02-0700.dump

PGPASSWORD=$(grep '^POSTGRES_PASSWORD=' compose/.env | cut -d= -f2) pg_restore \
  -h 127.0.0.1 -p 5432 -U orgchat -d orgchat \
  --no-owner --exit-on-error --no-privileges \
  "$DUMP"
```

If `pg_restore` complains about the database already existing, drop it
first (only safe on a fresh node):

```bash
PGPASSWORD=$(...) psql -h 127.0.0.1 -p 5432 -U orgchat -d postgres \
  -c "DROP DATABASE orgchat; CREATE DATABASE orgchat;"
```

### 3. Restore Qdrant (per collection)

```bash
SNAP_DIR=backups/qdrant/2026-05-02-0700  # one timestamped dir per backup run

for COL_DIR in "$SNAP_DIR"/*/; do
  COL=$(basename "$COL_DIR")
  SNAP=$(ls -1t "$COL_DIR" | head -n1)
  echo "Restoring $COL from $SNAP..."
  curl -sf -X POST \
    "http://127.0.0.1:6333/collections/$COL/snapshots/upload?priority=snapshot" \
    -H 'Content-Type: multipart/form-data' \
    -F "snapshot=@$COL_DIR$SNAP"
done
```

Pass `-H "api-key: $QDRANT_API_KEY"` on every call if Qdrant auth is on
(Wave 1a §4.1).

### 4. Restore async-ingest blobs (if applicable)

```bash
# If RAG_SYNC_INGEST=0 (default after Plan B Phase 6.2), there may be
# in-flight blobs the worker hadn't yet processed at backup time.
docker compose stop celery-worker
docker run --rm -v orgchat_ingest_blobs:/dst -v "$PWD/backups:/src" alpine \
  sh -c "cp -a /src/pre-wave-N-blobs-2026-05-02/. /dst/"
docker compose start celery-worker
```

### 5. Restore uploads + config

```bash
mkdir -p volumes/uploads
cp -a backups/pre-wave-N-uploads-2026-05-02/. volumes/uploads/
tar -xzf backups/pre-wave-N-config-2026-05-02.tgz -C .
```

### 6. Bring up the rest of the stack

```bash
docker compose -p orgchat up -d
```

### 7. Sanity verification

```bash
# Postgres: real doc count?
docker exec orgchat-postgres psql -U orgchat -d orgchat \
  -c "SELECT count(*) FROM kb_documents WHERE deleted_at IS NULL;"

# Qdrant: per-collection point counts?
for COL in $(curl -s http://127.0.0.1:6333/collections | jq -r '.result.collections[].name'); do
  COUNT=$(curl -s "http://127.0.0.1:6333/collections/$COL" \
    | jq -r '.result.points_count // .result.vectors_count')
  echo "$COL: $COUNT"
done

# RAG end-to-end: smoke test a query.
curl -X POST http://127.0.0.1:6100/api/rag/retrieve \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "kb_ids": [1]}'
```

---

## Estimated restore times

Sized for the live corpus snapshot from `psql query 2026-05-02`:

| Component | Volume | Estimated restore time |
|---|---|---|
| Postgres dump | ~50 MB compressed (260 docs in `kb_documents`, 12 migrations of metadata) | < 30s on local SSD |
| Qdrant `kb_1_v4` | ~2705 points × (1024-d dense + sparse + ColBERT) ≈ 200 MB | 1–2 min snapshot upload + 30s reindex |
| Qdrant `kb_1_v3` | ~2705 points (Plan A rollback target until 2026-05-10) | as above |
| Qdrant `kb_1_v2` + `kb_1_rebuild` | older targets, deleted by cron 2026-05-09 | n/a after that date |
| `chat_private` collection | < 100 points | < 30s |
| `volumes/uploads` | varies; current corpus ~500 MB | seconds (`cp -a`) |
| Async ingest blobs (`/var/ingest`) | sha-addressable, capped by `RAG_BLOB_RETENTION_DAYS` (30d) | < 1 min |

**Full restore wall-clock: ~10–15 minutes** assuming local-disk
backups. Network-attached backups add transfer time.

---

## Known caveats

### 1. Qdrant snapshots are PER-COLLECTION, not cluster-wide

Qdrant's snapshot API operates at the collection level. The Celery beat
task and `scripts/backup_qdrant.sh` both iterate every collection and
snapshot them one at a time. There is no atomic cluster-wide snapshot —
if data is being written during the snapshot loop, a kb_1 snapshot taken
at T+0s and a kb_2 snapshot taken at T+30s are not transactionally
consistent. For most production scenarios this is acceptable (KBs are
independent), but **do not rely on cross-collection point-count
invariants from a backup taken during active ingest**.

### 2. Custom-sharded collections (`kb_1_v4`) need shard-by-shard restore

Per Plan B Phase 5, `kb_1_v4` is a custom-sharded collection
(shard_key = `YYYY-MM`). Qdrant's snapshot API snapshots the whole
collection (all shards together), but **upload-restore lands as a
single shard** unless the operator manually re-creates the shard
structure first.

If a custom-sharded collection needs restore from snapshot:

1. Restore as a normal collection per the procedure above.
2. Run `scripts/reshard_kb_temporal.py --collection kb_1_v4 --reshard-from <restored>`
   to redistribute back into per-month shards.

This is a known limitation we'll address in Phase 5.x; for now, the
operator must run the reshard step explicitly.

### 3. Postgres dump format

`-Fc` (custom compressed) is intentional. It allows parallel restore
via `pg_restore -j N`, is ~70% smaller than plain SQL, and supports
`--data-only` / `--schema-only` modes for partial recovery. It does
**NOT** work with `psql < dump`; you must use `pg_restore`. The
restore drill (`scripts/restore_drill.sh`) uses the right command.

### 4. The drill DOES NOT exercise the application stack

`restore_drill.sh` proves Postgres + Qdrant can be reloaded into a
parallel scratch instance. It does **not** verify that `open-webui`
or `celery-worker` come back healthy from the restored data. Operators
should do a smoke test (`curl /api/rag/retrieve` after a restore) as a
Layer 5 check separate from Layer 4 (snapshot integrity).

### 5. Retention is a script-side concept

`--retention-days` only prunes files in the `--dest` directory. If you
keep multiple backup destinations (e.g. local + NAS), each needs its
own retention policy. Off-host backups are the operator's responsibility.

---

## Cross-references

* `ext/workers/snapshot_task.py` — Celery beat task source.
* `ext/workers/blob_gc_task.py` — sibling beat task for blob GC (run pattern).
* `scripts/backup_postgres.sh` / `scripts/backup_qdrant.sh` — host-side scripts.
* `scripts/restore_drill.sh` — verifies Layer 4 of the rollback architecture.
* `scripts/backup.sh` / `scripts/restore.sh` — older monolithic backup helpers (pre-Wave-3a, kept for the existing operator runbook).
* `docs/superpowers/plans/2026-05-02-bug-fix-campaign.md` §11.6 / §11.7 — the original review items this runbook closes.
* `docs/runbook/temporal-reshard-procedure.md` — sharded-collection notes related to caveat 2 above.
