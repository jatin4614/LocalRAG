# RAG Phase 0 + 1a + 4 — Rollback procedures

**Date:** 2026-04-21
**Companion:** `docs/rag-phase0-1a-4-execution-plan.md`
**Baseline:** git `b5fe768` on `main`, backup `backups/20260421-182222/` (CHECKSUMS verified)

This document covers three rollback scenarios sized by severity. Each starts with a **trigger**, a **scope** (what's affected vs what's safe), and **commands** safe to copy-paste.

Running services at backup time:
- `orgchat-open-webui` — main app
- `orgchat-vllm-chat` — chat/vision (Gemma 4 31B AWQ)
- `orgchat-tei` — bge-m3 embedder
- `orgchat-qdrant`, `orgchat-postgres`, `orgchat-redis` — state

Collections at backup time:
- `kb_1` (110 docs, 2590 points) — **the only KB to be touched by Phase 1a**
- `kb_eval` (130 points) — eval corpus, not affected
- `open-webui_files` (398 points) — upstream files, not affected

---

## Scenario A — Code rollback only (chunker/ingest/routes misbehave, data healthy)

**Trigger:**
- Unit tests fail after agent code lands but Qdrant has NOT been re-ingested yet.
- Health endpoint returns 500 but data counts look correct.
- Import errors on container start.

**Scope:**
- Revert code changes in `ext/` and `scripts/`.
- Postgres, Qdrant, Redis untouched — still at v2 chunks.
- No data restore required.

**Commands:**
```bash
cd /home/vogic/LocalRAG

# If changes aren't committed yet (most likely — agents left dirty tree):
git stash push -m "phase0-1a-4-rollback-$(date +%s)" -- \
    ext/services/chunker.py \
    ext/services/ingest.py \
    ext/services/extractor.py \
    ext/services/vector_store.py \
    ext/services/pipeline_version.py \
    ext/services/ood_signal.py \
    ext/services/chat_rag_bridge.py \
    ext/services/metrics.py \
    ext/routers/kb_admin.py \
    ext/workers/celery_app.py \
    ext/workers/scheduled_eval.py \
    scripts/reingest_all.py \
    scripts/delete_orphan_chunks.py \
    scripts/reembed_all.py \
    tests/eval/ \
    tests/unit/test_chunker_coalesce.py \
    tests/unit/test_ingest_heading_prepend.py

# Or, if already committed on a branch:
# git checkout b5fe768 -- ext/ scripts/ tests/

# Rebuild baseline image and redeploy
cd compose
docker compose build open-webui
docker compose up -d --force-recreate open-webui

# Verify
docker logs orgchat-open-webui --tail 100 | grep -iE "error|import" || echo "clean boot"
curl -f http://localhost:6100/health
```
**Recovery time:** ~5–10 min (rebuild dominates).

---

## Scenario B — Chunker changes deployed, kb_1 re-ingested, quality regressed

**Trigger (any one):**
- `baseline-post-phase1a.json` chunk_recall@10 < baseline − 2pp.
- Faithfulness drops > 5pp.
- `drift_pct > 20%` after `delete_orphan_chunks.py`.
- Users report noticeably worse answers in a 30-minute monitoring window.

**Scope:**
- Need to revert code (chunker/ingest back to v2).
- Need to restore `kb_1` Qdrant collection from snapshot (has all 2590 v2 chunks).
- Postgres `kb_documents.pipeline_version` still says `v3` for some rows — needs rewriting.

**Commands:**

### B.1 Stop ingest writes
```bash
cd /home/vogic/LocalRAG
docker compose stop open-webui celery-worker
```

### B.2 Revert code to baseline
```bash
# Fast path if work is uncommitted:
git restore -- ext/services/chunker.py \
             ext/services/ingest.py \
             ext/services/extractor.py \
             ext/services/vector_store.py \
             ext/services/pipeline_version.py

# Or, if committed on a branch:
# git checkout b5fe768 -- ext/services/chunker.py \
#                         ext/services/ingest.py \
#                         ext/services/extractor.py \
#                         ext/services/vector_store.py \
#                         ext/services/pipeline_version.py

# Rebuild image now so we can redeploy after data is restored
cd compose
docker compose build open-webui
```

### B.3 Restore kb_1 Qdrant snapshot
```bash
BK=/home/vogic/LocalRAG/backups/20260421-182222
SNAP=$(ls "$BK/qdrant/" | grep '^kb_1_')
echo "snapshot: $SNAP"

# Delete the current (post-1a) kb_1 collection
curl -sX DELETE http://localhost:6333/collections/kb_1

# Copy snapshot into Qdrant container and recover
docker cp "$BK/qdrant/$SNAP" "orgchat-qdrant:/qdrant/snapshots/kb_1/${SNAP#*_}"
curl -X PUT http://localhost:6333/collections/kb_1/snapshots/recover \
  -H 'Content-Type: application/json' \
  -d "{\"location\":\"file:///qdrant/snapshots/kb_1/${SNAP#*_}\"}"

# Verify point count restored to 2590
curl -s http://localhost:6333/collections/kb_1 | python3 -c "import sys,json; d=json.load(sys.stdin)['result']; print('points:', d['points_count'])"
```

### B.4 Restore Postgres pipeline_version stamps
```bash
# Fastest: restore kb_documents rows from the Postgres dump.
# We only need the pipeline_version column; full restore is overkill.
# Extract just the kb_documents table:
BK=/home/vogic/LocalRAG/backups/20260421-182222
gunzip -c "$BK/postgres.sql.gz" | grep -A100000 "COPY.*kb_documents" | head -200 > /tmp/kb_documents.sql

# Simplest correct option: rewrite pipeline_version back to v2 for kb_1 rows.
docker exec -i orgchat-postgres psql -U orgchat -d orgchat -c "
UPDATE kb_documents
SET pipeline_version = 'chunker=v2|extractor=v2|embedder=bge-m3|ctx=none'
WHERE kb_id = 1 AND deleted_at IS NULL;"

# Verify
docker exec orgchat-postgres psql -U orgchat -d orgchat -t -c "
SELECT pipeline_version, COUNT(*) FROM kb_documents
WHERE kb_id = 1 AND deleted_at IS NULL GROUP BY pipeline_version;"
```

### B.5 Redeploy baseline open-webui
```bash
cd /home/vogic/LocalRAG/compose
docker compose up -d --force-recreate open-webui celery-worker

# Smoke test
sleep 20
docker logs orgchat-open-webui --tail 50 | grep -iE "error" || echo "boot clean"
curl -f http://localhost:6100/health
```

### B.6 Verify retrieval quality restored
```bash
cd /home/vogic/LocalRAG
python -m tests.eval.run_all --golden tests/eval/golden_human.jsonl \
  --out tests/eval/results/rollback-verify-$(date +%s).json

# Compare to baseline-pre-phase1a.json — should match within noise
```

**Recovery time:** 30–45 min.

---

## Scenario C — Total failure (data corruption, uncertain state, Qdrant volume compromised)

**Trigger:**
- Qdrant shows collections in `red` or `yellow` status beyond 10 min.
- `drift_pct > 80%` (most chunks vanished).
- Qdrant container won't start.
- Postgres foreign-key violations referring to missing Qdrant data.
- Any scenario where B.3–B.4 don't restore correctly.

**Scope:** Full stack restore from the Apr-21 backup.

**Commands:**
```bash
cd /home/vogic/LocalRAG
BK=/home/vogic/LocalRAG/backups/20260421-182222

# 1. Hard stop everything app-side
docker compose stop open-webui celery-worker model-manager

# 2. Revert code tree to baseline SHA
git stash push -m "full-rollback-$(date +%s)" -- ext/ scripts/ tests/ docs/
# Expected diff: exactly the files listed in GIT_STATE.txt at backup time.

# 3. Restore Postgres
docker exec -i orgchat-postgres psql -U orgchat -d orgchat -c "
SET session_replication_role = replica;" || true

gunzip -c "$BK/postgres.sql.gz" | docker exec -i orgchat-postgres psql -U orgchat -d orgchat

# 4. Restore Qdrant — use backup script's restore, which upload-recovers every snapshot
bash scripts/restore.sh "$BK"
# (restore.sh requires confirming YES, and it uploads kb_1, kb_eval, open-webui_files)

# 5. Restore uploads volume
tar -xzf "$BK/uploads.tar.gz" -C volumes/

# 6. Rebuild and restart
cd compose
docker compose build open-webui
docker compose up -d

# 7. Verify full stack
sleep 30
curl -f http://localhost:6100/health
curl -s http://localhost:6333/collections | python3 -m json.tool
docker exec orgchat-postgres psql -U orgchat -d orgchat -c "SELECT COUNT(*) FROM kb_documents;"
```

**Recovery time:** 45–90 min.

---

## Decision matrix — which scenario do I use?

| Symptom | Scenario |
|---|---|
| Unit tests failing, containers rebuilt but not deployed | A |
| Container failing to start after deploy, data untouched | A |
| Reingest completed, quality worse than baseline | **B** |
| Orphan cleanup misbehaving, some chunks lost but Qdrant healthy | B (with snapshot recovery) |
| Qdrant red status / won't start | C |
| Postgres corruption or migration schema problem | C |
| Multiple symptoms or uncertain state | C |

When in doubt, Scenario C is safe — it restores everything to a verified snapshot.

---

## Post-rollback hygiene

After any rollback:

1. **Tag the rollback** — `git tag rollback-$(date +%Y%m%d-%H%M%S)` and push (optional).
2. **Preserve the failed state** — don't delete the uncommitted work or container logs. Copy to `/tmp/phase1a-failure-$(date +%s)/` for post-mortem.
3. **Rerun eval** — confirm `chunk_recall@10` and `faithfulness` match `baseline-pre-phase1a.json` within noise.
4. **Document the failure mode** — add a row to RAG.md §10 "Common failures" so the next attempt has the context.
5. **Retain the backup** until you've successfully re-attempted Phase 1a with a fix — do not delete `backups/20260421-182222/`.

---

## Pre-flight — one more sanity check before proceeding

Before any agent writes a line of code, verify:

```bash
cd /home/vogic/LocalRAG
BK=/home/vogic/LocalRAG/backups/20260421-182222

# Checksums intact
(cd "$BK" && sha256sum -c CHECKSUMS.sha256 --status && echo "backup OK")

# Current state still matches GIT_STATE.txt snapshot
[ "$(git rev-parse HEAD)" = "b5fe768"* ] || echo "WARNING: HEAD moved since backup"

# kb_1 point count still 2590 (nothing ran since backup)
curl -s http://localhost:6333/collections/kb_1 | \
  python3 -c "import sys,json; p=json.load(sys.stdin)['result']['points_count']; \
              print('kb_1 points:', p, 'OK' if p == 2590 else 'MOVED')"

# Postgres doc count still 110
docker exec orgchat-postgres psql -U orgchat -d orgchat -t -c \
  "SELECT COUNT(*) FROM kb_documents WHERE kb_id=1 AND deleted_at IS NULL;"
```

If any check fails, **stop and take a new backup** before proceeding.

---

## Contact — what to do if rollback itself fails

If Scenario C's snapshot recovery fails, the last defense is the raw Qdrant volume. It is NOT included in `scripts/backup.sh` by default. Take a volume copy NOW as belt-and-suspenders:

```bash
docker compose stop qdrant
docker run --rm -v orgchat_qdrant_data:/data:ro \
                -v "$PWD/backups/20260421-182222":/backup \
                alpine tar -czf /backup/qdrant_volume.tgz -C /data .
docker compose start qdrant
```

Then restore with:
```bash
docker compose stop qdrant
docker run --rm -v orgchat_qdrant_data:/data \
                -v "$PWD/backups/20260421-182222":/backup \
                alpine sh -c 'rm -rf /data/* && tar -xzf /backup/qdrant_volume.tgz -C /data'
docker compose start qdrant
```
