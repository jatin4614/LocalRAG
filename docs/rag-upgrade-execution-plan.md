# RAG Upgrade — Safe Execution Plan

**Date:** 2026-04-19
**Scope:** Implement the RAG pipeline upgrade described in `Ragupdate.md` **without modifying or breaking the currently running app on `main`**.
**Detailed task breakdown:** `~/.config/superpowers/worktrees/LocalRAG/rag-upgrade-p0/docs/superpowers/plans/2026-04-18-rag-pipeline-upgrade.md`

This document is the *safety + sequencing* plan. It answers: **how do we do this work without touching the live code or live data, and how do we recover if anything goes wrong?**

---

## 1. Guiding Principles

1. **Main's working tree is sacred.** No edits, no stashes, no rebases on `main` during this work. The 8 modified + 7 untracked files in `main`'s working tree stay exactly as-is.
2. **All code work happens in a git worktree on branch `rag-upgrade-p0`.** Separate working directory, separate branch, shared `.git` object store — no risk of accidentally committing to `main`.
3. **All runtime state (Postgres, Qdrant, Redis, uploaded blobs) is backed up before any migration or ingest change runs.**
4. **Every new behavior lands behind a feature flag that defaults OFF.** Merging to `main` later is code-only; behavior on `main` does not change until a flag is flipped.
5. **Eval harness goes in first.** We do not merge any quality change without a before/after number from the same golden set.
6. **Migrations are additive-only** (new columns, new tables). No drops, no renames, no type changes on existing columns.

---

## 2. Isolation Strategy (Hybrid A+B)

Chosen approach: **git worktree + full state backup**, work in-place against the same Docker stack.

### Why this combination
- **Worktree** gives code isolation — `main`'s files are never touched.
- **Backups** give data isolation — if a migration or ingest mutates Postgres/Qdrant in a way we don't like, we can restore.
- Running against the **same Docker stack** means we don't duplicate 50+ GB of model weights and don't burn a second GPU context.

### What is NOT isolated (and why it's acceptable)
- Postgres/Qdrant/Redis containers are shared between `main` and the worktree branch. → Acceptable because during the work window we are the only user; `main` is not serving live traffic.
- GPU VRAM is shared. → Acceptable; we do not load additional models for P0.

### If `main` must keep serving live users during the work window
Skip to Option C (namespaced parallel stack) — **not** covered in this plan. Requires a second compose project name, second set of ports, second Postgres/Qdrant/Redis volumes, second reverse-proxy vhost. Defer until/unless that requirement appears.

---

## 3. Pre-Flight (one-time, before any code change)

Run these in order. Each is recoverable; none modify `main`'s code.

### 3.1 Record current state
```bash
cd /home/vogic/LocalRAG
git rev-parse HEAD > /tmp/rag-upgrade-baseline-sha.txt
git status --short > /tmp/rag-upgrade-baseline-status.txt
docker compose -f compose/docker-compose.yml ps > /tmp/rag-upgrade-baseline-containers.txt
```

### 3.2 Backup runtime state
```bash
BACKUP_DIR=~/rag-upgrade-backups/$(date +%Y%m%d-%H%M%S)
mkdir -p "$BACKUP_DIR"

# Code + config (excludes venv, model weights, caches)
tar --exclude='.venv' --exclude='volumes/models' --exclude='__pycache__' \
    --exclude='node_modules' --exclude='.git' \
    -czf "$BACKUP_DIR/localrag-tree.tgz" -C /home/vogic LocalRAG

# Postgres full dump
docker exec orgchat-postgres pg_dumpall -U postgres | gzip > "$BACKUP_DIR/postgres.sql.gz"

# Qdrant snapshot per collection
for col in $(curl -s http://localhost:6333/collections | jq -r '.result.collections[].name'); do
  curl -s -X POST "http://localhost:6333/collections/$col/snapshots" > "$BACKUP_DIR/qdrant-$col.json"
done

# Record what we backed up
echo "baseline_sha=$(cat /tmp/rag-upgrade-baseline-sha.txt)" > "$BACKUP_DIR/MANIFEST"
echo "backup_at=$(date -Iseconds)" >> "$BACKUP_DIR/MANIFEST"
```

Rollback drill (do NOT run unless actually rolling back):
```bash
docker compose down
gunzip -c "$BACKUP_DIR/postgres.sql.gz" | docker exec -i orgchat-postgres psql -U postgres
# Qdrant: restore each snapshot via POST /collections/{name}/snapshots/recover
docker compose up -d
```

### 3.3 Verify worktree is clean
```bash
git -C ~/.config/superpowers/worktrees/LocalRAG/rag-upgrade-p0 status
git -C ~/.config/superpowers/worktrees/LocalRAG/rag-upgrade-p0 log -1
# Expect: clean tree, HEAD = ba28dd5
```

### 3.4 Baseline test suite
```bash
cd ~/.config/superpowers/worktrees/LocalRAG/rag-upgrade-p0
pytest -q 2>&1 | tee /tmp/rag-upgrade-baseline-tests.txt
# Expected: 5 failed, 91 passed (5 pre-existing auth failures — documented, out of scope)
```

---

## 4. Phased Workflow

All work happens in `~/.config/superpowers/worktrees/LocalRAG/rag-upgrade-p0`. Each phase ends with: (a) tests pass, (b) worktree commits clean, (c) eval numbers recorded.

### P0 — Foundations (must land before any quality work)

| # | Task | Risk to main | Flag | Rollback |
|---|------|--------------|------|----------|
| P0.0 | Preflight: fix 2 worktree-only test failures, confirm env | None — worktree only | — | git reset in worktree |
| P0.1 | **Eval harness** (`tests/eval/run_eval.py`, golden set, chunk_recall@K, MRR) | None | — | Delete eval dir |
| P0.2 | **Celery ingest worker** + blob store + DLQ | Additive: new service, new queue. `RAG_SYNC_INGEST=1` keeps existing sync path | `RAG_SYNC_INGEST` | Flip flag to 1 |
| P0.3 | **O(N) chunker** rewrite (fixes accidentally-quadratic decode) | Behavior-compatible; tokenizer unchanged | `RAG_CHUNKER_TOKENIZER=cl100k` | Revert file |
| P0.4 | **Structural extractors + `pipeline_version`** column (additive migration 004) | New column only, backfilled with current version string | — | Column is nullable; leave |
| P0.5 | **History-aware query rewrite** | Off by default | `RAG_DISABLE_REWRITE=1` | Flip flag |
| P0.6 | **Spotlighting** prompt-injection guards in retrieval context | Off by default | `RAG_SPOTLIGHT` | Flip flag |

**Exit criterion for P0:** eval harness green, Celery worker processes a real upload end-to-end with `RAG_SYNC_INGEST=0`, and `pytest` is no worse than baseline.

### P1 — Retrieval quality (gated on P0.1 eval numbers)

- P1.1 Hybrid search (BM25 via Qdrant's `bm25` + dense, RRF fusion) — flag `RAG_HYBRID`
- P1.2 Cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, CPU) — flag `RAG_RERANK`
- P1.3 MMR diversification — flag `RAG_MMR`
- P1.4 Context expansion (parent-document) — flag `RAG_CONTEXT_EXPAND`
- P1.5 Budget tokenizer aligned with chat model — flag `RAG_BUDGET_TOKENIZER`

Each P1 task must **beat baseline eval by ≥X%** (X to be set when P0.1 lands) before its flag default flips on.

### P2 — Multi-tenancy + operational hardening

- P2.1 `is_tenant=true` payload index on Qdrant
- P2.2 `owner_user_id` propagation through ingest + retrieval
- P2.3 Per-chat collection consolidation (single collection + tenant filter)
- P2.4 HNSW tuning, connection pooling, async client timeouts
- P2.5 Observability: retrieval spans, per-stage latency, per-KB hit counts
- P2.6 Semantic cache — flag `RAG_SEMCACHE`
- P2.7 Contextual retrieval for KBs — flag `RAG_CONTEXTUALIZE_KBS`
- P2.8 Blob retention policy — flag `RAG_RETAIN_BLOBS`

### P3 — Optional / experimental

- P3.1 Anthropic-style contextual retrieval (cost-bounded)
- P3.2 Quantization (scalar only; skip binary at 1024d)
- P3.3 HyDE
- P3.4 RAPTOR
- P3.5 RAGAS faithfulness scoring in eval

---

## 5. Merge-Back Protocol (worktree → main)

Do NOT fast-forward. When P0 is complete and eval numbers justify merge:

1. Worktree: ensure branch is rebased on current `main` (main will still have same 8 uncommitted files — those stay uncommitted).
2. Open a PR from `rag-upgrade-p0` against `main`.
3. Every new behavior is **flag-off by default** in the PR. Merge is a no-op in production until flags flip.
4. Re-run pre-flight backup before merge.
5. After merge, verify `docker compose up -d` brings stack up with flags off and existing chats work unchanged.
6. Flip flags one at a time with eval run between each.

---

## 6. Failure Modes and Recovery

| Failure | Detected by | Recovery |
|---------|-------------|----------|
| Migration applied partially | `pytest`, app crash on startup | Restore Postgres from `$BACKUP_DIR/postgres.sql.gz` |
| Qdrant collection corrupted by bad ingest | Eval recall drops to 0, search errors | Restore Qdrant snapshot from `$BACKUP_DIR/qdrant-*.json` |
| Celery queue stuck / Redis bloat | Worker logs, `redis-cli DBSIZE` | `docker compose restart redis`; DLQ drains handle it |
| Accidentally committed to `main` | `git log main` shows new commit | `git reset --hard $(cat /tmp/rag-upgrade-baseline-sha.txt)` (ONLY if not pushed) |
| Worktree branch corrupted | `git status` in worktree | `git worktree remove` + re-create from baseline SHA |
| Live-traffic regression (post-merge) | User reports, metrics | Flip all `RAG_*` flags off; code path reverts to baseline |

---

## 7. What This Plan Explicitly Does NOT Do

- Does not modify any file under `/home/vogic/LocalRAG/` tracked on `main`.
- Does not commit to `main`.
- Does not change Docker named volumes or container names.
- Does not drop or rename any Postgres column.
- Does not delete any Qdrant collection.
- Does not re-embed existing chunks (P0 only adds new `pipeline_version`; re-embedding is a P1+ decision gated on eval).
- Does not touch the upstream Open WebUI submodule.

---

## 8. Go / No-Go Checklist

Before I start P0.0, confirm:

- [ ] Approach approved: **worktree + full backup, in-place Docker stack**
- [ ] Backup directory location acceptable (default: `~/rag-upgrade-backups/`)
- [ ] Work window confirmed: **`main` has no live users during work**
- [ ] Owner for golden-set labeling (for P0.1 eval)
- [ ] Confirmed env values: `CHAT_MODEL`, `RAG_EMBEDDING_OPENAI_API_BASE_URL`
- [ ] OK to add new services to `docker-compose.yml` (Celery worker, Redis already present)

On "go" I start P0.0 in the worktree. No code outside the worktree is touched at any point.

---

## 9. References

- Detailed task-level plan: `~/.config/superpowers/worktrees/LocalRAG/rag-upgrade-p0/docs/superpowers/plans/2026-04-18-rag-pipeline-upgrade.md`
- Flaw analysis + target architecture: `/home/vogic/LocalRAG/Ragupdate.md`
- Project context: `/home/vogic/LocalRAG/CLAUDE.md`
