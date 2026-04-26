# Temporal Reshard Checklist (operator-facing)

Use this checklist in the off-hours window to execute Task 5.4
(`docs/runbook/temporal-reshard-procedure.md`).

**Pre-requisites verified:**

- [ ] Plan A Phase 3.7 re-ingest is complete (`kb_1` -> `kb_1_v3` alias active for >= 7 days).
- [ ] Plan A's `kb_1_v2` is still in its 14-day rollback window — DO NOT delete during this window.
- [ ] Phase 5.1, 5.2, 5.3, 5.5, 5.6, 5.7 merged to main.
- [ ] Phase 0 baseline + Phase 4 baseline both committed.
- [ ] `tests/eval/golden_evolution.jsonl` committed with at least 30 queries.
- [ ] Off-peak window confirmed; chat QPS low.
- [ ] `nvidia-smi` GPU 1 <= 50% (Phase 4 vllm-qu does not contend with this work but watch anyway).
- [ ] **Qdrant cluster mode is enabled** — `curl http://localhost:6333/cluster | jq .result.status` returns `"enabled"`. See procedure doc for the one-time enablement steps.

**Execution:**

- [ ] Step 1 — Snapshot `kb_1_v3` (per `docs/runbook/temporal-reshard-procedure.md` Section 1).
- [ ] Step 2 — Dry-run reshard against staging clone (Section 2). Inspect shard_key origin distribution; at least 90% of docs must have non-`ingest_default` origin or stop and fix the date convention upstream.
- [ ] Step 3 — Run actual reshard against staging (Section 3).
- [ ] Step 4 — `make eval-evolution KB_EVAL_ID=$STAGING_KB_ID`. Confirm gate passes (Section 4).
- [ ] Step 5 — Production reshard (Section 5). Monitor Qdrant disk I/O and `rag_qdrant_upsert_latency_seconds` p95.
- [ ] Step 6 — Per-shard count verification (Section 6).
- [ ] Step 7 — Apply tier configuration via the Python helper (Section 7). Then update Redis DB 5 cache (`tier:kb_1_v4:<sk>`) so the daily cron is a no-op on first run.
- [ ] Step 8 — `make eval` and `make eval-evolution` against the new collection (Section 8). Confirm gates pass.
- [ ] Step 9 — Alias swap (Section 9).
- [ ] Step 10 — Spot-check live retrieval (Section 10).
- [ ] Step 11 — Mark `kb_1_v3` read-only for 14 days (Section 11).

**Post-window:**

- [ ] Update `docs/runbook/flag-reference.md` and `docs/runbook/plan-b-flag-reference.md`:
  - Move Phase 5 flags out of "NOT YET SHIPPED" section.
  - Set `RAG_SHARDING_ENABLED=1` for `kb_1_v4` in per-KB config.
  - Set `RAG_TEMPORAL_LEVELS=1` for `kb_1_v4` in per-KB config.
  - Set `RAG_RAPTOR_TEMPORAL=1` for `kb_1_v4` (see flag audit notes).
- [ ] Set calendar reminder for Day 14 to drop `kb_1_v3` (and `kb_1_v2` if its window has also passed).
- [ ] Build the temporal RAPTOR tree for `kb_1_v4` (separate operation; see below).
- [ ] Start Celery Beat to activate the daily tier cron (`docs/runbook/tiered-storage-runbook.md`).
- [ ] Reload Prometheus to pick up `observability/prometheus/alerts-tiered-shards.yml`:
      `curl -s -XPOST http://localhost:9090/-/reload`.
- [ ] Announce completion to the team.

**Building the temporal RAPTOR tree:**

After resharding, the L0 chunks exist in `kb_1_v4` but L1-L4 nodes do not.
Build them via a script that mirrors the Plan A 3.7 pattern (snapshot ->
build -> upsert -> verify):

```bash
python - <<'PY'
import asyncio
from qdrant_client import AsyncQdrantClient
from ext.services.temporal_raptor import build_temporal_tree
# ... wire summarize via vllm-chat, embed via TEI, then upsert returned nodes
PY
```

Operator: a complete script for this is
`scripts/build_temporal_tree.py` (mirrored on the pattern from
Plan A 3.7's `scripts/reingest_kb.py`). If the script does not yet exist,
create it from the `build_temporal_tree` helper in
`ext/services/temporal_raptor.py`.

**If any step fails:** follow Rollback in `temporal-reshard-procedure.md`.
Log the failure mode in `docs/runbook/troubleshooting.md` under
"Temporal reshard issues."

## Phase 5 completion gate

Before announcing Phase 5 done, verify:

- [ ] All Phase 5 unit + integration tests pass.
- [ ] `scripts/reshard_kb_temporal.py --dry-run` against `kb_1_v3` succeeds; shard_key origin distribution shows >= 90% non-`ingest_default`.
- [ ] Production reshard executed during off-peak window. Per-shard counts match source.
- [ ] Tier config applied; `nvidia-smi` post-tier shows reduced GPU 1 memory or unchanged (cold tier moves data off GPU).
- [ ] `make eval-evolution KB_EVAL_ID=$KB_ID` shows `chunk_recall@10` >= +5pp on evolution stratum vs Plan A baseline.
- [ ] `make eval KB_EVAL_ID=$KB_ID` shows no per-intent regression > 2pp on `golden_starter`.
- [ ] Phase 5 baseline JSON committed at `tests/eval/results/phase-5-baseline.json`.
- [ ] Tier cron has run successfully for >= 3 consecutive days.
- [ ] Per-shard metrics emitting at `/metrics`.
