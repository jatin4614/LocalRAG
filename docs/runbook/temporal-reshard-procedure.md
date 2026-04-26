# Temporal Reshard Procedure (`kb_1_v3` → `kb_1_v4`)

**Plan B Phase 5.4.**

**Purpose:** migrate the post-Plan-A canonical collection `kb_1_v3` (dense + sparse + ColBERT + `context_prefix`) into a temporally sharded collection `kb_1_v4` with `shard_key="YYYY-MM"`.

**Schedule:** off-peak window. The reshard reads + writes the entire collection and must not contend with live ingestion.

## Pre-requisites verified

- [ ] Plan A is shipped and `kb_1` alias points to `kb_1_v3`.
- [ ] `kb_1_v2` is on disk (Plan A 14-day rollback target). DO NOT delete during this window.
- [ ] Phase 5.1 (`ensure_collection_temporal`), 5.2 (date extraction), 5.3 (tier helpers) are merged.
- [ ] Phase 0 eval baseline + Phase 4 baseline both committed.
- [ ] Off-peak window confirmed; chat QPS low.
- [ ] **Qdrant cluster (distributed) mode is enabled.** Custom sharding is a
      cluster-mode feature; standalone Qdrant returns 400 "Distributed mode
      disabled" on `create_shard_key`. To enable on a single-node deployment,
      set `QDRANT__CLUSTER__ENABLED=true` on the qdrant service env (compose
      file). Verify with `curl http://localhost:6333/cluster | jq .result.status`
      → must be `"enabled"`. **Note:** flipping cluster mode on requires a
      Qdrant restart and a clean storage directory the first time (Qdrant
      will not promote an existing standalone storage to cluster format
      in-place). Mitigation: snapshot all collections, recreate as a
      cluster-mode node, restore from snapshot.

## Throttle policy

The reshard reads + writes Qdrant; it does NOT call vllm-chat or any LLM. Throttle is by Qdrant load only (not chat p95).

## Step-by-step

### 1. Snapshot the source

```bash
SOURCE=kb_1_v3
curl -s -X POST "http://localhost:6333/collections/$SOURCE/snapshots" \
  | python -m json.tool
# Note the snapshot path; this is your absolute rollback target
```

### 2. Dry-run the reshard against staging clone

Make a clone first to validate the script doesn't blow up:

```bash
SOURCE=kb_1_v3
STAGING=kb_1_v3_staging
curl -X POST "http://localhost:6333/collections/$STAGING/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "<snapshot_path_from_step_1>"}'

python scripts/reshard_kb_temporal.py \
  --source $STAGING \
  --target kb_1_v4_staging \
  --dry-run
```

Expected: prints per-shard_key counts, ratio of `filename` vs `frontmatter` vs `body` vs `ingest_default` origins.

### 3. Run the actual reshard against staging

```bash
python scripts/reshard_kb_temporal.py \
  --source $STAGING \
  --target kb_1_v4_staging \
  --batch-size 256
```

### 4. Eval against staging target

Create a temporary KB row in Postgres pointing at `kb_1_v4_staging`, then:

```bash
make eval-evolution KB_EVAL_ID=$NEW_KB_ID
```

Compare against `tests/eval/results/phase-4-baseline.json`. Gate: `chunk_recall@10` ≥ +5pp on `golden_evolution.jsonl`, no per-intent regression > 2pp on `golden_starter.jsonl`.

### 5. If staging gate passes — production reshard

```bash
python scripts/reshard_kb_temporal.py \
  --source kb_1_v3 \
  --target kb_1_v4 \
  --batch-size 256
```

Monitor: `nvidia-smi` (no GPU change expected — script only touches Qdrant), Qdrant logs, `docker stats orgchat-qdrant`.

### 6. Verify counts per shard

```bash
for sk in $(curl -s "http://localhost:6333/collections/kb_1_v4/cluster" \
  | python -c "import json,sys;print('\n'.join(s['shard_key'] for s in json.load(sys.stdin)['result']['local_shards'] if s.get('shard_key')))"); do
  count=$(curl -s "http://localhost:6333/collections/kb_1_v4/points/count" \
    -X POST -H "Content-Type: application/json" \
    -d "{\"filter\": {\"must\": [{\"key\": \"shard_key\", \"match\": {\"value\": \"$sk\"}}]}}" \
    | python -c "import json,sys;print(json.load(sys.stdin)['result']['count'])")
  echo "$sk $count"
done
```

### 7. Apply tier configuration

```bash
python - <<PY
import asyncio
from ext.services.vector_store import VectorStore, classify_tier
from ext.services.temporal_shard import iter_shard_keys
import datetime as dt

async def apply():
    vs = VectorStore(url="http://localhost:6333", vector_size=1024)
    today = dt.date.today()
    for sk in iter_shard_keys("2024-01", f"{today.year:04d}-{today.month:02d}"):
        tier = classify_tier(sk)
        print(f"{sk} -> {tier}")
        await vs.apply_tier_config("kb_1_v4", sk, tier)

asyncio.run(apply())
PY
```

### 8. Run eval against the new production collection

Move the kb_id row in Postgres to point at `kb_1_v4` (or use the alias swap to make this transparent):

```bash
make eval KB_EVAL_ID=$KB_ID
make eval-evolution KB_EVAL_ID=$KB_ID
```

Gate: same as step 4.

### 9. Alias swap (production cutover)

```bash
curl -X POST http://localhost:6333/collections/aliases \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      {"delete_alias": {"alias_name": "kb_1"}},
      {"create_alias": {"collection_name": "kb_1_v4", "alias_name": "kb_1"}}
    ]
  }'
```

### 10. Confirm cutover

```bash
curl -s -X POST http://localhost:6100/api/rag/retrieve \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN" \
  -d '{"query":"OFC roadmap","selected_kb_config":[{"kb_id":1}],"top_k":3}' \
  | python -m json.tool

# Inspect a hit's payload — must include shard_key
```

### 11. Hold rollback window

- Keep `kb_1_v3` for 14 days (mark read-only at app level).
- Plan A's `kb_1_v2` is ALSO still in its 14-day window — don't delete that either.
- Monitor `retrieval_ndcg_daily` for 14 days.
- If regression detected: alias swap back: `kb_1 → kb_1_v3`.

## Rollback

```bash
curl -X POST http://localhost:6333/collections/aliases \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      {"delete_alias": {"alias_name": "kb_1"}},
      {"create_alias": {"collection_name": "kb_1_v3", "alias_name": "kb_1"}}
    ]
  }'
```

Total revert time: < 1 second.

## Known failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| `create_shard_key` fails with "already exists" | Idempotency edge — script handles, but a misclick may have run partial reshard | Inspect with `curl /collections/kb_1_v4/cluster`; resume by running script (idempotent via UUID5 IDs) |
| Target count > source count | Duplicate points across multiple shard_keys (race in date extraction) | Compare `(doc_id, chunk_index)` distinct count instead of raw point count; if duplicates exist, drop kb_1_v4 + re-run |
| `_v4` shard_key origin mostly `ingest_default` | Date extraction unable to recover dates from existing collection | Verify ingest used filename-with-date convention; consider a one-off re-derivation pass that mines content harder |
| Live retrieval slower after swap | Tiered config not yet applied | Run step 7; cold shards default to in-RAM until reconfigured |
