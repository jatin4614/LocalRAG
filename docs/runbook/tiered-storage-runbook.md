# Tiered Storage Runbook

Plan B Phase 5.8.

## Tiers

| Tier | Age | Storage | Quantization |
|---|---|---|---|
| hot  | 0-2 months | in-RAM HNSW (`memmap_threshold=0`) | none |
| warm | 3-11 months | mmap on SSD (`memmap_threshold=20_000`) | none |
| cold | >= 12 months | on-disk + mmap | INT8 scalar (`always_ram=False`) |

Defaults: `RAG_TIER_HOT_MONTHS=3`, `RAG_TIER_WARM_MONTHS=12`. Tunable per
environment via env vars on the Celery worker.

## Daily cron

Celery Beat fires `ext.workers.scheduled_eval.tier_storage_cron` at 03:00
local. The task shells out to `scripts/tier_storage_cron.py` for each
collection in `RAG_TIER_COLLECTIONS` (default: `kb_1_v4`).

The cron is idempotent — Redis DB 5 (`tier:<col>:<sk>` keys) caches the
last-applied tier per shard so repeated runs that observe no boundary
crossing are no-ops. Operators can also run the script manually:

```bash
# Apply transitions
python scripts/tier_storage_cron.py --collection kb_1_v4

# Preview without writing
python scripts/tier_storage_cron.py --collection kb_1_v4 --dry-run
```

### Operator activation checklist

- [ ] Confirm Phase 5.4 reshard has produced `kb_1_v4` (or whichever
      collection you intend to manage).
- [ ] Confirm `redis://redis:6379/5` is reachable from the worker host.
- [ ] Set `RAG_TIER_COLLECTIONS=kb_1_v4` (override default if
      managing multiple collections).
- [ ] Start the Celery Beat service (compose: `celery-beat`). The beat
      entry `tier-storage-cron-daily-kb_1_v4` is registered at module
      load but inert until Beat is running.
- [ ] Verify after the first 03:00 fire: `redis-cli -n 5 KEYS 'tier:kb_1_v4:*'`
      returns one key per shard.

## Verification

After a transition (or daily), check:

```bash
# Last applied tiers
redis-cli -n 5 KEYS 'tier:kb_1_v4:*' | head

# Each key's value
for key in $(redis-cli -n 5 KEYS 'tier:kb_1_v4:*'); do
  val=$(redis-cli -n 5 GET "$key")
  echo "$key: $val"
done

# Qdrant collection config
curl -s http://localhost:6333/collections/kb_1_v4 | python -m json.tool | \
  grep -A 20 'optimizers_config\|quantization_config'
```

## Common operations

### Force a re-tier (e.g. after changing month thresholds)

```bash
redis-cli -n 5 --scan --pattern 'tier:kb_1_v4:*' | xargs -r redis-cli -n 5 DEL
python scripts/tier_storage_cron.py --collection kb_1_v4
```

### Pin a specific shard to hot (e.g. for an incident)

```bash
python - <<'PY'
import asyncio
from ext.services.vector_store import VectorStore

async def main():
    vs = VectorStore(url='http://localhost:6333', vector_size=1024)
    await vs.apply_tier_config(
        collection='kb_1_v4', shard_key='2024-06', tier='hot',
    )

asyncio.run(main())
PY

# Then prevent the cron from reverting the next morning:
redis-cli -n 5 SET 'tier:kb_1_v4:2024-06' 'hot'
```

### Disable cron entirely

Stop the Celery Beat service (compose: `docker compose stop celery-beat`).
The beat entry remains in `app.conf.beat_schedule` but no scheduler
is running to fire it. To remove it permanently from the registered set,
unset `RAG_TIER_COLLECTIONS=` (empty) before the worker boots.

### Add additional collections

```bash
# Add to existing list (worker env)
RAG_TIER_COLLECTIONS=kb_1_v4,kb_2_v4

# One beat entry per collection is registered automatically
# (key: tier-storage-cron-daily-<collection>).
```

## Per-shard health metrics

Phase 5.9 exposes the following Prometheus metrics. After tiering is
applied you should see `rag_shard_tier{tier="hot"}=1` for recent shards
and `rag_shard_tier{tier="cold"}=1` for shards >= 12 months old.

| Metric | Type | Labels |
|---|---|---|
| `rag_shard_point_count` | Gauge | `collection, shard_key` |
| `rag_shard_search_latency_seconds` | Histogram | `collection, shard_key` |
| `rag_shard_upsert_latency_seconds` | Histogram | `collection, shard_key` |
| `rag_shard_tier` | Gauge (1=current, 0=not) | `collection, shard_key, tier` |

Alerts live in `observability/prometheus/alerts-tiered-shards.yml`.
After deploying the file, reload Prometheus:

```bash
# Either SIGHUP or POST /-/reload (depending on your Prometheus config)
curl -s -XPOST http://localhost:9090/-/reload
```
