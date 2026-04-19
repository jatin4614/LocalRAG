# RAG HNSW Tuning (P2.4)

`ext/services/vector_store.py` exposes six env-tunable knobs that control the
HNSW graph shape, per-query search width, and the async HTTP connection pool.
All defaults are chosen to be a safe baseline for a 10k–100k-chunk KB; none
change behavior on existing Qdrant collections until those collections are
recreated or explicitly re-optimized.

## Knobs

| Env var                           | Default | Effect |
|-----------------------------------|--------:|--------|
| `RAG_QDRANT_M`                    | `16`    | HNSW edges per node. Higher → better recall, more graph memory. Qdrant default. |
| `RAG_QDRANT_EF_CONSTRUCT`         | `200`   | Build-time search width. **Bumped from Qdrant's 100** → +2-3 pp recall at modest index-time cost. |
| `RAG_QDRANT_EF`                   | `128`   | Per-query search width (HNSW `ef`). Higher → better recall, slower. Applied via `SearchParams(hnsw_ef=…)`. |
| `RAG_QDRANT_FULL_SCAN_THRESHOLD`  | `10000` | If the filter keeps ≤ this many points, Qdrant runs a flat scan instead of HNSW. Qdrant default. |
| `RAG_QDRANT_ON_DISK_PAYLOAD`      | `false` | When `true`, payloads live on disk (RAM only for graph). Keep `false` unless memory-pressured. |
| `RAG_QDRANT_MAX_CONNS`            | `32`    | httpx connection pool size for `AsyncQdrantClient` (up from default ~100 shared → dedicated per-process). |

Client-side also: `timeout=30.0s` on `AsyncQdrantClient` so big batch upserts
don't fail on the default 5s deadline.

## Recommendations by KB size

### Small KB (< 10k chunks)

- Keep defaults. `full_scan_threshold=10000` means **filtered queries fall
  through to flat scan** anyway — indistinguishable latency, perfect recall.
- `RAG_QDRANT_ON_DISK_PAYLOAD=false` (RAM is cheap at this size).

### Medium KB (10k–100k chunks)

- Keep defaults. HNSW kicks in above 10k, and `m=16 + ef_construct=200 + ef=128`
  is the sweet spot for ~95-98 % recall at p95 < 15 ms.
- If you measure recall < 0.95 on your eval set, bump `RAG_QDRANT_EF` to 192
  first (per-query, no rebuild needed). If still short, raise
  `RAG_QDRANT_EF_CONSTRUCT` to 300 and reindex (see below).

### Large KB (> 100k chunks)

- `RAG_QDRANT_M=32` — better recall at the cost of ~2× graph memory.
- `RAG_QDRANT_EF_CONSTRUCT=400` — longer to build, but only once.
- Consider `RAG_QDRANT_ON_DISK_PAYLOAD=true` if host RAM is tight. Graph stays
  in RAM; payloads go to disk (adds ~2-5 ms per hit to fetch text).
- `RAG_QDRANT_MAX_CONNS=64` if you expect > 20 concurrent users.
- Monitor `rag_retrieval_latency_seconds` (Prometheus) before and after.

### High-concurrency deployments

- `RAG_QDRANT_MAX_CONNS` scales with concurrent ingest + chat. Rule of thumb:
  `max_conns ≥ 2 × max_concurrent_users` plus overhead for ingest.

## Reindexing after a tuning change

HNSW `m` / `ef_construct` only affect **new** inserts. Changing them on a live
collection doesn't rebuild existing points. Two options:

### Option A: recreate the collection

```bash
python scripts/reindex_hybrid.py --kb <kb_id> --force
```

This drops + recreates the Qdrant collection with the current env values,
then re-embeds every `kb_documents` row with `status='indexed'`.

### Option B: force optimizer to re-index in place

Call `VectorStore.optimize_collection(name)` — sets
`OptimizersConfigDiff(indexing_threshold=0)`, telling Qdrant to re-balance
the HNSW graph for all existing points. Cheaper than a full re-embed, but
only re-links edges — vectors themselves aren't re-embedded. Use after
raising `m` / `ef_construct` on a collection whose points are fine but whose
graph is stale.

## Measuring impact

Run the eval matrix with two env snapshots (before + after). The golden file
is at `tests/eval/golden.jsonl`:

```bash
# Baseline
RAG_QDRANT_EF_CONSTRUCT=100 RAG_QDRANT_EF=128 \
    python scripts/eval_retrieval.py --out /tmp/eval-before.json

# With tuning
RAG_QDRANT_EF_CONSTRUCT=200 RAG_QDRANT_EF=128 \
    python scripts/eval_retrieval.py --out /tmp/eval-after.json

# Compare
diff <(jq -r '.recall_at_5' /tmp/eval-before.json) \
     <(jq -r '.recall_at_5' /tmp/eval-after.json)
```

Tracked metrics (emitted via `ext/services/metrics.py`):

- `rag_retrieval_latency_seconds{stage="search"}` — p50/p95/p99
- `rag_retrieval_latency_seconds{stage="hybrid_search"}`
- `rag_retrieval_hits_total` — count, broken down by KB

If p95 regresses > 20 % after a tuning change, roll back. If recall barely
moves, the KB is probably small enough that full-scan already kicks in —
tuning HNSW won't help.

## FAQ

**Why did you bump `ef_construct` from 100 to 200 by default?** Because the
Qdrant project's own benchmarks show this range gives +2-3 pp recall for a
one-time index-build cost of ~2×. Build happens offline during ingest — a
user never waits for it. Query-time behavior is unchanged.

**Why didn't you raise `m`?** Because `m=32` doubles graph memory per
collection. For an org running dozens of KBs on one host, that adds up fast.
`m=16` is a conservative default; operators with large single-KB deployments
should raise it explicitly.

**Why is `on_disk_payload` false?** Because the payload includes the chunk
text (up to ~800 tokens ≈ 3.2 kB). Keeping it in RAM shaves 2-5 ms per hit
off the response. Flip it to `true` only when memory is the binding
constraint.
