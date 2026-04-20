# RAG Vector Quantization (P3.2)

`ext/services/vector_store.py` can create and query Qdrant collections with
**scalar INT8 quantization**. This shrinks vector RAM by ~4× with < 2% recall
loss at the default `quantile=0.99`. It's a one-line opt-in for new
collections and a one-command retrofit for existing ones.

## What it does

Qdrant stores one fp32 copy of every vector *and* a compressed index. With
scalar INT8 quantization:

- Every 4-byte float is clamped to the quantile window and mapped to a single
  byte (0-255).
- The compressed index lives in RAM (`always_ram=True`) — fast approximate
  search.
- The original fp32 vectors move to disk (`on_disk=True` on `VectorParams`)
  and are only paged in when a query opts into rescoring.

**Memory impact for 1024-d bge-m3 embeddings**

| Points | fp32 RAM | INT8 RAM | Reduction |
|--------|---------:|---------:|----------:|
| 10 k   |   41 MB  |   10 MB  |   4×      |
| 100 k  |  410 MB  |  100 MB  |   4×      |
| 1 M    |  4.1 GB  |  1.0 GB  |   4×      |

The index lookup is actually *faster* than fp32 because the 4×-smaller
footprint fits more of the graph in CPU cache.

## Why not binary?

Binary quantization gives another 8× on top of INT8 but "falls apart below
1024 dimensions" (Qdrant docs). bge-m3 is exactly 1024-d — borderline. We
intentionally skip binary here; it can be opted into later per-collection if
a KB grows past several million points AND we've measured no recall
regression on that KB's eval set.

## Enabling

### New collections: `RAG_QDRANT_QUANTIZE=1`

Set the env once and every collection created by `VectorStore.ensure_collection`
picks up:

```python
ScalarQuantization(
    scalar=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=True,
    )
)
```

Per-call override wins over env:

```python
await vs.ensure_collection("kb_42", with_quantization=True)   # force on
await vs.ensure_collection("kb_43", with_quantization=False)  # force off
```

### Existing collections: `scripts/enable_quantization.py`

Qdrant supports changing `quantization_config` on a live collection via
`update_collection`. The engine builds the compressed index in the background
with zero downtime and no data rewrite.

```bash
# Plan (default dry-run — makes no writes).
python scripts/enable_quantization.py --qdrant-url http://localhost:6333

# Apply to every non-system collection.
python scripts/enable_quantization.py --apply

# Or apply to a specific subset.
python scripts/enable_quantization.py \
    --collections kb_eval,kb_1_v2,kb_3_v2,chat_private \
    --apply

# Tune the outlier clamp (default 0.99).
python scripts/enable_quantization.py --quantile 0.95 --apply
```

Idempotent — re-running with the same config is a no-op on Qdrant's side.

## Per-query rescoring

The INT8 index is approximate. For queries where the last percent of recall
matters, Qdrant supports a two-pass "oversample + rescore":

1. Pull `N × oversampling` candidates using the INT8 index (fast).
2. Rescore those candidates with the original fp32 vectors (precise).
3. Return the top N by the rescored scores.

`VectorStore.search` and `VectorStore.hybrid_search` attach
`QuantizationSearchParams(rescore=True, oversampling=2.0)` to every query by
default. Both are togglable:

```python
# Per-call: skip rescoring for this one query (latency-critical path).
hits = await vs.search(name, vec, rescore=False)

# Globally: disable rescoring by default.
# RAG_QDRANT_RESCORE=0   # on the VectorStore side

# Increase oversampling (pull 3× candidates instead of 2×).
# RAG_QDRANT_OVERSAMPLING=3.0
```

On a collection *without* quantization the hint is a silent no-op, so these
knobs are safe to leave on everywhere.

## Env reference

| Env var | Default | Effect |
|---------|--------:|--------|
| `RAG_QDRANT_QUANTIZE`     | `0`    | Create new collections with scalar INT8 quantization. |
| `RAG_QDRANT_QUANTILE`     | `0.99` | Outlier-clamp cutoff. Lower → tighter scaling, more speed, less recall. |
| `RAG_QDRANT_RESCORE`      | `1`    | Attach `QuantizationSearchParams(rescore=True, …)` to every query. |
| `RAG_QDRANT_OVERSAMPLING` | `2.0`  | Candidate multiplier during rescore. Higher → better recall, more I/O. |

## Measurement

### Check a collection's quantization config

```bash
curl -s http://localhost:6333/collections/kb_1 | jq '.result.config.quantization_config'
```

After `enable_quantization.py --apply` this should show:

```json
{
  "scalar": {
    "type": "int8",
    "quantile": 0.99,
    "always_ram": true
  }
}
```

### Recall regression check

Use the eval harness in `scripts/eval_retrieval.py` with two env snapshots:

```bash
# Baseline (no quantization).
RAG_QDRANT_QUANTIZE=0 python scripts/eval_retrieval.py --out /tmp/eval-fp32.json

# Quantized + rescore.
RAG_QDRANT_QUANTIZE=1 RAG_QDRANT_RESCORE=1 \
    python scripts/eval_retrieval.py --out /tmp/eval-int8.json

diff <(jq -r '.recall_at_5' /tmp/eval-fp32.json) \
     <(jq -r '.recall_at_5' /tmp/eval-int8.json)
```

Expected drop at `quantile=0.99`: within 0-2 percentage points. If recall
drops more than 3 pp, raise the quantile (to 0.995), raise oversampling (to
3.0), or disable quantization for that KB.

## Recommendations by KB size

### Small KB (< 10 k points)

Don't bother. RAM saved is a few MB and the extra indirection costs
latency. Leave `RAG_QDRANT_QUANTIZE=0`.

### Medium KB (10 k – 100 k points)

`RAG_QDRANT_QUANTIZE=1` + default rescore. Saves 30–100 MB RAM per
collection, recall drop is noise.

### Large KB (> 100 k points) — the target case

`RAG_QDRANT_QUANTIZE=1` + `RAG_QDRANT_ON_DISK_PAYLOAD=true`. Combined with
`on_disk=True` on vectors, the only thing in RAM is the HNSW graph + INT8
index — a year-long KB at 1 M+ points stays tractable on a 32 GB host.

If recall matters more than speed, bump `RAG_QDRANT_OVERSAMPLING` to 3.0 or
tighten `RAG_QDRANT_QUANTILE` to 0.995.

## FAQ

**Does enabling quantization on a live collection cause downtime?** No —
Qdrant's `update_collection` is a settings-only change. The compressed
index is rebuilt in the background; queries keep hitting the fp32 path
until the new index is ready, then switch atomically. Zero data rewrite.

**What about the existing fp32 vectors on disk after I enable quantization?**
They're kept — that's what `on_disk=True` means. Rescoring paths read them
from disk. To free the disk space you'd have to recreate the collection
from scratch (Qdrant can't "delete" original vectors when quantization is
enabled; that would break rescoring).

**I enabled `RAG_QDRANT_QUANTIZE=1` but old collections aren't compressed.**
Expected. The env only affects `ensure_collection` on *new* collection
creation. Run `scripts/enable_quantization.py --apply` to retrofit
existing ones.

**Why is rescore=True the default instead of False?** Because the rescore
pass is very cheap (2× oversampling, disk read of ~2× limit vectors), and
it buys back most of the recall loss. Turning it off only makes sense in
latency-critical paths where ±2 pp recall is acceptable.
