# Cross-Encoder Reranker (P1.2)

A true cross-encoder reranker using `BAAI/bge-reranker-v2-m3` replaces the
legacy per-KB max-normalizer when `RAG_RERANK=1`. Default (flag unset or `0`)
is **byte-identical** to the pre-P1.2 release — the cross-encoder module is
not even imported.

This feature is **off by default**. Eval gating decides when to flip it on
cluster-wide.

## How it differs from the legacy reranker

The legacy `rerank()` function in `ext/services/reranker.py` is not actually
a reranker; it is a per-KB max-normalizer with a fast-path short-circuit:

* **Fast path:** if top-1 raw score / top-2 raw score > 2.0, return unchanged.
* **Otherwise:** max-normalize scores per KB, then flat-sort by the normalized
  score. Cheap, but it never consults the query content — it only rescales
  dense similarity scores.

The cross-encoder reranker actually scores each `(query, passage)` pair with
the bge-reranker-v2-m3 model, capturing full query-passage interaction. This
is what "rerank" means in most RAG literature.

## Enabling the cross-encoder reranker

1. **Install the `rerank` extra.** Pulls in `sentence-transformers` and
   `torch` (~1 GB+ on first install). The BAAI/bge-reranker-v2-m3 model
   itself (~560 MB) downloads on first call and caches under
   `~/.cache/huggingface/hub/`.
   ```bash
   pip install '.[rerank]'
   ```
2. **Set the flag.**
   ```bash
   export RAG_RERANK=1
   ```
3. **Restart the service.** The flag is read at call time, so hot toggling
   works within a running process, but production flips should be a restart.

### Advanced tuning

These env vars are read at model-load time only (first call after process
start). They do not need to be set for the feature flag to engage.

* `RAG_RERANK_MODEL` — override the default model name
  (`BAAI/bge-reranker-v2-m3`).
* `RAG_RERANK_MAX_LEN` — max sequence length passed to the CrossEncoder
  (default `512`).
* `RAG_RERANK_DEVICE` — `auto` (default) | `cpu` | `cuda` | `cuda:N`. In
  `auto` mode the reranker picks `cuda:0` when `torch.cuda.is_available()`
  and falls back to `cpu` otherwise. Explicit values bypass the probe.
* `RAG_RERANK_BATCH_SIZE` — override the auto default (32 on GPU, 8 on CPU).
  Read every call; adjustable without restart.

## GPU auto-select

With `RAG_RERANK_DEVICE=auto` (the default), the cross-encoder pins to
`cuda:0` whenever CUDA is available. First call loads the ~560 MB model
onto the GPU (~1-3 s after the HuggingFace cache is warm); subsequent
queries stay hot.

Force CPU with `RAG_RERANK_DEVICE=cpu` if you need to keep the GPU
exclusively for vLLM chat / embeddings. On shared-GPU deployments the
reranker uses ~1.2 GB of VRAM (weights + activations), which is
comfortable on an RTX 6000 Ada (32 GB) alongside Qwen2.5-14B-AWQ + TEI.

## Redis score cache

Cross-encoder scores are cached in Redis per `(model, query, passage)`
tuple. Repeated queries — or overlapping chunks across adjacent queries
— skip model inference entirely.

* Key format: `rerank:{model}:{sha1(query)[:16]}:{sha1(passage)[:16]}`
* Value: float score (ASCII bytes)
* TTL: `RAG_RERANK_CACHE_TTL` seconds (default 300)
* Connection URL: `RAG_REDIS_URL` (default `redis://redis:6379/0` — the
  in-cluster alias; override when running tests from the host)
* Kill switch: `RAG_RERANK_CACHE_DISABLED=1` bypasses entirely

The cache is **fail-open**: any Redis exception (timeout, connection
refused, corrupt value) is swallowed and the score is recomputed by the
model. Retrieval never fails because of the cache.

Warm queries (full cache hit over 30 pairs) complete in **<10 ms** —
dominated by the `MGET` + local deserialization.

## Latency

Cross-encoder scoring is `O(N)` model passes over `(query, passage)` pairs.
The pipeline reranks the top-30 retrieval candidates. Measured on a single
RTX 6000 Ada (Ampere, 32 GB) running Qwen2.5-14B-AWQ + TEI + bge-reranker-v2-m3:

| Scenario                           | Latency (top-30)   |
| ---------------------------------- | ------------------ |
| CPU, batch 8, cache cold           | ~11 000 ms         |
| CPU, batch 8, cache hit            | <10 ms             |
| GPU (cuda:0), batch 32, cache cold | **~30-50 ms**      |
| GPU (cuda:0), cache hit            | **<10 ms**         |
| Very first call (model download)   | ~5-15 s (one-time) |

Pre-warm in your deploy hook to pay the model-load cost at startup, not on
the first user query:

```python
from ext.services.cross_encoder_reranker import score_pairs
score_pairs("warmup", ["seed passage"])
```

## Expected quality gain

Reference eval (`tests/eval/golden.jsonl`, 50 queries against `kb_eval`)
comparing the three paths:

* **Dense-only + legacy reranker (default, P1.2 and earlier).** Baseline.
* **Dense + hybrid + legacy reranker (`RAG_HYBRID=1`, P1.1).** +3 to +10 pp
  chunk-recall@10 on lexical-heavy queries.
* **Dense + hybrid + cross-encoder (`RAG_HYBRID=1 RAG_RERANK=1`).** Another
  +5 to +15 pp chunk-recall@10 on tail queries (ambiguous phrasings,
  paraphrased questions). This is where the cross-encoder pays for itself.

Gate the default flip on pre/post eval:

```bash
RAG_RERANK=0 python tests/eval/run_eval.py --out baseline.json
RAG_RERANK=1 python tests/eval/run_eval.py --out rerank.json
```

## Fail-open design

`rerank_with_flag` never crashes retrieval:

* If `sentence-transformers` is not installed, the module import fails and
  we silently fall back to the legacy reranker.
* If the model download fails (offline, cache corruption), same thing.
* If inference itself raises, same thing.

This keeps retrieval working in degraded environments. To surface failures
during debugging, run with `PYTHONWARNINGS=error` or add logging at the
except block in `ext/services/reranker.py::rerank_with_flag`.

## Troubleshooting

### First query slow (~10-30 s)

Model download on first call. Caches under
`~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/`. Pre-warm via
the snippet above.

### Results identical to legacy path

Check the flag is actually engaged:

```bash
echo $RAG_RERANK
python -c "import os; print(os.environ.get('RAG_RERANK'))"
```

Only the exact string `"1"` enables the cross-encoder (not `"true"`, not
`"yes"`). If the flag is set but results look unchanged, check service logs
for a silent fail-open — the dispatcher in `rerank_with_flag` swallows
exceptions by design.

### Memory usage

The reranker loads ~1.2 GB when running (weights + activations). On CPU
this is host RAM; on GPU it's VRAM. For a 32 GB RTX 6000 Ada sharing with
Qwen2.5-14B-AWQ (12 GB) + TEI (3 GB), there is ample headroom.

Each additional uvicorn worker holds its own copy. For memory-tight deploys
(or to avoid duplicating the VRAM allocation), run the reranker in a single
worker behind a semaphore.

### Batch size

Default is **32 on GPU** and **8 on CPU** — both honoured automatically
based on `_resolve_device()`. Override with `RAG_RERANK_BATCH_SIZE=N`
(read every call, no restart needed).

Larger batches improve GPU utilisation but increase per-query latency for
very small result sets. The default 32 is a good fit for the typical
top-30 rerank window.

### Cache tuning

* Default TTL (300 s) is aimed at short-lived duplicate queries during an
  interactive chat. Bump it for retrieval patterns with heavy replay (agent
  loops, stable knowledge bases): `RAG_RERANK_CACHE_TTL=3600`.
* Flush on model swap: when `RAG_RERANK_MODEL` changes, old entries are
  keyed under the old model name and are simply dead weight (the new model
  won't find them). They expire naturally via TTL; for an immediate flush
  call `rerank_cache.clear_all(model="old-model")` or drop the `rerank:*`
  keys directly.
* Cache hit verification: `redis-cli --scan --pattern 'rerank:*' | head`
  from inside the orgchat-net (e.g. `docker exec orgchat-redis redis-cli`).

### Interaction with hybrid retrieval

Independent. `RAG_HYBRID=1` changes what `retrieve()` returns; `RAG_RERANK=1`
changes how those results are ordered before budget truncation. Enable
either, both, or neither.
