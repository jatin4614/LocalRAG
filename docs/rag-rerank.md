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

## Latency

Cross-encoder scoring is `O(N)` model passes over `(query, passage)` pairs.
The pipeline reranks the top-30 retrieval candidates, so expect:

* CPU inference, **batch size 8**: ~300-700 ms per query for top-30 on a
  modern x86 server. First call pays ~5-15 s for model download + load.
* GPU inference (if `torch.cuda.is_available()`): ~50-150 ms per query after
  warmup. Not the default target — keep the reranker on CPU so it does not
  compete with vLLM chat / embeddings for VRAM.

Even on CPU the reranker fits comfortably within typical RAG query budgets
(~1 s total). The worst case is the very first query after a cold start —
the ~560 MB model download dominates. Pre-warm in your deploy hook if this
matters:

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

The reranker loads ~1.2 GB of RAM when running (model weights + activations).
For a 32 GB host this is negligible, but note that each additional uvicorn
worker will hold its own copy. Consider running the reranker in a single
worker behind a semaphore for very memory-tight deploys.

### Interaction with hybrid retrieval

Independent. `RAG_HYBRID=1` changes what `retrieve()` returns; `RAG_RERANK=1`
changes how those results are ordered before budget truncation. Enable
either, both, or neither.
