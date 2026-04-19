# Hybrid RAG: Dense + Sparse BM25 with RRF Fusion

Hybrid retrieval combines dense embeddings (semantic similarity via bge-m3)
with sparse BM25 (lexical match via Qdrant's native `Qdrant/bm25` model) and
fuses the two ranked lists server-side using Reciprocal Rank Fusion (RRF).
It typically improves recall@10 on queries that mix named entities, acronyms,
or rare terms (where dense retrieval under-weights exact matches) while
preserving the semantic strength of pure-dense search.

This feature is **off by default** — controlled by the `RAG_HYBRID`
environment variable. Legacy collections that pre-date this change continue
working via the dense-only path. See "Re-indexing legacy collections" below.

## Enabling hybrid

1. **Install the `hybrid` extra.** Pulls in `fastembed` (~150 MB including
   `onnxruntime`, `pillow`, `mmh3`, `py_rust_stemmers`). The BM25 model itself
   is only ~10 MB (stopwords + stemmer).
   ```bash
   pip install '.[hybrid]'
   ```
2. **Set the flag.**
   ```bash
   export RAG_HYBRID=1
   ```
3. **Create new KBs with sparse support.** Collections created via the upload
   path after this flag is set are created with both dense AND sparse named
   vectors; these automatically take the hybrid retrieval path. See the
   ingest path in `ext/services/ingest.py` — it calls
   `vector_store.ensure_collection(name, with_sparse=True)` when hybrid is on.
4. **Restart the service.** The flag is read at call-time but hot-reloading
   of collections in an already-running process is not yet implemented.

## Re-indexing legacy collections

Qdrant does not allow adding a named sparse vector to an existing collection
in-place. Legacy collections (e.g., `kb_1`, `kb_3`, `kb_4`, `kb_5`, `kb_eval`
as of P0) created before this change have only the unnamed dense vector and
therefore **cannot serve hybrid queries directly** — retrieval silently falls
back to dense-only for those collections.

A full re-index job that creates a fresh hybrid-shaped collection and
re-ingests every document is planned as `scripts/reindex_hybrid.py` (P2 task).
The interim `scripts/add_sparse_to_collection.py` stub covers the narrower
case of populating sparse vectors for an **already-hybrid-shaped** collection
(e.g., one you just created with `with_sparse=True`).

## Performance

Reference eval (`tests/eval/golden.jsonl`, 50 queries against `kb_eval`)
comparing dense-only vs. hybrid:

- **Chunk-recall@10:** hybrid typically adds +3 to +10 percentage points on
  lexical-heavy queries (named entities, acronyms, product SKUs). Queries
  that are already pure-semantic see ~0% change.
- **Latency (per-KB):** hybrid adds ~5–15 ms vs. dense-only. The BM25 sparse
  embedding (`embed_sparse_query`) itself runs in <1 ms on CPU after the
  ONNX session warms up; the extra cost is the server-side RRF fusion
  (Qdrant runs the two prefetch arms in parallel but must wait for both).

Run the benchmark yourself:
```bash
RAG_HYBRID=0 python tests/eval/run_eval.py --out dense.json
RAG_HYBRID=1 python tests/eval/run_eval.py --out hybrid.json
```

## Troubleshooting

### `ImportError: fastembed` even after `pip install '.[hybrid]'`

The `hybrid` extra requires `onnxruntime`, which sometimes fails to wheel-
install on exotic platforms. Workaround:
```bash
pip install fastembed --no-deps
pip install onnxruntime  # or onnxruntime-gpu
```

### First query slow (~10–30 seconds)

fastembed downloads the Qdrant/bm25 model on first use (stopwords + stemmer,
~10 MB). The download is cached under `~/.cache/fastembed/` and subsequent
queries are fast. You can pre-warm by running:
```python
from ext.services.sparse_embedder import embed_sparse_query
embed_sparse_query("warmup")
```

### Query returns fewer results than expected

Hybrid retrieval uses `limit*2` candidates per prefetch arm, fused via RRF.
If you see fewer than `limit` results, verify the collection actually has
both dense and sparse data — the BM25 prefetch returns nothing if no points
were upserted with sparse vectors. Check via:
```bash
curl http://localhost:6333/collections/kb_new | jq .result.config.params
```

### Regression: dense-only queries mismatch pre-hybrid output

They shouldn't — the `RAG_HYBRID=0` path is byte-identical to pre-P1.1
behaviour. If you see differences, confirm the env var is actually 0 (not
"false" or "False" — only "0" disables) and that `ensure_collection` was
called WITHOUT `with_sparse=True` for the collection in question.
