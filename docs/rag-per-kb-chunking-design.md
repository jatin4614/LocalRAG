# Per-KB chunking configuration — design

**Status:** DRAFT — do not deploy until Option 1 (chunk_tokens=300 on kb_1) is signed off
**Motivation:** heterogeneous datasets need different chunk_tokens targets. The Option 1 result (300 wins for short-doc fact-heavy reports) proves 800 is not universal.

---

## 1. What we learned

The default `CHUNK_SIZE=800` is a one-size-fits-none compromise:

| Corpus profile | Good chunk target | Why |
|---|---|---|
| Short fact-heavy reports (2-3 pages, many specific numbers/names) | 200-300 tokens | Smaller chunks preserve per-fact embedding specificity; bge-m3 matches exact phrases. Too-large chunks average multiple facts → dilute. |
| Long policy / whitepaper (20+ pages, section-based) | 600-800 tokens | Large chunks capture full arguments; enough context for LLM to understand. |
| Conversational threads (chat logs, Q&A) | 300-500 tokens | Capture multi-turn context but not too much off-topic. |
| Code / API docs | 400-600 tokens | Function-level granularity; preserve signatures + doc comments together. |
| Tables-heavy (financial reports, CSV-like) | Don't chunk — use atomic tables (`block_type="table"`) | Already handled by Phase 1a. Table rows lose meaning if split. |

A single flag on the whole service can't serve all of these. Per-KB is the right scope because KB admins know their corpus character.

---

## 2. Design

### 2.1 Add two keys to `rag_config` JSONB schema

Edit `ext/services/kb_config.py`:

```python
ALLOWED_KEYS = {
    "rerank": bool, "rerank_top_k": int,
    "mmr": bool, "mmr_lambda": float,
    "context_expand": bool, "context_expand_window": int,
    "spotlight": bool, "semcache": bool,
    "hyde": bool, "hyde_n": int,
    "contextualize_on_ingest": bool,
    # NEW (Option 1 follow-up):
    "chunk_tokens": int,       # 100 ≤ v ≤ 2000
    "overlap_tokens": int,     # 0  ≤ v ≤ chunk_tokens // 2
}
```

Validation in `validate_config()`:

```python
if "chunk_tokens" in cfg:
    v = int(cfg["chunk_tokens"])
    if not 100 <= v <= 2000:
        raise ValueError("chunk_tokens must be 100..2000")
if "overlap_tokens" in cfg:
    v = int(cfg["overlap_tokens"])
    ct = int(cfg.get("chunk_tokens", 800))
    if v < 0 or v >= ct // 2:
        raise ValueError("overlap_tokens must be 0..chunk_tokens//2")
```

### 2.2 Read at ingest time

`ext/routers/upload.py` already has access to `kb_id` on every KB upload. Before calling `ingest_bytes`, fetch the target KB's `rag_config` and extract the chunk-size pair:

```python
kb_cfg = await load_kb_rag_config(kb_id, session)
chunk_tokens = kb_cfg.get("chunk_tokens") or int(os.environ.get("CHUNK_SIZE", "800"))
overlap_tokens = kb_cfg.get("overlap_tokens") or int(os.environ.get("CHUNK_OVERLAP", "100"))

await ingest_bytes(..., chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
```

Chat-private uploads have no KB — they keep the env default.

### 2.3 Reingest CLI already accepts these (as of this session's patch)

`scripts/reingest_all.py --chunk-tokens N --chunk-overlap M` is live. An admin can also run it with no explicit flags and the script will read the KB's rag_config automatically (after step 2.2 lands and we add a per-doc lookup inside the reingest loop).

### 2.4 Merge policy — what happens when multiple KBs are selected in a chat?

Chat retrieval (not ingest) doesn't use chunk_tokens — it uses whatever chunks the KB was ingested with. So the UNION/MAX merge policy in `kb_config.merge_configs()` is only about retrieval-time behavior (rerank, mmr, etc.). `chunk_tokens` is ingest-time only and does NOT participate in merge.

Document this in `RAG.md §7`:

> `chunk_tokens` and `overlap_tokens` are ingest-time per-KB settings. They do not merge across multiple selected KBs at chat time — each KB's chunks were produced with its own target, and retrieval just reads whatever is there.

---

## 3. How a KB admin picks the target

Short decision tree:

```
Is the corpus mostly docs ≤ 5 pages with dense facts? → 200-300
Is the corpus docs 5-20 pages with sections?        → 400-600
Is the corpus docs 20+ pages / whitepapers?         → 600-800
Mixed? Pick the MAJORITY profile; tables are atomic regardless.
Unsure?                                             → 400 (middle)
```

Operational workflow:

1. Upload 5-10 representative docs with default 800.
2. Run `python tests/eval/chunk_size_histogram.py --collection kb_N`.
3. If median chunk is < 50 tokens → fragmentation, increase target.
4. If median chunk is near 800 → docs are long, keep default.
5. If median chunk is 100-300 → corpus is short-doc, consider lowering to 200-300.
6. Ask a few specific queries via the UI; if answers are wrong but docs are correct, *smaller* target probably helps precision.

Record the decision in `knowledge_bases.rag_config`.

---

## 4. Admin API surface

Add a PATCH endpoint `PATCH /api/kb/{kb_id}/rag_config` (admin-guarded) that accepts the updated JSON and calls `validate_config()`. Already half-implemented in `ext/routers/kb_admin.py` — extend to accept `chunk_tokens` and `overlap_tokens`.

Example:

```bash
curl -X PATCH http://localhost:6100/api/kb/1/rag_config \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"chunk_tokens": 300, "overlap_tokens": 50}'
```

After setting, existing chunks stay as-is. Admin runs reingest to apply:

```bash
python scripts/reingest_all.py --kb 1
# (no explicit --chunk-tokens flag — reads from rag_config)
```

---

## 5. What this does NOT do

- **Does not auto-detect the right target.** An auto-tuner needs its own design — it would chunk at 3-4 candidate targets, run a held-out eval on each, pick the winner. Useful but a separate project (call it Phase 1c).
- **Does not migrate existing KBs automatically.** kb_1 becomes the reference case. Future KBs set their target at creation time.
- **Does not change retrieval-time behavior** — just ingest-time chunking.
- **Does not apply to `chat_private`** — chat-scoped uploads use the process-level default (fine; they're usually short anyway).

---

## 6. Rollout plan

After Option 1 signs off:

1. Add the two keys to `ALLOWED_KEYS` + validation. 30 min.
2. Plumb them through `upload.py` and (inside-loop) `reingest_all.py`. 30 min.
3. Unit tests: `test_kb_config_chunk_tokens.py` covering validation bounds + defaults. 30 min.
4. Document in `RAG.md §7`. 10 min.
5. Deploy, stamp kb_1 with `{"chunk_tokens": 300, "overlap_tokens": 50}` so future re-ingests auto-use the right value.
6. When the next KB is created, prompt the admin via UI/README for the chunk_tokens value.

Total effort: ~2 hours to ship. Zero downtime rollout (default behavior unchanged for KBs without the keys).

---

## 7. Future extensions (deferred)

- **Per-subtag overrides.** A KB might have a "specs" subtag (long) and a "daily_reports" subtag (short). Need a second-level config. Cost-benefit TBD.
- **Auto-profiling.** Ingest the first 5 docs at 400 tokens, measure chunk-size distribution, recommend a target. Fully automated.
- **Content-type-aware targets.** PDFs with lots of tables → lower target; markdown with heavy headings → higher. Would need block-type-weighted chunking. Real work.
