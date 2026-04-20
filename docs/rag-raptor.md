# RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval

**Status:** P3.4 — flag-gated off by default.
**Reference:** Sarthi et al. 2024, "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (https://arxiv.org/abs/2401.18059).

## What it does

At ingest, each uploaded document is expanded into a **tree of LLM
summaries** instead of a flat list of chunks:

```
level 0 (leaves)       chunk_0  chunk_1  chunk_2  chunk_3  ...  chunk_N
                          \  /     \  /     \  /
level 1 (cluster sum)   summary_A   summary_B   summary_C
                                \       |        /
level 2 (higher sum)           summary_AB_C
                                     |
root                             root_summary
```

Leaves are the original 800-token chunks, verbatim. Level-1 nodes are
summaries of **clusters of leaves** (via Gaussian Mixture Models over
the chunk embeddings). Level-2 clusters summaries, and so on. When the
recursion bottoms out at >1 surviving node, a single root summary is
emitted over all of them.

**Every node — leaf or summary — is upserted into the same Qdrant
collection** with a `chunk_level` payload field (0 = leaf, 1+ = summary).
Retrieval sees them all: the reranker and MMR see each node as a
"chunk" and pick the best fit, whether that's a specific leaf or a
higher-level summary that better captures the macro-topic of the query.

## Why

Flat chunking fails on queries whose answer **spans many chunks** — a
"what are our policies around X" query against a 200-page handbook
retrieves a handful of relevant leaves but misses the synthesis that
a human reader would produce after reading the whole thing. RAPTOR
fixes this by pre-computing those syntheses at ingest time. The
query-side hot path is unchanged — the retriever just sees more points
in the collection, and summary nodes naturally win the dense-retrieval
score race when the query is broad.

**Typical wins:**
  - Policy / spec / contract docs where the answer is "everywhere a
    little" rather than "in one 800-token window".
  - Long-form narrative docs (meeting transcripts, design memos) where
    the high-level thread matters more than any single paragraph.
  - Year-long KBs where old docs have been forgotten but their gist
    shows up at a summary level.

## Cost

**2–5× base ingest time.** Concretely for a 1000-chunk document with
default settings (cluster min 5 → ~20 clusters at level 1, ~4 at level
2, 1 root), expect around 25 summary LLM calls. At ~2–3 s/call on
Qwen2.5-14B, that's ~1 minute on top of the base extract/embed/upsert
time.

Per-document storage in Qdrant grows by ~25–30%. Retrieval latency is
unchanged (same collection, same HNSW graph; filtered HNSW on
`chunk_level` is a keyword filter).

## When NOT to use

  - Small-doc KBs where every chunk is already self-contained
    (FAQs, release notes, single-fact snippets). The summaries add
    noise without signal.
  - Time-sensitive ingest pipelines (customer uploads a doc and wants
    it searchable immediately). RAPTOR makes upload cost longer.
  - KBs that change frequently and need instant reindex. Re-running
    RAPTOR on each edit is expensive.

## Configuration

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `RAG_RAPTOR` | `0` | Enable RAPTOR at ingest. `1` turns it on. |
| `RAG_RAPTOR_MAX_LEVELS` | `3` | Cap on tree depth above leaves. |
| `RAG_RAPTOR_CLUSTER_MIN` | `5` | Minimum nodes at a level to trigger further clustering. |
| `RAG_RAPTOR_CONCURRENCY` | `4` | Max parallel LLM summarize calls. |

All four are read via the `flags` overlay, so `rag_config.raptor` on a
KB can override them per-upload without touching process env.

### Per-KB opt-in (recommended)

Store `{"RAG_RAPTOR": "1"}` in the KB's `rag_config` column. The ingest
path overlays the per-KB config before calling `ingest_bytes`, so
exactly the KBs that want RAPTOR pay the cost.

## Operational notes

### Fail-open

Every failure mode drops back to flat chunking:

  - `sklearn` not importable → single-cluster degenerate clustering
    → no tree expansion → flat ingest continues.
  - Chat endpoint unreachable → `_summarize_cluster` returns `None`
    → clusters without summaries are dropped → if ALL clusters fail,
    no tree nodes at that level; recursion stops gracefully.
  - `build_tree` crashes outright → caught at the ingest boundary,
    fall back to flat chunking, no data loss.
  - Summary-embedding call fails mid-tree → already-built summaries
    are kept (matchable on text), recursion stops.

### Retrieval transparency

The retriever, reranker, MMR, context-expand, and hybrid-RRF stacks
are **unchanged**. They see RAPTOR nodes as ordinary Qdrant points
with an extra `chunk_level` payload field. This is intentional:
RAPTOR is an ingest-time trick; retrieval stays dumb.

If you want to force leaf-only retrieval (e.g. for debugging or to A/B
flat vs. tree), add `{"chunk_level": 0}` to the query filter — `chunk_level`
is in the payload allowlist but is NOT indexed by default, so such a
filter is a full scan (fine for small query volumes, avoid as a default).

### Provenance

Summary nodes carry `source_chunk_ids: list[int]` in their payload —
the ORIGINAL leaf chunk indices the summary ultimately covers. This
enables an optional retrieval-time "expand to leaves" pass (not
implemented in P3.4 — a follow-up could fetch leaves by
`doc_id + chunk_index ∈ source_chunk_ids` when a summary wins the
top-k, giving the chat model both the synthesis and the supporting
evidence).

## Known limitations

  - GMM clustering on 1024-d vectors is slower than clustering on a
    UMAP-reduced 10-d projection (Sarthi et al. used UMAP+GMM).
    `umap-learn` pulls in numba + pillow and bloats the install, so
    we stick with plain GMM on 1024-d. Doc-level ingest is still
    interactive latency (seconds not minutes on 100–500 chunks).
  - No tree rebuild on partial doc edit. A re-upload replaces the
    whole tree (leaves are stamped with deterministic UUID5 ids
    derived from `doc_id:chunk_index`, so re-upsert overwrites cleanly).
  - Intermediate summaries do NOT carry a sparse BM25 vector (we don't
    compute sparse for synthetic text). They participate in dense
    retrieval only. Leaves keep their sparse vectors as before.
