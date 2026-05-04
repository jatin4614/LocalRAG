# Multi-entity comparative queries — chunker, budgets, summaries, subtopic axis

**Status:** Design — pending implementation
**Author:** Operator + Claude (interactive brainstorm 2026-05-04, empirical validation against KB 2)
**Scope:** `ext/services/{ingest,chat_rag_bridge,doc_summarizer,multi_query,entity_extractor,query_understanding,kb_config}.py`, `compose/.env`, `compose/docker-compose.yml`, `compose/.env.example`, optional new SQL migration. Five items grouped into three rollout phases. Estimated 2-3 weeks engineering.
**Successor to:** `2026-05-03-retrieval-quality-fix-design.md`. The 2026-05-03 spec landed entity_text_filter case-handling, soft-boost, synonyms, multi-entity coverage counter — all of which are now in production. This spec addresses the residual gap surfaced after Phase 1+2+3 of that spec shipped.
**Operator decision (2026-05-04):** fresh-start re-ingest authorized — KB 2 will be wiped and re-ingested with the new chunker config; no alias-cutover or rollback rails needed.

---

## 1. Background

### 1.1 Failure mode that triggered this work

Operator query against KB 2 / subtag 11 (the four 2026 monthly reports), 2026-05-04:

> *"Give out major updates from the report of apr 2026, for the following:
> 1. 75 INF bde 2. 5 PoK bde 3. 32 Inf Bde 4. 80 Inf Bde
> I want answers under fwg heads: 1. visits of senior military officials..."*

LLM answer surfaced:
- 75 Inf Bde — 7 visit facts (good)
- 5 PoK Bde — **0** facts ("no records")
- 32 Inf Bde — **0** facts ("no records")
- 80 Inf Bde — 2 facts

This is *after* the 2026-05-03 retrieval-quality fixes shipped. Direct Qdrant scrolls confirm Apr 26.docx contains:

| Brigade | Apr 26 chunks | with visit-keyword | with visit ∧ Apr-proximity |
|---|---:|---:|---:|
| 75 Inf Bde | 43 | 21 | 15 |
| **5 PoK Bde** | **103** | **49** | **28** |
| 32 Inf Bde | 19 | 10 | 6 |
| 80 Inf Bde | 34 | 17 | 8 |

So the data is plentiful and the entity_text_filter / synonym path is no longer the bottleneck. The gap is downstream.

### 1.2 Empirical investigation (2026-05-04 brainstorm)

Five live experiments against the deployed pipeline (see also the live results in the brainstorm transcript that produced this spec):

| Run | Chunker config | n chunks | 75 Inf | 5 PoK | 32 Inf | 80 Inf | Total facts | Answer chars |
|---|---|---:|:---:|:---:|:---:|:---:|:---:|---:|
| Baseline (production) | structured + coalesce | 270 | 5 | 0 | 0 | 2 | 7 | ~3700 |
| E6 — rerank off | structured + coalesce | 270 | 5 | 1 | 1 | 2 | 9 | ~4000 |
| Window 200/30 + coalesce | window 200t | 882 | 5 | 4 | 3 | 3 | 15 | 5492 |
| **Historic 800/100, no coalesce** | window 800t, no coalesce | **2796** | 5 | 3 | 5 | 3 | 16 | 5149 |
| **Window 200/30 + items 2+3** | window 200t + floor=8 + budget=35K | 882 | **7** | **5** | 3 | **4** | **19** | **6431** |

Key findings, all empirically validated:

1. **The `3e283c7 fix(ingest): coalesce small docx blocks before chunking` commit (May 3 2026) is the load-bearing regression.** It took Apr 26.docx from **2796 → 270 chunks** — exactly 10× as advertised. For broad questions over the whole doc this helped recall; for narrow multi-entity questions it is the primary cause of "no records" responses on low-frequency brigades. The cross-encoder is forced to score on chunks containing 4-6 events, so a 5 PoK Bde event buried in a chunk that's mostly about 75 Inf Bde drowns in other-event noise.

2. **Per-entity rerank-stage floor of 3 is too low** for the user's actual goal of "elaborate per-brigade enumeration". Multi-entity decompose returns 200 hits with floor=10, but the rerank-stage `_apply_entity_quota` in `ext/services/chat_rag_bridge.py:929-1020` caps each entity at floor=3 by default (`RAG_MULTI_ENTITY_RERANK_FLOOR=3`). With 4 brigades that's 12 chunks at the entity-floor — far short of the 8-10 facts/brigade the operator wants surfaced.

3. **Budget tokens 22000 is too tight** for multi-entity + subtopic queries. With 4 brigades + 5-6 subheadings × ~10 facts each, the LLM needs 6-8K of generation room and 30-40K of context. Today's `RAG_BUDGET_TOKENS=22000` truncates context aggressively, evicting later-ranked entity quota chunks.

4. **Doc summaries are 75-Inf-only.** Direct read of `level=doc` Qdrant points for all four 2026 monthly reports: every summary literally says "activities of the 75 Inf Bde, Sarupa" with **zero mention** of 5 PoK / 32 Inf / 80 Inf despite each appearing in dozens of underlying chunks. The auto-summariser (`ext/services/doc_summarizer.py`) prompts the chat-LLM with "Summarize the following document in 3 sentences" and the LLM picks the dominant entity. When the query routes to `intent=global` and the doc-summary index is queried, the resulting context is structurally biased toward 75 Inf Bde regardless of what the user asked.

5. **No subtopic axis.** The brigade query has 1+ subheadings ("visits of senior military officials"). Multi-entity decompose builds focus-suffixed sub-queries `<query>(focus on 5 PoK Bde)` — entity is the only axis. Subtopic stays buried in the dense embedding, and the cross-encoder can't separate visit-content from logistics-content for the same brigade.

The combination of (1)+(2)+(3) when applied together (window 200/30 chunker + floor=8 + budget=35K) lifted the brigade query from **7 → 19 total facts** (+170%) and **3700 → 6431 char** answer (+74%) with **every brigade having ≥3 facts** — see row 5 of the table above. (4) and (5) are predicted to compound on top of that, addressing the structural bias and subtopic-collapse class for any future comparative query.

### 1.3 Why a fresh-start re-ingest is the right call

The operator authorized wiping KB 2 (Postgres `kb_documents` rows + Qdrant `kb_2_v*` collections) and re-uploading the four monthly reports plus any other docs. This simplifies the design substantially:

- No alias-cutover or dual-collection live-migration plumbing.
- No `pipeline_version` reconciliation.
- No backfill script for doc summaries on existing rows.
- Existing test subtags (12 / 13 created during this spec's brainstorm) get cleaned up as part of the wipe.

The fresh start also means the new chunker config and the new doc-summary prompt apply uniformly — no mixed-vintage chunks where some are 87-char paragraph fragments and others are 1565-char structured blocks.

---

## 2. Goals + non-goals

### Goals

1. **Make the brigade-query class produce balanced, elaborate answers** — at least 8-10 grounded facts per named entity for queries that name 4+ entities under 1+ subheadings.
2. **Eliminate the structural bias of doc summaries** so `intent=global` queries don't silently demote all-but-one entity.
3. **Give multi-entity + multi-subtopic queries their own retrieval shape** — entity decompose times subtopic decompose, with quotas honoured at both axes.
4. **Lift the answer length** to match the elaborateness the operator expects — 6K-10K char answers for high-fanout queries — while keeping tighter answers for simple lookup queries.
5. **Land all changes behind feature flags / per-KB config** so other corpora (KB 3 / KB 8) see no behaviour change unless explicitly opted in.

### Non-goals

- LLM-as-reranker (deferred — would compound but is a separate engineering investment; see §10).
- Self-RAG / corrective loop (same — out of scope).
- Knowledge-graph augmentation (same).
- Embedder fine-tuning on the corpus (same).
- Query rewrite / HyDE changes — those paths are off by default and orthogonal.
- Touching the upstream Open WebUI patches; all changes live under `ext/`.

---

## 3. Architecture overview

```
Phase 1 (~3-5 days)              Phase 2 (~3-5 days)             Phase 3 (~5-7 days)
─────────────────                ──────────────────              ──────────────────
1. window chunker default    →   4. entity-coverage         →   5. subtopic-aware
   + small chunk_tokens          per-doc summary                 multi-axis decompose
2. rerank-floor bump
3. budget bump
                                                                 (rolling re-ingest of
                                                                  any earlier docs as
                                                                  prompts change)
```

### Surface-area changes by file

| File | Phase | Change |
|---|---|---|
| `compose/.env.example` | 1 | document `RAG_MULTI_ENTITY_RERANK_FLOOR=8`, `RAG_BUDGET_TOKENS=28000` (raised from 22000), `RAG_INGEST_BLOCK_MIN_TOKENS=200` (default unchanged but documented). Per-KB `rerank_top_k` set via PATCH not env. |
| `compose/docker-compose.yml` | 1 | new explicit env mapping for `RAG_MULTI_ENTITY_RERANK_FLOOR` (open-webui + celery-worker); `RAG_BUDGET_TOKENS` default raised |
| `ext/services/kb_config.py` | 1 | new per-KB key `multi_entity_rerank_floor` (per-KB override of the env, joins `VALID_INT_KEYS`); chunker defaults clarified in `resolve_chunk_params` docstring |
| `ext/services/chat_rag_bridge.py` | 1, 5 | read floor via `flags.get` so per-KB override wins; in P3, plumb subtopic axis into `_multi_entity_retrieve` and per-entity-quota helper |
| `ext/services/doc_summarizer.py` | 2 | new entity-coverage prompt; structured 5-7 sentence summary with explicit "Brigades named in this document: …" line; bumped chars budget |
| `ext/services/entity_extractor.py` | 3 | new helper `extract_subtopics(query)` — lightweight extraction of user-supplied subheadings (markdown / numbered / "under fwg heads" etc.) |
| `ext/services/multi_query.py` | 3 | `build_sub_queries` extended to N×M (entities × subtopics); `merge_with_quota` extended to two-axis quotas |
| `ext/services/query_understanding.py` | 3 | extend QU LLM prompt to also extract subtopics into `HybridClassification.subtopics: list[str]` |
| `ext/db/migrations/018_kb_subtopics.sql` | 3 (optional) | adds optional per-KB subtopic-stoplist column for corpora that want subtopic detection disabled |
| `scripts/wipe_and_reingest.py` | 1 | one-shot operator script — backs up Postgres, deletes Qdrant collections, deletes `kb_documents` rows, prompts operator to re-upload |

No `Dockerfile.celery` change. Embedder / TEI / fastembed paths untouched.

### Compatibility guarantees

- **Phase 1**: per-KB `chunking_strategy` / `chunk_tokens` / `multi_entity_rerank_floor` are explicit per-KB knobs. Other KBs (KB 3, KB 8) keep their current behaviour unless opted in. Env defaults stay backwards-compatible.
- **Phase 2**: new doc-summary prompt only fires for newly-ingested docs. Existing summaries on other KBs untouched. The fresh-start re-ingest of KB 2 picks up the new prompt automatically.
- **Phase 3**: subtopic axis is gated behind `RAG_SUBTOPIC_DECOMPOSE` (default `0`) and per-KB `subtopic_decompose: true`. KBs without the flag see byte-identical behaviour.
- The rerank-floor bump is the one process-global default change. Operator note: shipping `RAG_MULTI_ENTITY_RERANK_FLOOR=8` widens the post-rerank pool for *all* multi-entity queries on *all* KBs. Single-entity queries are unaffected (decompose doesn't fire). Worst case for non-multi-entity workloads is unchanged latency.

---

## 4. Phase 1 — chunker + budgets (~3-5 days)

### 4.1 Item 1 — Window chunker default + chunk_tokens=200, overlap_tokens=30

#### 4.1.1 What changes

- Default chunker for KB 2 becomes `chunking_strategy="window"` with `chunk_tokens=200`, `overlap_tokens=30`.
- Coalesce stays on (default `RAG_INGEST_BLOCK_MIN_TOKENS=200`) — the chunker's window splitter handles the rest. Disabling coalesce entirely (the historic 2796-chunk path) was empirically marginal on quality but 3× the index size; not worth the storage/latency cost.
- Per-KB `rag_config.chunking_strategy` and `chunk_tokens` already exist (`ext/services/kb_config.py:VALID_BOOL_KEYS` / `VALID_INT_KEYS`); no schema change.

#### 4.1.2 Why these specific numbers

| Setting | n chunks (Apr 26) | p50 chunk size | Total facts | Verdict |
|---|---:|---:|:---:|---|
| structured + coalesce (current) | 270 | 1565 chars (~390 t) | 7 | **Regression** |
| window 200/30 + coalesce | 882 | 487 chars (~120 t) | 15 | **Sweet spot** |
| window 800/100, no coalesce (historic) | 2796 | 116 chars (~30 t) | 16 | Marginal gain, 3× cost |

200 tokens / 30 overlap is the empirically validated sweet spot. p50 ~120 tokens means each chunk carries one focused event-and-its-assessment. The 30-token overlap preserves enough context across chunk boundaries that the cross-encoder still picks up event continuations.

#### 4.1.3 Per-KB application

```python
# After fresh-start wipe, before re-uploading docs:
PATCH /api/kb/2/config
{"chunking_strategy": "window", "chunk_tokens": 200, "overlap_tokens": 30}
```

Other KBs (KB 3, KB 8) keep their current settings — the per-KB stamp wins over env defaults. If/when an operator wants the same shape elsewhere, same PATCH call.

#### 4.1.4 What about the structured chunker?

The structured chunker (`ext/services/chunker_structured.py`) preserves table/code blocks as atomic units. It is the right default for KBs whose docs have substantial tabular or code content (security policy KBs, contract KBs). KB 2 monthly reports are mostly prose with few tables, so structured atomicity helps less than per-event granularity. Leave structured chunker as a per-KB option, not default.

### 4.2 Item 2 — `RAG_MULTI_ENTITY_RERANK_FLOOR` env bump 3 → 8

#### 4.2.1 What changes

- New env knob `RAG_MULTI_ENTITY_RERANK_FLOOR=8` in `compose/.env.example` and explicit mapping under `open-webui:` and `celery-worker:` env blocks.
- Existing read site `int(flags.get("RAG_MULTI_ENTITY_RERANK_FLOOR") or "3")` (in `ext/services/chat_rag_bridge.py:~2024`) unchanged — `flags.get` already supports per-KB override.
- New per-KB key `multi_entity_rerank_floor` added to `VALID_INT_KEYS` in `ext/services/kb_config.py` with bounds `[1, 50]`. KBs that want to override the env can stamp the per-KB value.

#### 4.2.2 Why 8

Per-entity floor multiplied by entity count = floor for the post-rerank pool. With 4 brigades:
- floor=3: pool floor 12 → after MMR (typically takes top 12-15) and context_expand and budget → user sees 1-3 facts per brigade.
- floor=8: pool floor 32 → after MMR and expand → user sees 4-7 facts per brigade.
- floor=12: pool floor 48 → diminishing returns; cross-encoder noise starts to dominate at the floor's tail.

Empirically validated at 8 in the brainstorm runs. Bounds upper of 50 is the same as `multi_entity_min_per_entity`'s — anything above that crowds out single-entity recall on the same KB.

### 4.3 Item 3 — Widen the rerank pool + budget headroom

#### 4.3.1 What changes (corrected after empirical re-check)

The 2026-05-04 brainstorm initially framed this as a budget bump, but the actual empirical trace shows budget was **never** the bottleneck. With 40 post-rerank chunks at ~120 tokens each + sibling expansion = ~5-6K tokens of retrieval context — well under either 22K or 35K budget. The lever that actually grew the answer was item 2 (floor 3 → 8 → final pool 12 → 40 chunks).

So item 3 has two parts:

- **3a. Per-KB `rerank_top_k` lift** — raise from 50 to 120 in KB 2's `rag_config`. This is the actual cap on post-rerank pool depth. With `_final_k = max(rerank_top_k, len(entities) * floor)`, `rerank_top_k=120` lets a 4-entity query keep up to 120 chunks past the cross-encoder cut (was 50). Budget then chooses how many fit; this gives MMR room to diversify and context-expand room to add siblings.
- **3b. Budget headroom bump 22000 → 28000** — for the 4-entity * 5-subtopic * sibling-expanded case, 120 chunks × 120 tokens + 60 sibling chunks × 120 tokens = ~22K tokens. 28K leaves 6K headroom for prompt-prefix + spotlight tags + intent preamble + datetime preamble. Conservative — leaves 4K of LLM-ctx room for response generation.

#### 4.3.2 LLM context-window arithmetic

Gemma-4-31B-it-AWQ has 32K ctx (CLAUDE.md §3). Budget for retrieval context + system prompt (~3-5K) + user message (~1K) + chat history (~2K typical) + response output must fit:

```
22000 (current budget) + 5000 (sys) + 1000 (user) + 2000 (hist) + max_response = 32000
                                                                ⇒ max_response ≤ 2000
```

That's tight. Long answers were getting truncated mid-sentence on the brigade query class.

After 3a + 3b:

```
28000 (new budget) + 5000 + 1000 + 2000 + max_response = 32000
                                         ⇒ max_response ≤ 4000  ← comfortable for elaborate answers
```

If the operator later swaps in a longer-context model (e.g. a 128K-ctx variant), revisit and lift budget to 50-60K.

#### 4.3.3 What about `RAG_GLOBAL_BUDGET_TOKENS`?

Global intent's budget is `RAG_GLOBAL_BUDGET_TOKENS=22000` (compose default). After Phase 2 (entity-coverage doc summaries), more doc summaries reach the LLM per query, so global budget could plausibly want a similar bump. But we won't know until item 4 ships and we measure. Leave at 22000 in Phase 1; revisit at the end of Phase 2 once the new summaries are flowing.

#### 4.3.4 Compose env mapping

`compose/.env.example` adds `RAG_RERANK_TOP_K_DEFAULT=120` (env-side default — per-KB stamp wins). Or stamp directly on KB 2 via `PATCH /api/kb/2/config`. Recommend the per-KB stamp — `rerank_top_k` per-KB already in `VALID_INT_KEYS` with bounds `[1, 1000]`, so this is config-only, no code change.

`RAG_BUDGET_TOKENS` already in compose env block (line ~599). Bump default 22000 → 28000.

### 4.4 Phase 1 deliverables

- `kb_config.py` adds `multi_entity_rerank_floor` to `VALID_INT_KEYS` with bounds `[1, 50]`.
- `chat_rag_bridge.py` no code change at the floor read site (already uses `flags.get`); add a one-line debug log when floor is non-default so the operator can see it firing.
- `compose/.env.example` and `compose/docker-compose.yml`:
  - new env mapping `RAG_MULTI_ENTITY_RERANK_FLOOR` (default 8) on `open-webui` + `celery-worker`.
  - `RAG_BUDGET_TOKENS` default raised 22000 → 28000.
  - `RAG_INGEST_BLOCK_MIN_TOKENS` mapping (already added during 2026-05-04 brainstorm) — keep at default 200.
- KB 2 `rag_config` PATCH:
  - `chunking_strategy: "window"`, `chunk_tokens: 200`, `overlap_tokens: 30`
  - `rerank_top_k: 120` (raised from 50)
  - `multi_entity_rerank_floor: 8` (or rely on env)
- Operator script `scripts/wipe_and_reingest.py` (see §7).
- Per-KB chat-completions smoke test: brigade query against KB 2 returns ≥4 facts/brigade across all 4 brigades, ≥6K total chars.
- Eval-gate: `make eval-gate` no >5pp nDCG@10 regression vs the committed baseline. Updated baseline if shape of the gold-set queries changes.

---

## 5. Phase 2 — entity-coverage doc summaries (~3-5 days)

### 5.1 Item 4 — Re-summarize per-doc with explicit entity coverage

#### 5.1.1 The current prompt

`ext/services/doc_summarizer.py:_SUMMARY_PROMPT`:

```
Summarize the following document in 3 sentences. Include the document
name, top-line content, and dates/entities/identifiers a reader would
need to know. Write as a single paragraph of plain prose — no bullets,
no preamble.
```

The chat-LLM picks the dominant story arc. For Apr 26.docx that's 75 Inf Bde because it leads the document and gets the most paragraph count. Other brigades vanish from the summary even though they're substantively present.

#### 5.1.2 The new prompt

```
Summarize this document for retrieval. Output two sections, no preamble:

ENTITIES: A comma-separated list of every named formation, brigade,
battalion, regiment, or unit that appears at least twice in the document.
Use the canonical name as it first appears.

SUMMARY: A 5-7 sentence paragraph covering: document name, reporting
period, every named entity from the ENTITIES list (one clause per entity
naming what activities they were involved in), and any cross-cutting
themes (training, construction, intel, etc.) that appear across multiple
entities. Do NOT favour the most-mentioned entity over others — every
ENTITIES-list member must be named in the SUMMARY.
```

Output shape:

```
ENTITIES: 75 Inf Bde, 5 PoK Bde, 32 Inf Bde, 80 Inf Bde, 12 Inf Div, FCNA, 651 Mjd Bn, 77 Mtn Fd Arty Regt, 47 BALUCH, 6 NLI, 41 FF, 25 PoK Bn

SUMMARY: Document "Apr 26.docx" provides the fourth monthly update for
the GOC for April 2026. 75 Inf Bde activity centred on operational
preparedness reviews by Brig Aamir Fareed and visits to 77 Mtn Fd Arty
Regt and 651 Mjd Bn. 5 PoK Bde reporting tracked coordination meetings
at brigade HQ Muzaffarabad including a 05 Apr 26 conference attended by
COs of 69 FF and 77 PUNJAB. 32 Inf Bde activity in the Kel sector
focused on UAS / drone reconnaissance missions by 6 NLI Shardi. 80 Inf
Bde activity included a brigade-commander visit to 59 BALUCH and
projected GOC FCNA visits to forward posts. Cross-cutting themes
include training cadres, troop rotations between forward companies, and
intelligence visits by 310 Corps Int Bn / 611 FS Sec.
```

#### 5.1.3 Mechanism

- `_SUMMARY_PROMPT` updated.
- `_MAX_BODY_CHARS` raised 16000 → 32000 (more body context = better entity coverage).
- Output parsed into a structured `{entities: [...], summary: "..."}` dict at ingest time.
- Qdrant `level=doc` point payload gets a new field `entities: list[str]` alongside existing `text`.
- `text` field becomes "ENTITIES: …\n\nSUMMARY: …" so the dense retriever sees both signals.
- Summary embeddings include the entities list — dense retrieval for "5 PoK Bde April 2026" matches doc summaries that name 5 PoK Bde even when 75 Inf Bde dominates the body.

#### 5.1.4 Why this kills the global-intent bias

Today: `intent=global` → query the doc-summary index → dense retrieval ranks by semantic similarity → 75-Inf-only summaries always rank top → drilldown fetches K chunks per top-summary → LLM sees 75-Inf chunks only.

After: doc summaries name every entity → dense retrieval can rank a 5 PoK Bde-mentioning summary high for a 5 PoK query → drilldown surfaces 5 PoK chunks. Same path, fixed signal.

#### 5.1.5 Validation

Same brigade query pre/post item 4. Expectation: when item 4 lands without items 1-3, the brigade query (still `intent=specific` per current classifier) is unaffected; but a query like *"compare 5 PoK Bde and 80 Inf Bde operational status across 2026"* (which routes to `intent=global`) goes from 0 → multiple facts per brigade.

#### 5.1.6 Cost

LLM call per doc, ~32K context + 500 token output. With 16 docs in KB 2, that's 16 calls × ~3-5 seconds each = ~60-80 seconds total ingest time. Acceptable for the fresh-start re-ingest; not a per-query cost.

### 5.2 Phase 2 deliverables

- `doc_summarizer.py` new prompt, raised `_MAX_BODY_CHARS`, new return shape `{entities, summary}`.
- `ingest.py` doc-summary upsert path consumes the structured return; populates `level=doc` Qdrant payload `entities` field.
- `qdrant_schema.py` payload index for `entities` (text-tokenized like the post-2026-05-03 `text` index) so future `/api/rag/retrieve` calls can `MatchText` on entity name.
- Postgres `kb_documents.doc_summary` mirrors the SUMMARY section (not ENTITIES — keep DB column reasonable).
- Phase 2 smoke: comparative query *"compare 5 PoK Bde and 80 Inf Bde operational status in April 2026"* surfaces facts from both brigades (today: 5 PoK = 0).

---

## 6. Phase 3 — subtopic-aware multi-axis decomposition (~5-7 days)

### 6.1 Item 5 — Subtopic axis as a first-class signal

#### 6.1.1 What changes

When the user query names N entities AND M subheadings (e.g. *"For 75 Inf, 5 PoK, 32 Inf, 80 Inf give updates under: 1. visits 2. operations 3. construction 4. intel"*), the bridge fans out N×M sub-queries instead of N. Each sub-query carries both axes in its focus suffix:

```
"Give all updates for April 2026 (focus on 5 PoK Bde — visits of senior officers)"
"Give all updates for April 2026 (focus on 5 PoK Bde — operations and exercises)"
"Give all updates for April 2026 (focus on 5 PoK Bde — construction)"
"Give all updates for April 2026 (focus on 5 PoK Bde — intelligence activities)"
... × 4 brigades = 16 sub-queries
```

Per-sub-query retrieval at `_per_kb=10` total = 160 candidate chunks. Per-(entity,subtopic) quota at `floor=2` = 32 chunks at the floor; merge-with-quota tops up to `total=200`. The cross-encoder then operates on a pool that's substantively richer in the (entity, subtopic) cells the user actually asked about.

#### 6.1.2 Subtopic detection

Three input shapes the operator query uses in practice:

```
under fwg heads:
1. visits of senior military officials
2. operations
3. construction
```

```
Give updates for the following:
- visits
- operations
- construction
```

```
visits of senior officers, operational activities, and construction in April 2026
```

`extract_subtopics(query)` lives in `ext/services/entity_extractor.py` (the regex side) and `query_understanding.py` (QU-LLM side). Three extractors in priority order:

1. **Numbered-list under heading**: regex matches `(?:^|under .*?)\s*\d+\.\s*([^\n]+)` after a "heads:"/"headings:"/"sections:" trigger.
2. **Bulleted list under heading**: regex matches `\s*[-*]\s*([^\n]+)` after the same trigger.
3. **QU-LLM extraction**: when `RAG_QU_SUBTOPIC_EXTRACT=1` AND regex returns `<2`, the QU-LLM is asked to enumerate subtopics. Soft-falls to regex on QU error.

Output is deduped, lowercased, capped at 8 (anything more is too much fan-out).

#### 6.1.3 Two-axis decompose gate

`should_decompose` in `ext/services/multi_query.py` extended:

```python
def should_decompose(*, entities, subtopics, flag_on, intent):
    if not flag_on: return ("none", False)
    if intent == "metadata": return ("none", False)
    n_e = len(entities); n_s = len(subtopics)
    if n_e >= 2 and n_s >= 2: return ("both", True)
    if n_e >= 2: return ("entity", True)
    if n_s >= 2: return ("subtopic", True)
    return ("none", False)
```

Three modes the bridge has to support:
- `entity` — current behaviour (unchanged).
- `subtopic` — only subtopics are the axis (e.g. *"give me visits, operations, construction across the corpus"* — single doc, multiple subtopics). Build M sub-queries with subtopic-only focus suffix.
- `both` — N×M as described above.

#### 6.1.4 Quota in two axes

`merge_with_quota` extended to `merge_with_two_axis_quota`:

```python
def merge_with_two_axis_quota(
    *,
    per_cell_hits: dict[tuple[str, str], list],   # (entity, subtopic) -> hits
    k_min_per_entity: int,
    k_min_per_subtopic: int,
    k_total: int,
) -> list:
```

Algorithm:
1. **Cell-quota pass**: for each (entity, subtopic) cell, take top `min(k_min_per_entity, k_min_per_subtopic) // 2` (rounded up) hits. Default 2 per cell; with 4×4 = 16 cells × 2 = 32 quota chunks.
2. **Entity floor recovery pass**: for each entity, ensure at least `k_min_per_entity` total chunks in pool — if a brigade has cells with no hits, pull more from cells that did have hits.
3. **Subtopic floor recovery pass**: same for subtopics.
4. **Top-up pass**: fill remaining slots up to `k_total` by score across deduped non-quota leftovers.
5. **Final sort**: by score desc, capped at `k_total`.

Pure function — same hit-shape contract as the existing `merge_with_quota` (any object with `.id` and `.score`).

#### 6.1.5 Cross-encoder rerank with two-axis quota

The post-rerank quota helper `_apply_entity_quota` in `chat_rag_bridge.py:929-1020` becomes `_apply_two_axis_quota` when both axes are present. Same per-axis floor logic. Per-cell substring matching for attribution: a chunk's text must contain *both* an entity-name variant AND a subtopic-keyword variant for it to count toward that cell. Synonyms (the 2026-05-03 spec's per-KB `synonyms` table) feed entity attribution.

For subtopic attribution, a small per-KB / per-corpus subtopic-keywords table:

```json
{
  "visits": ["visit", "vis", "inspection", "tour", "arrival", "arr at", "mov of CO", "attended UI mtg", "conference", "mtg/conf"],
  "operations": ["operation", "exercise", "ex", "drill", "deployment", "patrol", "mission", "QC msn", "drone msn"],
  "construction": ["construction", "constr", "bunker", "trench", "track", "OP", "fortification"],
  "intelligence": ["intel", "ISI", "ISPR", "FS Sec", "security check", "clearance", "Int Bn", "informer"]
}
```

This is the place where corpus-vocabulary mismatch (the deeper diagnosed issue from §1.2) gets addressed without resorting to LLM-as-reranker. Operator-curated, per-KB.

#### 6.1.6 Storage

New per-KB JSONB key `subtopic_keywords` in `knowledge_bases.rag_config` (no schema migration — existing JSONB column accepts new keys). Validated as `dict[str, list[str]]` in `kb_config.validate_config` with the same shape rules as `synonyms`.

Operator CLI `scripts/edit_kb_subtopics.py` — same shape as `edit_kb_synonyms.py` from the 2026-05-03 work. `--add`, `--remove`, `--list`, `--load FILE`. Initial seed for KB 2 included as a separate operator step.

#### 6.1.7 Knobs

- `RAG_SUBTOPIC_DECOMPOSE` env (default `0`).
- Per-KB `subtopic_decompose: true` in `rag_config` (`VALID_BOOL_KEYS`).
- `RAG_QU_SUBTOPIC_EXTRACT` env (default `0`) — enable QU-LLM subtopic extraction (vs. regex-only).
- Per-KB `subtopic_keywords` JSONB.

#### 6.1.8 Default-flip plan

Ship as `RAG_SUBTOPIC_DECOMPOSE=0` (off) initially. Operator manually verifies on the brigade query and ≥3 other multi-entity multi-subtopic queries. After comparing answer quality with item 5 on/off, flip to `1`. Per-KB stamp is the right place for KB 2 to opt in early.

### 6.2 Phase 3 deliverables

- `entity_extractor.py` adds `extract_subtopics(query)` with three priority extractors.
- `query_understanding.py` extends prompt and `HybridClassification` to include `subtopics: list[str]`.
- `multi_query.py` adds `should_decompose` two-axis return, `build_sub_queries_two_axis`, `merge_with_two_axis_quota`.
- `chat_rag_bridge.py` `_multi_entity_retrieve` becomes mode-aware (`entity` / `subtopic` / `both`); `_apply_entity_quota` becomes `_apply_two_axis_quota` when mode is `both`; signature accepts subtopics + subtopic_keywords + per-cell floors.
- `kb_config.py` adds `subtopic_decompose` (BOOL) and `subtopic_keywords` (DICT) keys; `validate_config` enforces shapes.
- `scripts/edit_kb_subtopics.py` CLI.
- Optional `ext/db/migrations/018_kb_subtopic_columns.sql` if we want explicit JSONB columns vs. nesting in `rag_config` (decision at implementation time).
- Per-KB seed for KB 2: subtopic_keywords table populated from §6.1.5 plus 4-6 more topics surfaced by operator review.
- Phase 3 smoke: brigade query with 4 explicit subheadings produces ≥6 facts per (brigade, subheading) cell where the corpus has ≥3 candidates.

---

## 7. Fresh-start re-ingest playbook

The operator authorized wiping KB 2 and re-uploading all docs after Phase 1 ships. This section is the runbook for that.

### 7.1 Prerequisites

- Phase 1 spec items merged (§4) — chunker default, env knobs, per-KB key for floor.
- Phase 2 spec items merged (§5) — new doc-summary prompt.
- (Optional) Phase 3 items merged (§6) — won't change ingest, only retrieval; can land later.

### 7.2 Operator script

`scripts/wipe_and_reingest.py` — one-shot, confirms-on-stdin before destructive ops:

```bash
.venv/bin/python scripts/wipe_and_reingest.py --kb 2 \
    --confirm-wipe \
    --backup-dir /tmp/kb2_backup_2026-05-XX
```

Steps the script runs:

1. Backs up Postgres `kb_documents` rows + `knowledge_bases.rag_config` for the KB to `--backup-dir/postgres.json`.
2. Calls Qdrant snapshot API for `kb_2_v2` (and any aliased collections); writes snapshot path to backup dir.
3. Prompts for explicit confirmation: `WIPE KB 2 (Y/n)`.
4. Deletes Qdrant collections for the KB (`kb_2`, `kb_2_v2`, `kb_2_v3`, `kb_2_v4`, `kb_2_rebuild`, etc.).
5. `DELETE FROM kb_documents WHERE kb_id = 2;` (Postgres FK cascade handles `kb_access` and chunk-references).
6. Re-applies the per-KB `rag_config` from the backup (so chunker/floor settings persist across the wipe).
7. Recreates the empty Qdrant collection `kb_2` with the current schema (see `ext/db/qdrant_schema.py`).
8. Logs the new pipeline_version that future ingests will stamp.

### 7.3 Re-upload sequence

Operator uploads docs via the existing admin upload UI / API. Each upload kicks off async ingest through Celery — see `compose/docker-compose.yml` celery-worker block. The new chunker config (window 200/30) and the new doc-summary prompt fire automatically.

Verification per doc:
- `docker compose -p orgchat exec -T postgres psql -U orgchat -d orgchat -c "SELECT id, filename, ingest_status, chunk_count FROM kb_documents WHERE kb_id=2 AND filename = 'Apr 26.docx';"`
- Expect `ingest_status=done`, `chunk_count` in the 800-1200 range for a typical monthly report.
- `curl /api/kb/2/documents/{id}/summary` returns the new ENTITIES + SUMMARY structured shape.

### 7.4 Post-ingest validation

Brigade query smoke:

```bash
JWT=$(...); CID=...
curl -X POST -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  -d @- /api/chat/completions <<'JSON'
{
  "model": "orgchat-chat", "stream": false,
  "chat_id": "...",
  "messages": [{"role": "user", "content": "Give out major updates from the report of apr 2026 for the following: 1. 75 INF bde 2. 5 PoK bde 3. 32 Inf Bde 4. 80 Inf Bde under: 1. visits of senior military officials"}],
  "rag_kb_config": [{"kb_id": 2, "subtag_ids": [11]}]
}
JSON
```

Expected (Phase 1+2 together): ≥5 facts per brigade, ≥6K char answer.
Expected (Phase 1+2+3 together): ≥6 facts per (brigade, subheading) cell where data exists.

---

## 8. Compatibility / fallback

- **Phase 1**: per-KB `chunking_strategy` / `chunk_tokens` / `multi_entity_rerank_floor` are explicit knobs. Other KBs unaffected.
- **Phase 2**: new doc-summary prompt fires only on freshly-ingested docs. Other KBs that don't re-ingest keep their old summaries.
- **Phase 3**: gated by `RAG_SUBTOPIC_DECOMPOSE=0` (default off). Per-KB opt-in. Fallthrough on missing subtopic keywords table = entity-only behaviour. QU-LLM subtopic extraction soft-fails to regex.
- **Rollback**: per phase, env knob reset + `git revert`. Re-ingest needed only if doc summaries got polluted by a buggy prompt — operator runs `scripts/wipe_and_reingest.py` again with the previous prompt.

The whole change is reversible piece-by-piece. The fresh-start re-ingest in §7 is the most destructive step but also the cleanest — no migration plumbing needed.

---

## 9. Verification gates

### Per-phase eval-gate

Each phase has `make eval-gate` re-run after deploy. Acceptable thresholds:

- **Phase 1**: nDCG@10 must not drop more than 5pp vs the committed baseline. Recall@30 expected to *rise* on the multi-entity gold-set queries; if it drops by >2pp anywhere else, investigate.
- **Phase 2**: doc-summary tier eval — golden set of comparative queries (e.g. *"compare X and Y operational status"*) must show entity coverage in returned doc-summary points. Add 8-10 such queries to the gold set.
- **Phase 3**: multi-axis gold set — queries with explicit subheadings must show ≥6 facts per (entity, subheading) cell where the corpus has ≥3 candidates. Add 4-6 such queries.

### Manual brigade-query smoke (per-phase)

Phase 1: brigade query → ≥4 facts/brigade across all 4. Empirically observed 19 facts total in the 2026-05-04 brainstorm; require ≥15.

Phase 2: comparative-query smoke (the §5.1.5 query) → ≥3 facts per brigade where today: 0.

Phase 3: brigade query with 4 subheadings → ≥3 (brigade, subheading) cells with ≥3 facts each, ≥0 cells with 0 facts where the corpus has ≥3 candidates.

### Latency budget

- Phase 1: no expected latency change. Brigade query observed ~50-130s in the brainstorm (LLM-bound, not retrieval-bound).
- Phase 2: ingest latency rises ~60-80s/doc per the §5.1.6 estimate. Query latency unchanged.
- Phase 3: query latency rises proportional to N×M sub-queries. Dispatched as `asyncio.gather` against the shared httpx pool, so wall-time grows ~log(N×M) given pool=32. Expected +5-15s on the brigade query at N=4, M=4. Within the latency-OK envelope the operator stated.

---

## 10. Out of scope / future work

The 2026-05-03 spec's §9 already enumerated the bigger structural items. Re-iterating with current relevance:

- **LLM-as-reranker for multi-entity intent** — replace `bge-reranker-v2-m3` with a small LLM scoring (chunk, query, entity, subtopic) tuples. Expected highest-ROI follow-up after this spec ships. Bumps quality further; ~3× rerank cost; manageable behind a per-intent flag.
- **Self-RAG / corrective-RAG** — LLM-judge that detects "missing entity coverage" in the draft answer and triggers a second focused retrieve. Useful when the post-rerank pool genuinely lacks an entity's content (vs today where pool has it but LLM doesn't surface it).
- **Listwise rerank with ILP** — joint optimization of relevance + entity-coverage + subtopic-coverage. Replaces the greedy quota in `merge_with_two_axis_quota` (§6.1.4) with a solver. Bigger payoff for queries with more axes (e.g. N×M×P with date as a third axis).
- **Cross-encoder / embedder fine-tuning on the corpus** — domain adaptation. Material quality gain on the corpus-vocabulary mismatch (the "mov of CO" / "UI mtg" terminology from §1.2). Large engineering investment.
- **Knowledge-graph augmented retrieval** — entity-relation graph extracted at ingest, joined with vector hits at query time. High payoff for highly-relational corpora (military org-chart, contract counterparty network).
- **Query rewrite for multi-entity queries** — LLM-rewrites *"give me visits for 75 Inf, 5 PoK, 32 Inf, 80 Inf"* into 4 single-entity queries before retrieval, runs them in parallel, merges. Different shape than the focus-suffix decomposition in this spec; trade-offs at implementation time.

If items 1-5 ship and the brigade-query class is still not meeting the operator's bar for elaborate answers, the next conversation should evaluate **LLM-as-reranker** first (highest ROI of the deferred items, given the rest of the pipeline will be at this sophistication).

---

## 11. Tracking + history

Predecessor: `docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md` — Phase 1+2+3 of which shipped before this spec was written.

Empirical evidence for the design comes from a 2026-05-04 brainstorm session (this conversation). Key live measurements documented in §1.2.

This spec's implementation plan will be at `docs/superpowers/plans/2026-05-04-multi-entity-elaborate-answers.md` (TBD by writing-plans skill).
