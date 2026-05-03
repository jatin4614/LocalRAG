# Retrieval-quality fix — entity_text_filter casing + soft-boost + synonyms

**Status:** Design — pending implementation
**Author:** Operator + Claude (interactive brainstorm 2026-05-03)
**Scope:** `ext/services/{vector_store,chat_rag_bridge,multi_query,kb_config}.py`,
`ext/services/metrics.py`, one new SQL migration, two operator scripts, and a
runtime alert. Six items grouped into three rollout phases. Estimated
1–2 weeks engineering.

---

## 1. Background

LocalRAG's multi-entity retrieval path (Phase 6.X "Methods 3–5" — entity
extraction, parallel sub-queries, per-entity quota — see CLAUDE.md §4) was
designed to make queries that name N entities each get coverage in the final
LLM context. Today that path **silently drops entities for which the user's
casing doesn't match the corpus's casing**.

### 1.1 Failure mode that triggered this work

Operator query (KB 2 / subtag 11, after the four 2026 monthly reports were
re-ingested with the block-coalescing fix):

> *"Give out major updates from the report of apr 2026, for the following:
> 1. 75 INF bde, 2. 5 PoK bde, 3. 32 Inf Bde, 4. 80 Inf Bde
> I want answers under fwg heads: 1. visits of senior military officials..."*

Pipeline trace observed:

```
multi-entity decompose entities=5 filter=True floor=10 total=200 -> 200 hits
context_expand cap active, expanded top 4 of 24
```

LLM answer surfaced 7 facts for 75 Inf Bde, **0 facts for 5 PoK Bde**,
4 facts for 32 Inf Bde, only 1 fact for 80 Inf Bde — despite direct
Qdrant scrolls confirming all four brigades have rich coverage in the
April 2026 shard.

### 1.2 Diagnosis (verified)

Per-entity sub-query hit counts via the deployed `vector_store.hybrid_search`
with the user's typed entity strings:

| Entity (user typing) | Hits returned by `entity_text_filter` |
|---|---:|
| `75 INF bde` | **0** |
| `5 PoK bde`  | **1** |
| `32 Inf Bde` | 12 |
| `80 Inf Bde` | 12 |

Same query with corpus-canonical casing:

| Entity | Hits |
|---|---:|
| `75 Inf Bde` | 12 |
| `5 PoK Bde`  | 12 |

**Root cause:** Qdrant's `MatchText` against the `text` payload uses default
case-sensitive whitespace tokenization, and the `text` field is **unindexed**
(verified via `GET /collections/kb_2` payload schema — only
`chat_id`/`deleted`/`doc_id`/`kb_id`/`owner_user_id`/`shard_key`/`subtag_id`
are indexed; `text` is not). Without a payload index, Qdrant scans + tokenizes
on the fly using the default tokenizer, which preserves case. So
`MatchText("75 INF bde")` produces tokens `[75, INF, bde]` while corpus chunks
contain tokens `[75, Inf, Bde, Sarupa, ...]` — zero overlap.

The per-entity rerank quota landed earlier today (commit `f416dbe`) cannot
compensate: it picks the top `floor=3` chunks per entity from the
cross-encoder output, but if the entity sub-query returned 0 chunks, the
quota has nothing to take.

### 1.3 The wider failure class this exposes

The entity-text filter is implemented as **hard exclusion** in
`vector_store._build_filter` — a chunk that doesn't textually match
the entity name (under Qdrant's tokenizer) is removed before reranking.
That's brittle to:

- **Casing variants** (the actual bug today)
- **Abbreviations**: `PoK` vs `POK` vs `Pakistan-Occupied Kashmir`
- **Paraphrases**: `75 Bde` vs `75 Infantry Brigade`
- **Tokenizer suffix attachment**: `Bde,` (with trailing comma) tokenized as
  one token can lose the match

A bug in any one of these layers wipes out the entity completely with no
graceful degradation.

### 1.4 One regression also surfaced

While running the brigade query the bridge logged:

```
ERROR KB retrieval failed: cannot access local variable '_do_decompose'
where it is not associated with a value
```

This fires for `intent=metadata` queries (`how many docs?`, `list files`).
The metadata-intent code path in `_run_pipeline` doesn't initialize
`_do_decompose`, but the per-entity quota patch at line ~2125 references it.
Result: every metadata-intent query returns 0 KB sources today. Regression
introduced by commit `f416dbe`. Bundled into this spec so the fix lands as
part of one coherent change.

---

## 2. Goals + non-goals

### Goals

1. **Make per-entity sub-queries robust to casing and common spelling
   variants** so the brigade query (and any analogous multi-entity query)
   gives every entity equal opportunity to surface in the final pool.
2. **Convert `entity_text_filter` from a hard exclusion to a soft signal**
   so a single layer's imperfection (regex extraction, casing, abbreviation,
   tokenizer quirk) does not silently zero an entity.
3. **Provide a per-KB synonym table** so domain-specific abbreviations
   (`PoK` ↔ `POK` ↔ `Pakistan-Occupied Kashmir`) become first-class.
4. **Fix the `_do_decompose` UnboundLocalError regression** on the
   metadata-intent path.
5. **Surface the failure class in observability** so the next regression
   pages within minutes instead of being noticed in the chat UI hours later.

### Non-goals

- LLM-as-reranker, self-RAG / corrective loop, agentic retrieval,
  knowledge-graph augmentation, listwise rerank with ILP, embedder
  fine-tuning. These are deferred — the diagnosis showed the gap is
  shallower than any of those.
- A formal golden eval set / `make eval-gate` extension. Operator will
  manually verify with the brigade query (see §8).
- Touching the OpenAI-compatible embedder, the global-intent path,
  metadata routing flags, or any other unrelated subsystem.

---

## 3. Architecture overview

```
Phase 1 (~1 day)               Phase 2 (~3-5 days)             Phase 3 (~1-2 days)
─────────────────              ──────────────────              ──────────────────
1. lowercase text index    →   3. soft-boost vs hard-filter →  6. multi-entity
2. suffix-strip in filter      4. per-KB synonym table            coverage counter
5. _do_decompose fix                                              + alert
                                                                  + manual-test
                                                                    recipe in spec
```

### Surface-area changes by file

| File | Phase | Change |
|---|---|---|
| `ext/services/vector_store.py` | 1, 2 | `_build_filter()` lowercases + strips suffixes + accepts variant set; new `_ensure_text_index()` helper |
| `ext/services/chat_rag_bridge.py` | 1, 2 | initialize `_do_decompose=False` upfront (Item 5); thread `RAG_ENTITY_TEXT_FILTER_MODE` and `synonyms` through to `_multi_entity_retrieve` and `_apply_entity_quota` (Item 3) |
| `ext/services/multi_query.py` | 2 | `_apply_entity_quota` accepts optional `entity_variants_map` for the boost-mode score adjustment |
| `ext/services/kb_config.py` | 2 | new key `entity_text_filter_mode` ("filter" \| "boost"); new key `synonyms` (list of equivalence classes); `expand_entity()` helper |
| `ext/services/metrics.py` | 3 | `rag_multi_entity_coverage_total{outcome,entity_count}` counter |
| `ext/db/migrations/013_kb_synonyms.sql` | 2 | adds `synonyms` JSONB column to `knowledge_bases` |
| `scripts/apply_text_index.py` | 1 | one-shot operator script — adds the lowercase text index to every KB collection |
| `scripts/edit_kb_synonyms.py` | 2 | operator CLI to seed/edit per-KB synonym table |
| `ext/routers/kb_admin.py` | 2 | `PATCH /api/kb/{kb_id}/synonyms` (admin-only) |
| `observability/prometheus/alerts-rag-quality.yml` | 3 | `MultiEntityCoverageEmpty` alert |

No `compose/.env` or `compose/docker-compose.yml` change is required for
Phase 1-2; the new env knobs (`RAG_ENTITY_TEXT_FILTER_MODE`,
`RAG_ENTITY_BOOST_ALPHA`, `RAG_ENTITY_TEXT_FILTER_STRIP_NOISE`) all default to
behaviour-preserving values.

### Compatibility guarantees

- **Phase 1**: fully backward-compatible. Lowercasing on the query side is
  idempotent for already-canonical names; suffix-strip is gated by
  `RAG_ENTITY_TEXT_FILTER_STRIP_NOISE` (default `1` but easily reverted).
- **Phase 2**: ships with `MODE=filter` as default. Boost mode is opt-in
  per-KB via `entity_text_filter_mode="boost"`. Synonym table empty by
  default — no-op until operator adds entries.
- **Phase 3**: pure additions (counter + alert + recipe). Zero impact on
  request path.
- All new knobs are per-KB. KB 3 / KB 8 see no behaviour change unless
  explicitly opted in.
- `_do_decompose` regression fix is unconditional and behaviour-restoring,
  not behaviour-changing.

---

## 4. Phase 1 — quick wins (~1 day)

### 4.1 Item 1 — Case-insensitive text matching

**Layer A: index side.** Operator script `scripts/apply_text_index.py`
walks every KB collection and creates a Qdrant payload index on the
`text` field with a lowercased word tokenizer.

```python
from qdrant_client import models as qm

client.create_payload_index(
    collection_name=name,
    field_name="text",
    field_schema=qm.TextIndexParams(
        type="text",
        tokenizer=qm.TokenizerType.WORD,
        lowercase=True,
        min_token_len=2,
        max_token_len=20,
    ),
)
```

Properties:
- Online operation in Qdrant — no downtime, no reindex needed; the new
  index covers all existing points immediately on first build.
- Idempotent — Qdrant returns 200 on `create_payload_index` when an index
  of the identical shape already exists (verify in script with a try/except
  on `UnexpectedResponse(409)`).
- Cheap — measured in minutes per 10K-point collection.

**Layer B: query side.** In `vector_store.py:_build_filter`:

```python
if text_filter and text_filter.strip():
    normalized = text_filter.strip().lower()
    must.append(qm.FieldCondition(
        key="text",
        match=qm.MatchText(text=normalized),
    ))
```

Defense in depth — works even on collections where the operator hasn't
yet run the index script (the comparison is now lowercase-vs-lowercase
even on the on-the-fly tokenizer fallback, modulo the tokenizer's own
behaviour).

### 4.2 Item 2 — Suffix strip

In `_build_filter`, strip a configurable trailing-noise list **from the
entity name BEFORE matching**. The semantic sub-query text (the
`"... (focus on X)"` string) is unaffected — it keeps the suffix because
that's signal for the dense retriever.

```python
_FILTER_SUFFIX_NOISE = {"bde", "bn", "regt", "coy", "div", "corps", "comd"}

def _strip_filter_suffix(s: str) -> str:
    """Drop trailing noise tokens; never returns empty (returns input unchanged
    if every token is a noise word)."""
    tokens = s.split()
    stripped = list(tokens)
    while stripped and stripped[-1].lower() in _FILTER_SUFFIX_NOISE:
        stripped.pop()
    return " ".join(stripped) if stripped else s
```

Knob: `RAG_ENTITY_TEXT_FILTER_STRIP_NOISE` (default `1`). Per-KB override
not exposed yet — the noise list is military-corpus specific; if a
non-military KB needs it disabled, operator sets the env var. (Per-KB
override added later if it becomes a real need.)

### 4.3 Item 5 — `_do_decompose` regression fix

In `chat_rag_bridge._run_pipeline`, the metadata-intent code path
short-circuits before the multi-entity decompose block runs, but the
per-entity quota call at line ~2125 still references `_do_decompose`,
`_entities`, and `_entity_floor`. Today that crashes with
`UnboundLocalError`.

Fix: initialize all three variables BEFORE the intent branch:

```python
# In _run_pipeline, at the top of the post-intent block:
_do_decompose: bool = False
_entities: list[str] = []
_entity_floor: int = 0
```

The non-metadata branch overwrites them as it does today. The metadata
branch leaves them at the safe defaults; the quota call then no-ops
(quota with `entities=[]` returns the input unchanged per the existing
contract).

### 4.4 Phase 1 deliverables

- `scripts/apply_text_index.py` — operator one-shot, idempotent
- `vector_store._build_filter` — lowercases entity, strips noise suffixes
- `chat_rag_bridge._run_pipeline` — initializes the three quota variables
- Unit tests:
  - `test_match_text_case_insensitive` — same chunk reachable via three
    casings of the same entity
  - `test_strip_suffix_basic` / `test_strip_suffix_keeps_when_only_noise` /
    `test_strip_suffix_recursive`
  - `test_metadata_intent_does_not_crash` — regression for `_do_decompose`
- Manual smoke after deploy: re-run the brigade query in user's casing;
  per-entity sub-query hits ≥ 5 for every brigade.

---

## 5. Phase 2 — structural change (~3–5 days)

### 5.1 Item 3 — Soft-boost vs hard-filter

#### 5.1.1 What changes

Today (`MODE=filter`): when a sub-query for entity `E` is dispatched,
`vector_store.hybrid_search` adds `MatchText(E)` to the Qdrant `must`
filter. Chunks not matching are **excluded** — they never enter the
candidate pool, never reach the cross-encoder.

New mode (`MODE=boost`): drop the Qdrant filter. Run normal hybrid
retrieval. After cross-encoder rerank, score each hit for entity-text
presence (Python pass) and use the score as a **rerank-score boost**.
Per-entity quota then operates on the boosted ranking.

#### 5.1.2 Mechanism

In `_multi_entity_retrieve`, when `MODE=boost`:

1. Skip the `text_filter` argument when calling `hybrid_search` for each
   sub-query. (Sub-query text — `"<original> (focus on E)"` — is unchanged.)
2. After the cross-encoder rerank step (which runs on the merged candidate
   pool), iterate the hits once and adjust scores in place:

   ```python
   def _entity_boost_score(text: str, variants: set[str]) -> float:
       """1.0 if chunk text contains any variant of the entity name;
       else 0.0. Lowercase comparison."""
       if not text or not variants:
           return 0.0
       text_low = text.lower()
       return 1.0 if any(v.lower() in text_low for v in variants) else 0.0

   alpha = float(flags.get("RAG_ENTITY_BOOST_ALPHA") or "0.3")
   for hit in reranked:
       chunk_text = (hit.payload or {}).get("text", "")
       boost = _entity_boost_score(chunk_text, variants_for_active_entity(hit))
       hit.score += alpha * boost
   ```

3. Per-entity quota (`_apply_entity_quota`) operates on the boosted
   `hit.score` ordering — same algorithm, different input ordering.

#### 5.1.3 Why this is structurally better than the filter

| Failure | `filter` (today) | `boost` (new) |
|---|---|---|
| Casing variant misses chunks | excluded forever | demoted but still in pool |
| Paraphrase mismatch | excluded forever | demoted but still in pool |
| Chunk mentions entity in surrounding context only | excluded forever | demoted but still in pool |
| Chunk doesn't mention entity at all | excluded — correct | demoted to bottom — also correct |
| Per-entity quota recovery if 0 matches | impossible (nothing to take) | quota uses pure rerank order; LLM gets weaker but non-empty answer |

#### 5.1.4 Knobs

- `RAG_ENTITY_TEXT_FILTER_MODE` = `filter` (initial default) | `boost`
- `RAG_ENTITY_BOOST_ALPHA` (default `0.3` — small enough that the
  cross-encoder still dominates ranking, large enough to surface intended
  entity)
- Per-KB: `entity_text_filter_mode` in `rag_config` JSONB

#### 5.1.5 Default-flip plan

Ship as `filter` initially. Operator manually verifies on the brigade
query in both modes. After comparing answer quality on ≥3 multi-entity
queries (operator judgment), flip env default to `boost`.

### 5.2 Item 4 — Per-KB synonym table

#### 5.2.1 Storage

New JSONB column `synonyms` on `knowledge_bases`. Migration
`013_kb_synonyms.sql`:

```sql
ALTER TABLE knowledge_bases
ADD COLUMN IF NOT EXISTS synonyms JSONB NOT NULL DEFAULT '[]'::jsonb;
```

Shape — equivalence classes (no canonical form, simpler):

```json
[
  ["5 PoK", "5 POK", "5 PoK Bde", "5 POK Bde", "Pakistan-Occupied Kashmir"],
  ["75 Inf", "75 INF", "75 Inf Bde", "75 Infantry Brigade"],
  ["Inf Bde", "Infantry Brigade"],
  ["PoK", "POK", "Pakistan Occupied Kashmir", "Pakistan-Occupied Kashmir"]
]
```

#### 5.2.2 Lookup helper

In `kb_config.py`:

```python
def expand_entity(entity: str, classes: list[list[str]]) -> set[str]:
    """Return entity + every equivalence-class member that contains it.
    Case-insensitive membership check."""
    out = {entity}
    e_low = entity.lower()
    for cls in classes:
        if any(v.lower() == e_low for v in cls):
            out.update(cls)
    return out
```

#### 5.2.3 Wiring

Used in two places, both via `_build_filter` reading the per-KB
`synonyms` from the request-scope flag overlay:

1. **`MODE=filter`**: replace the single `MatchText(entity)` with a
   `should` clause over the variants:

   ```python
   variants = expand_entity(text_filter, kb_synonyms)
   must.append(qm.Filter(should=[
       qm.FieldCondition(key="text", match=qm.MatchText(text=v.lower()))
       for v in variants
   ]))
   ```

   (Qdrant: a `Filter` with only `should` clauses passes any point matching
   at least one — verify against current qdrant-client docs at implementation
   time and adjust if the API requires a `min_should_match` parameter.)

2. **`MODE=boost`**: variants set passed to `_entity_boost_score` so any
   synonym match triggers the boost.

#### 5.2.4 Operator UX

- `scripts/edit_kb_synonyms.py --kb 2 --add '["5 PoK","5 POK","5 PoK Bde"]'`
  (and `--remove`, `--list`, `--load FILE`).
- `PATCH /api/kb/{kb_id}/synonyms` admin-only endpoint, body shape matches
  the JSONB column.
- Initial seed for KB 2 included as a separate operator step (see §7) so
  the brigade query works on first deploy after the upgrade.

#### 5.2.5 Risk + mitigation

- **Bad synonym mapping over-promotes irrelevant chunks** → caught by
  manual eval on the brigade query before flipping `MODE=boost` default.
- **Maintenance burden** — operator catches new abbreviations as they
  surface in real queries → eventually the table stabilizes per KB.
- **Per-KB scope** means a wrong synonym in KB 2 cannot affect KB 3.

### 5.3 Phase 2 deliverables

- Migration `013_kb_synonyms.sql` (idempotent)
- `kb_config.expand_entity()` + `kb_config.VALID_BOOL_KEYS` adds
  `entity_text_filter_mode`; `kb_config.VALID_KEYS` adds `synonyms`
- `vector_store._build_filter` uses variants set + `should` clause when
  variants > 1
- `chat_rag_bridge._multi_entity_retrieve` reads
  `RAG_ENTITY_TEXT_FILTER_MODE`, threads variants map through to
  the boost step
- `scripts/edit_kb_synonyms.py` operator CLI
- `kb_admin.py` `PATCH /api/kb/{kb_id}/synonyms`
- Unit tests:
  - `test_expand_entity_basic` / `test_expand_entity_case_insensitive` /
    `test_expand_entity_not_in_any_class_returns_self_only`
  - `test_filter_mode_uses_should_for_variants`
  - `test_boost_mode_promotes_matching_chunks_above_non_matching`
  - `test_boost_mode_does_not_exclude_non_matching`
  - `test_filter_mode_byte_identical_to_pre_change`
  - `test_alpha_zero_equals_no_boost`
  - `test_synonym_table_empty_no_op`
  - Migration test in `tests/integration/`

---

## 6. Phase 3 — observability + manual-test recipe (~1–2 days)

(Per operator decision: no formal golden-set / `make eval-gate` extension.
Manual verification only.)

### 6.1 Multi-entity coverage counter

In `ext/services/metrics.py`:

```python
rag_multi_entity_coverage_total = Counter(
    "rag_multi_entity_coverage_total",
    "Multi-entity quota outcome — full = every entity met its floor; "
    "partial = at least one entity got <floor but >0 chunks; "
    "empty = at least one entity got 0 chunks.",
    labelnames=("outcome", "entity_count"),
)
```

Bumped at the end of `_apply_entity_quota` after the final pool is built:

```python
counts_per_entity = {e: 0 for e in entities}
for hit in final_pool:
    text_low = (hit.payload.get("text") or "").lower()
    for e in entities:
        if any(v.lower() in text_low for v in expand_entity(e, kb_synonyms)):
            counts_per_entity[e] += 1
            break  # one entity attribution per chunk

n_zero    = sum(1 for c in counts_per_entity.values() if c == 0)
n_partial = sum(1 for c in counts_per_entity.values() if 0 < c < per_entity_floor)
outcome = "empty" if n_zero else ("partial" if n_partial else "full")
rag_multi_entity_coverage_total.labels(
    outcome=outcome, entity_count=str(len(entities))
).inc()
```

### 6.2 Prometheus alert

In `observability/prometheus/alerts-rag-quality.yml`:

```yaml
- alert: MultiEntityCoverageEmpty
  expr: rate(rag_multi_entity_coverage_total{outcome="empty"}[15m]) > 0.05
  for: 10m
  labels: {severity: warning, component: rag}
  annotations:
    summary: "Multi-entity queries returning 0 hits for at least one entity (>5% rate)"
    description: |
      Either entity_text_filter regressed, the synonym table is stale,
      or a new corpus added an unfamiliar abbreviation. Cross-reference
      rag_multi_entity_decompose_total for context. The failing entity
      name is NOT in the metric label (cardinality concern); inspect
      open-webui logs for "rag: multi-entity rerank quota active" lines
      to see which entity has the empty bucket.
```

### 6.3 Manual-test recipe

Documented inline in this spec for operator use after each phase ships.
Same recipe runs against the running stack via `curl`.

#### Step 0 — get a JWT

```bash
JWT=$(docker compose -p orgchat exec -T open-webui curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@orgchat.lan","password":"OrgChatAdmin2026!"}' \
  http://localhost:8080/api/v1/auths/signin \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['token'])")
```

#### Step 1 — Phase 1 verification (after item 1, 2, 5 deploy)

The brigade query in user's casing should return non-empty per-entity hits:

```bash
docker compose -p orgchat exec -T open-webui curl -s -X POST \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  -d '{"chat_id":"a767297f-63b4-4cae-b8af-34aaf1e12247",
       "query":"75 INF bde April 2026 visits",
       "selected_kb_config":[{"kb_id":2,"subtag_ids":[11]}],
       "top_k":12}' \
  http://localhost:8080/api/rag/retrieve \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('hits=',len(d['hits']))"
```

Expected: `hits=` ≥ 5 (was 0 before Phase 1).

Repeat for `5 PoK bde`, `32 Inf Bde`, `80 Inf Bde`. All must be ≥ 5.

Metadata-intent regression check:

```bash
docker compose -p orgchat exec -T open-webui curl -s -X POST \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  -d '{"chat_id":"a767297f-63b4-4cae-b8af-34aaf1e12247",
       "query":"how many documents are in the KB",
       "selected_kb_config":[{"kb_id":2,"subtag_ids":[]}],
       "top_k":3}' \
  http://localhost:8080/api/rag/retrieve \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('hits=',len(d['hits']))"
```

Expected: `hits=` ≥ 1 (catalog source). Was 0 before.

#### Step 2 — Phase 2 verification (after item 3, 4 deploy)

Seed KB 2 synonyms:

```bash
.venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --load - <<'JSON'
[
  ["5 PoK", "5 POK", "5 PoK Bde", "5 POK Bde", "Pakistan-Occupied Kashmir"],
  ["75 Inf", "75 INF", "75 Inf Bde", "75 Infantry Brigade"],
  ["32 Inf", "32 INF", "32 Inf Bde", "32 Infantry Brigade"],
  ["80 Inf", "80 INF", "80 Inf Bde", "80 Infantry Brigade"],
  ["Inf Bde", "Infantry Brigade"],
  ["Bn", "Battalion"],
  ["Regt", "Regiment"],
  ["Coy", "Company"],
  ["GOC", "General Officer Commanding"]
]
JSON
```

Re-run the brigade-query chat completion against each
`MODE=filter` and `MODE=boost`:

```bash
# MODE=filter (default)
# (no env override needed)

# MODE=boost — set per-KB via the admin endpoint
curl -s -X PATCH -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{"entity_text_filter_mode":"boost"}' \
  https://localhost/api/kb/2/config

# Re-issue chat completion with the brigade query and compare answers
```

Look for: 5 PoK Bde no longer empty; 80 Inf Bde mentions ≥ 3 chunks
worth of distinct facts.

#### Step 3 — Phase 3 verification (after counter+alert deploy)

```bash
# Issue 5 brigade queries; tail the Prometheus metric
for i in 1 2 3 4 5; do
  # ... chat completion call ...
done
sleep 30
docker compose -p orgchat exec -T open-webui curl -s \
  http://localhost:9464/metrics \
  | grep rag_multi_entity_coverage_total
```

Expected: `outcome="full"` count > `outcome="empty"` count after the
fixes are deployed; ratio inverted before deploy.

Trigger the alert deliberately to verify it routes correctly:

```bash
# Set MODE=filter and revert text-index changes on a test KB,
# issue a brigade query — should trip MultiEntityCoverageEmpty within 15m
```

### 6.4 Phase 3 deliverables

- `metrics.py` — new counter with `outcome` and `entity_count` labels
- `_apply_entity_quota` bumps the counter once per multi-entity request
- `alerts-rag-quality.yml` — new `MultiEntityCoverageEmpty` rule
- This spec's §6.3 doubles as the operator runbook (no separate `.md`)

---

## 7. Migration / rollout plan

1. **Land Phase 1** in one PR. Deploy. Run §6.3 step 1.
2. If brigade-query hits look good, **land Phase 2** in a second PR.
   Deploy. Apply migration `013_kb_synonyms.sql` via
   `scripts/apply_migrations.py`. Seed synonyms via §6.3 step 2.
3. Manually compare `MODE=filter` vs `MODE=boost` on 3 representative
   queries (operator judgment). If `boost` is at least as good, flip env
   default with a third PR (`compose/.env.example`).
4. **Land Phase 3** in a fourth PR. Deploy. Verify alert routes correctly.
5. After 14 days of stable operation, remove the `MODE=filter` code path
   if no operator has needed to revert. (Until then keep both for
   roll-back safety.)

Each phase commits in its own logical group following the existing
`fix(scope): …` convention seen in recent commits.

---

## 8. Compatibility / fallback

- **Failure of Phase 1 (Item 1) text index creation**: the script logs
  and continues — query-side lowercase still applies. No retrieval
  regression; just falls back to on-the-fly tokenization with
  lowercase-vs-lowercase comparison.
- **Failure of Phase 1 (Item 2) suffix strip**: gated by env;
  `RAG_ENTITY_TEXT_FILTER_STRIP_NOISE=0` reverts.
- **Failure of Phase 2 (Item 3) boost mode**: gated by env;
  `RAG_ENTITY_TEXT_FILTER_MODE=filter` reverts. Per-KB override beats env.
- **Failure of Phase 2 (Item 4) synonym table**: empty `synonyms` JSONB
  is no-op; `expand_entity` returns just the input. Wrong synonym entries
  can be removed via the operator script.
- **Failure of Phase 3 alert**: pure observability, no impact on request
  path.

The whole change is reversible piece-by-piece. No data migration
(synonyms is a new column with default `[]`).

---

## 9. Out of scope / future work

The following techniques came up in the brainstorm and were deferred
because the diagnosis showed the immediate gap is shallower:

- **LLM-as-reranker** — replace `bge-reranker-v2-m3` with a small LLM
  scoring chunk relevance. Higher quality, ~3× cost. Worth revisiting
  after Phase 2 ships and we measure the residual quality gap.
- **Self-RAG / corrective-RAG / loop-on-low-confidence** — single-pass
  today. Add an LLM-judge that detects "missing entity coverage" in the
  draft answer and triggers a second focused retrieve.
- **Agentic retrieval** — LLM as planner, issues N retrieves, judges,
  retries. Bigger payoff, bigger blast radius.
- **Listwise rerank with ILP/optimization for entity coverage** — today's
  per-entity quota is greedy. A listwise solver that jointly maximises
  coverage + relevance would reduce edge cases like 80 Inf having 18
  bullets while 5 PoK has 4.
- **Cross-encoder / embedder fine-tuning on the corpus** — domain
  adaptation. Material quality gain, large engineering investment.
- **KG-augmented retrieval** — entity-relation graph extracted at ingest,
  joined with vector hits at query time. Big lift, high payoff for
  highly-relational corpora.
- **Semantic chunking** — topic-shift detection vs window/structured.
  Modest gain over today's coalesce + window pipeline.

If Phase 1-3 ship and the brigade-query class is not fully solved, the
next conversation should evaluate LLM-as-reranker first (highest
return-on-investment of the deferred items, given the rest of the
pipeline is already this sophisticated).
