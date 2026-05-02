# Bug-fix Campaign — 2026-05-02

Master plan and live tracker for the senior-engineer code review remediation.
Originating review: 11 sections, 85 findings. Triaged 64 KEEP / 7 DEFER / 8 DROP.

**Source of truth for status.** Each fix is committed with `review §X.Y` in the
commit message; this doc tracks PR numbers, rollback tags, and per-fix decisions.

---

## Status legend

- `[ ]` not started
- `[~]` in progress
- `[x]` done (commit SHA in cell)
- `[-]` deferred (trigger noted)
- `[/]` dropped (justification noted)

---

## Decision log

| ID | Question | Answer | Source / date |
|---|---|---|---|
| Q1 | Scanned PDFs in corpus? | **No (meaningful)**. 23/260 docs <5 chunks but sample shows polaris-injection security probes + 1 short DOCX. §1.1 OCR fix downgrades from CRITICAL to KEEP-low. | psql query 2026-05-02 |
| Q2 | Multilingual corpus? | **Sometimes**. KEEP §2.5 (pysbd) + §10.9 (Tesseract languages). | user 2026-05-02 |
| Q3 | PR strategy? | PR-per-agent for Wave 2 (8 PRs); PR-per-step for Wave 3 (6 PRs). | user 2026-05-02 |
| Q4 | Backup destination? | Local disk. | user 2026-05-02 |
| Q5 | Wave-3 downtime window? | Full downtime acceptable while work is in progress. | user 2026-05-02 |

---

## Rollback registry

| Tag | Date | pg_dump | Qdrant snapshots | Image tags | Notes |
|---|---|---|---|---|---|
| `rollback-pre-wave-1` | 2026-05-02 (`de52ca5`) | n/a (Wave 1 = code-only, no schema) | n/a | snap latest as `:pre-wave-1` after first deploy | Tag set after committing Phase 6.X completions (`9084694`, `204f66f`, `de52ca5`) |
| `rollback-pre-wave-2` | _to set after Wave 1a tracker commit_ | n/a (Wave 2 = code-only) | n/a | snap as `:pre-wave-2` | Wave 1a merged into main; Agent G inherits Wave 1b USER directive scope |
| `rollback-pre-wave-3a` | _pending_ | **REQUIRED** | **REQUIRED** | snap as `:pre-wave-3a` | Snapshot infra must exist BEFORE 3a |
| `rollback-pre-wave-3b` | _pending_ | REQUIRED | REQUIRED | `:pre-wave-3b` | DB migrations |
| `rollback-pre-wave-3c` | _pending_ | REQUIRED | REQUIRED | `:pre-wave-3c` | Qdrant schema |
| `rollback-pre-wave-3d` | _pending_ | REQUIRED | not strictly needed | `:pre-wave-3d` | Intent classifier unification |
| `rollback-pre-wave-3e` | _pending_ | REQUIRED | not strictly needed | `:pre-wave-3e` | `_run_pipeline` registry refactor (flag-gated) |
| `rollback-pre-wave-3f` | _pending_ | not needed | not strictly needed | `:pre-wave-3f` | `reembed_all.py` checkpoint |

**Pre-wave-3 protocol** (codified later as `make pre-wave-snapshot N=3a`):
```
pg_dump -Fc --no-owner orgchat > backups/pre-wave-N-$(date +%F).dump
for col in $(curl -sf localhost:6333/collections | jq -r '.result.collections[].name'); do
    curl -X POST localhost:6333/collections/$col/snapshots
done
cp -a volumes/uploads backups/pre-wave-N-uploads-$(date +%F)
cp -a /var/lib/docker/volumes/orgchat_ingest_blobs/_data backups/pre-wave-N-blobs-$(date +%F)
tar -czf backups/pre-wave-N-config-$(date +%F).tgz compose/.env compose/caddy/
# Verify restore on scratch DB BEFORE proceeding (drill).
```

---

## Wave 1 — Security & first-boot (direct execution, ~1-2 hours)

Pre-flight tag: `rollback-pre-wave-1`. Code-only changes; revert via `git revert <sha>`.

| # | Item | Review § | Status | Commit |
|---|---|---|---|---|
| 1a.1 | Cert filename mismatch (orgchat.crt vs kairos.crt) | §11.1 | `[x]` | `cdb2801` |
| 1a.2 | Bind qdrant:6333 / open-webui:6100 / vllm-qu:8101 to 127.0.0.1 + Qdrant API key | §4.1, §11.2 | `[x]` | `0dd71f6` |
| 1a.3 | Reject default `change-me-*` secrets in bootstrap.sh | §9 + §11 | `[x]` | `14f4bc5` |
| 1a.4 | `.dockerignore` additions (.claude, .env*, node_modules, *.log, dist, build) | §10.3 | `[x]` | `0ee861f` |
| 1a.5 | Fix `Makefile:84` BASELINE default to existing file | §9.4 | `[x]` | `a46e470` |
| 1a.6 | Update `.env.example`: CHAT_MODEL → gemma-4 | §11.13 | `[x]` | `8e8af7e` |
| 1a.7 | Archive `compose/*.pre-gemma4` to `compose/archive/` | §11.14 | `[x]` | `e3c19cd` |
| 1a.8 | TEI `--max-input-length 8192` (compose) — moved up from §3.1 | §3.1 | `[x]` | `9f42974` |
| 1a.9 | `internal: true` on orgchat-net | §11.3 | `[-]` | Deferred to Wave 1c (validation cycle); risk: TEI/vllm boot probes may need outbound during cold-start even with HF_HUB_OFFLINE |
| 1b.1 | USER 1000:1000 in 7 Dockerfiles (build-test each) | §10.1 | `[-]` | **Folded into Wave 2 Agent G** — Agent G already touches all Dockerfiles; combining avoids two rounds of container rebuilds |
| 1b.2 | Shared volume UID mapping for ingest_blobs | §11.10 | `[-]` | Folded into Agent G with 1b.1 |

---

## Wave 2 — Code-only fixes (8 parallel agents, PR-per-agent)

Each agent runs in `worktree` isolation. PRs land independently. Pre-flight tag: `rollback-pre-wave-2`.

| Agent | Scope | KEEP items | Conditional | PR | Status |
|---|---|---|---|---|---|
| **A** Ingest non-schema | ingest.py, extractor.py, ingest_worker.py, upload.py | §1.1, 1.3, 1.6, 1.7, 1.10, 1.12, 1.13 | §1.1 (now low-pri per Q1) | _pending_ | `[ ]` |
| **B** Chunking | chunker.py, chunker_structured.py, kb_config.py | §2.1, 2.2, 2.4, 2.6, 2.7, 2.8 | §2.5 pysbd (KEEP per Q2) | _pending_ | `[ ]` |
| **C** Embedding | embedder.py, sparse_embedder.py, hyde.py, retry_policy.py, scripts/reembed_all.py | §3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10 | — | _pending_ | `[ ]` |
| **D** Retrieval quality | retriever.py, reranker.py, cross_encoder_reranker.py, mmr.py, rerank_cache.py, budget.py, query_understanding.py, orchestrator hygiene §7.4-7.6 | §5.1*, 5.2*, 5.3, 5.4, 5.7, 5.9, 5.10, 5.12, 5.13, 5.14, 5.15*, 5.16; §7.4, 7.5, 7.6 | (5.1, 5.2, 5.15 ship default-off) | _pending_ | `[ ]` |
| **E** LLM + injection hardening | spotlight.py, doc_summarizer.py, contextualizer.py, llm_recorder.py, system_prompt_analyst.txt, patches/000X | §6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10*, 6.11*, 6.12 | (6.10, 6.11 ship default-off) | _pending_ | `[ ]` |
| **F** Observability | prometheus.yml, alerts*.yml, alertmanager.yml, metrics.py, scheduled_eval.py, Loki/Promtail | §8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9*, 8.11, 8.12 | (8.9 ship default-off) | _pending_ | `[ ]` |
| **G** Infra hardening | docker-compose.yml, all Dockerfile.*, Caddyfile, .pre-commit-config.yaml | §10.2, 10.4, 10.5, 10.6, 10.7, 10.8, 11.4, 11.5, 11.9, 11.10, 11.11, 11.12, 11.15 | §10.9 (KEEP per Q2) | _pending_ | `[ ]` |
| **H** CI scaffolding | .github/workflows/ci.yml, conftest.py, pyproject.toml, Makefile | §9.1, 9.2, 9.3, 9.5, 9.6, 9.7, 9.10 | — | _pending_ | `[ ]` |

Merge order: H → G → F → A → B → C → D → E (CI + infra first, retrieval/LLM last so eval gate validates final state).

---

## Wave 3 — Schema + refactors (6 sequential PRs)

**3a is the gate.** No 3b/3c/3e step proceeds until 3a + restore drill complete.

| Step | Scope | KEEP items | PR | Status | Pre-flight |
|---|---|---|---|---|---|
| **3a** Foundational backups | snapshot_task.py, backup_postgres.sh, backup_qdrant.sh, beat schedule, restore drill | §4.2, 11.6, 11.7 | _pending_ | `[ ]` | rollback-pre-wave-3a |
| **3b** DB migrations | 013_blob_sha_unique.sql, 014_status_ladder_check.sql, 009_skip_reservation.sql + DOWN scripts | §1.4, 1.5, 4.6 | _pending_ | `[ ]` | rollback-pre-wave-3b (incl. Q1 dedup data fix) |
| **3c** Qdrant schema | vector_store.py, qdrant_schema.py, reconcile_qdrant_schema.py | §4.3, 4.4, 4.7 | _pending_ | `[ ]` | rollback-pre-wave-3c |
| **3d** Intent classifier unification (flag `RAG_INTENT_UNIFIED`) | chat_rag_bridge.py | §5.5 | _pending_ | `[ ]` | rollback-pre-wave-3d |
| **3e** `_run_pipeline` registry refactor (flag `RAG_PIPELINE_V2`) | chat_rag_bridge.py, ext/services/pipeline/ | §7.1, 7.2, 7.3, 7.7 | _pending_ | `[ ]` | rollback-pre-wave-3e |
| **3f** reembed_all.py checkpoint | scripts/reembed_all.py | §9.9 | _pending_ | `[ ]` | rollback-pre-wave-3f |

---

## Wave 4 — Triggered items (deferred)

| ID | Item | Trigger | Status |
|---|---|---|---|
| W4.1 | §1.8 contextualizer determinism | Flipping `contextualize=true` on any production KB | `[-]` |
| W4.2 | §1.11 content-sha UUID5 | Next planned re-embed cycle | `[-]` |
| W4.3 | §2.3 chunker dedup | Same as W4.2 (forces re-ingest) | `[-]` |
| W4.4 | §4.5 text_filter payload index | Flipping `RAG_MULTI_ENTITY_DECOMPOSE=1` | `[-]` |
| W4.5 | §5.6 multi-query bucket starvation | Same as W4.4 | `[-]` |
| W4.6 | §5.11 HyDE A/B baseline | Next quarterly eval cycle | `[-]` |
| W4.7 | §1.2 HTML extractor | First user request | `[-]` |
| W4.8 | §1.9 charset detection | First non-UTF8 doc complaint | `[-]` |

---

## Dropped (with justification)

| ID | Item | Why dropped |
|---|---|---|
| §2.9 | Recursive chunker | Feature-add, not bug fix; no demand observed |
| §4.8 | Per-collection HNSW override | Premature optimization at this scale |
| §4.9 | `payload_m` | Same |
| §5.8 | Query rewriter before RBAC | `RAG_DISABLE_REWRITE=1` default; no actual cost today |
| §6.1 | Chat-model fallback | Single-host, single-GPU; no fallback target exists |
| §8.13 | Logger `extra={}` migration | Cosmetic; can be done piecemeal as files are touched |
| §9.8 | Blue/green deploy | Single-host; not actionable today |
| §11.8 | celery-beat single-instance | Only one beat process; not a real risk |

---

## Feature-flag rollback handles (Wave 2 + 3)

Every behavior change ships behind a flag, default OFF for first deploy. Soak 7 days, eval gate passes, then flip ON. Flag stays for 30 days for instant rollback.

| Fix | Flag | Initial default |
|---|---|---|
| §5.1 rerank min score | `RAG_RERANK_MIN_SCORE` | unset (off) |
| §5.2 budget includes prompt | `RAG_BUDGET_INCLUDES_PROMPT` | `0` |
| §5.15 total pipeline timeout | `RAG_TOTAL_BUDGET_SEC` | `30` (generous) |
| §6.10 inline citations | `RAG_ENFORCE_CITATIONS` | `0` |
| §8.9 query/chunk debug log | `RAG_LOG_QUERY_TEXT` | `0` |
| §3.5 TEI circuit breaker | `RAG_CB_TEI_ENABLED` | `0` |
| §6.12 LLM circuit breaker | `RAG_CB_LLM_ENABLED` | `0` |
| Wave 3d intent unify | `RAG_INTENT_UNIFIED` | `0` |
| Wave 3e pipeline v2 | `RAG_PIPELINE_V2` | `0` |

---

## Change log

(Updated as commits land. Newest first.)

- 2026-05-02 `0dd71f6` — Wave 1a.2: bind 3 ports to loopback + QDRANT_API_KEY plumbing.
- 2026-05-02 `14f4bc5` — Wave 1a.3: reject change-me-* secrets in bootstrap; sketch QDRANT_API_KEY.
- 2026-05-02 `0ee861f` — Wave 1a.4: .dockerignore additions.
- 2026-05-02 `9f42974` — Wave 1a.8: TEI --max-input-length 8192 (highest-impact single fix).
- 2026-05-02 `e3c19cd` — Wave 1a.7: archive docker-compose.yml.pre-gemma4.
- 2026-05-02 `8e8af7e` — Wave 1a.6: env.example CHAT_MODEL → gemma-4.
- 2026-05-02 `a46e470` — Wave 1a.5: Makefile baseline default → baseline.json.
- 2026-05-02 `cdb2801` — Wave 1a.1: Caddyfile cert filename → orgchat.{crt,key}.
- 2026-05-02 `de52ca5` — campaign kickoff. Tracking doc created. Baseline tagged `rollback-pre-wave-1`.
- 2026-05-02 `204f66f` — Phase 6.X UI: chat skeleton "thinking" label.
- 2026-05-02 `9084694` — Phase 6.X backend: multi-entity decompose + MMR/expand caps + per-KB image_captions.
