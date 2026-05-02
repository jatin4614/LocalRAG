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

### Wave 2 round 6 (9 inline fixes — finish small remaining items)

- 2026-05-03 `6d1fb9f` — Inline §6.12: LLM circuit breaker on doc_summarizer (RAG_CB_LLM_ENABLED=0 default).
- 2026-05-03 `dd57147` — Inline §6.6: system prompt version stamping (sha256[:12] gauge + startup log).
- 2026-05-03 `898f45a` — Inline §11.9 + §2.4: Caddy rate-limit recipe + pipe-table column-count.
- 2026-05-03 `c610b29` — Inline §11.4: mem_limit + cpus + log rotation across base stack (YAML anchor).
- 2026-05-03 `e401266` — Inline §10.7: BuildKit secret for HF_TOKEN (both Dockerfiles).
- 2026-05-03 `d174383` — Inline §10.9: Tesseract multilingual packs (fra/deu/jpn/chi-sim/hin).
- 2026-05-03 `9d73335` — Inline §10.5 + §10.6: celery requirements.txt extraction + HEALTHCHECK.

### Round 6 deferred / out-of-scope:
- **§10.1** Wave 1b (USER 1000:1000 + volume UID) — needs container build verification → Wave 1c after the rest stabilizes.
- **§11.5** Docker secrets — needs operator-side coordination (secrets/ dir creation, bootstrap.sh updates); not pushable as a single safe commit.
- **§11.11** PVC quota — already mitigated by existing daily blob_gc beat schedule (`ext/workers/blob_gc_task.py:128`); operator can shorten cron via `RAG_BLOB_GC_CRON` env if needed.
- **§2.5** pysbd multilingual splitter — needs new pip dep; defer to a focused agent.
- **§6.10 / §6.11** RAG_ENFORCE_CITATIONS / RAG_ENFORCE_ABSTENTION — LLM-output post-processing complexity; flag-gated, defer.
- **§6.4** BLOCKED upstream — chat-call record_llm_call lives in open_webui/main.py; needs middleware patch.
- **§8.7** BLOCKED operator — daily-eval staleness alert needs node-exporter `--collector.textfile.directory` flag.

### Wave 2 round 5 (4 background agents quota-capped + 5 inline; 9 fixes shipped)

- 2026-05-02 `8cfcb2f` — Inline §11.12: open-webui depends_on tei + vllm-chat healthy.
- 2026-05-02 `bc2c5d0` — Inline §11.15: rename k8s/ → k8s.future/ + test fix.
- 2026-05-02 `f124f5a` — Inline §10.8: PYTHONDONTWRITEBYTECODE in openwebui runtime stage.
- 2026-05-02 `83c66e9` — Inline §2.8: chunk_index=None for doc-summary points.
- 2026-05-02 `a2fd533` — Inline §2.6 + §2.7: if-key-in-cfg precedence + RAG_CHUNK_MAX_TOKENS env ceiling.
- 2026-05-02 `839ca09` — G-2 §10.4: openwebui multi-stage build (3 stages).
- 2026-05-02 `61f2028` — G-2 §10.2: install cu128 torch BEFORE upstream requirements.
- 2026-05-02 `e5734ce` — B §2.2: real tokenizer in chunker_structured.
- 2026-05-02 `6fa4911` — B §2.1: CLAUDE.md fix — structured chunker IS wired.
- (quota-capped) — B residuals: §2.4 pipe-table strict, §2.5 pysbd multilingual.
- (quota-capped) — G-2 residuals: §10.5 celery req extraction, §10.6 healthchecks, §10.7 BuildKit secrets, §10.9 Tesseract langs.
- (quota-capped, 0 commits) — E-2: §6.6, §6.10, §6.11, §6.12.
- (quota-capped, 0 commits) — G-3: §11.4, §11.5, §11.9, §11.11.

### Wave 2 round 4 (4 background agents, 17 fixes shipped, 1 dedup-skip)

- 2026-05-02 `3bed794` — D-2 §5.15: RAG_TOTAL_BUDGET_SEC pipeline timeout (asyncio.wait_for + degraded fallback).
- 2026-05-02 `e25a9b6` — D-2 §5.9: MMR reuses Hit.vector (with_vectors=True path; skip TEI re-embed).
- 2026-05-02 `d4a0a6f` — D-2 §5.2: RAG_BUDGET_INCLUDES_PROMPT pre-deduct (reserved_tokens kwarg).
- 2026-05-02 `4e12829` — D-2 §5.1: RAG_RERANK_MIN_SCORE post-rerank floor (default OFF).
- 2026-05-02 (skipped) — D-2 §7.6: dedup with D-1's c48a9fa, no merge.
- 2026-05-02 `399b250` — C-2 §3.8: concurrent dense/sparse/colbert via asyncio.gather + to_thread.
- 2026-05-02 `fdd803a` — C-2 §3.6: EMBED_MODEL preflight via TEI /info.
- 2026-05-02 `4882b37` — C-2 §3.5: TEI circuit breaker (RAG_CB_TEI_ENABLED=0 default).
- 2026-05-02 `de24f93` — C-2 §3.3: reembed_all routes upserts through VectorStore (shard_key fix).
- 2026-05-02 `e795855` — C-2 §3.2: Redis-backed embed cache (RAG_EMBED_CACHE_ENABLED=0 default).
- 2026-05-02 `f3aa2c5` — F-2 §8.9: RAG_LOG_QUERY_TEXT optional debug log (default OFF, PII-safe).
- 2026-05-02 `9a95a5b` — F-2 §8.8: Promtail JSON parse + Loki structured metadata (trace_id label).
- 2026-05-02 `931f4fb` — F-2 §8.6: kb→(kb_count, kb_primary) cardinality fix on retrieval_hits_total.
- 2026-05-02 `e09db51` — A §1.13: image-caption heading_path page-aware lookup.
- 2026-05-02 `6bba8de` — A §1.12: refresh kb_documents.pipeline_version after contextualize.
- 2026-05-02 `454af1e` — A §1.7: UTF-8-safe error_message truncation in worker.
- 2026-05-02 `f99fcdf` — A §1.6: SQLAlchemy engine singleton in ingest_worker (lazy).

### Wave 2 round 3 (3 background agents, smaller charters; 13 fixes + 2 BLOCKED)

- 2026-05-02 `2f5da50` — D-1 §7.5: double-checked locking on bridge singletons.
- 2026-05-02 `5b7ce80` — D-1 §5.12: RRF dedup key disambiguates doc-summary levels.
- 2026-05-02 `0a48fbd` — D-1 §5.10: MMR uses np.dot on pre-normalized vectors.
- 2026-05-02 `85bc0d3` — D-1 §5.7: pipeline_version in rerank cache key.
- 2026-05-02 `66a1ea6` — E-1 §6.9 SECURITY: html.escape filename in get_source_context (+ patches/0006).
- 2026-05-02 `40a17ab` — E-1 §6.8 SECURITY: defang `</source>` + `<source` in spotlight.
- 2026-05-02 `94d866f` — E-1 §6.7: CHAT_MODEL preflight against /v1/models.
- 2026-05-02 `e2c7f5f` — E-1 §6.5: rename SSE event spacing → sse_event_interval_seconds.
- 2026-05-02 `33ab962` — E-1 §6.3: record_llm_call into doc_summarizer + query_understanding.
- 2026-05-02 `4cb65ae` — F-1 §8.12: Jaeger + Loki retention 168h.
- 2026-05-02 `ead0f8e` — F-1 §8.7 BLOCKED: daily-eval staleness alert needs node-exporter textfile collector.
- 2026-05-02 `b98e56e` — F-1 §8.4: rag_silent_failure baseline-ramp alert.
- 2026-05-02 `5c3d850` — F-1 §8.3: rag_retrieval_empty_total counter + alert.
- 2026-05-02 `52d1614` — F-1 §8.1: Alertmanager service + alerts-celery volume mount.
- 2026-05-02 (BLOCKED upstream) — E-1 §6.4: user-facing chat-call record_llm_call wrapping; lives in open_webui/main.py, needs middleware-level patch.
- 2026-05-02 (deferred) — D-1 §7.6: OTel inject for HyDE+contextualizer headers (not committed before agent stop; reassign to batch-2 agent).

- 2026-05-02 `04cd57b` — Round 2 H §9.7+§9.10: .pre-commit-config.yaml (ruff + gitleaks + standard hygiene).
- 2026-05-02 `a6f303f` — Round 2 H §9.6: targeted mypy stub overrides + flip ignore_missing_imports=false.
- 2026-05-02 `f302327` — Round 2 H §9.5: xfail-quarantine 2 pre-existing reds + tests/README update.
- 2026-05-02 `b3951ea` — Round 2 D §7.4: configure() idempotent guard.
- 2026-05-02 `1e305b4` — Round 2 D §5.16: QU retry uses full timeout (RAG_QU_RETRY_TIMEOUT_MS override).
- 2026-05-02 `7d1a47a` — Round 2 D §5.14: partition empty-text hits (preserve at score 0).
- 2026-05-02 `bbf416b` — Round 2 D §5.4 + §5.13: silent-failure on rerank fallback + repaired fast-path divzero guard.
- 2026-05-02 `f8a6046` — Round 2 D §5.3: silent-failure on retriever per-KB exception.
- 2026-05-02 `616bf47` — Round 2 C §3.10: np.ndarray.tolist() in colbert_embed.
- 2026-05-02 `093bd3b` — Round 2 F §8.5: tokenizer-fallback alert rule.
- 2026-05-02 `ab4d3b9` — Round 2 F §8.2: load alerts-celery.yml in prometheus rule_files.
- 2026-05-02 `7f6c47f` — Round 1 → main cherry-picks tracker update + tag `rollback-post-batch1`.
- 2026-05-02 `3ae1f42` — Agent H §9.2: clarify isolation-suite count in CLAUDE.md (cherry-picked from worktree).
- 2026-05-02 `007bad8` — Agent H §9.1: GH Actions ci.yml — lint + unit matrix + isolation gate (cherry-picked).
- 2026-05-02 `1f8a51a` — Agent H §9.3: flip integration tests default-on (cherry-picked).
- 2026-05-02 `d2d440e` — Agent C §3.7: HyDE renormalize averaged vector (cherry-picked).
- 2026-05-02 `411a57c` — Agent C §3.4: TEI 429 retried as transient (cherry-picked).
- 2026-05-02 `6a35705` — Agent F §8.11: scheduled_eval gauges parse harness schema (cherry-picked).
- 2026-05-02 `878d4cc` — Wave 1a complete tracker update + tag `rollback-pre-wave-2`.
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
