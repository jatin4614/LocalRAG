# Org Chat Assistant — Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a self-hosted, air-gapped, multi-user ChatGPT-like assistant with hierarchical Knowledge Bases, strict per-user isolation, and on-demand vision/STT on a single 32 GB RTX 6000 Ada — built as a hybrid thin-fork of Open WebUI.

**Architecture:** Hybrid thin-fork. `upstream/` is a git submodule pinned to the latest stable Open WebUI release; our KB schema, services, and routers live side-by-side in `ext/` and are wired in via a small, rebase-friendly patch set. Always-on services (vllm-chat 12 GB, tei 3 GB, postgres, redis, qdrant) stay hot; vllm-vision (8 GB) and whisper (4 GB) idle using **vLLM Sleep Mode** / a Torch `.cpu()` offload pattern, woken on demand by a `model-manager` sidecar that tracks idle time. Deployed via `docker compose` behind self-signed Caddy TLS on LAN.

**Tech Stack:** Open WebUI upstream (Svelte 5 + FastAPI), PostgreSQL 15, Redis 7, Qdrant, vLLM (chat + vision), HuggingFace TEI (BAAI/bge-m3), faster-whisper, Caddy 2, Docker Compose 3.8, Python ≥ 3.10 (D14 — 3.10.12 on build host), SQLAlchemy 2.0 async, pytest + pytest-asyncio + testcontainers + httpx.

---

## 0. How to read / revise this plan

Every decision below carries a **Status** (`confirmed` / `tentative` / `revisable`). If a tentative decision is overturned during execution, update the affected tasks in the relevant per-phase plan and note the change in the **Decision Log** at the bottom of this file. Nothing here is sacred — it's load-bearing scaffolding, not a contract.

**Per-phase plans** live next to this file (`docs/superpowers/plans/2026-04-16-phase-N-<name>.md`). Phase 1 is already written. Phases 2–7 get commissioned as each phase begins so the detailed steps reflect what we actually learned in the preceding phase. Avoids rot.

---

## 1. Scope check

This project covers ≥7 distinct subsystems (infra, auth+RBAC, KB admin, model manager, RAG pipeline, frontend UX, deployment). The writing-plans skill is explicit that multi-subsystem work should live in separate plans. The shape:

- **This file** = master overview + phase gates + cross-cutting concerns + decision log.
- **Per-phase plans** = bite-sized TDD tasks for one phase at a time.

Do not try to expand every phase here. Each phase plan is its own ≥15-task document.

---

## 2. Decision log

| # | Decision | Status | Revise by |
|---|----------|--------|-----------|
| D1 | Hybrid thin-fork (upstream submodule + `ext/` sibling modules + small patch set) | confirmed | Change before Phase 1 Task 2 |
| D2 | Replace Open WebUI's built-in `knowledge` tables with our `knowledge_bases`/`kb_subtags`/`kb_documents`/`kb_access` schema (migration drops upstream tables) | confirmed | Change before Phase 1 Task 12 |
| D3 | Upstream pinned to *latest stable Open WebUI release tag* (resolved at Phase 1 Task 2 execution time, captured in `UPSTREAM_VERSION` file) | confirmed | Change before Phase 1 Task 2 |
| D4 | Preflight model-weight check (Phase 1 Task 15) runs before first compose up; downloads cached to `./volumes/models/` so all later runs are offline | confirmed | Change before Phase 1 Task 15 |
| D5 | Self-signed TLS cert generated into `./volumes/certs/`, mounted into Caddy; no ACME | confirmed | Change before Phase 1 Task 6 |
| D6 | No CI — all tests run locally via `make test` / `pytest` | confirmed | Change before Phase 6 |
| D7 | Model manager = standalone Python FastAPI service in its own container; talks to vllm-vision + whisper via their HTTP sleep/wake endpoints (NO docker.sock mount) | confirmed | Change before Phase 3 |
| D8 | Test stack: pytest + pytest-asyncio + testcontainers + httpx async | confirmed | Change before Phase 1 Task 3 |
| D9 | Admin seed runs in Phase 1 via `scripts/seed_admin.py`, env-driven: `ADMIN_EMAIL`, `ADMIN_PASSWORD` (Argon2id) | confirmed | Change before Phase 1 Task 16 |
| D10 | Plans live under `docs/superpowers/plans/` (skill default) | confirmed | — |
| D11 | Phases 3 and 4 may run in parallel once Phase 1 lands (both depend on Phase 1 infra; neither depends on the other) | tentative | Revise after Phase 2 |
| D12 | Isolation test battery runs as unit tests per-phase where applicable; full cross-user sweep is Phase 6 gate | tentative | Revise after Phase 2 |
| D13 | Session-local (private) uploads: chunks land in a per-chat Qdrant namespace `chat_{chat_id}`; TTL-evicted 24h after chat's `updated_at` by a Redis-scheduled janitor (Phase 4) | tentative | Revise in Phase 4 design |
| D14 | Python requirement relaxed from 3.11 to ≥3.10 (host has 3.10.12, no sudo to install 3.11). `requires-python = ">=3.10,<3.13"`, Makefile `PYTHON ?= python3`, ruff `target-version = "py310"`. | confirmed | Bump to 3.11 if 3.10 blocks a required dep |

Revising a decision = edit this table + affected task(s) in the relevant per-phase plan + append an entry to the Decision Log section at the bottom.

---

## 3. Repo layout (hybrid thin-fork)

```
/home/vogic/LocalRAG/
├── CLAUDE.md                              (existing — project summary)
├── README.md                              (Task 1)
├── Makefile                               (Task 3)
├── pyproject.toml                         (Task 3)
├── .gitignore                             (Task 1)
├── .env.example                           (Task 5)
├── UPSTREAM_VERSION                       (Task 2 — pins upstream tag)
├── docs/
│   ├── superpowers/specs/…                (existing)
│   ├── superpowers/plans/…                (this file + per-phase)
│   └── runbook.md                         (Phase 7)
├── upstream/                              (git submodule → open-webui)
├── patches/                               (git format-patch, applied at build)
│   ├── 0001-disable-signup.patch
│   ├── 0002-drop-upstream-knowledge-tables.patch
│   ├── 0003-branding-assets.patch
│   ├── 0004-kb-chat-integration.patch
│   └── 0005-model-manager-hooks.patch
├── ext/                                   (our Python modules — imported by upstream backend)
│   ├── __init__.py
│   ├── config.py
│   ├── db/
│   │   ├── migrations/
│   │   │   └── 001_create_kb_schema.sql
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── kb.py
│   │       └── chat_ext.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── kb_service.py                  (Phase 2)
│   │   ├── rag_service.py                 (Phase 4)
│   │   └── model_manager_client.py        (Phase 3)
│   └── routers/
│       ├── __init__.py
│       ├── kb_admin.py                    (Phase 2)
│       └── kb_retrieval.py                (Phase 2)
├── model_manager/                         (Phase 3 — own container)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── idle_tracker.py
├── whisper_service/                       (Task 11 scaffold; Phase 3 completes)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
├── scripts/
│   ├── seed_admin.py                      (Task 16)
│   ├── preflight_models.py                (Task 15)
│   ├── gen_self_signed_cert.sh            (Task 6)
│   └── rebase_upstream.sh                 (Task 2)
├── compose/
│   ├── docker-compose.yml                 (Tasks 5–11)
│   ├── caddy/Caddyfile                    (Task 7)
│   └── .env.example                       (Task 5)
├── branding/                              (user supplies later — out of scope)
│   └── README.md
└── tests/
    ├── conftest.py                        (Task 3)
    ├── unit/
    │   ├── test_kb_models.py              (Task 13)
    │   ├── test_chat_kb_config.py         (Task 14)
    │   ├── test_preflight.py              (Task 15)
    │   └── test_seed_admin.py             (Task 16)
    └── integration/
        ├── test_compose_up.py             (Task 17)
        ├── test_kb_isolation.py           (Phase 6)
        └── test_rag_retrieval.py          (Phase 6)
```

---

## 4. Phase overview

| # | Phase | Deliverable | Gate (must pass) | Detailed plan | Est. tasks |
|---|-------|-------------|------------------|---------------|-----|
| 1 | Foundation | Repo skeleton, upstream submodule, compose stack up, KB schema migrated, admin seeded, preflight model check passes | `make up && make smoke` green; admin can log in to upstream UI | `2026-04-16-phase-1-foundation.md` **(written)** | 18 |
| 2 | KB management + RBAC | KB CRUD API, subtag CRUD, `kb_access` grants, KB selection at chat start | Admin creates KB, grants group, user in group selects KB; user outside group gets 403 | commissioned before Phase 2 | ~20 |
| 3 | Model manager (smart loading) | Sidecar service; vllm-vision / whisper wake on demand, sleep after idle | Image/audio message triggers wake ≤3 s; `/api/models/status` reflects state; idle 5 min → sleep | commissioned before Phase 3 | ~14 |
| 4 | RAG pipeline | Upload → extract → chunk → embed → upsert; parallel KB retrieval; cross-KB rerank; token budget; private-doc session namespace | KB doc ingests to `done`; chat with KB selected retrieves relevant chunk; token budget respected | commissioned before Phase 4 | ~25 |
| 5 | Frontend | KB selector component (multi-select hierarchical), chat-start flow, private-doc upload UI, streaming with RAG context | Playwright smoke: full flow from login → KB select → chat → answer cites KB | commissioned before Phase 5 | ~18 |
| 6 | Testing & isolation | Isolation battery, RBAC battery, RAG relevance, model-loading, E2E multi-user | All isolation tests pass with zero cross-user bleed; RBAC 403 on unauthorized KB | commissioned before Phase 6 | ~16 |
| 7 | Deployment runbook | `docs/runbook.md`, health checks, first-run bootstrap doc, troubleshooting, Phase-2 k8s overlay skeleton | Non-technical admin deploys from a fresh host using runbook alone in <60 min | commissioned before Phase 7 | ~10 |

**Phase dependencies:**
- 1 → 2 → 5 → 6 → 7 (strict)
- 1 → 3 (required by Phase 4's image/audio path)
- 1 → 4 (independent of 3 for text-only ingest; 3 is required for audio-doc ingest)
- 11 (D11): 3 and 4 may run in parallel after 1; confirm after Phase 2.

---

## 5. Cross-phase invariants

### 5.1 Isolation invariant (checked in every phase touching data)

> Every `kb_documents` row, every Qdrant point, every uploaded file is tagged with exactly one of: `owner_user_id` (private docs), `chat_id` (session-local docs), or `kb_id + subtag_id` (shared KB docs).
> Retrieval always filters by the current session's allowed set: `{owner_user_id == me}` ∪ `{chat_id == current}` ∪ `{kb_id ∈ my_accessible_kb_ids}`.
> Enforced at three layers: SQL (CHECK constraints + FKs), API (middleware reads `kb_access`), vector (Qdrant payload filter + post-check assertion).

Any PR that touches retrieval, upload, or chat state **must** include an isolation test in `tests/integration/test_kb_isolation.py` and run it green.

### 5.2 Token budget invariant

No RAG prompt is assembled without running the budget check in `ext/services/rag_service.py::budget_prompt`. Over-budget chunks are truncated (tiered rerank scores preserved); never silently dropped without a log line.

### 5.3 Observability invariant

Every service emits structured JSON logs (one event per line) to stdout. Prometheus metrics exposed on `/metrics` for: `rag_retrieval_latency_seconds`, `rag_rerank_latency_seconds`, `rag_embedding_failures_total`, `model_manager_wake_latency_seconds`, `model_manager_state` (gauge 0/1 per model).

### 5.4 Upstream rebase discipline

Patches under `patches/` MUST be minimal, well-named, and applied via `scripts/rebase_upstream.sh`. No edit ever goes directly into `upstream/`. If a patch grows unwieldy, split it — rebasing a 1000-line patch after an upstream bump is a liability.

---

## 6. Model manager — research note (Decision D7)

Web-searched April 2026:

- **vLLM ≥ 0.7.0 ships Sleep Mode** (docs: `docs.vllm.ai/en/latest/features/sleep_mode/`). Launch `vllm-vision` with `--enable-sleep-mode`; control via `POST /sleep` and `POST /wake_up`. Releases ~90% of GPU memory while keeping the server process running — no container restart, no model re-download, wake latency ≈ 1–3 s.
- **faster-whisper** has no sleep API. Our `whisper_service/app.py` wraps it with the same contract: `POST /sleep` calls `model.to("cpu")` and frees CUDA cache; `POST /wake_up` reloads to GPU.
- The `model-manager` sidecar is therefore a **pure HTTP coordinator** — no Docker socket, no privileged mount. Air-gap-safe, minimal attack surface.

If vLLM Sleep Mode misbehaves on this GPU (rare — known issues with some quantized vision models), the fallback is docker-py container stop/start; a task in Phase 3's plan captures that escape hatch.

Sources:
- https://docs.vllm.ai/en/latest/features/sleep_mode/
- https://github.com/vllm-project/vllm/issues/5491
- https://github.com/vllm-project/vllm/issues/15287

---

## 7. Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-16-org-chat-assistant-master-plan.md`. Phase 1 detail plan saved to `docs/superpowers/plans/2026-04-16-phase-1-foundation.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task in the Phase 1 plan, review between tasks, fast iteration.
2. **Inline Execution** — We execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Phases 2–7 detailed plans are commissioned just-in-time. Each per-phase plan repeats this handoff choice at its end.

---

## Decision Log (append entries here as decisions change)

*Example format:*
- `2026-04-16` — **D11 confirmed:** Phase 3 and Phase 4 ran in parallel successfully; no shared-state conflicts. Parallel execution now default for future phases with similar independence.

- `2026-04-16` — **Plan bug corrected during P1.T1/T2:** the original Task 1 `.gitignore` listed `upstream/`, but `git submodule add` refuses to add a submodule at a gitignored path. Implementer added a fixup commit during Task 2 to drop the line; Phase 1 plan and Task 1 test were patched to match. Lesson: submodule paths are tracked as gitlinks, not ignored.
- `2026-04-16` — **D14 added during P1.T3:** Python 3.11 not available on build host (only 3.10.12 present, no sudo). Relaxed `requires-python` to `>=3.10,<3.13`, Makefile default Python to `python3`, ruff target to `py310`. All deps (FastAPI, SQLAlchemy 2, Pydantic 2, testcontainers, asyncpg) support 3.10 — zero functional impact.
