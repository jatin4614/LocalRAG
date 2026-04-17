# Org Chat Assistant — KB-Based RAG Pipeline

**Project Type:** Self-hosted, air-gapped LLM chat with hierarchical knowledge bases  
**Status:** Design Complete → Implementation Phase  
**Target Users:** Organizations (20-200 users)  
**Hardware:** Single 32GB RTX 6000 Ada GPU + 125GB RAM  
**Repo Structure:** Thin fork of Open WebUI with KB extensions

---

## 1. What This Project Is

A **self-hosted, multi-user ChatGPT-like web application** for organizations that:
- Runs entirely on-premises (air-gapped, no external API calls)
- Organizes knowledge into **hierarchical Knowledge Bases (KBs)** with 2-level structure (KB → subtags)
- Enforces **strict per-user data isolation** (users cannot see each other's data)
- Supports **shared RBAC-gated knowledge collections** (admin assigns KB access to users/groups)
- Processes **documents, audio, and images** locally
- Uses **smart model loading** to fit 4 models (chat, vision, embeddings, STT) into 32GB VRAM

**Core Problem Solved:**
Traditional RAG systems retrieve from all documents indiscriminately. This system lets users **explicitly select which KBs to query per chat**, reducing retrieval scope (faster, cheaper, better results) while maintaining strict isolation.

---

## 2. Goals & Motivation

### Primary Goal
Enable organizations to self-host a ChatGPT-like assistant with:
- Local LLM inference (no cloud dependencies)
- Multi-tenancy with zero cross-user data leakage
- Fine-grained RAG over internal knowledge (wikis, docs, decisions)
- Flexible model scaling (Phase 1: 1×32GB → Phase 2: 4×48GB GPUs)

### Business Value
- **Compliance:** No data leaves the org (air-gapped)
- **Autonomy:** No vendor lock-in, full control of models/data
- **Cost:** Single GPU runs 5-10 concurrent users (cheaper than cloud APIs at scale)
- **Customization:** Can fine-tune models, control prompts, own the IP

### Technical Goals
1. ✅ **Hierarchical KBs:** Organize documents at 2 levels (KB + subtags)
2. ✅ **Per-Session KB Selection:** Users choose KBs at chat start → scoped retrieval
3. ✅ **Strict Isolation:** Row-level filtering + RBAC enforcement at 3 layers
4. ✅ **Smart Model Loading:** Always-running chat (12GB) + on-demand vision/whisper
5. ✅ **Parallel RAG:** Async retrieval across KBs with cross-KB ranking
6. ✅ **Token Budgeting:** Graceful prompt truncation (no cost surprises)
7. ✅ **Observability:** Full tracing for debugging, cost accounting

---

## 3. Architecture Overview

### Components (9 services)

| Service | Purpose | Model | VRAM | Status |
|---------|---------|-------|------|--------|
| **vllm-chat** | Chat completions | Qwen2.5-14B-AWQ | 12GB | Always-on |
| **tei** | Embeddings | BAAI/bge-m3 | 3GB | Always-on |
| **vllm-vision** | Image understanding | Qwen2-VL-7B | 8GB | On-demand |
| **whisper** | Speech-to-text | faster-whisper medium | 4GB | On-demand |
| **postgres** | Users, KBs, chats, docs | N/A | CPU | Always-on |
| **redis** | Sessions, rate-limits, jobs | N/A | CPU | Always-on |
| **qdrant** | Vector DB (RAG index) | N/A | CPU | Always-on |
| **model-manager** | Smart load/unload | N/A | CPU | Always-on |
| **caddy** | TLS termination, routing | N/A | CPU | Always-on |

**Why Smart Loading (Option B)?**
- Vision (8GB) + Whisper (4GB) = 12GB on-demand
- Always-running chat + embeddings = 15GB baseline
- Total peak: 23-27GB (safe within 32GB)
- Auto-unload after 5min inactivity (frees VRAM for chat batching)

### Data Flow

```
User Chat Message
  ↓
1. Load chat + validate RBAC (KB access check)
2. Embed query (TEI) → 1024-dim vector
3. Parallel retrieve from selected KBs (Qdrant)
4. Retrieve session-local private docs
5. Rerank cross-KB results (tiered strategy)
6. Budget tokens, truncate gracefully
7. Inject into system prompt
8. Call vllm-chat → stream response
9. Store message + audit log
```

### Database Schema

**4 new tables:**
- `knowledge_bases` — Org KBs (name, admin_id, description)
- `kb_subtags` — Organization within KB (e.g., "OFC", "Roadmap")
- `kb_documents` — Docs uploaded to KB/subtag (status tracking, soft-delete)
- `kb_access` — RBAC grants (user_id OR group_id → KB read/write)

**Modified tables:**
- `chats` — Added `selected_kb_config` JSONB (which KBs this chat uses)

**Isolation Invariant:**
- Every chunk in Qdrant tagged with `kb_id` + `subtag_id`
- Every document in `kb_documents` references exact KB/subtag
- Every retrieval filters by current user's `kb_access` grants
- Enforced at API + Qdrant + DB layers (defense in depth)

---

## 4. Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Thin fork of Open WebUI** | 90% done on day one, focus on KB features | Must track upstream changes |
| **Hierarchical 2-level KBs** | Balance flexibility (subtags) + simplicity | Can't do 3+ levels |
| **One Qdrant collection per KB** (not per subtag) | 100 collections vs. 2000 (manageability) | ~5-10% slower queries (metadata filtering) |
| **Per-session KB selection** (locked) | Users control scope, better results | Can't change KBs mid-chat (restart needed) |
| **Session-local private docs** (ephemeral) | No persistence overhead | Must re-upload per chat |
| **Smart model loading** (Option B) | Fits everything in 32GB safely | 2-3s delay on first image/audio |
| **Parallel async retrieval** | 25 queries in 100ms not 2500ms | Requires connection pooling |
| **Tiered reranking** | Optional reranker, fast path for confident queries | More code complexity |
| **Token budgeting** (strict) | Predictable costs, no surprises | Must truncate results (accept gracefully) |

---

## 5. Hardware Constraints & Solutions

**User Hardware:** 32GB RTX 6000 Ada (single machine)

**Challenge:** Chat (12GB) + Vision (8GB) + Embeddings (3GB) + Whisper (4GB) = 27GB + KV-cache = OOM risk

**Solution (Option B: Smart Loading):**
```
Baseline (15GB):
  vllm-chat: 12GB
  + tei: 3GB
  = 15GB + 17GB headroom

On-demand:
  vllm-vision: 8GB (load when image uploaded, unload after 5min idle)
  whisper: 4GB (load when audio uploaded, unload after 5min idle)

Result:
  ✓ Always-hot chat/embeddings (best UX)
  ✓ 2-3s delay on first image/audio (acceptable)
  ✓ 17GB headroom for KV-cache, batching
  ✓ Auto-cleanup prevents memory leak
```

---

## 6. Deliverables by Phase

### Phase 1: Foundation (DB + Docker + Basic Chat)
- ✅ KB schema migration (knowledge_bases, kb_subtags, kb_documents, kb_access)
- ✅ SQLAlchemy models
- ✅ Docker Compose for 32GB hardware (smart loading)
- ✅ Model manager service (on-demand loading/unload)
- **Deliverable:** Basic chat works, models load at startup

### Phase 2: KB Management (Admin API + RBAC)
- ✅ KB CRUD APIs (create, update, delete, list)
- ✅ Subtag management
- ✅ RBAC access control (assign KBs to users/groups)
- ✅ KB selection at chat start
- **Deliverable:** Admins can create KBs, users can select them

### Phase 3: Model Manager (Smart Loading)
- ✅ On-demand vision/whisper loading
- ✅ Auto-unload after inactivity
- ✅ Monitoring + status endpoints
- **Deliverable:** Vision/audio work without VRAM OOM

### Phase 4: RAG Pipeline (Upload → Retrieval)
- ✅ Document upload (KB + private per-session)
- ✅ Text extraction (PDF, DOCX, TXT, MD, XLSX)
- ✅ Semantic chunking (800 tokens, 100 overlap)
- ✅ Async embedding (TEI)
- ✅ Qdrant upsert (KB-scoped namespaces)
- ✅ Parallel KB retrieval
- ✅ Cross-KB ranking + normalization
- ✅ Token budgeting + graceful truncation
- **Deliverable:** RAG works, users get relevant answers from KBs

### Phase 5: Frontend Integration
- ✅ KB selector component (multi-select, hierarchical)
- ✅ Chat start flow (optional KB selection)
- ✅ Private doc upload (per-session)
- ✅ Streaming responses with RAG context
- **Deliverable:** Full chat UX with KB selection

### Phase 6: Testing & Isolation
- ✅ Isolation tests (user A can't see user B's KB docs)
- ✅ RBAC tests (access control enforcement)
- ✅ RAG retrieval tests (relevance, scoping)
- ✅ Model loading tests (on-demand behavior)
- ✅ E2E smoke tests (multi-user flows)
- ✅ Performance benchmarks (latency, throughput)
- **Deliverable:** All isolation tests pass, verified security

### Phase 7: Deployment & Runbook
- ✅ Docker Compose runbook (single-command startup)
- ✅ Health checks + monitoring
- ✅ First-run initialization (create admin, seed data)
- ✅ Troubleshooting guide
- **Deliverable:** Production-ready deployment docs

---

## 7. Tech Stack

| Layer | Technology | Version | Notes |
|-------|------------|---------|-------|
| **Frontend** | Svelte + TypeScript | 5.0 | Thin fork of Open WebUI |
| **Backend** | FastAPI + SQLAlchemy | 0.104 / 2.0 | Async, type-safe |
| **Database** | PostgreSQL | 15 | JSONB for KB config |
| **Cache/Queue** | Redis | 7 | Sessions, rate-limiting, ingest jobs |
| **Vector DB** | Qdrant | latest | Namespaced collections (KB-scoped) |
| **Chat Model** | vLLM | latest | Qwen2.5-14B-AWQ (12GB) |
| **Vision Model** | vLLM | latest | Qwen2-VL-7B (8GB, on-demand) |
| **Embeddings** | TEI | 0.4.0 | BAAI/bge-m3 (3GB) |
| **STT** | faster-whisper | latest | medium (4GB, on-demand) |
| **Deployment** | Docker Compose | 3.8 | Phase 1; Kubernetes for Phase 2 |

---

## 8. Key Design Patterns

### 1. Isolation Invariant
Every piece of data (chat, document, embedding, chunk) is tagged with `owner_user_id` XOR `collection_id` (exactly one).

**Enforcement (3 layers):**
- **DB layer:** CHECK constraints, foreign keys, soft deletes
- **API layer:** Middleware checks user's `kb_access` grants before retrieval
- **Vector layer:** Qdrant namespaces + payload filters + post-check

### 2. Smart Model Loading
Load on-demand, unload on inactivity (5-min timer).

**Implementation:**
- `ModelManager` service watches Docker daemon
- On image/audio message, async load model (user sees spinner)
- Background job unloads idle models → frees VRAM

### 3. RAG Pipeline (8-Stage)
1. Upload → 2. Chunk → 3. Embed → 4. Upsert → 5. RBAC check → 6. Retrieve (parallel) → 7. Rerank (tiered) → 8. Inject + infer

**Optimizations:**
- Parallel async retrieval (25 queries in 100ms, not 2500ms)
- Tiered reranking (fast path for confident queries)
- Token budgeting (strict limits, graceful truncation)

### 4. Observability
Full tracing for debugging + cost accounting.

**Metrics:**
- `rag_retrieval_latency_seconds` (by KB, by status)
- `rag_rerank_latency_seconds`
- `rag_embedding_failures_total`
- Structured JSON logs (event, duration, tokens, cost)

---

## 9. Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| **Namespace explosion** (100 KBs × 20 subtags = 2000 collections) | Hybrid: 1 Qdrant collection per KB + metadata filtering (~5% slower, vastly simpler ops) |
| **Multi-query fan-out latency** | Parallel async batch search (25 queries in 100ms) |
| **RBAC check overhead** | Cache `allowed_kb_ids` at session start (90% fewer DB hits) |
| **Cross-KB ranking weakness** | Global reranking + score normalization per KB |
| **Reranker bottleneck** | Tiered strategy: optional reranker (skip if confident) |
| **Token limit overflow** | Strict budgeting + graceful truncation |
| **Session cleanup leak** | TTL policy + background garbage collector |
| **Model VRAM contention** | Smart loading (on-demand) + auto-unload (5-min idle) |
| **No observability** | Prometheus metrics + structured JSON logging |
| **Upstream divergence** | Thin fork strategy (small patches, easy rebases) |

---

## 10. Testing Strategy

**Critical isolation tests (must-pass):**
- User A uploads doc → User B queries → zero hits from A's doc
- User A in group X → can access KB assigned to group X → cannot access KB assigned to group Y
- Session-local private docs not visible in other chats

**RBAC tests:**
- Admin creates KB, assigns to "engineering" group
- User in "engineering" can select + retrieve from KB
- User NOT in "engineering" gets 403 on KB selection

**RAG tests:**
- Chunk retrieval from correct KB/subtag
- Cross-KB ranking works (best chunks bubble up)
- Token budget respected (no overflow)

**E2E tests:**
- Multi-user chat flow (concurrent users, no interference)
- Image/audio trigger model loading (on-demand works)
- Chat with KB selection + private docs (both sources merged)

---

## 11. File Structure

```
O:\OpenClaw\
├── docs/
│   ├── superpowers/specs/
│   │   ├── 2026-04-12-org-chat-assistant-design.md     (original spec)
│   │   ├── 2026-04-16-kb-rag-pipeline-workflow.md       (detailed workflow)
│   │   └── 2026-04-16-kb-rag-pipeline-implementation.md (implementation plan)
│   └── runbook.md                                        (Phase 7 deliverable)
├── backend/
│   ├── migrations/
│   │   └── 001_create_kb_schema.sql                     (Phase 1)
│   ├── models/
│   │   ├── kb.py                                         (Phase 1)
│   │   └── chat.py                                       (Phase 1)
│   ├── services/
│   │   ├── model_manager.py                              (Phase 3)
│   │   ├── rag_service.py                                (Phase 4)
│   │   ├── kb_service.py                                 (Phase 2)
│   │   └── chat_service.py                               (Phase 2)
│   ├── routers/
│   │   ├── kb_admin.py                                   (Phase 2)
│   │   ├── kb_retrieval.py                               (Phase 2)
│   │   └── upload.py                                     (Phase 4)
│   ├── app.py                                            (modified: migrations, KB routes)
│   └── config.py                                         (Phase 1)
├── frontend/
│   ├── src/routes/chat/+page.svelte                     (Phase 5)
│   ├── src/lib/components/KBSelector.svelte             (Phase 5)
│   └── src/lib/stores/kb_store.ts                       (Phase 5)
├── compose/
│   ├── docker-compose.yml                               (Phase 1)
│   ├── .env.example                                      (Phase 1)
│   └── caddy/Caddyfile                                  (Phase 1)
├── tests/
│   ├── unit/
│   │   ├── test_kb_models.py                             (Phase 1)
│   │   ├── test_chat_kb_config.py                        (Phase 1)
│   │   └── test_model_manager.py                         (Phase 3)
│   └── integration/
│       ├── test_kb_isolation.py                          (Phase 6)
│       ├── test_rag_retrieval.py                         (Phase 6)
│       └── test_model_loading.py                         (Phase 3)
├── CLAUDE.md                                             (this file)
├── README.md                                             (user-facing overview)
└── .mcp.json                                             (sequential thinking MCP)
```

---

## 12. Success Criteria

### Phase Completion
- Each phase produces working, tested, deployable code
- Zero regression in prior phases
- All tests pass (unit + integration + E2E)

### Final Success (All Phases)
- ✅ Chat works for 5-10 concurrent users on single 32GB GPU
- ✅ Zero cross-user data leakage (isolation tests pass)
- ✅ RBAC enforcement verified (access control tests pass)
- ✅ RAG retrieval scoped by user-selected KBs
- ✅ Vision/Whisper load on-demand, auto-unload (no VRAM leaks)
- ✅ Single `docker-compose up` starts entire system
- ✅ Runbook allows non-technical admin to deploy
- ✅ Observability (metrics + logs) works end-to-end

---

## 13. Known Limitations & Future Work

**Not in Scope (This Project):**
- SSO/LDAP integration (local auth only)
- Document versioning (Phase 2 feature)
- Multi-GPU scaling (defer to Phase 2 + Kubernetes)
- Fine-tuning workflows (future)
- Compliance certifications (can layer later if needed)

**Future Phases:**
- **Phase 2:** Kubernetes deployment, 4×48GB GPUs, 70B models, higher concurrency
- **Phase 3:** Document versioning + rollback
- **Phase 4:** Custom fine-tuning pipelines
- **Phase 5:** SOC2/HIPAA compliance layer

---

## 14. Team & Decision-Making

**Primary Architect:** Jatin  
**Implementation:** Subagent-driven (Claude Code agents per task)  
**Review Checkpoints:** After each phase (isolation tests, performance benchmarks)  

**Key Decisions Made:**
- ✅ Option B (Smart Loading) for 32GB hardware
- ✅ Thin fork approach (not custom build or hard fork)
- ✅ Hybrid Qdrant collections (scalability over query speed)
- ✅ Parallel async retrieval (latency optimization)
- ✅ Tiered reranking (optional bottleneck)

---

## 15. Communication & Status

**Current Status:** Design complete, implementation plan ready  
**Next Action:** Start Phase 1 (DB schema + Docker)  
**Expected Duration:** ~3-4 weeks (full build)  

**Design Documents:**
- `2026-04-12-org-chat-assistant-design.md` — Original spec (sections 1-14)
- `2026-04-16-kb-rag-pipeline-workflow.md` — Detailed workflow (8-stage RAG, 10 mitigations)
- `2026-04-16-kb-rag-pipeline-implementation.md` — Implementation plan (7 phases, 33 tasks)

---

## Appendix: Quick Reference

### Docker Compose (Quick Start)
```bash
cd compose/
cp .env.example .env
# Edit .env with org name, domain
docker-compose up -d

# On-demand models start automatically when needed
# Always-running: vllm-chat + tei

# Check status
curl http://localhost:3000/health
```

### Key Endpoints
- Chat: `POST /api/chat/{chat_id}/messages`
- KB Admin: `POST /api/kb` (create), `GET /api/kb` (list)
- Upload: `POST /api/kb/{kb_id}/subtag/{subtag_id}/upload`
- Model Status: `GET /api/models/status`

### Environment Variables (Key)
```bash
DATABASE_URL=postgresql://postgres:...
OPENAI_API_BASE_URL=http://vllm-chat:8000/v1
RAG_EMBEDDING_OPENAI_API_BASE_URL=http://tei:80/v1
MODEL_UNLOAD_IDLE_SECS=300
```

---

**Ready for implementation. Awaiting phase-by-phase execution.**
