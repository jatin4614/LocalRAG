# Org Chat Assistant — Design Spec

**Date:** 2026-04-12
**Status:** Approved (pending user review)

## 1. Goal

Self-hosted, air-gapped, ChatGPT-like web assistant for an organization of 20–200 users, with strict per-user isolation, shared RBAC-gated knowledge collections, and document/audio/image input. No external network egress.

## 2. Starting Point

Thin fork of **Open WebUI**. Original request considered forking OpenClaw; Open WebUI is a much closer fit (native multi-user, RBAC, RAG, STT, vision, Ollama-native, Helm chart). Fork stays as small as possible — branding, disabled features, org defaults — so upstream upgrades remain cheap.

## 3. Architecture

Single Docker Compose stack on a Phase-1 GPU host. Phase-2 migration to Kubernetes (4×48GB GPUs).

Services:
1. **open-webui** — Svelte + FastAPI. Web UI, auth, chats, admin, RAG orchestration. Thin fork lives here.
2. **vllm-chat** — vLLM OpenAI-compatible server for chat model. GPU-attached. Port 8000 Docker-internal only. Continuous batching, PagedAttention.
3. **vllm-vision** — separate vLLM instance for vision-capable model (Qwen2-VL / LLaVA). Docker-internal only. Run as a second instance to avoid hot-swap latency.
4. **tei** — HuggingFace `text-embeddings-inference` server for embeddings. GPU or CPU.
5. **whisper** (faster-whisper server) — local STT.
6. **postgres** — users, chats, messages, collections, RBAC, audit log.
7. **qdrant** — vector store, namespaced per user / per collection.
8. **redis** — sessions, rate-limit counters, ingest job queue.
9. **caddy** — TLS, single LAN entry point. No public exposure.

**Why vLLM (no Ollama):** at 20–200 users Ollama's weak batching becomes the bottleneck. vLLM's continuous batching + PagedAttention gives 5–20× throughput on the same GPU for concurrent requests. Open WebUI talks to vLLM via its OpenAI-compatible endpoint (`/v1/chat/completions`) — still fully air-gapped because the base URL points at the internal service.

**Network posture:** compose network has no internet egress (host firewall + `internal: true` on non-public services). Only Caddy is reachable from org LAN.

**Isolation invariant:** every `documents` row, every Qdrant point, every uploaded file is tagged with `owner_user_id` XOR `collection_id`. Retrieval filters by current identity + group memberships, enforced in API layer (not UI).

**Phase-2 mapping:**
- Ollama → StatefulSet with one replica per GPU node, Service + round-robin.
- Postgres → operator (CloudNativePG) or internal managed.
- Qdrant → StatefulSet with PVC.
- Helm values overlay in `k8s/values.yaml`.

## 4. Roles, Groups, Data Model

### Roles
- `admin` — CRUD users, groups, collections; manage models; view audit log.
- `user` — chat; upload private docs; query collections they belong to.
- `pending` — registered, awaiting admin activation.

### Groups
Named sets of users (e.g., `engineering`, `hr`, `leadership`). A user may belong to many groups. Groups grant access to collections.

### Postgres schema (core tables)
- `users` — id, email, password_hash (Argon2id), role, status, created_at, last_login.
- `groups` — id, name, description.
- `user_groups` — user_id, group_id.
- `collections` — id, name, description, owner_admin_id.
- `collection_groups` — collection_id, group_id (read permission).
- `documents` — id, filename, owner_user_id NULL, collection_id NULL, mime, bytes, uploaded_at, ingest_status. CHECK: exactly one of owner_user_id / collection_id is non-null.
- `chats` — id, user_id, title, created_at, model.
- `messages` — id, chat_id, role, content, attachments, created_at.
- `audit_log` — id, ts, user_id, action, target_type, target_id, ip, metadata JSONB. Append-only, monthly partitioned.

### Vector store (Qdrant)
One collection per logical scope:
- `user_{user_id}` — private docs.
- `collection_{collection_id}` — shared.

Filtering at Qdrant payload level AND API-level post-check (defense in depth).

## 5. RAG Pipeline

1. Upload → file stored → ingest job enqueued in Redis.
2. Worker: extract text (pdf, docx, txt, md, xlsx) → semantic chunk (~800 tokens, 100 overlap) → embed via TEI (`BAAI/bge-m3` or `nomic-ai/nomic-embed-text-v1.5`) → upsert to the correct Qdrant namespace.
3. Query: resolve user's accessible scopes → top-k retrieval from each → rerank → inject into prompt.

## 6. Audio / Image

- **Audio:** browser records → `POST /api/audio/transcriptions` → Whisper → text becomes a user message.
- **Image:** attached to a message, routed to the `vllm-vision` service (e.g., `Qwen/Qwen2-VL-7B-Instruct` or `llava-hf/llava-v1.6-mistral-7b`) for that turn only. Stored per-user. Not RAG-indexed unless explicitly added as a document.

## 7. Auth, Sessions, RBAC Enforcement

### Auth flow
- Admin creates user → system generates one-time invite token → admin sends out-of-band.
- User opens invite link → sets password (min length + complexity) → lands in `pending` unless auto-activate is on.
- Login: email + password → Argon2id verify → HTTP-only, `SameSite=Lax`, `Secure` session cookie. TTL: 8h idle / 24h absolute. Server-side session store in Redis (admin-revocable).
- Password reset: admin-initiated only (no SMTP in air-gapped deploy).
- Sign-up page **disabled**: `ENABLE_SIGNUP=false` + route removed in fork.

### Enforcement rules
Every request → `user_id`, `role`, `group_ids`. Enforcement in one middleware + one query helper:
1. **Route-level:** admin routes require `role=admin`. Unit-tested.
2. **Row-level reads:** `documents` / `chats` / `messages` always filtered by `owner_user_id = :me` OR `collection_id IN (SELECT … WHERE group_id IN :my_groups)`. No bare `SELECT * FROM documents` in app code — lint rule + code review.
3. **Row-level writes:** uploads set `owner_user_id = :me` server-side only. Collection uploads require admin or explicit collection-write permission.
4. **Vector retrieval:** Qdrant `filter` for allowed namespaces; API asserts returned chunks are in allowed set before prompt injection.

### Audit log
Append-only. Events: login success/fail, user create/disable, password change, collection create/grant/revoke, document upload/delete, admin impersonation (disabled by default). Admin CSV/JSONL export.

### Rate limiting
Per-user token bucket in Redis: default 60 chat turns/hour, 20 doc uploads/hour. Tunable per role.

### Upload safety
MIME sniffing + extension allowlist. Max size 50MB (configurable). Open WebUI's existing CSRF/XSS protections retained; fork must not weaken them.

## 8. Repo Layout (Thin Fork)

```
<your-repo>/
├─ upstream/                 # submodule/subtree of open-webui
├─ patches/                  # git format-patch files
│   ├─ 0001-disable-signup.patch
│   ├─ 0002-remove-external-connectors.patch
│   └─ 0003-branding-assets.patch
├─ branding/                 # TODO: user-supplied assets
│   ├─ logo.svg              # TODO
│   ├─ logo-dark.svg         # TODO
│   ├─ favicon.ico           # TODO
│   └─ theme.css             # TODO: primary color vars
├─ compose/
│   ├─ docker-compose.yml
│   ├─ caddy/Caddyfile
│   └─ .env.example
├─ scripts/
│   ├─ rebase-upstream.sh
│   ├─ seed-admin.sh
│   └─ export-audit.sh
├─ k8s/
│   └─ values.yaml
└─ docs/
    └─ runbook.md
```

## 9. Branding Touchpoints

User will swap these files/values directly. Left as `TODO` in spec:

- `upstream/static/favicon/*` — favicon set. **TODO**
- `upstream/static/logo.svg`, `logo-dark.svg`, `splash.png` — logos. **TODO**
- `upstream/src/lib/constants.ts` — `WEBUI_NAME`. **TODO**
- `upstream/src/app.css` (or Tailwind config) — primary/accent color tokens. **TODO**
- `upstream/src/routes/auth/+page.svelte` — login page tagline. **TODO**
- `.env` — `WEBUI_NAME`, title fallbacks. **TODO**

Patch `0003-branding-assets.patch` captures these so rebases don't clobber them.

## 10. Environment Config

```
WEBUI_NAME=<TODO>
ENABLE_SIGNUP=false
DEFAULT_USER_ROLE=pending
ENABLE_OPENAI_API=true              # points at internal vLLM, not OpenAI
OPENAI_API_BASE_URL=http://vllm-chat:8000/v1
OPENAI_API_KEY=sk-internal-dummy    # vLLM accepts any token by default; set --api-key for auth
ENABLE_OLLAMA_API=false
ENABLE_WEB_SEARCH=false
ENABLE_IMAGE_GENERATION=false
ENABLE_RAG_WEB_LOADER=false
AUDIO_STT_ENGINE=whisper-local
RAG_EMBEDDING_ENGINE=openai         # TEI also speaks OpenAI-compatible /v1/embeddings
RAG_EMBEDDING_OPENAI_API_BASE_URL=http://tei:80/v1
RAG_EMBEDDING_MODEL=BAAI/bge-m3
VECTOR_DB=qdrant
DATABASE_URL=postgresql://...
REDIS_URL=redis://redis:6379
SESSION_SECRET=<generated>
```

## 11. Models

All served via vLLM (OpenAI-compatible) except embeddings (TEI) and STT (Whisper).

### Phase 1 — one 24GB GPU
- Chat: `Qwen/Qwen2.5-14B-Instruct-AWQ` (4-bit AWQ, fits with KV-cache headroom) — or `meta-llama/Llama-3.1-8B-Instruct` for more concurrency.
- Vision: `Qwen/Qwen2-VL-7B-Instruct` — loaded in a second vLLM instance if VRAM allows; otherwise run on CPU/swap in only when an image message arrives.
- Embeddings: TEI with `BAAI/bge-m3` (CPU-capable, GPU-accelerated if spare).
- STT: `faster-whisper` `medium` on GPU (or `small` on CPU).
- vLLM flags: `--max-model-len 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching`.

### Phase 2 — 4×48GB GPUs
- GPU 0–1: vLLM chat with `Qwen/Qwen2.5-72B-Instruct` or `meta-llama/Llama-3.3-70B-Instruct`, `--tensor-parallel-size 2`.
- GPU 2: vLLM vision (`Qwen/Qwen2-VL-72B-Instruct` or keep 7B with high concurrency).
- GPU 3: TEI embeddings + Whisper `large-v3` + headroom.
- vLLM flags: `--enable-prefix-caching --enable-chunked-prefill --max-num-seqs 256`.

## 12. Testing Strategy

- **Isolation tests (must-have):** user A uploads doc → user B must not retrieve any chunk from it, via API or chat. One test per endpoint touching documents/chats/collections.
- **RBAC tests:** user outside group X gets 403 on collection X's docs and zero RAG hits.
- **Auth tests:** session expiry, revocation, rate-limit trigger, CSRF.
- **Upload safety:** oversized file, wrong MIME, zip-bomb rejection.
- **Smoke E2E:** CI job spins up compose, seeds 2 users + 1 collection, runs isolation + RAG + audio + image flows. Runs on every patch change.

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Upstream breaking change to a patched file | Keep patches small and well-named; weekly rebase dry-run in CI. |
| Single-GPU VRAM exhaustion | Tune `--gpu-memory-utilization` and `--max-model-len`; run vision vLLM as a separate container started on demand; document limits in runbook. |
| vLLM cold start / model download on first boot | Pre-pull model weights into a persistent volume; `healthcheck` gates Open WebUI start-up until vLLM is ready. |
| Vision + chat competing for VRAM on Phase 1 | Load vision model only when an image message is sent (lazy second vLLM instance), or keep a small vision model permanently resident. |
| Audit log unbounded growth | Monthly partitioning + archival job. |
| Admin forgets to send invite | Panel shows "invite pending, copy link" explicitly. |
| Patch drift from upstream during long pauses | Lock upstream version; rebase deliberately, not continuously. |

## 14. Out of Scope (this spec)

- SSO / LDAP — defer until needed.
- External channel integrations (Slack, Teams, etc.) — explicitly unwanted.
- Self-service signup / SMTP email.
- Web-browsing / external tool calls.
- Image generation.
- Compliance certifications (SOC2/HIPAA) — can be layered later if required.
