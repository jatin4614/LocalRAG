# Learning Plan — Org Chat Assistant Project

**Audience:** you — comfortable coding, weaker on architecture & system design.
**Goal:** by the end, you can reason about the design in the spec, implement it, and make sensible changes when reality pushes back.
**Suggested pace:** 4–6 weeks, ~1 hour/day. Do the *Build* step in each section — reading without doing will not stick.

---

## How to use this plan

1. Do sections in order. Each one feeds the next.
2. Every section has: **Read**, **Watch** (optional), **Build**. Skip reading, do the Build — you'll come back to reading when you hit a wall.
3. When a term feels fuzzy, look it up *once*, then keep moving. Depth comes from doing, not from re-reading.
4. Keep a `learning-log.md` — one paragraph per day on what you tried and what broke.

---

## Week 1 — System design foundations

The single biggest leverage area for you. These concepts repeat everywhere in the spec.

**Concepts to internalize (not just recognize):**
- Client ↔ server ↔ database — why each layer exists, who owns state.
- Stateless vs stateful services — why auth tokens/sessions matter.
- Synchronous vs asynchronous work (request/response vs queues/workers).
- Horizontal vs vertical scaling; "where is the bottleneck" as a question you always ask.
- Caching — what it buys you, what it breaks (staleness, invalidation).
- Defense in depth — never rely on one layer for security.
- "Isolation invariant" — what it means for multi-tenant data and why it's a one-line bug away from a leak.

**Read (pick any one, don't read all):**
- *Designing Data-Intensive Applications*, Kleppmann — ch. 1, 2, 5, 7, 8. The canonical book.
- `github.com/donnemartin/system-design-primer` — free, dense, skim the TOC and read what's new to you.
- *System Design Interview*, Alex Xu — lighter, example-driven.

**Watch (optional):**
- "Gaurav Sen" or "ByteByteGo" on YouTube — 2–3 videos on load balancing, caching, and databases.

**Build:** sketch the spec's architecture on paper from memory. Which boxes exist, what goes between them, where does a user's request flow through on: (a) login, (b) sending a chat message, (c) uploading a PDF. If you get stuck, reread §3 of the spec, then try again.

---

## Week 2 — The web stack under the hood

Open WebUI is **Svelte (frontend) + FastAPI (Python backend)**. You don't need expertise in either — you need to *read* them comfortably.

**Read the official docs' first few pages of each (not all, just enough to navigate):**
- FastAPI — `fastapi.tiangolo.com` — routes, dependencies, background tasks, middleware.
- Svelte / SvelteKit — `svelte.dev/docs` and `kit.svelte.dev/docs` — components, routes, load functions. You'll mostly be editing, not writing from scratch.
- HTTP basics you probably half-know: methods, status codes, cookies, CORS, CSRF. MDN is the reference.

**Topics to actually understand:**
- How a session cookie flows from login → every subsequent request.
- Why `HttpOnly` + `SameSite` + `Secure` together matter.
- Middleware as the place where authn/authz happens before your route code runs.

**Build:** clone Open WebUI into a throwaway folder, run it with Docker, poke around the `/api/*` endpoints using your browser's DevTools Network tab. Log in, send a chat, upload a doc, watch the requests. Then find the matching Python handlers in the repo. *This is 80% of your onboarding.*

---

## Week 3 — The LLM serving stack

### vLLM
**Read:** `docs.vllm.ai` — quickstart, OpenAI-compatible server, engine args. Focus on:
- Continuous batching (what it is, why Ollama doesn't do it well).
- PagedAttention (you only need the intuition: KV cache is paged like virtual memory, so GPU doesn't waste VRAM).
- `--tensor-parallel-size`, `--max-model-len`, `--gpu-memory-utilization`, `--enable-prefix-caching`.

**Build:** run vLLM locally (even on a small model like `Qwen2.5-0.5B-Instruct`), hit `/v1/chat/completions` with `curl`, then with the OpenAI Python SDK. Stream a response.

### Embeddings + TEI
**Read:** HuggingFace `text-embeddings-inference` README. Understand what an embedding is (a vector representation of meaning) and what "cosine similarity" measures.

**Build:** embed 20 sentences with TEI, compute pairwise similarity, confirm synonyms cluster.

### Whisper
**Read:** `faster-whisper` README. Understand model sizes vs accuracy vs speed.

**Build:** transcribe a 30-second audio file.

---

## Week 4 — Data: Postgres, Qdrant, Redis

### Postgres
**Read:** the official tutorial (`postgresql.org/docs`), chapters: SQL basics, indexes, transactions, foreign keys, CHECK constraints. Then: `EXPLAIN ANALYZE`.

**Topics:**
- Row-level filtering as your multi-tenant guardrail — why every query that touches user data needs a `WHERE owner_user_id = …` clause.
- Partitioning (for the audit log).
- Migrations — Alembic is the usual tool for FastAPI projects.

**Build:** create the `users`, `documents`, `collections`, `collection_groups` tables from §4 of the spec by hand. Insert rows. Write the row-level filter query for "documents I can see." Break it on purpose (remove the filter) and see what that would leak.

### Qdrant (vector DB)
**Read:** `qdrant.tech/documentation` — quickstart, collections, filtering by payload, hybrid search.

**Build:** embed 10 paragraphs from two "users" into one collection, use a payload filter so queries from user A never see user B's chunks. This is the core of your RAG isolation test.

### Redis
**Read:** `redis.io/docs` — data types overview, TTLs, `INCR` for rate limiting, streams/lists for job queues.

**Build:** implement a per-user rate limiter (token bucket) in a script. 60 req/hour. Verify the 61st fails and rolls over after an hour.

---

## Week 5 — RAG in depth

**Read:**
- "Retrieval-Augmented Generation" — the original Meta paper (2020). Skim, not study.
- Any modern practical guide — search "RAG best practices 2024/2025." Focus on: chunking strategies, reranking, evaluation.
- Open WebUI's RAG docs — understand how it does chunking/retrieval so your fork doesn't fight it.

**Topics:**
- Chunking: fixed vs semantic; overlap; why 800 tokens is a reasonable default.
- Retrieval: top-k, reranking (cross-encoder), score thresholds.
- Evaluation: "answer contains the right citation" is a better signal than "answer looks right."

**Build:** wire the pieces from Weeks 3–4 into a toy RAG: upload 3 PDFs → chunk → embed with TEI → store in Qdrant with a `user_id` payload → query with vLLM, injecting top-3 chunks. Then add a second "user" and verify isolation.

---

## Week 6 — Deployment & ops

### Docker + Docker Compose
**Read:** `docs.docker.com` — compose basics, networks, volumes, healthchecks, `internal: true` networks, secrets.

**Topics:**
- Named volumes for model weights (huge — don't rebuild the image each time).
- Healthchecks for startup ordering (vLLM takes minutes to become ready).
- `depends_on: condition: service_healthy`.

**Build:** stand up a mini version of the full stack locally — Open WebUI + vLLM (tiny model) + TEI + Postgres + Qdrant + Redis + Caddy. Log in, chat, upload a doc. If this works, 60% of the project works.

### Kubernetes (only once Phase 1 is running)
**Read:** `kubernetes.io/docs` — concepts: Pod, Deployment, Service, ConfigMap, Secret, PVC, StatefulSet. Then Helm basics.

**Topics:**
- Why GPU workloads are usually StatefulSets.
- `nvidia/k8s-device-plugin` for GPU scheduling.
- Helm values files as the swap point between environments.

**Build:** deploy your Phase-1 compose to a local K8s (Minikube / Kind / k3d). Don't chase production-grade until it runs.

### Caddy
**Read:** `caddyserver.com/docs` — Caddyfile, automatic HTTPS (you'll use an internal CA or self-signed since you're air-gapped).

---

## Cross-cutting — Security, auth, multi-tenancy

Read these *while* you're working through the above, not after.

- OWASP Top 10 — at least skim. Specifically: broken access control, IDOR, SSRF (less relevant air-gapped but good habits).
- "Multi-tenant" articles — search for "multi-tenant SaaS row-level security." The pattern is the same even without cloud.
- Argon2id (password hashing) — why, not how to implement it.
- JWT vs server-side sessions — you're using server-side sessions for revocability; know the tradeoff.

**Red-team yourself:** once the RAG toy from Week 5 works, pretend you're user B trying to read user A's docs. What endpoints would you hit? What query params would you tamper with? Make sure each fails.

---

## What you can safely skip (for now)

- Deep ML theory (transformer internals, attention math). You're integrating, not training.
- Niche infra (service mesh, sidecars, advanced K8s operators) until Phase 2 forces them.
- Frontend design systems — Open WebUI already has one.
- Frontend framework wars — Svelte is fine; don't switch it.

---

## Checkpoint questions (try to answer without looking)

If you can answer these clearly in your own words, you're ready to start implementation:

1. When user A uploads a PDF and user B sends a question, name every service the bytes pass through and what each does.
2. Why does `ENABLE_SIGNUP=false` alone not make your deploy secure? What else matters?
3. What's the one SQL clause whose absence would leak data across users?
4. Why vLLM over Ollama at 50 concurrent users — in one sentence.
5. What breaks if the Qdrant `filter` clause is misconfigured, and what other layer catches it?
6. If vLLM is still loading when Open WebUI starts, what do users see, and how do you prevent it?

---

## Single-sentence study rule

If a topic doesn't help you answer one of the six questions above or build one of the Builds, it's not on the critical path yet. Park it.
