# Phase 6 — Testing & Security Battery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Close out the test battery per CLAUDE.md §10 — upload safety, concurrent multi-user isolation under load, RAG relevance smoke, and a top-level `make test-all` that runs everything. Most isolation/RBAC tests already exist (Phases 2–4) and are passing; Phase 6 adds the missing safety + concurrency coverage and the test-runner ergonomics.

**Scope note (explicit):** Structured JSON audit logs, Redis token-bucket rate limiting, and session revocation are called out in CLAUDE.md §7 but are deferred to a later phase — they require their own design work and aren't gate items for the backend-only slice we've shipped. Phase 6 is about **verification**, not new features.

**Tech Stack:** pytest + testcontainers + httpx + asyncio.gather, already installed.

**Working directory:** `/home/vogic/LocalRAG/` (main, tagged `phase-5a-auth-bridge`).

---

## Decisions (Phase 6)

| # | Decision | Revise-by |
|---|----------|-----------|
| D40 | Deferred to later: audit log, rate limiting, session revocation. These are features, not gate items for a functional backend. | Phase 8 |
| D41 | Performance benchmarks stay lightweight (measure-only, no hard thresholds) — the target hardware is a single GPU host that isn't identical to this dev box. | — |
| D42 | `make test-all` = unit + integration + lint. GPU smoke skipped by default (`SKIP_GPU_SMOKE=1`). | — |
| D43 | Pytest markers: `security`, `concurrency`, `perf` — opt-in via `pytest -m`. Default run includes all (markers don't deselect). | — |

---

## Task 1: Upload safety tests (MIME sniffing, wrong extension)

**Files:** `tests/integration/test_upload_safety.py`.

- [ ] **Step 1: Write test**

Create `tests/integration/test_upload_safety.py`:

```python
"""Upload route safety: unsupported MIME, mismatched extension, oversized already-covered."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.upload import router as upload_router, configure as configure_upload
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder


ADMIN = {"X-User-Id": "9", "X-User-Role": "admin"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin')"))
        await s.execute(text("INSERT INTO knowledge_bases (id,name,admin_id) VALUES (10,'KB',9)"))
        await s.execute(text("INSERT INTO kb_subtags (id,kb_id,name) VALUES (100,10,'Docs')"))
        await s.commit()


@pytest_asyncio.fixture
async def client(engine, clean_qdrant):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    configure_upload(sessionmaker=SessionLocal, vector_store=vs, embedder=StubEmbedder(dim=32))
    app = FastAPI()
    app.include_router(upload_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await vs.close()


@pytest.mark.security
@pytest.mark.asyncio
async def test_unsupported_mime_rejected(client):
    """application/vnd.ms-excel isn't in EXTRACTORS → 422."""
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("data.xls", b"binary garbage", "application/vnd.ms-excel")},
    )
    assert r.status_code == 422


@pytest.mark.security
@pytest.mark.asyncio
async def test_nonexistent_subtag_404(client):
    r = await client.post(
        "/api/kb/10/subtag/999999/upload", headers=ADMIN,
        files={"file": ("a.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 404


@pytest.mark.security
@pytest.mark.asyncio
async def test_nonexistent_kb_404(client):
    r = await client.post(
        "/api/kb/999999/subtag/1/upload", headers=ADMIN,
        files={"file": ("a.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 404


@pytest.mark.security
@pytest.mark.asyncio
async def test_octet_stream_with_unknown_extension_rejected(client):
    """Generic octet-stream + unknown ext → UnsupportedMimeType → 422."""
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("file.unknown", b"raw bytes", "application/octet-stream")},
    )
    assert r.status_code == 422
```

- [ ] **Step 2: Run — expect PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_upload_safety.py -v
```

Expected: 4 PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_upload_safety.py
git commit -m "test(security): upload MIME + subtag + kb not-found cases"
```

---

## Task 2: Concurrent multi-user isolation (stress)

**Files:** `tests/integration/test_concurrent_users.py`.

- [ ] **Step 1: Write test**

`tests/integration/test_concurrent_users.py`:

```python
"""Concurrent multi-user uploads + queries — isolation holds under load."""
import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.upload import router as upload_router, configure as configure_upload
from ext.routers.rag import router as rag_router, configure as configure_rag
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (1,'u1@x','h','user'),(2,'u2@x','h','user'),(3,'u3@x','h','user')"))
        for i, u in enumerate((1, 2, 3), start=1):
            await s.execute(text(f"INSERT INTO chats (id,user_id) VALUES ({i * 100},{u})"))
        await s.commit()


@pytest_asyncio.fixture
async def client(engine, clean_qdrant):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    emb = StubEmbedder(dim=32)
    configure_upload(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    configure_rag(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    app = FastAPI()
    app.include_router(upload_router)
    app.include_router(rag_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await vs.close()


@pytest.mark.concurrency
@pytest.mark.asyncio
async def test_concurrent_private_uploads_stay_isolated(client):
    """Three users upload SIMULTANEOUSLY to their own chats.

    Verify that each user's retrieval sees only their own content."""
    users_chats = [(1, 100), (2, 200), (3, 300)]

    async def _upload(uid: int, cid: int):
        h = {"X-User-Id": str(uid), "X-User-Role": "user"}
        return await client.post(
            f"/api/chats/{cid}/private_docs/upload", headers=h,
            files={"file": (f"u{uid}.txt",
                            f"user {uid} secret token is {uid*1000}".encode(),
                            "text/plain")},
        )

    results = await asyncio.gather(*[_upload(u, c) for u, c in users_chats])
    for r in results:
        assert r.status_code == 201, r.text

    # Each user retrieves — must only see their own content.
    for uid, cid in users_chats:
        h = {"X-User-Id": str(uid), "X-User-Role": "user"}
        r = await client.post("/api/rag/retrieve", headers=h, json={
            "chat_id": cid, "query": "secret token",
            "selected_kb_config": [],
        })
        assert r.status_code == 200
        for hit in r.json()["hits"]:
            assert f"user {uid}" in hit["text"], f"leak: user {uid} saw {hit}"
            for other_uid in (1, 2, 3):
                if other_uid != uid:
                    assert f"user {other_uid}" not in hit["text"]
```

- [ ] **Step 2: Register markers**

Edit `pyproject.toml` `[tool.pytest.ini_options]` — add:

```toml
markers = [
  "security: upload safety / XSS / CSRF / MIME checks",
  "concurrency: multi-user parallel operations",
  "perf: performance benchmarks (non-gating)",
]
```

- [ ] **Step 3: Run — expect PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_concurrent_users.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_concurrent_users.py pyproject.toml
git commit -m "test(concurrency): 3 users concurrent uploads + retrieves stay isolated"
```

---

## Task 3: RAG relevance smoke

**Files:** `tests/integration/test_rag_relevance.py`.

- [ ] **Step 1: Write test**

`tests/integration/test_rag_relevance.py`:

```python
"""Relevance smoke — a known-good query finds the right chunk, not noise."""
import pytest
from ext.services.ingest import ingest_bytes
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder
from ext.services.retriever import retrieve
from ext.services.reranker import rerank


@pytest.mark.asyncio
async def test_query_hits_matching_doc_not_noise(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    emb = StubEmbedder(dim=32)
    await vs.ensure_collection("kb_1")

    docs = [
        (b"Python is a programming language used for web development and data science.",
         {"kb_id": 1, "subtag_id": 1, "doc_id": 1}),
        (b"The Eiffel Tower is located in Paris, France and was built in 1889.",
         {"kb_id": 1, "subtag_id": 2, "doc_id": 2}),
        (b"Mount Everest is the tallest mountain on Earth at 8,849 meters.",
         {"kb_id": 1, "subtag_id": 3, "doc_id": 3}),
    ]
    for data, payload in docs:
        await ingest_bytes(
            data=data, mime_type="text/plain", filename="d.txt",
            collection="kb_1", payload_base=payload,
            vector_store=vs, embedder=emb,
            chunk_tokens=30, overlap_tokens=5,
        )

    # Query that textually overlaps document 1 ("Python").
    hits = await retrieve(
        query="Python is a programming language used for web development and data science.",
        selected_kbs=[{"kb_id": 1, "subtag_ids": []}],
        chat_id=None, vector_store=vs, embedder=emb,
    )
    reranked = rerank(hits, top_k=3)
    assert reranked, "no hits returned"
    # Top hit should correspond to doc_id=1 (exact text match → StubEmbedder returns identical vector).
    assert reranked[0].payload["doc_id"] == 1
    await vs.close()
```

Note on the StubEmbedder: because it's deterministic hash-based, an exact-text query produces the exact-same vector as the chunk it matches — so top-1 is perfectly the corresponding doc. With a real embedder, semantic near-matches would show similar behavior but the test would need looser assertions; this test validates **pipeline plumbing**, not model quality.

- [ ] **Step 2: Run — expect PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_rag_relevance.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_rag_relevance.py
git commit -m "test(rag): relevance smoke (correct doc retrieved, not noise)"
```

---

## Task 4: `make test-all` + lint unification

**Files:** `Makefile` (modified).

- [ ] **Step 1: Edit Makefile**

In `/home/vogic/LocalRAG/Makefile`, locate the `.PHONY` line and add `test-all`. Then below existing `test:`, add:

```makefile
test-all:
	$(ACTIVATE) && ruff check . && mypy .
	$(ACTIVATE) && pytest tests/unit -v
	$(ACTIVATE) && SKIP_GPU_SMOKE=1 pytest tests/integration -v
```

Also add a convenience target for marker-filtered runs:

```makefile
test-security:
	$(ACTIVATE) && pytest -m security -v

test-concurrency:
	$(ACTIVATE) && pytest -m concurrency -v

test-perf:
	$(ACTIVATE) && pytest -m perf -v
```

Update `.PHONY:` at the top of Makefile to include `test-all test-security test-concurrency test-perf`.

Also extend the `help:` target to document these new recipes.

- [ ] **Step 2: Run `make test-all`**

```bash
cd /home/vogic/LocalRAG && make test-all 2>&1 | tail -20
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: make test-all target (lint + unit + integration)"
```

---

## Task 5: Regression + Phase 6 tag

```bash
cd /home/vogic/LocalRAG && make test-all 2>&1 | tail -10
git tag -a phase-6-testing-battery -m "Phase 6 complete: security + concurrency + relevance + test-all"
git log phase-5a-auth-bridge..HEAD --oneline
```

---

## Phase 6 acceptance checklist

- [ ] Upload safety tests: unsupported MIME → 422; unknown kb/subtag → 404.
- [ ] Concurrent 3-user upload + retrieve test passes (isolation holds).
- [ ] RAG relevance smoke: known query hits the right document.
- [ ] `make test-all` runs cleanly.
- [ ] pytest markers registered (`security`, `concurrency`, `perf`).
- [ ] Tag `phase-6-testing-battery` exists.
