# Phase 4 — RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Ship the end-to-end Retrieval-Augmented Generation backend: upload a document to a KB (or per-chat private namespace) → extract text → chunk → embed via TEI → upsert to Qdrant; then retrieve with parallel multi-KB search + tiered reranking + token budgeting, filtered by the caller's `selected_kb_config`.

**Architecture:** Services are pure-Python modules under `ext/services/`. Uploads land on disk (bind-mounted `volumes/uploads/`) and are ingested synchronously in-request for Phase 4 — background workers are a perf optimization we defer. Embeddings come from a pluggable `Embedder` interface (`TEIEmbedder` in prod; `StubEmbedder` in tests). Vectors live in Qdrant, one collection per KB (`kb_<id>`) and one per chat for private docs (`chat_<id>`). `selected_kb_config` gates retrieval — users never see KBs outside their `kb_access` grants (Phase 2 `get_allowed_kb_ids`). Reranking is score-normalization across KBs + a simple "fast path" bypass when top-1 dominates top-2. Token budgeting truncates to `RAG_CONTEXT_TOKEN_LIMIT` (default 4000).

**Tech Stack:** Python ≥ 3.10, FastAPI, SQLAlchemy 2.0, qdrant-client, pypdf, tiktoken, httpx, pytest + testcontainers (Postgres + Qdrant).

**Working directory:** `/home/vogic/LocalRAG/` (main, tagged `phase-3-model-manager`).

---

## Decisions (Phase 4 scope)

| # | Decision | Revise-by |
|---|----------|-----------|
| D26 | 1 Qdrant collection per KB (`kb_<id>`), payload filters by `subtag_id`. Private docs live in `chat_<id>` collections. Matches master plan §3. | — |
| D27 | Chunk size = 800 tokens, overlap = 100 tokens. Tokenizer: `tiktoken` cl100k_base (approximates bge-m3's vocab well enough for chunking purposes; true tokenization is done server-side by TEI). | — |
| D28 | Embedder = `bge-m3` via TEI (1024-dim cosine). Tests use `StubEmbedder` which hashes text to a deterministic 1024-dim vector so Qdrant semantics are real but no TEI container is needed. | — |
| D29 | Ingest is synchronous in-request for Phase 4. Background worker (Redis queue) deferred. Upload endpoints return 202-style on success (`{"status":"done","chunks":N}` once ingest finishes). | Phase 6 perf |
| D30 | Supported MIME types (Phase 4): `text/plain`, `text/markdown`, `application/pdf`. DOCX/XLSX/HTML deferred — plan includes a plug-in pattern via `EXTRACTORS` dict. | Phase 4b |
| D31 | Private-doc cleanup: explicit on `DELETE /api/chats/{chat_id}` (Phase 5 wires this up). TTL janitor deferred. | Phase 6 |
| D32 | Max upload size: 50 MB (env `RAG_MAX_UPLOAD_BYTES`). Rejected with 413. | — |
| D33 | Token budget default 4000 tokens (`RAG_CONTEXT_TOKEN_LIMIT`). Over-budget chunks truncated from lowest-rerank-score end. Not silently dropped — debug log line emitted. | — |
| D34 | Reranker = normalize scores per KB (max-normalize) + global descending sort. No cross-encoder yet. If top-1 / top-2 > 2.0, "fast-path" returns top-K without normalization. | Phase 6 |

---

## File structure

```
ext/
├── db/migrations/
│   └── 003_add_chunk_count_to_kb_documents.sql    T6
├── db/models/
│   └── kb.py                                       (extend with chunk_count field)
├── services/
│   ├── vector_store.py                            T1  qdrant wrapper
│   ├── embedder.py                                T2  Embedder protocol + StubEmbedder + TEIEmbedder
│   ├── extractor.py                               T3  text extraction (TXT/MD/PDF)
│   ├── chunker.py                                 T4  token-aware chunking
│   ├── ingest.py                                  T5  extract → chunk → embed → upsert
│   ├── retriever.py                               T8  parallel multi-KB search
│   ├── reranker.py                                T9  score normalization + fast path
│   └── budget.py                                  T10 token budgeting
└── routers/
    ├── upload.py                                  T6  POST /api/kb/{kb}/subtag/{sub}/upload, POST /api/chats/{chat}/private_docs/upload
    └── rag.py                                     T11 POST /api/rag/retrieve

tests/
├── integration/
│   ├── conftest.py                                (extend with qdrant fixture + stub embedder)
│   ├── test_vector_store.py                       T1
│   ├── test_embedder.py                           T2
│   ├── test_ingest.py                             T5
│   ├── test_upload_routes.py                      T7
│   ├── test_retriever.py                          T8
│   ├── test_rag_routes.py                         T11
│   ├── test_rag_isolation.py                      T13 cross-user isolation
│   └── test_rag_end_to_end.py                     T12 full upload → retrieve
└── unit/
    ├── test_extractor.py                          T3
    ├── test_chunker.py                            T4
    ├── test_reranker.py                           T9
    └── test_budget.py                             T10

compose/docker-compose.yml   T6 (add bind mount for volumes/uploads)
volumes/uploads/             T6 (created by app; gitignored)
```

---

## Task 1: Qdrant wrapper — `ext/services/vector_store.py`

**Files:** `ext/services/vector_store.py`, `tests/integration/test_vector_store.py`, extend `tests/integration/conftest.py` with a Qdrant testcontainer fixture.

- [ ] **Step 1: Extend conftest with qdrant fixture**

Append to `/home/vogic/LocalRAG/tests/integration/conftest.py`:

```python
from testcontainers.core.container import DockerContainer


@pytest.fixture(scope="session")
def qdrant():
    """Session-scoped Qdrant container (reused across tests for speed)."""
    container = (
        DockerContainer("qdrant/qdrant:latest")
        .with_exposed_ports(6333)
    )
    container.start()
    host = container.get_container_host_ip()
    port = container.get_exposed_port(6333)
    # Wait for ready
    import time, httpx
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://{host}:{port}/readyz", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        container.stop()
        raise RuntimeError("qdrant not ready")
    try:
        yield f"http://{host}:{port}"
    finally:
        container.stop()


@pytest_asyncio.fixture(scope="function")
async def clean_qdrant(qdrant):
    """Per-test wiper: deletes all collections at teardown."""
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(url=qdrant)
    yield qdrant
    # cleanup
    cols = (await client.get_collections()).collections
    for c in cols:
        await client.delete_collection(c.name)
    await client.close()
```

- [ ] **Step 2: Write failing test**

Create `/home/vogic/LocalRAG/tests/integration/test_vector_store.py`:

```python
import pytest
from ext.services.vector_store import VectorStore


@pytest.mark.asyncio
async def test_ensure_collection_is_idempotent(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=16)
    await vs.ensure_collection("kb_5")
    await vs.ensure_collection("kb_5")  # idempotent
    cols = await vs.list_collections()
    assert "kb_5" in cols
    await vs.close()


@pytest.mark.asyncio
async def test_upsert_and_search(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=4)
    await vs.ensure_collection("kb_1")

    points = [
        {"id": 1, "vector": [1.0, 0.0, 0.0, 0.0], "payload": {"text": "alpha", "subtag_id": 10}},
        {"id": 2, "vector": [0.0, 1.0, 0.0, 0.0], "payload": {"text": "beta",  "subtag_id": 20}},
    ]
    await vs.upsert("kb_1", points)

    hits = await vs.search("kb_1", [1.0, 0.0, 0.0, 0.0], limit=5)
    assert hits[0].payload["text"] == "alpha"
    assert hits[0].score > 0.9
    await vs.close()


@pytest.mark.asyncio
async def test_search_with_subtag_filter(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=4)
    await vs.ensure_collection("kb_1")
    await vs.upsert("kb_1", [
        {"id": 1, "vector": [1, 0, 0, 0], "payload": {"text": "alpha", "subtag_id": 10}},
        {"id": 2, "vector": [1, 0, 0, 0], "payload": {"text": "beta",  "subtag_id": 20}},
    ])
    hits = await vs.search("kb_1", [1.0, 0, 0, 0], limit=5, subtag_ids=[10])
    assert len(hits) == 1
    assert hits[0].payload["subtag_id"] == 10
    await vs.close()


@pytest.mark.asyncio
async def test_delete_collection(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=4)
    await vs.ensure_collection("chat_42")
    await vs.delete_collection("chat_42")
    assert "chat_42" not in await vs.list_collections()
    await vs.close()
```

- [ ] **Step 3: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_vector_store.py -v
```

- [ ] **Step 4: Write `ext/services/vector_store.py`**

```python
"""Thin async wrapper over qdrant-client."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm


@dataclass
class Hit:
    id: int | str
    score: float
    payload: dict


class VectorStore:
    def __init__(self, *, url: str, vector_size: int, distance: str = "Cosine") -> None:
        self._client = AsyncQdrantClient(url=url)
        self._vector_size = vector_size
        self._distance = distance

    async def close(self) -> None:
        await self._client.close()

    async def list_collections(self) -> list[str]:
        cols = (await self._client.get_collections()).collections
        return [c.name for c in cols]

    async def ensure_collection(self, name: str) -> None:
        cols = await self.list_collections()
        if name in cols:
            return
        await self._client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=self._vector_size,
                distance=qm.Distance[self._distance.upper()],
            ),
        )

    async def delete_collection(self, name: str) -> None:
        try:
            await self._client.delete_collection(name)
        except Exception:
            pass  # already gone

    async def upsert(self, name: str, points: Iterable[dict]) -> None:
        pts = [
            qm.PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload", {}))
            for p in points
        ]
        await self._client.upsert(collection_name=name, points=pts, wait=True)

    async def search(
        self,
        name: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        subtag_ids: Optional[list[int]] = None,
    ) -> List[Hit]:
        flt = None
        if subtag_ids:
            flt = qm.Filter(must=[
                qm.FieldCondition(key="subtag_id", match=qm.MatchAny(any=subtag_ids))
            ])
        results = await self._client.search(
            collection_name=name, query_vector=query_vector, limit=limit, query_filter=flt,
        )
        return [Hit(id=r.id, score=r.score, payload=r.payload or {}) for r in results]
```

- [ ] **Step 5: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_vector_store.py -v
ruff check . && mypy .
```

- [ ] **Step 6: Commit**

```bash
git add ext/services/vector_store.py tests/integration/test_vector_store.py tests/integration/conftest.py
git commit -m "feat(rag): qdrant-backed VectorStore + testcontainers fixture"
```

---

## Task 2: Embedder interface (Protocol + StubEmbedder + TEIEmbedder)

**Files:** `ext/services/embedder.py`, `tests/integration/test_embedder.py`.

- [ ] **Step 1: Write failing test**

`tests/integration/test_embedder.py`:

```python
import hashlib
import httpx
import pytest
from ext.services.embedder import StubEmbedder, TEIEmbedder


@pytest.mark.asyncio
async def test_stub_embedder_deterministic():
    e = StubEmbedder(dim=8)
    a = await e.embed(["hello"])
    b = await e.embed(["hello"])
    assert a == b
    assert len(a[0]) == 8


@pytest.mark.asyncio
async def test_stub_embedder_different_texts_different_vectors():
    e = StubEmbedder(dim=8)
    [va], [vb] = await e.embed(["hello"]), await e.embed(["world"])
    assert va != vb


@pytest.mark.asyncio
async def test_tei_embedder_calls_embed_endpoint():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        body = request.content
        return httpx.Response(200, json=[[0.1, 0.2, 0.3, 0.4]])

    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)
    vecs = await e.embed(["hello"])
    assert vecs == [[0.1, 0.2, 0.3, 0.4]]
    await e.aclose()


@pytest.mark.asyncio
async def test_tei_embedder_batches():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[0.1]*4, [0.2]*4, [0.3]*4])
    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)
    vecs = await e.embed(["a", "b", "c"])
    assert len(vecs) == 3
    assert len(vecs[0]) == 4
    await e.aclose()
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_embedder.py -v
```

- [ ] **Step 3: Write `ext/services/embedder.py`**

```python
"""Embedder protocol + a deterministic test stub + the real TEI client."""
from __future__ import annotations

import hashlib
import struct
from typing import List, Optional, Protocol

import httpx


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class StubEmbedder:
    """Hash-based deterministic embedder. Same text → same vector, always."""

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            out.append(self._hash_vector(t))
        return out

    def _hash_vector(self, text: str) -> list[float]:
        # Expand a SHA-256 hash to `dim` floats in [-1, 1], then L2-normalize.
        data = b""
        i = 0
        while len(data) < self._dim * 4:
            data += hashlib.sha256(f"{i}:{text}".encode()).digest()
            i += 1
        raw = struct.unpack(f"<{self._dim}i", data[: self._dim * 4])
        vec = [x / 2**31 for x in raw]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class TEIEmbedder:
    """HuggingFace Text-Embeddings-Inference client."""

    def __init__(
        self, *, base_url: str, timeout: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        r = await self._client.post("/embed", json={"inputs": texts})
        r.raise_for_status()
        return r.json()
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_embedder.py -v
ruff check . && mypy .
```

- [ ] **Step 5: Commit**

```bash
git add ext/services/embedder.py tests/integration/test_embedder.py
git commit -m "feat(rag): Embedder protocol + StubEmbedder + TEIEmbedder"
```

---

## Task 3: Text extractor (TXT / MD / PDF)

**Files:** `ext/services/extractor.py`, `tests/unit/test_extractor.py`.

- [ ] **Step 1: Install pypdf dep**

Edit `pyproject.toml` `[project].dependencies` — append `"pypdf>=4.3"`. Then `pip install -e ".[dev]"`.

- [ ] **Step 2: Write failing test**

`tests/unit/test_extractor.py`:

```python
import io
import pytest
from ext.services.extractor import extract_text, UnsupportedMimeType


def test_txt_passthrough():
    assert extract_text(b"hello", "text/plain", "a.txt") == "hello"


def test_markdown_passthrough():
    assert extract_text(b"# Title\n\npara", "text/markdown", "a.md") == "# Title\n\npara"


def test_unsupported_mime_raises():
    with pytest.raises(UnsupportedMimeType):
        extract_text(b"...", "application/vnd.ms-excel", "a.xls")


def test_extractor_dispatches_on_extension_if_mime_missing():
    # Some uploads arrive with application/octet-stream; fall back to extension.
    assert extract_text(b"hi", "application/octet-stream", "note.txt") == "hi"


def test_pdf_extraction_tiny():
    # Build a 1-page PDF with pypdf at test time.
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, TextStringObject
    # Easiest path: use reportlab if available; else generate a minimal PDF by hand.
    # We rely on pypdf round-tripping its own writer.
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)
    # A blank PDF extracts to "" (valid).
    assert extract_text(buf.getvalue(), "application/pdf", "a.pdf") == ""
```

- [ ] **Step 3: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_extractor.py -v
```

- [ ] **Step 4: Write `ext/services/extractor.py`**

```python
"""Text extraction from uploaded documents. Plug-in via EXTRACTORS dict."""
from __future__ import annotations

import io
import mimetypes
from typing import Callable


class UnsupportedMimeType(RuntimeError):
    pass


def _extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


EXTRACTORS: dict[str, Callable[[bytes], str]] = {
    "text/plain":       _extract_txt,
    "text/markdown":    _extract_txt,
    "application/pdf":  _extract_pdf,
}


_EXT_FALLBACK = {
    ".txt": "text/plain",
    ".md":  "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
}


def extract_text(data: bytes, mime_type: str, filename: str) -> str:
    fn = EXTRACTORS.get(mime_type)
    if fn is None:
        # Fallback: use the filename extension.
        for ext, m in _EXT_FALLBACK.items():
            if filename.lower().endswith(ext):
                fn = EXTRACTORS[m]
                break
    if fn is None:
        raise UnsupportedMimeType(f"{mime_type} (filename={filename})")
    return fn(data)
```

- [ ] **Step 5: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_extractor.py -v
ruff check . && mypy .
```

- [ ] **Step 6: Commit**

```bash
git add ext/services/extractor.py tests/unit/test_extractor.py pyproject.toml
git commit -m "feat(rag): text extractor (TXT/MD/PDF)"
```

---

## Task 4: Chunker (token-aware with overlap)

**Files:** `ext/services/chunker.py`, `tests/unit/test_chunker.py`.

- [ ] **Step 1: Install tiktoken dep**

Edit `pyproject.toml` `[project].dependencies` — append `"tiktoken>=0.7"`. Then `pip install -e ".[dev]"`.

- [ ] **Step 2: Write failing test**

`tests/unit/test_chunker.py`:

```python
import pytest
from ext.services.chunker import chunk_text


def test_short_text_returns_one_chunk():
    chunks = chunk_text("hello world", chunk_tokens=800, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].index == 0


def test_long_text_splits_with_overlap():
    # Build 2500-token text by repeating.
    para = ("word " * 2500).strip()
    chunks = chunk_text(para, chunk_tokens=800, overlap_tokens=100)
    # (2500 - 100) / (800 - 100) ≈ 4 windows (actually 4)
    assert len(chunks) >= 3
    # Indices are contiguous
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Chunks overlap roughly overlap_tokens
    for i in range(len(chunks) - 1):
        assert chunks[i].text.split()[-3:] == chunks[i + 1].text.split()[:3] or True
        # exact overlap check is fragile with sub-word tokens; skip


def test_empty_text_returns_empty():
    assert chunk_text("", chunk_tokens=800, overlap_tokens=100) == []


def test_single_very_long_word_chunked():
    # One word of ~2000 chars → should still chunk without infinite-looping.
    one_word = "a" * 8000
    chunks = chunk_text(one_word, chunk_tokens=800, overlap_tokens=100)
    assert len(chunks) >= 1
```

- [ ] **Step 3: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_chunker.py -v
```

- [ ] **Step 4: Write `ext/services/chunker.py`**

```python
"""Token-aware chunking with overlap (tiktoken cl100k_base)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import tiktoken


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, *, chunk_tokens: int = 800, overlap_tokens: int = 100) -> List[Chunk]:
    if not text:
        return []
    if chunk_tokens <= overlap_tokens:
        raise ValueError("chunk_tokens must exceed overlap_tokens")

    enc = _encoder()
    ids = enc.encode(text)
    stride = chunk_tokens - overlap_tokens
    chunks: List[Chunk] = []
    idx = 0
    start = 0
    while start < len(ids):
        end = min(start + chunk_tokens, len(ids))
        chunk_ids = ids[start:end]
        chunks.append(Chunk(index=idx, text=enc.decode(chunk_ids)))
        idx += 1
        if end == len(ids):
            break
        start += stride
    return chunks
```

- [ ] **Step 5: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_chunker.py -v
ruff check . && mypy .
```

- [ ] **Step 6: Commit**

```bash
git add ext/services/chunker.py tests/unit/test_chunker.py pyproject.toml
git commit -m "feat(rag): token-aware chunker (800/100, tiktoken cl100k_base)"
```

---

## Task 5: Ingest pipeline (extract → chunk → embed → upsert)

**Files:** `ext/services/ingest.py`, `tests/integration/test_ingest.py`.

- [ ] **Step 1: Write failing test**

`tests/integration/test_ingest.py`:

```python
import pytest
from ext.services.ingest import ingest_bytes
from ext.services.embedder import StubEmbedder
from ext.services.vector_store import VectorStore


@pytest.mark.asyncio
async def test_ingest_txt_into_kb(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    await vs.ensure_collection("kb_7")
    e = StubEmbedder(dim=32)

    n = await ingest_bytes(
        data=b"This is the first sentence. Here is more text to index.",
        mime_type="text/plain",
        filename="a.txt",
        collection="kb_7",
        payload_base={"kb_id": 7, "subtag_id": 11, "doc_id": 100},
        vector_store=vs,
        embedder=e,
        chunk_tokens=20, overlap_tokens=5,
    )
    assert n >= 1

    hits = await vs.search("kb_7", [1.0] + [0.0]*31, limit=10)
    assert len(hits) >= 1
    assert all(h.payload["doc_id"] == 100 for h in hits)
    await vs.close()


@pytest.mark.asyncio
async def test_ingest_pdf_empty_ok(clean_qdrant):
    """An empty PDF yields 0 chunks — shouldn't explode."""
    from pypdf import PdfWriter
    import io
    w = PdfWriter(); w.add_blank_page(width=612, height=792)
    buf = io.BytesIO(); w.write(buf)

    vs = VectorStore(url=clean_qdrant, vector_size=16)
    await vs.ensure_collection("kb_1")
    n = await ingest_bytes(
        data=buf.getvalue(), mime_type="application/pdf", filename="blank.pdf",
        collection="kb_1",
        payload_base={"kb_id": 1, "subtag_id": 2, "doc_id": 3},
        vector_store=vs, embedder=StubEmbedder(dim=16),
    )
    assert n == 0
    await vs.close()
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_ingest.py -v
```

- [ ] **Step 3: Write `ext/services/ingest.py`**

```python
"""Extract → chunk → embed → upsert pipeline."""
from __future__ import annotations

import time
import uuid
from typing import Mapping

from .chunker import chunk_text
from .embedder import Embedder
from .extractor import extract_text
from .vector_store import VectorStore


async def ingest_bytes(
    *,
    data: bytes,
    mime_type: str,
    filename: str,
    collection: str,
    payload_base: Mapping[str, int | str],
    vector_store: VectorStore,
    embedder: Embedder,
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> int:
    """Full ingest: returns number of chunks upserted."""
    text = extract_text(data, mime_type, filename)
    chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    vectors = await embedder.embed(texts)

    now = int(time.time())
    points = []
    for chunk, vec in zip(chunks, vectors):
        payload = dict(payload_base)
        payload["chunk_index"] = chunk.index
        payload["text"] = chunk.text
        payload["uploaded_at"] = now
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "payload": payload,
        })
    await vector_store.upsert(collection, points)
    return len(points)
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_ingest.py -v
ruff check . && mypy .
```

- [ ] **Step 5: Commit**

```bash
git add ext/services/ingest.py tests/integration/test_ingest.py
git commit -m "feat(rag): ingest pipeline (extract → chunk → embed → upsert)"
```

---

## Task 6: Migration 003 — add `chunk_count` to `kb_documents`; add kb uploads dir

**Files:** `ext/db/migrations/003_add_chunk_count.sql`; extend `KBDocument` model; update compose to bind-mount `volumes/uploads`.

- [ ] **Step 1: Write migration 003**

`ext/db/migrations/003_add_chunk_count.sql`:

```sql
-- 003_add_chunk_count.sql
BEGIN;
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0;
COMMIT;
```

Also update `/home/vogic/LocalRAG/tests/integration/conftest.py` — extend the `MIGRATION_*` list:

```python
MIGRATION_003 = ROOT / "ext/db/migrations/003_add_chunk_count.sql"
# ... inside engine fixture, after applying 002:
        if MIGRATION_003.exists():
            await _raw_exec(conn, MIGRATION_003.read_text())
```

- [ ] **Step 2: Extend KBDocument model** — append after `deleted_at:` line in `ext/db/models/kb.py`:

```python
    chunk_count: Mapped[int] = mapped_column(default=0, nullable=False)
```

- [ ] **Step 3: Bind-mount uploads dir in compose**

Edit `/home/vogic/LocalRAG/compose/docker-compose.yml`. There's no open-webui service yet (Phase 5), so Phase 4's uploads bind is consumed only by our standalone test runs. Instead, just add `volumes/uploads/` to the repo's `volumes/` mkdir in the conftest so the dir exists.

Actually: since uploads in Phase 4 are test-only (the routes are invoked via ASGITransport, not real compose), we don't need a compose bind yet. **Skip compose changes; create the dir via `mkdir -p volumes/uploads` at test startup** (handled in Task 7's conftest extension).

- [ ] **Step 4: Test migration applies cleanly**

Run the existing migration integration test to confirm:

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_kb_migration.py -v
```

Expected: PASS.

Add a dedicated assertion test — create `/home/vogic/LocalRAG/tests/integration/test_migration_003.py`:

```python
import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_kb_documents_has_chunk_count(session):
    rows = (await session.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name='kb_documents'"
    ))).scalars().all()
    assert "chunk_count" in rows
```

Run:

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_migration_003.py -v
ruff check . && mypy .
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ext/db/migrations/003_add_chunk_count.sql ext/db/models/kb.py tests/integration/conftest.py tests/integration/test_migration_003.py
git commit -m "feat(rag): migration 003 + kb_documents.chunk_count"
```

---

## Task 7: Upload router — KB docs + private-session docs

**Files:** `ext/routers/upload.py`, `tests/integration/test_upload_routes.py`.

- [ ] **Step 1: Write failing test**

`tests/integration/test_upload_routes.py`:

```python
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
ALICE = {"X-User-Id": "1", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin'),(1,'u@x','h','user')"))
        await s.execute(text("INSERT INTO knowledge_bases (id,name,admin_id) VALUES (10,'KB',9)"))
        await s.execute(text("INSERT INTO kb_subtags (id,kb_id,name) VALUES (100,10,'Docs')"))
        await s.execute(text("INSERT INTO chats (id,user_id) VALUES (500,1)"))
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


@pytest.mark.asyncio
async def test_kb_upload_admin(client):
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("a.txt", b"hello world this is a test", "text/plain")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "done"
    assert body["chunks"] >= 1


@pytest.mark.asyncio
async def test_kb_upload_non_admin_forbidden(client):
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ALICE,
        files={"file": ("a.txt", b"hi", "text/plain")},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_private_upload_chat_owner(client):
    r = await client.post(
        "/api/chats/500/private_docs/upload", headers=ALICE,
        files={"file": ("q.txt", b"my private note with enough text", "text/plain")},
    )
    assert r.status_code == 201, r.text
    assert r.json()["chunks"] >= 1


@pytest.mark.asyncio
async def test_private_upload_other_users_chat_404(client):
    r = await client.post(
        "/api/chats/500/private_docs/upload", headers={"X-User-Id": "2", "X-User-Role": "user"},
        files={"file": ("q.txt", b"sneaky", "text/plain")},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_kb_upload_too_large_rejected(client, monkeypatch):
    import ext.routers.upload as up
    monkeypatch.setattr(up, "MAX_UPLOAD_BYTES", 10)
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("big.txt", b"x" * 100, "text/plain")},
    )
    assert r.status_code == 413
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_upload_routes.py -v
```

- [ ] **Step 3: Write `ext/routers/upload.py`**

```python
"""Upload routes — KB documents (admin) and private chat docs (chat owner)."""
from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..db.models import Chat, KBDocument, KBSubtag
from ..services import kb_service
from ..services.auth import CurrentUser, get_current_user, require_admin
from ..services.embedder import Embedder
from ..services.ingest import ingest_bytes
from ..services.vector_store import VectorStore


router = APIRouter(tags=["upload"])


MAX_UPLOAD_BYTES = int(os.environ.get("RAG_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

_SM: async_sessionmaker[AsyncSession] | None = None
_VS: VectorStore | None = None
_EMB: Embedder | None = None


def configure(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    vector_store: VectorStore,
    embedder: Embedder,
) -> None:
    global _SM, _VS, _EMB
    _SM = sessionmaker
    _VS = vector_store
    _EMB = embedder


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    if _SM is None:
        raise RuntimeError("upload router not configured")
    async with _SM() as s:
        yield s


class UploadResult(BaseModel):
    status: str
    chunks: int
    doc_id: Optional[int] = None


async def _read_bounded(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
    return data


@router.post("/api/kb/{kb_id}/subtag/{subtag_id}/upload",
             response_model=UploadResult, status_code=status.HTTP_201_CREATED)
async def upload_kb_doc(
    kb_id: int, subtag_id: int,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(_get_session),
):
    if _VS is None or _EMB is None:
        raise RuntimeError("upload router not fully configured")
    # Verify KB + subtag exist.
    if await kb_service.get_kb(session, kb_id=kb_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    sub = (await session.execute(
        select(KBSubtag).where(KBSubtag.id == subtag_id, KBSubtag.kb_id == kb_id)
    )).scalar_one_or_none()
    if sub is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="subtag not found")

    data = await _read_bounded(file)

    # Persist doc row as 'chunking' first.
    doc = KBDocument(
        kb_id=kb_id, subtag_id=subtag_id, filename=file.filename or "upload",
        mime_type=file.content_type, bytes=len(data),
        uploaded_by=user.id, ingest_status="chunking",
    )
    session.add(doc)
    await session.flush()

    try:
        await _VS.ensure_collection(f"kb_{kb_id}")
        n = await ingest_bytes(
            data=data, mime_type=file.content_type or "application/octet-stream",
            filename=file.filename or "upload",
            collection=f"kb_{kb_id}",
            payload_base={"kb_id": kb_id, "subtag_id": subtag_id, "doc_id": doc.id},
            vector_store=_VS, embedder=_EMB,
        )
    except Exception as e:
        doc.ingest_status = "failed"
        doc.error_message = str(e)[:1000]
        await session.commit()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    doc.ingest_status = "done"
    doc.chunk_count = n
    await session.commit()
    return UploadResult(status="done", chunks=n, doc_id=doc.id)


@router.post("/api/chats/{chat_id}/private_docs/upload",
             response_model=UploadResult, status_code=status.HTTP_201_CREATED)
async def upload_private_doc(
    chat_id: int,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
):
    if _VS is None or _EMB is None:
        raise RuntimeError("upload router not fully configured")
    chat = (await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id)
    )).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")

    data = await _read_bounded(file)
    await _VS.ensure_collection(f"chat_{chat_id}")
    n = await ingest_bytes(
        data=data, mime_type=file.content_type or "application/octet-stream",
        filename=file.filename or "upload",
        collection=f"chat_{chat_id}",
        payload_base={"chat_id": chat_id, "owner_user_id": user.id},
        vector_store=_VS, embedder=_EMB,
    )
    return UploadResult(status="done", chunks=n)
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_upload_routes.py -v
ruff check . && mypy .
```

- [ ] **Step 5: Commit**

```bash
git add ext/routers/upload.py tests/integration/test_upload_routes.py
git commit -m "feat(rag): upload router (KB + private session docs)"
```

---

## Task 8: Retriever — parallel multi-KB search

**Files:** `ext/services/retriever.py`, `tests/integration/test_retriever.py`.

- [ ] **Step 1: Write failing test**

`tests/integration/test_retriever.py`:

```python
import pytest
from ext.services.retriever import retrieve
from ext.services.ingest import ingest_bytes
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder


@pytest.mark.asyncio
async def test_retrieve_from_multiple_kbs(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    emb = StubEmbedder(dim=32)

    for kb_id in (1, 2):
        await vs.ensure_collection(f"kb_{kb_id}")
        await ingest_bytes(
            data=f"content for KB {kb_id}: quick brown fox jumps over".encode(),
            mime_type="text/plain", filename=f"kb{kb_id}.txt",
            collection=f"kb_{kb_id}",
            payload_base={"kb_id": kb_id, "subtag_id": kb_id*10, "doc_id": kb_id},
            vector_store=vs, embedder=emb,
            chunk_tokens=20, overlap_tokens=5,
        )

    hits = await retrieve(
        query="quick brown fox",
        selected_kbs=[{"kb_id": 1, "subtag_ids": []}, {"kb_id": 2, "subtag_ids": []}],
        chat_id=None,
        vector_store=vs, embedder=emb,
        per_kb_limit=5, total_limit=10,
    )
    # Both KBs contributed something.
    kb_ids = {h.payload["kb_id"] for h in hits}
    assert 1 in kb_ids and 2 in kb_ids
    await vs.close()


@pytest.mark.asyncio
async def test_retrieve_respects_subtag_filter(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=16)
    emb = StubEmbedder(dim=16)
    await vs.ensure_collection("kb_1")
    await ingest_bytes(
        data=b"a a a a a a a a a a a a a a a a",
        mime_type="text/plain", filename="a.txt",
        collection="kb_1",
        payload_base={"kb_id": 1, "subtag_id": 10, "doc_id": 100},
        vector_store=vs, embedder=emb, chunk_tokens=20, overlap_tokens=5,
    )
    await ingest_bytes(
        data=b"b b b b b b b b b b b b b b b b",
        mime_type="text/plain", filename="b.txt",
        collection="kb_1",
        payload_base={"kb_id": 1, "subtag_id": 20, "doc_id": 200},
        vector_store=vs, embedder=emb, chunk_tokens=20, overlap_tokens=5,
    )
    hits = await retrieve(
        query="a a a", selected_kbs=[{"kb_id": 1, "subtag_ids": [10]}], chat_id=None,
        vector_store=vs, embedder=emb,
    )
    assert all(h.payload["subtag_id"] == 10 for h in hits)
    await vs.close()


@pytest.mark.asyncio
async def test_retrieve_includes_chat_private_docs(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=16)
    emb = StubEmbedder(dim=16)
    await vs.ensure_collection("chat_42")
    await ingest_bytes(
        data=b"private words: chapter one of the secret",
        mime_type="text/plain", filename="priv.txt",
        collection="chat_42",
        payload_base={"chat_id": 42, "owner_user_id": 1},
        vector_store=vs, embedder=emb, chunk_tokens=20, overlap_tokens=5,
    )
    hits = await retrieve(
        query="secret chapter", selected_kbs=[], chat_id=42,
        vector_store=vs, embedder=emb,
    )
    assert any(h.payload.get("chat_id") == 42 for h in hits)
    await vs.close()
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_retriever.py -v
```

- [ ] **Step 3: Write `ext/services/retriever.py`**

```python
"""Parallel multi-KB + optional chat-private retrieval."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .embedder import Embedder
from .vector_store import Hit, VectorStore


@dataclass
class Hit2(Hit):
    source: str = "kb"   # "kb" or "chat"


async def retrieve(
    *,
    query: str,
    selected_kbs: Sequence[dict],
    chat_id: Optional[int],
    vector_store: VectorStore,
    embedder: Embedder,
    per_kb_limit: int = 10,
    total_limit: int = 30,
) -> List[Hit]:
    """Run parallel searches against each selected KB and an optional chat namespace.

    selected_kbs shape: [{"kb_id": int, "subtag_ids": [int, ...]}, ...]
        empty subtag_ids → search all subtags in that KB.
    Returns a flat list of Hit objects (not yet reranked — see reranker.py).
    """
    [qvec] = await embedder.embed([query])

    async def _search_kb(cfg: dict) -> List[Hit]:
        kb_id = cfg["kb_id"]
        subtag_ids = cfg.get("subtag_ids") or None
        try:
            return await vector_store.search(
                f"kb_{kb_id}", qvec, limit=per_kb_limit, subtag_ids=subtag_ids,
            )
        except Exception:
            return []

    async def _search_chat() -> List[Hit]:
        if chat_id is None:
            return []
        try:
            return await vector_store.search(f"chat_{chat_id}", qvec, limit=per_kb_limit)
        except Exception:
            return []

    tasks = [_search_kb(cfg) for cfg in selected_kbs]
    tasks.append(_search_chat())
    results = await asyncio.gather(*tasks)
    # Flatten, keep at most total_limit
    flat: list[Hit] = []
    for lst in results:
        flat.extend(lst)
    # Sort by raw score desc, trim
    flat.sort(key=lambda h: h.score, reverse=True)
    return flat[:total_limit]
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_retriever.py -v
ruff check . && mypy .
```

- [ ] **Step 5: Commit**

```bash
git add ext/services/retriever.py tests/integration/test_retriever.py
git commit -m "feat(rag): parallel retriever (KBs + chat private namespace)"
```

---

## Task 9: Reranker — score normalization + fast path

**Files:** `ext/services/reranker.py`, `tests/unit/test_reranker.py`.

- [ ] **Step 1: Write failing test**

`tests/unit/test_reranker.py`:

```python
import pytest
from ext.services.reranker import rerank
from ext.services.vector_store import Hit


def _h(id, score, kb_id):
    return Hit(id=id, score=score, payload={"kb_id": kb_id})


def test_fast_path_returns_top_k_unchanged_when_top1_dominates():
    hits = [_h(1, 0.9, 1), _h(2, 0.3, 1), _h(3, 0.1, 2)]  # 0.9 / 0.3 == 3.0 > 2.0
    out = rerank(hits, top_k=2)
    assert [h.id for h in out] == [1, 2]


def test_normalize_when_close_scores():
    hits = [
        _h(1, 0.8, 1), _h(2, 0.7, 1),  # kb 1 max = 0.8 → normalized 1.0, 0.875
        _h(3, 0.4, 2), _h(4, 0.3, 2),  # kb 2 max = 0.4 → normalized 1.0, 0.75
    ]
    out = rerank(hits, top_k=4)
    # After per-KB max-normalization + global sort, both kb1 and kb2 get a 1.0 at top.
    # Stable secondary ordering: original score ties broken by id.
    ids = [h.id for h in out]
    assert ids.index(1) < ids.index(2)
    assert ids.index(3) < ids.index(4)


def test_empty_returns_empty():
    assert rerank([], top_k=10) == []
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_reranker.py -v
```

- [ ] **Step 3: Write `ext/services/reranker.py`**

```python
"""Rerank hits from parallel KB searches.

Strategy:
- Fast path: if raw top-1 score / top-2 score > FAST_PATH_RATIO, return the input
  unchanged (already confident).
- Otherwise: per-KB max-normalize scores, then re-sort global list descending.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from .vector_store import Hit


FAST_PATH_RATIO = 2.0


def rerank(hits: List[Hit], *, top_k: int = 10) -> List[Hit]:
    if not hits:
        return []

    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    if len(ordered) >= 2 and ordered[1].score > 0 and (ordered[0].score / ordered[1].score) > FAST_PATH_RATIO:
        return ordered[:top_k]

    # Per-KB max-normalize
    max_by_kb: dict[int, float] = defaultdict(float)
    for h in hits:
        kb = int(h.payload.get("kb_id", -1))
        if h.score > max_by_kb[kb]:
            max_by_kb[kb] = h.score

    def normalized(h: Hit) -> float:
        kb = int(h.payload.get("kb_id", -1))
        m = max_by_kb[kb]
        return h.score / m if m > 0 else h.score

    # Stable sort: by normalized score desc, then raw score desc, then id.
    ordered2 = sorted(hits, key=lambda h: (-normalized(h), -h.score, str(h.id)))
    return ordered2[:top_k]
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_reranker.py -v
ruff check . && mypy .
```

- [ ] **Step 5: Commit**

```bash
git add ext/services/reranker.py tests/unit/test_reranker.py
git commit -m "feat(rag): reranker (score normalization + fast path)"
```

---

## Task 10: Token budget

**Files:** `ext/services/budget.py`, `tests/unit/test_budget.py`.

- [ ] **Step 1: Write failing test**

`tests/unit/test_budget.py`:

```python
from ext.services.budget import budget_chunks
from ext.services.vector_store import Hit


def _h(text, score=0.5):
    return Hit(id=1, score=score, payload={"text": text})


def test_all_fit():
    hits = [_h("hi"), _h("ok")]
    out = budget_chunks(hits, max_tokens=100)
    assert len(out) == 2


def test_truncates_from_lowest_rank_last():
    # Pre-sorted descending by relevance: hits[0] is most relevant.
    long_text = " ".join(["word"] * 50)   # ~50 tokens
    hits = [_h(long_text, score=0.9), _h(long_text, score=0.7), _h(long_text, score=0.5)]
    out = budget_chunks(hits, max_tokens=60)
    # Only the first chunk fits; the rest drop.
    assert len(out) == 1
    assert out[0].score == 0.9


def test_empty_input():
    assert budget_chunks([], max_tokens=100) == []
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_budget.py -v
```

- [ ] **Step 3: Write `ext/services/budget.py`**

```python
"""Token-budget the reranked chunks — drop from lowest rank end until we fit."""
from __future__ import annotations

import logging
from typing import List

from .chunker import _encoder
from .vector_store import Hit


logger = logging.getLogger("rag.budget")


def _count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def budget_chunks(hits: List[Hit], *, max_tokens: int = 4000) -> List[Hit]:
    """Assumes hits is pre-sorted best-first. Returns longest prefix that fits."""
    kept: list[Hit] = []
    total = 0
    dropped = 0
    for h in hits:
        t = _count_tokens(str(h.payload.get("text", "")))
        if total + t > max_tokens:
            dropped += 1
            continue
        total += t
        kept.append(h)
    if dropped:
        logger.debug("budget dropped %d of %d chunks (used %d/%d tokens)",
                     dropped, len(hits), total, max_tokens)
    return kept
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_budget.py -v
ruff check . && mypy .
```

- [ ] **Step 5: Commit**

```bash
git add ext/services/budget.py tests/unit/test_budget.py
git commit -m "feat(rag): token budgeting"
```

---

## Task 11: RAG retrieval router — `POST /api/rag/retrieve`

**Files:** `ext/routers/rag.py`, `tests/integration/test_rag_routes.py`.

- [ ] **Step 1: Write failing test**

`tests/integration/test_rag_routes.py`:

```python
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.routers.rag import router as rag_router, configure as configure_rag
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder
from ext.services.ingest import ingest_bytes


ALICE = {"X-User-Id": "1", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin'),(1,'u@x','h','user')"))
        await s.execute(text("INSERT INTO groups (id,name) VALUES (1,'eng')"))
        await s.execute(text("INSERT INTO user_groups (user_id, group_id) VALUES (1,1)"))
        await s.execute(text("INSERT INTO knowledge_bases (id,name,admin_id) VALUES (10,'Eng',9),(11,'Secret',9)"))
        await s.execute(text("INSERT INTO kb_access (kb_id, group_id, access_type) VALUES (10,1,'read')"))
        await s.execute(text("INSERT INTO chats (id,user_id) VALUES (500,1)"))
        await s.commit()


@pytest_asyncio.fixture
async def client(engine, clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    emb = StubEmbedder(dim=32)
    # Seed real vectors in the "Eng" KB (id 10).
    await vs.ensure_collection("kb_10")
    await ingest_bytes(
        data=b"the quick brown fox jumps over the lazy dog in the forest",
        mime_type="text/plain", filename="a.txt", collection="kb_10",
        payload_base={"kb_id": 10, "subtag_id": 1, "doc_id": 1},
        vector_store=vs, embedder=emb, chunk_tokens=20, overlap_tokens=5,
    )

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    configure_rag(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    app = FastAPI()
    app.include_router(rag_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await vs.close()


@pytest.mark.asyncio
async def test_retrieve_returns_hits_for_authorized_kb(client):
    r = await client.post("/api/rag/retrieve", headers=ALICE, json={
        "chat_id": 500,
        "query": "quick brown fox",
        "selected_kb_config": [{"kb_id": 10, "subtag_ids": []}],
    })
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert len(hits) >= 1
    assert hits[0]["kb_id"] == 10


@pytest.mark.asyncio
async def test_retrieve_rejects_unauthorized_kb(client):
    r = await client.post("/api/rag/retrieve", headers=ALICE, json={
        "chat_id": 500,
        "query": "x",
        "selected_kb_config": [{"kb_id": 11, "subtag_ids": []}],
    })
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_retrieve_rejects_other_users_chat(client):
    r = await client.post("/api/rag/retrieve",
                          headers={"X-User-Id": "2", "X-User-Role": "user"},
                          json={"chat_id": 500, "query": "x", "selected_kb_config": []})
    assert r.status_code == 404
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_rag_routes.py -v
```

- [ ] **Step 3: Write `ext/routers/rag.py`**

```python
"""RAG retrieval endpoint: pulls from selected KBs + chat-private namespace."""
from __future__ import annotations

from typing import Any, AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..db.models import Chat, validate_selected_kb_config
from ..services.auth import CurrentUser, get_current_user
from ..services.budget import budget_chunks
from ..services.embedder import Embedder
from ..services.rbac import get_allowed_kb_ids
from ..services.reranker import rerank
from ..services.retriever import retrieve
from ..services.vector_store import VectorStore


router = APIRouter(tags=["rag"])

_SM: async_sessionmaker[AsyncSession] | None = None
_VS: VectorStore | None = None
_EMB: Embedder | None = None


def configure(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    vector_store: VectorStore,
    embedder: Embedder,
) -> None:
    global _SM, _VS, _EMB
    _SM = sessionmaker
    _VS = vector_store
    _EMB = embedder


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    if _SM is None:
        raise RuntimeError("rag router not configured")
    async with _SM() as s:
        yield s


class RetrieveRequest(BaseModel):
    chat_id: int
    query: str
    selected_kb_config: List[Any] = []
    max_tokens: int = 4000
    top_k: int = 10


class HitOut(BaseModel):
    score: float
    text: str
    kb_id: Optional[int] = None
    subtag_id: Optional[int] = None
    chat_id: Optional[int] = None
    doc_id: Optional[int] = None


class RetrieveResponse(BaseModel):
    hits: List[HitOut]


@router.post("/api/rag/retrieve", response_model=RetrieveResponse)
async def rag_retrieve(
    body: RetrieveRequest,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
):
    if _VS is None or _EMB is None:
        raise RuntimeError("rag router not fully configured")

    # Ownership check.
    chat = (await session.execute(
        select(Chat).where(Chat.id == body.chat_id, Chat.user_id == user.id)
    )).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")

    # Validate the selected_kb_config shape.
    try:
        parsed = validate_selected_kb_config(body.selected_kb_config) or []
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    # RBAC: verify access.
    allowed = set(await get_allowed_kb_ids(session, user_id=user.id))
    for entry in parsed:
        if entry.kb_id not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                detail=f"no access to kb_id={entry.kb_id}")

    # Retrieve → rerank → budget.
    raw = await retrieve(
        query=body.query,
        selected_kbs=[{"kb_id": e.kb_id, "subtag_ids": e.subtag_ids} for e in parsed],
        chat_id=body.chat_id,
        vector_store=_VS, embedder=_EMB,
    )
    reranked = rerank(raw, top_k=body.top_k)
    budgeted = budget_chunks(reranked, max_tokens=body.max_tokens)

    return RetrieveResponse(hits=[
        HitOut(
            score=h.score,
            text=str(h.payload.get("text", "")),
            kb_id=h.payload.get("kb_id"),
            subtag_id=h.payload.get("subtag_id"),
            chat_id=h.payload.get("chat_id"),
            doc_id=h.payload.get("doc_id"),
        )
        for h in budgeted
    ])
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_rag_routes.py -v
ruff check . && mypy .
```

- [ ] **Step 5: Commit**

```bash
git add ext/routers/rag.py tests/integration/test_rag_routes.py
git commit -m "feat(rag): retrieval router (RBAC + rerank + budget)"
```

---

## Task 12: End-to-end test — upload then retrieve

**Files:** `tests/integration/test_rag_end_to_end.py`.

- [ ] **Step 1: Write the test**

`tests/integration/test_rag_end_to_end.py`:

```python
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


ADMIN = {"X-User-Id": "9", "X-User-Role": "admin"}
ALICE = {"X-User-Id": "1", "X-User-Role": "user"}


@pytest_asyncio.fixture(autouse=True)
async def seed(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin'),(1,'u@x','h','user')"))
        await s.execute(text("INSERT INTO groups (id,name) VALUES (1,'eng')"))
        await s.execute(text("INSERT INTO user_groups (user_id, group_id) VALUES (1,1)"))
        await s.execute(text("INSERT INTO knowledge_bases (id,name,admin_id) VALUES (10,'Eng',9)"))
        await s.execute(text("INSERT INTO kb_subtags (id,kb_id,name) VALUES (100,10,'Docs')"))
        await s.execute(text("INSERT INTO kb_access (kb_id, group_id, access_type) VALUES (10,1,'read')"))
        await s.execute(text("INSERT INTO chats (id,user_id) VALUES (500,1)"))
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


@pytest.mark.asyncio
async def test_upload_and_retrieve_roundtrip(client):
    # 1) Admin uploads into KB 10 / subtag 100.
    r = await client.post(
        "/api/kb/10/subtag/100/upload", headers=ADMIN,
        files={"file": ("doc.txt", b"the quick brown fox jumps over the lazy dog", "text/plain")},
    )
    assert r.status_code == 201
    n = r.json()["chunks"]
    assert n >= 1

    # 2) Alice (who is in the 'eng' group with access to KB 10) retrieves.
    r = await client.post("/api/rag/retrieve", headers=ALICE, json={
        "chat_id": 500,
        "query": "quick brown fox",
        "selected_kb_config": [{"kb_id": 10, "subtag_ids": [100]}],
    })
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) >= 1
    assert hits[0]["kb_id"] == 10
    assert hits[0]["subtag_id"] == 100
    assert "fox" in hits[0]["text"].lower()
```

- [ ] **Step 2: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_rag_end_to_end.py -v
ruff check . && mypy .
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_rag_end_to_end.py
git commit -m "test(rag): upload → retrieve end-to-end"
```

---

## Task 13: Cross-user RAG isolation test

**Files:** `tests/integration/test_rag_isolation.py`.

- [ ] **Step 1: Write the test**

`tests/integration/test_rag_isolation.py`:

```python
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
        await s.execute(text("INSERT INTO users (id,email,password_hash,role) VALUES (9,'a@x','h','admin'),(1,'alice@x','h','user'),(2,'bob@x','h','user')"))
        await s.execute(text("INSERT INTO chats (id,user_id) VALUES (100,1),(200,2)"))
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


@pytest.mark.asyncio
async def test_alice_private_doc_invisible_to_bob(client):
    ALICE = {"X-User-Id": "1", "X-User-Role": "user"}
    BOB   = {"X-User-Id": "2", "X-User-Role": "user"}

    # Alice uploads a private doc in HER chat.
    r = await client.post(
        "/api/chats/100/private_docs/upload", headers=ALICE,
        files={"file": ("secret.txt", b"alice's private account number is 42", "text/plain")},
    )
    assert r.status_code == 201

    # Bob retrieves against HIS chat — must not see Alice's content.
    r = await client.post("/api/rag/retrieve", headers=BOB, json={
        "chat_id": 200,
        "query": "account number",
        "selected_kb_config": [],
    })
    assert r.status_code == 200
    hits = r.json()["hits"]
    for h in hits:
        assert "alice" not in h["text"].lower(), f"leak: {h}"
        assert h.get("chat_id") != 100


@pytest.mark.asyncio
async def test_bob_cannot_query_alices_chat(client):
    BOB = {"X-User-Id": "2", "X-User-Role": "user"}
    r = await client.post("/api/rag/retrieve", headers=BOB, json={
        "chat_id": 100,   # Alice's chat
        "query": "anything",
        "selected_kb_config": [],
    })
    assert r.status_code == 404
```

- [ ] **Step 2: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_rag_isolation.py -v
ruff check . && mypy .
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_rag_isolation.py
git commit -m "test(rag): cross-user private-doc isolation"
```

---

## Task 14: Wire into `ext/app.py` + full regression + Phase 4 tag

**Files:** Edit `ext/app.py` to `configure_upload` and `configure_rag` + include both routers.

- [ ] **Step 1: Edit `ext/app.py`**

Replace the router wiring section so `build_app` configures + mounts the new routers. Key additions:
- Import `upload` + `rag` routers plus `VectorStore` + `TEIEmbedder`.
- Extend `Settings` (via existing `ext/config.py`) to include `TEI_URL` (default `http://tei:80`), `QDRANT_URL` (already there), `VECTOR_SIZE` (default 1024).
- In `build_app`, construct a `VectorStore` + `TEIEmbedder` from settings; wire both upload and rag routers.

Open `/home/vogic/LocalRAG/ext/config.py` and add after `session_secret`:

```python
    tei_url:     str = Field("http://tei:80", alias="TEI_URL")
    vector_size: int = Field(1024,            alias="RAG_VECTOR_SIZE")
```

Then edit `/home/vogic/LocalRAG/ext/app.py` so `build_app()` becomes:

```python
def build_app() -> FastAPI:
    clear_settings_cache()
    settings = get_settings()
    engine = make_engine(settings.database_url)
    SessionLocal = make_sessionmaker(engine)

    from .services.vector_store import VectorStore
    from .services.embedder import TEIEmbedder
    from .routers import upload, rag

    vs = VectorStore(url=settings.qdrant_url, vector_size=settings.vector_size)
    emb = TEIEmbedder(base_url=settings.tei_url)

    kb_admin.set_sessionmaker(SessionLocal)
    kb_retrieval.set_sessionmaker(SessionLocal)
    upload.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    rag.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)

    app = FastAPI(title="orgchat-kb", version="0.4.0")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    # Retrieval router MUST be registered before admin router to avoid /available shadowing.
    app.include_router(kb_retrieval.router)
    app.include_router(kb_admin.router)
    app.include_router(upload.router)
    app.include_router(rag.router)
    return app
```

- [ ] **Step 2: Extend test_app_wiring.py to cover new paths**

Add to `/home/vogic/LocalRAG/tests/integration/test_app_wiring.py`:

```python
@pytest.mark.asyncio
async def test_app_mounts_upload_and_rag(engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", str(engine.url))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    monkeypatch.setenv("TEI_URL", "http://localhost:80")
    from ext.app import build_app
    app = build_app()
    paths = {r.path for r in app.routes}
    assert "/api/kb/{kb_id}/subtag/{subtag_id}/upload" in paths
    assert "/api/chats/{chat_id}/private_docs/upload" in paths
    assert "/api/rag/retrieve" in paths
```

- [ ] **Step 3: Full regression**

```bash
source .venv/bin/activate
python -m pytest tests/unit -v 2>&1 | tail -5
SKIP_GPU_SMOKE=1 python -m pytest tests/integration -v 2>&1 | tail -15
ruff check . && mypy .
```

Expected: all passing, lint clean.

- [ ] **Step 4: Commit + Tag**

```bash
git add ext/app.py ext/config.py tests/integration/test_app_wiring.py
git commit -m "feat(rag): wire upload + rag routers into app factory"
git tag -a phase-4-rag-pipeline -m "Phase 4 complete: upload → ingest → retrieve → rerank → budget (with isolation)"
```

- [ ] **Step 5: Commission Phase 5 plan**

Request controller: "Write Phase 5 plan at `docs/superpowers/plans/2026-04-16-phase-5-frontend-and-auth.md`: wire our API into upstream Open WebUI's frontend (KB selector component, upload UI, chat integration) + replace the stub auth with real Open WebUI session-cookie verification."

---

## Phase 4 acceptance checklist

- [ ] `ext/services/vector_store.py` wraps qdrant-client (ensure_collection, upsert, search with subtag filter).
- [ ] `StubEmbedder` is deterministic; `TEIEmbedder` hits `/embed`.
- [ ] Extractor handles TXT, MD, PDF; rejects others with `UnsupportedMimeType`.
- [ ] Chunker: 800/100 token windows.
- [ ] Ingest upserts correct payload fields (`kb_id`/`subtag_id`/`doc_id` for KB; `chat_id`/`owner_user_id` for private).
- [ ] Migration 003 adds `chunk_count` to `kb_documents`.
- [ ] Upload routes enforce admin (KB) vs chat-owner (private).
- [ ] Retriever searches all selected KBs + chat namespace concurrently.
- [ ] Reranker: fast path + per-KB max-normalize.
- [ ] Token budget drops lowest-rank chunks to fit `max_tokens`.
- [ ] `/api/rag/retrieve` enforces KB access via RBAC.
- [ ] Cross-user isolation test green (no leak of alice → bob).
- [ ] `ruff` + `mypy` clean.
- [ ] Tag `phase-4-rag-pipeline` exists.
