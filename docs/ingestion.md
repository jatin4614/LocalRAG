# LocalRAG Ingestion Pipeline — Exhaustive Reference

**Audience:** Operators who need to debug a stuck ingest, tune per-KB settings, or understand exactly what happens between `POST /api/kb/{id}/subtag/{sid}/upload` and a chunk appearing in Qdrant.

**Source of truth for every claim:** `ext/`, `compose/`, `scripts/`. When this doc and the code disagree, the code wins.

---

## 1. Big Picture

Ingestion transforms raw document bytes into a set of dense+sparse+ColBERT vector points in Qdrant plus summary metadata in Postgres. The full pipeline is: **extract** structured text blocks from the uploaded file → **coalesce** small prose blocks to prevent chunk explosion → optionally **OCR** low-text PDF pages → **chunk** blocks into token-bounded windows → optionally **contextualize** chunks with a short LLM prefix → optionally expand into a **RAPTOR** tree or **doc summary** → **embed** in three parallel arms (TEI dense, fastembed BM25 sparse, fastembed ColBERT) → **upsert** deterministic UUIDv5 points into Qdrant → **write status** back to `kb_documents.ingest_status` in Postgres.

**Async path (default, `RAG_SYNC_INGEST=0`):** The FastAPI upload handler (`ext/routers/upload.py`) reads the file bytes, writes them atomically to `/var/ingest/{sha}` via the `BlobStore`, stamps `kb_documents.ingest_status='queued'`, and enqueues a Celery task (`ext/workers/ingest_worker.py:ingest_blob`). The Celery worker runs the full pipeline in a fresh `asyncio.run()` scope and writes the final status. The HTTP response returns `{status: "queued", task_id: "..."}` immediately — the caller must poll `GET /api/kb/{id}/ingest-stream` or watch the admin UI for completion.

**Sync path (`RAG_SYNC_INGEST=1`):** `ingest_bytes` runs in-process in the FastAPI worker. The HTTP response blocks until ingest completes (or fails). Suitable for small corpora or testing; not recommended under load because long-running ingest holds a FastAPI worker thread.

**Status ladder:** `pending → queued → chunking → embedding → done | failed`. This is a `CHECK` constraint enforced in DB migration `012_add_kb_documents_queued_status.sql`. The sequence: upload handler stamps `pending`, then `queued` after enqueue (async) or `done`/`failed` after completion (sync). The Celery worker stamps `chunking` when it starts extraction, `embedding` just before the embed+upsert phase, and `done` with `chunk_count` on success or `failed` with `error_message` on exhausted retries.

---

## 2. Flow Diagram

```mermaid
flowchart TD
    A[HTTP POST /api/kb/{id}/subtag/{sid}/upload] --> B{RAG_SYNC_INGEST?}

    B -- "=1 sync" --> S1[ingest_bytes in-process]
    B -- "=0 async default" --> C1[BlobStore.write to /var/ingest/sha]
    C1 --> C2[kb_documents.ingest_status = queued]
    C2 --> C3[ingest_blob.apply_async → Celery queue]
    C3 --> C4[ingest_blob picks up task]
    C4 --> C5[status = chunking]
    C5 --> S1

    S1 --> E1["extract(data, mime, filename)\next/services/extractor.py"]
    E1 -- "no blocks" --> DONE0[return 0]
    E1 -- "blocks" --> E2["_coalesce_small_blocks(blocks)\next/services/ingest.py:402\nRAG_INGEST_BLOCK_MIN_TOKENS=200\nRAG_INGEST_BLOCK_MAX_TOKENS=600"]

    E2 --> OCR{"RAG_OCR_ENABLED=1\n+ per-KB ocr_policy.enabled?"}
    OCR -- "no" --> CHUNK
    OCR -- "yes (PDF only)" --> OCR2["extract_pdf_with_ocr_fallback()\next/services/ingest.py:1166\nocr_policy.backend: tesseract/textract/document_ai"]
    OCR2 --> CHUNK

    CHUNK["chunk_text_for_kb() per block\next/services/chunker.py\nRAG_STRUCTURED_CHUNKER + rag_config.chunking_strategy\nDefault: window 800/100 tokens"] --> CTX

    CTX{"RAG_CONTEXTUALIZE_KBS=1\nor rag_config.contextualize=true?"} -- "yes" --> CTX2["contextualize_batch()\next/services/contextualizer.py\nconcurrency=8 RAG_CONTEXTUALIZE_CONCURRENCY\npipeline_version→ctx=contextual-v1"]
    CTX -- "no" --> RAPTOR
    CTX2 --> RAPTOR

    RAPTOR{"RAG_RAPTOR=1\nor temporal_raptor?"} -- "yes" --> RAPTOR2["build_tree()\next/services/raptor.py\nor temporal_raptor.py L0→L4\nRAG_RAPTOR_MAX_LEVELS / CLUSTER_MIN"]
    RAPTOR -- "no" --> DOCSUM
    RAPTOR2 --> DOCSUM

    DOCSUM{"RAG_DOC_SUMMARIES=1\n+ doc_id present?"} -- "yes" --> DOCSUM2["_emit_doc_summary_point()\next/services/ingest.py:1045\nsummarize→embed→upsert level=doc\nmirror to kb_documents.doc_summary"]
    DOCSUM -- "no" --> EMBED
    DOCSUM2 --> EMBED

    C5 --> EMBED2[status = embedding]
    EMBED2 --> EMBED

    EMBED["asyncio.gather — 3 parallel arms\next/services/ingest.py:756"] --> DENSE["TEIEmbedder.embed(texts)\next/services/embedder.py\nbge-m3 1024-d\nRAG_TEI_MAX_BATCH=32\nretry-with-halving on 424/429/5xx"]
    EMBED --> SPARSE["embed_sparse(texts) via asyncio.to_thread\next/services/sparse_embedder.py\nQdrant/bm25 fastembed\nRAG_HYBRID=1 + collection has bm25 slot"]
    EMBED --> COLBERT["colbert_embed(texts) via asyncio.to_thread\next/services/embedder.py\n128-d fastembed ColBERT\nRAG_COLBERT=1 + collection has colbert slot"]

    DENSE --> UPSERT
    SPARSE --> UPSERT
    COLBERT --> UPSERT

    UPSERT["vector_store.upsert() or upsert_temporal()\next/services/vector_store.py\nUUIDv5 deterministic IDs\nRAG_SHARDING_ENABLED → shard_key from temporal_shard.py"] --> STATUS

    STATUS["_update_doc_status(done, chunk_count=N)\next/workers/ingest_worker.py:124\nNullPool asyncpg — one fresh conn per call\ningest_status_update_failed_total{stage} on error"] --> BLOBGC["BlobStore.delete(sha)\n/var/ingest cleanup"]

    E1 -- "UnsupportedMimeType" --> FAIL
    CHUNK -- "empty after chunk" --> DONE0
    EMBED -- "TEI 424 exhausted at batch=1" --> FAIL
    UPSERT -- "Qdrant 5xx" --> FAIL
    FAIL["status=failed + error_message\nDLQ after 3 retries\ncelery_dlq_depth alert"] --> END
    DONE0 --> END[return chunk count]
    STATUS --> END
```

---

## 3. Per-Stage Detail

### 3.1 Upload Entry Point

**File:line:** `ext/routers/upload.py:152` (KB upload) and `:333` (private chat upload).

**What it does:** The FastAPI handler reads the uploaded file (streaming in 1 MB chunks up to `RAG_MAX_UPLOAD_BYTES=50MB`), creates a `KBDocument` row in Postgres, determines whether to run sync or async ingest, and returns an `UploadResult`.

**Inputs / outputs:**
- Input: multipart `UploadFile` with `Content-Type`, `kb_id`, `subtag_id`, JWT user claim.
- Output: HTTP 201 `{status, chunks, doc_id, task_id, sha}`.

**Gating:**
- `RAG_MAX_UPLOAD_BYTES` (default 50 MB); raises HTTP 413 if exceeded (`upload.py:135`).
- `RAG_SYNC_INGEST` read at module import time (`upload.py:81`); changing it requires a FastAPI process restart.
- `RAG_COLBERT=1` and `RAG_SHARDING_ENABLED=1` are checked when creating the Qdrant collection (`_ensure_kb_collection`, `upload.py:41`).

**Defaults:** Async path. Collection created with sparse support always; ColBERT and temporal sharding opt-in.

**Failure mode:** Duplicate sha raises HTTP 409 (checked against `uniq_kb_doc_blob_sha` partial index from migration 016). On sync ingest exception, stamps `ingest_status='failed'` + `error_message` and raises HTTP 422. On async enqueue failure (Redis down), the DB row is left as `pending` and must be manually re-queued.

**Performance:** File read is bounded by `RAG_MAX_UPLOAD_BYTES`. Blob write to `/var/ingest` is an fsync'd atomic write (`blob_store.py:47`); on a local SSD this is <100ms for a 50 MB file. Collection creation (`ensure_collection_temporal`) is idempotent and cached in `_known`.

**What can go wrong:**
- `RAG_SYNC_INGEST` is read at module import — environment change after process start has no effect until restart.
- `/var/ingest` must be writable by the `open-webui` container and readable by `celery-worker`. After the UID=1000 image bake (see §5), the volume may be root-owned from initial creation. Run `chown -R 1000:1000 /var/ingest` once as described in CLAUDE.md §8.

**Observability:**
- `rag_upload_bytes_total{kb=<kb_id>}` — incremented at `upload.py:188`.
- OTel span `upload.request` wrapping the whole handler; child spans `upload.read_bounded`, `upload.blob_write`, `upload.enqueue_celery`, `upload.ensure_collection`.

---

### 3.2 Celery Task Hand-off

**File:line:** `ext/workers/ingest_worker.py:288` — `ingest_blob` Celery task; `ext/workers/ingest_worker.py:124` — `_update_doc_status`.

**What it does:** `ingest_blob` is the Celery task that bridges the producer (FastAPI) and the ingest logic. It reads the blob by sha, updates status to `chunking`, calls `_do_ingest`, updates status to `done` on success or `failed` after exhausted retries, then deletes the blob.

**Inputs / outputs:**
- Input: `(sha, mime_type, filename, collection, payload_base)`. `payload_base` may contain `_chunk_tokens` / `_overlap_tokens` stashed by the upload route (popped before passing downstream).
- Output: `{status: "ok", chunks: N, sha: sha}`.

**Status transitions stamped by this task:**
- `queued → chunking` at task start (`ingest_worker.py:380`).
- `chunking → embedding` just before `_do_ingest` call (`ingest_worker.py:393`).
- `embedding → done` on success (`ingest_worker.py:434`).
- Any state → `failed` after exhausted retries (`ingest_worker.py:410`).

**The asyncpg / NullPool fix (commit `ebe4fee`):** Celery prefork workers run each task inside a fresh `asyncio.run()` call — a new event loop per task. SQLAlchemy's default `QueuePool` cached asyncpg connections bound to the loop alive at pool creation time. The second task in the same worker process tripped `RuntimeError: Event loop is closed` inside `_update_doc_status`. The error was swallowed by the broad `except Exception` that already existed to make status updates best-effort, so Qdrant ingest completed successfully but `ingest_status` was stuck at `chunking` forever. Fix: the singleton engine is created with `poolclass=NullPool` (`ingest_worker.py:120`). NullPool opens a fresh asyncpg connection per `engine.begin()` call (~5-10ms per status write, ~4 writes total per ingest — acceptable given the multi-second embedding time).

**Failure mode:**
- Blob missing: `Reject(requeue=False)` → DLQ, no retry. Log line: `"ingest: blob missing sha=..."`.
- Any `Exception` during `_do_ingest`: retry up to 3 times with exponential backoff (1s, 2s, 4s). After exhausted retries: `Reject(requeue=False)` → DLQ.
- Status write failure: logged at `ERROR` level + `ingest_status_update_failed_total{stage}` incremented. Ingest data in Qdrant is unaffected.

**Performance:** Task overhead (blob read, status writes) is ~50-100ms. The bulk of time is `_do_ingest` (see per-stage latency below).

**What can go wrong:**
- `DATABASE_URL` not set or wrong dialect: `_get_engine()` returns `None` and status writes are silently skipped.
- Celery worker OOM during ColBERT embed (fastembed + large batch): task crashes, retried, may DLQ. Inspect `docker compose logs celery-worker`.
- `INGEST_BLOB_ROOT` mismatch between producer and consumer: blob exists on the wrong volume. Default is `/var/ingest` in both; check that the named volume `ingest_blobs` is mounted to the same path in both services.

**Observability:**
- `ingest_status_update_failed_total{stage}` — `ext/services/metrics.py:681`.
- `IngestStatusUpdateFailing` Prometheus alert — `observability/prometheus/alerts-celery.yml:37`.
- OTel span `ingest.celery_task` with `doc_id`, `kb_id`, `sha`, `collection` attributes.
- SSE progress events via `ingest_progress.emit_sync`: `{stage: processing}` at start, `{stage: done, chunks: N}` or `{stage: failed}` at end. Consumed by `GET /api/kb/{id}/ingest-stream`.

---

### 3.3 Extract

**File:line:** `ext/services/extractor.py:1` — module; `:211` — `_blocks_txt`; `:226` — `_blocks_pdf`; `:311` — `_blocks_docx`; `:362` — `_blocks_xlsx`; `:383` — `_blocks_pptx`.

**What it does:** Converts raw bytes into a list of `ExtractedBlock` objects, each carrying `text`, optional `page` (1-based PDF/PPTX page number), `heading_path` (DOCX/MD heading stack), `sheet` (XLSX worksheet name), and `kind` ("prose" or "table").

**`ExtractedBlock` shape** (`extractor.py:39`):
```python
@dataclass
class ExtractedBlock:
    text: str
    page: Optional[int] = None
    heading_path: list[str] = field(default_factory=list)
    sheet: Optional[str] = None
    kind: str = "prose"   # "prose" | "table"
```

**Inputs / outputs:**
- Input: `(data: bytes, mime_type: str, filename: str)`.
- Output: `list[ExtractedBlock]`. Empty list if no text extracted.

**Per-format walkers:**
- **PDF** (`_blocks_pdf`, `:226`): Prefers `pymupdf` (de-duplicates overlay/watermark text via `_dedup_overlay_text`; 58% duplicate reduction observed on stamped PDFs), falls back to `pypdf`. Emits one block per page with `page=i+1`. `RAG_PDF_DEDUP=0` disables overlay dedup.
- **DOCX** (`_blocks_docx`, `:311`): Walks `doc.paragraphs` tracking heading level via `style.name == "Heading N"`. Headings update `heading_stack` but are NOT emitted as blocks. Prose paragraphs carry the current `heading_path`. Tables are emitted as TSV rows with `kind="table"`. Note: `python-docx` does not give a single ordered iterator over paragraphs and tables interleaved — tables are appended after all paragraphs.
- **XLSX** (`_blocks_xlsx`, `:362`): One `kind="table"` block per worksheet with `sheet=ws.title`. Empty sheets are skipped.
- **PPTX** (`_blocks_pptx`, `:383`): One block per slide; `page=slide_idx`; title becomes `heading_path`.
- **MD/TXT** (`_blocks_txt`, `:211`): Single block. No heading parsing (heading-aware chunking is the structured chunker's job).

**Failure mode:** `UnsupportedMimeType` raises for `.doc` files (`:195`). Other extraction errors propagate up and fail the ingest task (retried by Celery). Import failures (missing `pymupdf`) fall back to `pypdf` automatically.

**Performance:** PDF extraction is CPU-bound, typically 100-500ms per MB on the Celery worker. DOCX/XLSX are faster (~10-50ms). No GPU involved.

**What can go wrong:**
- `.doc` files raise immediately — user must convert to `.docx` first.
- Stamped/watermarked PDFs produce duplicate content when `pymupdf` is not installed (falls back to `pypdf` which includes watermarks verbatim).
- DOCX tables are emitted after paragraphs, not interleaved — if document order matters for your corpus, note that table content will follow all prose.

**Observability:** No dedicated metric. Errors are logged as `WARNING` by the Celery task's retry handler.

---

### 3.4 Block Coalesce

**File:line:** `ext/services/ingest.py:402` — `_coalesce_small_blocks`; `:376` — `_coalesce_block_min_tokens`; `:391` — `_coalesce_block_max_tokens`.

**What it does:** Merges adjacent small prose blocks into larger blocks before the chunking pass. Without this, DOCX prose paragraphs (median ~30 tokens in the 2026 corpus) each enter the 800-token window chunker as individual inputs, each producing one tiny chunk. The fix was introduced after `Apr 26.docx` produced 2,796 chunks before coalescing and ~270 chunks after — a 10× reduction.

**Algorithm:** Single forward walk of `blocks`. A "run" accumulates adjacent blocks when all of:
1. `kind == "prose"` (non-prose blocks pass through atomic).
2. Token count is below `min_tokens` threshold (default 200).
3. Same `heading_path` as the run leader (heading boundary interrupts the run).
4. Adding the block would not exceed `max_tokens` cap (default 600).

A merged block inherits the leader's `heading_path` and `sheet`, and the first non-None `page` in the run.

**Inputs / outputs:**
- Input: `list[ExtractedBlock]` from the extractor.
- Output: `list[ExtractedBlock]` (new list; input is not mutated).

**Gating:**
- `RAG_INGEST_BLOCK_MIN_TOKENS` (default 200): blocks below this are coalesce candidates.
- `RAG_INGEST_BLOCK_MAX_TOKENS` (default 600): merged block stops growing when it would exceed this.
- Non-prose blocks (`kind != "prose"`) always pass through unchanged.

**Defaults:** Enabled unconditionally for prose blocks. Non-configurable toggle (no off switch), but setting `RAG_INGEST_BLOCK_MIN_TOKENS=0` effectively disables it (every block would be "large").

**Failure mode:** The function is pure Python with no I/O. On any tokenizer error it would propagate upward and fail the ingest task. In practice this has never been observed.

**Performance:** O(N) over blocks; tokenizer call per block (cached tokenizer singleton). <10ms per document in practice.

**What can go wrong:** If `RAG_INGEST_BLOCK_MAX_TOKENS` is set lower than typical paragraph size, merged blocks may still be very short. The 200/600 defaults were chosen for the 2026 monthly report corpus; tune for your specific document shapes.

**Observability:** No dedicated metric. The 10× chunk reduction is visible in `rag_ingest_chunks_per_doc{kb}` histogram.

---

### 3.5 OCR

**File:line:** `ext/services/ocr.py:1` — module; `ext/services/ingest.py:1166` — `extract_pdf_with_ocr_fallback`.

**What it does:** For PDF documents, detects pages where text extraction returned fewer than `RAG_OCR_TRIGGER_CHARS` characters (default 50) and re-runs those pages through an OCR backend.

**Inputs / outputs:**
- Input: `(pdf_bytes, filename, ocr_policy: dict | None)`.
- Output: concatenated text string (not `ExtractedBlock` list — OCR result feeds the extractor's flat path).

**Gating:**
- `RAG_OCR_ENABLED=1` (default 0 — disabled).
- Per-KB `ocr_policy` JSONB (migration `011_add_kb_ocr_policy.sql`): `{enabled: true/false, backend: "tesseract"|"cloud:textract"|"cloud:document_ai", language: "eng", trigger_chars_per_page: 50}`.
- OCR only fires for `application/pdf` mime type.
- Per-KB `ocr_policy.enabled=false` suppresses OCR even when the global flag is on.

**Defaults:** Tesseract (air-gap safe), English language, 50-char threshold.

**Cloud backends:** Textract requires `TEXTRACT_REGION` + `AWS_ACCESS_KEY_ID`; raises `RuntimeError` if credentials are missing — never silently falls back to Tesseract (`ocr.py:94`). Document AI requires `GCP_PROJECT` + `DOCUMENTAI_PROCESSOR_ID` + service account credentials.

**Failure mode:** Tesseract runs in a `ThreadPoolExecutor(max_workers=2)` (`ocr.py:28`). On import failure of `pymupdf`/`pytesseract`/`PIL`, raises at call time and fails the ingest task. Cloud backends fail closed on missing credentials.

**Performance:** Tesseract at 200 DPI takes ~2-8s per page on CPU. A 20-page scanned PDF may take 1-2 minutes. Plan accordingly when enabling for scan-heavy corpora.

**What can go wrong:** The OCR path currently produces a flat string, not structured `ExtractedBlock`s — page metadata is not preserved for OCR'd pages. This means `page=` hints in Qdrant payloads will be absent for OCR'd content.

**Observability:** No dedicated Prometheus metric for OCR currently. Log line `"orgchat.ocr: ..."` in `celery-worker` logs.

---

### 3.6 Chunk

**File:line:** `ext/services/chunker.py:287` — `chunk_text`; `ext/services/ingest.py:594` — dispatch loop; `ext/services/chunker_structured.py:1` — structured chunker.

**What it does:** Converts each `ExtractedBlock.text` into a list of token-bounded `Chunk` objects. Two strategies are available.

**Window chunker (default):** Sentence-aware sliding window. Packs whole sentences until the next one would overflow `chunk_tokens`, then emits a chunk with `overlap_tokens` of tail-overlap with the previous chunk. A sentence larger than `chunk_tokens` is hard-split at token boundaries (`chunker.py:287`).

**Structured chunker (`RAG_STRUCTURED_CHUNKER=1` + `rag_config.chunking_strategy="structured"`):** Recognizes fenced code blocks, markdown pipe tables, HTML tables, and emits them as atomic chunks with `chunk_type` metadata (`code`, `table`, `image_caption`). Prose falls through to the window chunker. Oversized tables are split by row-groups with a `continuation` flag (`chunker_structured.py:67`).

**Token counting:** Both chunkers use the shared `ext.services.budget.get_tokenizer()` singleton — the same tokenizer used by the budget truncation pass. This ensures chunk sizes track real prompt token counts. Set `RAG_BUDGET_TOKENIZER=gemma-4` + `RAG_BUDGET_TOKENIZER_MODEL=...` to match the deployed chat model (default is `cl100k` which undercounts gemma-4 tokens by 10-15%).

**Multilingual sentence splitting:** `RAG_PYSBD_ENABLED=1` (default 0) activates `pysbd` for 23 languages including Hindi (danda `।`), CJK (`。`), Arabic. Language is auto-detected by Unicode block sniff of the first 512 chars (`chunker.py:114`). Missing `pysbd` wheel falls back to the English regex silently.

**Inputs / outputs:**
- Input: text string, `chunk_size_tokens`, `overlap_tokens`, optional `rag_config`.
- Output: list of `dict{text, chunk_type, language, continuation}` (structured) or `Chunk` objects (window).

**Gating:**
- Default: `chunk_tokens=800`, `overlap_tokens=100` (from `ingest_bytes` signature defaults and `CHUNK_SIZE`/`CHUNK_OVERLAP` env vars).
- Per-KB override: `rag_config.chunk_tokens` (100-2000, bounded by `RAG_CHUNK_MAX_TOKENS` default 2000) and `rag_config.overlap_tokens` (0-1000), resolved in `kb_config.resolve_chunk_params()` (`kb_config.py:429`).
- Structured chunker requires both `RAG_STRUCTURED_CHUNKER=1` AND `rag_config.chunking_strategy="structured"` (`ingest.py:584`).

**Failure mode:** Empty text → zero chunks → block skipped. Token overflow in hard-split path is handled gracefully. No external I/O.

**Performance:** O(N) in text length. Typically <5ms per block.

**What can go wrong:** If `RAG_BUDGET_TOKENIZER` does not match the deployed chat model, chunk sizes in tokens are wrong, causing budget truncation to evict correct chunks later. This is the most common misconfiguration silent failure.

**Observability:**
- `rag_ingest_chunks_per_doc{kb}` histogram — observable spread in Grafana.
- `TokenizerFallbackHigh` alert fires when `tokenizer_fallback_total` exceeds 10/hr (`alerts-retrieval.yml:37`).

---

### 3.7 Contextualize

**File:line:** `ext/services/contextualizer.py:1`; called from `ext/services/ingest.py:670` via `_maybe_contextualize_chunks`.

**What it does:** For each chunk, calls the chat model with a "situate this chunk in the document" prompt (Anthropic Contextual Retrieval pattern). The ~50-token LLM-generated context prefix is prepended to the chunk text before embedding. Mutates the `paired` list in place (`ingest.py:244`).

**Concurrency:** Up to `RAG_CONTEXTUALIZE_CONCURRENCY=8` concurrent LLM calls per document via `asyncio.gather` under a semaphore.

**Prefix-cache friendly:** The system message and document-header user message are byte-identical for all chunks of the same document — vllm's prefix cache reuses the KV prefix across the per-chunk requests.

**Gating:**
- Global: `RAG_CONTEXTUALIZE_KBS=1` (default 0).
- Per-KB: `rag_config.contextualize=true` (opt in) or `false` (opt out, overrides global). Resolved in `ingest.should_contextualize()` (`ingest.py:80`).
- `OPENAI_API_BASE_URL` must be set; if absent, contextualize is silently skipped (`ingest.py:226`).

**Side effects:** When contextualization ran, stamps `pipeline_version=ctx=contextual-v1` on every chunk point. Also updates `kb_documents.pipeline_version` via `_persist_doc_pipeline_version` (`ingest.py:155`) so the admin drift dashboard doesn't mis-classify the doc.

**Failure mode:** Fail-open at both per-chunk and batch level. Any error returns the original chunk unchanged. `context_augmented` stays `False`, so `pipeline_version` is not bumped to `contextual-v1`.

**Performance:** ~2-3s per chunk on a local Qwen model. For a 270-chunk document: 270/8 concurrency ≈ 34 batches × 2-3s ≈ 68-102s of LLM time. Cost: significant. Evaluate before enabling on large corpora.

**Observability:** No dedicated metric. Check `rag_stage_latency_seconds{stage="contextualize"}` if you add instrumentation, or `docker compose logs celery-worker` for `orgchat.contextualize` log lines.

---

### 3.8 RAPTOR / Temporal RAPTOR

**File:line:** `ext/services/raptor.py` (legacy flat); `ext/services/temporal_raptor.py` (current, Phase 5.5).

**What it does:** Builds a hierarchical tree of LLM summaries over the chunks. L0 = original chunks (untouched); L1 = per-month subtree summaries; L2 = per-quarter summaries; L3 = per-year; L4 = multi-year meta summary. Each node becomes a Qdrant point with `chunk_level` payload (`0` = leaf, `1+` = summary).

**Tree depth:** Controlled by `RAG_RAPTOR_MAX_LEVELS` (default 3), `RAG_RAPTOR_CLUSTER_MIN` (default 5), `RAG_RAPTOR_CONCURRENCY` (default 4).

**Gating:** `RAG_RAPTOR=1` via `flags.get` (default 0). When off, `raptor.py` is never imported — zero cost on the default path (`ingest.py:797`).

**Failure mode:** Fail-open at batch level (`ingest.py:821`). Any exception during tree build drops back to flat chunk ingest.

**Performance:** Each summary node requires one LLM call. A 3-year corpus can require many hundreds of summary calls. Only enable if temporal query quality justifies the ingest wall-time cost.

**Observability:** No dedicated metric currently. `rag_temporal_levels` gauge in production deployment.

---

### 3.9 Doc Summary

**File:line:** `ext/services/doc_summarizer.py:47` — `summarize_document`; `ext/services/ingest.py:1023` — caller; `ext/services/ingest.py:1045` — `_emit_doc_summary_point`.

**What it does:** After the chunk-level upsert succeeds, calls the chat model for a 3-sentence document summary, embeds it, and upserts one Qdrant point with `level="doc"` and `kind="doc_summary"`. Also mirrors the summary text to `kb_documents.doc_summary` (added in migration `008_add_doc_summary.sql`) so the admin UI can render it without a Qdrant round-trip.

**Summary prompt** (`doc_summarizer.py:38`): Instructs the model to include document name, top-line content, dates/entities/identifiers, as a single prose paragraph.

**Point ID:** `uuid5(NS, "doc:{doc_id}:doc_summary")` — deterministic and idempotent (`ingest.py:1129`).

**Gating:**
- `RAG_DOC_SUMMARIES=1` (default 0; enabled in production per CLAUDE.md §7).
- `doc_id` must be present (KB uploads only; not emitted for chat-private uploads).
- `OPENAI_API_BASE_URL` must be set.
- Optional per-KB: `rag_config.doc_summaries=true/false`.

**Why it matters:** Global-intent retrieval (`retrieve_kb_sources` with `intent="global"`) searches against `level="doc"` points. If doc summaries are missing, global intent yields no results.

**Failure mode:** Fail-open. Summary failure leaves the document at the chunk tier only — retrieval still works, global intent is degraded. Backfill script available at `scripts/backfill_doc_summaries.py`.

**Timeout:** `RAG_DOC_SUMMARY_TIMEOUT=30.0` seconds (default).

**Observability:** Warning log `"doc summary emit failed for doc_id=%s"`. No dedicated metric. Use `rag_retrieval_empty_total{intent="global"}` as a downstream signal.

---

### 3.10 Embed

**File:line:** `ext/services/ingest.py:694` — three concurrent arms; `ext/services/embedder.py:137` — `TEIEmbedder`; `ext/services/sparse_embedder.py:58` — `embed_sparse`; `ext/services/embedder.py` — `colbert_embed`.

**What it does:** Embeds all chunk texts in three parallel async arms via `asyncio.gather`. Arms that are disabled or fail return a list of `None` values — the upsert loop skips them.

**Dense (TEI):**
- Model: `bge-m3` (1024-d cosine). Served by `tei` container on GPU 1.
- Batched: `RAG_TEI_MAX_BATCH=32` (default). The embedder splits the chunk list into batches of 32 and concatenates results (`embedder.py:166`).
- Retry-with-halving (introduced 2026-05-03): On retryable errors (424 CUDA OOM, 429, 5xx, network), retries up to `RAG_EMBED_MAX_RETRIES=3` times at the same batch size with exponential backoff (0.5s, 1s, 2s). After exhausted retries at batch > 1, halves the batch and recurses. Recursion floor at batch=1 — a single-chunk failure surfaces as an exception. Order is preserved.
- Circuit breaker: `RAG_CB_TEI_ENABLED=1` wraps each batch dispatch. One breaker decision per user-facing `embed` call, not per retry.

**Sparse BM25 (fastembed):**
- Model: `Qdrant/bm25` (client-side, ~10 MB ONNX). Runs synchronously in `asyncio.to_thread` so it doesn't block the event loop (`ingest.py:722`).
- Lazy singleton init: first call downloads and loads the model; subsequent calls use the cached `_MODEL`.
- Gating: `RAG_HYBRID=1` (default 1) AND the target collection was created with sparse support. Falls back to `[None] * N` if either gate is closed.

**ColBERT (fastembed):**
- 128-d multi-vector late interaction. ~10 KB payload per chunk. Runs in `asyncio.to_thread`.
- Gating: `RAG_COLBERT=1` (default 0) AND the collection has the `colbert` named slot.
- RAPTOR summary nodes do not get ColBERT vectors (token-level late interaction on LLM paraphrases is not meaningful, `ingest.py:924`).

**Performance:** Dense is GPU-bound (TEI). Sparse and ColBERT are CPU-bound ONNX. Under `asyncio.gather`, total embed time ≈ max(dense, sparse, colbert) rather than their sum — a significant improvement over the sequential path.

**What can go wrong:**
- TEI OOM (424): before the retry-with-halving fix (commit `b82d8db`), this failed the entire ingest. Now it halves and retries.
- `RAG_TEI_MAX_BATCH` evolution: the batch cap was 4096 (too high, caused OOM), then 8192/4, then settled at 8192/32 (default 32) to balance throughput vs memory pressure.
- `fastembed` not installed: `SparseEmbeddingNotAvailable` raised, sparse arm returns `None` list — hybrid retrieval silently degrades to dense-only.

**Observability:**
- `embedder_retry_total{outcome, reason}` — counts retry attempts and outcomes (`metrics.py:740`).
- `embedder_halving_total{batch_size_class}` — counts batch halvings (`metrics.py:746`).
- `rag_qdrant_upsert_latency_seconds` — OTel-wrapped Qdrant upsert timing.

---

### 3.11 Upsert

**File:line:** `ext/services/vector_store.py:353` — `ensure_collection`; `ext/services/ingest.py:997` — upsert dispatch; `ext/services/temporal_shard.py:1` — shard_key derivation.

**What it does:** Writes the vector points to Qdrant. Two paths:
- Standard: `vector_store.upsert(collection, points)` — single upsert call for the whole batch.
- Temporal: `vector_store.upsert_temporal(collection, points, shard_key=doc_shard_key)` — routes the batch to the named temporal shard (`RAG_SHARDING_ENABLED=1`).

**Point ID:** Deterministic UUIDv5 from `doc:{doc_id}:chunk:{index}` (KB) or `chat:{chat_id}:chunk:{index}` (private). This makes re-ingest idempotent: upserting the same document twice overwrites the same points (`ingest.py:910`).

**Named vectors:** The upsert dict carries `vector` (dense 1024-d), optional `sparse_vector` (BM25 indices+values), optional `colbert_vector` (128-d multi-vector). The collection must have been created with the corresponding slots; if not, Qdrant raises `"Wrong input: Not existing vector name"`.

**Shard key derivation** (`temporal_shard.py`): Priority — filename date (`Jan 23.docx` → `2023-01`) → YAML frontmatter `date:` → first date in body's first 1000 chars → current month at ingest (tagged as `shard_key_origin="ingest_default"`). Format: `YYYY-MM`.

**Payload fields upserted** (from `build_point_payload`, `ingest.py:260`): `kb_id`, `doc_id`, `subtag_id`, `owner_user_id`, `chat_id`, `filename`, `level`, `chunk_index`, `text`, `context_prefix`, `page`, `heading_path`, `sheet`, `uploaded_at`, `deleted`, `model_version`, `chunk_type`, `language`, `continuation`, `shard_key` (when sharding on).

**HNSW config:** `m=16`, `ef_construct=200`, `ef=128` at query time. Override via `RAG_QDRANT_M`, `RAG_QDRANT_EF_CONSTRUCT`, `RAG_QDRANT_EF`.

**Failure mode:** Qdrant 5xx → exception propagates out of `ingest_bytes` → Celery retry. Qdrant 400 on schema mismatch (missing named vector slot, missing shard_key) → non-retryable, goes to DLQ. Check `docker compose logs celery-worker` for `"Shard key not specified"` or `"Wrong input: Not existing vector name"`.

**Observability:**
- `rag_qdrant_upsert_latency_seconds` histogram.
- `rag_ingest_chunks_total{collection, path}` — bumped after successful upsert.
- `RAG_SHARD_UPSERT_LATENCY{collection, shard_key}` — per-shard timing when sharding enabled.

---

### 3.12 Status Update

**File:line:** `ext/workers/ingest_worker.py:124` — `_update_doc_status`; `ext/workers/ingest_worker.py:92` — `_get_engine` (NullPool singleton).

**What it does:** Writes `kb_documents.ingest_status` (and optionally `chunk_count`, `error_message`) to Postgres after each pipeline stage boundary.

**NullPool detail:** The engine is cached as a module-level singleton (`_engine_singleton`). On first call, it creates a `NullPool` engine. Each subsequent `engine.begin()` opens a fresh asyncpg connection to Postgres, executes the `UPDATE`, and closes it. Cost: ~5-10ms per write. Writes occur at most 4 times per ingest task (chunking, embedding, done/failed).

**Failure mode:** Best-effort, fail-open. Any DB error logs at `ERROR` and increments `ingest_status_update_failed_total{stage}`. Ingest data in Qdrant is unaffected — only the Postgres visibility is wrong.

**Observability:**
- `ingest_status_update_failed_total{stage}` counter.
- `IngestStatusUpdateFailing` alert (`alerts-celery.yml:37`): fires at `rate > 0` for 5 minutes.

---

## 4. Per-KB `rag_config` Knobs at INGEST Time

The following keys in `knowledge_bases.rag_config` JSONB influence the ingest pipeline specifically. They are defined in `ext/services/kb_config.py` and marked as `INGEST_ONLY_KEYS` (never propagated to the request-scope flag overlay).

| Key | Type | Env var equivalent | What it controls | Default |
|-----|------|--------------------|-----------------|---------|
| `contextualize` | bool | `RAG_CONTEXTUALIZE_KBS` | Per-KB opt-in/out for chunk contextualization. Explicit `false` suppresses even when global env=1. | Inherits env |
| `contextualize_on_ingest` | bool | `RAG_CONTEXTUALIZE_KBS` | Legacy alias for `contextualize`; accepted but new code reads `contextualize`. | Inherits env |
| `chunk_tokens` | int (100-2000) | `CHUNK_SIZE` (800) | Token window size for this KB's ingest. | Inherits env |
| `overlap_tokens` | int (0-1000) | `CHUNK_OVERLAP` (100) | Overlap between consecutive chunks. Clipped to `chunk_tokens//4` if ≥ `chunk_tokens`. | Inherits env |
| `chunking_strategy` | `"window"` \| `"structured"` | n/a | When `"structured"` + `RAG_STRUCTURED_CHUNKER=1`, uses the table/code-aware chunker. | `"window"` |
| `doc_summaries` | bool | `RAG_DOC_SUMMARIES` | Whether ingest emits a `level="doc"` summary point. Required for global-intent retrieval. | Inherits env |
| `image_captions` | bool | `RAG_IMAGE_CAPTIONS` | Whether ingest calls the vision LLM for PDF images. Per-KB explicit `false` suppresses even if global=1. | Inherits env |

Source: `ext/services/kb_config.py:178` (`INGEST_ONLY_KEYS`).

**Note:** To inspect actual KB configs, run:
```sql
SELECT id, name, jsonb_pretty(rag_config) FROM knowledge_bases WHERE id IN (2,3,8);
```

Multi-KB merge policy (relevant when an ingest is triggered for a KB that shares config with others): booleans use `any()` (union), integers use `max()` (strictest wins). For ingest-only keys, the KB's own `rag_config` is read directly by the Celery worker (`_fetch_kb_rag_config`, `ingest_worker.py:183`) — the merge is not applied at ingest time.

---

## 5. Critical Paths and Failure Modes

### 5.1 The asyncpg Event-Loop Bug (commit `ebe4fee`)

**Symptom:** `ingest_status` stuck at `chunking` even though Qdrant has the chunks and retrieval works.

**Root cause:** Celery prefork workers run each task in a fresh `asyncio.run()` scope. SQLAlchemy's default `QueuePool` cached asyncpg connections bound to the loop alive at pool-creation time. The second task in the same worker process triggered `RuntimeError: Event loop is closed` (or `Future ... attached to a different loop`) inside `_update_doc_status`. This was silently swallowed by the `except Exception` that already existed to keep status updates non-critical. No log appeared at `WARNING` level (operators typically tail at `ERROR+`), so the failure was invisible.

**Fix:** `_get_engine()` creates the singleton with `poolclass=NullPool` (`ingest_worker.py:120`). Each `engine.begin()` opens a fresh connection on the current loop. Cost: ~5-10ms per status write — irrelevant versus the multi-second embedding time. The fix was paired with a counter (`ingest_status_update_failed_total`) and alert (`IngestStatusUpdateFailing`) so future failures at the `log.error` level are visible in Prometheus within minutes.

**Residual risk:** If `DATABASE_URL` is wrong or the engine cannot connect (Postgres down), `_get_engine()` still returns a cached singleton but all subsequent `engine.begin()` calls fail. The counter will fire; fix the connection and the worker recovers on the next task without restart.

### 5.2 TEI Batch-Cap Evolution

**Problem:** GPU 1 (24 GB shared by TEI, reranker, ColBERT, fastembed, vllm-qu) runs at ~95% steady state. Large embedding batches spike GPU activation memory and return HTTP 424 (`CUDA_ERROR_OUT_OF_MEMORY`) from TEI.

**Evolution:**
- Pre-fix: `RAG_TEI_MAX_BATCH` defaulted to 4096 (far too high). Single 424 failed the entire ingest task.
- Intermediate: Set to `8192/4` (still occasionally OOM'd under load).
- Current: `RAG_TEI_MAX_BATCH=32` (default) with retry-then-halve redundancy. On 424/429/5xx: retry up to `RAG_EMBED_MAX_RETRIES=3` times; if exhausted at batch > 1, halve and recurse. Recursion floor at batch=1 prevents infinite loop.

**Monitor:** `embedder_halving_total{batch_size_class="1"}` is a danger signal — TEI cannot embed even a single chunk, indicating model misload or pathological input (post-OCR garbage, extremely long token sequence).

### 5.3 Block Coalescing Fix

**Problem:** `python-docx` emits one `ExtractedBlock` per Word paragraph. On business prose (median ~30 tokens per paragraph), the 800-token window chunker received one tiny text input per paragraph and produced one tiny chunk each — no cross-paragraph packing occurred. `Apr 26.docx`: 2,796 chunks before fix → ~270 after (10× reduction). Tiny chunks have low recall because they lack enough surrounding context to match queries.

**Why structured chunker alone wasn't enough:** The structured chunker operates on the text string from each block, not across blocks. It handles tables and code as atomic units correctly, but it does not concatenate adjacent blocks from the extractor. The coalescer is the pre-chunker fix that runs before any chunker is called.

**Tuning knobs:** `RAG_INGEST_BLOCK_MIN_TOKENS=200` (blocks below this are candidates), `RAG_INGEST_BLOCK_MAX_TOKENS=600` (merged block ceiling). Both read at call time — operator can change them without code deploy.

### 5.4 asyncpg JSONB-as-String Quirk

**Problem:** When SQLAlchemy passes a Python `dict` to a Postgres `JSONB` column using raw SQL (not ORM models), asyncpg expects the value pre-serialized as a JSON string, not a native dict. Passing a native dict raises `asyncpg.exceptions._base.InterfaceError: cannot convert type dict to asyncpg value`.

**Where it bites:** `_update_doc_status` and `_fetch_kb_rag_config` in the Celery worker use raw SQL via `sqlalchemy.text`. The `_fetch_kb_rag_config` function returns `dict(cfg) if cfg else None` — asyncpg correctly returns JSONB as a Python dict when reading, so the read path is fine. The write path in operator scripts (`scripts/apply_migrations.py` etc.) must use `json.dumps(value)` when writing JSONB columns with raw SQL.

### 5.5 Image-Rebuild HF Cache Mount Path Change

After the `USER 1000:1000` bake (§10.1 of CLAUDE.md), the HF model cache mount target moved from `/root/.cache/huggingface` to `/home/orgchat/.cache/huggingface`. `/root/` is mode 700 in the new image — the UID=1000 user cannot traverse it, so any model load that tries the old path fails with `PermissionError`. This affects `open-webui` (cross-encoder, fastembed) and `celery-worker` (fastembed). Update `compose/docker-compose.yml` volume mounts accordingly and run the one-time `chown` described in CLAUDE.md §8.

---

## 6. Observability — Every Counter, Span, and Alert

### Prometheus Metrics (source: `ext/services/metrics.py`)

| Metric | Type | Labels | Triggered when |
|--------|------|--------|----------------|
| `rag_upload_bytes_total` | Counter | `kb` | Bytes accepted by upload handler |
| `rag_ingest_chunks_total` | Counter | `collection, path` | Chunks upserted (sync or celery) |
| `rag_ingest_chunks_per_doc` | Histogram | `kb` | Chunks produced per document |
| `rag_ingest_document_bytes` | Histogram | `kb, format` | Document size at upload |
| `rag_ingest_duration_seconds` | Histogram | `stage` | Per-stage ingest duration |
| `rag_ingest_failures_total` | Counter | `stage, reason` | Ingest failures by stage |
| `rag_ingest_queue_depth` | Gauge | — | Celery ingest queue depth |
| `ingest_status_update_failed_total` | Counter | `stage` | `_update_doc_status` write failures |
| `rag_qdrant_upsert_latency_seconds` | Histogram | — | Per-upsert call timing |
| `rag_shard_upsert_latency_seconds` | Histogram | `collection, shard_key` | Per-shard upsert timing |
| `embedder_retry_total` | Counter | `outcome, reason` | TEI embed retry attempts |
| `embedder_halving_total` | Counter | `batch_size_class` | TEI batch halving events |
| `rag_image_skip_total` | Counter | — | Images skipped (vision LLM unreachable) |
| `rag_snapshot_failure_total` | Counter | `collection` | Daily Qdrant snapshot failures |

### OTel Spans (propagated from HTTP → Celery via W3C traceparent)

| Span name | Where opened | Key attributes |
|-----------|-------------|----------------|
| `upload.request` | `upload.py:167` | `user_id, kb_id, subtag_id, filename` |
| `upload.read_bounded` | `upload.py:185` | — |
| `upload.blob_write` | `upload.py:299` | `size_bytes` |
| `upload.enqueue_celery` | `upload.py:303` | `collection` |
| `upload.ensure_collection` | `upload.py:270` | `collection` |
| `upload.ingest_sync` | `upload.py:272` | `collection, size_bytes` |
| `ingest.celery_task` | `ingest_worker.py:322` | `sha, collection, doc_id, kb_id, chunks` |

### Prometheus Alerts Covering Ingest

| Alert | File | Expression | Severity | Investigate when |
|-------|------|-----------|---------|-----------------|
| `IngestStatusUpdateFailing` | `alerts-celery.yml:37` | `rate(ingest_status_update_failed_total[5m]) > 0` for 5m | warning | Docs stuck at previous status; asyncpg loop issue or PG down |
| `CeleryDLQDepthHigh` | `alerts-celery.yml:5` | `celery_dlq_depth > 10` for 10m | warning | Documents permanently failing; check DLQ + celery-worker logs |
| `CeleryWorkerDown` | `alerts-celery.yml:12` | `up{job="celery-worker"} == 0` for 5m | critical | All async ingest is silently queuing (blobs accumulate in `/var/ingest`) |
| `CeleryUploadLatencyHigh` | `alerts-celery.yml:21` | p95 > 30s for 10m | warning | Worker saturated or slow |
| `RagRetrievalEmptyHigh` | `alerts-rag-quality.yml:11` | `rate(rag_retrieval_empty_total[5m]) > 0.1` for 10m | warning | Ingest broken: newly added docs not returning in queries |
| `TokenizerFallbackHigh` | `alerts-retrieval.yml:37` | `rate(tokenizer_fallback_total[1h]) > 10/3600` for 30m | warning | Chunk token counts wrong; HF cache mount or RAG_BUDGET_TOKENIZER misconfigured |

---

## 7. Operator Runbook Quick-Reference

### Watch a doc ingest in real time

```bash
# Watch SSE stream for a specific KB (token from /api/auth/... or set in .env)
curl -N -H "Authorization: Bearer $TOKEN" \
  "https://$DOMAIN/api/kb/1/ingest-stream"

# Or tail the worker logs directly
docker compose -p orgchat logs -f celery-worker open-webui
```

### Check current status of a document

```sql
SELECT id, filename, ingest_status, chunk_count, error_message, created_at
FROM kb_documents WHERE kb_id = 1 ORDER BY created_at DESC LIMIT 20;
```

### Recover a doc stuck at chunking / embedding (after NullPool fix is deployed)

If `ingest_status` is stuck but `celery-worker` logs show no active task for that `doc_id`, the fix is deployed but the status write failed for a transient reason. Verify chunks are in Qdrant (see below), then manually correct:

```sql
-- Verify Qdrant has the chunks first (count should match expected)
-- Then correct the status row
UPDATE kb_documents SET ingest_status='done', chunk_count=<N> WHERE id=<doc_id>;
```

This is safe because ingest is idempotent (UUIDv5 point IDs — re-ingest overwrites the same Qdrant points) and the Postgres row update does not touch Qdrant.

### Verify an ingest reached Qdrant

Use the Qdrant REST API with a `doc_id` payload filter:

```bash
curl -s -X POST "http://localhost:6333/collections/kb_1/points/count" \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "doc_id", "match": {"value": 42}}]}}'
```

For temporally sharded collections use the v4 alias: `kb_1_v4`.

### Re-ingest a document safely

Deterministic UUIDv5 point IDs make re-ingest idempotent at the Qdrant level — upserting the same `doc_id` + chunk index overwrites the same points without creating duplicates. Procedure:

1. Admin deletes the doc via `DELETE /api/kb/{id}/documents/{doc_id}` (hard-deletes end-to-end: DB row + Qdrant filter delete).
2. Re-upload the file via `POST /api/kb/{id}/subtag/{sid}/upload`.

Or if you only want to re-embed without re-extracting, use `scripts/reembed_all.py` (checks `pipeline_version` to skip already-current docs).

### Find ingest logs in real time

```bash
docker compose -p orgchat logs -f celery-worker open-webui 2>&1 | grep -E "orgchat\.(ingest|ocr|embed|qdrant)"
```

---

## 8. Deferred / Out of Scope

The following features are **not yet in the codebase** as of 2026-05-03:

- **Semantic chunking (topic-shift detection):** The structured chunker handles tables and code as atomic units but does not detect semantic topic shifts within prose. A potential future addition would use embedding similarity between sliding windows to detect content boundaries.
- **LLM-driven OCR layout reconstruction:** Current OCR (Tesseract, Textract, Document AI) returns plain text. Future work could use vision LLMs to reconstruct layout-sensitive content (multi-column PDFs, tables in scanned images) with structural metadata preserved.
- **Auto-generated synonyms:** The `synonyms` list in `rag_config` (`migration 017_kb_synonyms.sql`; `kb_config.py:166`) is operator-curated only. There is no mechanism to auto-generate synonym equivalence classes from corpus vocabulary.
- **Cross-language embedding fine-tune:** `bge-m3` supports multilingual embedding out of the box, but no per-corpus or cross-language fine-tuning pipeline exists. Domain-specific terminology (military abbreviations, legal references) is not adapted.
- **Per-corpus embedder fine-tune:** No fine-tuning loop for domain adaptation of the base `bge-m3` model. This would require a separate training pipeline and a mechanism to hot-swap the TEI model without a full re-ingest.
