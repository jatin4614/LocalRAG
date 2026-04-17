# RAG Pipeline Deep-Audit — Recommendations

**Scope:** `ext/` package of the LocalRAG fork of Open WebUI (ingestion → chunking → embedding → vector storage → retrieval → rerank → context assembly → LLM bridge).
**Method:** Line-level read of every file in `ext/services/`, `ext/routers/`, `ext/db/`, `ext/app.py`, `ext/config.py`, plus the two patches under `patches/` and the ingestion-relevant `compose/docker-compose.yml` settings.
**Output:** Findings grouped by pipeline stage. Each finding gives: **(a)** file + line, **(b)** problem, **(c)** impact, **(d)** concrete fix.

Severity legend — **P0 = correctness/data-loss/security**, **P1 = user-visible quality or scale failure**, **P2 = performance or maintainability**, **P3 = polish**.

---

## Executive Summary — Top 10 Things To Fix First

| # | Severity | Finding | Where |
|---|----------|---------|-------|
| 1 | **P0** | Production uses raw SQL against `"user"` / `chat`; several routers still use the compat `Chat` ORM (mapped to a non-existent `chats` table) → `POST /api/rag/retrieve`, `POST /api/chats/{id}/private_docs/upload` will 404 in production | `ext/routers/rag.py:77-79`, `ext/routers/upload.py:139-145` |
| 2 | **P0** | Whole-file `await file.read()` then size-check → trivially exploitable memory DoS (send a 10 GB upload, server buffers it all before rejecting) | `ext/routers/upload.py:55-62` |
| 3 | **P0** | `uuid.uuid4()` used as the Qdrant point id with no `doc_id` embedded → `DELETE /documents/{id}` cannot remove vectors (and the code comments this as a known TODO) — every deletion leaks embeddings forever | `ext/services/ingest.py:42-46`, `ext/routers/kb_admin.py:276` |
| 4 | **P0** | bge-m3 is called without query/passage differentiation → retrieval quality silently degraded (bge-m3 expects the *query* path to omit or use a different format than the passage path) | `ext/services/embedder.py:51-54`, `ext/services/retriever.py:27` |
| 5 | **P1** | `ingest_bytes` embeds ALL chunks in one TEI call and upserts them in one Qdrant call → any moderately large PDF (≥ TEI's `max_client_batch_size`, default 32) hits a `413` or times out | `ext/services/ingest.py:32-47` |
| 6 | **P1** | No payload indexes on `kb_id` / `subtag_id` / `doc_id` in Qdrant → filters run full scans as collections grow past ~100k points | `ext/services/vector_store.py:36-42` |
| 7 | **P1** | `retrieve()` embeds the query *before* checking whether any KB was selected; then silently swallows Qdrant errors → wasted GPU + indistinguishable "no hits" vs "backend down" | `ext/services/retriever.py:27, 37, 45` |
| 8 | **P1** | Reranker "normalization" divides by per-KB max — a tiny KB with one weak hit (0.3) gets normalized to 1.0 and outranks a strong hit (0.85) from a big KB → cross-KB ranking inversion | `ext/services/reranker.py:26-38` |
| 9 | **P1** | No RBAC cache → every assistant message incurs 2-3 DB roundtrips (`role` + `group_member` + `kb_access`) before even embedding the query | `ext/services/rbac.py:19-46` (called in `chat_rag_bridge.retrieve_kb_sources` per-message) |
| 10 | **P1** | Private chat documents are never actually routed through the middleware path — bridge hardcodes `chat_id=None` → the entire "session-local private docs" feature described in `CLAUDE.md §4` is dead in the LLM flow | `ext/services/chat_rag_bridge.py:90` |

---

## Stage 1 — Document Ingestion

### 1.1 Text Extraction — `ext/services/extractor.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 1.1.1 | **P0** | `54` | `application/msword` routes to `_extract_docx` (python-docx), but python-docx does **not** support `.doc` binary format | Crashes with `BadZipFile` on any legacy `.doc` upload | Either drop `.doc` from the allow-list, or add `antiword`/`textract` fallback for `application/msword` |
| 1.1.2 | **P0** | `13, 44` | `data.decode("utf-8", errors="replace")` silently replaces invalid bytes with `\ufffd` | Non-UTF8 files (Windows-1252 CSVs, UTF-16 BOM exports) become garbage-embedded; user can't tell | Detect encoding (`chardet`/`charset-normalizer`) and pick the best match; log when `errors` kick in |
| 1.1.3 | **P1** | `27` | DOCX extractor only iterates `doc.paragraphs`; ignores tables, headers, footers, textboxes, footnotes | Significant portion of business documents lost in extraction | Walk `doc.element.body` directly; concatenate table cells row-wise; include section headers |
| 1.1.4 | **P1** | `33-40` | XLSX reads **every row** of every worksheet in memory, then joins → 50 MB workbook → ~500 MB Python strings | RAM spike; event-loop blocked during load | Yield per-sheet (generator); write chunks to a temp file; or use streaming via `openpyxl` + incremental flush |
| 1.1.5 | **P1** | `20` | PDF extraction is synchronous, CPU-heavy, and runs on the request thread | 1 MB PDF ≈ 1-5 s of blocking I/O → blocks whole FastAPI worker | Run `extract_text` inside `asyncio.to_thread(...)` or a `ProcessPoolExecutor` |
| 1.1.6 | **P1** | — | No size/page limits → 500-page scanned PDF → minutes of CPU | One malicious or accidental upload stalls a worker | Add `MAX_PAGES` env; short-circuit if `len(reader.pages) > MAX_PAGES`; add per-file wall-clock budget |
| 1.1.7 | **P1** | `20` | `page.extract_text() or ""` swallows empty pages silently; no warning when 100 % of pages yield nothing (scanned PDF) | User uploads a scanned PDF, ingest says "done, 0 chunks", mystery "nothing matches" in chat | If `ratio(empty_pages/total) > 0.8`, set `ingest_status="failed"` with `error_message="likely a scanned/image PDF — OCR required"` |
| 1.1.8 | **P2** | — | No `.pptx`, `.html`, `.rtf`, `.epub`, images (OCR), audio transcript support | Limits "organizational knowledge" coverage — major gap vs design goal in `CLAUDE.md §2` | Add an `EXTRACTORS` entry per format; use `unstructured` / `tesseract` / `whisper` as optional deps |

### 1.2 Chunking — `ext/services/chunker.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 1.2.1 | **P1** | `22-42` | Pure token-window chunking with no structural awareness — splits mid-sentence, mid-heading, mid-table, mid-code-fence | Retrieval returns hits that lose their anchoring context; "what does the Config section say?" retrieves a chunk that starts mid-paragraph and ends mid-word | Add a semantic pre-split (Markdown headings → paragraphs → sentences), then pack until `chunk_tokens`; or use LangChain `RecursiveCharacterTextSplitter` with token length function |
| 1.2.2 | **P1** | `29` | `enc.encode(text)` encodes the **entire** document in one go — for a 10 MB doc this is 10-30 s of CPU **inside the async event loop** | One big ingest blocks everything | Wrap in `asyncio.to_thread()`; or chunk-encode in streaming fashion on paragraph boundaries |
| 1.2.3 | **P2** | `22` | Hardcoded 800/100; cannot vary per-KB or per-format (code benefits from smaller chunks, prose from larger) | Can't tune retrieval quality without redeploy | Accept `chunk_tokens`/`overlap_tokens` via `KBDocument.config JSONB` or KB-level setting |
| 1.2.4 | **P2** | `37` | `enc.decode(chunk_ids)` called once per chunk — on long docs this is O(N²-ish) since each decode walks a new slice | Not critical but wastes CPU | Use a proper streaming chunker that yields offsets, decoding once |
| 1.2.5 | **P3** | `11-15` | `Chunk` has `index, text` but no byte/char offsets into source | Cannot cite "chunk 3, lines 42-58" in UI | Add `start, end` char offsets from the pre-encoded raw text |

### 1.3 Embeddings — `ext/services/embedder.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 1.3.1 | **P0** | `51-54` | No differentiation between query and passage embedding (bge-m3 uses different internal handling; query-side may need `query: <text>` format for best retrieval). Raw text goes in both directions. | Retrieval recall/MRR measurably worse vs. running bge-m3 as intended | Add `embed_query(text)` vs `embed_passages(texts)` methods; for bge-m3 add the documented prefixes; update `retriever.retrieve` and `ingest.ingest_bytes` to call the right side |
| 1.3.2 | **P1** | `52` | Single POST sends **all** chunks in one body → will exceed TEI's `max_client_batch_size` (default 32) on any ~30+ chunk doc and fail with 413 | Large docs fail to ingest; no partial success | Chunk into batches of `min(N, TEI_MAX_BATCH)` with a configurable env; semaphore-limit concurrency |
| 1.3.3 | **P1** | `46` | `httpx.AsyncClient` created at construction with default timeouts; no retry, no backoff, no circuit breaker | One TEI hiccup → every upload in flight fails irretrievably | Wrap calls in `tenacity.retry(...)`; expose `TEI_RETRIES`, `TEI_BACKOFF` env; implement circuit breaker (e.g., `aiobreaker`) |
| 1.3.4 | **P1** | — | No embedding cache | Re-uploading the same doc re-embeds chunk-by-chunk; a /reembed endpoint would re-pay full cost | Cache `sha256(chunk_text) -> vector` in Redis with TTL; skip known hashes on upsert |
| 1.3.5 | **P2** | `46-49` | `httpx.AsyncClient` has no explicit `limits=httpx.Limits(...)` | Under 20 concurrent uploads/retrievals, default pool of ~100 is fine but there's no cap protecting TEI | Set `limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)` |
| 1.3.6 | **P2** | `46` | `aclose()` exists but no one calls it — app shutdown never returns the client | Connection leak on reload; not critical | Wire a FastAPI `lifespan` that calls `emb.aclose()` and `vs.close()` |
| 1.3.7 | **P3** | `25-33` | `StubEmbedder` builds the hash `data` bytestring one 32-byte block at a time via a `while` loop — fine for tests, but does `ceil(4096/32)=128` hashes per text | Test suite slower than needed | Replace with `hashlib.shake_128(text).digest(dim*4)` — one call |

### 1.4 Ingest Pipeline — `ext/services/ingest.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 1.4.1 | **P0** | `42-46` | Point IDs are random UUIDs with no `doc_id` prefix / payload-keyed deletion helper | `DELETE /documents/{id}` (kb_admin.py:261) marks SQL row deleted but vectors persist forever (orphans). Soft-deleted docs are still retrieved. | Derive ID as `uuid5(NAMESPACE, f"{doc_id}:{chunk_index}")` OR store `doc_id` as the ID directly in a `PayloadSelector` for batch delete; implement `vector_store.delete_by_doc(kb_id, doc_id)` and call it from the delete endpoint |
| 1.4.2 | **P1** | `33` | All chunks embedded in one call; no progress streaming | Big docs hit TEI batch limits; also, no back-pressure | Implement `_batch_embed(texts, size=32)` generator; stream points in waves into Qdrant |
| 1.4.3 | **P1** | `47` | `wait=True` on every upsert → blocks until Qdrant flushes to disk (≈100-300 ms per batch) | Adds latency proportional to chunk count | Use `wait=False` for bulk ingest, then one final `wait=True` sync at the end (or a post-ingest `collection.flush`) |
| 1.4.4 | **P1** | `27-47` | Ingest is sequential: extract → chunk → embed → upsert; no pipelining | Time = sum of each phase; could be `max(embed, upsert)` | Start upserting batch N while batch N+1 is still embedding; use an `asyncio.Queue` with 2 workers |
| 1.4.5 | **P1** | — | No idempotency / dedup by file hash | Same file uploaded twice → 2× embeddings, 2× cost, 2× storage, double-weighted retrieval | Compute `sha256(data)` before extraction; if `kb_documents` already has a row with this hash (not deleted) → 409 Conflict or return existing doc_id |
| 1.4.6 | **P1** | `14-25` | On any exception mid-embed, previously-upserted chunks stay in Qdrant but SQL row is flipped to "failed" | Orphaned partial vectors → skewed retrievals; no recovery | Wrap the whole ingest in a try/finally that calls `delete_by_doc(doc_id)` on failure |
| 1.4.7 | **P2** | `35` | `now = int(time.time())` seconds resolution | Two uploads in the same second sort nondeterministically | `datetime.now(tz=utc).isoformat()` or `time.time_ns()` |

### 1.5 Upload Router — `ext/routers/upload.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 1.5.1 | **P0** | `55-62` | `data = await file.read()` buffers the **entire** upload before checking size; then raises 413 | With `MAX_UPLOAD_BYTES=50 MB`, a 5 GB upload still allocates 5 GB before rejection → OOM crash | Stream in `async for chunk in file.stream():` and cut off at `MAX_UPLOAD_BYTES`; reject with `411` / `413` before full read. Also set `client_max_body_size` at Caddy layer |
| 1.5.2 | **P0** | `139-145` | Uses `select(Chat)` against the compat ORM which maps to `chats` (plural); upstream uses `chat` (singular) table with UUID PK | Endpoint 404s in production (the query returns nothing because the table literally isn't called `chats`) | Use the same raw-SQL pattern as `kb_retrieval.set_chat_kb_config`: `text('SELECT user_id FROM chat WHERE id = :cid')` |
| 1.5.3 | **P0** | `132` | `chat_id: int` path param, but upstream chat IDs are UUID strings | Every call fails with 422 at type-parse | Change to `chat_id: str`; cast only where needed |
| 1.5.4 | **P1** | `94-100` | DB row inserted **before** chunking/embedding; on extract failure, row is updated to `"failed"` but only if commit succeeds | `KBDocument.bytes` is correct but `chunk_count=0` row clutters admin UI | Insert row inside same transaction as the Qdrant upsert guard; rollback on pre-embed failure |
| 1.5.5 | **P1** | `23` | `MAX_UPLOAD_BYTES` read from env at import time | Can't change without restart; tests can't override per-test cleanly | Read inside request handler (or from `get_settings()`) |
| 1.5.6 | **P1** | `107, 151` | `file.content_type` is taken from the client verbatim; no MIME sniffing (`python-magic`) | Attacker uploads a PDF exploit labelled `text/plain` → pypdf never called, but the file gets embedded literally (nuisance), OR labels `.exe` as `application/pdf` → unexpected crash | Sniff from bytes (first ~1024) with `python-magic`; reject mismatches |
| 1.5.7 | **P1** | `104, 148` | `ensure_collection` called on **every** upload | Adds one `get_collections` + one compare per request; also has a race (two uploads to a new KB can both try to create) | Cache known collections in-process (a `set[str]`); on miss, `create_collection` with `try/except AlreadyExists` |
| 1.5.8 | **P1** | — | Admin is the only uploader → serial bottleneck; no queue | For a 200-doc bulk import, admin's HTTP connection must stay open for tens of minutes | Move ingest to a background worker (Redis/RQ or ARQ or Celery); return `202 Accepted` with a `doc_id`, expose `GET /documents/{id}/status` |
| 1.5.9 | **P1** | `94, 97` | `file.filename or "upload"` with no sanitization | Filename stored raw — later displayed in Svelte UI unescaped → XSS | `pathlib.Path(name).name`, strip control chars, cap length at 256 |
| 1.5.10 | **P2** | `116` | `error_message=str(e)[:1000]` truncates mid-multi-byte character | Text can split a UTF-8 codepoint → DB encoding warning | Truncate on char boundary: `str(e)[:1000].encode("utf-8", "ignore").decode("utf-8")` |
| 1.5.11 | **P2** | — | No per-user / per-KB upload rate limit | Single abusive admin can monopolize TEI | Redis-backed sliding window on `(user_id, endpoint)` |
| 1.5.12 | **P2** | `126-158` | `upload_private_doc` has no `ingest_status` tracking at all; no DB row, no error reporting | User can't see what happened; no retry path | Create a lightweight `chat_private_doc` table or reuse `kb_documents` with sentinel `kb_id=-1`; track status |

---

## Stage 2 — Vector Storage

### 2.1 VectorStore — `ext/services/vector_store.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 2.1.1 | **P1** | `32-42` | `ensure_collection` = `list_collections` → string compare → maybe `create`. Race between concurrent callers, and O(N collections) every call | New-KB first upload can 500 on second concurrent upload; extra Qdrant traffic | `try: create_collection(...) except AlreadyExists: pass`; cache positive hits in `self._known: set[str]` |
| 2.1.2 | **P1** | `36-42` | `VectorParams(size, distance)` only — no `hnsw_config`, no `optimizers_config`, no `quantization_config` | On 500k-point collections, search latency and RAM are both 2-5× worse than necessary | Set `hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=200)`; set `quantization_config=qm.ScalarQuantization(...)` for large KBs |
| 2.1.3 | **P0/P1** | `36-42` | No payload indexes created for `kb_id` / `subtag_id` / `doc_id` | `search(..., subtag_ids=[...])` filter is a full-scan past ~100k vectors | After `create_collection`, call `create_payload_index(collection_name, field_name="subtag_id", field_schema="integer")` for each filter field |
| 2.1.4 | **P1** | `57-77` | `search()` has no `score_threshold` parameter | Returns junk low-similarity hits that then waste budget tokens | Accept `score_threshold`, pass through to Qdrant; default ~0.2 for cosine |
| 2.1.5 | **P1** | `57-77` | Soft-deleted documents are still retrieved (no `doc_deleted_at` payload filter) | Admin deletes a doc, users still see its content in RAG responses | Add `payload["deleted"]=false` on upsert; flip to `true` in `delete_document`; filter `must_not=[FieldCondition(key="deleted", match=MatchValue(value=True))]` |
| 2.1.6 | **P1** | `44-48` | `delete_collection` swallows all exceptions silently | KB delete can fail (Qdrant down, permission error) and the admin sees success | Log at `ERROR`; surface at least a 500 to the caller if needed |
| 2.1.7 | **P2** | `20-22` | Single `AsyncQdrantClient`, no prefer_grpc flag, no gRPC pool | HTTP/REST is ~30-50 % slower than gRPC on high-throughput batch queries | Use `prefer_grpc=True` with `grpc_port` env |
| 2.1.8 | **P2** | — | No `search_batch` / `query_batch_points` method | `retriever.py` does `asyncio.gather` of N point queries — Qdrant has a server-side batch API that's a single request | Add `search_batch(collection, queries: list[vector], ...)`; use it in retriever |
| 2.1.9 | **P2** | `13-17` | `Hit` class hand-rolled instead of `pydantic.BaseModel` or `msgspec.Struct` | No validation, harder to serialize | Tiny cleanup, reuse elsewhere |
| 2.1.10 | **P3** | `75` | `with_payload=True` but no `with_payload=["text","kb_id",...]` allowlist | Returns every payload field; larger response; more parsing | Pass explicit allowlist of required fields |

---

## Stage 3 — Retrieval

### 3.1 Retriever — `ext/services/retriever.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 3.1.1 | **P1** | `27` | `embed([query])` happens **before** checking if `selected_kbs` and `chat_id` are both empty | Wastes ~30 ms per message on no-KB chats; adds TEI load | `if not selected_kbs and chat_id is None: return []`; then embed |
| 3.1.2 | **P1** | `32-37, 39-45` | `except Exception: return []` silently hides Qdrant errors | Retrieval "no hits" indistinguishable from infra outage; user sees an empty-context answer and thinks "KB has nothing" | `logger.exception` at minimum; consider raising so caller can 503 |
| 3.1.3 | **P1** | `47-49` | `asyncio.gather` with N tasks, no concurrency cap | 50 KBs selected → 50 parallel Qdrant calls; on production Qdrant can trigger per-client throttle | Bound with `asyncio.Semaphore(8)` or use `aiostream.stream.amerge` with cap |
| 3.1.4 | **P1** | `54` | Sort by raw `h.score` across KBs — cosine scores are NOT comparable across different corpora/collections with different distributions | Bigger KBs with more diverse content dominate the top-k even when a smaller KB has a near-perfect match | Delegate sorting to the reranker only; leave this step as flat concat |
| 3.1.5 | **P1** | — | No deduplication across KBs (same chunk indexed into multiple KBs by accident → duplicated in results) | Citation bloat, budget waste | Dedup by `(payload.doc_id, payload.chunk_index)` before returning |
| 3.1.6 | **P2** | `18-19` | `per_kb_limit=10`, `total_limit=30` are hardcoded default args | Tuning requires redeploy | Expose via `Settings`; pipe through from `rag.py` request body |
| 3.1.7 | **P2** | `39-45` | Chat namespace searched every call when `chat_id is not None`, even if no private docs exist | Unnecessary Qdrant round-trip | Track "has private docs" flag on the chat; skip when 0 |

### 3.2 Reranker — `ext/services/reranker.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 3.2.1 | **P1** | `23` | Fast-path triggers when `top1/top2 > 2.0` — but cosine is in [0,1]; a 0.9 vs 0.4 split triggers it even though 0.4 might be pure noise | A highly-confident wrong answer (embedding collapse) is propagated unchecked | Require `top1 > absolute_threshold` (e.g., 0.7) AND ratio condition; otherwise fall through |
| 3.2.2 | **P1** | `26-37` | Per-KB max-normalization: a KB whose best hit is 0.3 gets normalized to 1.0; it then outranks a 0.85 hit from a big KB (which normalizes to ≈ 0.85/0.95 = 0.89) | Reranker **actively inverts** good results — a small KB with a single weak hit wins vs. a large KB with strong hits | Drop normalization; use a real cross-encoder reranker (e.g., `BAAI/bge-reranker-v2-m3`) OR min-max globally, not per-KB |
| 3.2.3 | **P1** | — | "Tiered reranking" described in `CLAUDE.md §3.3` is not implemented — there is no cross-encoder, no reranker service, no reranker configured in compose | Retrieval quality ceiling is whatever embedding similarity gives you; no second-pass | Add a `reranker_service` (e.g., TEI-rerank image or `sentence-transformers` on CPU); call on top-K raw hits; fallback to score-normalization only when reranker is offline |
| 3.2.4 | **P2** | `28, 33` | `int(h.payload.get("kb_id", -1))` — if payload has a stringified id or None, `int(None)` raises | Any malformed payload crashes rerank | `int_or_default(h.payload.get("kb_id"), -1)` helper |
| 3.2.5 | **P3** | `37` | Tie-break includes `str(h.id)` for determinism but doesn't document it | Stable but confusing for debugging | Add a comment |

### 3.3 Budget — `ext/services/budget.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 3.3.1 | **P1** | `18-33` | Strictly drops from lowest-rank end — if chunk #3 is too big to fit but chunk #4 (smaller) would fit, #4 is *also* dropped because of the `continue` + position-based iteration | Wastes budget tokens; drops useful small chunks | Change to "greedy fit": keep iterating, skip chunks that don't fit individually but consider later smaller ones |
| 3.3.2 | **P1** | `24` | `_count_tokens` re-tokenizes `text` for every hit — no caching | For 30 hits of 200 tokens each, that's 30 encode calls on the event loop | Cache `token_count` on `Hit` during retrieval; or pre-compute in the reranker pass |
| 3.3.3 | **P1** | `7` | Imports `_encoder` (a private member) from `chunker` | Tight coupling; changing chunker encoder breaks budget | Expose a public `get_encoder()` in a shared `tokens.py`; use it in both |
| 3.3.4 | **P1** | `18` | `max_tokens=4000` default — far below modern chat context (32K / 128K); leaves most of the window for RAG empty | Worse answers than necessary; arbitrary "out of budget" drops | Raise default to ~8000; compute dynamically from `chat_model.context_length - chat_history_tokens - system_prompt_tokens` |
| 3.3.5 | **P2** | — | No chunk-level truncation — a 4001-token chunk when budget is 4000 → drop the whole chunk | Binary fail | Truncate long chunks at a sentence boundary when skipping would exceed budget |
| 3.3.6 | **P2** | — | No per-source fairness — one huge KB chunk can starve others | Top-1 always from one KB, answer feels narrow | Round-robin from top-K per KB with a per-KB min quota |

---

## Stage 4 — Context Assembly and LLM Bridge

### 4.1 RAG Bridge — `ext/services/chat_rag_bridge.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 4.1.1 | **P0** | `90` | Hardcoded `chat_id=None` — private chat documents never reach the LLM through the middleware path | The "session-local private docs" feature described in `CLAUDE.md §2.2` is **completely dead** for chat | Extract `chat_id` from the caller (middleware already has `metadata['chat_id']`); plumb through `retrieve_kb_sources(kb_config, query, user_id, chat_id)` |
| 4.1.2 | **P1** | `13-23` | Module-level globals `_vector_store / _embedder / _sessionmaker` — not testable without monkey-patching; not isolated per-request | Hard to hot-reload, hard to inject mocks | Use FastAPI dependencies; or attach services to `request.app.state` in the lifespan and read from there |
| 4.1.3 | **P1** | `98-100` | `except Exception: return []` silently swallows retrieval failure | Answer comes back with no KB context; user has no idea why | Log at `warning`+`exc_info`; surface a one-line notice into `sources` (`{name: "[RAG temporarily unavailable]"}`) so the UX is transparent |
| 4.1.4 | **P1** | `71-73` | RBAC check runs on every single message, synchronously before retrieval | 3 DB queries per message × busy chat = DB hot spot | Cache `get_allowed_kb_ids(user_id)` in Redis with 60 s TTL; key on `user_id`; invalidate on access-grant mutations |
| 4.1.5 | **P1** | `32-42` | `text('SELECT meta FROM chat WHERE id = :cid AND user_id = :uid')` returns JSON; `json.loads(row[0])` on exception path only | Malformed `meta` raises, caught by broad except, returns None; user sees "no KB" with no log clue | Narrow the except; log the parse failure specifically; consider validating the shape against `validate_selected_kb_config` |
| 4.1.6 | **P1** | `93-97` | Hardcoded `per_kb_limit=10, total_limit=30, top_k=10, max_tokens=4000` | Cannot tune without redeploy; no experimentation | Pull defaults from `Settings`; let admin override per-KB |
| 4.1.7 | **P2** | `106-128` | `sources_by_doc` groups chunks by doc but within a doc the order is insertion order (= rerank order), not chunk-index order | LLM sees doc fragments in a confusing order | Either preserve rerank order (current) but include `chunk_index` in each chunk's header for the prompt, OR sort by `chunk_index` asc inside each doc group |
| 4.1.8 | **P2** | `108-111` | `str(hit.payload.get("doc_id", "unknown"))`, `str(hit.payload.get("filename", ...))` — defaults mask ingest bugs | Silent "unknown" docs leak into UX | Assert at ingest that both are set; fail loud in prod |
| 4.1.9 | **P3** | — | No trace/correlation ID threaded through the pipeline | RAG failures can't be tied to a specific user/message | Add `contextvars` for `request_id` + `user_id`; log at each stage boundary |

### 4.2 RAG Router — `ext/routers/rag.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 4.2.1 | **P0** | `77-79` | `select(Chat).where(Chat.id == body.chat_id, Chat.user_id == user.id)` — compat `Chat` maps to `chats` (plural); production has `chat` (singular) | `POST /api/rag/retrieve` **404s in production** | Same fix as 1.5.2 — use raw SQL against `chat`, cast `chat_id` to `str` |
| 4.2.2 | **P1** | `48-52` | `chat_id: int` on request model | Clients must pass int; real upstream uses UUID strings | Change to `str` |
| 4.2.3 | **P2** | `94-101` | Duplicates logic that also exists in `chat_rag_bridge.retrieve_kb_sources` (embed → retrieve → rerank → budget) | Two places to keep in sync; bug fixes land in only one | Have `rag_retrieve` call `chat_rag_bridge.retrieve_kb_sources` internally, or extract a shared `run_rag(...)` helper |
| 4.2.4 | **P2** | `103-113` | No latency / timing info in response | Clients can't detect slow queries or monitor P95 | Return `{"hits": ..., "stage_ms": {"embed": 32, "search": 77, "rerank": 3, "budget": 1}}` |
| 4.2.5 | **P3** | — | No streaming — clients wait for all four phases | Could stream hits as they rank | Server-Sent-Events `text/event-stream` endpoint |

### 4.3 Middleware Patch — `patches/0001-mount-ext-routers.patch`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 4.3.1 | **P1** | `69-99` | Whole RAG injection wrapped in `try/except Exception: log.warning(...)` | Every failure silent to user; no metric, no alert | Emit a structured log event `kb_rag.inject.failed` with fields; raise if `ENV == "test"` |
| 4.3.2 | **P1** | `82` | `form_data.pop('kb_config', None)` mutates the upstream payload | Downstream middleware that might read this field loses it; also makes replay harder | Use `form_data.get('kb_config')` without pop — the upstream doesn't need it anyway |
| 4.3.3 | **P1** | `91-93` | `sources.extend(_kb_sources)` — blindly appends to upstream's sources list | If upstream already retrieved some file-sources (e.g., uploaded files in-chat), you double-count; if not, fine | Dedup by `source.id` before extending |
| 4.3.4 | **P2** | `50-60` | Extension wiring is in one big try/except at module-import time | If env is misconfigured, upstream boots with KB endpoints silently missing; users discover this via 404 | Fail-fast (raise) if `AUTH_MODE=jwt` env is missing but KB routes are wired; or health-check dependency |
| 4.3.5 | **P2** | `32-40` | Migration runner spawns `subprocess` on every lifespan start | Slow startup; weird subprocess errors hard to debug | Use alembic or call migration logic directly in-process |
| 4.3.6 | **P3** | — | `_var_orgchat` prefixed names throughout → unreadable | Patch is hostile to reviewers | Wrap in a function (`_wire_orgchat(app)`) defined once and called once |

---

## Stage 5 — Cross-Cutting: RBAC, Config, App, DB

### 5.1 RBAC — `ext/services/rbac.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 5.1.1 | **P1** | `20-46` | 3 DB queries per `get_allowed_kb_ids` call; called on *every* available-KB list, KB-config set, RAG request, and chat message | Dominates request latency for chatty users | Redis cache, key = `rbac:user:{id}`, TTL 60 s; invalidate on grant/revoke |
| 5.1.2 | **P1** | `20-21` | `SELECT role FROM "user"` — mixes raw SQL with ORM `select(KBAccess...)`; code is half-and-half | Harder to maintain; subtle schema drift | Commit to raw SQL throughout for upstream tables, keep ORM for our own `kb_*` tables only |
| 5.1.3 | **P1** | `26-31` | Admins bypass everything and fetch every non-deleted KB | For a 200-KB org, admin's `/api/kb/available` returns a huge list; no pagination | Paginate via `limit/offset` or cursor |
| 5.1.4 | **P2** | — | No audit log of RBAC decisions | Can't answer "why did user X retrieve from KB Y" | Emit `rbac.allow` / `rbac.deny` structured logs |

### 5.2 App Wiring — `ext/app.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 5.2.1 | **P1** | `16-46` vs `52-94` | `build_app` and `build_ext_routers` duplicate nearly all setup | Two sources of truth for DI; divergence bugs | Factor into `_configure_services(settings) -> Services` dataclass; both entry points call it |
| 5.2.2 | **P1** | — | No FastAPI `lifespan` — services (`vs`, `emb`) never `close()` | Connection leaks on graceful shutdown; log spam on SIGTERM | `@asynccontextmanager async def lifespan(app): yield; await emb.aclose(); await vs.close()` |
| 5.2.3 | **P1** | `32-34` | `/healthz` always returns `ok` — doesn't check Qdrant/TEI/DB | Liveness probe can't distinguish "FastAPI alive but dependencies dead" | Check: `await vs.list_collections()` (with timeout), `await emb._client.get("/")`, simple DB `SELECT 1` — respond 503 if any fail |
| 5.2.4 | **P1** | `36-39` and `86-91` | `kb_admin_ui` reads HTML file **per request** | Small but wasteful | Read once at startup into `BYTES` module var; return a `Response(BYTES, media_type='text/html')` |
| 5.2.5 | **P2** | `49` | `app = None` at module scope | Confusing dead code; some tools may import `.app` expecting a FastAPI instance | Remove or set `app = build_app()` |
| 5.2.6 | **P2** | `17, 65` | `clear_settings_cache()` on every wiring call | Invalidates the lru_cache; fine but masks the fact that `get_settings` is memoized | Only clear in tests |

### 5.3 Config — `ext/config.py`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 5.3.1 | **P1** | `14-26` | No validation of `database_url`/`qdrant_url`/`tei_url` format | Bad URL → cryptic error at first connection, not at boot | Add `pydantic.AnyUrl` validators; surface a clear error on startup |
| 5.3.2 | **P1** | `18-21` | `async_database_url` only handles `postgresql://` → `postgresql+asyncpg://`; doesn't consider `+psycopg`/`+aiopg` | Silent downgrade to asyncpg even if user intended otherwise | Parse with `sqlalchemy.engine.make_url` and inspect the dialect |
| 5.3.3 | **P2** | `10-11` | `env_file=None` — settings read only from process env | Can't use a `.env` file in dev without manually exporting | Set `env_file=".env"` in dev profile |
| 5.3.4 | **P2** | — | No central place to tune RAG knobs (batch sizes, top_k, thresholds) — they're hardcoded in various services | Hard to tune without editing multiple files | Add `Settings.rag_per_kb_limit`, `rag_total_limit`, `rag_top_k`, `rag_max_tokens`, `rag_score_threshold`, `embed_batch_size` |

### 5.4 DB Layer — `ext/db/*`

| # | Sev | Lines | Problem | Impact | Fix |
|---|---|---|---|---|---|
| 5.4.1 | **P0** | `ext/db/models/compat.py:19,42` | Compat ORM defines `users`, `chats` (plural) but production runs against upstream's `user`, `chat` (singular) | Every ORM query via compat returns empty in prod → silent 404s (already called out in 1.5.2, 4.2.1); `User` / `Chat` / `Group` are **dead code in production** | Delete compat module for production profiles; keep under `tests/` only; adjust routers to always use raw SQL or an ORM bound to real tables |
| 5.4.2 | **P1** | `ext/db/migrations/001_create_kb_schema.sql:15, 40, 50` | FKs point at `users(id)` (BIGINT), but upstream's real `user.id` is a UUID string | Migration will **fail to apply** on a real upstream DB (type mismatch) | Change column type to `VARCHAR(255)` and drop the `REFERENCES users(id)` constraint (or use `REFERENCES "user"(id)`) |
| 5.4.3 | **P1** | `ext/db/migrations/001_create_kb_schema.sql:50-51` | `kb_access.user_id BIGINT` and `group_id BIGINT` — but ORM and DTOs use `Optional[str]` UUIDs | Same mismatch; grants by string UUID will explode at insert | Make both `VARCHAR(255)` |
| 5.4.4 | **P1** | `ext/db/migrations/001_create_kb_schema.sql:62-64` | `ALTER TABLE chats ADD COLUMN ... selected_kb_config JSONB` — but upstream stores this inside `chat.meta` JSONB; separate column is dead | Two places to read/write KB config; the bridge already reads from `meta` (not `selected_kb_config`) | Drop the migration's `ALTER TABLE chats` section; standardize on `chat.meta.kb_config` |
| 5.4.5 | **P1** | `ext/db/models/kb.py:77-99` | `KBAccess.user_id` is `String(255)` (UUID) but FK in SQL is `BIGINT REFERENCES users(id)` (see 5.4.3) | ORM and migrations are inconsistent | Align: drop FK or change column types |
| 5.4.6 | **P1** | `ext/db/migrations/001_create_kb_schema.sql:47-59` | No composite index `kb_access(user_id, kb_id)` — RBAC lookup does `WHERE user_id = ... OR group_id IN (...)` | Scans can grow expensive with 10k+ grants | Add `CREATE INDEX ON kb_access (user_id, kb_id)` and `(group_id, kb_id)` |
| 5.4.7 | **P2** | `ext/db/models/kb.py:64` | `bytes` column — stored but never used | Dead field | Either display it in admin UI or drop |
| 5.4.8 | **P2** | `ext/db/session.py:8-9` | `create_async_engine(url, pool_pre_ping=True)` — no `pool_size`/`max_overflow` | Default pool is 5/10; under concurrent chat load will queue | Set `pool_size=10, max_overflow=20, pool_recycle=3600` |

---

## Stage 6 — Architectural & Scalability Gaps

| # | Sev | Area | Problem | Fix |
|---|---|---|---|---|
| 6.1 | **P1** | Caching | No caching layer anywhere — RBAC, embeddings, retrievals, collection-existence all hit backends every call | Introduce Redis with clear namespacing: `rbac:user:{id}`, `emb:sha256:{hash}`, `qdrant:known_collections` |
| 6.2 | **P1** | Background jobs | Uploads are synchronous HTTP — admin must keep connection open for minutes during bulk imports | ARQ or RQ worker; upload returns `202` + `doc_id`; `/status` endpoint polls |
| 6.3 | **P1** | Observability | No metrics, no distributed tracing, no structured logs — violates `CLAUDE.md §8.4 Observability` which lists Prometheus metrics by name but none are implemented | Add `prometheus_fastapi_instrumentator`; emit counters/histograms per the spec (e.g., `rag_retrieval_latency_seconds`) |
| 6.4 | **P1** | Health & Readiness | `/healthz` doesn't actually check dependencies | Separate `/livez` (cheap) from `/readyz` (full dependency probe) |
| 6.5 | **P1** | Reranker | Real cross-encoder reranker is specced but not implemented — rerank quality is capped by embedding similarity | Deploy bge-reranker-v2-m3 as a third GPU tenant or on CPU; wire through a `RerankerClient` |
| 6.6 | **P2** | Dedup | No file-hash-based dedup at upload time | Hash on receive; 409 Conflict if duplicate |
| 6.7 | **P2** | Private-doc cleanup | `chat_{id}` collection is never deleted when a chat is deleted | Chat-delete hook → `vs.delete_collection(f"chat_{id}")` |
| 6.8 | **P2** | Pagination | `/api/kb/available`, `/api/kb/{id}/documents`, `/api/kb/{id}/access` return unbounded lists | Add `limit`, `offset`/`cursor`; cap at 100 |
| 6.9 | **P2** | DI / testability | Module-level `_SM`, `_VS`, `_EMB`, `_sessionmaker` globals in routers and bridge | FastAPI `Depends(get_vector_store)` etc; attach to `app.state` in lifespan |
| 6.10 | **P2** | Rate-limiting | No per-user rate limit on `/api/rag/retrieve`, `/upload`, `/available` | `slowapi` with Redis backend |
| 6.11 | **P3** | Dead code | `app = None` in `ext/app.py`; `User`/`Group`/`UserGroup`/`Chat` compat models unused in prod; `bytes` column; `StubEmbedder` used only in tests but lives in prod package | Move to `tests/` or guard behind `TESTING=1` |
| 6.12 | **P3** | Evaluation harness | No golden-query benchmark → cannot regression-test retrieval quality | Add `tests/eval/queries.yaml` with expected top-1 doc; compute MRR@10 on each CI run |

---

## Stage 7 — Retrieval-Quality Improvements (non-bug, high ROI)

These aren't bugs, but address the design claim of "better results" in `CLAUDE.md §2`:

1. **Hybrid search** — pure dense fails on proper nouns, acronyms, exact quotes. Add BM25 (Qdrant supports it natively) and RRF-fuse with dense results. Large lift on KB-style queries.
2. **Multi-vector bge-m3** — bge-m3 supports dense + sparse + multi-vector. You're only using dense. Switch to TEI's multi-vector mode or run `FlagEmbedding` directly.
3. **Query rewriting** — for conversational follow-ups ("tell me more"), the query is not self-contained; rewrite it using chat history before embedding.
4. **Query expansion** — for short queries ("login bug"), expand to ("login authentication session-cookie bug") with an LLM pre-pass or a synonym dictionary per KB.
5. **MMR (Maximal Marginal Relevance)** — your top-K may be 10 near-duplicates of the same paragraph. Add MMR reranking with λ≈0.5 for diversity.
6. **Citation grounding** — model often generates hallucinated citations. Post-process the response against `sources` and strike any `[doc_id]` the model uses that isn't in the retrieved set.
7. **Context window prompt structure** — sources are currently grouped by doc but prepended raw. Use a structured template:

   ```
   # Retrieved Context
   ## Source 1 — {filename} (KB: {kb_name}, Subtag: {subtag_name})
   [chunk_idx=3, score=0.87]
   {chunk_text}
   ...
   ```

   Better citation, clearer to the LLM.

---

## Stage 8 — Suggested Order of Attack

**Phase A — Correctness (days 1-3)**
- 5.4.1, 5.4.2, 5.4.3, 5.4.4, 5.4.5 — fix DB schema / compat layer (the one that causes production 404s)
- 1.5.2, 1.5.3, 4.2.1, 4.2.2 — stop using compat `Chat` for prod routes
- 1.4.1 — chunk IDs keyed on doc so deletion actually works
- 4.1.1 — plumb `chat_id` through the bridge so private docs work
- 1.5.1 — stream uploads / cut off at size boundary

**Phase B — Quality (days 4-7)**
- 1.3.1 — bge-m3 query/passage handling
- 3.2.2, 3.2.3 — real reranker + fix broken normalization
- 3.1.1, 3.1.2 — retriever early-exit + non-silent errors
- 2.1.3, 2.1.5 — payload indexes + soft-delete filter in Qdrant
- 3.3.4 — raise token budget

**Phase C — Scale (days 8-14)**
- 1.3.2, 1.3.3, 1.4.2 — embedding batch/retry/pipeline
- 6.2 — background ingest worker
- 6.1, 4.1.4 — Redis caching for RBAC + embeddings + collection existence
- 6.3 — Prometheus / structured logs end-to-end
- 5.2.2, 5.2.3 — lifespan + real health checks
- 6.7 — chat deletion hooks → Qdrant cleanup

**Phase D — Polish (week 3+)**
- 7.x — hybrid search, query rewriting, MMR
- 6.12 — eval harness in CI
- 1.1.x — richer extractors (tables, OCR, pptx)

---

## Appendix — Files Audited

```
ext/app.py
ext/config.py
ext/_lazy_user_lookup.py
ext/db/base.py
ext/db/session.py
ext/db/models/__init__.py
ext/db/models/kb.py
ext/db/models/chat_ext.py
ext/db/models/compat.py
ext/db/migrations/001_create_kb_schema.sql
ext/db/migrations/002_soft_delete_kb.sql
ext/db/migrations/003_add_chunk_count.sql
ext/routers/kb_admin.py
ext/routers/kb_retrieval.py
ext/routers/upload.py
ext/routers/rag.py
ext/services/auth.py
ext/services/jwt_verifier.py
ext/services/rbac.py
ext/services/kb_service.py
ext/services/extractor.py
ext/services/chunker.py
ext/services/embedder.py
ext/services/ingest.py
ext/services/vector_store.py
ext/services/retriever.py
ext/services/reranker.py
ext/services/budget.py
ext/services/chat_rag_bridge.py
patches/0001-mount-ext-routers.patch
patches/0002-kb-selector-frontend.patch
compose/docker-compose.yml (relevant sections)
compose/.env.example
tests/unit/test_chunker.py, test_reranker.py, test_budget.py, test_extractor.py (spot checks)
```

*No source files were modified; this document is the sole output of the audit.*
