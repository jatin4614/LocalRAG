# KB-Based RAG Pipeline — Workflow & Architecture Overview

**Date:** 2026-04-16  
**Status:** Design Phase  
**Author:** System Design  
**Related:** `2026-04-12-org-chat-assistant-design.md` (original spec)

---

## 1. Executive Summary

This document describes the **new KB-centric RAG architecture** that replaces the flat collection-based model. Key changes:

- **Hierarchical KBs:** Admins create Knowledge Bases (KBs) with up to 2 levels of organization (subtags within each KB)
- **User-Scoped Retrieval:** Users explicitly select which KBs/subtags to retrieve from at chat start (optional)
- **Session-Local Private Docs:** Users can upload private docs/audio/images per-chat; these become a temporary knowledge base scoped to that chat only
- **Performance + Clarity:** Reduces vector search scope (faster, cheaper) while giving users explicit control over context

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Layer                              │
│  (Web UI: KB Selection → Chat Start → Message + Optional Uploads)│
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────┴───────────┐
        │                      │
        ▼                      ▼
┌───────────────────┐  ┌────────────────────┐
│  Admin Interface  │  │  Chat Interface    │
│  (KB Management)  │  │  (User Chat UX)    │
└────────┬──────────┘  └──────────┬─────────┘
         │                        │
         │                        │
    ┌────▼────────────────────────▼────────────┐
    │        Open WebUI (Orchestration)         │
    │  - Route KB uploads to workers            │
    │  - Handle chat session state              │
    │  - Enforce RBAC (kb_access checks)        │
    │  - Coordinate RAG pipeline                │
    └────┬───────────┬───────────┬──────────────┘
         │           │           │
    ┌────▼───┐  ┌────▼──────┐  ┌▼──────────────┐
    │ Redis  │  │ PostgreSQL │  │ Background   │
    │ (Jobs) │  │ (KB Schema)│  │ Worker Pool  │
    └────────┘  └───────────┘  └┬──────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                ┌───▼────┐  ┌───▼──────┐  ┌──▼─────┐
                │  TEI   │  │ Qdrant   │  │Whisper/│
                │(Embeds)│  │(Vectors) │  │Vision  │
                └────────┘  └──────────┘  └────────┘
```

---

## 3. Data Model Overview

### 3.1 Knowledge Base Schema

**Three new tables replace/extend existing `collections` model:**

#### `knowledge_bases`
```sql
CREATE TABLE knowledge_bases (
  id BIGSERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  admin_id BIGINT NOT NULL REFERENCES users(id),
  created_at TIMESTAMP DEFAULT now()
);
-- Example: id=5, name="Comn", admin_id=1
```

#### `kb_subtags`
```sql
CREATE TABLE kb_subtags (
  id BIGSERIAL PRIMARY KEY,
  kb_id BIGINT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT now(),
  UNIQUE(kb_id, name)  -- One "OFC" per KB
);
-- Example: kb_id=5, id=12, name="OFC"
--          kb_id=5, id=13, name="Comn reports"
```

#### `kb_documents`
```sql
CREATE TABLE kb_documents (
  id BIGSERIAL PRIMARY KEY,
  kb_id BIGINT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  subtag_id BIGINT NOT NULL REFERENCES kb_subtags(id) ON DELETE CASCADE,
  filename VARCHAR(512) NOT NULL,
  mime_type VARCHAR(100),
  bytes BIGINT,
  ingest_status VARCHAR(20) DEFAULT 'pending',  -- pending, chunking, embedding, done, failed
  error_message TEXT,
  uploaded_at TIMESTAMP DEFAULT now(),
  uploaded_by BIGINT NOT NULL REFERENCES users(id)
);
-- Example: kb_id=5, subtag_id=12, filename="Q2-roadmap.pdf"
```

#### `kb_access` (RBAC)
```sql
CREATE TABLE kb_access (
  id BIGSERIAL PRIMARY KEY,
  kb_id BIGINT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
  group_id BIGINT REFERENCES groups(id) ON DELETE CASCADE,
  access_type VARCHAR(20) DEFAULT 'read',  -- 'read', 'write' (future)
  granted_at TIMESTAMP DEFAULT now(),
  CHECK (user_id IS NOT NULL OR group_id IS NOT NULL)
  -- Exactly one of user_id or group_id must be non-null
);
-- Example: kb_id=5, group_id=2 (grant "Comn" KB to "engineering" group)
--          kb_id=7, user_id=10 (grant "Leadership" KB to user 10)
```

### 3.2 Chat & Session Schema (Modified)

#### `chats` (new field)
```sql
ALTER TABLE chats ADD COLUMN selected_kb_config JSONB;

-- Example value:
-- [
--   {
--     "kb_id": 5,
--     "subtag_ids": [12, 13]   // "OFC" + "Comn reports"
--   },
--   {
--     "kb_id": 7,
--     "subtag_ids": []  // All subtags in Engineering KB
--   }
-- ]
-- 
-- Empty subtag_ids = "retrieve from all subtags in this KB"
-- NULL selected_kb_config = "no KB selected, only private docs"
```

#### `messages` (unchanged)
```sql
-- Existing messages table
-- Attachments field can include private doc references
-- Private docs are embedded into session_{chat_id} namespace (Qdrant)
```

### 3.3 Qdrant Vector Store Namespacing

**Hybrid Approach: One Collection per KB + Metadata Filtering**

To avoid namespace explosion (100 KBs × 20 subtags = 2000 collections), use a **hybrid model**:

```
Qdrant Collections: kb_{kb_id}  ← One per KB (not per subtag)

Example:
  kb_5      ← Comn KB (all subtags inside)
  kb_7      ← Engineering KB (all subtags inside)
  kb_12     ← HR KB (all subtags inside)
  ...
  session_{chat_id}  ← Ephemeral, deleted when chat ends
```

**Why hybrid?**
- **100 KBs = 100 collections** (operationally manageable)
- **NOT** 100 × 20 = 2000 collections (performance nightmare)
- Metadata filtering in Qdrant is fast (built-in optimization)
- Trade-off: ~5-10% slower query vs. vastly simpler operations

**Qdrant Point Payload:**
```json
{
  "kb_id": 5,
  "subtag_id": 12,
  "doc_id": 42,
  "filename": "Q2-roadmap.pdf",
  "chunk_index": 0,
  "chunk_text": "Q2 focus: infrastructure modernization...",
  "uploaded_at": "2026-04-16T10:30:00Z",
  "source": "kb"  // or "session" for private docs
}
```

**Qdrant Search with Filtering:**
```python
# Search kb_5 but only subtag_12
Qdrant.search(
    collection="kb_5",
    vector=query_embedding,
    limit=5,
    filter={
        "field": "subtag_id",
        "match": {"value": 12}
    }
)
```

---

## 4. RAG Pipeline — 8-Stage Workflow

### **Stage 1: KB Upload & Document Registration**

**Trigger:** Admin uploads document to KB / User uploads private doc in chat

**Admin KB Upload:**
```
POST /api/kb/{kb_id}/subtag/{subtag_id}/upload
  Body: {file, filename}

Backend:
  1. Validate user is admin
  2. Insert into kb_documents:
     {kb_id, subtag_id, filename, ingest_status: "pending"}
  3. Store file to disk / object storage
  4. Enqueue Redis job: {doc_id, kb_id, subtag_id, file_path}
  5. Return 202 Accepted with job_id
```

**User Private Doc Upload (in chat):**
```
POST /api/chat/{chat_id}/upload
  Body: {file, filename}

Backend:
  1. Validate user owns chat
  2. Store file temporarily
  3. Enqueue Redis job: {chat_id, file_path, type: "session"}
  4. Return 202 Accepted
```

---

### **Stage 2: Text Extraction & Chunking**

**Worker (background process) picks up ingest job:**

```
Job: {doc_id, kb_id, subtag_id, file_path, type}

Worker:
  1. Load file from disk
  2. Extract text:
     - PDF: pdf2image → OCR (if scanned) or pdfplumber (if text)
     - DOCX: python-docx
     - TXT/MD: raw read
     - XLSX: tabula or openpyxl
  
  3. Semantic chunking:
     - Split text into logical chunks (~800 tokens)
     - Overlap: 100 tokens (preserve context between chunks)
     - Preserve document structure (headers, lists, etc.)
  
  4. Update kb_documents.ingest_status = "chunking"
```

**Output:**
```python
chunks = [
  {
    "text": "Q2 focus: infrastructure modernization...",
    "index": 0,
    "metadata": {kb_id, subtag_id, doc_id, filename}
  },
  {
    "text": "This includes network upgrades and...",
    "index": 1,
    "metadata": {...}
  },
  ...
]
```

---

### **Stage 3: Embedding (Vector Generation)**

**For each chunk, call TEI (Text Embeddings Inference):**

```
Worker:
  For each chunk in chunks:
    POST http://tei:80/v1/embeddings
      {
        "model": "BAAI/bge-m3",
        "input": chunk["text"]
      }
    
    Response: {embedding: [1024 floats]}
    
    chunk["embedding"] = embedding
```

**Why TEI?**
- OpenAI-compatible API (`/v1/embeddings`)
- Fast CPU or GPU-accelerated
- No external API calls (air-gapped)
- Deterministic (same text → same embedding)

**Update status:**
```
kb_documents.ingest_status = "embedding"
```

---

### **Stage 4: Vector Store Upsert (Qdrant)**

**Upsert all chunks into correct Qdrant namespace:**

```
Namespace: kb_{kb_id}_sub_{subtag_id}

For each chunk:
  Qdrant.upsert(
    collection: f"kb_{kb_id}_sub_{subtag_id}",
    points: [
      {
        id: f"doc_{doc_id}_chunk_{idx}",
        vector: chunk["embedding"],  // 1024-dim
        payload: {
          kb_id: chunk.metadata["kb_id"],
          subtag_id: chunk.metadata["subtag_id"],
          doc_id: chunk.metadata["doc_id"],
          chunk_text: chunk["text"],
          chunk_index: chunk["index"],
          filename: chunk.metadata["filename"],
          uploaded_at: now(),
          source: "kb"  // or "session" for private
        }
      }
    ]
  )

Final status:
  kb_documents.ingest_status = "done"
```

**Result:** Chunks are now searchable in Qdrant, isolated by KB/subtag namespace.

---

### **Stage 5: RBAC Check (Before Retrieval) — Optimized with Caching**

**Optimization: Cache Allowed KBs at Chat Start (Reduce DB Hits by 90%)**

Problem Solved: Repeated DB queries on every message = latency + cost overhead.

**At Chat Start (First Time):**
```
User starts chat:
  1. Fetch allowed KB IDs from DB:
     SELECT DISTINCT kb_id FROM kb_access
     WHERE user_id = 10 OR group_id IN (2, 3)
     → allowed_kb_ids = [5, 7, 12, ...]
  
  2. Store in Redis (session cache):
     cache.set(f"user:10:allowed_kbs", allowed_kb_ids, ttl=3600)
     (TTL: 1 hour = reasonable for org KB permission changes)
  
  3. Store in chat object:
     chat.allowed_kb_ids = allowed_kb_ids
```

**On Every Message (No DB Hit):**
```
Current user: user_id = 10, groups = [2, 3]
Selected KBs in chat: [
  {kb_id: 5, subtag_ids: [12, 13]},
  {kb_id: 7, subtag_ids: []}
]

Backend validation (from cache):
  allowed_kb_ids = cache.get(f"user:10:allowed_kbs")  ← Redis hit (O(1))
  
  For each {kb_id, subtag_ids} in selected_kb_config:
    if kb_id not in allowed_kb_ids:  ← Simple O(1) membership check
      Return 403 Forbidden ("No access to KB")
      Reject the message
  
  If all KBs pass: proceed to retrieval
```

**Cost Reduction:**
- First message: 1 DB query (cache miss)
- Messages 2-3600: 0 DB queries (cache hits)
- **Result: ~90% fewer DB hits per user session**

---

### **Stage 6: Similarity Search & Retrieval (Parallel + Optimized)**

**User sends message: "What's the Q2 roadmap?"**

**Optimization: Parallel Batch Search (Avoid Fan-Out Latency)**

Instead of sequential searches (5 KBs × 5 subtags = 25 serial queries = 2500ms latency), use async parallel execution:

```
1. Embed the query:
   POST http://tei:80/v1/embeddings
     {model: "BAAI/bge-m3", input: "What's the Q2 roadmap?"}
   
   query_embedding = [1024 floats]

2. Build parallel search tasks:
   
   tasks = []
   For each {kb_id, subtag_ids} in selected_kb_config:
     if subtag_ids is empty:
       # Search ALL subtags in this KB
       for each subtag in kb_subtags where kb_id:
         tasks.append(
           Qdrant.search_async(
             collection: f"kb_{kb_id}",
             vector: query_embedding,
             filter: {"subtag_id": subtag.id},  ← Metadata filter (not namespace)
             limit: 5
           )
         )
     else:
       # Search only selected subtags
       for subtag_id in subtag_ids:
         tasks.append(
           Qdrant.search_async(
             collection: f"kb_{kb_id}",
             vector: query_embedding,
             filter: {"subtag_id": subtag_id},  ← Metadata filter
             limit: 5
           )
         )

   # Add session-local private doc search
   tasks.append(
     Qdrant.search_async(
       collection: f"session_{chat_id}",
       vector: query_embedding,
       limit: 3
     )
   )

3. Execute all searches in parallel:
   all_chunks = await asyncio.gather(*tasks)
   # All 25+ queries execute concurrently (~100ms total, not 2500ms)
```

**Result:** List of chunks ranked by cosine similarity, retrieved ~25× faster via parallel execution.

---

### **Stage 7: Reranking & Cross-KB Normalization (Tiered Strategy)**

**Problem Solved:**
1. Scores from different KB/subtag namespaces aren't globally calibrated
2. Reranker adds latency; becomes a bottleneck
3. Cross-KB relevance ranking is weak

**Solution: Tiered Reranking (Optional but Smart)**

```
all_chunks = results from all KB/subtag searches
(Already sorted by Qdrant cosine similarity)

TIER 1: Fast Path (No Reranker)
  ───────────────────────────────────
  If latency < 100ms AND top_chunk.score > 0.85:
    Return top 8 chunks as-is
    ✓ Fast path (50ms total), high confidence
    Skip reranker overhead

TIER 2: Smart Normalization (Cross-KB Calibration)
  ────────────────────────────────────────────
  Else if multiple KBs selected:
    # Normalize scores across KBs (different namespaces have different score ranges)
    by_kb = group_chunks_by_kb(all_chunks)
    
    for kb_id in by_kb:
      max_score = max(chunk.score for chunk in by_kb[kb_id])
      min_score = min(chunk.score for chunk in by_kb[kb_id])
      
      for chunk in by_kb[kb_id]:
        chunk.normalized_score = (chunk.score - min_score) / (max_score - min_score + 1e-6)
    
    Sort all chunks by normalized_score
    Take top 8
    ✓ Medium path (100ms), globally ranked

TIER 3: Heavyweight Reranker (When Uncertain)
  ────────────────────────────────────────────
  Else (low confidence, many KBs):
    POST http://reranker:8000/rerank
      {
        query: "What's the Q2 roadmap?",
        documents: [chunk.text for chunk in all_chunks],
        top_k: 8
      }
    
    Reranked chunks = top 8 after global reranking
    ✓ Slow path (200ms), best quality
    Use lightweight reranker: bge-reranker-base or Qdrant RRF
```

**Deduplication:**
```
final_chunks = []
seen_docs = set()

For chunk in final_chunks_from_tiered_strategy:
  if (chunk.doc_id, chunk.chunk_index) not in seen_docs:
    final_chunks.append(chunk)
    seen_docs.add((chunk.doc_id, chunk.chunk_index))
    if len(final_chunks) >= 8:
      break
```

**Result:** Reranker is optional but available; fast queries don't pay latency cost; uncertain queries get better ranking.

---

### **Stage 8: Prompt Injection & LLM Inference (Token-Budgeted)**

**Problem Solved:** Appending all chunks without limit → token overflow → cost explosion.

**Solution: Strict Token Budgeting with Graceful Truncation**

```
MAX_PROMPT_TOKENS = 2048
RESERVED_FOR_RESPONSE = 512
AVAILABLE_FOR_CONTEXT = MAX_PROMPT_TOKENS - RESERVED_FOR_RESPONSE  # 1536 tokens

Build system prompt with RAG context:

system_prefix = """You are a helpful assistant for {org_name}.

Use the following context to answer the user's question. 
If the context doesn't contain relevant information, say so.

Context:
"""

system_tokens = count_tokens(system_prefix)
remaining_budget = AVAILABLE_FOR_CONTEXT - system_tokens  # ~1200 tokens left

context_parts = []
for chunk in final_chunks:
  chunk_str = f"- [{chunk.filename}] {chunk.chunk_text}\n"
  chunk_tokens = count_tokens(chunk_str)
  
  if chunk_tokens > remaining_budget:
    # Stop adding chunks, truncate gracefully
    context_parts.append("... [truncated for brevity]\n")
    logger.warn({
      "event": "context_truncated",
      "chunks_added": len(context_parts),
      "total_chunks": len(final_chunks),
      "tokens_used": count_tokens("\n".join(context_parts))
    })
    break
  
  context_parts.append(chunk_str)
  remaining_budget -= chunk_tokens

system_prompt = system_prefix + "\n".join(context_parts)
user_prompt = f"User question: {user_message}"

total_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)
assert total_tokens < MAX_PROMPT_TOKENS, f"Prompt overflow: {total_tokens} > {MAX_PROMPT_TOKENS}"

Send to vllm-chat:
  POST http://vllm-chat:8000/v1/chat/completions
    {
      model: "Qwen/Qwen2.5-14B-Instruct-AWQ",
      messages: [
        {role: "system", content: system_prompt},
        {role: "user", content: user_prompt}
      ],
      max_tokens: 512,  ← Reserved budget
      temperature: 0.7
    }

Response: LLM-generated answer with RAG context grounding
```

**Cost Control:**
- Predictable token usage (no surprises)
- Graceful truncation (best chunks included first)
- Cost per query capped at token budget

---

## 5. User Workflows

### 5.1 Chat Start Workflow

```
User opens Web UI → "New Chat" button

┌─────────────────────────────────────────┐
│ KB Selection Screen (Optional)           │
├─────────────────────────────────────────┤
│ Available KBs:                          │
│ ☐ Comn                                  │
│   ☐ OFC                                 │
│   ☐ Comn reports                        │
│   ☐ Outages                             │
│ ☐ Engineering                           │
│   ☐ (all subtags shown)                 │
│ ☐ HR                                    │
│   ☐ Policies                            │
│   ☐ Benefits                            │
│                                         │
│ [Select KBs] [Skip & Chat Without KB]   │
└─────────────────────────────────────────┘

User either:
  A. Selects KBs → [Select KBs] button → chat starts
  B. Skips → [Skip & Chat Without KB] → chat starts with NULL selected_kb_config

Chat created with selected_kb_config stored in DB
```

**Chat State in DB:**
```sql
INSERT INTO chats (user_id, selected_kb_config, model, created_at)
VALUES (
  10,
  '[{"kb_id": 5, "subtag_ids": [12, 13]}, {"kb_id": 7, "subtag_ids": []}]',
  'Qwen2.5-14B',
  now()
);
```

---

### 5.2 Message & RAG Retrieval Workflow

```
User types: "What's in the Q2 roadmap?"

┌────────────────────────────────────────────┐
│ 1. Message arrives at API                  │
│    POST /api/chat/{chat_id}/messages       │
│    Body: {content: "What's...?"}           │
└───────────┬────────────────────────────────┘
            │
┌───────────▼────────────────────────────────┐
│ 2. Load chat + check RBAC                  │
│    Fetch selected_kb_config from chat      │
│    Validate user has access to each KB     │
└───────────┬────────────────────────────────┘
            │
┌───────────▼────────────────────────────────┐
│ 3. Embed query (TEI)                       │
│    "What's in the Q2 roadmap?"             │
│    → query_embedding (1024 dims)           │
└───────────┬────────────────────────────────┘
            │
┌───────────▼────────────────────────────────┐
│ 4. Search Qdrant                           │
│    Query kb_5_sub_12, kb_5_sub_13, kb_7_* │
│    + session_{chat_id} (private docs)      │
│    → top 8-10 chunks                       │
└───────────┬────────────────────────────────┘
            │
┌───────────▼────────────────────────────────┐
│ 5. Rerank (optional)                       │
│    Order by relevance                      │
│    → final_chunks                          │
└───────────┬────────────────────────────────┘
            │
┌───────────▼────────────────────────────────┐
│ 6. Call vLLM with RAG context              │
│    System prompt: "Use context: ..."       │
│    → LLM response                          │
└───────────┬────────────────────────────────┘
            │
┌───────────▼────────────────────────────────┐
│ 7. Store message + stream response         │
│    INSERT messages (chat_id, role, content)│
│    Stream response to user                 │
└────────────────────────────────────────────┘
```

---

### 5.3 Private Document Upload (Per-Session)

```
User in chat → Click "Upload" → selects file

┌─────────────────────────────────────┐
│ Is file:                            │
│ - Document (text/image)?            │
│ - Audio?                            │
│ - Image?                            │
└────────┬────────────────────────────┘
         │
    ┌────┴────┬─────────┬──────────┐
    │          │         │          │
    ▼          ▼         ▼          ▼
  Text      Image     Audio    Vision
  File      File      File     Model
    │          │         │          │
    ├──────────┴─────────┼──────────┤
    │                    │          │
    ▼                    ▼          ▼
Extract          Whisper         Vision
Text           Transcribe       Process
    │                │            │
    └────────────────┼────────────┘
                     │
              ┌──────▼──────┐
              │ TEI Embed   │
              │ Chunks      │
              └──────┬──────┘
                     │
          ┌──────────▼──────────┐
          │ Qdrant Upsert       │
          │ session_{chat_id}   │
          └─────────────────────┘
```

**Result:** Private docs now queryable in this chat, invisible to other chats/users.

---

### 5.4 Admin KB Management Workflow

```
Admin logs in → "Knowledge Base Management" panel

┌──────────────────────────────────────┐
│ Create KB                            │
├──────────────────────────────────────┤
│ Name: "Comn"                         │
│ Description: "Communication docs"   │
│ [Create]                            │
└──────────────────────────────────────┘

→ KB created (id=5)

┌──────────────────────────────────────┐
│ Add Subtags to "Comn"               │
├──────────────────────────────────────┤
│ Subtag: "OFC"                        │
│ [Add]                                │
│                                      │
│ Subtag: "Comn reports"              │
│ [Add]                                │
│                                      │
│ Subtag: "Outages"                   │
│ [Add]                                │
└──────────────────────────────────────┘

→ Subtags created (id=12, 13, 14)

┌──────────────────────────────────────┐
│ Grant Access                         │
├──────────────────────────────────────┤
│ Grant "Comn" KB to:                 │
│ ☐ User: alice@org.com               │
│ ☐ Group: "comn-team"               │
│ [Grant]                             │
└──────────────────────────────────────┘

→ kb_access rows created
```

---

## 6. Isolation & Security

### 6.1 Three-Layer Isolation Enforcement

**Layer 1: RBAC Gate (API)**
```python
# Before any retrieval
for kb in user_selected_kbs:
    access = kb_access.filter(
        kb_id=kb.id,
        user_id__in=[current_user.id] + current_user.group_ids
    ).first()
    if not access:
        raise PermissionDenied(f"No access to KB {kb.id}")
```

**Layer 2: Namespace Isolation (Qdrant)**
```python
# Each KB/subtag is a separate Qdrant collection
# User can only query collections they were granted access to
Qdrant.search(
    collection=f"kb_{kb_id}_sub_{subtag_id}",  # Scoped namespace
    vector=query_embedding,
    limit=k
)
```

**Layer 3: Post-Check (API)**
```python
# Verify all returned chunks are from allowed KBs
for chunk in retrieved_chunks:
    assert chunk.payload.kb_id in allowed_kb_ids, \
        f"Unexpected chunk from KB {chunk.payload.kb_id}"
```

### 6.2 Session-Local Private Isolation

```
Chat A (user=10): session_A has private docs
Chat B (user=10): session_B has different private docs
Chat C (user=20): session_C is isolated from user 10's chats

→ No cross-session leakage
→ Private docs auto-deleted when chat archived/deleted
```

---

## 7. Key Architectural Decisions

| Decision | Rationale |
|---|---|
| **Two-level hierarchy** | Balances flexibility (subtags) with simplicity (not arbitrary depth) |
| **Namespace-per-KB/subtag** | Enables fast filtering; avoids O(n) searches across all docs |
| **Session-local private KB** | Users get temporary knowledge base per chat; auto-cleanup |
| **RBAC on KBs, not documents** | Cleaner permission model; reduce DB rows (grant to group vs. per-doc) |
| **Explicit KB selection** | Reduces retrieval scope; improves LLM answer quality; cost control for admins |
| **Qdrant over single vector DB** | Multiple namespaces enable isolation without row-level filtering in vectors |

---

## 8. Performance Implications

### 8.1 Admin Perspective (Cost Savings)

**Before (old collection model):**
```
User query → Qdrant searches ALL collections user has access to
→ 10,000 chunks searched
→ High latency, high compute cost
```

**After (KB-scoped model):**
```
User selects "Comn" KB only
→ Qdrant searches kb_5_sub_*
→ 500 chunks searched
→ 5x faster, 5x cheaper
```

### 8.2 User Perspective (Context Quality)

**Before:**
```
Query "Q2 roadmap" → RAG pulls chunks from:
  - Engineering KB
  - HR KB
  - Comn KB
  - Finance KB
  → Mixed results, LLM confused
```

**After:**
```
User selects "Engineering KB"
Query "Q2 roadmap" → RAG pulls only from Engineering
  → Focused, relevant results
  → LLM gives better answer
```

---

## 9. Mitigations & Risk Management

This section addresses 10 key risks identified in design review. Each mitigation is integrated into the architecture above.

### 9.1 Namespace Explosion → Hybrid Collections + Metadata Filtering

**Risk:** 100 KBs × 20 subtags = 2000 Qdrant collections = management nightmare + performance degradation.

**Mitigation:** Use **one Qdrant collection per KB**, not per subtag. Metadata filtering in Qdrant is fast.

```
Collections: kb_1, kb_2, ..., kb_100  (not kb_1_sub_1, kb_1_sub_2, ...)
Payload includes: {subtag_id: ...}
Search with filter: {"subtag_id": 12}
```

**Result:** 100 collections (vs. 2000), ~5-10% query latency increase, vastly simpler ops. *See Section 3.3.*

---

### 9.2 Multi-Query Fan-Out → Async Parallel Batch Search

**Risk:** 5 KBs × 5 subtags = 25 sequential Qdrant queries = 2500ms latency per request.

**Mitigation:** Execute all Qdrant searches in parallel using asyncio.

```python
tasks = [Qdrant.search_async(...) for each kb/subtag]
results = await asyncio.gather(*tasks)  # All parallel, ~100ms total
```

**Result:** ~25× faster retrieval. *See Stage 6.*

---

### 9.3 RBAC DB Overhead → Cache at Chat Start

**Risk:** `SELECT kb_access...` on every message = repeated DB hits + latency.

**Mitigation:** Fetch allowed KB IDs once at chat start, cache in Redis (1h TTL).

```
First message: SELECT kb_access → store in cache
Messages 2-3600: cache.get() → O(1) lookup, 0 DB hits
```

**Result:** ~90% fewer DB queries per user session. *See Stage 5.*

---

### 9.4 Cross-KB Ranking Weakness → Global Reranking + Normalization

**Risk:** Scores from different KB namespaces aren't comparable; weak cross-KB ranking.

**Mitigation:** Normalize scores per KB, then globally sort. Use reranker as optional tier.

```
Tier 1: Fast path (no reranker) if confident
Tier 2: Normalize scores across KBs
Tier 3: Heavy reranker if uncertain
```

**Result:** Better cross-KB relevance, optional latency. *See Stage 7.*

---

### 9.5 Reranker Bottleneck → Tiered Retrieval (Optional but Smart)

**Risk:** Reranker adds latency; becomes a required dependency.

**Mitigation:** Use tiered strategy—reranker only when needed.

```
If latency < 100ms AND confidence > 0.85: skip reranker
Else: normalize or rerank
```

**Result:** Reranker is optional optimization, not required path. *See Stage 7.*

---

### 9.6 Static Chunking → Adaptive Chunking (Optional)

**Risk:** Fixed 800-token chunks don't work for all doc types (PDFs vs. code vs. tables).

**Mitigation (Optional):** Detect doc type, apply adaptive chunking strategy.

```python
if doc_type == "code":
    split_on_function_boundaries()
elif doc_type == "table":
    split_by_rows()
else:
    default_recursive_split(800_tokens)
```

**Result:** Better chunk quality per doc type (future enhancement).

---

### 9.7 No Freshness / Versioning → Simple Versioning Layer

**Risk:** Documents never update; no way to rollback or track versions.

**Mitigation (Future):** Add simple versioning on kb_documents table.

```sql
ALTER TABLE kb_documents ADD COLUMN (
  version INT DEFAULT 1,
  superseded_by BIGINT REFERENCES kb_documents(id)
);
-- Old doc marked as superseded; new version re-embedded
```

**Result:** Support document updates without data loss. *Phase 2.*

---

### 9.8 Private Session Cleanup Risk → TTL + Background GC

**Risk:** session_{chat_id} vectors orphaned if cleanup fails → storage leak.

**Mitigation:** Use Qdrant TTL + periodic background garbage collector.

```python
# Option 1: Qdrant TTL (automatic)
Qdrant.upsert(..., point_ttl_seconds=86400)  # 24h auto-delete

# Option 2: Scheduled job (every hour)
@periodic_task
def cleanup_old_sessions():
    old_chats = Chat.objects.filter(archived=True, archived_at < now() - 1day)
    for chat_id in old_chats:
        Qdrant.delete_collection(f"session_{chat_id}")
```

**Result:** Zero orphaned vectors, automatic storage cleanup.

---

### 9.9 Token Limit Overflow → Strict Budgeting + Truncation

**Risk:** Appending all chunks into prompt → exceed token limits → cost spike.

**Mitigation:** Strict token budget per prompt, graceful truncation.

```
MAX_PROMPT = 2048, RESERVED_FOR_RESPONSE = 512
AVAILABLE_FOR_CONTEXT = 1536 tokens

For each chunk:
  if chunk_tokens < remaining_budget:
    append to prompt
  else:
    truncate gracefully, stop adding
```

**Result:** Predictable token costs, no surprises. *See Stage 8.*

---

### 9.10 No Observability → Prometheus + Structured Logging

**Risk:** No visibility into query latency, retrieval quality, embedding failures → hard to debug production issues.

**Mitigation:** Comprehensive observability stack.

```python
# Prometheus metrics
rag_retrieval_latency_seconds (by kb_id, status)
rag_rerank_latency_seconds
rag_embedding_failures_total

# Structured JSON logs
{
  "event": "rag_complete",
  "query_length": 42,
  "chunks_retrieved": 8,
  "top_chunk_score": 0.92,
  "llm_tokens": 512,
  "duration_ms": 245
}
```

**Result:** Production debugging is easy, quality metrics visible, cost tracking enabled.

---

## 10. Models & Infrastructure

All services are optimized for **32GB RTX 6000 Ada** with **Option B (Smart Loading)**: always-running baseline + on-demand models.

### 10.1 Hardware-Optimized Single GPU (32GB RTX 6000 Ada)

**Architecture: Always-Running + On-Demand Loading**

```
Always Running (15GB baseline, 17GB headroom):
├── vllm-chat: Qwen/Qwen2.5-14B-Instruct-AWQ (12GB)
├── tei: BAAI/bge-m3 embeddings (3GB)
└── Available for KV-cache, batching, overhead: ~17GB

On-Demand Loading (Loads when first used, auto-unloads after 5min inactivity):
├── vllm-vision: Qwen/Qwen2-VL-7B-Instruct (8GB) ← Loads on image upload
└── whisper: faster-whisper medium (4GB) ← Loads on audio upload
```

**Why This Approach?**
- ✅ Baseline services always hot for chat/RAG (best latency for primary use case)
- ✅ 17GB headroom for KV-cache, prompt caching, concurrent requests
- ✅ Vision/audio loads in ~2-3 seconds (acceptable UX, shows loading spinner)
- ✅ Auto-unload frees 8-12GB during idle periods
- ✅ Safe margins prevent OOM errors

**LLM Chat Model (Always Running):**
- Model: `Qwen/Qwen2.5-14B-Instruct-AWQ` (4-bit quantized)
- VRAM: 12GB
- Framework: vLLM (OpenAI-compatible)
- Port: 8000 (Docker-internal)
- Throughput: ~5-10 concurrent users
- vLLM flags: `--max-model-len 8192 --gpu-memory-utilization 0.9 --enable-prefix-caching --enable-chunked-prefill`

**Vision Model (On-Demand Loading):**
- Model: `Qwen/Qwen2-VL-7B-Instruct`
- VRAM: 8GB (loads when needed)
- Framework: vLLM (separate container, paused by default)
- Port: 8001 (Docker-internal)
- Auto-Unload: After 5 minutes of inactivity (frees 8GB)
- Behavior: First image request triggers async load (user sees spinner for 2-3s)

**Embeddings Model (Always Running):**
- Model: `BAAI/bge-m3` (1024 dimensions, optimized)
- VRAM: 3GB
- Framework: TEI (text-embeddings-inference, GPU-accelerated)
- Port: 80 (Docker-internal)
- Throughput: 100+ embeddings/second (batched)
- Used by: RAG chunk embedding + query embedding

**Speech-to-Text Model (On-Demand Loading):**
- Model: `faster-whisper` `medium`
- VRAM: 4GB (loads when needed)
- Framework: faster-whisper (OpenAI-compatible)
- Port: 8080 (Docker-internal)
- Auto-Unload: After 5 minutes of inactivity
- Behavior: First audio request triggers async load (user sees spinner for 1-2s)

### 10.2 Future: Multi-GPU Scaling (When Needed)

If your organization grows beyond 10-20 concurrent users, migrate to:
- **4×48GB GPU cluster** (one GPU per service: chat, vision, embeddings, whisper)
- **Kubernetes deployment** with auto-scaling
- **Larger models**: Qwen2.5-72B chat, Qwen2-VL-72B vision
- See original `2026-04-12-org-chat-assistant-design.md` Section 11 for Phase 2 details

**Current single-GPU setup supports:** 5-10 concurrent users with on-demand model loading

### 10.3 On-Demand Model Loading Mechanism

**Problem:** Vision (8GB) + Whisper (4GB) = 12GB. Adding to always-running 15GB = 27GB (risky, leaves no headroom).

**Solution:** Load models on-demand, auto-unload on inactivity.

**Implementation:**

```python
# services/model_manager.py

class ModelManager:
    def __init__(self):
        self.models = {
            'vision': {
                'container': 'vllm-vision',
                'status': 'unloaded',
                'last_used': None,
                'vram_gb': 8
            },
            'whisper': {
                'container': 'whisper',
                'status': 'unloaded',
                'last_used': None,
                'vram_gb': 4
            }
        }
    
    async def ensure_model_loaded(self, model_name: str):
        """Load model if needed, async to avoid blocking chat."""
        if self.models[model_name]['status'] == 'loaded':
            self.models[model_name]['last_used'] = now()
            return
        
        logger.info(f"Loading {model_name} model (~3s)...")
        
        # Scale down chat model's KV-cache to free VRAM
        await self.reduce_vllm_chat_cache()
        
        # Start container or resume paused container
        docker.container.start(self.models[model_name]['container'])
        
        # Wait for model to be ready (health check)
        await self.wait_for_readiness(model_name, timeout=10)
        
        self.models[model_name]['status'] = 'loaded'
        self.models[model_name]['last_used'] = now()
        logger.info(f"{model_name} model loaded successfully")
    
    @periodic_task(interval=300)  # Every 5 minutes
    async def cleanup_unused_models(self):
        """Auto-unload models after 5 min inactivity."""
        for model_name, config in self.models.items():
            if config['status'] == 'loaded':
                last_used_ago = (now() - config['last_used']).total_seconds()
                
                if last_used_ago > 300:  # 5 minutes
                    logger.info(f"Unloading {model_name} (freed {config['vram_gb']}GB)")
                    
                    # Pause container (keeps state, frees VRAM)
                    docker.container.pause(config['container'])
                    config['status'] = 'unloaded'
                    
                    # Restore KV-cache
                    await self.restore_vllm_chat_cache()

# In API handlers:

@app.post("/api/chat/{chat_id}/messages")
async def send_message(chat_id: str, message: Message):
    # If message has image
    if message.attachments and any(a.type == 'image' for a in message.attachments):
        logger.info("Image detected, ensuring vision model is loaded...")
        
        # This happens async; user sees spinner
        asyncio.create_task(
            model_manager.ensure_model_loaded('vision')
        )
        
        # Send "Processing image..." to user immediately
        return {"status": "processing", "message": "Loading image model..."}
    
    # If message has audio
    if message.attachments and any(a.type == 'audio' for a in message.attachments):
        logger.info("Audio detected, ensuring Whisper is loaded...")
        
        asyncio.create_task(
            model_manager.ensure_model_loaded('whisper')
        )
        
        return {"status": "processing", "message": "Loading audio model..."}
    
    # Normal chat (no vision/audio)
    return await process_chat_message(chat_id, message)
```

**Docker Compose Strategy:**

```yaml
services:
  vllm-vision:
    image: vllm/vllm-openai:latest
    environment:
      CUDA_VISIBLE_DEVICES: 0
    entrypoint: |
      python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2-VL-7B-Instruct \
        --port 8001
    restart: unless-stopped
    profiles: ["on-demand"]  # Not started by default
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 5s
      timeout: 10s
      retries: 3

  whisper:
    image: openai/whisper:latest
    environment:
      CUDA_VISIBLE_DEVICES: 0
      WHISPER_MODEL: medium
    ports:
      - "8080:8080"
    restart: unless-stopped
    profiles: ["on-demand"]  # Not started by default
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 5s
      timeout: 10s
      retries: 3
```

**Start Sequence:**

```bash
# Start core services (always-running)
docker-compose up -d vllm-chat tei postgres redis qdrant

# On-demand services NOT started initially
# They're started by model_manager.ensure_model_loaded() when needed
```

**VRAM Timeline:**

```
t=0: Chat starts
  ├─ vllm-chat: 12GB ✓
  ├─ tei: 3GB ✓
  └─ Free: 17GB

t=30s: User uploads image
  ├─ vllm-chat: 12GB (reduced KV-cache to 8GB)
  ├─ vllm-vision: 8GB (loading)
  ├─ tei: 3GB
  └─ Free: 1GB (tight during load)

t=33s: Vision model ready
  ├─ vllm-chat: 12GB (normal KV-cache restored)
  ├─ vllm-vision: 8GB (active)
  ├─ tei: 3GB
  └─ Free: 9GB

t=333s (5min later, no image requests):
  ├─ vllm-vision: unloaded (paused)
  └─ Free: 17GB (back to baseline)
```

---

### 10.4 Model Configuration in Environment (32GB RTX 6000 Ada)

```bash
# Core Services (Always Running)

# vLLM Chat
OPENAI_API_BASE_URL=http://vllm-chat:8000/v1
OPENAI_API_KEY=sk-internal-dummy
VLLM_CHAT_GPU_MEMORY_UTILIZATION=0.9
VLLM_CHAT_MAX_MODEL_LEN=8192
VLLM_CHAT_ENABLE_PREFIX_CACHING=true
VLLM_CHAT_ENABLE_CHUNKED_PREFILL=true

# TEI Embeddings
RAG_EMBEDDING_ENGINE=openai
RAG_EMBEDDING_OPENAI_API_BASE_URL=http://tei:80/v1
RAG_EMBEDDING_MODEL=BAAI/bge-m3

# On-Demand Services

# vLLM Vision (loaded on-demand)
VISION_API_BASE_URL=http://vllm-vision:8001/v1
VISION_AUTO_LOAD=true
VISION_AUTO_UNLOAD_IDLE_SECS=300

# Whisper STT (loaded on-demand)
AUDIO_STT_ENGINE=whisper-local
AUDIO_STT_OPENAI_API_BASE_URL=http://whisper:8080/v1
AUDIO_AUTO_LOAD=true
AUDIO_AUTO_UNLOAD_IDLE_SECS=300

# Reranker (CPU-based)
RAG_RERANKER_ENABLED=true
RAG_RERANKER_MODEL=BAAI/bge-reranker-base
```

### 10.5 Model Performance Baselines (32GB RTX 6000 Ada, Option B)

| Model | VRAM (GB) | Status | Throughput | Latency p99 | Notes |
|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct-AWQ | 12 | Always-on | 5-10 req/s | 200-400ms | Chat |
| BAAI/bge-m3 (TEI) | 3 | Always-on | 100+ req/s | 10-20ms | Embeddings |
| Qwen2-VL-7B-Instruct | 8 | On-demand | 2-5 req/s | 500-1000ms | Vision, ~2-3s load |
| faster-whisper medium | 4 | On-demand | 2-3 audio/s | 2000-5000ms | STT, ~1-2s load |
| **Baseline** | **15** | — | — | — | Chat + embeddings only |
| **Peak w/ Vision** | **23-27** | — | — | — | Safe within 32GB |
| **Peak w/ Whisper** | **19-23** | — | — | — | Safe within 32GB |

**Setup for 32GB:**
- Baseline VRAM: 15GB (chat + embeddings always hot)
- Headroom: 17GB (KV-cache, batching)
- Concurrent users: 5-10
- Auto-unload timer: 5 minutes

---

## 11. Migration Path (From Old Collections)

**Option 1: Preserve existing collections as KBs**
```
existing collection "engineering" 
  → new KB "Engineering"
  → auto-create default subtag "All"
  → all existing docs → subtag "All"
```

**Option 2: Manual migration**
```
Admins recreate KBs with desired structure
Old collection schema deprecated
```

**Option 3: Dual-mode (future)**
```
Support both old (collection-based) and new (KB-based)
Gradual migration user-by-user
```

---

## 12. Testing Strategy

### Critical Tests

1. **KB Access Control**
   - User A uploads doc to KB-X
   - User B (no access to KB-X) starts chat, tries to select KB-X
   - → 403 Forbidden

2. **Subtag Isolation**
   - KB-Y has subtags S1, S2
   - User selects only S1
   - Upload doc to S2
   - → Doc not retrieved in user's queries

3. **Session Privacy**
   - User A uploads private doc in Chat-1
   - User A in Chat-2 → private doc from Chat-1 not visible
   - User B → cannot see any of User A's private docs

4. **RBAC + Retrieval**
   - Admin grants KB-Z to "engineering" group
   - User in "engineering" selects KB-Z
   - → Retrieval succeeds
   - User NOT in "engineering" tries to select KB-Z
   - → Selection blocked

5. **Mixed KB + Private Doc Retrieval**
   - User selects KB-A
   - User uploads private doc
   - Query returns results from both KB-A and private doc

---

## 13. Out of Scope (This Phase)

- KB versioning (e.g., rollback to old KB state)
- KB-level rate limiting (future)
- Subtag three+ levels (keeping to two levels)
- SSO/LDAP integration (existing scope limitation)

---

## 14. Appendix: Example Data Flow

### Scenario: Q2 Roadmap Query

**Setup:**
- KB "Engineering" (id=7) with subtags "Roadmap" (id=21), "Bugs" (id=22)
- User "alice@org.com" (id=10) in group "engineering" (id=2)
- Doc "Q2-roadmap.pdf" uploaded to Engineering/Roadmap
- Chunks embedded and stored in `kb_7_sub_21` namespace

**Chat Session:**
```
1. Alice starts chat
   Selects: Engineering KB, Roadmap subtag only
   selected_kb_config = [{"kb_id": 7, "subtag_ids": [21]}]

2. Alice asks: "What's our Q2 plan?"

3. RBAC Check:
   SELECT * FROM kb_access 
   WHERE kb_id=7 AND (user_id=10 OR group_id=2)
   → Found (group 2 has access)
   ✓ Proceed

4. Embed query → query_embedding

5. Qdrant search:
   collection="kb_7_sub_21"
   vector=query_embedding
   limit=5
   → Returns top 5 chunks from Q2-roadmap.pdf

6. Rerank → Final 3 chunks

7. vLLM prompt:
   System: "You are helpful. Context:
   - Q2 focus: modernize infrastructure...
   - Timeline: Q2 weeks 1-13...
   - Budget: $2M allocated..."
   User: "What's our Q2 plan?"
   
   Response: "Your Q2 plan focuses on infrastructure 
   modernization with a $2M budget over 13 weeks..."

8. Message + response saved
```

---

## Next Steps

1. **Brainstorming Review** — User approves design
2. **Implementation Planning** — Create detailed implementation plan (models, APIs, migrations)
3. **Backend Implementation** — Schema, APIs, RAG pipeline
4. **Frontend Integration** — KB selection UI, upload flows
5. **Testing & Validation** — Isolation tests, performance tests
