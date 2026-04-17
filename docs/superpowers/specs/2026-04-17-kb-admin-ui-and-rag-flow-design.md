# KB Admin UI + E2E RAG Chat Flow — Design Spec

**Date:** 2026-04-17
**Status:** Approved
**Scope:** Two deliverables: (1) KB management UI inside Open WebUI admin portal, (2) RAG injection into chat so LLM answers are grounded in selected KB content.

---

## 1. E2E RAG Chat Flow

### How it works

When a user sends a message with KBs selected via the KBSelector:

1. `POST /api/chat/completions` hits upstream's backend
2. `process_chat_payload()` in `middleware.py` checks if the chat has `selected_kb_config`
3. If yes: calls our `retrieve(query, selected_kbs, chat_id)` → parallel Qdrant search → rerank → budget
4. Converts our hits into upstream's source format: `<source id="N" name="filename">chunk text</source>`
5. Calls upstream's `apply_source_context_to_messages()` which injects context into the system prompt via the RAG template
6. Sends augmented prompt to vllm-chat
7. LLM generates answer using the context
8. Response streams back — upstream's citation rendering shows sources automatically

### Patch point

ONE insertion in `upstream/backend/open_webui/utils/middleware.py` at `process_chat_payload()`, before the existing `chat_completion_files_handler` call (~line 2699). Approximately 40 lines of Python.

The patch:
- Reads `selected_kb_config` from our `kb_access` system (by querying the chat's stored config)
- Extracts the user's query from the last message
- Calls our async retriever (parallel Qdrant search + rerank + budget)
- Formats results as upstream-compatible source dicts
- Merges into the `sources` list that upstream's existing `apply_source_context_to_messages` consumes

### No frontend changes for RAG

Upstream already renders sources/citations. When we inject sources in the correct format, they appear automatically in the chat UI.

---

## 2. KB Admin UI

### Architecture

Svelte route at `/admin/kb` using Option C (thin Svelte shell + our API).

One SvelteKit route file + one main component calling our `/api/kb/*` endpoints. Matches upstream's admin design language (Tailwind dark theme, same layout patterns).

### Layout

```
/admin/kb page:
┌─────────────────────────────────────────────────────┐
│  KB Management                          [+ Create]  │
├──────────────┬──────────────────────────────────────┤
│  KB List     │  Detail Panel (shown on selection)   │
│              │                                      │
│  > Engineering│  Subtags: [Architecture] [Policies]  │
│    CompanyDocs│  [+ Add subtag]                     │
│    HR Docs   │                                      │
│              │  Documents:                           │
│              │  ┌──────────────────────────────────┐│
│              │  │ policy.docx  done  3 chunks [Del]││
│              │  │ arch.pdf     done  7 chunks [Del]││
│              │  └──────────────────────────────────┘│
│              │  [Drop files here or click to browse] │
│              │                                      │
│              │  Access Control:                      │
│              │  Engineering Team (group) read [Revoke│
│              │  [Grant to User ▼] [Grant to Group ▼]│
│              │                                      │
│              │  [Re-embed All]                       │
└──────────────┴──────────────────────────────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `upstream/src/routes/(app)/admin/kb/+page.svelte` | Route wrapper |
| `upstream/src/lib/components/admin/KB/KBAdmin.svelte` | Full admin component |
| Admin sidebar patch | Add "Knowledge Bases" nav link |
| `upstream/backend/open_webui/utils/middleware.py` patch | RAG injection |

### Backend additions

| Endpoint | Purpose |
|----------|---------|
| `POST /api/kb/{id}/documents/{did}/reembed` | Re-ingest a doc from stored file |
| `POST /api/kb/{id}/reembed-all` | Wipe + re-ingest all docs in KB |
| `GET /api/kb/{id}/documents` | List docs with status (already exists) |
| `DELETE /api/kb/{id}/documents/{did}` | Soft-delete + remove Qdrant vectors (already exists, needs Qdrant cleanup) |

File storage: uploads saved to `volumes/uploads/{kb_id}/{doc_id}/{filename}` so re-embed can re-read them.

---

## 3. Session Persistence

- **KBSelector on reload:** Component calls `GET /api/chats/{id}/kb_config` on mount to restore selection from DB.
- **Stale ingest:** On admin page load, docs with `ingest_status='chunking'` older than 10 min shown with "stale" warning. Admin can delete and re-upload.
- **Re-embed policy:** Manual Option A. Upload always creates new doc row. Delete removes chunks. "Replace" = delete + re-upload. "Re-embed All" = wipe all chunks for KB + re-ingest all non-deleted docs.

---

## 4. Edge Cases

| # | Case | Behavior |
|---|------|----------|
| 1 | Page reload with KBs selected | Restored from DB |
| 2 | Browser close mid-upload | In-flight doc marked failed on next load |
| 3 | Delete doc already retrieved | Future queries: not returned. Past messages: unchanged |
| 4 | Select KB → send → deselect → send | First: RAG context. Second: none |
| 5 | Admin revokes access mid-chat | Next message: RBAC re-checked, KB excluded |
| 6 | Unsupported file type | 422 with message |
| 7 | Oversized file | 413 |
| 8 | Two users same KB simultaneously | Both correct, no cross-contamination |
| 9 | Empty KB selected | No RAG context, LLM answers normally |
| 10 | KB with 100+ docs | Top-K retrieved within token budget |
| 11 | KB deleted while selected | Graceful: no crash, no context |
| 12 | Re-embed all | Old chunks deleted, fresh re-ingest |
| 13 | Same filename uploaded twice | Two doc rows, both indexed |
| 14 | TEI down during embed | Doc status=failed, admin retries |

---

## 5. Test Plan

### E2E RAG test (the core validation):

1. Admin creates KB "Engineering" + subtag "Architecture"
2. Admin uploads a document about microservices
3. Admin grants access to "Engineering Team" group
4. Alice (in group) opens chat, selects "Engineering" KB
5. Alice asks: "What architecture does our company use?"
6. **LLM responds citing the uploaded document content** (not a generic answer)
7. Bob (no group) asks same question → generic answer, no company context

### Isolation test:

8. Alice's private chat docs invisible to Bob
9. Bob cannot select Alice's granted KBs

### Persistence test:

10. Alice reloads page → KB selection restored
11. Admin deletes KB → Alice's next message gracefully degrades
