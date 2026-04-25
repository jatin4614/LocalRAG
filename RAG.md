1. Executive summary                              
                                                                                                                                                                      
  The system is a self-hosted, air-gapped organizational RAG chat assistant on a single 32/48 GB RTX 6000 Ada. It now has:
                                                                                                                                                                      
  - A live hybrid (dense + sparse BM25 + RRF fusion) retrieval path on the production KB.                                                                             
  - A working three-way intent router (metadata / global / specific) that picks the right retrieval shape for each query instead of forcing everything through top-k  
  chunk search.                                                                                                                                                       
  - A per-document summary index (108 summary points for 110 docs in Comn) that solves the "list every X" / "what reports do I have" aggregation-query pain from the
  previous week.                                                                                                                                                      
  - Subtag-scoped KB selection end-to-end: users can pick whole KBs, specific subtags within a KB, or no KB, with UI, API, validation, and retrieval all honoring the
  scope.                                                                                                                                                              
  - Lock-after-first-message on chat KB config (per design §2.4).
  - Contextual retrieval (Anthropic-style chunk contextualization) implemented in code, currently flagged OFF — turning it on requires a re-ingest window.            
  - RAPTOR (hierarchical summarization) implemented in code, flagged OFF.                                                                                             
  - Full observability: Prometheus metrics for every stage, SSE progress events, correlation IDs, structured JSON logs.                                               
  - Strict isolation: defense-in-depth at DB / API / Qdrant payload filter layers.                                                                                    
                                                                                                                                                                      
  Image: orgchat-open-webui:8f2ce71699f5 · Container healthy · 110 docs / 928 chunks / 108 summary points live.                                                       
                                                                                                                                                                      
  ---                                                                                                                                                                 
  2. What this session achieved (chronological)                                                                                                                     
                                                                                                                                                                      
  2a. Subtag scoping (Phases A + B of the earlier work)
                                                                                                                                                                      
  Backend (Phase A):                                                                                                                                                  
  - Catalog preamble (chat_rag_bridge.py) rewritten to respect subtag_ids — no more hallucinated out-of-scope docs. Header now reads "KB 4 → Roadmap, Legal: N        
  document(s) available" when subtags are picked.                                                                                                                     
  - New endpoint GET /api/kb/{kb_id}/subtags for users (admin-only version already existed).                                                                        
  - Two new guards on PUT /api/chats/{id}/kb_config: (1) subtag must belong to its KB (400 on mismatch), (2) config is immutable once the chat has any role=="user"   
  message in chat.chat.history.messages (409).                                                                                                                        
                                                                                                                                                                      
  Frontend (Phase B):                                                                                                                                                 
  - $lib/apis/kb/index.ts: added getKBSubtags() and the shared KBSelection type.                                                                                      
  - KBPickerModal.svelte + KBSelector.svelte: hierarchical picker with tri-state per KB (unchecked / whole / scoped).                                                 
  - Chat.svelte + Navbar.svelte: state shape migrated from selectedKBIds: number[] to selections: KBSelection[]. Server is now the single source of truth (no       
  localStorage duplication).                                                                                                                                          
  - Reactivity bugfix: {@const state/entry = rowState(kb.id)...} used function indirection; Svelte 4 doesn't traverse function bodies for deps, so the {@const} never 
  re-evaluated on selections mutation. Inlined the derivation so selections is read directly in the template.                                                         
                                                                                                                                                                      
  Schema cleanup: migration 007 drops the orphaned chats.selected_kb_config column from migration 001 (no-op in this deployment since the column was never actually 
  created, but idempotent for other installs).                                                                                                                        
                                                    
  2b. Option-C parallel tracks (today)                                                                                                                                
                                                    
  Track A — kb_1 collection rebuild:                                                                                                                                  
  - Snapshot backup at backups/kb_1_rebuild_2026-04-22/.
  - Collection reshaped from legacy dense-only unnamed vector → named dense (1024-d Cosine) + bm25 sparse (IDF modifier).                                             
  - indexed_vectors_count: 0 → 2589 (1-point segment-boundary artifact).                                                 
  - Payload indexes: kb_id/subtag_id/doc_id changed from KeywordIndexParams (which was silently indexing nothing because the values are int) →                        
  IntegerIndexParams(type="integer", lookup=True). Index populations: 0/2590 → 2590/2590 each.                                                                        
  - Alias swap: kb_1 → kb_1_rebuild, old collection dropped.                                                                                                          
                                                                                                                                                                      
  Track B — Tier 1 (per-doc summary index) + Tier 2 (intent router):                                                                                                  
  - Migration 008 — kb_documents.doc_summary TEXT NULL.                                                                                                               
  - ext/services/doc_summarizer.py — 3-sentence Gemma summary, fail-open.                                                                                             
  - ext/services/query_intent.py — regex classifier with 37 unit tests passing. classify() + classify_with_reason().                                                  
  - ext/services/kb_config.py — doc_summaries (INGEST_ONLY), intent_routing, intent_llm wired into per-KB overlay.                                                    
  - chat_rag_bridge.py::_run_pipeline — branches retrieval by intent: metadata skips retrieve, global hits summary index with wider top-k (50/100), specific uses the 
  existing pipeline.                                                                                                                                                  
  - ingest.py — _emit_doc_summary_point helper gated by RAG_DOC_SUMMARIES, deterministic UUIDv5 ID, mirrors summary into kb_documents.doc_summary.                    
  - scripts/backfill_doc_summaries.py — idempotent, bounded-concurrency backfill.                                                                                     
                                                                                                                                                                      
  Track-B follow-ups I had to land after the agent's report:                                                                                                          
  1. Backfill script constructor bugs: VectorStore(qdrant_url) → VectorStore(url=..., vector_size=...); Embedder(...) is a Protocol, needed TEIEmbedder(base_url=...).
  2. Summary points needed a sparse companion (sparse_vector computed via embed_sparse) — without it, VectorStore.upsert fell to the legacy unnamed-vector path which 
  Qdrant rejects on hybrid collections with "Not existing vector name".                                                                                               
  3. Specific-path contamination fix: _level_filter="chunk" on specific intent (Track B had None, which let summaries leak into specific queries since summary text   
  rich in dates/filenames outscored chunks).                                                                                                                        
  4. Catalog drop-out fix: early-exit at if not budgeted updated to skip both metadata AND global intents so the catalog preamble always flows on those paths.        
  5. HyDE + hybrid bias on global: global-path override scope now sets RAG_HYDE=0, RAG_HYDE_N=0, RAG_HYBRID=0 — HyDE's hypothetical answers look like chunks, and BM25
   ranks term-rich chunks above summaries. Pure dense-over-summaries is what global needs.                                                                            
  6. Critical _PAYLOAD_FIELDS bug: allowlist in vector_store.py was stripping level and kind out of returned payloads, so post-filters on level could never see the   
  value. Added both fields.                                                                                                                                           
  7. Qdrant-level pre-filter for level: added level param to _build_filter, threaded through search() and hybrid_search(), wired into retriever._search_one.          
  Summary/chunk filtering now happens server-side, not as a racy Python post-filter.                                                                                
  8. Dockerfile.openwebui: COPY scripts/apply_migrations.py → COPY scripts/ so future scripts ship with the image.                                                    
                                                                                                                  
  Backfill outcome: 105 of 107 remaining docs summarized (+ 3 from the initial smoke = 108 live summary points); 2 came back empty from Gemma (fail-open handled      
  them).                                                                                                                                                              
                                                                                                                                                                      
  ---                                                                                                                                                                 
  3. RAG architecture at a glance                                                                                                                                   
                                                                                                                                                                    
  ┌──────────────────────────────────────────────────────────────────┐
  │                        INGEST (upload.py)                        │                                                                                                
  │  bytes → extract → chunk (800/100) → [contextualize?]            │
  │    → embed (TEI bge-m3) + sparse (fastembed bm25)                │                                                                                                
  │    → upsert Qdrant kb_{kb_id} (dense+sparse, named vectors)      │                                                                                                
  │    → [emit doc_summary point if RAG_DOC_SUMMARIES=1]             │                                                                                                
  │    → Postgres kb_documents row                                   │                                                                                                
  └──────────────────────────────────────────────────────────────────┘                                                                                                
                                                                                                                                                                      
  ┌──────────────────────────────────────────────────────────────────┐                                                                                                
  │                   CHAT REQUEST                                    │                                                                                             
  │                     │                                              │                                                                                              
  │                     ▼                                              │                                                                                              
  │  chat_rag_bridge.retrieve_kb_sources                              │                                                                                             
  │    1. RBAC: get_allowed_kb_ids (filter kb_config)                │                                                                                                
  │    2. Load per-KB rag_config → merge → contextvars overlay        │                                                                                               
  │    3. [query rewrite if RAG_DISABLE_REWRITE=0]                    │                                                                                               
  │    4. Classify intent if RAG_INTENT_ROUTING=1                     │                                                                                               
  │                                                                    │                                                                                              
  │    ┌──────────────────────────────────────────────────────────┐   │                                                                                               
  │    │  intent == "metadata"                                    │   │                                                                                             
  │    │    → skip retrieve → [] + catalog preamble = answer      │   │                                                                                               
  │    │                                                          │   │                                                                                               
  │    │  intent == "global"                                      │   │                                                                                               
  │    │    → retrieve(level_filter="doc",                        │   │                                                                                               
  │    │               RAG_HYDE=0, RAG_HYBRID=0,                  │   │                                                                                               
  │    │               per_kb=50, total=100)                      │   │                                                                                               
  │    │    → skip rerank / MMR / expand (summaries self-contain) │   │                                                                                               
  │    │    → catalog preamble + summaries                        │   │                                                                                               
  │    │                                                          │   │                                                                                               
  │    │  intent == "specific"  (default)                         │   │                                                                                               
  │    │    → [HyDE? → 2× hypothetical + RRF fuse]                │   │                                                                                               
  │    │    → retrieve(level_filter="chunk",                      │   │                                                                                               
  │    │               per_kb=10, total=30)                       │   │                                                                                               
  │    │    → [rerank? bge-reranker-v2-m3 cross-encoder]          │   │                                                                                               
  │    │    → [MMR? λ=0.7 diversify]                              │   │                                                                                               
  │    │    → [context expand? ±window siblings]                  │   │                                                                                               
  │    │    → budget (5000 tokens)                                │   │                                                                                               
  │    │    → catalog preamble + chunks                           │   │                                                                                               
  │    └──────────────────────────────────────────────────────────┘   │                                                                                               
  │                                                                    │                                                                                              
  │    5. [Spotlight wrap? ZWJ defang + UNTRUSTED tags]               │                                                                                               
  │    6. Emit sources to LLM                                         │                                                                                               
  └──────────────────────────────────────────────────────────────────┘                                                                                              
                                                                                                                                                                      
  ---                                               
  4. Complete ingest pipeline                                                                                                                                         
                                                    
  ┌─────┬────────────────┬─────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────┐ 
  │  #  │     Stage      │                  File                   │                                         What happens                                         │   
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 1   │ Upload         │ routers/upload.py                       │ Size cap (RAG_MAX_UPLOAD_BYTES, 50MB default → 25MB in .env), MIME sniff, safe filename,     │   
  │     │ validation     │                                         │ RBAC (admin for KB, chat owner for private).                                                 │
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ 2   │ DB row         │ upload.py → kb_documents                │ Row inserted with ingest_status="chunking", pipeline_version, blob_sha.                      │
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ 3   │ Text           │ services/extractors/*.py                │ PDF (PyMuPDF), DOCX (python-docx), XLSX (openpyxl), TXT/MD (raw). Structural metadata        │
  │     │ extraction     │                                         │ captured: page, heading_path, sheet, block_type.                                             │   
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ 4   │ Chunking       │ services/chunker.py                     │ Token-based, 800 tokens/chunk, 100 overlap. chunker=v2 semantic-boundary-aware (avoid        │ 
  │     │                │                                         │ splitting mid-sentence / mid-table).                                                         │   
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 5   │ Contextualize  │ services/contextualizer.py              │ If RAG_CONTEXTUALIZE_KBS=1: Gemma prepends 50-100 tokens of doc context per chunk. Bounded   │   
  │     │ (opt)          │                                         │ concurrency = 8. ctx=contextual-v1 stamped on pipeline_version.                              │   
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤ 
  │ 6   │ Dense embed    │ services/embedder.TEIEmbedder           │ TEI HTTP POST /embed, client-side batched at RAG_TEI_MAX_BATCH=32 (server rejects ≥33).      │   
  │     │                │                                         │ bge-m3 1024-d.                                                                               │   
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤ 
  │ 7   │ Sparse embed   │ services/sparse_embedder.py             │ fastembed Qdrant/bm25: stopwords + stemmer + TF. IDF is applied server-side by Qdrant.       │   
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ 8   │ Deterministic  │ services/ingest.py                      │ Point ID = UUIDv5(NS, f"doc:{doc_id}:chunk:{i}"). Re-ingest overwrites.                      │ 
  │     │ IDs            │                                         │                                                                                              │   
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │     │                │                                         │ Named-vector form {dense: [...], bm25: SparseVector(indices, values)} on hybrid collections; │   
  │ 9   │ Upsert         │ vector_store.upsert                     │  legacy unnamed fallback on dense-only collections. Payload: text + kb_id + subtag_id +      │   
  │     │                │                                         │ doc_id + chat_id + owner_user_id + filename + page + heading_path + block_type + deleted.    │ 
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤   
  │     │ Summary point  │                                         │ If RAG_DOC_SUMMARIES=1: summarize full doc via Gemma → embed → upsert one point per doc with │   
  │ 10  │ (opt)          │ services/ingest._emit_doc_summary_point │  level="doc", kind="doc_summary", chunk_index=-1. Also UPDATE kb_documents SET doc_summary = │ 
  │     │                │                                         │  .... Deterministic ID UUIDv5(NS, f"doc:{doc_id}:doc_summary").                              │   
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 11  │ RAPTOR (opt)   │ services/raptor.py                      │ If RAG_RAPTOR=1: UMAP+GMM clustering, LLM summaries at N levels. Stamps chunk_level,         │ 
  │     │                │                                         │ source_chunk_ids on resulting points.                                                        │   
  ├─────┼────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 12  │ Status flip    │ upload.py                               │ kb_documents.ingest_status = "done", chunk_count set, commit.                                │   
  └─────┴────────────────┴─────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘   
                                                                                                                                                                    
  ---                                                                                                                                                                 
  5. Complete retrieval pipeline                    
                                                                                                                                                                    
  The 10-stage pipeline from RAG.md, extended with Tier 2 routing (intent classifier + branching):
                                                                                                                                                                      
  ┌─────┬───────────────────────────────────┬────────────────────────────────────────┬────────────────────────────┬───────────┬──────────────────────────────────┐    
  │  #  │               Stage               │               File / fn                │            Flag            │  Default  │            KB 1 live             │    
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤    
  │ 0   │ RBAC + per-KB config merge        │ chat_rag_bridge.py:311                 │ —                          │ always-on │ ✓                                │  
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤  
  │ 1   │ Query rewrite (history-aware)     │ query_rewriter.py                      │ RAG_DISABLE_REWRITE        │ 1 (OFF)   │ OFF                              │    
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤    
  │ 1b  │ Intent classify (Tier 2)          │ query_intent.classify                  │ RAG_INTENT_ROUTING         │ 0         │ ON                               │    
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤    
  │ 2   │ Embed + HyDE                      │ retriever.py, hyde.py                  │ RAG_HYDE, RAG_HYDE_N       │ 0 / 1     │ ON (n=2; forced OFF on global    │  
  │     │                                   │                                        │                            │           │ path)                            │    
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤  
  │ 3   │ Semantic cache lookup             │ retrieval_cache.py                     │ RAG_SEMCACHE, _TTL         │ 0 / 300   │ ON                               │    
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤  
  │ 4   │ Level pre-filter (Tier 1)         │ vector_store._build_filter             │ via intent                 │ None      │ doc/chunk by intent              │    
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤  
  │ 5   │ Parallel Qdrant fan-out (hybrid   │ retriever.py,                          │ RAG_HYBRID                 │ 1         │ ON (forced OFF on global path)   │    
  │     │ RRF)                              │ vector_store.hybrid_search             │                            │           │                                  │  
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤    
  │ 6   │ Rerank (cross-encoder)            │ reranker.py, cross_encoder_reranker.py │ RAG_RERANK                 │ 0         │ ON top_k=30 (OFF on global)      │  
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤    
  │ 7   │ MMR                               │ chat_rag_bridge.py:501, mmr.py         │ RAG_MMR, _LAMBDA           │ 0 / 0.7   │ ON (OFF on global)               │
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤    
  │ 8   │ Context expansion                 │ context_expand.py                      │ RAG_CONTEXT_EXPAND,        │ 0 / 1     │ ON (window=2, OFF on global)     │
  │     │                                   │                                        │ _WINDOW                    │           │                                  │    
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤
  │ 9   │ Token budget                      │ budget.py                              │ RAG_BUDGET_TOKENIZER       │ gemma-4   │ 5000 tokens                      │  
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤    
  │ 10  │ Spotlight wrap                    │ spotlight.py                           │ RAG_SPOTLIGHT              │ 0         │ ON                               │
  ├─────┼───────────────────────────────────┼────────────────────────────────────────┼────────────────────────────┼───────────┼──────────────────────────────────┤    
  │ 11  │ KB catalog preamble               │ chat_rag_bridge.py:676                 │ RAG_KB_CATALOG_MAX         │ 500       │ ON (subtag-aware)                │
  └─────┴───────────────────────────────────┴────────────────────────────────────────┴────────────────────────────┴───────────┴──────────────────────────────────┘    
                                                    
  Tier-2 branching logic (explicit)                                                                                                                                   
                                                    
  if RAG_INTENT_ROUTING == "0":                                                                                                                                       
      # Byte-identical to pre-Tier-2. _intent = "specific", _level_filter = None.
  else:                                                                                                                                                               
      _intent, _reason = classify_with_reason(query)
      if _intent == "metadata":                                                                                                                                       
          raw_hits = []           # catalog preamble answers
      elif _intent == "global":                                                                                                                                       
          # Disable query-vector contamination:     
          with flags.with_overrides({"RAG_HYDE":"0","RAG_HYDE_N":"0","RAG_HYBRID":"0"}):                                                                              
              raw_hits = retrieve(level_filter="doc", per_kb=50, total=100, ...)                                                                                      
          # Skip rerank/MMR/expand (summaries self-contained)                                                                                                         
      else:  # "specific"                                                                                                                                             
          raw_hits = retrieve(level_filter="chunk", per_kb=10, total=30, ...)                                                                                         
          # Full pipeline (rerank/MMR/expand if flagged)                                                                                                              
                                                                                                                                                                      
  Classification rules (regex fast-path):                                                                                                                             
                                                                                                                                                                      
  ┌──────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────┐   
  │  Label   │                                            Regex patterns (abridged)                                            │             Example              │ 
  ├──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┤   
  │ metadata │ ^(what|which|list|show) (reports?|files?|docs?), how many (reports?...), what files do (i|we) have, give me the │ "what reports do I have?"        │
  │          │  list                                                                                                           │                                  │ 
  ├──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┤   
  │ global   │ ^list (all|every), every (report|file|date), across all, summarize (all|the entire), enumerate                  │ "list every report from january" │   
  ├──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┤   
  │ specific │ default (no pattern matched)                                                                                    │ "what did jan 5 say about        │   
  │          │                                                                                                                 │ outages"                         │ 
  └──────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────┘   
   
  Order matters: metadata → global → specific. "list every report from january" = global (over-general metadata rule would have matched "list" otherwise; explicitly  
  tested in test_list_all_reports_is_global_not_metadata).
                                                                                                                                                                      
  ---                                               
  6. Complete flags reference                                                                                                                                       
                                                                                                                                                                      
  Code defaults (env-readable, overridable per-KB via rag_config)
                                                                                                                                                                      
  ┌───────────────────────────────┬───────────────┬───────────────────────────────────────────────────────┬──────────────┬──────────────────────────────────┐         
  │             Flag              │    Default    │                        Purpose                        │ KB 1 overlay │             Net live             │         
  ├───────────────────────────────┼───────────────┼───────────────────────────────────────────────────────┼──────────────┼──────────────────────────────────┤         
  │ RAG_HYBRID                    │ 1             │ Dense+sparse RRF on hybrid collections                │ —            │ ON                               │       
  ├───────────────────────────────┼───────────────┼───────────────────────────────────────────────────────┼──────────────┼──────────────────────────────────┤       
  │ RAG_RERANK                    │ 0             │ Cross-encoder rerank                                  │ true         │ ON                               │         
  ├───────────────────────────────┼───────────────┼───────────────────────────────────────────────────────┼──────────────┼──────────────────────────────────┤         
  │ RAG_RERANK_TOP_K              │ auto          │ Pool size for reranker                                │ 30           │ 30                               │         
  ├───────────────────────────────┼───────────────┼───────────────────────────────────────────────────────┼──────────────┼──────────────────────────────────┤         
  │ RAG_RERANK_DEVICE             │ auto → cuda:0 │ Reranker GPU                                          │ —            │ cuda:0                           │       
  ├───────────────────────────────┼───────────────┼───────────────────────────────────────────────────────┼──────────────┼──────────────────────────────────┤         
  │ RAG_MMR                       │ 0             │ Diversification                                       │ true         │ ON                               │
  ├────────────────────────────────┼─────────────┼─────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────┤     
  │ RAG_MMR                        │ 0           │ Diversification                                 │ true        │ ON                                           │
  ├────────────────────────────────┼─────────────┼─────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────┤     
  │ RAG_MMR_LAMBDA                 │ 0.7         │ MMR balance (higher = more relevance, less      │ 0.7         │ 0.7                                          │
  │                                │             │ diversity)                                      │             │                                              │     
  ├────────────────────────────────┼─────────────┼─────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────┤
  │ RAG_CONTEXT_EXPAND             │ 0           │ Add sibling chunks ±window                      │ true        │ ON                                           │     
  ├────────────────────────────────┼─────────────┼─────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────┤
  │ RAG_CONTEXT_EXPAND_WINDOW      │ 1           │ Window size                                     │ 2           │ 2                                            │     
  ├────────────────────────────────┼─────────────┼─────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────┤
  │ RAG_HYDE                       │ 0           │ Hypothetical-document embedding                 │ true        │ ON (n=2; global path forces off)             │     
  ├────────────────────────────────┼─────────────┼─────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────┤
  │ RAG_HYDE_N                     │ 1           │ Number of hypotheticals                         │ 2           │ 2                                            │     
  ├────────────────────────────────┼─────────────┼────────────────────────────────────────────────┼─────────────┼───────────────────────────────────────────────┤
  │ RAG_SPOTLIGHT                  │ 0           │ OWASP LLM01 defense: ZWJ defang + UNTRUSTED    │ true        │ ON                                            │     
  │                                │             │ tags                                           │             │                                               │
  ├────────────────────────────────┼─────────────┼────────────────────────────────────────────────┼─────────────┼───────────────────────────────────────────────┤     
  │ RAG_SEMCACHE                   │ 0           │ Redis semantic retrieval cache                 │ true        │ ON                                            │
  ├────────────────────────────────┼─────────────┼────────────────────────────────────────────────┼─────────────┼───────────────────────────────────────────────┤     
  │ RAG_SEMCACHE_TTL               │ 300         │ Cache entry TTL seconds                        │ —           │ 300                                           │
  │ RAG_DISABLE_REWRITE            │ 1                │ Disable history-aware rewrite                │ not         │ OFF (rewrite disabled)                     │     
  │                                │                  │                                              │ mappable    │                                            │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ RAG_INTENT_ROUTING             │ 0                │ Tier 2 intent router                         │ true        │ ON                                         │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ RAG_INTENT_LLM                 │ 0                │ LLM tiebreaker for intent classifier (stub)  │ false       │ OFF                                        │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ RAG_DOC_SUMMARIES              │ 0                │ Tier 1 emit per-doc summary on ingest        │ true        │ ON (ingest-only)                           │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ RAG_CONTEXTUALIZE_KBS          │ 0                │ Anthropic-style chunk contextualization      │ ingest-only │ OFF                                        │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ RAG_CONTEXTUALIZE_CONCURRENCY  │ 8                │ Concurrency cap for contextualize LLM calls  │ —           │ 8                                          │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ RAG_RAPTOR                     │ 0                │ Hierarchical summarization                   │ not         │ OFF                                        │
  │                                │                  │                                              │ mappable    │                                            │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_SYNC_INGEST                │ 1                │ Sync ingest (0 → Celery queue)               │ —           │ ON                                         │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_TEI_MAX_BATCH              │ 32               │ TEI server-side batch cap                    │ —           │ 32                                         │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_KB_CATALOG_MAX             │ 500              │ Cap per KB in catalog preamble               │ —           │ 500                                        │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_QDRANT_QUANTIZE            │ 0                │ INT8 scalar quantization                     │ —           │ OFF                                        │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_QDRANT_RESCORE             │ 1                │ Rescore with fp32 vectors                    │ —           │ ON (no-op sans quant)                      │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_QDRANT_OVERSAMPLING        │ 2.0              │ Rescore pool multiplier                      │ —           │ 2.0                                        │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_QDRANT_M                   │ 16               │ HNSW connectivity                            │ —           │ 16                                         │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_QDRANT_EF_CONSTRUCT        │ 200              │ HNSW build-time accuracy                     │ —           │ 200                                        │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_QDRANT_EF                  │ 128              │ HNSW query-time accuracy                     │ —           │ 128                                        │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_QDRANT_FULL_SCAN_THRESHOLD │ 10000            │ Below → brute force                          │ —           │ 10000 (kb_1 at 2590 uses HNSW since build  │   
  │                                │                  │                                              │             │ finished)                                  │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_QDRANT_MAX_CONNS           │ 32               │ HTTP pool size                               │ —           │ 32                                         │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_QDRANT_ON_DISK_PAYLOAD     │ 0                │ Spill payloads to disk                       │ —           │ OFF                                        │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤
  │ RAG_BUDGET_TOKENIZER           │ gemma-4          │ Tokenizer for budget math                    │ —           │ gemma-4 (HF gated; falls back to cl100k    │     
  │                                │                  │                                              │             │ w/o HF_TOKEN)                              │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ CHUNK_SIZE                     │ 800              │ Tokens per chunk                             │ —           │ 800                                        │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ CHUNK_OVERLAP                  │ 100              │ Overlap tokens                               │ —           │ 100                                        │     
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤   
  │ RAG_FILE_MAX_SIZE              │ 50MB code / 25MB │ Max upload                                   │ —           │ 25MB                                       │     
  │                                │  env             │                                              │             │                                            │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ RAG_VECTOR_SIZE                │ 1024             │ bge-m3 embedding dim                         │ —           │ 1024                                       │
  ├────────────────────────────────┼──────────────────┼──────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────┤     
  │ RAG_MAX_UPLOAD_BYTES           │ 50MB             │ Safety cap                                   │ —           │ 50MB                                       │
  └────────────────────────────────┴──────────────────┴──────────────────────────────────────────────┴─────────────┴────────────────────────────────────────────┘     
   
  Session-stage shortcuts (per-request overlays)                                                                                                                      
                                                    
  The global intent path temporarily overrides inside _run_pipeline:                                                                                                  
  {"RAG_HYDE":"0", "RAG_HYDE_N":"0", "RAG_HYBRID":"0"}
  # plus _short_circuit_quality = True disables rerank/MMR/expand                                                                                                     
                                                                                                                                                                      
  ---                                                                                                                                                                 
  7. Per-KB rag_config schema                                                                                                                                         
                                                                                                                                                                      
  Stored as JSONB on knowledge_bases.rag_config. Admin stamps preferences per KB; at request time the bridge merges configs across all selected KBs (UNION for bools,
  MAX for ints) and wraps the pipeline in flags.with_overrides.                                                                                                       
                                                    
  Valid keys (from ext/services/kb_config.py::VALID_BOOL_KEYS and VALID_INT_KEYS)                                                                                     
                                                    
  Boolean keys:                                                                                                                                                       
  - hybrid, rerank, mmr, context_expand, hyde, spotlight, semcache
  - contextualize (INGEST_ONLY — stripped from request overlay)                                                                                                       
  - doc_summaries (INGEST_ONLY)                                
  - intent_routing, intent_llm                                                                                                                                        
                                                    
  Integer keys:                                                                                                                                                       
  - rerank_top_k, mmr_lambda (float stored as number), context_expand_window, hyde_n
                                                                                                                                                                      
  Current KB 1 (Comn) state                         
                                                                                                                                                                      
  {                                                 
    "mmr": true, "hyde": true, "hyde_n": 2,                                                                                                                           
    "rerank": true, "rerank_top_k": 30,                                                                                                                               
    "semcache": true,                                                                                                                                               
    "spotlight": true,                                                                                                                                                
    "mmr_lambda": 0.7,                              
    "context_expand": true, "context_expand_window": 2,                                                                                                               
    "doc_summaries": true,                          
    "intent_routing": true                                                                                                                                            
  }
                                                                                                                                                                      
  Merge policy (kb_config.merge_configs)                                                                                                                              
                                                                                                                                                                    
  - Bool: UNION (any true wins — strictest-safety wins).                                                                                                              
  - Int: MAX (wider window / higher top_k wins).    
  - Request overlay: INGEST_ONLY keys stripped.                                                                                                                       
  - Process env remains untouched: all overrides go through contextvars, safe across asyncio.gather.
                                                                                                                                                                      
  ---                                               
  8. Selection config schema (chat.meta.kb_config)                                                                                                                    
                                                                                                                                                                      
  Stored as a list on chat.meta.kb_config:                                                                                                                          
                                                                                                                                                                      
  [                                                 
    { "kb_id": 4 },                           // whole KB 4                                                                                                         
    { "kb_id": 5, "subtag_ids": [] },         // whole KB 5 (same as above)                                                                                           
    { "kb_id": 7, "subtag_ids": [12, 15] }    // subtags 12+15 of KB 7 only                                                                                           
  ]                                                                                                                                                                   
                                                                                                                                                                      
  - null / missing → no RAG (pure LLM).                                                                                                                               
  - [] → no RAG.                                                                                                                                                    
  - Each entry validated: user must have kb_access to kb_id; each subtag_id must belong to kb_id.                                                                     
  - Config is locked once the chat has any role=="user" message — re-submit returns 409.                                                                              
                                                                                                                                                                      
  ---                                                                                                                                                                 
  9. Live infrastructure snapshot                                                                                                                                     
                                                                                                                                                                      
  Qdrant                                                                                                                                                            
                                                                                                                                                                      
  ┌────────────────────────┬──────────────────────┬─────────┬────────────┬────────┬───────────────────────────────────────────────────────────────────────────────┐   
  │       Collection       │        Points        │ Indexed │   Shape    │ Sparse │                                Payload indexes                                │ 
  ├────────────────────────┼──────────────────────┼─────────┼────────────┼────────┼───────────────────────────────────────────────────────────────────────────────┤   
  │ kb_1 (alias →          │ 2590 (+ 108 summary  │ 2589    │ dense+bm25 │ ✓      │ kb_id integer 2590/2590, subtag_id integer 2590/2590, doc_id integer          │
  │ kb_1_rebuild)          │ points)              │         │            │        │ 2590/2590, owner_user_id keyword 2590/2590                                    │ 
  ├────────────────────────┼──────────────────────┼─────────┼────────────┼────────┼───────────────────────────────────────────────────────────────────────────────┤   
  │ kb_eval                │ 130                  │ 130     │ dense+bm25 │ ✓      │ kb_id keyword 130/130                                                         │
  └────────────────────────┴──────────────────────┴─────────┴────────────┴────────┴───────────────────────────────────────────────────────────────────────────────┘   
                                                    
  Postgres                                                                                                                                                            
                                                    
  - 1 KB (comn, id=1), 1 subtag (reports, id=1), 110 docs (108 with doc_summary populated).                                                                           
  - Migrations applied: 001-008.
                                                                                                                                                                      
  VRAM (48 GB RTX 6000 Ada)                                                                                                                                           
                                                                                                                                                                      
  ┌──────────────────────────────────────┬──────────────────────┐                                                                                                     
  │                Tenant                │        Usage         │                                                                                                   
  ├──────────────────────────────────────┼──────────────────────┤                                                                                                     
  │ vllm-chat (Gemma 4 31B AWQ)          │ 28.9 GB              │
  ├──────────────────────────────────────┼──────────────────────┤                                                                                                     
  │ TEI (bge-m3)                         │ 1.6 GB               │                                                                                                     
  ├──────────────────────────────────────┼──────────────────────┤                                                                                                     
  │ frame workers + rtsp (other tenants) │ ~8.5 GB              │                                                                                                     
  ├──────────────────────────────────────┼──────────────────────┤                                                                                                     
  │ Reranker (lazy, after first chat)    │ +2.2 GB              │                                                                                                   
  ├──────────────────────────────────────┼──────────────────────┤                                                                                                     
  │ Total                                │ ~42 GB / 49 GB (86%) │
  └──────────────────────────────────────┴──────────────────────┘                                                                                                     
                                                    
  Redis                                                                                                                                                               
                                                    
  - Used for semcache + rerank-cache. Flushed recently; warms naturally with traffic.                                                                                 
   
  Containers                                                                                                                                                          
                                                    
  - orgchat-open-webui (image 8f2ce71699f5), orgchat-postgres, orgchat-qdrant, orgchat-redis, orgchat-whisper, kairos-tts, vllm-chat, TEI, observability stack        
  (Grafana/Jaeger/Loki/Prometheus).
                                                                                                                                                                      
  ---                                               
  10. End-to-end behavior by query type                                                                                                                             
                                       
  ┌───────────────────────────────────┬──────────┬───────────────────────────────────────────────────────────┬───────────────────────────────────────────────────┐ 
  │            Query shape            │  Intent  │                        What fires                         │                  What user sees                   │    
  ├───────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤    
  │ "what reports do I have?"         │ metadata │ Retrieve skipped; catalog preamble returned               │ Bulleted list of all filenames in selected scope, │    
  │                                   │          │                                                           │  grouped by KB/subtag                             │    
  ├───────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤    
  │ "list every date covered"         │ global   │ Dense search over level="doc" summary points, top-100;    │ Per-document summaries with dates, enumerated     │    
  │                                   │          │ catalog preamble + summaries                              │                                                   │    
  ├───────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤    
  │ "summarize everything about jan"  │ global   │ Same as above                                             │ Thematic per-document summaries                   │  
  ├───────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤    
  │ "what did 05 Jan 2026 report say  │ specific │ Dense + BM25 RRF + rerank + MMR + expand, level="chunk"   │ Specific chunks from the matching doc             │ 
  │ about outages"                    │          │ filter                                                    │                                                   │    
  ├───────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤    
  │ "show me the roi numbers"         │ specific │ Same                                                      │ Specific chunks with ROI content                  │  
  ├───────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤    
  │ "" (empty selection)              │ N/A      │ No retrieve, no catalog                                   │ Pure LLM                                          │
  └───────────────────────────────────┴──────────┴───────────────────────────────────────────────────────────┴───────────────────────────────────────────────────┘    
                                                    
  ---                                                                                                                                                                 
  11. Isolation & security (unchanged, confirmed working)
                                                                                                                                                                      
  - Row-level RBAC: every request calls get_allowed_kb_ids(user_id) → filters kb_config to allowed KBs before retrieve.
  - Qdrant payload filters: owner_user_id for chat-private; kb_id AND subtag_id for KB queries. Integer-typed indexes now populated for fast filtered-HNSW.           
  - Subtag belongs-to-KB: validated server-side on config PUT.                                                                                                        
  - Lock-after-first-message: 409 CONFLICT if chat has any role=="user" message.                                                                                      
  - Spotlight OWASP LLM01: ZWJ defang + <UNTRUSTED_RETRIEVED_CONTENT> tags wrap retrieved text (verified in today's sample output).                                   
                                                                                                                                                                      
  ---                                                                                                                                                                 
  12. Known limitations & suggested next steps                                                                                                                        
                                                                                                                                                                    
  ┌─────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────┬─────────────────────────┐  
  │  #  │                                                 Gap                                                 │         Effort         │        Priority         │
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤    
  │ 1   │ "Lost in the middle" context reordering — reorder so top-relevance chunks are at prompt start +     │ ~20 LOC                │ High (5-15% free lift)  │
  │     │ end, not middle                                                                                     │                        │                         │    
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤    
  │ 2   │ Bridge output metadata doesn't propagate level — debug reporting shows "chunk" for summaries        │ ~5 LOC                 │ Low (cosmetic)          │    
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤    
  │ 3   │ 2 docs have empty doc_summary — Gemma returned "" during backfill                                   │ Re-run backfill        │ Low                     │    
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤  
  │ 4   │ HF_TOKEN unset — Gemma tokenizer may silently fall back to cl100k → 10-15% token-count drift        │ Add token              │ Medium                  │    
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤
  │ 5   │ contextualize-on-ingest not yet enabled on kb_1 — 35-49% retrieval failure reduction available per  │ Re-ingest window (~5   │ High                    │    
  │     │ Anthropic                                                                                           │ min)                   │                         │
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤    
  │ 6   │ LLM tiebreaker for intent classifier is a stub                                                      │ ~50 LOC                │ Low (regex is doing     │
  │     │                                                                                                     │                        │ fine)                   │    
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤
  │ 7   │ classify_intent (original log-only stub) still exists alongside the new routing                     │ Remove                 │ Low (cleanup)           │    
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤    
  │ 8   │ RAPTOR not enabled — hierarchical thematic queries would benefit                                    │ Re-ingest + flag       │ Medium                  │  
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤    
  │ 9   │ Reranker v3.5 / jina-reranker-v2 A/B vs bge-reranker-v2-m3                                          │ Spike                  │ Medium                  │
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┤    
  │ 10  │ Per-doc summary cap at one per doc — doesn't stratify by section; future option to emit summaries   │ Design                 │ Low                     │
  │     │ per heading                                                                                         │                        │                         │    
  └─────┴─────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────┴─────────────────────────┘
                                                                                                                                                                      
  Suggested path forward                            
                                                                                                                                                                    
  - Phase 1 (week 1): Enable contextualize-on-ingest for kb_1 (re-ingest), ship "lost in the middle" reordering, add HF_TOKEN.                                        
  - Phase 2 (week 2): Propagate level into bridge output metadata; test intent LLM tiebreaker; re-run backfill on the 2 empty-summary docs.
  - Phase 3 (optional): RAPTOR on a test KB; jina-reranker-v2 A/B.                                                                                                    
                                                                                                                                                                      
  ---                                                                                                                                                                 
  13. Files of record                                                                                                                                                 
                                                                                                                                                                    
  - RAG.md — hand-maintained architecture doc (may need a refresh pass to reflect today's session).                                                                 
  - CLAUDE.md — project + design context (unchanged this session).                                                                                                    
  - ext/services/ — all RAG services; chat_rag_bridge.py is the orchestrator.                                                                                         
  - ext/db/migrations/ — 001 → 008.                                                                                                                                   
  - scripts/backfill_doc_summaries.py — idempotent backfill (now ships in image).                                                                                     
  - compose/docker-compose.yml + .env — service topology and tuning.                                                                                                  
  - Dockerfile.openwebui — now copies whole scripts/ dir.