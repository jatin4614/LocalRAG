-- Adds doc_summary column to kb_documents for Tier 1 per-doc summary index.
-- Populated by the doc_summarizer service at ingest (when RAG_DOC_SUMMARIES=1)
-- or retroactively by scripts/backfill_doc_summaries.py.
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS doc_summary TEXT NULL;
-- No index needed — summaries are read through kb_documents PK, not searched.
