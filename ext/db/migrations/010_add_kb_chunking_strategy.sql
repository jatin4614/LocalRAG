-- Plan B Phase 6.6 — per-KB chunking strategy
-- Adds the field to the rag_config JSONB blob (no schema change beyond default).
-- Use UPDATE to populate the default for existing rows.

UPDATE knowledge_bases
SET rag_config = COALESCE(rag_config, '{}'::jsonb) || '{"chunking_strategy": "window"}'::jsonb
WHERE NOT (rag_config ? 'chunking_strategy');

COMMENT ON COLUMN knowledge_bases.rag_config IS
  'Per-KB RAG config (JSONB). Includes chunking_strategy ("window" | "structured"), contextualize, colbert, etc.';
