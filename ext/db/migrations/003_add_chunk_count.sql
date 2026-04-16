-- 003_add_chunk_count.sql
BEGIN;
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0;
COMMIT;
