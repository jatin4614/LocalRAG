-- 002_soft_delete_kb.sql — add soft-delete to knowledge_bases.
BEGIN;
ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_kb_not_deleted ON knowledge_bases(id) WHERE deleted_at IS NULL;
COMMIT;
