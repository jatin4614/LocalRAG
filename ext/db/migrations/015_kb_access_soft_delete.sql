-- 015_kb_access_soft_delete.sql
-- Adds soft-delete support to kb_access for parity with knowledge_bases,
-- kb_documents, and kb_subtags. kb_access is currently hard-DELETEd in
-- service code; the column lets that switch to a tombstone update if
-- the operator wants an audit trail of revocations.
BEGIN;

ALTER TABLE kb_access ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_kb_access_not_deleted
  ON kb_access(id) WHERE deleted_at IS NULL;

COMMIT;
