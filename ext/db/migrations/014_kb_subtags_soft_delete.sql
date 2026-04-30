-- 014_kb_subtags_soft_delete.sql
-- Adds soft-delete support to kb_subtags so subtag removal mirrors KB
-- and document removal (already soft-delete via ``deleted_at``). The
-- partial index keeps the live-row scan path index-only.
BEGIN;

ALTER TABLE kb_subtags ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_kb_subtags_not_deleted
  ON kb_subtags(id) WHERE deleted_at IS NULL;

COMMIT;
