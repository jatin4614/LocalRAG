-- 013_partial_unique_kb_name.sql
-- Bug fix: bare UNIQUE(name) on knowledge_bases (migration 001) blocked
-- recreating a KB with the same name after soft-delete. The soft-deleted
-- row's name still occupied the unique slot, so create_kb() failed with
-- a duplicate-key violation. Replace with a partial unique index that
-- only applies to live (non-deleted) rows.
--
-- Idempotent. Safe to re-run.

BEGIN;

-- Drop the original constraint from migration 001. Postgres' default name
-- for an inline UNIQUE(name) on a new table is <table>_<column>_key.
ALTER TABLE knowledge_bases DROP CONSTRAINT IF EXISTS knowledge_bases_name_key;

-- Drop our own partial index too in case a prior application created it
-- (re-run safety / experimental rollouts).
DROP INDEX IF EXISTS uq_kb_name_live;

-- Partial unique index: enforce name uniqueness only across non-deleted
-- rows. Soft-deleted rows can keep their name without conflict, so a new
-- KB can claim the same name immediately after the soft-delete.
CREATE UNIQUE INDEX uq_kb_name_live
  ON knowledge_bases (name)
  WHERE deleted_at IS NULL;

COMMIT;
