-- 000_schema_migrations.sql
-- Migration history table. Always runs first via the alphabetical sort
-- in scripts/apply_migrations.py.
--
-- Each row records a successfully applied migration with the SHA-256
-- checksum of its source file. The applier:
--   * skips files whose name + checksum are already recorded
--   * aborts loudly if a recorded file's checksum has changed (drift)
--   * inserts on success after running the migration in its own tx
BEGIN;

CREATE TABLE IF NOT EXISTS schema_migrations (
  name TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  checksum TEXT
);

COMMIT;
