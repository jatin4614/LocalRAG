-- 004_add_pipeline_version.sql
-- Additive, idempotent. Adds kb_documents.pipeline_version for re-index gating.
-- NULL means "pre-upgrade document, version unknown" — that's the honest
-- signal. A later migration / reindex job can decide what to do with legacy rows.

BEGIN;

ALTER TABLE kb_documents
  ADD COLUMN IF NOT EXISTS pipeline_version TEXT;

COMMENT ON COLUMN kb_documents.pipeline_version IS
  'Composite pipeline version at ingest time; see ext/services/pipeline_version.py. NULL = legacy/unknown.';

COMMIT;
