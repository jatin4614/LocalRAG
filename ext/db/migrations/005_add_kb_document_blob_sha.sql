-- 005_add_kb_document_blob_sha.sql
-- Additive, idempotent. Adds kb_documents.blob_sha so the blob GC can
-- reliably free the sha256-addressed file for a soft-deleted document.
--
-- NULL values mean either (a) legacy rows ingested before this column
-- existed, or (b) rows created by the synchronous ingest path which does
-- not persist a blob. Both are tolerated by ext/services/blob_gc.py.
--
-- Also adds a partial index on deleted_at to make the GC scan efficient
-- (the WHERE clause keeps the index small — only soft-deleted rows).
--
-- deleted_at itself is already present on kb_documents (migration 001).

BEGIN;

ALTER TABLE kb_documents
  ADD COLUMN IF NOT EXISTS blob_sha TEXT;

COMMENT ON COLUMN kb_documents.blob_sha IS
  'sha256 hex of the original upload bytes in the BlobStore, or NULL for '
  'legacy / sync-ingest rows. Populated by async ingest; read by blob GC.';

CREATE INDEX IF NOT EXISTS idx_kb_documents_deleted_at
  ON kb_documents(deleted_at)
  WHERE deleted_at IS NOT NULL;

COMMIT;
