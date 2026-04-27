-- Plan B Phase 6.2 followup: add 'queued' to kb_documents.ingest_status
-- allowed values. The async ingest path (upload.py line 241) sets status
-- to 'queued' between dispatch and pickup, but the original check
-- constraint from Plan A predates async ingest and rejects 'queued'.
--
-- Safe: drops + recreates the check constraint. No data conversion needed
-- (existing rows stay valid since the new set is a strict superset).

ALTER TABLE kb_documents DROP CONSTRAINT IF EXISTS kb_documents_ingest_status_check;
ALTER TABLE kb_documents ADD CONSTRAINT kb_documents_ingest_status_check
  CHECK (ingest_status::text = ANY (ARRAY[
    'pending'::character varying,
    'queued'::character varying,
    'chunking'::character varying,
    'embedding'::character varying,
    'done'::character varying,
    'failed'::character varying
  ]::text[]));
