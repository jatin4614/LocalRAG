-- DOWN for migration 016 (review §1.4). Drops the partial unique index.
-- No data rows touched. Safe to re-apply UP after running this.
DROP INDEX IF EXISTS uniq_kb_doc_blob_sha;
