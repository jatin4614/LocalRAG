-- Wave 2 (review §1.4): partial UNIQUE index on (kb_id, blob_sha).
--
-- Re-uploading the same PDF twice into the same KB previously yielded
-- two kb_documents rows + two doc_ids + two parallel sets of Qdrant
-- points — silently doubling retrieval results for duplicates.
--
-- This migration is GUARDED: if pre-existing duplicates exist, the
-- index creation will fail. The operator must dedupe first using the
-- recipe in the NOTICE block below. Once the duplicates are gone,
-- re-run apply_migrations.py.
--
-- DOWN: ext/db/migrations/down/016_kb_documents_blob_sha_unique.sql
--       drops the index. Doc-level data is untouched (no rows deleted
--       by this migration's UP path).

-- Probe for duplicates BEFORE attempting the unique index.
DO $$
DECLARE
    dup_groups INT;
BEGIN
    SELECT count(*) INTO dup_groups FROM (
        SELECT kb_id, blob_sha
        FROM kb_documents
        WHERE blob_sha IS NOT NULL AND deleted_at IS NULL
        GROUP BY 1, 2
        HAVING count(*) > 1
    ) AS d;
    IF dup_groups > 0 THEN
        RAISE NOTICE 'Migration 016: % duplicate (kb_id, blob_sha) groups detected.', dup_groups;
        RAISE NOTICE 'Dedup recipe (run manually as superuser, then re-apply this migration):';
        RAISE NOTICE '  WITH ranked AS (';
        RAISE NOTICE '    SELECT id, kb_id, blob_sha, uploaded_at,';
        RAISE NOTICE '      ROW_NUMBER() OVER (PARTITION BY kb_id, blob_sha ORDER BY uploaded_at DESC, id DESC) AS rn';
        RAISE NOTICE '    FROM kb_documents';
        RAISE NOTICE '    WHERE blob_sha IS NOT NULL AND deleted_at IS NULL';
        RAISE NOTICE '  )';
        RAISE NOTICE '  UPDATE kb_documents SET deleted_at = now() WHERE id IN (SELECT id FROM ranked WHERE rn > 1);';
        RAISE NOTICE '';
        RAISE NOTICE 'Then run delete_orphan_chunks.py to GC the doubled Qdrant points.';
        RAISE EXCEPTION 'Migration 016 aborted: dedup duplicate (kb_id, blob_sha) rows first.';
    END IF;
END $$;

-- Partial unique index — only enforced for non-deleted, sha-bearing rows.
-- Wrapped IF NOT EXISTS so re-running the migration after a partial
-- failure is a no-op.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_kb_doc_blob_sha
  ON kb_documents (kb_id, blob_sha)
  WHERE blob_sha IS NOT NULL AND deleted_at IS NULL;
