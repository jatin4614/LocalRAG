-- 007_drop_orphan_selected_kb_config.sql
--
-- Cleanup for unused column `chats.selected_kb_config` and its GIN index
-- `idx_chats_kb_config`, originally declared in 001_create_kb_schema.sql but
-- never populated by any code path. The current KB selection state is stored
-- in upstream's singular `chat.meta.kb_config` (a JSON column), written by
-- `ext/routers/kb_retrieval.py::set_chat_kb_config`. The `chats` table column
-- has no reader and diverged from the canonical storage location.
--
-- Idempotent: no-op if the column / index don't exist.

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'chats' AND column_name = 'selected_kb_config'
  ) THEN
    DROP INDEX IF EXISTS idx_chats_kb_config;
    ALTER TABLE chats DROP COLUMN selected_kb_config;
    RAISE NOTICE 'dropped orphan chats.selected_kb_config';
  ELSE
    RAISE NOTICE 'chats.selected_kb_config absent; migration is a no-op';
  END IF;
END $$;
