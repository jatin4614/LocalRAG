-- 006_add_kb_rag_config.sql
-- Additive, idempotent. Adds a per-KB JSONB column where admins can
-- stamp retrieval-quality preferences (rerank on/off, MMR, context
-- expand window, etc.). Empty dict (the default) preserves byte-
-- identical behaviour — the merged-overrides resolver treats missing
-- keys as "inherit process-level flag" so no existing deployments
-- change unless an admin opts in.
--
-- Solves P3.0: small KBs benefit from fast-path retrieval, year-long
-- KBs need rerank + expand. One global flag can't satisfy both at
-- once; stamping the config on the KB lets the bridge compute a
-- per-request override from the UNION of the selected KBs' configs
-- (strictest wins — any KB requesting rerank enables it for that
-- request).
--
-- The GIN index lets admin UIs query "which KBs have rerank on?" and
-- also speeds any future migrations that need to scan by config key.

BEGIN;

ALTER TABLE knowledge_bases
  ADD COLUMN IF NOT EXISTS rag_config JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN knowledge_bases.rag_config IS
  'Per-KB retrieval quality overrides (rerank, mmr, context_expand, '
  'spotlight, semcache, etc.). Empty dict = inherit process-level '
  'defaults. Merged via UNION/MAX across the chat''s selected KBs by '
  'ext.services.kb_config at request time.';

CREATE INDEX IF NOT EXISTS idx_knowledge_bases_rag_config
  ON knowledge_bases USING gin (rag_config);

COMMIT;
