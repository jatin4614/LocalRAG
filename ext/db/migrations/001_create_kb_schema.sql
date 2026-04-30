-- 001_create_kb_schema.sql
-- Replaces Open WebUI's upstream `knowledge` schema with hierarchical KB tables.
-- Idempotent: safe to re-run.
--
-- IMPORTANT: ``admin_id``, ``uploaded_by``, ``kb_access.user_id`` are
-- VARCHAR(255) and ``kb_access.group_id`` is TEXT — they store the
-- upstream Open WebUI user/group identifiers (UUID strings managed by
-- upstream's auth layer). We deliberately do NOT add foreign keys to
-- ``users(id)`` / ``groups(id)`` because:
--   1. Upstream owns those tables; their schema is not ours to constrain.
--   2. Production was bootstrapped via SQLAlchemy ``create_all`` which
--      respected the ORM model types (String(255) / Text); migrations
--      must reproduce the same shape so ``\d`` matches production.
--   3. CASCADE behaviour for KB grants is enforced at the application
--      layer (rbac cache invalidation + soft-delete propagation).

BEGIN;

-- Drop upstream knowledge tables if present (D2).
DROP TABLE IF EXISTS knowledge_file CASCADE;
DROP TABLE IF EXISTS knowledge CASCADE;

CREATE TABLE IF NOT EXISTS knowledge_bases (
  id          BIGSERIAL    PRIMARY KEY,
  name        VARCHAR(255) NOT NULL,
  description TEXT,
  admin_id    VARCHAR(255) NOT NULL,
  created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
  UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS kb_subtags (
  id          BIGSERIAL    PRIMARY KEY,
  kb_id       BIGINT       NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  name        VARCHAR(255) NOT NULL,
  description TEXT,
  created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
  UNIQUE(kb_id, name)
);

CREATE TABLE IF NOT EXISTS kb_documents (
  id             BIGSERIAL    PRIMARY KEY,
  kb_id          BIGINT       NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  subtag_id      BIGINT       NOT NULL REFERENCES kb_subtags(id)       ON DELETE CASCADE,
  filename       VARCHAR(512) NOT NULL,
  mime_type      VARCHAR(100),
  bytes          BIGINT,
  ingest_status  VARCHAR(20)  NOT NULL DEFAULT 'pending'
                 CHECK (ingest_status IN ('pending','chunking','embedding','done','failed')),
  error_message  TEXT,
  uploaded_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
  uploaded_by    VARCHAR(255) NOT NULL,
  deleted_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_kb_documents_kb      ON kb_documents(kb_id)      WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_kb_documents_subtag  ON kb_documents(subtag_id)  WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_kb_documents_status  ON kb_documents(ingest_status) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS kb_access (
  id          BIGSERIAL    PRIMARY KEY,
  kb_id       BIGINT       NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  user_id     VARCHAR(255),
  group_id    TEXT,
  access_type VARCHAR(20)  NOT NULL DEFAULT 'read'
              CHECK (access_type IN ('read','write')),
  granted_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
  CHECK ( (user_id IS NOT NULL)::int + (group_id IS NOT NULL)::int = 1 )
);
CREATE INDEX IF NOT EXISTS idx_kb_access_user  ON kb_access(user_id)  WHERE user_id  IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kb_access_group ON kb_access(group_id) WHERE group_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kb_access_kb    ON kb_access(kb_id);

-- Extend chats with per-session KB config (§3.2 of workflow spec).
ALTER TABLE chats ADD COLUMN IF NOT EXISTS selected_kb_config JSONB;
CREATE INDEX IF NOT EXISTS idx_chats_kb_config ON chats USING GIN (selected_kb_config)
  WHERE selected_kb_config IS NOT NULL;

COMMIT;
