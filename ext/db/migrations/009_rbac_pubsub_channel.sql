-- 009_rbac_pubsub_channel.sql
-- Phase 1.5 — no schema change; documents the Redis pub/sub channel.
--
-- The RBAC cache in ext/services/rbac_cache.py listens on channel
-- 'rbac:invalidate' for user-id payloads; the kb_admin router publishes
-- on this channel after any kb_access mutation so all replicas drop
-- the affected per-user cache entry.
--
-- TTL safety net is RAG_RBAC_CACHE_TTL_SECS (default 30s) so a dropped
-- pub/sub message can leak stale grants for at most that window.
--
-- Cache key namespace: 'rbac:user:{user_id}' (Redis DB index 3, see
-- compose/docker-compose.yml RAG_RBAC_CACHE_REDIS_URL).
--
-- This file exists so the migration runner records the change in the
-- audit log even though there is no DDL to apply. Migration runners
-- that ignore comment-only files will skip it harmlessly.

SELECT 1;
