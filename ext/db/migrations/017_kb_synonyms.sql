-- Phase 2 / Item 4 — per-KB synonym/abbreviation table.
--
-- Stored as a JSONB array of equivalence classes. Each class is an array
-- of strings that all refer to the same thing. Used by the entity-text
-- filter / boost path to expand the user's entity name to all known
-- variants before MatchText (or boost-score evaluation).
--
-- Example for KB 2 (military intel):
--   [
--     ["5 PoK", "5 POK", "5 PoK Bde", "5 POK Bde", "Pakistan-Occupied Kashmir"],
--     ["75 Inf", "75 INF", "75 Inf Bde", "75 Infantry Brigade"],
--     ["Inf Bde", "Infantry Brigade"]
--   ]
--
-- Default `[]` is no-op.
--
-- Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2.1

ALTER TABLE knowledge_bases
    ADD COLUMN IF NOT EXISTS synonyms JSONB NOT NULL DEFAULT '[]'::jsonb;
