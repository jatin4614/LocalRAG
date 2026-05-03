-- Wave 2 (review §4.6): no-op placeholder so the migration sequence is
-- contiguous. The Plan A→B handoff jumped from 008_add_doc_summary.sql
-- to 010_add_kb_chunking_strategy.sql, leaving 009 missing. Operators
-- reading the migrations dir chronologically were forced to wonder
-- whether a migration had been deleted from a corrupt history.
--
-- This file does nothing intentionally. apply_migrations.py treats it
-- as applied (its checksum is recorded in schema_migrations) so future
-- migrations don't trip the gap detector. Do NOT delete this file —
-- removing it would re-create the chronology problem.

-- Intentional no-op (use SELECT 1 so the runner has a statement to execute).
SELECT 1 AS skip_009_intentional;
