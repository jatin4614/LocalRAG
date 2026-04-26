-- Plan B Phase 6.3 — per-KB OCR policy
ALTER TABLE knowledge_bases
ADD COLUMN IF NOT EXISTS ocr_policy JSONB NOT NULL DEFAULT '{
  "enabled": true,
  "backend": "tesseract",
  "language": "eng",
  "trigger_chars_per_page": 50
}'::jsonb;

COMMENT ON COLUMN knowledge_bases.ocr_policy IS
  'OCR fallback configuration per KB. Cloud backends (cloud:textract, cloud:document_ai) require operator opt-in and creds.';
