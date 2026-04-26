# OCR Runbook

Plan B Phase 6.8.

## When to enable OCR

Default: `RAG_OCR_ENABLED=1` (Plan B Phase 6.4 turns this on globally).
Per-KB override: `kb_config.ocr_policy.enabled = false` to disable for KBs known to be all-text (faster ingest).

## Symptoms a KB needs OCR

- Retrieval misses on docs that visually contain text but the search returns nothing relevant.
- Doc inspection shows pdfplumber extracted < 50 chars per page.
- Documents are scanned PDFs, image-only PDFs, or photos.

Detect:

```bash
python - <<PY
import asyncio
from qdrant_client import AsyncQdrantClient
from collections import defaultdict

async def main():
    qc = AsyncQdrantClient(url="http://localhost:6333")
    by_doc = defaultdict(int)
    sizes = defaultdict(int)
    offset = None
    while True:
        points, offset = await qc.scroll(
            collection_name="kb_1", limit=512, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points: break
        for p in points:
            payload = p.payload or {}
            did = payload.get("doc_id")
            txt = payload.get("text", "")
            by_doc[did] += 1
            sizes[did] += len(txt)
        if offset is None: break
    short = sum(1 for d in by_doc if sizes[d] / max(1, by_doc[d]) < 100)
    print(f"docs with avg chunk < 100 chars: {short}/{len(by_doc)}")

asyncio.run(main())
PY
```

If > 5% of docs are short — that KB likely needs OCR.

## Enable OCR for a KB

```bash
KB_ID=2
curl -X PATCH "http://localhost:6100/api/kb/$KB_ID/config" \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ocr_policy": {
      "enabled": true,
      "backend": "tesseract",
      "language": "eng",
      "trigger_chars_per_page": 50
    }
  }'
```

For non-English content, set `language` to a Tesseract language code (e.g. `"deu"` for German, `"chi_sim"` for Simplified Chinese — install the language pack first via `apt-get install tesseract-ocr-deu`).

## Use a cloud backend (opt-in only)

**Air-gap warning:** the cloud backends MAKE OUTBOUND CALLS. Only enable on hosts that are still connected.

AWS Textract:

```bash
# 1. Install boto3 in open-webui
docker exec orgchat-open-webui pip install boto3

# 2. Set creds in .env
cat >> compose/.env <<EOF
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
TEXTRACT_REGION=us-east-1
EOF

# 3. Restart
docker compose up -d --force-recreate open-webui

# 4. Per-KB switch
curl -X PATCH "http://localhost:6100/api/kb/$KB_ID/config" \
  -d '{"ocr_policy": {"backend": "cloud:textract"}}'
```

GCP Document AI:

```bash
docker exec orgchat-open-webui pip install google-cloud-documentai
# Mount service account JSON; set DOCUMENT_AI_PROJECT, DOCUMENT_AI_PROCESSOR
# Then PATCH ocr_policy.backend = "cloud:document_ai"
```

## Re-ingest existing docs after enabling OCR

```bash
python scripts/reingest_for_ocr.py \
  --kb-id $KB_ID \
  --short-text-threshold 100 \
  --api-base http://localhost:6100 \
  --admin-token $RAG_ADMIN_TOKEN
```

Inspect output: per-doc OCR latency + char count delta. Re-run eval after.

## Common failure modes

### Tesseract fails: "Could not find tesseract language data"

The container doesn't have the language pack.

```bash
docker exec -u root orgchat-open-webui apt-get install -y tesseract-ocr-<lang>
docker exec orgchat-open-webui python -c "import pytesseract; print(pytesseract.get_languages())"
```

### OCR text is gibberish

- Wrong DPI (default 200 in `ocr.py`); raise to 300 if scans are low-res
- Wrong language; verify with `pytesseract.get_languages()`
- Page is rotated; pre-process via PIL: `image.rotate(90)`

### OCR is too slow

- Reduce DPI to 150 (faster but lower quality)
- Use cloud backend (Textract / Document AI) — typically 5-10x faster than local Tesseract
- Per-KB: set `ocr_policy.enabled=false` for KBs that don't need it (skip the trigger check entirely)

### Cloud backend errors with "Access Denied"

- IAM role not attached to host (Textract)
- Service account JSON missing or wrong scope (Document AI)
- Region mismatch
