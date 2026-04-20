# Blob + Vector Garbage Collection (P2.8)

Closes the soft-delete lifecycle for KB documents. Once a `kb_documents`
row has been soft-deleted (`deleted_at` set) for more than
`RAG_BLOB_RETENTION_DAYS` days, a GC pass will:

1. Delete the on-disk blob via `BlobStore.delete(sha)` (if `blob_sha` is
   populated and the file still exists).
2. Hard-delete the Qdrant points for that `doc_id` via
   `VectorStore.delete_by_doc(collection, doc_id)`. In practice this is
   usually a no-op because the admin soft-delete path
   (`ext/routers/kb_admin.py`) already purges Qdrant at soft-delete time.
   We re-issue defensively — it's cheap, idempotent, and covers any prior
   failure.
3. Hard-delete the `kb_documents` row itself.

After a GC pass, the document is fully gone: no DB row, no blob, no
vectors. Nothing is recoverable.

## Retention policy

Retention is expressed in whole days via `RAG_BLOB_RETENTION_DAYS`
(default: 30). Rows whose `deleted_at` is strictly older than
`now() - retention_days` are eligible.

To keep soft-deleted rows around indefinitely for audit, set
`RAG_BLOB_RETENTION_DAYS=36500` (100 years) — effectively disables GC.

## Running manually

```bash
# Preview (default — no writes):
python scripts/blob_gc.py \
    --database-url "$DATABASE_URL" \
    --blob-root /var/ingest \
    --qdrant-url http://localhost:6333

# Actually delete:
python scripts/blob_gc.py --apply \
    --database-url "$DATABASE_URL" \
    --blob-root /var/ingest \
    --qdrant-url http://localhost:6333

# Override retention for a one-off sweep:
python scripts/blob_gc.py --apply --retention-days 90 ...

# Cap batch size (useful if the backlog is huge and you want to spread
# load over multiple passes):
python scripts/blob_gc.py --apply --limit 100 ...
```

The script prints a JSON summary on stdout:

```json
{
  "rows_processed": 42,
  "blobs_deleted": 30,
  "blobs_missing": 2,
  "blobs_none": 10,
  "qdrant_points_deleted": 42,
  "qdrant_errors": 0,
  "rows_deleted": 42,
  "dry_run": false,
  "retention_days": 30,
  "cutoff": "2026-03-20T00:00:00+00:00"
}
```

Field meanings:

| Field | Meaning |
|---|---|
| `rows_processed` | eligible rows the pass scanned |
| `blobs_deleted` | blob files freed (row had `blob_sha` and file existed) |
| `blobs_missing` | row had `blob_sha` but file was already gone |
| `blobs_none` | row had no `blob_sha` (legacy / sync-ingest) |
| `qdrant_points_deleted` | `delete_by_doc` returned success |
| `qdrant_errors` | `delete_by_doc` raised or returned 0 |
| `rows_deleted` | `kb_documents` rows hard-deleted |
| `cutoff` | ISO8601 of the retention cutoff this pass used |

## Celery periodic task

`ext/workers/blob_gc_task.py` registers `ext.workers.blob_gc_task.blob_gc`
on the `ingest` queue, plus a Celery Beat schedule entry
`blob-gc-daily` that runs the task daily at **03:17 UTC** by default.

**Beat is a separate process from the worker.** The compose file does not
include a beat container today. Until that ships, pick one of:

* **External cron**: wrap `scripts/blob_gc.py --apply` in a cron job on
  the host. Simplest option.
* **Manual trigger**: `celery -A ext.workers.celery_app call
  ext.workers.blob_gc_task.blob_gc` from a worker shell.
* **Run beat alongside worker**: add a sidecar
  `celery -A ext.workers.celery_app beat` process.

### Cron expression

Override the beat schedule via `RAG_BLOB_GC_CRON` (five space-separated
fields, same order as Celery's `crontab(minute, hour, day_of_week,
month_of_year, day_of_month)`):

```bash
# Run every Monday at 04:30 UTC:
export RAG_BLOB_GC_CRON="30 4 mon * *"

# Run every 6 hours:
export RAG_BLOB_GC_CRON="0 */6 * * *"
```

Invalid expressions fall back to the default without killing the worker.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `RAG_BLOB_RETENTION_DAYS` | `30` | Days a soft-deleted row must age before GC |
| `INGEST_BLOB_ROOT` | `/var/ingest` | Blob store root directory |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant REST URL |
| `RAG_VECTOR_SIZE` | `1024` | Embedding dimension |
| `RAG_BLOB_GC_LIMIT` | `1000` | Max rows processed per pass |
| `RAG_BLOB_GC_CRON` | `"17 3 * * *"` | Beat cron (5-field spec) |

## Audit considerations

This is a **hard delete** — nothing survives: no DB row, no blob, no
vectors. If your compliance / audit posture requires that a record of the
deletion persist, you have a few options:

1. **Extend retention** by setting `RAG_BLOB_RETENTION_DAYS` to a very
   large number (e.g. `3650` = 10 years). The soft-deleted row stays
   queryable for that whole window.
2. **Log the GC summary** off-host. The JSON summary contains enough
   detail (`rows_deleted`, `cutoff`) to satisfy most retention audits,
   and the Celery task result is captured by the Celery backend; pipe
   that to durable storage if needed.
3. **Pre-GC hook**: insert an audit row into a separate table before
   calling `run_gc`. Not implemented today — add if/when needed.

The default posture (30 days, hard delete) is appropriate for a
standard org RAG deployment where the legal purpose of soft-delete is to
undo accidental deletes, not to retain data against the user's will.

## Migration 005

`ext/db/migrations/005_add_kb_document_blob_sha.sql` adds:

* `kb_documents.blob_sha TEXT` — nullable; populated by async ingest.
* `idx_kb_documents_deleted_at` — partial index on `deleted_at IS NOT
  NULL`, keeps the GC query fast as the delete backlog grows.

`deleted_at` itself was added in migration 001 / 002 and is already
present.

## Known limitation: legacy rows without `blob_sha`

The current ingest pipeline does **not** yet persist `blob_sha` on the
`kb_documents` row. Synchronous ingest (the default, `RAG_SYNC_INGEST=1`)
never writes to the blob store at all, so those rows legitimately have no
blob to clean up — they are counted as `blobs_none` and GC'd without FS
work. Async ingest (`RAG_SYNC_INGEST=0`) writes bytes to the blob store
but the ingest worker **deletes** the blob on success, so the row would
point at a gone file anyway (counted as `blobs_missing`).

Until ingest is wired to (a) retain the blob after ingest and (b)
capture the sha into `kb_documents.blob_sha`, blob GC is essentially a
row-and-vector GC for soft-deleted docs. That's still useful — it's what
closes the soft-delete lifecycle — but the blob-file deletion path is
only exercised by manual `UPDATE kb_documents SET blob_sha = ...` today.
A follow-up commit should plumb `blob_sha` through ingest.
