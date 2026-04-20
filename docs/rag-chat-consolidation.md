# Per-Chat Collection Consolidation (P2.3)

## Why

Before P2.3, each chat got its own Qdrant collection (`chat_{chat_id}`). With
50 active users × 5 chats each, that's 250 collections — and each one incurs
fixed per-collection memory overhead (HNSW graphs, payload indexes). This
doesn't scale.

## After

One collection — `chat_private` — tenant-partitioned via Qdrant's
`is_tenant=True` payload indexes (P2.1) on `chat_id` + `owner_user_id`. The
server's filtered-HNSW then skips cross-tenant sub-graphs during search,
giving us O(tenant-size) lookups without the per-collection bloat.

## Backward compatibility

New uploads → `chat_private`.
Reads → **dual-read** from both `chat_private` (primary) and
`chat_{chat_id}` (legacy fallback). Merged + deduped by point id; higher
score wins on tie. Dual-read is idempotent: once a chat is migrated, the
legacy collection is empty (or deleted) and contributes nothing.

## Migration

Run against a live stack. Dry-run default.

```bash
# 1. Preview
python scripts/migrate_chat_collections.py --qdrant-url http://localhost:6333 --dry-run

# 2. Execute — copies points into chat_private, leaves sources in place
python scripts/migrate_chat_collections.py --qdrant-url http://localhost:6333 --apply

# 3. Verify retrieval still works (check a known chat's docs surface)

# 4. Delete legacy collections once verified
python scripts/migrate_chat_collections.py --qdrant-url http://localhost:6333 --apply --delete-source
```

The script:
- Copies points with their dense vectors, payloads, and ids preserved.
- Computes `bm25` sparse vectors via fastembed for hybrid support.
- Injects `chat_id = <parsed_from_collection_name>` when the payload is missing
  it (legacy data didn't stamp it).
- Skips any collection already named `chat_private` (idempotent).

## Tenant isolation

Inside `chat_private`, each retrieval sends a filter like:
```
must = [chat_id == <requested>, owner_user_id == <requesting user>]
```

Qdrant's `is_tenant=True` index turns both into O(tenant-size) probes. Two
users querying the same chat see zero cross-pollination — the `owner_user_id`
filter is defense-in-depth on top of our RBAC layer.

## Rollback

Private chat docs are ephemeral by design. If anything goes wrong with the
migration or `chat_private` collection, users can re-upload via the chat UI.
Legacy `chat_{chat_id}` collections remain untouched until you pass
`--delete-source`, so recovery from a bad migration is:
1. Don't pass `--delete-source` on first run.
2. If verification fails, delete `chat_private` (`curl -X DELETE
   http://localhost:6333/collections/chat_private`) and re-run after fixing.
3. Dual-read falls back to legacy collections automatically.

## Config

| Key                           | Default       | Effect                              |
| ----------------------------- | ------------- | ----------------------------------- |
| `CHAT_PRIVATE_COLLECTION`     | `chat_private`| Target collection name (constant)   |
| `RAG_HYBRID`                  | `1`           | Hybrid retrieval honored for this collection  |

No new env knobs — the consolidation is structural.
