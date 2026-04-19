# Eval seed corpus

Drop additional public-domain or permissively licensed `.md` / `.txt` files
here to grow the eval corpus. Everything under this directory is picked up
by `scripts/seed_eval_corpus.py` and ingested into the `kb_eval` Qdrant
collection alongside the worktree-docs allowlist.

## Rules
- **No copyrighted material.** Public domain (Project Gutenberg chapters,
  RFCs, government publications) or permissively licensed text only.
- Plain `.md` or `.txt`, UTF-8 encoded.
- Files are sorted alphabetically by relative path, so a new file gets a
  stable `doc_id` that won't shift other files' IDs.
- Keep individual files under ~100 KB of text so chunking stays snappy; if
  you have a long book, split into chapter files.

## Why this dir exists
The worktree documentation alone is ~13K lines of dense project docs which
produces ~80-100 chunks. For stronger eval signal we want **≥ 50 docs and
≥ 100 chunks of varied prose**. If your worktree docs aren't enough, drop
additional source texts here.

## Suggestions (not committed — add manually, air-gapped)
- Project Gutenberg chapter extracts: https://www.gutenberg.org (US public domain)
- RFCs: https://www.rfc-editor.org/rfc-index.html (public domain)
- NASA / NIH publications: most are public domain

## NOT to go here
- Customer data
- Copyrighted content
- Any file >1 MB (will slow the seed script for little gain)
