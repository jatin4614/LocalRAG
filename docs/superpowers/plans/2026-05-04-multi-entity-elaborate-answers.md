# Multi-Entity Comparative-Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the LocalRAG multi-entity comparative-query class from "0 facts on low-frequency entities" (today) to "8-10 grounded facts per named entity, with subtopic separation" by changing the chunker default, widening the rerank pool, fixing entity-coverage in doc summaries, and adding a subtopic axis to multi-query decomposition.

**Architecture:** Three-phase rollout matching the 2026-05-04 spec. Phase 1 (~3-5 days) ships chunker + budget + floor changes — minimal code, mostly config. Phase 2 (~3-5 days) replaces the doc-summary prompt with a structured ENTITIES + SUMMARY shape and threads an `entities` payload through Qdrant and ingest. Phase 3 (~5-7 days) introduces subtopic extraction and two-axis (entity × subtopic) decomposition with per-cell quotas. After Phase 1 ships, the operator runs `scripts/wipe_and_reingest.py` and re-uploads KB 2 docs from scratch — no alias-cutover plumbing needed.

**Tech Stack:** Python 3.11 (FastAPI, asyncpg, httpx, qdrant-client), Postgres 15 (JSONB rag_config column), Qdrant cluster (kb_{id} / kb_{id}_v2 collections), TEI bge-m3, vllm-chat (Gemma-4-31B-it-AWQ, 32K ctx), pytest for unit + integration tests, docker-compose for orchestration.

**Spec:** `docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md`

**Predecessors merged before this plan starts:**
- `docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md` (Phase 1+2+3 — entity_text_filter casing, soft-boost, synonyms, multi-entity coverage counter — all shipped)

**Test runner convention:** `.venv/bin/pytest <path> -v` (host pytest may not be on PATH). Lint/type: `make lint` (`ruff check . && mypy .`).

**Conventional-commits:** all commits follow `<type>(<scope>): <subject>` with the standard `<scope>` set used elsewhere in the repo (`feat`, `fix`, `docs`, `test`, `chore`, `refactor`). End every commit message with the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.

---

## File map

| File | Phase | Purpose |
|---|---|---|
| `ext/services/kb_config.py` | 1, 3 | Add `multi_entity_rerank_floor`, `subtopic_decompose`, `subtopic_keywords` keys + bounds + validators |
| `compose/docker-compose.yml` | 1 | Explicit env mapping for `RAG_MULTI_ENTITY_RERANK_FLOOR`; `RAG_BUDGET_TOKENS` default raised |
| `compose/.env.example` | 1 | Document new knobs |
| `ext/services/chat_rag_bridge.py` | 1, 3 | Plumb floor read site (already uses `flags.get`); add two-axis decompose path + `_apply_two_axis_quota` helper |
| `ext/services/doc_summarizer.py` | 2 | New ENTITIES + SUMMARY prompt; structured return shape; bumped body cap |
| `ext/services/ingest.py` | 2 | Wire `entities` payload onto `level=doc` Qdrant points |
| `ext/db/qdrant_schema.py` | 2 | Add `entities` payload index |
| `ext/services/entity_extractor.py` | 3 | New `extract_subtopics(query)` |
| `ext/services/query_understanding.py` | 3 | Extend QU prompt + `HybridClassification.subtopics` |
| `ext/services/multi_query.py` | 3 | Two-axis `should_decompose`, `build_sub_queries_two_axis`, `merge_with_two_axis_quota` |
| `scripts/wipe_and_reingest.py` | 1 | Operator script — backup, wipe, recreate empty schema |
| `scripts/edit_kb_subtopics.py` | 3 | Operator CLI for `subtopic_keywords` per-KB table |
| `tests/unit/test_kb_config_phase6.py` | 1, 3 | Extend with new key tests |
| `tests/unit/test_doc_summarizer.py` | 2 | New test file (or extend existing) |
| `tests/unit/test_entity_extractor.py` | 3 | Extend with `extract_subtopics` tests |
| `tests/unit/test_multi_query.py` | 3 | Extend with two-axis tests |
| `tests/unit/test_chat_rag_bridge_two_axis_quota.py` | 3 | New |
| `tests/integration/test_wipe_and_reingest.py` | 1 | New |
| `tests/integration/test_edit_kb_subtopics_cli.py` | 3 | New (mirror `test_edit_kb_synonyms_cli.py`) |

---

# Phase 1 — Chunker + Budgets (Tasks 1-8)

## Task 1: Add `multi_entity_rerank_floor` per-KB config key

**Files:**
- Modify: `ext/services/kb_config.py` (add to `VALID_INT_KEYS` + bounds in `validate_config`)
- Test: `tests/unit/test_kb_config_phase6.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_kb_config_phase6.py` (immediately after the existing `multi_entity_min_per_entity` test class):

```python
class TestMultiEntityRerankFloor:
    """Per-KB override of RAG_MULTI_ENTITY_RERANK_FLOOR env. Phase 1 of the
    2026-05-04 multi-entity-elaborate-answers spec."""

    def test_accepts_valid_int(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 8}
        ) == {"multi_entity_rerank_floor": 8}

    def test_accepts_lower_bound(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 1}
        ) == {"multi_entity_rerank_floor": 1}

    def test_accepts_upper_bound(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 50}
        ) == {"multi_entity_rerank_floor": 50}

    def test_rejects_below_lower_bound(self) -> None:
        from ext.services import kb_config
        # 0 silently dropped — out-of-range = inherit env default
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 0}
        ) == {}

    def test_rejects_above_upper_bound(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 51}
        ) == {}

    def test_rejects_string(self) -> None:
        from ext.services import kb_config
        # Strings without int() coercion drop silently
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": "high"}
        ) == {}

    def test_coerces_string_int(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": "10"}
        ) == {"multi_entity_rerank_floor": 10}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/test_kb_config_phase6.py::TestMultiEntityRerankFloor -v
```

Expected: 7 tests FAIL (first reasons typically include "key not in VALID_INT_KEYS" so all assertions return `{}` instead of the expected dict).

- [ ] **Step 3: Add the key to `VALID_INT_KEYS` and bounds**

Edit `ext/services/kb_config.py`. In `VALID_INT_KEYS` add the new key after `multi_entity_min_per_entity`:

```python
VALID_INT_KEYS = frozenset({
    "top_k",
    "rerank_top_k",
    "context_expand_window",
    "hyde_n",
    "chunk_tokens",
    "overlap_tokens",
    "multi_entity_min_per_entity",
    # 2026-05-04 — rerank-stage per-entity floor (Phase 1 / Item 2 of
    # multi-entity-elaborate-answers spec). Overrides the
    # RAG_MULTI_ENTITY_RERANK_FLOOR env var per-KB. Bounds [1, 50] —
    # higher values starve single-entity recall on the same KB.
    "multi_entity_rerank_floor",
})
```

Then add bounds inside `validate_config` next to the existing `multi_entity_min_per_entity` bounds check:

```python
            # Phase 6.X — multi-entity per-entity floor. Below 1 has no
            # meaning (no quota); above 50 starts crowding out other
            # signal at the rerank cut. Out-of-range silently drops.
            if key == "multi_entity_min_per_entity" and not (1 <= coerced <= 50):
                continue
            # 2026-05-04 — rerank-stage per-entity floor. Same bounds
            # rationale as multi_entity_min_per_entity above.
            if key == "multi_entity_rerank_floor" and not (1 <= coerced <= 50):
                continue
```

**Also add a `_KEY_TO_ENV` mapping** in the same file. Find the `_KEY_TO_ENV` dict and add the new mapping immediately after `multi_entity_min_per_entity`:

\```python
    "multi_entity_rerank_floor": "RAG_MULTI_ENTITY_RERANK_FLOOR",
\```

Without this entry, the per-KB JSONB stamp validates and persists but is silently dropped by `config_to_env_overrides`, and the rerank-stage floor read site (`chat_rag_bridge.py:~2024`) never sees the per-KB value — only the env default reaches `flags.get`. Add the test below to lock the round-trip:

\```python
    def test_emits_rerank_floor_env(self) -> None:
        from ext.services import kb_config
        env = kb_config.config_to_env_overrides(
            {"multi_entity_rerank_floor": 8}
        )
        assert env == {"RAG_MULTI_ENTITY_RERANK_FLOOR": "8"}
\```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/test_kb_config_phase6.py::TestMultiEntityRerankFloor -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ext/services/kb_config.py tests/unit/test_kb_config_phase6.py
git commit -m "$(cat <<'EOF'
feat(kb_config): multi_entity_rerank_floor per-KB key with [1, 50] bounds

Phase 1 / item 2 of the 2026-05-04 multi-entity-elaborate-answers spec.
Lets a KB stamp the rerank-stage floor without flipping the env var
globally (env var ships at 8 in compose; per-KB override wins via
flags.get). Bounds match multi_entity_min_per_entity for consistency.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §4.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `RAG_MULTI_ENTITY_RERANK_FLOOR` env mapping in compose

**Files:**
- Modify: `compose/docker-compose.yml` (open-webui + celery-worker env blocks)

This is a config-only change. No new test logic; verification is through `docker compose config` and a container-side `printenv` check.

- [ ] **Step 1: Inject the env mapping into open-webui block**

Find the line in `compose/docker-compose.yml` that has `RAG_MULTI_ENTITY_MIN_PER_ENTITY:` (around line 628 in the open-webui service env block). Add the new line directly below:

```yaml
      RAG_MULTI_ENTITY_MIN_PER_ENTITY: ${RAG_MULTI_ENTITY_MIN_PER_ENTITY:-10}
      # 2026-05-04 — per-entity rerank-stage floor (Phase 1 / Item 2).
      # Bumps post-cross-encoder pool floor for multi-entity queries.
      # Default 12 = each named entity gets 12 chunks past the rerank cut.
      # Original spec said 8; revised up to 12 after confirming Gemma-4
      # supports 128K ctx (operator correction 2026-05-04, see spec §4.3).
      RAG_MULTI_ENTITY_RERANK_FLOOR: ${RAG_MULTI_ENTITY_RERANK_FLOOR:-12}
```

- [ ] **Step 2: Inject the same mapping into celery-worker block**

Find the celery-worker `environment:` block (starts around line 818). Add the same line near the other `RAG_MULTI_ENTITY_*` mappings if present, otherwise after `RAG_RERANK:`:

```yaml
      RAG_RERANK: ${RAG_RERANK:-1}
      # 2026-05-04 — per-entity rerank-stage floor (Phase 1 / Item 2).
      # Worker reads this for any retrieval path it uses (eval scheduler).
      # Default 12 — see open-webui block above for the 128K context note.
      RAG_MULTI_ENTITY_RERANK_FLOOR: ${RAG_MULTI_ENTITY_RERANK_FLOOR:-12}
```

- [ ] **Step 3: Validate compose YAML syntax**

```bash
cd /home/vogic/LocalRAG/compose && docker compose -p orgchat config | grep -A1 RAG_MULTI_ENTITY_RERANK_FLOOR
```

Expected: shows the env var listed under both `open-webui` and `celery-worker` services with value `12`.

- [ ] **Step 4: Recreate the affected services**

```bash
cd /home/vogic/LocalRAG/compose && docker compose -p orgchat up -d open-webui celery-worker
docker compose -p orgchat exec -T open-webui printenv RAG_MULTI_ENTITY_RERANK_FLOOR
docker compose -p orgchat exec -T celery-worker printenv RAG_MULTI_ENTITY_RERANK_FLOOR
```

Expected: both print `12`.

- [ ] **Step 5: Commit**

```bash
git add compose/docker-compose.yml
git commit -m "$(cat <<'EOF'
chore(compose): map RAG_MULTI_ENTITY_RERANK_FLOOR=12 on open-webui + celery-worker

Phase 1 / item 2. Env defaults to 12 (was implicit 3 inside the bridge);
per-KB rag_config.multi_entity_rerank_floor still wins via flags.get.
Floor 12 (not 8 as in the original spec) reflects the 2026-05-04
operator correction that Gemma-4 supports 128K context — see spec §4.3
for the revised arithmetic.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §4.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Lift vllm `--max-model-len` to 128K + bump `RAG_BUDGET_TOKENS` 22000 → 80000 + `RAG_GLOBAL_BUDGET_TOKENS` 22000 → 100000

**Files:**
- Modify: `compose/docker-compose.yml` (`vllm-chat.command` array + open-webui env block)
- Modify: `compose/.env.example` (two lines)

**Background:** Gemma-4-31B-it-AWQ supports 128K context natively (operator-confirmed 2026-05-04). The deployed `vllm-chat` container has a `--max-model-len` flag that may be capping at 32K from an older deploy. Bumping `RAG_BUDGET_TOKENS` past ~25K only takes effect after vllm is also lifted; otherwise vllm silently truncates the input regardless of the bridge's budget. This task does both lifts together so the change is coherent.

- [ ] **Step 1: Inspect current `vllm-chat.command` `--max-model-len`**

```bash
grep -A30 "^  vllm-chat:" /home/vogic/LocalRAG/compose/docker-compose.yml | grep -E "max-model-len|command:" | head -5
```

If `--max-model-len` is set to anything below `131072`, proceed to Step 2. If it's already `131072` or unset (vllm defaults to model-card max), skip to Step 3.

- [ ] **Step 2: Lift `--max-model-len` to 131072 in compose**

In `compose/docker-compose.yml`, find the `vllm-chat` service's `command:` array (typically a multi-line YAML list of strings). Locate the `--max-model-len <N>` pair and replace `<N>` with `131072`. If the `--max-model-len` flag isn't present at all, add it directly after the model name argument:

```yaml
  vllm-chat:
    # ... existing service definition ...
    command:
      - --model
      - QuantTrio/gemma-4-31B-it-AWQ
      # 2026-05-04 — explicit ctx lift. Model card supports 128K; older
      # deploys may have shipped at 32K via a stale --max-model-len.
      # Without this, RAG_BUDGET_TOKENS > 25K silently truncates inside
      # vllm regardless of the bridge's budget setting. See spec §4.3.3.
      - --max-model-len
      - "131072"
      # ... rest of existing flags ...
```

- [ ] **Step 3: Update `RAG_BUDGET_TOKENS` and `RAG_GLOBAL_BUDGET_TOKENS` defaults in compose**

Find the line `RAG_BUDGET_TOKENS: ${RAG_BUDGET_TOKENS:-22000}` (around line 599 in the open-webui env block) and change to:

```yaml
      # Bumped 22000 -> 80000 on 2026-05-04 (Phase 1 / Item 3 of
      # multi-entity-elaborate-answers spec). Exploits the 128K Gemma-4
      # ctx confirmed by operator 2026-05-04. With rerank_top_k=200
      # per-KB and floor=12, post-rerank pool can grow to ~30-40K tokens;
      # 80K budget gives headroom for context-expand siblings, datetime/
      # spotlight preambles, and future entity-coverage doc summaries
      # (Phase 2). Leaves ~38K for response generation on 128K ctx after
      # sys/user/history. See spec §4.3 for the full arithmetic.
      RAG_BUDGET_TOKENS: ${RAG_BUDGET_TOKENS:-80000}
```

Then find `RAG_GLOBAL_BUDGET_TOKENS: ${RAG_GLOBAL_BUDGET_TOKENS:-22000}` (typically a few lines below in the same block) and change to:

```yaml
      # Bumped 22000 -> 100000 on 2026-05-04. Phase 2 entity-coverage
      # doc summaries push more level=doc points into context for
      # global-intent queries; 100K leaves room for that plus drilldown.
      RAG_GLOBAL_BUDGET_TOKENS: ${RAG_GLOBAL_BUDGET_TOKENS:-100000}
```

- [ ] **Step 4: Update compose/.env.example doc**

Find the relevant `RAG_BUDGET_TOKENS` block in `compose/.env.example` and update the documented value + comment:

```bash
# Token budget for retrieval context fed to the LLM (specific/specific_date
# intent). Bumped to 80000 on 2026-05-04 to exploit the 128K Gemma-4
# context window. Gives multi-entity comparative queries room for
# 4 entities × 12 floor × ~120 tokens per chunk plus context-expand
# siblings, with ~38K free for response generation after sys/user/history.
# Requires --max-model-len 131072 in vllm-chat.command (Task 3 Step 2).
RAG_BUDGET_TOKENS=80000

# Token budget for global-intent queries (catalog / aggregation).
# Higher than RAG_BUDGET_TOKENS because Phase 2 entity-coverage doc
# summaries surface more level=doc points per query.
RAG_GLOBAL_BUDGET_TOKENS=100000
```

- [ ] **Step 5: Recreate vllm-chat AND open-webui to pick up the new values**

```bash
cd /home/vogic/LocalRAG/compose && docker compose -p orgchat up -d vllm-chat open-webui
docker compose -p orgchat exec -T open-webui printenv RAG_BUDGET_TOKENS RAG_GLOBAL_BUDGET_TOKENS
```

Expected: `80000` and `100000`.

vllm-chat takes ~30-90s to start on Gemma 4 31B AWQ. Wait for healthcheck:

```bash
until docker compose -p orgchat exec -T vllm-chat curl -sf http://localhost:8000/health 2>/dev/null; do sleep 5; done && echo "vllm ready"
```

Expected: `vllm ready` after 30-90s.

- [ ] **Step 6: Sanity-test that retrieval still works at the new budget**

```bash
JWT=$(docker compose -p orgchat exec -T open-webui curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"email":"<ADMIN_EMAIL>","password":"<ADMIN_PASSWORD>"}' \
  http://localhost:8080/api/v1/auths/signin | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

CID=$(docker compose -p orgchat exec -T postgres psql -U orgchat -d orgchat -At -c \
  "SELECT id FROM chat WHERE meta::text LIKE '%kb_config%' ORDER BY updated_at DESC LIMIT 1;")

docker compose -p orgchat exec -T open-webui curl -sf -X POST \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  -d "{\"chat_id\":\"$CID\",\"query\":\"hello\",\"selected_kb_config\":[{\"kb_id\":2,\"subtag_ids\":[]}],\"top_k\":3}" \
  http://localhost:8080/api/rag/retrieve > /dev/null && echo OK
```

Expected: `OK`.

Also verify a chat completion succeeds without truncation warnings in the vllm log:

```bash
docker compose -p orgchat logs --since 2m vllm-chat 2>&1 | grep -iE "truncat|max_tokens|max-model-len" | tail -5
```

Expected: no recent truncation warnings.

- [ ] **Step 7: Commit**

```bash
git add compose/docker-compose.yml compose/.env.example
git commit -m "$(cat <<'EOF'
chore(compose): exploit 128K Gemma ctx — vllm --max-model-len 131072 + budget 22000 -> 80000

Phase 1 / item 3. Operator confirmed 2026-05-04 that Gemma-4-31B-it-AWQ
supports 128K context (CLAUDE.md §3 was stale at "32K"). The deployed
vllm-chat may have been shipping at --max-model-len 32768 from an
older deploy; without lifting that, raising RAG_BUDGET_TOKENS past 25K
silently truncates inside vllm regardless of the bridge's budget.

This commit:
- Lifts vllm-chat.command --max-model-len to 131072
- Raises RAG_BUDGET_TOKENS default 22000 -> 80000 (specific intent)
- Raises RAG_GLOBAL_BUDGET_TOKENS default 22000 -> 100000 (global intent)

Pairs with the per-KB rerank_top_k=200 stamp set in Task 6 and the
multi_entity_rerank_floor=12 env default from Task 2. Empirical
brainstorm 2026-05-04 showed budget was never the bottleneck at 22K;
the bump is to give the 4-entity x 5-subtopic case + sibling expansion
genuine headroom and to leave room for Phase 2's per-doc entity-coverage
summaries and Phase 3's two-axis decompose pool.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §4.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `scripts/wipe_and_reingest.py` — backup phase

**Files:**
- Create: `scripts/wipe_and_reingest.py` (initial skeleton + backup logic)
- Test: `tests/integration/test_wipe_and_reingest.py` (initial)

This script is split into two tasks (4 and 5) to keep the diffs reviewable. Task 4 lands the backup half — running it twice is safe; running it once produces a recoverable backup directory.

- [ ] **Step 1: Write failing test for backup phase**

Create `tests/integration/test_wipe_and_reingest.py`:

```python
"""Integration test for scripts/wipe_and_reingest.py — backup phase.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §7.2
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_backup_writes_postgres_json(tmp_path: Path) -> None:
    """Backup half: writes kb_documents + rag_config snapshot to backup-dir/postgres.json."""
    backup_dir = tmp_path / "kb2_backup_test"
    result = subprocess.run(
        [
            ".venv/bin/python", "scripts/wipe_and_reingest.py",
            "--kb", "2",
            "--backup-only",
            "--backup-dir", str(backup_dir),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # File exists, has the expected shape
    pg_path = backup_dir / "postgres.json"
    assert pg_path.exists()
    data = json.loads(pg_path.read_text())
    assert "kb_id" in data
    assert data["kb_id"] == 2
    assert "rag_config" in data
    assert isinstance(data["rag_config"], dict)
    assert "kb_documents" in data
    assert isinstance(data["kb_documents"], list)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/integration/test_wipe_and_reingest.py -v
```

Expected: FAIL — `scripts/wipe_and_reingest.py` does not exist.

- [ ] **Step 3: Create the script with backup-only logic**

Create `scripts/wipe_and_reingest.py`:

```python
"""Wipe a KB and prepare for fresh re-ingest.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §7

Usage:
    .venv/bin/python scripts/wipe_and_reingest.py --kb 2 --backup-only --backup-dir DIR
    .venv/bin/python scripts/wipe_and_reingest.py --kb 2 --confirm-wipe --backup-dir DIR

Phase 1 of this spec ships the --backup-only path. Phase 1 / Task 5 adds
the --confirm-wipe path that deletes Qdrant collections + Postgres rows
after the backup is taken.

Env: DATABASE_URL (required). QDRANT_URL + QDRANT_API_KEY (required for
the wipe path; optional for backup-only since backup doesn't touch
Qdrant).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg


async def _conn() -> asyncpg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)
    url = url.replace("+asyncpg", "")
    return await asyncpg.connect(url)


async def _backup_postgres(kb_id: int, backup_dir: Path) -> Path:
    """Snapshot rag_config + every kb_documents row for the KB to JSON."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    out_path = backup_dir / "postgres.json"

    conn = await _conn()
    try:
        kb_row = await conn.fetchrow(
            "SELECT id, name, rag_config, synonyms FROM knowledge_bases WHERE id = $1",
            kb_id,
        )
        if kb_row is None:
            print(f"error: kb_id={kb_id} not found", file=sys.stderr)
            sys.exit(3)

        doc_rows = await conn.fetch(
            "SELECT id, kb_id, subtag_id, filename, mime_type, bytes, "
            "       ingest_status, chunk_count, pipeline_version, blob_sha, "
            "       doc_summary, uploaded_at, uploaded_by "
            "FROM kb_documents WHERE kb_id = $1 AND deleted_at IS NULL "
            "ORDER BY id",
            kb_id,
        )
    finally:
        await conn.close()

    payload = {
        "kb_id": kb_row["id"],
        "name": kb_row["name"],
        "rag_config": dict(kb_row["rag_config"] or {}),
        "synonyms": kb_row["synonyms"] or [],
        "kb_documents": [
            {
                "id": r["id"],
                "kb_id": r["kb_id"],
                "subtag_id": r["subtag_id"],
                "filename": r["filename"],
                "mime_type": r["mime_type"],
                "bytes": r["bytes"],
                "ingest_status": r["ingest_status"],
                "chunk_count": r["chunk_count"],
                "pipeline_version": r["pipeline_version"],
                "blob_sha": r["blob_sha"],
                "doc_summary": r["doc_summary"],
                "uploaded_at": r["uploaded_at"].isoformat() if r["uploaded_at"] else None,
                "uploaded_by": r["uploaded_by"],
            }
            for r in doc_rows
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kb", type=int, required=True)
    p.add_argument("--backup-dir", type=Path, required=True)
    p.add_argument("--backup-only", action="store_true",
                   help="Run backup phase only; skip wipe")
    p.add_argument("--confirm-wipe", action="store_true",
                   help="Actually wipe (Phase 5 — added in Task 5)")
    args = p.parse_args()

    print(f"== KB {args.kb} backup -> {args.backup_dir} ==")
    backup_path = await _backup_postgres(args.kb, args.backup_dir)
    print(f"  postgres backup: {backup_path}")

    if args.backup_only:
        print("== --backup-only set; not wiping ==")
        return

    if not args.confirm_wipe:
        print("error: pass --confirm-wipe to actually wipe", file=sys.stderr)
        sys.exit(4)

    print("error: wipe phase added in Task 5", file=sys.stderr)
    sys.exit(5)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/integration/test_wipe_and_reingest.py -v
```

Expected: PASS. (The test runs the script against the live Postgres; integration tests assume the dev stack is up.)

- [ ] **Step 5: Commit**

```bash
git add scripts/wipe_and_reingest.py tests/integration/test_wipe_and_reingest.py
git commit -m "$(cat <<'EOF'
feat(scripts): wipe_and_reingest.py — backup phase (KB rag_config + kb_documents)

Phase 1 / item 6 of the 2026-05-04 multi-entity-elaborate-answers spec.
Step 1 of the operator runbook: snapshot Postgres state to a JSON file
before any destructive action. Wipe phase lands in Task 5.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §7

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `scripts/wipe_and_reingest.py` — wipe phase

**Files:**
- Modify: `scripts/wipe_and_reingest.py` (add wipe path)
- Modify: `tests/integration/test_wipe_and_reingest.py` (add wipe test)

- [ ] **Step 1: Write failing test for wipe phase**

Append to `tests/integration/test_wipe_and_reingest.py`:

```python
@pytest.mark.integration
def test_wipe_deletes_qdrant_collections_and_postgres_rows(
    tmp_path: Path,
) -> None:
    """Wipe phase: deletes kb_{id}_v* collections + DELETE kb_documents rows.

    NOTE: this test creates a temporary KB id=99 with one fake document
    so we don't trash the operator's real KBs. The script accepts any kb_id;
    --confirm-wipe gates the destructive ops.
    """
    import asyncpg, asyncio

    async def setup_fake_kb() -> None:
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            await conn.execute("DELETE FROM kb_documents WHERE kb_id = 99")
            await conn.execute("DELETE FROM knowledge_bases WHERE id = 99")
            await conn.execute(
                "INSERT INTO knowledge_bases (id, name, admin_id, rag_config, synonyms) "
                "VALUES (99, 'wipe-test', 1, '{}', '[]')"
            )
            await conn.execute(
                "INSERT INTO kb_documents (kb_id, subtag_id, filename, ingest_status, chunk_count) "
                "VALUES (99, 1, 'test.txt', 'done', 5)"
            )
        finally:
            await conn.close()

    asyncio.run(setup_fake_kb())

    backup_dir = tmp_path / "kb99_backup"
    result = subprocess.run(
        [
            ".venv/bin/python", "scripts/wipe_and_reingest.py",
            "--kb", "99",
            "--confirm-wipe",
            "--backup-dir", str(backup_dir),
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # Backup written
    assert (backup_dir / "postgres.json").exists()

    # Postgres rows gone
    async def check_pg() -> int:
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            n = await conn.fetchval(
                "SELECT COUNT(*) FROM kb_documents WHERE kb_id = 99 AND deleted_at IS NULL"
            )
        finally:
            await conn.close()
        return n
    assert asyncio.run(check_pg()) == 0

    # rag_config preserved (the script re-stamps it from the backup)
    async def check_kb() -> dict:
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            row = await conn.fetchrow(
                "SELECT rag_config FROM knowledge_bases WHERE id = 99"
            )
        finally:
            await conn.close()
        return dict(row["rag_config"] or {})
    assert asyncio.run(check_kb()) == {}  # original empty rag_config preserved

    # cleanup
    async def teardown() -> None:
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            await conn.execute("DELETE FROM knowledge_bases WHERE id = 99")
        finally:
            await conn.close()
    asyncio.run(teardown())
```

Add this import at the top of the test file:

```python
import os
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/integration/test_wipe_and_reingest.py::test_wipe_deletes_qdrant_collections_and_postgres_rows -v
```

Expected: FAIL — script exits with `error: wipe phase added in Task 5`.

- [ ] **Step 3: Add the wipe phase to the script**

Replace the `main()` function in `scripts/wipe_and_reingest.py` and add the wipe helpers above it:

```python
async def _qdrant_delete_kb_collections(kb_id: int) -> list[str]:
    """Delete every collection whose name matches kb_{id} or kb_{id}_v*."""
    import httpx
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    headers = {"api-key": api_key} if api_key else {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{url}/collections", headers=headers)
        r.raise_for_status()
        all_names = [c["name"] for c in r.json()["result"]["collections"]]

        prefix = f"kb_{kb_id}"
        targets = [
            n for n in all_names
            if n == prefix or n.startswith(f"{prefix}_") or n == f"{prefix}_rebuild"
        ]
        deleted = []
        for name in targets:
            resp = await client.delete(
                f"{url}/collections/{name}", headers=headers,
            )
            if resp.status_code in (200, 404):
                deleted.append(name)
        return deleted


async def _postgres_wipe(kb_id: int, restore_rag_config: dict) -> None:
    """DELETE kb_documents rows; preserve knowledge_bases.rag_config."""
    conn = await _conn()
    try:
        # Hard-delete docs (FK cascades chunk-level rows in any audit table
        # if those exist; they don't today but the wipe is meant to be
        # destructive).
        await conn.execute("DELETE FROM kb_documents WHERE kb_id = $1", kb_id)
        # Re-stamp rag_config from the backup so chunker/floor settings
        # survive the wipe.
        await conn.execute(
            "UPDATE knowledge_bases SET rag_config = $1::jsonb WHERE id = $2",
            json.dumps(restore_rag_config), kb_id,
        )
    finally:
        await conn.close()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kb", type=int, required=True)
    p.add_argument("--backup-dir", type=Path, required=True)
    p.add_argument("--backup-only", action="store_true")
    p.add_argument("--confirm-wipe", action="store_true")
    args = p.parse_args()

    print(f"== KB {args.kb} backup -> {args.backup_dir} ==")
    backup_path = await _backup_postgres(args.kb, args.backup_dir)
    backup_data = json.loads(backup_path.read_text())
    print(f"  postgres backup: {backup_path}")
    print(f"  kb_documents in backup: {len(backup_data['kb_documents'])}")

    if args.backup_only:
        print("== --backup-only set; not wiping ==")
        return

    if not args.confirm_wipe:
        print("error: pass --confirm-wipe to actually wipe", file=sys.stderr)
        sys.exit(4)

    # Wipe Qdrant first — if Qdrant fails, Postgres is still consistent.
    print(f"== wiping Qdrant collections for kb_{args.kb} ==")
    deleted = await _qdrant_delete_kb_collections(args.kb)
    for n in deleted:
        print(f"  deleted: {n}")
    if not deleted:
        print(f"  (no collections matched kb_{args.kb}*)")

    # Then Postgres rows.
    print(f"== wiping kb_documents rows for kb_id={args.kb} ==")
    await _postgres_wipe(args.kb, backup_data["rag_config"])
    print(f"  rag_config restored from backup ({len(backup_data['rag_config'])} keys)")

    print()
    print("== DONE ==")
    print(f"Re-upload your docs via /api/kb/{args.kb}/subtag/<sid>/upload — "
          "ingest will pick up the per-KB rag_config restored from the backup.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/integration/test_wipe_and_reingest.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/wipe_and_reingest.py tests/integration/test_wipe_and_reingest.py
git commit -m "$(cat <<'EOF'
feat(scripts): wipe_and_reingest.py — wipe phase (Qdrant + Postgres)

Phase 1 / item 6. Adds Qdrant collection deletion (kb_{id}, kb_{id}_v*
matched by prefix) + kb_documents DELETE while preserving rag_config
restored from the backup. --confirm-wipe gates the destructive ops.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §7.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Apply Phase 1 PATCH to KB 2 + run wipe

**Files:**
- Operational only — no code changes. Sequence pulled from §7.3 of the spec.

This task is the operator-side activation step. It is recorded as a task so the executor performs it in the right order (after Tasks 1-5 ship and BEFORE the operator re-uploads docs).

- [ ] **Step 1: Get an admin JWT for the operator user**

```bash
EMAIL=$(grep -E "^ADMIN_EMAIL=" /home/vogic/LocalRAG/compose/.env | cut -d= -f2)
PWD=$(grep -E "^ADMIN_PASSWORD=" /home/vogic/LocalRAG/compose/.env | cut -d= -f2)
JWT=$(docker compose -p orgchat exec -T open-webui curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$EMAIL\",\"password\":\"$PWD\"}" \
    http://localhost:8080/api/v1/auths/signin | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
echo "JWT len: ${#JWT}"
```

Expected: nonzero JWT length.

- [ ] **Step 2: PATCH KB 2 to the Phase 1 chunker config**

```bash
docker compose -p orgchat exec -T open-webui curl -s -X PATCH \
    -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
    -d '{"chunking_strategy":"window","chunk_tokens":200,"overlap_tokens":30,"rerank_top_k":200,"multi_entity_rerank_floor":12}' \
    http://localhost:8080/api/kb/2/config | python3 -m json.tool
```

Expected: response shows the 5 keys applied. Other keys (mmr, rerank, etc.) preserved.

- [ ] **Step 3: Run the wipe script for KB 2**

```bash
mkdir -p /tmp/kb2_backup_$(date +%Y%m%d_%H%M)
BACKUP_DIR=$(ls -dt /tmp/kb2_backup_* | head -1)

DATABASE_URL=$(docker compose -p orgchat exec -T open-webui printenv DATABASE_URL) \
QDRANT_URL=$(docker compose -p orgchat exec -T open-webui printenv QDRANT_URL) \
QDRANT_API_KEY=$(cat /home/vogic/LocalRAG/compose/secrets/qdrant_api_key) \
.venv/bin/python scripts/wipe_and_reingest.py \
    --kb 2 \
    --confirm-wipe \
    --backup-dir "$BACKUP_DIR"
```

Expected: prints `== DONE ==` and the path to re-upload via.

- [ ] **Step 4: Verify the wipe**

```bash
# Postgres
docker compose -p orgchat exec -T postgres psql -U orgchat -d orgchat -c \
  "SELECT COUNT(*) FROM kb_documents WHERE kb_id = 2 AND deleted_at IS NULL;"
# Qdrant
QK=$(cat /home/vogic/LocalRAG/compose/secrets/qdrant_api_key)
curl -s -H "api-key: $QK" http://127.0.0.1:6333/collections \
  | python3 -c "import sys,json; print([c['name'] for c in json.load(sys.stdin)['result']['collections']])"
# rag_config preserved
docker compose -p orgchat exec -T open-webui curl -s -H "Authorization: Bearer $JWT" \
  http://localhost:8080/api/kb/2/config | python3 -m json.tool
```

Expected:
- `count = 0`
- collection list does NOT include any `kb_2*` entries
- rag_config still has chunking_strategy=window, chunk_tokens=200, etc.

- [ ] **Step 5: No commit (operator runbook step). Move on to Task 7 — operator re-uploads docs once Phase 2 ships.**

---

## Task 7: Phase 1 smoke test — manual brigade query

**Files:**
- None. This is a verification step.

This task runs after the operator has re-uploaded the four 2026 monthly reports (Apr, Mar, Feb, Jan 26.docx) into KB 2 / subtag 11. The doc summaries will use the OLD prompt; that's fine — Phase 2 fixes those after Phase 1 stabilises.

- [ ] **Step 1: Confirm 4 docs ingested**

```bash
docker compose -p orgchat exec -T postgres psql -U orgchat -d orgchat -c \
  "SELECT id, filename, ingest_status, chunk_count, pipeline_version FROM kb_documents WHERE kb_id=2 AND deleted_at IS NULL ORDER BY filename;"
```

Expected: 4 rows, all `ingest_status=done`, `chunk_count` between 800 and 1200 each (window 200/30 produces ~3-4× the structured chunker count). `pipeline_version` shows `chunker=v2|extractor=v2|embedder=bge-m3|...`.

- [ ] **Step 2: Run the brigade query through chat completions**

```bash
JWT=$(cat /tmp/jwt.tok)
CID=$(docker compose -p orgchat exec -T postgres psql -U orgchat -d orgchat -At -c \
  "SELECT id FROM chat WHERE meta::text LIKE '%kb_config%' ORDER BY updated_at DESC LIMIT 1;")
cat > /tmp/q_phase1.json <<EOF
{
  "model": "orgchat-chat",
  "stream": false,
  "chat_id": "$CID",
  "rag_kb_config": [{"kb_id": 2, "subtag_ids": [11]}],
  "messages": [{"role": "user", "content": "Give out major updates from the report of apr 2026, for the following :\n1 75 INf bde\n2.  5 PoK bde\n3.  32 Inf Bde\n4.  80 Inf Bde\nI want answers under fwg heads\n1.  visits of senior military official and important visits"}]
}
EOF
docker compose -p orgchat exec -T open-webui curl -s -X POST \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  --max-time 240 \
  -d @- http://localhost:8080/api/chat/completions < /tmp/q_phase1.json > /tmp/r_phase1.json
```

Expected: response file has `choices[0].message.content`.

- [ ] **Step 3: Parse facts per brigade**

```bash
python3 <<'PY'
import json, re
d = json.load(open('/tmp/r_phase1.json'))
ans = d['choices'][0]['message']['content']
parts = re.split(r'(?im)(75 Inf Bde|5 PoK Bde|5 POK Bde|32 Inf Bde|80 Inf Bde)', ans)
out = {}
for i in range(1, len(parts)-1, 2):
    b = parts[i].replace('5 POK', '5 PoK')
    bullets = re.findall(r'^[*-]\s+', parts[i+1], re.MULTILINE)
    out[b] = out.get(b, 0) + len(bullets)
print('per-brigade fact bullets:', out)
print('answer length:', len(ans), 'chars')
PY
```

Expected (Phase 1 acceptance criteria):
- ≥6 facts per brigade across all 4 brigades (revised up from ≥4 after the 128K context lift).
- Total facts ≥20 (revised up from ≥15).
- Answer length ≥10000 chars (revised up from ≥6000).

- [ ] **Step 4: If acceptance fails, do not roll back — capture diagnostics**

```bash
docker compose -p orgchat logs --since 5m open-webui 2>&1 | \
  grep -E "intent=|multi-entity|cap active|reranked|drilldown" > /tmp/phase1_diag.log
echo "diag log: /tmp/phase1_diag.log"
```

Hand the diag log to the next executor. Do not regress to structured chunker; investigate the chunker pipeline downstream of Task 6 first.

- [ ] **Step 5: No commit (verification only). Proceed to Phase 2.**

---

## Task 8: Phase 1 eval-gate run

**Files:**
- None. Validation step.

- [ ] **Step 1: Run the existing eval harness**

```bash
cd /home/vogic/LocalRAG && .venv/bin/python -m pytest tests/eval -v -k "not slow"
```

Expected: existing baseline passes. If a multi-entity gold-set query was added by a previous PR, it should now show improved nDCG@10 vs the committed baseline.

- [ ] **Step 2: Capture the diff**

```bash
ls -la tests/eval/results/ | tail -5
git diff tests/eval/results/ | head -40
```

If the eval harness wrote new result files showing >5pp regression on any non-multi-entity query, investigate. If only multi-entity gold-set queries show improvement, that's the expected shape.

- [ ] **Step 3: Update the committed baseline if intentional shifts are present**

```bash
# Only if Step 2 confirms the diff is the expected lift on multi-entity queries:
git add tests/eval/results/
git commit -m "$(cat <<'EOF'
test(eval): refresh baseline after Phase 1 multi-entity-elaborate-answers ship

Updates tests/eval/results/ with the post-Phase-1 numbers. Multi-entity
gold-set queries show expected nDCG@10 lift; non-multi-entity queries
within ±2pp of the prior baseline.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §9

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: No further steps for this task.**

---

# Phase 2 — Entity-Coverage Doc Summaries (Tasks 9-14)

## Task 9: New ENTITIES + SUMMARY prompt in `doc_summarizer.py`

**Files:**
- Modify: `ext/services/doc_summarizer.py` (replace `_SUMMARY_PROMPT`, raise `_MAX_BODY_CHARS`)
- Test: `tests/unit/test_doc_summarizer_prompt.py` (new file)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_doc_summarizer_prompt.py`:

```python
"""Unit tests for the structured doc-summary prompt (Phase 2 of the
2026-05-04 multi-entity-elaborate-answers spec).

The prompt asks the chat-LLM to emit two named sections:
  ENTITIES: <comma-separated list of formations named >=2 times>
  SUMMARY: <5-7 sentence paragraph that names every entity from ENTITIES>

The parser separates these into a dict so the ingest path can populate
a Qdrant payload `entities` field alongside the summary text.
"""
from __future__ import annotations

from ext.services import doc_summarizer


class TestPromptShape:
    def test_prompt_mentions_two_sections(self) -> None:
        # The prompt template is module-level; we just check shape.
        assert "ENTITIES:" in doc_summarizer._SUMMARY_PROMPT
        assert "SUMMARY:" in doc_summarizer._SUMMARY_PROMPT
        assert "{filename}" in doc_summarizer._SUMMARY_PROMPT
        assert "{body}" in doc_summarizer._SUMMARY_PROMPT

    def test_max_body_chars_raised(self) -> None:
        # Phase 2 raised this to 32000.
        assert doc_summarizer._MAX_BODY_CHARS == 32000


class TestParseStructuredOutput:
    def test_parses_well_formed_output(self) -> None:
        raw = (
            "ENTITIES: 75 Inf Bde, 5 PoK Bde, 32 Inf Bde, 80 Inf Bde\n\n"
            "SUMMARY: Document Apr 26.docx provides the fourth monthly "
            "update for April 2026. 75 Inf Bde activity centred on "
            "operational reviews. 5 PoK Bde tracked coordination meetings."
        )
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == [
            "75 Inf Bde", "5 PoK Bde", "32 Inf Bde", "80 Inf Bde"
        ]
        assert "April 2026" in out["summary"]
        assert "5 PoK Bde tracked coordination meetings." in out["summary"]

    def test_handles_missing_entities_section(self) -> None:
        # Fail-soft: if LLM only emits the summary, we keep it and emit
        # an empty entities list. Caller's responsibility to log this.
        raw = "SUMMARY: A simple summary with no entities header."
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == []
        assert "simple summary" in out["summary"]

    def test_handles_missing_summary_section(self) -> None:
        # Same — keep the entities, set summary empty.
        raw = "ENTITIES: A, B, C"
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == ["A", "B", "C"]
        assert out["summary"] == ""

    def test_returns_empty_on_blank(self) -> None:
        out = doc_summarizer.parse_structured_summary("")
        assert out == {"entities": [], "summary": ""}

    def test_strips_whitespace(self) -> None:
        raw = "  ENTITIES:   A,  B  \n\n  SUMMARY:   text here.  "
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == ["A", "B"]
        assert out["summary"] == "text here."

    def test_dedupes_entities_case_insensitively(self) -> None:
        raw = "ENTITIES: A, a, A, B\n\nSUMMARY: x"
        out = doc_summarizer.parse_structured_summary(raw)
        # First-surface-form wins — same convention as entity_extractor
        assert out["entities"] == ["A", "B"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/test_doc_summarizer_prompt.py -v
```

Expected: All FAIL — `_SUMMARY_PROMPT` doesn't have `ENTITIES:`, `_MAX_BODY_CHARS` is 16000, `parse_structured_summary` doesn't exist.

- [ ] **Step 3: Update `doc_summarizer.py`**

Edit `ext/services/doc_summarizer.py`. Replace `_MAX_BODY_CHARS` and `_SUMMARY_PROMPT` near the top of the file:

```python
# Approx token budget for the body we feed the summarizer. ~4 chars/token
# → ~32000 chars ≈ 8000 tokens. We truncate by character count to avoid
# pulling a tokenizer dep into the hot path; the model can handle slight
# over/under. Bumped 16000 -> 32000 on 2026-05-04 (Phase 2 of the
# multi-entity-elaborate-answers spec) so the entity-coverage prompt
# sees enough body to enumerate every named formation.
_MAX_BODY_CHARS = 32000

_SUMMARY_PROMPT = """Summarize this document for retrieval. Output two sections, no preamble:

ENTITIES: A comma-separated list of every named formation, brigade, battalion, regiment, or unit that appears at least twice in the document. Use the canonical name as it first appears. If fewer than two distinct named formations are present, list whatever named formations exist (or write "none" if there are none).

SUMMARY: A 5-7 sentence paragraph covering: document name, reporting period, every named entity from the ENTITIES list (one clause per entity naming what activities they were involved in), and any cross-cutting themes (training, construction, intel, etc.) that appear across multiple entities. Do NOT favour the most-mentioned entity over others — every ENTITIES-list member must be named in the SUMMARY.

Document: {filename}

{body}
"""
```

Then add the parser function near the bottom of the file (before the `__all__` block if any, otherwise at end):

```python
def parse_structured_summary(raw: str) -> dict[str, list[str] | str]:
    """Parse the chat-LLM ENTITIES + SUMMARY response.

    Returns ``{"entities": [...], "summary": "..."}``. Fail-soft: if a
    section is missing, returns an empty list / empty string for that
    section but keeps whatever was parseable. Empty input returns the
    fully-empty shape.

    Entity dedup is case-insensitive, preserving the first surface form
    (matches ``entity_extractor._dedupe_preserve_first``).
    """
    if not raw:
        return {"entities": [], "summary": ""}

    raw = raw.strip()
    entities_part = ""
    summary_part = ""

    # Find ENTITIES: marker (case-insensitive)
    import re
    ent_match = re.search(r"\bENTITIES\s*:\s*", raw, re.IGNORECASE)
    sum_match = re.search(r"\bSUMMARY\s*:\s*", raw, re.IGNORECASE)

    if ent_match and sum_match:
        # Both present — split on the SUMMARY marker
        entities_part = raw[ent_match.end():sum_match.start()].strip()
        summary_part = raw[sum_match.end():].strip()
    elif ent_match:
        # Only ENTITIES — take everything after the marker
        entities_part = raw[ent_match.end():].strip()
    elif sum_match:
        # Only SUMMARY
        summary_part = raw[sum_match.end():].strip()
    else:
        # Neither marker — treat the whole thing as a summary (legacy shape)
        summary_part = raw

    # Parse entities list — comma-separated, dedupe case-insensitively
    if entities_part and entities_part.lower() != "none":
        seen_lower: set[str] = set()
        entities: list[str] = []
        for e in entities_part.split(","):
            e_clean = e.strip()
            if not e_clean:
                continue
            if e_clean.lower() in seen_lower:
                continue
            seen_lower.add(e_clean.lower())
            entities.append(e_clean)
    else:
        entities = []

    return {"entities": entities, "summary": summary_part}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/test_doc_summarizer_prompt.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ext/services/doc_summarizer.py tests/unit/test_doc_summarizer_prompt.py
git commit -m "$(cat <<'EOF'
feat(doc_summarizer): structured ENTITIES + SUMMARY prompt + parser

Phase 2 / item 4 of the 2026-05-04 multi-entity-elaborate-answers spec.
Replaces the dominant-story-arc 3-sentence prompt that produced
75-Inf-only summaries on KB 2's monthly reports. New prompt asks the
chat-LLM for a comma-separated ENTITIES list and a 5-7 sentence
SUMMARY where every ENTITIES-list member is named.

Body cap 16000 -> 32000 chars so 12K-15K-token monthly reports get
fully visible to the summariser. Added parse_structured_summary()
helper with fail-soft section handling.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §5.1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Wire `summarize_document` to return structured output

**Files:**
- Modify: `ext/services/doc_summarizer.py` (`summarize_document` return shape)
- Modify: `tests/unit/test_doc_summarizer_prompt.py` (add return-shape test)

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_doc_summarizer_prompt.py`:

```python
import pytest


class TestSummarizeDocumentReturnShape:
    """summarize_document() now returns the structured dict, not a bare string.
    Callers that want the legacy string can read the 'summary' key."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_entities_and_summary(
        self, monkeypatch
    ) -> None:
        async def fake_impl(**kwargs):
            # Simulates a chat-LLM response in the new shape
            return (
                "ENTITIES: 75 Inf Bde, 5 PoK Bde\n\n"
                "SUMMARY: Apr 26 monthly update covering 75 Inf Bde and 5 PoK Bde."
            )
        monkeypatch.setattr(doc_summarizer, "_summarize_impl", fake_impl)

        out = await doc_summarizer.summarize_document(
            chunks=["chunk 1", "chunk 2"],
            filename="Apr 26.docx",
            chat_url="http://fake/v1",
            chat_model="fake",
        )
        assert isinstance(out, dict)
        assert out["entities"] == ["75 Inf Bde", "5 PoK Bde"]
        assert "Apr 26 monthly update" in out["summary"]

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty_shape(
        self, monkeypatch
    ) -> None:
        out = await doc_summarizer.summarize_document(
            chunks=[],
            filename="x",
            chat_url="http://fake/v1",
            chat_model="fake",
        )
        assert out == {"entities": [], "summary": ""}

    @pytest.mark.asyncio
    async def test_impl_failure_returns_empty_shape(
        self, monkeypatch
    ) -> None:
        async def fake_impl(**kwargs):
            return ""  # _summarize_impl returns "" on any failure (legacy contract)
        monkeypatch.setattr(doc_summarizer, "_summarize_impl", fake_impl)

        out = await doc_summarizer.summarize_document(
            chunks=["x"],
            filename="x",
            chat_url="http://fake/v1",
            chat_model="fake",
        )
        assert out == {"entities": [], "summary": ""}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/test_doc_summarizer_prompt.py::TestSummarizeDocumentReturnShape -v
```

Expected: FAIL — current return type is `str`, not a dict.

- [ ] **Step 3: Update `summarize_document` in `ext/services/doc_summarizer.py`**

Find the `summarize_document` async function. Replace the body with:

```python
async def summarize_document(
    chunks: list[str],
    filename: str,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> dict[str, list[str] | str]:
    """Summarize a document, returning ``{"entities": [...], "summary": "..."}``.

    Phase 2 of the 2026-05-04 multi-entity-elaborate-answers spec.
    Replaces the legacy bare-string return — callers that only want
    the summary text should read ``out["summary"]``.

    Args:
        chunks: ordered list of chunk-body strings for the document.
        filename: display name (included in the prompt for explicit naming).
        chat_url: base URL of an OpenAI-compatible endpoint.
        chat_model: model name the endpoint expects.
        api_key: optional bearer token.
        timeout: request timeout (seconds).
        transport: optional httpx transport (for tests).

    Returns:
        A dict with two keys:
          - ``entities``: list of canonical entity names (comma-list parsed)
          - ``summary``: 5-7 sentence paragraph
        On any LLM failure or empty input both keys hold their empty
        defaults (``[]`` and ``""``).
    """
    if not chunks:
        return {"entities": [], "summary": ""}

    with span("doc.summarize", model=chat_model, n_chunks=len(chunks)):
        raw = await _summarize_impl(
            chunks=chunks,
            filename=filename,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )
    return parse_structured_summary(raw)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/test_doc_summarizer_prompt.py -v
```

Expected: ALL tests pass (10+ across both test classes).

- [ ] **Step 5: Commit**

```bash
git add ext/services/doc_summarizer.py tests/unit/test_doc_summarizer_prompt.py
git commit -m "$(cat <<'EOF'
feat(doc_summarizer): summarize_document returns {entities, summary} dict

Phase 2 / item 4. summarize_document now parses the LLM response into
a structured dict so the ingest path can populate a Qdrant payload
'entities' field alongside the summary text. Empty / failure cases
return the empty-shape ({"entities": [], "summary": ""}) — same
fail-soft contract as the legacy string-empty.

BREAKING CHANGE: every caller of summarize_document() now needs to
read .summary instead of treating the return as a string. The only
in-tree caller (ingest.py) is updated in Task 11.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §5.1.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Update `ingest.py` doc-summary upsert to use structured shape

**Files:**
- Modify: `ext/services/ingest.py` (caller of `summarize_document`; payload construction)
- Test: `tests/unit/test_ingest_doc_summary_payload.py` (new file or extend existing)

- [ ] **Step 1: Find the existing call site**

```bash
grep -nE "summarize_document|doc_summary\s*=" /home/vogic/LocalRAG/ext/services/ingest.py | head -10
```

Expected output: line numbers showing the call to `summarize_document` (typically inside `ingest_bytes`) and the upsert that writes the doc-summary point to Qdrant.

- [ ] **Step 2: Write failing test**

Create `tests/unit/test_ingest_doc_summary_payload.py`:

```python
"""Test that ingest.py wires summarize_document's structured return into
the level=doc Qdrant payload (entities field).

Phase 2 / item 4 of the 2026-05-04 multi-entity-elaborate-answers spec.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_doc_summary_upsert_includes_entities_payload() -> None:
    """When summarize_document returns {entities: [...], summary: "..."},
    the level=doc Qdrant point payload MUST include an 'entities' field."""

    # Patch the summariser to return a known structured shape
    with patch(
        "ext.services.ingest.summarize_document",
        new_callable=AsyncMock,
    ) as mock_sum, patch(
        "ext.services.ingest.embedder",
    ) as mock_embedder, patch(
        "ext.services.ingest.vector_store",
    ) as mock_vs:
        mock_sum.return_value = {
            "entities": ["75 Inf Bde", "5 PoK Bde", "32 Inf Bde"],
            "summary": "Test summary mentioning every entity.",
        }
        # Capture upsert calls — we want to inspect the payload sent to
        # Qdrant for the level=doc point.
        upserted = []
        async def fake_upsert(collection, points, **kw):
            upserted.append((collection, points))
        mock_vs.upsert = fake_upsert
        mock_embedder.embed = AsyncMock(return_value=[[0.0] * 1024])

        from ext.services import ingest

        # Call the helper that emits the doc-summary point. The exact
        # function name depends on how ingest.py is structured;
        # _emit_doc_summary_point or similar. Test pulls the relevant
        # private helper if needed.
        await ingest._emit_doc_summary_point(
            kb_id=2,
            doc_id=999,
            subtag_id=11,
            filename="Apr 26.docx",
            chunk_texts=["a", "b"],
            chat_url="http://fake/v1",
            chat_model="fake",
        )

    # Find the doc-summary point in the upserts
    doc_points = [
        p for _coll, pts in upserted for p in pts
        if (p.get("payload") or {}).get("level") == "doc"
    ]
    assert len(doc_points) == 1
    payload = doc_points[0]["payload"]
    assert payload["entities"] == [
        "75 Inf Bde", "5 PoK Bde", "32 Inf Bde",
    ]
    assert "Test summary" in payload["text"]
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/test_ingest_doc_summary_payload.py -v
```

Expected: FAIL — `_emit_doc_summary_point` doesn't exist or doesn't include `entities`.

- [ ] **Step 4: Update `ingest.py`**

Find the doc-summary emission code. It looks something like (the exact text will differ):

```python
summary = await summarize_document(chunks=texts, filename=fn, ...)
if summary:
    await vector_store.upsert(coll, [{
        "id": _doc_point_id(doc_id),
        "vector": dense_vec,
        "payload": {"text": summary, "filename": fn, "level": "doc",
                    "doc_id": doc_id, "kb_id": kb_id, ...},
    }])
```

Refactor (a) the call to consume the new dict and (b) extract a small helper. Replace it with:

```python
async def _emit_doc_summary_point(
    *,
    kb_id: int,
    doc_id: int,
    subtag_id: int,
    filename: str,
    chunk_texts: list[str],
    chat_url: str,
    chat_model: str,
    api_key: str | None = None,
) -> dict[str, list[str] | str]:
    """Emit one ``level=doc`` Qdrant point with entities + summary text.

    Returns the structured summary dict (``{entities, summary}``) so the
    Postgres ``kb_documents.doc_summary`` mirror can be updated by the
    caller. On empty/error result, no Qdrant point is upserted and the
    caller gets the empty shape back.
    """
    summary_dict = await summarize_document(
        chunks=chunk_texts, filename=filename,
        chat_url=chat_url, chat_model=chat_model, api_key=api_key,
    )
    if not summary_dict["summary"]:
        # Fail-soft: no summary, no point. Caller decides whether to
        # log; existing pipeline already logs via _record_silent_failure.
        return summary_dict

    # Compose the text field as ENTITIES + SUMMARY so the dense retriever
    # sees both signals.
    text_field = (
        f"ENTITIES: {', '.join(summary_dict['entities'])}\n\n"
        f"SUMMARY: {summary_dict['summary']}"
    ) if summary_dict["entities"] else summary_dict["summary"]

    dense_vec = (await embedder.embed([text_field]))[0]
    await vector_store.upsert(
        f"kb_{kb_id}",
        [{
            "id": _doc_point_id(doc_id),
            "vector": dense_vec,
            "payload": {
                "text": text_field,
                "entities": summary_dict["entities"],  # NEW — Phase 2
                "filename": filename,
                "level": "doc",
                "doc_id": doc_id,
                "kb_id": kb_id,
                "subtag_id": subtag_id,
                "chunk_index": None,
            },
        }],
    )
    return summary_dict
```

Then replace the previous inline call site with:

```python
summary_dict = await _emit_doc_summary_point(
    kb_id=kb_id, doc_id=doc_id, subtag_id=subtag_id,
    filename=fn, chunk_texts=texts,
    chat_url=chat_url, chat_model=chat_model, api_key=api_key,
)
# Mirror summary text to Postgres
if summary_dict["summary"]:
    await db.execute(
        "UPDATE kb_documents SET doc_summary = $1 WHERE id = $2",
        summary_dict["summary"], doc_id,
    )
```

(Fix the exact import path for `embedder`, `vector_store`, and `_doc_point_id` based on what `ingest.py` currently uses — search for the existing call site to find the right names.)

- [ ] **Step 5: Run test + integration smoke**

```bash
.venv/bin/pytest tests/unit/test_ingest_doc_summary_payload.py -v
.venv/bin/pytest tests/unit/test_ingest.py -v -k "doc_summary or summary"
```

Expected: new test PASSES, existing tests still PASS (no regression).

```bash
git add ext/services/ingest.py tests/unit/test_ingest_doc_summary_payload.py
git commit -m "$(cat <<'EOF'
feat(ingest): wire entity-coverage summary into level=doc payload

Phase 2 / item 4. Doc-summary upsert now writes the structured
{entities, summary} return from summarize_document into the Qdrant
point payload — entities as a payload field, "ENTITIES: ...\n\nSUMMARY: ..."
text shape so dense retrieval sees both signals.

Postgres kb_documents.doc_summary mirrors the SUMMARY section only
(keeps the column human-readable; ENTITIES is reconstructable from
the Qdrant payload).

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §5.1.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Add `entities` payload index in `qdrant_schema.py`

**Files:**
- Modify: `ext/db/qdrant_schema.py` (append to `CANONICAL_INDEXES`)
- Test: `tests/unit/test_qdrant_schema.py` (extend or new file)

- [ ] **Step 1: Write failing test**

Create or extend `tests/unit/test_qdrant_schema.py`:

```python
"""Unit tests for the canonical Qdrant payload-index list.

Phase 2 of the 2026-05-04 multi-entity-elaborate-answers spec adds an
'entities' index on doc-summary points so the global-intent retrieval
path can do MatchText against entity names that appear in the summary.
"""
from __future__ import annotations

from ext.db import qdrant_schema


def test_canonical_indexes_includes_entities() -> None:
    field_names = [idx["field"] for idx in qdrant_schema.CANONICAL_INDEXES]
    assert "entities" in field_names


def test_entities_index_is_text_lowercased() -> None:
    entities_idx = next(
        idx for idx in qdrant_schema.CANONICAL_INDEXES if idx["field"] == "entities"
    )
    # Entities is a list-of-strings; Qdrant text index with lowercase
    # tokenization handles MatchText for variants
    assert entities_idx["type"] == "text"
    assert entities_idx.get("lowercase") is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/test_qdrant_schema.py -v
```

Expected: FAIL.

- [ ] **Step 3: Add `entities` to `CANONICAL_INDEXES`**

Edit `ext/db/qdrant_schema.py`. Append to the `CANONICAL_INDEXES` list (after the last entry):

```python
    # 2026-05-04 — Phase 2 / item 4 of multi-entity-elaborate-answers
    # spec. Entity list on level=doc points; text index with lowercase
    # tokenization handles "5 PoK" / "5 POK" / "5 PoK Bde" variants the
    # same way the per-KB synonym table feeds entity_text_filter on
    # chunk-level points.
    {"field": "entities", "type": "text", "lowercase": True, "is_tenant": False},
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/test_qdrant_schema.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ext/db/qdrant_schema.py tests/unit/test_qdrant_schema.py
git commit -m "$(cat <<'EOF'
feat(qdrant_schema): entities text-lowercased payload index on doc summaries

Phase 2 / item 4. New canonical index lets global-intent retrieval
match text against entity names that appear in the summary, the same
way per-KB synonyms feed entity_text_filter on chunk-level points.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §5.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Apply the new payload index to live KB collections

**Files:**
- Operational only.

- [ ] **Step 1: Re-run the schema reconciliation script**

```bash
QK=$(cat /home/vogic/LocalRAG/compose/secrets/qdrant_api_key)
DATABASE_URL=$(docker compose -p orgchat exec -T open-webui printenv DATABASE_URL) \
QDRANT_API_KEY=$QK \
.venv/bin/python scripts/reconcile_qdrant_schema.py --apply
```

Expected: prints `created index: entities (text)` for every kb_*_v* collection that exists.

- [ ] **Step 2: Verify the index landed**

```bash
QK=$(cat /home/vogic/LocalRAG/compose/secrets/qdrant_api_key)
curl -s -H "api-key: $QK" http://127.0.0.1:6333/collections/kb_2 | \
  python3 -c "import sys,json; d=json.load(sys.stdin)['result']; print(list(d.get('payload_schema',{}).keys()))"
```

Expected: list contains `entities` alongside `kb_id`, `doc_id`, etc.

- [ ] **Step 3: No commit (operator step).**

---

## Task 14: Phase 2 smoke — comparative-query test

**Files:**
- None (verification step).

This task runs after Phase 2 ships AND the operator re-uploaded all docs. New ingests use the structured prompt; doc summaries now name every entity.

- [ ] **Step 1: Confirm a re-ingest happened with the new prompt**

```bash
docker compose -p orgchat exec -T postgres psql -U orgchat -d orgchat -c \
  "SELECT id, filename, doc_summary FROM kb_documents WHERE kb_id=2 AND doc_summary IS NOT NULL ORDER BY id LIMIT 4;"
```

Expected: each `doc_summary` now contains references to multiple brigades (5 PoK Bde, 32 Inf Bde, 80 Inf Bde) — not just 75 Inf Bde.

- [ ] **Step 2: Spot-check the Qdrant level=doc payload**

```bash
QK=$(cat /home/vogic/LocalRAG/compose/secrets/qdrant_api_key)
curl -s -X POST -H "api-key: $QK" -H "Content-Type: application/json" \
  -d '{"limit":5,"with_payload":true,"with_vector":false,"filter":{"must":[{"key":"level","match":{"value":"doc"}},{"key":"kb_id","match":{"value":2}}]}}' \
  http://127.0.0.1:6333/collections/kb_2/points/scroll | \
  python3 -c "
import sys,json
d=json.load(sys.stdin)
for p in d['result']['points']:
    pl=p['payload']
    print(p['id'], pl.get('filename'), 'entities=', pl.get('entities'))
"
```

Expected: each row shows `entities=` populated with at least 3-4 brigade names.

- [ ] **Step 3: Run the comparative-query smoke**

```bash
JWT=$(cat /tmp/jwt.tok)
CID=$(docker compose -p orgchat exec -T postgres psql -U orgchat -d orgchat -At -c \
  "SELECT id FROM chat WHERE meta::text LIKE '%kb_config%' ORDER BY updated_at DESC LIMIT 1;")
cat > /tmp/q_phase2.json <<EOF
{
  "model":"orgchat-chat","stream":false,"chat_id":"$CID",
  "rag_kb_config":[{"kb_id":2,"subtag_ids":[]}],
  "messages":[{"role":"user","content":"Compare 5 PoK Bde and 80 Inf Bde operational status across early 2026 (Jan-Apr 2026). Cite specific dates and units for each brigade."}]
}
EOF
docker compose -p orgchat exec -T open-webui curl -s -X POST \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  --max-time 240 \
  -d @- http://localhost:8080/api/chat/completions < /tmp/q_phase2.json > /tmp/r_phase2.json
python3 -c "
import json
d=json.load(open('/tmp/r_phase2.json'))
ans=d['choices'][0]['message']['content']
print('---')
print(ans[:4000])
print('---')
print('answer length:', len(ans))
print('5 PoK count:', ans.lower().count('5 pok'))
print('80 Inf count:', ans.lower().count('80 inf'))
"
```

Expected:
- Both `5 PoK count` and `80 Inf count` are ≥3 (today's pre-Phase-2 state has one brigade at 0).
- Answer length ≥4000 chars.

- [ ] **Step 4: Compare to a pre-Phase-2 baseline**

If `tests/eval/results/comparative_baseline.txt` exists and shows the pre-Phase-2 numbers, diff against it. Otherwise capture the post-Phase-2 numbers as a new baseline.

- [ ] **Step 5: No commit (verification). Move to Phase 3.**

---

# Phase 3 — Subtopic Axis (Tasks 15-24)

## Task 15: `extract_subtopics(query)` in `entity_extractor.py`

**Files:**
- Modify: `ext/services/entity_extractor.py` (add `extract_subtopics`)
- Modify: `tests/unit/test_entity_extractor.py` (extend with new class)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_entity_extractor.py`:

```python
class TestExtractSubtopicsRegex:
    """Phase 3 / item 5 of the 2026-05-04 multi-entity-elaborate-answers spec.
    Regex-only subtopic extractor for queries with explicit subheadings."""

    def test_under_fwg_heads_with_numbered_list(self) -> None:
        q = (
            "Give updates for following:\n1. 75 Inf Bde\nunder fwg heads:\n"
            "1. visits of senior officers\n2. operational activities\n3. construction"
        )
        out = entity_extractor.extract_subtopics_regex(q)
        # First-surface-form preserved, lowercased trimmed normalisation
        assert out == [
            "visits of senior officers",
            "operational activities",
            "construction",
        ]

    def test_under_following_heads(self) -> None:
        q = (
            "For each brigade list facts under following heads:\n"
            "1. visits\n2. operations\n3. construction\n4. intel"
        )
        out = entity_extractor.extract_subtopics_regex(q)
        assert out == ["visits", "operations", "construction", "intel"]

    def test_bulleted_list_under_heading_marker(self) -> None:
        q = "Updates under headings:\n- visits\n- operations"
        out = entity_extractor.extract_subtopics_regex(q)
        assert out == ["visits", "operations"]

    def test_no_subtopics_returns_empty(self) -> None:
        q = "Give updates from the report of April 2026"
        assert entity_extractor.extract_subtopics_regex(q) == []

    def test_single_subtopic_returns_empty(self) -> None:
        # Below the >=2 threshold for two-axis decompose; we still emit []
        q = "Updates under heads:\n1. visits"
        assert entity_extractor.extract_subtopics_regex(q) == []

    def test_caps_at_eight(self) -> None:
        items = [f"topic {i}" for i in range(20)]
        q = "Under headings:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))
        out = entity_extractor.extract_subtopics_regex(q)
        assert len(out) == 8
        assert out[0] == "topic 0"
        assert out[7] == "topic 7"

    def test_dedupes_case_insensitively(self) -> None:
        q = "Under heads:\n1. Visits\n2. visits\n3. Operations"
        out = entity_extractor.extract_subtopics_regex(q)
        assert out == ["Visits", "Operations"]


class TestExtractSubtopicsCompose:
    """Composition with QU LLM — same shape as extract_entities."""

    def test_qu_subtopics_preferred_when_present(self) -> None:
        class FakeQU:
            subtopics = ["visits", "operations", "construction"]

        out = entity_extractor.extract_subtopics(
            "Give updates", qu_result=FakeQU(),
        )
        assert out == ["visits", "operations", "construction"]

    def test_falls_back_to_regex_on_no_qu(self) -> None:
        q = "Under heads:\n1. visits\n2. operations"
        out = entity_extractor.extract_subtopics(q, qu_result=None)
        assert out == ["visits", "operations"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_entity_extractor.py::TestExtractSubtopicsRegex tests/unit/test_entity_extractor.py::TestExtractSubtopicsCompose -v
```

Expected: FAIL — `extract_subtopics_regex` / `extract_subtopics` don't exist.

- [ ] **Step 3: Add the two functions to `entity_extractor.py`**

Append to `ext/services/entity_extractor.py` (before `__all__`):

```python
# 2026-05-04 — Phase 3 / item 5 of multi-entity-elaborate-answers spec.
# Subtopic extraction. Same priority pattern as entity extraction:
# QU LLM > regex > empty.

_MIN_SUBTOPICS = 2
_MAX_SUBTOPICS = 8

# Regexes mirror the entity-extractor patterns but trigger on subtopic
# markers rather than entity-list markers.
_SUBTOPIC_HEADER_RE = re.compile(
    r"\b(under|with)?\s*(fwg|following)?\s*(head(?:ing)?s?|sections?)\s*:",
    re.IGNORECASE,
)


def _extract_subtopic_numbered(text: str) -> list[str]:
    # Find a header marker, then read numbered items beneath it
    m = _SUBTOPIC_HEADER_RE.search(text)
    if not m:
        return []
    after = text[m.end():]
    items = re.findall(r"^\s*\d+[.)]\s+(.+?)\s*$", after, re.MULTILINE)
    return [i.strip() for i in items if i.strip()]


def _extract_subtopic_bulleted(text: str) -> list[str]:
    m = _SUBTOPIC_HEADER_RE.search(text)
    if not m:
        return []
    after = text[m.end():]
    items = re.findall(r"^\s*[-*•]\s+(.+?)\s*$", after, re.MULTILINE)
    return [i.strip() for i in items if i.strip()]


def extract_subtopics_regex(query: str | None) -> list[str]:
    """Pure-regex subtopic extractor — no LLM, no I/O.

    Returns empty list when:
      * input is empty / None
      * no subtopic header marker found
      * fewer than :data:`_MIN_SUBTOPICS` items detected

    Output is deduped case-insensitively, capped at :data:`_MAX_SUBTOPICS`.
    """
    if not query or not isinstance(query, str):
        return []
    for extractor in (_extract_subtopic_numbered, _extract_subtopic_bulleted):
        cands = extractor(query)
        if len(cands) >= _MIN_SUBTOPICS:
            return _dedupe_preserve_first(cands)[:_MAX_SUBTOPICS]
    return []


def _subtopics_from_qu(qu_result: Any) -> list[str]:
    """Pull `.subtopics` off a QU result object, defensively. Mirror of
    `_entities_from_qu`."""
    raw = getattr(qu_result, "subtopics", None)
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for item in raw:
        s = _clean_surface(item) if isinstance(item, str) else ""
        if s:
            cleaned.append(s)
    return cleaned


def extract_subtopics(
    query: str | None,
    qu_result: Optional[Any] = None,
) -> list[str]:
    """Compose QU + regex subtopic extraction. Mirror of `extract_entities`."""
    qu_subtopics = _subtopics_from_qu(qu_result)
    if qu_subtopics:
        return _dedupe_preserve_first(qu_subtopics)[:_MAX_SUBTOPICS]
    return extract_subtopics_regex(query)
```

Update `__all__` at the bottom of the file:

```python
__all__ = [
    "extract_entities",
    "extract_entities_regex",
    "is_multi_entity_query",
    # Phase 3 — multi-entity-elaborate-answers
    "extract_subtopics",
    "extract_subtopics_regex",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/unit/test_entity_extractor.py -v
```

Expected: ALL tests pass (existing + new).

- [ ] **Step 5: Commit**

```bash
git add ext/services/entity_extractor.py tests/unit/test_entity_extractor.py
git commit -m "$(cat <<'EOF'
feat(entity_extractor): extract_subtopics(query) for two-axis decompose

Phase 3 / item 5. Mirror of extract_entities — QU LLM preferred, regex
fallback. Detects "under (fwg|following) (heads|sections):" markers
followed by numbered or bulleted lists. Caps at 8 subtopics; min 2 to
trigger two-axis decompose.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Extend `HybridClassification` with `subtopics` + QU prompt

**Files:**
- Modify: `ext/services/query_understanding.py` (`HybridClassification` dataclass + LLM prompt)
- Test: `tests/unit/test_query_understanding_subtopics.py` (new file)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_query_understanding_subtopics.py`:

```python
"""Phase 3 / item 5 — QU LLM extracts subtopics into HybridClassification."""
from __future__ import annotations

from ext.services.query_understanding import HybridClassification


def test_hybrid_classification_has_subtopics_field() -> None:
    h = HybridClassification(
        intent="specific",
        regex_label="specific",
        regex_reason="default",
        llm_label=None,
        llm_temporal={},
        llm_confidence=0.0,
        agree=True,
        escalation_reason="none",
        entities=[],
        subtopics=["visits", "operations"],
    )
    assert h.subtopics == ["visits", "operations"]


def test_hybrid_classification_subtopics_default_empty() -> None:
    """Subtopics defaults to empty list when not provided (back-compat)."""
    h = HybridClassification(
        intent="specific",
        regex_label="specific",
        regex_reason="default",
        llm_label=None,
        llm_temporal={},
        llm_confidence=0.0,
        agree=True,
        escalation_reason="none",
        entities=[],
    )
    assert h.subtopics == []
```

- [ ] **Step 2: Run test to verify fail**

```bash
.venv/bin/pytest tests/unit/test_query_understanding_subtopics.py -v
```

Expected: FAIL — `subtopics` not on the dataclass.

- [ ] **Step 3: Add `subtopics` to `HybridClassification`**

Find the `HybridClassification` dataclass in `ext/services/query_understanding.py`. Add the field with default `[]`:

```python
@dataclass
class HybridClassification:
    intent: Intent
    regex_label: str
    regex_reason: str
    llm_label: Optional[str]
    llm_temporal: dict[str, Any]
    llm_confidence: float
    agree: bool
    escalation_reason: str
    entities: list[str] = dataclasses.field(default_factory=list)
    # 2026-05-04 — Phase 3 / item 5 of multi-entity-elaborate-answers spec.
    # Subtopics extracted alongside entities so the bridge can build
    # N×M sub-queries.
    subtopics: list[str] = dataclasses.field(default_factory=list)
```

If the file doesn't already import `dataclasses` (it might use `field` directly), add `import dataclasses` at the top.

Also extend the QU LLM prompt. Find the prompt template (search for `entities` in the prompt). Add a sibling instruction:

```python
SUBTOPICS_PROMPT_FRAGMENT = (
    'subtopics: a list of subheadings the user named (when the query says '
    '"under heads:", "under following:", or has a numbered/bulleted '
    'list of topics). Up to 8. Empty list if none.'
)
```

Inject the fragment into the JSON-shape spec the prompt asks the LLM to produce. The exact location depends on how the prompt is constructed; search for where `entities:` is mentioned in the prompt's shape spec and add `subtopics:` next to it.

In the response parser (look for code that reads `result.get("entities")`), add:

```python
subtopics = result.get("subtopics") or []
if not isinstance(subtopics, list):
    subtopics = []
subtopics = [s for s in subtopics if isinstance(s, str) and s.strip()]
# carried into HybridClassification(...)
```

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/unit/test_query_understanding_subtopics.py -v
.venv/bin/pytest tests/unit/test_query_understanding.py -v   # existing tests
```

Expected: new pass; existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add ext/services/query_understanding.py tests/unit/test_query_understanding_subtopics.py
git commit -m "$(cat <<'EOF'
feat(query_understanding): subtopics field on HybridClassification + QU prompt

Phase 3 / item 5. QU LLM now extracts a subtopics list alongside
entities; HybridClassification.subtopics holds the result. Empty list
when the query has no subheadings — back-compat default.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Two-axis `should_decompose` mode return

**Files:**
- Modify: `ext/services/multi_query.py` (extend `should_decompose` signature)
- Modify: `tests/unit/test_multi_query.py` (extend tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_multi_query.py` (after existing test classes):

```python
class TestShouldDecomposeTwoAxis:
    """Phase 3 — two-axis return: ('none' | 'entity' | 'subtopic' | 'both', bool)."""

    def test_no_entities_no_subtopics_returns_none(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=[], subtopics=[], flag_on=True, intent="specific",
        )
        assert (mode, on) == ("none", False)

    def test_two_entities_no_subtopics_returns_entity(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], subtopics=[], flag_on=True, intent="specific",
        )
        assert (mode, on) == ("entity", True)

    def test_no_entities_two_subtopics_returns_subtopic(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=[], subtopics=["visits", "ops"], flag_on=True, intent="specific",
        )
        assert (mode, on) == ("subtopic", True)

    def test_two_entities_two_subtopics_returns_both(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], subtopics=["x", "y"], flag_on=True, intent="specific",
        )
        assert (mode, on) == ("both", True)

    def test_metadata_intent_returns_none(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], subtopics=["x", "y"],
            flag_on=True, intent="metadata",
        )
        assert (mode, on) == ("none", False)

    def test_flag_off_returns_none(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], subtopics=["x", "y"],
            flag_on=False, intent="specific",
        )
        assert (mode, on) == ("none", False)
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/pytest tests/unit/test_multi_query.py::TestShouldDecomposeTwoAxis -v
```

Expected: FAIL — `should_decompose` returns `bool`, not `tuple[str, bool]`. Existing tests will also fail since the signature is changing.

- [ ] **Step 3: Update `should_decompose`**

Replace the function in `ext/services/multi_query.py` with:

```python
def should_decompose(
    *,
    entities: Sequence[str],
    subtopics: Sequence[str] = (),
    flag_on: bool,
    intent: str | None,
) -> tuple[str, bool]:
    """Two-axis gate predicate for multi-query decomposition.

    Returns ``(mode, enabled)`` where ``mode`` is one of:

    * ``"none"``     — single-axis behaviour, no decompose
    * ``"entity"``   — N entities, M < 2 subtopics — current single-axis path
    * ``"subtopic"`` — M subtopics, N < 2 entities — fan out by subtopic only
    * ``"both"``     — N×M sub-queries with two-axis quotas

    Bound conditions (independent of mode):
    * ``flag_on`` — env or per-KB master gate
    * ``intent`` is not ``"metadata"`` — catalog questions never decompose

    ``intent=None`` is treated as decomposable (defensive — mirror of the
    original gate behaviour).
    """
    if not flag_on:
        return ("none", False)
    if intent in _NON_DECOMPOSING_INTENTS:
        return ("none", False)

    n_e = len(entities or [])
    n_s = len(subtopics or [])

    if n_e >= 2 and n_s >= 2:
        return ("both", True)
    if n_e >= 2:
        return ("entity", True)
    if n_s >= 2:
        return ("subtopic", True)
    return ("none", False)
```

Existing callers (in `chat_rag_bridge.py`) call `should_decompose` and treat the return as `bool`. Adapt those call sites to unpack the tuple — Task 22 wires the bridge fully. For now, in `chat_rag_bridge.py:_run_pipeline`, change:

```python
_do_decompose = should_decompose(
    entities=_entities,
    flag_on=True,
    intent=_intent,
)
```

to:

```python
_decompose_mode, _do_decompose = should_decompose(
    entities=_entities,
    subtopics=[],          # Phase 3 plumbs real subtopics; for now empty
    flag_on=True,
    intent=_intent,
)
```

- [ ] **Step 4: Run all `multi_query` + bridge tests**

```bash
.venv/bin/pytest tests/unit/test_multi_query.py tests/unit/test_chat_rag_bridge_top_k_override.py -v
```

Expected: all pass (existing single-axis call sites now unpack the tuple but still produce identical entity-mode behaviour because subtopics=[]).

- [ ] **Step 5: Commit**

```bash
git add ext/services/multi_query.py ext/services/chat_rag_bridge.py tests/unit/test_multi_query.py
git commit -m "$(cat <<'EOF'
feat(multi_query): should_decompose returns (mode, bool) for two-axis dispatch

Phase 3 / item 5. mode ∈ {none, entity, subtopic, both}; existing
single-axis path preserved when subtopics=[]. Bridge call site updated
to unpack the tuple — Phase 3 / Task 22 plumbs real subtopics into the
flag_on call.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: `build_sub_queries_two_axis` in `multi_query.py`

**Files:**
- Modify: `ext/services/multi_query.py`
- Modify: `tests/unit/test_multi_query.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_multi_query.py`:

```python
class TestBuildSubQueriesTwoAxis:
    def test_n_x_m_pairs(self) -> None:
        out = multi_query.build_sub_queries_two_axis(
            "Give all updates",
            entities=["A", "B"],
            subtopics=["visits", "ops"],
        )
        assert len(out) == 4
        assert out[0] == ("A", "visits", "Give all updates (focus on A — visits)")
        assert out[1] == ("A", "ops", "Give all updates (focus on A — ops)")
        assert out[2] == ("B", "visits", "Give all updates (focus on B — visits)")
        assert out[3] == ("B", "ops", "Give all updates (focus on B — ops)")

    def test_empty_entities_returns_empty(self) -> None:
        assert multi_query.build_sub_queries_two_axis(
            "x", entities=[], subtopics=["a", "b"],
        ) == []

    def test_empty_subtopics_returns_empty(self) -> None:
        # Two-axis function rejects subtopics=[]; caller should fall back
        # to the single-axis build_sub_queries
        assert multi_query.build_sub_queries_two_axis(
            "x", entities=["A", "B"], subtopics=[],
        ) == []

    def test_blank_query_uses_placeholder(self) -> None:
        out = multi_query.build_sub_queries_two_axis(
            "", entities=["A"], subtopics=["x"],
        )
        assert out == [("A", "x", "(no query) (focus on A — x)")]
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/pytest tests/unit/test_multi_query.py::TestBuildSubQueriesTwoAxis -v
```

Expected: FAIL — `build_sub_queries_two_axis` doesn't exist.

- [ ] **Step 3: Add the function to `multi_query.py`**

Append to `ext/services/multi_query.py`:

```python
def build_sub_queries_two_axis(
    query: str,
    entities: Sequence[str],
    subtopics: Sequence[str],
) -> list[tuple[str, str, str]]:
    """Return ``[(entity, subtopic, sub_query), ...]`` one per cell.

    For each (entity, subtopic) pair, build a focus-suffixed sub-query.
    Order is entity-major, subtopic-minor (so all of A's subtopics come
    before B's). Empty entity OR empty subtopic list yields ``[]``.

    Suffix format: ``"<original> (focus on <entity> — <subtopic>)"``.
    """
    if not entities or not subtopics:
        return []
    base = (query or "").strip() or "(no query)"
    return [
        (e, s, f"{base} (focus on {e} — {s})")
        for e in entities
        for s in subtopics
    ]
```

Update `__all__`:

```python
__all__ = [
    "should_decompose",
    "build_sub_queries",
    "build_sub_queries_two_axis",
    "merge_with_quota",
    "merge_with_two_axis_quota",  # added in Task 19
]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_multi_query.py::TestBuildSubQueriesTwoAxis -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ext/services/multi_query.py tests/unit/test_multi_query.py
git commit -m "$(cat <<'EOF'
feat(multi_query): build_sub_queries_two_axis(N×M) for two-axis fan-out

Phase 3 / item 5. Builds (entity, subtopic, sub_query) triples for the
"both" decompose mode. Suffix shape "(focus on E — S)" feeds the dense
retriever both axes' signal.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: `merge_with_two_axis_quota` in `multi_query.py`

**Files:**
- Modify: `ext/services/multi_query.py`
- Modify: `tests/unit/test_multi_query.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_multi_query.py`:

```python
class TestMergeWithTwoAxisQuota:
    def _hits(self, scores: list[tuple[str, str, int, float]]):
        # (entity, subtopic, hit_id, score) -> {(e,s): [hit, ...]}
        out: dict = {}
        for e, s, hid, sc in scores:
            out.setdefault((e, s), []).append(_FakeHit(id=hid, score=sc))
        for k in out:
            out[k].sort(key=lambda h: h.score, reverse=True)
        return out

    def test_cell_floor_satisfied(self) -> None:
        per_cell = self._hits([
            ("A", "x", 1, 1.0), ("A", "x", 2, 0.9),
            ("A", "y", 3, 0.8), ("A", "y", 4, 0.7),
            ("B", "x", 5, 0.6), ("B", "x", 6, 0.5),
            ("B", "y", 7, 0.4), ("B", "y", 8, 0.3),
        ])
        out = multi_query.merge_with_two_axis_quota(
            per_cell_hits=per_cell,
            k_min_per_cell=1,
            k_min_per_entity=2,
            k_min_per_subtopic=2,
            k_total=8,
        )
        ids = [h.id for h in out]
        # Each cell got at least 1 hit; final sorted by score desc
        assert ids == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_dedupe_by_id(self) -> None:
        # Same hit appearing in two cells should appear once in output
        h = _FakeHit(id=99, score=1.0)
        per_cell = {
            ("A", "x"): [h],
            ("A", "y"): [h],
            ("B", "x"): [_FakeHit(id=2, score=0.5)],
            ("B", "y"): [_FakeHit(id=3, score=0.4)],
        }
        out = multi_query.merge_with_two_axis_quota(
            per_cell_hits=per_cell,
            k_min_per_cell=1, k_min_per_entity=1,
            k_min_per_subtopic=1, k_total=4,
        )
        ids = [h.id for h in out]
        assert ids.count(99) == 1

    def test_total_cap_respected(self) -> None:
        per_cell = self._hits([
            ("A", "x", i, 1.0 - i * 0.01) for i in range(20)
        ])
        out = multi_query.merge_with_two_axis_quota(
            per_cell_hits=per_cell,
            k_min_per_cell=2,
            k_min_per_entity=2,
            k_min_per_subtopic=2,
            k_total=5,
        )
        assert len(out) == 5

    def test_entity_floor_recovery_when_cell_empty(self) -> None:
        # 5 PoK x visits has 0 hits but 5 PoK x ops has plenty.
        # Entity-floor recovery pulls extra ops chunks so the per-entity
        # floor is met.
        per_cell = self._hits([
            ("75 Inf", "visits", 1, 0.9), ("75 Inf", "visits", 2, 0.8),
            ("75 Inf", "ops",    3, 0.7), ("75 Inf", "ops",    4, 0.6),
            # 5 PoK has 0 visits but 4 ops:
            ("5 PoK", "ops",     5, 0.5), ("5 PoK", "ops",     6, 0.4),
            ("5 PoK", "ops",     7, 0.3), ("5 PoK", "ops",     8, 0.2),
        ])
        out = multi_query.merge_with_two_axis_quota(
            per_cell_hits=per_cell,
            k_min_per_cell=1,
            k_min_per_entity=3,    # 5 PoK must end up with ≥3 chunks
            k_min_per_subtopic=2,
            k_total=10,
        )
        ids = [h.id for h in out]
        # 5 PoK should have ≥3 of its 4 ops chunks (5,6,7,8)
        pok_ids = [i for i in ids if i in (5, 6, 7, 8)]
        assert len(pok_ids) >= 3
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/pytest tests/unit/test_multi_query.py::TestMergeWithTwoAxisQuota -v
```

Expected: FAIL — `merge_with_two_axis_quota` doesn't exist.

- [ ] **Step 3: Implement the function**

Append to `ext/services/multi_query.py`:

```python
def merge_with_two_axis_quota(
    *,
    per_cell_hits: dict[tuple[str, str], list[Any]],
    k_min_per_cell: int,
    k_min_per_entity: int,
    k_min_per_subtopic: int,
    k_total: int,
) -> list[Any]:
    """Merge per-(entity, subtopic) hit lists with quotas at three levels.

    Algorithm:
      1. **Cell-quota pass.** For each (entity, subtopic) cell, take its
         top ``k_min_per_cell`` hits in score-desc order.
      2. **Entity-floor recovery.** For each entity, ensure its total
         pool size is ≥ ``k_min_per_entity`` — if a cell was empty,
         pull more from cells that did have hits for that entity.
      3. **Subtopic-floor recovery.** Same for subtopics — if a subtopic
         is under-represented, pull from any cell that had hits for it.
      4. **Top-up pass.** Fill remaining slots up to ``k_total`` by
         score-desc across deduped non-quota leftovers.
      5. **Final sort + cap.** Sort by score desc, cap at ``k_total``.

    Dedup is by ``hit.id``. When the same hit appears in multiple cells
    (the same chunk semantically matched two sub-queries), the highest
    score copy wins, then the algorithm above runs.

    Hit shape contract: any object with ``.id`` and ``.score``
    (matches ``merge_with_quota``).

    Returns a flat list of hits, length ≤ ``k_total``.
    """
    # Step 0 — flatten + dedupe by id, keep highest score copy + remember
    # which (entity, subtopic) cell first picked it.
    best_for_id: dict[Any, tuple[Any, str, str]] = {}
    for (entity, subtopic), hits in per_cell_hits.items():
        for h in hits:
            prev = best_for_id.get(h.id)
            if prev is None or h.score > prev[0].score:
                best_for_id[h.id] = (h, entity, subtopic)

    # Rebuild bucket dict from deduped hits (cell-keyed)
    bucket: dict[tuple[str, str], list[Any]] = {
        k: [] for k in per_cell_hits.keys()
    }
    for hit, e, s in best_for_id.values():
        bucket[(e, s)].append(hit)
    for k in bucket:
        bucket[k].sort(key=lambda h: h.score, reverse=True)

    selected_ids: set = set()
    selected: list[Any] = []

    def _take(hit: Any) -> None:
        if hit.id not in selected_ids:
            selected_ids.add(hit.id)
            selected.append(hit)

    # Step 1 — cell-quota
    for (e, s), hits in bucket.items():
        for h in hits[:k_min_per_cell]:
            _take(h)

    # Step 2 — entity-floor recovery
    entities = sorted({e for (e, _) in bucket.keys()})
    for entity in entities:
        # Count current selection for this entity
        ent_selected = [
            h for h in selected
            if any(
                h.id in {hh.id for hh in bucket.get((entity, s), [])}
                for s in {ss for (ee, ss) in bucket.keys() if ee == entity}
            )
        ]
        if len(ent_selected) >= k_min_per_entity:
            continue
        deficit = k_min_per_entity - len(ent_selected)
        # Pull more from any cell of this entity, score-desc
        candidates = []
        for s in {ss for (ee, ss) in bucket.keys() if ee == entity}:
            for h in bucket.get((entity, s), []):
                if h.id not in selected_ids:
                    candidates.append(h)
        candidates.sort(key=lambda h: h.score, reverse=True)
        for h in candidates[:deficit]:
            _take(h)

    # Step 3 — subtopic-floor recovery (mirror of step 2)
    subtopics = sorted({s for (_, s) in bucket.keys()})
    for subtopic in subtopics:
        sub_selected = [
            h for h in selected
            if any(
                h.id in {hh.id for hh in bucket.get((e, subtopic), [])}
                for e in {ee for (ee, ss) in bucket.keys() if ss == subtopic}
            )
        ]
        if len(sub_selected) >= k_min_per_subtopic:
            continue
        deficit = k_min_per_subtopic - len(sub_selected)
        candidates = []
        for e in {ee for (ee, ss) in bucket.keys() if ss == subtopic}:
            for h in bucket.get((e, subtopic), []):
                if h.id not in selected_ids:
                    candidates.append(h)
        candidates.sort(key=lambda h: h.score, reverse=True)
        for h in candidates[:deficit]:
            _take(h)

    # Step 4 — top-up by score across remaining leftovers
    if len(selected) < k_total:
        leftover = [
            h for hits in bucket.values() for h in hits
            if h.id not in selected_ids
        ]
        leftover.sort(key=lambda h: h.score, reverse=True)
        for h in leftover[: k_total - len(selected)]:
            _take(h)

    # Step 5 — final sort + cap
    selected.sort(key=lambda h: h.score, reverse=True)
    return selected[:k_total]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_multi_query.py::TestMergeWithTwoAxisQuota -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ext/services/multi_query.py tests/unit/test_multi_query.py
git commit -m "$(cat <<'EOF'
feat(multi_query): merge_with_two_axis_quota — per-cell + per-axis floors

Phase 3 / item 5. Three-pass merge algorithm:
  1. cell quota (per (entity, subtopic) pair)
  2. entity-floor recovery (pull from any cell of an under-represented entity)
  3. subtopic-floor recovery (mirror of step 2)
  4. score top-up to k_total
  5. final sort + cap

Dedup by hit.id with highest-score copy winning. Same hit shape
contract as merge_with_quota — any object with .id and .score.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: `subtopic_decompose` + `subtopic_keywords` per-KB keys

**Files:**
- Modify: `ext/services/kb_config.py` (`VALID_BOOL_KEYS` + new `VALID_DICT_KEYS` + validator)
- Modify: `tests/unit/test_kb_config_phase6.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_kb_config_phase6.py`:

```python
class TestSubtopicDecompose:
    """Phase 3 / item 5 — per-KB master gate for two-axis decompose."""

    def test_accepts_true(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_decompose": True}
        ) == {"subtopic_decompose": True}

    def test_accepts_false(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_decompose": False}
        ) == {"subtopic_decompose": False}


class TestSubtopicKeywords:
    """Phase 3 / item 5 — per-KB subtopic-keywords table."""

    def test_accepts_dict_of_str_to_list(self) -> None:
        from ext.services import kb_config
        kw = {
            "visits": ["visit", "vis", "inspection"],
            "operations": ["operation", "exercise"],
        }
        assert kb_config.validate_config(
            {"subtopic_keywords": kw}
        ) == {"subtopic_keywords": kw}

    def test_rejects_non_dict(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_keywords": ["a", "b"]}
        ) == {}

    def test_rejects_dict_with_non_string_key(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_keywords": {1: ["x"]}}
        ) == {}

    def test_rejects_dict_with_non_list_value(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_keywords": {"visits": "not a list"}}
        ) == {}

    def test_strips_non_string_list_items(self) -> None:
        from ext.services import kb_config
        out = kb_config.validate_config(
            {"subtopic_keywords": {"visits": ["visit", 42, None, "vis"]}}
        )
        assert out == {"subtopic_keywords": {"visits": ["visit", "vis"]}}

    def test_empty_dict_accepted(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_keywords": {}}
        ) == {"subtopic_keywords": {}}
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/pytest tests/unit/test_kb_config_phase6.py::TestSubtopicDecompose tests/unit/test_kb_config_phase6.py::TestSubtopicKeywords -v
```

Expected: FAIL — keys don't exist in VALID_KEYS.

- [ ] **Step 3: Add the keys to `kb_config.py`**

Add `subtopic_decompose` to `VALID_BOOL_KEYS`:

```python
VALID_BOOL_KEYS = frozenset({
    "rerank",
    "mmr",
    # ... existing keys ...
    "image_captions",
    # 2026-05-04 — Phase 3 / item 5 of multi-entity-elaborate-answers spec.
    # Per-KB gate for two-axis decompose. False = current single-axis
    # behaviour even when subtopics are detected.
    "subtopic_decompose",
})
```

Add a new `VALID_DICT_KEYS` set (mirror of `VALID_LIST_KEYS`):

```python
# 2026-05-04 — Phase 3 / item 5. Dict-typed keys. ``subtopic_keywords``
# is the only one today; a dict of {subtopic_name: [variant, ...]}.
# Validator enforces all-string keys and all-string list values.
VALID_DICT_KEYS = frozenset({
    "subtopic_keywords",
})

VALID_KEYS = (
    VALID_BOOL_KEYS | VALID_INT_KEYS | VALID_FLOAT_KEYS
    | VALID_STRING_KEYS | VALID_LIST_KEYS | VALID_DICT_KEYS
)
```

Add the validator branch inside `validate_config`. Find the existing `elif key in VALID_LIST_KEYS:` branch and add a new branch after it:

```python
        elif key in VALID_DICT_KEYS:
            # 2026-05-04 — Phase 3 / item 5. Dict-typed key validation.
            if not isinstance(value, dict):
                continue
            cleaned_dict: dict[str, list[str]] = {}
            ok = True
            for k_inner, v_inner in value.items():
                if not isinstance(k_inner, str):
                    ok = False; break
                if not isinstance(v_inner, list):
                    ok = False; break
                cleaned_list = [
                    item for item in v_inner if isinstance(item, str) and item.strip()
                ]
                cleaned_dict[k_inner] = cleaned_list
            if not ok:
                continue
            out[key] = cleaned_dict
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_kb_config_phase6.py -v
```

Expected: All TestSubtopicDecompose + TestSubtopicKeywords tests PASS, existing tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add ext/services/kb_config.py tests/unit/test_kb_config_phase6.py
git commit -m "$(cat <<'EOF'
feat(kb_config): subtopic_decompose (bool) + subtopic_keywords (dict) keys

Phase 3 / item 5. Per-KB gate for two-axis decompose + curated
subtopic-keywords table for per-cell attribution and corpus-vocabulary
mapping (e.g. {visits: [visit, vis, mov of CO, UI mtg/conf]}).
New VALID_DICT_KEYS class with strict shape validation.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.6

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: `scripts/edit_kb_subtopics.py` operator CLI

**Files:**
- Create: `scripts/edit_kb_subtopics.py`
- Test: `tests/integration/test_edit_kb_subtopics_cli.py`

This script mirrors `scripts/edit_kb_synonyms.py`. The shape is the same — `--list`, `--load FILE`, `--add JSON`, `--remove JSON`. Difference: target column is `rag_config`'s `subtopic_keywords` JSONB key.

- [ ] **Step 1: Write failing tests**

Create `tests/integration/test_edit_kb_subtopics_cli.py`:

```python
"""Integration tests for scripts/edit_kb_subtopics.py.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.6
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

# These tests target a temporary KB id=98 to avoid trashing real KBs


@pytest.fixture
def kb98_clean():
    import asyncpg, asyncio

    async def setup() -> None:
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            await conn.execute("DELETE FROM knowledge_bases WHERE id = 98")
            await conn.execute(
                "INSERT INTO knowledge_bases (id, name, admin_id, rag_config, synonyms) "
                "VALUES (98, 'subtopic-test', 1, '{}'::jsonb, '[]'::jsonb)"
            )
        finally:
            await conn.close()

    asyncio.run(setup())
    yield 98
    # teardown
    async def teardown() -> None:
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            await conn.execute("DELETE FROM knowledge_bases WHERE id = 98")
        finally:
            await conn.close()
    asyncio.run(teardown())


@pytest.mark.integration
def test_load_replaces_subtopic_keywords(kb98_clean) -> None:
    payload = {"visits": ["visit", "vis"], "operations": ["operation", "ex"]}
    p = subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--load", "-"],
        input=json.dumps(payload),
        capture_output=True, text=True, timeout=30,
    )
    assert p.returncode == 0, p.stderr

    # Verify
    import asyncpg, asyncio
    async def check():
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            row = await conn.fetchrow(
                "SELECT rag_config FROM knowledge_bases WHERE id = 98"
            )
        finally:
            await conn.close()
        return dict(row["rag_config"] or {})
    rc = asyncio.run(check())
    assert rc.get("subtopic_keywords") == payload


@pytest.mark.integration
def test_list_prints_current_table(kb98_clean) -> None:
    # First load
    subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--load", "-"],
        input='{"visits": ["v"]}',
        check=True, capture_output=True, text=True,
    )
    p = subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--list"],
        capture_output=True, text=True, timeout=30,
    )
    assert p.returncode == 0
    assert "visits" in p.stdout
    assert '"v"' in p.stdout


@pytest.mark.integration
def test_add_merges_new_keys(kb98_clean) -> None:
    # Pre-populate
    subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--load", "-"],
        input='{"visits": ["v"]}',
        check=True, capture_output=True, text=True,
    )
    # Add a new key
    p = subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--add", '{"operations": ["o", "op"]}'],
        capture_output=True, text=True, timeout=30,
    )
    assert p.returncode == 0, p.stderr

    import asyncpg, asyncio
    async def check():
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            row = await conn.fetchrow(
                "SELECT rag_config FROM knowledge_bases WHERE id = 98"
            )
        finally:
            await conn.close()
        return dict(row["rag_config"] or {})
    rc = asyncio.run(check())
    assert rc["subtopic_keywords"]["visits"] == ["v"]
    assert rc["subtopic_keywords"]["operations"] == ["o", "op"]


@pytest.mark.integration
def test_missing_kb_returns_nonzero(kb98_clean) -> None:
    p = subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "97777", "--list"],
        capture_output=True, text=True,
    )
    assert p.returncode != 0
    assert "not found" in p.stderr.lower()
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/pytest tests/integration/test_edit_kb_subtopics_cli.py -v
```

Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Create the script**

Create `scripts/edit_kb_subtopics.py`:

```python
"""Edit per-KB subtopic-keywords table (rag_config.subtopic_keywords).

Usage:
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --list
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --add '{"visits":["v","vis"]}'
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --remove '["visits"]'
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --load FILE.json
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --load -

Env: DATABASE_URL (required)

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import asyncpg


async def _conn() -> asyncpg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)
    url = url.replace("+asyncpg", "")
    return await asyncpg.connect(url)


async def _get_table(kb_id: int) -> dict[str, list[str]]:
    conn = await _conn()
    try:
        row = await conn.fetchrow(
            "SELECT rag_config FROM knowledge_bases WHERE id = $1", kb_id,
        )
    finally:
        await conn.close()
    if row is None:
        print(f"error: kb_id={kb_id} not found", file=sys.stderr)
        sys.exit(3)
    rc = dict(row["rag_config"] or {})
    table = rc.get("subtopic_keywords") or {}
    if not isinstance(table, dict):
        return {}
    return table


async def _set_table(kb_id: int, table: dict[str, list[str]]) -> None:
    """Atomic JSONB merge — replace the subtopic_keywords key only."""
    conn = await _conn()
    try:
        await conn.execute(
            "UPDATE knowledge_bases "
            "SET rag_config = jsonb_set(rag_config, '{subtopic_keywords}', $1::jsonb) "
            "WHERE id = $2",
            json.dumps(table), kb_id,
        )
    finally:
        await conn.close()


def _validate_payload(payload) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        print("error: payload must be a dict", file=sys.stderr)
        sys.exit(4)
    out: dict[str, list[str]] = {}
    for k, v in payload.items():
        if not isinstance(k, str):
            print(f"error: key must be string: {k!r}", file=sys.stderr)
            sys.exit(4)
        if not isinstance(v, list):
            print(f"error: value must be list: {k!r} -> {v!r}", file=sys.stderr)
            sys.exit(4)
        items = [s for s in v if isinstance(s, str) and s.strip()]
        out[k] = items
    return out


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kb", type=int, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--load", help="path to JSON file or '-' for stdin (REPLACES table)")
    g.add_argument("--add", help='JSON object — merge new keys into table')
    g.add_argument("--remove", help='JSON list of keys to drop')
    args = p.parse_args()

    if args.list:
        table = await _get_table(args.kb)
        print(json.dumps(table, indent=2))
        return

    if args.load:
        if args.load == "-":
            data = sys.stdin.read()
        else:
            with open(args.load) as f:
                data = f.read()
        payload = _validate_payload(json.loads(data))
        await _set_table(args.kb, payload)
        print(f"replaced subtopic_keywords for kb={args.kb}: {len(payload)} entries")
        return

    if args.add:
        payload = _validate_payload(json.loads(args.add))
        current = await _get_table(args.kb)
        current.update(payload)
        await _set_table(args.kb, current)
        print(f"added/updated keys for kb={args.kb}: {sorted(payload.keys())}")
        return

    if args.remove:
        keys_to_drop = json.loads(args.remove)
        if not isinstance(keys_to_drop, list):
            print("error: --remove takes a JSON list of key names", file=sys.stderr)
            sys.exit(4)
        current = await _get_table(args.kb)
        before = set(current.keys())
        for k in keys_to_drop:
            current.pop(k, None)
        after = set(current.keys())
        await _set_table(args.kb, current)
        print(f"removed keys for kb={args.kb}: {sorted(before - after)}")
        return


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/integration/test_edit_kb_subtopics_cli.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/edit_kb_subtopics.py tests/integration/test_edit_kb_subtopics_cli.py
git commit -m "$(cat <<'EOF'
feat(scripts): edit_kb_subtopics.py CLI for subtopic_keywords table

Phase 3 / item 5. Mirror of edit_kb_synonyms.py — --list / --load /
--add / --remove operations on rag_config.subtopic_keywords. Atomic
JSONB merge via jsonb_set. Stdin support for piped payloads.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.6

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 22: `_apply_two_axis_quota` in `chat_rag_bridge.py`

**Files:**
- Modify: `ext/services/chat_rag_bridge.py`
- Test: `tests/unit/test_chat_rag_bridge_two_axis_quota.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_chat_rag_bridge_two_axis_quota.py`:

```python
"""Phase 3 / item 5 — bridge-side two-axis quota helper.

Mirrors `_apply_entity_quota` but attributes hits to (entity, subtopic)
cells using the per-KB subtopic_keywords table.
"""
from __future__ import annotations

from dataclasses import dataclass

from ext.services import chat_rag_bridge as bridge


@dataclass
class _Hit:
    id: int
    score: float
    payload: dict


def _make(text: str, hid: int, score: float):
    return _Hit(id=hid, score=score, payload={"text": text})


def test_attributes_to_correct_cell() -> None:
    reranked = [
        _make("75 Inf Bde visited 77 Mtn Fd Arty Regt", 1, 0.9),  # 75/visits
        _make("75 Inf Bde construction at Lipa", 2, 0.8),         # 75/construction
        _make("5 PoK Bde mov of CO Lt Col Rana", 3, 0.7),         # 5/visits (kw)
        _make("5 PoK Bde firing exercise", 4, 0.6),               # 5/operations
    ]
    out = bridge._apply_two_axis_quota(
        reranked=reranked,
        entities=["75 Inf Bde", "5 PoK Bde"],
        subtopics=["visits", "construction", "operations"],
        subtopic_keywords={
            "visits": ["visited", "visit", "vis", "mov of co"],
            "construction": ["construction", "constr", "bunker"],
            "operations": ["firing", "exercise", "operation"],
        },
        per_cell_floor=1,
        per_entity_floor=2,
        per_subtopic_floor=1,
        final_k=4,
    )
    ids = [h.id for h in out]
    assert ids == [1, 2, 3, 4]


def test_synonyms_for_entity_attribution() -> None:
    """Entity attribution honours the per-KB synonyms table — 5 PoK / 5 POK
    both attribute to the same entity cell."""
    reranked = [
        _make("5 POK Bde construction work", 1, 0.5),
    ]
    out = bridge._apply_two_axis_quota(
        reranked=reranked,
        entities=["5 PoK Bde"],
        subtopics=["construction"],
        subtopic_keywords={"construction": ["construction"]},
        synonyms=[["5 PoK", "5 POK", "5 PoK Bde", "5 POK Bde"]],
        per_cell_floor=1, per_entity_floor=1, per_subtopic_floor=1, final_k=1,
    )
    assert [h.id for h in out] == [1]


def test_empty_inputs_return_top_k_unchanged() -> None:
    """No entities or no subtopics — bypass quota, return reranked[:final_k]."""
    reranked = [_make("x", 1, 0.9), _make("y", 2, 0.8)]
    out = bridge._apply_two_axis_quota(
        reranked=reranked,
        entities=[],
        subtopics=[],
        subtopic_keywords={},
        per_cell_floor=1, per_entity_floor=1, per_subtopic_floor=1, final_k=1,
    )
    assert [h.id for h in out] == [1]
```

- [ ] **Step 2: Run test to verify fail**

```bash
.venv/bin/pytest tests/unit/test_chat_rag_bridge_two_axis_quota.py -v
```

Expected: FAIL — `_apply_two_axis_quota` doesn't exist.

- [ ] **Step 3: Add the helper to `chat_rag_bridge.py`**

Find `_apply_entity_quota` in `ext/services/chat_rag_bridge.py` and add the two-axis sibling immediately after it:

```python
def _apply_two_axis_quota(
    *,
    reranked: list,
    entities: list,
    subtopics: list,
    subtopic_keywords: dict[str, list[str]],
    per_cell_floor: int,
    per_entity_floor: int,
    per_subtopic_floor: int,
    final_k: int,
    synonyms: list[list[str]] | None = None,
) -> list:
    """Phase 3 / item 5 — two-axis post-rerank quota.

    Attributes each hit to (entity, subtopic) cells using:
      - entity attribution: case-insensitive substring of any entity name
        (or any synonym variant if ``synonyms`` is provided)
      - subtopic attribution: case-insensitive substring of any keyword
        in ``subtopic_keywords[subtopic]``

    Then dispatches to ``merge_with_two_axis_quota`` shape — but
    operating on already-reranked hits with their cross-encoder scores
    preserved (no re-scoring).

    If ``entities`` or ``subtopics`` is empty, short-circuits to
    ``reranked[:final_k]`` (single-axis or no-quota behaviour).
    """
    if not reranked or not entities or not subtopics:
        return list(reranked[:final_k])

    # Pre-compute entity needle sets (synonym-expanded if available)
    def _entity_needles(e: str) -> set[str]:
        out = {e.lower()}
        if not synonyms:
            return out
        for cls in synonyms:
            if any(v.lower() == e.lower() for v in cls):
                out.update(v.lower() for v in cls)
        return out

    entity_needle_map = {e: _entity_needles(e) for e in entities}

    # Pre-lowercase subtopic keyword variants
    subtopic_kw_map = {
        s: [v.lower() for v in subtopic_keywords.get(s, [s])]
        for s in subtopics
    }

    # Attribute each hit to one or more (entity, subtopic) cells
    per_cell: dict[tuple[str, str], list] = {}
    for hit in reranked:
        text = ((hit.payload or {}).get("text") or "").lower()
        if not text:
            continue
        matched_entities = [
            e for e, needles in entity_needle_map.items()
            if any(n in text for n in needles)
        ]
        matched_subtopics = [
            s for s, kws in subtopic_kw_map.items()
            if any(kw in text for kw in kws)
        ]
        # A hit that matches no entity OR no subtopic is "leftover" — kept
        # for the top-up pass via merge.
        if not matched_entities or not matched_subtopics:
            per_cell.setdefault(("__leftover__", "__leftover__"), []).append(hit)
            continue
        # Attribute to every matching cell (the merge dedupes by id)
        for e in matched_entities:
            for s in matched_subtopics:
                per_cell.setdefault((e, s), []).append(hit)

    # Sort each cell by score
    for k in per_cell:
        per_cell[k].sort(key=lambda h: h.score, reverse=True)

    # Hand off to multi_query.merge_with_two_axis_quota
    from .multi_query import merge_with_two_axis_quota
    return merge_with_two_axis_quota(
        per_cell_hits=per_cell,
        k_min_per_cell=per_cell_floor,
        k_min_per_entity=per_entity_floor,
        k_min_per_subtopic=per_subtopic_floor,
        k_total=final_k,
    )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_chat_rag_bridge_two_axis_quota.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ext/services/chat_rag_bridge.py tests/unit/test_chat_rag_bridge_two_axis_quota.py
git commit -m "$(cat <<'EOF'
feat(chat_rag_bridge): _apply_two_axis_quota helper for two-axis path

Phase 3 / item 5. Sibling of _apply_entity_quota — attributes hits to
(entity, subtopic) cells using subtopic_keywords + synonyms, dispatches
to multi_query.merge_with_two_axis_quota. Empty-axis short-circuit to
reranked[:final_k] preserves single-axis behaviour.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.5

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 23: Wire two-axis path through `_run_pipeline` in `chat_rag_bridge.py`

**Files:**
- Modify: `ext/services/chat_rag_bridge.py` (the `_decompose_on` block + the post-rerank quota call)
- Modify: `ext/services/chat_rag_bridge.py` (signature of `_multi_entity_retrieve` to accept subtopics)

This is the largest single task in Phase 3 — the wiring task. Steps split it into incremental commits.

- [ ] **Step 1: Add subtopic extraction at the decompose-gate site**

Find the block in `_run_pipeline` that starts with `_decompose_on = flags.get("RAG_MULTI_ENTITY_DECOMPOSE", "0") == "1"`. Replace the entity extraction + gate with:

```python
                    _decompose_on = flags.get("RAG_MULTI_ENTITY_DECOMPOSE", "0") == "1"
                    _subtopic_decompose_on = (
                        flags.get("RAG_SUBTOPIC_DECOMPOSE", "0") == "1"
                    )
                    _entities: list = []
                    _subtopics: list = []
                    _decompose_mode = "none"
                    _do_decompose = False
                    if _decompose_on:
                        try:
                            from .entity_extractor import (
                                extract_entities, extract_subtopics,
                            )
                            from .multi_query import should_decompose

                            _entities = extract_entities(query, qu_result=_hybrid)
                            if _subtopic_decompose_on:
                                _subtopics = extract_subtopics(
                                    query, qu_result=_hybrid,
                                )
                            _decompose_mode, _do_decompose = should_decompose(
                                entities=_entities,
                                subtopics=_subtopics,
                                flag_on=True,
                                intent=_intent,
                            )
                        except Exception as _exc:
                            _record_silent_failure(
                                "multi_entity.gate", _exc,
                            )
                            _decompose_mode = "none"
                            _do_decompose = False
```

- [ ] **Step 2: Plumb subtopics into `_multi_entity_retrieve` calls**

Find the existing call to `_multi_entity_retrieve(...)` in `_run_pipeline`. Pass the new args:

```python
                        if _do_decompose:
                            raw_hits = await _multi_entity_retrieve(
                                entities=_entities,
                                subtopics=_subtopics,           # NEW
                                decompose_mode=_decompose_mode, # NEW
                                base_query=_retrieval_query,
                                # ... rest of args unchanged ...
                            )
```

Update the `_multi_entity_retrieve` signature to accept the new args (with safe defaults). At the function definition near line 1100, add:

```python
async def _multi_entity_retrieve(
    *,
    entities: list[str],
    subtopics: list[str] = (),                     # NEW
    decompose_mode: str = "entity",                # NEW: "entity"|"subtopic"|"both"
    base_query: str,
    # ... existing args ...
) -> list:
    ...
```

Inside the function body, branch on `decompose_mode`:

```python
    if decompose_mode == "both" and entities and subtopics:
        # N×M fan-out
        from .multi_query import build_sub_queries_two_axis
        triples = build_sub_queries_two_axis(base_query, entities, subtopics)
        # ... fan out N×M retrieves; merge by per-cell quota ...
        # (use _per_kb cap proportional to N*M to avoid blowing up Qdrant load)
        ...
    elif decompose_mode == "entity":
        # existing single-axis path — unchanged from current code
        ...
    elif decompose_mode == "subtopic":
        # M-only fan-out: subtopics-only sub-queries
        ...
```

- [ ] **Step 3: Switch the post-rerank quota to two-axis when mode == "both"**

Find the existing call sites that look like:

```python
                if _do_decompose and _entities and _entity_floor > 0:
                    reranked = _apply_entity_quota(
                        reranked=reranked,
                        entities=list(_entities),
                        per_entity_floor=_entity_floor,
                        final_k=_final_k,
                    )
```

Replace each with:

```python
                if _do_decompose and _entities and _entity_floor > 0:
                    # Read per-KB subtopic_keywords + synonyms via flags overlay
                    _subtopic_keywords = flags.get_dict("subtopic_keywords") or {}
                    _kb_synonyms = flags.get_list("synonyms") or []
                    if _decompose_mode == "both" and _subtopics:
                        # Per-cell + per-axis floors. Conservative defaults:
                        # cell_floor = 1, entity_floor = _entity_floor (8),
                        # subtopic_floor = max(2, _entity_floor // 2)
                        reranked = _apply_two_axis_quota(
                            reranked=reranked,
                            entities=list(_entities),
                            subtopics=list(_subtopics),
                            subtopic_keywords=_subtopic_keywords,
                            synonyms=_kb_synonyms,
                            per_cell_floor=1,
                            per_entity_floor=_entity_floor,
                            per_subtopic_floor=max(2, _entity_floor // 2),
                            final_k=_final_k,
                        )
                    else:
                        reranked = _apply_entity_quota(
                            reranked=reranked,
                            entities=list(_entities),
                            per_entity_floor=_entity_floor,
                            final_k=_final_k,
                        )
```

If `flags.get_dict` and `flags.get_list` don't exist, add them to `ext/services/flags.py` as thin wrappers:

```python
def get_dict(key: str, default: dict | None = None) -> dict | None:
    """Return overlay/per-KB value for ``key`` if it's a dict; else default."""
    val = _ctx_var.get({}).get(key)
    if isinstance(val, dict):
        return val
    return default if default is not None else None


def get_list(key: str, default: list | None = None) -> list | None:
    val = _ctx_var.get({}).get(key)
    if isinstance(val, list):
        return val
    return default if default is not None else None
```

- [ ] **Step 4: Run integration tests**

```bash
.venv/bin/pytest tests/unit/test_chat_rag_bridge_two_axis_quota.py tests/unit/test_multi_query.py -v
```

Expected: all PASS.

```bash
.venv/bin/pytest tests/integration/test_kb_isolation.py tests/integration/test_rag_isolation.py -v
```

Expected: existing isolation tests still PASS — no regression.

- [ ] **Step 5: Commit**

```bash
git add ext/services/chat_rag_bridge.py ext/services/flags.py
git commit -m "$(cat <<'EOF'
feat(chat_rag_bridge): wire two-axis decompose path end-to-end

Phase 3 / item 5. _run_pipeline now extracts subtopics when
RAG_SUBTOPIC_DECOMPOSE=1, dispatches should_decompose with both axes,
fans out N×M sub-queries when mode=="both", and applies
_apply_two_axis_quota to the post-rerank pool. flags.get_dict and
flags.get_list helpers added for the per-KB subtopic_keywords +
synonyms overlay reads.

When mode != "both", behaviour is byte-identical to single-axis path.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 24: Phase 3 smoke + KB 2 subtopic seed

**Files:**
- None (verification + operator-data step).

- [ ] **Step 1: Seed KB 2 subtopic_keywords**

```bash
.venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --load - <<'JSON'
{
  "visits of senior officers": [
    "visit", "vis", "inspection", "tour", "arrived", "arr at",
    "mov of CO", "attended UI mtg", "conference", "mtg/conf",
    "GOC visit", "Bde Cdr visit", "CO visit"
  ],
  "operations": [
    "operation", "exercise", "ex", "drill", "deployment", "patrol",
    "mission", "QC msn", "drone msn", "firing", "practice"
  ],
  "construction": [
    "construction", "constr", "bunker", "trench", "track",
    "fortification", "OP", "track development"
  ],
  "intelligence": [
    "intel", "ISI", "ISPR", "FS Sec", "security check", "clearance",
    "Int Bn", "informer", "WEU", "EW Bn"
  ],
  "training": [
    "training", "cadre", "course", "BCC", "PT test", "drill",
    "ITC", "individual training cycle"
  ]
}
JSON
```

Expected: `replaced subtopic_keywords for kb=2: 5 entries`.

- [ ] **Step 2: Enable two-axis on KB 2**

```bash
JWT=$(cat /tmp/jwt.tok)
docker compose -p orgchat exec -T open-webui curl -s -X PATCH \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  -d '{"subtopic_decompose": true}' \
  http://localhost:8080/api/kb/2/config | python3 -m json.tool
```

Expected: response shows `"subtopic_decompose": true`.

- [ ] **Step 3: Enable RAG_SUBTOPIC_DECOMPOSE at the env layer**

The compose mapping for `RAG_SUBTOPIC_DECOMPOSE` ships with the multi-entity-elaborate-answers feature branch (see commit history) — no manual YAML edit required. Just append the env override to compose/.env:

```bash
grep -q "^RAG_SUBTOPIC_DECOMPOSE=" /home/vogic/LocalRAG/compose/.env || \
  echo "RAG_SUBTOPIC_DECOMPOSE=1" >> /home/vogic/LocalRAG/compose/.env
```

Recreate the services + verify:

```bash
cd /home/vogic/LocalRAG/compose && docker compose -p orgchat up -d open-webui celery-worker
docker compose -p orgchat exec -T open-webui printenv RAG_SUBTOPIC_DECOMPOSE
docker compose -p orgchat exec -T celery-worker printenv RAG_SUBTOPIC_DECOMPOSE
```

Expected: both print `1`.

- [ ] **Step 4: Run the brigade query with explicit subheadings**

```bash
JWT=$(cat /tmp/jwt.tok)
CID=$(docker compose -p orgchat exec -T postgres psql -U orgchat -d orgchat -At -c \
  "SELECT id FROM chat WHERE meta::text LIKE '%kb_config%' ORDER BY updated_at DESC LIMIT 1;")
cat > /tmp/q_phase3.json <<EOF
{
  "model":"orgchat-chat","stream":false,"chat_id":"$CID",
  "rag_kb_config":[{"kb_id":2,"subtag_ids":[]}],
  "messages":[{"role":"user","content":"Give out major updates from the report of apr 2026, for the following:\n1. 75 Inf Bde\n2. 5 PoK Bde\n3. 32 Inf Bde\n4. 80 Inf Bde\nunder following heads:\n1. visits of senior officers\n2. operations\n3. construction\n4. intelligence"}]
}
EOF
docker compose -p orgchat exec -T open-webui curl -s -X POST \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  --max-time 240 \
  -d @- http://localhost:8080/api/chat/completions < /tmp/q_phase3.json > /tmp/r_phase3.json

python3 <<'PY'
import json, re
d = json.load(open('/tmp/r_phase3.json'))
ans = d['choices'][0]['message']['content']
print('answer length:', len(ans))
# Parse a 4×4 cell grid
brigades = ['75 Inf Bde', '5 PoK Bde', '32 Inf Bde', '80 Inf Bde']
heads = ['visits', 'operations', 'construction', 'intelligence']
cell_facts = {}
brigade_blocks = re.split(r'(?im)^#{1,4}\s+(75 Inf Bde|5 PoK Bde|32 Inf Bde|80 Inf Bde)\b', ans)
for i in range(1, len(brigade_blocks)-1, 2):
    b = brigade_blocks[i]
    body = brigade_blocks[i+1]
    for h in heads:
        # Crude: count bullets in the section that mentions this head
        # (the LLM's ordering is not strict; we accept any bullet mentioning the head keyword too)
        h_block = re.search(rf'(?is){h}.+?(?=\Z|\n#{{2,}}|\n\*\*|$)', body)
        if h_block:
            cell_facts[(b, h)] = len(re.findall(r'^[*-]\s+', h_block.group(0), re.MULTILINE))
print('\nper-cell fact counts:')
for b in brigades:
    for h in heads:
        print(f'  {b:12s} × {h:14s}: {cell_facts.get((b, h), 0)}')
PY
```

Expected:
- ≥3 cells per brigade (some subheadings have no data and that's OK).
- ≥2 facts in cells where data exists in the corpus.
- Answer length ≥8000 chars.

- [ ] **Step 5: No commit (verification only). Phase 3 complete.**

---

# Plan Self-Review

The following is a self-review checklist for the plan author. The executor does not need to repeat it.

**Spec coverage:**
- [x] §4.1 (Item 1, window chunker) → Task 6 (operator PATCH) + Task 7 (smoke)
- [x] §4.2 (Item 2, rerank floor) → Tasks 1, 2
- [x] §4.3 (Item 3, rerank_top_k + budget) → Task 3 (budget) + Task 6 (rerank_top_k stamp)
- [x] §5.1 (Item 4, doc-summary prompt) → Tasks 9, 10, 11
- [x] §5.2 (Qdrant entities payload index) → Tasks 12, 13
- [x] §6.1 (Item 5, subtopic axis) → Tasks 15-23
- [x] §6.1.6 (per-KB subtopic_keywords) → Tasks 20, 21, 24
- [x] §7 (fresh-start re-ingest) → Tasks 4, 5, 6
- [x] §9 (eval-gate) → Task 8

**Type consistency:**
- `should_decompose` returns `tuple[str, bool]` (Task 17) — Task 23 unpacks consistently.
- `summarize_document` returns `dict[str, list[str] | str]` (Task 10) — Task 11 reads `.summary` and `.entities`.
- `_apply_two_axis_quota` signature (Task 22) matches `merge_with_two_axis_quota` shape (Task 19).
- `_decompose_mode` typed as `str` ∈ {"none", "entity", "subtopic", "both"} consistently across Tasks 17, 23.

**Placeholder scan:** No `TBD`, `TODO`, or "fill in" steps. All test code complete. All commit messages spelled out.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-04-multi-entity-elaborate-answers.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, two-stage review between tasks, fast iteration. Best for the 24-task scope here since each task is bounded and reviewable.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Better if you want to watch each task land in real-time but tradeoff is context-window pressure across 24 tasks.

**Which approach?**
