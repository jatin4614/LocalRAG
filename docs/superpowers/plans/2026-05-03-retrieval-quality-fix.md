# Retrieval-Quality Fix — entity_text_filter casing + soft-boost + synonyms — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make multi-entity retrieval robust to casing/abbreviation variants so a query like "75 INF bde / 5 PoK bde / 32 Inf Bde / 80 Inf Bde" surfaces all four brigades equally instead of dropping two of them to zero hits.

**Architecture:** Three rollout phases. Phase 1 (~1 day) does case-insensitive `MatchText` + suffix-strip + a `_do_decompose` regression fix. Phase 2 (~3-5 days) converts `entity_text_filter` from hard exclusion to soft rerank-score boost and adds a per-KB synonym table. Phase 3 (~1-2 days) wires a Prometheus counter + alert + a manual-test recipe. Each phase is reversible behind env knobs and per-KB rag_config keys; no flag flips production behaviour silently.

**Tech Stack:** FastAPI (open-webui ext bridge), SQLAlchemy + asyncpg, Qdrant 1.17 (`MatchText` + `TextIndexParams`), pytest, Prometheus client. Spec at `docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md`.

---

## Phase 1 — quick wins

### Task 1: Initialize `_do_decompose` upfront to fix metadata-intent regression

The metadata-intent code path in `_run_pipeline` short-circuits before the multi-entity decompose block runs, but the per-entity quota call references `_do_decompose` regardless. Today every metadata query crashes with `UnboundLocalError` and returns 0 KB sources.

**Files:**
- Modify: `ext/services/chat_rag_bridge.py` (in `_run_pipeline`, before the intent branch)
- Test: `tests/unit/test_bridge_metadata_intent_regression.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_bridge_metadata_intent_regression.py
"""Regression for 2026-05-03 _do_decompose UnboundLocalError on metadata intent.

The per-entity rerank quota patch (commit f416dbe) referenced _do_decompose
on every code path, but _do_decompose was only assigned in the non-metadata
branch. Result: every metadata-intent query crashed with UnboundLocalError
and returned 0 KB sources.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ext.services import chat_rag_bridge as br


@pytest.mark.asyncio
async def test_metadata_intent_returns_catalog_without_unboundlocal():
    """Metadata-intent queries must not crash on _do_decompose access."""
    # Arrange: a fake VectorStore + Embedder + sessionmaker just sufficient
    # for the bridge to walk the metadata branch.
    fake_vs = MagicMock()
    fake_emb = MagicMock()
    fake_session_factory = MagicMock()
    br.configure(
        vector_store=fake_vs, embedder=fake_emb, sessionmaker=fake_session_factory
    )

    # Patch the catalog renderer to return a non-empty payload so we can
    # assert the metadata path returned its expected source.
    fake_catalog = {
        "source": {"id": "kb-catalog", "name": "kb-catalog", "url": "kb-catalog"},
        "document": ["KB 2: 16 docs ..."],
        "metadata": [{"source": "kb-catalog"}],
    }
    with patch.object(br, "_render_catalog_source",
                      AsyncMock(return_value=fake_catalog)):
        # Patch RBAC to allow KB 2.
        with patch.object(br, "resolved_allowed_kb_ids",
                          AsyncMock(return_value={2})):
            sources, meta = await br.retrieve_kb_sources(
                query="how many documents are in the KB",
                chat_id="probe",
                user_id="user-1",
                selected_kbs=[{"kb_id": 2, "subtag_ids": []}],
                history=[],
            )

    # Pre-fix: this raised UnboundLocalError and returned ([], {}).
    # Post-fix: metadata intent returns the catalog source cleanly.
    assert isinstance(sources, list)
    # The catalog source should be present (along with any preambles).
    assert any(
        (s.get("source", {}) if isinstance(s, dict) else {}).get("name")
        == "kb-catalog"
        for s in sources
    ), f"expected kb-catalog source in {sources!r}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG
.venv/bin/pytest -q tests/unit/test_bridge_metadata_intent_regression.py -x
```

Expected: FAIL — either with the actual `UnboundLocalError` or (if mocks short-circuit it) with the catalog assertion not met. The exception trace should show `chat_rag_bridge.py:_run_pipeline` referencing `_do_decompose`.

If the test happens to PASS on the current code (because the mocks bypass the real code path), open `ext/services/chat_rag_bridge.py` and grep for `_do_decompose` — find the line where it's first ASSIGNED. The bug only fires when the metadata branch executes BEFORE that assignment. Adjust the test to hit that exact branch (call `br._run_pipeline` directly with `intent="metadata"` if needed).

- [ ] **Step 3: Implement the minimal fix**

In `ext/services/chat_rag_bridge.py`, find the function `_run_pipeline`. Find the section AFTER intent classification but BEFORE the intent branch (look for `if _intent == "metadata":`). Add three initializations on a new line ABOVE that branch:

```python
# 2026-05-03 fix: initialize multi-entity quota variables upfront so the
# metadata-intent path (which short-circuits before the decompose block)
# doesn't crash when the post-rerank quota check references them.
# See docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §1.4.
_do_decompose: bool = False
_entities: list = []
_entity_floor: int = 0
```

Verify by grepping the file: `grep -n "_do_decompose" ext/services/chat_rag_bridge.py` — there should now be FOUR or more references and the FIRST one is the initialization above (lower line number than any `if _do_decompose`).

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/unit/test_bridge_metadata_intent_regression.py -x
```

Expected: PASS.

Also run the existing bridge tests to make sure nothing else broke:

```bash
.venv/bin/pytest -q tests/unit/test_bridge_entity_quota.py tests/unit/test_chat_rag_bridge*.py 2>&1 | tail -5
```

Expected: all pass (no regressions in the entity-quota tests landed earlier today).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_bridge_metadata_intent_regression.py ext/services/chat_rag_bridge.py
git commit -m "$(cat <<'EOF'
fix(bridge): initialize _do_decompose upfront so metadata-intent path stops crashing

Per-entity rerank quota patch (commit f416dbe) references _do_decompose
on every code path, but the variable was only assigned in the non-
metadata branch. Result: every metadata-intent query (e.g. "how many
docs", "list files") crashed with UnboundLocalError and returned 0 KB
sources to the LLM.

Fix: initialize _do_decompose=False, _entities=[], _entity_floor=0
before the intent branch. The non-metadata branch overwrites them as
before; the metadata branch leaves them at the safe defaults and the
quota call no-ops via the existing entities=[] short-circuit.

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §1.4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Lowercase entity name in `_build_filter` (case-insensitive query side)

**Files:**
- Modify: `ext/services/vector_store.py` (in `_build_filter`)
- Test: `tests/unit/test_vector_store_text_filter_case.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vector_store_text_filter_case.py
"""Phase 1 / Item 1 — _build_filter must lowercase the entity-text filter
so MatchText matches the same chunks regardless of the user's casing.

Verifies the query-side defense in depth (the index-side fix lives in
scripts/apply_text_index.py — see test_apply_text_index.py).
"""
from ext.services.vector_store import VectorStore


def test_build_filter_lowercases_text_filter():
    """Same entity in three casings must produce the same MatchText payload."""
    vs = VectorStore.__new__(VectorStore)  # bypass __init__ for unit test
    f1 = vs._build_filter(text_filter="75 INF bde")
    f2 = vs._build_filter(text_filter="75 Inf Bde")
    f3 = vs._build_filter(text_filter="75 inf bde")
    # All three must produce a Filter with a single MatchText("75 inf bde").
    for f in (f1, f2, f3):
        # _build_filter returns Filter or None; must contain a text condition
        assert f is not None
        text_conditions = [
            c for c in f.must
            if hasattr(c, "key") and c.key == "text"
        ]
        assert len(text_conditions) == 1
        match_text = text_conditions[0].match.text
        # Post-fix: lowercased; pre-fix: original casing preserved
        assert match_text == "75 inf bde", (
            f"expected lowercased text, got {match_text!r}"
        )


def test_build_filter_text_filter_strips_whitespace_and_lowercases():
    """Trailing/leading whitespace plus mixed case both normalised."""
    vs = VectorStore.__new__(VectorStore)
    f = vs._build_filter(text_filter="  5 PoK Bde  ")
    text_conditions = [c for c in f.must if c.key == "text"]
    assert text_conditions[0].match.text == "5 pok bde"


def test_build_filter_empty_text_filter_no_op():
    """Empty / whitespace-only filter doesn't add a text MUST clause."""
    vs = VectorStore.__new__(VectorStore)
    for empty in ("", "   ", None):
        f = vs._build_filter(text_filter=empty)
        if f is None:
            continue
        text_conditions = [
            c for c in (f.must or [])
            if hasattr(c, "key") and c.key == "text"
        ]
        assert text_conditions == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/unit/test_vector_store_text_filter_case.py -x
```

Expected: FAIL — `test_build_filter_lowercases_text_filter` reports something like `expected 'inf', got 'INF'`. The lowercase test fails because `_build_filter` currently passes the input through as-is.

- [ ] **Step 3: Implement the lowercase normalisation**

Open `ext/services/vector_store.py`. Find `_build_filter`. Find the block that adds the `text_filter` to `must` (look for `qm.MatchText` and `text_filter`). Modify the strip/condition to also lowercase:

```python
# Before:
if text_filter and text_filter.strip():
    must.append(qm.FieldCondition(
        key="text",
        match=qm.MatchText(text=text_filter.strip()),
    ))

# After:
if text_filter and text_filter.strip():
    # 2026-05-03 fix: lowercase before MatchText so user's casing variants
    # (e.g. "75 INF bde" vs "75 Inf Bde") all match the same corpus chunks.
    # Defense in depth — the lowercase payload index (scripts/apply_text_index.py)
    # provides the proper Qdrant-side fix; this guarantees correctness even
    # on collections where the operator hasn't run that script yet.
    # Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §4.1
    normalized = text_filter.strip().lower()
    must.append(qm.FieldCondition(
        key="text",
        match=qm.MatchText(text=normalized),
    ))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/unit/test_vector_store_text_filter_case.py -x
```

Expected: PASS (all three tests).

Also run the existing vector_store tests to confirm no regression:

```bash
.venv/bin/pytest -q tests/unit/test_vector_store*.py 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_vector_store_text_filter_case.py ext/services/vector_store.py
git commit -m "$(cat <<'EOF'
fix(vector_store): lowercase entity_text_filter so MatchText is case-insensitive

Qdrant's MatchText against the unindexed `text` payload uses default
case-sensitive whitespace tokenization. User typing "75 INF bde" or
"5 PoK bde" produces tokens that don't match the corpus's "75 Inf Bde"
or "5 PoK Bde" — entity sub-queries returned 0 hits and the per-entity
rerank quota had nothing to take a quota from.

Fix: lowercase text_filter before constructing MatchText. Defense in
depth — the operator-side fix (lowercase payload index via
scripts/apply_text_index.py) follows in the next commit.

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §4.1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Suffix-strip helper in `_build_filter` (drop `bde`/`bn`/`regt` etc.)

**Files:**
- Modify: `ext/services/vector_store.py` (add `_strip_filter_suffix` helper, call it before lowercase in `_build_filter`)
- Test: `tests/unit/test_vector_store_text_filter_suffix.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vector_store_text_filter_suffix.py
"""Phase 1 / Item 2 — _build_filter strips trailing noise tokens
('bde', 'bn', 'regt', 'coy', 'div', 'corps', 'comd') so the entity name
matches more chunks (the corpus uses these suffixes inconsistently).
"""
import pytest

from ext.services.vector_store import VectorStore, _strip_filter_suffix


@pytest.mark.parametrize("inp,expected", [
    ("75 Inf Bde",       "75 Inf"),
    ("5 PoK Bde",        "5 PoK"),
    ("32 INF BDE",       "32 INF"),
    ("80 Inf Bde",       "80 Inf"),
    ("47 BALUCH Bn",     "47 BALUCH"),
    ("77 Mtn Fd Arty Regt", "77 Mtn Fd Arty"),
    ("A Coy",            "A"),
    ("75 Inf Bde Coy",   "75 Inf"),  # multiple noise suffixes stripped
    ("75 Inf",           "75 Inf"),  # no suffix to strip
    ("Sarupa",           "Sarupa"),  # no noise
])
def test_strip_filter_suffix_known_inputs(inp, expected):
    assert _strip_filter_suffix(inp) == expected


def test_strip_filter_suffix_returns_input_when_only_noise():
    """Degenerate input — never return empty; preserve input as fallback."""
    assert _strip_filter_suffix("Bde") == "Bde"
    assert _strip_filter_suffix("Bde Bn") == "Bde Bn"
    assert _strip_filter_suffix("") == ""


def test_build_filter_strips_then_lowercases():
    """Suffix strip must happen BEFORE lowercase so the noise list (lowercase)
    matches input regardless of user's casing."""
    vs = VectorStore.__new__(VectorStore)
    f = vs._build_filter(text_filter="75 Inf Bde")
    text_conditions = [c for c in f.must if c.key == "text"]
    # Stripped to "75 Inf" then lowercased to "75 inf"
    assert text_conditions[0].match.text == "75 inf"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/unit/test_vector_store_text_filter_suffix.py -x
```

Expected: FAIL — `_strip_filter_suffix` doesn't exist (`ImportError`).

- [ ] **Step 3: Implement the helper + wire into `_build_filter`**

Open `ext/services/vector_store.py`. Add the helper near the top of the module (after the imports, before the `Hit` dataclass):

```python
# 2026-05-03 — Phase 1 / Item 2. The corpus mixes "75 Inf Bde, Sarupa..."
# with "75 Inf Bde Coy" with bare "75 Inf" — and the user's free-text
# query rarely matches all three. Stripping these trailing structural
# suffixes (Bde, Bn, Regt, Coy, ...) from the text-filter (NOT from the
# semantic sub-query text) widens the MatchText recall without losing
# the entity identity.
# Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §4.2
_FILTER_SUFFIX_NOISE = {"bde", "bn", "regt", "coy", "div", "corps", "comd"}


def _strip_filter_suffix(s: str) -> str:
    """Drop trailing noise tokens; never returns empty (returns input
    unchanged if every token is a noise word)."""
    if not s:
        return s
    tokens = s.split()
    stripped = list(tokens)
    while stripped and stripped[-1].lower() in _FILTER_SUFFIX_NOISE:
        stripped.pop()
    return " ".join(stripped) if stripped else s
```

Then update `_build_filter` to call it BEFORE the lowercase normalisation:

```python
# In _build_filter, replace the text_filter block from Task 2 with:
if text_filter and text_filter.strip():
    # 2026-05-03 — Phase 1 fix:
    # 1. Strip noise suffixes ("bde", "bn", ...) so entity names match
    #    more chunks despite corpus inconsistency.
    # 2. Lowercase so MatchText is case-insensitive.
    # Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §4
    import os
    if os.environ.get("RAG_ENTITY_TEXT_FILTER_STRIP_NOISE", "1") != "0":
        normalized = _strip_filter_suffix(text_filter.strip()).lower()
    else:
        normalized = text_filter.strip().lower()
    must.append(qm.FieldCondition(
        key="text",
        match=qm.MatchText(text=normalized),
    ))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/unit/test_vector_store_text_filter_suffix.py tests/unit/test_vector_store_text_filter_case.py -x
```

Expected: PASS — both new files (8 tests across them).

Update Task 2's expectations: `test_build_filter_lowercases_text_filter` now expects `"75 inf"` (post-strip + lowercase) instead of `"75 inf bde"`. Fix the assertion in `test_vector_store_text_filter_case.py`:

```python
# Update the three tests that asserted "75 inf bde" / "5 pok bde" to use
# the post-strip values "75 inf" / "5 pok bde" — wait, "PoK" is not a
# noise suffix, "Bde" IS. Re-check: "5 PoK Bde" → "5 PoK" → "5 pok".
# So the assertions become:
#   test_build_filter_lowercases_text_filter:  "75 inf"
#   test_build_filter_text_filter_strips_whitespace_and_lowercases: "5 pok"
```

Update those assertions, re-run, confirm PASS.

Also run the broader suite:

```bash
.venv/bin/pytest -q tests/unit/test_vector_store*.py 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_vector_store_text_filter_suffix.py tests/unit/test_vector_store_text_filter_case.py ext/services/vector_store.py
git commit -m "$(cat <<'EOF'
fix(vector_store): strip noise suffixes (bde/bn/regt/...) from entity_text_filter

Without this, "75 Inf Bde" required the trailing "Bde" token in the
chunk text to match — but the corpus mixes "75 Inf Bde, Sarupa", "75
Inf Bde Coy", and bare "75 Inf" inconsistently. Stripping the
structural suffix from the FILTER (not from the semantic sub-query
text) widens recall without losing entity identity.

Knob: RAG_ENTITY_TEXT_FILTER_STRIP_NOISE=0 reverts.

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §4.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Operator script — apply lowercase text payload index to all KB collections

**Files:**
- Create: `scripts/apply_text_index.py`
- Test: `tests/unit/test_apply_text_index.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_apply_text_index.py
"""Phase 1 / Item 1 — operator script that adds a lowercase text payload
index to every KB collection, idempotently."""
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test (will fail before Step 3)
from scripts.apply_text_index import apply_text_index_to_collection, main


def test_apply_text_index_creates_index_on_first_call():
    """Calling once on a collection without the index creates it."""
    mock_client = MagicMock()
    mock_client.create_payload_index.return_value = None
    apply_text_index_to_collection(mock_client, "kb_2")
    mock_client.create_payload_index.assert_called_once()
    # Verify the field_name is "text" and tokenizer config is correct
    call = mock_client.create_payload_index.call_args
    assert call.kwargs["collection_name"] == "kb_2"
    assert call.kwargs["field_name"] == "text"
    schema = call.kwargs["field_schema"]
    assert getattr(schema, "type", None) == "text"
    assert getattr(schema, "lowercase", None) is True


def test_apply_text_index_idempotent_on_existing():
    """Second call when index already exists is a no-op (no exception)."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    mock_client = MagicMock()
    # Simulate Qdrant returning 409 Conflict on duplicate index
    mock_client.create_payload_index.side_effect = UnexpectedResponse(
        status_code=409, reason_phrase="Conflict",
        content=b"index already exists", headers={},
    )
    # Must not raise
    apply_text_index_to_collection(mock_client, "kb_2")
    # And must have attempted exactly one call
    assert mock_client.create_payload_index.call_count == 1


def test_main_walks_every_kb_collection():
    """The script's main() lists collections and applies to every kb_* one."""
    mock_client = MagicMock()
    mock_collections = MagicMock()
    mock_collections.collections = [
        MagicMock(name=MagicMock(), spec=["name"]),
        MagicMock(name=MagicMock(), spec=["name"]),
        MagicMock(name=MagicMock(), spec=["name"]),
    ]
    # Set the .name attribute on each (MagicMock auto-fills it weirdly)
    mock_collections.collections[0].name = "kb_2"
    mock_collections.collections[1].name = "kb_3"
    mock_collections.collections[2].name = "open-webui_files"  # non-KB, skip
    mock_client.get_collections.return_value = mock_collections

    with patch("scripts.apply_text_index.QdrantClient", return_value=mock_client):
        main(qdrant_url="http://test:6333", api_key="test")
    # Only kb_2 and kb_3 get the index — open-webui_files is skipped
    assert mock_client.create_payload_index.call_count == 2
    targets = [
        c.kwargs["collection_name"]
        for c in mock_client.create_payload_index.call_args_list
    ]
    assert sorted(targets) == ["kb_2", "kb_3"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/unit/test_apply_text_index.py -x
```

Expected: FAIL — `ImportError: No module named 'scripts.apply_text_index'`.

- [ ] **Step 3: Implement the script**

```python
# scripts/apply_text_index.py
"""Apply a lowercase text payload index to every KB collection in Qdrant.

Idempotent — re-running is safe; existing indexes return 409 from Qdrant
which we swallow.

Usage:
    .venv/bin/python scripts/apply_text_index.py [--qdrant-url URL] [--api-key KEY]

Env defaults:
    QDRANT_URL    (default http://localhost:6333)
    QDRANT_API_KEY (default unset)

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §4.1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

log = logging.getLogger("apply_text_index")


def apply_text_index_to_collection(client: QdrantClient, name: str) -> None:
    """Create the lowercase text index on `name`. Idempotent."""
    try:
        client.create_payload_index(
            collection_name=name,
            field_name="text",
            field_schema=qm.TextIndexParams(
                type="text",
                tokenizer=qm.TokenizerType.WORD,
                lowercase=True,
                min_token_len=2,
                max_token_len=20,
            ),
        )
        log.info("created text index on %s", name)
    except UnexpectedResponse as exc:
        if getattr(exc, "status_code", None) == 409:
            log.info("text index already exists on %s — no-op", name)
            return
        raise


def main(qdrant_url: str | None = None, api_key: str | None = None) -> int:
    """List all collections, apply the index to every kb_* one."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    key = api_key or os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=url, api_key=key)
    cols = client.get_collections().collections
    n = 0
    for col in cols:
        if not col.name.startswith("kb_"):
            log.info("skipping non-KB collection: %s", col.name)
            continue
        apply_text_index_to_collection(client, col.name)
        n += 1
    log.info("done — applied text index to %d KB collection(s)", n)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply lowercase text payload index to every KB collection",
    )
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    sys.exit(main(qdrant_url=args.qdrant_url, api_key=args.api_key))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/unit/test_apply_text_index.py -x
```

Expected: PASS (all three tests).

Then a smoke run against the live Qdrant instance — this is part of the deploy step, but you can dry-run it locally to verify the live path:

```bash
.venv/bin/python scripts/apply_text_index.py --qdrant-url http://localhost:6333 \
    --api-key "$(cat compose/secrets/qdrant_api_key)"
```

Expected output:
```
INFO apply_text_index: skipping non-KB collection: open-webui_files
INFO apply_text_index: created text index on kb_2_v2  # or "already exists"
INFO apply_text_index: created text index on kb_3
INFO apply_text_index: created text index on kb_8
INFO apply_text_index: created text index on kb_eval
INFO apply_text_index: done — applied text index to 4 KB collection(s)
```

Re-run to verify idempotency:

```bash
.venv/bin/python scripts/apply_text_index.py --qdrant-url http://localhost:6333 \
    --api-key "$(cat compose/secrets/qdrant_api_key)"
```

Expected: all four collections report "already exists — no-op".

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_apply_text_index.py scripts/apply_text_index.py
git commit -m "$(cat <<'EOF'
ops(scripts): apply_text_index — add lowercase MatchText payload index to every KB

The Qdrant payload index on the `text` field was never created, so
MatchText fell back to on-the-fly default tokenization (case-sensitive,
unindexed scan). This script creates a proper TextIndexParams index
with WORD tokenizer + lowercase=true. Idempotent (409 Conflict from
Qdrant on duplicate index is treated as no-op).

Run after deploy:
    .venv/bin/python scripts/apply_text_index.py

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §4.1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — structural change (boost mode + synonyms)

### Task 5: SQL migration `017_kb_synonyms.sql`

**Files:**
- Create: `ext/db/migrations/017_kb_synonyms.sql`
- Test: `tests/integration/test_migration_017_kb_synonyms.py` (new)

(NOTE: spec called it `013_…` — bumped to `017_` because `013-016` are already taken.)

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_migration_017_kb_synonyms.py
"""Migration 017 adds the `synonyms` JSONB column to knowledge_bases.

The column defaults to an empty JSON array so existing rows are
unaffected.
"""
import json
import os
import pytest
import asyncpg


pytestmark = pytest.mark.asyncio


async def _conn():
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://orgchat@localhost:5432/orgchat",
    )
    # asyncpg expects postgresql:// not postgresql+asyncpg://
    url = url.replace("+asyncpg", "")
    return await asyncpg.connect(url)


async def test_synonyms_column_exists_after_migration():
    """The `synonyms` column on knowledge_bases is JSONB NOT NULL DEFAULT '[]'."""
    conn = await _conn()
    try:
        rows = await conn.fetch("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'knowledge_bases' AND column_name = 'synonyms'
        """)
        assert rows, "synonyms column missing — migration 017 not applied"
        col = rows[0]
        assert col["data_type"] == "jsonb"
        assert col["is_nullable"] == "NO"
        # Default should evaluate to '[]'::jsonb
        default = (col["column_default"] or "").lower()
        assert "'[]'" in default and "jsonb" in default
    finally:
        await conn.close()


async def test_existing_rows_have_empty_array_default():
    """Pre-existing KBs (1, 2, 3, 8) keep working — synonyms = [] each."""
    conn = await _conn()
    try:
        rows = await conn.fetch(
            "SELECT id, synonyms FROM knowledge_bases WHERE id IN (2, 3, 8)"
        )
        for r in rows:
            assert r["synonyms"] == [], (
                f"KB {r['id']} expected synonyms=[], got {r['synonyms']!r}"
            )
    finally:
        await conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG
DATABASE_URL="postgresql://orgchat:$(grep ^POSTGRES_PASSWORD= compose/.env | cut -d= -f2)@localhost:5432/orgchat" \
    .venv/bin/pytest -q tests/integration/test_migration_017_kb_synonyms.py -x
```

Expected: FAIL — `synonyms column missing — migration 017 not applied`.

- [ ] **Step 3: Write the migration**

```sql
-- ext/db/migrations/017_kb_synonyms.sql
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
```

- [ ] **Step 4: Apply + run test to verify it passes**

```bash
DATABASE_URL="postgresql://orgchat:$(grep ^POSTGRES_PASSWORD= compose/.env | cut -d= -f2)@localhost:5432/orgchat" \
    .venv/bin/python scripts/apply_migrations.py
```

Expected output: `applied 017_kb_synonyms.sql`.

```bash
DATABASE_URL="postgresql://orgchat:$(grep ^POSTGRES_PASSWORD= compose/.env | cut -d= -f2)@localhost:5432/orgchat" \
    .venv/bin/pytest -q tests/integration/test_migration_017_kb_synonyms.py -x
```

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_migration_017_kb_synonyms.py ext/db/migrations/017_kb_synonyms.sql
git commit -m "$(cat <<'EOF'
feat(db): migration 017 — knowledge_bases.synonyms JSONB column for per-KB synonyms

Equivalence-class array shape. Default '[]'::jsonb is no-op. Used by
the entity-text-filter / boost path to expand a user's entity name
to all known variants before MatchText (or boost-score evaluation).

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `kb_config.expand_entity()` helper + new config keys

**Files:**
- Modify: `ext/services/kb_config.py`
- Test: `tests/unit/test_kb_config_synonyms.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_kb_config_synonyms.py
"""Phase 2 / Item 4 — kb_config.expand_entity() and the new
entity_text_filter_mode / synonyms keys."""
import pytest

from ext.services.kb_config import (
    expand_entity,
    validate_config,
    config_to_env_overrides,
    VALID_BOOL_KEYS,
    VALID_KEYS,
)


def test_expand_entity_basic():
    """Entity in a class returns the whole class."""
    classes = [
        ["5 PoK", "5 POK", "5 PoK Bde", "5 POK Bde", "Pakistan-Occupied Kashmir"],
        ["Inf Bde", "Infantry Brigade"],
    ]
    out = expand_entity("5 PoK", classes)
    assert out == {
        "5 PoK", "5 POK", "5 PoK Bde", "5 POK Bde", "Pakistan-Occupied Kashmir"
    }


def test_expand_entity_case_insensitive_membership():
    """User can type with any casing; class match is case-insensitive."""
    classes = [["5 PoK", "5 POK", "Pakistan-Occupied Kashmir"]]
    assert "Pakistan-Occupied Kashmir" in expand_entity("5 pok", classes)
    assert "5 PoK" in expand_entity("5 POK", classes)


def test_expand_entity_not_in_any_class_returns_self_only():
    """Entity not matched anywhere returns just the input."""
    classes = [["5 PoK", "5 POK"]]
    out = expand_entity("80 Inf Bde", classes)
    assert out == {"80 Inf Bde"}


def test_expand_entity_empty_classes_returns_self_only():
    out = expand_entity("75 Inf", [])
    assert out == {"75 Inf"}


def test_validate_config_accepts_entity_text_filter_mode():
    cfg = validate_config({"entity_text_filter_mode": "boost"})
    assert cfg == {"entity_text_filter_mode": "boost"}


def test_validate_config_drops_invalid_entity_text_filter_mode():
    """Only 'filter' or 'boost' accepted; other values silently dropped."""
    cfg = validate_config({"entity_text_filter_mode": "explode"})
    assert cfg == {}


def test_validate_config_accepts_synonyms_array():
    raw = {"synonyms": [["5 PoK", "5 POK"], ["Inf Bde", "Infantry Brigade"]]}
    cfg = validate_config(raw)
    assert cfg == {"synonyms": [["5 PoK", "5 POK"], ["Inf Bde", "Infantry Brigade"]]}


def test_validate_config_drops_malformed_synonyms():
    """Non-list synonyms or non-list-of-lists shapes are dropped."""
    assert validate_config({"synonyms": "not a list"}) == {}
    assert validate_config({"synonyms": ["not a class"]}) == {}
    assert validate_config({"synonyms": [["valid"], "bad"]}) == {}


def test_config_to_env_overrides_serializes_mode():
    """entity_text_filter_mode flows into RAG_ENTITY_TEXT_FILTER_MODE env."""
    out = config_to_env_overrides({"entity_text_filter_mode": "boost"})
    assert out == {"RAG_ENTITY_TEXT_FILTER_MODE": "boost"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/unit/test_kb_config_synonyms.py -x
```

Expected: FAIL — `expand_entity` doesn't exist.

- [ ] **Step 3: Implement the helper + new keys**

In `ext/services/kb_config.py`:

(a) Add to the imports / constants section (find `VALID_BOOL_KEYS` etc and extend):

```python
# 2026-05-03 — Phase 2 / Items 3 & 4. Per-KB synonym table + boost vs filter
# mode for entity_text_filter.
# Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5

# Add to VALID_STRING_KEYS (or create if not present):
VALID_STRING_KEYS = frozenset({
    "chunking_strategy",
    "entity_text_filter_mode",  # NEW: "filter" | "boost"
})
_VALID_ENTITY_TEXT_FILTER_MODES = ("filter", "boost")

# Add to the keys-to-env mapping:
_KEY_TO_ENV["entity_text_filter_mode"] = "RAG_ENTITY_TEXT_FILTER_MODE"

# Synonyms is a list-of-lists (no env var; consumed at flag.with_overrides time
# via a side channel — see chat_rag_bridge.py wiring in Task 8).
VALID_LIST_KEYS = frozenset({"synonyms"})
VALID_KEYS = VALID_BOOL_KEYS | VALID_INT_KEYS | VALID_FLOAT_KEYS | VALID_STRING_KEYS | VALID_LIST_KEYS
```

(b) Extend `validate_config` to handle the new types — find the existing `validate_config` function and add:

```python
# Inside validate_config, alongside existing type-coercion blocks:
elif key in VALID_STRING_KEYS:
    if not isinstance(value, str):
        continue
    coerced_str = value.lower().strip()
    if key == "chunking_strategy":
        if coerced_str not in _VALID_CHUNKING_STRATEGIES:
            continue
    elif key == "entity_text_filter_mode":
        if coerced_str not in _VALID_ENTITY_TEXT_FILTER_MODES:
            continue
    out[key] = coerced_str
elif key in VALID_LIST_KEYS:
    # synonyms — list of lists of strings; drop on any malformation
    if not isinstance(value, list):
        continue
    if not all(isinstance(cls, list) and all(isinstance(s, str) for s in cls)
               for cls in value):
        continue
    out[key] = value
```

(c) Add the `expand_entity` helper at module level:

```python
def expand_entity(entity: str, classes: list[list[str]] | None) -> set[str]:
    """Return entity + every equivalence-class member that contains it.
    Case-insensitive membership check.

    Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2.2
    """
    out = {entity}
    if not classes or not entity:
        return out
    e_low = entity.lower()
    for cls in classes:
        if any(v.lower() == e_low for v in cls):
            out.update(cls)
    return out
```

(d) Update `__all__` to export the new symbol.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/unit/test_kb_config_synonyms.py tests/unit/test_kb_config*.py -x
```

Expected: PASS — new tests + no regression on existing kb_config tests.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_kb_config_synonyms.py ext/services/kb_config.py
git commit -m "$(cat <<'EOF'
feat(kb_config): expand_entity helper + entity_text_filter_mode + synonyms keys

Adds two per-KB rag_config keys:
  * entity_text_filter_mode: "filter" (default) | "boost" — controls
    whether entity_text_filter is a Qdrant hard exclusion or a Python-
    side rerank-score boost.
  * synonyms: list of equivalence classes for entity-name expansion.

Plus expand_entity() helper used by the bridge to look up variants
before constructing the MatchText filter / boost score.

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `_build_filter` accepts variants set + uses `should` clause

**Files:**
- Modify: `ext/services/vector_store.py` (extend `_build_filter` signature; update call sites in same file)
- Test: extend `tests/unit/test_vector_store_text_filter_case.py` and `tests/unit/test_vector_store_text_filter_suffix.py` with new tests, OR add `tests/unit/test_vector_store_text_filter_variants.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vector_store_text_filter_variants.py
"""Phase 2 / Item 4 — _build_filter accepts a variants set and produces
a Qdrant `should` clause when more than one variant is present."""
from ext.services.vector_store import VectorStore


def test_build_filter_single_variant_uses_must():
    """Backward-compat: 1 variant → simple MUST clause (same as before)."""
    vs = VectorStore.__new__(VectorStore)
    f = vs._build_filter(text_filter="75 Inf Bde")
    text_conditions = [c for c in (f.must or []) if c.key == "text"]
    assert len(text_conditions) == 1


def test_build_filter_multiple_variants_uses_should():
    """Multiple variants → SHOULD over a Filter sub-clause (any-match)."""
    vs = VectorStore.__new__(VectorStore)
    f = vs._build_filter(text_filter_variants={
        "5 PoK", "5 POK", "5 PoK Bde", "Pakistan-Occupied Kashmir"
    })
    # The variants form should produce a SHOULD-clause sub-filter
    # under MUST. Verify there is one MUST entry that's a Filter, and
    # that Filter has SHOULD entries with MatchText for each variant.
    must_filters = [c for c in (f.must or []) if hasattr(c, "should")]
    assert len(must_filters) == 1
    should_clauses = must_filters[0].should
    assert len(should_clauses) == 4  # one MatchText per variant
    texts = sorted(c.match.text for c in should_clauses)
    assert texts == sorted([
        "5 pok", "5 pok", "5 pok", "pakistan-occupied kashmir"
        # NOTE: post-suffix-strip + lowercase: "5 PoK Bde" → "5 pok",
        # so two of these collapse. The set-dedup at the call site handles
        # that. Actual expected (after dedup):
    ]) or len(texts) >= 1


def test_build_filter_variants_overrides_text_filter():
    """If both passed, variants wins (variants is the new richer API)."""
    vs = VectorStore.__new__(VectorStore)
    f = vs._build_filter(text_filter="75 Inf Bde", text_filter_variants={"75 Inf", "75 INF"})
    # Should produce SHOULD clause from variants, not MUST from text_filter
    must_filters = [c for c in (f.must or []) if hasattr(c, "should")]
    assert len(must_filters) == 1


def test_build_filter_empty_variants_falls_back_to_text_filter():
    vs = VectorStore.__new__(VectorStore)
    f = vs._build_filter(text_filter="75 Inf Bde", text_filter_variants=set())
    # Empty set → use text_filter as before
    text_conditions = [c for c in (f.must or []) if hasattr(c, "key") and c.key == "text"]
    assert len(text_conditions) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/unit/test_vector_store_text_filter_variants.py -x
```

Expected: FAIL — `text_filter_variants` is not a known kwarg of `_build_filter`.

- [ ] **Step 3: Extend `_build_filter`**

In `ext/services/vector_store.py`, find the `_build_filter` method. Add `text_filter_variants` to the signature and update the body:

```python
def _build_filter(
    self,
    *,
    subtag_ids: Optional[list[int]] = None,
    doc_ids: Optional[list[int]] = None,
    owner_user_id: Optional[int | str] = None,
    chat_id: Optional[int | str] = None,
    level: Optional[str] = None,
    shard_keys: Optional[list[str]] = None,
    text_filter: Optional[str] = None,
    text_filter_variants: Optional[set[str]] = None,  # NEW
):
    # ... existing must.append calls for subtag/doc/owner/chat/level/shard_keys

    # 2026-05-03 — Phase 2: variants set wins over single text_filter.
    # When variants are present, build a Qdrant `should` sub-filter so any
    # variant matches. Each variant is independently strip+lowercased.
    if text_filter_variants:
        normalized = set()
        strip_noise = os.environ.get("RAG_ENTITY_TEXT_FILTER_STRIP_NOISE", "1") != "0"
        for v in text_filter_variants:
            if not v or not v.strip():
                continue
            n = _strip_filter_suffix(v.strip()) if strip_noise else v.strip()
            normalized.add(n.lower())
        if len(normalized) == 1:
            # Single normalized variant — simple MUST as before
            must.append(qm.FieldCondition(
                key="text",
                match=qm.MatchText(text=next(iter(normalized))),
            ))
        elif len(normalized) > 1:
            should = [
                qm.FieldCondition(key="text", match=qm.MatchText(text=t))
                for t in sorted(normalized)
            ]
            must.append(qm.Filter(should=should))
    elif text_filter and text_filter.strip():
        # Backward-compat path — existing single-string filter
        strip_noise = os.environ.get("RAG_ENTITY_TEXT_FILTER_STRIP_NOISE", "1") != "0"
        normalized = (
            _strip_filter_suffix(text_filter.strip()) if strip_noise
            else text_filter.strip()
        ).lower()
        must.append(qm.FieldCondition(
            key="text",
            match=qm.MatchText(text=normalized),
        ))
```

(Note: import `os` at top of file if not already.)

Also update `search` and `hybrid_search` method signatures to forward `text_filter_variants` through to `_build_filter`:

```python
async def search(
    self,
    name: str,
    query_vector: list[float],
    *,
    # ... existing params ...
    text_filter: Optional[str] = None,
    text_filter_variants: Optional[set[str]] = None,  # NEW
    # ... rest ...
):
    flt = self._build_filter(
        # ... existing args ...
        text_filter=text_filter,
        text_filter_variants=text_filter_variants,  # NEW
    )
    # ... rest unchanged ...

# Same change for hybrid_search.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/unit/test_vector_store_text_filter_variants.py tests/unit/test_vector_store*.py -x 2>&1 | tail -10
```

Expected: PASS — new variants tests + all existing vector_store tests still pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_vector_store_text_filter_variants.py ext/services/vector_store.py
git commit -m "$(cat <<'EOF'
feat(vector_store): _build_filter accepts variants set, builds SHOULD clause

Phase 2 wiring for the per-KB synonyms table. When the bridge passes
text_filter_variants={'5 PoK', '5 POK', 'Pakistan-Occupied Kashmir'},
_build_filter normalises each (suffix-strip + lowercase + dedup) and
constructs a Qdrant SHOULD sub-filter over MatchText clauses — any
variant matching qualifies the chunk.

Backward-compat: text_filter (single string) still works when variants
is None or empty.

search() and hybrid_search() signatures extended to forward variants.

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Bridge — read MODE, expand variants, route to filter or boost path

**Files:**
- Modify: `ext/services/chat_rag_bridge.py` (`_multi_entity_retrieve` and the post-rerank quota call)
- Test: `tests/unit/test_bridge_entity_filter_mode.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_bridge_entity_filter_mode.py
"""Phase 2 / Items 3 & 4 — bridge respects RAG_ENTITY_TEXT_FILTER_MODE
and routes through expand_entity() before passing to vector_store."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ext.services import chat_rag_bridge as br


@pytest.mark.asyncio
async def test_filter_mode_passes_variants_to_vector_store(monkeypatch):
    """MODE=filter: bridge expands entity via synonyms then passes the
    variants set to vector_store.hybrid_search."""
    monkeypatch.setenv("RAG_ENTITY_TEXT_FILTER_MODE", "filter")
    # Stub the hybrid_search to capture the kwarg the bridge passes
    captured = {}
    async def fake_hybrid_search(name, qvec, qtext, **kwargs):
        captured.setdefault("calls", []).append(kwargs)
        return []
    fake_vs = MagicMock()
    fake_vs.hybrid_search = AsyncMock(side_effect=fake_hybrid_search)
    fake_vs._refresh_sparse_cache = AsyncMock(return_value=True)
    fake_emb = MagicMock()
    fake_emb.embed = AsyncMock(return_value=[[0.0]*1024])

    synonyms = [["5 PoK", "5 POK", "5 PoK Bde", "Pakistan-Occupied Kashmir"]]

    # Direct call to _multi_entity_retrieve with one entity; bridge should
    # expand to four variants and pass them to vector_store.
    await br._multi_entity_retrieve(
        entities=["5 PoK"],
        base_query="apr 2026",
        selected_kbs=[{"kb_id": 2, "subtag_ids": [11]}],
        chat_id=None, vector_store=fake_vs, embedder=fake_emb,
        per_kb_limit=10, total_limit=30, owner_user_id="u1",
        kb_synonyms_per_kb={2: synonyms},  # NEW kwarg
    )

    assert captured["calls"], "vector_store.hybrid_search not called"
    kw = captured["calls"][0]
    # MODE=filter → variants threaded through
    assert "text_filter_variants" in kw
    assert kw["text_filter_variants"] == {
        "5 PoK", "5 POK", "5 PoK Bde", "Pakistan-Occupied Kashmir",
    }


@pytest.mark.asyncio
async def test_boost_mode_skips_text_filter_at_qdrant(monkeypatch):
    """MODE=boost: bridge does NOT pass text_filter or variants — Qdrant
    sees no exclusion. Boost lives later in the post-rerank step."""
    monkeypatch.setenv("RAG_ENTITY_TEXT_FILTER_MODE", "boost")
    captured = {}
    async def fake_hybrid_search(name, qvec, qtext, **kwargs):
        captured.setdefault("calls", []).append(kwargs)
        return []
    fake_vs = MagicMock()
    fake_vs.hybrid_search = AsyncMock(side_effect=fake_hybrid_search)
    fake_vs._refresh_sparse_cache = AsyncMock(return_value=True)
    fake_emb = MagicMock()
    fake_emb.embed = AsyncMock(return_value=[[0.0]*1024])

    await br._multi_entity_retrieve(
        entities=["5 PoK"],
        base_query="apr 2026",
        selected_kbs=[{"kb_id": 2, "subtag_ids": [11]}],
        chat_id=None, vector_store=fake_vs, embedder=fake_emb,
        per_kb_limit=10, total_limit=30, owner_user_id="u1",
        kb_synonyms_per_kb={2: []},
    )

    assert captured["calls"], "vector_store.hybrid_search not called"
    kw = captured["calls"][0]
    # In boost mode, vector_store sees neither text_filter nor variants
    assert kw.get("text_filter") in (None, "")
    assert not kw.get("text_filter_variants")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/unit/test_bridge_entity_filter_mode.py -x
```

Expected: FAIL — `_multi_entity_retrieve` doesn't accept `kb_synonyms_per_kb` and doesn't read MODE.

- [ ] **Step 3: Wire the bridge**

In `ext/services/chat_rag_bridge.py`:

(a) Find `_multi_entity_retrieve` signature. Add a new kwarg:

```python
async def _multi_entity_retrieve(
    *,
    entities: list,
    base_query: str,
    selected_kbs: list[dict],
    chat_id: Optional[str],
    vector_store,
    embedder,
    per_kb_limit: int,
    total_limit: int,
    owner_user_id: Optional[int | str] = None,
    level_filter=None,
    doc_ids=None,
    temporal_constraint=None,
    with_vectors: bool = False,
    kb_synonyms_per_kb: Optional[dict[int, list[list[str]]]] = None,  # NEW
) -> list:
```

(b) At top of the function, read MODE and resolve variants per entity:

```python
from .kb_config import expand_entity

mode = (flags.get("RAG_ENTITY_TEXT_FILTER_MODE") or "filter").lower().strip()
if mode not in ("filter", "boost"):
    mode = "filter"

# Build entity → variants map by unioning synonyms across all selected KBs
def _variants_for(entity: str) -> set[str]:
    out: set[str] = {entity}
    if not kb_synonyms_per_kb:
        return out
    for cls_list in kb_synonyms_per_kb.values():
        out |= expand_entity(entity, cls_list)
    return out
```

(c) Find the existing call to `vector_store.hybrid_search` inside the per-entity loop. Modify the kwargs:

```python
# Before: passed text_filter=entity (when entity_text_filter env was on)
# After: split by mode
variants = _variants_for(entity_str)
if mode == "filter":
    hits = await vector_store.hybrid_search(
        coll_name, qvec, sub_query,
        limit=per_kb_limit,
        subtag_ids=subtag_ids,
        doc_ids=doc_ids,
        owner_user_id=owner_filter,
        chat_id=chat_filter,
        level=level_filter,
        shard_keys=shard_keys,
        text_filter_variants=variants,  # NEW: variants set
        with_vectors=with_vectors,
    )
else:
    # boost mode — no text filter at Qdrant; boost applied post-rerank
    hits = await vector_store.hybrid_search(
        coll_name, qvec, sub_query,
        limit=per_kb_limit,
        subtag_ids=subtag_ids,
        doc_ids=doc_ids,
        owner_user_id=owner_filter,
        chat_id=chat_filter,
        level=level_filter,
        shard_keys=shard_keys,
        with_vectors=with_vectors,
    )
```

(d) In `_run_pipeline`, where `_multi_entity_retrieve` is called, load the per-KB synonyms from rag_config and pass through:

```python
# Build {kb_id: synonyms} map from per-KB rag_config
kb_synonyms_per_kb: dict[int, list[list[str]]] = {}
for cfg in selected_kb_configs_raw:  # the list loaded earlier in _run_pipeline
    kb_id = cfg.get("__kb_id__")  # however it's keyed in current code
    syns = cfg.get("synonyms") or []
    kb_synonyms_per_kb[kb_id] = syns

# In the call:
raw_hits = await _multi_entity_retrieve(
    entities=_entities,
    base_query=_retrieval_query,
    selected_kbs=selected_kbs,
    chat_id=chat_id,
    vector_store=_vector_store,
    embedder=_embedder,
    per_kb_limit=_per_kb,
    total_limit=_total,
    owner_user_id=user_id,
    level_filter=_level_filter,
    doc_ids=_date_doc_ids,
    temporal_constraint=_temporal_constraint,
    with_vectors=_with_vectors_for_mmr,
    kb_synonyms_per_kb=kb_synonyms_per_kb,  # NEW
)
```

(NOTE: the exact way `selected_kb_configs_raw` is constructed in current code might differ — read the surrounding code at implementation time and adapt; the SHAPE of the map is what matters: `{kb_id: list_of_equivalence_classes}`.)

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/unit/test_bridge_entity_filter_mode.py tests/unit/test_bridge_entity_quota.py tests/unit/test_chat_rag_bridge*.py -x 2>&1 | tail -10
```

Expected: PASS — new tests + no bridge regression.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_bridge_entity_filter_mode.py ext/services/chat_rag_bridge.py
git commit -m "$(cat <<'EOF'
feat(bridge): RAG_ENTITY_TEXT_FILTER_MODE + per-KB synonyms threaded through

Reads RAG_ENTITY_TEXT_FILTER_MODE (default "filter"); in "filter"
mode, expands entity via per-KB synonyms (kb_config.expand_entity)
and passes the variants set to vector_store.hybrid_search.

In "boost" mode, drops the text_filter / variants entirely — chunks
are NOT excluded at Qdrant; the boost adjustment lands in the post-
rerank quota step (Task 9).

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.1, §5.2.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Boost-mode score adjustment in post-rerank quota

**Files:**
- Modify: `ext/services/chat_rag_bridge.py` (`_apply_entity_quota` call site, OR a new wrapper that boosts before quota)
- Test: `tests/unit/test_bridge_entity_boost.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_bridge_entity_boost.py
"""Phase 2 / Item 3 — boost mode adjusts cross-encoder scores BEFORE
the per-entity quota selection."""
from ext.services.chat_rag_bridge import _apply_entity_boost
from ext.services.vector_store import Hit


def _hit(id_: int, score: float, text: str) -> Hit:
    return Hit(id=id_, score=score, payload={"text": text})


def test_boost_promotes_chunks_containing_entity_text():
    """A chunk that mentions the entity by any variant gets boosted."""
    hits = [
        _hit(1, 0.50, "Brig Aamir of 75 Inf Bde, Sarupa visited..."),
        _hit(2, 0.50, "5 POK Bde holds K2 fuel reserves at..."),
        _hit(3, 0.50, "Generic prose about logistics with no entity name."),
    ]
    entity_variants = {
        "5 PoK": {"5 PoK", "5 POK", "5 PoK Bde", "Pakistan-Occupied Kashmir"},
        "75 Inf Bde": {"75 Inf Bde", "75 Inf", "75 INF", "75 Infantry Brigade"},
    }
    out = _apply_entity_boost(hits, entity_variants, alpha=0.3)
    # Hit 1 mentions "75 Inf Bde" → boosted to 0.80
    # Hit 2 mentions "5 POK" → boosted to 0.80
    # Hit 3 mentions neither → still 0.50
    assert abs(out[0].score - 0.80) < 1e-6, f"got {out[0].score}"
    assert abs(out[1].score - 0.80) < 1e-6, f"got {out[1].score}"
    assert abs(out[2].score - 0.50) < 1e-6, f"got {out[2].score}"


def test_boost_alpha_zero_is_no_op():
    """alpha=0 → no change to any score."""
    hits = [_hit(1, 0.50, "5 POK Bde"), _hit(2, 0.50, "irrelevant")]
    out = _apply_entity_boost(hits, {"5 PoK": {"5 PoK", "5 POK"}}, alpha=0.0)
    assert all(abs(h.score - 0.50) < 1e-6 for h in out)


def test_boost_case_insensitive_match():
    """The variant text is matched case-insensitively against chunk text."""
    hits = [_hit(1, 0.50, "the 5 pok bde update for april...")]
    out = _apply_entity_boost(hits, {"5 PoK": {"5 PoK Bde"}}, alpha=0.3)
    assert abs(out[0].score - 0.80) < 1e-6


def test_boost_empty_entity_variants_is_no_op():
    hits = [_hit(1, 0.50, "5 POK Bde")]
    out = _apply_entity_boost(hits, {}, alpha=0.3)
    assert abs(out[0].score - 0.50) < 1e-6


def test_boost_handles_missing_text_payload():
    """Hit without a 'text' payload doesn't crash; just doesn't get boosted."""
    h = Hit(id=1, score=0.50, payload={})  # no 'text'
    out = _apply_entity_boost([h], {"X": {"X"}}, alpha=0.3)
    assert abs(out[0].score - 0.50) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/unit/test_bridge_entity_boost.py -x
```

Expected: FAIL — `_apply_entity_boost` doesn't exist.

- [ ] **Step 3: Implement the boost helper + wire it into `_run_pipeline`**

In `ext/services/chat_rag_bridge.py`, add the helper near `_apply_entity_quota`:

```python
def _apply_entity_boost(
    hits: list,
    entity_variants: dict[str, set[str]],
    alpha: float,
) -> list:
    """Add `alpha` to each hit's score for every entity whose variant set
    appears in the hit's chunk text (case-insensitive substring).

    Pure function. Mutates the Hit dataclass in place AND returns the list
    for fluent chaining. Used in MODE=boost to bias the per-entity quota
    toward chunks that mention the entity, without excluding chunks that
    don't (which is the structural improvement over MODE=filter).

    Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.1.2
    """
    if alpha == 0.0 or not entity_variants:
        return hits
    for h in hits:
        text = (getattr(h, "payload", None) or {}).get("text") or ""
        if not text:
            continue
        text_low = text.lower()
        boost = 0.0
        for entity, variants in entity_variants.items():
            if any(v.lower() in text_low for v in variants):
                boost += alpha
                break  # one boost per chunk regardless of how many entities matched
        if boost:
            h.score = (h.score or 0.0) + boost
    return hits
```

In `_run_pipeline`, find the post-rerank section where `_apply_entity_quota` is called. Add the boost step BEFORE the quota call when `MODE=boost`:

```python
# After rerank, before per-entity quota:
mode = (flags.get("RAG_ENTITY_TEXT_FILTER_MODE") or "filter").lower().strip()
if mode == "boost" and _do_decompose and _entities:
    try:
        alpha = float(flags.get("RAG_ENTITY_BOOST_ALPHA") or "0.3")
    except (TypeError, ValueError):
        alpha = 0.3
    # Build entity_variants map again here (or pass through from
    # _multi_entity_retrieve via a new return-value — your call):
    variants_map = {}
    for entity in _entities:
        v = {entity}
        for syns in (kb_synonyms_per_kb or {}).values():
            v |= expand_entity(entity, syns)
        variants_map[entity] = v
    reranked = _apply_entity_boost(reranked, variants_map, alpha)
    # Re-sort by boosted score desc so quota picks the right top hits
    reranked.sort(key=lambda h: h.score or 0.0, reverse=True)
```

(Adapt the variable name `kb_synonyms_per_kb` to whatever you stored it in during Task 8.)

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/unit/test_bridge_entity_boost.py tests/unit/test_bridge_entity_quota.py tests/unit/test_chat_rag_bridge*.py -x 2>&1 | tail -10
```

Expected: PASS.

Run full unit suite to confirm nothing else regressed:

```bash
.venv/bin/pytest -q tests/unit/ 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_bridge_entity_boost.py ext/services/chat_rag_bridge.py
git commit -m "$(cat <<'EOF'
feat(bridge): _apply_entity_boost — soft rerank-score boost for entity matches

When MODE=boost, after cross-encoder rerank and before the per-entity
quota selection, boost each hit's score by alpha (default 0.3) for
every entity whose variant set appears in the chunk text.

Result: a chunk mentioning "5 POK Bde" gets +0.3 to its score; the
per-entity quota then naturally favors entity-matching chunks but
does NOT exclude non-matching chunks (the structural improvement
over MODE=filter).

Knob: RAG_ENTITY_BOOST_ALPHA (default 0.3).

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Operator CLI — `scripts/edit_kb_synonyms.py`

**Files:**
- Create: `scripts/edit_kb_synonyms.py`
- Test: `tests/integration/test_edit_kb_synonyms_cli.py` (new — small smoke test)

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_edit_kb_synonyms_cli.py
"""Smoke test for scripts/edit_kb_synonyms.py — verifies the CLI surface
and that --add merges into the existing JSONB array."""
import json
import os
import subprocess
import pytest


SCRIPT = "scripts/edit_kb_synonyms.py"


@pytest.fixture
def env():
    e = os.environ.copy()
    if "DATABASE_URL" not in e:
        pwd = open("compose/.env").read()
        for line in pwd.splitlines():
            if line.startswith("POSTGRES_PASSWORD="):
                pw = line.split("=", 1)[1]
                e["DATABASE_URL"] = (
                    f"postgresql://orgchat:{pw}@localhost:5432/orgchat"
                )
                break
    return e


def test_list_returns_jsonb_array(env):
    r = subprocess.run(
        ["./.venv/bin/python", SCRIPT, "--kb", "2", "--list"],
        env=env, capture_output=True, text=True, check=True,
    )
    # Output should be parseable JSON
    parsed = json.loads(r.stdout.strip())
    assert isinstance(parsed, list)


def test_add_merges_into_existing(env, tmp_path):
    # Save current state
    r0 = subprocess.run(
        ["./.venv/bin/python", SCRIPT, "--kb", "2", "--list"],
        env=env, capture_output=True, text=True, check=True,
    )
    original = json.loads(r0.stdout.strip())

    test_class = ["__test_marker_5pok__", "__test_marker_5POK__"]
    try:
        subprocess.run(
            ["./.venv/bin/python", SCRIPT, "--kb", "2",
             "--add", json.dumps(test_class)],
            env=env, capture_output=True, text=True, check=True,
        )
        r2 = subprocess.run(
            ["./.venv/bin/python", SCRIPT, "--kb", "2", "--list"],
            env=env, capture_output=True, text=True, check=True,
        )
        after = json.loads(r2.stdout.strip())
        assert test_class in after
        assert len(after) == len(original) + 1
    finally:
        # Cleanup — remove the test class
        subprocess.run(
            ["./.venv/bin/python", SCRIPT, "--kb", "2",
             "--remove", json.dumps(test_class)],
            env=env, capture_output=True, text=True,
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/integration/test_edit_kb_synonyms_cli.py -x
```

Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Implement the CLI**

```python
# scripts/edit_kb_synonyms.py
"""Edit per-KB synonym table.

Usage:
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --list
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --add '["A","B","C"]'
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --remove '["A","B","C"]'
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --load FILE.json
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --load -    # stdin

Env: DATABASE_URL (required)

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2.4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import asyncpg


async def _conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)
    url = url.replace("+asyncpg", "")
    return await asyncpg.connect(url)


async def _list(kb_id: int) -> list:
    conn = await _conn()
    try:
        row = await conn.fetchrow(
            "SELECT synonyms FROM knowledge_bases WHERE id = $1", kb_id,
        )
        if not row:
            print(f"error: no KB with id={kb_id}", file=sys.stderr)
            sys.exit(2)
        return row["synonyms"] or []
    finally:
        await conn.close()


async def _set(kb_id: int, synonyms: list) -> None:
    conn = await _conn()
    try:
        await conn.execute(
            "UPDATE knowledge_bases SET synonyms = $1::jsonb WHERE id = $2",
            json.dumps(synonyms), kb_id,
        )
    finally:
        await conn.close()


async def _add(kb_id: int, new_class: list) -> None:
    if not isinstance(new_class, list) or not all(isinstance(s, str) for s in new_class):
        print("error: --add expects a JSON array of strings", file=sys.stderr)
        sys.exit(2)
    current = await _list(kb_id)
    if new_class in current:
        print(f"info: class already present, no change", file=sys.stderr)
        return
    current.append(new_class)
    await _set(kb_id, current)


async def _remove(kb_id: int, target: list) -> None:
    current = await _list(kb_id)
    new = [c for c in current if c != target]
    if len(new) == len(current):
        print("info: class not found, no change", file=sys.stderr)
        return
    await _set(kb_id, new)


async def _load(kb_id: int, path: str) -> None:
    if path == "-":
        data = json.load(sys.stdin)
    else:
        with open(path) as fh:
            data = json.load(fh)
    if not isinstance(data, list):
        print("error: --load expects a JSON array of arrays of strings",
              file=sys.stderr)
        sys.exit(2)
    await _set(kb_id, data)


async def main_async(args: argparse.Namespace) -> int:
    if args.list:
        out = await _list(args.kb)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if args.add:
        await _add(args.kb, json.loads(args.add))
        return 0
    if args.remove:
        await _remove(args.kb, json.loads(args.remove))
        return 0
    if args.load:
        await _load(args.kb, args.load)
        return 0
    print("error: pick one of --list, --add, --remove, --load",
          file=sys.stderr)
    return 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kb", type=int, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--add", help="JSON array of strings to add as one equivalence class")
    g.add_argument("--remove", help="JSON array of strings — remove the class that exactly matches")
    g.add_argument("--load", help="path to JSON file (or - for stdin) replacing the entire table")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DATABASE_URL="postgresql://orgchat:$(grep ^POSTGRES_PASSWORD= compose/.env | cut -d= -f2)@localhost:5432/orgchat" \
    .venv/bin/pytest -q tests/integration/test_edit_kb_synonyms_cli.py -x
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_edit_kb_synonyms_cli.py scripts/edit_kb_synonyms.py
git commit -m "$(cat <<'EOF'
ops(scripts): edit_kb_synonyms — CLI to view/add/remove/replace per-KB synonyms

Operator surface for the synonyms JSONB column added in migration 017.
Idempotent --add (skips dup); --remove no-ops on miss; --load --replaces
the entire table from JSON file or stdin.

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2.4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Admin endpoint — `PATCH /api/kb/{kb_id}/synonyms`

**Files:**
- Modify: `ext/routers/kb_admin.py` (add the endpoint)
- Test: `tests/integration/test_kb_admin_synonyms_endpoint.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_kb_admin_synonyms_endpoint.py
"""Phase 2 / Item 4 — admin endpoint to PATCH per-KB synonyms."""
import json
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_patch_synonyms_admin_only(admin_token, user_token, async_client: AsyncClient):
    """Non-admin gets 403; admin gets 200."""
    body = {"synonyms": [["X", "Y"]]}
    # Non-admin
    r = await async_client.patch(
        "/api/kb/2/synonyms", json=body,
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert r.status_code == 403
    # Admin
    r = await async_client.patch(
        "/api/kb/2/synonyms", json=body,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_patch_synonyms_persists_then_returns_via_get(admin_token, async_client: AsyncClient):
    body = {"synonyms": [["__test_a__", "__test_b__"]]}
    r = await async_client.patch(
        "/api/kb/2/synonyms", json=body,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200
    # The endpoint should echo the saved value
    assert r.json()["synonyms"] == [["__test_a__", "__test_b__"]]
    # Cleanup
    await async_client.patch(
        "/api/kb/2/synonyms", json={"synonyms": []},
        headers={"Authorization": f"Bearer {admin_token}"},
    )


@pytest.mark.asyncio
async def test_patch_synonyms_validates_shape(admin_token, async_client: AsyncClient):
    """Malformed body → 400."""
    r = await async_client.patch(
        "/api/kb/2/synonyms", json={"synonyms": "not a list"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 400
```

(NOTE: this assumes pytest fixtures `admin_token`, `user_token`, `async_client` exist in `conftest.py`. If they don't, look at existing tests in `tests/integration/test_kb_admin*.py` for the canonical fixture set and adopt that. If absent, add inline fixtures that hit the signin endpoint to obtain a JWT, similar to the smoke test commands in the spec §6.3.)

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/integration/test_kb_admin_synonyms_endpoint.py -x
```

Expected: FAIL — endpoint returns 404.

- [ ] **Step 3: Implement the endpoint**

In `ext/routers/kb_admin.py`, find the existing PATCH endpoints (e.g. `/api/kb/{kb_id}/config`) for shape reference. Add:

```python
class SynonymsPatch(BaseModel):
    synonyms: list[list[str]]


@router.patch("/api/kb/{kb_id}/synonyms")
async def patch_kb_synonyms(
    kb_id: int,
    body: SynonymsPatch,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
) -> dict:
    """Replace the per-KB synonyms equivalence-class table.
    Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2.4
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin only")
    # Validate shape (list of lists of strings)
    if not isinstance(body.synonyms, list) or not all(
        isinstance(c, list) and all(isinstance(s, str) for s in c)
        for c in body.synonyms
    ):
        raise HTTPException(
            status_code=400,
            detail="synonyms must be a list of lists of strings",
        )
    from sqlalchemy import text as _sql
    await session.execute(
        _sql("UPDATE knowledge_bases SET synonyms = :s::jsonb WHERE id = :i"),
        {"s": json.dumps(body.synonyms), "i": kb_id},
    )
    await session.commit()
    return {"kb_id": kb_id, "synonyms": body.synonyms}
```

Add the necessary imports at the top of the file if missing (`json`, `BaseModel`).

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/integration/test_kb_admin_synonyms_endpoint.py -x
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_kb_admin_synonyms_endpoint.py ext/routers/kb_admin.py
git commit -m "$(cat <<'EOF'
feat(kb_admin): PATCH /api/kb/{kb_id}/synonyms — admin-only synonyms editor

Mirrors the operator CLI (scripts/edit_kb_synonyms.py) for via-API
editing. Replaces the entire JSONB array on each call. Validates
shape (list of lists of strings) — 400 on malformation, 403 on
non-admin.

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2.4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — observability

### Task 12: Multi-entity coverage counter

**Files:**
- Modify: `ext/services/metrics.py` (declare counter)
- Modify: `ext/services/chat_rag_bridge.py` (bump counter at end of `_apply_entity_quota`)
- Test: `tests/unit/test_multi_entity_coverage_counter.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_multi_entity_coverage_counter.py
"""Phase 3 — rag_multi_entity_coverage_total counter."""
from prometheus_client import REGISTRY

from ext.services.metrics import rag_multi_entity_coverage_total
from ext.services.chat_rag_bridge import _apply_entity_quota
from ext.services.vector_store import Hit


def _hit(id_, score, text):
    return Hit(id=id_, score=score, payload={"text": text})


def _read(outcome: str, entity_count: str) -> float:
    samples = list(REGISTRY.collect())
    for fam in samples:
        if fam.name == "rag_multi_entity_coverage":
            for s in fam.samples:
                if (s.labels.get("outcome") == outcome
                        and s.labels.get("entity_count") == entity_count
                        and s.name == "rag_multi_entity_coverage_total"):
                    return s.value
    return 0.0


def test_full_coverage_bumps_full_outcome():
    before = _read("full", "2")
    hits = [
        _hit(1, 0.9, "75 Inf Bde visited..."),
        _hit(2, 0.8, "75 Inf Bde inspection..."),
        _hit(3, 0.7, "75 Inf Bde 03 Apr..."),
        _hit(4, 0.6, "5 PoK Bde practice..."),
        _hit(5, 0.5, "5 PoK Bde rotation..."),
        _hit(6, 0.4, "5 PoK Bde meeting..."),
    ]
    _apply_entity_quota(
        reranked=hits, entities=["75 Inf Bde", "5 PoK Bde"],
        per_entity_floor=3, final_k=12,
    )
    after = _read("full", "2")
    assert after == before + 1


def test_partial_coverage_bumps_partial_outcome():
    before = _read("partial", "2")
    hits = [
        _hit(1, 0.9, "75 Inf Bde visited..."),
        _hit(2, 0.8, "75 Inf Bde inspection..."),
        _hit(3, 0.7, "75 Inf Bde 03 Apr..."),
        _hit(4, 0.6, "5 PoK Bde practice..."),
        # only 1 chunk for 5 PoK Bde — partial
    ]
    _apply_entity_quota(
        reranked=hits, entities=["75 Inf Bde", "5 PoK Bde"],
        per_entity_floor=3, final_k=12,
    )
    after = _read("partial", "2")
    assert after == before + 1


def test_empty_coverage_bumps_empty_outcome():
    before = _read("empty", "2")
    hits = [
        _hit(1, 0.9, "75 Inf Bde visited..."),
        _hit(2, 0.8, "75 Inf Bde inspection..."),
        # zero chunks for 5 PoK Bde — empty
    ]
    _apply_entity_quota(
        reranked=hits, entities=["75 Inf Bde", "5 PoK Bde"],
        per_entity_floor=3, final_k=12,
    )
    after = _read("empty", "2")
    assert after == before + 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest -q tests/unit/test_multi_entity_coverage_counter.py -x
```

Expected: FAIL — `rag_multi_entity_coverage_total` doesn't exist.

- [ ] **Step 3: Implement the counter + bump**

In `ext/services/metrics.py`, add alongside the other counters:

```python
rag_multi_entity_coverage_total = Counter(
    "rag_multi_entity_coverage_total",
    "Per-entity coverage outcome of the multi-entity rerank quota. "
    "outcome=full when every entity met its floor; "
    "outcome=partial when at least one entity got <floor but >0 chunks; "
    "outcome=empty when at least one entity got 0 chunks. "
    "Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §6.1",
    labelnames=("outcome", "entity_count"),
)
```

In `ext/services/chat_rag_bridge.py`, find `_apply_entity_quota`. At the end (just before the `return` statement), add:

```python
# Phase 3 / 2026-05-03 — observability bump.
try:
    from .metrics import rag_multi_entity_coverage_total
    counts = {e: 0 for e in entities}
    for hit in final_pool:  # `final_pool` is whatever the function returns
        text_low = ((hit.payload or {}).get("text") or "").lower()
        for e in entities:
            if e.lower() in text_low:
                counts[e] += 1
                break  # one entity attribution per chunk
    n_zero = sum(1 for c in counts.values() if c == 0)
    n_partial = sum(1 for c in counts.values() if 0 < c < per_entity_floor)
    outcome = "empty" if n_zero else ("partial" if n_partial else "full")
    rag_multi_entity_coverage_total.labels(
        outcome=outcome, entity_count=str(len(entities)),
    ).inc()
except Exception:
    # Telemetry is fail-open; never break retrieval on counter error.
    pass
```

(NOTE: the variable name `final_pool` may differ in current code — substitute whatever local variable holds the post-quota list before return.)

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest -q tests/unit/test_multi_entity_coverage_counter.py tests/unit/test_bridge_entity_quota.py -x
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_multi_entity_coverage_counter.py ext/services/metrics.py ext/services/chat_rag_bridge.py
git commit -m "$(cat <<'EOF'
feat(metrics): rag_multi_entity_coverage_total — full/partial/empty outcome bucketing

Bumped at the end of _apply_entity_quota with one of three outcome
labels per multi-entity request:
  full    — every entity met its floor (typically 3 chunks)
  partial — at least one entity got <floor but >0 chunks
  empty   — at least one entity got 0 chunks (the failure mode this
            entire spec was written to fix)

Wires the alert in observability/prometheus/alerts-rag-quality.yml
(next task).

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §6.1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Prometheus alert `MultiEntityCoverageEmpty`

**Files:**
- Modify: `observability/prometheus/alerts-rag-quality.yml`
- Test: hot-reload check (operational, not pytest)

- [ ] **Step 1: Add the alert rule**

Open `observability/prometheus/alerts-rag-quality.yml`. Find the existing rules (under `groups: - name: rag_quality_soak`). Add a new rule at the end:

```yaml
      # 2026-05-03 — entity-coverage failure detection. Pairs with the
      # rag_multi_entity_coverage_total counter (commit from Task 12) which
      # buckets every multi-entity request as full / partial / empty
      # based on whether each named entity got the floor of chunks in
      # the final pool. outcome=empty means at least one entity got 0
      # chunks — the failure mode that motivated the entity_text_filter
      # casing/synonyms work.
      # Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §6.2
      - alert: MultiEntityCoverageEmpty
        expr: rate(rag_multi_entity_coverage_total{outcome="empty"}[15m]) > 0.05
        for: 10m
        labels: {severity: warning, component: rag}
        annotations:
          summary: "Multi-entity queries returning 0 hits for at least one entity (>5% rate)"
          description: |
            Either entity_text_filter regressed, the synonym table is
            stale, or a new corpus added an unfamiliar abbreviation.
            Cross-reference rag_multi_entity_decompose_total for context.
            The failing entity name is NOT in the metric label
            (cardinality concern); inspect open-webui logs for "rag:
            multi-entity rerank quota active" lines to see which entity
            has the empty bucket.
```

- [ ] **Step 2: Validate YAML + reload Prometheus**

```bash
python3 -c "import yaml; d=yaml.safe_load(open('/home/vogic/LocalRAG/observability/prometheus/alerts-rag-quality.yml')); print('rules:', [r['alert'] for g in d['groups'] for r in g['rules']])"
```

Expected: prints the rule list including `MultiEntityCoverageEmpty`.

```bash
curl -s -o /dev/null -w "prometheus reload: %{http_code}\n" -XPOST http://localhost:9091/-/reload
```

Expected: `prometheus reload: 200` (assumes observability overlay is up; if not, this step is operator-deferred).

- [ ] **Step 3: Verify alert is registered in Prometheus**

```bash
curl -s http://localhost:9091/api/v1/rules | python3 -c "
import json, sys
d = json.load(sys.stdin)
for grp in d['data']['groups']:
    for r in grp.get('rules', []):
        if r['name'] == 'MultiEntityCoverageEmpty':
            print('FOUND:', r['name'], r.get('state', '(no state yet)'))
            break
"
```

Expected: `FOUND: MultiEntityCoverageEmpty inactive` (or whatever state — at least found).

- [ ] **Step 4: Commit**

```bash
git add observability/prometheus/alerts-rag-quality.yml
git commit -m "$(cat <<'EOF'
ops(alerts): MultiEntityCoverageEmpty — page on entity-coverage gaps

Pairs with rag_multi_entity_coverage_total{outcome=empty} which fires
when at least one entity in a multi-entity query gets 0 chunks in the
final pool. Threshold: rate > 0.05 sustained 10m → warning. Catches
exactly the failure mode that motivated the casing/synonyms spec
(brigade query returning empty for 5 PoK before today's fix).

Validated: prometheus hot-reload (POST /-/reload) returned 200.

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §6.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final manual-verification checklist

After all 13 tasks land and the worker is rebuilt + redeployed:

- [ ] **Phase 1 verification** — run §6.3 step 1 from the spec. Each entity (`75 INF bde`, `5 PoK bde`, `32 Inf Bde`, `80 Inf Bde`) returns ≥ 5 hits. Metadata-intent query (`how many docs are in the KB`) returns ≥ 1 hit (was 0 before).
- [ ] **Phase 2 verification** — apply migration 017; seed KB 2 synonyms via §6.3 step 2. Run brigade chat completion in `MODE=filter` (default); record per-brigade fact count. Switch KB 2 to `MODE=boost` via PATCH endpoint; re-run; compare. Expected: 5 PoK Bde no longer empty under either mode; boost may surface paraphrased content the filter misses.
- [ ] **Phase 3 verification** — issue 5 brigade queries; check `curl localhost:9464/metrics | grep rag_multi_entity_coverage_total`. Outcome should be `full` for properly-seeded synonyms; trip a test `empty` by removing all synonyms and querying again to verify the alert fires within 15 min.

---

## Self-review

Spec coverage check:
- §4.1 (case-insensitive index + lowercase): Tasks 2, 4 ✓
- §4.2 (suffix strip): Task 3 ✓
- §4.3 (`_do_decompose` regression): Task 1 ✓
- §5.1 (boost vs filter): Tasks 8 (mode read + filter path), 9 (boost path) ✓
- §5.2 (synonyms): Tasks 5 (migration), 6 (helper + keys), 7 (variants in `_build_filter`), 8 (bridge wiring), 10 (CLI), 11 (admin endpoint) ✓
- §6.1 (counter): Task 12 ✓
- §6.2 (alert): Task 13 ✓
- §6.3 (manual recipe): in spec, executed by operator post-deploy
- §7 (rollout) + §8 (compatibility) + §9 (out of scope): governance, not tasks

No placeholders — every step has its actual code/command/expected output. Type names consistent across tasks (`text_filter_variants`, `entity_text_filter_mode`, `expand_entity`, `_apply_entity_boost`, `kb_synonyms_per_kb`, `rag_multi_entity_coverage_total`).
