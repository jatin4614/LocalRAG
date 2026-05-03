"""Per-KB ``rag_config`` resolution.

Each ``knowledge_bases`` row carries an optional ``rag_config`` JSONB
column (see migration 006) where admins stamp retrieval-quality
preferences — e.g. "this KB is huge, always rerank with a window=2
context expansion". At request time the bridge collects the configs of
all KBs the chat selected, merges them into a single override dict, and
wraps the retrieval pipeline in ``flags.with_overrides(...)`` so each
stage reads the correct values without touching ``os.environ`` (shared
process state, would leak across concurrent requests).

Multi-KB merge policy (UNION / MAX — strictest wins)
----------------------------------------------------
If the user selects multiple KBs for one chat, the merge picks whichever
setting runs the HEAVIER pipeline. The reasoning:

* Rerank/MMR/expand are opt-in quality boosts. If ANY selected KB needs
  rerank (``true``), the chat runs rerank — we can't re-rank only half
  the candidates.
* For numeric thresholds (``context_expand_window``, ``mmr_lambda``,
  ``rerank_top_k``) the larger value wins so the most-demanding KB's
  budget is honoured. The strictest KB effectively "pulls up" the rest.
* Spotlight / semcache: same union rule; treat ``true`` as dominant.

This is conservative on latency (you pay the most-expensive KB's cost)
but correct on quality (no silent downgrade when a small-KB-specific
fast path would otherwise suppress a heavier KB's preference).

Valid keys (anything else is silently dropped — admin UI should refuse
to set unknown keys, but we belt-and-brace at the service layer too)::

    rerank                bool    → RAG_RERANK
    rerank_top_k          int     → RAG_RERANK_TOP_K
    mmr                   bool    → RAG_MMR
    mmr_lambda            float   → RAG_MMR_LAMBDA
    context_expand        bool    → RAG_CONTEXT_EXPAND
    context_expand_window int     → RAG_CONTEXT_EXPAND_WINDOW
    spotlight             bool    → RAG_SPOTLIGHT
    semcache              bool    → RAG_SEMCACHE
    contextualize_on_ingest bool  → RAG_CONTEXTUALIZE_KBS  (ingest-side;
                                     recorded but NOT applied via the
                                     flag overlay — the overlay is
                                     request-scoped and ingest is a
                                     separate process path)
    hyde                  bool    → RAG_HYDE   (P3.3 — enable Hypothetical
                                     Document Embeddings per KB)
    hyde_n                int     → RAG_HYDE_N (number of hypothetical
                                     excerpts to average; default 1)

An empty config ``{}`` means "inherit process defaults" (no overlay
entry emitted — ``flags.get`` falls through to ``os.environ``).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

# Whitelist — any other key is filtered out. Keep this list in sync with
# the PATCH /api/kb/{kb_id}/config validator and the docs in
# ``docs/rag-per-kb-config.md``.
VALID_BOOL_KEYS = frozenset({
    "rerank",
    "mmr",
    "context_expand",
    "spotlight",
    "semcache",
    "contextualize_on_ingest",
    # Phase 3.3: short-form per-KB ingest gate consumed directly by
    # ``ingest.should_contextualize``. We keep ``contextualize_on_ingest``
    # accepted (legacy) but the new code path reads ``contextualize`` —
    # explicit ``True``/``False`` here overrides ``RAG_CONTEXTUALIZE_KBS``
    # in either direction (e.g. global ON, KB OFF → skip).
    "contextualize",
    # P3.3: HyDE (Hypothetical Document Embeddings) — per-KB override so a
    # year-long, abstract-query-heavy KB can opt in without flipping the
    # global RAG_HYDE process flag.
    "hyde",
    # Tier 1/2 (doc-summary index + intent router). ``doc_summaries`` is
    # ingest-only (it controls whether ingest emits a per-doc summary
    # point); the other two are request-scope flags read by the bridge.
    "doc_summaries",
    "intent_routing",
    "intent_llm",
    # Phase 6.X (multi-entity decomposition — Method 3). When set,
    # the bridge fans out a multi-entity query into N parallel
    # retrievals (one per named entity) and merges with a per-entity
    # quota. Default off; opt in per-KB for corpora with multi-entity
    # query shapes (military reports listing N brigades, contracts
    # citing N parties, etc.).
    "multi_entity_decompose",
    # Phase 6.X (per-entity Qdrant text filter — Method 4). When set,
    # each per-entity sub-query in the decompose fan-out adds a
    # ``must.match.text`` filter on the entity name. Hard precision
    # boost: chunks not literally naming the entity are excluded.
    # Pairs naturally with multi_entity_decompose; can be off if you
    # want decomposition without the lexical hard-cut.
    "entity_text_filter",
    # Phase 6.X (QU LLM entity extraction — Method 5). When set, the
    # bridge prefers ``HybridClassification.entities`` from the QU LLM
    # (already populated as part of intent classification — no second
    # LLM call). Off → regex extractor only. Soft-falls to regex on
    # any QU side error.
    "qu_entity_extract",
    # Phase 6.X — per-KB image caption extraction (ingest-only).
    # When True, the ingest path calls the vision LLM for every image
    # in PDFs and emits ``chunk_type="image_caption"`` chunks. When
    # False, all image extraction is suppressed for this KB even if
    # the global ``RAG_IMAGE_CAPTIONS=1``. Useful for text-only KBs
    # (security policies with org charts but no critical diagrams)
    # where the per-image LLM round-trips slow ingest 5-10× and
    # routinely OOM the worker. Per-KB explicit value (True or False)
    # wins over the env flag.
    "image_captions",
})
VALID_INT_KEYS = frozenset({
    # Pre-rerank pull cap. Overrides the intent-driven _per_kb default
    # (10 / 30 / 50 for default / specific_date / global). Use to widen
    # recall on KBs with many entities-per-query (military reports
    # listing 4 brigades; legal contracts citing N parties) where the
    # default 10-30 starves low-frequency entities into top-12 eviction.
    # MAX merge across selected KBs.
    "top_k",
    "rerank_top_k",
    "context_expand_window",
    # P3.3: number of hypothetical-doc generations to average. Higher N
    # improves retrieval quality on abstract queries at the cost of N
    # extra chat calls (parallel, so wall-time cost is roughly constant).
    "hyde_n",
    # Phase 1a / per-KB chunking. Ingest-only (not propagated to the flag
    # overlay) — see INGEST_ONLY_KEYS below. Corpora differ wildly: daily
    # fact-heavy reports want ~200-300 tokens, whitepapers ~600-800.
    # Bounds enforced in validate_config.
    "chunk_tokens",
    "overlap_tokens",
    # Phase 6.X (multi-entity decomposition — Method 3). Per-entity
    # quota floor in ``merge_with_quota``. Each entity is guaranteed
    # at least this many hits in the merged candidate set (or all of
    # its hits, whichever is fewer). Default 10. Bounds [1, 50] —
    # higher than 50 starts crowding out single-entity recall.
    "multi_entity_min_per_entity",
})
VALID_FLOAT_KEYS = frozenset({
    "mmr_lambda",
})
# Phase 6.X — string-typed enum keys. ``chunking_strategy`` is the only
# member today; the validator enforces it's one of
# ``_VALID_CHUNKING_STRATEGIES`` and silently drops anything else.
# Keep this small — string sprawl in JSONB is the road to schema rot.
# 2026-05-03 — Phase 2 / Item 4. ``entity_text_filter_mode`` added:
# "filter" | "boost" controls whether entity_text_filter hard-excludes
# non-matching chunks or applies a Python-side rerank-score boost.
VALID_STRING_KEYS = frozenset({
    "chunking_strategy",
    "entity_text_filter_mode",  # NEW: "filter" | "boost"
})
# 2026-05-03 — Phase 2 / Item 4. ``synonyms`` is a list of equivalence
# classes for entity-name expansion. Used by future query-rewrite work;
# the column is in DB already (migration 017) and the per-KB key threads
# it through. No env var — consumed via per-request payload.
VALID_LIST_KEYS = frozenset({
    "synonyms",
})
VALID_KEYS = (
    VALID_BOOL_KEYS | VALID_INT_KEYS | VALID_FLOAT_KEYS
    | VALID_STRING_KEYS | VALID_LIST_KEYS
)

# Keys that are NOT propagated into the request-scope overlay because the
# underlying flag is read by the ingest process, not the retrieval hot
# path. We still accept them in rag_config (so admin UIs can stamp them
# for future ingest runs) but they do not influence a live chat request.
INGEST_ONLY_KEYS = frozenset({
    "contextualize_on_ingest",
    # Phase 3.3 short-form alias — ingest-only, consumed by
    # ``ingest.should_contextualize``. Stripping it from the overlay
    # keeps the request hot path free of an env var that wouldn't
    # influence retrieval anyway and avoids confusing operators reading
    # the overlay in logs.
    "contextualize",
    "chunk_tokens",
    "overlap_tokens",
    # doc_summaries affects what ingest emits (per-doc summary points),
    # not request-time routing — strip it from the flag overlay so it
    # doesn't leak into the retrieval hot path.
    "doc_summaries",
    # Phase 6.6 — read directly by ``ingest.chunk_text_for_kb`` via
    # ``get_chunking_strategy``. Not propagated to the flag overlay
    # because the request hot path never branches on it.
    "chunking_strategy",
    # Phase 6.X — read by ``ingest.should_caption_images``. Ingest-side
    # gate; the request hot path doesn't branch on it.
    "image_captions",
})

# Mapping from JSON config key -> RAG_* env var name. Values are stringified
# at overlay time because ``flags.get`` / ``os.environ.get`` return strings.
_KEY_TO_ENV: dict[str, str] = {
    "rerank": "RAG_RERANK",
    "rerank_top_k": "RAG_RERANK_TOP_K",
    "top_k": "RAG_TOP_K",
    "mmr": "RAG_MMR",
    "mmr_lambda": "RAG_MMR_LAMBDA",
    "context_expand": "RAG_CONTEXT_EXPAND",
    "context_expand_window": "RAG_CONTEXT_EXPAND_WINDOW",
    "spotlight": "RAG_SPOTLIGHT",
    "semcache": "RAG_SEMCACHE",
    "contextualize_on_ingest": "RAG_CONTEXTUALIZE_KBS",
    "hyde": "RAG_HYDE",
    "hyde_n": "RAG_HYDE_N",
    "doc_summaries": "RAG_DOC_SUMMARIES",
    # Phase 6.X — multi-entity decomposition / text filter / QU
    # entity extract (Methods 3, 4, 5). All four keys are
    # request-scoped: the bridge reads them via ``flags.get(...)``
    # before deciding whether to fan out and how many sub-query hits
    # each entity is guaranteed.
    "multi_entity_decompose": "RAG_MULTI_ENTITY_DECOMPOSE",
    "entity_text_filter": "RAG_ENTITY_TEXT_FILTER",
    "qu_entity_extract": "RAG_QU_ENTITY_EXTRACT",
    "multi_entity_min_per_entity": "RAG_MULTI_ENTITY_MIN_PER_ENTITY",
    # 2026-05-03 — Phase 2 / Item 4. entity_text_filter mode toggle.
    "entity_text_filter_mode": "RAG_ENTITY_TEXT_FILTER_MODE",
    # The QU LLM flags (RAG_QU_*) are cluster-wide and not exposed as
    # per-KB rag_config keys.
}
# Note: `synonyms` is intentionally NOT in _KEY_TO_ENV. The synonym
# table has no env-level analogue (it's per-KB only); kb_config.py
# threads it via the merged dict, not via the request-scoped env overlay.


def validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Strip unknown keys and coerce values to the expected Python type.

    Returns a fresh dict containing only whitelisted keys with coerced
    values. Does not raise on bad input — unknown keys are dropped,
    un-coerceable values are dropped. Callers that want hard validation
    should check that the returned dict has the same keys as the input.

    Admin PATCH handlers call this before persisting so a caller can't
    sneak arbitrary JSON into the JSONB column.
    """
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in VALID_KEYS:
            continue
        if key in VALID_BOOL_KEYS:
            if isinstance(value, bool):
                out[key] = value
            elif isinstance(value, str):
                out[key] = value.lower() in ("1", "true", "yes", "on")
            elif isinstance(value, (int, float)):
                out[key] = bool(value)
            # else: drop
        elif key in VALID_INT_KEYS:
            try:
                coerced = int(value)
            except (TypeError, ValueError):
                continue
            # Per-KB chunk-size bounds. 100 floor avoids over-fragmentation;
            # 2000 ceiling keeps chunks below typical embedder max_seq_len.
            # overlap_tokens must be strictly less than chunk_tokens and
            # non-negative. Values outside bounds are dropped silently so
            # the KB inherits process defaults rather than ingesting at a
            # corrupt size.
            # Wave 2 (review §2.7): RAG_CHUNK_MAX_TOKENS env-driven ceiling.
            # bge-m3 supports up to 8192; the historical 2000 cap was tied to
            # cl100k-tokenizer / older models. Operators can raise this when
            # ingesting long-form docs that benefit from larger chunks.
            import os as _os_local
            try:
                _chunk_max = int(_os_local.environ.get("RAG_CHUNK_MAX_TOKENS", "2000"))
            except (TypeError, ValueError):
                _chunk_max = 2000
            if key == "chunk_tokens" and not (100 <= coerced <= _chunk_max):
                continue
            if key == "overlap_tokens":
                # Cross-field check is handled at apply time (ingest reads
                # both); at validate time we only reject absurd values.
                if coerced < 0 or coerced > 1000:
                    continue
            # H5 bounds for retrieval-side knobs. Out-of-range silently
            # drops the key so the KB inherits process defaults instead
            # of running with a corrupt config.
            if key == "rerank_top_k" and not (1 <= coerced <= 1000):
                continue
            # top_k caps the pre-rerank pull. 200 ceiling matches the
            # heaviest realistic budget — anything more and the rerank
            # cross-encoder dominates per-request latency without
            # measurably improving recall.
            if key == "top_k" and not (1 <= coerced <= 200):
                continue
            if key == "context_expand_window" and not (0 <= coerced <= 100):
                continue
            if key == "hyde_n" and not (1 <= coerced <= 10):
                continue
            # Phase 6.X — multi-entity per-entity floor. Below 1 has no
            # meaning (no quota); above 50 starts crowding out other
            # signal at the rerank cut. Out-of-range silently drops.
            if key == "multi_entity_min_per_entity" and not (1 <= coerced <= 50):
                continue
            out[key] = coerced
        elif key in VALID_FLOAT_KEYS:
            try:
                coerced_float = float(value)
            except (TypeError, ValueError):
                continue
            # H5: mmr_lambda is a convex-combination weight; values
            # outside [0, 1] don't make sense and would skew the
            # reranker's diversity-vs-relevance balance.
            if key == "mmr_lambda" and not (0.0 <= coerced_float <= 1.0):
                continue
            out[key] = coerced_float
        elif key in VALID_STRING_KEYS:
            if not isinstance(value, str):
                continue
            coerced_str = value.lower().strip()
            # Enum check per key. Unknown values silently drop so a
            # hand-edited row never silently switches behaviour on an
            # admin who didn't expect it.
            if key == "chunking_strategy":
                if coerced_str not in _VALID_CHUNKING_STRATEGIES:
                    continue
            elif key == "entity_text_filter_mode":
                # 2026-05-03 — Phase 2 / Item 4.
                if coerced_str not in _VALID_ENTITY_TEXT_FILTER_MODES:
                    continue
            out[key] = coerced_str
        elif key in VALID_LIST_KEYS:
            # 2026-05-03 — Phase 2 / Item 4. synonyms — list of lists of
            # strings; drop on any malformation so a bad row never reaches
            # the retrieval hot path.
            if not isinstance(value, list):
                continue
            if not all(
                isinstance(cls, list) and all(isinstance(s, str) for s in cls)
                for cls in value
            ):
                continue
            out[key] = value
    return out


def merge_configs(configs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge per-KB configs into a single effective config.

    Policy:
      * Booleans: ``any(...)``  (union — if any KB wants it, enable it)
      * Ints:     ``max(...)``  (strictest — largest window wins)
      * Floats:   ``max(...)``  (same reasoning — higher MMR lambda
                                 biases relevance over diversity; we pick
                                 the value closer to the "pure relevance"
                                 extreme to be conservative)

    Each input dict is ``validate_config``-cleaned first so unknown/
    malformed keys are dropped before the merge. Empty/None inputs are
    skipped (treated as "no opinion").

    Returns the merged config. An empty merge (no non-empty inputs) is
    an empty dict — the caller interprets this as "use process defaults".
    """
    merged: dict[str, Any] = {}
    for raw in configs:
        if not raw:
            continue
        cleaned = validate_config(raw)
        for key, value in cleaned.items():
            if key in VALID_BOOL_KEYS:
                merged[key] = bool(merged.get(key, False)) or bool(value)
            elif key in VALID_INT_KEYS:
                prev = merged.get(key)
                merged[key] = max(int(value), int(prev)) if prev is not None else int(value)
            elif key in VALID_FLOAT_KEYS:
                prev = merged.get(key)
                merged[key] = max(float(value), float(prev)) if prev is not None else float(value)
            elif key in VALID_STRING_KEYS:
                prev = merged.get(key)
                # ``chunking_strategy``: ``"structured"`` beats ``"window"`` so a
                # KB that explicitly opted into structured chunking isn't
                # silently downgraded by another KB stuck on the default.
                # Other future string keys: first non-empty wins.
                if key == "chunking_strategy" and value == "structured":
                    merged[key] = value
                elif prev is None:
                    merged[key] = value
            elif key in VALID_LIST_KEYS:
                # 2026-05-03 — Phase 2 / Item 4. Synonyms: concatenate all
                # equivalence-class lists across selected KBs. Additive —
                # if KB-A defines brigade variants and KB-B defines unit
                # variants, a multi-KB query should see both tables.
                # Duplicates are harmless: expand_entity returns a set so
                # identical members across classes are naturally deduped at
                # call time. Simpler than any class-merging algorithm and
                # avoids the head-based dedup bug (see review of 5c3c6ae).
                merged[key] = merged.get(key, []) + list(value)
    return merged


def config_to_env_overrides(merged: Mapping[str, Any]) -> dict[str, str]:
    """Convert a merged config dict into ``{RAG_*: "string"}`` for the flag overlay.

    Booleans serialize to ``"1"``/``"0"`` — matching the existing env-var
    conventions read by retriever.py, reranker.py, etc. Ints/floats are
    ``str(...)``-converted. Unknown keys (shouldn't happen after
    ``validate_config``) are dropped defensively.

    Ingest-only keys (``contextualize_on_ingest``) are skipped: the flag
    overlay is request-scoped, so setting it would have no effect and
    would confuse operators reading the overlay in logs.
    """
    out: dict[str, str] = {}
    for key, value in merged.items():
        if key in INGEST_ONLY_KEYS:
            continue
        env = _KEY_TO_ENV.get(key)
        if env is None:
            continue
        if isinstance(value, bool):
            out[env] = "1" if value else "0"
        else:
            out[env] = str(value)
    return out


def resolve_chunk_params(
    raw_config: Mapping[str, Any] | None,
    *,
    env_chunk_size: int | None = None,
    env_chunk_overlap: int | None = None,
) -> tuple[int, int]:
    """Resolve ``(chunk_tokens, overlap_tokens)`` for a KB ingest call.

    Priority, highest wins:

    1. ``raw_config`` values (after ``validate_config`` cleans them), if the
       KB explicitly stamped ``chunk_tokens`` / ``overlap_tokens``.
    2. Process env defaults ``CHUNK_SIZE`` / ``CHUNK_OVERLAP``.
    3. Hard defaults ``800`` / ``100`` (matches the ``ingest_bytes``
       signature defaults so behaviour is unchanged for KBs without the
       keys and no env override).

    The two args are resolved independently — a KB that sets only
    ``chunk_tokens`` inherits the env overlap, and vice versa.

    Fail-closed: overlap that would exceed ``chunk_tokens // 2`` (the
    chunker's hard contract) is clipped down to ``chunk_tokens // 4`` so
    the call is still valid. A caller should have run ``validate_config``
    already so we never hit that path in practice — this is a safety net
    against hand-edited JSONB.
    """
    import os as _os

    cfg: Mapping[str, Any] = validate_config(raw_config) if raw_config else {}

    if env_chunk_size is None:
        try:
            env_chunk_size = int(_os.environ.get("CHUNK_SIZE", "800"))
        except (TypeError, ValueError):
            env_chunk_size = 800
    if env_chunk_overlap is None:
        try:
            env_chunk_overlap = int(_os.environ.get("CHUNK_OVERLAP", "100"))
        except (TypeError, ValueError):
            env_chunk_overlap = 100

    # Wave 2 (review §2.6): use `if "key" in cfg` precedence — `cfg.get(key)
    # or env_default` falls through to env when the operator legitimately set
    # 0 (overlap_tokens=0 is a valid value). Surprising bug if the operator
    # explicitly disables overlap and the env default re-enables it.
    chunk_tokens = (
        int(cfg["chunk_tokens"]) if "chunk_tokens" in cfg else env_chunk_size
    )
    overlap_tokens = (
        int(cfg["overlap_tokens"]) if "overlap_tokens" in cfg else env_chunk_overlap
    )

    # Safety clip — chunker raises if overlap >= chunk_tokens. Clip down
    # rather than raise so one bad row doesn't block a whole reingest.
    if overlap_tokens >= chunk_tokens:
        overlap_tokens = max(0, chunk_tokens // 4)

    return chunk_tokens, overlap_tokens


def with_overrides(overrides: Mapping[str, str]):
    """Re-export of ``ext.services.flags.with_overrides`` for callers that
    only need the KB-config surface.

    This is a courtesy alias — the bridge imports both ``merge_configs``
    and ``with_overrides`` from ``kb_config`` for locality. The actual
    implementation lives in ``flags.py``.
    """
    from .flags import with_overrides as _with_overrides
    return _with_overrides(overrides)


_VALID_CHUNKING_STRATEGIES = ("window", "structured")
# 2026-05-03 — Phase 2 / Item 4. Valid values for ``entity_text_filter_mode``.
# "filter" → Qdrant hard exclusion (original behaviour).
# "boost"  → Python-side rerank-score boost (experimental, non-default).
# Default code path is "filter off entirely" per validation results; this
# constant exists for per-KB operator use and future rollback.
_VALID_ENTITY_TEXT_FILTER_MODES = ("filter", "boost")


def get_chunking_strategy(rag_config: dict | None) -> str:
    """Return 'window' (default) or 'structured'.

    Plan B Phase 6.6. Reads ``chunking_strategy`` from a KB's
    ``rag_config`` JSONB blob (migration 010). Unknown values fall back
    to ``"window"`` so a hand-edited row never silently switches the
    chunker on an admin who didn't expect it.
    """
    if not rag_config:
        return "window"
    raw = (rag_config.get("chunking_strategy") or "window").lower().strip()
    if raw not in _VALID_CHUNKING_STRATEGIES:
        return "window"
    return raw


def expand_entity(entity: str, classes: list[list[str]] | None) -> set[str]:
    """Return entity + every equivalence-class member that contains it.

    Case-insensitive membership check. If ``entity`` is found in any class
    in ``classes``, the full class is merged into the output set.  If it
    appears in no class, only the original string is returned.

    Args:
        entity: The entity name as typed by the user (any casing).
        classes: Per-KB synonym table — a list of equivalence classes, each
                 class being a list of canonical + variant strings. Typically
                 sourced from ``rag_config["synonyms"]`` after
                 ``validate_config``.  ``None`` or ``[]`` is safe.

    Returns:
        A set containing ``entity`` plus every variant from matching classes.

    Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2.2
    """
    out: set[str] = {entity}
    if not classes or not entity:
        return out
    e_low = entity.lower()
    for cls in classes:
        if any(v.lower() == e_low for v in cls):
            out.update(cls)
    return out


def get_ocr_policy(kb_id: int, db_session) -> dict | None:
    """Return the per-KB OCR policy or None if disabled.

    Plan B Phase 6.3. Reads from the ``ocr_policy`` column added in
    migration 011. Returns None when the KB doesn't exist OR when the
    policy explicitly disables OCR (``enabled=False``). Otherwise
    returns the policy dict so the ingest path can pick a backend +
    language.
    """
    from ..db.models import KnowledgeBase
    kb = db_session.query(KnowledgeBase).filter_by(id=kb_id).first()
    if not kb:
        return None
    policy = kb.ocr_policy or {}
    if not policy.get("enabled", True):
        return None
    return policy


__all__ = [
    "VALID_KEYS",
    "VALID_LIST_KEYS",
    "merge_configs",
    "config_to_env_overrides",
    "validate_config",
    "resolve_chunk_params",
    "with_overrides",
    "expand_entity",
    "get_ocr_policy",
    "get_chunking_strategy",
]
