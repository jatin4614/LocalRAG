"""Tests for temporal_raptor.build_temporal_tree + helpers.

Plan B Phase 5.5.
"""
from __future__ import annotations

import pytest

from ext.services.temporal_raptor import (
    build_meta_prompt,
    build_quarter_prompt,
    build_temporal_tree,
    build_year_prompt,
    group_chunks_by_shard_key,
    quarter_for_shard_key,
)


def test_quarter_for_shard_key():
    assert quarter_for_shard_key("2026-01") == ("2026-Q1", 2026, 1)
    assert quarter_for_shard_key("2026-04") == ("2026-Q2", 2026, 2)
    assert quarter_for_shard_key("2026-07") == ("2026-Q3", 2026, 3)
    assert quarter_for_shard_key("2026-10") == ("2026-Q4", 2026, 4)


def test_group_chunks_by_shard_key():
    chunks = [
        {"text": "a", "shard_key": "2026-01", "doc_id": 1, "chunk_index": 0},
        {"text": "b", "shard_key": "2026-01", "doc_id": 2, "chunk_index": 0},
        {"text": "c", "shard_key": "2026-02", "doc_id": 3, "chunk_index": 0},
    ]
    grouped = group_chunks_by_shard_key(chunks)
    assert set(grouped.keys()) == {"2026-01", "2026-02"}
    assert len(grouped["2026-01"]) == 2
    assert len(grouped["2026-02"]) == 1


def test_quarter_prompt_includes_change_vs_prior():
    prompt = build_quarter_prompt(
        quarter_label="2026-Q2",
        month_summaries=["April: ...", "May: ...", "June: ..."],
        prior_quarter_summary="2026-Q1: ...",
    )
    assert "2026-Q2" in prompt
    assert "April" in prompt and "May" in prompt and "June" in prompt
    assert (
        "prior quarter" in prompt.lower() or "compared to" in prompt.lower()
    )


def test_quarter_prompt_handles_no_prior():
    prompt = build_quarter_prompt(
        quarter_label="2024-Q1",
        month_summaries=["Jan", "Feb", "Mar"],
        prior_quarter_summary=None,
    )
    # Should not break — first quarter in corpus has no prior
    assert "2024-Q1" in prompt
    assert "no prior" in prompt.lower() or "first" in prompt.lower()


def test_year_prompt_synthesizes_quarters():
    prompt = build_year_prompt(
        year=2025,
        quarter_summaries=[
            "2025-Q1: ...", "2025-Q2: ...", "2025-Q3: ...", "2025-Q4: ...",
        ],
    )
    assert "2025" in prompt
    assert (
        "year-in-review" in prompt.lower() or "annual" in prompt.lower()
        or "trends" in prompt.lower()
    )


def test_meta_prompt_3_year():
    prompt = build_meta_prompt(
        year_summaries=["2024: ...", "2025: ...", "2026: ..."],
    )
    assert (
        "3-year" in prompt or "three-year" in prompt.lower()
        or "long-term" in prompt.lower()
    )


@pytest.mark.asyncio
async def test_build_temporal_tree_emits_all_levels():
    """Synthetic test — verify the tree has L1, L2, L3, L4 nodes for a multi-year corpus."""

    async def fake_embed(texts):
        return [[0.1] * 4 for _ in texts]

    async def fake_summarize(prompt: str) -> str:
        return f"summary[{prompt[:30]}]"

    chunks = []
    for year in (2024, 2025, 2026):
        for month in (1, 4, 7, 10):  # one per quarter
            for chunk_idx in range(3):
                chunks.append({
                    "text": f"chunk y={year} m={month} i={chunk_idx}",
                    "shard_key": f"{year:04d}-{month:02d}",
                    "doc_id": year * 100 + month,
                    "chunk_index": chunk_idx,
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                })

    nodes = await build_temporal_tree(
        chunks=chunks,
        summarize=fake_summarize,
        embed=fake_embed,
        chat_model="qwen3-4b-qu",  # any model id — we mock summarize
    )

    levels = {n["payload"]["level"] for n in nodes}
    # Levels include 1 (monthly), 2 (quarterly), 3 (yearly)
    assert 1 in levels and 2 in levels and 3 in levels
    # Meta level only when more than 1 year present
    assert 4 in levels


@pytest.mark.asyncio
async def test_build_temporal_tree_single_year_no_meta():
    """Single-year corpus has no L4 meta node."""

    async def fake_embed(texts):
        return [[0.1] * 4 for _ in texts]

    async def fake_summarize(prompt: str) -> str:
        return "summary"

    chunks = [
        {
            "text": f"chunk m={month}",
            "shard_key": f"2026-{month:02d}",
            "doc_id": month,
            "chunk_index": 0,
            "embedding": [0.1, 0.2, 0.3, 0.4],
        }
        for month in (1, 4, 7, 10)
    ]
    nodes = await build_temporal_tree(
        chunks=chunks, summarize=fake_summarize, embed=fake_embed,
        chat_model="x",
    )
    levels = {n["payload"]["level"] for n in nodes}
    assert 4 not in levels  # no meta for single year
