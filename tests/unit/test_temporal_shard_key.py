"""Tests for temporal_shard.extract_shard_key + helpers.

Plan B Phase 5.2.
"""
from __future__ import annotations

import datetime as dt
import pytest

from ext.services.temporal_shard import (
    extract_shard_key,
    parse_shard_key,
    iter_shard_keys,
    ShardKeyOrigin,
)


class TestExtractShardKey:
    def test_filename_dmy(self):
        sk, origin = extract_shard_key(
            filename="05 Jan 2026.docx", body="irrelevant"
        )
        assert sk == "2026-01"
        assert origin is ShardKeyOrigin.FILENAME

    def test_filename_iso(self):
        sk, origin = extract_shard_key(
            filename="2024-08-15-summary.md", body=""
        )
        assert sk == "2024-08"
        assert origin is ShardKeyOrigin.FILENAME

    def test_filename_mdy(self):
        sk, origin = extract_shard_key(
            filename="January 5 2026 report.pdf", body=""
        )
        assert sk == "2026-01"

    def test_yaml_frontmatter(self):
        body = "---\ndate: 2025-03-20\ntitle: Q1 Update\n---\n\nContent..."
        sk, origin = extract_shard_key(filename="random.md", body=body)
        assert sk == "2025-03"
        assert origin is ShardKeyOrigin.FRONTMATTER

    def test_first_body_date(self):
        body = "Meeting on June 12, 2025. Attendees: ..."
        sk, origin = extract_shard_key(filename="meeting.txt", body=body)
        assert sk == "2025-06"
        assert origin is ShardKeyOrigin.BODY

    def test_fallback_to_now(self):
        body = "No dates here at all"
        sk, origin = extract_shard_key(filename="random.txt", body=body)
        # Should be current month
        now = dt.date.today()
        assert sk == f"{now.year:04d}-{now.month:02d}"
        assert origin is ShardKeyOrigin.INGEST_DEFAULT

    def test_filename_takes_priority_over_body(self):
        # Filename: 2026-01, body has 2025-03 frontmatter — filename wins
        body = "---\ndate: 2025-03-20\n---\n"
        sk, origin = extract_shard_key(
            filename="05 Jan 2026.docx", body=body,
        )
        assert sk == "2026-01"


class TestParseShardKey:
    def test_valid(self):
        y, m = parse_shard_key("2024-07")
        assert y == 2024 and m == 7

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_shard_key("2024-7")  # need zero-pad

    def test_invalid_month(self):
        with pytest.raises(ValueError):
            parse_shard_key("2024-13")


class TestIterShardKeys:
    def test_year_range(self):
        keys = list(iter_shard_keys(start="2024-01", end="2024-04"))
        assert keys == ["2024-01", "2024-02", "2024-03", "2024-04"]

    def test_year_boundary(self):
        keys = list(iter_shard_keys(start="2024-11", end="2025-02"))
        assert keys == ["2024-11", "2024-12", "2025-01", "2025-02"]

    def test_full_36_months(self):
        keys = list(iter_shard_keys(start="2024-01", end="2026-12"))
        assert len(keys) == 36
