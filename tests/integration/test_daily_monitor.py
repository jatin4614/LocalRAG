import json
import subprocess
import os
from pathlib import Path

import pytest


def _golden_subset_has_real_rows() -> bool:
    """Return True if the daily subset contains at least one real query row.

    The placeholder file ships with a single ``{"_comment": ...}`` row until the
    operator labels ``golden_starter.jsonl`` and copies 20 queries over per Plan
    A Task 1.8 Step 1. Until then, the harness has nothing to score and would
    KeyError on its summary print, so we skip the integration test cleanly.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "eval"
        / "golden_daily_subset.jsonl"
    )
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "query" in row:
            return True
    return False


@pytest.mark.integration
def test_daily_eval_cron_runs_end_to_end(tmp_path, monkeypatch):
    """Execute the daily cron against a live stack and verify output file shape."""
    if not _golden_subset_has_real_rows():
        pytest.skip(
            "golden_daily_subset.jsonl is placeholder-only — operator must "
            "populate 20 queries from golden_starter.jsonl per Plan A Task 1.8 "
            "Step 1 before this test can exercise the cron end-to-end."
        )
    out_dir = tmp_path
    monkeypatch.setenv("KB_EVAL_ID", os.environ.get("KB_EVAL_ID", "1"))
    monkeypatch.setenv("API_BASE", os.environ.get("API_BASE", "http://localhost:6100"))
    # Point the textfile output to tmp
    script = Path(__file__).resolve().parents[2] / "scripts" / "daily_eval_cron.sh"
    # Copy script to tmp and patch OUT_DIR
    body = script.read_text().replace(
        'OUT_DIR="/var/lib/node_exporter/textfile_collector"',
        f'OUT_DIR="{out_dir}"',
    )
    patched = tmp_path / "daily_eval_cron.sh"
    patched.write_text(body)
    patched.chmod(0o755)

    r = subprocess.run(["bash", str(patched)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr

    prom_file = out_dir / "retrieval_daily.prom"
    assert prom_file.exists()
    content = prom_file.read_text()
    assert "retrieval_ndcg_daily" in content
    assert 'intent="__global__"' in content
