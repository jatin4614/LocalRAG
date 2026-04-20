"""Smoke tests for scripts/preflight_models.py CLI behaviour.

These tests exercise the wrapper behaviour that scripts/bootstrap.sh relies on:
  * --dry-run prints the plan and exits 0 without touching the network.
  * --verify-only exits 2 and lists missing models when the cache is empty.

Both tests run fully offline; they are safe to include in the default unit
suite.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "preflight_models.py"


def test_dry_run_prints_plan():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    stdout_lower = r.stdout.lower()
    assert "cache:" in stdout_lower
    # Should mention at least one known model family.
    assert any(
        tag in stdout_lower for tag in ("qwen", "bge", "whisper")
    ), f"expected qwen/bge/whisper in stdout, got:\n{r.stdout}"


def test_verify_only_returns_2_when_empty(tmp_path):
    # The script reads MODEL_CACHE from the environment (via its _env helper)
    # and falls back to ./volumes/models. Point it at an empty tmp dir so we
    # do not depend on any host-side cache state.
    env = {**os.environ, "MODEL_CACHE": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--verify-only"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert r.returncode == 2, (
        f"expected exit 2 for empty cache, got {r.returncode}\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    combined = (r.stdout + r.stderr).lower()
    assert "missing" in combined, f"expected 'missing' in output, got:\n{r.stdout}{r.stderr}"


def test_mutually_exclusive_flags_exit_4():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--verify-only"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 4, (
        f"expected exit 4 for conflicting flags, got {r.returncode}\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
