"""Smoke tests for ``scripts/reembed_all.py``.

The script must be callable with ``--help`` without a live Qdrant/TEI.
This guards against accidental import-time side effects that would
make the operational tool unusable when someone just wants the usage.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "reembed_all.py"


def test_script_exists() -> None:
    assert SCRIPT.is_file()


def test_script_help_returns_zero() -> None:
    """``python scripts/reembed_all.py --help`` exits 0 and prints usage."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--pipeline-version" in proc.stdout
    assert "--kb" in proc.stdout
    assert "--all" in proc.stdout
    assert "--apply" in proc.stdout


def test_script_parses_ast() -> None:
    """Syntax sanity — parse the script without executing it."""
    import ast
    text = SCRIPT.read_text()
    ast.parse(text)  # raises on any SyntaxError


def test_script_rejects_without_kb_or_all() -> None:
    """Mutually-exclusive --kb/--all group requires one of them."""
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--pipeline-version", "test",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    # argparse exits non-zero and prints the error to stderr.
    assert proc.returncode != 0
    assert "one of the arguments" in proc.stderr or "required" in proc.stderr


def test_script_rejects_kb_plus_all() -> None:
    """Mutually-exclusive: --kb and --all at the same time → error."""
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--kb", "1",
            "--all",
            "--pipeline-version", "test",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "not allowed" in proc.stderr or "argument" in proc.stderr
