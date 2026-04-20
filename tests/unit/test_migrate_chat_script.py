"""CLI/parser smoke for scripts/migrate_chat_collections.py (P2.3)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "migrate_chat_collections.py"


def test_help_parses_cleanly() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0
    assert "--qdrant-url" in r.stdout
    assert "--dry-run" in r.stdout
    assert "--apply" in r.stdout
    assert "--delete-source" in r.stdout


def test_dry_run_and_apply_mutually_exclusive() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--apply"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode != 0
    assert "not allowed" in r.stderr or "argument" in r.stderr


def test_module_imports_cleanly() -> None:
    """Import the script as a module — verifies top-level doesn't do network I/O."""
    spec = importlib.util.spec_from_file_location("migrate_chat_collections", SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Key helpers are exposed
    assert callable(getattr(mod, "_is_legacy_chat"))
    assert callable(getattr(mod, "_chat_id_from_collection"))
    # Never treats chat_private as legacy
    assert mod._is_legacy_chat("chat_private") is False
    assert mod._is_legacy_chat("chat_abc-def") is True
    assert mod._chat_id_from_collection("chat_abc-def") == "abc-def"
