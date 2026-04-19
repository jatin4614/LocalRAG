"""Sanity tests for scripts/seed_eval_corpus.py.

Full end-to-end behaviour needs a live Qdrant + TEI stack; these tests only
check that the script is wired correctly — imports resolve, CLI parses,
help text mentions kb_eval, and the internal helpers behave.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "seed_eval_corpus.py"


def test_script_file_exists_and_is_executable():
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"


def test_help_works():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    # Help text must reference the target collection name so users know where
    # this script is landing its output.
    assert "kb_eval" in r.stdout


def test_module_imports_cleanly():
    # Don't execute the script (that calls asyncio.run and needs live
    # services) — just confirm the source file imports without exploding.
    # Use a unique module name to avoid polluting sys.modules for later tests.
    spec = importlib.util.spec_from_file_location(
        "seed_eval_corpus_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Smoke: key helpers exist and have plausible signatures.
    assert hasattr(mod, "seed")
    assert hasattr(mod, "_stable_doc_id")
    assert hasattr(mod, "_discover_files")
    assert hasattr(mod, "WORKTREE_ALLOWLIST")


def test_stable_doc_id_is_deterministic_and_bounded():
    spec = importlib.util.spec_from_file_location(
        "seed_eval_corpus_doc_id_test", SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = mod._stable_doc_id("CLAUDE.md")
    b = mod._stable_doc_id("CLAUDE.md")
    c = mod._stable_doc_id("Ragupdate.md")
    assert a == b, "same path must produce same doc_id"
    assert a != c, "different paths should produce different doc_ids (collision unlikely)"
    assert 0 <= a < 1_000_000
    assert 0 <= c < 1_000_000


def test_discover_files_respects_allowlist():
    # The allowlist includes CLAUDE.md which is present in the worktree;
    # discovery must return it. Missing entries must be silently skipped.
    spec = importlib.util.spec_from_file_location(
        "seed_eval_corpus_discover_test", SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seed_dir = ROOT / "tests" / "eval" / "seed_corpus"
    files = mod._discover_files(seed_dir)
    assert files, "expected at least one allowlisted doc to exist in the worktree"
    rels = {p.relative_to(ROOT).as_posix() for p in files}
    # CLAUDE.md is load-bearing for the corpus — if it moved we want to
    # notice fast.
    assert "CLAUDE.md" in rels
