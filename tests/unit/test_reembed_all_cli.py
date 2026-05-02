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


def test_script_imports_vector_store() -> None:
    """Bug-fix campaign §3.3: script must route upserts through
    ``VectorStore.upsert`` so custom-sharded targets (kb_1_v4 since
    2026-04-26) auto-derive the shard_key from each point's payload.
    Direct ``qdrant.upsert`` would 400 with ``Shard key not specified``.

    Asserts the source file imports VectorStore — a static guard against
    an accidental revert to the raw client path.
    """
    text = SCRIPT.read_text()
    # Either explicit import, or referenced by name when constructed.
    assert "VectorStore" in text, (
        "scripts/reembed_all.py must use ext.services.vector_store.VectorStore "
        "to route upserts through the sharding-aware path."
    )
    # The script should NOT call qdrant.upsert directly any more.
    # ``qdrant.scroll`` and ``qdrant.get_collections`` remain fine.
    import re
    bad = re.search(r"\bqdrant\.upsert\b", text)
    assert bad is None, (
        "scripts/reembed_all.py still calls qdrant.upsert directly — "
        "this bypasses VectorStore's shard_key derivation and will fail "
        "on custom-sharded collections."
    )


def test_script_signature_matches_vector_store_upsert() -> None:
    """``VectorStore.upsert(name, points: list[dict])`` is the contract.

    Locks the import + signature so the script doesn't drift from the
    helper it depends on. Without this, a future refactor of VectorStore
    could silently break the operator script.
    """
    import inspect

    # Repo-relative import; the script does the same on its sys.path
    # fix-up so this is the same surface it will see at runtime.
    sys.path.insert(0, str(ROOT))
    try:
        from ext.services.vector_store import VectorStore
    finally:
        sys.path.remove(str(ROOT))

    sig = inspect.signature(VectorStore.upsert)
    params = list(sig.parameters)
    # First param after self is 'name', then keyword-only options.
    assert params[0] == "self"
    assert params[1] == "name"
    assert "points" in params
