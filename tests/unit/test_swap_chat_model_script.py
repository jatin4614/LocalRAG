"""CLI sanity + dry-run behaviour for scripts/swap_chat_model.py.

Never touches the real compose/.env or downloads weights. We drive the
CLI with a temp .env fixture and --dry-run to verify parsing + refusal
logic.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "swap_chat_model.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("swap_chat_model_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_file_exists() -> None:
    assert SCRIPT.is_file(), f"missing: {SCRIPT}"


def test_help_parses_cleanly() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, f"stderr={r.stderr!r}"
    # Key switches advertised in docs must appear in --help
    assert "--model" in r.stdout
    assert "--tokenizer" in r.stdout
    assert "--dry-run" in r.stdout
    assert "--force" in r.stdout


def test_module_imports_cleanly() -> None:
    mod = _load_module()
    # Public entry point + a couple helpers we rely on in tests
    assert callable(mod.main)
    assert callable(mod._read_env_var)
    assert callable(mod._patch_env_var)


def _make_env(tmp_path: Path, *, chat_model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ") -> Path:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# generated fixture\n"
        f"CHAT_MODEL={chat_model}\n"
        "EMBED_MODEL=BAAI/bge-m3\n"
        "RAG_BUDGET_TOKENIZER=qwen\n",
        encoding="utf-8",
    )
    return env_path


def test_refuses_when_model_unchanged(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "--env",
            str(env),
            "--skip-preflight",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    # Script exits non-zero and does NOT modify .env
    assert r.returncode != 0
    assert "Qwen/Qwen2.5-14B-Instruct-AWQ" in (r.stdout + r.stderr)
    # .env untouched
    assert env.read_text() == (
        "# generated fixture\n"
        "CHAT_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ\n"
        "EMBED_MODEL=BAAI/bge-m3\n"
        "RAG_BUDGET_TOKENIZER=qwen\n"
    )
    # No backup written
    assert not (env.with_suffix(env.suffix + ".bak")).exists()


def test_dry_run_does_not_modify_env(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    original = env.read_text()
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "google/gemma-3-12b-it",
            "--tokenizer",
            "gemma",
            "--env",
            str(env),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, f"stderr={r.stderr!r}"
    assert env.read_text() == original
    assert not (env.with_suffix(env.suffix + ".bak")).exists()
    # Plan should mention both old + new models
    out = r.stdout
    assert "Qwen/Qwen2.5-14B-Instruct-AWQ" in out
    assert "google/gemma-3-12b-it" in out


def test_missing_env_returns_error_code(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.env"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "google/gemma-3-12b-it",
            "--env",
            str(missing),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode != 0
    assert str(missing) in r.stderr or ".env not found" in r.stderr


def test_patch_env_var_idempotent(tmp_path: Path) -> None:
    mod = _load_module()
    env = _make_env(tmp_path)
    # First patch: change the value
    changed1 = mod._patch_env_var(env, "CHAT_MODEL", "google/gemma-3-12b-it")
    assert changed1 is True
    assert "CHAT_MODEL=google/gemma-3-12b-it" in env.read_text()
    # Second patch: same value -> no-op
    changed2 = mod._patch_env_var(env, "CHAT_MODEL", "google/gemma-3-12b-it")
    assert changed2 is False


def test_patch_env_var_appends_missing_key(tmp_path: Path) -> None:
    mod = _load_module()
    env = tmp_path / ".env"
    env.write_text("FOO=bar\n", encoding="utf-8")
    changed = mod._patch_env_var(env, "RAG_BUDGET_TOKENIZER", "gemma")
    assert changed is True
    text = env.read_text()
    assert "FOO=bar" in text
    assert "RAG_BUDGET_TOKENIZER=gemma" in text


def test_read_env_var_ignores_comments(tmp_path: Path) -> None:
    mod = _load_module()
    env = tmp_path / ".env"
    env.write_text(
        "# CHAT_MODEL=commented-out\n"
        "CHAT_MODEL=real-value\n",
        encoding="utf-8",
    )
    assert mod._read_env_var(env, "CHAT_MODEL") == "real-value"


def test_tokenizer_hint_from_repo_name() -> None:
    mod = _load_module()
    assert mod._tokenizer_hint("Qwen/Qwen2.5-14B-Instruct-AWQ") == "qwen"
    assert mod._tokenizer_hint("google/gemma-3-12b-it") == "gemma"
    assert mod._tokenizer_hint("meta-llama/Llama-3.3-70B-Instruct-AWQ") == "llama"
    assert mod._tokenizer_hint("random/unknown-model") is None
