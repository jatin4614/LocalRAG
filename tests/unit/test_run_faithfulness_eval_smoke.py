"""Smoke tests for tests/eval/run_faithfulness_eval.py (P3.5).

Full end-to-end behaviour needs a live Qdrant + TEI + chat model; these tests
only check that the runner is wired correctly — CLI --help parses and the
module imports cleanly without triggering any network calls.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "eval" / "run_faithfulness_eval.py"


def test_script_file_exists():
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"


def test_help_works():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    # Help text should mention the key options so users can discover them.
    assert "--golden" in r.stdout
    assert "--chat-url" in r.stdout
    assert "--chat-model" in r.stdout
    assert "--out" in r.stdout


def test_module_imports_cleanly():
    # Don't execute main() (that calls asyncio.run and needs live services);
    # just confirm the source file imports without exploding. Use a unique
    # module name so we don't pollute sys.modules for other tests.
    spec = importlib.util.spec_from_file_location(
        "run_faithfulness_eval_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Key symbols exist.
    assert hasattr(mod, "main")
    assert hasattr(mod, "_retrieve_pipeline")
    assert hasattr(mod, "_build_context")
    assert hasattr(mod, "_generate_answer")
    assert hasattr(mod, "_parse_args")


def test_build_context_joins_hits_with_separator():
    spec = importlib.util.spec_from_file_location(
        "run_faithfulness_eval_build_ctx", SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _FakeHit:
        def __init__(self, text):
            self.payload = {"text": text}

    hits = [_FakeHit("first chunk"), _FakeHit("second chunk"), _FakeHit("")]
    ctx = mod._build_context(hits)
    # Empty payload dropped; remaining joined by "\n\n---\n\n".
    assert "first chunk" in ctx
    assert "second chunk" in ctx
    assert "---" in ctx
    # Empty-text hit must not leave a dangling separator at the tail.
    assert not ctx.rstrip().endswith("---")


def test_build_context_handles_missing_payload():
    spec = importlib.util.spec_from_file_location(
        "run_faithfulness_eval_missing_payload", SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _FakeHit:
        payload = None

    ctx = mod._build_context([_FakeHit()])
    assert ctx == ""


def test_parse_args_defaults_match_spec():
    spec = importlib.util.spec_from_file_location(
        "run_faithfulness_eval_parse_args", SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ns = mod._parse_args([])
    assert ns.golden == "tests/eval/golden.jsonl"
    assert ns.qdrant_url == "http://localhost:6333"
    # Chat endpoint defaults to the vllm-chat container IP we already use
    # elsewhere in the eval harness.
    assert ns.chat_url == "http://172.19.0.7:8000/v1"
    assert ns.chat_model == "orgchat-chat"
    assert ns.k == 10
    assert ns.concurrency == 4
    assert ns.label == "faithfulness_baseline"
