import json
import subprocess
import sys
from pathlib import Path


def _run_gate(tmp_path: Path, baseline: dict, latest: dict) -> tuple[int, str]:
    bp = tmp_path / "baseline.json"
    lp = tmp_path / "latest.json"
    sp = tmp_path / "slo.md"
    bp.write_text(json.dumps(baseline))
    lp.write_text(json.dumps(latest))
    sp.write_text("# slo stub")
    r = subprocess.run(
        [sys.executable, "-m", "tests.eval.gate",
         "--baseline", str(bp), "--latest", str(lp), "--slo", str(sp)],
        capture_output=True, text=True,
    )
    return r.returncode, r.stdout + r.stderr


def test_gate_passes_on_no_regression(tmp_path):
    b = {"global": {"chunk_recall@10": 0.80, "p95_latency_ms": 900},
         "by_intent": {"specific": {"n": 30, "chunk_recall@10": 0.85},
                       "metadata": {"n": 7, "chunk_recall@10": 0.75}}}
    rc, out = _run_gate(tmp_path, b, b)
    assert rc == 0, out


def test_gate_fails_on_global_regression(tmp_path):
    b = {"global": {"chunk_recall@10": 0.80, "p95_latency_ms": 900},
         "by_intent": {"specific": {"n": 30, "chunk_recall@10": 0.85},
                       "metadata": {"n": 7, "chunk_recall@10": 0.75}}}
    l = {"global": {"chunk_recall@10": 0.77, "p95_latency_ms": 900},
         "by_intent": {"specific": {"n": 30, "chunk_recall@10": 0.85},
                       "metadata": {"n": 7, "chunk_recall@10": 0.75}}}
    rc, out = _run_gate(tmp_path, b, l)
    assert rc == 1, out
    assert "GLOBAL regression" in out


def test_gate_fails_on_metadata_floor_breach(tmp_path):
    b = {"global": {"chunk_recall@10": 0.80, "p95_latency_ms": 900},
         "by_intent": {"metadata": {"n": 7, "chunk_recall@10": 0.75}}}
    l = {"global": {"chunk_recall@10": 0.80, "p95_latency_ms": 900},
         "by_intent": {"metadata": {"n": 7, "chunk_recall@10": 0.65}}}
    rc, out = _run_gate(tmp_path, b, l)
    assert rc == 1, out
    assert "FLOOR breach" in out
