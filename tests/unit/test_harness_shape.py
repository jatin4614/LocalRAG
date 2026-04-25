import inspect

from tests.eval import harness


def test_harness_exports_run_eval():
    assert callable(harness.run_eval)
    sig = inspect.signature(harness.run_eval)
    params = set(sig.parameters)
    assert {"golden_path", "kb_id", "api_base_url"}.issubset(params)


def test_aggregate_handles_empty():
    assert harness._aggregate([]) == {"n": 0}


def test_aggregate_emits_keys():
    rows = [
        {"chunk_recall@k": 1.0, "mrr@k": 1.0, "ndcg@k": 1.0, "latency_ms": 100},
        {"chunk_recall@k": 0.5, "mrr@k": 0.5, "ndcg@k": 0.5, "latency_ms": 200},
    ]
    agg = harness._aggregate(rows)
    assert agg["n"] == 2
    assert agg["chunk_recall@10"] == 0.75
    assert agg["p95_latency_ms"] >= 100
