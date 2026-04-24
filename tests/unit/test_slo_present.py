from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLO = ROOT / "docs" / "runbook" / "slo.md"


def test_slo_file_exists():
    assert SLO.exists(), "SLO document missing at docs/runbook/slo.md"


def test_slo_has_required_sections():
    content = SLO.read_text()
    required = [
        "## Retrieval latency budget",
        "## Cost budget",
        "## Error rate ceiling",
        "## Quality floor per intent",
        "## Post-plan projection",
    ]
    for section in required:
        assert section in content, f"SLO missing section: {section}"


def test_slo_has_concrete_numbers():
    content = SLO.read_text().lower()
    assert "p50" in content and "p95" in content and "p99" in content, \
        "SLO must specify p50/p95/p99 retrieval latency"
    assert "chunk_recall@10" in content, "SLO must specify retrieval quality floor"
