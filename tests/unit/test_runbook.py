from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "docs" / "runbook.md"


def test_runbook_exists():
    assert RUN.is_file()


def test_runbook_has_required_sections():
    content = RUN.read_text().lower()
    for heading in [
        "prerequisites",
        "first-run",
        "troubleshooting",
        "backup",
        "upgrad",
    ]:
        assert heading in content, f"runbook missing section: {heading!r}"


def test_runbook_mentions_bootstrap_script():
    content = RUN.read_text()
    assert "bootstrap.sh" in content
    assert "compose/.env" in content
