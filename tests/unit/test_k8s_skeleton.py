from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_k8s_values_has_phase2_services():
    content = (ROOT / "k8s.future" / "values.yaml").read_text()
    for svc in ["postgres", "redis", "qdrant", "vllmChat", "vllmVision",
                "tei", "whisper", "modelManager", "openWebUI"]:
        assert svc in content


def test_k8s_readme_marks_as_skeleton():
    content = (ROOT / "k8s.future" / "README.md").read_text().lower()
    assert "skeleton" in content or "template" in content
    assert "phase 2" in content
