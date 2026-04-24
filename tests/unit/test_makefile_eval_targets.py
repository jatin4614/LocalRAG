from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MF = ROOT / "Makefile"


def test_makefile_has_eval_targets():
    content = MF.read_text()
    for tgt in ["eval:", "eval-baseline:", "eval-gate:", "eval-seed:"]:
        assert tgt in content, f"Makefile missing target: {tgt}"
