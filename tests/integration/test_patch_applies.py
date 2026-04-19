import subprocess
from pathlib import Path

import pytest
pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def _reset_upstream():
    tag = (ROOT / "UPSTREAM_VERSION").read_text().strip()
    subprocess.run(
        ["git", "-C", str(ROOT / "upstream"), "checkout", tag, "--", "."],
        check=True, capture_output=True,
    )


def test_apply_patches_succeeds():
    _reset_upstream()
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "apply_patches.sh")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"apply failed:\n{r.stderr}\n{r.stdout}"
    main_py = (ROOT / "upstream" / "backend" / "open_webui" / "main.py").read_text()
    assert "orgchat extension wiring" in main_py
    _reset_upstream()


def test_apply_patches_is_idempotent():
    _reset_upstream()
    r1 = subprocess.run(["bash", str(ROOT / "scripts" / "apply_patches.sh")],
                        capture_output=True, text=True)
    assert r1.returncode == 0
    r2 = subprocess.run(["bash", str(ROOT / "scripts" / "apply_patches.sh")],
                        capture_output=True, text=True)
    assert r2.returncode == 0
    assert "already applied" in r2.stdout
    _reset_upstream()
