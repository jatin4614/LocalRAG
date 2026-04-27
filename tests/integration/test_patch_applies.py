import subprocess
from pathlib import Path

import pytest
pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream"
UPSTREAM_MAIN = UPSTREAM / "backend" / "open_webui" / "main.py"
UPSTREAM_GIT = UPSTREAM / ".git"


def _upstream_checked_out() -> bool:
    """Both the working tree AND the git metadata must be present.

    B10 — the original check only looked at ``UPSTREAM_MAIN``. That is true
    in some main-repo states even when the submodule was never initialised
    (e.g. an old artifact tree without ``.git``), in which case
    ``apply_patches.sh`` fails partway through with a confusing
    ``checkout`` error rather than a clean skip. Requiring ``.git`` as
    well makes the precondition match the spec ("upstream/ as a real
    git submodule") and the script's actual requirements.
    """
    return UPSTREAM_MAIN.exists() and UPSTREAM_GIT.exists()


def _reset_upstream():
    tag = (ROOT / "UPSTREAM_VERSION").read_text().strip()
    subprocess.run(
        ["git", "-C", str(UPSTREAM), "checkout", tag, "--", "."],
        check=True, capture_output=True,
    )


@pytest.mark.skipif(
    not _upstream_checked_out(),
    reason="upstream submodule not initialised in this worktree "
    "(git submodule update --init upstream)",
)
def test_apply_patches_succeeds():
    _reset_upstream()
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "apply_patches.sh")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"apply failed:\n{r.stderr}\n{r.stdout}"
    main_py = UPSTREAM_MAIN.read_text()
    assert "orgchat extension wiring" in main_py
    _reset_upstream()


@pytest.mark.skipif(
    not _upstream_checked_out(),
    reason="upstream submodule not initialised in this worktree "
    "(git submodule update --init upstream)",
)
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
