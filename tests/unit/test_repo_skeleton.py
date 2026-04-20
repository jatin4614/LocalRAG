from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_git_repo_initialized():
    # .git is a directory in a normal clone, but a file (gitdir: ...) in a git worktree.
    assert (ROOT / ".git").exists(), ".git missing — run `git init`"

def test_gitignore_ignores_volumes_and_venv():
    content = (ROOT / ".gitignore").read_text()
    for entry in ["volumes/", ".venv/", "__pycache__/", ".pytest_cache/", "*.pyc"]:
        assert entry in content, f".gitignore missing entry: {entry}"
    assert "upstream/" not in content, (
        "upstream/ must NOT be gitignored — it's a git submodule path"
    )

def test_readme_exists():
    readme = ROOT / "README.md"
    assert readme.exists()
    assert "Org Chat Assistant" in readme.read_text()
