from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DIRS = [
    "ext",
    "ext/db",
    "ext/db/migrations",
    "ext/db/models",
    "ext/services",
    "ext/routers",
    "model_manager",
    "whisper_service",
    "scripts",
    "compose",
    "compose/caddy",
    "branding",
    "volumes",  # gitignored, but present for bind mounts
]

EXPECTED_INITS = [
    "ext/__init__.py",
    "ext/db/__init__.py",
    "ext/db/models/__init__.py",
    "ext/services/__init__.py",
    "ext/routers/__init__.py",
]

def test_directories_exist():
    for d in EXPECTED_DIRS:
        assert (ROOT / d).is_dir(), f"missing dir: {d}"

def test_init_files_exist():
    for f in EXPECTED_INITS:
        assert (ROOT / f).is_file(), f"missing __init__: {f}"

def test_ext_importable():
    import sys
    sys.path.insert(0, str(ROOT))
    import ext  # noqa: F401
