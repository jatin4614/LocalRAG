"""Unit tests for ``ext.services.blob_store.BlobStore``."""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import threading
from pathlib import Path

import pytest

from ext.services.blob_store import BlobStore


@pytest.fixture()
def store(tmp_path: Path) -> BlobStore:
    return BlobStore(str(tmp_path / "blobs"))


def test_write_read_roundtrip(store: BlobStore) -> None:
    data = b"hello, world" * 1000
    sha = store.write(data)
    assert sha == hashlib.sha256(data).hexdigest()
    assert store.exists(sha)
    assert store.read(sha) == data


def test_write_is_idempotent(store: BlobStore) -> None:
    """Writing the same bytes twice yields the same sha and a single file."""
    data = b"same bytes"
    sha1 = store.write(data)
    sha2 = store.write(data)
    assert sha1 == sha2
    # Exactly one file exists at the address.
    p = Path(store.path(sha1))
    assert p.is_file()
    # No orphan .tmp.* files left behind.
    orphans = list(p.parent.glob(f"{sha1}.tmp.*"))
    assert orphans == []


def test_concurrent_writes_same_content_agree(store: BlobStore) -> None:
    """N threads writing identical bytes all end up with the same sha and
    leave exactly one file on disk (atomic rename semantics)."""
    data = b"concurrent payload " * 500
    expected = hashlib.sha256(data).hexdigest()
    results: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(store.write(data))
        except BaseException as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker) for _ in range(8)]
        for f in futures:
            f.result(timeout=10)

    assert errors == []
    assert all(r == expected for r in results)
    # Only one final file remains.
    p = Path(store.path(expected))
    assert p.is_file()
    siblings = [f for f in p.parent.iterdir() if f.name != expected]
    assert siblings == [], f"leftover files: {siblings}"


def test_delete_removes_blob(store: BlobStore) -> None:
    data = b"to delete"
    sha = store.write(data)
    assert store.exists(sha)
    store.delete(sha)
    assert not store.exists(sha)


def test_delete_is_idempotent(store: BlobStore) -> None:
    """Deleting a non-existent blob must not raise."""
    fake_sha = "0" * 64
    store.delete(fake_sha)  # first call: missing
    store.delete(fake_sha)  # second call: still missing
    # Also: delete after write-then-delete is still idempotent.
    sha = store.write(b"x")
    store.delete(sha)
    store.delete(sha)


def test_read_missing_raises_file_not_found(store: BlobStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.read("0" * 64)


def test_exists_false_for_missing(store: BlobStore) -> None:
    assert store.exists("deadbeef" * 8) is False


def test_path_uses_prefix_fanout(store: BlobStore) -> None:
    """Path layout is ``root/{sha[:2]}/{sha}`` for directory fan-out."""
    sha = "a" * 64
    p = Path(store.path(sha))
    assert p.name == sha
    assert p.parent.name == "aa"
    assert p.parent.parent == store.root


def test_empty_bytes_roundtrip(store: BlobStore) -> None:
    sha = store.write(b"")
    assert sha == hashlib.sha256(b"").hexdigest()
    assert store.read(sha) == b""


def test_final_file_is_atomic_no_tmp_leftover(store: BlobStore, tmp_path: Path) -> None:
    """After a successful write, no ``.tmp.*`` files should remain in the
    blob's subdir."""
    data = b"atomicity check " * 200
    sha = store.write(data)
    subdir = Path(store.path(sha)).parent
    tmps = list(subdir.glob("*.tmp.*"))
    assert tmps == [], f"atomic rename failed, leftovers: {tmps}"


def test_store_creates_root_lazily(tmp_path: Path) -> None:
    """BlobStore should create its root at construction time."""
    root = tmp_path / "new_root" / "nested"
    assert not root.exists()
    BlobStore(str(root))
    assert root.is_dir()


def test_write_fsyncs_before_rename(store: BlobStore) -> None:
    """Smoke test that writes succeed on a real fs (implicitly exercises fsync
    path; we can't easily assert fsync directly from user-space)."""
    data = os.urandom(4096)
    sha = store.write(data)
    assert store.read(sha) == data
