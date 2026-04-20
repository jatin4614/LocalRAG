"""sha256-addressed filesystem blob store with atomic writes.

Write procedure:
    1. compute sha256 of data
    2. write to ``{path}.tmp.{uuid4}``
    3. ``os.fsync`` the file
    4. ``os.replace(tmp, final)`` — atomic on POSIX

Files fan out into two-char subdirectories (``root/{sha[:2]}/{sha}``) to
keep any single directory from exceeding filesystem-friendly sizes under
realistic load.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path


class BlobStore:
    """A sha256-addressed blob store rooted at ``root``.

    All writes are atomic (POSIX ``os.replace``). Idempotent: writing the
    same bytes twice returns the same sha and leaves exactly one file on disk.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sha: str) -> str:
        """Return the on-disk absolute path for ``sha`` (not guaranteed to exist)."""
        return str(self.root / sha[:2] / sha)

    def write(self, data: bytes) -> str:
        """Write ``data`` atomically and return its sha256 hex digest.

        If a blob with that sha already exists on disk, this is a no-op.
        """
        sha = hashlib.sha256(data).hexdigest()
        final = Path(self.path(sha))
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            return sha
        tmp = final.with_name(f"{final.name}.tmp.{uuid.uuid4().hex}")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            # Write full payload, then fsync before rename.
            view = memoryview(data)
            pos = 0
            while pos < len(view):
                pos += os.write(fd, view[pos:])
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.replace(str(tmp), str(final))
        except FileNotFoundError:
            # Another writer already moved it into place between our check and
            # replace — that's fine, content is identical.
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        return sha

    def read(self, sha: str) -> bytes:
        """Read and return the bytes for ``sha``. Raises FileNotFoundError if absent."""
        return Path(self.path(sha)).read_bytes()

    def exists(self, sha: str) -> bool:
        """Return True iff a blob for ``sha`` exists on disk."""
        return Path(self.path(sha)).is_file()

    def delete(self, sha: str) -> None:
        """Delete the blob for ``sha`` if present. Idempotent (``missing_ok=True``)."""
        Path(self.path(sha)).unlink(missing_ok=True)
