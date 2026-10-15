"""SHA-256 helpers used by the manifest checks and the frozen-replay tests."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 22) -> str:
    """Return the hex SHA-256 of a file, read in chunks.

    Raises ``FileNotFoundError`` if the path does not exist.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise ``ValueError`` if the file's digest differs from ``expected``."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path}\n  expected {expected}\n  actual   {actual}"
        )
