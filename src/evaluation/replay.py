"""Load public replay settings and verify released artifact checksums.

The settings contain only inputs needed by the notebooks. Artifact hashes are stored separately
in ``data_manifest/artifact_checksums.json``; internal paper-review notes are not required.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..utils.hashing import sha256_file
from ..utils.paths import project_root


def load_artifact_manifest() -> dict:
    """Read the released path-to-SHA-256 list. Missing or malformed lists fail closed."""
    path = project_root() / "data_manifest" / "artifact_checksums.json"
    manifest = json.loads(path.read_text())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"{path} has no artifact checksums")
    for relative_path, digest in artifacts.items():
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"artifact path must stay inside the checkout: {relative_path}")
        if not isinstance(digest, str) or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid SHA-256 for {relative_path}")
    return manifest


def load_settings() -> dict:
    """Read the public settings after checking their recorded checksum."""
    root = project_root()
    relative = "configs/evaluation_settings.json"
    path = root / relative
    expected = load_artifact_manifest()["artifacts"][relative]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"evaluation settings checksum mismatch: expected {expected}, actual {actual}")
    return json.loads(path.read_text())


def verify_artifacts(manifest: dict | None = None) -> list[dict]:
    """Check every artifact in the public checksum list.

    Returns one record per artifact with ``status`` in ``ok`` / ``missing`` / ``checksum_mismatch``.
    Does not raise, so a caller can report the full picture rather than the first failure.
    """
    manifest = load_artifact_manifest() if manifest is None else manifest
    root = project_root()
    records: list[dict] = []
    for relative_path, expected in manifest["artifacts"].items():
        path = root / relative_path
        if not path.is_file():
            status = "missing"
            actual = None
        else:
            actual = sha256_file(path)
            status = "ok" if actual == expected else "checksum_mismatch"
        records.append({"artifact": relative_path, "status": status,
                        "expected": expected, "actual": actual})
    return records


def assert_artifacts_intact(manifest: dict | None = None) -> None:
    """Raise ``AssertionError`` naming every artifact that is missing or altered."""
    problems = [r for r in verify_artifacts(manifest) if r["status"] != "ok"]
    if problems:
        lines = "\n".join(f"  {r['status']:<18} {r['artifact']}" for r in problems)
        raise AssertionError(f"{len(problems)} released artifact(s) not intact:\n{lines}")


