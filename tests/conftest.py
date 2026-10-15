"""Shared fixtures. All paths resolve through src.utils.paths so no absolute path is baked in."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import project_root  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return project_root()


@pytest.fixture(scope="session")
def settings(root: Path) -> dict:
    from src.evaluation.replay import load_settings
    return load_settings()


@pytest.fixture(scope="session")
def artifact_manifest(root: Path) -> dict:
    from src.evaluation.replay import load_artifact_manifest
    return load_artifact_manifest()


@pytest.fixture(scope="session")
def expected(root: Path) -> dict:
    return json.loads((root / "tests" / "expected_results.json").read_text())


@pytest.fixture(scope="session")
def tables(root: Path) -> Path:
    return root / "results" / "paper_tables"
