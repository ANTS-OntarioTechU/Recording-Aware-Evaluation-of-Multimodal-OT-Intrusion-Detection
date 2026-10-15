"""Project-root resolution.

Every notebook and test resolves paths through here so that no absolute path is baked into
the release. Set ``PROVICS_ROOT`` to override, otherwise the repository root is inferred from
this file's location.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "PROVICS_ROOT"


def project_root() -> Path:
    """Return the repository root.

    Reads ``PROVICS_ROOT`` if set, otherwise walks up from this file. Raises ``FileNotFoundError``
    if the resolved directory does not contain ``configs/evaluation_settings.json``, which is
    the marker used to confirm a complete checkout.
    """
    root = Path(os.environ[_ENV_VAR]).expanduser() if _ENV_VAR in os.environ \
        else Path(__file__).resolve().parents[2]
    marker = root / "configs" / "evaluation_settings.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"{root} does not look like a checkout of this repository: {marker} is missing. "
            f"Set {_ENV_VAR} to the repository root."
        )
    return root


def results_dir() -> Path:
    """Directory holding canonical submitted-paper artifacts."""
    return project_root() / "results"


def regenerated_dir() -> Path:
    """Directory for newly produced output, kept separate from canonical artifacts."""
    d = project_root() / "results" / "regenerated"
    d.mkdir(parents=True, exist_ok=True)
    return d
