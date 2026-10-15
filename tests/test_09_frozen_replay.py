"""9. Deterministic replay of the frozen score-based tables and figures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.replay import assert_artifacts_intact, verify_artifacts
from src.utils.hashing import sha256_file


def test_every_released_artifact_is_intact(artifact_manifest):
    assert_artifacts_intact(artifact_manifest)


def test_verification_covers_all_canonical_results(root, artifact_manifest):
    records = verify_artifacts(artifact_manifest)
    expected = {p.relative_to(root).as_posix() for p in (root / "results").rglob("*")
                if p.is_file() and "regenerated" not in p.parts and p.name != ".DS_Store"}
    assert expected <= {r["artifact"] for r in records}
    assert all(r["status"] == "ok" for r in records)


def test_frozen_scores_are_readable_and_stable(root):
    """The persisted per-window score streams load and have consistent shapes."""
    score_files = sorted((root / "results/frozen_scores").glob("A_M*_w*.npz"))
    assert len(score_files) == 16, f"expected 16 score streams, found {len(score_files)}"
    for path in score_files:
        with np.load(path) as bundle:
            assert {"attack", "benign"} <= set(bundle.files)
            assert bundle["attack"].ndim == 1 and bundle["benign"].ndim == 1
            assert np.isfinite(bundle["attack"]).any()


def test_replay_of_delta_cw_from_the_persisted_table(root, expected):
    """Recompute the headline range from the table and compare with the recorded value."""
    summary = pd.read_csv(root / "results/paper_tables/window_sensitivity_summary.csv")
    cell_means = summary.groupby(["window_s", "modality"])["delta_cw"].mean()
    entry = next(e for e in expected["entries"] if e["id"] == "delta_cw_range")
    assert round(float(cell_means.min()), 3) == entry["expected"][0]
    assert round(float(cell_means.max()), 3) == entry["expected"][1]
    assert (cell_means > 0).all(), "Delta_CW must be positive in every cell"


def test_paper_figures_present_and_checksummed(root, artifact_manifest):
    figures = [a for a in artifact_manifest["artifacts"]
               if a.startswith("results/paper_figures/") and a.endswith(".pdf")]
    assert len(figures) == 3, f"expected three paper figures, found {figures}"
    for relative in figures:
        path = root / relative
        assert path.is_file()
        expected_hash = artifact_manifest["artifacts"][relative]
        assert sha256_file(path) == expected_hash
