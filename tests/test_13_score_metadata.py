"""Saved score arrays must use their own canonical metadata, not the inventory grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.hashing import sha256_file

WINDOWS = [5, 10, 30, 60]
PHASES = [("attack", "attack22h"), ("benign", "benign48h")]


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("array_key,phase", PHASES)
def test_metadata_grid_is_present_checksummed_and_ordered(
        root, settings, artifact_manifest, window, array_key, phase):
    relative = settings["score_metadata"][f"{array_key}_pattern"].format(window_s=window)
    path = root / relative
    assert path.is_file()
    assert sha256_file(path) == artifact_manifest["artifacts"][relative]
    meta = pd.read_parquet(path)
    assert meta["window_id"].is_unique
    assert meta["phase"].eq(phase).all()
    starts = pd.to_datetime(meta["window_start"], utc=True)
    ends = pd.to_datetime(meta["window_end"], utc=True)
    assert starts.is_monotonic_increasing and starts.is_unique
    assert starts.diff().dropna().dt.total_seconds().eq(window).all()
    assert (ends - starts).dt.total_seconds().eq(window).all()
    assert ((starts.astype("int64") % (window * 1_000_000_000)) == 0).all()


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("modality", ["M1", "M2", "M3", "M4"])
def test_each_score_stream_matches_the_declared_metadata(root, settings, window, modality):
    with np.load(root / f"results/frozen_scores/A_{modality}_w{window}.npz") as scores:
        for key, _phase in PHASES:
            relative = settings["score_metadata"][f"{key}_pattern"].format(window_s=window)
            meta = pd.read_parquet(root / relative)
            assert scores[key].ndim == 1
            assert len(scores[key]) == len(meta), (modality, window, key)


def test_inventory_metadata_are_not_declared_as_score_alignment(settings):
    for key in ("attack", "benign"):
        inventory = settings["score_metadata"]["inventory_only"][key]
        score_grid = settings["score_metadata"][f"{key}_pattern"].format(window_s=60)
        assert inventory != score_grid
    assert settings["score_metadata"]["availability_masks_included"] is False
