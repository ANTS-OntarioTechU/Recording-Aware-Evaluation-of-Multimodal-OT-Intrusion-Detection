"""2. Window-grid alignment and modality availability."""
from __future__ import annotations

import pandas as pd

MASTER_ATTACK_WINDOWS = 1246
MASTER_BENIGN_WINDOWS = 3117
LABELED_ACTION_WINDOWS = 128


def test_master_grid_sizes(root):
    attack = pd.read_parquet(root / "results/frozen_scores/window_meta_attack22h.parquet")
    benign = pd.read_parquet(root / "results/frozen_scores/window_meta_benign48h.parquet")
    assert len(attack) == MASTER_ATTACK_WINDOWS
    assert len(benign) == MASTER_BENIGN_WINDOWS
    assert int((attack["label"] == 1).sum()) == LABELED_ACTION_WINDOWS
    assert int((benign["label"] == 1).sum()) == 0


def test_grid_is_ordered_and_aligned(root):
    """The master grid is 60-s aligned and ordered, with one documented telemetry gap.

    It is deliberately NOT fully contiguous: a single 960 s step spans the physical-telemetry
    outage of 2026-04-16 08:49-09:05 UTC. That gap is why the master grid holds 1,246 windows
    while the later analysis grid, rebuilt across the whole physical-state span, holds 1,261.
    These metadata are not an alignment map for the later score arrays.
    """
    attack = pd.read_parquet(root / "results/frozen_scores/window_meta_attack22h.parquet")
    starts = pd.to_datetime(attack["window_start"], utc=True)
    assert starts.is_monotonic_increasing, "master grid is not time-ordered"

    steps = starts.diff().dropna().dt.total_seconds()
    assert (steps % 60 == 0).all(), "every step must be a whole number of 60 s windows"

    gaps = steps[steps > 60]
    assert len(gaps) == 1, f"expected exactly one documented gap, found {list(gaps)}"
    assert float(gaps.iloc[0]) == 960.0, "the documented gap is 960 s (16 window slots)"

    missing_windows = int(gaps.iloc[0] / 60) - 1
    assert missing_windows == 15, (
        f"the gap should account for the 15 windows by which the Revision 5 grid exceeds the "
        f"master grid, found {missing_windows}")
    gap_start = starts[steps.index[steps > 60][0] - 1]
    assert str(gap_start.date()) == "2026-04-16", "gap is not on the documented date"


def test_availability_matches_settings(settings):
    """Availability percentages are recorded settings, not recomputed here."""
    availability = {k: v["availability_pct"] for k, v in settings["modalities"].items()
                    if isinstance(v, dict) and "availability_pct" in v}
    assert availability == {"M1": 94.9, "M2": 77.6, "M3": 94.9, "M4": 98.7}


def test_window_durations_declared(settings):
    assert settings["windowing"]["durations_seconds"] == [5, 10, 30, 60]
