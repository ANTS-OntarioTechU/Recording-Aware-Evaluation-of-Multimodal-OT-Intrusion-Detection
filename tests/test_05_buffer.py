"""5. The 300-second same-recording buffer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.background import non_action_mask


def test_settings_declare_300s_buffer(settings):
    assert settings["windowing"]["same_recording_background_buffer_seconds"] == 300
    assert "seconds" in settings["windowing"]["buffer_scaling"]


def test_buffer_excludes_windows_adjacent_to_actions(root):
    """No window within 300 s of a labeled action may survive into the background pool."""
    meta = pd.read_parquet(root / "results/frozen_scores/window_meta_attack22h.parquet")
    events = pd.read_csv(root / "data_manifest/ground_truth_events.csv",
                         parse_dates=["start", "end"])
    mask = non_action_mask(meta, events, buffer_seconds=300)
    assert not (meta["label"].values[mask] == 1).any(), "labeled window survived the buffer"

    starts = pd.to_datetime(meta["window_start"], utc=True).values
    ends = pd.to_datetime(meta["window_end"], utc=True).values
    span = np.timedelta64(300, "s")
    for _, event in events.iterrows():
        onset = pd.Timestamp(event["start"]).to_datetime64()
        offset = pd.Timestamp(event["end"]).to_datetime64()
        too_close = (ends > onset - span) & (starts < offset + span)
        assert not (too_close & mask).any(), f"window within 300 s of {event['event_id']} retained"


def test_widening_the_buffer_shrinks_the_background(root):
    meta = pd.read_parquet(root / "results/frozen_scores/window_meta_attack22h.parquet")
    events = pd.read_csv(root / "data_manifest/ground_truth_events.csv",
                         parse_dates=["start", "end"])
    counts = [non_action_mask(meta, events, b).sum() for b in (300, 600, 900)]
    assert counts[0] > counts[1] > counts[2], f"background did not shrink monotonically: {counts}"
