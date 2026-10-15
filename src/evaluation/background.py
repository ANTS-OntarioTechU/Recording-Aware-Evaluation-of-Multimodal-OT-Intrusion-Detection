"""Exclude labeled actions and their fixed temporal buffer from background windows.

Longer persistence or recovery can remain unlabeled outside this buffer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_naive_datetime64(value) -> np.datetime64:
    """Convert to timezone-naive UTC ``datetime64[ns]``."""
    stamp = pd.Timestamp(value)
    if stamp.tz is not None:
        stamp = stamp.tz_convert(None)
    return stamp.to_datetime64()


def non_action_mask(window_meta: pd.DataFrame, events: pd.DataFrame,
                    buffer_seconds: int = 300) -> np.ndarray:
    """Retain unlabeled windows outside the fixed buffer around labeled events."""
    starts = window_meta["window_start"].values
    ends = window_meta["window_end"].values
    keep = window_meta["label"].values == 0
    span = np.timedelta64(int(buffer_seconds), "s")
    for _, event in events.iterrows():
        onset = _as_naive_datetime64(event["start"])
        offset = _as_naive_datetime64(event["end"])
        keep &= ~((ends > onset - span) & (starts < offset + span))
    return keep
