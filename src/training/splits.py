"""Original campaign and within-recording partition definitions."""
from __future__ import annotations
import numpy as np
import pandas as pd


FORWARD = [("C2", ["C1"]), ("C3", ["C1", "C2"]), ("C4", ["C1", "C2", "C3"])]


def campaign_of_window(meta: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """Assign every attack-recording window to the campaign whose SPAN contains it.

    Campaign span = [first event start, last event end] of that campaign. Windows between
    campaigns are assigned to the preceding campaign's tail so the forward split is a clean
    chronological cut with no unassigned gap.
    """
    spans = (events.groupby("campaign_id")
                   .agg(s=("start", "min"), e=("end", "max"))
                   .sort_values("s"))
    ws = meta["window_start"].values
    out = np.array([""] * len(meta), dtype=object)
    order = list(spans.index)
    bounds = [pd.Timestamp(x).to_datetime64() for x in spans["s"]]
    for i, c in enumerate(order):
        lo = bounds[i]
        # NOTE: no far-future sentinel -- datetime64[ns] saturates above ~2262-04-11, which
        # silently made the final campaign's mask empty. The last campaign is left open-ended.
        sel = (ws >= lo) if i + 1 == len(order) else ((ws >= lo) & (ws < bounds[i + 1]))
        out[sel] = c
    out[ws < bounds[0]] = "PRE"          # before any campaign = pre-action segment
    return out


def forward_split(meta: pd.DataFrame, events: pd.DataFrame, step: str) -> dict:
    """Train on all campaigns strictly before `step`; hold `step` out."""
    camp = campaign_of_window(meta, events)
    train_c = dict(FORWARD)[step]
    tr = np.isin(camp, train_c)
    ho = camp == step
    return dict(step=step, train_campaigns=train_c, campaign=camp,
                train_idx=np.flatnonzero(tr), holdout_idx=np.flatnonzero(ho),
                n_train=int(tr.sum()), n_holdout=int(ho.sum()))


def time_blocked_split(n: int, train_frac: float = 0.70) -> tuple:
    """Chronological cut inside ONE recording. Never shuffles adjacent windows."""
    k = int(round(n * train_frac))
    return np.arange(0, k), np.arange(k, n)


def pre_action_mask(meta: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """Attack-recording windows ending before the FIRST labeled event."""
    t0 = pd.Timestamp(events["start"].min()).to_datetime64()
    return meta["window_end"].values < t0
