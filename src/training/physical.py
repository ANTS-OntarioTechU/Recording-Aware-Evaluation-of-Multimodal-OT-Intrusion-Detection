"""Original physical feature construction.

22 physical variables x 9 summaries plus sample count = 199 features, as in the
released feature-list artifact. This is not a mean-only 22-feature replacement.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple


def m4_window_features(df: pd.DataFrame, meta: pd.DataFrame, cols: List[str],
                       w: int, pair_cols: List[Tuple[str, str]] | None = None) -> pd.DataFrame:
    """Causal within-window M4 features. No timestamps, no ids, no labels leak in."""
    t = df["_time"].values
    idx = np.searchsorted(t, meta["window_start"].values, "left")
    idx_e = np.searchsorted(t, meta["window_end"].values, "left")
    X = np.full((len(meta), len(cols) * 9 + 1), np.nan, dtype=np.float32)
    V = df[cols].to_numpy(np.float64)
    for i, (a, b) in enumerate(zip(idx, idx_e)):
        if b <= a:
            X[i, -1] = 0.0
            continue
        seg = V[a:b]
        d = np.diff(seg, axis=0) if b - a > 1 else np.zeros((1, seg.shape[1]))
        blk = np.concatenate([
            np.nanmean(seg, 0), np.nanstd(seg, 0),
            np.nanmin(seg, 0), np.nanmax(seg, 0),
            np.nanmedian(seg, 0),
            np.nanpercentile(seg, 75, axis=0) - np.nanpercentile(seg, 25, axis=0),
            seg[-1] - seg[0],
            np.nanmean(np.abs(d), 0), np.nanmax(np.abs(d), 0),
        ])
        X[i, :-1] = blk
        X[i, -1] = b - a
    names = ([f"{c}__{s}" for s in ["mean","std","min","max","med","iqr","delta","mad","maxjump"] for c in cols]
             + ["n_samples_m4"])
    return pd.DataFrame(X, columns=names, index=meta.index)
