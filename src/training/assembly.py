"""Original cyber feature assembly on a supplied metadata grid."""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from .featurize import SCALAR_M3, SCALAR_PROV, DICT_M3, DICT_PROV, fit_vocab, build_matrix


SPEC = {"M1": (SCALAR_PROV, DICT_PROV), "M2": (SCALAR_PROV, DICT_PROV),
        "M3": (SCALAR_M3, DICT_M3)}


def epoch_index(meta: pd.DataFrame, w: int) -> np.ndarray:
    """Map each master window to the absolute epoch window index used by the streamers."""
    ts = meta["window_start"].astype("int64").to_numpy() // 10**9
    return (ts // w).astype(np.int64)


def load_records(win_dir: str, mod: str, phase: str, w: int) -> dict:
    p = os.path.join(win_dir, "%s_%s_w%d.json" % (mod, phase, w))
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def fit_vocabs(records: dict, mod: str, train_keys) -> dict:
    """Fit categorical vocabularies on TRAIN windows only."""
    _, dicts = SPEC[mod]
    sub = {k: records[k] for k in train_keys if k in records}
    return {field: fit_vocab(sub, field, top_k) for field, top_k in dicts.items()}


def build_view(records: dict, meta: pd.DataFrame, w: int, mod: str, vocabs: dict) -> pd.DataFrame:
    scalars, dicts = SPEC[mod]
    idx = epoch_index(meta, w)
    return build_matrix(records, idx, scalars, dicts, vocabs, prefix=mod.lower())


def availability(records: dict, meta: pd.DataFrame, w: int) -> np.ndarray:
    idx = epoch_index(meta, w)
    return np.array([1.0 if str(i) in records else 0.0 for i in idx])
