"""Original counter-to-feature definitions; vocabularies use training windows only."""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import Counter


SCALAR_M3 = ["n","req","resp","n_client_ports","n_units","n_distinct_reg","reg_min","reg_max",
             "reg_entropy","read_frac","write_frac","wc_mean","wc_max","wc_std","dt_mean",
             "dt_std","dt_med","dt_p95","dt_max","burstiness","fano","subsec_active",
             "tid_reset_frac","tid_span"]


SCALAR_PROV = ["n_edges","n_vertices","edges_per_vertex","new_vertex_rate","n_new_vertices",
               "proc_frac","root_frac","outdeg_mean","outdeg_max","outdeg_std","indeg_mean",
               "indeg_max","indeg_std","op_entropy","name_entropy","bucket_entropy",
               "subtype_entropy","n_distinct_ops","n_distinct_names","dt_mean","dt_std","dt_max"]


DICT_M3   = {"fc": 16, "reg": 32}


DICT_PROV = {"ops": 24, "etypes": 6, "pnames": 30, "pbuckets": 17, "subtypes": 8}


def fit_vocab(records: dict, field: str, top_k: int) -> list:
    c = Counter()
    for r in records.values():
        c.update({str(k): v for k, v in r.get(field, {}).items()})
    return [k for k, _ in c.most_common(top_k)]


def build_matrix(records: dict, index: np.ndarray, scalars: list, dicts: dict,
                 vocabs: dict, prefix: str) -> pd.DataFrame:
    """`index` = master window indices; missing windows -> NaN scalars, 0 counts, present=0."""
    cols, data = [], []
    for s in scalars:
        cols.append(f"{prefix}_{s}")
        data.append(np.array([records.get(str(i), records.get(i, {})).get(s, np.nan) for i in index], float))
    for field in dicts:
        voc = vocabs[field]
        mats = np.zeros((len(index), len(voc) + 2), float)
        for r_i, i in enumerate(index):
            d = records.get(str(i), records.get(i, {})).get(field, {})
            if not d:
                continue
            tot = sum(d.values()) or 1
            for j, k in enumerate(voc):
                mats[r_i, j] = d.get(k, d.get(int(k) if str(k).lstrip("-").isdigit() else k, 0))
            mats[r_i, -2] = tot - mats[r_i, :-2].sum()      # out-of-vocabulary mass
            mats[r_i, -1] = len(d)                           # distinct categories
        for j, k in enumerate(voc):
            cols.append(f"{prefix}_{field}_{k}"); data.append(mats[:, j])
        cols.append(f"{prefix}_{field}__oov"); data.append(mats[:, -2])
        cols.append(f"{prefix}_{field}__ndistinct"); data.append(mats[:, -1])
        # normalised shares are scale-free and travel better across recordings
        tot = mats[:, :-1].sum(1, keepdims=True); tot[tot == 0] = 1
        for j, k in enumerate(voc):
            cols.append(f"{prefix}_{field}_{k}__frac"); data.append(mats[:, j] / tot[:, 0])
    df = pd.DataFrame(np.vstack(data).T, columns=cols)
    df[f"{prefix}__present"] = np.array(
        [1.0 if (str(i) in records or i in records) else 0.0 for i in index])
    return df
