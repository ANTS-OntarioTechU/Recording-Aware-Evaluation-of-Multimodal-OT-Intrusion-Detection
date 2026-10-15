"""Original recording-membership probe and optional attribution functions.

The target is recording membership, not attack activity. Fitting is explicit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from . import splits as sp5


TRAIN_FRAC = 0.70


FAMILIES = {
 "event_rate":      ["_n_edges","_n_vertices","_edges_per_vertex","_n_new_vertices","_new_vertex_rate",
                     "_n$","_req","_resp","n_samples_m4"],
 "timing":          ["_dt_mean","_dt_std","_dt_med","_dt_p95","_dt_max","_burstiness","_fano","_subsec_active"],
 "topology_degree": ["_outdeg_","_indeg_","_proc_frac","_root_frac"],
 "operation_mix":   ["_ops_","_etypes_","_op_entropy","_n_distinct_ops"],
 "modbus_fn_addr":  ["_fc_","_reg_","_reg_entropy","_reg_min","_reg_max","_n_distinct_reg","_read_frac",
                     "_write_frac","_wc_","_n_units","_n_client_ports","_tid_reset_frac","_tid_span"],
 "process_summary": ["_pnames_","_pbuckets_","_subtypes_","_name_entropy","_bucket_entropy",
                     "_subtype_entropy","_n_distinct_names"],
 "vocab_oov":       ["__oov","__ndistinct"],
 "physical_state":  ["m4_"],
}


def assign_family(col: str) -> str:
    import re
    for fam, pats in FAMILIES.items():
        for p in pats:
            if p.endswith("$"):
                if re.search(p, col):
                    return fam
            elif p in col:
                return fam
    return "other"


def models(seed: int) -> dict:
    return {
      "logistic_regression": ("scale", LogisticRegression(C=1.0, class_weight="balanced",
                                                          max_iter=2000, random_state=seed)),
      "random_forest":       ("raw",   RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                              class_weight="balanced_subsample",
                                                              random_state=seed, n_jobs=-1)),
      "mlp":                 ("scale", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                                     solver="adam", max_iter=500,
                                                     early_stopping=True, random_state=seed)),
    }


def _prep(Xtr, Xte, how):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0.0)
    f = lambda A: np.where(np.isfinite(A), A, med)
    if how == "scale":
        sc = StandardScaler().fit(f(Xtr))
        return sc.transform(f(Xtr)), sc.transform(f(Xte))
    return f(Xtr), f(Xte)


def session_probe(Xa_pre: pd.DataFrame, Xb: pd.DataFrame, seed: int = 42) -> tuple:
    """Time-blocked probe. Returns (results_df, fitted_artifacts)."""
    na, nb = len(Xa_pre), len(Xb)
    tra, tea = sp5.time_blocked_split(na, TRAIN_FRAC)
    trb, teb = sp5.time_blocked_split(nb, TRAIN_FRAC)
    cols = list(Xa_pre.columns)
    A = Xa_pre.to_numpy(np.float64); B = Xb[cols].to_numpy(np.float64)
    Xtr = np.vstack([B[trb], A[tra]]); ytr = np.r_[np.zeros(len(trb)), np.ones(len(tra))]
    Xte = np.vstack([B[teb], A[tea]]); yte = np.r_[np.zeros(len(teb)), np.ones(len(tea))]
    rows, art = [], {}
    for name, (how, clf) in models(seed).items():
        Ztr, Zte = _prep(Xtr, Xte, how)
        clf.fit(Ztr, ytr)
        s = clf.predict_proba(Zte)[:, 1]
        base = float(yte.mean()); ap = float(average_precision_score(yte, s))
        pred = (s >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        rows.append(dict(model=name, seed=seed, auc=float(roc_auc_score(yte, s)),
                         pr_auc=ap, pr_lift=ap / base, base_rate=base,
                         n_train=len(ytr), n_test=len(yte),
                         n_test_pos=int(yte.sum()), n_test_neg=int((yte == 0).sum()),
                         tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                         score_mean_pos=float(s[yte == 1].mean()), score_mean_neg=float(s[yte == 0].mean()),
                         n_features=len(cols)))
        art[name] = dict(model=clf, scores=s, y=yte, prep=how)
    return pd.DataFrame(rows), art, (Xtr, ytr, Xte, yte, cols)


def family_ablation(Xtr, ytr, Xte, yte, cols, seed: int = 42) -> pd.DataFrame:
    """Held-out AUC when ONE family is removed, and when ONLY that family is kept."""
    fam = np.array([assign_family(c) for c in cols])
    rows = []
    full_h, full = _prep(Xtr, Xte, "scale")
    base_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed).fit(full_h, ytr)
    base_auc = float(roc_auc_score(yte, base_clf.predict_proba(full)[:, 1]))
    for f in sorted(set(fam)):
        for mode, sel in (("drop", fam != f), ("only", fam == f)):
            if sel.sum() < 2:
                continue
            Ztr, Zte = _prep(Xtr[:, sel], Xte[:, sel], "scale")
            c = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed).fit(Ztr, ytr)
            a = float(roc_auc_score(yte, c.predict_proba(Zte)[:, 1]))
            rows.append(dict(family=f, mode=mode, n_features=int(sel.sum()),
                             auc=a, full_auc=base_auc, delta_vs_full=a - base_auc, seed=seed))
    return pd.DataFrame(rows)


def permutation_importance_grouped(Xtr, ytr, Xte, yte, cols, seed: int = 42,
                                   n_repeat: int = 5) -> pd.DataFrame:
    """Held-out grouped permutation importance (semantic family, not individual features)."""
    rng = np.random.default_rng(seed)
    fam = np.array([assign_family(c) for c in cols])
    Ztr, Zte = _prep(Xtr, Xte, "scale")
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed).fit(Ztr, ytr)
    base = float(roc_auc_score(yte, clf.predict_proba(Zte)[:, 1]))
    rows = []
    for f in sorted(set(fam)):
        sel = fam == f
        drops = []
        for r in range(n_repeat):
            Z = Zte.copy()
            idx = rng.permutation(len(Z))
            Z[:, sel] = Z[idx][:, sel]
            drops.append(base - float(roc_auc_score(yte, clf.predict_proba(Z)[:, 1])))
        rows.append(dict(family=f, n_features=int(sel.sum()), base_auc=base,
                         importance_mean=float(np.mean(drops)), importance_sd=float(np.std(drops)),
                         seed=seed, n_repeat=n_repeat))
    return pd.DataFrame(rows)
