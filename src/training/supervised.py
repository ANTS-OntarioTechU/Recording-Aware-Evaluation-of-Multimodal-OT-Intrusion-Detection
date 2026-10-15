"""Original supervised diagnostic: LR, RF and MLP, with earlier attack campaigns.

The separate benign reference remains retrospective in calendar time. Calling
supervised_forward explicitly fits models; importing this module does not.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from . import splits as sp5


def _models(seed):
    return {"logistic_regression": ("scale", LogisticRegression(C=1.0, class_weight="balanced",
                                                                max_iter=2000, random_state=seed)),
            "random_forest": ("raw", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                            class_weight="balanced_subsample",
                                                            random_state=seed, n_jobs=-1)),
            "mlp": ("scale", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                           early_stopping=True, random_state=seed))}


def _prep(Xtr, Xte, how):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0.0)
    f = lambda A: np.where(np.isfinite(A), A, med)
    if how == "scale":
        sc = StandardScaler().fit(f(Xtr))
        return sc.transform(f(Xtr)), sc.transform(f(Xte))
    return f(Xtr), f(Xte)


def supervised_forward(Xa, Xb, meta_a, events, camp, lab, bg, benign_train_idx, benign_eval_idx,
                       modality, w, seed=42) -> pd.DataFrame:
    """Train on labeled action windows of PAST campaigns vs benign train; evaluate held-out step
    against BOTH negative populations."""
    A = Xa.to_numpy(np.float64); Bm = Xb[list(Xa.columns)].to_numpy(np.float64)
    rows = []
    for step, train_c in sp5.FORWARD:
        tr_pos = lab & np.isin(camp, train_c)
        if tr_pos.sum() < 5:
            continue
        Xtr = np.vstack([Bm[benign_train_idx], A[tr_pos]])
        ytr = np.r_[np.zeros(len(benign_train_idx)), np.ones(int(tr_pos.sum()))]
        ho_pos = lab & (camp == step)
        ho_bg  = bg  & (camp == step)
        if ho_pos.sum() < 3:
            continue
        for name, (how, clf) in _models(seed).items():
            Xte_cross = np.vstack([Bm[benign_eval_idx], A[ho_pos]])
            yte_cross = np.r_[np.zeros(len(benign_eval_idx)), np.ones(int(ho_pos.sum()))]
            Xte_within = np.vstack([A[ho_bg], A[ho_pos]])
            yte_within = np.r_[np.zeros(int(ho_bg.sum())), np.ones(int(ho_pos.sum()))]
            Ztr, Zc = _prep(Xtr, Xte_cross, how)
            clf.fit(Ztr, ytr)
            sc = clf.predict_proba(Zc)[:, 1]
            _, Zw = _prep(Xtr, Xte_within, how)
            sw = clf.predict_proba(Zw)[:, 1]
            def m(y, s):
                if len(np.unique(y)) < 2:
                    return np.nan, np.nan
                ap = float(average_precision_score(y, s))
                return float(roc_auc_score(y, s)), ap / float(y.mean())
            ca, cl = m(yte_cross, sc); wa, wl = m(yte_within, sw)
            rows.append(dict(window_s=w, modality=modality, model=name, seed=seed, step=step,
                             n_train_pos=int(tr_pos.sum()), n_holdout_pos=int(ho_pos.sum()),
                             n_holdout_bg=int(ho_bg.sum()),
                             cross_auc=ca, cross_pr_lift=cl, within_auc=wa, within_pr_lift=wl,
                             delta_cw=(ca - wa) if np.isfinite(ca) and np.isfinite(wa) else np.nan))
    return pd.DataFrame(rows)
