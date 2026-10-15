"""Original primary-detector fitting functions, with no module-level execution.

The estimator parameters, training-median imputation, scaling and score orientation
are unchanged. This is a retrospective separate-benign-reference fit, not a
strictly calendar-forward reference. The first benign 70% is reused for calibration.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


SEEDS = (42, 43, 44, 45, 46)


IF_PARAMS = dict(n_estimators=300, max_samples=256, bootstrap=False)


BENIGN_TRAIN_FRAC = 0.70


MODALITIES = ("M1", "M2", "M3", "M4")


def _prep_fit(Xtr, Xall):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    def imp(A):
        A = np.where(np.isfinite(A), A, med)
        return A
    sc = StandardScaler().fit(imp(Xtr))
    return sc.transform(imp(Xtr)), sc.transform(imp(Xall))


def score_view(Xa_df, Xb_df, tr_b, seed):
    """Fit on benign TRAIN rows; score attack rows and benign rows."""
    Xb = Xb_df.to_numpy(np.float64); Xa = Xa_df.to_numpy(np.float64)
    Ztr, Zb = _prep_fit(Xb[tr_b], Xb)
    _,  Za = _prep_fit(Xb[tr_b], Xa)
    clf = IsolationForest(random_state=seed, **IF_PARAMS).fit(Ztr)
    return -clf.score_samples(Za), -clf.score_samples(Zb), Ztr, clf
