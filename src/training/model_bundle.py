"""Save the fitted primary estimator together with its preprocessing state."""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler


def bundle_primary(model, benign_features, fit_index):
    """Rebuild preprocessing state only; this function never fits an IDS estimator.

    The original score_view returns an estimator and scores, but not its scaler.
    These are the same training-only median and StandardScaler operations. The
    bundle is useful only with the recorded feature order and sklearn version.
    """
    training = benign_features.to_numpy(np.float64)[fit_index]
    median = np.nanmedian(training,axis=0)
    median = np.where(np.isfinite(median),median,0.0)
    imputed = np.where(np.isfinite(training),training,median)
    scaler = StandardScaler().fit(imputed)
    return {'model':model,'median':median,'scaler':scaler,
        'feature_names':list(benign_features.columns),'score_orientation':'-score_samples',
        'sklearn_version':sklearn.__version__,'reference':'first 70% of separate benign recording'}


def score_bundle(bundle, features):
    """Score with a locally created, trusted bundle; never fit or reorder columns silently."""
    if list(features.columns) != bundle['feature_names']:
        raise ValueError('Feature names/order do not match the model bundle')
    values = features.to_numpy(np.float64)
    values = np.where(np.isfinite(values),values,bundle['median'])
    return -bundle['model'].score_samples(bundle['scaler'].transform(values))
