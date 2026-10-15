"""6. Numerical discrimination metrics; alarm-rerun helpers are not released."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import auc_and_lift, cross_within


def test_delta_cw_is_cross_minus_within():
    rng = np.random.default_rng(0)
    positives = rng.normal(2, 1, 80)
    result = cross_within(positives, rng.normal(0, 1, 400), rng.normal(1.2, 1, 300))
    assert result["delta_cw"] == pytest.approx(result["cross_auc"] - result["within_auc"])


def test_sparse_cell_is_reported_not_dropped():
    out = auc_and_lift([1.0, 2.0], [0.0, 0.5, 0.7])
    assert np.isnan(out["auc"]) and out["n_pos"] == 2, "small groups report NaN with counts"
