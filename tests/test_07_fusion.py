"""7. Fusion-score construction."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.fusion import (ALL_VIEWS, CYBER_VIEWS, all_view_sum_z, cyber_only_sum_z,
                                 late_or_max_z, standardise)


def test_standardisation_uses_calibration_statistics_only():
    calibration = np.array([10.0, 12.0, 14.0])
    scores = np.array([12.0, 20.0])
    z = standardise(scores, calibration)
    assert z[0] == pytest.approx(0.0), "calibration mean maps to zero"
    assert z[1] > 4.0, "a held-out value far above calibration must not be rescaled by itself"


def test_cyber_only_excludes_the_physical_view():
    z = {"M1": np.array([1.0]), "M2": np.array([2.0]), "M3": np.array([3.0]),
         "M4": np.array([100.0])}
    assert cyber_only_sum_z(z)[0] == pytest.approx(6.0)
    assert all_view_sum_z(z)[0] == pytest.approx(106.0)


def test_late_or_is_the_maximum():
    z = {"M1": np.array([1.0, 5.0]), "M2": np.array([4.0, 2.0]),
         "M3": np.array([0.0, 0.0]), "M4": np.array([-1.0, 3.0])}
    assert list(late_or_max_z(z)) == [4.0, 5.0]


def test_view_sets_match_the_settings(settings):
    assert CYBER_VIEWS == ("M1", "M2", "M3")
    assert ALL_VIEWS == ("M1", "M2", "M3", "M4")
    assert "M1, M2, M3" in settings["fusion"]["cyber_only_sum_z"]
    assert "M1, M2, M3, M4" in settings["fusion"]["all_view_sum_z"]


def test_zero_variance_calibration_does_not_produce_infinities():
    z = standardise([1.0, 2.0], [5.0, 5.0, 5.0])
    assert np.isfinite(z).all()


def test_empty_calibration_raises():
    with pytest.raises(ValueError):
        standardise([1.0], [np.nan, np.nan])
