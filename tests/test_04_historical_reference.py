"""4. Action screening in the alternative historical-reference construction audit."""
from __future__ import annotations

import pandas as pd


def test_no_labeled_windows_in_fit_or_calibration(root):
    audit = pd.read_csv(root / "results/paper_tables/v6_forward_partition_audit.csv")
    assert (audit["labelled_in_fit"] == 0).all(), "labeled attack windows leaked into fitting"
    assert (audit["labelled_in_cal"] == 0).all(), "labeled attack windows leaked into calibration"


def test_reference_totals_match_split_sizes(root):
    audit = pd.read_csv(root / "results/paper_tables/v6_forward_partition_audit.csv")
    declared = {"C2": (278, 120), "C3": (358, 154), "C4": (396, 170)}
    for row in audit.itertuples():
        n_fit, n_cal = declared[row.held_out]
        assert row.n_fit == n_fit and row.n_cal == n_cal
        assert row.n_fit + row.n_cal == row.n_tr_neg_total


def test_primary_reference_is_distinct_from_historical_audit(settings):
    reference = settings["primary_reference"]
    assert reference["recording"] == "separate_benign"
    assert reference["fit_fraction"] == 0.70
    assert reference["calibration"] == "benign_fitting_partition"
    assert reference["refit_between_campaigns"] is False
    assert settings["historical_reference_audit"]["role"] == (
        "alternative_reference_not_primary_fitting_partitions")
