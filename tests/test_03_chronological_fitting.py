"""3. Chronology of the alternative historical-reference construction audit.

These counts do not describe the benign-trained headline detector partitions.
"""
from __future__ import annotations

import pandas as pd


def test_calibration_precedes_heldout_campaign(root):
    """Calibration must close before the held-out campaign begins, for every forward step."""
    audit = pd.read_csv(root / "results/paper_tables/v6_forward_partition_audit.csv")
    assert len(audit) == 3, "expected three forward steps"
    for row in audit.itertuples():
        cal_last = pd.Timestamp(row.cal_last)
        heldout_start = pd.Timestamp(row.heldout_start)
        assert cal_last <= heldout_start, (
            f"{row.held_out}: calibration ends {cal_last} but the held-out campaign "
            f"starts {heldout_start}")


def test_fitting_precedes_calibration(root):
    audit = pd.read_csv(root / "results/paper_tables/v6_forward_partition_audit.csv")
    for row in audit.itertuples():
        assert pd.Timestamp(row.fit_last) <= pd.Timestamp(row.cal_first), (
            f"{row.held_out}: fitting window overlaps calibration")


def test_forward_campaigns_declared(settings):
    assert settings["forward_campaigns"] == ["C2", "C3", "C4"]
