"""8. Schema and leakage guard."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.schema_guard import REVIEWED_EXEMPTIONS, enforce, guard_frame, inspect

BANNED = ["attack", "attacker", "label", "phase", "ground_truth", "target",
          "old_value", "new_value", "scenario", "file", "split",
          "src_ip", "dst_ip", "mac_addr", "src_port", "trans_id", "transaction_id",
          "capture_path", "proc_pid", "ppid", "event_id", "sha256_hash",
          "exe_path", "cmdline", "window_start", "timestamp", "m1__present", "availability_m2"]


@pytest.mark.parametrize("column", BANNED)
def test_every_banned_column_is_rejected(column):
    assert column in inspect([column]).rejected, f"{column} was not rejected"


def test_underscore_does_not_defeat_the_boundary():
    """`\\bip\\b` does not fire on dst_ip and `\\bpid\\b` does not fire on proc_pid."""
    for column in ("dst_ip", "proc_pid", "src_port", "trans_id"):
        assert column in inspect([column]).rejected


def test_real_paper_features_are_all_accepted(root):
    """The 2,100 feature columns used for the submitted paper must survive the guard."""
    listing = json.loads(
        (root / "configs/frozen_specs/feature_lists_post_screening.json").read_text())
    columns = [c for key, value in listing.items() if key != "_meta" for c in value["features"]]
    assert len(columns) > 2000
    report = inspect(columns)
    assert not report.rejected, f"guard rejected real features: {sorted(report.rejected)[:10]}"


def test_exemptions_are_documented():
    for column, reason in REVIEWED_EXEMPTIONS.items():
        assert len(reason) > 40, f"{column} needs a written justification"


def test_guard_frame_fails_closed():
    frame = pd.DataFrame({"m3_n": [1], "label": [0]})
    with pytest.raises(AssertionError):
        guard_frame(frame, "unit test")


def test_guard_reports_are_readable():
    report = inspect(["m3_n", "label", "m2_pnames_sshd-session"])
    text = report.to_text("unit test")
    assert "REJECT" in text and "EXEMPT" in text


def test_enforce_names_the_matrix():
    with pytest.raises(AssertionError, match="M2 attack w60"):
        enforce(["attack_flag"], "M2 attack w60")
