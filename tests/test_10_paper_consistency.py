"""10. Paper-result consistency against the machine-readable expected-results file.

Each entry names a paper location, the source artifact, the expected value and a tolerance that
reflects the rounding used in the manuscript. A regenerated value that drifts beyond tolerance
fails the suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _regenerate(entry_id: str, tables) -> object:
    """Recompute one expected value from its source artifact."""
    if entry_id in {"delta_cw_range", "m4_delta_cw_range", "pre_attack_cyber_auc",
                    "pre_attack_m4_auc"}:
        summary = pd.read_csv(tables / "window_sensitivity_summary.csv")
        cells = summary.groupby(["window_s", "modality"]).agg(
            dcw=("delta_cw", "mean"), pre=("pre_auc", "mean")).reset_index()
        if entry_id == "delta_cw_range":
            return [round(float(cells.dcw.min()), 3), round(float(cells.dcw.max()), 3)]
        if entry_id == "m4_delta_cw_range":
            m4 = cells[cells.modality == "M4"]
            return [round(float(m4.dcw.min()), 3), round(float(m4.dcw.max()), 3)]
        subset = cells[cells.modality != "M4"] if entry_id == "pre_attack_cyber_auc" \
            else cells[cells.modality == "M4"]
        return [round(float(subset.pre.min()), 3), round(float(subset.pre.max()), 3)]

    if entry_id == "membership_probe_auc":
        probe = pd.read_csv(tables / "session_probe_results.csv")
        subset = probe[(probe.modality != "M4")
                       & probe.model.isin(["logistic_regression", "random_forest"])]
        return round(float(subset.auc.min()), 3)

    if entry_id in {"cyber_only_gap", "all_view_gap", "all_view_occupancy"}:
        anchor = pd.read_csv(tables / "physical_anchor_comparison.csv")
        if entry_id == "cyber_only_gap":
            return [round(float(anchor[anchor.window_s == w].cyber_delta_cw.iloc[0]), 3)
                    for w in (5, 10, 30, 60)]
        if entry_id == "all_view_gap":
            return round(float(anchor.fused_delta_cw.mean()), 2)
        return [round(float(anchor.fused_occupancy.min()), 2),
                round(float(anchor.fused_occupancy.max()), 2)]

    if entry_id in {"any_overlap_vs_one_to_one", "longest_episode_windows"}:
        matching = pd.read_csv(tables / "event_matching_sensitivity.csv")
        at60 = matching[matching.window_s == 60]
        if entry_id == "any_overlap_vs_one_to_one":
            recall = at60.groupby(["family", "rule"]).event_recall.mean().unstack()
            return [round(float(recall.loc["sum_z_fusion", "any_overlap"]), 3),
                    round(float(recall.loc["sum_z_fusion", "one_to_one"]), 3)]
        row = at60[(at60.family == "sum_z_fusion") & (at60.rule == "any_overlap")]
        return int(round(float(row.max_episode_windows.mean())))

    if entry_id in {"q_align_min", "q_align_m3_cyber"}:
        nulls = pd.read_csv(tables / "temporal_nulls.csv")
        at60 = nulls[(nulls.window_s == 60) & (nulls.null == "duration_matched")]
        by_family = at60.groupby("family").p_empirical.mean()
        if entry_id == "q_align_min":
            return round(float(by_family.loc["sum_z_fusion"]), 6)
        return [round(float(by_family.loc["M3"]), 4),
                round(float(by_family.loc["cyber_only_fusion"]), 4)]

    if entry_id == "bootstrap_occupancy":
        boot = pd.read_csv(tables / "calibration_bootstrap.csv")
        out = []
        for family in ("cyber_only_fusion", "sum_z_fusion"):
            values = boot[(boot.family == family) & (boot.budget_q == 0.014)].outside_action_occupancy
            out.append([round(float(np.percentile(values, 2.5)), 2),
                        round(float(np.percentile(values, 97.5)), 2)])
        return out

    if entry_id in {"smd_post_match", "physical_state_probe"}:
        support = pd.read_csv(tables / "operating_support_summary.csv")
        if entry_id == "smd_post_match":
            return [round(float(support.max_abs_smd_post.min()), 2),
                    round(float(support.max_abs_smd_post.max()), 2)]
        return [round(float(support.propensity_separation_auc.min()), 3),
                round(float(support.propensity_separation_auc.max()), 3)]

    if entry_id in {"buffer_sensitivity_positive", "buffer_max_shift"}:
        buffer = pd.read_csv(tables / "supervisor_buffer_sensitivity_summary.csv")
        pivot = buffer.pivot_table(index=["window_s", "modality"], columns="buffer_s",
                                   values="median_delta_cw")
        if entry_id == "buffer_sensitivity_positive":
            return int((pivot.min(axis=1) > 0).sum())
        return round(float((pivot[900] - pivot[300]).abs().max()), 4)

    if entry_id in {"hai_eligible", "hai_sign_changes"}:
        hai = pd.read_csv(tables / "hai_portable_core_full.csv")
        eligible = hai[hai.status == "complete"]
        if entry_id == "hai_eligible":
            return [int(len(eligible)), int(len(hai))]
        return int(sum(1 for _, group in eligible.groupby("target")
                       if not ((group.delta_cw > 0).all() or (group.delta_cw < 0).all())))

    if entry_id == "synthetic_control":
        synthetic = pd.read_csv(tables / "e2_synthetic_session_confounding.csv")
        return [round(float(synthetic[synthetic.lam == 0].auc_cross_mean.iloc[0]), 3),
                round(float(synthetic[synthetic.lam == 2].auc_cross_mean.iloc[0]), 3),
                round(float(synthetic.auc_within_mean.min()), 3),
                round(float(synthetic.auc_within_mean.max()), 3)]

    raise KeyError(f"no regeneration rule for {entry_id}")


def _close(actual, target, tolerance) -> bool:
    if isinstance(target, list):
        return (len(actual) == len(target)
                and all(_close(a, t, tolerance) for a, t in zip(actual, target)))
    return abs(float(actual) - float(target)) <= tolerance


def test_expected_results_file_is_complete(expected):
    assert len(expected["entries"]) >= 20
    for entry in expected["entries"]:
        for field in ("id", "paper", "source", "quantity", "expected", "tol", "paper_text"):
            assert field in entry, f"{entry.get('id')} missing {field}"


@pytest.mark.parametrize("entry_id", [
    "delta_cw_range", "m4_delta_cw_range", "pre_attack_cyber_auc", "pre_attack_m4_auc",
    "membership_probe_auc", "cyber_only_gap", "all_view_gap", "all_view_occupancy",
    "any_overlap_vs_one_to_one", "longest_episode_windows", "q_align_min", "q_align_m3_cyber",
    "bootstrap_occupancy", "smd_post_match", "physical_state_probe",
    "buffer_sensitivity_positive", "buffer_max_shift", "hai_eligible", "hai_sign_changes",
    "synthetic_control"])
def test_paper_value_still_holds(entry_id, expected, tables):
    entry = next(e for e in expected["entries"] if e["id"] == entry_id)
    actual = _regenerate(entry_id, tables)
    assert _close(actual, entry["expected"], entry["tol"]), (
        f"{entry_id} ({entry['paper']}): regenerated {actual} vs recorded {entry['expected']} "
        f"(tolerance {entry['tol']})")


def test_buffer_reporting_matches_corrected_paper_text(expected):
    """The rounded bound in the current paper agrees with the unchanged artifact value."""
    entry = next(e for e in expected["entries"] if e["id"] == "buffer_max_shift")
    assert entry["paper_text"] == "changes the median gap by less than 0.009"
    assert "paper_discrepancy" not in entry
    assert entry["expected"] == 0.0088
    assert "0.008827" in entry["reporting_note"]
