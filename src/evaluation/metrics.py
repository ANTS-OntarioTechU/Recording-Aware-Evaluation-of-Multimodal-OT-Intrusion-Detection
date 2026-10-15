"""Cross-recording and same-recording discrimination.

The definitions here are the ones used for the submitted paper. They are a refactor of the
archived implementation; no arithmetic is changed.

Both comparisons hold the fitted detector and the attack-positive windows fixed. Only the
background population differs:

* ``AUC_cross``  — attack-action windows vs benign-reference recording windows
* ``AUC_within`` — attack-action windows vs same-recording background outside the exclusion buffer
* ``Delta_CW``   — ``AUC_cross - AUC_within``

A positive ``Delta_CW`` means discrimination is stronger when the background comes from a separate
recording than when it comes from the same one.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

MIN_GROUP = 3  # below this a cell is reported as infeasible rather than as a number


def auc_and_lift(positive: Sequence[float], negative: Sequence[float]) -> dict[str, float]:
    """ROC-AUC, average precision and PR lift for one positive/negative split.

    Non-finite scores are dropped. Returns NaN metrics (not an exception) when either group has
    fewer than ``MIN_GROUP`` finite values, so a sparse cell is reported rather than silently
    omitted.

    PR lift is ``average_precision / positive_base_rate``, which is comparable across cells with
    different prevalence; raw PR-AUC is not.
    """
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < MIN_GROUP or len(neg) < MIN_GROUP:
        return {"auc": np.nan, "pr_auc": np.nan, "pr_lift": np.nan,
                "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    labels = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    scores = np.r_[neg, pos]
    base_rate = float(labels.mean())
    average_precision = float(average_precision_score(labels, scores))
    return {"auc": float(roc_auc_score(labels, scores)),
            "pr_auc": average_precision,
            "pr_lift": average_precision / base_rate,
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def cross_within(action_scores: Sequence[float],
                 benign_reference_scores: Sequence[float],
                 same_recording_background_scores: Sequence[float]) -> dict[str, float]:
    """The central diagnostic: identical positives scored against two backgrounds.

    ``action_scores`` must be the same array in both comparisons; passing different positive
    populations would make ``delta_cw`` meaningless.
    """
    cross = auc_and_lift(action_scores, benign_reference_scores)
    within = auc_and_lift(action_scores, same_recording_background_scores)
    gap = (cross["auc"] - within["auc"]
           if np.isfinite(cross["auc"]) and np.isfinite(within["auc"]) else np.nan)
    return {"cross_auc": cross["auc"], "cross_pr_lift": cross["pr_lift"],
            "cross_n_neg": cross["n_neg"],
            "within_auc": within["auc"], "within_pr_lift": within["pr_lift"],
            "within_n_neg": within["n_neg"],
            "n_action": cross["n_pos"], "delta_cw": gap}


def separability(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """Direction-invariant separability, ``max(AUC, 1 - AUC)``.

    Used for the external HAI check, where recording separability has no intrinsic direction but
    an anomaly score does.
    """
    auc = auc_and_lift(scores_a, scores_b)["auc"]
    return float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
