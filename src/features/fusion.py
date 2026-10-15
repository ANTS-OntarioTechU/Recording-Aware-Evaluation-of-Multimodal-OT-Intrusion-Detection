"""Score-level fusion families.

All four fusions are built from the same per-modality score streams, so a comparison between them
changes only the combination rule. Standardisation uses chronologically prior calibration data
only; no future window contributes to ``mu`` or ``sigma``.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

CYBER_VIEWS: tuple[str, ...] = ("M1", "M2", "M3")
ALL_VIEWS: tuple[str, ...] = ("M1", "M2", "M3", "M4")


def standardise(scores: Sequence[float], calibration_scores: Sequence[float]) -> np.ndarray:
    """Return ``(scores - mu) / sigma`` with ``mu``/``sigma`` taken from calibration data.

    ``sigma`` of zero is replaced by 1.0 so a constant view yields zeros rather than infinities.
    """
    calibration = np.asarray(calibration_scores, dtype=float)
    calibration = calibration[np.isfinite(calibration)]
    if calibration.size == 0:
        raise ValueError("calibration_scores contains no finite value")
    mu = float(calibration.mean())
    sigma = float(calibration.std())
    return (np.asarray(scores, dtype=float) - mu) / (sigma if sigma > 0 else 1.0)


def cyber_only_sum_z(z_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Sum of standardised scores over M1-M3."""
    return _stack_sum(z_scores, CYBER_VIEWS)


def all_view_sum_z(z_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Sum of standardised scores over M1-M4."""
    return _stack_sum(z_scores, ALL_VIEWS)


def late_or_max_z(z_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Maximum standardised score across the available views."""
    present = [z_scores[v] for v in ALL_VIEWS if v in z_scores]
    if not present:
        raise ValueError("no view available for late-OR fusion")
    return np.nanmax(np.vstack(present), axis=0)


def cca_residual(cyber_z: np.ndarray, physical_z: np.ndarray,
                 calibration_index: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Departure from the calibration-time cyber/physical relationship.

    A one-component CCA is fitted on calibration rows only; the fused score is the absolute
    discrepancy between the two canonical variates. A large value means the two evidence blocks
    disagree relative to how they co-varied during calibration.

    ``cyber_z`` is (n_windows, n_cyber_views); ``physical_z`` is (n_windows, 1).
    """
    from sklearn.cross_decomposition import CCA

    finite = (np.isfinite(cyber_z[calibration_index]).all(axis=1)
              & np.isfinite(physical_z[calibration_index]).ravel())
    model = CCA(n_components=1, max_iter=max_iter)
    model.fit(np.nan_to_num(cyber_z[calibration_index][finite]),
              np.nan_to_num(physical_z[calibration_index][finite]))
    u, v = model.transform(np.nan_to_num(cyber_z), np.nan_to_num(physical_z))
    return np.abs(u.ravel() - v.ravel())


def _stack_sum(z_scores: Mapping[str, np.ndarray], views: Sequence[str]) -> np.ndarray:
    present = [z_scores[v] for v in views if v in z_scores]
    if not present:
        raise ValueError(f"none of {tuple(views)} present in z_scores")
    return np.nansum(np.vstack(present), axis=0)
