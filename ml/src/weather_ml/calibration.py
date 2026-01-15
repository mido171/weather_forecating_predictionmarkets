"""Calibration helpers."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_calibrators(
    prob_by_bin: dict[str, np.ndarray],
    y_true_by_bin: dict[str, np.ndarray],
) -> dict[str, IsotonicRegression]:
    calibrators: dict[str, IsotonicRegression] = {}
    for name, probs in prob_by_bin.items():
        y_true = y_true_by_bin.get(name)
        if y_true is None:
            continue
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probs, y_true)
        calibrators[name] = calibrator
    return calibrators


def apply_calibrators(
    prob_by_bin: dict[str, np.ndarray],
    calibrators: dict[str, IsotonicRegression],
) -> dict[str, np.ndarray]:
    calibrated: dict[str, np.ndarray] = {}
    for name, probs in prob_by_bin.items():
        calibrator = calibrators.get(name)
        if calibrator is None:
            calibrated[name] = probs
        else:
            calibrated[name] = calibrator.predict(probs)
    return calibrated
