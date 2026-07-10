from __future__ import annotations

import math

import pandas as pd

from scripts.run_hkg_t24_0085_long_history_feature_station_residual_bridge import (
    bucket_signal,
    priority_score,
    safe_corr,
    stability_label,
)


def test_safe_corr_requires_enough_rows_and_variation() -> None:
    n, corr = safe_corr(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), min_rows=3)
    assert n == 2
    assert math.isnan(corr)

    n, corr = safe_corr(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0]), min_rows=3)
    assert n == 3
    assert math.isnan(corr)


def test_bucket_signal_reports_spread_for_ordered_values() -> None:
    feature = pd.Series(range(100), dtype=float)
    target = pd.Series([0.0] * 50 + [2.0] * 50)

    signal = bucket_signal(feature, target, min_rows=10)

    assert signal["bucket_count"] == 5
    assert signal["bucket_target_spread"] == 2.0


def test_stability_and_priority_are_directional() -> None:
    assert stability_label(0.2, 0.1) == "same_sign"
    assert stability_label(0.2, -0.1) == "sign_flip"

    score = priority_score(
        {
            "corr_best_error": 0.1,
            "corr_best_abs_error": -0.2,
            "corr_correction_abs_improvement": 0.3,
            "bucket_best_abs_error_spread": 0.4,
            "bucket_improvement_spread": 0.5,
            "target_stability": "same_sign",
        }
    )

    assert score > 0.9
