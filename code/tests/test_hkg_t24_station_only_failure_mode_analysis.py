from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_only_failure_mode_analysis import (
    apply_tertile,
    leakage_audit,
    quantile_edges,
    score_group,
)


def test_apply_tertile_uses_low_mid_high_and_missing() -> None:
    values = pd.Series([None, 0.0, 1.0, 2.0, 3.0])

    out = apply_tertile(values, (1.0, 2.0))

    assert out.tolist() == ["missing", "low", "low", "mid", "high"]


def test_quantile_edges_rejects_small_or_constant_samples() -> None:
    low, high = quantile_edges(pd.Series([1.0, 1.0, 1.0]))

    assert low != low
    assert high != high


def test_score_group_reports_error_shape() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=130),
            "group": ["a"] * 130,
            "error_c": [1.0] * 100 + [3.0] * 30,
        }
    )
    frame["abs_error_c"] = frame["error_c"].abs()

    out = score_group(frame, ["group"], min_rows=120)

    assert len(out) == 1
    assert out.iloc[0]["n"] == 130
    assert out.iloc[0]["mae"] > 1.0
    assert out.iloc[0]["share_abs_error_ge_2c"] == 30 / 130


def test_leakage_audit_marks_target_heat_bucket_as_diagnostic_only() -> None:
    frame = pd.DataFrame({"target_date": pd.to_datetime(["2023-12-31"])})
    thresholds = pd.DataFrame(
        [
            {
                "bucket_column": "availability_bucket",
                "threshold_fit_window": "<= 1999-12-31",
                "deployability": "deployable_pre_cutoff",
            },
            {
                "bucket_column": "heat_bucket_pre2000_target",
                "threshold_fit_window": "<= 1999-12-31",
                "deployability": "diagnostic_outcome_only_not_a_forecast_feature",
            },
        ]
    )

    audit = leakage_audit(frame, thresholds)

    assert audit["passed"].astype(bool).all()
