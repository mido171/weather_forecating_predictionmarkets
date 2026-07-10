from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_only_pressure_high_uncertainty_guard import (
    PRESSURE_HIGH,
    PressureGuardSpec,
    compute_pressure_guard,
    correction_from_state,
    score_pressure_guard,
)


def test_correction_from_state_requires_min_prior_rows() -> None:
    spec = PressureGuardSpec(
        "test",
        ("pressure_spread_bucket",),
        PRESSURE_HIGH,
        min_prior_rows=3,
        mean_shrinkage=0.0,
        scale_shrinkage=0.0,
        mean_cap_c=10.0,
        min_sigma_multiplier=0.5,
        max_sigma_multiplier=2.0,
    )

    correction, multiplier, raw_mean, raw_multiplier = correction_from_state(
        count=2,
        residual_sum=4.0,
        abs_sum=4.0,
        expected_abs_sum=2.0,
        spec=spec,
    )

    assert correction == 0.0
    assert multiplier == 1.0
    assert raw_mean != raw_mean
    assert raw_multiplier != raw_multiplier


def test_pressure_guard_excludes_current_row_from_mean_and_scale() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "pressure_spread_bucket": [PRESSURE_HIGH, PRESSURE_HIGH, PRESSURE_HIGH],
            "residual_to_add_c": [1.0, 3.0, 90.0],
            "reference_expected_abs_c": [1.0, 1.0, 1.0],
        }
    )
    spec = PressureGuardSpec(
        "test",
        ("pressure_spread_bucket",),
        PRESSURE_HIGH,
        min_prior_rows=1,
        mean_shrinkage=0.0,
        scale_shrinkage=0.0,
        mean_cap_c=100.0,
        min_sigma_multiplier=0.5,
        max_sigma_multiplier=10.0,
    )

    corrections, multipliers, prior_rows, raw_means, raw_multipliers = compute_pressure_guard(frame, spec)

    assert corrections.tolist() == [0.0, 1.0, 2.0]
    assert multipliers.tolist() == [1.0, 1.0, 2.0]
    assert prior_rows.tolist() == [0, 1, 2]
    assert raw_means[0] != raw_means[0]
    assert raw_multipliers[0] != raw_multipliers[0]


def test_promotion_gate_requires_interval_calibration_improvement() -> None:
    predictions = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-02-01"]),
            "target_tmax_c": [20.0, 20.0, 20.0],
            "global_bias_repaired_prediction_c": [19.0, 21.0, 20.0],
            "distribution_sigma_c": [10.0, 10.0, 10.0],
            "candidate_prediction_c": [20.0, 20.0, 20.0],
            "candidate_sigma_c": [10.0, 10.0, 10.0],
            "pressure_high_window": [True, True, False],
            "mean_residual_correction_c": [1.0, -1.0, 0.0],
            "sigma_multiplier": [1.0, 1.0, 1.0],
            "guard_id": ["candidate", "candidate", "candidate"],
            "group_columns": ["test", "test", "test"],
            "active_pressure_bucket": [PRESSURE_HIGH, PRESSURE_HIGH, PRESSURE_HIGH],
            "window_days": [float("nan"), float("nan"), float("nan")],
            "min_prior_rows": [1, 1, 1],
        }
    )

    scoreboard, _ = score_pressure_guard(predictions)

    assert scoreboard.iloc[0]["pressure_high_delta_mae_vs_0058"] < 0.0
    assert (
        scoreboard.iloc[0]["pressure_high_coverage_distance"]
        == scoreboard.iloc[0]["reference_pressure_high_coverage_distance"]
    )
    assert not bool(scoreboard.iloc[0]["promotion_gate_passed"])
