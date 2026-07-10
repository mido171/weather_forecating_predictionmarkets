from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_only_late_period_bias_repair import (
    BiasRepairSpec,
    apply_bias_repair,
    correction_from_prior,
    score_candidates,
)


def test_correction_from_prior_requires_enough_strict_prior_rows() -> None:
    prior = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "residual_to_add_c": [1.0, 3.0],
        }
    )
    spec = BiasRepairSpec(
        "test",
        "expanding",
        None,
        None,
        None,
        min_prior_rows=3,
        shrinkage=1.0,
        cap_c=2.0,
    )

    correction, prior_rows, raw = correction_from_prior(prior, pd.Timestamp("2020-01-03"), spec)

    assert correction == 0.0
    assert prior_rows == 2
    assert raw != raw


def test_apply_bias_repair_excludes_current_row_from_correction() -> None:
    anchor = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "target_tmax_c": [11.0, 13.0, 100.0],
            "point_forecast_c": [10.0, 10.0, 10.0],
            "fold_id": ["a", "a", "a"],
            "year": [2020, 2020, 2020],
            "month": [1, 1, 1],
            "season": ["DJF", "DJF", "DJF"],
            "residual_to_add_c": [1.0, 3.0, 90.0],
        }
    )
    spec = BiasRepairSpec(
        "test",
        "expanding",
        None,
        None,
        None,
        min_prior_rows=1,
        shrinkage=0.0,
        cap_c=100.0,
    )

    out = apply_bias_repair(anchor, spec)

    assert out["residual_correction_c"].tolist() == [0.0, 1.0, 2.0]
    assert out["prior_rows"].tolist() == [0, 1, 2]


def test_score_candidates_promotion_gate_blocks_large_early_degradation() -> None:
    anchor = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2017-12-31", "2018-01-01"]),
            "target_tmax_c": [10.0, 10.0],
            "point_forecast_c": [10.0, 8.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2017-12-31", "2018-01-01"]),
            "target_tmax_c": [10.0, 10.0],
            "anchor_prediction_c": [10.0, 8.0],
            "candidate_prediction_c": [11.0, 10.0],
            "residual_correction_c": [1.0, 2.0],
            "prior_rows": [10, 10],
            "fold_id": ["a", "b"],
            "year": [2017, 2018],
            "month": [12, 1],
            "season": ["DJF", "DJF"],
            "correction_id": ["candidate", "candidate"],
            "family": ["test", "test"],
            "group_column": ["", ""],
            "window_days": ["", ""],
            "half_life_days": ["", ""],
            "min_prior_rows": [1, 1],
            "shrinkage": [0.0, 0.0],
            "cap_c": [2.0, 2.0],
        }
    )

    scoreboard, _ = score_candidates(predictions, anchor)

    assert not bool(scoreboard.iloc[0]["promotion_gate_passed"])
    assert scoreboard.iloc[0]["late_delta_mae_vs_anchor"] < 0.0
    assert scoreboard.iloc[0]["early_delta_mae_vs_anchor"] > 0.02
