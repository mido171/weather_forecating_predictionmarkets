from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_station_only_guarded_candidate_stack import (
    StackSpec,
    apply_stack,
    score_stack,
)


def test_apply_stack_combines_point_and_sigma_components() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01"]),
            "target_tmax_c": [20.0],
            "base_0058_prediction_c": [19.0],
            "base_sigma_c": [2.0],
            "feb_mar_correction_c": [1.0],
            "pressure_correction_c": [0.5],
            "nearby_temp_correction_c": [-0.2],
            "pressure_sigma_multiplier": [0.9],
            "nearby_temp_sigma_multiplier": [0.8],
        }
    )
    spec = StackSpec(
        "test",
        feb_mar_weight=0.5,
        pressure_mean_weight=1.0,
        temp_mean_weight=1.0,
        pressure_sigma_power=1.0,
        temp_sigma_power=1.0,
    )

    out = apply_stack(frame, spec)

    assert out.loc[0, "candidate_prediction_c"] == 19.8
    assert out.loc[0, "candidate_sigma_c"] == pytest.approx(1.44)
    assert out.loc[0, "total_point_correction_vs_0058_c"] == pytest.approx(0.8)
    assert out.loc[0, "total_sigma_multiplier"] == pytest.approx(0.72)


def test_score_stack_promotion_gate_blocks_large_fold_damage() -> None:
    predictions = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
            "target_tmax_c": [20.0, 20.0, 20.0, 20.0],
            "base_0058_prediction_c": [19.0, 19.0, 20.0, 20.0],
            "base_sigma_c": [10.0, 10.0, 10.0, 10.0],
            "candidate_prediction_c": [20.0, 20.0, 20.1, 20.1],
            "candidate_sigma_c": [10.0, 10.0, 10.0, 10.0],
            "stack_id": ["candidate", "candidate", "candidate", "candidate"],
            "fold_id": ["fold_a", "fold_a", "fold_b", "fold_b"],
            "feb_mar_weight": [1.0, 1.0, 1.0, 1.0],
            "pressure_mean_weight": [0.0, 0.0, 0.0, 0.0],
            "nearby_temp_mean_weight": [0.0, 0.0, 0.0, 0.0],
            "pressure_sigma_power": [0.0, 0.0, 0.0, 0.0],
            "nearby_temp_sigma_power": [0.0, 0.0, 0.0, 0.0],
            "total_point_correction_vs_0058_c": [1.0, 1.0, 0.1, 0.1],
            "total_sigma_multiplier": [1.0, 1.0, 1.0, 1.0],
            "transition_target_window": [True, True, False, False],
            "pressure_high_window": [False, False, False, False],
            "nearby_temp_bucket": ["nearby_temp_mid", "nearby_temp_mid", "nearby_temp_mid", "nearby_temp_mid"],
        }
    )

    scoreboard, folds, _ = score_stack(predictions)

    assert scoreboard.iloc[0]["delta_mae_vs_0058"] < 0.0
    assert scoreboard.iloc[0]["fold_delta_max"] > 0.015
    assert not bool(scoreboard.iloc[0]["promotion_gate_passed"])
    assert len(folds) == 2
