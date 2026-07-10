from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_only_february_march_transition_specialist import (
    TransitionSpec,
    activation_weight,
    compute_transition_correction,
    score_candidates,
)


def test_transition_correction_excludes_current_row() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-02-01", "2020-02-02", "2020-02-03"]),
            "transition_month_bucket": ["february", "february", "february"],
            "residual_to_add_c": [1.0, 3.0, 90.0],
        }
    )
    spec = TransitionSpec(
        "test",
        ("transition_month_bucket",),
        "feb_mar_only",
        min_prior_rows=1,
        shrinkage=0.0,
        cap_c=100.0,
    )

    corrections, active_weights, prior_rows, raw_means = compute_transition_correction(frame, spec)

    assert corrections.tolist() == [0.0, 1.0, 2.0]
    assert active_weights.tolist() == [1.0, 1.0, 1.0]
    assert prior_rows.tolist() == [0, 1, 2]
    assert raw_means[0] != raw_means[0]


def test_activation_weight_blocks_non_transition_months() -> None:
    assert activation_weight(pd.Timestamp("2020-02-15"), "feb_mar_only") == 1.0
    assert activation_weight(pd.Timestamp("2020-03-15"), "feb_mar_only") == 1.0
    assert activation_weight(pd.Timestamp("2020-01-15"), "feb_mar_only") == 0.0
    assert activation_weight(pd.Timestamp("2020-04-15"), "feb_mar_only") == 0.0


def test_score_candidates_promotion_gate_blocks_adjacent_month_damage() -> None:
    predictions = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-15", "2020-02-15", "2020-05-15"]),
            "target_tmax_c": [20.0, 20.0, 20.0],
            "anchor_prediction_c": [20.0, 18.0, 20.0],
            "candidate_prediction_c": [20.1, 20.0, 20.0],
            "residual_correction_c": [0.1, 2.0, 0.0],
            "transition_target_window": [False, True, False],
            "adjacent_window": [True, False, False],
            "correction_id": ["candidate", "candidate", "candidate"],
            "group_columns": ["test", "test", "test"],
            "activation": ["test", "test", "test"],
            "window_days": [float("nan"), float("nan"), float("nan")],
            "min_prior_rows": [1, 1, 1],
            "shrinkage": [0.0, 0.0, 0.0],
            "cap_c": [2.0, 2.0, 2.0],
        }
    )

    scoreboard, _ = score_candidates(predictions)

    assert not bool(scoreboard.iloc[0]["promotion_gate_passed"])
    assert scoreboard.iloc[0]["transition_delta_mae_vs_anchor"] < 0.0
    assert scoreboard.iloc[0]["adjacent_delta_mae_vs_anchor"] > 0.005
