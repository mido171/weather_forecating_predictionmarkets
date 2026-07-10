from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_composite_expert_stack import merge_extra_experts, run_blend_grid


def test_merge_extra_experts_pivots_predictions_by_expert_id() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "forecast_source_family": ["press_archive", "press_archive"],
            "target_tmax_c": [20.0, 21.0],
            "forecast_max_c": [19.0, 20.0],
            "official_raw": [19.0, 20.0],
        }
    )
    long = pd.DataFrame(
        {
            "target_date": ["2020-01-01", "2020-01-02"],
            "expert_id": ["extra_a", "extra_a"],
            "candidate_prediction_c": [19.5, 20.5],
        }
    )

    out = merge_extra_experts(frame, long)

    assert out["extra_a"].to_list() == [19.5, 20.5]
    assert len(out) == 2


def test_run_blend_grid_scores_each_mode_source_and_history_combination() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "forecast_source_family": ["press_archive"] * 6,
            "target_tmax_c": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "official_raw": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "expert_a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )

    scoreboard, predictions = run_blend_grid(
        frame,
        experts=["official_raw", "expert_a"],
        min_histories=(2,),
    )

    assert len(scoreboard) == 4
    assert set(scoreboard["mode"]) == {"best", "inverse_mae"}
    assert set(scoreboard["same_source"]) == {False, True}
    assert len(predictions) == 24
