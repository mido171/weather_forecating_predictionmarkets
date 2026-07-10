from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0095_mam_error_direction_split_lab import (
    DirectionSplitSpec,
    apply_direction_split,
    mode_allows_direction,
    prior_direction,
    select_strong_pairs,
)


def test_prior_direction_uses_symmetric_threshold() -> None:
    assert prior_direction(0.11, 0.10) == "overforecast"
    assert prior_direction(-0.11, 0.10) == "underforecast"
    assert prior_direction(0.05, 0.10) == "neutral"


def test_mode_allows_only_requested_direction() -> None:
    assert mode_allows_direction("bidirectional", "overforecast")
    assert mode_allows_direction("bidirectional", "underforecast")
    assert mode_allows_direction("overforecast_only", "overforecast")
    assert not mode_allows_direction("overforecast_only", "underforecast")
    assert mode_allows_direction("underforecast_only", "underforecast")
    assert not mode_allows_direction("underforecast_only", "overforecast")


def test_select_strong_pairs_prefers_hardened_improving_0094_pairs() -> None:
    pairs = pd.DataFrame(
        [
            {"pair_name": "fallback_pair", "pair_priority": 9.0, "mam_new_frame_valid_rows": 100},
            {"pair_name": "winning_pair", "pair_priority": 1.0, "mam_new_frame_valid_rows": 100},
        ]
    )
    scoreboard = pd.DataFrame(
        [
            {
                "candidate_class": "0094_expanded_high_error_interaction",
                "candidate_id": "a",
                "pair_name": "winning_pair",
                "hardened_gate_passed": True,
                "delta_mae_vs_0093_base": -0.1,
                "mae": 1.0,
                "rmse": 1.2,
            },
            {
                "candidate_class": "0094_expanded_high_error_interaction",
                "candidate_id": "b",
                "pair_name": "fallback_pair",
                "hardened_gate_passed": False,
                "delta_mae_vs_0093_base": -0.2,
                "mae": 0.9,
                "rmse": 1.1,
            },
        ]
    )

    selected = select_strong_pairs(pairs, scoreboard)

    assert selected.iloc[0]["pair_name"] == "winning_pair"
    assert "fallback_pair" in selected["pair_name"].tolist()


def test_direction_split_uses_prior_active_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-02-28", "2020-03-01", "2020-03-02", "2020-03-03"]),
            "forecast_source_family": ["press_archive"] * 4,
            "season": ["DJF", "MAM", "MAM", "MAM"],
            "frame_segment": ["newly_available_official_frame"] * 4,
            "era_bucket": ["a"] * 4,
            "target_tmax_c": [10.0, 18.0, 19.0, 18.0],
            "forecast_max_c": [30.0, 20.0, 21.0, 21.0],
            "candidate_prediction_c": [30.0, 20.0, 21.0, 21.0],
            "base_residual_c": [20.0, 2.0, 2.0, 3.0],
            "feature_a__x__feature_b__bucket": [0.0, 0.0, 0.0, 0.0],
        }
    )
    spec = DirectionSplitSpec(
        candidate_id="test",
        pair_name="feature_a__x__feature_b",
        feature_a="feature_a",
        feature_b="feature_b",
        group_a="target_memory",
        group_b="isd_station_network",
        active_gate="mam_all",
        direction_mode="overforecast_only",
        min_history=1,
        direction_threshold_c=0.0,
        shrink_rows=0.0,
        correction_cap_c=10.0,
    )

    prediction, diagnostics = apply_direction_split(frame, spec)

    assert prediction.tolist() == [30.0, 20.0, 19.0, 19.0]
    assert diagnostics["prior_rows"].tolist() == [0, 0, 1, 2]
    assert diagnostics["prior_direction"].tolist() == ["inactive", "neutral", "overforecast", "overforecast"]


def test_direction_split_can_block_wrong_direction() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-03-01", "2020-03-02"]),
            "forecast_source_family": ["press_archive"] * 2,
            "season": ["MAM", "MAM"],
            "frame_segment": ["current_0081_frame"] * 2,
            "era_bucket": ["a"] * 2,
            "target_tmax_c": [22.0, 22.0],
            "forecast_max_c": [20.0, 20.0],
            "candidate_prediction_c": [20.0, 20.0],
            "base_residual_c": [-2.0, -2.0],
            "feature_a__x__feature_b__bucket": [0.0, 0.0],
        }
    )
    spec = DirectionSplitSpec(
        candidate_id="test",
        pair_name="feature_a__x__feature_b",
        feature_a="feature_a",
        feature_b="feature_b",
        group_a="target_memory",
        group_b="isd_station_network",
        active_gate="mam_all",
        direction_mode="overforecast_only",
        min_history=1,
        direction_threshold_c=0.0,
        shrink_rows=0.0,
        correction_cap_c=10.0,
    )

    prediction, diagnostics = apply_direction_split(frame, spec)

    assert prediction.tolist() == [20.0, 20.0]
    assert diagnostics["prior_direction"].tolist() == ["neutral", "underforecast"]
