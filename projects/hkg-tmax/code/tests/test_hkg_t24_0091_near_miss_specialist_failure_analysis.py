from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0091_near_miss_specialist_failure_analysis import (
    candidate_failure_details,
    delta_columns,
    design_queue,
    readable_slice,
)


def make_scoreboard() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["0088_0087_interaction_champion", "candidate_a", "candidate_b"],
            "feature": ["", "isd_morning_to_midday_temp_rise_c", "target_lag45_tmax_c"],
            "family": ["", "isd_station_network", "target_memory"],
            "context_mode": ["", "source_season_feature", "feature"],
            "mae": [1.0, 0.99, 1.01],
            "rmse": [1.2, 1.19, 1.21],
            "delta_mae_vs_0088_base": [0.0, -0.01, 0.01],
            "old_frame_delta_mae_vs_0088_base": [0.0, -0.02, 0.0],
            "season_MAM_delta_mae_vs_0088_base": [0.0, 0.03, -0.01],
            "hardened_gate_passed": ["False", "False", "False"],
            "season_no_regression_passed": ["True", "False", "True"],
        }
    )


def test_delta_columns_and_readable_slice() -> None:
    cols = delta_columns(make_scoreboard())

    assert "season_MAM_delta_mae_vs_0088_base" in cols
    assert "delta_mae_vs_0088_base" not in cols
    assert readable_slice("season_MAM_delta_mae_vs_0088_base") == "season_MAM"


def test_candidate_failure_details_finds_blocking_slice() -> None:
    details = candidate_failure_details(make_scoreboard())
    candidate = details[details["candidate_id"].eq("candidate_a")].iloc[0]

    assert bool(candidate["full_improves"]) is True
    assert candidate["worst_failed_slice"] == "season_MAM"
    assert candidate["worst_failed_delta_mae"] == 0.03
    assert "no-correction guard" in candidate["recommendation"]


def test_design_queue_prefers_full_improving_near_misses() -> None:
    queue = design_queue(candidate_failure_details(make_scoreboard()))

    assert queue.iloc[0]["candidate_id"] == "candidate_a"
