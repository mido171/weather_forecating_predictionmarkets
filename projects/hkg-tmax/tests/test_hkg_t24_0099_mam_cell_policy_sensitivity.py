from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0095_mam_error_direction_split_lab import DirectionSplitSpec
from scripts.run_hkg_t24_0099_mam_cell_policy_sensitivity import (
    CORRECTION_CAP_GRID,
    DIRECTION_THRESHOLD_GRID,
    MIN_HISTORY_GRID,
    float_token,
    make_sensitivity_specs,
    summarize_sensitivity,
)


def base_spec() -> DirectionSplitSpec:
    return DirectionSplitSpec(
        candidate_id="base",
        pair_name="feature_a__x__feature_b",
        feature_a="feature_a",
        feature_b="feature_b",
        group_a="target_memory",
        group_b="upper_air_ceiling",
        active_gate="mam_all",
        direction_mode="overforecast_only",
        min_history=80,
        direction_threshold_c=0.10,
        shrink_rows=80.0,
        correction_cap_c=0.25,
    )


def test_float_token_is_short_and_stable() -> None:
    assert float_token(0.10) == "0p1"
    assert float_token(0.05) == "0p05"
    assert float_token(1.00) == "1"


def test_make_sensitivity_specs_crosses_adjacent_grid_once() -> None:
    specs = make_sensitivity_specs(base_spec())

    assert len(specs) == len(MIN_HISTORY_GRID) * len(DIRECTION_THRESHOLD_GRID) * len(CORRECTION_CAP_GRID)
    assert len({spec.candidate_id for spec in specs}) == len(specs)
    assert any(
        spec.min_history == 80 and spec.direction_threshold_c == 0.10 and spec.correction_cap_c == 0.25
        for spec in specs
    )
    assert {spec.shrink_rows for spec in specs} == {80.0}
    assert {spec.direction_mode for spec in specs} == {"overforecast_only"}


def test_summarize_sensitivity_reports_improvement_and_hardened_counts() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "min_history": 60,
                "direction_threshold_c": 0.05,
                "correction_cap_c": 0.20,
                "mae": 0.99,
                "hardened_gate_passed": True,
                "specialist_active_rows": 5,
            },
            {
                "candidate_id": "b",
                "min_history": 60,
                "direction_threshold_c": 0.10,
                "correction_cap_c": 0.25,
                "mae": 1.02,
                "hardened_gate_passed": False,
                "specialist_active_rows": 7,
            },
            {
                "candidate_id": "c",
                "min_history": 80,
                "direction_threshold_c": 0.10,
                "correction_cap_c": 0.25,
                "mae": 0.98,
                "hardened_gate_passed": True,
                "specialist_active_rows": 9,
            },
        ]
    )

    summary = summarize_sensitivity(candidates, input_0095_mae=1.0)
    min60 = summary[(summary["group_dimension"].eq("min_history")) & (summary["group_value"].eq(60))].iloc[0]

    assert min60["candidate_count"] == 2
    assert min60["best_mae"] == 0.99
    assert min60["improves_vs_0095_count"] == 1
    assert min60["hardened_count"] == 1
