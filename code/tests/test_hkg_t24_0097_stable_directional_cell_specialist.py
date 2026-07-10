from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0095_mam_error_direction_split_lab import DirectionSplitSpec
from scripts.run_hkg_t24_0097_stable_directional_cell_specialist import (
    apply_stable_cell_specialist,
    build_spec_from_scoreboard_row,
    load_cell_sets,
    parse_bucket_label,
)


def test_parse_bucket_label_accepts_only_canonical_bucket_labels() -> None:
    assert parse_bucket_label("bucket_14") == "bucket_14"
    assert parse_bucket_label("bucket_014") == "bucket_14"
    assert parse_bucket_label("missing") is None
    assert parse_bucket_label("bucket_x") is None


def test_load_cell_sets_filters_stable_and_damaging_statuses() -> None:
    frame = pd.DataFrame(
        [
            {"pair_bucket_label": "bucket_14", "prior_direction": "overforecast", "status": "stable_improving"},
            {"pair_bucket_label": "bucket_10", "prior_direction": "overforecast", "status": "neutral"},
            {"pair_bucket_label": "bucket_5", "prior_direction": "underforecast", "status": "damaging"},
            {"pair_bucket_label": "missing", "prior_direction": "neutral", "status": "damaging"},
        ]
    )

    stable, damaging = load_cell_sets(frame)

    assert stable == {("bucket_14", "overforecast")}
    assert damaging == {("bucket_5", "underforecast")}


def test_build_spec_from_scoreboard_row_preserves_best_0095_parameters() -> None:
    row = pd.Series(
        {
            "pair_name": "feature_a__x__feature_b",
            "feature_a": "feature_a",
            "feature_b": "feature_b",
            "group_a": "target_memory",
            "group_b": "upper_air_ceiling",
            "active_gate": "mam_all",
            "direction_mode": "overforecast_only",
            "min_history": 80.0,
            "direction_threshold_c": 0.1,
            "shrink_rows": 80.0,
            "correction_cap_c": 0.25,
        }
    )

    spec = build_spec_from_scoreboard_row(row)

    assert spec.pair_name == "feature_a__x__feature_b"
    assert spec.min_history == 80
    assert spec.direction_threshold_c == 0.1
    assert spec.correction_cap_c == 0.25
    assert spec.candidate_id.startswith("stablecell_")


def test_stable_cell_specialist_uses_prior_rows_only_and_blocks_non_stable_cells() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-03-01", "2020-03-02", "2020-03-03"]),
            "forecast_source_family": ["press_archive"] * 3,
            "season": ["MAM", "MAM", "MAM"],
            "frame_segment": ["current_0081_frame"] * 3,
            "era_bucket": ["a"] * 3,
            "target_tmax_c": [18.0, 19.0, 18.0],
            "forecast_max_c": [20.0, 21.0, 21.0],
            "candidate_prediction_c": [20.0, 21.0, 21.0],
            "base_residual_c": [2.0, 2.0, 3.0],
            "feature_a__x__feature_b__bucket": [0.0, 0.0, 1.0],
        }
    )
    spec = DirectionSplitSpec(
        candidate_id="test",
        pair_name="feature_a__x__feature_b",
        feature_a="feature_a",
        feature_b="feature_b",
        group_a="target_memory",
        group_b="upper_air_ceiling",
        active_gate="mam_all",
        direction_mode="overforecast_only",
        min_history=1,
        direction_threshold_c=0.0,
        shrink_rows=0.0,
        correction_cap_c=10.0,
    )

    prediction, diagnostics = apply_stable_cell_specialist(
        frame,
        spec,
        stable_cells={("bucket_0", "overforecast")},
        damaging_cells=set(),
    )

    assert prediction.tolist() == [20.0, 19.0, 21.0]
    assert diagnostics["prior_rows"].tolist() == [0, 1, 0]
    assert diagnostics["stable_cell_allowed"].tolist() == [False, True, False]
    assert diagnostics["specialist_active"].tolist() == [False, True, False]


def test_stable_cell_specialist_damaging_guard_blocks_matching_cell() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-03-01", "2020-03-02"]),
            "forecast_source_family": ["press_archive", "press_archive"],
            "season": ["MAM", "MAM"],
            "frame_segment": ["current_0081_frame", "current_0081_frame"],
            "era_bucket": ["a", "a"],
            "target_tmax_c": [18.0, 19.0],
            "forecast_max_c": [20.0, 21.0],
            "candidate_prediction_c": [20.0, 21.0],
            "base_residual_c": [2.0, 2.0],
            "feature_a__x__feature_b__bucket": [0.0, 0.0],
        }
    )
    spec = DirectionSplitSpec(
        candidate_id="test",
        pair_name="feature_a__x__feature_b",
        feature_a="feature_a",
        feature_b="feature_b",
        group_a="target_memory",
        group_b="upper_air_ceiling",
        active_gate="mam_all",
        direction_mode="overforecast_only",
        min_history=1,
        direction_threshold_c=0.0,
        shrink_rows=0.0,
        correction_cap_c=10.0,
    )

    prediction, diagnostics = apply_stable_cell_specialist(
        frame,
        spec,
        stable_cells={("bucket_0", "overforecast")},
        damaging_cells={("bucket_0", "overforecast")},
    )

    assert prediction.tolist() == [20.0, 21.0]
    assert diagnostics["stable_cell_allowed"].tolist() == [False, True]
    assert diagnostics["damaging_cell_blocked"].tolist() == [False, True]
    assert diagnostics["specialist_active"].tolist() == [False, False]
