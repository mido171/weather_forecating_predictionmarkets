from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_regime_interaction_atlas import (
    balanced_select,
    bin_codes_from_edges,
    build_feature_frame,
    should_score_pair,
    summarize_cell_outcome,
    summarize_codes,
)


def test_balanced_select_limits_repeated_groups_before_fallback() -> None:
    frame = pd.DataFrame(
        {
            "feature": ["a1", "a2", "a3", "b1", "c1"],
            "group": ["a", "a", "a", "b", "c"],
            "priority_score": [10.0, 9.0, 8.0, 7.0, 6.0],
        }
    )

    out = balanced_select(frame, limit=4, group_columns=["group"], per_group_limit=1)

    assert out["feature"].tolist()[:3] == ["a1", "b1", "c1"]
    assert len(out) == 4


def test_should_score_pair_skips_duplicate_same_station_same_attribute() -> None:
    gate = pd.Series(
        {
            "feature_id": "gate",
            "source_family": "station_trajectory",
            "source_attribute": "air_temperature_c",
            "station_ids": "450070-99999",
        }
    )
    response = pd.Series(
        {
            "feature_id": "response",
            "source_family": "station_trajectory",
            "source_attribute": "air_temperature_c",
            "station_ids": "450070-99999",
        }
    )
    other_station = response.copy()
    other_station["station_ids"] = "450110-99999"

    assert not should_score_pair(gate, response)
    assert should_score_pair(gate, other_station)


def test_summarize_cell_outcome_filters_cells_under_min_rows() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2000-01-01", periods=10),
            "gate": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0],
            "response": [0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0],
            "outcome": [1.0, 2.0, 100.0, 3.0, 8.0, 9.0, 20.0, 21.0, 22.0, 23.0],
        }
    )

    summary, cells = summarize_cell_outcome(
        frame,
        gate_feature_id="gate",
        response_feature_id="response",
        gate_edges=(2.0, 7.0),
        response_edges=(2.0, 7.0),
        outcome_column="outcome",
        min_rows=2,
    )

    assert len(cells) == 5
    assert summary["valid_cells"] == 3
    assert summary["high_cell"] == "high/high"
    assert summary["low_cell"] == "low/low"
    assert summary["spread"] == 20.0


def test_summarize_codes_matches_cell_spread_logic() -> None:
    gate_codes = bin_codes_from_edges(pd.Series([0.0, 0.0, 5.0, 5.0, 10.0, 10.0]), (2.0, 7.0))
    response_codes = bin_codes_from_edges(pd.Series([0.0, 0.0, 5.0, 5.0, 10.0, 10.0]), (2.0, 7.0))
    outcome = pd.Series([1.0, 2.0, 8.0, 10.0, 20.0, 24.0]).to_numpy(dtype=float)

    summary, cells = summarize_codes(outcome, gate_codes, response_codes, min_rows=2)

    assert len(cells) == 3
    assert summary["valid_cells"] == 3
    assert summary["high_cell"] == "high/high"
    assert summary["low_cell"] == "low/low"
    assert summary["spread"] == 20.5


def test_build_feature_frame_uses_train_safe_existing_feature_values() -> None:
    station_frame = pd.DataFrame(
        {
            "station_id": ["a", "a", "b", "b"],
            "target_date": pd.to_datetime(["1999-01-02", "1999-01-03", "1999-01-02", "1999-01-03"]),
            "target_tmax_c": [20.0, 21.0, 20.0, 21.0],
            "past_doy_count": [10, 10, 10, 10],
            "past_doy_mean_tmax_c": [19.0, 19.0, 19.0, 19.0],
            "target_anomaly_vs_past_doy_c": [1.0, 2.0, 1.0, 2.0],
            "temp": [5.0, 6.0, 9.0, 10.0],
        }
    )
    catalog = pd.DataFrame(
        [
            {
                "feature_id": "traj_a_temp_lag",
                "source_family": "station_trajectory",
                "station_id": "a",
                "station_a": "",
                "station_b": "",
                "station_ids": "a",
                "source_attribute": "temp",
                "transform": "lag_1d",
                "raw_feature_name": "temp__lag_1d",
                "display_name": "a temp lag",
            },
            {
                "feature_id": "pair_temp_a_minus_b",
                "source_family": "station_pair_spread",
                "station_id": "",
                "station_a": "a",
                "station_b": "b",
                "station_ids": "a,b",
                "source_attribute": "temp",
                "transform": "station_a_minus_station_b",
                "raw_feature_name": "temp",
                "display_name": "a minus b temp",
            },
        ]
    )

    out, enriched = build_feature_frame(station_frame, catalog)

    assert out["traj_a_temp_lag"].tolist()[0] != out["traj_a_temp_lag"].tolist()[0]
    assert out["traj_a_temp_lag"].tolist()[1] == 5.0
    assert out["pair_temp_a_minus_b"].tolist() == [-4.0, -4.0]
    assert enriched["feature_id"].tolist() == ["traj_a_temp_lag", "pair_temp_a_minus_b"]
