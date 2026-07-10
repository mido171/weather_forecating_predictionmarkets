from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_network_forecast_residual_interaction_mining import (
    InteractionSpec,
    past_only_group_residual_predictions,
    past_only_tercile_bucket,
    state_columns_for_specs,
    station_network_candidate_columns,
)


def test_past_only_tercile_bucket_excludes_current_date_distribution() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    values = pd.Series([1.0, 2.0, 3.0, 100.0])

    buckets = past_only_tercile_bucket(values, pd.Series(dates), min_history=3)

    assert buckets.to_list() == ["missing", "missing", "missing", "high"]


def test_past_only_group_residual_predictions_excludes_same_date_rows() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press", "press", "rss"],
            "target_tmax_c": [10.0, 100.0, 0.0],
            "official_raw": [0.0, 0.0, 0.0],
            "anchor_0038_c": [0.0, 0.0, 0.0],
            "anchor_residual_c": [10.0, 100.0, 0.0],
            "bucket_feature": ["high", "high", "high"],
        }
    )

    predictions = past_only_group_residual_predictions(
        frame,
        InteractionSpec(
            feature="bucket_feature",
            state_cols=(),
            same_source=False,
            min_cell_history=1,
            shrinkage=0.0,
            correction_clip_c=200.0,
        ),
    )

    assert predictions.loc[2, "residual_correction_c"] == 10.0
    assert predictions.loc[2, "candidate_prediction_c"] == 10.0


def test_past_only_group_residual_predictions_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "rss"],
            "target_tmax_c": [10.0, 0.0],
            "official_raw": [0.0, 0.0],
            "anchor_0038_c": [0.0, 0.0],
            "anchor_residual_c": [10.0, 0.0],
            "bucket_feature": ["high", "high"],
        }
    )

    predictions = past_only_group_residual_predictions(
        frame,
        InteractionSpec(
            feature="bucket_feature",
            state_cols=(),
            same_source=True,
            min_cell_history=1,
            shrinkage=0.0,
            correction_clip_c=200.0,
        ),
    )

    assert predictions.loc[1, "residual_correction_c"] == 0.0
    assert predictions.loc[1, "prior_cell_rows"] == 0


def test_station_network_candidate_columns_keeps_lagged_network_signals_only() -> None:
    rows = 501
    frame = pd.DataFrame(
        {
            "target_tmax_c": list(range(rows)),
            "official_raw": list(range(rows)),
            "target_lag7_tmax_c": list(range(rows)),
            "derived_air_temperature_network_spread": [value / 10 for value in range(rows)],
            "isd_station_air_temperature_c_450050_99999": [20.0 + value / 100 for value in range(rows)],
        }
    )

    columns = station_network_candidate_columns(frame)

    assert "target_tmax_c" not in columns
    assert "official_raw" not in columns
    assert "target_lag7_tmax_c" in columns
    assert "derived_air_temperature_network_spread" in columns
    assert "isd_station_air_temperature_c_450050_99999" in columns


def test_state_columns_for_specs_deduplicates_missing_optional_states() -> None:
    frame = pd.DataFrame({"meta_text_signal_state": ["sunny"], "some_other_column": [1]})

    states = state_columns_for_specs(frame)

    assert states == ((), ("meta_text_signal_state",))
