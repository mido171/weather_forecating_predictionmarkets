from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_deep_signal_forensics import (
    parse_station_feature,
    past_only_feature_bucket_predictions,
    qcut_labels,
)


def test_parse_station_feature_extracts_station_and_metric() -> None:
    assert parse_station_feature("isd_station_air_temperature_c_450050_99999") == (
        "450050_99999",
        "air_temperature_c",
    )
    assert parse_station_feature("isd_air_temp_range_c") is None


def test_qcut_labels_returns_ordered_quantile_names() -> None:
    labels = qcut_labels(pd.Series(range(100)), 5, "q")

    assert labels.value_counts().sort_index().to_dict() == {
        "q1": 20,
        "q2": 20,
        "q3": 20,
        "q4": 20,
        "q5": 20,
    }


def test_past_only_feature_bucket_predictions_use_prior_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "target_tmax_c": [10.0, 12.0, 10.0, 12.0, 99.0, 12.0],
            "point_forecast": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "season": ["DJF"] * 6,
            "feature_x": [0.0, 10.0, 0.0, 10.0, 10.0, 10.0],
        }
    )

    preds = past_only_feature_bucket_predictions(
        frame,
        "feature_x",
        bins=2,
        season_conditioned=False,
        min_history=2,
        min_group=1,
    )

    assert pd.isna(preds.loc[0, "candidate_prediction_c"])
    assert preds.loc[2, "candidate_prediction_c"] == pytest.approx(10.0)
    assert preds.loc[4, "candidate_prediction_c"] == pytest.approx(12.0)
    assert preds.loc[5, "candidate_prediction_c"] == pytest.approx(41.0)
