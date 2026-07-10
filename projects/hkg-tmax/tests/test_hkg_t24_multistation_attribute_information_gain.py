from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_beastmode_signal_discovery import require_no_confirmation_dates
from scripts.run_hkg_t24_multistation_attribute_information_gain import (
    parse_station_attribute,
    past_only_feature_bucket_prediction,
)


def test_parse_station_attribute_extracts_station_and_attribute() -> None:
    parsed = parse_station_attribute("isd_station_dew_point_c_450070_99999")
    assert parsed == ("450070_99999", "dew_point_c")


def test_parse_station_attribute_rejects_non_station_feature() -> None:
    assert parse_station_attribute("isd_air_temp_mean_c") is None


def test_past_only_feature_bucket_prediction_excludes_current_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "forecast_source_family": ["press_archive"] * 5,
            "season": ["DJF"] * 5,
            "target_tmax_c": [11.0, 11.0, 100.0, 11.0, 11.0],
            "forecast_max_c": [10.0, 10.0, 10.0, 10.0, 10.0],
            "feature_a": [1.0, 1.1, 1.05, 1.2, 1.3],
        }
    )

    out = past_only_feature_bucket_prediction(
        frame,
        feature="feature_a",
        bins=2,
        season_conditioned=False,
        min_history=2,
        min_bucket_rows=1,
    )

    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(11.0)
    assert out.loc[2, "past_rows_used"] == 1


def test_confirmation_dates_are_blocked() -> None:
    with pytest.raises(RuntimeError):
        require_no_confirmation_dates([pd.Timestamp("2024-01-01")], context="test")
