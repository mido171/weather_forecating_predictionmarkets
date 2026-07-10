from __future__ import annotations

import math

import pandas as pd

from scripts.run_hkg_t24_station_lag_slope_information_atlas import (
    add_variant_columns,
    rolling_slope,
    station_attribute_variants,
)


def test_rolling_slope_returns_linear_slope_only_after_full_window() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 5.0])

    slope = rolling_slope(values, 3)

    assert math.isnan(slope.iloc[0])
    assert math.isnan(slope.iloc[1])
    assert slope.iloc[2] == 1.0
    assert slope.iloc[3] == 1.5


def test_station_attribute_variants_use_past_station_sequence() -> None:
    values = pd.Series([10.0, 13.0, 18.0, 19.0])

    variants = station_attribute_variants(values, "temp")

    assert math.isnan(variants["temp__lag_1d"].iloc[0])
    assert variants["temp__lag_1d"].iloc[1:].tolist() == [10.0, 13.0, 18.0]
    assert math.isnan(variants["temp__delta_1d"].iloc[0])
    assert variants["temp__delta_1d"].iloc[1:].tolist() == [3.0, 5.0, 1.0]
    assert variants["temp__rolling_mean_3d"].tolist()[0] != variants["temp__rolling_mean_3d"].tolist()[0]
    assert variants["temp__rolling_mean_3d"].iloc[2] == (10.0 + 13.0 + 18.0) / 3.0


def test_add_variant_columns_keeps_station_histories_separate() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["a", "a", "b", "b"],
            "local_date": pd.to_datetime(["1990-01-01", "1990-01-02", "1990-01-01", "1990-01-02"]),
            "target_date": pd.to_datetime(["1990-01-02", "1990-01-03", "1990-01-02", "1990-01-03"]),
            "target_tmax_c": [20.0, 21.0, 20.0, 21.0],
            "past_doy_count": [10, 10, 10, 10],
            "past_doy_mean_tmax_c": [19.0, 19.0, 19.0, 19.0],
            "target_anomaly_vs_past_doy_c": [1.0, 2.0, 1.0, 2.0],
        }
    )
    for column in [
        "air_temperature_c_latest_before_1500",
        "dew_point_c_latest_before_1500",
        "sea_level_pressure_hpa_latest_before_1500",
        "wind_speed_mps_latest_before_1500",
        "wind_u_mps_latest_before_1500",
        "wind_v_mps_latest_before_1500",
        "temp_dew_spread_c_latest_before_1500",
        "air_temperature_c_latest_before_1500_change_1d",
        "dew_point_c_latest_before_1500_change_1d",
        "sea_level_pressure_hpa_latest_before_1500_change_1d",
        "wind_speed_mps_latest_before_1500_change_1d",
        "air_temperature_c_latest_before_1500_minus_network_median",
        "dew_point_c_latest_before_1500_minus_network_median",
        "sea_level_pressure_hpa_latest_before_1500_minus_network_median",
        "wind_speed_mps_latest_before_1500_minus_network_median",
    ]:
        frame[column] = [1.0, 2.0, 100.0, 200.0]

    out, catalog = add_variant_columns(frame)

    feature = "air_temperature_c_latest_before_1500__lag_1d"
    station_b = out[out["station_id"].eq("b")].sort_values("target_date")
    assert station_b[feature].tolist()[0] != station_b[feature].tolist()[0]
    assert station_b[feature].tolist()[1] == 100.0
    assert catalog[["station_id", "feature_name"]].drop_duplicates().shape[0] == len(catalog)
