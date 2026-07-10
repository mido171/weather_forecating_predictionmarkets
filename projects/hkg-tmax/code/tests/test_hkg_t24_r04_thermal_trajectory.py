from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_r04_thermal_trajectory import (
    build_origin_feature_row,
    cutoff_observed_at_for_origin,
    origin_date_for_target,
    solar_geometry_features,
)


def test_origin_date_and_cutoff_observed_cap_are_tminus1_1440() -> None:
    target_date = pd.Timestamp("2023-07-15")
    origin_date = origin_date_for_target(target_date)

    assert origin_date == pd.Timestamp("2023-07-14")
    assert cutoff_observed_at_for_origin(origin_date) == pd.Timestamp("2023-07-14 14:40:00")


def test_solar_geometry_features_are_deterministic_and_finite() -> None:
    features = solar_geometry_features(180)

    assert 0 < features["day_length_hours"] < 24
    assert 0 < features["noon_solar_elevation_deg"] <= 90


def test_build_origin_feature_row_rejects_post_cutoff_observation_by_ignoring_it() -> None:
    temp = pd.DataFrame(
        [
            {
                "local_date": pd.Timestamp("2023-07-14"),
                "observed_at_naive": pd.Timestamp("2023-07-14 14:40:00"),
                "minute_of_day": 14 * 60 + 40,
                "value": 32.0,
            },
            {
                "local_date": pd.Timestamp("2023-07-14"),
                "observed_at_naive": pd.Timestamp("2023-07-14 14:50:00"),
                "minute_of_day": 14 * 60 + 50,
                "value": 40.0,
            },
        ]
    )
    since = pd.DataFrame(
        [
            {
                "local_date": pd.Timestamp("2023-07-14"),
                "observed_at_naive": pd.Timestamp("2023-07-14 14:40:00"),
                "variable": "temperature_since_midnight_max_c",
                "value": 32.5,
            },
        ]
    )

    row = build_origin_feature_row(
        target_date=pd.Timestamp("2023-07-15"),
        target_tmax_c=33.0,
        temp_group=temp,
        since_group=since,
    )

    assert row is not None
    assert row["hko_latest_temp_c"] == 32.0
    assert row["latest_observed_at_hkt"] == pd.Timestamp("2023-07-14 14:40:00")
