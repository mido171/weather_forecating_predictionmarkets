from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_r03_tmax_anatomy import (
    build_since_midnight_max,
    parse_hkt_observed_at,
    zip_entry_timestamp_hkt,
)


def test_zip_entry_timestamp_hkt_parses_data_gov_archive_name() -> None:
    parsed = zip_entry_timestamp_hkt(
        "https%3A%2F%2Fdata.weather.gov.hk%2FweatherAPI%2F/20200701-2357-latest_1min_temperature.csv"
    )

    assert parsed is not None
    assert parsed.isoformat() == "2020-07-01T23:57:00+08:00"


def test_parse_hkt_observed_at_accepts_compact_hko_timestamp() -> None:
    parsed = parse_hkt_observed_at("202007011530")

    assert parsed is not None
    assert parsed.isoformat() == "2020-07-01T15:30:00+08:00"


def test_since_midnight_max_preserves_raw_carryover_and_late_final() -> None:
    frame = pd.DataFrame(
        [
            {
                "station": "HK Observatory",
                "variable": "temperature_since_midnight_max_c",
                "local_date": pd.Timestamp("2020-07-01"),
                "observed_at_hkt": pd.Timestamp("2020-07-01 00:00:00+08:00"),
                "value": 34.9,
            },
            {
                "station": "HK Observatory",
                "variable": "temperature_since_midnight_max_c",
                "local_date": pd.Timestamp("2020-07-01"),
                "observed_at_hkt": pd.Timestamp("2020-07-01 15:30:00+08:00"),
                "value": 32.7,
            },
            {
                "station": "HK Observatory",
                "variable": "temperature_since_midnight_max_c",
                "local_date": pd.Timestamp("2020-07-01"),
                "observed_at_hkt": pd.Timestamp("2020-07-01 23:40:00+08:00"),
                "value": 32.7,
            },
        ]
    )

    result = build_since_midnight_max(frame).iloc[0]

    assert result["since_midnight_feed_raw_max_c"] == 34.9
    assert result["since_midnight_late_final_value_c"] == 32.7
    assert bool(result["since_midnight_midnight_carryover_suspected"]) is True
