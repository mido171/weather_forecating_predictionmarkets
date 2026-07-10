from __future__ import annotations

from datetime import UTC, datetime

from hkg_tmax.source_normalization import (
    _forecast_rows_from_9day_description,
    _forecast_rows_from_local_description,
    parse_igra_level_line,
    parse_isd_line,
)


def test_parse_igra_key_pressure_level_with_quality_flags() -> None:
    line = "10 -9999  85000  1522B   80B-9999    37   310    40 "

    parsed = parse_igra_level_line(line)

    assert parsed["level_type"] == 10
    assert parsed["pressure_hpa"] == 850.0
    assert parsed["geopotential_height_m"] == 1522
    assert parsed["temperature_c"] == 8.0
    assert parsed["temperature_flag"] == "B"
    assert parsed["dewpoint_depression_c"] == 3.7
    assert parsed["wind_direction_deg"] == 310
    assert parsed["wind_speed_mps"] == 4.0


def test_parse_noaa_isd_core_line_extracts_station_time_and_thermal_fields() -> None:
    line = (
        "0173596730999992025010100004+21730+112770FM-12+001899999"
        "V0200371N0050199999999013900199+01661+01131101891"
        "ADDAA124999999AY101999AY201999"
    )

    parsed = parse_isd_line(line)

    assert parsed is not None
    assert parsed["station_id"] == "596730-99999"
    assert parsed["observed_at_utc"] == datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    assert parsed["latitude"] == 21.73
    assert parsed["longitude"] == 112.77
    assert parsed["air_temperature_c"] == 16.6
    assert parsed["dew_point_c"] == 11.3
    assert parsed["sea_level_pressure_hpa"] == 1018.9
    assert parsed["wind_direction_deg"] == 20
    assert parsed["wind_speed_mps"] == 5.0


def test_extract_english_rss_forecast_temperature_ranges() -> None:
    base = {
        "source_id": "datagov_hko_historical_rss_9day_forecast_en_archive",
        "published_at_hkt": "2021-04-13T11:30:00+08:00",
    }
    description = """
    Date/Month: 14/04 (Wednesday)<br/>
    Wind: East force 4.<br/>
    Weather: Cloudy.<br/>
    Temp range: 23 - 26 C<br/>
    Date/Month: 15/04 (Thursday)<br/>
    Temp range: 22 - 25 C<br/>
    """

    rows = _forecast_rows_from_9day_description(base, description)

    assert [(row["forecast_date"], row["forecast_max_temperature_c"]) for row in rows] == [
        ("2021-04-14", 26.0),
        ("2021-04-15", 25.0),
    ]


def test_extract_local_rss_forecast_temperature_range() -> None:
    base = {
        "source_id": "datagov_hko_historical_rss_local_forecast_en_archive",
        "published_at_hkt": "2020-06-01T00:00:00+08:00",
    }
    description = """
    Weather forecast for Hong Kong (Monday, 1 Jun 2020):<br/>
    Mainly cloudy. Temperatures will range between 28 and 32 degrees.
    """

    rows = _forecast_rows_from_local_description(base, description)

    assert rows[0]["forecast_date"] == "2020-06-01"
    assert rows[0]["forecast_min_temperature_c"] == 28.0
    assert rows[0]["forecast_max_temperature_c"] == 32.0
