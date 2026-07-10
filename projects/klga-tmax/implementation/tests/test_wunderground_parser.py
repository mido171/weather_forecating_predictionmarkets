from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

from klga_tmax.ingestion.hash_keys import sha256_hex
from klga_tmax.providers.wunderground.models import WundergroundRawDayResponse
from klga_tmax.providers.wunderground.parser import (
    DAILY_LABEL_METHOD_COMPUTED,
    parse_wunderground_response,
)
from klga_tmax.providers.wunderground.persistence import fetch_window_status
from klga_tmax.registry.cutoffs import target_local_day_window_utc


def _fixture_payload() -> dict:
    return json.loads(
        (Path(__file__).parent / "fixtures" / "weathercom_historical_observations_fixture.json").read_text(
            encoding="utf-8"
        )
    )


def _raw_response(payload: dict) -> WundergroundRawDayResponse:
    body = json.dumps(payload, sort_keys=True)
    return WundergroundRawDayResponse(
        station_id="KLGA",
        wunderground_station_id="KLGA",
        weathercom_location_id="KLGA:9:US",
        start_local_date=date(2023, 11, 14),
        end_local_date=date(2023, 11, 14),
        units="e",
        endpoint_url_redacted="https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=REDACTED",
        retrieved_at_utc=datetime(2023, 11, 15, 12, 0, tzinfo=timezone.utc),
        http_status=200,
        content_type="application/json",
        response_body_text=body,
        response_body_sha256=sha256_hex(body),
        response_size_bytes=len(body.encode("utf-8")),
        payload_json=payload,
        attempts=1,
    )


def test_parser_normalizes_intraday_rows_and_daily_tmax() -> None:
    parsed = parse_wunderground_response(_raw_response(_fixture_payload()), canonical_station_id="KLGA")

    assert parsed.observations_count == 2
    assert len(parsed.daily_actuals) == 1
    daily = parsed.daily_actuals[0]
    assert daily.local_date == date(2023, 11, 14)
    assert daily.daily_high_f == 55
    assert daily.settlement_high_f_whole == 55
    assert daily.daily_high_source_field == "hourly_temp_max"
    assert daily.label_method == DAILY_LABEL_METHOD_COMPUTED
    assert daily.daily_low_f == 54
    assert daily.daily_max_wind_gust_mph == 18.0
    assert daily.observations_count == 2
    assert daily.quality_flag == "suspect"
    assert daily.validation_status == "suspect"
    assert daily.provider_max_temp_values_json[0]["max_temp"] == 56.0


def test_daily_label_availability_is_local_day_end_plus_24_hours() -> None:
    parsed = parse_wunderground_response(_raw_response(_fixture_payload()), canonical_station_id="KLGA")
    daily = parsed.daily_actuals[0]
    _, local_day_end_utc = target_local_day_window_utc(date(2023, 11, 14))
    assert daily.provider_available_at_utc == local_day_end_utc + timedelta(hours=24)


def test_intraday_availability_uses_configured_lag() -> None:
    parsed = parse_wunderground_response(
        _raw_response(_fixture_payload()),
        canonical_station_id="KLGA",
        intraday_lag_minutes=45,
    )
    first = parsed.intraday_observations[0]
    assert first.temp_f == 54.0
    assert first.condition_text == "Mostly Cloudy"
    assert first.cloud_cover_text == "BKN"
    assert first.uv_index == 1.0
    assert first.solar_radiation is None
    assert first.provider_available_at_utc == first.observation_time_utc + timedelta(minutes=45)


def test_parser_filters_provider_sentinel_low_values_from_daily_normalization() -> None:
    payload = _fixture_payload()
    payload["observations"][0]["min_temp"] = -98

    parsed = parse_wunderground_response(_raw_response(payload), canonical_station_id="KLGA")

    daily = parsed.daily_actuals[0]
    assert daily.daily_high_f == 55
    assert daily.daily_low_f == 54
    assert daily.quality_flag == "suspect"


def test_parser_never_uses_provider_max_temp_as_daily_high() -> None:
    payload = _fixture_payload()
    payload["observations"][0]["temp"] = 66
    payload["observations"][0]["max_temp"] = 96
    payload["observations"][1]["temp"] = 56
    payload["observations"][1]["max_temp"] = 96
    response = WundergroundRawDayResponse(
        **{
            **_raw_response(payload).__dict__,
            "start_local_date": date(2023, 11, 14),
            "end_local_date": date(2023, 11, 14),
        }
    )

    parsed = parse_wunderground_response(response, canonical_station_id="KLGA")

    daily = parsed.daily_actuals[0]
    assert daily.daily_high_f == 66
    assert daily.daily_low_f == 56
    assert daily.daily_high_source_field == "hourly_temp_max"
    assert daily.raw_daily_json["daily_high_rule"] == "max bounded hourly temp field only; provider max_temp ignored"
    assert {item["max_temp"] for item in daily.provider_max_temp_values_json} == {96.0}


def test_parser_nulls_out_of_range_constrained_intraday_fields() -> None:
    payload = _fixture_payload()
    payload["observations"][0]["temp"] = 144
    payload["observations"][0]["dewPt"] = 144
    payload["observations"][0]["wspd"] = 749
    payload["observations"][0]["rh"] = 101
    payload["observations"][0]["precip_hrly"] = 99

    parsed = parse_wunderground_response(_raw_response(payload), canonical_station_id="KLGA")

    first = parsed.intraday_observations[0]
    assert first.temp_f is None
    assert first.dewpoint_f is None
    assert first.wind_speed_mph is None
    assert first.humidity_pct is None
    assert first.precipitation_in is None
    assert first.quality_flag == "suspect"
    assert "temp_out_of_range" in (first.quality_note or "")
    assert "dewpoint_out_of_range" in (first.quality_note or "")
    assert "wind_speed_out_of_range" in (first.quality_note or "")
    daily = parsed.daily_actuals[0]
    assert daily.daily_max_wind_speed_mph == 13.0
    assert daily.daily_precipitation_in is None


def test_provider_no_data_error_is_not_a_failed_window() -> None:
    payload = {
        "metadata": {"status_code": 400},
        "success": False,
        "errors": [
            {
                "error": {
                    "code": "NDF-0001",
                    "message": "There was no data found for your historical observations query.",
                }
            }
        ],
    }
    response = _raw_response(payload)
    response = WundergroundRawDayResponse(
        **{
            **response.__dict__,
            "http_status": 400,
            "error_type": "HTTP_400",
            "error_message": json.dumps(payload),
        }
    )

    parsed = parse_wunderground_response(response, canonical_station_id="KLGA")

    assert response.provider_no_data is True
    assert parsed.daily_actuals == ()
    assert parsed.intraday_observations == ()
    assert fetch_window_status(response, parsed) == "no_data"
