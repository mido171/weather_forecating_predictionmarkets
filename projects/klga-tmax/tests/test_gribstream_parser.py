from __future__ import annotations

from datetime import date, datetime, timezone
import gzip
import json

from klga_tmax.providers.gribstream.models import GribStreamRawResponse, ResolvedSelector
from klga_tmax.providers.gribstream.parser import parse_gribstream_response
from klga_tmax.providers.gribstream.plan import T1245_CUTOFF_ID, build_chunk, model_spec_by_id
from klga_tmax.providers.gribstream.plan import build_runs_chunk


def _temperature_selector() -> tuple[ResolvedSelector, ...]:
    return (
        ResolvedSelector(
            alias="temperature_2m",
            request_variables=(
                {
                    "name": "TMP",
                    "level": "2 m above ground",
                    "info": "",
                    "alias": "temperature_2m",
                },
            ),
            variable_name="TMP",
            variable_level="2 m above ground",
            variable_info="",
            shared_parameter="temperature_2m",
            unit_hint="K",
        ),
    )


def test_parser_preserves_member_zero_and_maps_nbmqmd_06z_to_target_day(tmp_path) -> None:
    chunk = build_chunk(
        spec=model_spec_by_id("nbmqmd"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=_temperature_selector(),
        members=(0, 1),
        cutoff_id=T1245_CUTOFF_ID,
    )
    raw_path = tmp_path / "nbmqmd.ndjson.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "forecasted_at": "2026-06-28T11:00:00Z",
                    "forecasted_time": "2026-06-29T06:00:00Z",
                    "lat": 40.7769,
                    "lon": -73.8740,
                    "name": "GP_KLGA_EXACT",
                    "member": 0,
                    "temperature_2m": 301.15,
                    "index_updated_at": "2026-06-28T11:20:00Z",
                }
            )
            + "\n"
        )
    response = GribStreamRawResponse(
        chunk=chunk,
        endpoint_url_redacted="https://gribstream.com/api/v2/nbmqmd/timeseries",
        retrieved_at_utc=datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc),
        http_status=200,
        content_type="application/ndjson",
        response_body_sha256="sha256",
        response_size_bytes=100,
        raw_storage_uri=str(raw_path),
        attempts=1,
    )

    parsed = parse_gribstream_response(response)

    assert parsed.row_count_raw == 1
    assert len(parsed.values) == 1
    value = parsed.values[0]
    assert value.member == "0"
    assert value.target_date == date(2026, 6, 28)
    assert value.cutoff_id == T1245_CUTOFF_ID
    assert value.cutoff_utc.isoformat() == "2026-06-28T12:45:00+00:00"
    assert any(
        gap.get("gap_type") == "missing_timeseries_value"
        and gap.get("member") == "1"
        and gap.get("variable_alias") == "temperature_2m"
        for gap in parsed.gaps
    )


def test_runs_parser_filters_extra_native_horizon_and_uses_row_level_asof(tmp_path) -> None:
    chunk = build_runs_chunk(
        spec=model_spec_by_id("gefsatmosmean"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 29),
        coordinate_tier_name="B",
        selectors=_temperature_selector(),
        cutoff_id=T1245_CUTOFF_ID,
        group_index=0,
    )
    raw_path = tmp_path / "gefs-runs.ndjson.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "forecasted_at": "2026-06-28T06:00:00Z",
                    "forecasted_time": "2026-06-28T18:00:00Z",
                    "lat": 40.7769,
                    "lon": -73.8740,
                    "name": "GP_KLGA_EXACT",
                    "temperature_2m": 300.15,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "forecasted_at": "2026-06-28T06:00:00Z",
                    "forecasted_time": "2026-06-28T21:00:00Z",
                    "lat": 40.7769,
                    "lon": -73.8740,
                    "name": "GP_KLGA_EXACT",
                    "temperature_2m": 301.15,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "forecasted_at": "2026-06-29T06:00:00Z",
                    "forecasted_time": "2026-06-29T18:00:00Z",
                    "lat": 40.7769,
                    "lon": -73.8740,
                    "name": "GP_KLGA_EXACT",
                    "temperature_2m": 302.15,
                }
            )
            + "\n"
        )
    response = GribStreamRawResponse(
        chunk=chunk,
        endpoint_url_redacted="https://gribstream.com/api/v2/gefsatmosmean/runs",
        retrieved_at_utc=datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc),
        http_status=200,
        content_type="application/ndjson",
        response_body_sha256="sha256",
        response_size_bytes=200,
        raw_storage_uri=str(raw_path),
        attempts=1,
    )

    parsed = parse_gribstream_response(response)

    assert parsed.row_count_raw == 3
    assert len(parsed.values) == 2
    assert {value.forecasted_time_utc.hour for value in parsed.values} == {18}
    assert {value.endpoint_type for value in parsed.values} == {"runs"}
    assert [value.as_of_utc.isoformat() for value in parsed.values] == [
        "2026-06-28T08:45:00+00:00",
        "2026-06-29T08:45:00+00:00",
    ]
