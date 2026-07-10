from __future__ import annotations

import gzip
import json
from pathlib import Path

import httpx

from hkg_tmax.gribstream.catalog import select_exact_temperature_2m
from hkg_tmax.gribstream.client import (
    GribStreamClient,
    RetryConfig,
    canonical_request_json,
    request_sha256,
    retry_delay_seconds,
    sanitize_text,
)
from hkg_tmax.gribstream.normalizer import normalize_runs_rows
from hkg_tmax.gribstream.store import POINT_VALUE_UPSERT_SQL


def test_selector_resolution_requires_exact_gfs_2m_temperature() -> None:
    payload = {
        "default_request": {
            "variables": [
                {"name": "TMP", "level": "0.02 mb", "info": ""},
                {"name": "TMP", "level": "2 m above ground", "info": ""},
            ],
        },
    }

    selector = select_exact_temperature_2m(payload, dataset="gfs", retrieved_at_utc="2026-06-24T00:00:00Z")

    assert selector.native_name == "TMP"
    assert selector.native_level == "2 m above ground"
    assert selector.as_request_variable() == {
        "name": "TMP",
        "level": "2 m above ground",
        "info": "",
        "alias": "temperature_2m",
    }


def test_canonical_request_hash_is_stable_and_secret_safe() -> None:
    left = {"b": [2, 1], "a": {"z": "value"}}
    right = {"a": {"z": "value"}, "b": [2, 1]}

    assert canonical_request_json(left) == canonical_request_json(right)
    assert request_sha256(left) == request_sha256(right)
    assert "secret-token" not in sanitize_text("failed with secret-token", "secret-token")


def test_429_without_retry_after_defaults_to_three_to_five_minute_pause() -> None:
    delay = retry_delay_seconds(
        status_code=429,
        retry_after=None,
        attempt_number=1,
        config=RetryConfig(default_rate_limit_pause_seconds=300, min_rate_limit_pause_seconds=180),
    )

    assert delay == 300


def test_client_retries_429_and_writes_ndjson_gzip(tmp_path: Path) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer test-token"
        if calls == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            headers={"content-type": "application/ndjson"},
            text=(
                '{"forecasted_at":"2026-06-23T00:00:00Z","forecasted_time":"2026-06-23T01:00:00Z","name":"hkg","temperature_2m":300.1}\n'
                '{"forecasted_at":"2026-06-23T00:00:00Z","forecasted_time":"2026-06-23T02:00:00Z","name":"hkg","temperature_2m":300.2}\n'
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gribstream = GribStreamClient(
        "test-token",
        base_url="https://example.test",
        retry_config=RetryConfig(max_attempts=2, min_interval_seconds=0, default_rate_limit_pause_seconds=300),
        http_client=client,
        sleeper=sleeps.append,
    )
    output_path = tmp_path / "object.ndjson.gz"

    manifest = gribstream.post_runs_to_gzip(
        dataset="gfs",
        payload={"coordinates": [], "variables": []},
        output_path=output_path,
        request_hash="abc",
    )

    assert calls == 2
    assert sleeps == [300]
    assert manifest.row_count == 2
    with gzip.open(output_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert [row["temperature_2m"] for row in rows] == [300.1, 300.2]


def test_normalizer_sorts_out_of_order_rows_and_computes_lead_minutes() -> None:
    rows = [
        {
            "forecasted_at": "2026-06-23T00:00:00Z",
            "forecasted_time": "2026-06-23T03:00:00Z",
            "name": "b",
            "temperature_2m": "301.0",
        },
        {
            "forecasted_at": "2026-06-23T00:00:00Z",
            "forecasted_time": "2026-06-23T01:00:00Z",
            "name": "a",
            "temperature_2m": "300.0",
        },
    ]

    result = normalize_runs_rows(rows, value_alias="temperature_2m", location_ids_by_code={"a": 10, "b": 11})

    assert not result.rejected_rows
    assert [point.location_code for point in result.points] == ["a", "b"]
    assert [point.lead_minutes for point in result.points] == [60, 180]
    assert result.points[0].value == 300.0


def test_point_value_sql_is_idempotent_upsert() -> None:
    assert "ON CONFLICT (model_run_id, valid_time_utc, location_id, selector_id, member_number)" in POINT_VALUE_UPSERT_SQL
    assert "DO UPDATE SET" in POINT_VALUE_UPSERT_SQL
