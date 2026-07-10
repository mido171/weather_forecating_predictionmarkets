from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from hkg_tmax.gribstream.client import (
    GribStreamClient,
    GribStreamRequestError,
    RetryConfig,
    fs_path,
    request_sha256,
    sanitize_text,
    sha256_file,
)
from hkg_tmax.paths import ProjectPaths
from hkg_tmax_db.connection import apply_migration, import_psycopg, redact_database_url

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
SMOKE_SCRIPT = REPO_ROOT / "scripts/run_tactical_gribstream_h24n_smoke.py"
TACTICAL_MIGRATION = REPO_ROOT / "db/migrations/postgres/20260625_0007_tactical_gribstream_h24n_schema.sql"
EXPERIMENT_ROOT = (
    REPO_ROOT
    / "experiments"
    / "campaigns"
    / "hkg-t24"
    / "0214_tactical_h24n_gribstream_backfill"
)
FIRST_WEEK_ROOT = EXPERIMENT_ROOT / "first_week_pull"
REQUEST_ROOT = FIRST_WEEK_ROOT / "request_payloads"
RAW_ROOT = PROJECT_PATHS.data_root / "_pipeline_internal" / "raw" / "gribstream_tactical_first_week"
SECRET_FILE = REPO_ROOT / "secrets/local/gribstream.env"
API_EVENT_LOG = EXPERIMENT_ROOT / "logs/gribstream_first_week_api_events.jsonl"
RESULTS_CSV = FIRST_WEEK_ROOT / "first_week_results.csv"
SUMMARY_JSON = FIRST_WEEK_ROOT / "first_week_summary.json"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"
ACQUISITION_VERSION = "tactical_h24n_v1"
HKT = ZoneInfo("Asia/Hong_Kong")


BACKFILL_WINDOWS: dict[str, tuple[str, str]] = {
    "gfs": ("2021-03-22T00:00:00Z", "2026-06-22T00:00:00Z"),
    "gefsatmosmean": ("2020-10-01T18:00:00Z", "2026-06-21T18:00:00Z"),
    "gefsatmos": ("2020-10-01T18:00:00Z", "2026-06-21T18:00:00Z"),
    "ifsoper": ("2024-02-28T18:00:00Z", "2026-06-21T18:00:00Z"),
    "ifsenfo": ("2024-03-01T18:00:00Z", "2026-06-21T18:00:00Z"),
    "aifsoper": ("2025-02-25T18:00:00Z", "2026-06-21T18:00:00Z"),
    "aifsenfo": ("2025-07-02T18:00:00Z", "2026-06-21T18:00:00Z"),
    "aigfssfc": ("2026-04-16T18:00:00Z", "2026-06-21T18:00:00Z"),
    "aigfspres": ("2026-04-16T18:00:00Z", "2026-06-21T18:00:00Z"),
    "aigefssfc": ("2025-06-01T18:00:00Z", "2026-06-21T18:00:00Z"),
    "graphcast": ("2024-04-25T18:00:00Z", "2026-05-05T00:00:00Z"),
    "fourcastnetgfs": ("2024-05-02T18:00:00Z", "2026-03-01T12:00:00Z"),
}

SPECIAL_RUN_TIMES: dict[str, list[str]] = {
    # CWA WRF only has a rolling recent history; this is a live/prospective probe, not a historical week.
    "cwawrf15": [
        "2026-06-22T18:00:00Z",
        "2026-06-23T18:00:00Z",
        "2026-06-24T18:00:00Z",
    ],
    # NBMOC is probe-only in the tactical plan.
    "nbmoc": [
        "2026-06-17T18:00:00Z",
        "2026-06-18T18:00:00Z",
        "2026-06-19T18:00:00Z",
        "2026-06-20T18:00:00Z",
        "2026-06-21T18:00:00Z",
        "2026-06-22T18:00:00Z",
        "2026-06-23T18:00:00Z",
    ],
}

FORECAST_WIDE_ALIAS_MAP = {
    "temperature_2m_k": "temperature_2m_k",
    "temperature_2m_mean_k": "temperature_2m_k",
    "member_temperature_2m_k": "temperature_2m_k",
    "interval_tmax_2m_k": "interval_tmax_2m_k",
    "interval_tmax_mean_k": "interval_tmax_2m_k",
    "member_interval_tmax_k": "interval_tmax_2m_k",
    "dewpoint_2m_k": "dewpoint_2m_k",
    "dewpoint_2m_mean_k": "dewpoint_2m_k",
    "rh_2m_mean_pct": "relative_humidity_2m_pct",
    "u_wind_10m_mps": "u_wind_10m_mps",
    "u10_mean_mps": "u_wind_10m_mps",
    "v_wind_10m_mps": "v_wind_10m_mps",
    "v10_mean_mps": "v_wind_10m_mps",
    "mslp_pa": "mslp_pa",
    "mslp_mean_pa": "mslp_pa",
    "low_cloud_pct": "low_cloud_pct",
    "accumulated_precip_kg_m2": "accumulated_precip_kg_m2",
    "downward_shortwave_w_m2": "downward_shortwave_w_m2",
    "net_shortwave_w_m2": "net_shortwave_w_m2",
    "total_precip_m": "total_precip_m",
    "shortwave_down_j_m2": "shortwave_down_j_m2",
    "total_column_water_vapour_kg_m2": "total_column_water_vapour_kg_m2",
    "pwat_mean_kg_m2": "pwat_kg_m2",
    "pwat_kg_m2": "pwat_kg_m2",
    "temperature_925_k": "temperature_925_k",
    "temperature_850_k": "temperature_850_k",
    "relative_humidity_700_pct": "relative_humidity_700_pct",
    "geopotential_height_500_m": "geopotential_height_500_m",
}

FORECAST_VALUE_COLUMNS = tuple(sorted(set(FORECAST_WIDE_ALIAS_MAP.values())))
ROW_METADATA_KEYS = {"forecasted_at", "forecasted_time", "lat", "lon", "name", "member"}


def load_smoke_module() -> Any:
    spec = importlib.util.spec_from_file_location("tactical_smoke_specs", SMOKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load smoke specs from {SMOKE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tactical_smoke_specs"] = module
    spec.loader.exec_module(module)
    return module


SMOKE = load_smoke_module()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_directory(path: Path) -> None:
    os.makedirs(fs_path(path), exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_directory(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_gribstream_token() -> str | None:
    if SECRET_FILE.exists():
        for line in SECRET_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("GRIBSTREAM_API_KEY="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token
    env_key = os.environ.get("GRIBSTREAM_API_KEY")
    if env_key:
        return env_key.strip()
    env_token = os.environ.get("GRIBSTREAM_API_TOKEN")
    return env_token.strip() if env_token else None


def effective_spec(dataset: str) -> Any:
    spec = SMOKE.MODEL_SPECS[dataset]
    variable_spec = SMOKE.VariableSpec
    if dataset == "aigfssfc":
        # Live catalog exposes no DPT selector for aigfssfc.
        return replace(
            spec,
            variables=(
                variable_spec("TMP", "2 m above ground", "temperature_2m_k"),
                variable_spec("UGRD", "10 m above ground", "u_wind_10m_mps"),
                variable_spec("VGRD", "10 m above ground", "v_wind_10m_mps"),
                variable_spec("PRMSL", "mean sea level", "mslp_pa"),
            ),
        )
    if dataset == "nbmoc":
        # Live catalog exposes WIND/WDIR, not UGRD/VGRD/PRMSL, for this probe.
        return replace(
            spec,
            variables=(
                variable_spec("TMP", "2 m above ground", "temperature_2m_k", "50% level"),
                variable_spec("DPT", "2 m above ground", "dewpoint_2m_k", "50% level"),
                variable_spec("WIND", "10 m above ground", "wind_10m_mps"),
                variable_spec("WDIR", "10 m above ground", "wind_direction_10m_deg"),
            ),
        )
    return spec


def run_times_for_dataset(dataset: str, week_days: int) -> list[str]:
    if dataset in SPECIAL_RUN_TIMES:
        return SPECIAL_RUN_TIMES[dataset][:week_days]
    start_text, end_text = BACKFILL_WINDOWS[dataset]
    start = parse_utc(start_text)
    end = parse_utc(end_text)
    run_times: list[str] = []
    current = start
    while current <= end and len(run_times) < week_days:
        run_times.append(iso_z(current))
        current += timedelta(days=1)
    return run_times


def build_payload(spec: Any, run_times: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timesList": run_times,
        "minLeadTime": spec.min_lead,
        "maxLeadTime": spec.max_lead,
        "coordinates": SMOKE.coordinates_for_policy(spec.location_policy),
        "variables": [variable.as_request_variable() for variable in spec.variables],
    }
    if spec.members:
        payload["members"] = list(spec.members)
    return payload


def expected_credits(spec: Any, payload: dict[str, Any]) -> int:
    return (
        len(payload["timesList"])
        * spec.expected_steps
        * len(payload["variables"])
        * math.ceil(len(payload["coordinates"]) / 500)
        * max(len(payload.get("members", [])), 1)
    )


def expected_rows(spec: Any, payload: dict[str, Any]) -> int:
    return (
        len(payload["timesList"])
        * spec.expected_steps
        * len(payload["coordinates"])
        * max(len(payload.get("members", [])), 1)
    )


def raw_object_path(dataset: str, run_times: list[str], request_hash: str) -> Path:
    safe_start = run_times[0].replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    safe_end = run_times[-1].replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    return RAW_ROOT / dataset / f"run_window_utc={safe_start}_to_{safe_end}" / f"{request_hash}.ndjson.gz"


def read_ndjson_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(fs_path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def infer_credits_from_rows(rows: list[dict[str, Any]], payload: dict[str, Any]) -> int:
    if not rows:
        return 0
    coord_bucket = math.ceil(len(payload["coordinates"]) / 500)
    variables = len(payload["variables"])
    requested_members = payload.get("members")
    total = 0
    run_times = {str(row.get("forecasted_at")) for row in rows if row.get("forecasted_at")}
    for run_time in run_times:
        run_rows = [row for row in rows if str(row.get("forecasted_at")) == run_time]
        valid_times = {str(row.get("forecasted_time")) for row in run_rows if row.get("forecasted_time")}
        if requested_members:
            members = {str(row.get("member")) for row in run_rows if row.get("member") is not None}
            member_count = len(members)
        else:
            member_count = 1
        total += len(valid_times) * variables * coord_bucket * max(member_count, 1)
    return total


def choose_chunk_id(cursor: Any, request_hash: str, dataset: str) -> str:
    cursor.execute("SELECT chunk_id FROM nwp_tactical.acquisition_chunk WHERE request_sha256 = %s", (request_hash,))
    row = cursor.fetchone()
    if row:
        return str(row[0])
    return f"firstweek_{dataset}_{request_hash[:12]}"


def upsert_chunk_and_raw(
    database_url: str,
    *,
    spec: Any,
    payload: dict[str, Any],
    request_hash: str,
    status: str,
    expected_row_count: int,
    expected_credit_count: int,
    actual_credit_count: int,
    raw_path: Path | None,
    response_sha256: str | None,
    row_count: int,
    http_status: int | None,
    elapsed_seconds: float,
    error_class: str | None = None,
    error_message: str | None = None,
) -> tuple[str, int | None]:
    psycopg = import_psycopg()
    from psycopg.types.json import Jsonb

    response_object_id: int | None = None
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            chunk_id = choose_chunk_id(cursor, request_hash, spec.dataset)
            request_with_metrics = dict(payload)
            request_with_metrics["_first_week_metrics"] = {
                "actual_credit_estimate": actual_credit_count,
                "elapsed_seconds": elapsed_seconds,
            }
            cursor.execute(
                """
                INSERT INTO nwp_tactical.acquisition_chunk (
                    chunk_id, acquisition_version, dataset_code, run_times_utc,
                    min_lead_hours, max_lead_hours, location_policy, variable_bundle_id,
                    member_policy, members, expected_rows, expected_credits, request_json,
                    request_sha256, status, raw_object_uri, response_sha256, http_status,
                    row_count, error_class, error_message, started_at_utc, completed_at_utc
                )
                VALUES (
                    %s, %s, %s, %s::timestamptz[],
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, now(), now()
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    raw_object_uri = EXCLUDED.raw_object_uri,
                    response_sha256 = EXCLUDED.response_sha256,
                    http_status = EXCLUDED.http_status,
                    row_count = EXCLUDED.row_count,
                    error_class = EXCLUDED.error_class,
                    error_message = EXCLUDED.error_message,
                    request_json = EXCLUDED.request_json,
                    expected_rows = EXCLUDED.expected_rows,
                    expected_credits = EXCLUDED.expected_credits,
                    completed_at_utc = now()
                """,
                (
                    chunk_id,
                    ACQUISITION_VERSION,
                    spec.dataset,
                    payload["timesList"],
                    int(spec.min_lead.rstrip("h")),
                    int(spec.max_lead.rstrip("h")),
                    spec.location_policy,
                    "first_week_required_v1",
                    spec.member_policy,
                    list(spec.members) if spec.members else None,
                    expected_row_count,
                    expected_credit_count,
                    Jsonb(request_with_metrics),
                    request_hash,
                    status,
                    raw_path.as_posix() if raw_path else None,
                    response_sha256,
                    http_status,
                    row_count,
                    error_class,
                    error_message,
                ),
            )
            if raw_path and response_sha256 is not None:
                cursor.execute(
                    """
                    INSERT INTO nwp_tactical.raw_response_object (
                        chunk_id, object_uri, byte_size, sha256, content_type, retrieved_at_utc, row_count
                    )
                    VALUES (%s, %s, %s, %s, 'application/ndjson', now(), %s)
                    ON CONFLICT (chunk_id, sha256) DO UPDATE SET
                        object_uri = EXCLUDED.object_uri,
                        byte_size = EXCLUDED.byte_size,
                        row_count = EXCLUDED.row_count
                    RETURNING response_object_id
                    """,
                    (
                        chunk_id,
                        raw_path.as_posix(),
                        os.path.getsize(fs_path(raw_path)),
                        response_sha256,
                        row_count,
                    ),
                )
                response_object_id = int(cursor.fetchone()[0])
        connection.commit()
    return chunk_id, response_object_id


def value_or_none(row: dict[str, Any], alias: str) -> float | None:
    value = row.get(alias)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def wide_record_from_row(
    *,
    dataset: str,
    row: dict[str, Any],
    requested_coordinates: dict[str, tuple[float, float]],
    source_response_object_id: int | None,
) -> dict[str, Any]:
    run_time = parse_utc(str(row["forecasted_at"]))
    valid_time = parse_utc(str(row["forecasted_time"]))
    location_code = str(row.get("name") or "hko_center")
    requested_latitude, requested_longitude = requested_coordinates.get(
        location_code,
        (float(row.get("lat")), float(row.get("lon"))),
    )
    values = {column: None for column in FORECAST_VALUE_COLUMNS}
    for alias, column in FORECAST_WIDE_ALIAS_MAP.items():
        if alias in row:
            values[column] = value_or_none(row, alias)
    member_raw = row.get("member", 0)
    try:
        member_number = int(member_raw)
    except (TypeError, ValueError):
        member_number = 0
    return {
        "dataset_code": dataset,
        "acquisition_version": ACQUISITION_VERSION,
        "target_date_hkt": valid_time.astimezone(HKT).date().isoformat(),
        "run_time_utc": iso_z(run_time),
        "valid_time_utc": iso_z(valid_time),
        "lead_hours": (valid_time - run_time).total_seconds() / 3600.0,
        "location_code": location_code,
        "requested_latitude": requested_latitude,
        "requested_longitude": requested_longitude,
        "returned_latitude": value_or_none(row, "lat"),
        "returned_longitude": value_or_none(row, "lon"),
        "member_number": member_number,
        "raw_values_jsonb": {
            key: value
            for key, value in row.items()
            if key not in ROW_METADATA_KEYS
        },
        "source_response_object_id": source_response_object_id,
        **values,
    }


def insert_forecast_wide(database_url: str, dataset: str, rows: list[dict[str, Any]], payload: dict[str, Any], response_object_id: int | None) -> int:
    if not rows:
        return 0
    psycopg = import_psycopg()
    from psycopg.types.json import Jsonb

    requested_coordinates = {
        str(coord["name"]): (float(coord["lat"]), float(coord["lon"]))
        for coord in payload["coordinates"]
    }
    records = [
        wide_record_from_row(
            dataset=dataset,
            row=row,
            requested_coordinates=requested_coordinates,
            source_response_object_id=response_object_id,
        )
        for row in rows
    ]
    columns = [
        "dataset_code",
        "acquisition_version",
        "target_date_hkt",
        "run_time_utc",
        "valid_time_utc",
        "lead_hours",
        "location_code",
        "requested_latitude",
        "requested_longitude",
        "returned_latitude",
        "returned_longitude",
        "member_number",
        *FORECAST_VALUE_COLUMNS,
        "raw_values_jsonb",
        "source_response_object_id",
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    update_columns = [
        "target_date_hkt",
        "lead_hours",
        "requested_latitude",
        "requested_longitude",
        "returned_latitude",
        "returned_longitude",
        *FORECAST_VALUE_COLUMNS,
        "raw_values_jsonb",
        "source_response_object_id",
        "quality_status",
    ]
    update_clause = ", ".join(f"{column} = EXCLUDED.{column}" for column in update_columns)
    sql = f"""
        INSERT INTO nwp_tactical.forecast_wide (
            {", ".join(columns)}, quality_status
        )
        VALUES ({placeholders}, 'raw_valid')
        ON CONFLICT (
            dataset_code, acquisition_version, run_time_utc, valid_time_utc, location_code, member_number
        )
        DO UPDATE SET {update_clause}
    """
    values = []
    for record in records:
        values.append(
            [
                Jsonb(record[column]) if column == "raw_values_jsonb" else record.get(column)
                for column in columns
            ]
        )
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(sql, values)
        connection.commit()
    return len(records)


def full_backfill_projection(datasets: list[str], week_days: int) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        spec = effective_spec(dataset)
        if dataset in BACKFILL_WINDOWS:
            start, end = BACKFILL_WINDOWS[dataset]
            total_runs = (parse_utc(end).date() - parse_utc(start).date()).days + 1
        elif dataset == "cwawrf15":
            total_runs = len(SPECIAL_RUN_TIMES["cwawrf15"])
        elif dataset == "nbmoc":
            total_runs = len(SPECIAL_RUN_TIMES["nbmoc"])
        else:
            total_runs = 0
        sample_runs = max(len(run_times_for_dataset(dataset, week_days)), 1)
        payload = build_payload(spec, run_times_for_dataset(dataset, min(week_days, sample_runs)))
        credits_per_run = expected_credits(spec, payload) / len(payload["timesList"])
        rows_per_run = expected_rows(spec, payload) / len(payload["timesList"])
        dataset_rows.append(
            {
                "dataset": dataset,
                "stage": spec.stage,
                "projected_run_count": total_runs,
                "credits_per_run": credits_per_run,
                "rows_per_run": rows_per_run,
                "projected_total_credits": int(credits_per_run * total_runs),
                "projected_total_rows": int(rows_per_run * total_runs),
            }
        )
    return {
        "datasets": dataset_rows,
        "projected_total_credits": sum(row["projected_total_credits"] for row in dataset_rows),
        "projected_total_rows": sum(row["projected_total_rows"] for row in dataset_rows),
    }


def run(args: argparse.Namespace) -> int:
    token = load_gribstream_token()
    ensure_directory(FIRST_WEEK_ROOT)
    ensure_directory(REQUEST_ROOT)
    if args.apply_schema:
        apply_migration(args.database_url, TACTICAL_MIGRATION)
    if not token:
        write_json(SUMMARY_JSON, {"status": "blocked", "reason": "missing_gribstream_api_key", "updated_at_utc": utc_now_iso()})
        return 2

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(SMOKE.MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")

    results: list[dict[str, Any]] = []
    started_wall = time.perf_counter()
    retry_config = RetryConfig(
        max_attempts=args.api_max_attempts,
        min_interval_seconds=args.api_min_interval_seconds,
        default_rate_limit_pause_seconds=args.pause_on_429_seconds,
        max_retry_delay_seconds=args.max_retry_after_seconds,
    )
    with GribStreamClient(token, retry_config=retry_config, event_log_path=API_EVENT_LOG) as client:
        for dataset in datasets:
            spec = effective_spec(dataset)
            run_times = run_times_for_dataset(dataset, args.week_days)
            payload = build_payload(spec, run_times)
            req_hash = request_sha256(payload)
            request_path = REQUEST_ROOT / f"{dataset}_{req_hash[:12]}.json"
            write_json(request_path, payload)
            raw_path = raw_object_path(dataset, run_times, req_hash)
            expected_credit_count = expected_credits(spec, payload)
            expected_row_count = expected_rows(spec, payload)
            raw_source = "api_fetched"
            http_status: int | None = None
            row_count = 0
            inserted_rows = 0
            actual_credit_count = 0
            response_hash: str | None = None
            elapsed_seconds = 0.0
            chunk_id = ""
            response_object_id: int | None = None
            try:
                request_started = time.perf_counter()
                if os.path.exists(fs_path(raw_path)) and args.reuse_existing:
                    raw_source = "raw_reused"
                    http_status = 200
                else:
                    client.post_runs_to_gzip(
                        dataset=dataset,
                        payload=payload,
                        output_path=raw_path,
                        request_hash=req_hash,
                    )
                    http_status = 200
                elapsed_seconds = time.perf_counter() - request_started
                rows = read_ndjson_gzip(raw_path)
                row_count = len(rows)
                actual_credit_count = infer_credits_from_rows(rows, payload)
                response_hash = sha256_file(raw_path)
                returned_runs = {str(row.get("forecasted_at")) for row in rows if row.get("forecasted_at")}
                wrong_runs = sorted(returned_runs - set(run_times))
                status = "completed" if rows and not wrong_runs else "completed_empty" if not rows else "failed"
                chunk_id, response_object_id = upsert_chunk_and_raw(
                    args.database_url,
                    spec=spec,
                    payload=payload,
                    request_hash=req_hash,
                    status=status,
                    expected_row_count=expected_row_count,
                    expected_credit_count=expected_credit_count,
                    actual_credit_count=actual_credit_count,
                    raw_path=raw_path,
                    response_sha256=response_hash,
                    row_count=row_count,
                    http_status=http_status,
                    elapsed_seconds=elapsed_seconds,
                    error_class="unexpected_forecasted_at" if wrong_runs else None,
                    error_message=";".join(wrong_runs[:10]) if wrong_runs else None,
                )
                inserted_rows = insert_forecast_wide(args.database_url, dataset, rows, payload, response_object_id)
                result = {
                    "dataset": dataset,
                    "stage": spec.stage,
                    "status": status,
                    "http_status": http_status,
                    "run_time_count": len(run_times),
                    "first_run_time": run_times[0],
                    "last_run_time": run_times[-1],
                    "row_count": row_count,
                    "forecast_wide_rows_upserted": inserted_rows,
                    "expected_rows": expected_row_count,
                    "estimated_credits_consumed": actual_credit_count,
                    "expected_credits": expected_credit_count,
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "credits_per_minute": round(actual_credit_count / max(elapsed_seconds / 60.0, 1e-9), 3),
                    "rows_per_second": round(row_count / max(elapsed_seconds, 1e-9), 3),
                    "request_sha256": req_hash,
                    "chunk_id": chunk_id,
                    "raw_path": raw_path.as_posix(),
                    "source": raw_source,
                    "error_class": "",
                    "error_message": "",
                }
            except GribStreamRequestError as exc:
                elapsed_seconds = time.perf_counter() - request_started
                chunk_id, _response_object_id = upsert_chunk_and_raw(
                    args.database_url,
                    spec=spec,
                    payload=payload,
                    request_hash=req_hash,
                    status="failed",
                    expected_row_count=expected_row_count,
                    expected_credit_count=expected_credit_count,
                    actual_credit_count=0,
                    raw_path=None,
                    response_sha256=None,
                    row_count=0,
                    http_status=exc.status_code,
                    elapsed_seconds=elapsed_seconds,
                    error_class=exc.error_class,
                    error_message=sanitize_text(str(exc), token),
                )
                result = {
                    "dataset": dataset,
                    "stage": spec.stage,
                    "status": "failed",
                    "http_status": exc.status_code,
                    "run_time_count": len(run_times),
                    "first_run_time": run_times[0],
                    "last_run_time": run_times[-1],
                    "row_count": 0,
                    "forecast_wide_rows_upserted": 0,
                    "expected_rows": expected_row_count,
                    "estimated_credits_consumed": 0,
                    "expected_credits": expected_credit_count,
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "credits_per_minute": 0,
                    "rows_per_second": 0,
                    "request_sha256": req_hash,
                    "chunk_id": chunk_id,
                    "raw_path": "",
                    "source": "api_error",
                    "error_class": exc.error_class,
                    "error_message": sanitize_text(str(exc), token),
                }
                results.append(result)
                if exc.status_code in {401, 403, 429}:
                    break
                continue
            results.append(result)

    total_wall_seconds = time.perf_counter() - started_wall
    projection = full_backfill_projection(datasets, args.week_days)
    measured_credits = sum(int(row["estimated_credits_consumed"]) for row in results)
    measured_rows = sum(int(row["row_count"]) for row in results)
    measured_fetch_seconds = sum(float(row["elapsed_seconds"]) for row in results)
    effective_credits_per_minute = measured_credits / max(total_wall_seconds / 60.0, 1e-9)
    projected_total_credits = int(projection["projected_total_credits"])
    projected_minutes_at_measured_speed = projected_total_credits / max(effective_credits_per_minute, 1e-9)
    summary = {
        "status": "passed" if all(row["status"] in {"completed", "completed_empty"} for row in results) else "failed",
        "updated_at_utc": utc_now_iso(),
        "database": redact_database_url(args.database_url),
        "datasets_requested": datasets,
        "total_wall_seconds": round(total_wall_seconds, 3),
        "measured_fetch_seconds_sum": round(measured_fetch_seconds, 3),
        "measured_credits_consumed_estimate": measured_credits,
        "measured_rows_returned": measured_rows,
        "measured_forecast_wide_rows_upserted": sum(int(row["forecast_wide_rows_upserted"]) for row in results),
        "effective_credits_per_minute": round(effective_credits_per_minute, 3),
        "effective_rows_per_second": round(measured_rows / max(total_wall_seconds, 1e-9), 3),
        "projected_total_credits_for_current_scope": projected_total_credits,
        "projected_total_rows_for_current_scope": projection["projected_total_rows"],
        "projected_minutes_at_measured_speed": round(projected_minutes_at_measured_speed, 2),
        "projected_hours_at_measured_speed": round(projected_minutes_at_measured_speed / 60.0, 2),
        "daily_limit_for_3_days_from_provider": 768000,
        "three_day_credit_allowance": 2304000,
        "projected_credit_headroom_vs_three_day_allowance": 2304000 - projected_total_credits,
        "results_csv": RESULTS_CSV.as_posix(),
        "api_event_log": API_EVENT_LOG.as_posix(),
        "projection": projection,
    }
    write_csv(
        RESULTS_CSV,
        results,
        [
            "dataset",
            "stage",
            "status",
            "http_status",
            "run_time_count",
            "first_run_time",
            "last_run_time",
            "row_count",
            "forecast_wide_rows_upserted",
            "expected_rows",
            "estimated_credits_consumed",
            "expected_credits",
            "elapsed_seconds",
            "credits_per_minute",
            "rows_per_second",
            "request_sha256",
            "chunk_id",
            "raw_path",
            "source",
            "error_class",
            "error_message",
        ],
    )
    write_json(SUMMARY_JSON, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if summary["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and measure the first tactical GribStream week.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--week-days", type=int, default=7)
    parser.add_argument(
        "--datasets",
        default="gfs,gefsatmosmean,gefsatmos,ifsoper,ifsenfo,cwawrf15,aifsoper,aifsenfo,aigfssfc,aigfspres,aigefssfc,graphcast,fourcastnetgfs,nbmoc",
    )
    parser.add_argument("--api-min-interval-seconds", type=float, default=12.0)
    parser.add_argument("--api-max-attempts", type=int, default=2)
    parser.add_argument("--pause-on-429-seconds", type=float, default=300.0)
    parser.add_argument("--max-retry-after-seconds", type=float, default=1800.0)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
