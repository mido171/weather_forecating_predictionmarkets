from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from hkg_tmax.gribstream.client import (
    GribStreamClient,
    GribStreamRequestError,
    RetryConfig,
    request_sha256,
    sanitize_text,
    sha256_file,
)
from hkg_tmax.paths import ProjectPaths
from hkg_tmax_db.connection import import_psycopg, redact_database_url

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
EXPERIMENT_ROOT = (
    REPO_ROOT
    / "experiments"
    / "campaigns"
    / "hkg-t24"
    / "0214_tactical_h24n_gribstream_backfill"
)
REQUEST_ROOT = EXPERIMENT_ROOT / "request_payloads"
RAW_ROOT = PROJECT_PATHS.data_root / "_pipeline_internal" / "raw" / "gribstream_tactical_smoke"
SECRET_FILE = REPO_ROOT / "secrets/local/gribstream.env"
API_EVENT_LOG = EXPERIMENT_ROOT / "logs/gribstream_api_events.jsonl"
SMOKE_RESULTS_CSV = EXPERIMENT_ROOT / "smoke_api_results.csv"
CATALOG_PREFLIGHT_JSON = EXPERIMENT_ROOT / "catalog_preflight.json"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"


@dataclass(frozen=True)
class VariableSpec:
    name: str
    level: str
    alias: str
    info: str = ""

    def as_request_variable(self) -> dict[str, Any]:
        return {"name": self.name, "level": self.level, "info": self.info, "alias": self.alias}


@dataclass(frozen=True)
class ModelSpec:
    dataset: str
    stage: str
    smoke_run_time: str
    min_lead: str
    max_lead: str
    expected_steps: int
    location_policy: str
    member_policy: str
    variables: tuple[VariableSpec, ...]
    members: tuple[int, ...] = ()


HKO_LAT = 22.301944
HKO_LON = 114.174167


STENCIL_12: tuple[tuple[str, float, float], ...] = (
    ("hko_center", 0.00, 0.00),
    ("local_n", 0.25, 0.00),
    ("local_s", -0.25, 0.00),
    ("local_e", 0.00, 0.25),
    ("local_w", 0.00, -0.25),
    ("local_ne", 0.25, 0.25),
    ("local_nw", 0.25, -0.25),
    ("local_se", -0.25, 0.25),
    ("local_sw", -0.25, -0.25),
    ("inland_nw_far", 0.50, -0.50),
    ("marine_s_far", -0.50, 0.00),
    ("marine_e_far", 0.00, 0.50),
)


MODEL_SPECS: dict[str, ModelSpec] = {
    "gfs": ModelSpec(
        dataset="gfs",
        stage="core",
        smoke_run_time="2021-03-23T00:00:00Z",
        min_lead="15h",
        max_lead="39h",
        expected_steps=25,
        location_policy="deterministic_12_point_stencil",
        member_policy="deterministic",
        variables=(
            VariableSpec("TMP", "2 m above ground", "temperature_2m_k"),
            VariableSpec("TMAX", "2 m above ground", "interval_tmax_2m_k"),
            VariableSpec("DPT", "2 m above ground", "dewpoint_2m_k"),
            VariableSpec("UGRD", "10 m above ground", "u_wind_10m_mps"),
            VariableSpec("VGRD", "10 m above ground", "v_wind_10m_mps"),
            VariableSpec("PRMSL", "mean sea level", "mslp_pa"),
            VariableSpec("LCDC", "low cloud layer", "low_cloud_pct"),
            VariableSpec("APCP", "surface", "accumulated_precip_kg_m2"),
            VariableSpec("DSWRF", "surface", "downward_shortwave_w_m2"),
            VariableSpec("TMP", "925 mb", "temperature_925_k"),
            VariableSpec("TMP", "850 mb", "temperature_850_k"),
            VariableSpec("RH", "700 mb", "relative_humidity_700_pct"),
            VariableSpec("HGT", "500 mb", "geopotential_height_500_m"),
        ),
    ),
    "gefsatmosmean": ModelSpec(
        dataset="gefsatmosmean",
        stage="core",
        smoke_run_time="2020-10-01T18:00:00Z",
        min_lead="24h",
        max_lead="45h",
        expected_steps=8,
        location_policy="deterministic_12_point_stencil",
        member_policy="mean",
        variables=(
            VariableSpec("TMP", "2 m above ground", "temperature_2m_mean_k", "ens mean"),
            VariableSpec("TMAX", "2 m above ground", "interval_tmax_mean_k", "ens mean"),
            VariableSpec("DPT", "2 m above ground", "dewpoint_2m_mean_k", "ens mean"),
            VariableSpec("RH", "2 m above ground", "rh_2m_mean_pct", "ens mean"),
            VariableSpec("UGRD", "10 m above ground", "u10_mean_mps", "ens mean"),
            VariableSpec("VGRD", "10 m above ground", "v10_mean_mps", "ens mean"),
            VariableSpec("PRMSL", "mean sea level", "mslp_mean_pa", "ens mean"),
            VariableSpec(
                "PWAT",
                "entire atmosphere (considered as a single layer)",
                "pwat_mean_kg_m2",
                "ens mean",
            ),
        ),
    ),
    "gefsatmos": ModelSpec(
        dataset="gefsatmos",
        stage="core",
        smoke_run_time="2020-10-01T18:00:00Z",
        min_lead="24h",
        max_lead="45h",
        expected_steps=8,
        location_policy="hko_center_only",
        member_policy="members_0_30",
        variables=(VariableSpec("TMAX", "2 m above ground", "member_interval_tmax_k"),),
        members=tuple(range(31)),
    ),
    "ifsoper": ModelSpec(
        dataset="ifsoper",
        stage="core",
        smoke_run_time="2024-02-28T18:00:00Z",
        min_lead="21h",
        max_lead="45h",
        expected_steps=9,
        location_policy="deterministic_12_point_stencil",
        member_policy="deterministic",
        variables=(
            VariableSpec("2t", "sfc", "temperature_2m_k"),
            VariableSpec("2d", "sfc", "dewpoint_2m_k"),
            VariableSpec("10u", "sfc", "u_wind_10m_mps"),
            VariableSpec("10v", "sfc", "v_wind_10m_mps"),
            VariableSpec("msl", "sfc", "mslp_pa"),
            VariableSpec("tp", "sfc", "total_precip_m"),
            VariableSpec("ssrd", "sfc", "shortwave_down_j_m2"),
            VariableSpec("tcwv", "sfc", "total_column_water_vapour_kg_m2"),
            VariableSpec("t", "pl 925", "temperature_925_k"),
            VariableSpec("t", "pl 850", "temperature_850_k"),
            VariableSpec("r", "pl 700", "relative_humidity_700_pct"),
            VariableSpec("gh", "pl 500", "geopotential_height_500_m"),
        ),
    ),
    "ifsenfo": ModelSpec(
        dataset="ifsenfo",
        stage="core",
        smoke_run_time="2024-03-01T18:00:00Z",
        min_lead="24h",
        max_lead="45h",
        expected_steps=8,
        location_policy="hko_center_only",
        member_policy="members_0_50",
        variables=(VariableSpec("2t", "sfc", "member_temperature_2m_k"),),
        members=tuple(range(51)),
    ),
    "cwawrf15": ModelSpec(
        dataset="cwawrf15",
        stage="prospective",
        smoke_run_time="2026-06-23T18:00:00Z",
        min_lead="18h",
        max_lead="42h",
        expected_steps=5,
        location_policy="deterministic_12_point_stencil",
        member_policy="prospective_latest_complete",
        variables=(
            VariableSpec("TMP", "2 m above ground", "temperature_2m_k"),
            VariableSpec("DPT", "2 m above ground", "dewpoint_2m_k"),
            VariableSpec("UGRD", "10 m above ground", "u_wind_10m_mps"),
            VariableSpec("VGRD", "10 m above ground", "v_wind_10m_mps"),
            VariableSpec("PRMSL", "mean sea level", "mslp_pa"),
            VariableSpec("APCP", "surface", "accumulated_precip_kg_m2"),
            VariableSpec("NSWRF", "surface", "net_shortwave_w_m2"),
            VariableSpec("TMP", "850 mb", "temperature_850_k"),
            VariableSpec("RH", "700 mb", "relative_humidity_700_pct"),
            VariableSpec("HGT", "500 mb", "geopotential_height_500_m"),
        ),
    ),
    "aifsoper": ModelSpec(
        dataset="aifsoper",
        stage="optional",
        smoke_run_time="2025-02-25T18:00:00Z",
        min_lead="18h",
        max_lead="42h",
        expected_steps=5,
        location_policy="deterministic_12_point_stencil",
        member_policy="deterministic",
        variables=(
            VariableSpec("2t", "sfc", "temperature_2m_k"),
            VariableSpec("2d", "sfc", "dewpoint_2m_k"),
            VariableSpec("10u", "sfc", "u_wind_10m_mps"),
            VariableSpec("10v", "sfc", "v_wind_10m_mps"),
            VariableSpec("msl", "sfc", "mslp_pa"),
            VariableSpec("tp", "sfc", "total_precip_m"),
            VariableSpec("ssrd", "sfc", "shortwave_down_j_m2"),
            VariableSpec("t", "pl 850", "temperature_850_k"),
        ),
    ),
    "aifsenfo": ModelSpec(
        dataset="aifsenfo",
        stage="optional",
        smoke_run_time="2025-07-02T18:00:00Z",
        min_lead="24h",
        max_lead="42h",
        expected_steps=4,
        location_policy="hko_center_only",
        member_policy="members_0_50",
        variables=(VariableSpec("2t", "sfc", "member_temperature_2m_k"),),
        members=tuple(range(51)),
    ),
    "aigfssfc": ModelSpec(
        dataset="aigfssfc",
        stage="optional",
        smoke_run_time="2026-04-16T18:00:00Z",
        min_lead="18h",
        max_lead="42h",
        expected_steps=5,
        location_policy="deterministic_12_point_stencil",
        member_policy="deterministic",
        variables=(
            VariableSpec("TMP", "2 m above ground", "temperature_2m_k"),
            VariableSpec("DPT", "2 m above ground", "dewpoint_2m_k"),
            VariableSpec("UGRD", "10 m above ground", "u_wind_10m_mps"),
            VariableSpec("VGRD", "10 m above ground", "v_wind_10m_mps"),
            VariableSpec("PRMSL", "mean sea level", "mslp_pa"),
        ),
    ),
    "aigfspres": ModelSpec(
        dataset="aigfspres",
        stage="optional",
        smoke_run_time="2026-04-16T18:00:00Z",
        min_lead="18h",
        max_lead="42h",
        expected_steps=5,
        location_policy="deterministic_12_point_stencil",
        member_policy="deterministic",
        variables=(
            VariableSpec("TMP", "850 mb", "temperature_850_k"),
            VariableSpec("HGT", "500 mb", "geopotential_height_500_m"),
        ),
    ),
    "aigefssfc": ModelSpec(
        dataset="aigefssfc",
        stage="optional",
        smoke_run_time="2025-06-01T18:00:00Z",
        min_lead="24h",
        max_lead="42h",
        expected_steps=4,
        location_policy="hko_center_only",
        member_policy="members_0_30",
        variables=(VariableSpec("TMP", "2 m above ground", "member_temperature_2m_k"),),
        members=tuple(range(31)),
    ),
    "graphcast": ModelSpec(
        dataset="graphcast",
        stage="optional",
        smoke_run_time="2024-04-25T18:00:00Z",
        min_lead="18h",
        max_lead="42h",
        expected_steps=5,
        location_policy="deterministic_12_point_stencil",
        member_policy="deterministic",
        variables=(
            VariableSpec("TMP", "2 m above ground", "temperature_2m_k"),
            VariableSpec("UGRD", "10 m above ground", "u_wind_10m_mps"),
            VariableSpec("VGRD", "10 m above ground", "v_wind_10m_mps"),
            VariableSpec("PRMSL", "mean sea level", "mslp_pa"),
            VariableSpec("TMP", "850 mb", "temperature_850_k"),
            VariableSpec("HGT", "500 mb", "geopotential_height_500_m"),
        ),
    ),
    "fourcastnetgfs": ModelSpec(
        dataset="fourcastnetgfs",
        stage="optional",
        smoke_run_time="2024-05-02T18:00:00Z",
        min_lead="18h",
        max_lead="42h",
        expected_steps=5,
        location_policy="deterministic_12_point_stencil",
        member_policy="deterministic",
        variables=(
            VariableSpec("TMP", "2 m above ground", "temperature_2m_k"),
            VariableSpec("UGRD", "10 m above ground", "u_wind_10m_mps"),
            VariableSpec("VGRD", "10 m above ground", "v_wind_10m_mps"),
            VariableSpec("PRMSL", "mean sea level", "mslp_pa"),
            VariableSpec("TMP", "850 mb", "temperature_850_k"),
            VariableSpec("HGT", "500 mb", "geopotential_height_500_m"),
        ),
    ),
    "nbmoc": ModelSpec(
        dataset="nbmoc",
        stage="probe",
        smoke_run_time="2026-06-23T18:00:00Z",
        min_lead="18h",
        max_lead="42h",
        expected_steps=5,
        location_policy="nbmoc_probe_3_point",
        member_policy="probe",
        variables=(
            VariableSpec("TMP", "2 m above ground", "temperature_2m_k"),
            VariableSpec("UGRD", "10 m above ground", "u_wind_10m_mps"),
            VariableSpec("VGRD", "10 m above ground", "v_wind_10m_mps"),
            VariableSpec("PRMSL", "mean sea level", "mslp_pa"),
        ),
    ),
}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def fs_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


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


def load_gribstream_token() -> str | None:
    env_key = os.environ.get("GRIBSTREAM_API_KEY")
    if env_key:
        return env_key.strip()
    if SECRET_FILE.exists():
        for line in SECRET_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("GRIBSTREAM_API_KEY="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token
    env_token = os.environ.get("GRIBSTREAM_API_TOKEN")
    return env_token.strip() if env_token else None


def coordinates_for_policy(policy: str) -> list[dict[str, Any]]:
    if policy == "hko_center_only":
        return [{"lat": HKO_LAT, "lon": HKO_LON, "name": "hko_center"}]
    if policy == "nbmoc_probe_3_point":
        selected = [item for item in STENCIL_12 if item[0] in {"hko_center", "marine_s_far", "marine_e_far"}]
    else:
        selected = list(STENCIL_12)
    return [
        {
            "lat": round(HKO_LAT + lat_offset, 6),
            "lon": round(HKO_LON + lon_offset, 6),
            "name": name,
        }
        for name, lat_offset, lon_offset in selected
    ]


def build_payload(spec: ModelSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timesList": [spec.smoke_run_time],
        "minLeadTime": spec.min_lead,
        "maxLeadTime": spec.max_lead,
        "coordinates": coordinates_for_policy(spec.location_policy),
        "variables": [variable.as_request_variable() for variable in spec.variables],
    }
    if spec.members:
        payload["members"] = list(spec.members)
    return payload


def expected_credits(spec: ModelSpec, payload: dict[str, Any]) -> int:
    return (
        spec.expected_steps
        * len(payload["variables"])
        * math.ceil(len(payload["coordinates"]) / 500)
        * max(len(payload.get("members", [])), 1)
    )


def raw_object_path(spec: ModelSpec, request_hash: str) -> Path:
    safe_run = spec.smoke_run_time.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    return RAW_ROOT / spec.dataset / f"run_time_utc={safe_run}" / f"{request_hash}.ndjson.gz"


def read_ndjson_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(fs_path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def collect_catalog_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"name", "code", "short_name"} and isinstance(nested, str):
                names.add(nested)
            else:
                names.update(collect_catalog_names(nested))
    elif isinstance(value, list):
        for item in value:
            names.update(collect_catalog_names(item))
    return names


def catalog_preflight(datasets: list[str]) -> dict[str, Any]:
    results: dict[str, Any] = {"updated_at_utc": utc_now_iso(), "datasets": {}}
    with httpx.Client(timeout=httpx.Timeout(20.0, read=60.0)) as client:
        for dataset in datasets:
            spec = MODEL_SPECS[dataset]
            dataset_result: dict[str, Any] = {
                "dataset_status": "not_checked",
                "parameter_status": "not_checked",
                "missing_parameter_names": [],
                "expected_members": len(spec.members),
            }
            try:
                meta_response = client.get(f"https://gribstream.com/api/v2/catalog/datasets/{dataset}")
                dataset_result["dataset_http_status"] = meta_response.status_code
                dataset_result["dataset_status"] = "ok" if meta_response.status_code == 200 else "failed"
                if meta_response.status_code == 200 and spec.members:
                    meta = meta_response.json()
                    catalog_members = meta.get("members") or []
                    member_count = int(meta.get("member_count") or len(catalog_members) or 0)
                    dataset_result["catalog_member_count"] = member_count
                    dataset_result["member_status"] = (
                        "ok"
                        if member_count >= len(spec.members) or member_count + 1 == len(spec.members)
                        else "member_count_mismatch"
                    )
                time.sleep(0.4)
                params_response = client.get(f"https://gribstream.com/api/v2/catalog/datasets/{dataset}/parameters")
                dataset_result["parameter_http_status"] = params_response.status_code
                if params_response.status_code == 200:
                    names = collect_catalog_names(params_response.json())
                    required_names = {variable.name for variable in spec.variables}
                    missing = sorted(required_names - names)
                    dataset_result["parameter_status"] = "ok" if not missing else "missing_names"
                    dataset_result["missing_parameter_names"] = missing
                else:
                    dataset_result["parameter_status"] = "failed"
            except Exception as exc:  # noqa: BLE001 - preflight must preserve per-dataset evidence
                dataset_result["dataset_status"] = "exception"
                dataset_result["error_class"] = type(exc).__name__
                dataset_result["error_message"] = sanitize_text(str(exc), None)
            results["datasets"][dataset] = dataset_result
            time.sleep(0.4)
    write_json(CATALOG_PREFLIGHT_JSON, results)
    return results


def upsert_chunk(
    database_url: str,
    *,
    spec: ModelSpec,
    payload: dict[str, Any],
    req_hash: str,
    status: str,
    expected_rows: int,
    expected_credit_count: int,
    raw_path: Path | None = None,
    response_sha256: str | None = None,
    row_count: int | None = None,
    http_status: int | None = None,
    error_class: str | None = None,
    error_message: str | None = None,
) -> None:
    psycopg = import_psycopg()
    from psycopg.types.json import Jsonb

    chunk_id = f"smoke_{spec.dataset}_{req_hash[:12]}"
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO nwp_tactical.acquisition_chunk (
                    chunk_id, dataset_code, run_times_utc, min_lead_hours, max_lead_hours,
                    location_policy, variable_bundle_id, member_policy, members, expected_rows,
                    expected_credits, request_json, request_sha256, status, raw_object_uri,
                    response_sha256, http_status, row_count, error_class, error_message,
                    started_at_utc, completed_at_utc
                )
                VALUES (
                    %s, %s, %s::timestamptz[], %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    now(), now()
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    raw_object_uri = EXCLUDED.raw_object_uri,
                    response_sha256 = EXCLUDED.response_sha256,
                    http_status = EXCLUDED.http_status,
                    row_count = EXCLUDED.row_count,
                    error_class = EXCLUDED.error_class,
                    error_message = EXCLUDED.error_message,
                    completed_at_utc = now()
                """,
                (
                    chunk_id,
                    spec.dataset,
                    payload["timesList"],
                    int(spec.min_lead.rstrip("h")),
                    int(spec.max_lead.rstrip("h")),
                    spec.location_policy,
                    "required_v1",
                    spec.member_policy,
                    list(spec.members) if spec.members else None,
                    expected_rows,
                    expected_credit_count,
                    Jsonb(payload),
                    req_hash,
                    status,
                    raw_path.as_posix() if raw_path else None,
                    response_sha256,
                    http_status,
                    row_count,
                    error_class,
                    error_message,
                ),
            )
            if raw_path and response_sha256 and row_count is not None:
                cursor.execute(
                    """
                    INSERT INTO nwp_tactical.raw_response_object (
                        chunk_id, object_uri, byte_size, sha256, content_type, retrieved_at_utc, row_count
                    )
                    VALUES (%s, %s, %s, %s, 'application/ndjson', now(), %s)
                    ON CONFLICT (chunk_id, sha256) DO NOTHING
                    """,
                    (
                        chunk_id,
                        raw_path.as_posix(),
                        os.path.getsize(fs_path(raw_path)),
                        response_sha256,
                        row_count,
                    ),
                )
        connection.commit()


def run(args: argparse.Namespace) -> int:
    token = load_gribstream_token()
    ensure_directory(EXPERIMENT_ROOT / "logs")
    ensure_directory(REQUEST_ROOT)
    if not token:
        write_json(EXPERIMENT_ROOT / "smoke_status.json", {"status": "blocked", "reason": "missing_gribstream_api_key"})
        return 2

    all_datasets = list(MODEL_SPECS)
    preflight = catalog_preflight(all_datasets)
    smoke_datasets = [item.strip() for item in args.smoke_datasets.split(",") if item.strip()]
    unknown = sorted(set(smoke_datasets) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown smoke dataset(s): {', '.join(unknown)}")
    smoke_datasets = smoke_datasets[: args.max_data_calls]

    results: list[dict[str, Any]] = []
    retry_config = RetryConfig(
        max_attempts=args.api_max_attempts,
        min_interval_seconds=args.api_min_interval_seconds,
        default_rate_limit_pause_seconds=args.pause_on_429_seconds,
        max_retry_delay_seconds=args.max_retry_after_seconds,
    )
    with GribStreamClient(token, retry_config=retry_config, event_log_path=API_EVENT_LOG) as client:
        for dataset in smoke_datasets:
            spec = MODEL_SPECS[dataset]
            payload = build_payload(spec)
            req_hash = request_sha256(payload)
            request_path = REQUEST_ROOT / f"{dataset}_{req_hash[:12]}.json"
            write_json(request_path, payload)
            raw_path = raw_object_path(spec, req_hash)
            expected_credit_count = expected_credits(spec, payload)
            expected_rows = (
                spec.expected_steps
                * len(payload["coordinates"])
                * max(len(payload.get("members", [])), 1)
            )
            try:
                if os.path.exists(fs_path(raw_path)):
                    manifest = None
                    http_status = 200
                    raw_source = "raw_reused"
                else:
                    manifest = client.post_runs_to_gzip(
                        dataset=dataset,
                        payload=payload,
                        output_path=raw_path,
                        request_hash=req_hash,
                    )
                    http_status = manifest.http_status
                    raw_source = "api_fetched"
                rows = read_ndjson_gzip(raw_path)
                returned_runs = sorted({str(row.get("forecasted_at", "")) for row in rows})
                wrong_runs = [value for value in returned_runs if value != spec.smoke_run_time]
                status = "completed" if rows and not wrong_runs else "completed_empty" if not rows else "failed"
                response_hash = sha256_file(raw_path)
                upsert_chunk(
                    args.database_url,
                    spec=spec,
                    payload=payload,
                    req_hash=req_hash,
                    status=status,
                    expected_rows=expected_rows,
                    expected_credit_count=expected_credit_count,
                    raw_path=raw_path,
                    response_sha256=response_hash,
                    row_count=len(rows),
                    http_status=http_status,
                    error_class=None if not wrong_runs else "unexpected_forecasted_at",
                    error_message=";".join(wrong_runs[:5]) if wrong_runs else None,
                )
                results.append(
                    {
                        "dataset": dataset,
                        "stage": spec.stage,
                        "status": status,
                        "http_status": http_status,
                        "row_count": len(rows),
                        "expected_rows": expected_rows,
                        "expected_credits": expected_credit_count,
                        "request_sha256": req_hash,
                        "raw_path": raw_path.as_posix(),
                        "source": raw_source,
                        "wrong_forecasted_at_count": len(wrong_runs),
                    }
                )
            except GribStreamRequestError as exc:
                upsert_chunk(
                    args.database_url,
                    spec=spec,
                    payload=payload,
                    req_hash=req_hash,
                    status="failed",
                    expected_rows=expected_rows,
                    expected_credit_count=expected_credit_count,
                    http_status=exc.status_code,
                    error_class=exc.error_class,
                    error_message=sanitize_text(str(exc), token),
                )
                results.append(
                    {
                        "dataset": dataset,
                        "stage": spec.stage,
                        "status": "failed",
                        "http_status": exc.status_code,
                        "row_count": 0,
                        "expected_rows": expected_rows,
                        "expected_credits": expected_credit_count,
                        "request_sha256": req_hash,
                        "raw_path": "",
                        "error_class": exc.error_class,
                        "error_message": sanitize_text(str(exc), token),
                    }
                )
                if exc.status_code in {401, 403, 429}:
                    break
    write_csv(
        SMOKE_RESULTS_CSV,
        results,
        [
            "dataset",
            "stage",
            "status",
            "http_status",
            "row_count",
            "expected_rows",
            "expected_credits",
            "request_sha256",
            "raw_path",
            "source",
            "wrong_forecasted_at_count",
            "error_class",
            "error_message",
        ],
    )
    status = {
        "status": "passed" if results and all(row["status"] == "completed" for row in results) else "failed",
        "updated_at_utc": utc_now_iso(),
        "database": redact_database_url(args.database_url),
        "catalog_preflight_path": CATALOG_PREFLIGHT_JSON.as_posix(),
        "smoke_results_path": SMOKE_RESULTS_CSV.as_posix(),
        "executed_data_calls": len(results),
        "requested_data_call_limit": args.max_data_calls,
        "api_shape_note": "The GribStream /api/v2/{dataset}/runs endpoint is per dataset, so 2-3 data calls cannot fetch every model.",
        "catalog_preflight_summary": {
            dataset: preflight["datasets"][dataset]["dataset_status"]
            for dataset in all_datasets
        },
    }
    write_json(EXPERIMENT_ROOT / "smoke_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if status["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny tactical H24N GribStream smoke test.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--max-data-calls", type=int, default=3)
    parser.add_argument(
        "--smoke-datasets",
        default="gfs,ifsenfo,aifsoper",
        help="Comma-separated dataset codes to execute, capped by --max-data-calls.",
    )
    parser.add_argument("--api-min-interval-seconds", type=float, default=12.0)
    parser.add_argument("--api-max-attempts", type=int, default=2)
    parser.add_argument("--pause-on-429-seconds", type=float, default=300.0)
    parser.add_argument("--max-retry-after-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
