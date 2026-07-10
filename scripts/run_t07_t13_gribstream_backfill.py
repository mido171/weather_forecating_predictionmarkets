from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from hkg_tmax.gribstream.catalog import ResolvedSelector, stable_json_sha
from hkg_tmax.gribstream.client import (
    GribStreamClient,
    GribStreamRequestError,
    RetryConfig,
    canonical_request_json,
    request_sha256,
    sanitize_text,
    sha256_file,
)
from hkg_tmax.gribstream.normalizer import normalize_runs_ndjson_gzip
from hkg_tmax.gribstream.store import (
    ingest_response,
    load_location_ids,
    mark_request_failed,
    register_request_started,
)
from hkg_tmax.gribstream.planner import load_canonical_locations
from hkg_tmax_db.connection import import_psycopg, redact_database_url


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments/0214_t07_t13_gribstream_backfill"
RAW_ROOT = REPO_ROOT / "data/raw/gribstream"
STATUS_PATH = EXPERIMENT_ROOT / "logs/t07_t13_status.json"
API_EVENT_LOG = EXPERIMENT_ROOT / "logs/gribstream_api_events.jsonl"
LEDGER_PATH = EXPERIMENT_ROOT / "resume_ledger.jsonl"
PLAN_PATH = EXPERIMENT_ROOT / "planned_chunks.csv"
MANIFEST_PATH = EXPERIMENT_ROOT / "executed_chunks.csv"
BLOCKERS_PATH = EXPERIMENT_ROOT / "blockers.csv"
SECRET_FILE = REPO_ROOT / "secrets/local/gribstream.env"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"
TRANSIENT_STOP_STATUSES = {429}


CORE_SURFACE = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "u_wind_10m",
    "v_wind_10m",
    "mean_sea_level_pressure",
    "surface_pressure",
    "cloud_cover_total",
    "total_precipitation",
]

CORE_PRESSURE = [
    "temperature_925hpa",
    "temperature_850hpa",
    "temperature_700hpa",
    "temperature_500hpa",
    "relative_humidity_925hpa",
    "relative_humidity_850hpa",
    "relative_humidity_700hpa",
    "geopotential_height_850hpa",
    "geopotential_height_500hpa",
]


DATASETS: dict[str, dict[str, Any]] = {
    "gfs": {
        "task_id": "T07",
        "provider": "NOAA",
        "archive_start": "2021-03-22",
        "archive_end": "2026-06-23",
        "members": None,
        "params": CORE_SURFACE + CORE_PRESSURE,
        "kickoff_params": CORE_SURFACE[:5],
        "priority": 30,
    },
    "gefsatmosmean": {
        "task_id": "T08",
        "provider": "NOAA",
        "archive_start": "2020-10-01",
        "archive_end": "2026-06-23",
        "members": None,
        "params": CORE_SURFACE[:7] + CORE_PRESSURE[:4],
        "kickoff_params": CORE_SURFACE[:4],
        "priority": 40,
    },
    "gefsatmos": {
        "task_id": "T08",
        "provider": "NOAA",
        "archive_start": "2020-10-01",
        "archive_end": "2026-06-23",
        "members": "catalog",
        "params": ["temperature_2m", "dew_point_2m", "u_wind_10m", "v_wind_10m"],
        "kickoff_params": ["temperature_2m"],
        "priority": 45,
    },
    "ifsoper": {
        "task_id": "T09",
        "provider": "ECMWF",
        "archive_start": "2024-02-28",
        "archive_end": "2026-06-23",
        "members": None,
        "params": CORE_SURFACE[:7] + CORE_PRESSURE[:4],
        "kickoff_params": CORE_SURFACE[:4],
        "priority": 50,
    },
    "ifsenfo": {
        "task_id": "T09",
        "provider": "ECMWF",
        "archive_start": "2024-03-01",
        "archive_end": "2026-06-23",
        "members": "catalog",
        "params": ["temperature_2m", "dew_point_2m", "u_wind_10m", "v_wind_10m"],
        "kickoff_params": ["temperature_2m"],
        "priority": 55,
    },
    "graphcast": {
        "task_id": "T10",
        "provider": "Google/ECMWF-derived",
        "archive_start": "2024-04-25",
        "archive_end": "2026-05-05",
        "members": None,
        "params": ["temperature_2m", "u_wind_10m", "v_wind_10m", "mean_sea_level_pressure"],
        "kickoff_params": ["temperature_2m"],
        "priority": 60,
    },
    "fourcastnetgfs": {
        "task_id": "T10",
        "provider": "NVIDIA/NOAA-derived",
        "archive_start": "2024-05-02",
        "archive_end": "2026-03-01",
        "members": None,
        "params": ["temperature_2m", "u_wind_10m", "v_wind_10m", "mean_sea_level_pressure"],
        "kickoff_params": ["temperature_2m"],
        "priority": 61,
    },
    "aifsoper": {
        "task_id": "T10",
        "provider": "ECMWF",
        "archive_start": "2025-02-25",
        "archive_end": "2026-06-23",
        "members": None,
        "params": CORE_SURFACE[:8],
        "kickoff_params": ["temperature_2m"],
        "priority": 62,
    },
    "aifsenfo": {
        "task_id": "T10",
        "provider": "ECMWF",
        "archive_start": "2025-07-02",
        "archive_end": "2026-06-23",
        "members": "catalog",
        "params": ["temperature_2m"],
        "kickoff_params": ["temperature_2m"],
        "priority": 63,
    },
    "aigfssfc": {
        "task_id": "T10",
        "provider": "NOAA",
        "archive_start": "2026-04-16",
        "archive_end": "2026-06-23",
        "members": None,
        "params": ["temperature_2m", "dew_point_2m", "u_wind_10m", "v_wind_10m", "mean_sea_level_pressure"],
        "kickoff_params": ["temperature_2m"],
        "priority": 64,
    },
    "aigefssfc": {
        "task_id": "T10",
        "provider": "NOAA",
        "archive_start": "2025-06-01",
        "archive_end": "2026-06-23",
        "members": "catalog",
        "params": ["temperature_2m"],
        "kickoff_params": ["temperature_2m"],
        "priority": 65,
    },
    "cwawrf15": {
        "task_id": "T11",
        "provider": "CWA",
        "archive_start": "2026-06-21",
        "archive_end": "2026-06-23",
        "members": None,
        "params": CORE_SURFACE[:9] + CORE_PRESSURE[:4],
        "kickoff_params": CORE_SURFACE[:9] + CORE_PRESSURE[:4],
        "priority": 10,
    },
    "gdas": {
        "task_id": "T12",
        "provider": "NOAA",
        "archive_start": "2026-06-23",
        "archive_end": "2026-06-23",
        "members": None,
        "params": ["temperature_2m", "dew_point_2m", "u_wind_10m", "v_wind_10m"],
        "kickoff_params": ["temperature_2m"],
        "priority": 70,
    },
    "cdas": {
        "task_id": "T12",
        "provider": "NOAA",
        "archive_start": "2026-06-23",
        "archive_end": "2026-06-23",
        "members": None,
        "params": ["temperature_2m", "dew_point_2m", "u_wind_10m", "v_wind_10m"],
        "kickoff_params": ["temperature_2m"],
        "priority": 71,
    },
    "nbmoc": {
        "task_id": "T12",
        "provider": "NOAA",
        "archive_start": "2026-06-23",
        "archive_end": "2026-06-23",
        "members": None,
        "params": ["temperature_2m", "dew_point_2m", "cloud_cover_total"],
        "kickoff_params": ["temperature_2m"],
        "priority": 72,
    },
    "nbmparoc": {
        "task_id": "T12",
        "provider": "NOAA",
        "archive_start": "2026-06-23",
        "archive_end": "2026-06-23",
        "members": None,
        "params": ["temperature_2m", "dew_point_2m", "cloud_cover_total"],
        "kickoff_params": ["temperature_2m"],
        "priority": 73,
    },
    "uvi": {
        "task_id": "T12",
        "provider": "NOAA",
        "archive_start": "2026-06-23",
        "archive_end": "2026-06-23",
        "members": None,
        "params": [],
        "kickoff_params": [],
        "priority": 80,
    },
}


@dataclass(frozen=True)
class SharedResolution:
    selector: ResolvedSelector
    request_fragment: dict[str, Any]
    output_alias: str
    output_unit: str
    dataset_catalog_sha256: str
    dataset_meta: dict[str, Any]


@dataclass(frozen=True)
class ChunkJob:
    task_id: str
    dataset: str
    provider: str
    shared_parameter: str
    run_date: date
    forecasted_from: str
    forecasted_until: str
    min_lead_time: str
    max_lead_time: str
    members: tuple[int, ...]
    priority: int
    stage: str
    estimated_credits: int
    estimated_rows: int
    request_sha256: str
    raw_path: Path
    payload: dict[str, Any]
    resolution: SharedResolution


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


def write_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    with open(fs_path(path), "w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with open(fs_path(path), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def load_gribstream_token() -> str | None:
    env_key = os.environ.get("GRIBSTREAM_API_KEY")
    if env_key:
        return env_key.strip()
    if not SECRET_FILE.exists():
        env_token = os.environ.get("GRIBSTREAM_API_TOKEN")
        return env_token.strip() if env_token else None
    for line in SECRET_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("GRIBSTREAM_API_KEY="):
            token = line.split("=", 1)[1].strip()
            if token:
                return token
    env_token = os.environ.get("GRIBSTREAM_API_TOKEN")
    return env_token.strip() if env_token else None


def write_status(stage: str, status: str, **details: Any) -> None:
    write_json(
        STATUS_PATH,
        {
            "stage": stage,
            "status": status,
            "updated_at_utc": utc_now_iso(),
            "pid": os.getpid(),
            **details,
        },
    )


def git_output(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def stable_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"),
    ).hexdigest()


def date_range_days(start: str, end: str) -> list[date]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if end_date < start_date:
        return []
    return [start_date + timedelta(days=offset) for offset in range((end_date - start_date).days + 1)]


def chunks(items: list[int], size: int) -> list[tuple[int, ...]]:
    if not items:
        return [()]
    return [tuple(items[index : index + size]) for index in range(0, len(items), size)]


def extract_request_fragment(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("resolved_request", "default_request", "request"):
        value = payload.get(key)
        if isinstance(value, dict) and value.get("variables"):
            return value
    if payload.get("variables"):
        return payload
    return None


def extract_visible_alias(fragment: dict[str, Any], fallback: str) -> str:
    for expression in fragment.get("expressions", []):
        if not expression.get("hidden", False) and expression.get("alias"):
            return str(expression["alias"])
    for variable in fragment.get("variables", []):
        if not variable.get("hidden", False) and variable.get("alias"):
            return str(variable["alias"])
    return fallback


def selector_from_fragment(
    *,
    dataset: str,
    shared_parameter: str,
    fragment: dict[str, Any],
    source_payload: dict[str, Any],
    unit: str,
    retrieved_at_utc: str,
) -> ResolvedSelector:
    alias = extract_visible_alias(fragment, shared_parameter)
    visible_variables = [item for item in fragment.get("variables", []) if not item.get("hidden", False)]
    expressions = [item for item in fragment.get("expressions", []) if not item.get("hidden", False)]
    if len(visible_variables) == 1 and not expressions:
        variable = visible_variables[0]
        native_name = str(variable["name"])
        native_level = str(variable["level"])
        native_info = str(variable.get("info", ""))
    else:
        native_name = f"EXPR:{alias}"
        native_level = "derived"
        native_info = stable_sha(fragment)[:16]
    return ResolvedSelector(
        dataset=dataset,
        semantic_variable=shared_parameter,
        semantic_family="shared_parameter",
        native_name=native_name,
        native_level=native_level,
        native_info=native_info,
        alias=alias,
        native_unit=unit,
        source_sha256=stable_sha({"dataset": dataset, "shared_parameter": shared_parameter, "fragment": fragment}),
        retrieved_at_utc=retrieved_at_utc,
        source_json=source_payload,
    )


def resolve_shared_parameter(
    client: httpx.Client,
    *,
    dataset: str,
    shared_parameter: str,
    dataset_meta: dict[str, Any],
) -> SharedResolution | None:
    response = client.get(
        f"https://gribstream.com/api/v2/catalog/shared-parameters/{shared_parameter}",
        params={"dataset": dataset, "alias": shared_parameter},
    )
    if response.status_code != 200:
        return None
    payload = response.json()
    fragment = extract_request_fragment(payload)
    if fragment is None:
        return None
    unit = str(payload.get("units") or payload.get("unit") or "")
    retrieved_at = utc_now_iso()
    selector = selector_from_fragment(
        dataset=dataset,
        shared_parameter=shared_parameter,
        fragment=fragment,
        source_payload=payload,
        unit=unit,
        retrieved_at_utc=retrieved_at,
    )
    return SharedResolution(
        selector=selector,
        request_fragment=fragment,
        output_alias=selector.alias,
        output_unit=unit,
        dataset_catalog_sha256=stable_sha(dataset_meta),
        dataset_meta=dataset_meta,
    )


def fetch_dataset_meta(client: httpx.Client, dataset: str) -> dict[str, Any] | None:
    response = client.get(f"https://gribstream.com/api/v2/catalog/datasets/{dataset}")
    if response.status_code != 200:
        return None
    return response.json()


def build_payload(
    *,
    job_base: dict[str, Any],
    coordinates: list[dict[str, Any]],
    resolution: SharedResolution,
    members: tuple[int, ...],
) -> dict[str, Any]:
    payload = {
        **job_base,
        "coordinates": coordinates,
        "variables": resolution.request_fragment.get("variables", []),
    }
    expressions = resolution.request_fragment.get("expressions", [])
    if expressions:
        payload["expressions"] = expressions
    if members:
        payload["members"] = list(members)
    return payload


def estimate_credits(payload: dict[str, Any], coordinate_count: int) -> int:
    start = datetime.fromisoformat(payload["forecastedFrom"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(payload["forecastedUntil"].replace("Z", "+00:00"))
    run_count = int((end - start).total_seconds() // (6 * 3600)) + 1
    max_lead_hours = int(str(payload["maxLeadTime"]).rstrip("h"))
    min_lead_hours = int(str(payload["minLeadTime"]).rstrip("h"))
    lead_count = max_lead_hours - min_lead_hours + 1
    parameter_count = len(payload.get("variables", [])) + len(payload.get("expressions", []))
    member_count = max(len(payload.get("members", [])), 1)
    return run_count * lead_count * max(parameter_count, 1) * math.ceil(coordinate_count / 500) * member_count


def raw_object_path(dataset: str, run_date: date, shared_parameter: str, request_hash: str) -> Path:
    return (
        RAW_ROOT
        / dataset
        / "runs"
        / f"run_date={run_date.isoformat()}"
        / shared_parameter
        / f"{request_hash}.ndjson.gz"
    )


def existing_request_hashes() -> set[str]:
    if not MANIFEST_PATH.exists():
        return set()
    with open(fs_path(MANIFEST_PATH), newline="", encoding="utf-8") as handle:
        return {row["request_sha256"] for row in csv.DictReader(handle) if row.get("status") == "completed"}


def write_csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_directory(path.parent)
    with open(fs_path(path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    ensure_directory(path.parent)
    exists = path.exists()
    with open(fs_path(path), "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_jobs(
    *,
    database_url: str,
    mode: str,
    member_chunk_size: int,
    existing_hashes: set[str] | None = None,
    credit_budget: int | None = None,
    datasets: set[str] | None = None,
) -> tuple[list[ChunkJob], list[dict[str, Any]]]:
    locations = load_canonical_locations(database_url)
    coordinates = [location.as_gribstream_coordinate() for location in locations]
    jobs: list[ChunkJob] = []
    blockers: list[dict[str, Any]] = [
        {
            "dataset": "hko_arwf",
            "shared_parameter": "",
            "blocker": "T13 is an HKO ARWF exact-vintage collector task, not a GribStream dataset; it requires a separate HKO endpoint collector.",
        },
    ]
    completed_hashes = existing_hashes or set()
    prepared: list[dict[str, Any]] = []

    def make_job(
        *,
        dataset: str,
        meta: dict[str, Any],
        resolution: SharedResolution,
        shared_parameter: str,
        run_day: date,
        member_group: tuple[int, ...],
    ) -> ChunkJob | None:
        job_base = {
            "forecastedFrom": f"{run_day.isoformat()}T00:00:00Z",
            "forecastedUntil": f"{run_day.isoformat()}T18:00:00Z",
            "minLeadTime": "0h",
            "maxLeadTime": "84h",
        }
        payload = build_payload(
            job_base=job_base,
            coordinates=coordinates,
            resolution=resolution,
            members=member_group,
        )
        req_hash = request_sha256(payload)
        if req_hash in completed_hashes:
            return None
        credits = estimate_credits(payload, len(coordinates))
        return ChunkJob(
            task_id=str(meta["task_id"]),
            dataset=dataset,
            provider=str(meta["provider"]),
            shared_parameter=shared_parameter,
            run_date=run_day,
            forecasted_from=job_base["forecastedFrom"],
            forecasted_until=job_base["forecastedUntil"],
            min_lead_time=job_base["minLeadTime"],
            max_lead_time=job_base["maxLeadTime"],
            members=member_group,
            priority=int(meta["priority"]),
            stage=mode,
            estimated_credits=credits,
            estimated_rows=4 * 85 * max(len(member_group), 1) * len(coordinates),
            request_sha256=req_hash,
            raw_path=raw_object_path(dataset, run_day, shared_parameter, req_hash),
            payload=payload,
            resolution=resolution,
        )

    with httpx.Client(timeout=httpx.Timeout(20.0, read=60.0)) as public_client:
        for dataset, meta in DATASETS.items():
            if datasets is not None and dataset not in datasets:
                continue
            dataset_meta = fetch_dataset_meta(public_client, dataset)
            if dataset_meta is None:
                blockers.append({"dataset": dataset, "shared_parameter": "", "blocker": "dataset_catalog_not_available"})
                continue
            catalog_members = dataset_meta.get("members") if meta["members"] == "catalog" else None
            member_groups = chunks([int(value) for value in (catalog_members or [])], member_chunk_size)
            if meta["members"] == "catalog" and not catalog_members:
                blockers.append({"dataset": dataset, "shared_parameter": "", "blocker": "ensemble_members_missing_in_catalog"})
                continue
            if dataset == "uvi":
                blockers.append({"dataset": dataset, "shared_parameter": "", "blocker": "uvi has no shared Tmax-relevant parameter mapping; registered for T12 only"})
                continue
            if mode == "kickoff":
                run_dates = (
                    date_range_days(meta["archive_start"], meta["archive_end"])
                    if dataset == "cwawrf15"
                    else [date.fromisoformat(meta["archive_end"])]
                )
                parameters = meta["kickoff_params"]
            else:
                run_dates = date_range_days(meta["archive_start"], meta["archive_end"])
                parameters = meta["params"]
            if dataset == "cwawrf15":
                run_dates = list(reversed(run_dates))
            resolutions: dict[str, SharedResolution] = {}
            for shared_parameter in parameters:
                resolution = resolve_shared_parameter(
                    public_client,
                    dataset=dataset,
                    shared_parameter=shared_parameter,
                    dataset_meta=dataset_meta,
                )
                if resolution is None:
                    blockers.append(
                        {
                            "dataset": dataset,
                            "shared_parameter": shared_parameter,
                            "blocker": "shared_parameter_not_supported_or_unresolved",
                        },
                    )
                    continue
                resolutions[shared_parameter] = resolution
            prepared.append(
                {
                    "dataset": dataset,
                    "meta": meta,
                    "member_groups": member_groups,
                    "run_dates": run_dates,
                    "parameters": [parameter for parameter in parameters if parameter in resolutions],
                    "resolutions": resolutions,
                    "combo_index": 0,
                    "date_index": 0,
                },
            )

    if mode == "full":
        planned_credits = 0
        plan_credit_limit = max(int((credit_budget or 85000) * 1.25), 1)
        states = [state for state in prepared if state["parameters"] and state["run_dates"]]
        states.sort(key=lambda state: (int(state["meta"]["priority"]), state["dataset"]))
        while states and planned_credits < plan_credit_limit:
            next_states: list[dict[str, Any]] = []
            progressed = False
            for state in states:
                combos = [
                    (parameter, member_group)
                    for parameter in state["parameters"]
                    for member_group in state["member_groups"]
                ]
                while state["date_index"] < len(state["run_dates"]):
                    if state["combo_index"] >= len(combos):
                        state["combo_index"] = 0
                        state["date_index"] += 1
                        continue
                    parameter, member_group = combos[state["combo_index"]]
                    state["combo_index"] += 1
                    job = make_job(
                        dataset=state["dataset"],
                        meta=state["meta"],
                        resolution=state["resolutions"][parameter],
                        shared_parameter=parameter,
                        run_day=state["run_dates"][state["date_index"]],
                        member_group=member_group,
                    )
                    if job is None:
                        continue
                    jobs.append(job)
                    planned_credits += job.estimated_credits
                    progressed = True
                    break
                if state["date_index"] < len(state["run_dates"]):
                    next_states.append(state)
                if planned_credits >= plan_credit_limit:
                    break
            if not progressed:
                break
            states = next_states
    else:
        for state in prepared:
            for run_day in state["run_dates"]:
                for shared_parameter in state["parameters"]:
                    for member_group in state["member_groups"]:
                        job = make_job(
                            dataset=state["dataset"],
                            meta=state["meta"],
                            resolution=state["resolutions"][shared_parameter],
                            shared_parameter=shared_parameter,
                            run_day=run_day,
                            member_group=member_group,
                        )
                        if job is not None:
                            jobs.append(job)
        jobs.sort(
            key=lambda job: (
                job.priority,
                job.run_date,
                job.dataset,
                job.shared_parameter,
                ",".join(str(member) for member in job.members),
            ),
        )
    return jobs, blockers


def write_plan(jobs: list[ChunkJob], blockers: list[dict[str, Any]]) -> None:
    write_csv_rows(
        PLAN_PATH,
        [
            {
                "task_id": job.task_id,
                "dataset": job.dataset,
                "shared_parameter": job.shared_parameter,
                "run_date": job.run_date.isoformat(),
                "members": " ".join(str(member) for member in job.members),
                "estimated_credits": job.estimated_credits,
                "estimated_rows": job.estimated_rows,
                "request_sha256": job.request_sha256,
                "raw_path": repo_rel(job.raw_path),
                "stage": job.stage,
            }
            for job in jobs
        ],
        [
            "task_id",
            "dataset",
            "shared_parameter",
            "run_date",
            "members",
            "estimated_credits",
            "estimated_rows",
            "request_sha256",
            "raw_path",
            "stage",
        ],
    )
    write_csv_rows(BLOCKERS_PATH, blockers, ["dataset", "shared_parameter", "blocker"])


def response_manifest_for_existing(job: ChunkJob) -> Any:
    from hkg_tmax.gribstream.client import ResponseManifest
    from hkg_tmax.gribstream.normalizer import iter_ndjson_gzip

    rows = iter_ndjson_gzip(job.raw_path)
    return ResponseManifest(
        provider="GribStream",
        dataset=job.dataset,
        endpoint="runs",
        request_sha256=job.request_sha256,
        object_path=job.raw_path,
        byte_size=os.path.getsize(fs_path(job.raw_path)),
        sha256=sha256_file(job.raw_path),
        content_type="application/ndjson",
        retrieved_at_utc=utc_now_iso(),
        row_count=len(rows),
        http_status=200,
        attempt_count=0,
    )


def run_job(
    *,
    job: ChunkJob,
    client: GribStreamClient,
    database_url: str,
    location_ids: dict[str, int],
) -> dict[str, Any]:
    request_id = register_request_started(
        database_url,
        provider="GribStream",
        model_code=job.dataset,
        endpoint="runs",
        canonical_payload=job.payload,
        request_hash=job.request_sha256,
    )
    if os.path.exists(fs_path(job.raw_path)):
        manifest = response_manifest_for_existing(job)
        source = "raw_reused"
    else:
        manifest = client.post_runs_to_gzip(
            dataset=job.dataset,
            payload=job.payload,
            output_path=job.raw_path,
            request_hash=job.request_sha256,
        )
        source = "api_fetched"
    normalized = normalize_runs_ndjson_gzip(
        manifest.object_path,
        value_alias=job.resolution.output_alias,
        location_ids_by_code=location_ids,
    )
    summary = ingest_response(
        database_url,
        request_id=request_id,
        selector=job.resolution.selector,
        manifest=manifest,
        points=normalized.points,
        rejected_rows=normalized.rejected_rows,
        provider=job.provider,
        archive_start=DATASETS[job.dataset]["archive_start"],
        disposition=f"{job.task_id}_{job.stage}",
        catalog_snapshot_sha256=job.resolution.dataset_catalog_sha256,
    )
    return {
        "status": "completed",
        "source": source,
        "task_id": job.task_id,
        "dataset": job.dataset,
        "shared_parameter": job.shared_parameter,
        "run_date": job.run_date.isoformat(),
        "members": " ".join(str(member) for member in job.members),
        "estimated_credits": job.estimated_credits,
        "row_count": manifest.row_count,
        "point_rows": summary.inserted_or_updated_points,
        "rejected_rows": summary.rejected_rows,
        "request_sha256": job.request_sha256,
        "raw_path": repo_rel(job.raw_path),
        "response_sha256": manifest.sha256,
        "completed_at_utc": utc_now_iso(),
    }


def secret_scan(token: str | None) -> dict[str, Any]:
    if not token:
        return {"status": "skipped_no_token", "matches": []}
    roots = [
        EXPERIMENT_ROOT,
        REPO_ROOT / "scripts/run_t07_t13_gribstream_backfill.py",
        REPO_ROOT / "scripts/check_t07_t13_gribstream_status.py",
    ]
    matches: list[str] = []
    for root in roots:
        files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
        for path in files:
            if token in path.read_text(encoding="utf-8", errors="ignore"):
                matches.append(repo_rel(path))
    return {"status": "passed" if not matches else "failed", "matches": matches}


def write_runbook(args: argparse.Namespace, jobs: list[ChunkJob], blockers: list[dict[str, Any]]) -> None:
    write_text(
        EXPERIMENT_ROOT / "README.md",
        "# T07-T13 GribStream Backfill Supervisor\n\n"
        "This folder tracks the one-thread GribStream acquisition run covering T07-T12 and the T13 non-GribStream blocker/status note.\n\n"
        f"Mode: `{args.mode}`\n\n"
        f"Planned chunks: {len(jobs)}\n\n"
        f"Initial blockers: {len(blockers)}\n",
    )
    write_text(
        EXPERIMENT_ROOT / "operator_runbook.md",
        "# Operator Runbook\n\n"
        "Check status without API calls:\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\check_t07_t13_gribstream_status.py\n"
        "```\n\n"
        "Resume the same strategy:\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\run_t07_t13_gribstream_backfill.py --mode kickoff\n"
        "```\n\n"
        "The runner uses one authenticated thread, canonical request SHA-256 keys, atomic raw gzip writes, and DB upserts.",
    )


def run(args: argparse.Namespace) -> int:
    if not args.allow_legacy_broad_runner:
        ensure_directory(EXPERIMENT_ROOT / "logs")
        write_status(
            "retired",
            "blocked",
            reason=(
                "retired_after_tactical_h24n_plan; use "
                "scripts/run_tactical_gribstream_h24n_smoke.py and the consolidated T07-T12 task"
            ),
        )
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "legacy broad T07-T13 runner is retired",
                    "replacement": "scripts/run_tactical_gribstream_h24n_smoke.py",
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return 2
    token = load_gribstream_token()
    ensure_directory(EXPERIMENT_ROOT / "logs")
    if not token:
        write_status("startup", "blocked", reason="missing_gribstream_api_key")
        return 2
    write_status(
        "planning",
        "running",
        mode=args.mode,
        database=redact_database_url(args.database_url),
        credit_budget=args.credit_budget,
    )
    done_hashes = existing_request_hashes()
    jobs, blockers = build_jobs(
        database_url=args.database_url,
        mode=args.mode,
        member_chunk_size=args.member_chunk_size,
        existing_hashes=done_hashes,
        credit_budget=args.credit_budget,
        datasets=set(args.datasets.split(",")) if args.datasets else None,
    )
    write_plan(jobs, blockers)
    write_runbook(args, jobs, blockers)
    location_ids = load_location_ids(args.database_url)
    remaining = [job for job in jobs if job.request_sha256 not in done_hashes]
    spent = 0
    completed = 0
    failed = 0
    skipped_budget = 0
    append_jsonl(
        LEDGER_PATH,
        {
            "event": "run_started",
            "mode": args.mode,
            "planned_chunks": len(jobs),
            "remaining_chunks": len(remaining),
            "blockers": len(blockers),
            "credit_budget": args.credit_budget,
            "updated_at_utc": utc_now_iso(),
        },
    )
    retry_config = RetryConfig(
        max_attempts=args.api_max_attempts,
        min_interval_seconds=args.api_min_interval_seconds,
        default_rate_limit_pause_seconds=args.pause_on_429_seconds,
        max_retry_delay_seconds=args.max_retry_after_seconds,
    )
    manifest_columns = [
        "status",
        "source",
        "task_id",
        "dataset",
        "shared_parameter",
        "run_date",
        "members",
        "estimated_credits",
        "row_count",
        "point_rows",
        "rejected_rows",
        "request_sha256",
        "raw_path",
        "response_sha256",
        "completed_at_utc",
        "error_class",
        "error_message",
    ]
    with GribStreamClient(token, retry_config=retry_config, event_log_path=API_EVENT_LOG) as client:
        for index, job in enumerate(remaining, start=1):
            if args.max_chunks and completed + failed >= args.max_chunks:
                break
            if spent + job.estimated_credits > args.credit_budget:
                skipped_budget += 1
                continue
            write_status(
                "fetching",
                "running",
                completed_chunks=completed,
                failed_chunks=failed,
                remaining_chunks=len(remaining) - index + 1,
                spent_estimated_credits=spent,
                current_dataset=job.dataset,
                current_parameter=job.shared_parameter,
                current_run_date=job.run_date.isoformat(),
                current_members=list(job.members),
                current_request_sha256=job.request_sha256,
            )
            try:
                row = run_job(job=job, client=client, database_url=args.database_url, location_ids=location_ids)
                append_csv_row(MANIFEST_PATH, row, manifest_columns)
                append_jsonl(LEDGER_PATH, {"event": "chunk_completed", **row})
                completed += 1
                spent += job.estimated_credits
            except GribStreamRequestError as exc:
                mark_request_failed(
                    args.database_url,
                    request_hash=job.request_sha256,
                    error_class=exc.error_class,
                    error_message=str(exc),
                )
                row = {
                    "status": "failed",
                    "source": "api_or_ingest",
                    "task_id": job.task_id,
                    "dataset": job.dataset,
                    "shared_parameter": job.shared_parameter,
                    "run_date": job.run_date.isoformat(),
                    "members": " ".join(str(member) for member in job.members),
                    "estimated_credits": job.estimated_credits,
                    "row_count": 0,
                    "point_rows": 0,
                    "rejected_rows": 0,
                    "request_sha256": job.request_sha256,
                    "raw_path": repo_rel(job.raw_path),
                    "response_sha256": "",
                    "completed_at_utc": utc_now_iso(),
                    "error_class": exc.error_class,
                    "error_message": sanitize_text(str(exc), token),
                }
                append_csv_row(MANIFEST_PATH, row, manifest_columns)
                append_jsonl(LEDGER_PATH, {"event": "chunk_failed", **row})
                failed += 1
                if exc.status_code in {401, 403}:
                    write_status("auth_failed", "blocked", failed_chunk=row)
                    break
                if exc.status_code in TRANSIENT_STOP_STATUSES:
                    write_status("rate_limited", "paused_or_stopped", failed_chunk=row)
                    break
            except Exception as exc:  # noqa: BLE001 - runner must preserve chunk evidence and continue cautiously
                row = {
                    "status": "failed",
                    "source": "local_exception",
                    "task_id": job.task_id,
                    "dataset": job.dataset,
                    "shared_parameter": job.shared_parameter,
                    "run_date": job.run_date.isoformat(),
                    "members": " ".join(str(member) for member in job.members),
                    "estimated_credits": job.estimated_credits,
                    "row_count": 0,
                    "point_rows": 0,
                    "rejected_rows": 0,
                    "request_sha256": job.request_sha256,
                    "raw_path": repo_rel(job.raw_path),
                    "response_sha256": "",
                    "completed_at_utc": utc_now_iso(),
                    "error_class": type(exc).__name__,
                    "error_message": sanitize_text(str(exc), token),
                }
                append_csv_row(MANIFEST_PATH, row, manifest_columns)
                append_jsonl(LEDGER_PATH, {"event": "chunk_failed", **row})
                failed += 1
                if args.stop_on_local_error:
                    raise
    scan = secret_scan(token)
    write_json(
        EXPERIMENT_ROOT / "run_summary.json",
        {
            "mode": args.mode,
            "git_commit": git_output("rev-parse", "HEAD"),
            "planned_chunks": len(jobs),
            "completed_this_run": completed,
            "failed_this_run": failed,
            "skipped_budget_this_run": skipped_budget,
            "spent_estimated_credits": spent,
            "blockers": len(blockers),
            "secret_scan": scan,
            "updated_at_utc": utc_now_iso(),
        },
    )
    write_status(
        "complete",
        "partial" if failed or skipped_budget or blockers else "passed",
        completed_this_run=completed,
        failed_this_run=failed,
        skipped_budget_this_run=skipped_budget,
        spent_estimated_credits=spent,
        blockers=len(blockers),
        secret_scan=scan["status"],
    )
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T07-T13 one-thread GribStream backfill supervisor.")
    parser.add_argument("--mode", choices=["kickoff", "full"], default="kickoff")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--credit-budget", type=int, default=85000)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--member-chunk-size", type=int, default=5)
    parser.add_argument("--api-min-interval-seconds", type=float, default=12.0)
    parser.add_argument("--api-max-attempts", type=int, default=3)
    parser.add_argument("--pause-on-429-seconds", type=float, default=300.0)
    parser.add_argument("--max-retry-after-seconds", type=float, default=1800.0)
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset codes to plan, e.g. gfs")
    parser.add_argument("--stop-on-local-error", action="store_true")
    parser.add_argument(
        "--allow-legacy-broad-runner",
        action="store_true",
        help="Explicit escape hatch for the retired broad 0-84h legacy runner.",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
