from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from . import db
from .config import (
    COORD_TOLERANCE_DEGREES,
    DEFAULT_FETCH_THREADS,
    GRIBSTREAM_RESPONSE_COMPRESSED,
    GRIBSTREAM_RESPONSE_FORMAT,
    STATION,
    TEMPERATURE_NATIVE_UNIT,
    isoformat_utc,
    kelvin_to_f,
    local_day_window_utc,
    localize_forecast_time,
    parse_utc,
    settlement_asof_utc,
    utc_now,
)
from .gribstream_client import FetchTimeseriesResult, GribstreamClient, GribstreamRequestError
from .model_catalog import ModelSpec, VariableSpec, eligible_specs_for_date, historical_model_specs

LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class FetchJob:
    spec: ModelSpec
    station_id: str
    settlement_date_local: date
    as_of_utc: str
    from_time_utc: str
    until_time_utc: str
    payload: dict[str, object]
    request_id: str


@dataclass(frozen=True)
class FetchJobResult:
    request_row: dict[str, object]
    raw_rows: list[dict[str, object]]


@dataclass
class ModelProgress:
    total: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    raw_rows: int = 0


def _thread_client() -> GribstreamClient:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = GribstreamClient()
        _THREAD_LOCAL.client = client
    return client


def _stable_payload_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _request_id(job_key: str) -> str:
    return hashlib.sha256(job_key.encode("utf-8")).hexdigest()


def _coordinate_matches(lat: float, lon: float, station_lat: float, station_lon: float) -> bool:
    return abs(lat - station_lat) <= COORD_TOLERANCE_DEGREES and abs(lon - station_lon) <= COORD_TOLERANCE_DEGREES


def _requested_variables(spec: ModelSpec, settlement_date_local: date) -> list[VariableSpec]:
    requested = [spec.snapshot_var]
    native_tmax = spec.native_tmax_for_date(settlement_date_local)
    if native_tmax is not None and native_tmax.header not in {var.header for var in requested}:
        requested.append(native_tmax)
    return requested


def _variable_headers(spec: ModelSpec, settlement_date_local: date) -> list[str]:
    return [variable.header for variable in _requested_variables(spec, settlement_date_local)]


def build_timeseries_payload(
    spec: ModelSpec,
    settlement_date_local: date,
    *,
    from_time_utc: str,
    until_time_utc: str,
    as_of_utc: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "fromTime": from_time_utc,
        "untilTime": until_time_utc,
        "asOf": as_of_utc,
        "coordinates": [
            {
                "lat": STATION.latitude,
                "lon": STATION.longitude,
                "name": STATION.station_id,
            }
        ],
        "variables": [
            {
                "name": variable.name,
                "level": variable.level,
                "info": variable.info,
            }
            for variable in _requested_variables(spec, settlement_date_local)
        ],
    }
    if spec.ensemble_members is not None:
        payload["members"] = list(spec.ensemble_members)
    return payload


def _build_job(
    spec: ModelSpec,
    settlement_date_local: date,
    from_time_utc: str,
    until_time_utc: str,
) -> FetchJob:
    as_of_utc = isoformat_utc(settlement_asof_utc(settlement_date_local))
    payload = build_timeseries_payload(
        spec,
        settlement_date_local,
        from_time_utc=from_time_utc,
        until_time_utc=until_time_utc,
        as_of_utc=as_of_utc,
    )
    payload_json = _stable_payload_json(payload)
    request_id = _request_id(
        f"{spec.model_code}|{STATION.station_id}|{settlement_date_local.isoformat()}|{payload_json}"
    )
    return FetchJob(
        spec=spec,
        station_id=STATION.station_id,
        settlement_date_local=settlement_date_local,
        as_of_utc=as_of_utc,
        from_time_utc=from_time_utc,
        until_time_utc=until_time_utc,
        payload=payload,
        request_id=request_id,
    )


def _result_hour_summary(raw_rows: list[dict[str, object]]) -> str:
    if not raw_rows:
        return "valid_hours=0 native_rows=0 snapshot_rows=0 range=none"
    times = sorted({str(row["forecasted_time_utc"]) for row in raw_rows})
    variable_counts = Counter(str(row["variable_name"]) for row in raw_rows)
    native_rows = sum(count for name, count in variable_counts.items() if name in {"TMAX", "mx2t3"})
    snapshot_rows = sum(count for name, count in variable_counts.items() if name in {"TMP", "2t"})
    return (
        f"valid_hours={len(times)} "
        f"snapshot_rows={snapshot_rows} "
        f"native_rows={native_rows} "
        f"range={times[0]}..{times[-1]}"
    )


def _log_job_plan(jobs: list[FetchJob], *, max_workers: int) -> None:
    if not jobs:
        LOGGER.info("No pending fetch jobs. workers=%d", max_workers)
        return
    by_model: dict[str, list[FetchJob]] = defaultdict(list)
    for job in jobs:
        by_model[job.spec.model_code].append(job)
    LOGGER.info(
        "Fetch plan pending_jobs=%d workers=%d models=%d",
        len(jobs),
        max_workers,
        len(by_model),
    )
    for model_code in sorted(by_model):
        model_jobs = sorted(by_model[model_code], key=lambda job: job.settlement_date_local)
        sample = model_jobs[0]
        LOGGER.info(
            "Fetch plan model=%s pending=%d date_range=%s..%s asof=13:00Z vars=%s members=%s window_rule=local_midnight_to_next_midnight_utc saved_rows=all_valid_times_within_local_day",
            model_code,
            len(model_jobs),
            model_jobs[0].settlement_date_local,
            model_jobs[-1].settlement_date_local,
            _variable_headers(sample.spec, sample.settlement_date_local),
            list(sample.spec.ensemble_members) if sample.spec.ensemble_members is not None else None,
        )


def _format_progress_snapshot(
    total_jobs: int,
    completed: int,
    succeeded: int,
    failed: int,
    inserted_rows: int,
    progress_by_model: dict[str, ModelProgress],
) -> str:
    overall_pct = (completed / total_jobs * 100.0) if total_jobs else 100.0
    model_bits: list[str] = []
    for model_code in sorted(progress_by_model):
        progress = progress_by_model[model_code]
        pct = (progress.completed / progress.total * 100.0) if progress.total else 100.0
        remaining = progress.total - progress.completed
        model_bits.append(
            f"{model_code} {progress.completed}/{progress.total} {pct:.1f}% left={remaining} ok={progress.succeeded} fail={progress.failed} rows={progress.raw_rows}"
        )
    return (
        f"overall {completed}/{total_jobs} {overall_pct:.1f}% left={total_jobs - completed} "
        f"ok={succeeded} fail={failed} raw_rows={inserted_rows} | " + " | ".join(model_bits)
    )


def _raw_rows_from_result(job: FetchJob, result: FetchTimeseriesResult) -> list[dict[str, object]]:
    inserted_at = isoformat_utc(utc_now())
    raw_rows: list[dict[str, object]] = []
    for parsed_row in result.rows:
        if not _coordinate_matches(parsed_row.lat, parsed_row.lon, STATION.latitude, STATION.longitude):
            continue
        forecasted_local = localize_forecast_time(parsed_row.forecasted_time_utc, STATION.timezone_name)
        if forecasted_local.date() != job.settlement_date_local:
            continue
        lead_minutes = int(
            (parsed_row.forecasted_time_utc - parsed_row.forecasted_at_utc).total_seconds() // 60
        )
        raw_rows.append(
            {
                "request_id": job.request_id,
                "model_code": job.spec.model_code,
                "station_id": job.station_id,
                "settlement_date_local": job.settlement_date_local.isoformat(),
                "as_of_utc": job.as_of_utc,
                "forecasted_at_utc": isoformat_utc(parsed_row.forecasted_at_utc),
                "forecasted_time_utc": isoformat_utc(parsed_row.forecasted_time_utc),
                "forecasted_time_local": forecasted_local.isoformat(),
                "forecasted_date_local": forecasted_local.date().isoformat(),
                "lat": parsed_row.lat,
                "lon": parsed_row.lon,
                "coord_name": parsed_row.coord_name,
                "variable_name": parsed_row.variable.name,
                "variable_level": parsed_row.variable.level,
                "variable_info": parsed_row.variable.info,
                "member": parsed_row.member,
                "value_native": parsed_row.value_native,
                "unit_native": TEMPERATURE_NATIVE_UNIT,
                "value_f": kelvin_to_f(parsed_row.value_native),
                "lead_minutes": lead_minutes,
                "inserted_at_utc": inserted_at,
            }
        )
    return raw_rows


def _execute_fetch_job(job: FetchJob) -> FetchJobResult:
    started_at = isoformat_utc(utc_now())
    LOGGER.info(
        "Fetch start model=%s date=%s asof=%s from=%s until=%s vars=%s members=%s",
        job.spec.model_code,
        job.settlement_date_local,
        job.as_of_utc,
        job.from_time_utc,
        job.until_time_utc,
        _variable_headers(job.spec, job.settlement_date_local),
        list(job.spec.ensemble_members) if job.spec.ensemble_members is not None else None,
    )
    request_row: dict[str, object] = {
        "request_id": job.request_id,
        "model_code": job.spec.model_code,
        "station_id": job.station_id,
        "settlement_date_local": job.settlement_date_local.isoformat(),
        "endpoint": "timeseries",
        "as_of_utc": job.as_of_utc,
        "from_time_utc": job.from_time_utc,
        "until_time_utc": job.until_time_utc,
        "http_status": None,
        "attempts": 0,
        "success": 0,
        "row_count": 0,
        "error_text": None,
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "response_format": GRIBSTREAM_RESPONSE_FORMAT,
        "response_compressed": GRIBSTREAM_RESPONSE_COMPRESSED,
    }
    try:
        result = _thread_client().fetch_timeseries_with_meta(job.spec.model_code, job.payload)
        raw_rows = _raw_rows_from_result(job, result)
        request_row.update(
            {
                "http_status": result.http_status,
                "attempts": result.attempts,
                "success": 1,
                "row_count": len(raw_rows),
                "finished_at_utc": isoformat_utc(utc_now()),
            }
        )
        return FetchJobResult(request_row=request_row, raw_rows=raw_rows)
    except GribstreamRequestError as exc:
        request_row.update(
            {
                "http_status": getattr(exc, "status_code", None),
                "attempts": getattr(exc, "attempts", REQUEST_ATTEMPTS_FALLBACK) or REQUEST_ATTEMPTS_FALLBACK,
                "success": 0,
                "error_text": str(exc)[:1000],
                "finished_at_utc": isoformat_utc(utc_now()),
            }
        )
        return FetchJobResult(request_row=request_row, raw_rows=[])
    except Exception as exc:  # pragma: no cover - defensive failure path
        request_row.update(
            {
                "attempts": REQUEST_ATTEMPTS_FALLBACK,
                "success": 0,
                "error_text": f"{exc.__class__.__name__}: {exc}"[:1000],
                "finished_at_utc": isoformat_utc(utc_now()),
            }
        )
        return FetchJobResult(request_row=request_row, raw_rows=[])


REQUEST_ATTEMPTS_FALLBACK = 1


def _truth_windows(
    connection,
    start_date: date,
    end_date: date,
) -> list[tuple[date, str, str]]:
    truth_rows = db.load_truth_rows(
        connection,
        STATION.station_id,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    if not truth_rows:
        raise ValueError(
            "No truth rows found. Run fetch-truth before fetch-forecasts."
        )
    return [
        (
            date.fromisoformat(str(row["settlement_date_local"])),
            str(row["local_day_start_utc"]),
            str(row["local_day_end_utc"]),
        )
        for row in truth_rows
    ]


def _jobs_for_truth_windows(
    connection,
    start_date: date,
    end_date: date,
    *,
    include_live_only: bool,
) -> list[FetchJob]:
    specs = historical_model_specs() if not include_live_only else eligible_specs_for_date(end_date, True)
    jobs: list[FetchJob] = []
    for settlement_date_local, from_time_utc, until_time_utc in _truth_windows(connection, start_date, end_date):
        candidate_specs: Iterable[ModelSpec]
        if include_live_only:
            candidate_specs = eligible_specs_for_date(settlement_date_local, include_live_only=True)
        else:
            candidate_specs = specs
        for spec in candidate_specs:
            if spec.archive_start > settlement_date_local:
                continue
            jobs.append(_build_job(spec, settlement_date_local, from_time_utc, until_time_utc))
    return jobs


def fetch_historical_forecasts(
    connection,
    *,
    start_date: date,
    end_date: date,
    max_workers: int = DEFAULT_FETCH_THREADS,
) -> dict[str, int]:
    jobs = _jobs_for_truth_windows(connection, start_date, end_date, include_live_only=False)
    existing_request_ids = db.successful_request_ids(connection)
    pending_jobs = [job for job in jobs if job.request_id not in existing_request_ids]
    LOGGER.info(
        "Historical fetch start jobs_total=%d pending=%d workers=%d",
        len(jobs),
        len(pending_jobs),
        max_workers,
    )
    _log_job_plan(pending_jobs, max_workers=max_workers)
    return _run_jobs(connection, pending_jobs, max_workers=max_workers)


def fetch_prediction_date_forecasts(
    connection,
    target_date_local: date,
    *,
    include_live_only: bool = True,
    max_workers: int = DEFAULT_FETCH_THREADS,
) -> dict[str, int]:
    truth_rows = db.load_truth_rows(
        connection,
        STATION.station_id,
        target_date_local.isoformat(),
        target_date_local.isoformat(),
    )
    if truth_rows:
        from_time_utc = str(truth_rows[0]["local_day_start_utc"])
        until_time_utc = str(truth_rows[0]["local_day_end_utc"])
    else:
        start_utc, end_utc = local_day_window_utc(target_date_local, STATION.timezone_name)
        from_time_utc = isoformat_utc(start_utc)
        until_time_utc = isoformat_utc(end_utc)
    jobs = [
        _build_job(spec, target_date_local, from_time_utc, until_time_utc)
        for spec in eligible_specs_for_date(target_date_local, include_live_only=include_live_only)
    ]
    existing_request_ids = db.successful_request_ids(connection)
    pending_jobs = [job for job in jobs if job.request_id not in existing_request_ids]
    LOGGER.info(
        "Prediction-date fetch start date=%s jobs_total=%d pending=%d workers=%d",
        target_date_local,
        len(jobs),
        len(pending_jobs),
        max_workers,
    )
    _log_job_plan(pending_jobs, max_workers=max_workers)
    return _run_jobs(connection, pending_jobs, max_workers=max_workers)


def _run_jobs(connection, jobs: list[FetchJob], *, max_workers: int) -> dict[str, int]:
    if not jobs:
        return {"requested": 0, "succeeded": 0, "failed": 0, "rows": 0}
    succeeded = 0
    failed = 0
    inserted_rows = 0
    completed = 0
    progress_by_model: dict[str, ModelProgress] = {}
    for job in jobs:
        progress = progress_by_model.setdefault(job.spec.model_code, ModelProgress())
        progress.total += 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_execute_fetch_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            result = future.result()
            db.upsert_gribstream_request(connection, result.request_row)
            db.insert_gribstream_raw_forecasts(connection, result.raw_rows)
            completed += 1
            inserted_rows += len(result.raw_rows)
            model_progress = progress_by_model[job.spec.model_code]
            model_progress.completed += 1
            model_progress.raw_rows += len(result.raw_rows)
            if int(result.request_row["success"]) == 1:
                succeeded += 1
                model_progress.succeeded += 1
                LOGGER.info(
                    "Fetch success model=%s date=%s rows=%d attempts=%s %s",
                    job.spec.model_code,
                    job.settlement_date_local,
                    len(result.raw_rows),
                    result.request_row["attempts"],
                    _result_hour_summary(result.raw_rows),
                )
            else:
                failed += 1
                model_progress.failed += 1
                LOGGER.error(
                    "Fetch failed model=%s date=%s attempts=%s status=%s error=%s",
                    job.spec.model_code,
                    job.settlement_date_local,
                    result.request_row["attempts"],
                    result.request_row["http_status"],
                    result.request_row["error_text"],
                )
            db.commit(connection)
            LOGGER.info(
                "Fetch progress %s",
                _format_progress_snapshot(
                    len(jobs),
                    completed,
                    succeeded,
                    failed,
                    inserted_rows,
                    progress_by_model,
                ),
            )
    db.commit(connection)
    return {
        "requested": len(jobs),
        "succeeded": succeeded,
        "failed": failed,
        "rows": inserted_rows,
    }
