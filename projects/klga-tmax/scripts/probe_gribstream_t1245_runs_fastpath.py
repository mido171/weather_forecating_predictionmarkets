from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy import create_engine, text

from klga_tmax.providers.gribstream.catalog import resolve_all_selectors
from klga_tmax.providers.gribstream.config import load_gribstream_settings
from klga_tmax.providers.gribstream.plan import (
    MODEL_SPECS,
    T1245_CUTOFF_ID,
    as_of_utc,
    build_chunk,
    default_members,
    effective_target_start,
    model_spec_by_id,
    valid_times_for_target,
)
from klga_tmax.registry.materialize_targets import iter_dates


DEFAULT_JOB_ID = "klga_t1245utc_full_backfill_v1"
UTC = timezone.utc


@dataclass(frozen=True)
class RunsProbe:
    model_id: str
    endpoint: str
    target_start_date: date
    target_end_date: date
    request_payload: dict[str, Any]
    target_days: int
    intended_row_keys: set[tuple[str, str]]
    estimated_current_shape_credits: int


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def floor_to_hour(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def floor_to_cycle(value: datetime, cycle_hours: tuple[int, ...]) -> datetime:
    value = value.astimezone(UTC)
    for hour in sorted(cycle_hours, reverse=True):
        candidate = value.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate <= value:
            return candidate
    previous = value - timedelta(days=1)
    return previous.replace(hour=max(cycle_hours), minute=0, second=0, microsecond=0)


def run_time_for_target(model_id: str, target_date: date) -> datetime:
    spec = model_spec_by_id(model_id)
    if spec.fetch_shape == "urma_peak_temp":
        raise ValueError("URMA uses lead-zero valid-time run list, not one cycle per target date")
    cutoff_as_of = as_of_utc(target_date, spec, cutoff_id=T1245_CUTOFF_ID)
    if cutoff_as_of is None:
        raise ValueError(f"{model_id} has no asOf-backed run cycle")
    if model_id in {"gfs", "gefsatmos", "gefsatmosmean", "aigefssfc", "aigfssfc"}:
        return floor_to_cycle(cutoff_as_of, (0, 6, 12, 18))
    if model_id in {"ifsoper", "ifsenfo", "aifsoper", "aifsenfo"}:
        return floor_to_cycle(cutoff_as_of, (0, 12))
    if model_id == "rap":
        # At T_1245UTC the nominal asOf is 11Z, but live /timeseries checks
        # select earlier long-enough RAP cycles for the full NY peak window.
        return floor_to_hour(cutoff_as_of - timedelta(hours=3))
    if model_id == "nbmqmd":
        # max-18h valid at T+1 06Z resolves to the 06Z NBM QMD package in
        # live /timeseries checks, not the 11Z hourly cutoff floor.
        return floor_to_hour(cutoff_as_of - timedelta(hours=5))
    return floor_to_hour(cutoff_as_of)


def desired_pairs_for_targets(model_id: str, start_date: date, end_date: date) -> tuple[list[str], set[tuple[str, str]], list[int]]:
    spec = model_spec_by_id(model_id)
    run_times: list[str] = []
    desired_pairs: set[tuple[str, str]] = set()
    lead_hours: list[int] = []
    if spec.fetch_shape == "urma_peak_temp":
        for target_date in iter_dates(start_date, end_date):
            for valid_time in valid_times_for_target(spec, target_date, cutoff_id=T1245_CUTOFF_ID):
                run_iso = iso_z(valid_time)
                run_times.append(run_iso)
                desired_pairs.add((run_iso, run_iso))
                lead_hours.append(0)
        return run_times, desired_pairs, lead_hours

    for target_date in iter_dates(start_date, end_date):
        run_time = run_time_for_target(model_id, target_date)
        run_iso = iso_z(run_time)
        run_times.append(run_iso)
        for valid_time in valid_times_for_target(spec, target_date, cutoff_id=T1245_CUTOFF_ID):
            valid_iso = iso_z(valid_time)
            desired_pairs.add((run_iso, valid_iso))
            lead = int(round((valid_time - run_time).total_seconds() / 3600.0))
            lead_hours.append(lead)
    return run_times, desired_pairs, lead_hours


def completed_days_by_model(database_url: str | None, job_id: str) -> dict[str, int]:
    if not database_url:
        return {}
    engine = create_engine(database_url)
    rows: dict[str, int] = {}
    with engine.begin() as conn:
        for row in conn.execute(
            text(
                """
                SELECT model_id,
                       COALESCE(SUM(target_end_date - target_start_date + 1), 0)::int AS days
                FROM audit.gribstream_backfill_chunks
                WHERE job_id = :job_id
                  AND status IN ('completed','completed_empty','skipped')
                GROUP BY model_id
                """
            ),
            {"job_id": job_id},
        ).mappings():
            rows[str(row["model_id"])] = int(row["days"])
    return rows


def base_request_parts(model_id: str, target_date: date, selectors_by_model: dict[str, tuple[Any, ...]]) -> dict[str, Any]:
    spec = model_spec_by_id(model_id)
    chunk = build_chunk(
        spec=spec,
        target_start_date=target_date,
        target_end_date=target_date,
        coordinate_tier_name="B",
        selectors=selectors_by_model[model_id],
        members=default_members(spec),
        cutoff_id=T1245_CUTOFF_ID,
    )
    payload = dict(chunk.request_payload)
    payload.pop("asOf", None)
    payload.pop("timesList", None)
    return payload


def build_runs_probe(
    *,
    model_id: str,
    target_start_date: date,
    target_end_date: date,
    selectors_by_model: dict[str, tuple[Any, ...]],
) -> RunsProbe:
    spec = model_spec_by_id(model_id)
    base = base_request_parts(model_id, target_start_date, selectors_by_model)
    run_times, desired_pairs, lead_hours = desired_pairs_for_targets(model_id, target_start_date, target_end_date)
    if not run_times:
        raise ValueError(f"no run times built for {model_id}")
    payload = dict(base)
    payload["timesList"] = run_times
    payload["minLeadTime"] = f"{min(lead_hours)}h"
    payload["maxLeadTime"] = f"{max(lead_hours)}h"
    payload["includeMetadata"] = ["index_updated_at"]
    target_days = (target_end_date - target_start_date).days + 1
    return RunsProbe(
        model_id=model_id,
        endpoint="runs",
        target_start_date=target_start_date,
        target_end_date=target_end_date,
        request_payload=payload,
        target_days=target_days,
        intended_row_keys=desired_pairs,
        estimated_current_shape_credits=target_days * spec.expected_credits_per_day,
    )


def decode_body(raw_body: bytes, headers: Any) -> bytes:
    encoding = str(headers.get("Content-Encoding") or "").lower()
    if "gzip" in encoding:
        return gzip.decompress(raw_body)
    return raw_body


def response_rows(decoded_body: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in decoded_body.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str] | None:
    run_time = row.get("forecasted_at")
    valid_time = row.get("forecasted_time")
    if not run_time or not valid_time:
        return None
    try:
        return (iso_z(parse_utc(str(run_time))), iso_z(parse_utc(str(valid_time))))
    except ValueError:
        return (str(run_time), str(valid_time))


def post_probe(token: str, base_url: str, probe: RunsProbe, *, timeout_seconds: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{probe.model_id}/{probe.endpoint}"
    body = json.dumps(probe.request_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/ndjson",
            "Accept-Encoding": "gzip",
            "User-Agent": "klga-tmax-t1245-runs-fastpath-smoke/1.0",
        },
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read()
            elapsed = time.perf_counter() - started
            decoded_body = decode_body(raw_body, response.headers)
            rows = response_rows(decoded_body)
            returned_keys = Counter(key for item in rows if (key := row_key(item)) is not None)
            exact_rows = sum(count for key, count in returned_keys.items() if key in probe.intended_row_keys)
            extra_keys = sum(1 for key in returned_keys if key not in probe.intended_row_keys)
            return {
                "model_id": probe.model_id,
                "endpoint": probe.endpoint,
                "status_code": int(response.status),
                "target_start_date": probe.target_start_date.isoformat(),
                "target_end_date": probe.target_end_date.isoformat(),
                "target_days": probe.target_days,
                "request_run_times": len(probe.request_payload["timesList"]),
                "min_lead_time": probe.request_payload["minLeadTime"],
                "max_lead_time": probe.request_payload["maxLeadTime"],
                "elapsed_seconds": round(elapsed, 3),
                "request_bytes": len(body),
                "wire_bytes": len(raw_body),
                "decoded_bytes": len(decoded_body),
                "row_count": len(rows),
                "exact_needed_rows": exact_rows,
                "extra_forecast_time_keys": extra_keys,
                "distinct_forecasted_at": len({key[0] for key in returned_keys}),
                "distinct_forecasted_time": len({key[1] for key in returned_keys}),
                "estimated_current_shape_credits": probe.estimated_current_shape_credits,
                "content_type": response.headers.get("Content-Type"),
                "content_encoding": response.headers.get("Content-Encoding"),
                "retry_after": response.headers.get("Retry-After"),
                "error_preview": "",
            }
    except HTTPError as exc:
        raw_body = exc.read()
        elapsed = time.perf_counter() - started
        decoded_body = decode_body(raw_body, exc.headers)
        return {
            "model_id": probe.model_id,
            "endpoint": probe.endpoint,
            "status_code": int(exc.code),
            "target_start_date": probe.target_start_date.isoformat(),
            "target_end_date": probe.target_end_date.isoformat(),
            "target_days": probe.target_days,
            "request_run_times": len(probe.request_payload["timesList"]),
            "min_lead_time": probe.request_payload["minLeadTime"],
            "max_lead_time": probe.request_payload["maxLeadTime"],
            "elapsed_seconds": round(elapsed, 3),
            "request_bytes": len(body),
            "wire_bytes": len(raw_body),
            "decoded_bytes": len(decoded_body),
            "row_count": 0,
            "exact_needed_rows": 0,
            "extra_forecast_time_keys": 0,
            "distinct_forecasted_at": 0,
            "distinct_forecasted_time": 0,
            "estimated_current_shape_credits": probe.estimated_current_shape_credits,
            "content_type": exc.headers.get("Content-Type"),
            "content_encoding": exc.headers.get("Content-Encoding"),
            "retry_after": exc.headers.get("Retry-After"),
            "error_preview": decoded_body[:500].decode("utf-8", errors="replace"),
        }
    except (TimeoutError, URLError, OSError) as exc:
        elapsed = time.perf_counter() - started
        return {
            "model_id": probe.model_id,
            "endpoint": probe.endpoint,
            "status_code": None,
            "target_start_date": probe.target_start_date.isoformat(),
            "target_end_date": probe.target_end_date.isoformat(),
            "target_days": probe.target_days,
            "request_run_times": len(probe.request_payload["timesList"]),
            "min_lead_time": probe.request_payload["minLeadTime"],
            "max_lead_time": probe.request_payload["maxLeadTime"],
            "elapsed_seconds": round(elapsed, 3),
            "request_bytes": len(body),
            "wire_bytes": 0,
            "decoded_bytes": 0,
            "row_count": 0,
            "exact_needed_rows": 0,
            "extra_forecast_time_keys": 0,
            "distinct_forecasted_at": 0,
            "distinct_forecasted_time": 0,
            "estimated_current_shape_credits": probe.estimated_current_shape_credits,
            "content_type": "",
            "content_encoding": "",
            "retry_after": "",
            "error_preview": str(exc)[:500],
        }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def projection_rows(
    *,
    results: list[dict[str, Any]],
    end_date: date,
    completed_days: dict[str, int],
) -> list[dict[str, Any]]:
    measured_by_model = {row["model_id"]: row for row in results if row.get("status_code") == 200 and row.get("target_days")}
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        target_start = effective_target_start(spec, cutoff_id=T1245_CUTOFF_ID)
        total_days = max(0, (end_date - target_start).days + 1)
        already_done = min(completed_days.get(spec.model_id, 0), total_days)
        remaining_days = max(0, total_days - already_done)
        batch_days = spec.default_chunk_days
        full_chunks = math.ceil(total_days / batch_days) if total_days else 0
        remaining_chunks = math.ceil(remaining_days / batch_days) if remaining_days else 0
        measured = measured_by_model.get(spec.model_id)
        measured_seconds = float(measured["elapsed_seconds"]) if measured else None
        measured_days = int(measured["target_days"]) if measured else None
        scaled_seconds_per_chunk = None
        projected_remaining_seconds = None
        projected_full_seconds = None
        if measured_seconds is not None and measured_days:
            scaled_seconds_per_chunk = measured_seconds * (batch_days / measured_days)
            projected_remaining_seconds = scaled_seconds_per_chunk * remaining_chunks
            projected_full_seconds = scaled_seconds_per_chunk * full_chunks
        rows.append(
            {
                "model_id": spec.model_id,
                "target_from": target_start.isoformat(),
                "target_through": end_date.isoformat(),
                "total_days": total_days,
                "completed_days_in_current_job": already_done,
                "remaining_days": remaining_days,
                "fast_batch_days": batch_days,
                "projected_full_chunks": full_chunks,
                "projected_remaining_chunks": remaining_chunks,
                "measured_smoke_days": measured_days,
                "measured_elapsed_seconds": measured_seconds,
                "scaled_seconds_per_fast_chunk": round(scaled_seconds_per_chunk, 3) if scaled_seconds_per_chunk else None,
                "projected_remaining_seconds_api_only": round(projected_remaining_seconds, 1) if projected_remaining_seconds else None,
                "projected_full_seconds_api_only": round(projected_full_seconds, 1) if projected_full_seconds else None,
                "expected_credits_remaining_current_shape": remaining_days * spec.expected_credits_per_day,
                "expected_credits_full_current_shape": total_days * spec.expected_credits_per_day,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a KLGA T_1245UTC /runs fast-path after the HKG batching pattern.")
    parser.add_argument("--models", required=True)
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--database-url")
    parser.add_argument("--max-models", type=int, default=1)
    parser.add_argument("--max-probe-days", type=int, default=1)
    parser.add_argument("--spacing-seconds", type=float, default=float(os.environ.get("GRIBSTREAM_SPACING_SECONDS", "12")))
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Acknowledge bounded provider requests and optional read-only DB state lookup.",
    )
    args = parser.parse_args()

    if not args.execute:
        parser.error("live probe is disabled; re-run with --execute")
    if args.models.strip().lower() in {"all", "*"}:
        parser.error("--models must list explicit model IDs; 'all' is not allowed")
    model_ids = tuple(item.strip() for item in args.models.split(",") if item.strip())
    if args.max_models < 1 or len(model_ids) > args.max_models:
        parser.error(
            f"model scope contains {len(model_ids)} models but --max-models is {args.max_models}"
        )
    if args.end_date < args.start_date:
        parser.error("--end-date must be on or after --start-date")
    if args.max_probe_days < 1:
        parser.error("--max-probe-days must be >= 1")
    if args.spacing_seconds < 12:
        parser.error("--spacing-seconds must be >= 12")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be > 0")
    unknown = sorted(set(model_ids) - {spec.model_id for spec in MODEL_SPECS})
    if unknown:
        parser.error(f"unknown model ids: {', '.join(unknown)}")

    settings = load_gribstream_settings(require_api_token=True)
    token = settings.api_token
    if token is None:  # Defensive; require_api_token=True already fails closed.
        raise RuntimeError("GribStream token validation unexpectedly returned no token")
    output_root = args.output_root or (
        settings.artifact_root / "runs_fastpath_smoke"
    )

    selectors, selector_gaps, snapshots = resolve_all_selectors(settings, model_ids=model_ids)
    if selector_gaps:
        write_json(args.output_root / "selector_gaps.json", selector_gaps)
    missing = [model_id for model_id in model_ids if model_id not in selectors]
    if missing:
        raise RuntimeError(f"selector resolution failed for: {', '.join(missing)}")

    completed_days = completed_days_by_model(args.database_url, args.job_id)
    probes: list[RunsProbe] = []
    for model_id in model_ids:
        spec = model_spec_by_id(model_id)
        smoke_start = args.start_date
        smoke_start = max(smoke_start, effective_target_start(spec, cutoff_id=T1245_CUTOFF_ID))
        smoke_end = min(
            args.end_date,
            smoke_start + timedelta(days=args.max_probe_days - 1),
        )
        if smoke_start > args.end_date:
            continue
        probes.append(
            build_runs_probe(
                model_id=model_id,
                target_start_date=smoke_start,
                target_end_date=smoke_end,
                selectors_by_model=selectors,
            )
        )

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, probe in enumerate(probes, start=1):
        if index > 1 and args.spacing_seconds > 0:
            time.sleep(args.spacing_seconds)
        result = post_probe(token, settings.base_url, probe, timeout_seconds=args.timeout_seconds)
        result["probe_index"] = index
        result["fast_batch_days"] = model_spec_by_id(probe.model_id).default_chunk_days
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status_code"] in {401, 403, 429}:
            break

    total_elapsed = time.perf_counter() - started
    projections = projection_rows(results=results, end_date=args.end_date, completed_days=completed_days)
    total_remaining_api_seconds = sum(
        float(row["projected_remaining_seconds_api_only"] or 0.0) for row in projections
    )
    total_full_api_seconds = sum(float(row["projected_full_seconds_api_only"] or 0.0) for row in projections)
    projected_remaining_chunks = sum(int(row["projected_remaining_chunks"]) for row in projections)
    projected_full_chunks = sum(int(row["projected_full_chunks"]) for row in projections)
    spacing_remaining_seconds = max(0, projected_remaining_chunks - 1) * max(0.0, args.spacing_seconds)
    spacing_full_seconds = max(0, projected_full_chunks - 1) * max(0.0, args.spacing_seconds)
    summary = {
        "ok": all(row.get("status_code") == 200 for row in results),
        "generated_at_utc": iso_z(datetime.now(UTC)),
        "cutoff_id": T1245_CUTOFF_ID,
        "endpoint_strategy": "POST /api/v2/{model}/runs with exact model-run timesList",
        "models_requested": list(model_ids),
        "models_probed": [row["model_id"] for row in results],
        "selector_gap_count": len(selector_gaps),
        "catalog_snapshots": len(snapshots),
        "spacing_seconds": args.spacing_seconds,
        "smoke_calls": len(results),
        "smoke_wall_seconds": round(total_elapsed, 3),
        "smoke_status_counts": dict(Counter(str(row.get("status_code")) for row in results)),
        "projected_full_chunks": projected_full_chunks,
        "projected_remaining_chunks_after_current_job": projected_remaining_chunks,
        "projected_full_api_seconds_from_measured_calls": round(total_full_api_seconds, 1),
        "projected_remaining_api_seconds_from_measured_calls": round(total_remaining_api_seconds, 1),
        "projected_full_wall_seconds_with_spacing": round(total_full_api_seconds + spacing_full_seconds, 1),
        "projected_remaining_wall_seconds_with_spacing": round(total_remaining_api_seconds + spacing_remaining_seconds, 1),
        "projected_full_hours_with_spacing": round((total_full_api_seconds + spacing_full_seconds) / 3600.0, 2),
        "projected_remaining_hours_with_spacing": round(
            (total_remaining_api_seconds + spacing_remaining_seconds) / 3600.0, 2
        ),
        "expected_full_credits_current_shape": sum(
            int(row["expected_credits_full_current_shape"]) for row in projections
        ),
        "expected_remaining_credits_current_shape": sum(
            int(row["expected_credits_remaining_current_shape"]) for row in projections
        ),
        "notes": [
            "Projection uses measured API response time scaled by each model's configured fast batch size.",
            "This smoke does not persist rows to bronze/silver and does not resume the paused /timeseries job.",
            "Rows from /runs may include extra native forecast horizons inside minLeadTime/maxLeadTime; downstream must filter to intended forecasted_at/forecasted_time pairs.",
        ],
    }

    write_csv(output_root / "smoke_results.csv", results)
    write_csv(output_root / "projection_by_model.csv", projections)
    write_json(output_root / "smoke_summary.json", summary)
    write_json(output_root / "request_payloads_summary.json", [
        {
            "model_id": probe.model_id,
            "endpoint": probe.endpoint,
            "target_start_date": probe.target_start_date.isoformat(),
            "target_end_date": probe.target_end_date.isoformat(),
            "timesList_count": len(probe.request_payload["timesList"]),
            "minLeadTime": probe.request_payload["minLeadTime"],
            "maxLeadTime": probe.request_payload["maxLeadTime"],
            "variable_count": len(probe.request_payload.get("variables", [])),
            "expression_count": len(probe.request_payload.get("expressions", [])),
            "coordinate_count": len(probe.request_payload.get("coordinates", [])),
            "member_count": len(probe.request_payload.get("members", [])) or 1,
        }
        for probe in probes
    ])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
