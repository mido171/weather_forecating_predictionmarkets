from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

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
FIRST_WEEK_SCRIPT = REPO_ROOT / "scripts/run_tactical_gribstream_first_week.py"
TACTICAL_MIGRATION = REPO_ROOT / "db/migrations/postgres/20260625_0007_tactical_gribstream_h24n_schema.sql"
EXPERIMENT_ROOT = REPO_ROOT / "experiments/0214_tactical_h24n_gribstream_backfill"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"
ACQUISITION_VERSION = "tactical_h24n_v1"

DEFAULT_DATASETS = (
    "gfs",
    "gefsatmosmean",
    "gefsatmos",
    "ifsoper",
    "ifsenfo",
    "cwawrf15",
    "aifsoper",
    "aifsenfo",
    "aigfssfc",
    "aigfspres",
    "aigefssfc",
    "graphcast",
    "fourcastnetgfs",
    "nbmoc",
)

BATCH_SIZE_DAYS = {
    "gfs": 14,
    "gefsatmosmean": 31,
    "gefsatmos": 5,
    "ifsoper": 14,
    "ifsenfo": 5,
    "cwawrf15": 3,
    "aifsoper": 10,
    "aifsenfo": 5,
    "aigfssfc": 31,
    "aigfspres": 14,
    "aigefssfc": 5,
    "graphcast": 14,
    "fourcastnetgfs": 14,
    "nbmoc": 7,
}


def load_first_week_module() -> Any:
    spec = importlib.util.spec_from_file_location("tactical_first_week", FIRST_WEEK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared first-week module from {FIRST_WEEK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tactical_first_week"] = module
    spec.loader.exec_module(module)
    return module


FW = load_first_week_module()


def utc_now_iso() -> str:
    return datetime.now(FW.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_directory(path: Path) -> None:
    os.makedirs(fs_path(path), exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with open(fs_path(path), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_directory(path.parent)
    with open(fs_path(path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def chunked(items: list[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def raw_object_path(raw_root: Path, dataset: str, run_times: list[str], request_hash: str) -> Path:
    safe_start = run_times[0].replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    safe_end = run_times[-1].replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    return raw_root / dataset / f"run_window_utc={safe_start}_to_{safe_end}" / f"{request_hash}.ndjson.gz"


def read_ndjson_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(fs_path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def choose_chunk_id(cursor: Any, request_hash: str, dataset: str, prefix: str) -> str:
    cursor.execute("SELECT chunk_id FROM nwp_tactical.acquisition_chunk WHERE request_sha256 = %s", (request_hash,))
    row = cursor.fetchone()
    if row:
        return str(row[0])
    return f"{prefix}_{dataset}_{request_hash[:12]}"


def upsert_chunk_and_raw(
    database_url: str,
    *,
    prefix: str,
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
            chunk_id = choose_chunk_id(cursor, request_hash, spec.dataset, prefix)
            request_with_metrics = dict(payload)
            request_with_metrics["_batch_smoke_metrics"] = {
                "actual_credit_estimate": actual_credit_count,
                "elapsed_seconds": elapsed_seconds,
                "batch_prefix": prefix,
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
                    f"{prefix}_required_v1",
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


def sanity_check_rows(spec: Any, payload: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    requested_runs = set(payload["timesList"])
    returned_runs = {str(row.get("forecasted_at")) for row in rows if row.get("forecasted_at")}
    wrong_runs = sorted(returned_runs - requested_runs)
    if wrong_runs:
        issues.append(f"unexpected_run_times={wrong_runs[:5]}")

    min_lead = int(spec.min_lead.rstrip("h"))
    max_lead = int(spec.max_lead.rstrip("h"))
    for row in rows:
        try:
            run_time = FW.parse_utc(str(row["forecasted_at"]))
            valid_time = FW.parse_utc(str(row["forecasted_time"]))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"bad_time_row={type(exc).__name__}")
            break
        lead_hours = (valid_time - run_time).total_seconds() / 3600.0
        if lead_hours < min_lead - 1e-6 or lead_hours > max_lead + 1e-6:
            issues.append(f"lead_out_of_range={lead_hours}")
            break

    requested_location_names = {str(coord["name"]) for coord in payload["coordinates"]}
    returned_location_names = {str(row.get("name")) for row in rows if row.get("name")}
    unknown_locations = sorted(returned_location_names - requested_location_names)
    if unknown_locations:
        issues.append(f"unexpected_locations={unknown_locations[:5]}")

    requested_members = payload.get("members")
    if requested_members:
        returned_members = {int(row["member"]) for row in rows if row.get("member") is not None}
        missing_members = sorted(set(requested_members) - returned_members)
        if rows and missing_members:
            issues.append(f"missing_members={missing_members[:10]}")
    return issues


def update_progress(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, {**payload, "updated_at_utc": utc_now_iso()})


def summarize_db(database_url: str, chunk_ids: list[str]) -> dict[str, Any]:
    if not chunk_ids:
        return {}
    psycopg = import_psycopg()
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, count(*)
                FROM nwp_tactical.acquisition_chunk
                WHERE chunk_id = ANY(%s)
                GROUP BY status
                ORDER BY status
                """,
                (chunk_ids,),
            )
            chunk_status = {str(status): int(count) for status, count in cursor.fetchall()}
            cursor.execute(
                """
                SELECT count(*)
                FROM nwp_tactical.raw_response_object
                WHERE chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
            raw_objects = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT fw.dataset_code, count(*), min(fw.run_time_utc)::text, max(fw.run_time_utc)::text
                FROM nwp_tactical.forecast_wide fw
                WHERE fw.source_response_object_id IN (
                    SELECT response_object_id
                    FROM nwp_tactical.raw_response_object
                    WHERE chunk_id = ANY(%s)
                )
                GROUP BY fw.dataset_code
                ORDER BY fw.dataset_code
                """,
                (chunk_ids,),
            )
            dataset_rows = [
                {"dataset": row[0], "rows": int(row[1]), "min_run_time": row[2], "max_run_time": row[3]}
                for row in cursor.fetchall()
            ]
    return {
        "chunk_status": chunk_status,
        "raw_response_objects": raw_objects,
        "forecast_wide_rows_by_dataset": dataset_rows,
    }


def run(args: argparse.Namespace) -> int:
    token = FW.load_gribstream_token()
    output_root = EXPERIMENT_ROOT / args.output_name
    request_root = output_root / "req"
    raw_root = PROJECT_PATHS.data_root / "_pipeline_internal" / "raw" / f"gribstream_tactical_{args.output_name}"
    event_log = EXPERIMENT_ROOT / "logs" / f"gribstream_{args.output_name}_api_events.jsonl"
    results_csv = output_root / "batch_results.csv"
    summary_json = output_root / "batch_summary.json"
    progress_json = output_root / "progress.json"
    ensure_directory(output_root)
    ensure_directory(request_root)
    if args.apply_schema:
        apply_migration(args.database_url, TACTICAL_MIGRATION)
    if not token:
        update_progress(progress_json, {"status": "blocked", "reason": "missing_gribstream_api_key"})
        return 2

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(FW.SMOKE.MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")

    planned_chunks: list[tuple[str, Any, list[str]]] = []
    for dataset in datasets:
        run_times = FW.run_times_for_dataset(dataset, args.days)
        spec = FW.effective_spec(dataset)
        size = BATCH_SIZE_DAYS[dataset]
        for run_time_chunk in chunked(run_times, size):
            planned_chunks.append((dataset, spec, run_time_chunk))

    results: list[dict[str, Any]] = []
    completed_chunk_ids: list[str] = []
    started = time.perf_counter()
    retry_config = RetryConfig(
        max_attempts=args.api_max_attempts,
        min_interval_seconds=args.api_min_interval_seconds,
        default_rate_limit_pause_seconds=args.pause_on_429_seconds,
        max_retry_delay_seconds=args.max_retry_after_seconds,
    )
    update_progress(
        progress_json,
        {
            "status": "running",
            "planned_chunks": len(planned_chunks),
            "completed_chunks": 0,
            "datasets": datasets,
            "output_root": output_root.as_posix(),
        },
    )

    with GribStreamClient(token, retry_config=retry_config, event_log_path=event_log) as client:
        for index, (dataset, spec, run_times) in enumerate(planned_chunks, start=1):
            payload = FW.build_payload(spec, run_times)
            request_hash = request_sha256(payload)
            request_path = request_root / f"{index:04d}_{request_hash[:12]}.json"
            write_json(request_path, payload)
            raw_path = raw_object_path(raw_root, dataset, run_times, request_hash)
            expected_credit_count = FW.expected_credits(spec, payload)
            expected_row_count = FW.expected_rows(spec, payload)
            elapsed_seconds = 0.0
            row_count = 0
            inserted_rows = 0
            actual_credit_count = 0
            chunk_id = ""
            status = "failed"
            http_status: int | None = None
            source = "api_fetched"
            error_class = ""
            error_message = ""
            sanity_issues: list[str] = []
            try:
                request_started = time.perf_counter()
                if args.reuse_existing and os.path.exists(fs_path(raw_path)):
                    source = "raw_reused"
                    http_status = 200
                else:
                    client.post_runs_to_gzip(
                        dataset=dataset,
                        payload=payload,
                        output_path=raw_path,
                        request_hash=request_hash,
                    )
                    http_status = 200
                elapsed_seconds = time.perf_counter() - request_started
                rows = read_ndjson_gzip(raw_path)
                row_count = len(rows)
                actual_credit_count = FW.infer_credits_from_rows(rows, payload)
                response_hash = sha256_file(raw_path)
                sanity_issues = sanity_check_rows(spec, payload, rows)
                status = "completed" if rows and not sanity_issues else "completed_empty" if not rows else "failed"
                chunk_id, response_object_id = upsert_chunk_and_raw(
                    args.database_url,
                    prefix=args.output_name,
                    spec=spec,
                    payload=payload,
                    request_hash=request_hash,
                    status=status,
                    expected_row_count=expected_row_count,
                    expected_credit_count=expected_credit_count,
                    actual_credit_count=actual_credit_count,
                    raw_path=raw_path,
                    response_sha256=response_hash,
                    row_count=row_count,
                    http_status=http_status,
                    elapsed_seconds=elapsed_seconds,
                    error_class=";".join(sanity_issues) if sanity_issues else None,
                    error_message=";".join(sanity_issues) if sanity_issues else None,
                )
                if rows:
                    inserted_rows = FW.insert_forecast_wide(args.database_url, dataset, rows, payload, response_object_id)
                completed_chunk_ids.append(chunk_id)
            except GribStreamRequestError as exc:
                elapsed_seconds = time.perf_counter() - request_started
                http_status = exc.status_code
                error_class = exc.error_class
                error_message = sanitize_text(str(exc), token)
                chunk_id, _response_object_id = upsert_chunk_and_raw(
                    args.database_url,
                    prefix=args.output_name,
                    spec=spec,
                    payload=payload,
                    request_hash=request_hash,
                    status="failed",
                    expected_row_count=expected_row_count,
                    expected_credit_count=expected_credit_count,
                    actual_credit_count=0,
                    raw_path=None,
                    response_sha256=None,
                    row_count=0,
                    http_status=http_status,
                    elapsed_seconds=elapsed_seconds,
                    error_class=error_class,
                    error_message=error_message,
                )
                completed_chunk_ids.append(chunk_id)
                status = "failed"
            result = {
                "chunk_index": index,
                "dataset": dataset,
                "status": status,
                "http_status": http_status,
                "run_time_count": len(run_times),
                "first_run_time": run_times[0],
                "last_run_time": run_times[-1],
                "row_count": row_count,
                "forecast_wide_rows_upserted": inserted_rows,
                "estimated_credits_consumed": actual_credit_count,
                "expected_credits": expected_credit_count,
                "expected_rows": expected_row_count,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "request_sha256": request_hash,
                "chunk_id": chunk_id,
                "raw_path": raw_path.as_posix(),
                "source": source,
                "sanity_issue_count": len(sanity_issues),
                "sanity_issues": ";".join(sanity_issues),
                "error_class": error_class,
                "error_message": error_message,
            }
            results.append(result)
            write_csv(
                results_csv,
                results,
                [
                    "chunk_index",
                    "dataset",
                    "status",
                    "http_status",
                    "run_time_count",
                    "first_run_time",
                    "last_run_time",
                    "row_count",
                    "forecast_wide_rows_upserted",
                    "estimated_credits_consumed",
                    "expected_credits",
                    "expected_rows",
                    "elapsed_seconds",
                    "request_sha256",
                    "chunk_id",
                    "raw_path",
                    "source",
                    "sanity_issue_count",
                    "sanity_issues",
                    "error_class",
                    "error_message",
                ],
            )
            status_counts = Counter(row["status"] for row in results)
            update_progress(
                progress_json,
                {
                    "status": "running",
                    "planned_chunks": len(planned_chunks),
                    "completed_chunks": len(results),
                    "last_dataset": dataset,
                    "last_chunk_status": status,
                    "status_counts": dict(status_counts),
                    "estimated_credits_consumed": sum(int(row["estimated_credits_consumed"]) for row in results),
                    "rows_returned": sum(int(row["row_count"]) for row in results),
                    "results_csv": results_csv.as_posix(),
                },
            )
            print(json.dumps(result, sort_keys=True), flush=True)
            if http_status in {401, 403, 429}:
                break

    total_seconds = time.perf_counter() - started
    status_counts = Counter(row["status"] for row in results)
    db_summary = summarize_db(args.database_url, completed_chunk_ids)
    summary = {
        "status": "passed" if results and all(row["status"] in {"completed", "completed_empty"} for row in results) else "failed",
        "updated_at_utc": utc_now_iso(),
        "database": redact_database_url(args.database_url),
        "output_name": args.output_name,
        "days_requested": args.days,
        "planned_chunks": len(planned_chunks),
        "completed_chunks": len(results),
        "status_counts": dict(status_counts),
        "total_wall_seconds": round(total_seconds, 3),
        "estimated_credits_consumed": sum(int(row["estimated_credits_consumed"]) for row in results),
        "expected_credits": sum(int(row["expected_credits"]) for row in results),
        "rows_returned": sum(int(row["row_count"]) for row in results),
        "forecast_wide_rows_upserted": sum(int(row["forecast_wide_rows_upserted"]) for row in results),
        "api_event_log": event_log.as_posix(),
        "results_csv": results_csv.as_posix(),
        "progress_json": progress_json.as_posix(),
        "raw_root": raw_root.as_posix(),
        "db_summary": db_summary,
    }
    write_json(summary_json, summary)
    update_progress(progress_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tactical GribStream model-specific batched smoke pulls.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--days", type=int, default=70)
    parser.add_argument("--output-name", default="batch_smoke_10w")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--api-min-interval-seconds", type=float, default=12.0)
    parser.add_argument("--api-max-attempts", type=int, default=2)
    parser.add_argument("--pause-on-429-seconds", type=float, default=300.0)
    parser.add_argument("--max-retry-after-seconds", type=float, default=1800.0)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
