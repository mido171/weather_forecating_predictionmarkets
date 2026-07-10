from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text

from klga_tmax.providers.gribstream.backfill import parse_model_ids
from klga_tmax.providers.gribstream.catalog import resolve_all_selectors, spec_summary_rows
from klga_tmax.providers.gribstream.client import (
    GribStreamRequestError,
    GribStreamTimeseriesClient,
    OneThreadRateLimiter,
)
from klga_tmax.providers.gribstream.config import load_gribstream_settings
from klga_tmax.providers.gribstream.parser import parse_gribstream_response
from klga_tmax.providers.gribstream.persistence import (
    chunk_row_id,
    existing_completed_request,
    insert_catalog_snapshots,
    mark_chunk_running,
    mark_chunk_terminal,
    model_status,
    persist_gribstream_response,
    refresh_job_status,
    upsert_job_plan,
    upsert_source_gaps,
)
from klga_tmax.providers.gribstream.plan import (
    DEFAULT_END_DATE,
    T1245_CUTOFF_ID,
    build_runs_job_plan,
)


DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
DEFAULT_JOB_ID = "klga_t1245utc_runs_fast_backfill_v1"


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def utc_now_label() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def status_payload(connection, job_id: str) -> dict[str, Any]:
    status = refresh_job_status(connection, job_id)
    per_model = model_status(connection, job_id)
    total = int(status.get("chunks_total", 0))
    completed = int(status.get("chunks_completed", 0))
    failed = int(status.get("chunks_failed", 0))
    remaining = max(0, total - completed - failed)
    return {
        "job_status": status,
        "model_status": per_model,
        "chunks_total": total,
        "chunks_completed": completed,
        "chunks_failed": failed,
        "chunks_remaining": remaining,
        "percent_complete": round((completed / total) * 100.0, 3) if total else 100.0,
        "percent_left": round((remaining / total) * 100.0, 3) if total else 0.0,
    }


def print_event(event: str, payload: dict[str, Any]) -> None:
    print(json.dumps({"event": event, "at_utc": utc_now_label(), **json_safe(payload)}, sort_keys=True), flush=True)


def run(args: argparse.Namespace) -> int:
    database_url = args.database_url or os.environ.get("KLGA_DB_URL") or DEFAULT_DATABASE_URL
    settings = load_gribstream_settings(require_api_token=True)
    settings = replace(
        settings,
        spacing_seconds=args.spacing_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    model_ids = parse_model_ids(args.models)
    selectors, selector_gaps, snapshots = resolve_all_selectors(settings, model_ids=model_ids)
    plan = build_runs_job_plan(
        job_id=args.job_id,
        end_date=args.end_date,
        coordinate_tier_name=args.coordinate_tier,
        selectors_by_model=selectors,
        model_ids=model_ids,
        start_date_override=args.start_date,
        chunk_days_override=args.chunk_days,
        cutoff_id=T1245_CUTOFF_ID,
    )
    all_gaps = tuple(selector_gaps) + plan.selector_gaps
    config = {
        "endpoint": "/runs",
        "cutoff_id": T1245_CUTOFF_ID,
        "end_date": args.end_date.isoformat(),
        "coordinate_tier": args.coordinate_tier.upper(),
        "model_ids": list(model_ids) if model_ids else "all",
        "start_date_override": args.start_date.isoformat() if args.start_date else None,
        "chunk_days_override": args.chunk_days,
        "resume": args.resume,
        "max_chunks": args.max_chunks,
        "spacing_seconds": args.spacing_seconds,
        "specs": spec_summary_rows(end_date=args.end_date, cutoff_id=T1245_CUTOFF_ID),
        "notes": [
            "HKG-style fast path: POST /api/v2/{model}/runs with model-run timesList.",
            "Rows are filtered to expected run/valid pairs before silver persistence.",
            "Synoptic models are split into exact lead groups to avoid extra native horizons where possible.",
        ],
    }
    engine = create_engine(database_url, future=True, pool_pre_ping=True)
    with engine.begin() as connection:
        plan_counts = upsert_job_plan(connection, plan, config=config)
        inserted_catalogs = insert_catalog_snapshots(connection, snapshots)
        inserted_gaps = upsert_source_gaps(connection, all_gaps)
        current_status = status_payload(connection, args.job_id)
    print_event(
        "plan_seeded",
        {
            "job_id": args.job_id,
            "planned_chunks": len(plan.chunks),
            "estimated_credits": sum(chunk.estimated_credits for chunk in plan.chunks),
            "selector_gaps": len(all_gaps),
            "catalog_snapshots": len(snapshots),
            "plan_counts": plan_counts,
            "catalog_snapshots_inserted": inserted_catalogs,
            "gaps_inserted": inserted_gaps,
            **current_status,
        },
    )

    client = GribStreamTimeseriesClient(
        settings,
        rate_limiter=OneThreadRateLimiter(spacing_seconds=settings.spacing_seconds),
    )
    processed = 0
    fetched = 0
    skipped = 0
    rows_upserted = 0
    availability_rows = 0
    gaps_upserted = 0
    stopped_reason: str | None = None
    started = time.perf_counter()

    for chunk_index, chunk in enumerate(plan.chunks, start=1):
        if args.max_chunks is not None and processed >= args.max_chunks:
            break
        db_chunk_id = chunk_row_id(args.job_id, chunk)
        with engine.begin() as connection:
            current_status = connection.execute(
                text(
                    """
                    SELECT status
                    FROM audit.gribstream_backfill_chunks
                    WHERE job_id = :job_id AND chunk_id = :chunk_id
                    """
                ),
                {"job_id": args.job_id, "chunk_id": db_chunk_id},
            ).scalar_one_or_none()
            if current_status in {"completed", "completed_empty", "skipped"}:
                skipped += 1
                processed += 1
                progress = status_payload(connection, args.job_id)
                print_event(
                    "chunk_skipped_terminal",
                    {
                        "job_id": args.job_id,
                        "chunk_index": chunk_index,
                        "model_id": chunk.model_id,
                        "target_start_date": chunk.target_start_date.isoformat(),
                        "target_end_date": chunk.target_end_date.isoformat(),
                        "request_sha256": chunk.request_sha256,
                        "prior_status": current_status,
                        **progress,
                    },
                )
                continue
            if args.resume and existing_completed_request(connection, chunk.request_sha256):
                mark_chunk_terminal(connection, chunk_id=db_chunk_id, status="skipped")
                skipped += 1
                processed += 1
                progress = status_payload(connection, args.job_id)
                print_event(
                    "chunk_skipped_request_sha",
                    {
                        "job_id": args.job_id,
                        "chunk_index": chunk_index,
                        "model_id": chunk.model_id,
                        "target_start_date": chunk.target_start_date.isoformat(),
                        "target_end_date": chunk.target_end_date.isoformat(),
                        "request_sha256": chunk.request_sha256,
                        **progress,
                    },
                )
                continue
            mark_chunk_running(connection, job_id=args.job_id, chunk=chunk)

        try:
            fetch_started = time.perf_counter()
            response = client.fetch_chunk(chunk)
            fetch_seconds = time.perf_counter() - fetch_started
            parse_started = time.perf_counter()
            parsed = parse_gribstream_response(response)
            parse_seconds = time.perf_counter() - parse_started
            persist_started = time.perf_counter()
            with engine.begin() as connection:
                persisted = persist_gribstream_response(
                    connection,
                    job_id=args.job_id,
                    response=response,
                    parsed=parsed,
                )
                progress = status_payload(connection, args.job_id)
            persist_seconds = time.perf_counter() - persist_started
            fetched += 1
            processed += 1
            rows_upserted += persisted.rows_upserted
            availability_rows += persisted.availability_rows_upserted
            gaps_upserted += persisted.gaps_upserted
            elapsed = time.perf_counter() - started
            completed = int(progress["chunks_completed"])
            total = max(1, int(progress["chunks_total"]))
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta_seconds = (total - completed) / rate if rate > 0 else None
            print_event(
                "chunk_completed",
                {
                    "job_id": args.job_id,
                    "chunk_index": chunk_index,
                    "model_id": chunk.model_id,
                    "target_start_date": chunk.target_start_date.isoformat(),
                    "target_end_date": chunk.target_end_date.isoformat(),
                    "endpoint_type": chunk.endpoint_type,
                    "http_status": persisted.http_status,
                    "status": persisted.status,
                    "rows_upserted_this_chunk": persisted.rows_upserted,
                    "availability_rows_this_chunk": persisted.availability_rows_upserted,
                    "gaps_this_chunk": persisted.gaps_upserted,
                    "raw_storage_uri": response.raw_storage_uri,
                    "request_sha256": chunk.request_sha256,
                    "elapsed_seconds": round(elapsed, 1),
                    "fetch_seconds": round(fetch_seconds, 3),
                    "parse_seconds": round(parse_seconds, 3),
                    "persist_seconds": round(persist_seconds, 3),
                    "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
                    **progress,
                },
            )
        except GribStreamRequestError as exc:
            response = exc.response
            if response.http_status in {401, 403}:
                terminal_status = "auth_failed"
                stopped_reason = f"auth_failed_http_{response.http_status}"
            elif response.http_status == 429:
                terminal_status = "rate_limited"
                stopped_reason = "rate_limited_http_429"
            else:
                terminal_status = "failed"
            with engine.begin() as connection:
                mark_chunk_terminal(
                    connection,
                    chunk_id=db_chunk_id,
                    status=terminal_status,
                    response=response,
                    error_type=response.error_type,
                    error_message=response.error_message,
                )
                progress = status_payload(connection, args.job_id)
            processed += 1
            print_event(
                "chunk_failed",
                {
                    "job_id": args.job_id,
                    "chunk_index": chunk_index,
                    "model_id": chunk.model_id,
                    "target_start_date": chunk.target_start_date.isoformat(),
                    "target_end_date": chunk.target_end_date.isoformat(),
                    "terminal_status": terminal_status,
                    "http_status": response.http_status,
                    "error_type": response.error_type,
                    "error_message": response.error_message,
                    "retry_after": None,
                    "request_sha256": chunk.request_sha256,
                    **progress,
                },
            )
            if terminal_status in {"auth_failed", "rate_limited"}:
                break
        except Exception as exc:  # noqa: BLE001 - preserve chunk evidence and continue unless requested otherwise.
            with engine.begin() as connection:
                mark_chunk_terminal(
                    connection,
                    chunk_id=db_chunk_id,
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                progress = status_payload(connection, args.job_id)
            processed += 1
            print_event(
                "chunk_failed",
                {
                    "job_id": args.job_id,
                    "chunk_index": chunk_index,
                    "model_id": chunk.model_id,
                    "target_start_date": chunk.target_start_date.isoformat(),
                    "target_end_date": chunk.target_end_date.isoformat(),
                    "terminal_status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "request_sha256": chunk.request_sha256,
                    **progress,
                },
            )
            if args.stop_on_local_error:
                stopped_reason = f"local_error_{type(exc).__name__}"
                break

    with engine.begin() as connection:
        final_status = status_payload(connection, args.job_id)
    print_event(
        "run_finished",
        {
            "job_id": args.job_id,
            "processed_chunks_this_run": processed,
            "fetched_chunks_this_run": fetched,
            "skipped_chunks_this_run": skipped,
            "rows_upserted_this_run": rows_upserted,
            "availability_rows_this_run": availability_rows,
            "gaps_upserted_this_run": gaps_upserted,
            "stopped_reason": stopped_reason,
            "elapsed_seconds": round(time.perf_counter() - started, 1),
            **final_status,
        },
    )
    return 0 if stopped_reason is None and int(final_status["chunks_failed"]) == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the KLGA T_1245UTC HKG-style GribStream /runs fast backfill.")
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--database-url", default=os.environ.get("KLGA_DB_URL") or DEFAULT_DATABASE_URL)
    parser.add_argument("--models", default="all")
    parser.add_argument("--end-date", type=parse_date, default=DEFAULT_END_DATE)
    parser.add_argument("--start-date", type=parse_date, default=None)
    parser.add_argument("--coordinate-tier", default="B")
    parser.add_argument("--chunk-days", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--spacing-seconds", type=float, default=float(os.environ.get("GRIBSTREAM_SPACING_SECONDS", "2")))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.environ.get("GRIBSTREAM_TIMEOUT_SECONDS", "240")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("GRIBSTREAM_MAX_RETRIES", "3")))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-local-error", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
