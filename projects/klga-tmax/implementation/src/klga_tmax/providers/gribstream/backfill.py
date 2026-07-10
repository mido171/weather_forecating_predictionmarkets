from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from klga_tmax.providers.gribstream.catalog import resolve_all_selectors, spec_summary_rows
from klga_tmax.providers.gribstream.client import (
    GribStreamRequestError,
    GribStreamTimeseriesClient,
    OneThreadRateLimiter,
)
from klga_tmax.providers.gribstream.config import GribStreamSettings
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
    DEFAULT_CUTOFF_ID,
    DEFAULT_END_DATE,
    MODEL_SPECS,
    T1245_CUTOFF_ID,
    TMAX_THIN_FEATURE_PROFILE,
    TMAX_THIN_JOB_ID,
    TMAX_THIN_MODEL_SPECS,
    TMAX_THIN_PERSISTENCE_MODE,
    build_tmax_thin_runs_job_plan,
    build_job_plan,
    cutoff_profile_by_id,
    tmax_thin_spec_summary_rows,
)
from klga_tmax.registry.materialize_targets import materialize_target_instances


def parse_model_ids(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or raw.strip().lower() in {"", "all", "*"}:
        return None
    valid = {spec.model_id for spec in MODEL_SPECS}
    selected: list[str] = []
    for item in raw.split(","):
        model_id = item.strip().lower()
        if not model_id:
            continue
        if model_id not in valid:
            raise ValueError(f"unknown GribStream model {model_id}")
        selected.append(model_id)
    return tuple(dict.fromkeys(selected))


def parse_tmax_thin_model_ids(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or raw.strip().lower() in {"", "all", "*"}:
        return None
    valid = {spec.model_id for spec in TMAX_THIN_MODEL_SPECS}
    selected: list[str] = []
    for item in raw.split(","):
        model_id = item.strip().lower()
        if not model_id:
            continue
        if model_id not in valid:
            raise ValueError(f"unknown Tmax-thin GribStream model {model_id}")
        selected.append(model_id)
    return tuple(dict.fromkeys(selected))


def prepare_gribstream_plan(
    *,
    engine: Engine,
    settings: GribStreamSettings,
    job_id: str,
    end_date: date = DEFAULT_END_DATE,
    coordinate_tier: str = "B",
    model_ids: tuple[str, ...] | None = None,
    start_date_override: date | None = None,
    chunk_days_override: int | None = None,
    cutoff_id: str = DEFAULT_CUTOFF_ID,
    persist: bool = True,
) -> dict[str, Any]:
    cutoff_profile_by_id(cutoff_id)
    selectors, selector_gaps, snapshots = resolve_all_selectors(settings, model_ids=model_ids)
    plan = build_job_plan(
        job_id=job_id,
        end_date=end_date,
        coordinate_tier_name=coordinate_tier,
        selectors_by_model=selectors,
        model_ids=model_ids,
        start_date_override=start_date_override,
        chunk_days_override=chunk_days_override,
        cutoff_id=cutoff_id,
    )
    all_gaps = tuple(selector_gaps) + plan.selector_gaps
    config = {
        "endpoint": "/timeseries",
        "cutoff_id": plan.cutoff_id,
        "end_date": end_date.isoformat(),
        "coordinate_tier": coordinate_tier.upper(),
        "model_ids": list(model_ids) if model_ids else "all",
        "start_date_override": start_date_override.isoformat() if start_date_override else None,
        "chunk_days_override": chunk_days_override,
        "specs": spec_summary_rows(end_date=end_date, cutoff_id=cutoff_id),
    }
    row_counts: dict[str, Any] = {
        "job_id": job_id,
        "chunks_planned": len(plan.chunks),
        "selector_gaps": len(all_gaps),
        "catalog_snapshots": len(snapshots),
        "estimated_credits": sum(chunk.estimated_credits for chunk in plan.chunks),
        "models_planned": len({chunk.model_id for chunk in plan.chunks}),
    }
    if persist:
        with engine.begin() as connection:
            row_counts.update(upsert_job_plan(connection, plan, config=config))
            row_counts["audit.gribstream_catalog_snapshots"] = insert_catalog_snapshots(connection, snapshots)
            row_counts["audit.gribstream_source_gaps"] = int(row_counts.get("audit.gribstream_source_gaps", 0)) + upsert_source_gaps(connection, all_gaps)
            row_counts["model_status"] = model_status(connection, job_id)
    return row_counts


def prepare_tmax_thin_plan(
    *,
    engine: Engine,
    settings: GribStreamSettings,
    job_id: str = TMAX_THIN_JOB_ID,
    end_date: date = DEFAULT_END_DATE,
    model_ids: tuple[str, ...] | None = None,
    start_date_override: date | None = None,
    chunk_days_override: int | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    cutoff_profile_by_id(T1245_CUTOFF_ID)
    selectors, selector_gaps, snapshots = resolve_all_selectors(
        settings,
        model_ids=model_ids,
        model_specs=TMAX_THIN_MODEL_SPECS,
    )
    plan = build_tmax_thin_runs_job_plan(
        job_id=job_id,
        end_date=end_date,
        selectors_by_model=selectors,
        model_ids=model_ids,
        start_date_override=start_date_override,
        chunk_days_override=chunk_days_override,
    )
    all_gaps = tuple(selector_gaps) + plan.selector_gaps
    config = {
        "endpoint": "/runs",
        "feature_profile": TMAX_THIN_FEATURE_PROFILE,
        "persistence_mode": TMAX_THIN_PERSISTENCE_MODE,
        "cutoff_id": plan.cutoff_id,
        "end_date": end_date.isoformat(),
        "coordinate_tier": plan.coordinate_tier,
        "model_ids": list(model_ids) if model_ids else "all",
        "start_date_override": start_date_override.isoformat() if start_date_override else None,
        "chunk_days_override": chunk_days_override,
        "specs": tmax_thin_spec_summary_rows(end_date=end_date),
    }
    row_counts: dict[str, Any] = {
        "job_id": job_id,
        "feature_profile": TMAX_THIN_FEATURE_PROFILE,
        "persistence_mode": TMAX_THIN_PERSISTENCE_MODE,
        "chunks_planned": len(plan.chunks),
        "selector_gaps": len(all_gaps),
        "catalog_snapshots": len(snapshots),
        "estimated_credits": sum(chunk.estimated_credits for chunk in plan.chunks),
        "models_planned": len({chunk.model_id for chunk in plan.chunks}),
    }
    if persist:
        with engine.begin() as connection:
            if plan.chunks:
                materialize_target_instances(
                    connection,
                    start_date=plan.start_date,
                    end_date=plan.end_date,
                    replace=False,
                )
            row_counts.update(upsert_job_plan(connection, plan, config=config))
            row_counts["audit.gribstream_catalog_snapshots"] = insert_catalog_snapshots(connection, snapshots)
            row_counts["audit.gribstream_source_gaps"] = int(row_counts.get("audit.gribstream_source_gaps", 0)) + upsert_source_gaps(connection, all_gaps)
            row_counts["model_status"] = model_status(connection, job_id)
    return row_counts


def run_gribstream_backfill(
    *,
    engine: Engine,
    settings: GribStreamSettings,
    job_id: str,
    end_date: date = DEFAULT_END_DATE,
    coordinate_tier: str = "B",
    model_ids: tuple[str, ...] | None = None,
    start_date_override: date | None = None,
    chunk_days_override: int | None = None,
    cutoff_id: str = DEFAULT_CUTOFF_ID,
    max_chunks: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    cutoff_profile_by_id(cutoff_id)
    selectors, selector_gaps, snapshots = resolve_all_selectors(settings, model_ids=model_ids)
    plan = build_job_plan(
        job_id=job_id,
        end_date=end_date,
        coordinate_tier_name=coordinate_tier,
        selectors_by_model=selectors,
        model_ids=model_ids,
        start_date_override=start_date_override,
        chunk_days_override=chunk_days_override,
        cutoff_id=cutoff_id,
    )
    config = {
        "endpoint": "/timeseries",
        "cutoff_id": plan.cutoff_id,
        "end_date": end_date.isoformat(),
        "coordinate_tier": coordinate_tier.upper(),
        "model_ids": list(model_ids) if model_ids else "all",
        "start_date_override": start_date_override.isoformat() if start_date_override else None,
        "chunk_days_override": chunk_days_override,
        "resume": resume,
        "max_chunks": max_chunks,
        "specs": spec_summary_rows(end_date=end_date, cutoff_id=cutoff_id),
    }
    with engine.begin() as connection:
        upsert_job_plan(connection, plan, config=config)
        insert_catalog_snapshots(connection, snapshots)
        upsert_source_gaps(connection, tuple(selector_gaps) + plan.selector_gaps)

    client = GribStreamTimeseriesClient(
        settings,
        rate_limiter=OneThreadRateLimiter(spacing_seconds=settings.spacing_seconds),
    )
    fetched = 0
    skipped = 0
    rows_upserted = 0
    availability_rows = 0
    gaps_upserted = 0
    stopped_reason: str | None = None

    for chunk in plan.chunks:
        if max_chunks is not None and fetched >= max_chunks:
            break
        with engine.begin() as connection:
            db_chunk_id = chunk_row_id(job_id, chunk)
            current_status = connection.execute(
                text(
                    """
                    SELECT status
                    FROM audit.gribstream_backfill_chunks
                    WHERE job_id = :job_id
                      AND chunk_id = :chunk_id
                    """
                ),
                {"job_id": job_id, "chunk_id": db_chunk_id},
            ).scalar_one_or_none()
            if current_status in {"completed", "completed_empty", "skipped"}:
                skipped += 1
                continue
            if resume and existing_completed_request(connection, chunk.request_sha256):
                mark_chunk_terminal(
                    connection,
                    chunk_id=db_chunk_id,
                    status="skipped",
                )
                refresh_job_status(connection, job_id)
                skipped += 1
                continue
            mark_chunk_running(connection, job_id=job_id, chunk=chunk)
        try:
            response = client.fetch_chunk(chunk)
            parsed = parse_gribstream_response(response)
            with engine.begin() as connection:
                persisted = persist_gribstream_response(
                    connection,
                    job_id=job_id,
                    response=response,
                    parsed=parsed,
                )
                status = refresh_job_status(connection, job_id)
            fetched += 1
            rows_upserted += persisted.rows_upserted
            availability_rows += persisted.availability_rows_upserted
            gaps_upserted += persisted.gaps_upserted
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
                    chunk_id=chunk_row_id(job_id, chunk),
                    status=terminal_status,
                    response=response,
                    error_type=response.error_type,
                    error_message=response.error_message,
                )
                status = refresh_job_status(connection, job_id)
            if terminal_status in {"auth_failed", "rate_limited"}:
                break
            fetched += 1
        except Exception as exc:
            with engine.begin() as connection:
                mark_chunk_terminal(
                    connection,
                    chunk_id=chunk_row_id(job_id, chunk),
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                status = refresh_job_status(connection, job_id)
            fetched += 1

    with engine.begin() as connection:
        final_status = refresh_job_status(connection, job_id)
        per_model = model_status(connection, job_id)
    return {
        "job_id": job_id,
        "chunks_fetched": fetched,
        "chunks_skipped_completed": skipped,
        "rows_upserted": rows_upserted,
        "availability_rows_upserted": availability_rows,
        "gaps_upserted": gaps_upserted,
        "stopped_reason": stopped_reason,
        "job_status": final_status,
        "model_status": per_model,
    }


def run_tmax_thin_backfill(
    *,
    engine: Engine,
    settings: GribStreamSettings,
    job_id: str = TMAX_THIN_JOB_ID,
    end_date: date = DEFAULT_END_DATE,
    model_ids: tuple[str, ...] | None = None,
    start_date_override: date | None = None,
    chunk_days_override: int | None = None,
    max_chunks: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    cutoff_profile_by_id(T1245_CUTOFF_ID)
    selectors, selector_gaps, snapshots = resolve_all_selectors(
        settings,
        model_ids=model_ids,
        model_specs=TMAX_THIN_MODEL_SPECS,
    )
    plan = build_tmax_thin_runs_job_plan(
        job_id=job_id,
        end_date=end_date,
        selectors_by_model=selectors,
        model_ids=model_ids,
        start_date_override=start_date_override,
        chunk_days_override=chunk_days_override,
    )
    config = {
        "endpoint": "/runs",
        "feature_profile": TMAX_THIN_FEATURE_PROFILE,
        "persistence_mode": TMAX_THIN_PERSISTENCE_MODE,
        "cutoff_id": plan.cutoff_id,
        "end_date": end_date.isoformat(),
        "coordinate_tier": plan.coordinate_tier,
        "model_ids": list(model_ids) if model_ids else "all",
        "start_date_override": start_date_override.isoformat() if start_date_override else None,
        "chunk_days_override": chunk_days_override,
        "resume": resume,
        "max_chunks": max_chunks,
        "specs": tmax_thin_spec_summary_rows(end_date=end_date),
    }
    with engine.begin() as connection:
        if plan.chunks:
            materialize_target_instances(
                connection,
                start_date=plan.start_date,
                end_date=plan.end_date,
                replace=False,
            )
        upsert_job_plan(connection, plan, config=config)
        insert_catalog_snapshots(connection, snapshots)
        upsert_source_gaps(connection, tuple(selector_gaps) + plan.selector_gaps)

    client = GribStreamTimeseriesClient(
        settings,
        rate_limiter=OneThreadRateLimiter(spacing_seconds=settings.spacing_seconds),
    )
    fetched = 0
    skipped = 0
    feature_rows_upserted = 0
    availability_rows = 0
    gaps_upserted = 0
    stopped_reason: str | None = None

    for chunk in plan.chunks:
        if max_chunks is not None and fetched >= max_chunks:
            break
        with engine.begin() as connection:
            db_chunk_id = chunk_row_id(job_id, chunk)
            current_status = connection.execute(
                text(
                    """
                    SELECT status
                    FROM audit.gribstream_backfill_chunks
                    WHERE job_id = :job_id
                      AND chunk_id = :chunk_id
                    """
                ),
                {"job_id": job_id, "chunk_id": db_chunk_id},
            ).scalar_one_or_none()
            if current_status in {"completed", "completed_empty", "skipped"}:
                skipped += 1
                continue
            if resume and existing_completed_request(connection, chunk.request_sha256):
                mark_chunk_terminal(
                    connection,
                    chunk_id=db_chunk_id,
                    status="skipped",
                )
                refresh_job_status(connection, job_id)
                skipped += 1
                continue
            mark_chunk_running(connection, job_id=job_id, chunk=chunk)
        try:
            response = client.fetch_chunk(chunk)
            parsed = parse_gribstream_response(response)
            with engine.begin() as connection:
                persisted = persist_gribstream_response(
                    connection,
                    job_id=job_id,
                    response=response,
                    parsed=parsed,
                    persistence_mode=TMAX_THIN_PERSISTENCE_MODE,
                )
                refresh_job_status(connection, job_id)
            fetched += 1
            feature_rows_upserted += persisted.rows_upserted
            availability_rows += persisted.availability_rows_upserted
            gaps_upserted += persisted.gaps_upserted
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
                    chunk_id=chunk_row_id(job_id, chunk),
                    status=terminal_status,
                    response=response,
                    error_type=response.error_type,
                    error_message=response.error_message,
                )
                refresh_job_status(connection, job_id)
            if terminal_status in {"auth_failed", "rate_limited"}:
                break
            fetched += 1
        except Exception as exc:
            with engine.begin() as connection:
                mark_chunk_terminal(
                    connection,
                    chunk_id=chunk_row_id(job_id, chunk),
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                refresh_job_status(connection, job_id)
            fetched += 1

    with engine.begin() as connection:
        final_status = refresh_job_status(connection, job_id)
        per_model = model_status(connection, job_id)
    return {
        "job_id": job_id,
        "feature_profile": TMAX_THIN_FEATURE_PROFILE,
        "persistence_mode": TMAX_THIN_PERSISTENCE_MODE,
        "chunks_fetched": fetched,
        "chunks_skipped_completed": skipped,
        "feature_rows_upserted": feature_rows_upserted,
        "availability_rows_upserted": availability_rows,
        "gaps_upserted": gaps_upserted,
        "stopped_reason": stopped_reason,
        "job_status": final_status,
        "model_status": per_model,
    }
