from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.ingestion.bronze import CurrentBronzeRecord, decide_bronze_revision
from klga_tmax.ingestion.hash_keys import canonical_json, payload_hash, sha256_hex
from klga_tmax.providers.gribstream.client import PARSER_VERSION
from klga_tmax.providers.gribstream.catalog import CatalogSnapshot
from klga_tmax.providers.gribstream.features import (
    FEATURE_BUILD_VERSION,
    build_tmax_thin_gold_features,
)
from klga_tmax.providers.gribstream.models import (
    GribStreamChunk,
    GribStreamGoldFeature,
    GribStreamJobPlan,
    GribStreamRawResponse,
    ParsedGribStreamResponse,
    PersistedGribStreamChunk,
)
from klga_tmax.registry.materialize_targets import materialize_target_instances
from klga_tmax.utils.git import current_git_sha

SOURCE_NAME = "gribstream"
PROVIDER_NAME = "gribstream"
ENDPOINT_NAME = "timeseries"
THIN_FEATURE_SET_NAME = "klga_tmax_gribstream_tmax_thin"
THIN_FORMULA_CONTRACT_HASH = "gribstream_tmax_thin_v1"


def chunk_row_id(job_id: str, chunk: GribStreamChunk) -> str:
    return f"gs_chunk_{sha256_hex(canonical_json({'job_id': job_id, 'request_sha256': chunk.request_sha256}))[:32]}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _source_request_id(response: GribStreamRawResponse) -> str:
    identity = {
        "source": SOURCE_NAME,
        "endpoint": response.chunk.endpoint_type,
        "model_id": response.chunk.model_id,
        "request_sha256": response.chunk.request_sha256,
        "retrieved_at_utc": response.retrieved_at_utc.isoformat(),
        "http_status": response.http_status,
        "body_hash": response.response_body_sha256,
    }
    return f"gs_req_{sha256_hex(canonical_json(identity))[:32]}"


def _provider_record_key(chunk: GribStreamChunk) -> str:
    return (
        f"{chunk.endpoint_type}:{chunk.model_id}:{chunk.cutoff_id}:"
        f"{chunk.target_start_date.isoformat()}:{chunk.target_end_date.isoformat()}:"
        f"{chunk.coordinate_tier}:{chunk.request_sha256}"
    )


def insert_catalog_snapshots(connection: Connection, snapshots: tuple[CatalogSnapshot, ...]) -> int:
    inserted = 0
    for snapshot in snapshots:
        result = connection.execute(
            text(
                """
                INSERT INTO audit.gribstream_catalog_snapshots (
                    model_id,
                    catalog_kind,
                    catalog_url,
                    payload_sha256,
                    payload_json,
                    retrieved_at_utc,
                    status,
                    error_message
                )
                VALUES (
                    :model_id,
                    :catalog_kind,
                    :catalog_url,
                    :payload_sha256,
                    CAST(:payload_json AS jsonb),
                    :retrieved_at_utc,
                    :status,
                    :error_message
                )
                ON CONFLICT (catalog_url, payload_sha256) DO NOTHING
                """
            ),
            {
                "model_id": snapshot.model_id,
                "catalog_kind": snapshot.catalog_kind,
                "catalog_url": snapshot.catalog_url,
                "payload_sha256": snapshot.payload_sha256,
                "payload_json": _json_dumps(snapshot.payload_json),
                "retrieved_at_utc": snapshot.retrieved_at_utc,
                "status": snapshot.status,
                "error_message": snapshot.error_message,
            },
        )
        inserted += result.rowcount or 0
    return inserted


def upsert_job_plan(connection: Connection, plan: GribStreamJobPlan, *, config: dict[str, Any]) -> dict[str, int]:
    estimated_credits = sum(chunk.estimated_credits for chunk in plan.chunks)
    connection.execute(
        text(
            """
            DELETE FROM audit.gribstream_backfill_chunks
            WHERE job_id = :job_id
              AND status NOT IN ('completed','completed_empty')
            """
        ),
        {"job_id": plan.job_id},
    )
    connection.execute(
        text(
            """
            INSERT INTO audit.gribstream_backfill_jobs (
                job_id,
                cutoff_id,
                start_date,
                end_date,
                coordinate_tier,
                status,
                planned_chunks,
                completed_chunks,
                failed_chunks,
                estimated_credits,
                row_counts_json,
                config_json
            )
            VALUES (
                :job_id,
                :cutoff_id,
                :start_date,
                :end_date,
                :coordinate_tier,
                'planned',
                :planned_chunks,
                0,
                0,
                :estimated_credits,
                '{}'::jsonb,
                CAST(:config_json AS jsonb)
            )
            ON CONFLICT (job_id) DO UPDATE SET
                cutoff_id = EXCLUDED.cutoff_id,
                start_date = EXCLUDED.start_date,
                end_date = EXCLUDED.end_date,
                coordinate_tier = EXCLUDED.coordinate_tier,
                planned_chunks = EXCLUDED.planned_chunks,
                estimated_credits = EXCLUDED.estimated_credits,
                config_json = EXCLUDED.config_json,
                updated_at = now()
            """
        ),
        {
            "job_id": plan.job_id,
            "cutoff_id": plan.cutoff_id,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "coordinate_tier": plan.coordinate_tier,
            "planned_chunks": len(plan.chunks),
            "estimated_credits": estimated_credits,
            "config_json": _json_dumps(config),
        },
    )
    chunks = 0
    for chunk in plan.chunks:
        chunks += upsert_planned_chunk(connection, plan.job_id, chunk)
    gaps = upsert_source_gaps(connection, plan.selector_gaps)
    return {
        "audit.gribstream_backfill_jobs": 1,
        "audit.gribstream_backfill_chunks": chunks,
        "audit.gribstream_source_gaps": gaps,
    }


def upsert_planned_chunk(connection: Connection, job_id: str, chunk: GribStreamChunk) -> int:
    db_chunk_id = chunk_row_id(job_id, chunk)
    result = connection.execute(
        text(
            """
            INSERT INTO audit.gribstream_backfill_chunks (
                chunk_id,
                job_id,
                model_id,
                target_start_date,
                target_end_date,
                cutoff_id,
                endpoint_type,
                coordinate_tier,
                as_of_utc,
                valid_time_count,
                variable_count,
                member_count,
                estimated_credits,
                request_sha256,
                request_json,
                status
            )
            VALUES (
                :chunk_id,
                :job_id,
                :model_id,
                :target_start_date,
                :target_end_date,
                :cutoff_id,
                :endpoint_type,
                :coordinate_tier,
                :as_of_utc,
                :valid_time_count,
                :variable_count,
                :member_count,
                :estimated_credits,
                :request_sha256,
                CAST(:request_json AS jsonb),
                'planned'
            )
            ON CONFLICT (job_id, request_sha256) DO UPDATE SET
                job_id = EXCLUDED.job_id,
                endpoint_type = EXCLUDED.endpoint_type,
                estimated_credits = EXCLUDED.estimated_credits,
                request_json = EXCLUDED.request_json,
                updated_at = now()
            WHERE audit.gribstream_backfill_chunks.status NOT IN ('completed','completed_empty')
            """
        ),
        {
            "chunk_id": db_chunk_id,
            "job_id": job_id,
            "model_id": chunk.model_id,
            "target_start_date": chunk.target_start_date,
            "target_end_date": chunk.target_end_date,
            "cutoff_id": chunk.cutoff_id,
            "endpoint_type": chunk.endpoint_type,
            "coordinate_tier": chunk.coordinate_tier,
            "as_of_utc": chunk.as_of_utc,
            "valid_time_count": len(chunk.valid_times_utc),
            "variable_count": len(chunk.selectors),
            "member_count": len(chunk.members) if chunk.members else 1,
            "estimated_credits": chunk.estimated_credits,
            "request_sha256": chunk.request_sha256,
            "request_json": _json_dumps(chunk.request_payload),
        },
    )
    return result.rowcount or 0


def claim_next_chunk(connection: Connection, job_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        text(
            """
            UPDATE audit.gribstream_backfill_chunks
            SET
                status = 'running',
                attempts = attempts + 1,
                started_at_utc = COALESCE(started_at_utc, now()),
                updated_at = now()
            WHERE chunk_id = (
                SELECT chunk_id
                FROM audit.gribstream_backfill_chunks
                WHERE job_id = :job_id
                  AND status IN ('planned','failed')
                ORDER BY model_id, target_start_date
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
            """
        ),
        {"job_id": job_id},
    ).mappings().first()
    return dict(row) if row is not None else None


def mark_chunk_running(connection: Connection, *, job_id: str, chunk: GribStreamChunk) -> None:
    connection.execute(
        text(
            """
            UPDATE audit.gribstream_backfill_chunks
            SET
                status = 'running',
                attempts = attempts + 1,
                started_at_utc = COALESCE(started_at_utc, now()),
                updated_at = now()
            WHERE chunk_id = :chunk_id
              AND job_id = :job_id
              AND status IN ('planned','failed','running')
            """
        ),
        {
            "chunk_id": chunk_row_id(job_id, chunk),
            "job_id": job_id,
        },
    )


def existing_completed_request(connection: Connection, request_sha256: str) -> bool:
    count = connection.execute(
        text(
            """
            SELECT count(*)
            FROM audit.gribstream_backfill_chunks
            WHERE request_sha256 = :request_sha256
              AND status IN ('completed','completed_empty')
            """
        ),
        {"request_sha256": request_sha256},
    ).scalar_one()
    return int(count) > 0


def _insert_source_request(
    connection: Connection,
    response: GribStreamRawResponse,
    source_request_id: str,
) -> None:
    connection.execute(
        text(
            """
            INSERT INTO bronze.source_requests (
                source_request_id,
                source_name,
                source_endpoint,
                request_method,
                request_params_json,
                request_headers_redacted,
                retrieved_at_utc,
                provider_response_timestamp,
                http_status,
                response_content_type,
                response_body_sha256,
                response_size_bytes,
                raw_storage_uri,
                parser_version
            )
            VALUES (
                :source_request_id,
                :source_name,
                :source_endpoint,
                'POST',
                CAST(:request_params_json AS jsonb),
                CAST(:request_headers_redacted AS jsonb),
                :retrieved_at_utc,
                NULL,
                :http_status,
                :content_type,
                :response_body_sha256,
                :response_size_bytes,
                :raw_storage_uri,
                :parser_version
            )
            ON CONFLICT (source_request_id) DO NOTHING
            """
        ),
        {
            "source_request_id": source_request_id,
            "source_name": SOURCE_NAME,
            "source_endpoint": response.endpoint_url_redacted,
            "request_params_json": _json_dumps(response.chunk.request_payload),
            "request_headers_redacted": _json_dumps(
                {
                    "Accept": "application/ndjson",
                    "Content-Type": "application/json",
                    "Authorization": "Bearer REDACTED",
                }
            ),
            "retrieved_at_utc": response.retrieved_at_utc,
            "http_status": response.http_status,
            "content_type": response.content_type,
            "response_body_sha256": response.response_body_sha256,
            "response_size_bytes": response.response_size_bytes,
            "raw_storage_uri": response.raw_storage_uri,
            "parser_version": PARSER_VERSION,
        },
    )


def _current_source_record(connection: Connection, provider_record_key: str) -> CurrentBronzeRecord | None:
    row = connection.execute(
        text(
            """
            SELECT source_record_id, payload_hash, revision_number, is_current
            FROM bronze.source_records
            WHERE source_name = :source_name
              AND provider_name = :provider_name
              AND endpoint_name = :endpoint_name
              AND provider_record_key = :provider_record_key
            ORDER BY is_current DESC, revision_number DESC, acquired_at_utc DESC
            LIMIT 1
            """
        ),
        {
            "source_name": SOURCE_NAME,
            "provider_name": PROVIDER_NAME,
            "endpoint_name": provider_record_key.split(":", 1)[0],
            "provider_record_key": provider_record_key,
        },
    ).mappings().first()
    if row is None:
        return None
    return CurrentBronzeRecord(
        source_record_id=row["source_record_id"],
        payload_hash=row["payload_hash"],
        revision_number=int(row["revision_number"]),
        is_current=bool(row["is_current"]),
    )


def _insert_or_reuse_source_record(
    connection: Connection,
    response: GribStreamRawResponse,
    source_request_id: str,
) -> UUID:
    provider_record_key = _provider_record_key(response.chunk)
    current = _current_source_record(connection, provider_record_key)
    new_payload_hash = payload_hash(response.raw_storage_uri)
    decision = decide_bronze_revision(current_record=current, new_payload_hash=new_payload_hash)
    if decision.action == "return_existing" and decision.source_record_id is not None:
        return decision.source_record_id
    if decision.mark_prior_current_false and decision.supersedes_source_record_id is not None:
        connection.execute(
            text("UPDATE bronze.source_records SET is_current = false WHERE source_record_id = :source_record_id"),
            {"source_record_id": decision.supersedes_source_record_id},
        )
    row = connection.execute(
        text(
            """
            INSERT INTO bronze.source_records (
                source_request_id,
                source_name,
                provider_name,
                endpoint_name,
                provider_record_key,
                request_hash,
                payload_hash,
                payload_format,
                payload_json,
                payload_text,
                payload_uri,
                provider_issued_at_utc,
                provider_valid_at_utc,
                provider_available_at_utc,
                acquired_at_utc,
                revision_number,
                supersedes_source_record_id,
                is_current
            )
            VALUES (
                :source_request_id,
                :source_name,
                :provider_name,
                :endpoint_name,
                :provider_record_key,
                :request_hash,
                :payload_hash,
                'binary_uri',
                NULL,
                NULL,
                :payload_uri,
                :provider_issued_at_utc,
                NULL,
                :provider_available_at_utc,
                :acquired_at_utc,
                :revision_number,
                :supersedes_source_record_id,
                true
            )
            RETURNING source_record_id
            """
        ),
        {
            "source_request_id": source_request_id,
            "source_name": SOURCE_NAME,
            "provider_name": PROVIDER_NAME,
            "endpoint_name": response.chunk.endpoint_type,
            "provider_record_key": provider_record_key,
            "request_hash": response.chunk.request_sha256,
            "payload_hash": new_payload_hash,
            "payload_uri": response.raw_storage_uri,
            "provider_issued_at_utc": response.chunk.as_of_utc if response.chunk.endpoint_type == "timeseries" else None,
            "provider_available_at_utc": response.retrieved_at_utc,
            "acquired_at_utc": response.retrieved_at_utc,
            "revision_number": decision.revision_number,
            "supersedes_source_record_id": decision.supersedes_source_record_id,
        },
    ).one()
    return row.source_record_id


def _upsert_value(
    connection: Connection,
    *,
    value,
    source_request_id: str,
    source_record_id: UUID,
    request_sha256: str,
    ingested_at_utc: datetime,
) -> int:
    result = connection.execute(
        text(
            """
            INSERT INTO silver.grib_forecast_values (
                model_id,
                endpoint_type,
                target_date,
                cutoff_id,
                cutoff_utc,
                as_of_utc,
                coordinate_tier,
                grid_point_id,
                lat,
                lon,
                forecasted_at_utc,
                forecasted_time_utc,
                forecast_hour,
                member,
                variable_alias,
                variable_name,
                variable_level,
                variable_info,
                unit_original,
                value_original,
                unit_canonical,
                value_canonical,
                index_updated_at_utc,
                provider_available_at_utc,
                effective_available_at_utc,
                our_ingested_at_utc,
                availability_method,
                source_request_id,
                source_record_id,
                request_sha256,
                raw_row_hash,
                raw_row_json,
                quality_flag,
                quality_note
            )
            VALUES (
                :model_id,
                :endpoint_type,
                :target_date,
                :cutoff_id,
                :cutoff_utc,
                :as_of_utc,
                :coordinate_tier,
                :grid_point_id,
                :lat,
                :lon,
                :forecasted_at_utc,
                :forecasted_time_utc,
                :forecast_hour,
                :member,
                :variable_alias,
                :variable_name,
                :variable_level,
                :variable_info,
                :unit_original,
                :value_original,
                :unit_canonical,
                :value_canonical,
                :index_updated_at_utc,
                :provider_available_at_utc,
                :effective_available_at_utc,
                :our_ingested_at_utc,
                :availability_method,
                :source_request_id,
                :source_record_id,
                :request_sha256,
                :raw_row_hash,
                CAST(:raw_row_json AS jsonb),
                :quality_flag,
                :quality_note
            )
            ON CONFLICT (raw_row_hash) DO UPDATE SET
                value_original = EXCLUDED.value_original,
                value_canonical = EXCLUDED.value_canonical,
                source_request_id = EXCLUDED.source_request_id,
                source_record_id = EXCLUDED.source_record_id,
                raw_row_json = EXCLUDED.raw_row_json,
                quality_flag = EXCLUDED.quality_flag,
                quality_note = EXCLUDED.quality_note,
                updated_at = now()
            """
        ),
        {
            **value.__dict__,
            "endpoint_type": value.endpoint_type,
            "raw_row_json": _json_dumps(value.raw_row_json),
            "our_ingested_at_utc": ingested_at_utc,
            "source_request_id": source_request_id,
            "source_record_id": source_record_id,
            "request_sha256": request_sha256,
        },
    )
    return result.rowcount or 0


VALUE_UPSERT_SQL = text(
    """
    INSERT INTO silver.grib_forecast_values (
        model_id,
        endpoint_type,
        target_date,
        cutoff_id,
        cutoff_utc,
        as_of_utc,
        coordinate_tier,
        grid_point_id,
        lat,
        lon,
        forecasted_at_utc,
        forecasted_time_utc,
        forecast_hour,
        member,
        variable_alias,
        variable_name,
        variable_level,
        variable_info,
        unit_original,
        value_original,
        unit_canonical,
        value_canonical,
        index_updated_at_utc,
        provider_available_at_utc,
        effective_available_at_utc,
        our_ingested_at_utc,
        availability_method,
        source_request_id,
        source_record_id,
        request_sha256,
        raw_row_hash,
        raw_row_json,
        quality_flag,
        quality_note
    )
    VALUES (
        :model_id,
        :endpoint_type,
        :target_date,
        :cutoff_id,
        :cutoff_utc,
        :as_of_utc,
        :coordinate_tier,
        :grid_point_id,
        :lat,
        :lon,
        :forecasted_at_utc,
        :forecasted_time_utc,
        :forecast_hour,
        :member,
        :variable_alias,
        :variable_name,
        :variable_level,
        :variable_info,
        :unit_original,
        :value_original,
        :unit_canonical,
        :value_canonical,
        :index_updated_at_utc,
        :provider_available_at_utc,
        :effective_available_at_utc,
        :our_ingested_at_utc,
        :availability_method,
        :source_request_id,
        :source_record_id,
        :request_sha256,
        :raw_row_hash,
        CAST(:raw_row_json AS jsonb),
        :quality_flag,
        :quality_note
    )
    ON CONFLICT (raw_row_hash) DO UPDATE SET
        value_original = EXCLUDED.value_original,
        value_canonical = EXCLUDED.value_canonical,
        source_request_id = EXCLUDED.source_request_id,
        source_record_id = EXCLUDED.source_record_id,
        raw_row_json = EXCLUDED.raw_row_json,
        quality_flag = EXCLUDED.quality_flag,
        quality_note = EXCLUDED.quality_note,
        updated_at = now()
    """
)


VALUE_COLUMNS = (
    "model_id",
    "endpoint_type",
    "target_date",
    "cutoff_id",
    "cutoff_utc",
    "as_of_utc",
    "coordinate_tier",
    "grid_point_id",
    "lat",
    "lon",
    "forecasted_at_utc",
    "forecasted_time_utc",
    "forecast_hour",
    "member",
    "variable_alias",
    "variable_name",
    "variable_level",
    "variable_info",
    "unit_original",
    "value_original",
    "unit_canonical",
    "value_canonical",
    "index_updated_at_utc",
    "provider_available_at_utc",
    "effective_available_at_utc",
    "our_ingested_at_utc",
    "availability_method",
    "source_request_id",
    "source_record_id",
    "request_sha256",
    "raw_row_hash",
    "raw_row_json",
    "quality_flag",
    "quality_note",
)


VALUE_COPY_STAGE_DDL = text(
    """
    CREATE TEMP TABLE IF NOT EXISTS grib_forecast_values_stage (
        model_id text NOT NULL,
        endpoint_type text NOT NULL,
        target_date date NOT NULL,
        cutoff_id text NOT NULL,
        cutoff_utc timestamptz NOT NULL,
        as_of_utc timestamptz,
        coordinate_tier text NOT NULL,
        grid_point_id text NOT NULL,
        lat double precision NOT NULL,
        lon double precision NOT NULL,
        forecasted_at_utc timestamptz NOT NULL,
        forecasted_time_utc timestamptz NOT NULL,
        forecast_hour double precision NOT NULL,
        member text NOT NULL,
        variable_alias text NOT NULL,
        variable_name text NOT NULL,
        variable_level text,
        variable_info text,
        unit_original text,
        value_original double precision,
        unit_canonical text,
        value_canonical double precision,
        index_updated_at_utc timestamptz,
        provider_available_at_utc timestamptz NOT NULL,
        effective_available_at_utc timestamptz NOT NULL,
        our_ingested_at_utc timestamptz NOT NULL,
        availability_method text NOT NULL,
        source_request_id text NOT NULL,
        source_record_id uuid,
        request_sha256 text NOT NULL,
        raw_row_hash text NOT NULL,
        raw_row_json jsonb NOT NULL,
        quality_flag text NOT NULL,
        quality_note text
    ) ON COMMIT DROP
    """
)


VALUE_COPY_UPSERT_SQL = text(
    f"""
    INSERT INTO silver.grib_forecast_values (
        {", ".join(VALUE_COLUMNS)}
    )
    SELECT
        {", ".join(VALUE_COLUMNS)}
    FROM grib_forecast_values_stage
    ON CONFLICT (raw_row_hash) DO NOTHING
    """
)


AVAILABILITY_INSERT_SQL = text(
    """
    INSERT INTO silver.availability_ledger (
        source_record_id,
        source_name,
        provider_name,
        canonical_record_key,
        station_id,
        model_name,
        run_time_utc,
        valid_time_utc,
        forecast_hour,
        member,
        variable_name,
        provider_available_at_utc,
        acquired_at_utc,
        effective_available_at_utc,
        availability_method,
        source_lag_seconds,
        is_revision_current
    )
    VALUES (
        :source_record_id,
        :source_name,
        :provider_name,
        :canonical_record_key,
        :station_id,
        :model_name,
        :run_time_utc,
        :valid_time_utc,
        :forecast_hour,
        :member,
        :variable_name,
        :provider_available_at_utc,
        :acquired_at_utc,
        :effective_available_at_utc,
        :availability_method,
        :source_lag_seconds,
        true
    )
    ON CONFLICT DO NOTHING
    """
)


def _value_params(
    *,
    value,
    source_request_id: str,
    source_record_id: UUID,
    request_sha256: str,
    ingested_at_utc: datetime,
) -> dict[str, Any]:
    return {
        **value.__dict__,
        "raw_row_json": _json_dumps(value.raw_row_json),
        "our_ingested_at_utc": ingested_at_utc,
        "source_request_id": source_request_id,
        "source_record_id": source_record_id,
        "request_sha256": request_sha256,
    }


def _availability_params(*, value, source_record_id: UUID, ingested_at_utc: datetime) -> dict[str, Any]:
    canonical_record_key = (
        f"gribstream:{value.model_id}:{value.cutoff_id}:{value.target_date.isoformat()}:"
        f"{value.grid_point_id}:{value.forecasted_at_utc.isoformat()}:{value.forecasted_time_utc.isoformat()}"
    )
    return {
        "source_record_id": source_record_id,
        "source_name": SOURCE_NAME,
        "provider_name": PROVIDER_NAME,
        "canonical_record_key": canonical_record_key,
        "station_id": value.grid_point_id,
        "model_name": value.model_id,
        "run_time_utc": value.forecasted_at_utc,
        "valid_time_utc": value.forecasted_time_utc,
        "forecast_hour": int(round(value.forecast_hour)),
        "member": value.member,
        "variable_name": value.variable_name,
        "provider_available_at_utc": value.provider_available_at_utc,
        "acquired_at_utc": ingested_at_utc,
        "effective_available_at_utc": value.effective_available_at_utc,
        "availability_method": value.availability_method,
        "source_lag_seconds": int((value.effective_available_at_utc - value.forecasted_at_utc).total_seconds()),
    }


def _bulk_upsert_values(
    connection: Connection,
    *,
    values: tuple[Any, ...],
    source_request_id: str,
    source_record_id: UUID,
    request_sha256: str,
    ingested_at_utc: datetime,
) -> int:
    if not values:
        return 0
    params = [
        _value_params(
            value=value,
            source_request_id=source_request_id,
            source_record_id=source_record_id,
            request_sha256=request_sha256,
            ingested_at_utc=ingested_at_utc,
        )
        for value in values
    ]
    raw_connection = connection.connection.driver_connection
    connection.execute(VALUE_COPY_STAGE_DDL)
    connection.execute(text("TRUNCATE grib_forecast_values_stage"))
    copy_columns = ", ".join(VALUE_COLUMNS)
    with raw_connection.cursor() as cursor:
        with cursor.copy(f"COPY grib_forecast_values_stage ({copy_columns}) FROM STDIN") as copy:
            for param in params:
                copy.write_row([param[column] for column in VALUE_COLUMNS])
    connection.execute(VALUE_COPY_UPSERT_SQL)
    return len(params)


def _bulk_insert_availability(
    connection: Connection,
    *,
    values: tuple[Any, ...],
    source_record_id: UUID,
    ingested_at_utc: datetime,
) -> int:
    if not values:
        return 0
    params = [
        _availability_params(value=value, source_record_id=source_record_id, ingested_at_utc=ingested_at_utc)
        for value in values
    ]
    connection.execute(AVAILABILITY_INSERT_SQL, params)
    return len(params)


def _insert_availability_from_silver(
    connection: Connection,
    *,
    source_request_id: str,
    source_record_id: UUID,
    ingested_at_utc: datetime,
) -> int:
    candidate_count = connection.execute(
        text(
            """
            SELECT count(*)
            FROM silver.grib_forecast_values
            WHERE source_request_id = :source_request_id
              AND source_record_id = :source_record_id
            """
        ),
        {
            "source_request_id": source_request_id,
            "source_record_id": source_record_id,
        },
    ).scalar_one()
    connection.execute(
        text(
            """
            INSERT INTO silver.availability_ledger (
                source_record_id,
                source_name,
                provider_name,
                canonical_record_key,
                station_id,
                model_name,
                run_time_utc,
                valid_time_utc,
                forecast_hour,
                member,
                variable_name,
                provider_available_at_utc,
                acquired_at_utc,
                effective_available_at_utc,
                availability_method,
                source_lag_seconds,
                is_revision_current
            )
            SELECT
                source_record_id,
                :source_name,
                :provider_name,
                'gribstream:' || model_id || ':' || cutoff_id || ':' || target_date::text || ':' ||
                    grid_point_id || ':' || forecasted_at_utc::text || ':' || forecasted_time_utc::text,
                grid_point_id,
                model_id,
                forecasted_at_utc,
                forecasted_time_utc,
                round(forecast_hour)::int,
                member,
                variable_name,
                provider_available_at_utc,
                :acquired_at_utc,
                effective_available_at_utc,
                availability_method,
                EXTRACT(EPOCH FROM (effective_available_at_utc - forecasted_at_utc))::int,
                true
            FROM silver.grib_forecast_values
            WHERE source_request_id = :source_request_id
              AND source_record_id = :source_record_id
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "source_name": SOURCE_NAME,
            "provider_name": PROVIDER_NAME,
            "acquired_at_utc": ingested_at_utc,
            "source_request_id": source_request_id,
            "source_record_id": source_record_id,
        },
    )
    return int(candidate_count)


def _insert_availability_from_value_stage(
    connection: Connection,
    *,
    ingested_at_utc: datetime,
) -> int:
    candidate_count = connection.execute(text("SELECT count(*) FROM grib_forecast_values_stage")).scalar_one()
    connection.execute(
        text(
            """
            INSERT INTO silver.availability_ledger (
                source_record_id,
                source_name,
                provider_name,
                canonical_record_key,
                station_id,
                model_name,
                run_time_utc,
                valid_time_utc,
                forecast_hour,
                member,
                variable_name,
                provider_available_at_utc,
                acquired_at_utc,
                effective_available_at_utc,
                availability_method,
                source_lag_seconds,
                is_revision_current
            )
            SELECT
                source_record_id,
                :source_name,
                :provider_name,
                'gribstream:' || model_id || ':' || cutoff_id || ':' || target_date::text || ':' ||
                    grid_point_id || ':' || forecasted_at_utc::text || ':' || forecasted_time_utc::text,
                grid_point_id,
                model_id,
                forecasted_at_utc,
                forecasted_time_utc,
                round(forecast_hour)::int,
                member,
                variable_name,
                provider_available_at_utc,
                :acquired_at_utc,
                effective_available_at_utc,
                availability_method,
                EXTRACT(EPOCH FROM (effective_available_at_utc - forecasted_at_utc))::int,
                true
            FROM grib_forecast_values_stage
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "source_name": SOURCE_NAME,
            "provider_name": PROVIDER_NAME,
            "acquired_at_utc": ingested_at_utc,
        },
    )
    return int(candidate_count)


def _replace_availability_row(
    connection: Connection,
    *,
    value,
    source_record_id: UUID,
    ingested_at_utc: datetime,
) -> None:
    canonical_record_key = (
        f"gribstream:{value.model_id}:{value.cutoff_id}:{value.target_date.isoformat()}:"
        f"{value.grid_point_id}:{value.forecasted_at_utc.isoformat()}:{value.forecasted_time_utc.isoformat()}"
    )
    connection.execute(
        text(
            """
            DELETE FROM silver.availability_ledger
            WHERE source_name = :source_name
              AND provider_name = :provider_name
              AND canonical_record_key = :canonical_record_key
              AND variable_name = :variable_name
              AND COALESCE(member, '') = COALESCE(:member, '')
              AND COALESCE(model_name, '') = COALESCE(:model_name, '')
              AND COALESCE(station_id, '') = COALESCE(:station_id, '')
              AND COALESCE(run_time_utc, '1900-01-01'::timestamptz) = COALESCE(:run_time_utc, '1900-01-01'::timestamptz)
              AND COALESCE(valid_time_utc, '1900-01-01'::timestamptz) = COALESCE(:valid_time_utc, '1900-01-01'::timestamptz)
            """
        ),
        {
            "source_name": SOURCE_NAME,
            "provider_name": PROVIDER_NAME,
            "canonical_record_key": canonical_record_key,
            "variable_name": value.variable_name,
            "member": value.member,
            "model_name": value.model_id,
            "station_id": value.grid_point_id,
            "run_time_utc": value.forecasted_at_utc,
            "valid_time_utc": value.forecasted_time_utc,
        },
    )
    connection.execute(
        text(
            """
            INSERT INTO silver.availability_ledger (
                source_record_id,
                source_name,
                provider_name,
                canonical_record_key,
                station_id,
                model_name,
                run_time_utc,
                valid_time_utc,
                forecast_hour,
                member,
                variable_name,
                provider_available_at_utc,
                acquired_at_utc,
                effective_available_at_utc,
                availability_method,
                source_lag_seconds,
                is_revision_current
            )
            VALUES (
                :source_record_id,
                :source_name,
                :provider_name,
                :canonical_record_key,
                :station_id,
                :model_name,
                :run_time_utc,
                :valid_time_utc,
                :forecast_hour,
                :member,
                :variable_name,
                :provider_available_at_utc,
                :acquired_at_utc,
                :effective_available_at_utc,
                :availability_method,
                :source_lag_seconds,
                true
            )
            """
        ),
        {
            "source_record_id": source_record_id,
            "source_name": SOURCE_NAME,
            "provider_name": PROVIDER_NAME,
            "canonical_record_key": canonical_record_key,
            "station_id": value.grid_point_id,
            "model_name": value.model_id,
            "run_time_utc": value.forecasted_at_utc,
            "valid_time_utc": value.forecasted_time_utc,
            "forecast_hour": int(round(value.forecast_hour)),
            "member": value.member,
            "variable_name": value.variable_name,
            "provider_available_at_utc": value.provider_available_at_utc,
            "acquired_at_utc": ingested_at_utc,
            "effective_available_at_utc": value.effective_available_at_utc,
            "availability_method": value.availability_method,
            "source_lag_seconds": int((value.effective_available_at_utc - value.forecasted_at_utc).total_seconds()),
        },
    )


def upsert_source_gaps(connection: Connection, gaps: tuple[dict[str, Any], ...]) -> int:
    inserted = 0
    for gap in gaps:
        params = {
            "model_id": gap.get("model_id"),
            "target_start_date": gap.get("target_start_date"),
            "target_end_date": gap.get("target_end_date"),
            "cutoff_id": gap.get("cutoff_id"),
            "grid_point_id": gap.get("grid_point_id"),
            "variable_alias": gap.get("variable_alias"),
            "variable_name": gap.get("variable_name"),
            "member": gap.get("member"),
            "gap_type": gap.get("gap_type", "unknown"),
            "gap_reason": gap.get("gap_reason", ""),
            "evidence_json": _json_dumps(gap),
        }
        connection.execute(
            text(
                """
                DELETE FROM audit.gribstream_source_gaps
                WHERE model_id = :model_id
                  AND COALESCE(target_start_date, '1900-01-01'::date) = COALESCE(:target_start_date, '1900-01-01'::date)
                  AND COALESCE(target_end_date, '1900-01-01'::date) = COALESCE(:target_end_date, '1900-01-01'::date)
                  AND COALESCE(cutoff_id, '') = COALESCE(:cutoff_id, '')
                  AND COALESCE(grid_point_id, '') = COALESCE(:grid_point_id, '')
                  AND COALESCE(variable_alias, '') = COALESCE(:variable_alias, '')
                  AND COALESCE(member, '') = COALESCE(:member, '')
                  AND gap_type = :gap_type
                """
            ),
            params,
        )
        result = connection.execute(
            text(
                """
                INSERT INTO audit.gribstream_source_gaps (
                    model_id,
                    target_start_date,
                    target_end_date,
                    cutoff_id,
                    grid_point_id,
                    variable_alias,
                    variable_name,
                    member,
                    gap_type,
                    gap_reason,
                    evidence_json,
                    first_detected_at_utc,
                    last_detected_at_utc
                )
                VALUES (
                    :model_id,
                    :target_start_date,
                    :target_end_date,
                    :cutoff_id,
                    :grid_point_id,
                    :variable_alias,
                    :variable_name,
                    :member,
                    :gap_type,
                    :gap_reason,
                    CAST(:evidence_json AS jsonb),
                    now(),
                    now()
                )
                """
            ),
            params,
        )
        inserted += result.rowcount or 0
    return inserted


def mark_chunk_terminal(
    connection: Connection,
    *,
    chunk_id: str,
    status: str,
    response: GribStreamRawResponse | None = None,
    rows_upserted: int = 0,
    availability_rows: int = 0,
    gaps: int = 0,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    connection.execute(
        text(
            """
            UPDATE audit.gribstream_backfill_chunks
            SET
                status = :status,
                http_status = :http_status,
                error_type = :error_type,
                error_message = :error_message,
                source_request_id = :source_request_id,
                source_record_id = :source_record_id,
                raw_storage_uri = :raw_storage_uri,
                rows_upserted = :rows_upserted,
                availability_rows_upserted = :availability_rows_upserted,
                gaps_upserted = :gaps_upserted,
                finished_at_utc = now(),
                updated_at = now()
            WHERE chunk_id = :chunk_id
            """
        ),
        {
            "chunk_id": chunk_id,
            "status": status,
            "http_status": response.http_status if response else None,
            "error_type": error_type or (response.error_type if response else None),
            "error_message": error_message or (response.error_message if response else None),
            "source_request_id": None,
            "source_record_id": None,
            "raw_storage_uri": response.raw_storage_uri if response else None,
            "rows_upserted": rows_upserted,
            "availability_rows_upserted": availability_rows,
            "gaps_upserted": gaps,
        },
    )


def _target_instance_id(connection: Connection, *, target_date, cutoff_id: str) -> UUID:
    row = connection.execute(
        text(
            """
            SELECT target_instance_id
            FROM gold.target_instances
            WHERE target_date = :target_date
              AND cutoff_id = :cutoff_id
            """
        ),
        {"target_date": target_date, "cutoff_id": cutoff_id},
    ).mappings().first()
    if row is None:
        materialize_target_instances(connection, start_date=target_date, end_date=target_date, replace=False)
        row = connection.execute(
            text(
                """
                SELECT target_instance_id
                FROM gold.target_instances
                WHERE target_date = :target_date
                  AND cutoff_id = :cutoff_id
                """
            ),
            {"target_date": target_date, "cutoff_id": cutoff_id},
        ).mappings().first()
    if row is None:
        raise RuntimeError(f"missing gold.target_instances row for {target_date} {cutoff_id}")
    return row["target_instance_id"]


def _upsert_thin_feature_version(connection: Connection, feature_names: list[str]) -> UUID:
    row = connection.execute(
        text(
            """
            INSERT INTO registry.feature_versions (
                feature_set_name,
                feature_version,
                source_code_git_sha,
                formula_contract_hash,
                feature_names
            )
            VALUES (
                :feature_set_name,
                :feature_version,
                :source_code_git_sha,
                :formula_contract_hash,
                :feature_names
            )
            ON CONFLICT (feature_set_name, feature_version) DO UPDATE SET
                source_code_git_sha = EXCLUDED.source_code_git_sha,
                formula_contract_hash = EXCLUDED.formula_contract_hash,
                feature_names = (
                    SELECT ARRAY(
                        SELECT DISTINCT unnest(registry.feature_versions.feature_names || EXCLUDED.feature_names)
                        ORDER BY 1
                    )
                )
            RETURNING feature_version_id
            """
        ),
        {
            "feature_set_name": THIN_FEATURE_SET_NAME,
            "feature_version": FEATURE_BUILD_VERSION,
            "source_code_git_sha": current_git_sha(),
            "formula_contract_hash": THIN_FORMULA_CONTRACT_HASH,
            "feature_names": sorted(set(feature_names)),
        },
    ).mappings().one()
    return row["feature_version_id"]


def _bulk_upsert_gold_features(
    connection: Connection,
    *,
    features: tuple[GribStreamGoldFeature, ...],
    source_request_id: str,
    source_record_id: UUID,
    response: GribStreamRawResponse,
) -> int:
    if not features:
        return 0
    rows = []
    for feature in features:
        trace = {
            **feature.source_trace_json,
            "source_request_id": source_request_id,
            "source_record_id": str(source_record_id),
            "request_sha256": response.chunk.request_sha256,
            "raw_storage_uri": response.raw_storage_uri,
            "endpoint_type": response.chunk.endpoint_type,
            "feature_profile": response.chunk.feature_profile,
            "persistence_mode": response.chunk.persistence_mode,
            "model_id": response.chunk.model_id,
        }
        rows.append(
            {
                "target_instance_id": _target_instance_id(
                    connection,
                    target_date=feature.target_date,
                    cutoff_id=feature.cutoff_id,
                ),
                "feature_family": feature.feature_family,
                "feature_name": feature.feature_name,
                "feature_value": feature.feature_value,
                "feature_unit": feature.feature_unit,
                "feature_available": feature.feature_available,
                "source_latest_valid_time_utc": feature.source_latest_valid_time_utc,
                "source_latest_run_time_utc": feature.source_latest_run_time_utc,
                "source_age_hours": feature.source_age_hours,
                "source_latency_minutes": feature.source_latency_minutes,
                "feature_build_version": feature.feature_build_version,
                "max_source_available_at_utc": feature.max_source_available_at_utc,
                "source_trace_json": _json_dumps(trace),
            }
        )
    connection.execute(
        text(
            """
            INSERT INTO gold.feature_values (
                target_instance_id,
                feature_family,
                feature_name,
                feature_value,
                feature_unit,
                feature_available,
                source_latest_valid_time_utc,
                source_latest_run_time_utc,
                source_age_hours,
                source_latency_minutes,
                feature_build_version,
                max_source_available_at_utc,
                source_trace_json
            )
            VALUES (
                :target_instance_id,
                :feature_family,
                :feature_name,
                :feature_value,
                :feature_unit,
                :feature_available,
                :source_latest_valid_time_utc,
                :source_latest_run_time_utc,
                :source_age_hours,
                :source_latency_minutes,
                :feature_build_version,
                :max_source_available_at_utc,
                CAST(:source_trace_json AS jsonb)
            )
            ON CONFLICT (target_instance_id, feature_name, feature_build_version)
            DO UPDATE SET
                feature_family = EXCLUDED.feature_family,
                feature_value = EXCLUDED.feature_value,
                feature_unit = EXCLUDED.feature_unit,
                feature_available = EXCLUDED.feature_available,
                source_latest_valid_time_utc = EXCLUDED.source_latest_valid_time_utc,
                source_latest_run_time_utc = EXCLUDED.source_latest_run_time_utc,
                source_age_hours = EXCLUDED.source_age_hours,
                source_latency_minutes = EXCLUDED.source_latency_minutes,
                max_source_available_at_utc = EXCLUDED.max_source_available_at_utc,
                source_trace_json = EXCLUDED.source_trace_json
            """
        ),
        rows,
    )
    return len(rows)


def _upsert_feature_matrix_for_features(
    connection: Connection,
    *,
    features: tuple[GribStreamGoldFeature, ...],
) -> int:
    if not features:
        return 0
    feature_version_id = _upsert_thin_feature_version(
        connection,
        sorted({feature.feature_name for feature in features}),
    )
    target_ids = sorted(
        {
            _target_instance_id(connection, target_date=feature.target_date, cutoff_id=feature.cutoff_id)
            for feature in features
        },
        key=str,
    )
    upserted = 0
    for target_instance_id in target_ids:
        target = connection.execute(
            text(
                """
                SELECT settlement_high_f_whole, label_available, label_revision_sensitive
                FROM gold.target_instances
                WHERE target_instance_id = :target_instance_id
                """
            ),
            {"target_instance_id": target_instance_id},
        ).mappings().one()
        feature_rows = connection.execute(
            text(
                """
                SELECT feature_name, feature_value, feature_available
                FROM gold.feature_values
                WHERE target_instance_id = :target_instance_id
                  AND feature_build_version = :feature_build_version
                """
            ),
            {
                "target_instance_id": target_instance_id,
                "feature_build_version": FEATURE_BUILD_VERSION,
            },
        ).mappings().all()
        vector = {row["feature_name"]: row["feature_value"] for row in feature_rows}
        availability = {row["feature_name"]: bool(row["feature_available"]) for row in feature_rows}
        result = connection.execute(
            text(
                """
                INSERT INTO gold.feature_matrix (
                    target_instance_id,
                    feature_version_id,
                    feature_vector_json,
                    feature_availability_json,
                    label_high_temp_f,
                    label_available,
                    label_revision_sensitive
                )
                VALUES (
                    :target_instance_id,
                    :feature_version_id,
                    CAST(:feature_vector_json AS jsonb),
                    CAST(:feature_availability_json AS jsonb),
                    :label_high_temp_f,
                    :label_available,
                    :label_revision_sensitive
                )
                ON CONFLICT (target_instance_id, feature_version_id)
                DO UPDATE SET
                    feature_vector_json = EXCLUDED.feature_vector_json,
                    feature_availability_json = EXCLUDED.feature_availability_json,
                    label_high_temp_f = EXCLUDED.label_high_temp_f,
                    label_available = EXCLUDED.label_available,
                    label_revision_sensitive = EXCLUDED.label_revision_sensitive
                """
            ),
            {
                "target_instance_id": target_instance_id,
                "feature_version_id": feature_version_id,
                "feature_vector_json": _json_dumps(vector),
                "feature_availability_json": _json_dumps(availability),
                "label_high_temp_f": target["settlement_high_f_whole"],
                "label_available": target["label_available"],
                "label_revision_sensitive": target["label_revision_sensitive"],
            },
        )
        upserted += result.rowcount or 0
    return upserted


def persist_gribstream_response(
    connection: Connection,
    *,
    job_id: str,
    response: GribStreamRawResponse,
    parsed: ParsedGribStreamResponse,
    persistence_mode: str | None = None,
) -> PersistedGribStreamChunk:
    source_request_id = _source_request_id(response)
    _insert_source_request(connection, response, source_request_id)
    source_record_id = _insert_or_reuse_source_record(connection, response, source_request_id)
    ingested_at_utc = datetime.now(timezone.utc)
    mode = persistence_mode or response.chunk.persistence_mode
    if mode == "gold_only":
        gold_features = parsed.gold_features or build_tmax_thin_gold_features(parsed.values)
        rows = _bulk_upsert_gold_features(
            connection,
            features=gold_features,
            source_request_id=source_request_id,
            source_record_id=source_record_id,
            response=response,
        )
        _upsert_feature_matrix_for_features(connection, features=gold_features)
        availability_rows = _bulk_insert_availability(
            connection,
            values=parsed.values,
            source_record_id=source_record_id,
            ingested_at_utc=ingested_at_utc,
        )
    else:
        rows = _bulk_upsert_values(
            connection,
            values=parsed.values,
            source_request_id=source_request_id,
            source_record_id=source_record_id,
            request_sha256=response.chunk.request_sha256,
            ingested_at_utc=ingested_at_utc,
        )
        availability_rows = _insert_availability_from_value_stage(connection, ingested_at_utc=ingested_at_utc)
    gaps = upsert_source_gaps(connection, parsed.gaps)
    status = "completed" if rows else "completed_empty"
    connection.execute(
        text(
            """
            UPDATE audit.gribstream_backfill_chunks
            SET
                status = :status,
                http_status = :http_status,
                source_request_id = :source_request_id,
                source_record_id = :source_record_id,
                raw_storage_uri = :raw_storage_uri,
                rows_upserted = :rows_upserted,
                availability_rows_upserted = :availability_rows_upserted,
                gaps_upserted = :gaps_upserted,
                finished_at_utc = now(),
                updated_at = now()
            WHERE job_id = :job_id
              AND request_sha256 = :request_sha256
            """
        ),
        {
            "job_id": job_id,
            "status": status,
            "http_status": response.http_status,
            "source_request_id": source_request_id,
            "source_record_id": source_record_id,
            "raw_storage_uri": response.raw_storage_uri,
            "rows_upserted": rows,
            "availability_rows_upserted": availability_rows,
            "gaps_upserted": gaps,
            "request_sha256": response.chunk.request_sha256,
        },
    )
    return PersistedGribStreamChunk(
        chunk_id=chunk_row_id(job_id, response.chunk),
        source_request_id=source_request_id,
        source_record_id=source_record_id,
        status=status,
        rows_upserted=rows,
        availability_rows_upserted=availability_rows,
        gaps_upserted=gaps,
        http_status=response.http_status,
        error_type=response.error_type,
        error_message=response.error_message,
    )


def refresh_job_status(connection: Connection, job_id: str) -> dict[str, int | str]:
    rows = connection.execute(
        text(
            """
            SELECT status, count(*) AS count, COALESCE(sum(rows_upserted), 0) AS rows_upserted
            FROM audit.gribstream_backfill_chunks
            WHERE job_id = :job_id
            GROUP BY status
            """
        ),
        {"job_id": job_id},
    ).mappings().all()
    counts = {row["status"]: int(row["count"]) for row in rows}
    row_total = sum(int(row["rows_upserted"]) for row in rows)
    completed = counts.get("completed", 0) + counts.get("completed_empty", 0) + counts.get("skipped", 0)
    failed = counts.get("failed", 0) + counts.get("rate_limited", 0) + counts.get("auth_failed", 0) + counts.get("selector_missing", 0)
    total = sum(counts.values())
    if failed:
        status = "blocked" if counts.get("rate_limited", 0) or counts.get("auth_failed", 0) else "failed"
    elif total and completed == total:
        status = "completed"
    else:
        status = "running"
    connection.execute(
        text(
            """
            UPDATE audit.gribstream_backfill_jobs
            SET
                status = :status,
                completed_chunks = :completed_chunks,
                failed_chunks = :failed_chunks,
                row_counts_json = CAST(:row_counts_json AS jsonb),
                finished_at_utc = CASE WHEN :status IN ('completed','failed','blocked') THEN now() ELSE finished_at_utc END,
                updated_at = now()
            WHERE job_id = :job_id
            """
        ),
        {
            "job_id": job_id,
            "status": status,
            "completed_chunks": completed,
            "failed_chunks": failed,
            "row_counts_json": _json_dumps({"rows_upserted": row_total, **counts}),
        },
    )
    return {"job_id": job_id, "status": status, "chunks_total": total, "chunks_completed": completed, "chunks_failed": failed, "rows_upserted": row_total, **counts}


def job_status(connection: Connection, job_id: str | None = None) -> list[dict[str, Any]]:
    where = "WHERE job_id = :job_id" if job_id else ""
    rows = connection.execute(
        text(
            f"""
            SELECT
                job_id,
                cutoff_id,
                start_date,
                end_date,
                coordinate_tier,
                status,
                planned_chunks,
                completed_chunks,
                failed_chunks,
                estimated_credits,
                row_counts_json,
                started_at_utc,
                finished_at_utc
            FROM audit.gribstream_backfill_jobs
            {where}
            ORDER BY started_at_utc DESC
            LIMIT 20
            """
        ),
        {"job_id": job_id} if job_id else {},
    ).mappings().all()
    return [dict(row) for row in rows]


def model_status(connection: Connection, job_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        text(
            """
            SELECT
                model_id,
                min(target_start_date) AS from_date,
                max(target_end_date) AS through_date,
                count(*) AS chunks,
                count(*) FILTER (WHERE status IN ('completed','completed_empty','skipped')) AS completed,
                count(*) FILTER (WHERE status IN ('planned','running','failed')) AS remaining,
                count(*) FILTER (WHERE status IN ('rate_limited','auth_failed','selector_missing')) AS blocked,
                COALESCE(sum(estimated_credits), 0) AS estimated_credits,
                COALESCE(sum(rows_upserted), 0) AS rows_upserted
            FROM audit.gribstream_backfill_chunks
            WHERE job_id = :job_id
            GROUP BY model_id
            ORDER BY model_id
            """
        ),
        {"job_id": job_id},
    ).mappings().all()
    return [dict(row) for row in rows]
