"""PostgreSQL persistence for GribStream raw and normalized rows."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hkg_tmax.gribstream.catalog import ResolvedSelector
from hkg_tmax.gribstream.client import ResponseManifest, sanitize_text
from hkg_tmax.gribstream.normalizer import NormalizedPoint, RejectedRow
from hkg_tmax_db.connection import import_psycopg


POINT_VALUE_UPSERT_SQL = """
    INSERT INTO nwp_core.point_value (
        model_run_id,
        valid_time_utc,
        lead_minutes,
        location_id,
        selector_id,
        member_number,
        value,
        response_object_id,
        quality_status
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'raw_valid')
    ON CONFLICT (model_run_id, valid_time_utc, location_id, selector_id, member_number)
    DO UPDATE SET
        value = EXCLUDED.value,
        response_object_id = EXCLUDED.response_object_id,
        quality_status = 'raw_valid'
"""


@dataclass(frozen=True)
class IngestSummary:
    request_id: str
    response_object_id: int
    model_id: int
    selector_id: int
    inserted_or_updated_points: int
    rejected_rows: int


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def register_request_started(
    database_url: str,
    *,
    provider: str,
    model_code: str,
    endpoint: str,
    canonical_payload: dict[str, Any],
    request_hash: str,
) -> str:
    psycopg = import_psycopg()
    request_id = str(uuid.uuid4())
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO raw_audit.acquisition_request (
                    request_id,
                    provider,
                    model_code,
                    endpoint,
                    canonical_request_json,
                    request_sha256,
                    status,
                    attempt_count,
                    started_at_utc
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'running', 1, now())
                ON CONFLICT (request_sha256)
                DO UPDATE SET
                    status = CASE
                        WHEN raw_audit.acquisition_request.status = 'completed'
                        THEN raw_audit.acquisition_request.status
                        ELSE 'running'
                    END,
                    attempt_count = raw_audit.acquisition_request.attempt_count + 1,
                    started_at_utc = COALESCE(raw_audit.acquisition_request.started_at_utc, EXCLUDED.started_at_utc),
                    error_class = NULL,
                    error_message = NULL
                RETURNING request_id
                """,
                (request_id, provider, model_code, endpoint, _jsonb(canonical_payload), request_hash),
            )
            stored_request_id = str(cursor.fetchone()[0])
        connection.commit()
    return stored_request_id


def mark_request_failed(
    database_url: str,
    *,
    request_hash: str,
    error_class: str,
    error_message: str,
) -> None:
    psycopg = import_psycopg()
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE raw_audit.acquisition_request
                SET status = 'failed',
                    completed_at_utc = now(),
                    error_class = %s,
                    error_message = %s
                WHERE request_sha256 = %s
                """,
                (error_class, sanitize_text(error_message), request_hash),
            )
        connection.commit()


def load_location_ids(database_url: str) -> dict[str, int]:
    psycopg = import_psycopg()
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT location_code, location_id FROM catalog.location")
            rows = cursor.fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def _ensure_model(
    cursor: Any,
    selector: ResolvedSelector,
    *,
    provider: str = "NOAA",
    domain: str = "Global",
    model_type: str = "forecast",
    native_resolution: str = "catalog-resolved",
    archive_start: str = "2021-03-22",
    disposition: str = "acquisition",
    catalog_snapshot_sha256: str | None = None,
) -> int:
    model_catalog_sha256 = catalog_snapshot_sha256 or selector.source_sha256
    cursor.execute(
        """
        INSERT INTO catalog.weather_model (
            provider,
            model_code,
            domain,
            model_type,
            native_resolution,
            archive_start,
            disposition,
            catalog_snapshot_sha256
        )
        VALUES (%s, %s, %s, %s, %s, %s::date, %s, %s)
        ON CONFLICT (provider, model_code, catalog_snapshot_sha256) DO NOTHING
        """,
        (
            provider,
            selector.dataset,
            domain,
            model_type,
            native_resolution,
            archive_start,
            disposition,
            model_catalog_sha256,
        ),
    )
    cursor.execute(
        """
        SELECT model_id
        FROM catalog.weather_model
        WHERE provider = %s AND model_code = %s AND catalog_snapshot_sha256 = %s
        """,
        (provider, selector.dataset, model_catalog_sha256),
    )
    return int(cursor.fetchone()[0])


def _ensure_variable(cursor: Any, selector: ResolvedSelector) -> int:
    cursor.execute(
        """
        INSERT INTO catalog.variable (semantic_variable, semantic_family, canonical_unit, value_role)
        VALUES (%s, %s, %s, 'forecast')
        ON CONFLICT (semantic_family, semantic_variable, value_role) DO NOTHING
        """,
        (selector.semantic_variable, selector.semantic_family, selector.native_unit),
    )
    cursor.execute(
        """
        SELECT variable_id
        FROM catalog.variable
        WHERE semantic_family = %s AND semantic_variable = %s AND value_role = 'forecast'
        """,
        (selector.semantic_family, selector.semantic_variable),
    )
    return int(cursor.fetchone()[0])


def _ensure_selector(cursor: Any, selector: ResolvedSelector, model_id: int, variable_id: int) -> int:
    cursor.execute(
        """
        SELECT selector_id
        FROM catalog.variable_selector_snapshot
        WHERE model_id = %s
          AND semantic_variable = %s
          AND native_name = %s
          AND native_level = %s
          AND native_info = %s
          AND source_sha256 = %s
        ORDER BY retrieved_at_utc ASC
        LIMIT 1
        """,
        (
            model_id,
            selector.semantic_variable,
            selector.native_name,
            selector.native_level,
            selector.native_info,
            selector.source_sha256,
        ),
    )
    row = cursor.fetchone()
    if row is not None:
        return int(row[0])
    cursor.execute(
        """
        INSERT INTO catalog.variable_selector_snapshot (
            model_id,
            variable_id,
            semantic_variable,
            native_name,
            native_level,
            native_info,
            native_unit,
            retrieved_at_utc,
            source_sha256
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING selector_id
        """,
        (
            model_id,
            variable_id,
            selector.semantic_variable,
            selector.native_name,
            selector.native_level,
            selector.native_info,
            selector.native_unit,
            selector.retrieved_at_utc,
            selector.source_sha256,
        ),
    )
    return int(cursor.fetchone()[0])


def _ensure_response_object(cursor: Any, request_id: str, manifest: ResponseManifest) -> int:
    cursor.execute(
        """
        INSERT INTO raw_audit.response_object (
            request_id,
            object_uri,
            byte_size,
            sha256,
            content_type,
            retrieved_at_utc,
            first_seen_at_utc,
            row_count
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (request_id, sha256) DO NOTHING
        RETURNING response_object_id
        """,
        (
            request_id,
            manifest.object_path.as_posix(),
            manifest.byte_size,
            manifest.sha256,
            manifest.content_type,
            manifest.retrieved_at_utc,
            manifest.retrieved_at_utc,
            manifest.row_count,
        ),
    )
    row = cursor.fetchone()
    if row is not None:
        return int(row[0])
    cursor.execute(
        """
        SELECT response_object_id
        FROM raw_audit.response_object
        WHERE request_id = %s AND sha256 = %s
        """,
        (request_id, manifest.sha256),
    )
    return int(cursor.fetchone()[0])


def _ensure_model_run(cursor: Any, model_id: int, run_time_utc: datetime, first_seen_at_utc: str) -> int:
    cursor.execute(
        """
        INSERT INTO nwp_core.model_run (
            model_id,
            run_time_utc,
            first_seen_at_utc,
            availability_grade,
            availability_contract_version,
            model_version
        )
        VALUES (%s, %s, %s, 'C_RUN_TIME_ONLY', 'hkg_t24_1500hkt_v1', '')
        ON CONFLICT (model_id, run_time_utc, model_version)
        DO UPDATE SET first_seen_at_utc = COALESCE(nwp_core.model_run.first_seen_at_utc, EXCLUDED.first_seen_at_utc)
        RETURNING model_run_id
        """,
        (model_id, run_time_utc, first_seen_at_utc),
    )
    return int(cursor.fetchone()[0])


def ingest_response(
    database_url: str,
    *,
    request_id: str,
    selector: ResolvedSelector,
    manifest: ResponseManifest,
    points: tuple[NormalizedPoint, ...],
    rejected_rows: tuple[RejectedRow, ...],
    provider: str = "NOAA",
    domain: str = "Global",
    model_type: str = "forecast",
    native_resolution: str = "catalog-resolved",
    archive_start: str = "2021-03-22",
    disposition: str = "acquisition",
    catalog_snapshot_sha256: str | None = None,
    point_batch_size: int = 5000,
) -> IngestSummary:
    psycopg = import_psycopg()
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            model_id = _ensure_model(
                cursor,
                selector,
                provider=provider,
                domain=domain,
                model_type=model_type,
                native_resolution=native_resolution,
                archive_start=archive_start,
                disposition=disposition,
                catalog_snapshot_sha256=catalog_snapshot_sha256,
            )
            variable_id = _ensure_variable(cursor, selector)
            selector_id = _ensure_selector(cursor, selector, model_id, variable_id)
            response_object_id = _ensure_response_object(cursor, request_id, manifest)
            model_run_ids: dict[datetime, int] = {}
            point_params: list[tuple[Any, ...]] = []
            for point in points:
                model_run_id = model_run_ids.get(point.run_time_utc)
                if model_run_id is None:
                    model_run_id = _ensure_model_run(
                        cursor,
                        model_id,
                        point.run_time_utc,
                        manifest.retrieved_at_utc,
                    )
                    model_run_ids[point.run_time_utc] = model_run_id
                point_params.append(
                    (
                        model_run_id,
                        point.valid_time_utc,
                        point.lead_minutes,
                        point.location_id,
                        selector_id,
                        point.member_number,
                        point.value,
                        response_object_id,
                    ),
                )
                if len(point_params) >= point_batch_size:
                    cursor.executemany(POINT_VALUE_UPSERT_SQL, point_params)
                    point_params.clear()
            if point_params:
                cursor.executemany(POINT_VALUE_UPSERT_SQL, point_params)
            for row in rejected_rows:
                cursor.execute(
                    """
                    INSERT INTO quarantine.rejected_payload (
                        request_id,
                        response_object_id,
                        rejection_class,
                        rejection_reason,
                        evidence_json
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        request_id,
                        response_object_id,
                        row.rejection_class,
                        row.rejection_reason,
                        _jsonb(row.evidence | {"row_number": row.row_number}),
                    ),
                )
            cursor.execute(
                """
                UPDATE raw_audit.acquisition_request
                SET status = %s,
                    completed_at_utc = now(),
                    error_class = NULL,
                    error_message = NULL
                WHERE request_id = %s
                """,
                ("completed_empty" if manifest.row_count == 0 else "completed", request_id),
            )
        connection.commit()
    return IngestSummary(
        request_id=request_id,
        response_object_id=response_object_id,
        model_id=model_id,
        selector_id=selector_id,
        inserted_or_updated_points=len(points),
        rejected_rows=len(rejected_rows),
    )
