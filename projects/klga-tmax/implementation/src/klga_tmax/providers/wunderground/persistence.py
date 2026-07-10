from __future__ import annotations

from datetime import date, datetime, timezone
import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.ingestion.bronze import CurrentBronzeRecord, decide_bronze_revision
from klga_tmax.ingestion.hash_keys import canonical_json, payload_hash, sha256_hex
from klga_tmax.providers.wunderground.client import PARSER_VERSION
from klga_tmax.providers.wunderground.models import (
    ParsedWundergroundResponse,
    PersistedWundergroundWindow,
    WundergroundDailyActual,
    WundergroundIntradayObservation,
    WundergroundRawDayResponse,
)
from klga_tmax.registry.materialize_targets import iter_dates

SOURCE_NAME = "wunderground"
PROVIDER_NAME = "weathercom"
ENDPOINT_NAME = "historical_observations"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _source_request_id(response: WundergroundRawDayResponse) -> str:
    identity = {
        "source": SOURCE_NAME,
        "provider": PROVIDER_NAME,
        "station_id": response.station_id,
        "wunderground_station_id": response.wunderground_station_id,
        "weathercom_location_id": response.weathercom_location_id,
        "start_date": response.start_local_date.isoformat(),
        "end_date": response.end_local_date.isoformat(),
        "units": response.units,
        "retrieved_at_utc": response.retrieved_at_utc.isoformat(),
        "http_status": response.http_status,
        "body_hash": response.response_body_sha256,
    }
    return f"wu_req_{sha256_hex(canonical_json(identity))[:32]}"


def _provider_record_key(response: WundergroundRawDayResponse) -> str:
    return (
        f"weathercom_historical:{response.station_id}:"
        f"{response.start_local_date.isoformat()}:{response.end_local_date.isoformat()}:{response.units}"
    )


def _request_hash(response: WundergroundRawDayResponse) -> str:
    return sha256_hex(
        canonical_json(
            {
                "weathercom_location_id": response.weathercom_location_id,
                "start_date": response.start_local_date.isoformat(),
                "end_date": response.end_local_date.isoformat(),
                "units": response.units,
                "endpoint": "historical_observations",
            }
        )
    )


def _insert_source_request(
    connection: Connection,
    response: WundergroundRawDayResponse,
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
                'GET',
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
            "request_params_json": _json_dumps(
                {
                    "station_id": response.station_id,
                    "wunderground_station_id": response.wunderground_station_id,
                    "weathercom_location_id": response.weathercom_location_id,
                    "start_date": response.start_local_date.isoformat(),
                    "end_date": response.end_local_date.isoformat(),
                    "units": response.units,
                }
            ),
            "request_headers_redacted": _json_dumps(
                {"Accept": "application/json", "Accept-Encoding": "gzip"}
            ),
            "retrieved_at_utc": response.retrieved_at_utc,
            "http_status": response.http_status,
            "content_type": response.content_type,
            "response_body_sha256": response.response_body_sha256,
            "response_size_bytes": response.response_size_bytes,
            "raw_storage_uri": f"db://bronze.source_records/{_provider_record_key(response)}",
            "parser_version": PARSER_VERSION,
        },
    )


def _current_source_record(
    connection: Connection,
    provider_record_key: str,
) -> CurrentBronzeRecord | None:
    row = connection.execute(
        text(
            """
            SELECT source_record_id, payload_hash, revision_number, is_current
            FROM bronze.source_records
            WHERE source_name = :source_name
              AND provider_name = :provider_name
              AND endpoint_name = :endpoint_name
              AND provider_record_key = :provider_record_key
              AND is_current = true
            """
        ),
        {
            "source_name": SOURCE_NAME,
            "provider_name": PROVIDER_NAME,
            "endpoint_name": ENDPOINT_NAME,
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
    response: WundergroundRawDayResponse,
    source_request_id: str,
) -> UUID:
    provider_record_key = _provider_record_key(response)
    new_payload_hash = payload_hash(response.payload_json) if response.payload_json is not None else sha256_hex(response.response_body_text)
    current = _current_source_record(connection, provider_record_key)
    decision = decide_bronze_revision(current_record=current, new_payload_hash=new_payload_hash)
    if decision.action == "return_existing" and decision.source_record_id is not None:
        return decision.source_record_id
    if decision.mark_prior_current_false and decision.supersedes_source_record_id is not None:
        connection.execute(
            text(
                """
                UPDATE bronze.source_records
                SET is_current = false
                WHERE source_record_id = :source_record_id
                """
            ),
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
                :payload_format,
                CAST(:payload_json AS jsonb),
                :payload_text,
                NULL,
                NULL,
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
            "endpoint_name": ENDPOINT_NAME,
            "provider_record_key": provider_record_key,
            "request_hash": _request_hash(response),
            "payload_hash": new_payload_hash,
            "payload_format": "json" if response.payload_json is not None else "text",
            "payload_json": _json_dumps(response.payload_json) if response.payload_json is not None else None,
            "payload_text": None if response.payload_json is not None else response.response_body_text,
            "provider_available_at_utc": response.retrieved_at_utc,
            "acquired_at_utc": response.retrieved_at_utc,
            "revision_number": decision.revision_number,
            "supersedes_source_record_id": decision.supersedes_source_record_id,
        },
    ).one()
    return row.source_record_id


def _insert_fetch_window(
    connection: Connection,
    *,
    job_id: str,
    response: WundergroundRawDayResponse,
    status: str,
    source_request_id: str | None,
    source_record_id: UUID | None,
    daily_rows: int,
    intraday_rows: int,
    observations_count: int,
) -> UUID:
    row = connection.execute(
        text(
            """
            INSERT INTO audit.wu_fetch_windows (
                job_id,
                station_id,
                wunderground_station_id,
                weathercom_location_id,
                window_start_date,
                window_end_date,
                units,
                status,
                attempts,
                http_status,
                error_type,
                error_message,
                source_request_id,
                source_record_id,
                observations_count,
                daily_rows_upserted,
                intraday_rows_upserted,
                started_at_utc,
                finished_at_utc
            )
            VALUES (
                :job_id,
                :station_id,
                :wunderground_station_id,
                :weathercom_location_id,
                :window_start_date,
                :window_end_date,
                :units,
                :status,
                :attempts,
                :http_status,
                :error_type,
                :error_message,
                :source_request_id,
                :source_record_id,
                :observations_count,
                :daily_rows_upserted,
                :intraday_rows_upserted,
                :started_at_utc,
                :finished_at_utc
            )
            ON CONFLICT (job_id, station_id, window_start_date, window_end_date)
            DO UPDATE SET
                status = EXCLUDED.status,
                attempts = EXCLUDED.attempts,
                http_status = EXCLUDED.http_status,
                error_type = EXCLUDED.error_type,
                error_message = EXCLUDED.error_message,
                source_request_id = EXCLUDED.source_request_id,
                source_record_id = EXCLUDED.source_record_id,
                observations_count = EXCLUDED.observations_count,
                daily_rows_upserted = EXCLUDED.daily_rows_upserted,
                intraday_rows_upserted = EXCLUDED.intraday_rows_upserted,
                finished_at_utc = EXCLUDED.finished_at_utc,
                updated_at = now()
            RETURNING wu_fetch_window_id
            """
        ),
        {
            "job_id": job_id,
            "station_id": response.station_id,
            "wunderground_station_id": response.wunderground_station_id,
            "weathercom_location_id": response.weathercom_location_id,
            "window_start_date": response.start_local_date,
            "window_end_date": response.end_local_date,
            "units": response.units,
            "status": status,
            "attempts": response.attempts,
            "http_status": response.http_status,
            "error_type": response.error_type,
            "error_message": response.error_message,
            "source_request_id": source_request_id,
            "source_record_id": source_record_id,
            "observations_count": observations_count,
            "daily_rows_upserted": daily_rows,
            "intraday_rows_upserted": intraday_rows,
            "started_at_utc": response.retrieved_at_utc,
            "finished_at_utc": datetime.now(timezone.utc),
        },
    ).one()
    return row.wu_fetch_window_id


def _upsert_intraday(
    connection: Connection,
    *,
    row: WundergroundIntradayObservation,
    source_request_id: str,
    source_record_id: UUID,
    ingested_at_utc: datetime,
) -> int:
    result = connection.execute(
        text(
            """
            INSERT INTO silver.wu_intraday_observations (
                station_id,
                wunderground_station_id,
                weathercom_location_id,
                observation_time_local,
                observation_time_utc,
                local_date,
                timezone_name,
                temp_f,
                dewpoint_f,
                humidity_pct,
                wind_speed_mph,
                wind_gust_mph,
                wind_direction_deg,
                pressure_in,
                precipitation_in,
                condition_text,
                cloud_cover_text,
                uv_index,
                solar_radiation,
                raw_observation_json,
                provider_available_at_utc,
                our_ingested_at_utc,
                source_request_id,
                source_record_id,
                quality_flag,
                quality_note
            )
            VALUES (
                :station_id,
                :wunderground_station_id,
                :weathercom_location_id,
                :observation_time_local,
                :observation_time_utc,
                :local_date,
                :timezone_name,
                :temp_f,
                :dewpoint_f,
                :humidity_pct,
                :wind_speed_mph,
                :wind_gust_mph,
                :wind_direction_deg,
                :pressure_in,
                :precipitation_in,
                :condition_text,
                :cloud_cover_text,
                :uv_index,
                :solar_radiation,
                CAST(:raw_observation_json AS jsonb),
                :provider_available_at_utc,
                :our_ingested_at_utc,
                :source_request_id,
                :source_record_id,
                :quality_flag,
                :quality_note
            )
            ON CONFLICT (station_id, observation_time_utc)
            DO UPDATE SET
                temp_f = EXCLUDED.temp_f,
                dewpoint_f = EXCLUDED.dewpoint_f,
                humidity_pct = EXCLUDED.humidity_pct,
                wind_speed_mph = EXCLUDED.wind_speed_mph,
                wind_gust_mph = EXCLUDED.wind_gust_mph,
                wind_direction_deg = EXCLUDED.wind_direction_deg,
                pressure_in = EXCLUDED.pressure_in,
                precipitation_in = EXCLUDED.precipitation_in,
                condition_text = EXCLUDED.condition_text,
                cloud_cover_text = EXCLUDED.cloud_cover_text,
                uv_index = EXCLUDED.uv_index,
                solar_radiation = EXCLUDED.solar_radiation,
                raw_observation_json = EXCLUDED.raw_observation_json,
                provider_available_at_utc = EXCLUDED.provider_available_at_utc,
                source_request_id = EXCLUDED.source_request_id,
                source_record_id = EXCLUDED.source_record_id,
                quality_flag = EXCLUDED.quality_flag,
                quality_note = EXCLUDED.quality_note,
                updated_at = now()
            """
        ),
        {
            **row.__dict__,
            "raw_observation_json": _json_dumps(row.raw_observation_json),
            "our_ingested_at_utc": ingested_at_utc,
            "source_request_id": source_request_id,
            "source_record_id": source_record_id,
        },
    )
    _replace_availability_row(
        connection,
        source_record_id=source_record_id,
        canonical_record_key=f"wu_intraday:{row.station_id}:{row.observation_time_utc.isoformat()}",
        station_id=row.station_id,
        variable_name="intraday_observation",
        provider_available_at_utc=row.provider_available_at_utc,
        acquired_at_utc=ingested_at_utc,
        effective_available_at_utc=row.provider_available_at_utc,
        source_lag_seconds=int((row.provider_available_at_utc - row.observation_time_utc).total_seconds()),
        valid_time_utc=row.observation_time_utc,
    )
    return result.rowcount or 0


def _upsert_daily(
    connection: Connection,
    *,
    row: WundergroundDailyActual,
    source_request_id: str,
    source_record_id: UUID,
    ingested_at_utc: datetime,
) -> tuple[int, int]:
    current = connection.execute(
        text(
            """
            SELECT daily_high_f, source_request_id, source_record_id
            FROM silver.wu_daily_actuals
            WHERE station_id = :station_id AND local_date = :local_date
            """
        ),
        {"station_id": row.station_id, "local_date": row.local_date},
    ).mappings().first()
    revisions = 0
    if current is not None and current["daily_high_f"] != row.daily_high_f:
        connection.execute(
            text(
                """
                INSERT INTO silver.wu_daily_actual_revisions (
                    station_id,
                    local_date,
                    previous_daily_high_f,
                    new_daily_high_f,
                    previous_source_request_id,
                    new_source_request_id,
                    previous_source_record_id,
                    new_source_record_id,
                    detected_at_utc,
                    note
                )
                VALUES (
                    :station_id,
                    :local_date,
                    :previous_daily_high_f,
                    :new_daily_high_f,
                    :previous_source_request_id,
                    :new_source_request_id,
                    :previous_source_record_id,
                    :new_source_record_id,
                    :detected_at_utc,
                    :note
                )
                """
            ),
            {
                "station_id": row.station_id,
                "local_date": row.local_date,
                "previous_daily_high_f": current["daily_high_f"],
                "new_daily_high_f": row.daily_high_f,
                "previous_source_request_id": current["source_request_id"],
                "new_source_request_id": source_request_id,
                "previous_source_record_id": current["source_record_id"],
                "new_source_record_id": source_record_id,
                "detected_at_utc": ingested_at_utc,
                "note": "Wunderground daily high changed on refetch.",
            },
        )
        revisions = 1

    result = connection.execute(
        text(
            """
            INSERT INTO silver.wu_daily_actuals (
                station_id,
                wunderground_station_id,
                weathercom_location_id,
                local_date,
                timezone_name,
                local_day_start_utc,
                local_day_end_utc,
                daily_high_f,
                settlement_high_f_whole,
                daily_low_f,
                daily_avg_temp_f,
                daily_high_dewpoint_f,
                daily_low_dewpoint_f,
                daily_precipitation_in,
                daily_max_wind_speed_mph,
                daily_max_wind_gust_mph,
                daily_avg_wind_speed_mph,
                daily_dominant_wind_direction_deg,
                label_method,
                daily_high_source_field,
                provider_available_at_utc,
                our_ingested_at_utc,
                source_request_id,
                source_record_id,
                source_daily_summary_json,
                raw_daily_json,
                observations_count,
                quality_flag,
                quality_note
            )
            VALUES (
                :station_id,
                :wunderground_station_id,
                :weathercom_location_id,
                :local_date,
                :timezone_name,
                :local_day_start_utc,
                :local_day_end_utc,
                :daily_high_f,
                :settlement_high_f_whole,
                :daily_low_f,
                :daily_avg_temp_f,
                :daily_high_dewpoint_f,
                :daily_low_dewpoint_f,
                :daily_precipitation_in,
                :daily_max_wind_speed_mph,
                :daily_max_wind_gust_mph,
                :daily_avg_wind_speed_mph,
                :daily_dominant_wind_direction_deg,
                :label_method,
                :daily_high_source_field,
                :provider_available_at_utc,
                :our_ingested_at_utc,
                :source_request_id,
                :source_record_id,
                CAST(:source_daily_summary_json AS jsonb),
                CAST(:raw_daily_json AS jsonb),
                :observations_count,
                :quality_flag,
                :quality_note
            )
            ON CONFLICT (station_id, local_date)
            DO UPDATE SET
                wunderground_station_id = EXCLUDED.wunderground_station_id,
                weathercom_location_id = EXCLUDED.weathercom_location_id,
                timezone_name = EXCLUDED.timezone_name,
                local_day_start_utc = EXCLUDED.local_day_start_utc,
                local_day_end_utc = EXCLUDED.local_day_end_utc,
                daily_high_f = EXCLUDED.daily_high_f,
                settlement_high_f_whole = EXCLUDED.settlement_high_f_whole,
                daily_low_f = EXCLUDED.daily_low_f,
                daily_avg_temp_f = EXCLUDED.daily_avg_temp_f,
                daily_high_dewpoint_f = EXCLUDED.daily_high_dewpoint_f,
                daily_low_dewpoint_f = EXCLUDED.daily_low_dewpoint_f,
                daily_precipitation_in = EXCLUDED.daily_precipitation_in,
                daily_max_wind_speed_mph = EXCLUDED.daily_max_wind_speed_mph,
                daily_max_wind_gust_mph = EXCLUDED.daily_max_wind_gust_mph,
                daily_avg_wind_speed_mph = EXCLUDED.daily_avg_wind_speed_mph,
                daily_dominant_wind_direction_deg = EXCLUDED.daily_dominant_wind_direction_deg,
                label_method = EXCLUDED.label_method,
                daily_high_source_field = EXCLUDED.daily_high_source_field,
                provider_available_at_utc = EXCLUDED.provider_available_at_utc,
                our_ingested_at_utc = EXCLUDED.our_ingested_at_utc,
                source_request_id = EXCLUDED.source_request_id,
                source_record_id = EXCLUDED.source_record_id,
                source_daily_summary_json = EXCLUDED.source_daily_summary_json,
                raw_daily_json = EXCLUDED.raw_daily_json,
                observations_count = EXCLUDED.observations_count,
                quality_flag = EXCLUDED.quality_flag,
                quality_note = EXCLUDED.quality_note,
                updated_at = now()
            """
        ),
        {
            **row.__dict__,
            "our_ingested_at_utc": ingested_at_utc,
            "source_request_id": source_request_id,
            "source_record_id": source_record_id,
            "source_daily_summary_json": _json_dumps(row.source_daily_summary_json),
            "raw_daily_json": _json_dumps(row.raw_daily_json),
        },
    )

    _replace_availability_row(
        connection,
        source_record_id=source_record_id,
        canonical_record_key=f"wu_daily_actual:{row.station_id}:{row.local_date.isoformat()}",
        station_id=row.station_id,
        variable_name="daily_high_f",
        provider_available_at_utc=row.provider_available_at_utc,
        acquired_at_utc=ingested_at_utc,
        effective_available_at_utc=row.provider_available_at_utc,
        source_lag_seconds=int((row.provider_available_at_utc - row.local_day_end_utc).total_seconds()),
        valid_time_utc=row.local_day_end_utc,
    )
    if row.station_id == "KLGA" and row.daily_high_f is not None:
        _upsert_target_daily_actual(
            connection,
            row=row,
            source_record_id=source_record_id,
        )
        _refresh_target_instances_for_label(connection, row=row)
    return result.rowcount or 0, revisions


def _replace_availability_row(
    connection: Connection,
    *,
    source_record_id: UUID,
    canonical_record_key: str,
    station_id: str,
    variable_name: str,
    provider_available_at_utc: datetime,
    acquired_at_utc: datetime,
    effective_available_at_utc: datetime,
    source_lag_seconds: int,
    valid_time_utc: datetime,
) -> None:
    connection.execute(
        text(
            """
            DELETE FROM silver.availability_ledger
            WHERE source_name = :source_name
              AND provider_name = :provider_name
              AND canonical_record_key = :canonical_record_key
              AND variable_name = :variable_name
              AND station_id = :station_id
            """
        ),
        {
            "source_name": SOURCE_NAME,
            "provider_name": PROVIDER_NAME,
            "canonical_record_key": canonical_record_key,
            "variable_name": variable_name,
            "station_id": station_id,
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
                NULL,
                NULL,
                :valid_time_utc,
                NULL,
                NULL,
                :variable_name,
                :provider_available_at_utc,
                :acquired_at_utc,
                :effective_available_at_utc,
                'conservative_lag_rule',
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
            "station_id": station_id,
            "valid_time_utc": valid_time_utc,
            "variable_name": variable_name,
            "provider_available_at_utc": provider_available_at_utc,
            "acquired_at_utc": acquired_at_utc,
            "effective_available_at_utc": effective_available_at_utc,
            "source_lag_seconds": source_lag_seconds,
        },
    )


def _upsert_target_daily_actual(
    connection: Connection,
    *,
    row: WundergroundDailyActual,
    source_record_id: UUID,
) -> None:
    current = connection.execute(
        text(
            """
            SELECT high_temp_f, revision_number
            FROM silver.target_daily_actuals
            WHERE target_date = :target_date
              AND station_id = 'KLGA'
              AND source_name = :source_name
              AND is_current = true
            """
        ),
        {"target_date": row.local_date, "source_name": SOURCE_NAME},
    ).mappings().first()
    if current is not None and current["high_temp_f"] == row.daily_high_f:
        return
    next_revision = int(current["revision_number"]) + 1 if current is not None else 1
    if current is not None:
        connection.execute(
            text(
                """
                UPDATE silver.target_daily_actuals
                SET is_current = false
                WHERE target_date = :target_date
                  AND station_id = 'KLGA'
                  AND source_name = :source_name
                  AND is_current = true
                """
            ),
            {"target_date": row.local_date, "source_name": SOURCE_NAME},
        )
    connection.execute(
        text(
            """
            INSERT INTO silver.target_daily_actuals (
                target_date,
                station_id,
                source_name,
                high_temp_f,
                low_temp_f,
                source_available_at_utc,
                source_record_id,
                revision_number,
                is_current
            )
            VALUES (
                :target_date,
                'KLGA',
                :source_name,
                :high_temp_f,
                :low_temp_f,
                :source_available_at_utc,
                :source_record_id,
                :revision_number,
                true
            )
            """
        ),
        {
            "target_date": row.local_date,
            "source_name": SOURCE_NAME,
            "high_temp_f": row.daily_high_f,
            "low_temp_f": row.daily_low_f,
            "source_available_at_utc": row.provider_available_at_utc,
            "source_record_id": source_record_id,
            "revision_number": next_revision,
        },
    )


def _refresh_target_instances_for_label(connection: Connection, *, row: WundergroundDailyActual) -> None:
    connection.execute(
        text(
            """
            UPDATE gold.target_instances
            SET
                settlement_high_f_whole = :settlement_high_f_whole,
                settlement_high_available_at_utc = :settlement_high_available_at_utc,
                label_available = :settlement_high_available_at_utc <= cutoff_utc,
                label_revision_sensitive = true
            WHERE target_date = :target_date
              AND target_station_id = 'KLGA'
            """
        ),
        {
            "target_date": row.local_date,
            "settlement_high_f_whole": row.settlement_high_f_whole,
            "settlement_high_available_at_utc": row.provider_available_at_utc,
        },
    )


def _update_coverage_rows(
    connection: Connection,
    *,
    response: WundergroundRawDayResponse,
    parsed: ParsedWundergroundResponse,
    source_request_id: str | None,
    source_record_id: UUID | None,
    fetch_window_id: UUID,
    status: str,
) -> int:
    daily_by_date = {row.local_date: row for row in parsed.daily_actuals}
    intraday_counts: dict[date, int] = {}
    for row in parsed.intraday_observations:
        intraday_counts[row.local_date] = intraday_counts.get(row.local_date, 0) + 1
    updated = 0
    for local_date in iter_dates(response.start_local_date, response.end_local_date):
        if status == "failed":
            coverage_status = "failed"
            quality_flag = "failed"
            daily_present = False
        elif local_date in daily_by_date:
            daily_present = daily_by_date[local_date].daily_high_f is not None
            coverage_status = "saved" if daily_present else "no_data"
            quality_flag = daily_by_date[local_date].quality_flag
        else:
            coverage_status = "no_data"
            quality_flag = "missing"
            daily_present = False
        connection.execute(
            text(
                """
                INSERT INTO audit.wu_station_date_coverage (
                    station_id,
                    local_date,
                    wunderground_station_id,
                    weathercom_location_id,
                    status,
                    source_request_id,
                    source_record_id,
                    wu_fetch_window_id,
                    daily_actual_present,
                    intraday_observation_count,
                    first_attempt_at_utc,
                    last_attempt_at_utc,
                    last_success_at_utc,
                    last_error_type,
                    last_error_message,
                    quality_flag
                )
                VALUES (
                    :station_id,
                    :local_date,
                    :wunderground_station_id,
                    :weathercom_location_id,
                    :status,
                    :source_request_id,
                    :source_record_id,
                    :wu_fetch_window_id,
                    :daily_actual_present,
                    :intraday_observation_count,
                    :first_attempt_at_utc,
                    :last_attempt_at_utc,
                    :last_success_at_utc,
                    :last_error_type,
                    :last_error_message,
                    :quality_flag
                )
                ON CONFLICT (station_id, local_date)
                DO UPDATE SET
                    wunderground_station_id = EXCLUDED.wunderground_station_id,
                    weathercom_location_id = EXCLUDED.weathercom_location_id,
                    status = EXCLUDED.status,
                    source_request_id = EXCLUDED.source_request_id,
                    source_record_id = EXCLUDED.source_record_id,
                    wu_fetch_window_id = EXCLUDED.wu_fetch_window_id,
                    daily_actual_present = EXCLUDED.daily_actual_present,
                    intraday_observation_count = EXCLUDED.intraday_observation_count,
                    first_attempt_at_utc = COALESCE(
                        audit.wu_station_date_coverage.first_attempt_at_utc,
                        EXCLUDED.first_attempt_at_utc
                    ),
                    last_attempt_at_utc = EXCLUDED.last_attempt_at_utc,
                    last_success_at_utc = EXCLUDED.last_success_at_utc,
                    last_error_type = EXCLUDED.last_error_type,
                    last_error_message = EXCLUDED.last_error_message,
                    quality_flag = EXCLUDED.quality_flag,
                    updated_at = now()
                """
            ),
            {
                "station_id": response.station_id,
                "local_date": local_date,
                "wunderground_station_id": response.wunderground_station_id,
                "weathercom_location_id": response.weathercom_location_id,
                "status": coverage_status,
                "source_request_id": source_request_id,
                "source_record_id": source_record_id,
                "wu_fetch_window_id": fetch_window_id,
                "daily_actual_present": daily_present,
                "intraday_observation_count": intraday_counts.get(local_date, 0),
                "first_attempt_at_utc": response.retrieved_at_utc,
                "last_attempt_at_utc": response.retrieved_at_utc,
                "last_success_at_utc": response.retrieved_at_utc if coverage_status == "saved" else None,
                "last_error_type": response.error_type if coverage_status == "failed" else None,
                "last_error_message": response.error_message if coverage_status == "failed" else None,
                "quality_flag": quality_flag,
            },
        )
        updated += 1
    return updated


def fetch_window_status(
    response: WundergroundRawDayResponse,
    parsed: ParsedWundergroundResponse,
) -> str:
    if response.success:
        return "succeeded" if parsed.daily_actuals or parsed.intraday_observations else "no_data"
    if response.provider_no_data:
        return "no_data"
    return "failed"


def persist_wunderground_response(
    connection: Connection,
    *,
    job_id: str,
    response: WundergroundRawDayResponse,
    parsed: ParsedWundergroundResponse,
) -> PersistedWundergroundWindow:
    source_request_id = _source_request_id(response)
    _insert_source_request(connection, response, source_request_id)
    source_record_id = _insert_or_reuse_source_record(connection, response, source_request_id)

    daily_rows = 0
    intraday_rows = 0
    revisions = 0
    ingested_at_utc = datetime.now(timezone.utc)
    if response.success:
        for intraday in parsed.intraday_observations:
            intraday_rows += _upsert_intraday(
                connection,
                row=intraday,
                source_request_id=source_request_id,
                source_record_id=source_record_id,
                ingested_at_utc=ingested_at_utc,
            )
        for daily in parsed.daily_actuals:
            changed, revision_count = _upsert_daily(
                connection,
                row=daily,
                source_request_id=source_request_id,
                source_record_id=source_record_id,
                ingested_at_utc=ingested_at_utc,
            )
            daily_rows += changed
            revisions += revision_count

    status = fetch_window_status(response, parsed)
    fetch_window_id = _insert_fetch_window(
        connection,
        job_id=job_id,
        response=response,
        status=status,
        source_request_id=source_request_id,
        source_record_id=source_record_id,
        daily_rows=daily_rows,
        intraday_rows=intraday_rows,
        observations_count=parsed.observations_count,
    )
    coverage_rows = _update_coverage_rows(
        connection,
        response=response,
        parsed=parsed,
        source_request_id=source_request_id,
        source_record_id=source_record_id,
        fetch_window_id=fetch_window_id,
        status=status,
    )
    return PersistedWundergroundWindow(
        source_request_id=source_request_id,
        source_record_id=source_record_id,
        fetch_window_id=fetch_window_id,
        status=status,
        daily_rows_upserted=daily_rows,
        intraday_rows_upserted=intraday_rows,
        coverage_rows_updated=coverage_rows,
        revisions_inserted=revisions,
        observations_count=parsed.observations_count,
        error_type=response.error_type,
        error_message=response.error_message,
    )


def mark_station_dates_not_fetched(
    connection: Connection,
    *,
    station_id: str,
    wunderground_station_id: str,
    weathercom_location_id: str,
    start_date: date,
    end_date: date,
) -> int:
    result = connection.execute(
        text(
            """
            INSERT INTO audit.wu_station_date_coverage (
                station_id,
                local_date,
                wunderground_station_id,
                weathercom_location_id,
                status,
                quality_flag
            )
            SELECT
                :station_id,
                generated.local_date::date,
                :wunderground_station_id,
                :weathercom_location_id,
                'not_fetched',
                'missing'
            FROM generate_series(
                CAST(:start_date AS date),
                CAST(:end_date AS date),
                interval '1 day'
            ) AS generated(local_date)
            ON CONFLICT (station_id, local_date) DO NOTHING
            """
        ),
        {
            "station_id": station_id,
            "wunderground_station_id": wunderground_station_id,
            "weathercom_location_id": weathercom_location_id,
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    return result.rowcount or 0


def window_already_complete(
    connection: Connection,
    *,
    station_id: str,
    start_date: date,
    end_date: date,
) -> bool:
    total_days = sum(1 for _ in iter_dates(start_date, end_date))
    complete_count = connection.execute(
        text(
            """
            SELECT count(*)
            FROM audit.wu_station_date_coverage
            WHERE station_id = :station_id
              AND local_date BETWEEN :start_date AND :end_date
              AND status IN ('saved','no_data')
            """
        ),
        {"station_id": station_id, "start_date": start_date, "end_date": end_date},
    ).scalar_one()
    return int(complete_count) == total_days
