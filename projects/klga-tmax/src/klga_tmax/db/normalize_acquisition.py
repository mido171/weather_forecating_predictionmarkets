from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from klga_tmax.constants import PROJECT_ROOT

NY_TZ = ZoneInfo("America/New_York")


class AcquisitionContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class TableContract:
    source_table: str
    required_columns: tuple[str, ...]


ACQUISITION_TABLE_MAP_PATH = PROJECT_ROOT / "config" / "acquisition_table_map.yaml"

ACQUISITION_CONTRACTS: tuple[TableContract, ...] = (
    TableContract(
        "silver.iem_mos_forecast_rows",
        (
            "station_id",
            "mos_station_id",
            "source_product",
            "endpoint_model",
            "cutoff_id",
            "run_time_utc",
            "forecast_valid_time_utc",
            "raw_values_jsonb",
            "provider_available_at_utc",
            "effective_available_at_utc",
            "raw_row_hash",
        ),
    ),
)


def _split_table(qualified_table: str) -> tuple[str, str]:
    schema, table = qualified_table.split(".", 1)
    return schema, table


def inspect_acquisition_contract(connection: Connection) -> None:
    if not ACQUISITION_TABLE_MAP_PATH.exists():
        raise AcquisitionContractError(f"missing acquisition map: {ACQUISITION_TABLE_MAP_PATH}")

    inspector = inspect(connection)
    for contract in ACQUISITION_CONTRACTS:
        schema, table = _split_table(contract.source_table)
        if table not in set(inspector.get_table_names(schema=schema)):
            raise AcquisitionContractError(f"missing acquisition source table {contract.source_table}")
        columns = {column["name"] for column in inspector.get_columns(table, schema=schema)}
        missing = [column for column in contract.required_columns if column not in columns]
        if missing:
            raise AcquisitionContractError(
                f"missing acquisition columns for {contract.source_table}: {', '.join(missing)}"
            )


def normalize_acquisition(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None = None,
    observation_start_date: date | None = None,
    mos_start_date: date | None = None,
) -> dict[str, int]:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    mos_start = mos_start_date or start_date
    if observation_start_date is not None and observation_start_date > end_date:
        raise ValueError("observation_start_date must be on or before end_date")
    if mos_start > end_date:
        raise ValueError("mos_start_date must be on or before end_date")
    inspect_acquisition_contract(connection)
    row_counts: dict[str, int] = {}
    row_counts["silver.mos_guidance"] = _normalize_mos_guidance(
        connection, start_date=mos_start, end_date=end_date, cutoff_id=cutoff_id
    )
    return row_counts


def _normalize_target_daily_actuals(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
) -> int:
    result = connection.execute(
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
            SELECT
                local_date,
                station_id,
                'wunderground',
                COALESCE(settlement_high_f_whole, daily_high_f),
                daily_low_f,
                provider_available_at_utc,
                source_record_id,
                1,
                true
            FROM silver.wu_daily_actuals
            WHERE station_id = 'KLGA'
              AND local_date BETWEEN :start_date AND :end_date
              AND COALESCE(settlement_high_f_whole, daily_high_f) IS NOT NULL
              AND provider_available_at_utc IS NOT NULL
            ON CONFLICT (target_date, station_id, source_name, revision_number)
            DO UPDATE SET
                high_temp_f = EXCLUDED.high_temp_f,
                low_temp_f = EXCLUDED.low_temp_f,
                source_available_at_utc = EXCLUDED.source_available_at_utc,
                source_record_id = EXCLUDED.source_record_id,
                is_current = true
            """
        ),
        {"start_date": start_date, "end_date": end_date},
    )
    return result.rowcount or 0


def _normalize_station_daily_actuals(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
) -> int:
    result = connection.execute(
        text(
            """
            INSERT INTO silver.station_daily_actuals (
                target_date,
                station_id,
                source_name,
                high_temp_f,
                low_temp_f,
                avg_temp_f,
                precip_in,
                max_wind_speed_mph,
                max_wind_gust_mph,
                provider_available_at_utc,
                effective_available_at_utc,
                source_request_id,
                source_record_id,
                revision_number,
                is_current,
                quality_flag,
                source_trace_json
            )
            SELECT
                local_date,
                station_id,
                'wunderground',
                COALESCE(settlement_high_f_whole, daily_high_f),
                daily_low_f,
                daily_avg_temp_f,
                daily_precipitation_in,
                daily_max_wind_speed_mph,
                daily_max_wind_gust_mph,
                provider_available_at_utc,
                provider_available_at_utc,
                source_request_id,
                source_record_id,
                1,
                true,
                quality_flag,
                jsonb_build_object(
                    'source_table', 'silver.wu_daily_actuals',
                    'availability_rule', 'wunderground_provider_available_at_utc'
                )
            FROM silver.wu_daily_actuals
            WHERE local_date BETWEEN :start_date AND :end_date
              AND provider_available_at_utc IS NOT NULL
            ON CONFLICT (target_date, station_id, source_name, revision_number)
            DO UPDATE SET
                high_temp_f = EXCLUDED.high_temp_f,
                low_temp_f = EXCLUDED.low_temp_f,
                avg_temp_f = EXCLUDED.avg_temp_f,
                precip_in = EXCLUDED.precip_in,
                max_wind_speed_mph = EXCLUDED.max_wind_speed_mph,
                max_wind_gust_mph = EXCLUDED.max_wind_gust_mph,
                provider_available_at_utc = EXCLUDED.provider_available_at_utc,
                effective_available_at_utc = EXCLUDED.effective_available_at_utc,
                source_request_id = EXCLUDED.source_request_id,
                source_record_id = EXCLUDED.source_record_id,
                is_current = true,
                quality_flag = EXCLUDED.quality_flag,
                source_trace_json = EXCLUDED.source_trace_json,
                updated_at = now()
            """
        ),
        {"start_date": start_date, "end_date": end_date},
    )
    return result.rowcount or 0


def _normalize_station_observations(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
) -> int:
    result = connection.execute(
        text(
            """
            INSERT INTO silver.station_observations (
                station_id,
                source_name,
                observation_time_utc,
                local_date,
                temp_f,
                dewpoint_f,
                humidity_pct,
                wind_speed_mph,
                wind_gust_mph,
                wind_direction_deg,
                pressure_in,
                precipitation_in,
                condition_text,
                provider_available_at_utc,
                effective_available_at_utc,
                source_request_id,
                source_record_id,
                raw_row_hash,
                quality_flag,
                source_trace_json
            )
            SELECT
                station_id,
                'wunderground',
                observation_time_utc,
                local_date,
                temp_f,
                dewpoint_f,
                humidity_pct,
                wind_speed_mph,
                wind_gust_mph,
                wind_direction_deg,
                pressure_in,
                precipitation_in,
                condition_text,
                provider_available_at_utc,
                provider_available_at_utc,
                source_request_id,
                source_record_id,
                md5(concat_ws('|', station_id, observation_time_utc::text, source_request_id, COALESCE(source_record_id::text, ''))),
                quality_flag,
                jsonb_build_object(
                    'source_table', 'silver.wu_intraday_observations',
                    'availability_rule', 'wunderground_provider_available_at_utc'
                )
            FROM silver.wu_intraday_observations
            WHERE local_date BETWEEN :start_date AND :end_date
              AND provider_available_at_utc IS NOT NULL
            ON CONFLICT DO NOTHING
            """
        ),
        {"start_date": start_date, "end_date": end_date},
    )
    return result.rowcount or 0


def _normalize_mos_guidance(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
) -> int:
    valid_start_utc = datetime.combine(start_date, time.min, NY_TZ).astimezone(timezone.utc)
    valid_end_utc = datetime.combine(end_date, time.min, NY_TZ).astimezone(timezone.utc) + timedelta(days=1)
    params: dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "valid_start_utc": valid_start_utc,
        "valid_end_utc": valid_end_utc,
        "cutoff_id": cutoff_id,
    }
    result = connection.execute(
        text(
            """
            INSERT INTO silver.mos_guidance (
                station_id,
                mos_station_id,
                source_product,
                endpoint_model,
                cutoff_id,
                run_time_utc,
                forecast_valid_time_utc,
                target_date,
                raw_values_jsonb,
                tmax_f,
                tmp_f,
                dpt_f,
                wsp_kt,
                pop,
                qpf,
                tstm_prob,
                provider_available_at_utc,
                effective_available_at_utc,
                availability_method,
                source_request_id,
                source_record_id,
                request_sha256,
                raw_row_hash,
                source_trace_json
            )
            SELECT
                station_id,
                mos_station_id,
                source_product,
                endpoint_model,
                cutoff_id,
                run_time_utc,
                forecast_valid_time_utc,
                (forecast_valid_time_utc AT TIME ZONE 'America/New_York')::date,
                raw_values_jsonb,
                n_x_f,
                tmp_f,
                dpt_f,
                wsp_kt,
                pop,
                qpf,
                tstm_prob,
                provider_available_at_utc,
                effective_available_at_utc,
                availability_method,
                source_request_id,
                source_record_id,
                request_sha256,
                raw_row_hash,
                jsonb_build_object(
                    'source_table', 'silver.iem_mos_forecast_rows',
                    'availability_method', availability_method
                )
            FROM silver.iem_mos_forecast_rows
            WHERE forecast_valid_time_utc >= :valid_start_utc
              AND forecast_valid_time_utc < :valid_end_utc
              AND (forecast_valid_time_utc AT TIME ZONE 'America/New_York')::date
                  BETWEEN :start_date AND :end_date
              AND (CAST(:cutoff_id AS text) IS NULL OR cutoff_id = CAST(:cutoff_id AS text))
            ON CONFLICT DO NOTHING
            """
        ),
        params,
    )
    return result.rowcount or 0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
