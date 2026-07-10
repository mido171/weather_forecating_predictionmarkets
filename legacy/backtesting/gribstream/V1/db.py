from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import DB_PATH, SCHEMA_PATH, ensure_directories, isoformat_utc, utc_now
from .model_catalog import model_catalog_rows

sqlite3.register_adapter(bool, int)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    ensure_directories()
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    connection.commit()


def seed_model_catalog(connection: sqlite3.Connection) -> None:
    rows = model_catalog_rows(created_at_utc=isoformat_utc(utc_now()))
    sql = """
        INSERT INTO model_catalog (
            model_code, family, role, archive_start, snapshot_var_name, snapshot_var_level,
            snapshot_var_info, native_tmax_var_name, native_tmax_var_level,
            native_tmax_var_info, native_tmax_available_from, ensemble_members_json,
            enabled_backtest, enabled_live, notes, created_at_utc
        ) VALUES (
            :model_code, :family, :role, :archive_start, :snapshot_var_name, :snapshot_var_level,
            :snapshot_var_info, :native_tmax_var_name, :native_tmax_var_level,
            :native_tmax_var_info, :native_tmax_available_from, :ensemble_members_json,
            :enabled_backtest, :enabled_live, :notes, :created_at_utc
        )
        ON CONFLICT(model_code) DO UPDATE SET
            family=excluded.family,
            role=excluded.role,
            archive_start=excluded.archive_start,
            snapshot_var_name=excluded.snapshot_var_name,
            snapshot_var_level=excluded.snapshot_var_level,
            snapshot_var_info=excluded.snapshot_var_info,
            native_tmax_var_name=excluded.native_tmax_var_name,
            native_tmax_var_level=excluded.native_tmax_var_level,
            native_tmax_var_info=excluded.native_tmax_var_info,
            native_tmax_available_from=excluded.native_tmax_available_from,
            ensemble_members_json=excluded.ensemble_members_json,
            enabled_backtest=excluded.enabled_backtest,
            enabled_live=excluded.enabled_live,
            notes=excluded.notes
    """
    connection.executemany(sql, rows)
    connection.commit()


def table_row_count(connection: sqlite3.Connection, table_name: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) AS c FROM {table_name}").fetchone()
    return int(row["c"])


def list_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row["name"]) for row in rows]


def load_truth_rows(connection: sqlite3.Connection, station_id: str, start_date: str, end_date: str) -> list[sqlite3.Row]:
    cursor = connection.execute(
        """
        SELECT *
        FROM nws_daily_settlements
        WHERE station_id = ?
          AND settlement_date_local BETWEEN ? AND ?
        ORDER BY settlement_date_local
        """,
        (station_id, start_date, end_date),
    )
    return cursor.fetchall()


def successful_request_ids(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT request_id FROM gribstream_requests WHERE success = 1"
    ).fetchall()
    return {str(row["request_id"]) for row in rows}


def upsert_nws_daily_settlements(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO nws_daily_settlements (
            station_id, settlement_date_local, timezone, local_day_start_utc, local_day_end_utc,
            actual_tmax_native, actual_tmax_native_unit, actual_tmax_f, source, ingested_at_utc
        ) VALUES (
            :station_id, :settlement_date_local, :timezone, :local_day_start_utc, :local_day_end_utc,
            :actual_tmax_native, :actual_tmax_native_unit, :actual_tmax_f, :source, :ingested_at_utc
        )
        ON CONFLICT(station_id, settlement_date_local) DO UPDATE SET
            timezone=excluded.timezone,
            local_day_start_utc=excluded.local_day_start_utc,
            local_day_end_utc=excluded.local_day_end_utc,
            actual_tmax_native=excluded.actual_tmax_native,
            actual_tmax_native_unit=excluded.actual_tmax_native_unit,
            actual_tmax_f=excluded.actual_tmax_f,
            source=excluded.source,
            ingested_at_utc=excluded.ingested_at_utc
    """
    connection.executemany(sql, rows)
    connection.commit()


def upsert_gribstream_request(connection: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    sql = """
        INSERT INTO gribstream_requests (
            request_id, model_code, station_id, settlement_date_local, endpoint, as_of_utc,
            from_time_utc, until_time_utc, http_status, attempts, success, row_count,
            error_text, started_at_utc, finished_at_utc, response_format, response_compressed
        ) VALUES (
            :request_id, :model_code, :station_id, :settlement_date_local, :endpoint, :as_of_utc,
            :from_time_utc, :until_time_utc, :http_status, :attempts, :success, :row_count,
            :error_text, :started_at_utc, :finished_at_utc, :response_format, :response_compressed
        )
        ON CONFLICT(request_id) DO UPDATE SET
            model_code=excluded.model_code,
            station_id=excluded.station_id,
            settlement_date_local=excluded.settlement_date_local,
            endpoint=excluded.endpoint,
            as_of_utc=excluded.as_of_utc,
            from_time_utc=excluded.from_time_utc,
            until_time_utc=excluded.until_time_utc,
            http_status=excluded.http_status,
            attempts=excluded.attempts,
            success=excluded.success,
            row_count=excluded.row_count,
            error_text=excluded.error_text,
            started_at_utc=excluded.started_at_utc,
            finished_at_utc=excluded.finished_at_utc,
            response_format=excluded.response_format,
            response_compressed=excluded.response_compressed
    """
    connection.execute(sql, row)


def insert_gribstream_raw_forecasts(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT OR IGNORE INTO gribstream_raw_forecasts (
            request_id, model_code, station_id, settlement_date_local, as_of_utc, forecasted_at_utc,
            forecasted_time_utc, forecasted_time_local, forecasted_date_local, lat, lon, coord_name,
            variable_name, variable_level, variable_info, member, value_native, unit_native,
            value_f, lead_minutes, inserted_at_utc
        ) VALUES (
            :request_id, :model_code, :station_id, :settlement_date_local, :as_of_utc, :forecasted_at_utc,
            :forecasted_time_utc, :forecasted_time_local, :forecasted_date_local, :lat, :lon, :coord_name,
            :variable_name, :variable_level, :variable_info, :member, :value_native, :unit_native,
            :value_f, :lead_minutes, :inserted_at_utc
        )
    """
    connection.executemany(sql, rows)


def delete_range_rows(
    connection: sqlite3.Connection,
    table_name: str,
    station_id: str,
    start_date: str,
    end_date: str,
) -> None:
    connection.execute(
        f"""
        DELETE FROM {table_name}
        WHERE station_id = ?
          AND settlement_date_local BETWEEN ? AND ?
        """,
        (station_id, start_date, end_date),
    )


def replace_daily_model_tmax(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO daily_model_tmax (
            station_id, settlement_date_local, model_code, family, as_of_utc, local_day_start_utc,
            local_day_end_utc, native_tmax_f, snapshot_tmax_f, interpolated_tmax_f,
            selected_raw_tmax_f, selected_method, snapshot_row_count, native_row_count,
            model_available, notes, created_at_utc
        ) VALUES (
            :station_id, :settlement_date_local, :model_code, :family, :as_of_utc, :local_day_start_utc,
            :local_day_end_utc, :native_tmax_f, :snapshot_tmax_f, :interpolated_tmax_f,
            :selected_raw_tmax_f, :selected_method, :snapshot_row_count, :native_row_count,
            :model_available, :notes, :created_at_utc
        )
        ON CONFLICT(station_id, settlement_date_local, model_code) DO UPDATE SET
            family=excluded.family,
            as_of_utc=excluded.as_of_utc,
            local_day_start_utc=excluded.local_day_start_utc,
            local_day_end_utc=excluded.local_day_end_utc,
            native_tmax_f=excluded.native_tmax_f,
            snapshot_tmax_f=excluded.snapshot_tmax_f,
            interpolated_tmax_f=excluded.interpolated_tmax_f,
            selected_raw_tmax_f=excluded.selected_raw_tmax_f,
            selected_method=excluded.selected_method,
            snapshot_row_count=excluded.snapshot_row_count,
            native_row_count=excluded.native_row_count,
            model_available=excluded.model_available,
            notes=excluded.notes,
            created_at_utc=excluded.created_at_utc
    """
    connection.executemany(sql, rows)


def replace_model_daily_errors(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO model_daily_errors (
            station_id, settlement_date_local, model_code, selected_raw_tmax_f,
            actual_tmax_f, error_f, abs_error_f, squared_error_f, created_at_utc
        ) VALUES (
            :station_id, :settlement_date_local, :model_code, :selected_raw_tmax_f,
            :actual_tmax_f, :error_f, :abs_error_f, :squared_error_f, :created_at_utc
        )
        ON CONFLICT(station_id, settlement_date_local, model_code) DO UPDATE SET
            selected_raw_tmax_f=excluded.selected_raw_tmax_f,
            actual_tmax_f=excluded.actual_tmax_f,
            error_f=excluded.error_f,
            abs_error_f=excluded.abs_error_f,
            squared_error_f=excluded.squared_error_f,
            created_at_utc=excluded.created_at_utc
    """
    connection.executemany(sql, rows)


def replace_daily_model_weights(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO daily_model_weights (
            station_id, settlement_date_local, model_code, family, train_start_date, train_end_date,
            train_n_days, ew_bias_f, ew_mae_f, ew_rmse_f, bias_corrected_tmax_f, raw_weight,
            model_cap_applied, family_cap_applied, final_weight, included_in_blend,
            exclusion_reason, created_at_utc
        ) VALUES (
            :station_id, :settlement_date_local, :model_code, :family, :train_start_date, :train_end_date,
            :train_n_days, :ew_bias_f, :ew_mae_f, :ew_rmse_f, :bias_corrected_tmax_f, :raw_weight,
            :model_cap_applied, :family_cap_applied, :final_weight, :included_in_blend,
            :exclusion_reason, :created_at_utc
        )
        ON CONFLICT(station_id, settlement_date_local, model_code) DO UPDATE SET
            family=excluded.family,
            train_start_date=excluded.train_start_date,
            train_end_date=excluded.train_end_date,
            train_n_days=excluded.train_n_days,
            ew_bias_f=excluded.ew_bias_f,
            ew_mae_f=excluded.ew_mae_f,
            ew_rmse_f=excluded.ew_rmse_f,
            bias_corrected_tmax_f=excluded.bias_corrected_tmax_f,
            raw_weight=excluded.raw_weight,
            model_cap_applied=excluded.model_cap_applied,
            family_cap_applied=excluded.family_cap_applied,
            final_weight=excluded.final_weight,
            included_in_blend=excluded.included_in_blend,
            exclusion_reason=excluded.exclusion_reason,
            created_at_utc=excluded.created_at_utc
    """
    connection.executemany(sql, rows)


def replace_daily_prediction_components(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO daily_prediction_components (
            station_id, settlement_date_local, model_code, family, selected_raw_tmax_f,
            bias_corrected_tmax_f, final_weight, weighted_contribution_f, created_at_utc
        ) VALUES (
            :station_id, :settlement_date_local, :model_code, :family, :selected_raw_tmax_f,
            :bias_corrected_tmax_f, :final_weight, :weighted_contribution_f, :created_at_utc
        )
        ON CONFLICT(station_id, settlement_date_local, model_code) DO UPDATE SET
            family=excluded.family,
            selected_raw_tmax_f=excluded.selected_raw_tmax_f,
            bias_corrected_tmax_f=excluded.bias_corrected_tmax_f,
            final_weight=excluded.final_weight,
            weighted_contribution_f=excluded.weighted_contribution_f,
            created_at_utc=excluded.created_at_utc
    """
    connection.executemany(sql, rows)


def replace_daily_predictions(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO daily_predictions (
            station_id, settlement_date_local, as_of_utc, actual_tmax_f, equal_weight_blend_f,
            inverse_rmse_blend_f, family_capped_blend_f, nbm_only_f, hrrr_only_f, rap_only_f,
            gfs_only_f, best_single_model_code, best_single_model_pred_f, family_capped_error_f,
            family_capped_abs_error_f, created_at_utc
        ) VALUES (
            :station_id, :settlement_date_local, :as_of_utc, :actual_tmax_f, :equal_weight_blend_f,
            :inverse_rmse_blend_f, :family_capped_blend_f, :nbm_only_f, :hrrr_only_f, :rap_only_f,
            :gfs_only_f, :best_single_model_code, :best_single_model_pred_f, :family_capped_error_f,
            :family_capped_abs_error_f, :created_at_utc
        )
        ON CONFLICT(station_id, settlement_date_local) DO UPDATE SET
            as_of_utc=excluded.as_of_utc,
            actual_tmax_f=excluded.actual_tmax_f,
            equal_weight_blend_f=excluded.equal_weight_blend_f,
            inverse_rmse_blend_f=excluded.inverse_rmse_blend_f,
            family_capped_blend_f=excluded.family_capped_blend_f,
            nbm_only_f=excluded.nbm_only_f,
            hrrr_only_f=excluded.hrrr_only_f,
            rap_only_f=excluded.rap_only_f,
            gfs_only_f=excluded.gfs_only_f,
            best_single_model_code=excluded.best_single_model_code,
            best_single_model_pred_f=excluded.best_single_model_pred_f,
            family_capped_error_f=excluded.family_capped_error_f,
            family_capped_abs_error_f=excluded.family_capped_abs_error_f,
            created_at_utc=excluded.created_at_utc
    """
    connection.executemany(sql, rows)


def replace_metrics_summary(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO metrics_summary (
            metric_scope, metric_name, evaluation_start, evaluation_end, n_days, mae_f, rmse_f,
            bias_f, median_abs_error_f, within_0_5f, within_1f, within_2f, created_at_utc
        ) VALUES (
            :metric_scope, :metric_name, :evaluation_start, :evaluation_end, :n_days, :mae_f, :rmse_f,
            :bias_f, :median_abs_error_f, :within_0_5f, :within_1f, :within_2f, :created_at_utc
        )
        ON CONFLICT(metric_scope, metric_name, evaluation_start, evaluation_end) DO UPDATE SET
            n_days=excluded.n_days,
            mae_f=excluded.mae_f,
            rmse_f=excluded.rmse_f,
            bias_f=excluded.bias_f,
            median_abs_error_f=excluded.median_abs_error_f,
            within_0_5f=excluded.within_0_5f,
            within_1f=excluded.within_1f,
            within_2f=excluded.within_2f,
            created_at_utc=excluded.created_at_utc
    """
    connection.executemany(sql, rows)


def replace_coverage_summary(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO coverage_summary (
            model_code, role, archive_start, first_date_fetched, last_date_fetched,
            fetched_day_count, scored_day_count, notes, created_at_utc
        ) VALUES (
            :model_code, :role, :archive_start, :first_date_fetched, :last_date_fetched,
            :fetched_day_count, :scored_day_count, :notes, :created_at_utc
        )
        ON CONFLICT(model_code) DO UPDATE SET
            role=excluded.role,
            archive_start=excluded.archive_start,
            first_date_fetched=excluded.first_date_fetched,
            last_date_fetched=excluded.last_date_fetched,
            fetched_day_count=excluded.fetched_day_count,
            scored_day_count=excluded.scored_day_count,
            notes=excluded.notes,
            created_at_utc=excluded.created_at_utc
    """
    connection.executemany(sql, rows)


def commit(connection: sqlite3.Connection) -> None:
    connection.commit()
