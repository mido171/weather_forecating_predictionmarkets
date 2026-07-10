from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


logger = logging.getLogger("ml_live.db")


@dataclass(frozen=True)
class MysqlConfig:
    host: str
    port: int
    database: str
    user: str
    password: str


class MySqlStore:
    def __init__(self, cfg: MysqlConfig) -> None:
        self._engine = create_engine(
            f"mysql+pymysql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}",
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self._create_tables()

    @property
    def engine(self) -> Engine:
        return self._engine

    def _create_tables(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS live_gribstream_hourly_raw (
                station_id VARCHAR(8) NOT NULL,
                model VARCHAR(32) NOT NULL,
                asof_utc DATETIME NOT NULL,
                forecasted_at DATETIME NOT NULL,
                forecasted_time DATETIME NOT NULL,
                lat DOUBLE,
                lon DOUBLE,
                var_key VARCHAR(64) NOT NULL,
                value DOUBLE,
                ingested_at DATETIME NOT NULL,
                PRIMARY KEY (station_id, model, asof_utc, forecasted_time, var_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_features_daily (
                station_id VARCHAR(8) NOT NULL,
                target_date_local DATE NOT NULL,
                asof_utc DATETIME NOT NULL,
                gfs_tmax_f DOUBLE,
                hrrr_tmax_f DOUBLE,
                rap_tmax_f DOUBLE,
                nbm_tmax_f DOUBLE,
                gefsatmosmean_tmax_f DOUBLE,
                gefsatmos_tmp_spread_f DOUBLE,
                mos_gfs_tmax_f DOUBLE,
                mos_nam_tmax_f DOUBLE,
                gfs_n_x_max DOUBLE,
                nam_n_x_max DOUBLE,
                ingested_at DATETIME NOT NULL,
                PRIMARY KEY (station_id, target_date_local, asof_utc)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_predictions (
                station_id VARCHAR(8) NOT NULL,
                target_date_local DATE NOT NULL,
                asof_utc DATETIME NOT NULL,
                mu_hat_f DOUBLE,
                sigma_hat_f DOUBLE,
                sigma_emos_f DOUBLE,
                emos_c DOUBLE,
                emos_d DOUBLE,
                rolling_bias_45 DOUBLE,
                rolling_rmse_45 DOUBLE,
                created_at DATETIME NOT NULL,
                PRIMARY KEY (station_id, target_date_local, asof_utc)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_truth_cli (
                station_id VARCHAR(8) NOT NULL,
                target_date_local DATE NOT NULL,
                actual_tmax_f DOUBLE,
                observed_at DATETIME NOT NULL,
                source VARCHAR(64),
                PRIMARY KEY (station_id, target_date_local)
            )
            """,
        ]
        with self._engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))

    def upsert_gribstream_hourly_raw(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        sql = """
            INSERT INTO live_gribstream_hourly_raw
            (station_id, model, asof_utc, forecasted_at, forecasted_time, lat, lon, var_key, value, ingested_at)
            VALUES
            (:station_id, :model, :asof_utc, :forecasted_at, :forecasted_time, :lat, :lon, :var_key, :value, :ingested_at)
            ON DUPLICATE KEY UPDATE
                forecasted_at=VALUES(forecasted_at),
                lat=VALUES(lat),
                lon=VALUES(lon),
                value=VALUES(value),
                ingested_at=VALUES(ingested_at)
        """
        with self._engine.begin() as conn:
            conn.execute(text(sql), rows)

    def upsert_live_features_daily(self, row: dict) -> None:
        sql = """
            INSERT INTO live_features_daily
            (station_id, target_date_local, asof_utc, gfs_tmax_f, hrrr_tmax_f, rap_tmax_f, nbm_tmax_f,
             gefsatmosmean_tmax_f, gefsatmos_tmp_spread_f, mos_gfs_tmax_f, mos_nam_tmax_f,
             gfs_n_x_max, nam_n_x_max, ingested_at)
            VALUES
            (:station_id, :target_date_local, :asof_utc, :gfs_tmax_f, :hrrr_tmax_f, :rap_tmax_f, :nbm_tmax_f,
             :gefsatmosmean_tmax_f, :gefsatmos_tmp_spread_f, :mos_gfs_tmax_f, :mos_nam_tmax_f,
             :gfs_n_x_max, :nam_n_x_max, :ingested_at)
            ON DUPLICATE KEY UPDATE
                gfs_tmax_f=VALUES(gfs_tmax_f),
                hrrr_tmax_f=VALUES(hrrr_tmax_f),
                rap_tmax_f=VALUES(rap_tmax_f),
                nbm_tmax_f=VALUES(nbm_tmax_f),
                gefsatmosmean_tmax_f=VALUES(gefsatmosmean_tmax_f),
                gefsatmos_tmp_spread_f=VALUES(gefsatmos_tmp_spread_f),
                mos_gfs_tmax_f=VALUES(mos_gfs_tmax_f),
                mos_nam_tmax_f=VALUES(mos_nam_tmax_f),
                gfs_n_x_max=VALUES(gfs_n_x_max),
                nam_n_x_max=VALUES(nam_n_x_max),
                ingested_at=VALUES(ingested_at)
        """
        with self._engine.begin() as conn:
            conn.execute(text(sql), row)

    def upsert_live_predictions(self, row: dict) -> None:
        sql = """
            INSERT INTO live_predictions
            (station_id, target_date_local, asof_utc, mu_hat_f, sigma_hat_f, sigma_emos_f,
             emos_c, emos_d, rolling_bias_45, rolling_rmse_45, created_at)
            VALUES
            (:station_id, :target_date_local, :asof_utc, :mu_hat_f, :sigma_hat_f, :sigma_emos_f,
             :emos_c, :emos_d, :rolling_bias_45, :rolling_rmse_45, :created_at)
            ON DUPLICATE KEY UPDATE
                mu_hat_f=VALUES(mu_hat_f),
                sigma_hat_f=VALUES(sigma_hat_f),
                sigma_emos_f=VALUES(sigma_emos_f),
                emos_c=VALUES(emos_c),
                emos_d=VALUES(emos_d),
                rolling_bias_45=VALUES(rolling_bias_45),
                rolling_rmse_45=VALUES(rolling_rmse_45),
                created_at=VALUES(created_at)
        """
        with self._engine.begin() as conn:
            conn.execute(text(sql), row)

    def upsert_live_truth_cli(self, row: dict) -> None:
        sql = """
            INSERT INTO live_truth_cli
            (station_id, target_date_local, actual_tmax_f, observed_at, source)
            VALUES
            (:station_id, :target_date_local, :actual_tmax_f, :observed_at, :source)
            ON DUPLICATE KEY UPDATE
                actual_tmax_f=VALUES(actual_tmax_f),
                observed_at=VALUES(observed_at),
                source=VALUES(source)
        """
        with self._engine.begin() as conn:
            conn.execute(text(sql), row)

    def upsert_cli_daily(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        sql = """
            INSERT INTO cli_daily
            (station_id, target_date_local, tmax_f, tmin_f, report_issued_at_utc, truth_source_url,
             raw_payload_hash, retrieved_at_utc, updated_at_utc)
            VALUES
            (:station_id, :target_date_local, :tmax_f, :tmin_f, :report_issued_at_utc, :truth_source_url,
             :raw_payload_hash, :retrieved_at_utc, :updated_at_utc)
            ON DUPLICATE KEY UPDATE
                tmax_f=VALUES(tmax_f),
                tmin_f=VALUES(tmin_f),
                report_issued_at_utc=VALUES(report_issued_at_utc),
                truth_source_url=VALUES(truth_source_url),
                raw_payload_hash=VALUES(raw_payload_hash),
                retrieved_at_utc=VALUES(retrieved_at_utc),
                updated_at_utc=VALUES(updated_at_utc)
        """
        with self._engine.begin() as conn:
            conn.execute(text(sql), rows)

    def upsert_mos_daily_values(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        sql = """
            INSERT INTO mos_daily_value
            (station_id, station_zoneid, model, asof_utc, runtime_utc, target_date_local, variable_code,
             value_min, value_max, value_mean, value_median, sample_count,
             first_forecast_time_utc, last_forecast_time_utc, raw_payload_hash_ref, retrieved_at_utc)
            VALUES
            (:station_id, :station_zoneid, :model, :asof_utc, :runtime_utc, :target_date_local, :variable_code,
             :value_min, :value_max, :value_mean, :value_median, :sample_count,
             :first_forecast_time_utc, :last_forecast_time_utc, :raw_payload_hash_ref, :retrieved_at_utc)
            ON DUPLICATE KEY UPDATE
                station_zoneid=VALUES(station_zoneid),
                asof_utc=VALUES(asof_utc),
                value_min=VALUES(value_min),
                value_max=VALUES(value_max),
                value_mean=VALUES(value_mean),
                value_median=VALUES(value_median),
                sample_count=VALUES(sample_count),
                first_forecast_time_utc=VALUES(first_forecast_time_utc),
                last_forecast_time_utc=VALUES(last_forecast_time_utc),
                raw_payload_hash_ref=VALUES(raw_payload_hash_ref),
                retrieved_at_utc=VALUES(retrieved_at_utc)
        """
        with self._engine.begin() as conn:
            conn.execute(text(sql), rows)

    def upsert_gribstream_daily_feature(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        sql = """
            INSERT INTO gribstream_daily_feature
            (station_id, zone_id, target_date_local, asof_utc, model_code, metric,
             value_f, value_k, source_forecasted_at_utc, window_start_utc, window_end_utc,
             min_horizon_hours, max_horizon_hours, request_json, request_sha256, response_sha256,
             retrieved_at_utc, notes)
            VALUES
            (:station_id, :zone_id, :target_date_local, :asof_utc, :model_code, :metric,
             :value_f, :value_k, :source_forecasted_at_utc, :window_start_utc, :window_end_utc,
             :min_horizon_hours, :max_horizon_hours, :request_json, :request_sha256, :response_sha256,
             :retrieved_at_utc, :notes)
            ON DUPLICATE KEY UPDATE
                zone_id=VALUES(zone_id),
                value_f=VALUES(value_f),
                value_k=VALUES(value_k),
                source_forecasted_at_utc=VALUES(source_forecasted_at_utc),
                window_start_utc=VALUES(window_start_utc),
                window_end_utc=VALUES(window_end_utc),
                min_horizon_hours=VALUES(min_horizon_hours),
                max_horizon_hours=VALUES(max_horizon_hours),
                request_json=VALUES(request_json),
                request_sha256=VALUES(request_sha256),
                response_sha256=VALUES(response_sha256),
                retrieved_at_utc=VALUES(retrieved_at_utc),
                notes=VALUES(notes)
        """
        with self._engine.begin() as conn:
            conn.execute(text(sql), rows)

    def fetch_features_history(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        asof_utc: datetime,
    ) -> pd.DataFrame:
        sql = """
            SELECT f.*
            FROM live_features_daily f
            JOIN (
                SELECT target_date_local, MAX(asof_utc) AS max_asof
                FROM live_features_daily
                WHERE station_id = :station_id
                  AND target_date_local BETWEEN :start_date AND :end_date
                  AND asof_utc <= :asof_utc
                GROUP BY target_date_local
            ) sel
            ON f.station_id = :station_id
            AND f.target_date_local = sel.target_date_local
            AND f.asof_utc = sel.max_asof
        """
        params = {
            "station_id": station_id,
            "start_date": start_date,
            "end_date": end_date,
            "asof_utc": asof_utc,
        }
        df = pd.read_sql(text(sql), self._engine, params=params)
        return _coerce_date_column(df, "target_date_local")

    def fetch_predictions_history(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        asof_utc: datetime,
    ) -> pd.DataFrame:
        sql = """
            SELECT p.*
            FROM live_predictions p
            JOIN (
                SELECT target_date_local, MAX(asof_utc) AS max_asof
                FROM live_predictions
                WHERE station_id = :station_id
                  AND target_date_local BETWEEN :start_date AND :end_date
                  AND asof_utc <= :asof_utc
                GROUP BY target_date_local
            ) sel
            ON p.station_id = :station_id
            AND p.target_date_local = sel.target_date_local
            AND p.asof_utc = sel.max_asof
        """
        params = {
            "station_id": station_id,
            "start_date": start_date,
            "end_date": end_date,
            "asof_utc": asof_utc,
        }
        df = pd.read_sql(text(sql), self._engine, params=params)
        return _coerce_date_column(df, "target_date_local")

    def fetch_predictions_range(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        sql = """
            SELECT station_id, target_date_local, asof_utc, mu_hat_f, sigma_hat_f
            FROM live_predictions
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_date AND :end_date
        """
        params = {"station_id": station_id, "start_date": start_date, "end_date": end_date}
        df = pd.read_sql(text(sql), self._engine, params=params)
        if df.empty:
            return df
        df = _coerce_date_column(df, "target_date_local")
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
        return df

    def fetch_truth_history(self, station_id: str, start_date: date, end_date: date) -> pd.DataFrame:
        sql = """
            SELECT *
            FROM live_truth_cli
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_date AND :end_date
        """
        params = {"station_id": station_id, "start_date": start_date, "end_date": end_date}
        df = pd.read_sql(text(sql), self._engine, params=params)
        return _coerce_date_column(df, "target_date_local")

    def fetch_cli_truth_history(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        sql = """
            SELECT station_id, target_date_local, tmax_f
            FROM cli_daily
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_date AND :end_date
        """
        params = {"station_id": station_id, "start_date": start_date, "end_date": end_date}
        df = pd.read_sql(text(sql), self._engine, params=params)
        df = _coerce_date_column(df, "target_date_local")
        if "tmax_f" in df.columns:
            df = df.rename(columns={"tmax_f": "actual_tmax_f"})
        return df

    def fetch_gribstream_daily_feature_history(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        asof_utc: datetime,
    ) -> pd.DataFrame:
        sql = """
            SELECT station_id, target_date_local, asof_utc, model_code, metric, value_f
            FROM gribstream_daily_feature
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_date AND :end_date
              AND asof_utc <= :asof_utc
              AND model_code IN ('nbm', 'hrrr', 'rap', 'gefsatmosmean', 'gefsatmos')
              AND metric IN ('TMAX_F', 'TMP_SPREAD_F')
        """
        params = {
            "station_id": station_id,
            "start_date": start_date,
            "end_date": end_date,
            "asof_utc": asof_utc,
        }
        df = pd.read_sql(text(sql), self._engine, params=params)
        if df.empty:
            return df
        df = _coerce_date_column(df, "target_date_local")
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)

        def expected_asof(day: date) -> datetime:
            return datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc) - timedelta(days=1)

        expected_map = {day: expected_asof(day) for day in df["target_date_local"].unique()}
        df = df[df["asof_utc"] == df["target_date_local"].map(expected_map)]
        if df.empty:
            return df

        feature_map = {
            ("nbm", "TMAX_F"): "nbm_tmax_f",
            ("hrrr", "TMAX_F"): "hrrr_tmax_f",
            ("rap", "TMAX_F"): "rap_tmax_f",
            ("gefsatmosmean", "TMAX_F"): "gefsatmosmean_tmax_f",
            ("gefsatmos", "TMP_SPREAD_F"): "gefsatmos_tmp_spread_f",
        }
        rows = {}
        for _, row in df.iterrows():
            key = (
                row["station_id"],
                row["target_date_local"],
                row["asof_utc"],
            )
            if key not in rows:
                rows[key] = {
                    "station_id": row["station_id"],
                    "target_date_local": row["target_date_local"],
                    "asof_utc": row["asof_utc"],
                }
            column = feature_map.get((str(row["model_code"]).lower(), str(row["metric"])))
            if column:
                rows[key][column] = row["value_f"]
        return pd.DataFrame(rows.values())

    def fetch_mos_n_x_history(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        asof_utc: datetime,
    ) -> pd.DataFrame:
        dates = pd.date_range(start_date, end_date, freq="D").date
        sql = """
            SELECT value_max
            FROM mos_daily_value
            WHERE station_id = :station_id
              AND model = :model
              AND target_date_local = :target_date_local
              AND variable_code = 'n_x'
              AND runtime_utc <= :runtime_utc
            ORDER BY runtime_utc DESC
            LIMIT 1
        """
        rows = []
        with self._engine.begin() as conn:
            for target_date in dates:
                expected_asof = datetime(
                    target_date.year, target_date.month, target_date.day, 12, 0, tzinfo=timezone.utc
                ) - timedelta(days=1)
                if expected_asof > asof_utc:
                    continue
                params_base = {
                    "station_id": station_id,
                    "target_date_local": target_date,
                    "runtime_utc": expected_asof.replace(tzinfo=None),
                }
                gfs = conn.execute(text(sql), {**params_base, "model": "GFS"}).scalar()
                nam = conn.execute(text(sql), {**params_base, "model": "NAM"}).scalar()
                rows.append(
                    {
                        "station_id": station_id,
                        "target_date_local": target_date,
                        "asof_utc": expected_asof,
                        "gfs_n_x_max": float(gfs) if gfs is not None else None,
                        "nam_n_x_max": float(nam) if nam is not None else None,
                    }
                )
        return pd.DataFrame(rows)

    def fetch_station_registry(self, station_id: str) -> dict | None:
        sql = """
            SELECT station_id, zone_id, issuedby, wfo_site
            FROM station_registry
            WHERE station_id = :station_id
        """
        params = {"station_id": station_id}
        with self._engine.begin() as conn:
            row = conn.execute(text(sql), params).mappings().first()
            return dict(row) if row else None


def _coerce_date_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column]).dt.date
    return df
