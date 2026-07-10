from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


logger = logging.getLogger("ml_live.db.csv")


DATE_COLUMNS = {
    "target_date_local",
}

DATETIME_COLUMNS = {
    "asof_utc",
    "forecasted_at",
    "forecasted_time",
    "ingested_at",
    "created_at",
    "observed_at",
    "report_issued_at_utc",
    "retrieved_at_utc",
    "updated_at_utc",
    "runtime_utc",
    "first_forecast_time_utc",
    "last_forecast_time_utc",
    "source_forecasted_at_utc",
    "window_start_utc",
    "window_end_utc",
}

TABLE_COLUMNS: dict[str, list[str]] = {
    "live_gribstream_hourly_raw": [
        "station_id",
        "model",
        "asof_utc",
        "forecasted_at",
        "forecasted_time",
        "lat",
        "lon",
        "var_key",
        "value",
        "ingested_at",
    ],
    "live_features_daily": [
        "station_id",
        "target_date_local",
        "asof_utc",
        "gfs_tmax_f",
        "hrrr_tmax_f",
        "rap_tmax_f",
        "nbm_tmax_f",
        "gefsatmosmean_tmax_f",
        "gefsatmos_tmp_spread_f",
        "mos_gfs_tmax_f",
        "mos_nam_tmax_f",
        "gfs_n_x_max",
        "nam_n_x_max",
        "ingested_at",
    ],
    "live_predictions": [
        "station_id",
        "target_date_local",
        "asof_utc",
        "mu_hat_f",
        "sigma_hat_f",
        "sigma_emos_f",
        "emos_c",
        "emos_d",
        "rolling_bias_45",
        "rolling_rmse_45",
        "created_at",
    ],
    "live_truth_cli": [
        "station_id",
        "target_date_local",
        "actual_tmax_f",
        "observed_at",
        "source",
    ],
    "cli_daily": [
        "station_id",
        "target_date_local",
        "tmax_f",
        "tmin_f",
        "report_issued_at_utc",
        "truth_source_url",
        "raw_payload_hash",
        "retrieved_at_utc",
        "updated_at_utc",
    ],
    "mos_daily_value": [
        "station_id",
        "station_zoneid",
        "model",
        "asof_utc",
        "runtime_utc",
        "target_date_local",
        "variable_code",
        "value_min",
        "value_max",
        "value_mean",
        "value_median",
        "sample_count",
        "first_forecast_time_utc",
        "last_forecast_time_utc",
        "raw_payload_hash_ref",
        "retrieved_at_utc",
    ],
    "gribstream_daily_feature": [
        "station_id",
        "zone_id",
        "target_date_local",
        "asof_utc",
        "model_code",
        "metric",
        "value_f",
        "value_k",
        "source_forecasted_at_utc",
        "window_start_utc",
        "window_end_utc",
        "min_horizon_hours",
        "max_horizon_hours",
        "request_json",
        "request_sha256",
        "response_sha256",
        "retrieved_at_utc",
        "notes",
    ],
    "station_registry": [
        "station_id",
        "zone_id",
        "issuedby",
        "wfo_site",
    ],
}

PRIMARY_KEYS: dict[str, list[str]] = {
    "live_gribstream_hourly_raw": [
        "station_id",
        "model",
        "asof_utc",
        "forecasted_time",
        "var_key",
    ],
    "live_features_daily": [
        "station_id",
        "target_date_local",
        "asof_utc",
    ],
    "live_predictions": [
        "station_id",
        "target_date_local",
        "asof_utc",
    ],
    "live_truth_cli": [
        "station_id",
        "target_date_local",
    ],
    "cli_daily": [
        "station_id",
        "target_date_local",
    ],
    "mos_daily_value": [
        "station_id",
        "model",
        "target_date_local",
        "variable_code",
        "runtime_utc",
    ],
    "gribstream_daily_feature": [
        "station_id",
        "target_date_local",
        "asof_utc",
        "model_code",
        "metric",
    ],
    "station_registry": [
        "station_id",
    ],
}


class CsvStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def _table_path(self, table: str) -> Path:
        return self._base_dir / f"{table}.csv"

    def _normalize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date
        for col in DATETIME_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        return df

    def _format_frame_for_write(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in DATE_COLUMNS:
            if col in out.columns:
                out[col] = out[col].apply(
                    lambda value: value.isoformat() if pd.notna(value) else None
                )
        for col in DATETIME_COLUMNS:
            if col in out.columns:
                series = pd.to_datetime(out[col], utc=True, errors="coerce")
                out[col] = series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return out

    def _preferred_columns(self, table: str, df_existing: pd.DataFrame, df_new: pd.DataFrame) -> list[str]:
        base = TABLE_COLUMNS.get(table, [])
        extras = sorted(
            {col for col in df_existing.columns}.union(df_new.columns) - set(base)
        )
        return base + extras

    def _read_table(self, table: str) -> pd.DataFrame:
        path = self._table_path(table)
        columns = TABLE_COLUMNS.get(table, [])
        if not path.exists():
            return pd.DataFrame(columns=columns)
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame(columns=columns)
        df = self._normalize_frame(df)
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def _write_table(self, table: str, df: pd.DataFrame) -> None:
        path = self._table_path(table)
        df_out = self._format_frame_for_write(df)
        df_out.to_csv(path, index=False)

    def _upsert(self, table: str, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        df_new = pd.DataFrame(rows)
        if df_new.empty:
            return
        df_existing = self._read_table(table)
        columns = self._preferred_columns(table, df_existing, df_new)
        for col in columns:
            if col not in df_existing.columns:
                df_existing[col] = pd.NA
            if col not in df_new.columns:
                df_new[col] = pd.NA
        df_existing = self._normalize_frame(df_existing)
        df_new = self._normalize_frame(df_new)
        pk_cols = PRIMARY_KEYS.get(table, [])
        if pk_cols:
            df_new = df_new.drop_duplicates(subset=pk_cols, keep="last")
            df_existing = df_existing.set_index(pk_cols, drop=False) if not df_existing.empty else df_existing
            df_new = df_new.set_index(pk_cols, drop=False)
            if not df_existing.empty:
                df_existing.update(df_new)
                new_only = df_new.loc[~df_new.index.isin(df_existing.index)]
                df_combined = pd.concat([df_existing, new_only], axis=0)
            else:
                df_combined = df_new
            df_combined = df_combined.reset_index(drop=True)
        else:
            df_combined = pd.concat([df_existing, df_new], axis=0).reset_index(drop=True)
        df_combined = df_combined[columns]
        self._write_table(table, df_combined)

    def upsert_gribstream_hourly_raw(self, rows: Iterable[dict]) -> None:
        self._upsert("live_gribstream_hourly_raw", rows)

    def upsert_live_features_daily(self, row: dict) -> None:
        self._upsert("live_features_daily", [row])

    def upsert_live_predictions(self, row: dict) -> None:
        self._upsert("live_predictions", [row])

    def upsert_live_truth_cli(self, row: dict) -> None:
        self._upsert("live_truth_cli", [row])

    def upsert_cli_daily(self, rows: Iterable[dict]) -> None:
        self._upsert("cli_daily", rows)

    def upsert_mos_daily_values(self, rows: Iterable[dict]) -> None:
        self._upsert("mos_daily_value", rows)

    def upsert_gribstream_daily_feature(self, rows: Iterable[dict]) -> None:
        self._upsert("gribstream_daily_feature", rows)

    def fetch_features_history(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        asof_utc: datetime,
    ) -> pd.DataFrame:
        df = self._read_table("live_features_daily")
        if df.empty:
            return df
        df = df[
            (df["station_id"] == station_id)
            & (df["target_date_local"] >= start_date)
            & (df["target_date_local"] <= end_date)
        ]
        if df.empty:
            return df
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
        df = df[df["asof_utc"] <= asof_utc]
        if df.empty:
            return df
        df["max_asof"] = df.groupby("target_date_local")["asof_utc"].transform("max")
        df = df[df["asof_utc"] == df["max_asof"]].drop(columns=["max_asof"])
        return _coerce_date_column(df, "target_date_local")

    def fetch_predictions_history(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        asof_utc: datetime,
    ) -> pd.DataFrame:
        df = self._read_table("live_predictions")
        if df.empty:
            return df
        df = df[
            (df["station_id"] == station_id)
            & (df["target_date_local"] >= start_date)
            & (df["target_date_local"] <= end_date)
        ]
        if df.empty:
            return df
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
        df = df[df["asof_utc"] <= asof_utc]
        if df.empty:
            return df
        df["max_asof"] = df.groupby("target_date_local")["asof_utc"].transform("max")
        df = df[df["asof_utc"] == df["max_asof"]].drop(columns=["max_asof"])
        return _coerce_date_column(df, "target_date_local")

    def fetch_predictions_range(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        df = self._read_table("live_predictions")
        if df.empty:
            return df
        df = df[
            (df["station_id"] == station_id)
            & (df["target_date_local"] >= start_date)
            & (df["target_date_local"] <= end_date)
        ]
        if df.empty:
            return df
        df = _coerce_date_column(df, "target_date_local")
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
        cols = ["station_id", "target_date_local", "asof_utc", "mu_hat_f", "sigma_hat_f"]
        return df[[col for col in cols if col in df.columns]]

    def fetch_truth_history(self, station_id: str, start_date: date, end_date: date) -> pd.DataFrame:
        df = self._read_table("live_truth_cli")
        if df.empty:
            return df
        df = df[
            (df["station_id"] == station_id)
            & (df["target_date_local"] >= start_date)
            & (df["target_date_local"] <= end_date)
        ]
        return _coerce_date_column(df, "target_date_local")

    def fetch_cli_truth_history(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        df = self._read_table("cli_daily")
        if df.empty:
            return df
        df = df[
            (df["station_id"] == station_id)
            & (df["target_date_local"] >= start_date)
            & (df["target_date_local"] <= end_date)
        ]
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
        df = self._read_table("gribstream_daily_feature")
        if df.empty:
            return df
        df = df[
            (df["station_id"] == station_id)
            & (df["target_date_local"] >= start_date)
            & (df["target_date_local"] <= end_date)
        ]
        if df.empty:
            return df
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
        df = df[df["asof_utc"] <= asof_utc]
        if df.empty:
            return df
        df = df[
            df["model_code"].str.lower().isin(
                ["nbm", "hrrr", "rap", "gefsatmosmean", "gefsatmos"]
            )
            & df["metric"].isin(["TMAX_F", "TMP_SPREAD_F"])
        ]
        if df.empty:
            return df

        def expected_asof(day: date) -> datetime:
            return datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc) - timedelta(days=1)

        df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
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
        rows: dict[tuple, dict] = {}
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
        df = self._read_table("mos_daily_value")
        if df.empty:
            return df
        df = df[
            (df["station_id"] == station_id)
            & (df["variable_code"] == "n_x")
            & (df["target_date_local"] >= start_date)
            & (df["target_date_local"] <= end_date)
        ]
        if df.empty:
            return df
        df["runtime_utc"] = pd.to_datetime(df["runtime_utc"], utc=True)
        df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
        rows = []
        for target_date in pd.date_range(start_date, end_date, freq="D").date:
            expected_asof = datetime(
                target_date.year, target_date.month, target_date.day, 12, 0, tzinfo=timezone.utc
            ) - timedelta(days=1)
            if expected_asof > asof_utc:
                continue
            row_base = {
                "station_id": station_id,
                "target_date_local": target_date,
                "asof_utc": expected_asof,
                "gfs_n_x_max": None,
                "nam_n_x_max": None,
            }
            for model, key in [("GFS", "gfs_n_x_max"), ("NAM", "nam_n_x_max")]:
                subset = df[
                    (df["model"] == model)
                    & (df["target_date_local"] == target_date)
                    & (df["runtime_utc"] <= expected_asof)
                ]
                if subset.empty:
                    continue
                latest = subset.sort_values("runtime_utc", ascending=False).iloc[0]
                value = latest.get("value_max")
                row_base[key] = float(value) if pd.notna(value) else None
            rows.append(row_base)
        return pd.DataFrame(rows)

    def fetch_station_registry(self, station_id: str) -> dict | None:
        df = self._read_table("station_registry")
        if df.empty:
            return None
        df = df[df["station_id"] == station_id]
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        return {key: row.get(key) for key in ["station_id", "zone_id", "issuedby", "wfo_site"]}


def _coerce_date_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column]).dt.date
    return df
