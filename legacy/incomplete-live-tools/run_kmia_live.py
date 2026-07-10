from __future__ import annotations

# PowerShell one-liner (CSV mode):
# $repo="C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets"; $env:PYTHONPATH=$repo; $env:MYSQL_HOST="localhost"; $env:MYSQL_PORT="3306"; $env:MYSQL_DB="weather_predictionmarkets"; $env:MYSQL_USER="root"; $env:MYSQL_PASSWORD="root"; $env:GRIBSTREAM_TOKEN="a379b700999ff2263bc5816c3ca21d25427ce5f7"; $env:GRIBSTREAM_ACCEPT="application/ndjson"; python "C:\Users\ahmad\Documents\weather\run_kmia_live.py" --station KMIA --target-date 20260211 --verbose --csv

import argparse
import json
import os
import math
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from importlib.util import find_spec


def _ensure_dependencies() -> None:
    required = {
        "joblib": "joblib",
        "numpy": "numpy",
        "pandas": "pandas",
        "yaml": "PyYAML",
        "requests": "requests",
        "sqlalchemy": "SQLAlchemy",
        "pymysql": "pymysql",
        "lightgbm": "lightgbm",
        "sklearn": "scikit-learn",
        "tzdata": "tzdata",
    }
    missing = [pip_name for module, pip_name in required.items() if find_spec(module) is None]
    if not missing:
        return
    print(f"Missing dependencies detected: {', '.join(missing)}", file=sys.stderr)
    print("Attempting to install missing dependencies...", file=sys.stderr)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_ensure_dependencies()

import joblib
import numpy as np
import pandas as pd
import yaml

from ml_live.calibration.emos_w45 import calibrate
from ml_live.db.csv_store import CsvStore
from ml_live.db.mysql import MysqlConfig, MySqlStore
from ml_live.features.e92_features import build_feature_vector
from zoneinfo import ZoneInfo

from ml_live.fetch.gribstream import (
    DailyValue,
    compute_daily_tmax,
    compute_gefs_mean_tmax,
    compute_gefs_spread,
    fetch_forecast_tmp_with_raw,
    resolve_station_coordinates,
)
from ml_live.fetch.iem_cli import fetch_cli_year
from ml_live.fetch.iem_mos import (
    build_daily_rows_for_runtime,
    fetch_mos_payload,
    mos_window_utc,
    select_runtime,
)
from ml_live.modeling import artifacts as artifact_utils
from ml_live.runtime.clock import (
    asof_from_target_date,
    build_clock,
    parse_target_date,
    resolve_asof_utc,
    standard_time_window_utc,
)
from ml_live.runtime.logging import configure_logging
from ml_live.runtime.paths import config_path, models_dir, repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KMIA live pipeline.")
    parser.add_argument("--station", default="KMIA", help="Station ID (default: KMIA).")
    parser.add_argument("--target-date", help="Target date local (YYYYMMDD or YYYY-MM-DD).")
    parser.add_argument("--asof-utc", help="Override as-of UTC (ISO format).")
    parser.add_argument("--config", help="Path to live_kmia.yaml config.")
    parser.add_argument("--thresholds", help="Comma-separated integer thresholds.")
    parser.add_argument(
        "--store",
        choices=["mysql", "csv"],
        default="mysql",
        help="Storage backend (default: mysql).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Use CSV store instead of MySQL.",
    )
    parser.add_argument(
        "--csv-dir",
        help="CSV store directory (default: artifacts/live_store).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--output-json",
        help="Optional path to write the final structured result JSON.",
    )
    parser.add_argument(
        "--stdout-json",
        choices=["full", "none"],
        default="full",
        help="Emit structured result JSON to stdout (default: full).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_yaml_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _find_station_meta(config: dict, station_id: str) -> dict | None:
    stations = (config.get("gribstream") or {}).get("stations") or []
    for station in stations:
        sid = station.get("stationId")
        if sid and sid.upper() == station_id.upper():
            return station
    return None


def _resolve_tmax_setting(config: dict, ingestion_cfg: dict) -> dict:
    live_cfg = config.get("gribstream") or {}
    tmax_cfg = live_cfg.get("tmax_forecast") or live_cfg.get("tmax-forecast") or {}
    if tmax_cfg:
        return tmax_cfg
    return ((ingestion_cfg.get("gribstream") or {}).get("tmax-forecast")) or {}


def parse_asof_utc(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    return datetime.fromisoformat(cleaned).astimezone(timezone.utc)


def parse_thresholds(value: str | None) -> list[int]:
    if not value:
        return []
    thresholds = []
    for token in value.split(","):
        token = token.strip()
        if token:
            thresholds.append(int(token))
    return thresholds


REQUIRED_HISTORY_COLUMNS = [
    "nbm_tmax_f",
    "hrrr_tmax_f",
    "rap_tmax_f",
    "gefsatmosmean_tmax_f",
    "gefsatmos_tmp_spread_f",
    "gfs_n_x_max",
    "nam_n_x_max",
]


def _bootstrap_history_from_dataset(
    store: MySqlStore,
    station_id: str,
    logger,
) -> bool:
    try:
        e92_dir = artifact_utils.find_e92_run_dir()
        dataset_path = artifact_utils.resolve_dataset_path(e92_dir)
    except FileNotFoundError as exc:
        logger.warning("Bootstrap skipped: %s", exc)
        return False

    df = pd.read_parquet(dataset_path)
    df = df[df["station_id"].str.upper() == station_id.upper()]
    if df.empty:
        logger.warning("Bootstrap skipped: no dataset rows for station=%s", station_id)
        return False

    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    ingested_at = datetime.now(timezone.utc)

    for _, row in df.iterrows():
        features_row = {
            "station_id": station_id,
            "target_date_local": row["target_date_local"],
            "asof_utc": row["asof_utc"].to_pydatetime(),
            "gfs_tmax_f": None,
            "hrrr_tmax_f": row.get("hrrr_tmax_f"),
            "rap_tmax_f": row.get("rap_tmax_f"),
            "nbm_tmax_f": row.get("nbm_tmax_f"),
            "gefsatmosmean_tmax_f": row.get("gefsatmosmean_tmax_f"),
            "gefsatmos_tmp_spread_f": row.get("gefsatmos_tmp_spread_f"),
            "mos_gfs_tmax_f": row.get("gfs_n_x_max"),
            "mos_nam_tmax_f": row.get("nam_n_x_max"),
            "gfs_n_x_max": row.get("gfs_n_x_max"),
            "nam_n_x_max": row.get("nam_n_x_max"),
            "ingested_at": ingested_at,
        }
        store.upsert_live_features_daily(features_row)

        actual_tmax_f = row.get("actual_tmax_f")
        if actual_tmax_f is not None and not pd.isna(actual_tmax_f):
            observed_at = row["asof_utc"].to_pydatetime() + timedelta(days=1)
            truth_row = {
                "station_id": station_id,
                "target_date_local": row["target_date_local"],
                "actual_tmax_f": float(actual_tmax_f),
                "observed_at": observed_at,
                "source": "training_dataset",
            }
            store.upsert_live_truth_cli(truth_row)

    logger.info("Bootstrapped history from dataset=%s rows=%d", dataset_path, len(df))
    return True


def _load_history_from_core_tables(
    store: MySqlStore,
    station_id: str,
    start_date: date,
    end_date: date,
    asof_utc: datetime,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grib = store.fetch_gribstream_daily_feature_history(station_id, start_date, end_date, asof_utc)
    mos = store.fetch_mos_n_x_history(station_id, start_date, end_date, asof_utc)
    if grib.empty and mos.empty:
        logger.warning("Core table history is empty for station=%s", station_id)
        return pd.DataFrame(), pd.DataFrame()
    if grib.empty:
        features = mos
    elif mos.empty:
        features = grib
    else:
        features = grib.merge(
            mos,
            on=["station_id", "target_date_local", "asof_utc"],
            how="left",
        )
    truth = store.fetch_cli_truth_history(station_id, start_date, end_date)
    return features, truth


def _required_dates(start_date: date, end_date: date) -> list[date]:
    dates: list[date] = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current = current + timedelta(days=1)
    return dates


def _find_missing_history_dates(
    history_features: pd.DataFrame,
    history_truth: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[list[date], list[date]]:
    required = set(_required_dates(start_date, end_date))
    truth_dates = set()
    if not history_truth.empty and "target_date_local" in history_truth.columns:
        truth = history_truth.copy()
        truth["target_date_local"] = pd.to_datetime(truth["target_date_local"]).dt.date
        if "actual_tmax_f" in truth.columns:
            truth = truth[truth["actual_tmax_f"].notna()]
        truth_dates = set(truth["target_date_local"])
    missing_truth = sorted(required - truth_dates)

    if history_features.empty:
        return sorted(required), missing_truth

    features = history_features.copy()
    if "target_date_local" not in features.columns:
        return sorted(required), missing_truth
    features["target_date_local"] = pd.to_datetime(features["target_date_local"]).dt.date

    for col in REQUIRED_HISTORY_COLUMNS:
        if col not in features.columns:
            return sorted(required), missing_truth

    complete = features.dropna(subset=REQUIRED_HISTORY_COLUMNS)
    feature_dates = set(complete["target_date_local"])
    missing_features = sorted(required - feature_dates)
    return missing_features, missing_truth


def _expected_asof_for_date(target_date_local: date) -> datetime:
    return asof_from_target_date(target_date_local)


def _ensure_core_history(
    store: MySqlStore,
    station_id: str,
    start_date: date,
    end_date: date,
    asof_utc: datetime,
    iem_base: str,
    station_registry: dict | None,
    station_zoneid: str,
    zone_id: str,
    grib_base: str,
    token: str,
    grib_accept: str,
    grib_auth_scheme: str,
    min_horizon_hours: int,
    resolve_max_horizon,
    resolve_min_points,
    gefs_members: list[int],
    lat: float,
    lon: float,
    mos_models: list[str],
    deterministic_models: list[str],
    include_gefs_mean: bool,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_features, core_truth = _load_history_from_core_tables(
        store, station_id, start_date, end_date, asof_utc, logger
    )
    missing_features, missing_truth = _find_missing_history_dates(
        core_features, core_truth, start_date, end_date
    )
    if missing_truth:
        logger.warning(
            "Missing CLI truth for %d dates. Backfilling via IEM.", len(missing_truth)
        )
        _backfill_missing_cli_truth(
            store=store,
            station_id=station_id,
            missing_dates=missing_truth,
            iem_base=iem_base,
            station_registry=station_registry,
            logger=logger,
        )
        core_features, core_truth = _load_history_from_core_tables(
            store, station_id, start_date, end_date, asof_utc, logger
        )

    missing_features, missing_truth = _find_missing_history_dates(
        core_features, core_truth, start_date, end_date
    )
    if missing_features:
        logger.warning(
            "Missing history features for %d dates. Backfilling via APIs.", len(missing_features)
        )
        _backfill_missing_core_features(
            store=store,
            station_id=station_id,
            station_zoneid=station_zoneid,
            missing_dates=missing_features,
            zone_id=zone_id,
            grib_base=grib_base,
            token=token,
            grib_accept=grib_accept,
            grib_auth_scheme=grib_auth_scheme,
            min_horizon_hours=min_horizon_hours,
            resolve_max_horizon=resolve_max_horizon,
            resolve_min_points=resolve_min_points,
            gefs_members=gefs_members,
            lat=lat,
            lon=lon,
            iem_base=iem_base,
            mos_models=mos_models,
            deterministic_models=deterministic_models,
            include_gefs_mean=include_gefs_mean,
            logger=logger,
        )
        core_features, core_truth = _load_history_from_core_tables(
            store, station_id, start_date, end_date, asof_utc, logger
        )

    missing_features, missing_truth = _find_missing_history_dates(
        core_features, core_truth, start_date, end_date
    )
    if missing_truth:
        raise ValueError(
            "Missing CLI truth for required dates: "
            + ", ".join([d.isoformat() for d in missing_truth])
        )
    if missing_features:
        raise ValueError(
            "Missing history features after backfill for dates: "
            + ", ".join([d.isoformat() for d in missing_features])
        )
    return core_features, core_truth

def _resolve_truth_source_url(day, station_registry: dict | None) -> str | None:
    if day.truth_source_url:
        return day.truth_source_url
    if not station_registry:
        return None
    site = station_registry.get("wfo_site")
    issuedby = station_registry.get("issuedby")
    if not site or not issuedby:
        return None
    return f"https://forecast.weather.gov/product.php?site={site}&product=CLI&issuedby={issuedby}"


def _backfill_missing_cli_truth(
    store: MySqlStore,
    station_id: str,
    missing_dates: list[date],
    iem_base: str,
    station_registry: dict | None,
    logger,
) -> None:
    if not missing_dates:
        return
    missing_set = set(missing_dates)
    rows: list[dict] = []
    for year in sorted({d.year for d in missing_dates}):
        payload = fetch_cli_year(iem_base, station_id, year)
        retrieved_at = datetime.now(timezone.utc)
        updated_at = retrieved_at
        for day in payload.days:
            if day.target_date_local not in missing_set:
                continue
            rows.append(
                {
                    "station_id": station_id,
                    "target_date_local": day.target_date_local,
                    "tmax_f": day.tmax_f,
                    "tmin_f": day.tmin_f,
                    "report_issued_at_utc": day.report_issued_at_utc,
                    "truth_source_url": _resolve_truth_source_url(day, station_registry),
                    "raw_payload_hash": payload.raw_payload_hash,
                    "retrieved_at_utc": retrieved_at,
                    "updated_at_utc": updated_at,
                }
            )
    if not rows:
        raise ValueError(
            "CLI backfill failed; no records returned for dates: "
            + ", ".join([d.isoformat() for d in missing_dates])
        )
    store.upsert_cli_daily(rows)
    covered = {row["target_date_local"] for row in rows}
    remaining = sorted(missing_set - covered)
    if remaining:
        raise ValueError(
            "CLI backfill missing dates: " + ", ".join([d.isoformat() for d in remaining])
        )
    logger.info("Backfilled CLI truth rows=%d", len(rows))


def _sync_live_truth_cli(
    store: MySqlStore,
    station_id: str,
    start_date: date,
    end_date: date,
    logger,
) -> None:
    truth = store.fetch_cli_truth_history(station_id, start_date, end_date)
    if truth.empty:
        raise ValueError(
            f"Missing CLI truth for station={station_id} range={start_date}..{end_date}"
        )
    now = datetime.now(timezone.utc)
    updated = 0
    for _, row in truth.iterrows():
        value = row.get("actual_tmax_f")
        if value is None or pd.isna(value):
            continue
        store.upsert_live_truth_cli(
            {
                "station_id": station_id,
                "target_date_local": row["target_date_local"],
                "actual_tmax_f": float(value),
                "observed_at": now,
                "source": "cli_daily",
            }
        )
        updated += 1
    logger.info("Synced live_truth_cli rows=%d", updated)


def _find_missing_prediction_dates(
    preds: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> list[date]:
    required = set(_required_dates(start_date, end_date))
    if preds.empty:
        return sorted(required)
    df = preds.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df = df[df["mu_hat_f"].notna() & df["sigma_hat_f"].notna()]
    df["expected_asof"] = df["target_date_local"].apply(_expected_asof_for_date)
    df = df[df["asof_utc"] == df["expected_asof"]]
    present = set(df["target_date_local"])
    return sorted(required - present)


def _backfill_missing_predictions(
    store: MySqlStore,
    station_id: str,
    missing_dates: list[date],
    feature_cols: list[str],
    history_features: pd.DataFrame,
    history_truth: pd.DataFrame,
    mu_model,
    sigma_model,
    logger,
) -> None:
    if not missing_dates:
        return
    history_features = history_features.copy()
    history_truth = history_truth.copy()
    history_features["target_date_local"] = pd.to_datetime(
        history_features["target_date_local"]
    ).dt.date
    history_truth["target_date_local"] = pd.to_datetime(
        history_truth["target_date_local"]
    ).dt.date
    total = 0
    for target_date_local in missing_dates:
        asof_utc = _expected_asof_for_date(target_date_local)
        day_rows = history_features[
            (history_features["target_date_local"] == target_date_local)
            & (history_features["asof_utc"] == asof_utc)
        ]
        if day_rows.empty:
            raise ValueError(
                f"Missing base features for prediction backfill date={target_date_local}"
            )
        base_row = day_rows.iloc[0].to_dict()
        feature_df = build_feature_vector(
            feature_cols,
            base_row,
            history_features,
            history_truth,
            target_date_local,
        )
        mu_hat_f = float(mu_model.predict(feature_df.to_numpy(dtype=float))[0])
        sigma_hat_f = float(max(0.5, sigma_model.predict(feature_df.to_numpy(dtype=float))[0]))
        store.upsert_live_predictions(
            {
                "station_id": station_id,
                "target_date_local": target_date_local,
                "asof_utc": asof_utc,
                "mu_hat_f": mu_hat_f,
                "sigma_hat_f": sigma_hat_f,
                "sigma_emos_f": None,
                "emos_c": None,
                "emos_d": None,
                "rolling_bias_45": None,
                "rolling_rmse_45": None,
                "created_at": datetime.now(timezone.utc),
            }
        )
        total += 1
    logger.info("Backfilled predictions rows=%d", total)

def _fetch_mos_daily_rows(
    iem_base: str,
    station_id: str,
    station_zoneid: str,
    model: str,
    target_date: date,
    asof_utc: datetime,
) -> tuple[list[dict], float]:
    window_start_utc, window_end_utc = mos_window_utc(target_date, station_zoneid)
    request_start_utc = asof_utc if asof_utc < window_start_utc else window_start_utc
    payload = fetch_mos_payload(iem_base, station_id, model, request_start_utc, window_end_utc)
    runtime = select_runtime(
        payload.entries,
        asof_utc,
        window_start_utc,
        window_end_utc,
        ZoneInfo(station_zoneid),
        target_date,
    )
    if runtime is None:
        raise ValueError(
            f"Missing MOS runtime for station={station_id} model={model} target_date={target_date}"
        )
    retrieved_at = datetime.now(timezone.utc)
    rows = build_daily_rows_for_runtime(
        payload,
        station_zoneid,
        asof_utc,
        runtime,
        target_date,
        window_start_utc,
        window_end_utc,
        retrieved_at,
    )
    if not rows:
        raise ValueError(
            f"Missing MOS values for station={station_id} model={model} target_date={target_date}"
        )
    n_x_row = next((row for row in rows if row["variable_code"] == "n_x"), None)
    if n_x_row is None or n_x_row["value_max"] is None:
        raise ValueError(
            f"Missing MOS n_x for station={station_id} model={model} target_date={target_date}"
        )
    return rows, float(n_x_row["value_max"])


def _build_gribstream_row(
    station_id: str,
    zone_id: str,
    target_date: date,
    asof_utc: datetime,
    model_code: str,
    metric: str,
    daily: DailyValue,
    min_horizon_hours: int,
    max_horizon_hours: int,
    meta: dict[str, Any],
    notes: str | None,
) -> dict:
    window_start_utc, window_end_utc = standard_time_window_utc(target_date, zone_id)
    return {
        "station_id": station_id,
        "zone_id": zone_id,
        "target_date_local": target_date,
        "asof_utc": asof_utc,
        "model_code": model_code,
        "metric": metric,
        "value_f": daily.value_f,
        "value_k": daily.value_k,
        "source_forecasted_at_utc": asof_utc,
        "window_start_utc": window_start_utc,
        "window_end_utc": window_end_utc,
        "min_horizon_hours": min_horizon_hours,
        "max_horizon_hours": max_horizon_hours,
        "request_json": meta["request_json"],
        "request_sha256": meta["request_sha256"],
        "response_sha256": meta["response_sha256"],
        "retrieved_at_utc": meta["retrieved_at_utc"],
        "notes": notes,
    }


def _fetch_gribstream_daily_features(
    station_id: str,
    zone_id: str,
    target_date: date,
    asof_utc: datetime,
    lat: float,
    lon: float,
    grib_base: str,
    token: str,
    grib_accept: str,
    grib_auth_scheme: str,
    min_horizon_hours: int,
    resolve_max_horizon,
    resolve_min_points,
    gefs_members: list[int],
    deterministic_models: list[str],
    include_gefs_mean: bool,
    include_hourly_raw: bool,
    logger,
) -> tuple[dict[str, float | None], list[dict], list[dict]]:
    feature_values: dict[str, float | None] = {
        "gfs_tmax_f": None,
        "hrrr_tmax_f": None,
        "rap_tmax_f": None,
        "nbm_tmax_f": None,
        "gefsatmosmean_tmax_f": None,
        "gefsatmos_tmp_spread_f": None,
    }
    grib_rows: list[dict] = []
    hourly_rows: list[dict] = []

    required_models = {"nbm", "hrrr", "rap"}
    for model in deterministic_models:
        try:
            df, meta = fetch_forecast_tmp_with_raw(
                grib_base,
                model,
                token,
                station_id,
                lat,
                lon,
                asof_utc,
                min_horizon=min_horizon_hours,
                max_horizon=resolve_max_horizon(model),
                members=None,
                accept=grib_accept,
                auth_scheme=grib_auth_scheme,
            )
            daily = compute_daily_tmax(df, target_date, zone_id, resolve_min_points(model))
        except Exception as exc:
            if model.lower() in required_models:
                raise
            logger.warning("Skipping optional model=%s error=%s", model, exc)
            continue
        notes = f"pointsUsed={daily.points}"
        if model.lower() == "gfs":
            feature_values["gfs_tmax_f"] = daily.value_f
        elif model.lower() == "hrrr":
            feature_values["hrrr_tmax_f"] = daily.value_f
            grib_rows.append(
                _build_gribstream_row(
                    station_id,
                    zone_id,
                    target_date,
                    asof_utc,
                    "hrrr",
                    "TMAX_F",
                    daily,
                    min_horizon_hours,
                    resolve_max_horizon(model),
                    meta,
                    notes,
                )
            )
        elif model.lower() == "rap":
            feature_values["rap_tmax_f"] = daily.value_f
            grib_rows.append(
                _build_gribstream_row(
                    station_id,
                    zone_id,
                    target_date,
                    asof_utc,
                    "rap",
                    "TMAX_F",
                    daily,
                    min_horizon_hours,
                    resolve_max_horizon(model),
                    meta,
                    notes,
                )
            )
        elif model.lower() == "nbm":
            feature_values["nbm_tmax_f"] = daily.value_f
            grib_rows.append(
                _build_gribstream_row(
                    station_id,
                    zone_id,
                    target_date,
                    asof_utc,
                    "nbm",
                    "TMAX_F",
                    daily,
                    min_horizon_hours,
                    resolve_max_horizon(model),
                    meta,
                    notes,
                )
            )
        logger.info("GribStream tmax model=%s tmax_f=%.2f rows=%d", model, daily.value_f, len(df))

        if include_hourly_raw:
            for _, row in df.iterrows():
                member = int(row.get("member", 0))
                var_key = f"TMP|2 m above ground|member={member}"
                hourly_rows.append(
                    {
                        "station_id": station_id,
                        "model": model,
                        "asof_utc": asof_utc,
                        "forecasted_at": row["forecasted_at"].to_pydatetime(),
                        "forecasted_time": row["forecasted_time"].to_pydatetime(),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "var_key": var_key,
                        "value": float(row["tmp_k"]),
                        "ingested_at": datetime.now(timezone.utc),
                    }
                )

    mean_daily: DailyValue | None = None
    mean_meta: dict[str, Any] | None = None
    mean_notes: str | None = None
    df_mean_for_raw: pd.DataFrame | None = None
    if include_gefs_mean:
        try:
            df_mean, meta_mean = fetch_forecast_tmp_with_raw(
                grib_base,
                "gefsatmosmean",
                token,
                station_id,
                lat,
                lon,
                asof_utc,
                min_horizon=min_horizon_hours,
                max_horizon=resolve_max_horizon("gefsatmosmean"),
                members=None,
                accept=grib_accept,
                auth_scheme=grib_auth_scheme,
            )
            mean_daily = compute_daily_tmax(df_mean, target_date, zone_id, resolve_min_points("gefsatmosmean"))
            mean_meta = meta_mean
            mean_notes = f"pointsUsed={mean_daily.points} meanSource=gefsatmosmean"
            df_mean_for_raw = df_mean
        except Exception as exc:
            logger.warning(
                "GEFS mean fallback to members station=%s target_date=%s error=%s",
                station_id,
                target_date.isoformat(),
                exc,
            )

    df_members, meta_members = fetch_forecast_tmp_with_raw(
        grib_base,
        "gefsatmos",
        token,
        station_id,
        lat,
        lon,
        asof_utc,
        min_horizon=min_horizon_hours,
        max_horizon=resolve_max_horizon("gefsatmos"),
        members=gefs_members,
        accept=grib_accept,
        auth_scheme=grib_auth_scheme,
    )
    spread_daily = compute_gefs_spread(
        df_members, target_date, zone_id, gefs_members, resolve_min_points("gefsatmos")
    )
    spread_notes = f"pointsUsed={spread_daily.points} membersExpected={len(gefs_members)}"
    grib_rows.append(
        _build_gribstream_row(
            station_id,
            zone_id,
            target_date,
            asof_utc,
            "gefsatmos",
            "TMP_SPREAD_F",
            spread_daily,
            min_horizon_hours,
            resolve_max_horizon("gefsatmos"),
            meta_members,
            spread_notes,
        )
    )
    feature_values["gefsatmos_tmp_spread_f"] = spread_daily.value_f
    logger.info(
        "GribStream GEFS spread model=gefsatmos spread_f=%.2f members=%d rows=%d",
        spread_daily.value_f,
        df_members["member"].nunique(),
        len(df_members),
    )

    mean_from_members = compute_gefs_mean_tmax(
        df_members, target_date, zone_id, gefs_members, resolve_min_points("gefsatmos")
    )
    if mean_daily is None or mean_meta is None:
        mean_daily = mean_from_members
        mean_meta = meta_members
        mean_notes = (
            f"pointsUsed={mean_daily.points} membersExpected={len(gefs_members)} meanSource=gefsatmos"
        )
    grib_rows.append(
        _build_gribstream_row(
            station_id,
            zone_id,
            target_date,
            asof_utc,
            "gefsatmosmean",
            "TMAX_F",
            mean_daily,
            min_horizon_hours,
            resolve_max_horizon("gefsatmosmean"),
            mean_meta,
            mean_notes,
        )
    )
    feature_values["gefsatmosmean_tmax_f"] = mean_daily.value_f
    logger.info(
        "GribStream GEFS mean tmax_f=%.2f members=%d rows=%d",
        mean_daily.value_f,
        df_members["member"].nunique(),
        len(df_members),
    )

    if include_hourly_raw:
        for _, row in df_members.iterrows():
            member = int(row.get("member", 0))
            var_key = f"TMP|2 m above ground|member={member}"
            hourly_rows.append(
                {
                    "station_id": station_id,
                    "model": "gefsatmos",
                    "asof_utc": asof_utc,
                    "forecasted_at": row["forecasted_at"].to_pydatetime(),
                    "forecasted_time": row["forecasted_time"].to_pydatetime(),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "var_key": var_key,
                    "value": float(row["tmp_k"]),
                    "ingested_at": datetime.now(timezone.utc),
                }
            )
        if include_gefs_mean and df_mean_for_raw is not None:
            for _, row in df_mean_for_raw.iterrows():
                var_key = "TMP|2 m above ground|member=0"
                hourly_rows.append(
                    {
                        "station_id": station_id,
                        "model": "gefsatmosmean",
                        "asof_utc": asof_utc,
                        "forecasted_at": row["forecasted_at"].to_pydatetime(),
                        "forecasted_time": row["forecasted_time"].to_pydatetime(),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "var_key": var_key,
                        "value": float(row["tmp_k"]),
                        "ingested_at": datetime.now(timezone.utc),
                    }
                )

    return feature_values, grib_rows, hourly_rows


def _backfill_missing_core_features(
    store: MySqlStore,
    station_id: str,
    station_zoneid: str,
    missing_dates: list[date],
    zone_id: str,
    grib_base: str,
    token: str,
    grib_accept: str,
    grib_auth_scheme: str,
    min_horizon_hours: int,
    resolve_max_horizon,
    resolve_min_points,
    gefs_members: list[int],
    lat: float,
    lon: float,
    iem_base: str,
    mos_models: list[str],
    deterministic_models: list[str],
    include_gefs_mean: bool,
    logger,
) -> None:
    if not missing_dates:
        return
    total_grib = 0
    total_mos = 0
    for target_date in missing_dates:
        asof_utc = asof_from_target_date(target_date)
        feature_values, grib_rows, _ = _fetch_gribstream_daily_features(
            station_id=station_id,
            zone_id=zone_id,
            target_date=target_date,
            asof_utc=asof_utc,
            lat=lat,
            lon=lon,
            grib_base=grib_base,
            token=token,
            grib_accept=grib_accept,
            grib_auth_scheme=grib_auth_scheme,
            min_horizon_hours=min_horizon_hours,
            resolve_max_horizon=resolve_max_horizon,
            resolve_min_points=resolve_min_points,
            gefs_members=gefs_members,
            deterministic_models=deterministic_models,
            include_gefs_mean=include_gefs_mean,
            include_hourly_raw=False,
            logger=logger,
        )
        if grib_rows:
            store.upsert_gribstream_daily_feature(grib_rows)
            total_grib += len(grib_rows)

        mos_rows: list[dict] = []
        for model in mos_models:
            rows, value_max = _fetch_mos_daily_rows(
                iem_base=iem_base,
                station_id=station_id,
                station_zoneid=station_zoneid,
                model=model,
                target_date=target_date,
                asof_utc=asof_utc,
            )
            mos_rows.extend(rows)
            logger.info(
                "MOS daily %s n_x_max=%.2f target_date_local=%s",
                model,
                value_max,
                target_date.isoformat(),
            )
        if mos_rows:
            store.upsert_mos_daily_values(mos_rows)
            total_mos += len(mos_rows)

        logger.info(
            "Backfilled core features target_date_local=%s gefsmean=%.2f spread=%.2f",
            target_date.isoformat(),
            feature_values.get("gefsatmosmean_tmax_f") or float("nan"),
            feature_values.get("gefsatmos_tmp_spread_f") or float("nan"),
        )

    logger.info("Backfilled core tables grib_rows=%d mos_rows=%d", total_grib, total_mos)

def main() -> int:
    args = parse_args()
    logger = configure_logging(verbose=args.verbose)

    cfg_path = Path(args.config) if args.config else config_path()
    config = load_yaml(cfg_path)
    station_id = args.station or config.get("station_id") or "KMIA"
    ingestion_cfg = load_yaml_optional(
        repo_root() / "ingestion-service" / "src" / "main" / "resources" / "application.yml"
    )
    station_meta = _find_station_meta(ingestion_cfg, station_id)

    if args.asof_utc:
        asof_utc = parse_asof_utc(args.asof_utc)
    elif args.target_date:
        asof_utc = asof_from_target_date(parse_target_date(args.target_date))
    else:
        asof_utc = resolve_asof_utc()

    clock = build_clock(asof_utc)
    target_date = clock.target_date_local
    logger.info(
        "Runtime clock asof_utc=%s target_date_local=%s from=%s until=%s",
        clock.asof_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        target_date.isoformat(),
        clock.from_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        clock.until_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    thresholds = parse_thresholds(args.thresholds)
    if not thresholds:
        thresholds = [85, 86, 87]

    store_choice = (args.store or "mysql").lower()
    if args.csv:
        store_choice = "csv"
    if store_choice == "csv":
        csv_dir = args.csv_dir or os.getenv("LIVE_CSV_DIR") or Path(__file__).resolve().parent
        store = CsvStore(Path(csv_dir))
        logger.info("Using CSV store dir=%s", store.base_dir)
    else:
        mysql_cfg = MysqlConfig(
            host=os.getenv("MYSQL_HOST", config.get("mysql", {}).get("host", "localhost")),
            port=int(os.getenv("MYSQL_PORT", config.get("mysql", {}).get("port", 3306))),
            database=os.getenv("MYSQL_DB", config.get("mysql", {}).get("database", "weather_predictionmarkets")),
            user=os.getenv("MYSQL_USER", config.get("mysql", {}).get("user", "root")),
            password=os.getenv("MYSQL_PASSWORD", config.get("mysql", {}).get("password", "")),
        )
        store = MySqlStore(mysql_cfg)
    station_registry = store.fetch_station_registry(station_id)

    token = (
        os.getenv("GRIBSTREAM_TOKEN")
        or os.getenv("GRIBSTREAM_API_TOKEN")
        or None
    )
    if not token:
        raise RuntimeError("GRIBSTREAM_TOKEN or GRIBSTREAM_API_TOKEN env var is required")

    grib_cfg = config.get("gribstream", {})
    grib_base = (
        grib_cfg.get("base_url")
        or (ingestion_cfg.get("gribstream") or {}).get("baseUrl")
        or "https://gribstream.com"
    )
    grib_accept = (
        os.getenv("GRIBSTREAM_ACCEPT")
        or grib_cfg.get("accept")
        or grib_cfg.get("default_accept")
        or "text/csv"
    )
    grib_auth_scheme = (
        os.getenv("GRIBSTREAM_AUTH_SCHEME")
        or grib_cfg.get("auth_scheme")
        or grib_cfg.get("authScheme")
        or (ingestion_cfg.get("gribstream") or {}).get("authScheme")
        or "Bearer"
    )
    models = grib_cfg.get("models", ["nbm", "hrrr", "rap", "gefsatmosmean", "gefsatmos"])
    models_lower = [model.lower() for model in models]
    if "gefsatmos" not in models_lower:
        models.append("gefsatmos")
        models_lower.append("gefsatmos")
    include_gefs_mean = "gefsatmosmean" in models_lower
    deterministic_models = [model for model in models if model.lower() not in {"gefsatmos", "gefsatmosmean"}]
    core_deterministic_models = [
        model for model in deterministic_models if model.lower() in {"nbm", "hrrr", "rap"}
    ]
    gefs_members = grib_cfg.get("gefs_members", list(range(0, 31)))

    coords = grib_cfg.get("station_coords", {}).get(station_id)
    if coords:
        lat, lon = coords["lat"], coords["lon"]
    elif station_meta:
        lat = float(station_meta.get("latitude"))
        lon = float(station_meta.get("longitude"))
    else:
        lat, lon = resolve_station_coordinates(station_id)
    zone_id = (
        (station_registry or {}).get("zone_id")
        or (grib_cfg.get("station_zones") or {}).get(station_id)
        or (station_meta or {}).get("zoneId")
        or ("America/New_York" if station_id.upper() == "KMIA" else "UTC")
    )
    station_zoneid = zone_id
    logger.info("Station metadata station=%s zone_id=%s lat=%.4f lon=%.4f", station_id, zone_id, lat, lon)

    tmax_cfg = _resolve_tmax_setting(config, ingestion_cfg)
    min_horizon_hours = int(tmax_cfg.get("min-horizon-hours", tmax_cfg.get("min_horizon_hours", 12)))
    max_horizon_hours = int(tmax_cfg.get("max-horizon-hours", tmax_cfg.get("max_horizon_hours", 48)))
    rap_max_horizon_hours = int(
        tmax_cfg.get("rap-max-horizon-hours", tmax_cfg.get("rap_max_horizon_hours", 51))
    )
    hrrr_max_horizon_hours = int(
        tmax_cfg.get("hrrr-max-horizon-hours", tmax_cfg.get("hrrr_max_horizon_hours", 48))
    )
    min_points_per_day = int(tmax_cfg.get("min-points-per-day", tmax_cfg.get("min_points_per_day", 8)))
    rap_min_points_per_day = int(
        tmax_cfg.get("rap-min-points-per-day", tmax_cfg.get("rap_min_points_per_day", 5))
    )
    gefs_min_points_per_day = int(
        tmax_cfg.get("gefs-min-points-per-day", tmax_cfg.get("gefs_min_points_per_day", 4))
    )

    ingested_at = datetime.now(timezone.utc)

    def resolve_max_horizon(model_code: str) -> int:
        if model_code.lower() == "rap":
            return rap_max_horizon_hours
        if model_code.lower() == "hrrr":
            return hrrr_max_horizon_hours
        return max_horizon_hours

    def resolve_min_points(model_code: str) -> int:
        if model_code.lower() == "rap":
            return rap_min_points_per_day
        if model_code.lower() in {"gefsatmos", "gefsatmosmean"}:
            return gefs_min_points_per_day
        return min_points_per_day

    grib_features, grib_rows, hourly_rows = _fetch_gribstream_daily_features(
        station_id=station_id,
        zone_id=zone_id,
        target_date=target_date,
        asof_utc=clock.asof_utc,
        lat=lat,
        lon=lon,
        grib_base=grib_base,
        token=token,
        grib_accept=grib_accept,
        grib_auth_scheme=grib_auth_scheme,
        min_horizon_hours=min_horizon_hours,
        resolve_max_horizon=resolve_max_horizon,
        resolve_min_points=resolve_min_points,
        gefs_members=gefs_members,
        deterministic_models=deterministic_models,
        include_gefs_mean=include_gefs_mean,
        include_hourly_raw=True,
        logger=logger,
    )
    if hourly_rows:
        store.upsert_gribstream_hourly_raw(hourly_rows)
    if grib_rows:
        store.upsert_gribstream_daily_feature(grib_rows)

    iem_cfg = config.get("iem", {})
    iem_base = iem_cfg.get("base_url", "https://mesonet.agron.iastate.edu")
    mos_models = config.get("mos_models", ["GFS", "NAM"])
    mos_rows: list[dict] = []
    mos_values: dict[str, float | None] = {}
    for model in mos_models:
        rows, value_max = _fetch_mos_daily_rows(
            iem_base=iem_base,
            station_id=station_id,
            station_zoneid=station_zoneid,
            model=model,
            target_date=target_date,
            asof_utc=clock.asof_utc,
        )
        mos_rows.extend(rows)
        mos_values[model] = value_max
    if mos_rows:
        store.upsert_mos_daily_values(mos_rows)

    gfs_n_x_max = mos_values.get("GFS")
    nam_n_x_max = mos_values.get("NAM")
    logger.info(
        "MOS daily tmax GFS=%.2f NAM=%.2f target_date_local=%s",
        gfs_n_x_max if gfs_n_x_max is not None else float("nan"),
        nam_n_x_max if nam_n_x_max is not None else float("nan"),
        target_date.isoformat(),
    )

    features_row = {
        "station_id": station_id,
        "target_date_local": target_date,
        "asof_utc": clock.asof_utc,
        "gfs_tmax_f": grib_features.get("gfs_tmax_f"),
        "hrrr_tmax_f": grib_features.get("hrrr_tmax_f"),
        "rap_tmax_f": grib_features.get("rap_tmax_f"),
        "nbm_tmax_f": grib_features.get("nbm_tmax_f"),
        "gefsatmosmean_tmax_f": grib_features.get("gefsatmosmean_tmax_f"),
        "gefsatmos_tmp_spread_f": grib_features.get("gefsatmos_tmp_spread_f"),
        "mos_gfs_tmax_f": gfs_n_x_max,
        "mos_nam_tmax_f": nam_n_x_max,
        "gfs_n_x_max": gfs_n_x_max,
        "nam_n_x_max": nam_n_x_max,
        "ingested_at": ingested_at,
    }
    store.upsert_live_features_daily(features_row)

    station_models_dir = models_dir(station_id)
    feature_cols_path = station_models_dir / "e92_feature_columns.json"
    mu_model_path = station_models_dir / "e92_mu_model.joblib"
    sigma_model_path = station_models_dir / "e92_sigma_model.joblib"

    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing feature columns file: {feature_cols_path}")
    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))

    bias_window_days = 60
    truth_lag_days = 2
    emos_window_days = 45
    support_end = target_date - timedelta(days=truth_lag_days)
    support_start = target_date - timedelta(
        days=(emos_window_days + bias_window_days + truth_lag_days)
    )

    history_features, history_truth = _ensure_core_history(
        store=store,
        station_id=station_id,
        start_date=support_start,
        end_date=support_end,
        asof_utc=clock.asof_utc,
        iem_base=iem_base,
        station_registry=station_registry,
        station_zoneid=station_zoneid,
        zone_id=zone_id,
        grib_base=grib_base,
        token=token,
        grib_accept=grib_accept,
        grib_auth_scheme=grib_auth_scheme,
        min_horizon_hours=min_horizon_hours,
        resolve_max_horizon=resolve_max_horizon,
        resolve_min_points=resolve_min_points,
        gefs_members=gefs_members,
        lat=lat,
        lon=lon,
        mos_models=mos_models,
        deterministic_models=deterministic_models,
        include_gefs_mean=include_gefs_mean,
        logger=logger,
    )
    _sync_live_truth_cli(store, station_id, support_start, support_end, logger)

    feature_df = build_feature_vector(
        feature_cols,
        features_row,
        history_features,
        history_truth,
        target_date,
    )

    mu_model = joblib.load(mu_model_path)
    sigma_model = joblib.load(sigma_model_path)
    mu_hat_f = float(mu_model.predict(feature_df.to_numpy(dtype=float))[0])
    sigma_hat_f = float(max(0.5, sigma_model.predict(feature_df.to_numpy(dtype=float))[0]))

    prediction_row = {
        "station_id": station_id,
        "target_date_local": target_date,
        "asof_utc": clock.asof_utc,
        "mu_hat_f": mu_hat_f,
        "sigma_hat_f": sigma_hat_f,
        "sigma_emos_f": None,
        "emos_c": None,
        "emos_d": None,
        "rolling_bias_45": None,
        "rolling_rmse_45": None,
        "created_at": datetime.now(timezone.utc),
    }
    store.upsert_live_predictions(prediction_row)

    emos_end = target_date - timedelta(days=2)
    emos_start = emos_end - timedelta(days=44)
    preds_existing = store.fetch_predictions_range(station_id, emos_start, emos_end)
    missing_preds = _find_missing_prediction_dates(preds_existing, emos_start, emos_end)
    if missing_preds:
        logger.warning(
            "Missing predictions for %d dates. Backfilling via models.", len(missing_preds)
        )
        _backfill_missing_predictions(
            store=store,
            station_id=station_id,
            missing_dates=missing_preds,
            feature_cols=feature_cols,
            history_features=history_features,
            history_truth=history_truth,
            mu_model=mu_model,
            sigma_model=sigma_model,
            logger=logger,
        )

    pred_history = store.fetch_predictions_history(station_id, emos_start, emos_end, clock.asof_utc)
    truth_history = store.fetch_truth_history(station_id, emos_start, emos_end)
    emos_history = pred_history.merge(truth_history, on=["station_id", "target_date_local"], how="inner")
    emos_history = emos_history.dropna(subset=["mu_hat_f", "sigma_hat_f", "actual_tmax_f"])

    if len(emos_history) < 45:
        logger.warning(
            "Insufficient EMOS history rows=%s (need 45). Skipping EMOS calibration.",
            len(emos_history),
        )
        sigma_emos_f = None
        emos_c = None
        emos_d = None
        rolling_bias = None
        rolling_rmse = None
    else:
        emos_result = calibrate(emos_history, sigma_hat_f, sigma_floor=0.5)
        sigma_emos_f = emos_result.sigma_emos
        emos_c = emos_result.c
        emos_d = emos_result.d
        rolling_bias = emos_result.rolling_bias
        rolling_rmse = emos_result.rolling_rmse

        prediction_row.update(
            {
                "sigma_emos_f": sigma_emos_f,
                "emos_c": emos_c,
                "emos_d": emos_d,
                "rolling_bias_45": rolling_bias,
                "rolling_rmse_45": rolling_rmse,
                "created_at": datetime.now(timezone.utc),
            }
        )
        store.upsert_live_predictions(prediction_row)

    final_sigma = sigma_emos_f if sigma_emos_f is not None else sigma_hat_f
    threshold_probs = {
        f"P_ge_{k}": float(1.0 - _normal_cdf((k - mu_hat_f) / final_sigma))
        for k in thresholds
    }

    output = {
        "station_id": station_id,
        "asof_utc": clock.asof_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_date_local": target_date.isoformat(),
        "mu_hat_f": mu_hat_f,
        "sigma_hat_f": sigma_hat_f,
        "emos_window_days": 45,
        "emos_c": emos_c,
        "emos_d": emos_d,
        "sigma_emos_f": sigma_emos_f,
        "rolling_bias_45": rolling_bias,
        "rolling_rmse_45": rolling_rmse,
        "normal_dist": {
            "mu": mu_hat_f,
            "sigma": final_sigma,
            "variance": None if final_sigma is None else float(final_sigma**2),
        },
        "threshold_probs": threshold_probs,
    }
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output["output_json_path"] = str(output_path)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        logger.info("Wrote output JSON to %s", output_path)

    if args.stdout_json == "full":
        print(json.dumps(output, indent=2))
    return 0


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))


if __name__ == "__main__":
    raise SystemExit(main())
