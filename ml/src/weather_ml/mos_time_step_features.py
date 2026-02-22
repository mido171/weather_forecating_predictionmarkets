"""Time-step MOS feature extraction using mos_forecast_value."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from .mos_config import MosDatasetConfig
from .mos_utils import normalize_sql, sha256_hex


def fetch_mos_time_step_rows(
    engine,
    cfg: MosDatasetConfig,
    start_target: date,
    end_target: date,
) -> tuple[pd.DataFrame, str]:
    models = cfg.models or ["GFS", "NAM"]
    variables = cfg.variables or []
    model_placeholders = ", ".join([f":m{i}" for i in range(len(models))])
    var_placeholders = ", ".join([f":v{i}" for i in range(len(variables))])
    offset = int(cfg.climate_day_utc_offset_hours)
    start_utc = datetime.combine(start_target, datetime.min.time(), tzinfo=timezone.utc) - timedelta(
        hours=offset
    )
    end_utc = datetime.combine(
        end_target + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    ) - timedelta(hours=offset)

    sql = f"""
        SELECT station_id, model, runtime_utc, forecast_time_utc, variable_code,
               value_num, value_text, raw_payload_hash_ref, retrieved_at_utc
        FROM weather_predictionmarkets.mos_forecast_value
        WHERE station_id = :station_id
          AND forecast_time_utc BETWEEN :start_utc AND :end_utc
          AND UPPER(model) IN ({model_placeholders})
          AND LOWER(variable_code) IN ({var_placeholders})
        ORDER BY forecast_time_utc
    """
    params: dict[str, Any] = {
        "station_id": cfg.station_id,
        "start_utc": start_utc,
        "end_utc": end_utc,
    }
    for i, model in enumerate(models):
        params[f"m{i}"] = model
    for i, var in enumerate(variables):
        params[f"v{i}"] = var
    sql_hash = sha256_hex(normalize_sql(sql))
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return df, sql_hash
    df["forecast_time_utc"] = pd.to_datetime(df["forecast_time_utc"], utc=True)
    df["runtime_utc"] = pd.to_datetime(df["runtime_utc"], utc=True)
    df["retrieved_at_utc"] = pd.to_datetime(df["retrieved_at_utc"], utc=True)
    df["model"] = df["model"].astype(str).str.lower()
    df["variable_code"] = df["variable_code"].astype(str).str.lower()
    df["value_num"] = pd.to_numeric(df.get("value_num"), errors="coerce")
    df["target_date_local"] = (
        df["forecast_time_utc"] + pd.to_timedelta(offset, unit="h")
    ).dt.date

    expected_runtime = (
        pd.to_datetime(df["target_date_local"])
        + pd.to_timedelta(cfg.mos_runtime_hour_utc, unit="h")
        + pd.to_timedelta(cfg.mos_runtime_minute_utc, unit="m")
        - pd.Timedelta(days=1)
    ).dt.tz_localize("UTC")
    df["runtime_expected_utc"] = expected_runtime
    df["runtime_match"] = df["runtime_utc"] == df["runtime_expected_utc"]

    runtime_policy = (cfg.mos_runtime_policy or "latest_before_cutoff").lower()
    if runtime_policy == "exact":
        df = df[df["runtime_match"]]
    elif runtime_policy == "prefer_exact":
        df = df.sort_values(
            ["target_date_local", "model", "variable_code", "runtime_match", "runtime_utc", "retrieved_at_utc"],
            ascending=[True, True, True, True, True, True],
        )
        df = df.drop_duplicates(
            subset=["target_date_local", "model", "variable_code", "forecast_time_utc"],
            keep="last",
        )
    else:
        df = df.sort_values(
            ["target_date_local", "model", "variable_code", "runtime_utc", "retrieved_at_utc"],
            ascending=[True, True, True, True, True],
        )
        df = df.drop_duplicates(
            subset=["target_date_local", "model", "variable_code", "forecast_time_utc"],
            keep="last",
        )

    df = df[(df["target_date_local"] >= start_target) & (df["target_date_local"] <= end_target)]
    return df, sql_hash


def build_time_step_features(ts_df: pd.DataFrame, cfg: MosDatasetConfig) -> pd.DataFrame:
    if ts_df.empty:
        return pd.DataFrame()
    offset = int(cfg.climate_day_utc_offset_hours)
    ts_df = ts_df.copy()
    ts_df["lst_hour"] = (ts_df["forecast_time_utc"] + pd.to_timedelta(offset, unit="h")).dt.hour

    features = []
    for (target_date_local, model, variable), group in ts_df.groupby(
        ["target_date_local", "model", "variable_code"]
    ):
        values = group["value_num"].astype(float)
        if values.notna().sum() == 0:
            continue
        row = {
            "target_date_local": target_date_local,
            "model": model,
            "variable_code": variable,
            "ts_min": float(values.min()),
            "ts_max": float(values.max()),
            "ts_mean": float(values.mean()),
            "ts_std": float(values.std()) if values.notna().sum() > 1 else np.nan,
        }
        idx_max = values.idxmax()
        idx_min = values.idxmin()
        row["ts_time_of_max"] = float(group.loc[idx_max, "lst_hour"]) if idx_max is not None else np.nan
        row["ts_time_of_min"] = float(group.loc[idx_min, "lst_hour"]) if idx_min is not None else np.nan
        row["ts_diurnal_amp"] = row["ts_max"] - row["ts_min"]
        features.append(row)

    if not features:
        return pd.DataFrame()

    feat_df = pd.DataFrame(features)
    pivot = feat_df.pivot_table(
        index=["target_date_local"],
        columns=["model", "variable_code"],
        values=["ts_min", "ts_max", "ts_mean", "ts_std", "ts_time_of_max", "ts_time_of_min", "ts_diurnal_amp"],
        aggfunc="first",
    )
    pivot.columns = [
        f"mos_ts_{metric}_{model}_{var}"
        for metric, model, var in pivot.columns.to_list()
    ]
    pivot = pivot.reset_index()
    return pivot


def add_mos_time_step_features(
    base_df: pd.DataFrame,
    engine,
    cfg: MosDatasetConfig,
    start_target: date,
    end_target: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ts_df, sql_hash = fetch_mos_time_step_rows(engine, cfg, start_target, end_target)
    if ts_df.empty:
        return base_df, {"time_step_rows": 0, "sql_hash": sql_hash}
    features = build_time_step_features(ts_df, cfg)
    if features.empty:
        return base_df, {"time_step_rows": int(len(ts_df)), "sql_hash": sql_hash}
    merged = base_df.merge(features, on="target_date_local", how="left")
    return merged, {"time_step_rows": int(len(ts_df)), "sql_hash": sql_hash}
