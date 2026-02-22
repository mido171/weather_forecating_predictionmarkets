"""Truth, climatology, and observed-history features."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import re
import logging
from sqlalchemy import text

from .mos_config import MosDatasetConfig
from .mos_utils import normalize_sql, sha256_hex


LOGGER = logging.getLogger(__name__)


def fetch_truth_rows(engine, cfg: MosDatasetConfig, start_date: date, end_date: date) -> tuple[pd.DataFrame, str]:
    table = (cfg.truth_table or "station_daily_truth").strip()
    if not re.match(r"^[A-Za-z0-9_]+$", table):
        raise ValueError(f"Invalid truth_table name: {table}")
    sql = f"""
        SELECT station_id, station_zoneid, date_local, tmax_f, tmin_f
        FROM weather_predictionmarkets.{table}
        WHERE station_id = :station_id
          AND date_local BETWEEN :start_date AND :end_date
        ORDER BY date_local
    """
    params = {
        "station_id": cfg.station_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    sql_hash = sha256_hex(normalize_sql(sql))
    try:
        df = pd.read_sql(text(sql), engine, params=params)
    except Exception as exc:
        if table != "station_daily_truth":
            LOGGER.warning("Truth table %s unavailable (%s); falling back to station_daily_truth.", table, exc)
            sql = """
                SELECT station_id, station_zoneid, date_local, tmax_f, tmin_f
                FROM weather_predictionmarkets.station_daily_truth
                WHERE station_id = :station_id
                  AND date_local BETWEEN :start_date AND :end_date
                ORDER BY date_local
            """
            sql_hash = sha256_hex(normalize_sql(sql))
            df = pd.read_sql(text(sql), engine, params=params)
        else:
            raise
    if df.empty:
        return df, sql_hash
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    df["tmax_f"] = pd.to_numeric(df.get("tmax_f"), errors="coerce")
    df["tmin_f"] = pd.to_numeric(df.get("tmin_f"), errors="coerce")
    return df, sql_hash


def compute_climo_by_doy(truth: pd.DataFrame) -> pd.DataFrame:
    if truth.empty:
        return pd.DataFrame()
    truth = truth.copy()
    truth["date_local"] = pd.to_datetime(truth["date_local"])
    truth["year"] = truth["date_local"].dt.year
    truth["doy"] = truth["date_local"].dt.dayofyear

    doy_groups: dict[int, dict[str, Any]] = defaultdict(lambda: {"years": [], "values": []})
    for _, row in truth.iterrows():
        if pd.isna(row["tmax_f"]):
            continue
        doy = int(row["doy"])
        doy_groups[doy]["years"].append(int(row["year"]))
        doy_groups[doy]["values"].append(float(row["tmax_f"]))

    doy_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for doy, payload in doy_groups.items():
        years = np.array(payload["years"], dtype=int)
        values = np.array(payload["values"], dtype=float)
        order = np.argsort(years)
        doy_arrays[doy] = (years[order], values[order])

    rows = []
    for _, row in truth.iterrows():
        doy = int(row["doy"])
        year = int(row["year"])
        years, values = doy_arrays.get(doy, (np.array([], dtype=int), np.array([], dtype=float)))
        if years.size == 0:
            stats = [np.nan] * 5
        else:
            idx = np.searchsorted(years, year, side="left")
            vals = values[:idx]
            if vals.size == 0:
                stats = [np.nan] * 5
            else:
                stats = [
                    float(np.mean(vals)),
                    float(np.median(vals)),
                    float(np.std(vals)),
                    float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    float(vals.size),
                ]
        rows.append(
            {
                "date_local": row["date_local"].date(),
                "climo_mean": stats[0],
                "climo_median": stats[1],
                "climo_std": stats[2],
                "climo_iqr": stats[3],
                "climo_count_years": stats[4],
            }
        )
    return pd.DataFrame(rows)


def compute_obs_features(
    cal_df: pd.DataFrame,
    truth: pd.DataFrame,
    cfg: MosDatasetConfig,
) -> pd.DataFrame:
    if truth.empty:
        return pd.DataFrame()
    truth = truth.copy()
    truth = truth.sort_values("date_local")
    truth_indexed = truth.set_index("date_local")

    obs_features = pd.DataFrame(index=truth_indexed.index)
    series = truth_indexed["tmax_f"].astype(float)

    for window in cfg.obs_windows_days or []:
        roll = series.rolling(window=window, min_periods=1)
        obs_features[f"obs_tmax_roll_mean_{window}"] = roll.mean()
        obs_features[f"obs_tmax_roll_median_{window}"] = roll.median()
        obs_features[f"obs_tmax_roll_min_{window}"] = roll.min()
        obs_features[f"obs_tmax_roll_max_{window}"] = roll.max()
        obs_features[f"obs_tmax_roll_std_{window}"] = roll.std()
        obs_features[f"obs_tmax_roll_iqr_{window}"] = roll.quantile(0.75) - roll.quantile(0.25)
        obs_features[f"obs_tmax_roll_p10_{window}"] = roll.quantile(0.10)
        obs_features[f"obs_tmax_roll_p25_{window}"] = roll.quantile(0.25)
        obs_features[f"obs_tmax_roll_p75_{window}"] = roll.quantile(0.75)
        obs_features[f"obs_tmax_roll_p90_{window}"] = roll.quantile(0.90)

    for window in cfg.obs_slope_windows_days or []:
        slope = series.rolling(window=window, min_periods=2).apply(_slope, raw=True)
        obs_features[f"obs_tmax_slope_{window}"] = slope

    obs_features["obs_tmax_last"] = series
    obs_features["obs_tmax_prev"] = series.shift(1)
    obs_features["obs_tmax_delta_1"] = obs_features["obs_tmax_last"] - obs_features["obs_tmax_prev"]
    obs_features["obs_tmax_abs_delta_1"] = obs_features["obs_tmax_delta_1"].abs()
    if "obs_tmax_roll_mean_30" in obs_features.columns:
        obs_features["obs_tmax_vs_mean_30"] = (
            obs_features["obs_tmax_last"] - obs_features["obs_tmax_roll_mean_30"]
        )
    if "obs_tmax_roll_std_30" in obs_features.columns:
        obs_features["obs_tmax_z_30"] = (
            obs_features["obs_tmax_last"] - obs_features["obs_tmax_roll_mean_30"]
        ) / (obs_features["obs_tmax_roll_std_30"] + 1e-6)

    obs_features["obs_persist_ema_7"] = series.ewm(span=7, adjust=False).mean()

    obs_features = obs_features.reset_index().rename(columns={"date_local": "obs_date"})

    climo = compute_climo_by_doy(truth)
    climo = climo.rename(
        columns={
            "climo_mean": "obs_climo_mean",
            "climo_median": "obs_climo_median",
            "climo_std": "obs_climo_std",
            "climo_iqr": "obs_climo_iqr",
            "climo_count_years": "obs_climo_count_years",
        }
    )

    cal_df = cal_df.copy()
    cal_df["obs_cutoff_date"] = cal_df["asof_date_local"].apply(
        lambda d: d - timedelta(days=cfg.obs_cutoff_lag_days)
    )
    merged = cal_df.merge(obs_features, left_on="obs_cutoff_date", right_on="obs_date", how="left")
    merged = merged.merge(
        climo,
        left_on="obs_cutoff_date",
        right_on="date_local",
        how="left",
        suffixes=("", "_climo_t"),
    )
    merged["obs_tmax_anom_last_vs_climo"] = (
        merged["obs_tmax_last"] - merged["obs_climo_mean"]
    )

    climo_target = compute_climo_by_doy(truth)
    climo_target = climo_target.rename(
        columns={
            "date_local": "target_date_local",
            "climo_mean": "obs_climo_d_mean",
            "climo_median": "obs_climo_d_median",
            "climo_std": "obs_climo_d_std",
            "climo_iqr": "obs_climo_d_iqr",
            "climo_count_years": "obs_climo_d_count_years",
        }
    )
    merged = merged.merge(climo_target, on="target_date_local", how="left")
    merged["obs_climo_target_baseline"] = merged["obs_climo_d_mean"]
    merged["obs_persist_tmax"] = merged["obs_tmax_last"]
    merged["obs_persist_blend"] = (
        0.7 * merged["obs_persist_tmax"] + 0.3 * merged["obs_climo_target_baseline"]
    )

    obs_cols = [col for col in merged.columns if col.startswith("obs_")]
    return merged[["target_date_local", "asof_date_local"] + obs_cols]


def _slope(values: np.ndarray) -> float:
    y = values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    y = y[mask]
    x = np.arange(len(values))[mask]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)
