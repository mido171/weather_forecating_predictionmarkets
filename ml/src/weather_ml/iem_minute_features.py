from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from .mos_config import MosDatasetConfig


@dataclass(frozen=True)
class MinuteWindow:
    start_min: int
    end_min: int


def _slope_minutes(mins: np.ndarray, temps: np.ndarray, start_min: int, end_min: int) -> float:
    mask = (mins >= start_min) & (mins < end_min)
    if mask.sum() < 2:
        return np.nan
    x = mins[mask].astype(float) / 60.0
    y = temps[mask].astype(float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return np.nan


def _value_at_minute(series: pd.Series, target_min: int, tolerance: int = 5) -> float:
    if target_min in series.index:
        return float(series.loc[target_min])
    nearby = series.loc[(series.index >= target_min - tolerance) & (series.index <= target_min + tolerance)]
    if not nearby.empty:
        return float(nearby.iloc[0])
    return np.nan


def _compute_climo_by_doy(values: pd.DataFrame, value_col: str) -> pd.Series:
    values = values.copy()
    values["utc_date"] = pd.to_datetime(values["utc_date"])
    values["year"] = values["utc_date"].dt.year
    values["doy"] = values["utc_date"].dt.dayofyear
    rows = []
    for _, row in values.iterrows():
        doy = int(row["doy"])
        year = int(row["year"])
        subset = values[(values["doy"] == doy) & (values["year"] < year)]
        if subset.empty:
            rows.append(np.nan)
        else:
            rows.append(float(np.nanmean(subset[value_col].to_numpy(dtype=float))))
    return pd.Series(rows, index=values.index, dtype=float)


def _load_minute_files(path: Path) -> Iterable[Path]:
    return sorted(path.glob("*.csv"))


def compute_iem_minute_features(
    cal_df: pd.DataFrame,
    cfg: MosDatasetConfig,
    truth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not cfg.iem_minute_path:
        return pd.DataFrame()

    minute_path = Path(cfg.iem_minute_path)
    if not minute_path.exists():
        raise FileNotFoundError(f"IEM minute path not found: {minute_path}")

    zone = ZoneInfo(cfg.iem_minute_timezone or cfg.station_zoneid)
    time_col = cfg.iem_minute_time_col
    temp_col = cfg.iem_minute_temp_col
    eps = float(cfg.iem_minute_plateau_eps_f)
    drop_thr = float(cfg.iem_minute_drop_thr_f)
    auc_thresholds = cfg.iem_minute_auc_thresholds_f or [85.0, 88.0]
    last_hours = int(cfg.iem_minute_last_hours)

    target_dates = pd.to_datetime(cal_df["target_date_local"]).dt.date
    min_target = target_dates.min()
    max_target = target_dates.max()

    local_start = min_target - timedelta(days=1)
    local_end = max_target - timedelta(days=1)

    utc_start = datetime.combine(min_target, time(0, 0), tzinfo=timezone.utc)
    utc_end = datetime.combine(max_target, time(23, 59), tzinfo=timezone.utc)

    local_rows: list[dict] = []
    utc_rows: list[dict] = []

    for file_path in _load_minute_files(minute_path):
        df = pd.read_csv(file_path, usecols=[time_col, temp_col], low_memory=False)
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
        df = df.dropna(subset=[time_col, temp_col])

        df = df[(df[time_col] >= utc_start) & (df[time_col] <= utc_end)]
        if df.empty:
            continue

        df["utc_date"] = df[time_col].dt.date
        df["utc_minute"] = df[time_col].dt.hour * 60 + df[time_col].dt.minute

        local_dt = df[time_col].dt.tz_convert(zone)
        df["local_date"] = local_dt.dt.date
        df["local_minute"] = local_dt.dt.hour * 60 + local_dt.dt.minute
        df["local_hour"] = local_dt.dt.hour

        df_local = df[(df["local_date"] >= local_start) & (df["local_date"] <= local_end)]
        if not df_local.empty:
            for local_date, group in df_local.groupby("local_date"):
                temps = group[temp_col].to_numpy(dtype=float)
                mins = group["local_minute"].to_numpy(dtype=int)
                tmax = float(np.nanmax(temps)) if temps.size else np.nan
                tmin = float(np.nanmin(temps)) if temps.size else np.nan
                tmean = float(np.nanmean(temps)) if temps.size else np.nan
                tmed = float(np.nanmedian(temps)) if temps.size else np.nan
                tmax_time = np.nan
                if temps.size:
                    idx = np.where(temps == tmax)[0]
                    if idx.size:
                        tmax_time = float(mins[idx[0]])

                plateau = float(np.sum(temps >= (tmax - eps))) if temps.size else np.nan

                series = pd.Series(temps, index=mins).groupby(level=0).mean()
                series = series.reindex(range(1440))
                diff30 = series.shift(30) - series
                max_drop_30 = float(np.nanmax(diff30.to_numpy(dtype=float))) if diff30.notna().any() else np.nan
                drop_cnt_30 = float(np.sum(diff30 >= drop_thr)) if diff30.notna().any() else np.nan

                def slope_window(start_h: int, end_h: int) -> float:
                    return _slope_minutes(mins, temps, start_h * 60, end_h * 60)

                hourly = group.groupby("local_hour")[temp_col].mean()
                hourly_vals = {f"iem_hour_{h:02d}": float(hourly.get(h, np.nan)) for h in range(24)}

                row = {
                    "local_date": local_date,
                    "iem_tmax": tmax,
                    "iem_tmin": tmin,
                    "iem_tmean": tmean,
                    "iem_tmed": tmed,
                    "iem_range": tmax - tmin if np.isfinite(tmax) and np.isfinite(tmin) else np.nan,
                    "iem_tmax_time_local_min": tmax_time,
                    "iem_plateau_mins_eps": plateau,
                    "iem_slope_09_12": slope_window(9, 12),
                    "iem_slope_12_15": slope_window(12, 15),
                    "iem_slope_15_18": slope_window(15, 18),
                    "iem_slope_18_21": slope_window(18, 21),
                    "iem_slope_21_24": slope_window(21, 24),
                    "iem_drop_cnt_30": drop_cnt_30,
                    "iem_max_drop_30": max_drop_30,
                }
                for thr in auc_thresholds:
                    row[f"iem_auc_{int(thr)}"] = float(np.nansum(np.maximum(temps - thr, 0.0)) / 60.0)
                row.update(hourly_vals)
                local_rows.append(row)

        df_utc = df[(df["utc_date"] >= min_target) & (df["utc_date"] <= max_target)]
        if not df_utc.empty:
            for utc_date, group in df_utc.groupby("utc_date"):
                temps = group[temp_col].to_numpy(dtype=float)
                mins = group["utc_minute"].to_numpy(dtype=int)
                series = pd.Series(temps, index=mins).groupby(level=0).mean()
                temp_00z = _value_at_minute(series, 0)
                temp_03z = _value_at_minute(series, 180)
                temp_06z = _value_at_minute(series, 360)
                slope_last60 = _slope_minutes(mins, temps, 300, 360)
                slope_last180 = _slope_minutes(mins, temps, 180, 360)
                mask_last180 = (mins >= 180) & (mins <= 360)
                std_last180 = float(np.nanstd(temps[mask_last180])) if mask_last180.any() else np.nan
                row = {
                    "utc_date": utc_date,
                    "iem_temp_00z": temp_00z,
                    "iem_temp_03z": temp_03z,
                    "iem_temp_06z": temp_06z,
                    "iem_slope_last60": slope_last60,
                    "iem_slope_last180": slope_last180,
                    "iem_std_last180": std_last180,
                    "iem_cool_00_06": temp_00z - temp_06z if np.isfinite(temp_00z) and np.isfinite(temp_06z) else np.nan,
                }
                utc_rows.append(row)

    local_df = pd.DataFrame(local_rows)
    utc_df = pd.DataFrame(utc_rows)

    if local_df.empty and utc_df.empty:
        return pd.DataFrame()

    if not utc_df.empty:
        utc_df["utc_date"] = pd.to_datetime(utc_df["utc_date"])
        utc_df = utc_df.sort_values("utc_date").reset_index(drop=True)
        utc_df["iem_climo_06z"] = _compute_climo_by_doy(utc_df, "iem_temp_06z")
        utc_df["iem_night_warm_anom"] = utc_df["iem_temp_06z"] - utc_df["iem_climo_06z"]
        utc_df["target_date_local"] = utc_df["utc_date"].dt.date

    minute_df = pd.DataFrame()
    if not local_df.empty:
        local_df["local_date"] = pd.to_datetime(local_df["local_date"]).dt.date
        local_df = local_df.sort_values("local_date").reset_index(drop=True)
        local_df["target_date_local"] = local_df["local_date"] + timedelta(days=1)
        rename_map = {c: f"iem_tminus1_{c}" for c in local_df.columns if c not in ["local_date", "target_date_local"]}
        local_df = local_df.rename(columns=rename_map)

        minute_df = local_df[["target_date_local"] + list(rename_map.values())].copy()

        if truth is not None and not truth.empty:
            truth_map = truth.set_index("date_local")["tmax_f"].to_dict()
            iem_tmax = local_df["iem_tminus1_iem_tmax"].rename("iem_tmax_raw")
            diff_series = []
            for d, tmax in zip(local_df["local_date"], iem_tmax):
                nws = truth_map.get(d)
                nws_val = float(nws) if nws is not None else np.nan
                tmax_val = float(tmax) if tmax is not None else np.nan
                diff_series.append(
                    nws_val - tmax_val if np.isfinite(nws_val) and np.isfinite(tmax_val) else np.nan
                )
            diff_series = pd.Series(diff_series, index=local_df.index, dtype=float)
            diff_ewma_7 = diff_series.ewm(span=7, adjust=False).mean().shift(1)
            diff_ewma_30 = diff_series.ewm(span=30, adjust=False).mean().shift(1)
            diff_vol_30 = diff_series.rolling(window=30, min_periods=5).std().shift(1)
            minute_df["iem_diff_tminus1"] = diff_series.to_numpy(dtype=float)
            minute_df["iem_diff_ewma_7_tminus1"] = diff_ewma_7.to_numpy(dtype=float)
            minute_df["iem_diff_ewma_30_tminus1"] = diff_ewma_30.to_numpy(dtype=float)
            minute_df["iem_diff_vol_30_tminus1"] = diff_vol_30.to_numpy(dtype=float)

    if not utc_df.empty:
        utc_keep = [
            "target_date_local",
            "iem_temp_00z",
            "iem_temp_03z",
            "iem_temp_06z",
            "iem_slope_last60",
            "iem_slope_last180",
            "iem_std_last180",
            "iem_cool_00_06",
            "iem_night_warm_anom",
        ]
        minute_df = minute_df.merge(utc_df[utc_keep], on="target_date_local", how="outer")

    return minute_df
