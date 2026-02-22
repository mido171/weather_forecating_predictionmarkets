from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


TIME_COL_CANDIDATES = [
    "valid(utc)",
    "valid_utc",
    "valid",
    "timestamp",
    "ts",
    "time",
    "date_time",
    "datetime",
    "date",
]

TEMP_COL_CANDIDATES = [
    "tmpf",
    "temp_f",
    "temperature",
    "airtemp",
    "tmp",
    "temp",
]

STATION_COL_CANDIDATES = [
    "station",
    "station_id",
    "stid",
]


def _lower_cols(df: pd.DataFrame) -> dict[str, str]:
    return {c.lower(): c for c in df.columns}


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    col_map = _lower_cols(df)
    for name in candidates:
        if name.lower() in col_map:
            return col_map[name.lower()]
    return None


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _standardize_frame(df: pd.DataFrame, station_id: str) -> pd.DataFrame:
    time_col = _find_col(df, TIME_COL_CANDIDATES)
    if time_col is None:
        raise ValueError("No timestamp column found.")
    temp_col = _find_col(df, TEMP_COL_CANDIDATES)
    if temp_col is None:
        raise ValueError("No temperature column found.")
    station_col = _find_col(df, STATION_COL_CANDIDATES)
    if station_col is not None:
        station_vals = df[station_col].astype(str).str.upper()
        df = df[station_vals.isin({station_id, station_id.replace("K", "")})].copy()
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    tmpf = pd.to_numeric(df[temp_col], errors="coerce")
    out = pd.DataFrame({"ts_utc": ts, "tmpf": tmpf})
    out = out[out["ts_utc"].notna()].copy()
    return out


def _resample_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts_utc").set_index("ts_utc")
    tmp = df["tmpf"].resample("5min").median()
    out = tmp.to_frame("tmpf").reset_index()
    return out


def _ols_slope(minutes: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 8:
        return np.nan
    x = minutes[mask] / 60.0
    y = values[mask].astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _day_window(df: pd.DataFrame, start_min: int, end_min: int) -> pd.DataFrame:
    return df[(df["local_minute_of_day"] >= start_min) & (df["local_minute_of_day"] < end_min)]


def _utc_window(df: pd.DataFrame, start_min: int, end_min: int) -> pd.DataFrame:
    return df[(df["utc_minute_of_day"] >= start_min) & (df["utc_minute_of_day"] <= end_min)]


@dataclass
class MinuteDailyFeatures:
    local_date: datetime.date
    iem_tmax: float
    iem_tmin: float
    iem_tmean: float
    iem_tmed: float
    iem_range: float
    tmax_time_min: float
    plateau_05: float
    heat_12_15: float
    heat_15_18: float
    cool_18_21: float
    max_drop_30: float
    drop_cnt_15_19: float
    max_ts_utc: datetime


def _compute_daily_features(group: pd.DataFrame) -> MinuteDailyFeatures:
    group = group.sort_values("ts_utc")
    temps = group["tmpf"].to_numpy(dtype=float)
    iem_tmax = float(np.nanmax(temps)) if temps.size else np.nan
    iem_tmin = float(np.nanmin(temps)) if temps.size else np.nan
    iem_tmean = float(np.nanmean(temps)) if temps.size else np.nan
    iem_tmed = float(np.nanmedian(temps)) if temps.size else np.nan
    iem_range = iem_tmax - iem_tmin if np.isfinite(iem_tmax) and np.isfinite(iem_tmin) else np.nan
    tmax_mask = temps >= (iem_tmax - 0.1) if np.isfinite(iem_tmax) else np.zeros_like(temps, dtype=bool)
    if tmax_mask.any():
        tmax_time_min = float(group.loc[tmax_mask, "local_minute_of_day"].min())
    else:
        tmax_time_min = np.nan
    plateau_05 = float(np.sum(temps >= (iem_tmax - 0.5)) * 5) if np.isfinite(iem_tmax) else np.nan

    heat_12_15 = _ols_slope(
        _day_window(group, 12 * 60, 15 * 60)["local_minute_of_day"].to_numpy(dtype=float),
        _day_window(group, 12 * 60, 15 * 60)["tmpf"].to_numpy(dtype=float),
    )
    heat_15_18 = _ols_slope(
        _day_window(group, 15 * 60, 18 * 60)["local_minute_of_day"].to_numpy(dtype=float),
        _day_window(group, 15 * 60, 18 * 60)["tmpf"].to_numpy(dtype=float),
    )
    cool_18_21 = _ols_slope(
        _day_window(group, 18 * 60, 21 * 60)["local_minute_of_day"].to_numpy(dtype=float),
        _day_window(group, 18 * 60, 21 * 60)["tmpf"].to_numpy(dtype=float),
    )

    drop = group["tmpf"] - group["tmpf"].shift(-6)
    max_drop_30 = float(np.nanmax(drop.to_numpy(dtype=float))) if drop.notna().any() else np.nan

    window_15_19 = _day_window(group, 15 * 60, 19 * 60)
    drop_15_19 = window_15_19["tmpf"] - window_15_19["tmpf"].shift(-6)
    drop_cnt_15_19 = float(np.sum(drop_15_19 >= 2.0)) if drop_15_19.notna().any() else np.nan

    max_ts_utc = group["ts_utc"].max()
    return MinuteDailyFeatures(
        local_date=group["local_date"].iloc[0],
        iem_tmax=iem_tmax,
        iem_tmin=iem_tmin,
        iem_tmean=iem_tmean,
        iem_tmed=iem_tmed,
        iem_range=iem_range,
        tmax_time_min=tmax_time_min,
        plateau_05=plateau_05,
        heat_12_15=heat_12_15,
        heat_15_18=heat_15_18,
        cool_18_21=cool_18_21,
        max_drop_30=max_drop_30,
        drop_cnt_15_19=drop_cnt_15_19,
        max_ts_utc=max_ts_utc,
    )


def _compute_early_features(group: pd.DataFrame) -> dict[str, float | datetime]:
    group = group.sort_values("ts_utc")
    utc_date = group["utc_date"].iloc[0]
    day_00 = _utc_window(group, 0, 10)
    day_03 = _utc_window(group, 170, 190)
    day_06 = _utc_window(group, 350, 360)
    t00 = float(np.nanmedian(day_00["tmpf"])) if not day_00.empty else np.nan
    t03 = float(np.nanmedian(day_03["tmpf"])) if not day_03.empty else np.nan
    t06 = float(np.nanmedian(day_06["tmpf"])) if not day_06.empty else np.nan
    night_drop = t00 - t06 if np.isfinite(t00) and np.isfinite(t06) else np.nan
    last180 = _utc_window(group, 180, 360)
    slope_last180 = _ols_slope(
        last180["utc_minute_of_day"].to_numpy(dtype=float),
        last180["tmpf"].to_numpy(dtype=float),
    )
    std_last180 = float(np.nanstd(last180["tmpf"])) if not last180.empty else np.nan
    max_ts_utc = group[group["utc_minute_of_day"] <= 360]["ts_utc"].max()
    return {
        "utc_date": utc_date,
        "T00": t00,
        "T03": t03,
        "T06": t06,
        "night_drop_00_06": night_drop,
        "slope_last180": slope_last180,
        "std_last180": std_last180,
        "max_ts_utc_early": max_ts_utc,
    }


def _ewma_half_life(values: pd.Series, half_life_days: int) -> pd.Series:
    alpha = 1 - np.exp(np.log(0.5) / float(half_life_days))
    out = []
    prev = np.nan
    for val in values:
        if np.isnan(prev):
            prev = val
        else:
            prev = alpha * val + (1 - alpha) * prev
        out.append(prev)
    return pd.Series(out, index=values.index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build condensed minute features for E37.")
    parser.add_argument("--minute-dir", required=True, help="Directory with per-year minute CSVs")
    parser.add_argument("--truth-features-csv", required=True, help="Path to E37 features.csv (for truth)")
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument("--station-id", default="KMIA")
    parser.add_argument("--tz", default="America/New_York")
    args = parser.parse_args()

    minute_dir = Path(args.minute_dir)
    files = sorted([p for p in minute_dir.iterdir() if p.is_file()])
    if not files:
        raise ValueError(f"No files found in {minute_dir}")

    tz = ZoneInfo(args.tz)

    daily_rows: list[MinuteDailyFeatures] = []
    early_rows: list[dict[str, float | datetime]] = []

    for path in files:
        df = _read_file(path)
        df = _standardize_frame(df, args.station_id)
        if df.empty:
            continue
        df = _resample_5m(df)
        df["ts_local"] = df["ts_utc"].dt.tz_convert(tz)
        df["local_date"] = df["ts_local"].dt.date
        df["local_minute_of_day"] = df["ts_local"].dt.hour * 60 + df["ts_local"].dt.minute
        df["utc_date"] = df["ts_utc"].dt.date
        df["utc_minute_of_day"] = df["ts_utc"].dt.hour * 60 + df["ts_utc"].dt.minute

        for _, group in df.groupby("local_date", sort=True):
            daily_rows.append(_compute_daily_features(group))

        for _, group in df.groupby("utc_date", sort=True):
            early_rows.append(_compute_early_features(group))

    daily_df = pd.DataFrame([r.__dict__ for r in daily_rows])
    early_df = pd.DataFrame(early_rows)

    if daily_df.empty or early_df.empty:
        raise ValueError("Minute features could not be computed (empty daily or early table).")

    # Collapse any duplicates
    daily_df = daily_df.sort_values("local_date").groupby("local_date", as_index=False).first()
    early_df = early_df.sort_values("utc_date").groupby("utc_date", as_index=False).first()

    daily_df["target_date_local"] = pd.to_datetime(daily_df["local_date"]) + pd.Timedelta(days=1)
    daily_df["target_date_local"] = daily_df["target_date_local"].dt.date
    daily_df = daily_df.rename(
        columns={
            "iem_tmax": "iem_tmax_t1",
            "iem_tmin": "iem_tmin_t1",
            "iem_range": "iem_range_t1",
            "tmax_time_min": "tmax_time_min_t1",
            "plateau_05": "plateau_05_t1",
            "heat_12_15": "heat_12_15_t1",
            "heat_15_18": "heat_15_18_t1",
            "cool_18_21": "cool_18_21_t1",
            "max_drop_30": "max_drop_30_t1",
            "drop_cnt_15_19": "drop_cnt_15_19_t1",
            "max_ts_utc": "max_ts_utc_t1",
        }
    )
    daily_df = daily_df[
        [
            "target_date_local",
            "iem_tmax_t1",
            "iem_tmin_t1",
            "iem_range_t1",
            "tmax_time_min_t1",
            "plateau_05_t1",
            "heat_12_15_t1",
            "heat_15_18_t1",
            "cool_18_21_t1",
            "max_drop_30_t1",
            "drop_cnt_15_19_t1",
            "max_ts_utc_t1",
        ]
    ]

    early_df["target_date_local"] = pd.to_datetime(early_df["utc_date"]).dt.date
    early_df = early_df[
        [
            "target_date_local",
            "T00",
            "T03",
            "T06",
            "night_drop_00_06",
            "slope_last180",
            "std_last180",
            "max_ts_utc_early",
        ]
    ]

    minute = daily_df.merge(early_df, on="target_date_local", how="outer")

    # Translator features (IEM vs NWS truth)
    truth = pd.read_csv(args.truth_features_csv, parse_dates=["target_date_local"])
    truth["target_date_local"] = truth["target_date_local"].dt.date
    truth_map = truth.set_index("target_date_local")["y_actual_tmax_f"]
    diff_df = daily_df[["target_date_local", "iem_tmax_t1"]].copy()
    diff_df["local_date"] = pd.to_datetime(diff_df["target_date_local"]) - pd.Timedelta(days=1)
    diff_df["local_date"] = diff_df["local_date"].dt.date
    diff_df["y_tmax"] = diff_df["local_date"].map(truth_map)
    diff_df["diff"] = diff_df["y_tmax"] - diff_df["iem_tmax_t1"]
    diff_series = diff_df.set_index("local_date")["diff"].sort_index()
    ewma_30 = _ewma_half_life(diff_series, 30)
    diff_std_30 = diff_series.rolling(30, min_periods=5).std()
    diff_features = pd.DataFrame(
        {
            "diff_lag1": diff_series,
            "diff_ewma_30": ewma_30,
            "diff_std_30": diff_std_30,
        }
    )
    diff_features.index = pd.to_datetime(diff_features.index) + pd.Timedelta(days=1)
    diff_features["target_date_local"] = diff_features.index.date
    diff_features = diff_features.reset_index(drop=True)
    minute = minute.merge(diff_features, on="target_date_local", how="left")
    minute["T06_adj"] = minute["T06"] + minute["diff_ewma_30"]

    # Indices using train-only z-scores
    train_mask = (
        (pd.to_datetime(minute["target_date_local"]) >= pd.Timestamp("2002-01-22"))
        & (pd.to_datetime(minute["target_date_local"]) <= pd.Timestamp("2019-12-31"))
    )
    def zscore(series: pd.Series) -> pd.Series:
        train_vals = series[train_mask]
        mean = train_vals.mean()
        std = train_vals.std()
        if std == 0 or np.isnan(std):
            return series * 0.0
        return (series - mean) / std

    z_range = zscore(-minute["iem_range_t1"])
    z_plateau = zscore(minute["plateau_05_t1"])
    z_drop_cnt = zscore(minute["drop_cnt_15_19_t1"])
    z_max_drop = zscore(minute["max_drop_30_t1"])
    z_heat_12_15 = zscore(-minute["heat_12_15_t1"])
    z_heat_diff = zscore(minute["heat_12_15_t1"] - minute["heat_15_18_t1"])
    minute["MRI_suppress"] = (
        1.2 * z_range
        + 1.0 * z_plateau
        + 1.0 * z_drop_cnt
        + 0.8 * z_max_drop
        + 0.6 * z_heat_12_15
        + 0.6 * z_heat_diff
    )
    z_tmax_time = zscore(minute["tmax_time_min_t1"])
    z_heat_15_18 = zscore(minute["heat_15_18_t1"])
    minute["MRI_late"] = 1.0 * z_tmax_time + 0.8 * z_heat_15_18 - 0.6 * z_drop_cnt

    # Leakage assertions
    decision_utc = pd.to_datetime(minute["target_date_local"]).dt.tz_localize(timezone.utc) + pd.Timedelta(
        hours=6
    )
    minute["decision_utc"] = decision_utc
    minute["max_minute_ts_used_utc"] = minute[["max_ts_utc_t1", "max_ts_utc_early"]].max(axis=1)
    minute["leak_violation"] = minute["max_minute_ts_used_utc"] > minute["decision_utc"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    minute.to_parquet(out_path, index=False)

    audit = {
        "rows": int(len(minute)),
        "leak_violation_count": int(minute["leak_violation"].sum()),
    }
    audit_path = out_path.with_suffix(".audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
