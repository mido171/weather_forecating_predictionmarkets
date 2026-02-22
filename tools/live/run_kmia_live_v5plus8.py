from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from importlib.util import find_spec


def _ensure_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
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

import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from zoneinfo import ZoneInfo


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_target_date(value: str) -> date:
    cleaned = value.strip()
    if "-" in cleaned:
        return datetime.strptime(cleaned, "%Y-%m-%d").date()
    return datetime.strptime(cleaned, "%Y%m%d").date()


def _half_life_alpha(days: float) -> float:
    if days <= 0:
        return 1.0
    return 1.0 - math.exp(math.log(0.5) / days)


def _ewma_half_life(values: pd.Series, half_life_days: int) -> pd.Series:
    alpha = 1 - math.exp(math.log(0.5) / float(half_life_days))
    out = []
    prev = np.nan
    for val in values:
        if np.isnan(prev):
            prev = val
        else:
            prev = alpha * val + (1 - alpha) * prev
        out.append(prev)
    return pd.Series(out, index=values.index)


def _blend(a: float | None, b: float | None) -> float | None:
    if a is None or np.isnan(a):
        return b
    if b is None or np.isnan(b):
        return a
    return 0.5 * (a + b)


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


def _read_minute_csv(path: Path, station_id: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["station", "valid(UTC)", "tmpf"], dtype={"station": "string"})
    df = df[df["station"].str.upper().isin({station_id.upper().replace("K", ""), station_id.upper()})]
    df = df.rename(columns={"valid(UTC)": "valid_utc"})
    df["ts_utc"] = pd.to_datetime(df["valid_utc"], errors="coerce", utc=True)
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df[["ts_utc", "tmpf"]].dropna(subset=["ts_utc"]).copy()
    return df


def _fetch_iem_minute(
    station: str,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
    params = {
        "station": station.replace("K", ""),
        "vars": "tmpf",
        "sts": start_utc.strftime("%Y-%m-%d %H:%M"),
        "ets": end_utc.strftime("%Y-%m-%d %H:%M"),
        "what": "download",
        "tz": "UTC",
        "delim": "comma",
    }
    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    if "valid(UTC)" not in df.columns or "tmpf" not in df.columns:
        raise ValueError("IEM response missing required columns.")
    df = df.rename(columns={"valid(UTC)": "valid_utc"})
    df["ts_utc"] = pd.to_datetime(df["valid_utc"], errors="coerce", utc=True)
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    return df[["ts_utc", "tmpf"]].dropna(subset=["ts_utc"]).copy()


@dataclass
class MinuteDailyFeatures:
    local_date: date
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
    )


def _compute_early_features(group: pd.DataFrame) -> dict[str, float]:
    group = group.sort_values("ts_utc")
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
    return {
        "T00": t00,
        "T03": t03,
        "T06": t06,
        "night_drop_00_06": night_drop,
        "slope_last180": slope_last180,
        "std_last180": std_last180,
    }


def load_minute_features(
    *,
    minute_dir: Path,
    station_id: str,
    target_date: date,
    tz: ZoneInfo,
    feature_store: pd.DataFrame,
    truth_series: pd.Series,
) -> dict[str, float]:
    max_fs_date = pd.to_datetime(feature_store["target_date_local"]).max().date()
    diff_start = max(max_fs_date + timedelta(days=1), date(2026, 1, 1))
    if diff_start > target_date - timedelta(days=1):
        diff_start = target_date - timedelta(days=1)

    years = {target_date.year, (target_date - timedelta(days=1)).year}
    years |= {diff_start.year}
    years = sorted(years)

    start_local = datetime.combine(diff_start, time(0, 0), tzinfo=tz)
    end_local = datetime.combine(target_date - timedelta(days=1), time(23, 59), tzinfo=tz)
    end_utc = datetime.combine(target_date, time(6, 0), tzinfo=timezone.utc)
    start_utc = start_local.astimezone(timezone.utc)
    max_utc = max(end_local.astimezone(timezone.utc), end_utc)

    frames: list[pd.DataFrame] = []
    for year in years:
        path = minute_dir / f"MIA_tmpf_1min_UTC_{year}.csv"
        if path.exists():
            df = _read_minute_csv(path, station_id)
        else:
            df = _fetch_iem_minute(station_id, start_utc, max_utc)
        df = df[(df["ts_utc"] >= start_utc) & (df["ts_utc"] <= max_utc)].copy()
        frames.append(df)

    if not frames:
        raise ValueError("No minute data available for requested dates.")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts_utc")
    df = df.set_index("ts_utc").resample("5min").median().reset_index()
    df["ts_local"] = df["ts_utc"].dt.tz_convert(tz)
    df["local_date"] = df["ts_local"].dt.date
    df["local_minute_of_day"] = df["ts_local"].dt.hour * 60 + df["ts_local"].dt.minute
    df["utc_date"] = df["ts_utc"].dt.date
    df["utc_minute_of_day"] = df["ts_utc"].dt.hour * 60 + df["ts_utc"].dt.minute

    t1_date = target_date - timedelta(days=1)
    need_early = df[df["utc_date"] == target_date].empty
    need_t1 = df[df["local_date"] == t1_date].empty
    if need_early or need_t1:
        supplemental_frames: list[pd.DataFrame] = []
        if need_early:
            early_start = datetime.combine(target_date, time(0, 0), tzinfo=timezone.utc)
            early_end = datetime.combine(target_date, time(6, 0), tzinfo=timezone.utc)
            supplemental_frames.append(_fetch_iem_minute(station_id, early_start, early_end))
        if need_t1:
            t1_start_local = datetime.combine(t1_date, time(0, 0), tzinfo=tz)
            t1_end_local = datetime.combine(t1_date, time(23, 59), tzinfo=tz)
            supplemental_frames.append(
                _fetch_iem_minute(
                    station_id,
                    t1_start_local.astimezone(timezone.utc),
                    t1_end_local.astimezone(timezone.utc),
                )
            )
        if supplemental_frames:
            extra = pd.concat(supplemental_frames, ignore_index=True)
            extra = extra.sort_values("ts_utc")
            extra = extra.set_index("ts_utc").resample("5min").median().reset_index()
            extra["ts_local"] = extra["ts_utc"].dt.tz_convert(tz)
            extra["local_date"] = extra["ts_local"].dt.date
            extra["local_minute_of_day"] = extra["ts_local"].dt.hour * 60 + extra["ts_local"].dt.minute
            extra["utc_date"] = extra["ts_utc"].dt.date
            extra["utc_minute_of_day"] = extra["ts_utc"].dt.hour * 60 + extra["ts_utc"].dt.minute
            df = pd.concat([df, extra], ignore_index=True)
            df = df.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")

    daily_needed = pd.date_range(diff_start, target_date - timedelta(days=1), freq="D").date
    daily_groups = df[df["local_date"].isin(daily_needed)].groupby("local_date", sort=True)
    daily_rows = [_compute_daily_features(group) for _, group in daily_groups]
    daily_df = pd.DataFrame([r.__dict__ for r in daily_rows])
    if daily_df.empty:
        raise ValueError("No daily minute features computed.")

    early_group = df[df["utc_date"] == target_date]
    if early_group.empty:
        available_min = df["ts_utc"].min()
        available_max = df["ts_utc"].max()
        raise ValueError(
            "No early-minute data for target UTC date. "
            f"Target={target_date} available_range={available_min}..{available_max}. "
            "Re-download minute data or supply complete files for the target window."
        )
    early_feats = _compute_early_features(early_group)

    daily_df = daily_df.sort_values("local_date").groupby("local_date", as_index=False).first()
    t1_row = daily_df[daily_df["local_date"] == (target_date - timedelta(days=1))]
    if t1_row.empty:
        available_days = sorted(set(daily_df["local_date"]))[:5]
        raise ValueError(
            "Missing T-1 daily minute data. "
            f"T-1={target_date - timedelta(days=1)} available_days_head={available_days}. "
            "Re-download minute data or supply complete files for the target window."
        )
    t1 = t1_row.iloc[0]

    fs = feature_store.copy()
    fs_dates = pd.to_datetime(fs["target_date_local"]).dt.date
    diff_hist = pd.Series(
        fs["feat_iem_diff_tminus1"].to_numpy(dtype=float),
        index=fs_dates - timedelta(days=1),
    )
    diff_hist = diff_hist[diff_hist.index.notnull()].sort_index()

    extra_dates = [d for d in daily_df["local_date"] if d not in diff_hist.index]
    if extra_dates:
        extra_df = daily_df[daily_df["local_date"].isin(extra_dates)].copy()
        extra_df["y_tmax"] = extra_df["local_date"].map(truth_series)
        extra_df["diff"] = extra_df["y_tmax"] - extra_df["iem_tmax"]
        extra_series = pd.Series(extra_df["diff"].to_numpy(dtype=float), index=extra_df["local_date"])
        diff_series = pd.concat([diff_hist, extra_series]).sort_index()
    else:
        diff_series = diff_hist

    diff_series = diff_series[diff_series.index <= (target_date - timedelta(days=1))]
    ewma_30 = _ewma_half_life(diff_series, 30)
    diff_std_30 = diff_series.rolling(30, min_periods=5).std()

    diff_lag1 = float(diff_series.loc[target_date - timedelta(days=1)]) if (target_date - timedelta(days=1)) in diff_series.index else np.nan
    diff_ewma_30 = float(ewma_30.loc[target_date - timedelta(days=1)]) if (target_date - timedelta(days=1)) in ewma_30.index else np.nan
    diff_std_30_val = float(diff_std_30.loc[target_date - timedelta(days=1)]) if (target_date - timedelta(days=1)) in diff_std_30.index else np.nan

    train_mask = (
        (pd.to_datetime(feature_store["target_date_local"]) >= pd.Timestamp("2002-01-22"))
        & (pd.to_datetime(feature_store["target_date_local"]) <= pd.Timestamp("2019-12-31"))
    )

    def zscore(val: float, col: str, negate: bool = False) -> float:
        series = pd.to_numeric(feature_store[col], errors="coerce")
        train_vals = series[train_mask]
        mean = float(train_vals.mean())
        std = float(train_vals.std())
        if std == 0 or np.isnan(std):
            return 0.0
        v = -val if negate else val
        return (v - mean) / std

    z_range = zscore(float(t1["iem_range"]), "iem_range_t1", negate=True)
    z_plateau = zscore(float(t1["plateau_05"]), "plateau_05_t1")
    z_drop_cnt = zscore(float(t1["drop_cnt_15_19"]), "drop_cnt_15_19_t1")
    z_max_drop = zscore(float(t1["max_drop_30"]), "max_drop_30_t1")
    z_heat_12_15 = zscore(float(t1["heat_12_15"]), "heat_12_15_t1", negate=True)
    z_heat_diff = zscore(float(t1["heat_12_15"] - t1["heat_15_18"]), "heat_12_15_t1")
    mri_suppress = (
        1.2 * z_range
        + 1.0 * z_plateau
        + 1.0 * z_drop_cnt
        + 0.8 * z_max_drop
        + 0.6 * z_heat_12_15
        + 0.6 * z_heat_diff
    )
    z_tmax_time = zscore(float(t1["tmax_time_min"]), "tmax_time_min_t1")
    z_heat_15_18 = zscore(float(t1["heat_15_18"]), "heat_15_18_t1")
    mri_late = 1.0 * z_tmax_time + 0.8 * z_heat_15_18 - 0.6 * z_drop_cnt

    return {
        "iem_tmax_t1": float(t1["iem_tmax"]),
        "iem_tmin_t1": float(t1["iem_tmin"]),
        "iem_range_t1": float(t1["iem_range"]),
        "tmax_time_min_t1": float(t1["tmax_time_min"]),
        "plateau_05_t1": float(t1["plateau_05"]),
        "heat_12_15_t1": float(t1["heat_12_15"]),
        "heat_15_18_t1": float(t1["heat_15_18"]),
        "cool_18_21_t1": float(t1["cool_18_21"]),
        "max_drop_30_t1": float(t1["max_drop_30"]),
        "drop_cnt_15_19_t1": float(t1["drop_cnt_15_19"]),
        "T00": float(early_feats["T00"]),
        "T03": float(early_feats["T03"]),
        "T06": float(early_feats["T06"]),
        "night_drop_00_06": float(early_feats["night_drop_00_06"]),
        "slope_last180": float(early_feats["slope_last180"]),
        "std_last180": float(early_feats["std_last180"]),
        "diff_lag1": diff_lag1,
        "diff_ewma_30": diff_ewma_30,
        "diff_std_30": diff_std_30_val,
        "T06_adj": float(early_feats["T06"]) + diff_ewma_30 if np.isfinite(diff_ewma_30) else np.nan,
        "MRI_suppress": float(mri_suppress),
        "MRI_late": float(mri_late),
    }


def load_station_truth(engine_url: str, station_id: str, start_date: date, end_date: date) -> pd.Series:
    engine = create_engine(engine_url, pool_pre_ping=True)
    sql = """
        SELECT date_local, tmax_f
        FROM station_daily_truth
        WHERE station_id = :station_id
          AND date_local BETWEEN :start_date AND :end_date
    """
    df = pd.read_sql(
        text(sql),
        engine,
        params={
            "station_id": station_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    return pd.Series(df["tmax_f"].to_numpy(dtype=float), index=df["date_local"])


def fetch_mos_rows(
    engine_url: str,
    station_id: str,
    models: list[str],
    var_codes: list[str],
    target_date: date,
) -> pd.DataFrame:
    engine = create_engine(engine_url, pool_pre_ping=True)
    placeholders_models = ", ".join([f":m{i}" for i in range(len(models))])
    placeholders_vars = ", ".join([f":v{i}" for i in range(len(var_codes))])
    sql = f"""
        SELECT id, station_id, model, variable_code, target_date_local, asof_utc, runtime_utc, retrieved_at_utc,
               value_mean, value_max, value_min
        FROM mos_daily_value
        WHERE station_id = :station_id
          AND model IN ({placeholders_models})
          AND variable_code IN ({placeholders_vars})
          AND target_date_local = :target_date
    """
    params: dict[str, Any] = {
        "station_id": station_id,
        "target_date": target_date.isoformat(),
    }
    params.update({f"m{i}": m for i, m in enumerate(models)})
    params.update({f"v{i}": v for i, v in enumerate(var_codes)})
    return pd.read_sql(text(sql), engine, params=params)


def select_latest_mos(df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    df = df.copy()
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True, errors="coerce")
    df["runtime_utc"] = pd.to_datetime(df["runtime_utc"], utc=True, errors="coerce")
    df["retrieved_at_utc"] = pd.to_datetime(df["retrieved_at_utc"], utc=True, errors="coerce")
    df["model"] = df["model"].astype(str).str.lower()
    df["variable_code"] = df["variable_code"].astype(str).str.lower()
    df = df[df["asof_utc"] <= cutoff]
    df = df.sort_values(
        ["model", "variable_code", "asof_utc", "runtime_utc", "retrieved_at_utc", "id"]
    )
    latest = df.groupby(["model", "variable_code"], as_index=False).tail(1)
    return latest


def build_mos_features(
    mos_df: pd.DataFrame,
    decision_utc: datetime,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    max_vars = {"tmp", "p12", "q12"}
    min_vars = {"cig", "vis"}

    def _value(row: pd.Series, var: str) -> float:
        if var in max_vars:
            return float(row["value_max"]) if pd.notna(row["value_max"]) else float(row["value_mean"])
        if var in min_vars:
            return float(row["value_min"]) if pd.notna(row["value_min"]) else float(row["value_mean"])
        return float(row["value_mean"]) if pd.notna(row["value_mean"]) else float(row["value_max"])

    buckets = [0, 12, 24, 36, 48]
    bucket_values: dict[str, dict[str, float]] = {}
    meta: dict[str, Any] = {"latest_asof_used": None}

    for bucket in buckets:
        cutoff = decision_utc - timedelta(hours=bucket)
        latest = select_latest_mos(mos_df, cutoff)
        if latest.empty:
            continue
        bucket_vals: dict[str, float] = {}
        for _, row in latest.iterrows():
            model = row["model"].lower()
            var = row["variable_code"].lower()
            key = f"{model}_{var}_b{bucket}"
            bucket_vals[key] = _value(row, var)
        bucket_values[f"b{bucket}"] = bucket_vals
        max_asof = latest["asof_utc"].max()
        if max_asof is not pd.NaT:
            meta["latest_asof_used"] = max_asof if meta["latest_asof_used"] is None else max(meta["latest_asof_used"], max_asof)

    def get_val(model: str, var: str, bucket: int) -> float:
        return bucket_values.get(f"b{bucket}", {}).get(f"{model}_{var}_b{bucket}", np.nan)

    gfs_tmp_max = get_val("gfs", "tmp", 0)
    nam_tmp_max = get_val("nam", "tmp", 0)
    gfs_tmp_min = get_val("gfs", "tmp", 0)
    nam_tmp_min = get_val("nam", "tmp", 0)
    gfs_tmp_mean = get_val("gfs", "tmp", 0)
    nam_tmp_mean = get_val("nam", "tmp", 0)

    gfs_dpt_mean = get_val("gfs", "dpt", 0)
    nam_dpt_mean = get_val("nam", "dpt", 0)
    gfs_wdr_mean = get_val("gfs", "wdr", 0)
    nam_wdr_mean = get_val("nam", "wdr", 0)
    gfs_wsp_mean = get_val("gfs", "wsp", 0)
    nam_wsp_mean = get_val("nam", "wsp", 0)
    gfs_p12_max = get_val("gfs", "p12", 0)
    nam_p12_max = get_val("nam", "p12", 0)
    gfs_q12_max = get_val("gfs", "q12", 0)
    nam_q12_max = get_val("nam", "q12", 0)
    gfs_cig_min = get_val("gfs", "cig", 0)
    nam_cig_min = get_val("nam", "cig", 0)

    tmp_max_mean_models = _blend(gfs_tmp_max, nam_tmp_max)
    tmp_min_mean_models = _blend(gfs_tmp_min, nam_tmp_min)
    tmp_mean_mean_models = _blend(gfs_tmp_mean, nam_tmp_mean)
    dpt_mean_models = _blend(gfs_dpt_mean, nam_dpt_mean)

    wsp_mean = _blend(gfs_wsp_mean, nam_wsp_mean)
    wdr_mean = _blend(gfs_wdr_mean, nam_wdr_mean)
    wdr_rad = math.radians(wdr_mean) if wdr_mean is not None and np.isfinite(wdr_mean) else np.nan
    u = -wsp_mean * math.sin(wdr_rad) if np.isfinite(wsp_mean) and np.isfinite(wdr_rad) else np.nan
    v = -wsp_mean * math.cos(wdr_rad) if np.isfinite(wsp_mean) and np.isfinite(wdr_rad) else np.nan

    features = {
        "feat_tmp_max_mean_models": tmp_max_mean_models,
        "feat_tmp_min_mean_models": tmp_min_mean_models,
        "feat_tmp_mean_mean_models": tmp_mean_mean_models,
        "feat_tmp_range_mean_models": tmp_max_mean_models - tmp_min_mean_models if np.isfinite(tmp_max_mean_models) and np.isfinite(tmp_min_mean_models) else np.nan,
        "feat_dpt_mean_models": dpt_mean_models,
        "feat_dd_models": tmp_max_mean_models - dpt_mean_models if np.isfinite(tmp_max_mean_models) and np.isfinite(dpt_mean_models) else np.nan,
        "feat_wsp_mean": wsp_mean,
        "feat_wdr_mean": wdr_mean,
        "feat_u": u,
        "feat_v": v,
        "feat_p12_max": _blend(gfs_p12_max, nam_p12_max),
        "feat_q12_max": _blend(gfs_q12_max, nam_q12_max),
        "feat_cig_min": _blend(gfs_cig_min, nam_cig_min),
        "feat_onshore": 1.0 if np.isfinite(wdr_mean) and 45 <= wdr_mean <= 135 else 0.0,
    }

    tmp_buckets = {}
    for bucket in [0, 12, 24, 36]:
        tmp_buckets[f"feat_tmp_max_gfs_b{bucket}"] = get_val("gfs", "tmp", bucket)
        tmp_buckets[f"feat_tmp_max_nam_b{bucket}"] = get_val("nam", "tmp", bucket)

    return features, tmp_buckets, meta


def compute_base(
    feature_store: pd.DataFrame,
    tmp_buckets: dict[str, float],
    target_date: date,
) -> float:
    hist = feature_store.copy()
    hist["target_date_local"] = pd.to_datetime(hist["target_date_local"]).dt.date
    hist = hist[hist["target_date_local"] <= (target_date - timedelta(days=1))]
    if hist.empty:
        raise ValueError("No historical rows available for baseline.")
    last = hist.iloc[-1]
    alpha_15 = _half_life_alpha(15.0)
    corrected = []
    for model in ["gfs", "nam"]:
        for bucket in [0, 12, 24, 36]:
            raw_prev = float(last.get(f"feat_tmp_max_{model}_b{bucket}", np.nan))
            corr_prev = float(last.get(f"feat_tmp_corr_{model}_b{bucket}", np.nan))
            bias_prev = corr_prev - raw_prev if np.isfinite(corr_prev) and np.isfinite(raw_prev) else np.nan
            err_prev = float(last["y_actual_tmax_f"]) - raw_prev if np.isfinite(raw_prev) else np.nan
            if np.isnan(bias_prev):
                bias_prev = 0.0
            if np.isnan(err_prev):
                bias_t = bias_prev
            else:
                bias_t = (1 - alpha_15) * bias_prev + alpha_15 * err_prev
            raw_t = tmp_buckets.get(f"feat_tmp_max_{model}_b{bucket}", np.nan)
            corrected.append(raw_t + bias_t if np.isfinite(raw_t) else np.nan)
    corrected = np.array(corrected, dtype=float)
    return float(np.nanmedian(corrected))


def train_models(
    df: pd.DataFrame,
    *,
    seed: int,
    feature_cols: list[str],
    gate_features: list[str],
    base_col: str,
) -> tuple[Any, Any, Any]:
    import sys

    sys.path.append("ml")
    from ml import run_mos_45_suite as base

    split = base.split_by_date(
        df,
        train_start="2002-01-22",
        train_end="2019-12-31",
        val_start="2020-01-01",
        val_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2025-12-31",
    )
    train_mask = split["train_mask"]
    val_mask = split["val_mask"]

    y = pd.to_numeric(df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float)
    gate_label = (pd.to_numeric(df.get("feat_onshore"), errors="coerce") > 0.5).astype(int).to_numpy(dtype=int)

    gate_df = base.ensure_columns(df, gate_features)
    gate_X, _ = base.impute_features(gate_df[gate_features], train_mask)
    X_gate = gate_X.to_numpy(dtype=float)
    gate_model = base.train_lgbm_classifier(
        X_gate[train_mask],
        gate_label[train_mask],
        X_gate[val_mask],
        gate_label[val_mask],
        seed=seed,
    )

    exp_df = base.ensure_columns(df, feature_cols)
    exp_X, _ = base.impute_features(exp_df[feature_cols], train_mask)
    X = exp_X.to_numpy(dtype=float)
    base_vals = pd.to_numeric(df.get(base_col), errors="coerce").to_numpy(dtype=float)
    base_mean = float(np.nanmean(y[train_mask]))
    base_vals = np.where(np.isnan(base_vals), base_mean, base_vals)

    def fit(mask: np.ndarray, alpha: float) -> Any:
        train_idx = mask & train_mask
        val_idx = mask & val_mask
        return base.train_lgbm_quantile(
            X[train_idx],
            y[train_idx] - base_vals[train_idx],
            X[val_idx],
            y[val_idx] - base_vals[val_idx],
            seed=seed,
            alpha=alpha,
        )

    model_on_10 = fit(gate_label == 1, 0.1)
    model_on_50 = fit(gate_label == 1, 0.5)
    model_on_90 = fit(gate_label == 1, 0.9)
    model_off_10 = fit(gate_label == 0, 0.1)
    model_off_50 = fit(gate_label == 0, 0.5)
    model_off_90 = fit(gate_label == 0, 0.9)

    models = {
        "on_10": model_on_10,
        "on_50": model_on_50,
        "on_90": model_on_90,
        "off_10": model_off_10,
        "off_50": model_off_50,
        "off_90": model_off_90,
    }
    return gate_model, models, split


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V5+8 live pipeline for KMIA.")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--target-date", required=True, help="Target date local (YYYYMMDD or YYYY-MM-DD).")
    parser.add_argument("--minute-dir", default="data/iem_minute_data/MIA/tmpf/UTC/yearly")
    parser.add_argument(
        "--winner-dir",
        default="artifacts/experiments/winners/V5_PLUS8_20260219T222321Z",
        help="Winner artifact root used for feature store and calibration (no overrides).",
    )
    parser.add_argument(
        "--feature-store",
        default="",
        help="(Disabled) Feature store override. The script always uses the winner manifest.",
    )
    parser.add_argument(
        "--calibration-json",
        default="",
        help="(Disabled) Calibration override. The script always uses the winner calibration.",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--thresholds", help="Comma-separated thresholds for probabilities (e.g., 80,85,90)")
    args = parser.parse_args()

    station_id = args.station.upper()
    target_date = parse_target_date(args.target_date)
    decision_utc = datetime.combine(target_date, time(6, 0), tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)

    db_url = os.getenv("MYSQL_URL") or f"mysql+pymysql://{os.getenv('MYSQL_USER','root')}:{os.getenv('MYSQL_PASSWORD','root')}@{os.getenv('MYSQL_HOST','localhost')}:{os.getenv('MYSQL_PORT','3306')}/{os.getenv('MYSQL_DB','weather_predictionmarkets')}"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("artifacts/live_v5plus8") / utc_now_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Target date: {target_date} decision_utc: {decision_utc.isoformat()} now_utc: {now_utc.isoformat()}")

    repo_root = Path(__file__).resolve().parents[2]

    def _resolve_repo_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return repo_root / path

    if args.feature_store:
        raise ValueError("Feature store override is disabled. Use --winner-dir only.")
    if args.calibration_json:
        raise ValueError("Calibration override is disabled. Use --winner-dir only.")

    winner_dir = _resolve_repo_path(args.winner_dir)
    manifest_path = winner_dir / "config_snapshot" / "manifest.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Winner manifest not found: {manifest_path}")

    manifest_text = manifest_path.read_text(encoding="utf-8")
    feature_store_value = ""
    for line in manifest_text.splitlines():
        if line.strip().startswith("feature_store="):
            feature_store_value = line.split("=", 1)[1].strip()
            break
    if not feature_store_value:
        raise ValueError(f"feature_store not found in manifest: {manifest_path}")

    feature_store_path = _resolve_repo_path(feature_store_value)
    if not feature_store_path.exists():
        raise FileNotFoundError(f"Feature store parquet not found: {feature_store_path}")

    calibration_path = winner_dir / "calibration_report.json"
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration report not found: {calibration_path}")

    if args.verbose:
        print(f"Winner dir: {winner_dir}")
        print(f"Feature store: {feature_store_path}")
        print(f"Calibration: {calibration_path}")

    feature_store = pd.read_parquet(feature_store_path)
    truth_series = load_station_truth(db_url, station_id, date(2002, 1, 1), target_date)
    tz = ZoneInfo("America/New_York")

    minute_features = load_minute_features(
        minute_dir=_resolve_repo_path(args.minute_dir),
        station_id=station_id,
        target_date=target_date,
        tz=tz,
        feature_store=feature_store,
        truth_series=truth_series,
    )

    mos_df = fetch_mos_rows(
        db_url,
        station_id,
        ["gfs", "nam"],
        ["tmp", "dpt", "wdr", "wsp", "p12", "q12", "cig", "vis"],
        target_date,
    )
    mos_features, tmp_buckets, mos_meta = build_mos_features(mos_df, decision_utc)

    if args.verbose:
        latest_asof = mos_meta.get("latest_asof_used")
        if latest_asof is None:
            print("MOS as-of check: No MOS rows found for target date.")
        else:
            ok = latest_asof <= decision_utc
            print(f"MOS as-of check: latest_asof={latest_asof} decision_utc={decision_utc} ok={ok}")

    base_val = compute_base(feature_store, tmp_buckets, target_date)

    doy = target_date.timetuple().tm_yday
    cal_d_doy_sin = math.sin(2 * math.pi * doy / 365.0)
    cal_d_doy_cos = math.cos(2 * math.pi * doy / 365.0)

    target_row = {
        "target_date_local": target_date,
        "y_actual_tmax_f": np.nan,
        "feat_le_median_biascorr": base_val,
        "cal_d_doy_sin": cal_d_doy_sin,
        "cal_d_doy_cos": cal_d_doy_cos,
    }
    target_row.update(mos_features)
    target_row.update(tmp_buckets)
    target_row.update(minute_features)

    df = feature_store.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df = pd.concat([df, pd.DataFrame([target_row])], ignore_index=True)

    gate_features = ["feat_u", "feat_v", "feat_wsp_mean", "cal_d_doy_sin", "cal_d_doy_cos"]
    expert_features = [
        "feat_dd_models",
        "feat_tmp_range_mean_models",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
        "cal_d_doy_sin",
        "cal_d_doy_cos",
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
        "T00",
        "T03",
        "T06",
        "night_drop_00_06",
        "slope_last180",
        "std_last180",
        "T06_adj",
        "diff_lag1",
        "diff_ewma_30",
        "diff_std_30",
        "MRI_suppress",
        "MRI_late",
    ]

    gate_model, models, split = train_models(
        df,
        seed=42,
        feature_cols=expert_features,
        gate_features=gate_features,
        base_col="feat_le_median_biascorr",
    )

    from ml import run_mos_45_suite as base

    train_mask = split["train_mask"]
    val_mask = split["val_mask"]
    exp_df = base.ensure_columns(df, expert_features)
    exp_X, _ = base.impute_features(exp_df[expert_features], train_mask)
    X = exp_X.to_numpy(dtype=float)
    gate_df = base.ensure_columns(df, gate_features)
    gate_X, _ = base.impute_features(gate_df[gate_features], train_mask)
    X_gate = gate_X.to_numpy(dtype=float)
    p_gate = gate_model.predict_proba(X_gate)[:, 1]

    def resid_pred(model_on: Any, model_off: Any) -> np.ndarray:
        r_on = model_on.predict(X)
        r_off = model_off.predict(X)
        return p_gate * r_on + (1 - p_gate) * r_off

    r10 = resid_pred(models["on_10"], models["off_10"])
    r50 = resid_pred(models["on_50"], models["off_50"])
    r90 = resid_pred(models["on_90"], models["off_90"])
    spread = r90 - r10

    y = pd.to_numeric(df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float)
    base_vals = pd.to_numeric(df["feat_le_median_biascorr"], errors="coerce").to_numpy(dtype=float)
    k_grid = [0.0, 0.2, 0.4, 0.6, 0.8]
    best_k = 0.0
    best_mae = 1e9
    for k in k_grid:
        w = np.exp(-k * spread)
        pred = base_vals + w * r50
        mae = np.nanmean(np.abs(y[val_mask] - pred[val_mask]))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    w_final = np.exp(-best_k * spread)
    pred_v5p8 = base_vals + w_final * r50

    target_idx = len(df) - 1
    point_pred = float(pred_v5p8[target_idx])

    calib = json.loads(calibration_path.read_text(encoding="utf-8"))
    tau_vals = calib.get("cqr_hybrid_details", {}).get("tau_alpha_val", {})
    tau_05 = float(tau_vals.get("tau_0.05", 0.0))
    tau_10 = float(tau_vals.get("tau_0.10", 0.0))
    q05 = float(base_vals[target_idx] + r10[target_idx] - tau_05)
    q95 = float(base_vals[target_idx] + r90[target_idx] + tau_05)
    q10 = float(base_vals[target_idx] + r10[target_idx] - tau_10)
    q90 = float(base_vals[target_idx] + r90[target_idx] + tau_10)
    q50 = float(base_vals[target_idx] + r50[target_idx])
    quantiles = [q05, q10, q50, q90, q95]
    quantiles_sorted = sorted(quantiles)
    q05, q10, q50, q90, q95 = quantiles_sorted[0], quantiles_sorted[1], quantiles_sorted[2], quantiles_sorted[3], quantiles_sorted[4]

    output = {
        "station_id": station_id,
        "target_date_local": target_date.isoformat(),
        "decision_utc": decision_utc.isoformat(),
        "latest_mos_asof": mos_meta.get("latest_asof_used").isoformat() if mos_meta.get("latest_asof_used") else None,
        "mos_asof_ok": bool(mos_meta.get("latest_asof_used") and mos_meta.get("latest_asof_used") <= decision_utc),
        "point_forecast": point_pred,
        "p_gate": float(p_gate[target_idx]),
        "base": float(base_vals[target_idx]),
        "v5p8_k": float(best_k),
        "distribution": {
            "q05": q05,
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "q95": q95,
            "tau_05": tau_05,
            "tau_10": tau_10,
        },
    }

    thresholds = []
    if args.thresholds:
        thresholds = [int(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    if thresholds:
        q_levels = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
        q_vals = np.array([q05, q10, q50, q90, q95])
        probs = {}
        for thr in thresholds:
            cdf = float(np.interp(thr, q_vals, q_levels, left=0.0, right=1.0))
            probs[f"ge_{thr}"] = 1.0 - cdf
        output["threshold_probs"] = probs

    out_path = out_dir / "live_prediction.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.verbose:
        print(json.dumps(output, indent=2))
    else:
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
