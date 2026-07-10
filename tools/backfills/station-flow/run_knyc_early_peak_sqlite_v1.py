from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Downcasting object dtype arrays on \\.fillna, \\.ffill, \\.bfill is deprecated.*",
)


STATIONS = ["KNYC", "KLGA", "KJFK", "KEWR", "KTEB", "KHPN", "KISP", "KBDR"]
MAIN_STATION = "KNYC"
NEIGHBORS = [s for s in STATIONS if s != MAIN_STATION]
COASTAL = ["KJFK", "KISP", "KBDR"]
INLAND = ["KEWR", "KTEB", "KHPN"]
ALL_CUTOFFS = list(range(0, 1440, 30))
TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2025-12-31"
MAX_STALENESS_MINUTES = 90
FFILL_LIMIT_STEPS = MAX_STALENESS_MINUTES // 30

DENSE_STATE_COLS = [
    "temp",
    "dew_pt",
    "rh",
    "pressure",
    "vis",
    "wspd",
    "wdir",
    "clds",
    "wx_phrase",
    "wdir_cardinal",
]
EVENT_STATE_COLS = ["gust", "precip_hrly", "uv_index", "uv_desc"]
RAW_OBS_COLS = DENSE_STATE_COLS + EVENT_STATE_COLS
ALLOWED_CLDS = {"CLR", "FEW", "SCT", "BKN", "OVC"}
CLDS_ORD = {"CLR": 0.0, "FEW": 1.0, "SCT": 2.0, "BKN": 3.0, "OVC": 4.0}
ALLOWED_WDIR_CARD = {
    "CALM",
    "VAR",
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
}
MARINE_CARDINALS = {"E", "ENE", "ESE", "SE", "SSE", "S"}
CONTINENTAL_CARDINALS = {"W", "WNW", "WSW", "NW", "NNW", "N"}
PAIRWISE_QA_FIELDS = ["temp", "dew_pt", "rh", "pressure", "vis", "wspd", "wdir", "clds", "wx_phrase"]


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    results_root: Path
    run_id: str
    train_end: str = TRAIN_END
    val_start: str = VAL_START
    val_end: str = VAL_END
    test_start: str = TEST_START
    test_end: str = TEST_END


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a fresh KNYC early-peak model from EarlyPeak SQLite only.")
    p.add_argument("--data-root", default=r"D:\Ahmed\data\sqlite\EarlyPeak")
    p.add_argument("--results-root", default=r"D:\Ahmed\data\sqlite\EarlyPeak\results")
    p.add_argument("--run-id", default="")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_logger(level_name: str, log_path: Path | None = None) -> logging.Logger:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logger = logging.getLogger("knyc_early_peak_v1")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def read_sql_table(db_path: Path, sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def load_station_source(data_root: Path, station_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    station_dir = data_root / station_id
    dbs = sorted(station_dir.glob("*.sqlite"))
    if not dbs:
        raise FileNotFoundError(f"No sqlite DB found for {station_id} under {station_dir}")
    db_path = dbs[0]
    obs = read_sql_table(
        db_path,
        """
        SELECT station_id, request_location_id, valid_time_utc, valid_time_local, target_date_local,
               cutoff_minutes_local, temp, dew_pt, rh, pressure, vis, wspd, wdir, gust, precip_hrly,
               clds, wx_phrase, uv_index, uv_desc, wdir_cardinal
        FROM wu_observations_30m
        ORDER BY target_date_local, cutoff_minutes_local, valid_time_local
        """,
    )
    truth = read_sql_table(
        db_path,
        """
        SELECT station_id, station_usw, target_date_local, settled_tmax, truth_source, source_record_id, retrieved_at_utc
        FROM nws_settled_tmax_daily
        ORDER BY target_date_local
        """,
    )
    summary = {
        "station_id": station_id,
        "db_path": str(db_path),
        "obs_rows": int(len(obs)),
        "obs_min_date": str(obs["target_date_local"].min()),
        "obs_max_date": str(obs["target_date_local"].max()),
        "obs_days": int(obs["target_date_local"].nunique()),
        "truth_rows": int(len(truth)),
        "truth_min_date": str(truth["target_date_local"].min()),
        "truth_max_date": str(truth["target_date_local"].max()),
    }
    return obs, truth, summary


def _sanitize_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v).strip()


def _parse_wx_flags(text_value: Any) -> dict[str, float]:
    txt = _sanitize_text(text_value).lower()
    has = lambda parts: any(p in txt for p in parts)
    return {
        "wx_precip_flag": float(has(("rain", "drizzle", "snow", "sleet", "hail", "freezing", "wintry", "mix"))),
        "wx_convective_flag": float(has(("thunder", "t-storm", "tstorm", "storm", "showers"))),
        "wx_fog_flag": float(has(("fog", "mist"))),
        "wx_haze_flag": float(has(("haze", "smoke", "dust"))),
        "wx_frozen_flag": float(has(("snow", "sleet", "freezing", "hail", "ice", "wintry"))),
        "wx_windy_flag": float(has(("windy", "breezy"))),
    }


def clean_station_obs(obs: pd.DataFrame, station_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = obs.copy()
    df["station_id"] = station_id
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.strftime("%Y-%m-%d")
    df["valid_time_local"] = pd.to_datetime(df["valid_time_local"], errors="coerce")
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], errors="coerce")
    df["cutoff_minutes_local"] = pd.to_numeric(df["cutoff_minutes_local"], errors="coerce").astype("Int64")

    for col in ["temp", "dew_pt", "rh", "pressure", "vis", "wspd", "wdir", "gust", "precip_hrly", "uv_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["clds"] = df["clds"].map(_sanitize_text).str.upper()
    df["wx_phrase"] = df["wx_phrase"].map(_sanitize_text)
    df["uv_desc"] = df["uv_desc"].map(_sanitize_text)
    df["wdir_cardinal"] = df["wdir_cardinal"].map(_sanitize_text).str.upper()

    rule_counts: list[dict[str, Any]] = []

    def _null_invalid(mask: pd.Series, field_name: str) -> None:
        count = int(mask.fillna(False).sum())
        if count > 0:
            df.loc[mask, field_name] = np.nan
        rule_counts.append({"station_id": station_id, "field_name": field_name, "invalidated_rows": count})

    _null_invalid((df["temp"] < -20) | (df["temp"] > 110), "temp")
    _null_invalid((df["dew_pt"] < -40) | (df["dew_pt"] > 90), "dew_pt")
    _null_invalid(df["dew_pt"] > (df["temp"] + 3), "dew_pt")
    _null_invalid((df["rh"] < 2) | (df["rh"] > 100), "rh")
    _null_invalid((df["pressure"] < 27.5) | (df["pressure"] > 31.5), "pressure")
    _null_invalid((df["vis"] < 0) | (df["vis"] > 50), "vis")
    _null_invalid((df["wspd"] < 0) | (df["wspd"] > 80), "wspd")
    _null_invalid((df["wdir"] < 0) | (df["wdir"] > 360), "wdir")
    _null_invalid((df["gust"] < 0) | (df["gust"] > 100), "gust")
    _null_invalid((df["precip_hrly"] < 0) | (df["precip_hrly"] > 6), "precip_hrly")
    _null_invalid((df["uv_index"] < 0) | (df["uv_index"] > 15), "uv_index")

    bad_clds = ~df["clds"].isin(ALLOWED_CLDS) & df["clds"].ne("")
    rule_counts.append({"station_id": station_id, "field_name": "clds", "invalidated_rows": int(bad_clds.sum())})
    df.loc[bad_clds, "clds"] = ""

    bad_card = ~df["wdir_cardinal"].isin(ALLOWED_WDIR_CARD) & df["wdir_cardinal"].ne("")
    rule_counts.append({"station_id": station_id, "field_name": "wdir_cardinal", "invalidated_rows": int(bad_card.sum())})
    df.loc[bad_card, "wdir_cardinal"] = ""

    df = df.sort_values(["target_date_local", "cutoff_minutes_local", "valid_time_local"])
    df = df.groupby(["target_date_local", "cutoff_minutes_local"], as_index=False).tail(1).reset_index(drop=True)
    return df, pd.DataFrame(rule_counts)


def _cumulative_extrema(values: pd.Series, mode: str) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    cur = np.nan
    for i, v in enumerate(arr):
        if np.isfinite(v):
            if not np.isfinite(cur):
                cur = v
            else:
                cur = max(cur, v) if mode == "max" else min(cur, v)
        out[i] = cur
    return out


def _last_extreme_minute(values: pd.Series, cutoffs: pd.Series, mode: str) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mins = pd.to_numeric(cutoffs, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    cur = np.nan
    cur_minute = np.nan
    for i, v in enumerate(arr):
        if np.isfinite(v):
            if not np.isfinite(cur):
                cur = v
                cur_minute = mins[i]
            else:
                better = v >= cur if mode == "max" else v <= cur
                if better:
                    cur = v
                    cur_minute = mins[i]
        out[i] = cur_minute
    return out


def build_station_panel(
    obs_clean: pd.DataFrame,
    target_dates: list[str],
    station_id: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    started = time.perf_counter()
    if logger is not None:
        logger.info(
            "BUILD_PANEL_START station=%s target_days=%s source_rows=%s",
            station_id,
            len(target_dates),
            len(obs_clean),
        )

    base = pd.MultiIndex.from_product(
        [pd.Index(target_dates, name="target_date_local"), ALL_CUTOFFS],
        names=["target_date_local", "cutoff_minutes_local"],
    ).to_frame(index=False)
    keep_cols = ["target_date_local", "cutoff_minutes_local", "valid_time_local", "valid_time_utc"] + RAW_OBS_COLS
    grid = base.merge(obs_clean[keep_cols], on=["target_date_local", "cutoff_minutes_local"], how="left", sort=False)
    grid = grid.sort_values(["target_date_local", "cutoff_minutes_local"]).reset_index(drop=True)
    grid["station_id"] = station_id
    day_key = grid["target_date_local"]

    if logger is not None:
        logger.info(
            "BUILD_PANEL_PROGRESS station=%s stage=merge_done rows=%s elapsed_sec=%.1f",
            station_id,
            len(grid),
            time.perf_counter() - started,
        )

    grid["has_actual_row"] = grid["valid_time_local"].notna().astype(int)
    grid["last_actual_cutoff"] = grid["cutoff_minutes_local"].where(grid["has_actual_row"].eq(1))
    grid["last_actual_cutoff"] = grid.groupby(day_key, sort=False)["last_actual_cutoff"].ffill()
    grid["age_any_minutes"] = grid["cutoff_minutes_local"] - grid["last_actual_cutoff"]
    grid.loc[grid["last_actual_cutoff"].isna(), "age_any_minutes"] = np.nan
    grid["has_recent_30m"] = (grid["age_any_minutes"] <= 30).fillna(False).astype(int)
    grid["has_recent_60m"] = (grid["age_any_minutes"] <= 60).fillna(False).astype(int)
    grid["has_recent_90m"] = (grid["age_any_minutes"] <= 90).fillna(False).astype(int)

    for col in DENSE_STATE_COLS:
        grid[f"{col}_current"] = grid.groupby(day_key, sort=False)[col].ffill(limit=FFILL_LIMIT_STEPS)
        last_valid = grid["cutoff_minutes_local"].where(grid[col].notna())
        last_valid = last_valid.groupby(day_key, sort=False).ffill()
        grid[f"age_{col}_minutes"] = grid["cutoff_minutes_local"] - last_valid
        grid.loc[last_valid.isna(), f"age_{col}_minutes"] = np.nan

    grid["gust_current"] = grid["gust"]
    grid["precip_hrly_current"] = grid["precip_hrly"]

    if logger is not None:
        logger.info(
            "BUILD_PANEL_PROGRESS station=%s stage=current_state_done elapsed_sec=%.1f",
            station_id,
            time.perf_counter() - started,
        )

    temp_group = grid.groupby(day_key, sort=False)["temp"]
    grid["temp_high_sofar"] = temp_group.cummax()
    grid["temp_low_sofar"] = temp_group.cummin()
    grid["last_tmax_cutoff"] = grid["cutoff_minutes_local"].where(
        grid["temp"].notna() & np.isclose(grid["temp"], grid["temp_high_sofar"], equal_nan=False)
    )
    grid["last_tmax_cutoff"] = grid.groupby(day_key, sort=False)["last_tmax_cutoff"].ffill()
    grid["time_since_tmax_minutes"] = grid["cutoff_minutes_local"] - grid["last_tmax_cutoff"]
    grid.loc[grid["last_tmax_cutoff"].isna(), "time_since_tmax_minutes"] = np.nan
    grid["obs_count_sofar"] = grid["temp"].notna().groupby(day_key, sort=False).cumsum().astype(float)
    grid["temp_gap_from_high"] = grid["temp_current"] - grid["temp_high_sofar"]
    grid["temp_gap_from_low"] = grid["temp_current"] - grid["temp_low_sofar"]

    for lag_min in [60, 120, 180, 240]:
        steps = lag_min // 30
        lagged = grid.groupby(day_key, sort=False)["temp_current"].shift(steps)
        grid[f"temp_delta_{lag_min}"] = grid["temp_current"] - lagged
    grid["temp_slope_60"] = grid["temp_delta_60"]
    grid["temp_slope_120"] = grid["temp_delta_120"] / 2.0
    grid["dewpoint_depression"] = grid["temp_current"] - grid["dew_pt_current"]
    grid["dew_pt_delta_60"] = grid["dew_pt_current"] - grid.groupby(day_key, sort=False)["dew_pt_current"].shift(2)
    grid["dew_pt_delta_120"] = grid["dew_pt_current"] - grid.groupby(day_key, sort=False)["dew_pt_current"].shift(4)
    grid["pressure_delta_60"] = grid["pressure_current"] - grid.groupby(day_key, sort=False)["pressure_current"].shift(2)
    grid["pressure_delta_180"] = grid["pressure_current"] - grid.groupby(day_key, sort=False)["pressure_current"].shift(6)
    grid["vis_delta_60"] = grid["vis_current"] - grid.groupby(day_key, sort=False)["vis_current"].shift(2)
    grid["wspd_delta_60"] = grid["wspd_current"] - grid.groupby(day_key, sort=False)["wspd_current"].shift(2)

    temp_diff = grid.groupby(day_key, sort=False)["temp_current"].diff()
    rise_flag = pd.Series(np.where(temp_diff.notna(), (temp_diff > 0).astype(float), np.nan), index=grid.index)
    fall_flag = pd.Series(np.where(temp_diff.notna(), (temp_diff < 0).astype(float), np.nan), index=grid.index)
    for window_min in [120, 240]:
        steps = (window_min // 30) + 1
        rolling = grid.groupby(day_key, sort=False)["temp_current"].rolling(steps, min_periods=1)
        grid[f"temp_mean_{window_min}"] = rolling.mean().reset_index(level=0, drop=True)
        grid[f"temp_std_{window_min}"] = rolling.std().reset_index(level=0, drop=True)
        temp_max = rolling.max().reset_index(level=0, drop=True)
        temp_min = rolling.min().reset_index(level=0, drop=True)
        grid[f"temp_range_{window_min}"] = temp_max - temp_min
        grid[f"share_rising_{window_min}"] = rise_flag.groupby(day_key, sort=False).rolling(steps, min_periods=1).mean().reset_index(level=0, drop=True)
        grid[f"share_falling_{window_min}"] = fall_flag.groupby(day_key, sort=False).rolling(steps, min_periods=1).mean().reset_index(level=0, drop=True)

    if logger is not None:
        logger.info(
            "BUILD_PANEL_PROGRESS station=%s stage=path_features_done elapsed_sec=%.1f",
            station_id,
            time.perf_counter() - started,
        )

    precip_actual = pd.to_numeric(grid["precip_hrly"], errors="coerce")
    precip_positive = (precip_actual.fillna(0.0) > 0).astype(float)
    precip_reported = precip_actual.notna().astype(float)
    gust_actual = pd.to_numeric(grid["gust"], errors="coerce")
    gust_present = gust_actual.notna().astype(float)
    for window_min in [60, 180, 360]:
        steps = (window_min // 30) + 1
        grid[f"any_precip_{window_min}"] = precip_positive.groupby(day_key, sort=False).rolling(steps, min_periods=1).max().reset_index(level=0, drop=True)
    steps_180 = (180 // 30) + 1
    grid["precip_sum_180"] = precip_actual.fillna(0.0).groupby(day_key, sort=False).rolling(steps_180, min_periods=1).sum().reset_index(level=0, drop=True)
    grid["precip_report_count_180"] = precip_reported.groupby(day_key, sort=False).rolling(steps_180, min_periods=1).sum().reset_index(level=0, drop=True)
    steps_120 = (120 // 30) + 1
    grid["any_gust_120"] = gust_present.groupby(day_key, sort=False).rolling(steps_120, min_periods=1).max().reset_index(level=0, drop=True)
    grid["max_gust_120"] = gust_actual.groupby(day_key, sort=False).rolling(steps_120, min_periods=1).max().reset_index(level=0, drop=True)
    grid["gust_report_count_240"] = gust_present.groupby(day_key, sort=False).rolling((240 // 30) + 1, min_periods=1).sum().reset_index(level=0, drop=True)

    grid["vis_capped10"] = grid["vis_current"].clip(upper=10)
    grid["vis_lt_1"] = (grid["vis_current"] < 1).fillna(False).astype(float)
    grid["vis_lt_3"] = (grid["vis_current"] < 3).fillna(False).astype(float)
    grid["clds_ord"] = grid["clds_current"].map(CLDS_ORD)
    grid["clds_clear_flag"] = grid["clds_current"].isin(["CLR", "FEW"]).astype(float)
    grid["clds_overcast_flag"] = (grid["clds_current"] == "OVC").astype(float)

    wx_flags = pd.DataFrame([_parse_wx_flags(v) for v in grid["wx_phrase_current"]], index=grid.index)
    grid = pd.concat([grid, wx_flags], axis=1)

    grid["wdir_sin"] = np.sin(np.deg2rad(grid["wdir_current"].astype(float)))
    grid["wdir_cos"] = np.cos(np.deg2rad(grid["wdir_current"].astype(float)))
    grid["marine_sector_flag"] = grid["wdir_cardinal_current"].isin(MARINE_CARDINALS).astype(float)
    grid["continental_sector_flag"] = grid["wdir_cardinal_current"].isin(CONTINENTAL_CARDINALS).astype(float)

    keep = [
        "station_id",
        "target_date_local",
        "cutoff_minutes_local",
        "has_actual_row",
        "age_any_minutes",
        "has_recent_30m",
        "has_recent_60m",
        "has_recent_90m",
        "temp_current",
        "dew_pt_current",
        "rh_current",
        "pressure_current",
        "vis_current",
        "vis_capped10",
        "wspd_current",
        "wdir_current",
        "gust",
        "gust_current",
        "precip_hrly",
        "precip_hrly_current",
        "clds_current",
        "wx_phrase_current",
        "wdir_cardinal_current",
        "age_temp_minutes",
        "temp_high_sofar",
        "temp_low_sofar",
        "temp_gap_from_high",
        "temp_gap_from_low",
        "time_since_tmax_minutes",
        "obs_count_sofar",
        "temp_delta_60",
        "temp_delta_120",
        "temp_delta_180",
        "temp_delta_240",
        "temp_slope_60",
        "temp_slope_120",
        "dewpoint_depression",
        "dew_pt_delta_60",
        "dew_pt_delta_120",
        "pressure_delta_60",
        "pressure_delta_180",
        "vis_delta_60",
        "wspd_delta_60",
        "temp_mean_120",
        "temp_mean_240",
        "temp_std_120",
        "temp_std_240",
        "temp_range_120",
        "temp_range_240",
        "share_rising_120",
        "share_rising_240",
        "share_falling_120",
        "share_falling_240",
        "any_precip_60",
        "any_precip_180",
        "any_precip_360",
        "precip_sum_180",
        "precip_report_count_180",
        "any_gust_120",
        "max_gust_120",
        "gust_report_count_240",
        "vis_lt_1",
        "vis_lt_3",
        "clds_ord",
        "clds_clear_flag",
        "clds_overcast_flag",
        "wx_precip_flag",
        "wx_convective_flag",
        "wx_fog_flag",
        "wx_haze_flag",
        "wx_frozen_flag",
        "wx_windy_flag",
        "wdir_sin",
        "wdir_cos",
        "marine_sector_flag",
        "continental_sector_flag",
    ]
    out = grid[keep].reset_index(drop=True)
    if logger is not None:
        logger.info(
            "BUILD_PANEL_DONE station=%s rows=%s elapsed_sec=%.1f",
            station_id,
            len(out),
            time.perf_counter() - started,
        )
    return out


def build_knyc_labels(knyc_obs_clean: pd.DataFrame, knyc_panel: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_by_day = {d: g.sort_values("cutoff_minutes_local").copy() for d, g in knyc_obs_clean.groupby("target_date_local", sort=False)}
    high_by_row = knyc_panel[["target_date_local", "cutoff_minutes_local", "temp_high_sofar"]].copy()
    truth_map = truth_df.set_index("target_date_local")["settled_tmax"].to_dict()
    label_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []

    for day, y_val in sorted(truth_map.items()):
        obs_day = obs_by_day.get(day)
        settled = float(y_val) if y_val is not None and not pd.isna(y_val) else np.nan
        if obs_day is None:
            case = "no_obs"
            discordant = 1
            max_clean = np.nan
            t_neg = np.nan
            t_pos = np.nan
            width = np.nan
            base_weight = 0.0
        else:
            valid = obs_day.loc[obs_day["temp"].notna(), ["cutoff_minutes_local", "temp"]].copy()
            if valid.empty or not np.isfinite(settled):
                case = "missing_temp_or_truth"
                discordant = 1
                max_clean = np.nan
                t_neg = np.nan
                t_pos = np.nan
                width = np.nan
                base_weight = 0.0
            else:
                max_clean = float(valid["temp"].max())
                if max_clean < settled - 1 or max_clean > settled + 1:
                    case = "discordant"
                    discordant = 1
                    t_neg = np.nan
                    t_pos = np.nan
                    width = np.nan
                    base_weight = 0.0
                elif max_clean >= settled:
                    case = "exact" if max_clean == settled else "overshoot_1f"
                    discordant = 0
                    t_pos = int(valid.loc[valid["temp"] >= settled, "cutoff_minutes_local"].min())
                    before = valid.loc[(valid["cutoff_minutes_local"] < t_pos) & (valid["temp"] < settled), "cutoff_minutes_local"]
                    t_neg = int(before.max()) if not before.empty else 0
                    width = max(0, t_pos - t_neg)
                    base_weight = 1.0 if case == "exact" else 0.6
                else:
                    case = "undershoot_1f"
                    discordant = 0
                    near = valid.loc[valid["temp"] == settled - 1, "cutoff_minutes_local"]
                    t_first = int(near.min())
                    t_last = int(near.max())
                    prev = valid.loc[(valid["cutoff_minutes_local"] < t_first) & (valid["temp"] <= settled - 2), "cutoff_minutes_local"]
                    t_neg = int(prev.max()) if not prev.empty else 0
                    t_pos = t_last
                    width = max(0, t_pos - t_neg)
                    base_weight = 0.7

        penalty = 1.0
        if np.isfinite(width):
            if width <= 30:
                penalty = 1.0
            elif width <= 60:
                penalty = 0.9
            elif width <= 90:
                penalty = 0.75
            elif width <= 120:
                penalty = 0.6
            else:
                penalty = 0.4

        day_rows.append(
            {
                "target_date_local": day,
                "settled_tmax": settled,
                "max_clean_temp": max_clean,
                "label_case": case,
                "discordant_day": discordant,
                "t_neg_minute": t_neg,
                "t_pos_minute": t_pos,
                "ambiguity_width_minutes": width,
                "label_weight_base": base_weight * penalty,
            }
        )

        for cutoff in ALL_CUTOFFS:
            if discordant:
                y = np.nan
                ambiguous = 0
                weight = 0.0
            elif cutoff <= t_neg:
                y = 0.0
                ambiguous = 0
                weight = base_weight * penalty
            elif cutoff >= t_pos:
                y = 1.0
                ambiguous = 0
                weight = base_weight * penalty
            else:
                y = np.nan
                ambiguous = 1
                weight = 0.0
            label_rows.append(
                {
                    "target_date_local": day,
                    "cutoff_minutes_local": cutoff,
                    "settled_tmax": settled,
                    "label_binary": y,
                    "label_ambiguous": ambiguous,
                    "discordant_day": discordant,
                    "label_case": case,
                    "label_weight": weight,
                }
            )

    labels = pd.DataFrame(label_rows)
    labels = labels.merge(high_by_row, on=["target_date_local", "cutoff_minutes_local"], how="left")
    labels["remaining_rise_target"] = np.maximum(0.0, labels["settled_tmax"] - labels["temp_high_sofar"])
    return labels, pd.DataFrame(day_rows)


def _feature_group(feature_name: str) -> str:
    if feature_name.startswith("knyc_"):
        return "knyc_core"
    if feature_name.startswith("neighbor_"):
        return "neighbor_summary"
    if feature_name.startswith("coastal_") or feature_name.startswith("inland_") or feature_name.startswith("network_"):
        return "network_structure"
    if feature_name.startswith("prior_"):
        return "historical_prior"
    if feature_name.startswith("regime_"):
        return "regime"
    if feature_name in {"year", "month", "doy", "cutoff_minutes_local", "cutoff_sin", "cutoff_cos", "doy_sin", "doy_cos"}:
        return "calendar"
    return "misc"


def select_model_feature_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    return [c for c in df.columns if c not in excluded and is_numeric_dtype(df[c]) and df[c].notna().any()]


def assemble_feature_table(
    station_panels: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = station_panels[MAIN_STATION].copy()
    labels = labels.copy()
    rename_knyc = {
        c: f"knyc_{c}" for c in base.columns if c not in {"station_id", "target_date_local", "cutoff_minutes_local"}
    }
    feat = base.drop(columns=["station_id"]).rename(columns=rename_knyc)

    for neighbor in NEIGHBORS:
        panel = station_panels[neighbor].copy()
        keep = [
            "target_date_local",
            "cutoff_minutes_local",
            "temp_current",
            "temp_high_sofar",
            "temp_gap_from_high",
            "temp_delta_60",
            "temp_delta_120",
            "dewpoint_depression",
            "pressure_delta_180",
            "wspd_current",
            "clds_overcast_flag",
            "any_precip_180",
            "wx_convective_flag",
            "wx_fog_flag",
            "marine_sector_flag",
            "age_temp_minutes",
            "has_recent_90m",
        ]
        rename = {c: f"{neighbor.lower()}_{c}" for c in keep if c not in {"target_date_local", "cutoff_minutes_local"}}
        feat = feat.merge(panel[keep].rename(columns=rename), on=["target_date_local", "cutoff_minutes_local"], how="left")

    feat["target_date_local"] = pd.to_datetime(feat["target_date_local"])
    labels["target_date_local"] = pd.to_datetime(labels["target_date_local"])
    feat["cutoff_sin"] = np.sin(2 * np.pi * feat["cutoff_minutes_local"] / 1440.0)
    feat["cutoff_cos"] = np.cos(2 * np.pi * feat["cutoff_minutes_local"] / 1440.0)
    feat["year"] = feat["target_date_local"].dt.year
    feat["month"] = feat["target_date_local"].dt.month
    feat["doy"] = feat["target_date_local"].dt.dayofyear
    feat["doy_sin"] = np.sin(2 * np.pi * feat["doy"] / 366.0)
    feat["doy_cos"] = np.cos(2 * np.pi * feat["doy"] / 366.0)

    all_temp_cols = ["knyc_temp_current"] + [f"{s.lower()}_temp_current" for s in NEIGHBORS]
    all_high_cols = ["knyc_temp_high_sofar"] + [f"{s.lower()}_temp_high_sofar" for s in NEIGHBORS]
    all_delta_cols = ["knyc_temp_delta_60"] + [f"{s.lower()}_temp_delta_60" for s in NEIGHBORS]
    feat["network_temp_mean"] = feat[all_temp_cols].mean(axis=1)
    feat["network_temp_std"] = feat[all_temp_cols].std(axis=1)
    feat["network_temp_min"] = feat[all_temp_cols].min(axis=1)
    feat["network_temp_max"] = feat[all_temp_cols].max(axis=1)
    feat["network_high_mean"] = feat[all_high_cols].mean(axis=1)
    feat["network_delta60_mean"] = feat[all_delta_cols].mean(axis=1)
    feat["network_delta60_std"] = feat[all_delta_cols].std(axis=1)
    feat["network_available_count"] = feat[all_temp_cols].notna().sum(axis=1)
    feat["network_cooling_count"] = (feat[all_delta_cols] < 0).sum(axis=1)
    feat["network_warming_count"] = (feat[all_delta_cols] > 0).sum(axis=1)

    coastal_temp_cols = [f"{s.lower()}_temp_current" for s in COASTAL]
    inland_temp_cols = [f"{s.lower()}_temp_current" for s in INLAND]
    coastal_delta_cols = [f"{s.lower()}_temp_delta_60" for s in COASTAL]
    inland_delta_cols = [f"{s.lower()}_temp_delta_60" for s in INLAND]
    coastal_dewdep_cols = [f"{s.lower()}_dewpoint_depression" for s in COASTAL]
    inland_dewdep_cols = [f"{s.lower()}_dewpoint_depression" for s in INLAND]
    coastal_pressure_cols = [f"{s.lower()}_pressure_delta_180" for s in COASTAL]
    inland_pressure_cols = [f"{s.lower()}_pressure_delta_180" for s in INLAND]
    coastal_overcast_cols = [f"{s.lower()}_clds_overcast_flag" for s in COASTAL]
    inland_overcast_cols = [f"{s.lower()}_clds_overcast_flag" for s in INLAND]
    coastal_marine_cols = [f"{s.lower()}_marine_sector_flag" for s in COASTAL]
    inland_marine_cols = [f"{s.lower()}_marine_sector_flag" for s in INLAND]

    feat["coastal_temp_mean"] = feat[coastal_temp_cols].mean(axis=1)
    feat["inland_temp_mean"] = feat[inland_temp_cols].mean(axis=1)
    feat["coastal_delta60_mean"] = feat[coastal_delta_cols].mean(axis=1)
    feat["inland_delta60_mean"] = feat[inland_delta_cols].mean(axis=1)
    feat["coastal_dewdep_mean"] = feat[coastal_dewdep_cols].mean(axis=1)
    feat["inland_dewdep_mean"] = feat[inland_dewdep_cols].mean(axis=1)
    feat["coastal_pressure180_mean"] = feat[coastal_pressure_cols].mean(axis=1)
    feat["inland_pressure180_mean"] = feat[inland_pressure_cols].mean(axis=1)
    feat["coastal_overcast_share"] = feat[coastal_overcast_cols].mean(axis=1)
    feat["inland_overcast_share"] = feat[inland_overcast_cols].mean(axis=1)
    feat["coastal_marine_share"] = feat[coastal_marine_cols].mean(axis=1)
    feat["inland_marine_share"] = feat[inland_marine_cols].mean(axis=1)
    feat["coastal_minus_inland_temp"] = feat["coastal_temp_mean"] - feat["inland_temp_mean"]
    feat["coastal_minus_inland_delta60"] = feat["coastal_delta60_mean"] - feat["inland_delta60_mean"]
    feat["coastal_minus_inland_dewdep"] = feat["coastal_dewdep_mean"] - feat["inland_dewdep_mean"]
    feat["coastal_minus_inland_pressure180"] = feat["coastal_pressure180_mean"] - feat["inland_pressure180_mean"]
    feat["knyc_minus_coastal_temp"] = feat["knyc_temp_current"] - feat["coastal_temp_mean"]
    feat["knyc_minus_inland_temp"] = feat["knyc_temp_current"] - feat["inland_temp_mean"]
    feat["knyc_minus_coastal_delta60"] = feat["knyc_temp_delta_60"] - feat["coastal_delta60_mean"]
    feat["knyc_minus_inland_delta60"] = feat["knyc_temp_delta_60"] - feat["inland_delta60_mean"]
    feat["knyc_minus_coastal_dewdep"] = feat["knyc_dewpoint_depression"] - feat["coastal_dewdep_mean"]
    feat["knyc_minus_inland_dewdep"] = feat["knyc_dewpoint_depression"] - feat["inland_dewdep_mean"]
    feat["knyc_minus_klga_temp"] = feat["knyc_temp_current"] - feat["klga_temp_current"]
    feat["knyc_minus_klga_delta60"] = feat["knyc_temp_delta_60"] - feat["klga_temp_delta_60"]
    feat["neighbor_temp_gap_to_high_mean"] = feat[[f"{s.lower()}_temp_gap_from_high" for s in NEIGHBORS]].mean(axis=1)
    feat["neighbor_recent_obs_count"] = feat[[f"{s.lower()}_has_recent_90m" for s in NEIGHBORS]].sum(axis=1)
    feat["neighbor_age_temp_mean"] = feat[[f"{s.lower()}_age_temp_minutes" for s in NEIGHBORS]].mean(axis=1)

    feat = feat.merge(labels, on=["target_date_local", "cutoff_minutes_local"], how="left")
    feat["remaining_rise_target"] = pd.to_numeric(feat["remaining_rise_target"], errors="coerce")
    feat = feat.sort_values(["target_date_local", "cutoff_minutes_local"]).reset_index(drop=True)
    feat["target_date_local_str"] = feat["target_date_local"].dt.strftime("%Y-%m-%d")

    for prior_name, group_cols, value_col in [
        ("prior_p_yes_cutoff", ["cutoff_minutes_local"], "label_binary"),
        ("prior_p_yes_month_cutoff", ["month", "cutoff_minutes_local"], "label_binary"),
        ("prior_remaining_mean_cutoff", ["cutoff_minutes_local"], "remaining_rise_target"),
        ("prior_remaining_mean_month_cutoff", ["month", "cutoff_minutes_local"], "remaining_rise_target"),
    ]:
        feat[prior_name] = np.nan
        for _, idx in feat.groupby(group_cols, sort=False).groups.items():
            sub = feat.loc[idx, value_col]
            valid = sub.notna().astype(float)
            csum = sub.fillna(0.0).cumsum().shift(1)
            ccnt = valid.cumsum().shift(1)
            feat.loc[idx, prior_name] = csum / ccnt

    global_p = float(feat["label_binary"].dropna().mean())
    global_rem = float(feat["remaining_rise_target"].dropna().mean())
    feat["prior_p_yes_cutoff"] = feat["prior_p_yes_cutoff"].fillna(global_p)
    feat["prior_p_yes_month_cutoff"] = feat["prior_p_yes_month_cutoff"].fillna(feat["prior_p_yes_cutoff"])
    feat["prior_remaining_mean_cutoff"] = feat["prior_remaining_mean_cutoff"].fillna(global_rem)
    feat["prior_remaining_mean_month_cutoff"] = feat["prior_remaining_mean_month_cutoff"].fillna(feat["prior_remaining_mean_cutoff"])

    manifest_rows = []
    exclude = {
        "target_date_local",
        "target_date_local_str",
        "cutoff_minutes_local",
        "settled_tmax",
        "label_binary",
        "label_ambiguous",
        "discordant_day",
        "label_case",
        "label_weight",
        "temp_high_sofar",
        "remaining_rise_target",
    }
    for col in feat.columns:
        if col in exclude:
            continue
        manifest_rows.append(
            {
                "feature_name": col,
                "feature_group": _feature_group(col),
                "dtype": str(feat[col].dtype),
                "non_null_rows": int(feat[col].notna().sum()),
            }
        )
    return feat, pd.DataFrame(manifest_rows)


def add_regime_features(train_df: pd.DataFrame, full_df: pd.DataFrame, base_feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    regime_cols = [
        "knyc_temp_current",
        "knyc_temp_gap_from_high",
        "knyc_temp_delta_60",
        "knyc_dewpoint_depression",
        "knyc_pressure_delta_180",
        "coastal_minus_inland_temp",
        "coastal_minus_inland_delta60",
        "network_temp_std",
        "coastal_overcast_share",
        "neighbor_recent_obs_count",
    ]
    regime_cols = [c for c in regime_cols if c in base_feature_cols]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[regime_cols]))
    kmeans = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=4096, n_init=10)
    kmeans.fit(X_train)
    X_all = scaler.transform(imputer.transform(full_df[regime_cols]))
    distances = kmeans.transform(X_all)
    full_out = full_df.copy()
    for i in range(distances.shape[1]):
        full_out[f"regime_dist_{i}"] = distances[:, i]
    full_out["regime_id"] = distances.argmin(axis=1).astype(float)
    meta = {"regime_feature_inputs": regime_cols, "n_clusters": 8}
    artifact = {"imputer": imputer, "scaler": scaler, "kmeans": kmeans, "regime_feature_inputs": regime_cols}
    return full_out, meta, artifact


def assign_split(df: pd.DataFrame) -> pd.Series:
    date_str = df["target_date_local"].dt.strftime("%Y-%m-%d")
    split = pd.Series("other", index=df.index, dtype=object)
    split.loc[date_str <= TRAIN_END] = "train"
    split.loc[(date_str >= VAL_START) & (date_str <= VAL_END)] = "val"
    split.loc[(date_str >= TEST_START) & (date_str <= TEST_END)] = "test"
    split.loc[date_str > TEST_END] = "future_holdout"
    return split


def _class_balanced_weights(y: pd.Series, base_weights: pd.Series) -> np.ndarray:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    pos_mult = (neg / pos) if pos > 0 else 1.0
    out = base_weights.to_numpy(dtype=float).copy()
    out[y.to_numpy(dtype=float) == 1.0] *= pos_mult
    return out


def fit_and_predict_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    artifacts_dir: Path,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    started = time.perf_counter()
    use = df.loc[df["label_binary"].isin([0.0, 1.0]) & (df["split"].isin(["train", "val", "test"]))].copy()
    train = use.loc[use["split"] == "train"].copy()
    val = use.loc[use["split"] == "val"].copy()
    test = use.loc[use["split"] == "test"].copy()
    if logger is not None:
        logger.info(
            "TRAIN_SPLITS train_rows=%s val_rows=%s test_rows=%s feature_count=%s",
            len(train),
            len(val),
            len(test),
            len(feature_cols),
        )

    X_train = train[feature_cols]
    X_val = val[feature_cols]
    X_test = test[feature_cols]
    y_train = train["label_binary"].astype(int)
    y_val = val["label_binary"].astype(int)
    y_test = test["label_binary"].astype(int)
    w_train = _class_balanced_weights(y_train, train["label_weight"].fillna(1.0))

    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=400, n_jobs=1, random_state=42)),
        ]
    )
    if logger is not None:
        logger.info("TRAIN_MODEL_START model=baseline_logistic")
    logistic.fit(X_train, y_train, model__sample_weight=w_train)
    p_val_log = logistic.predict_proba(X_val)[:, 1]
    p_test_log = logistic.predict_proba(X_test)[:, 1]
    joblib.dump(logistic, artifacts_dir / "baseline_logistic.joblib")
    if logger is not None:
        logger.info("TRAIN_MODEL_DONE model=baseline_logistic elapsed_sec=%.1f", time.perf_counter() - started)

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=120,
        random_state=42,
        verbosity=-1,
    )
    if logger is not None:
        logger.info("TRAIN_MODEL_START model=lgbm_peak_classifier")
    lgbm.fit(X_train, y_train, sample_weight=w_train)
    p_val_lgbm = lgbm.predict_proba(X_val)[:, 1]
    p_test_lgbm = lgbm.predict_proba(X_test)[:, 1]
    lgbm.booster_.save_model(str(artifacts_dir / "lgbm_peak_classifier.txt"))
    if logger is not None:
        logger.info("TRAIN_MODEL_DONE model=lgbm_peak_classifier elapsed_sec=%.1f", time.perf_counter() - started)

    reg_train = train.loc[train["remaining_rise_target"].notna()].copy()
    Xr_train = reg_train[feature_cols]
    yr_train = reg_train["remaining_rise_target"].astype(float)
    wr_train = reg_train["label_weight"].fillna(1.0).to_numpy(dtype=float)

    q50 = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=120,
        random_state=42,
        verbosity=-1,
    )
    q90 = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.9,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=120,
        random_state=42,
        verbosity=-1,
    )
    if logger is not None:
        logger.info("TRAIN_MODEL_START model=lgbm_remaining_q50_q90")
    q50.fit(Xr_train, yr_train, sample_weight=wr_train)
    q90.fit(Xr_train, yr_train, sample_weight=wr_train)
    q50.booster_.save_model(str(artifacts_dir / "lgbm_remaining_q50.txt"))
    q90.booster_.save_model(str(artifacts_dir / "lgbm_remaining_q90.txt"))
    if logger is not None:
        logger.info("TRAIN_MODEL_DONE model=lgbm_remaining_q50_q90 elapsed_sec=%.1f", time.perf_counter() - started)

    val_q50 = np.clip(q50.predict(X_val), 0.0, None)
    val_q90 = np.clip(q90.predict(X_val), 0.0, None)
    test_q50 = np.clip(q50.predict(X_test), 0.0, None)
    test_q90 = np.clip(q90.predict(X_test), 0.0, None)

    meta_X_val = pd.DataFrame(
        {
            "p_lgbm": p_val_lgbm,
            "q50": val_q50,
            "q90": val_q90,
            "q_span": val_q90 - val_q50,
            "prior_yes": val["prior_p_yes_month_cutoff"].to_numpy(dtype=float),
            "cutoff_sin": val["cutoff_sin"].to_numpy(dtype=float),
            "cutoff_cos": val["cutoff_cos"].to_numpy(dtype=float),
        }
    )
    meta = LogisticRegression(max_iter=400, random_state=42)
    if logger is not None:
        logger.info("TRAIN_MODEL_START model=lgbm_stack_peak_remaining_meta")
    meta.fit(meta_X_val, y_val)
    joblib.dump(meta, artifacts_dir / "meta_logistic.joblib")

    meta_X_test = pd.DataFrame(
        {
            "p_lgbm": p_test_lgbm,
            "q50": test_q50,
            "q90": test_q90,
            "q_span": test_q90 - test_q50,
            "prior_yes": test["prior_p_yes_month_cutoff"].to_numpy(dtype=float),
            "cutoff_sin": test["cutoff_sin"].to_numpy(dtype=float),
            "cutoff_cos": test["cutoff_cos"].to_numpy(dtype=float),
        }
    )
    p_val_stack = meta.predict_proba(meta_X_val)[:, 1]
    p_test_stack = meta.predict_proba(meta_X_test)[:, 1]
    if logger is not None:
        logger.info("TRAIN_MODEL_DONE model=lgbm_stack_peak_remaining_meta elapsed_sec=%.1f", time.perf_counter() - started)

    pred_rows: list[pd.DataFrame] = []
    for model_name, eval_split, sub_df, probs, q50_vals, q90_vals in [
        ("baseline_logistic", "val", val, p_val_log, None, None),
        ("baseline_logistic", "test", test, p_test_log, None, None),
        ("lgbm_peak_classifier", "val", val, p_val_lgbm, val_q50, val_q90),
        ("lgbm_peak_classifier", "test", test, p_test_lgbm, test_q50, test_q90),
        ("lgbm_stack_peak_remaining", "val_meta_fit", val, p_val_stack, val_q50, val_q90),
        ("lgbm_stack_peak_remaining", "test", test, p_test_stack, test_q50, test_q90),
    ]:
        out = sub_df[
            [
                "target_date_local",
                "cutoff_minutes_local",
                "settled_tmax",
                "label_binary",
                "label_case",
                "label_weight",
                "remaining_rise_target",
                "knyc_temp_high_sofar",
                "knyc_temp_current",
                "split",
            ]
        ].copy()
        out["model_name"] = model_name
        out["eval_split"] = eval_split
        out["p_pred"] = probs
        out["y_pred"] = (out["p_pred"] >= 0.5).astype(int)
        out["remaining_q50_pred"] = q50_vals if q50_vals is not None else np.nan
        out["remaining_q90_pred"] = q90_vals if q90_vals is not None else np.nan
        pred_rows.append(out)

    model_rows = [
        {
            "model_name": "baseline_logistic",
            "model_family": "logistic_regression",
            "artifact_path": str(artifacts_dir / "baseline_logistic.joblib"),
            "params_json": json.dumps({"max_iter": 400}),
        },
        {
            "model_name": "lgbm_peak_classifier",
            "model_family": "lightgbm_classifier",
            "artifact_path": str(artifacts_dir / "lgbm_peak_classifier.txt"),
            "params_json": json.dumps(lgbm.get_params(), sort_keys=True),
        },
        {
            "model_name": "lgbm_stack_peak_remaining",
            "model_family": "lgbm_classifier_plus_quantile_stack",
            "artifact_path": str(artifacts_dir / "meta_logistic.joblib"),
            "params_json": json.dumps({"meta_features": list(meta_X_val.columns)}, sort_keys=True),
        },
    ]

    importances: list[dict[str, Any]] = []
    gains = lgbm.booster_.feature_importance(importance_type="gain")
    splits = lgbm.booster_.feature_importance(importance_type="split")
    for feat_name, gain, split_count in zip(feature_cols, gains, splits):
        importances.append(
            {
                "model_name": "lgbm_peak_classifier",
                "feature_name": feat_name,
                "importance_gain": float(gain),
                "importance_split": int(split_count),
            }
        )

    predictions = pd.concat(pred_rows, ignore_index=True)
    if logger is not None:
        logger.info("TRAIN_ALL_MODELS_DONE predictions_rows=%s total_elapsed_sec=%.1f", len(predictions), time.perf_counter() - started)
    return train, val, test, model_rows, importances, predictions


def _safe_auc(y_true: np.ndarray, p_pred: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, p_pred))


def _ece(y_true: np.ndarray, p_pred: np.ndarray, bins: int = 10) -> float:
    df = pd.DataFrame({"y": y_true, "p": p_pred})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True, labels=False)
    out = 0.0
    n = len(df)
    if n == 0:
        return np.nan
    for _, g in df.groupby("bin", observed=False):
        if g.empty:
            continue
        out += (len(g) / n) * abs(g["y"].mean() - g["p"].mean())
    return float(out)


def compute_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    pred = predictions.copy()
    pred["target_date_local"] = pd.to_datetime(pred["target_date_local"])
    pred["month"] = pred["target_date_local"].dt.month

    for (model_name, eval_split), g in pred.groupby(["model_name", "eval_split"], sort=False):
        y = g["label_binary"].astype(int).to_numpy()
        p = g["p_pred"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average="binary", zero_division=0)
        overall_rows.extend(
            [
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "row_count", "metric_value": float(len(g))},
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "positive_rate", "metric_value": float(np.mean(y))},
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "logloss", "metric_value": float(log_loss(y, p, labels=[0, 1]))},
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "brier", "metric_value": float(brier_score_loss(y, p))},
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "accuracy", "metric_value": float(accuracy_score(y, yp))},
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "balanced_accuracy", "metric_value": float(balanced_accuracy_score(y, yp))},
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "precision", "metric_value": float(pr)},
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "recall", "metric_value": float(rc)},
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "f1", "metric_value": float(f1)},
                {"model_name": model_name, "eval_split": eval_split, "metric_name": "ece_10bin", "metric_value": float(_ece(y, p, bins=10))},
            ]
        )
        auc = _safe_auc(y, p)
        if auc is not None:
            overall_rows.append({"model_name": model_name, "eval_split": eval_split, "metric_name": "roc_auc", "metric_value": auc})

        for cutoff, cg in g.groupby("cutoff_minutes_local", sort=False):
            y_c = cg["label_binary"].astype(int).to_numpy()
            p_c = cg["p_pred"].to_numpy(dtype=float)
            yp_c = cg["y_pred"].to_numpy(dtype=int)
            pr_c, rc_c, f1_c, _ = precision_recall_fscore_support(y_c, yp_c, average="binary", zero_division=0)
            cutoff_rows.extend(
                [
                    {"model_name": model_name, "eval_split": eval_split, "cutoff_minutes_local": int(cutoff), "metric_name": "row_count", "metric_value": float(len(cg))},
                    {"model_name": model_name, "eval_split": eval_split, "cutoff_minutes_local": int(cutoff), "metric_name": "positive_rate", "metric_value": float(np.mean(y_c))},
                    {"model_name": model_name, "eval_split": eval_split, "cutoff_minutes_local": int(cutoff), "metric_name": "logloss", "metric_value": float(log_loss(y_c, p_c, labels=[0, 1]))},
                    {"model_name": model_name, "eval_split": eval_split, "cutoff_minutes_local": int(cutoff), "metric_name": "brier", "metric_value": float(brier_score_loss(y_c, p_c))},
                    {"model_name": model_name, "eval_split": eval_split, "cutoff_minutes_local": int(cutoff), "metric_name": "precision", "metric_value": float(pr_c)},
                    {"model_name": model_name, "eval_split": eval_split, "cutoff_minutes_local": int(cutoff), "metric_name": "recall", "metric_value": float(rc_c)},
                    {"model_name": model_name, "eval_split": eval_split, "cutoff_minutes_local": int(cutoff), "metric_name": "f1", "metric_value": float(f1_c)},
                ]
            )
            auc_c = _safe_auc(y_c, p_c)
            if auc_c is not None:
                cutoff_rows.append({"model_name": model_name, "eval_split": eval_split, "cutoff_minutes_local": int(cutoff), "metric_name": "roc_auc", "metric_value": auc_c})

        for month, mg in g.groupby("month", sort=False):
            y_m = mg["label_binary"].astype(int).to_numpy()
            p_m = mg["p_pred"].to_numpy(dtype=float)
            month_rows.extend(
                [
                    {"model_name": model_name, "eval_split": eval_split, "month": int(month), "metric_name": "row_count", "metric_value": float(len(mg))},
                    {"model_name": model_name, "eval_split": eval_split, "month": int(month), "metric_name": "positive_rate", "metric_value": float(np.mean(y_m))},
                    {"model_name": model_name, "eval_split": eval_split, "month": int(month), "metric_name": "logloss", "metric_value": float(log_loss(y_m, p_m, labels=[0, 1]))},
                    {"model_name": model_name, "eval_split": eval_split, "month": int(month), "metric_name": "brier", "metric_value": float(brier_score_loss(y_m, p_m))},
                ]
            )

        for thr in [0.8, 0.9, 0.95]:
            sub = g.loc[g["p_pred"] >= thr].copy()
            cov = float(len(sub) / len(g)) if len(g) else np.nan
            precision_yes = float(sub["label_binary"].mean()) if not sub.empty else np.nan
            false_yes = sub.loc[sub["label_binary"] == 0]
            avg_remaining_false_yes = float(false_yes["remaining_rise_target"].mean()) if not false_yes.empty else np.nan
            utility_rows.append(
                {
                    "model_name": model_name,
                    "eval_split": eval_split,
                    "threshold": thr,
                    "coverage": cov,
                    "precision_yes": precision_yes,
                    "false_yes_count": int(len(false_yes)),
                    "avg_remaining_rise_false_yes": avg_remaining_false_yes,
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(cutoff_rows), pd.DataFrame(month_rows), pd.DataFrame(utility_rows)


def compute_source_pairwise_qa(station_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(STATIONS):
        for b in STATIONS[i + 1 :]:
            pa = station_panels[a][
                [
                    "target_date_local",
                    "cutoff_minutes_local",
                    "temp_current",
                    "dew_pt_current",
                    "rh_current",
                    "pressure_current",
                    "vis_current",
                    "wspd_current",
                    "wdir_current",
                    "clds_current",
                    "wx_phrase_current",
                ]
            ].copy()
            pb = station_panels[b][
                [
                    "target_date_local",
                    "cutoff_minutes_local",
                    "temp_current",
                    "dew_pt_current",
                    "rh_current",
                    "pressure_current",
                    "vis_current",
                    "wspd_current",
                    "wdir_current",
                    "clds_current",
                    "wx_phrase_current",
                ]
            ].copy()
            pa = pa.rename(columns={c: f"a_{c}" for c in pa.columns if c not in {"target_date_local", "cutoff_minutes_local"}})
            pb = pb.rename(columns={c: f"b_{c}" for c in pb.columns if c not in {"target_date_local", "cutoff_minutes_local"}})
            m = pa.merge(pb, on=["target_date_local", "cutoff_minutes_local"], how="inner")
            temp_valid = m["a_temp_current"].notna() & m["b_temp_current"].notna()
            temp_equal_ratio = float((m.loc[temp_valid, "a_temp_current"] == m.loc[temp_valid, "b_temp_current"]).mean()) if temp_valid.any() else np.nan
            core_equal = []
            for field in ["temp_current", "dew_pt_current", "rh_current", "pressure_current", "vis_current", "wspd_current", "wdir_current", "clds_current", "wx_phrase_current"]:
                fa = f"a_{field}"
                fb = f"b_{field}"
                both = m[fa].notna() & m[fb].notna()
                if both.any():
                    core_equal.append((m.loc[both, fa] == m.loc[both, fb]).mean())
            rows.append(
                {
                    "station_a": a,
                    "station_b": b,
                    "overlap_rows": int(len(m)),
                    "temp_equal_ratio": temp_equal_ratio,
                    "core_equal_ratio_mean": float(np.mean(core_equal)) if core_equal else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_results_db(
    db_path: Path,
    *,
    run_manifest: dict[str, Any],
    station_summary: pd.DataFrame,
    cleaning_rules: pd.DataFrame,
    source_pairwise_qa: pd.DataFrame,
    day_labels: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    model_registry: pd.DataFrame,
    metrics_overall: pd.DataFrame,
    metrics_cutoff: pd.DataFrame,
    metrics_month: pd.DataFrame,
    utility_table: pd.DataFrame,
    feature_importance: pd.DataFrame,
    predictions: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        table_frames = [
            ("run_manifest", pd.DataFrame([run_manifest])),
            ("source_station_summary", station_summary),
            ("cleaning_field_summary", cleaning_rules),
            ("source_pairwise_qa", source_pairwise_qa),
            ("label_day_summary", day_labels),
            ("feature_manifest", feature_manifest),
            ("dataset_summary", dataset_summary),
            ("model_registry", model_registry),
            ("model_metrics_overall", metrics_overall),
            ("model_metrics_by_cutoff", metrics_cutoff),
            ("model_metrics_by_month", metrics_month),
            ("model_high_confidence_utility", utility_table),
            ("model_feature_importance", feature_importance),
            ("model_predictions", predictions),
        ]
        for table_name, frame in table_frames:
            if logger is not None:
                logger.info("WRITE_TABLE table=%s rows=%s", table_name, len(frame))
            frame.to_sql(table_name, conn, index=False)
        conn.execute("CREATE INDEX idx_predictions_model_split ON model_predictions(model_name, eval_split)")
        conn.execute("CREATE INDEX idx_predictions_day_cutoff ON model_predictions(target_date_local, cutoff_minutes_local)")
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    results_root = Path(args.results_root).resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("knyc_peak_v1_%Y%m%dT%H%M%SZ")
    run_dir = results_root / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logger = init_logger(args.log_level, log_path)
    cfg = RunConfig(data_root=data_root, results_root=results_root, run_id=run_id)
    run_started = time.perf_counter()

    logger.info("RUN_START run_id=%s data_root=%s log_path=%s", run_id, data_root, log_path)

    station_summaries = []
    cleaning_frames = []
    station_panels: dict[str, pd.DataFrame] = {}
    station_truth: dict[str, pd.DataFrame] = {}
    station_obs_clean: dict[str, pd.DataFrame] = {}

    for station in STATIONS:
        station_started = time.perf_counter()
        obs_raw, truth_raw, summary = load_station_source(data_root, station)
        obs_clean, cleaning = clean_station_obs(obs_raw, station)
        station_summaries.append(summary)
        cleaning_frames.append(cleaning)
        station_truth[station] = truth_raw.copy()
        station_obs_clean[station] = obs_clean
        logger.info(
            "LOAD_AND_CLEAN station=%s obs_rows=%s obs_clean_rows=%s truth_rows=%s invalidated_total=%s elapsed_sec=%.1f",
            station,
            len(obs_raw),
            len(obs_clean),
            len(truth_raw),
            int(cleaning["invalidated_rows"].sum()),
            time.perf_counter() - station_started,
        )

    knyc_truth = station_truth[MAIN_STATION].copy()
    knyc_truth["target_date_local"] = pd.to_datetime(knyc_truth["target_date_local"]).dt.strftime("%Y-%m-%d")
    target_dates = sorted(knyc_truth["target_date_local"].tolist())

    logger.info("BUILD_STATION_PANELS days=%s", len(target_dates))
    for station in STATIONS:
        station_panels[station] = build_station_panel(station_obs_clean[station], target_dates, station, logger=logger)

    logger.info("BUILD_LABELS_START")
    labels, day_labels = build_knyc_labels(station_obs_clean[MAIN_STATION], station_panels[MAIN_STATION], knyc_truth)
    logger.info(
        "BUILD_LABELS_DONE rows=%s day_rows=%s discordant_days=%s ambiguous_rows=%s",
        len(labels),
        len(day_labels),
        int(day_labels["discordant_day"].sum()),
        int(labels["label_ambiguous"].sum()),
    )

    logger.info("ASSEMBLE_FEATURES_START")
    features, feature_manifest = assemble_feature_table(station_panels, labels)
    features["split"] = assign_split(features)
    logger.info("ASSEMBLE_FEATURES_DONE rows=%s columns=%s", len(features), len(features.columns))

    excluded_feature_cols = {
        "target_date_local",
        "target_date_local_str",
        "cutoff_minutes_local",
        "settled_tmax",
        "label_binary",
        "label_ambiguous",
        "discordant_day",
        "label_case",
        "label_weight",
        "temp_high_sofar",
        "remaining_rise_target",
        "split",
    }

    base_feature_cols = select_model_feature_columns(features, excluded_feature_cols)

    train_rows_for_regime = features.loc[(features["split"] == "train") & features["label_binary"].isin([0.0, 1.0])].copy()
    logger.info("REGIME_FEATURES_START train_rows=%s base_feature_count=%s", len(train_rows_for_regime), len(base_feature_cols))
    features, regime_meta, regime_artifact = add_regime_features(train_rows_for_regime, features, base_feature_cols)
    joblib.dump(regime_artifact, artifacts_dir / "regime_model.joblib")
    logger.info("REGIME_FEATURES_DONE n_clusters=%s", regime_meta["n_clusters"])

    feature_cols = select_model_feature_columns(features, excluded_feature_cols)

    feature_manifest = feature_manifest.set_index("feature_name")
    for col in feature_cols:
        if col not in feature_manifest.index:
            feature_manifest.loc[col, :] = {
                "feature_group": _feature_group(col),
                "dtype": str(features[col].dtype),
                "non_null_rows": int(features[col].notna().sum()),
            }
    feature_manifest["used_in_model"] = feature_manifest.index.isin(feature_cols).astype(int)
    feature_manifest = feature_manifest.reset_index().rename(columns={"index": "feature_name"})

    logger.info("TRAIN_MODELS feature_count=%s", len(feature_cols))
    train_df, val_df, test_df, model_rows, importances, predictions = fit_and_predict_models(
        features,
        feature_cols,
        artifacts_dir,
        logger=logger,
    )

    logger.info("COMPUTE_METRICS_START")
    metrics_overall, metrics_cutoff, metrics_month, utility_table = compute_metrics(predictions)
    logger.info(
        "COMPUTE_METRICS_DONE overall_rows=%s cutoff_rows=%s month_rows=%s utility_rows=%s",
        len(metrics_overall),
        len(metrics_cutoff),
        len(metrics_month),
        len(utility_table),
    )

    dataset_summary_rows = []
    for split_name, sub in features.groupby("split", sort=False):
        used = sub.loc[sub["label_binary"].isin([0.0, 1.0])]
        dataset_summary_rows.append(
            {
                "split_name": split_name,
                "rows_total": int(len(sub)),
                "rows_labeled": int(len(used)),
                "days_total": int(sub["target_date_local"].nunique()),
                "days_labeled": int(used["target_date_local"].nunique()),
                "positive_rows": int((used["label_binary"] == 1).sum()),
                "negative_rows": int((used["label_binary"] == 0).sum()),
                "ambiguous_rows": int((sub["label_ambiguous"] == 1).sum()),
                "discordant_rows": int((sub["discordant_day"] == 1).sum()),
            }
        )

    source_pairwise_qa = compute_source_pairwise_qa(station_panels)
    station_summary_df = pd.DataFrame(station_summaries)
    cleaning_rules_df = pd.concat(cleaning_frames, ignore_index=True)
    model_registry_df = pd.DataFrame(model_rows)
    feature_importance_df = pd.DataFrame(importances)
    dataset_summary_df = pd.DataFrame(dataset_summary_rows)

    results_db_path = results_root / f"{run_id}.sqlite"
    run_manifest = {
        "run_id": run_id,
        "generated_at_utc": utc_now_str(),
        "data_root": str(data_root),
        "results_root": str(results_root),
        "results_db_path": str(results_db_path),
        "main_station": MAIN_STATION,
        "neighbor_stations": ",".join(NEIGHBORS),
        "train_end": cfg.train_end,
        "val_start": cfg.val_start,
        "val_end": cfg.val_end,
        "test_start": cfg.test_start,
        "test_end": cfg.test_end,
        "feature_count": int(len(feature_cols)),
        "regime_meta_json": json.dumps(regime_meta, sort_keys=True),
        "notes": "Fresh KNYC early-peak v1 pipeline built from EarlyPeak SQLite only.",
    }

    logger.info("WRITE_RESULTS_DB_START path=%s", results_db_path)
    write_results_db(
        results_db_path,
        run_manifest=run_manifest,
        station_summary=station_summary_df,
        cleaning_rules=cleaning_rules_df,
        source_pairwise_qa=source_pairwise_qa,
        day_labels=day_labels,
        feature_manifest=feature_manifest,
        dataset_summary=dataset_summary_df,
        model_registry=model_registry_df,
        metrics_overall=metrics_overall,
        metrics_cutoff=metrics_cutoff,
        metrics_month=metrics_month,
        utility_table=utility_table,
        feature_importance=feature_importance_df,
        predictions=predictions,
        logger=logger,
    )

    summary = {
        "run_id": run_id,
        "results_db_path": str(results_db_path),
        "artifact_dir": str(artifacts_dir),
        "log_path": str(log_path),
        "feature_count": len(feature_cols),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "models": model_registry_df.to_dict(orient="records"),
    }
    (results_root / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("RUN_DONE run_id=%s total_elapsed_sec=%.1f", run_id, time.perf_counter() - run_started)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
