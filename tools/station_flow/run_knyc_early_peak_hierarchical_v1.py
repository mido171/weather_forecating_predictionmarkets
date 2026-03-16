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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Downcasting object dtype arrays on \\.fillna, \\.ffill, \\.bfill is deprecated.*",
)


ALL_STATIONS = ["KNYC", "KLGA", "KJFK", "KEWR", "KTEB", "KHPN", "KISP", "KBDR"]
MAIN_STATION = "KNYC"
PRIMARY_NEIGHBORS = ["KLGA", "KJFK", "KEWR", "KTEB", "KHPN", "KISP", "KBDR"]
COASTAL = ["KJFK", "KISP", "KBDR"]
INLAND = ["KEWR", "KTEB", "KHPN"]
ALL_CUTOFFS = list(range(0, 1440, 30))
MAX_STALENESS_MIN = 90
FFILL_LIMIT_STEPS = MAX_STALENESS_MIN // 30
PRETEST_END = "2023-12-31"
CALIB_START = "2022-01-01"
CALIB_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"
OOF_START = "2005-01-01"
EASTERN = ZoneInfo("America/New_York")

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
PAIRWISE_FIELDS = [
    "temp",
    "dew_pt",
    "rh",
    "pressure",
    "vis",
    "wspd",
    "wdir",
    "gust",
    "precip_hrly",
    "clds",
    "wx_phrase",
    "wdir_cardinal",
]
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

RISE_BIN_LABELS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10-11",
    "12-13",
    "14-15",
    "16-18",
    "19-21",
    "22-25",
    "26+",
]
RISE_BIN_REP = {
    "0": 0.0,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "6": 6.0,
    "7": 7.0,
    "8": 8.0,
    "9": 9.0,
    "10-11": 10.5,
    "12-13": 12.5,
    "14-15": 14.5,
    "16-18": 17.0,
    "19-21": 20.0,
    "22-25": 23.5,
    "26+": 26.0,
}
BRIDGE_BIN_LABELS = ["<=-4", "-3", "-2", "-1", "0", "+1", "+2", "+3", ">=+4"]
BRIDGE_BIN_REP = {"<=-4": -4.0, "-3": -3.0, "-2": -2.0, "-1": -1.0, "0": 0.0, "+1": 1.0, "+2": 2.0, "+3": 3.0, ">=+4": 4.0}
KLGA_DUP_THRESHOLD = 0.999
OFFICIAL_THRESHOLDS = [0.80, 0.90, 0.95, 0.98]
OOF_FOLDS = [
    ("fold_1", "2004-12-31", "2005-01-01", "2010-12-31"),
    ("fold_2", "2010-12-31", "2011-01-01", "2016-12-31"),
    ("fold_3", "2016-12-31", "2017-01-01", "2021-12-31"),
    ("fold_4", "2021-12-31", "2022-01-01", "2023-12-31"),
]


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    results_root: Path
    run_id: str


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_end: str
    valid_start: str
    valid_end: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fresh hierarchical KNYC early-peak pipeline from EarlyPeak SQLite only.")
    p.add_argument("--data-root", default=r"D:\Ahmed\data\sqlite\EarlyPeak")
    p.add_argument("--results-root", default=r"D:\Ahmed\data\sqlite\EarlyPeak\results")
    p.add_argument("--run-id", default="")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_logger(level_name: str, log_path: Path) -> logging.Logger:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logger = logging.getLogger("knyc_early_peak_hierarchical_v1")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def log_event(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    suffix = " ".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
    if suffix:
        logger.info("%s %s", event, suffix)
    else:
        logger.info("%s", event)


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
        "db_path": str(dbs[0]),
        "obs_rows": int(len(obs)),
        "obs_days": int(obs["target_date_local"].nunique()),
        "obs_min_date": str(obs["target_date_local"].min()),
        "obs_max_date": str(obs["target_date_local"].max()),
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


def _mode_text(series: pd.Series) -> str:
    txt = series.map(_sanitize_text)
    txt = txt[txt.ne("")]
    if txt.empty:
        return ""
    mode = txt.mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(txt.iloc[-1])


def _parse_wx_flags(text_value: Any) -> dict[str, float]:
    txt = _sanitize_text(text_value).lower()
    has = lambda parts: any(p in txt for p in parts)
    return {
        "wx_precip_flag": float(has(("rain", "drizzle", "showers", "snow", "sleet", "hail", "freezing", "wintry", "mix"))),
        "wx_convective_flag": float(has(("thunder", "t-storm", "tstorm", "storm"))),
        "wx_fog_flag": float(has(("fog", "mist"))),
        "wx_haze_flag": float(has(("haze", "smoke", "dust"))),
        "wx_frozen_flag": float(has(("snow", "sleet", "freezing", "hail", "ice", "wintry"))),
        "wx_windy_flag": float(has(("windy", "breezy"))),
    }


def _null_invalid(df: pd.DataFrame, mask: pd.Series, field_name: str, rule_name: str, rows: list[dict[str, Any]], station_id: str) -> None:
    count = int(mask.fillna(False).sum())
    if count > 0:
        df.loc[mask, field_name] = np.nan
    rows.append({"station_id": station_id, "field_name": field_name, "rule_name": rule_name, "invalidated_rows": count})


def _apply_spike_filter(df: pd.DataFrame, station_id: str, field_name: str, threshold: float, rows: list[dict[str, Any]]) -> None:
    prev = df.groupby("target_date_local", sort=False)[field_name].shift(1)
    nxt = df.groupby("target_date_local", sort=False)[field_name].shift(-1)
    mask = (
        df[field_name].notna()
        & prev.notna()
        & nxt.notna()
        & ((df[field_name] - prev).abs() > threshold)
        & ((df[field_name] - nxt).abs() > threshold)
        & (((df[field_name] - prev) * (df[field_name] - nxt)) > 0)
    )
    _null_invalid(df, mask, field_name, f"spike_gt_{threshold}", rows, station_id)


def downcast_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "float64":
            out[col] = pd.to_numeric(out[col], downcast="float")
        elif out[col].dtype == "int64":
            out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


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
    rule_rows: list[dict[str, Any]] = []

    _null_invalid(df, (df["temp"] < -40) | (df["temp"] > 130), "temp", "hard_range", rule_rows, station_id)
    _null_invalid(df, (df["dew_pt"] < -50) | (df["dew_pt"] > 100), "dew_pt", "hard_range", rule_rows, station_id)
    _null_invalid(df, df["dew_pt"] > (df["temp"] + 3), "dew_pt", "dew_gt_temp_plus_3", rule_rows, station_id)
    _null_invalid(df, (df["rh"] < 1) | (df["rh"] > 100), "rh", "hard_range", rule_rows, station_id)
    _null_invalid(df, (df["pressure"] < 25) | (df["pressure"] > 35), "pressure", "hard_range", rule_rows, station_id)
    _null_invalid(df, (df["vis"] < 0) | (df["vis"] > 60), "vis", "hard_range", rule_rows, station_id)
    _null_invalid(df, (df["wspd"] < 0) | (df["wspd"] > 80), "wspd", "hard_range", rule_rows, station_id)
    _null_invalid(df, (df["wdir"] < 0) | (df["wdir"] > 360), "wdir", "hard_range", rule_rows, station_id)
    _null_invalid(df, (df["gust"] < 0) | (df["gust"] > 120), "gust", "hard_range", rule_rows, station_id)
    _null_invalid(df, (df["precip_hrly"] < 0) | (df["precip_hrly"] > 10), "precip_hrly", "hard_range", rule_rows, station_id)
    _null_invalid(df, (df["uv_index"] < 0) | (df["uv_index"] > 15), "uv_index", "drop_contaminated", rule_rows, station_id)

    bad_clds = ~df["clds"].isin(ALLOWED_CLDS) & df["clds"].ne("")
    _null_invalid(df, bad_clds, "clds", "invalid_category", rule_rows, station_id)
    df["clds"] = df["clds"].fillna("")

    bad_card = ~df["wdir_cardinal"].isin(ALLOWED_WDIR_CARD) & df["wdir_cardinal"].ne("")
    _null_invalid(df, bad_card, "wdir_cardinal", "invalid_category", rule_rows, station_id)
    df["wdir_cardinal"] = df["wdir_cardinal"].fillna("")

    df = df.sort_values(["target_date_local", "cutoff_minutes_local", "valid_time_local"]).reset_index(drop=True)
    _apply_spike_filter(df, station_id, "temp", 15.0, rule_rows)
    _apply_spike_filter(df, station_id, "dew_pt", 12.0, rule_rows)
    _apply_spike_filter(df, station_id, "pressure", 0.25, rule_rows)
    _apply_spike_filter(df, station_id, "vis", 20.0, rule_rows)
    _apply_spike_filter(df, station_id, "wspd", 35.0, rule_rows)

    numeric_cols = ["temp", "dew_pt", "rh", "pressure", "vis", "wspd", "wdir", "gust", "precip_hrly", "uv_index"]
    agg: dict[str, Any] = {"station_id": "last", "request_location_id": "last", "valid_time_utc": "last", "valid_time_local": "last"}
    for col in numeric_cols:
        agg[col] = "median"
    for col in ["clds", "wx_phrase", "uv_desc", "wdir_cardinal"]:
        agg[col] = _mode_text
    df = (
        df.groupby(["target_date_local", "cutoff_minutes_local"], as_index=False)
        .agg(agg)
        .sort_values(["target_date_local", "cutoff_minutes_local"])
        .reset_index(drop=True)
    )
    return downcast_frame(df), pd.DataFrame(rule_rows)


def compute_klga_dedupe_audit(knyc_obs: pd.DataFrame, klga_obs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    a = knyc_obs[["target_date_local", "cutoff_minutes_local"] + PAIRWISE_FIELDS].rename(columns={c: f"knyc_{c}" for c in PAIRWISE_FIELDS})
    b = klga_obs[["target_date_local", "cutoff_minutes_local"] + PAIRWISE_FIELDS].rename(columns={c: f"klga_{c}" for c in PAIRWISE_FIELDS})
    merged = a.merge(b, on=["target_date_local", "cutoff_minutes_local"], how="inner")
    rows = []
    field_equal = {}
    for field in PAIRWISE_FIELDS:
        left = merged[f"knyc_{field}"]
        right = merged[f"klga_{field}"]
        both = left.notna() & right.notna()
        equal_ratio = float((left[both] == right[both]).mean()) if both.any() else np.nan
        field_equal[field] = equal_ratio
        rows.append({"record_type": "field", "field_name": field, "overlap_rows": int(len(merged)), "equal_ratio": equal_ratio})

    exact_row_equal = np.ones(len(merged), dtype=bool)
    valid_any = np.zeros(len(merged), dtype=bool)
    for field in PAIRWISE_FIELDS:
        left = merged[f"knyc_{field}"]
        right = merged[f"klga_{field}"]
        same = (left == right) | (left.isna() & right.isna())
        exact_row_equal &= same.to_numpy()
        valid_any |= (left.notna() | right.notna()).to_numpy()
    exact_row_equal_ratio = float(exact_row_equal[valid_any].mean()) if valid_any.any() else np.nan

    knyc_daily = knyc_obs.groupby("target_date_local", sort=False)["temp"].max().rename("knyc_final_peak")
    klga_daily = klga_obs.groupby("target_date_local", sort=False)["temp"].max().rename("klga_final_peak")
    daily = knyc_daily.to_frame().merge(klga_daily.to_frame(), left_index=True, right_index=True, how="inner")
    daily_equal_ratio = float((daily["knyc_final_peak"] == daily["klga_final_peak"]).mean()) if not daily.empty else np.nan

    summary = {
        "record_type": "summary",
        "overlap_rows": int(len(merged)),
        "exact_row_equal_ratio": exact_row_equal_ratio,
        "temp_equal_ratio": field_equal.get("temp"),
        "daily_final_peak_equal_ratio": daily_equal_ratio,
        "quarantine_klga": int(
            np.isfinite(exact_row_equal_ratio) and exact_row_equal_ratio >= KLGA_DUP_THRESHOLD
            or np.isfinite(field_equal.get("temp", np.nan)) and field_equal["temp"] >= KLGA_DUP_THRESHOLD
        ),
    }
    rows.append(summary)
    return pd.DataFrame(rows), summary


def _is_dst_local_date(day_str: str) -> int:
    dt = datetime.fromisoformat(f"{day_str}T12:00:00").replace(tzinfo=EASTERN)
    return int(bool(dt.dst()))


def _dst_transition_dates(dates: list[str]) -> set[str]:
    out: set[str] = set()
    prev_flag: int | None = None
    for d in sorted(dates):
        flag = _is_dst_local_date(d)
        if prev_flag is not None and flag != prev_flag:
            out.add(d)
        prev_flag = flag
    return out


def _shifted_climate_date(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    day_str = ts.strftime("%Y-%m-%d")
    if _is_dst_local_date(day_str) == 1 and ts.hour < 1:
        return (ts.date() - timedelta(days=1)).isoformat()
    return ts.date().isoformat()


def compute_dst_alignment_audit(knyc_obs: pd.DataFrame, knyc_truth: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    obs = knyc_obs.copy()
    truth = knyc_truth.copy()
    truth["target_date_local"] = pd.to_datetime(truth["target_date_local"]).dt.strftime("%Y-%m-%d")
    truth = truth[["target_date_local", "settled_tmax"]].dropna()

    civil = obs.groupby("target_date_local", sort=False)["temp"].max().rename("wu_final_peak")
    shifted_key = obs["valid_time_local"].map(_shifted_climate_date)
    shifted = obs.assign(shifted_target_date=shifted_key).groupby("shifted_target_date", sort=False)["temp"].max().rename("wu_final_peak")

    transition_dates = _dst_transition_dates(sorted(truth["target_date_local"].unique().tolist()))
    rows = []
    metrics: dict[str, dict[str, float]] = {}
    for mapping_name, series in [("civil", civil), ("shifted_dst_climate_day", shifted)]:
        merged = truth.merge(series.to_frame(), left_on="target_date_local", right_index=True, how="left")
        merged["abs_diff"] = (merged["settled_tmax"] - merged["wu_final_peak"]).abs()
        merged["transition_flag"] = merged["target_date_local"].isin(transition_dates).astype(int)
        for segment_name, sub in [("all", merged), ("transition_days", merged.loc[merged["transition_flag"] == 1])]:
            if sub.empty:
                mae = np.nan
                within_0 = np.nan
                within_1 = np.nan
            else:
                mae = float(sub["abs_diff"].mean())
                within_0 = float((sub["abs_diff"] == 0).mean())
                within_1 = float((sub["abs_diff"] <= 1).mean())
            rows.append(
                {
                    "mapping_name": mapping_name,
                    "segment_name": segment_name,
                    "row_count": int(len(sub)),
                    "mae_abs_diff": mae,
                    "within_0f": within_0,
                    "within_1f": within_1,
                }
            )
            if segment_name == "all":
                metrics[mapping_name] = {"mae": mae, "within_1": within_1}

    civil_mae = metrics["civil"]["mae"]
    shifted_mae = metrics["shifted_dst_climate_day"]["mae"]
    civil_within1 = metrics["civil"]["within_1"]
    shifted_within1 = metrics["shifted_dst_climate_day"]["within_1"]
    choose_shifted = np.isfinite(shifted_mae) and (
        (shifted_mae + 0.05 < civil_mae) or (abs(shifted_mae - civil_mae) <= 0.05 and shifted_within1 > civil_within1 + 0.005)
    )
    chosen = "shifted_dst_climate_day" if choose_shifted else "civil"
    summary = {
        "chosen_mapping": chosen,
        "civil_mae_abs_diff": civil_mae,
        "shifted_mae_abs_diff": shifted_mae,
        "civil_within_1f": civil_within1,
        "shifted_within_1f": shifted_within1,
        "dst_transition_day_count": len(transition_dates),
    }
    return pd.DataFrame(rows), summary


def build_station_panel(obs_clean: pd.DataFrame, target_dates: list[str], station_id: str, logger: logging.Logger) -> pd.DataFrame:
    started = time.perf_counter()
    log_event(logger, "BUILD_PANEL_START", station=station_id, target_days=len(target_dates), source_rows=len(obs_clean))
    base = pd.MultiIndex.from_product(
        [pd.Index(target_dates, name="target_date_local"), ALL_CUTOFFS],
        names=["target_date_local", "cutoff_minutes_local"],
    ).to_frame(index=False)
    keep = ["target_date_local", "cutoff_minutes_local", "valid_time_local", "valid_time_utc"] + RAW_OBS_COLS
    grid = base.merge(obs_clean[keep], on=["target_date_local", "cutoff_minutes_local"], how="left", sort=False)
    grid = grid.sort_values(["target_date_local", "cutoff_minutes_local"]).reset_index(drop=True)
    grid["station_id"] = station_id
    grid["dst_flag"] = grid["target_date_local"].map(_is_dst_local_date).astype("int8")
    grid["month"] = pd.to_datetime(grid["target_date_local"]).dt.month.astype("int8")
    grid["doy"] = pd.to_datetime(grid["target_date_local"]).dt.dayofyear.astype("int16")
    grid["cutoff_sin"] = np.sin(2.0 * np.pi * grid["cutoff_minutes_local"] / 1440.0)
    grid["cutoff_cos"] = np.cos(2.0 * np.pi * grid["cutoff_minutes_local"] / 1440.0)
    grid["doy_sin"] = np.sin(2.0 * np.pi * grid["doy"] / 366.0)
    grid["doy_cos"] = np.cos(2.0 * np.pi * grid["doy"] / 366.0)

    day_key = grid["target_date_local"]
    grid["has_actual_row"] = grid["valid_time_local"].notna().astype("int8")
    grid["last_actual_cutoff"] = grid["cutoff_minutes_local"].where(grid["has_actual_row"].eq(1))
    grid["last_actual_cutoff"] = grid.groupby(day_key, sort=False)["last_actual_cutoff"].ffill()
    grid["age_any_minutes"] = grid["cutoff_minutes_local"] - grid["last_actual_cutoff"]
    grid.loc[grid["last_actual_cutoff"].isna(), "age_any_minutes"] = np.nan
    grid["has_recent_30m"] = (grid["age_any_minutes"] <= 30).fillna(False).astype("int8")
    grid["has_recent_60m"] = (grid["age_any_minutes"] <= 60).fillna(False).astype("int8")
    grid["has_recent_90m"] = (grid["age_any_minutes"] <= 90).fillna(False).astype("int8")

    dense_numeric = ["temp", "dew_pt", "rh", "pressure", "vis", "wspd", "wdir"]
    dense_text = ["clds", "wx_phrase", "wdir_cardinal"]
    for col in dense_numeric + dense_text:
        grid[f"{col}_current"] = grid.groupby(day_key, sort=False)[col].ffill(limit=FFILL_LIMIT_STEPS)
    for col in dense_numeric:
        last_valid = grid["cutoff_minutes_local"].where(grid[col].notna())
        last_valid = last_valid.groupby(day_key, sort=False).ffill()
        grid[f"age_{col}_minutes"] = grid["cutoff_minutes_local"] - last_valid
        grid.loc[last_valid.isna(), f"age_{col}_minutes"] = np.nan

    temp_group = grid.groupby(day_key, sort=False)["temp"]
    grid["temp_high_sofar"] = temp_group.cummax()
    grid["temp_low_sofar"] = temp_group.cummin()
    grid["current_minus_high"] = grid["temp_current"] - grid["temp_high_sofar"]
    grid["current_minus_low"] = grid["temp_current"] - grid["temp_low_sofar"]
    grid["current_at_high_flag"] = (grid["temp_current"] == grid["temp_high_sofar"]).fillna(False).astype("int8")
    grid["new_high_count_sofar"] = grid.groupby(day_key, sort=False)["current_at_high_flag"].cumsum().astype("float32")
    last_high_cutoff = grid["cutoff_minutes_local"].where(grid["current_at_high_flag"].eq(1))
    last_high_cutoff = last_high_cutoff.groupby(day_key, sort=False).ffill()
    grid["time_since_high_minutes"] = grid["cutoff_minutes_local"] - last_high_cutoff
    grid.loc[last_high_cutoff.isna(), "time_since_high_minutes"] = np.nan
    grid["obs_count_sofar"] = grid["temp"].notna().groupby(day_key, sort=False).cumsum().astype("float32")
    grid["actual_obs_share_sofar"] = grid["obs_count_sofar"] / ((grid["cutoff_minutes_local"] / 30.0) + 1.0)

    for lag_min in [30, 60, 120, 180]:
        steps = lag_min // 30
        lag_temp = grid.groupby(day_key, sort=False)["temp_current"].shift(steps)
        grid[f"temp_delta_{lag_min}"] = grid["temp_current"] - lag_temp
        lag_dew = grid.groupby(day_key, sort=False)["dew_pt_current"].shift(steps)
        grid[f"dew_pt_delta_{lag_min}"] = grid["dew_pt_current"] - lag_dew
        lag_press = grid.groupby(day_key, sort=False)["pressure_current"].shift(steps)
        grid[f"pressure_delta_{lag_min}"] = grid["pressure_current"] - lag_press
        lag_vis = grid.groupby(day_key, sort=False)["vis_current"].shift(steps)
        grid[f"vis_delta_{lag_min}"] = grid["vis_current"] - lag_vis
        lag_wspd = grid.groupby(day_key, sort=False)["wspd_current"].shift(steps)
        grid[f"wspd_delta_{lag_min}"] = grid["wspd_current"] - lag_wspd

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

    precip_actual = pd.to_numeric(grid["precip_hrly"], errors="coerce")
    precip_positive = (precip_actual.fillna(0.0) > 0).astype(float)
    gust_actual = pd.to_numeric(grid["gust"], errors="coerce")
    gust_present = gust_actual.notna().astype(float)
    for window_min in [60, 180, 360]:
        steps = (window_min // 30) + 1
        grid[f"any_precip_{window_min}"] = precip_positive.groupby(day_key, sort=False).rolling(steps, min_periods=1).max().reset_index(level=0, drop=True)
    grid["precip_sum_180"] = precip_actual.fillna(0.0).groupby(day_key, sort=False).rolling((180 // 30) + 1, min_periods=1).sum().reset_index(level=0, drop=True)
    grid["any_gust_120"] = gust_present.groupby(day_key, sort=False).rolling((120 // 30) + 1, min_periods=1).max().reset_index(level=0, drop=True)
    grid["max_gust_120"] = gust_actual.groupby(day_key, sort=False).rolling((120 // 30) + 1, min_periods=1).max().reset_index(level=0, drop=True)

    grid["dewpoint_depression"] = grid["temp_current"] - grid["dew_pt_current"]
    grid["vis_capped10"] = grid["vis_current"].clip(upper=10)
    grid["vis_lt_1"] = (grid["vis_current"] < 1).fillna(False).astype(float)
    grid["vis_lt_3"] = (grid["vis_current"] < 3).fillna(False).astype(float)
    grid["clds_ord"] = grid["clds_current"].map(CLDS_ORD)
    grid["clds_clear_flag"] = grid["clds_current"].isin(["CLR", "FEW", "SCT"]).astype(float)
    grid["clds_overcast_flag"] = grid["clds_current"].isin(["BKN", "OVC"]).astype(float)

    wx_flags = pd.DataFrame([_parse_wx_flags(v) for v in grid["wx_phrase_current"]], index=grid.index)
    grid = pd.concat([grid, wx_flags], axis=1)

    grid["wdir_sin"] = np.sin(np.deg2rad(grid["wdir_current"].astype(float)))
    grid["wdir_cos"] = np.cos(np.deg2rad(grid["wdir_current"].astype(float)))
    grid["marine_sector_flag"] = grid["wdir_cardinal_current"].isin(MARINE_CARDINALS).astype(float)
    grid["continental_sector_flag"] = grid["wdir_cardinal_current"].isin(CONTINENTAL_CARDINALS).astype(float)

    keep_cols = [
        "station_id",
        "target_date_local",
        "cutoff_minutes_local",
        "month",
        "doy",
        "dst_flag",
        "cutoff_sin",
        "cutoff_cos",
        "doy_sin",
        "doy_cos",
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
        "age_temp_minutes",
        "temp_high_sofar",
        "temp_low_sofar",
        "current_minus_high",
        "current_minus_low",
        "current_at_high_flag",
        "time_since_high_minutes",
        "new_high_count_sofar",
        "obs_count_sofar",
        "actual_obs_share_sofar",
        "temp_delta_30",
        "temp_delta_60",
        "temp_delta_120",
        "temp_delta_180",
        "dew_pt_delta_60",
        "dew_pt_delta_120",
        "pressure_delta_60",
        "pressure_delta_120",
        "pressure_delta_180",
        "vis_delta_60",
        "wspd_delta_60",
        "wspd_delta_120",
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
        "any_gust_120",
        "max_gust_120",
        "dewpoint_depression",
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
    out = downcast_frame(grid[keep_cols].reset_index(drop=True))
    log_event(logger, "BUILD_PANEL_DONE", station=station_id, rows=len(out), elapsed_sec=round(time.perf_counter() - started, 1))
    return out


def encode_rise_bin(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    v = float(value)
    if v <= 0:
        return "0"
    if v < 10:
        return str(int(round(v)))
    if v <= 11:
        return "10-11"
    if v <= 13:
        return "12-13"
    if v <= 15:
        return "14-15"
    if v <= 18:
        return "16-18"
    if v <= 21:
        return "19-21"
    if v <= 25:
        return "22-25"
    return "26+"


def encode_bridge_bin(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    v = float(value)
    if v <= -4:
        return "<=-4"
    if v >= 4:
        return ">=+4"
    if v > 0:
        return f"+{int(round(v))}"
    return str(int(round(v)))


def build_station_wu_labels(station_id: str, obs_clean: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_lookup = panel[["target_date_local", "cutoff_minutes_local", "temp_high_sofar"]].copy()
    label_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    obs_groups = {d: g.sort_values("cutoff_minutes_local").copy() for d, g in obs_clean.groupby("target_date_local", sort=False)}

    for day, _day_panel in panel.groupby("target_date_local", sort=False):
        obs_day = obs_groups.get(day)
        if obs_day is None:
            continue
        valid = obs_day.loc[obs_day["temp"].notna(), ["cutoff_minutes_local", "temp"]].copy()
        if valid.empty:
            continue
        valid = valid.drop_duplicates("cutoff_minutes_local", keep="last").sort_values("cutoff_minutes_local")
        final_peak = float(valid["temp"].max())
        tau_star = int(valid.loc[valid["temp"] == final_peak, "cutoff_minutes_local"].min())
        obs_count = int(len(valid))
        gaps = valid["cutoff_minutes_local"].diff().dropna().astype(float)
        max_gap_all = float(gaps.max()) if not gaps.empty else 0.0
        near_peak = valid.loc[valid["cutoff_minutes_local"].between(max(0, tau_star - 120), min(1410, tau_star + 120))]
        peak_gaps = near_peak["cutoff_minutes_local"].diff().dropna().astype(float)
        max_gap_near_peak = float(peak_gaps.max()) if not peak_gaps.empty else 0.0
        strict = int(obs_count >= 18 and max_gap_all <= 180 and max_gap_near_peak <= 90)
        tolerant = int(obs_count >= 16 and max_gap_near_peak <= 180)

        day_rows.append(
            {
                "station_id": station_id,
                "target_date_local": day,
                "wu_final_peak": final_peak,
                "wu_tau_star_cutoff": tau_star,
                "obs_count_valid_temp": obs_count,
                "max_gap_all_minutes": max_gap_all,
                "max_gap_near_peak_minutes": max_gap_near_peak,
                "strict_quality_flag": strict,
                "tolerant_quality_flag": tolerant,
            }
        )

        day_high = high_lookup.loc[high_lookup["target_date_local"] == day]
        day_high_map = dict(zip(day_high["cutoff_minutes_local"], day_high["temp_high_sofar"]))
        for cutoff in ALL_CUTOFFS:
            high_sofar = day_high_map.get(cutoff, np.nan)
            remaining_rise = np.nan if pd.isna(high_sofar) else max(0.0, final_peak - float(high_sofar))
            label_rows.append(
                {
                    "station_id": station_id,
                    "target_date_local": day,
                    "cutoff_minutes_local": cutoff,
                    "wu_peak_by_cutoff": float(cutoff >= tau_star),
                    "wu_remaining_rise": remaining_rise,
                    "wu_remaining_rise_bin": encode_rise_bin(remaining_rise),
                    "wu_final_peak": final_peak,
                    "wu_tau_star_cutoff": tau_star,
                    "strict_quality_flag": strict,
                    "tolerant_quality_flag": tolerant,
                }
            )
    return downcast_frame(pd.DataFrame(label_rows)), downcast_frame(pd.DataFrame(day_rows))


def assign_regime_tag(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    p = prefix
    marine_push = (
        (df[f"{p}coastal_minus_inland_temp"] < -2.0)
        & (df[f"{p}coastal_marine_share"] > 0.34)
        & (df[f"{p}knyc_current_minus_high"] < -0.5)
    )
    precip_fog = (
        (df[f"{p}knyc_any_precip_180"] > 0)
        | (df[f"{p}knyc_wx_fog_flag"] > 0)
        | (df[f"{p}knyc_vis_lt_3"] > 0)
    )
    cloud_suppressed = (
        (df[f"{p}knyc_clds_overcast_flag"] > 0)
        & (df[f"{p}knyc_dewpoint_depression"] < 12)
        & (df[f"{p}knyc_temp_delta_60"] <= 0.5)
    )
    clear_dry = (
        (df[f"{p}knyc_clds_clear_flag"] > 0)
        & (df[f"{p}knyc_dewpoint_depression"] >= 15)
        & (df[f"{p}knyc_temp_delta_60"] > 0)
    )
    postfrontal = (
        (df[f"{p}knyc_pressure_delta_180"] > 0.05)
        & (df[f"{p}knyc_dew_pt_delta_120"] < -2)
        & (df[f"{p}knyc_wspd_current"] >= 8)
    )
    warm_adv = (
        (df[f"{p}knyc_pressure_delta_180"] < -0.05)
        & (df[f"{p}knyc_temp_delta_60"] > 0.5)
        & (df[f"{p}inland_delta60_mean"] > 0.5)
    )
    tag = pd.Series("HUMID_CLEAR", index=df.index, dtype=object)
    tag.loc[clear_dry.fillna(False)] = "CLEAR_DRY_MIXING"
    tag.loc[cloud_suppressed.fillna(False)] = "CLOUD_SUPPRESSED"
    tag.loc[precip_fog.fillna(False)] = "PRECIP_FOG_SUPPRESSED"
    tag.loc[marine_push.fillna(False)] = "MARINE_PUSH"
    tag.loc[postfrontal.fillna(False)] = "POSTFRONTAL_DRYING"
    tag.loc[warm_adv.fillna(False)] = "WARM_ADVECTION_LATE_RISE"
    return tag


def build_stage_a_frame(station_panels: dict[str, pd.DataFrame], station_labels: dict[str, pd.DataFrame], station_dummies: list[str]) -> pd.DataFrame:
    frames = []
    for station_id, panel in station_panels.items():
        labels = station_labels[station_id]
        merged = panel.merge(labels, on=["station_id", "target_date_local", "cutoff_minutes_local"], how="inner")
        merged["date_dt"] = pd.to_datetime(merged["target_date_local"])
        merged["station_code"] = station_id
        frames.append(merged)
    df = pd.concat(frames, ignore_index=True)
    dummies = pd.get_dummies(df["station_code"], prefix="station", dtype=float)
    for col in station_dummies:
        if col not in dummies.columns:
            dummies[col] = 0.0
    df = pd.concat([df, dummies[station_dummies]], axis=1)
    return downcast_frame(df)


def _neighbor_raw_cols() -> list[str]:
    return [
        "temp_current",
        "current_minus_high",
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


def build_stage_b_frame(
    knyc_panel: pd.DataFrame,
    knyc_labels: pd.DataFrame,
    station_panels: dict[str, pd.DataFrame],
    stage_a_preds: pd.DataFrame,
    active_neighbors: list[str],
) -> pd.DataFrame:
    knyc = knyc_panel.merge(knyc_labels, on=["station_id", "target_date_local", "cutoff_minutes_local"], how="inner").copy()
    knyc["date_dt"] = pd.to_datetime(knyc["target_date_local"])
    knyc = knyc.rename(columns={c: f"knyc_{c}" for c in knyc.columns if c not in {"target_date_local", "cutoff_minutes_local", "date_dt"}})

    for neighbor in active_neighbors:
        raw = station_panels[neighbor][["target_date_local", "cutoff_minutes_local"] + _neighbor_raw_cols()].copy()
        raw = raw.rename(columns={c: f"{neighbor.lower()}_{c}" for c in _neighbor_raw_cols()})
        knyc = knyc.merge(raw, on=["target_date_local", "cutoff_minutes_local"], how="left")

    stage_a_keep = stage_a_preds.loc[stage_a_preds["station_id"].isin(active_neighbors)].copy()
    stage_a_keep = stage_a_keep[
        [
            "station_id",
            "target_date_local",
            "cutoff_minutes_local",
            "prediction_source",
            "stage_a_p_peak",
            "stage_a_rise_mean",
            "stage_a_p_rise_gt1",
            "stage_a_p_rise_gt2",
        ]
    ]
    for neighbor in active_neighbors:
        sub = stage_a_keep.loc[stage_a_keep["station_id"] == neighbor].drop(columns=["station_id"])
        sub = sub.rename(
            columns={
                "prediction_source": f"{neighbor.lower()}_stage_a_prediction_source",
                "stage_a_p_peak": f"{neighbor.lower()}_stage_a_p_peak",
                "stage_a_rise_mean": f"{neighbor.lower()}_stage_a_rise_mean",
                "stage_a_p_rise_gt1": f"{neighbor.lower()}_stage_a_p_rise_gt1",
                "stage_a_p_rise_gt2": f"{neighbor.lower()}_stage_a_p_rise_gt2",
            }
        )
        knyc = knyc.merge(sub, on=["target_date_local", "cutoff_minutes_local"], how="left")

    coastal_temp = [f"{s.lower()}_temp_current" for s in COASTAL if s in active_neighbors]
    inland_temp = [f"{s.lower()}_temp_current" for s in INLAND if s in active_neighbors]
    coastal_delta = [f"{s.lower()}_temp_delta_60" for s in COASTAL if s in active_neighbors]
    inland_delta = [f"{s.lower()}_temp_delta_60" for s in INLAND if s in active_neighbors]
    coastal_marine = [f"{s.lower()}_marine_sector_flag" for s in COASTAL if s in active_neighbors]
    inland_marine = [f"{s.lower()}_marine_sector_flag" for s in INLAND if s in active_neighbors]
    coastal_overcast = [f"{s.lower()}_clds_overcast_flag" for s in COASTAL if s in active_neighbors]
    inland_overcast = [f"{s.lower()}_clds_overcast_flag" for s in INLAND if s in active_neighbors]
    coastal_peak = [f"{s.lower()}_stage_a_p_peak" for s in COASTAL if s in active_neighbors]
    inland_peak = [f"{s.lower()}_stage_a_p_peak" for s in INLAND if s in active_neighbors]
    coastal_rise = [f"{s.lower()}_stage_a_rise_mean" for s in COASTAL if s in active_neighbors]
    inland_rise = [f"{s.lower()}_stage_a_rise_mean" for s in INLAND if s in active_neighbors]
    neighbor_p_peak_cols = [f"{s.lower()}_stage_a_p_peak" for s in active_neighbors]
    neighbor_rise_cols = [f"{s.lower()}_stage_a_rise_mean" for s in active_neighbors]
    neighbor_age_cols = [f"{s.lower()}_age_temp_minutes" for s in active_neighbors]
    neighbor_delta_cols = [f"{s.lower()}_temp_delta_60" for s in active_neighbors]

    knyc["coastal_temp_mean"] = knyc[coastal_temp].mean(axis=1) if coastal_temp else np.nan
    knyc["inland_temp_mean"] = knyc[inland_temp].mean(axis=1) if inland_temp else np.nan
    knyc["coastal_delta60_mean"] = knyc[coastal_delta].mean(axis=1) if coastal_delta else np.nan
    knyc["inland_delta60_mean"] = knyc[inland_delta].mean(axis=1) if inland_delta else np.nan
    knyc["coastal_marine_share"] = knyc[coastal_marine].mean(axis=1) if coastal_marine else np.nan
    knyc["inland_marine_share"] = knyc[inland_marine].mean(axis=1) if inland_marine else np.nan
    knyc["coastal_overcast_share"] = knyc[coastal_overcast].mean(axis=1) if coastal_overcast else np.nan
    knyc["inland_overcast_share"] = knyc[inland_overcast].mean(axis=1) if inland_overcast else np.nan
    knyc["coastal_peak_frac"] = knyc[coastal_peak].mean(axis=1) if coastal_peak else np.nan
    knyc["inland_peak_frac"] = knyc[inland_peak].mean(axis=1) if inland_peak else np.nan
    knyc["coastal_rise_mean"] = knyc[coastal_rise].mean(axis=1) if coastal_rise else np.nan
    knyc["inland_rise_mean"] = knyc[inland_rise].mean(axis=1) if inland_rise else np.nan
    knyc["neighbor_peak_frac"] = knyc[neighbor_p_peak_cols].mean(axis=1) if neighbor_p_peak_cols else np.nan
    knyc["neighbor_peak_max"] = knyc[neighbor_p_peak_cols].max(axis=1) if neighbor_p_peak_cols else np.nan
    knyc["neighbor_rise_mean"] = knyc[neighbor_rise_cols].mean(axis=1) if neighbor_rise_cols else np.nan
    knyc["neighbor_rise_max"] = knyc[neighbor_rise_cols].max(axis=1) if neighbor_rise_cols else np.nan
    knyc["neighbor_recent_count"] = (knyc[neighbor_age_cols] <= 90).sum(axis=1) if neighbor_age_cols else 0
    knyc["neighbor_warming_count"] = (knyc[neighbor_delta_cols] > 0).sum(axis=1) if neighbor_delta_cols else 0
    knyc["neighbor_cooling_count"] = (knyc[neighbor_delta_cols] < 0).sum(axis=1) if neighbor_delta_cols else 0
    knyc["coastal_minus_inland_temp"] = knyc["coastal_temp_mean"] - knyc["inland_temp_mean"]
    knyc["coastal_minus_inland_delta60"] = knyc["coastal_delta60_mean"] - knyc["inland_delta60_mean"]
    knyc["knyc_minus_coastal_temp"] = knyc["knyc_temp_current"] - knyc["coastal_temp_mean"]
    knyc["knyc_minus_inland_temp"] = knyc["knyc_temp_current"] - knyc["inland_temp_mean"]
    knyc["network_temp_std"] = knyc[["knyc_temp_current"] + [f"{s.lower()}_temp_current" for s in active_neighbors]].std(axis=1)
    knyc["network_delta60_std"] = knyc[["knyc_temp_delta_60"] + [f"{s.lower()}_temp_delta_60" for s in active_neighbors]].std(axis=1)

    knyc = knyc.rename(
        columns={
            "knyc_station_id": "station_id",
            "knyc_wu_peak_by_cutoff": "wu_peak_by_cutoff",
            "knyc_wu_remaining_rise": "wu_remaining_rise",
            "knyc_wu_remaining_rise_bin": "wu_remaining_rise_bin",
            "knyc_wu_final_peak": "wu_final_peak",
            "knyc_wu_tau_star_cutoff": "wu_tau_star_cutoff",
            "knyc_strict_quality_flag": "strict_quality_flag",
            "knyc_tolerant_quality_flag": "tolerant_quality_flag",
            "knyc_current_minus_high": "knyc_current_minus_high",
        }
    )
    knyc["regime_tag"] = assign_regime_tag(knyc)
    regime_dummies = pd.get_dummies(knyc["regime_tag"], prefix="regime", dtype=float)
    knyc = pd.concat([knyc, regime_dummies], axis=1)
    return downcast_frame(knyc)


def build_stage_c_frame(stage_b_frame: pd.DataFrame, knyc_truth: pd.DataFrame, knyc_day_summary: pd.DataFrame, stage_b_preds: pd.DataFrame) -> pd.DataFrame:
    truth = knyc_truth.copy()
    truth["target_date_local"] = pd.to_datetime(truth["target_date_local"]).dt.strftime("%Y-%m-%d")
    truth = truth[["target_date_local", "settled_tmax"]]
    day = knyc_day_summary[["target_date_local", "wu_final_peak", "strict_quality_flag", "tolerant_quality_flag"]].copy()
    base = stage_b_frame.merge(truth, on="target_date_local", how="left").merge(day, on=["target_date_local", "strict_quality_flag", "tolerant_quality_flag"], how="left", suffixes=("", "_day"))
    preds = stage_b_preds[
        [
            "target_date_local",
            "cutoff_minutes_local",
            "prediction_source",
            "stage_b_p_peak",
            "stage_b_rise_mean",
            "stage_b_rise_q90",
            "stage_b_p_rise_gt1",
            "stage_b_p_rise_gt2",
        ]
    ].copy()
    base = base.merge(preds, on=["target_date_local", "cutoff_minutes_local"], how="left")
    base["bridge_delta"] = base["settled_tmax"] - base["wu_final_peak"]
    base["bridge_bin"] = base["bridge_delta"].map(encode_bridge_bin)
    base["bridge_abs"] = base["bridge_delta"].abs()
    return downcast_frame(base)


def build_stage_d_frame(
    stage_c_frame: pd.DataFrame,
    stage_c_preds: pd.DataFrame,
    neighbor_label_rows: pd.DataFrame,
    dst_transition_dates: set[str],
) -> pd.DataFrame:
    base = stage_c_frame.merge(
        stage_c_preds[
            ["target_date_local", "cutoff_minutes_local", "prediction_source", "stage_c_bridge_mean", "stage_c_bridge_entropy"]
            + [f"stage_c_bridge_p_{label}" for label in BRIDGE_BIN_LABELS]
        ],
        on=["target_date_local", "cutoff_minutes_local"],
        how="left",
    )
    label_pivot = neighbor_label_rows.pivot_table(
        index=["target_date_local", "cutoff_minutes_local"],
        columns="station_id",
        values="wu_peak_by_cutoff",
        aggfunc="first",
    )
    label_pivot.columns = [f"neighbor_label_{c}" for c in label_pivot.columns]
    label_pivot = label_pivot.reset_index()
    base = base.merge(label_pivot, on=["target_date_local", "cutoff_minutes_local"], how="left")
    neighbor_label_cols = [c for c in base.columns if c.startswith("neighbor_label_")]
    if neighbor_label_cols:
        comp = base[neighbor_label_cols].eq(base["wu_peak_by_cutoff"], axis=0)
        base["neighbor_same_state_frac"] = comp.mean(axis=1)
    else:
        base["neighbor_same_state_frac"] = np.nan

    base["quality_weight"] = np.where(base["strict_quality_flag"] == 1, 1.0, np.where(base["tolerant_quality_flag"] == 1, 0.5, 0.0))
    base["bridge_weight"] = np.select(
        [
            base["bridge_abs"] == 0,
            base["bridge_abs"] == 1,
            base["bridge_abs"] == 2,
            base["bridge_abs"] == 3,
        ],
        [1.0, 0.8, 0.5, 0.2],
        default=0.0,
    )
    base["concordance_weight"] = 0.5 + 0.5 * base["neighbor_same_state_frac"].fillna(0.5)
    risky_regime = base["regime_tag"].isin(["MARINE_PUSH", "WARM_ADVECTION_LATE_RISE"]).astype(int)
    base["regime_weight"] = np.where(risky_regime == 1, 0.5, 1.0)
    base["official_weight"] = base["quality_weight"] * base["bridge_weight"] * base["concordance_weight"] * base["regime_weight"]
    base["weak_official_target"] = base["wu_peak_by_cutoff"]
    base["silver_subset_flag"] = (
        (base["strict_quality_flag"] == 1)
        & (base["bridge_abs"] <= 1)
        & (~base["target_date_local"].isin(dst_transition_dates))
        & (~base["regime_tag"].isin(["WARM_ADVECTION_LATE_RISE"]))
    ).astype(int)
    base["dst_ambiguous_flag"] = base["target_date_local"].isin(dst_transition_dates).astype(int)
    base["fresh_station_count"] = 1 + base["neighbor_recent_count"].fillna(0)
    age_term = np.clip(base["knyc_age_temp_minutes"].fillna(180.0), 0.0, 180.0)
    network_term = np.clip(base["fresh_station_count"] / 7.0, 0.0, 1.0)
    base["quality_score"] = np.clip(1.0 - (age_term / 180.0), 0.0, 1.0) * 0.55 + network_term * 0.45
    base["future_wu_rise"] = base["wu_remaining_rise"]
    base["future_official_gap"] = np.maximum(0.0, base["settled_tmax"] - base["knyc_temp_high_sofar"])
    base["hard_midday_flag"] = base["cutoff_minutes_local"].between(690, 930).astype(int)
    base["actionability_flag"] = ((base["quality_score"] >= 0.70) & (base["fresh_station_count"] >= 5) & (base["dst_ambiguous_flag"] == 0)).astype(int)
    return downcast_frame(base)


def stage_a_feature_columns(station_dummies: list[str]) -> list[str]:
    return [
        "cutoff_minutes_local",
        "cutoff_sin",
        "cutoff_cos",
        "month",
        "doy_sin",
        "doy_cos",
        "dst_flag",
        "has_actual_row",
        "age_any_minutes",
        "has_recent_30m",
        "has_recent_60m",
        "has_recent_90m",
        "temp_current",
        "dew_pt_current",
        "rh_current",
        "pressure_current",
        "vis_capped10",
        "wspd_current",
        "age_temp_minutes",
        "temp_high_sofar",
        "temp_low_sofar",
        "current_minus_high",
        "current_minus_low",
        "current_at_high_flag",
        "time_since_high_minutes",
        "new_high_count_sofar",
        "obs_count_sofar",
        "actual_obs_share_sofar",
        "temp_delta_30",
        "temp_delta_60",
        "temp_delta_120",
        "temp_delta_180",
        "dew_pt_delta_60",
        "dew_pt_delta_120",
        "pressure_delta_60",
        "pressure_delta_120",
        "pressure_delta_180",
        "vis_delta_60",
        "wspd_delta_60",
        "wspd_delta_120",
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
        "any_gust_120",
        "max_gust_120",
        "dewpoint_depression",
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
    ] + station_dummies


def stage_b_feature_columns(active_neighbors: list[str], regime_cols: list[str]) -> list[str]:
    cols = [
        "cutoff_minutes_local",
        "knyc_cutoff_sin",
        "knyc_cutoff_cos",
        "knyc_month",
        "knyc_doy_sin",
        "knyc_doy_cos",
        "knyc_dst_flag",
        "knyc_has_actual_row",
        "knyc_age_any_minutes",
        "knyc_has_recent_30m",
        "knyc_has_recent_60m",
        "knyc_has_recent_90m",
        "knyc_temp_current",
        "knyc_dew_pt_current",
        "knyc_rh_current",
        "knyc_pressure_current",
        "knyc_vis_capped10",
        "knyc_wspd_current",
        "knyc_age_temp_minutes",
        "knyc_temp_high_sofar",
        "knyc_temp_low_sofar",
        "knyc_current_minus_high",
        "knyc_current_minus_low",
        "knyc_current_at_high_flag",
        "knyc_time_since_high_minutes",
        "knyc_new_high_count_sofar",
        "knyc_obs_count_sofar",
        "knyc_actual_obs_share_sofar",
        "knyc_temp_delta_30",
        "knyc_temp_delta_60",
        "knyc_temp_delta_120",
        "knyc_temp_delta_180",
        "knyc_dew_pt_delta_60",
        "knyc_dew_pt_delta_120",
        "knyc_pressure_delta_60",
        "knyc_pressure_delta_120",
        "knyc_pressure_delta_180",
        "knyc_vis_delta_60",
        "knyc_wspd_delta_60",
        "knyc_wspd_delta_120",
        "knyc_temp_mean_120",
        "knyc_temp_mean_240",
        "knyc_temp_std_120",
        "knyc_temp_std_240",
        "knyc_temp_range_120",
        "knyc_temp_range_240",
        "knyc_share_rising_120",
        "knyc_share_rising_240",
        "knyc_share_falling_120",
        "knyc_share_falling_240",
        "knyc_any_precip_60",
        "knyc_any_precip_180",
        "knyc_any_precip_360",
        "knyc_precip_sum_180",
        "knyc_any_gust_120",
        "knyc_max_gust_120",
        "knyc_dewpoint_depression",
        "knyc_vis_lt_1",
        "knyc_vis_lt_3",
        "knyc_clds_ord",
        "knyc_clds_clear_flag",
        "knyc_clds_overcast_flag",
        "knyc_wx_precip_flag",
        "knyc_wx_convective_flag",
        "knyc_wx_fog_flag",
        "knyc_wx_haze_flag",
        "knyc_wx_frozen_flag",
        "knyc_wx_windy_flag",
        "knyc_wdir_sin",
        "knyc_wdir_cos",
        "knyc_marine_sector_flag",
        "knyc_continental_sector_flag",
        "coastal_temp_mean",
        "inland_temp_mean",
        "coastal_delta60_mean",
        "inland_delta60_mean",
        "coastal_marine_share",
        "inland_marine_share",
        "coastal_overcast_share",
        "inland_overcast_share",
        "coastal_peak_frac",
        "inland_peak_frac",
        "coastal_rise_mean",
        "inland_rise_mean",
        "neighbor_peak_frac",
        "neighbor_peak_max",
        "neighbor_rise_mean",
        "neighbor_rise_max",
        "neighbor_recent_count",
        "neighbor_warming_count",
        "neighbor_cooling_count",
        "coastal_minus_inland_temp",
        "coastal_minus_inland_delta60",
        "knyc_minus_coastal_temp",
        "knyc_minus_inland_temp",
        "network_temp_std",
        "network_delta60_std",
    ]
    for neighbor in active_neighbors:
        prefix = neighbor.lower()
        cols.extend(
            [
                f"{prefix}_stage_a_p_peak",
                f"{prefix}_stage_a_rise_mean",
                f"{prefix}_stage_a_p_rise_gt1",
                f"{prefix}_stage_a_p_rise_gt2",
                f"{prefix}_temp_current",
                f"{prefix}_current_minus_high",
                f"{prefix}_temp_delta_60",
                f"{prefix}_temp_delta_120",
                f"{prefix}_dewpoint_depression",
                f"{prefix}_pressure_delta_180",
                f"{prefix}_wspd_current",
                f"{prefix}_clds_overcast_flag",
                f"{prefix}_any_precip_180",
                f"{prefix}_wx_convective_flag",
                f"{prefix}_wx_fog_flag",
                f"{prefix}_marine_sector_flag",
                f"{prefix}_age_temp_minutes",
                f"{prefix}_has_recent_90m",
            ]
        )
    return cols + regime_cols


def stage_c_feature_columns(regime_cols: list[str]) -> list[str]:
    cols = [
        "cutoff_minutes_local",
        "knyc_cutoff_sin",
        "knyc_cutoff_cos",
        "knyc_month",
        "knyc_doy_sin",
        "knyc_doy_cos",
        "knyc_dst_flag",
        "knyc_temp_current",
        "knyc_current_minus_high",
        "knyc_time_since_high_minutes",
        "knyc_temp_delta_60",
        "knyc_temp_delta_120",
        "knyc_temp_range_120",
        "knyc_dewpoint_depression",
        "knyc_pressure_delta_180",
        "knyc_wspd_current",
        "knyc_clds_overcast_flag",
        "knyc_any_precip_180",
        "knyc_wx_fog_flag",
        "knyc_wx_haze_flag",
        "knyc_marine_sector_flag",
        "knyc_age_temp_minutes",
        "knyc_has_recent_90m",
        "coastal_minus_inland_temp",
        "coastal_minus_inland_delta60",
        "knyc_minus_coastal_temp",
        "knyc_minus_inland_temp",
        "neighbor_peak_frac",
        "neighbor_rise_mean",
        "neighbor_recent_count",
        "network_temp_std",
        "network_delta60_std",
        "stage_b_p_peak",
        "stage_b_rise_mean",
        "stage_b_rise_q90",
        "stage_b_p_rise_gt1",
        "stage_b_p_rise_gt2",
    ]
    return cols + regime_cols


def stage_d_feature_columns(regime_cols: list[str]) -> list[str]:
    cols = [
        "cutoff_minutes_local",
        "knyc_month",
        "knyc_doy_sin",
        "knyc_doy_cos",
        "stage_b_p_peak",
        "stage_b_rise_mean",
        "stage_b_rise_q90",
        "stage_b_p_rise_gt1",
        "stage_b_p_rise_gt2",
        "stage_c_bridge_mean",
        "stage_c_bridge_entropy",
        "neighbor_same_state_frac",
        "coastal_peak_frac",
        "inland_peak_frac",
        "coastal_rise_mean",
        "inland_rise_mean",
        "neighbor_peak_frac",
        "neighbor_rise_mean",
        "neighbor_recent_count",
        "knyc_current_minus_high",
        "knyc_time_since_high_minutes",
        "knyc_age_temp_minutes",
        "quality_score",
        "fresh_station_count",
        "dst_ambiguous_flag",
        "actionability_flag",
    ]
    cols.extend([f"stage_c_bridge_p_{label}" for label in BRIDGE_BIN_LABELS])
    return cols + regime_cols


def _feature_manifest(feature_cols: list[str], stage_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        rows.append(
            {
                "stage_name": stage_name,
                "feature_name": col,
                "dtype": str(frame[col].dtype) if col in frame.columns else "missing",
                "non_null_rows": int(frame[col].notna().sum()) if col in frame.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def _mask_date_between(date_series: pd.Series, start: str, end: str) -> pd.Series:
    ds = date_series.dt.strftime("%Y-%m-%d")
    return (ds >= start) & (ds <= end)


def _mask_date_le(date_series: pd.Series, end: str) -> pd.Series:
    return date_series.dt.strftime("%Y-%m-%d") <= end


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


def _binary_weights(y: pd.Series, base_weights: pd.Series) -> np.ndarray:
    out = base_weights.fillna(1.0).to_numpy(dtype=float)
    counts = y.value_counts().to_dict()
    total = float(sum(counts.values()))
    for cls, count in counts.items():
        if count > 0:
            out[y.to_numpy() == cls] *= total / (len(counts) * float(count))
    return out


def _multiclass_weights(y: pd.Series, base_weights: pd.Series) -> np.ndarray:
    out = base_weights.fillna(1.0).to_numpy(dtype=float)
    counts = y.value_counts().to_dict()
    total = float(sum(counts.values()))
    for cls, count in counts.items():
        if count > 0:
            out[y.to_numpy() == cls] *= total / (len(counts) * float(count))
    return out


def _pmf_mean(prob: np.ndarray, labels: list[str], rep_map: dict[str, float]) -> np.ndarray:
    reps = np.array([rep_map[label] for label in labels], dtype=float)
    return prob @ reps


def _pmf_quantile(prob: np.ndarray, labels: list[str], rep_map: dict[str, float], q: float) -> np.ndarray:
    reps = np.array([rep_map[label] for label in labels], dtype=float)
    cdf = np.cumsum(prob, axis=1)
    idx = (cdf >= q).argmax(axis=1)
    return reps[idx]


def _multiclass_metrics_rows(
    pred_df: pd.DataFrame,
    *,
    stage_name: str,
    target_col: str,
    pred_mean_col: str,
    prob_prefix: str,
    label_order: list[str],
    split_col: str = "prediction_source",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []
    prob_cols = [f"{prob_prefix}{label}" for label in label_order]
    for split_name, g in pred_df.groupby(split_col, sort=False):
        sub = g.loc[g[target_col].isin(label_order)].copy()
        if sub.empty:
            continue
        y = pd.Categorical(sub[target_col], categories=label_order).codes
        p = sub[prob_cols].to_numpy(dtype=float)
        top1 = p.argmax(axis=1)
        pred_label = np.array(label_order, dtype=object)[top1]
        overall_rows.extend(
            [
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "row_count", "metric_value": float(len(sub))},
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "multiclass_logloss", "metric_value": float(log_loss(y, p, labels=np.arange(len(label_order))))},
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "top1_accuracy", "metric_value": float((pred_label == sub[target_col].to_numpy()).mean())},
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "mean_abs_error_meanpred", "metric_value": float((sub[pred_mean_col] - sub["target_numeric"]).abs().mean())},
            ]
        )
        for cutoff, cg in sub.groupby("cutoff_minutes_local", sort=False):
            y_c = pd.Categorical(cg[target_col], categories=label_order).codes
            p_c = cg[prob_cols].to_numpy(dtype=float)
            pred_c = np.array(label_order, dtype=object)[p_c.argmax(axis=1)]
            cutoff_rows.extend(
                [
                    {"stage_name": stage_name, "split_name": split_name, "cutoff_minutes_local": int(cutoff), "metric_name": "row_count", "metric_value": float(len(cg))},
                    {"stage_name": stage_name, "split_name": split_name, "cutoff_minutes_local": int(cutoff), "metric_name": "multiclass_logloss", "metric_value": float(log_loss(y_c, p_c, labels=np.arange(len(label_order))))},
                    {"stage_name": stage_name, "split_name": split_name, "cutoff_minutes_local": int(cutoff), "metric_name": "top1_accuracy", "metric_value": float((pred_c == cg[target_col].to_numpy()).mean())},
                    {"stage_name": stage_name, "split_name": split_name, "cutoff_minutes_local": int(cutoff), "metric_name": "mean_abs_error_meanpred", "metric_value": float((cg[pred_mean_col] - cg["target_numeric"]).abs().mean())},
                ]
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(cutoff_rows)


def _binary_metrics_rows(
    pred_df: pd.DataFrame,
    *,
    stage_name: str,
    target_col: str,
    prob_col: str,
    split_col: str = "prediction_source",
    station_breakout: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    for split_name, g in pred_df.groupby(split_col, sort=False):
        sub = g.loc[g[target_col].isin([0.0, 1.0])].copy()
        if sub.empty:
            continue
        y = sub[target_col].astype(int).to_numpy()
        p = sub[prob_col].to_numpy(dtype=float)
        yp = (p >= 0.5).astype(int)
        overall_rows.extend(
            [
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "row_count", "metric_value": float(len(sub))},
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "positive_rate", "metric_value": float(np.mean(y))},
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "logloss", "metric_value": float(log_loss(y, p, labels=[0, 1]))},
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "brier", "metric_value": float(brier_score_loss(y, p))},
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "accuracy", "metric_value": float(accuracy_score(y, yp))},
                {"stage_name": stage_name, "split_name": split_name, "metric_name": "ece_10bin", "metric_value": float(_ece(y, p))},
            ]
        )
        auc = _safe_auc(y, p)
        if auc is not None:
            overall_rows.append({"stage_name": stage_name, "split_name": split_name, "metric_name": "roc_auc", "metric_value": auc})
        for cutoff, cg in sub.groupby("cutoff_minutes_local", sort=False):
            y_c = cg[target_col].astype(int).to_numpy()
            p_c = cg[prob_col].to_numpy(dtype=float)
            yp_c = (p_c >= 0.5).astype(int)
            cutoff_rows.extend(
                [
                    {"stage_name": stage_name, "split_name": split_name, "cutoff_minutes_local": int(cutoff), "metric_name": "row_count", "metric_value": float(len(cg))},
                    {"stage_name": stage_name, "split_name": split_name, "cutoff_minutes_local": int(cutoff), "metric_name": "logloss", "metric_value": float(log_loss(y_c, p_c, labels=[0, 1]))},
                    {"stage_name": stage_name, "split_name": split_name, "cutoff_minutes_local": int(cutoff), "metric_name": "brier", "metric_value": float(brier_score_loss(y_c, p_c))},
                    {"stage_name": stage_name, "split_name": split_name, "cutoff_minutes_local": int(cutoff), "metric_name": "accuracy", "metric_value": float(accuracy_score(y_c, yp_c))},
                ]
            )
            auc_c = _safe_auc(y_c, p_c)
            if auc_c is not None:
                cutoff_rows.append({"stage_name": stage_name, "split_name": split_name, "cutoff_minutes_local": int(cutoff), "metric_name": "roc_auc", "metric_value": auc_c})
        if station_breakout and "station_id" in sub.columns:
            for station_id, sg in sub.groupby("station_id", sort=False):
                y_s = sg[target_col].astype(int).to_numpy()
                p_s = sg[prob_col].to_numpy(dtype=float)
                yp_s = (p_s >= 0.5).astype(int)
                station_rows.extend(
                    [
                        {"stage_name": stage_name, "split_name": split_name, "station_id": station_id, "metric_name": "row_count", "metric_value": float(len(sg))},
                        {"stage_name": stage_name, "split_name": split_name, "station_id": station_id, "metric_name": "logloss", "metric_value": float(log_loss(y_s, p_s, labels=[0, 1]))},
                        {"stage_name": stage_name, "split_name": split_name, "station_id": station_id, "metric_name": "brier", "metric_value": float(brier_score_loss(y_s, p_s))},
                        {"stage_name": stage_name, "split_name": split_name, "station_id": station_id, "metric_name": "accuracy", "metric_value": float(accuracy_score(y_s, yp_s))},
                    ]
                )
    return pd.DataFrame(overall_rows), pd.DataFrame(cutoff_rows), pd.DataFrame(station_rows)


def _fit_lgbm_binary(X_train: pd.DataFrame, y_train: pd.Series, w_train: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=150,
        random_state=42,
        verbosity=-1,
        n_jobs=4,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model


def _fit_lgbm_multiclass(X_train: pd.DataFrame, y_train: pd.Series, w_train: np.ndarray, num_class: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_class,
        n_estimators=220,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=180,
        random_state=42,
        verbosity=-1,
        n_jobs=4,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model


def run_stage_a(
    stage_a_df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[FoldSpec],
    artifacts_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_event(logger, "STAGE_A_START", rows=len(stage_a_df), feature_count=len(feature_cols))
    work = stage_a_df.loc[(stage_a_df["date_dt"] >= pd.Timestamp(OOF_START)) & stage_a_df["strict_quality_flag"].eq(1)].copy()
    work["rise_target_numeric"] = work["wu_remaining_rise"].astype(float)
    work["rise_class"] = pd.Categorical(work["wu_remaining_rise_bin"], categories=RISE_BIN_LABELS).codes
    work = work.loc[work["rise_class"] >= 0].copy()

    oof_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for idx, fold in enumerate(folds, start=1):
        train_mask = _mask_date_le(work["date_dt"], fold.train_end)
        valid_mask = _mask_date_between(work["date_dt"], fold.valid_start, fold.valid_end)
        train = work.loc[train_mask].copy()
        valid = work.loc[valid_mask].copy()
        if train.empty or valid.empty:
            continue
        log_event(logger, "STAGE_A_FOLD_START", fold=fold.fold_id, fold_index=idx, total_folds=len(folds), train_rows=len(train), valid_rows=len(valid))
        X_train = train[feature_cols]
        X_valid = valid[feature_cols]
        y_train_peak = train["wu_peak_by_cutoff"].astype(int)
        w_train_peak = _binary_weights(y_train_peak, pd.Series(np.ones(len(train)), index=train.index))

        peak_model = _fit_lgbm_binary(X_train, y_train_peak, w_train_peak)
        p_valid_peak = peak_model.predict_proba(X_valid)[:, 1]

        y_train_rise = train["rise_class"].astype(int)
        w_train_rise = _multiclass_weights(y_train_rise, pd.Series(np.ones(len(train)), index=train.index))
        rise_model = _fit_lgbm_multiclass(X_train, y_train_rise, w_train_rise, len(RISE_BIN_LABELS))
        p_valid_rise = rise_model.predict_proba(X_valid)
        rise_mean = _pmf_mean(p_valid_rise, RISE_BIN_LABELS, RISE_BIN_REP)
        rise_q90 = _pmf_quantile(p_valid_rise, RISE_BIN_LABELS, RISE_BIN_REP, 0.90)
        idx_gt1 = [i for i, label in enumerate(RISE_BIN_LABELS) if RISE_BIN_REP[label] > 1.0]
        idx_gt2 = [i for i, label in enumerate(RISE_BIN_LABELS) if RISE_BIN_REP[label] > 2.0]

        fold_pred = valid[
            [
                "station_id",
                "target_date_local",
                "cutoff_minutes_local",
                "date_dt",
                "wu_peak_by_cutoff",
                "wu_remaining_rise",
                "wu_remaining_rise_bin",
            ]
        ].copy()
        fold_pred["prediction_source"] = "oof_pretest"
        fold_pred["fold_id"] = fold.fold_id
        fold_pred["stage_a_p_peak"] = p_valid_peak
        fold_pred["stage_a_rise_mean"] = rise_mean
        fold_pred["stage_a_rise_q90"] = rise_q90
        fold_pred["stage_a_p_rise_gt1"] = p_valid_rise[:, idx_gt1].sum(axis=1)
        fold_pred["stage_a_p_rise_gt2"] = p_valid_rise[:, idx_gt2].sum(axis=1)
        for i, label in enumerate(RISE_BIN_LABELS):
            fold_pred[f"stage_a_rise_p_{label}"] = p_valid_rise[:, i]
        oof_frames.append(fold_pred)

        for feat, gain, split_count in zip(feature_cols, peak_model.booster_.feature_importance("gain"), peak_model.booster_.feature_importance("split")):
            importance_rows.append(
                {"stage_name": "stage_a_peak", "model_name": f"stage_a_peak_{fold.fold_id}", "feature_name": feat, "importance_gain": float(gain), "importance_split": int(split_count)}
            )
        model_rows.extend(
            [
                {"stage_name": "stage_a_peak", "model_name": f"stage_a_peak_{fold.fold_id}", "artifact_path": "", "split_name": fold.fold_id, "notes": "OOF fold model"},
                {"stage_name": "stage_a_rise", "model_name": f"stage_a_rise_{fold.fold_id}", "artifact_path": "", "split_name": fold.fold_id, "notes": "OOF fold model"},
            ]
        )
        log_event(logger, "STAGE_A_FOLD_DONE", fold=fold.fold_id)

    pretest_preds = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()

    pretest_train = work.loc[_mask_date_le(work["date_dt"], PRETEST_END)].copy()
    test_mask = stage_a_df["date_dt"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)) & stage_a_df["strict_quality_flag"].eq(1)
    test_frame = stage_a_df.loc[test_mask].copy()
    X_train_all = pretest_train[feature_cols]
    X_test = test_frame[feature_cols]
    y_train_peak_all = pretest_train["wu_peak_by_cutoff"].astype(int)
    w_train_peak_all = _binary_weights(y_train_peak_all, pd.Series(np.ones(len(pretest_train)), index=pretest_train.index))
    peak_final = _fit_lgbm_binary(X_train_all, y_train_peak_all, w_train_peak_all)
    p_test_peak = peak_final.predict_proba(X_test)[:, 1]
    peak_final.booster_.save_model(str(artifacts_dir / "stageA_wu_local_peak_final.txt"))

    y_train_rise_all = pd.Categorical(pretest_train["wu_remaining_rise_bin"], categories=RISE_BIN_LABELS).codes
    w_train_rise_all = _multiclass_weights(pd.Series(y_train_rise_all), pd.Series(np.ones(len(pretest_train)), index=pretest_train.index))
    rise_final = _fit_lgbm_multiclass(X_train_all, pd.Series(y_train_rise_all), w_train_rise_all, len(RISE_BIN_LABELS))
    p_test_rise = rise_final.predict_proba(X_test)
    rise_final.booster_.save_model(str(artifacts_dir / "stageA_wu_local_rise_final.txt"))

    idx_gt1 = [i for i, label in enumerate(RISE_BIN_LABELS) if RISE_BIN_REP[label] > 1.0]
    idx_gt2 = [i for i, label in enumerate(RISE_BIN_LABELS) if RISE_BIN_REP[label] > 2.0]
    test_pred = test_frame[
        ["station_id", "target_date_local", "cutoff_minutes_local", "date_dt", "wu_peak_by_cutoff", "wu_remaining_rise", "wu_remaining_rise_bin"]
    ].copy()
    test_pred["prediction_source"] = "test"
    test_pred["fold_id"] = "final_pretest_fit"
    test_pred["stage_a_p_peak"] = p_test_peak
    test_pred["stage_a_rise_mean"] = _pmf_mean(p_test_rise, RISE_BIN_LABELS, RISE_BIN_REP)
    test_pred["stage_a_rise_q90"] = _pmf_quantile(p_test_rise, RISE_BIN_LABELS, RISE_BIN_REP, 0.90)
    test_pred["stage_a_p_rise_gt1"] = p_test_rise[:, idx_gt1].sum(axis=1)
    test_pred["stage_a_p_rise_gt2"] = p_test_rise[:, idx_gt2].sum(axis=1)
    for i, label in enumerate(RISE_BIN_LABELS):
        test_pred[f"stage_a_rise_p_{label}"] = p_test_rise[:, i]

    deploy_mask = stage_a_df["date_dt"] <= pd.Timestamp(TEST_END)
    deploy_frame = stage_a_df.loc[deploy_mask & stage_a_df["strict_quality_flag"].eq(1)].copy()
    X_deploy = deploy_frame[feature_cols]
    y_deploy_peak = deploy_frame["wu_peak_by_cutoff"].astype(int)
    w_deploy_peak = _binary_weights(y_deploy_peak, pd.Series(np.ones(len(deploy_frame)), index=deploy_frame.index))
    peak_deploy = _fit_lgbm_binary(X_deploy, y_deploy_peak, w_deploy_peak)
    y_deploy_rise = pd.Categorical(deploy_frame["wu_remaining_rise_bin"], categories=RISE_BIN_LABELS).codes
    w_deploy_rise = _multiclass_weights(pd.Series(y_deploy_rise), pd.Series(np.ones(len(deploy_frame)), index=deploy_frame.index))
    rise_deploy = _fit_lgbm_multiclass(X_deploy, pd.Series(y_deploy_rise), w_deploy_rise, len(RISE_BIN_LABELS))
    joblib.dump(
        {"feature_cols": feature_cols, "rise_labels": RISE_BIN_LABELS, "peak_model": peak_deploy, "rise_model": rise_deploy},
        artifacts_dir / "stageA_deploy_bundle.joblib",
    )

    model_rows.extend(
        [
            {
                "stage_name": "stage_a_peak",
                "model_name": "stageA_wu_local_peak_final",
                "artifact_path": str(artifacts_dir / "stageA_wu_local_peak_final.txt"),
                "split_name": "pretest_to_test",
                "notes": "Final pretest fit for test inference",
            },
            {
                "stage_name": "stage_a_rise",
                "model_name": "stageA_wu_local_rise_final",
                "artifact_path": str(artifacts_dir / "stageA_wu_local_rise_final.txt"),
                "split_name": "pretest_to_test",
                "notes": "Final pretest fit for test inference",
            },
            {
                "stage_name": "stage_a_deploy",
                "model_name": "stageA_deploy_bundle",
                "artifact_path": str(artifacts_dir / "stageA_deploy_bundle.joblib"),
                "split_name": "through_2025",
                "notes": "Deployment bundle fit through 2025",
            },
        ]
    )

    preds_all = pd.concat([pretest_preds, test_pred], ignore_index=True)
    peak_metrics_overall, peak_metrics_cutoff, peak_metrics_station = _binary_metrics_rows(preds_all, stage_name="stage_a_peak", target_col="wu_peak_by_cutoff", prob_col="stage_a_p_peak", station_breakout=True)
    rise_metrics_overall, rise_metrics_cutoff = _multiclass_metrics_rows(
        preds_all.assign(target_numeric=preds_all["wu_remaining_rise"].astype(float)),
        stage_name="stage_a_rise",
        target_col="wu_remaining_rise_bin",
        pred_mean_col="stage_a_rise_mean",
        prob_prefix="stage_a_rise_p_",
        label_order=RISE_BIN_LABELS,
    )
    metrics = pd.concat([peak_metrics_overall, rise_metrics_overall], ignore_index=True)
    metrics_cutoff = pd.concat([peak_metrics_cutoff, rise_metrics_cutoff], ignore_index=True)
    feature_importance = pd.DataFrame(importance_rows)
    log_event(logger, "STAGE_A_DONE", prediction_rows=len(preds_all), metric_rows=len(metrics))
    return downcast_frame(preds_all), downcast_frame(metrics), downcast_frame(metrics_cutoff), downcast_frame(peak_metrics_station), pd.DataFrame(model_rows), feature_importance


def run_stage_b(
    stage_b_df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[FoldSpec],
    artifacts_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_event(logger, "STAGE_B_START", rows=len(stage_b_df), feature_count=len(feature_cols))
    work = stage_b_df.loc[(stage_b_df["date_dt"] >= pd.Timestamp(OOF_START)) & stage_b_df["strict_quality_flag"].eq(1)].copy()
    work["rise_target_numeric"] = work["wu_remaining_rise"].astype(float)
    work["rise_class"] = pd.Categorical(work["wu_remaining_rise_bin"], categories=RISE_BIN_LABELS).codes
    work = work.loc[work["rise_class"] >= 0].copy()
    oof_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for fold in folds:
        train = work.loc[_mask_date_le(work["date_dt"], fold.train_end)].copy()
        valid = work.loc[_mask_date_between(work["date_dt"], fold.valid_start, fold.valid_end)].copy()
        if train.empty or valid.empty:
            continue
        log_event(logger, "STAGE_B_FOLD_START", fold=fold.fold_id, train_rows=len(train), valid_rows=len(valid))
        X_train = train[feature_cols]
        X_valid = valid[feature_cols]
        y_train_peak = train["wu_peak_by_cutoff"].astype(int)
        w_train_peak = _binary_weights(y_train_peak, pd.Series(np.ones(len(train)), index=train.index))
        peak_model = _fit_lgbm_binary(X_train, y_train_peak, w_train_peak)
        p_valid_peak = peak_model.predict_proba(X_valid)[:, 1]

        y_train_rise = train["rise_class"].astype(int)
        w_train_rise = _multiclass_weights(y_train_rise, pd.Series(np.ones(len(train)), index=train.index))
        rise_model = _fit_lgbm_multiclass(X_train, y_train_rise, w_train_rise, len(RISE_BIN_LABELS))
        p_valid_rise = rise_model.predict_proba(X_valid)

        idx_gt1 = [i for i, label in enumerate(RISE_BIN_LABELS) if RISE_BIN_REP[label] > 1.0]
        idx_gt2 = [i for i, label in enumerate(RISE_BIN_LABELS) if RISE_BIN_REP[label] > 2.0]
        fold_pred = valid[
            ["target_date_local", "cutoff_minutes_local", "date_dt", "wu_peak_by_cutoff", "wu_remaining_rise", "wu_remaining_rise_bin"]
        ].copy()
        fold_pred["prediction_source"] = "oof_pretest"
        fold_pred["fold_id"] = fold.fold_id
        fold_pred["stage_b_p_peak"] = p_valid_peak
        fold_pred["stage_b_rise_mean"] = _pmf_mean(p_valid_rise, RISE_BIN_LABELS, RISE_BIN_REP)
        fold_pred["stage_b_rise_q90"] = _pmf_quantile(p_valid_rise, RISE_BIN_LABELS, RISE_BIN_REP, 0.90)
        fold_pred["stage_b_p_rise_gt1"] = p_valid_rise[:, idx_gt1].sum(axis=1)
        fold_pred["stage_b_p_rise_gt2"] = p_valid_rise[:, idx_gt2].sum(axis=1)
        for i, label in enumerate(RISE_BIN_LABELS):
            fold_pred[f"stage_b_rise_p_{label}"] = p_valid_rise[:, i]
        oof_frames.append(fold_pred)

        for feat, gain, split_count in zip(feature_cols, peak_model.booster_.feature_importance("gain"), peak_model.booster_.feature_importance("split")):
            importance_rows.append(
                {"stage_name": "stage_b_peak", "model_name": f"stage_b_peak_{fold.fold_id}", "feature_name": feat, "importance_gain": float(gain), "importance_split": int(split_count)}
            )
        model_rows.extend(
            [
                {"stage_name": "stage_b_peak", "model_name": f"stage_b_peak_{fold.fold_id}", "artifact_path": "", "split_name": fold.fold_id, "notes": "OOF fold model"},
                {"stage_name": "stage_b_rise", "model_name": f"stage_b_rise_{fold.fold_id}", "artifact_path": "", "split_name": fold.fold_id, "notes": "OOF fold model"},
            ]
        )
        log_event(logger, "STAGE_B_FOLD_DONE", fold=fold.fold_id)

    pretest_preds = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    pretest_train = work.loc[_mask_date_le(work["date_dt"], PRETEST_END)].copy()
    test = stage_b_df.loc[stage_b_df["date_dt"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)) & stage_b_df["strict_quality_flag"].eq(1)].copy()
    X_train_all = pretest_train[feature_cols]
    X_test = test[feature_cols]
    y_train_peak = pretest_train["wu_peak_by_cutoff"].astype(int)
    w_train_peak = _binary_weights(y_train_peak, pd.Series(np.ones(len(pretest_train)), index=pretest_train.index))
    peak_final = _fit_lgbm_binary(X_train_all, y_train_peak, w_train_peak)
    peak_final.booster_.save_model(str(artifacts_dir / "stageB_knyc_wu_peak_final.txt"))
    p_test_peak = peak_final.predict_proba(X_test)[:, 1]

    y_train_rise = pd.Categorical(pretest_train["wu_remaining_rise_bin"], categories=RISE_BIN_LABELS).codes
    w_train_rise = _multiclass_weights(pd.Series(y_train_rise), pd.Series(np.ones(len(pretest_train)), index=pretest_train.index))
    rise_final = _fit_lgbm_multiclass(X_train_all, pd.Series(y_train_rise), w_train_rise, len(RISE_BIN_LABELS))
    rise_final.booster_.save_model(str(artifacts_dir / "stageB_knyc_wu_rise_final.txt"))
    p_test_rise = rise_final.predict_proba(X_test)
    idx_gt1 = [i for i, label in enumerate(RISE_BIN_LABELS) if RISE_BIN_REP[label] > 1.0]
    idx_gt2 = [i for i, label in enumerate(RISE_BIN_LABELS) if RISE_BIN_REP[label] > 2.0]
    test_pred = test[["target_date_local", "cutoff_minutes_local", "date_dt", "wu_peak_by_cutoff", "wu_remaining_rise", "wu_remaining_rise_bin"]].copy()
    test_pred["prediction_source"] = "test"
    test_pred["fold_id"] = "final_pretest_fit"
    test_pred["stage_b_p_peak"] = p_test_peak
    test_pred["stage_b_rise_mean"] = _pmf_mean(p_test_rise, RISE_BIN_LABELS, RISE_BIN_REP)
    test_pred["stage_b_rise_q90"] = _pmf_quantile(p_test_rise, RISE_BIN_LABELS, RISE_BIN_REP, 0.90)
    test_pred["stage_b_p_rise_gt1"] = p_test_rise[:, idx_gt1].sum(axis=1)
    test_pred["stage_b_p_rise_gt2"] = p_test_rise[:, idx_gt2].sum(axis=1)
    for i, label in enumerate(RISE_BIN_LABELS):
        test_pred[f"stage_b_rise_p_{label}"] = p_test_rise[:, i]

    deploy = stage_b_df.loc[(stage_b_df["date_dt"] <= pd.Timestamp(TEST_END)) & stage_b_df["strict_quality_flag"].eq(1)].copy()
    X_deploy = deploy[feature_cols]
    y_deploy_peak = deploy["wu_peak_by_cutoff"].astype(int)
    peak_deploy = _fit_lgbm_binary(X_deploy, y_deploy_peak, _binary_weights(y_deploy_peak, pd.Series(np.ones(len(deploy)), index=deploy.index)))
    y_deploy_rise = pd.Categorical(deploy["wu_remaining_rise_bin"], categories=RISE_BIN_LABELS).codes
    rise_deploy = _fit_lgbm_multiclass(
        X_deploy,
        pd.Series(y_deploy_rise),
        _multiclass_weights(pd.Series(y_deploy_rise), pd.Series(np.ones(len(deploy)), index=deploy.index)),
        len(RISE_BIN_LABELS),
    )
    joblib.dump(
        {"feature_cols": feature_cols, "rise_labels": RISE_BIN_LABELS, "peak_model": peak_deploy, "rise_model": rise_deploy},
        artifacts_dir / "stageB_deploy_bundle.joblib",
    )

    model_rows.extend(
        [
            {
                "stage_name": "stage_b_peak",
                "model_name": "stageB_knyc_wu_peak_final",
                "artifact_path": str(artifacts_dir / "stageB_knyc_wu_peak_final.txt"),
                "split_name": "pretest_to_test",
                "notes": "Final pretest fit for KNYC test inference",
            },
            {
                "stage_name": "stage_b_rise",
                "model_name": "stageB_knyc_wu_rise_final",
                "artifact_path": str(artifacts_dir / "stageB_knyc_wu_rise_final.txt"),
                "split_name": "pretest_to_test",
                "notes": "Final pretest fit for KNYC test inference",
            },
            {
                "stage_name": "stage_b_deploy",
                "model_name": "stageB_deploy_bundle",
                "artifact_path": str(artifacts_dir / "stageB_deploy_bundle.joblib"),
                "split_name": "through_2025",
                "notes": "Deployment bundle fit through 2025",
            },
        ]
    )
    preds_all = pd.concat([pretest_preds, test_pred], ignore_index=True)
    peak_metrics_overall, peak_metrics_cutoff, _ = _binary_metrics_rows(preds_all, stage_name="stage_b_peak", target_col="wu_peak_by_cutoff", prob_col="stage_b_p_peak")
    rise_metrics_overall, rise_metrics_cutoff = _multiclass_metrics_rows(
        preds_all.assign(target_numeric=preds_all["wu_remaining_rise"].astype(float)),
        stage_name="stage_b_rise",
        target_col="wu_remaining_rise_bin",
        pred_mean_col="stage_b_rise_mean",
        prob_prefix="stage_b_rise_p_",
        label_order=RISE_BIN_LABELS,
    )
    metrics = pd.concat([peak_metrics_overall, rise_metrics_overall], ignore_index=True)
    metrics_cutoff = pd.concat([peak_metrics_cutoff, rise_metrics_cutoff], ignore_index=True)
    feature_importance = pd.DataFrame(importance_rows)
    log_event(logger, "STAGE_B_DONE", prediction_rows=len(preds_all), metric_rows=len(metrics))
    return downcast_frame(preds_all), downcast_frame(metrics), downcast_frame(metrics_cutoff), pd.DataFrame(model_rows), feature_importance


def run_stage_c(
    stage_c_df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[FoldSpec],
    artifacts_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_event(logger, "STAGE_C_START", rows=len(stage_c_df), feature_count=len(feature_cols))
    work = stage_c_df.loc[(stage_c_df["date_dt"] >= pd.Timestamp(OOF_START)) & stage_c_df["strict_quality_flag"].eq(1) & stage_c_df["bridge_bin"].isin(BRIDGE_BIN_LABELS)].copy()
    work["target_numeric"] = work["bridge_delta"].astype(float)
    work["bridge_class"] = pd.Categorical(work["bridge_bin"], categories=BRIDGE_BIN_LABELS).codes

    oof_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = work.loc[_mask_date_le(work["date_dt"], fold.train_end)].copy()
        valid = work.loc[_mask_date_between(work["date_dt"], fold.valid_start, fold.valid_end)].copy()
        if train.empty or valid.empty:
            continue
        log_event(logger, "STAGE_C_FOLD_START", fold=fold.fold_id, train_rows=len(train), valid_rows=len(valid))
        X_train = train[feature_cols]
        X_valid = valid[feature_cols]
        y_train = train["bridge_class"].astype(int)
        w_train = _multiclass_weights(y_train, pd.Series(np.ones(len(train)), index=train.index))
        model = _fit_lgbm_multiclass(X_train, y_train, w_train, len(BRIDGE_BIN_LABELS))
        p_valid = model.predict_proba(X_valid)
        fold_pred = valid[["target_date_local", "cutoff_minutes_local", "date_dt", "bridge_delta", "bridge_bin", "target_numeric"]].copy()
        fold_pred["prediction_source"] = "oof_pretest"
        fold_pred["fold_id"] = fold.fold_id
        fold_pred["stage_c_bridge_mean"] = _pmf_mean(p_valid, BRIDGE_BIN_LABELS, BRIDGE_BIN_REP)
        fold_pred["stage_c_bridge_q90"] = _pmf_quantile(p_valid, BRIDGE_BIN_LABELS, BRIDGE_BIN_REP, 0.90)
        fold_pred["stage_c_bridge_entropy"] = -(np.clip(p_valid, 1e-12, 1.0) * np.log(np.clip(p_valid, 1e-12, 1.0))).sum(axis=1)
        for i, label in enumerate(BRIDGE_BIN_LABELS):
            fold_pred[f"stage_c_bridge_p_{label}"] = p_valid[:, i]
        oof_frames.append(fold_pred)
        for feat, gain, split_count in zip(feature_cols, model.booster_.feature_importance("gain"), model.booster_.feature_importance("split")):
            importance_rows.append({"stage_name": "stage_c_bridge", "model_name": f"stage_c_bridge_{fold.fold_id}", "feature_name": feat, "importance_gain": float(gain), "importance_split": int(split_count)})
        model_rows.append({"stage_name": "stage_c_bridge", "model_name": f"stage_c_bridge_{fold.fold_id}", "artifact_path": "", "split_name": fold.fold_id, "notes": "OOF fold model"})
        log_event(logger, "STAGE_C_FOLD_DONE", fold=fold.fold_id)

    pretest_preds = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    pretest_train = work.loc[_mask_date_le(work["date_dt"], PRETEST_END)].copy()
    test = stage_c_df.loc[(stage_c_df["date_dt"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END))) & stage_c_df["strict_quality_flag"].eq(1) & stage_c_df["bridge_bin"].isin(BRIDGE_BIN_LABELS)].copy()
    test["target_numeric"] = test["bridge_delta"].astype(float)
    X_train_all = pretest_train[feature_cols]
    X_test = test[feature_cols]
    y_train_all = pd.Categorical(pretest_train["bridge_bin"], categories=BRIDGE_BIN_LABELS).codes
    model_final = _fit_lgbm_multiclass(
        X_train_all,
        pd.Series(y_train_all),
        _multiclass_weights(pd.Series(y_train_all), pd.Series(np.ones(len(pretest_train)), index=pretest_train.index)),
        len(BRIDGE_BIN_LABELS),
    )
    model_final.booster_.save_model(str(artifacts_dir / "stageC_bridge_delta_final.txt"))
    p_test = model_final.predict_proba(X_test)
    test_pred = test[["target_date_local", "cutoff_minutes_local", "date_dt", "bridge_delta", "bridge_bin", "target_numeric"]].copy()
    test_pred["prediction_source"] = "test"
    test_pred["fold_id"] = "final_pretest_fit"
    test_pred["stage_c_bridge_mean"] = _pmf_mean(p_test, BRIDGE_BIN_LABELS, BRIDGE_BIN_REP)
    test_pred["stage_c_bridge_q90"] = _pmf_quantile(p_test, BRIDGE_BIN_LABELS, BRIDGE_BIN_REP, 0.90)
    test_pred["stage_c_bridge_entropy"] = -(np.clip(p_test, 1e-12, 1.0) * np.log(np.clip(p_test, 1e-12, 1.0))).sum(axis=1)
    for i, label in enumerate(BRIDGE_BIN_LABELS):
        test_pred[f"stage_c_bridge_p_{label}"] = p_test[:, i]

    deploy = stage_c_df.loc[(stage_c_df["date_dt"] <= pd.Timestamp(TEST_END)) & stage_c_df["strict_quality_flag"].eq(1) & stage_c_df["bridge_bin"].isin(BRIDGE_BIN_LABELS)].copy()
    y_deploy = pd.Categorical(deploy["bridge_bin"], categories=BRIDGE_BIN_LABELS).codes
    deploy_model = _fit_lgbm_multiclass(
        deploy[feature_cols],
        pd.Series(y_deploy),
        _multiclass_weights(pd.Series(y_deploy), pd.Series(np.ones(len(deploy)), index=deploy.index)),
        len(BRIDGE_BIN_LABELS),
    )
    joblib.dump({"feature_cols": feature_cols, "bridge_labels": BRIDGE_BIN_LABELS, "bridge_model": deploy_model}, artifacts_dir / "stageC_deploy_bundle.joblib")
    model_rows.extend(
        [
            {"stage_name": "stage_c_bridge", "model_name": "stageC_bridge_delta_final", "artifact_path": str(artifacts_dir / "stageC_bridge_delta_final.txt"), "split_name": "pretest_to_test", "notes": "Final pretest fit for test inference"},
            {"stage_name": "stage_c_deploy", "model_name": "stageC_deploy_bundle", "artifact_path": str(artifacts_dir / "stageC_deploy_bundle.joblib"), "split_name": "through_2025", "notes": "Deployment bundle fit through 2025"},
        ]
    )

    preds_all = pd.concat([pretest_preds, test_pred], ignore_index=True)
    metrics_overall, metrics_cutoff = _multiclass_metrics_rows(preds_all, stage_name="stage_c_bridge", target_col="bridge_bin", pred_mean_col="stage_c_bridge_mean", prob_prefix="stage_c_bridge_p_", label_order=BRIDGE_BIN_LABELS)
    feature_importance = pd.DataFrame(importance_rows)
    log_event(logger, "STAGE_C_DONE", prediction_rows=len(preds_all), metric_rows=len(metrics_overall))
    return downcast_frame(preds_all), downcast_frame(metrics_overall), downcast_frame(metrics_cutoff), pd.DataFrame(model_rows), feature_importance


def wilson_lower_bound(success: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return np.nan
    phat = success / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def run_stage_d(
    stage_d_df: pd.DataFrame,
    feature_cols: list[str],
    artifacts_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_event(logger, "STAGE_D_START", rows=len(stage_d_df), feature_count=len(feature_cols))
    train = stage_d_df.loc[
        (stage_d_df["date_dt"] >= pd.Timestamp(OOF_START))
        & _mask_date_le(stage_d_df["date_dt"], "2021-12-31")
        & stage_d_df["weak_official_target"].isin([0.0, 1.0])
        & (stage_d_df["official_weight"] > 0)
    ].copy()
    calib = stage_d_df.loc[
        _mask_date_between(stage_d_df["date_dt"], CALIB_START, CALIB_END)
        & stage_d_df["weak_official_target"].isin([0.0, 1.0])
        & (stage_d_df["official_weight"] > 0)
    ].copy()
    test = stage_d_df.loc[
        _mask_date_between(stage_d_df["date_dt"], TEST_START, TEST_END)
        & stage_d_df["weak_official_target"].isin([0.0, 1.0])
        & (stage_d_df["official_weight"] > 0)
    ].copy()

    raw_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, C=0.25, random_state=42)),
        ]
    )
    raw_model.fit(train[feature_cols], train["weak_official_target"].astype(int), model__sample_weight=train["official_weight"].to_numpy(dtype=float))
    p_calib_raw = raw_model.predict_proba(calib[feature_cols])[:, 1]
    p_test_raw = raw_model.predict_proba(test[feature_cols])[:, 1]

    silver_calib = calib.loc[calib["silver_subset_flag"] == 1].copy()
    calib_model = None
    calibration_status = "identity"
    p_test = p_test_raw.copy()
    if silver_calib["weak_official_target"].nunique() >= 2 and len(silver_calib) >= 200:
        silver_mask = calib["silver_subset_flag"] == 1
        logit_calib = np.log(np.clip(p_calib_raw[silver_mask], 1e-6, 1 - 1e-6) / (1 - np.clip(p_calib_raw[silver_mask], 1e-6, 1 - 1e-6)))
        calib_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=300, C=1.0, random_state=42)),
            ]
        )
        calib_model.fit(logit_calib.reshape(-1, 1), silver_calib["weak_official_target"].astype(int))
        logit_test = np.log(np.clip(p_test_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(p_test_raw, 1e-6, 1 - 1e-6)))
        p_test = calib_model.predict_proba(logit_test.reshape(-1, 1))[:, 1]
        calibration_status = "silver_logistic"

    p_test = 0.85 * test["stage_b_p_peak"].to_numpy(dtype=float) + 0.15 * p_test
    p_test = np.clip(p_test, 0.0, 1.0)

    pred = test[
        [
            "target_date_local",
            "cutoff_minutes_local",
            "date_dt",
            "weak_official_target",
            "official_weight",
            "silver_subset_flag",
            "future_wu_rise",
            "future_official_gap",
            "bridge_delta",
            "quality_score",
            "regime_tag",
            "actionability_flag",
            "hard_midday_flag",
            "stage_b_p_peak",
            "stage_b_rise_mean",
            "stage_b_rise_q90",
            "stage_c_bridge_mean",
            "stage_c_bridge_entropy",
        ]
    ].copy()
    pred["prediction_source"] = "test"
    pred["stage_d_official_prob"] = p_test
    pred["stage_d_official_pred"] = (p_test >= 0.5).astype(int)
    pred["calibration_status"] = calibration_status

    silver = pred.loc[pred["silver_subset_flag"] == 1].copy()
    silver_overall, silver_cutoff, _ = _binary_metrics_rows(silver, stage_name="stage_d_official_silver", target_col="weak_official_target", prob_col="stage_d_official_prob")

    threshold_rows: list[dict[str, Any]] = []
    severity_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    for thr in OFFICIAL_THRESHOLDS:
        sub = pred.loc[pred["stage_d_official_prob"] >= thr].copy()
        silver_sub = sub.loc[sub["silver_subset_flag"] == 1]
        precision_yes = float(silver_sub["weak_official_target"].mean()) if not silver_sub.empty else np.nan
        threshold_rows.append(
            {
                "threshold": thr,
                "coverage_all_test": float(len(sub) / len(pred)) if len(pred) else np.nan,
                "coverage_silver_test": float(len(silver_sub) / len(silver)) if len(silver) else np.nan,
                "silver_precision_yes": precision_yes,
                "silver_yes_count": int(len(silver_sub)),
                "silver_yes_wilson_lb": wilson_lower_bound(int(silver_sub["weak_official_target"].sum()), int(len(silver_sub))) if not silver_sub.empty else np.nan,
            }
        )
        severity_rows.append(
            {
                "threshold": thr,
                "row_count": int(len(sub)),
                "future_wu_rise_gt0_rate": float((sub["future_wu_rise"] > 0).mean()) if not sub.empty else np.nan,
                "future_wu_rise_gt1_rate": float((sub["future_wu_rise"] > 1).mean()) if not sub.empty else np.nan,
                "future_wu_rise_gt2_rate": float((sub["future_wu_rise"] > 2).mean()) if not sub.empty else np.nan,
                "future_wu_rise_gt4_rate": float((sub["future_wu_rise"] > 4).mean()) if not sub.empty else np.nan,
                "future_official_gap_gt0_rate": float((sub["future_official_gap"] > 0).mean()) if not sub.empty else np.nan,
                "future_official_gap_gt1_rate": float((sub["future_official_gap"] > 1).mean()) if not sub.empty else np.nan,
                "future_official_gap_gt2_rate": float((sub["future_official_gap"] > 2).mean()) if not sub.empty else np.nan,
                "future_official_gap_gt4_rate": float((sub["future_official_gap"] > 4).mean()) if not sub.empty else np.nan,
                "avg_future_wu_rise": float(sub["future_wu_rise"].mean()) if not sub.empty else np.nan,
                "avg_future_official_gap": float(sub["future_official_gap"].mean()) if not sub.empty else np.nan,
            }
        )
        false_yes = sub.loc[(sub["silver_subset_flag"] == 1) & (sub["weak_official_target"] == 0)].copy()
        for _, row in false_yes.iterrows():
            casebook_rows.append(
                {
                    "threshold": thr,
                    "target_date_local": row["target_date_local"],
                    "cutoff_minutes_local": int(row["cutoff_minutes_local"]),
                    "stage_d_official_prob": float(row["stage_d_official_prob"]),
                    "future_wu_rise": float(row["future_wu_rise"]),
                    "future_official_gap": float(row["future_official_gap"]),
                    "regime_tag": row["regime_tag"],
                    "quality_score": float(row["quality_score"]),
                }
            )

    midday_rows: list[dict[str, Any]] = []
    for thr in OFFICIAL_THRESHOLDS:
        sub = pred.loc[(pred["hard_midday_flag"] == 1) & (pred["stage_d_official_prob"] >= thr)]
        silver_sub = sub.loc[sub["silver_subset_flag"] == 1]
        midday_rows.append(
            {
                "threshold": thr,
                "coverage_hard_midday": float(len(sub) / int((pred["hard_midday_flag"] == 1).sum())) if int((pred["hard_midday_flag"] == 1).sum()) else np.nan,
                "silver_precision_yes_hard_midday": float(silver_sub["weak_official_target"].mean()) if not silver_sub.empty else np.nan,
                "future_wu_rise_gt2_rate_hard_midday": float((sub["future_wu_rise"] > 2).mean()) if not sub.empty else np.nan,
                "future_official_gap_gt2_rate_hard_midday": float((sub["future_official_gap"] > 2).mean()) if not sub.empty else np.nan,
            }
        )

    deploy = stage_d_df.loc[
        (stage_d_df["date_dt"] <= pd.Timestamp(TEST_END))
        & stage_d_df["weak_official_target"].isin([0.0, 1.0])
        & (stage_d_df["official_weight"] > 0)
    ].copy()
    deploy_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, C=0.25, random_state=42)),
        ]
    )
    deploy_model.fit(deploy[feature_cols], deploy["weak_official_target"].astype(int), model__sample_weight=deploy["official_weight"].to_numpy(dtype=float))
    joblib.dump({"feature_cols": feature_cols, "raw_model": deploy_model, "calibration_model": calib_model, "anchor_mix_weight": 0.85}, artifacts_dir / "stageD_deploy_bundle.joblib")
    joblib.dump(raw_model, artifacts_dir / "stageD_official_raw_model.joblib")
    if calib_model is not None:
        joblib.dump(calib_model, artifacts_dir / "stageD_official_calibrator.joblib")

    model_registry = pd.DataFrame(
        [
            {"stage_name": "stage_d_official", "model_name": "stageD_official_raw_model", "artifact_path": str(artifacts_dir / "stageD_official_raw_model.joblib"), "split_name": "train_to_calib_test", "notes": "Raw official correction head"},
            {"stage_name": "stage_d_official", "model_name": "stageD_official_calibrator", "artifact_path": str(artifacts_dir / "stageD_official_calibrator.joblib") if calib_model is not None else "", "split_name": "calibration", "notes": calibration_status},
            {"stage_name": "stage_d_deploy", "model_name": "stageD_deploy_bundle", "artifact_path": str(artifacts_dir / "stageD_deploy_bundle.joblib"), "split_name": "through_2025", "notes": "Deployment bundle fit through 2025"},
        ]
    )
    model_core: LogisticRegression = raw_model.named_steps["model"]
    feature_importance = pd.DataFrame(
        [
            {"stage_name": "stage_d_official", "model_name": "stageD_official_raw_model", "feature_name": feat, "importance_gain": float(abs(coef)), "importance_split": 0}
            for feat, coef in zip(feature_cols, model_core.coef_[0])
        ]
    )
    log_event(logger, "STAGE_D_DONE", prediction_rows=len(pred), silver_rows=len(silver))
    return (
        downcast_frame(pred),
        downcast_frame(silver_overall),
        downcast_frame(silver_cutoff),
        downcast_frame(pd.DataFrame(threshold_rows).merge(pd.DataFrame(midday_rows), on="threshold", how="left")),
        downcast_frame(pd.DataFrame(severity_rows)),
        downcast_frame(pd.DataFrame(casebook_rows)),
        model_registry,
        feature_importance,
    )


def write_results_db(db_path: Path, table_frames: list[tuple[str, pd.DataFrame]], logger: logging.Logger) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        for table_name, frame in table_frames:
            log_event(logger, "WRITE_TABLE", table=table_name, rows=len(frame))
            frame.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stage_a_predictions ON stage_a_predictions(target_date_local, cutoff_minutes_local, station_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stage_b_predictions ON stage_b_predictions(target_date_local, cutoff_minutes_local)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stage_c_predictions ON stage_c_predictions(target_date_local, cutoff_minutes_local)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stage_d_predictions ON stage_d_predictions(target_date_local, cutoff_minutes_local)")
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    results_root = Path(args.results_root).resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("knyc_peak_hierarchical_v1_%Y%m%dT%H%M%SZ")
    run_dir = results_root / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logger = init_logger(args.log_level, log_path)
    run_started = time.perf_counter()

    log_event(logger, "RUN_START", run_id=run_id, data_root=str(data_root), results_root=str(results_root), log_path=str(log_path))
    fold_specs = [FoldSpec(*spec) for spec in OOF_FOLDS]
    fold_registry = pd.DataFrame([{"fold_id": f.fold_id, "train_end": f.train_end, "valid_start": f.valid_start, "valid_end": f.valid_end} for f in fold_specs])

    station_summaries = []
    cleaning_frames = []
    station_obs_clean: dict[str, pd.DataFrame] = {}
    station_truth: dict[str, pd.DataFrame] = {}
    for station in ALL_STATIONS:
        t0 = time.perf_counter()
        obs_raw, truth_raw, summary = load_station_source(data_root, station)
        obs_clean, cleaning = clean_station_obs(obs_raw, station)
        station_summaries.append(summary)
        cleaning_frames.append(cleaning)
        station_obs_clean[station] = obs_clean
        station_truth[station] = truth_raw.copy()
        log_event(logger, "LOAD_AND_CLEAN_DONE", station=station, obs_rows=len(obs_raw), obs_clean_rows=len(obs_clean), truth_rows=len(truth_raw), invalidated=int(cleaning["invalidated_rows"].sum()), elapsed_sec=round(time.perf_counter() - t0, 1))

    dst_audit, dst_summary = compute_dst_alignment_audit(station_obs_clean[MAIN_STATION], station_truth[MAIN_STATION])
    klga_audit, klga_summary = compute_klga_dedupe_audit(station_obs_clean["KNYC"], station_obs_clean["KLGA"])
    active_stations = [s for s in ALL_STATIONS if not (s == "KLGA" and klga_summary["quarantine_klga"] == 1)]
    active_neighbors = [s for s in PRIMARY_NEIGHBORS if s in active_stations and s != MAIN_STATION]
    log_event(logger, "PROVENANCE_GATES", chosen_day_mapping=dst_summary["chosen_mapping"], klga_quarantined=klga_summary["quarantine_klga"], active_station_count=len(active_stations), active_neighbors=",".join(active_neighbors))

    knyc_truth = station_truth[MAIN_STATION].copy()
    knyc_truth["target_date_local"] = pd.to_datetime(knyc_truth["target_date_local"]).dt.strftime("%Y-%m-%d")
    target_dates = sorted(knyc_truth["target_date_local"].tolist())
    dst_transition_dates = _dst_transition_dates(target_dates)

    station_panels: dict[str, pd.DataFrame] = {}
    station_label_rows: dict[str, pd.DataFrame] = {}
    station_day_summaries: dict[str, pd.DataFrame] = {}
    panel_summary_rows: list[dict[str, Any]] = []
    for station in active_stations:
        panel = build_station_panel(station_obs_clean[station], target_dates, station, logger)
        labels, day_summary = build_station_wu_labels(station, station_obs_clean[station], panel)
        station_panels[station] = panel
        station_label_rows[station] = labels
        station_day_summaries[station] = day_summary
        panel_summary_rows.append({"station_id": station, "panel_rows": int(len(panel)), "panel_days": int(panel["target_date_local"].nunique()), "label_rows": int(len(labels)), "day_summary_rows": int(len(day_summary)), "strict_days": int(day_summary["strict_quality_flag"].sum()), "tolerant_days": int(day_summary["tolerant_quality_flag"].sum())})
        log_event(logger, "PANEL_AND_LABELS_DONE", station=station, panel_rows=len(panel), strict_days=int(day_summary["strict_quality_flag"].sum()))

    station_dummies = [f"station_{s}" for s in active_stations]
    stage_a_df = build_stage_a_frame(station_panels, station_label_rows, station_dummies)
    stage_a_feature_cols = [c for c in stage_a_feature_columns(station_dummies) if c in stage_a_df.columns and is_numeric_dtype(stage_a_df[c])]
    stage_a_preds, stage_a_metrics, stage_a_cutoff_metrics, stage_a_station_metrics, stage_a_registry, stage_a_importance = run_stage_a(stage_a_df, stage_a_feature_cols, fold_specs, artifacts_dir, logger)

    stage_b_df = build_stage_b_frame(station_panels[MAIN_STATION], station_label_rows[MAIN_STATION], station_panels, stage_a_preds, active_neighbors)
    stage_b_regime_cols = sorted([c for c in stage_b_df.columns if c.startswith("regime_")])
    stage_b_feature_cols = [c for c in stage_b_feature_columns(active_neighbors, stage_b_regime_cols) if c in stage_b_df.columns and is_numeric_dtype(stage_b_df[c])]
    stage_b_preds, stage_b_metrics, stage_b_cutoff_metrics, stage_b_registry, stage_b_importance = run_stage_b(stage_b_df, stage_b_feature_cols, fold_specs, artifacts_dir, logger)

    stage_c_df = build_stage_c_frame(stage_b_df, station_truth[MAIN_STATION], station_day_summaries[MAIN_STATION], stage_b_preds)
    stage_c_regime_cols = sorted([c for c in stage_c_df.columns if c.startswith("regime_")])
    stage_c_feature_cols = [c for c in stage_c_feature_columns(stage_c_regime_cols) if c in stage_c_df.columns and is_numeric_dtype(stage_c_df[c])]
    stage_c_preds, stage_c_metrics, stage_c_cutoff_metrics, stage_c_registry, stage_c_importance = run_stage_c(stage_c_df, stage_c_feature_cols, fold_specs, artifacts_dir, logger)

    neighbor_exact = pd.concat([station_label_rows[s] for s in active_neighbors], ignore_index=True) if active_neighbors else pd.DataFrame(columns=["station_id", "target_date_local", "cutoff_minutes_local", "wu_peak_by_cutoff"])
    stage_d_df = build_stage_d_frame(stage_c_df, stage_c_preds, neighbor_exact[["station_id", "target_date_local", "cutoff_minutes_local", "wu_peak_by_cutoff"]], dst_transition_dates)
    stage_d_regime_cols = sorted([c for c in stage_d_df.columns if c.startswith("regime_")])
    stage_d_feature_cols = [c for c in stage_d_feature_columns(stage_d_regime_cols) if c in stage_d_df.columns and is_numeric_dtype(stage_d_df[c])]
    (
        stage_d_preds,
        stage_d_silver_metrics,
        stage_d_silver_cutoff_metrics,
        stage_d_threshold_metrics,
        stage_d_severity_metrics,
        stage_d_casebook,
        stage_d_registry,
        stage_d_importance,
    ) = run_stage_d(stage_d_df, stage_d_feature_cols, artifacts_dir, logger)

    stage_feature_manifest = pd.concat(
        [
            _feature_manifest(stage_a_feature_cols, "stage_a", stage_a_df),
            _feature_manifest(stage_b_feature_cols, "stage_b", stage_b_df),
            _feature_manifest(stage_c_feature_cols, "stage_c", stage_c_df),
            _feature_manifest(stage_d_feature_cols, "stage_d", stage_d_df),
        ],
        ignore_index=True,
    )

    artifact_registry = []
    for path in sorted(artifacts_dir.glob("*")):
        artifact_registry.append({"artifact_name": path.name, "artifact_path": str(path), "size_bytes": int(path.stat().st_size), "kind": path.suffix.lstrip(".") or "file"})

    dataset_summary = pd.DataFrame(
        [
            {"dataset_name": "stage_a", "rows": int(len(stage_a_df)), "strict_rows": int((stage_a_df["strict_quality_flag"] == 1).sum()), "date_min": str(stage_a_df["target_date_local"].min()), "date_max": str(stage_a_df["target_date_local"].max())},
            {"dataset_name": "stage_b", "rows": int(len(stage_b_df)), "strict_rows": int((stage_b_df["strict_quality_flag"] == 1).sum()), "date_min": str(stage_b_df["target_date_local"].min()), "date_max": str(stage_b_df["target_date_local"].max())},
            {"dataset_name": "stage_c", "rows": int(len(stage_c_df)), "strict_rows": int((stage_c_df["strict_quality_flag"] == 1).sum()), "date_min": str(stage_c_df["target_date_local"].min()), "date_max": str(stage_c_df["target_date_local"].max())},
            {"dataset_name": "stage_d", "rows": int(len(stage_d_df)), "strict_rows": int((stage_d_df["strict_quality_flag"] == 1).sum()), "date_min": str(stage_d_df["target_date_local"].min()), "date_max": str(stage_d_df["target_date_local"].max())},
        ]
    )
    provenance_gate = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "chosen_day_mapping": dst_summary["chosen_mapping"],
                "klga_quarantined": klga_summary["quarantine_klga"],
                "active_stations_csv": ",".join(active_stations),
                "active_neighbors_csv": ",".join(active_neighbors),
                "dst_transition_day_count": len(dst_transition_dates),
            }
        ]
    )

    wu_day_summary = pd.concat([station_day_summaries[s] for s in active_stations], ignore_index=True)
    model_registry = pd.concat([stage_a_registry, stage_b_registry, stage_c_registry, stage_d_registry], ignore_index=True)
    feature_importance = pd.concat([stage_a_importance, stage_b_importance, stage_c_importance, stage_d_importance], ignore_index=True)

    results_db_path = results_root / f"{run_id}.sqlite"
    run_manifest = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "generated_at_utc": utc_now_str(),
                "data_root": str(data_root),
                "results_root": str(results_root),
                "results_db_path": str(results_db_path),
                "log_path": str(log_path),
                "artifact_dir": str(artifacts_dir),
                "main_station": MAIN_STATION,
                "active_neighbors_csv": ",".join(active_neighbors),
                "notes": "Fresh hierarchical WU-first bridge KNYC pipeline from EarlyPeak SQLite only.",
            }
        ]
    )

    table_frames = [
        ("run_manifest", run_manifest),
        ("provenance_gate", provenance_gate),
        ("fold_registry", fold_registry),
        ("dst_alignment_audit", dst_audit),
        ("klga_dedupe_audit", klga_audit),
        ("source_station_summary", pd.DataFrame(station_summaries)),
        ("cleaning_field_summary", pd.concat(cleaning_frames, ignore_index=True)),
        ("station_panel_summary", pd.DataFrame(panel_summary_rows)),
        ("wu_day_summary", wu_day_summary),
        ("feature_manifest", stage_feature_manifest),
        ("dataset_summary", dataset_summary),
        ("model_registry", model_registry),
        ("stage_a_metrics_overall", stage_a_metrics),
        ("stage_a_metrics_by_cutoff", stage_a_cutoff_metrics),
        ("stage_a_metrics_by_station", stage_a_station_metrics),
        ("stage_b_metrics_overall", stage_b_metrics),
        ("stage_b_metrics_by_cutoff", stage_b_cutoff_metrics),
        ("stage_c_metrics_overall", stage_c_metrics),
        ("stage_c_metrics_by_cutoff", stage_c_cutoff_metrics),
        ("stage_d_silver_metrics_overall", stage_d_silver_metrics),
        ("stage_d_silver_metrics_by_cutoff", stage_d_silver_cutoff_metrics),
        ("stage_d_threshold_metrics", stage_d_threshold_metrics),
        ("stage_d_full_universe_severity", stage_d_severity_metrics),
        ("stage_d_false_yes_casebook", stage_d_casebook),
        ("stage_a_predictions", stage_a_preds),
        ("stage_b_predictions", stage_b_preds),
        ("stage_c_predictions", stage_c_preds),
        ("stage_d_predictions", stage_d_preds),
        ("model_feature_importance", feature_importance),
        ("artifact_registry", pd.DataFrame(artifact_registry)),
    ]
    log_event(logger, "WRITE_RESULTS_DB_START", path=str(results_db_path))
    write_results_db(results_db_path, table_frames, logger)

    best_stage_b = stage_b_metrics.loc[(stage_b_metrics["stage_name"] == "stage_b_peak") & (stage_b_metrics["split_name"] == "test") & (stage_b_metrics["metric_name"] == "logloss"), "metric_value"]
    summary = {
        "run_id": run_id,
        "results_db_path": str(results_db_path),
        "log_path": str(log_path),
        "artifact_dir": str(artifacts_dir),
        "active_stations": active_stations,
        "active_neighbors": active_neighbors,
        "chosen_day_mapping": dst_summary["chosen_mapping"],
        "klga_quarantined": int(klga_summary["quarantine_klga"]),
        "stage_b_test_logloss": float(best_stage_b.iloc[0]) if not best_stage_b.empty else None,
        "stage_d_thresholds": stage_d_threshold_metrics.to_dict(orient="records"),
    }
    (results_root / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log_event(logger, "RUN_DONE", run_id=run_id, total_elapsed_sec=round(time.perf_counter() - run_started, 1))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
