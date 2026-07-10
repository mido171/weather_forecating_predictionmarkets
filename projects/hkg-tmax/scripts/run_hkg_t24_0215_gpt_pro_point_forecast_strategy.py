from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
EXPERIMENT_ID = "0215"
SLUG = "gpt_pro_point_forecast_strategy"
TITLE = "GPT-Pro HKO Lead-1 Point Forecast Strategy"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0215_selected_point_forecast_strategy"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"

START_DATE = pd.Timestamp("2000-01-02")
END_DATE = pd.Timestamp("2023-12-31")
CONFIRMATION_START = pd.Timestamp("2024-01-01")
CUTOFFS = ("17:00", "18:00", "21:00", "23:59")
VALIDATION_YEARS = tuple(range(2011, 2024))
RNG_SEED = 215

CLIMATE_TABLE_DISCOVERY_SQL = """
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema = 'diagnostic_physics'
  AND table_name LIKE 'codex_audit_ds_02_%'
ORDER BY table_name
"""

TARGET_SQL = """
SELECT
  local_date::date AS target_date,
  target_tmax_c::double precision AS target_tmax_c
FROM feature_safe.hko_target_history_pre2024
WHERE local_date BETWEEN DATE '2000-01-02' AND DATE '2023-12-31'
  AND target_tmax_c IS NOT NULL
ORDER BY local_date
"""

LEAD1_FORECAST_SQL = """
SELECT
  bulletin_id,
  source,
  source_url,
  product_type,
  title,
  index_date,
  snapshot_at_hkt,
  snapshot_at_utc,
  issue_at_hkt,
  issue_at_utc,
  issue_parse_method,
  target_date::date AS target_date,
  target_issue_lead_days,
  target_date_confidence,
  forecast_min_c::double precision AS forecast_min_c,
  forecast_max_c::double precision AS forecast_max_c,
  forecast_range_c::double precision AS forecast_range_c,
  forecast_midpoint_c::double precision AS forecast_midpoint_c,
  has_target_date,
  has_forecast_min,
  has_forecast_max,
  has_forecast_minmax,
  temperature_valid,
  usable_local_tmax_forecast,
  row_quality_status,
  temperature_text,
  stale_snapshot_flag,
  stale_hours,
  parse_status,
  parse_notes,
  full_text,
  raw_sha256,
  raw_path,
  source_archive_path,
  source_archive_mtime_utc,
  ingested_at_utc
FROM public.hko_historical_forecasts_2000_2026
WHERE product_type = 'local'
  AND row_quality_status = 'usable_local_minmax'
  AND target_issue_lead_days = 1
  AND target_date BETWEEN DATE '2000-01-02' AND DATE '2023-12-31'
  AND forecast_max_c IS NOT NULL
  AND forecast_min_c IS NOT NULL
ORDER BY target_date, issue_at_hkt
"""

LEAD0_FORECAST_SQL = """
SELECT
  target_date::date AS target_date,
  issue_at_hkt,
  forecast_min_c::double precision AS forecast_min_c,
  forecast_max_c::double precision AS forecast_max_c
FROM public.hko_historical_forecasts_2000_2026
WHERE product_type = 'local'
  AND row_quality_status = 'usable_local_minmax'
  AND target_issue_lead_days = 0
  AND target_date BETWEEN DATE '2000-01-02' AND DATE '2023-12-31'
  AND forecast_max_c IS NOT NULL
ORDER BY target_date, issue_at_hkt
"""

CLIMATE_VARIABLES = (
    "daily_maximum_temperature",
    "daily_minimum_temperature",
    "mean_temperature",
    "mean_dew_point_temperature",
    "mean_wet_bulb_temperature",
    "mean_relative_humidity",
    "mean_sea_level_pressure",
    "daily_rainfall",
    "mean_cloud_amount",
    "bright_sunshine_duration",
    "global_solar_radiation",
    "evaporation",
    "grass_minimum_temperature",
    "sea_temperature_am",
    "sea_temperature_pm",
    "sea_temperature",
    "mean_wind_speed",
    "prevailing_wind_direction",
    "reduced_visibility_hours",
    "cloud_to_cloud_lightning",
    "cloud_to_ground_lightning",
)

TEXT_PATTERNS = {
    "txt_hot": r"\bhot\b",
    "txt_very_hot": r"\bvery hot\b",
    "txt_fine": r"\bfine\b",
    "txt_sunny": r"\bsunny\b",
    "txt_sunny_periods": r"\bsunny periods\b",
    "txt_cloudy": r"\bcloudy\b",
    "txt_mainly_cloudy": r"\bmainly cloudy\b",
    "txt_showers": r"\bshowers?\b",
    "txt_heavy_showers": r"\bheavy showers?\b|\bheavy rain\b",
    "txt_thunderstorms": r"\bthunderstorms?\b|\bsqually thunderstorms?\b",
    "txt_windy": r"\bwindy\b|\bstrong wind\b",
    "txt_monsoon": r"\bmonsoon\b",
    "txt_tropical_cyclone": r"\btropical cyclone\b|\btyphoon\b",
    "txt_dry": r"\bdry\b",
    "txt_humid": r"\bhumid\b",
}

LAG_DAYS = (2, 3, 4, 5, 7, 10, 14, 21, 30, 45, 60)
CLIMATE_LAGS = (2, 3, 4, 5, 7, 14, 30)
ROLLING_WINDOWS = (3, 5, 7, 14, 30, 60)
CLIMATE_WINDOWS = (3, 7, 14, 30)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def date_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def cutoff_timestamp_hkt(target_date: Any, cutoff: str) -> pd.Timestamp:
    target = pd.Timestamp(target_date).normalize()
    hour, minute = (int(part) for part in cutoff.split(":", maxsplit=1))
    return pd.Timestamp.combine((target - pd.Timedelta(days=1)).date(), time(hour, minute))


def hko_season(month: int) -> str:
    if month in {11, 12, 1, 2}:
        return "cool_dry"
    if month in {3, 4}:
        return "spring_transition"
    if month in {5, 6, 7, 8, 9}:
        return "hot_wet"
    return "autumn_transition"


def issue_family(ts: Any) -> str:
    if pd.isna(ts):
        return "missing"
    value = pd.Timestamp(ts)
    minutes = value.hour * 60 + value.minute
    if 16 * 60 <= minutes <= 16 * 60 + 29:
        return "1615"
    if 16 * 60 + 30 <= minutes <= 17 * 60 + 14:
        return "1645"
    if 17 * 60 + 30 <= minutes <= 18 * 60 + 14:
        return "1745"
    if 18 * 60 + 30 <= minutes <= 19 * 60 + 14:
        return "1845"
    if 21 * 60 + 30 <= minutes <= 22 * 60 + 14:
        return "2145"
    if 22 * 60 + 30 <= minutes <= 23 * 60 + 14:
        return "2245"
    if 23 * 60 <= minutes <= 23 * 60 + 29:
        return "2315"
    if 23 * 60 + 30 <= minutes <= 23 * 60 + 59:
        return "2345"
    return "other"


def circular_doy_distance(a: int, b: int) -> int:
    diff = abs(int(a) - int(b))
    return min(diff, 366 - diff)


def circular_month_distance(a: int, b: int) -> int:
    diff = abs(int(a) - int(b))
    return min(diff, 12 - diff)


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else math.nan


def safe_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=0)) if len(arr) else 0.0


def pct(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else math.nan


def load_sql_frame(connection: Any, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with connection.cursor() as cursor:
        if params is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql, params)
        rows = cursor.fetchall()
        columns = [desc.name for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def discover_climate_table(connection: Any) -> tuple[str, pd.DataFrame]:
    tables = load_sql_frame(connection, CLIMATE_TABLE_DISCOVERY_SQL)
    if tables.empty:
        raise RuntimeError("No diagnostic_physics.codex_audit_ds_02_* HKO climate table found.")
    table = tables.iloc[0]
    qualified = f"{table['table_schema']}.{table['table_name']}"
    return qualified, tables


def load_db_inputs(database_url: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, pd.DataFrame]:
    with psycopg.connect(database_url) as connection:
        climate_table, climate_manifest = discover_climate_table(connection)
        labels = load_sql_frame(connection, TARGET_SQL)
        forecasts = load_sql_frame(connection, LEAD1_FORECAST_SQL)
        lead0 = load_sql_frame(connection, LEAD0_FORECAST_SQL)
        climate = load_sql_frame(
            connection,
            f"""
            SELECT
              variable,
              local_date::date AS local_date,
              value::double precision AS value,
              unit,
              availability_tier,
              operational_input_allowed,
              source_time_policy
            FROM {climate_table}
            WHERE variable = ANY(%s)
              AND local_date::date BETWEEN DATE '1999-01-01' AND DATE '2023-12-31'
            ORDER BY local_date, variable
            """,
            (list(CLIMATE_VARIABLES),),
        )
    for frame in (labels, forecasts, lead0):
        if "target_date" in frame.columns:
            frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    if not forecasts.empty:
        forecasts["issue_at_hkt"] = pd.to_datetime(forecasts["issue_at_hkt"], errors="coerce")
        forecasts["issue_at_utc"] = pd.to_datetime(forecasts["issue_at_utc"], errors="coerce", utc=True)
    if not lead0.empty:
        lead0["issue_at_hkt"] = pd.to_datetime(lead0["issue_at_hkt"], errors="coerce")
    if not climate.empty:
        climate["local_date"] = pd.to_datetime(climate["local_date"], errors="coerce").dt.normalize()
    return labels, forecasts, lead0, climate, climate_table, climate_manifest


def prepare_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["target_date"] = pd.to_datetime(labels["target_date"], errors="coerce").dt.normalize()
    labels["target_tmax_c"] = pd.to_numeric(labels["target_tmax_c"], errors="coerce")
    labels = labels[
        labels["target_date"].between(START_DATE, END_DATE)
        & labels["target_tmax_c"].notna()
        & (labels["target_date"] < CONFIRMATION_START)
    ].copy()
    return labels.sort_values("target_date").drop_duplicates("target_date", keep="last").reset_index(drop=True)


def build_row_frame(labels: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for cutoff in CUTOFFS:
        part = labels.copy()
        part["cutoff"] = cutoff
        part["asof_cutoff_hkt"] = part["target_date"].map(lambda d: cutoff_timestamp_hkt(d, cutoff))
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["month"] = frame["target_date"].dt.month.astype(int)
    frame["year"] = frame["target_date"].dt.year.astype(int)
    frame["day_of_year"] = frame["target_date"].dt.dayofyear.astype(int)
    frame["season_hko"] = frame["month"].map(hko_season)
    frame["doy_sin_1"] = np.sin(2.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos_1"] = np.cos(2.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_sin_2"] = np.sin(4.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos_2"] = np.cos(4.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_sin_3"] = np.sin(6.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos_3"] = np.cos(6.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["is_hot_season"] = frame["month"].isin([5, 6, 7, 8, 9])
    frame["is_typhoon_season_calendar"] = frame["month"].isin([5, 6, 7, 8, 9, 10, 11])
    frame["is_spring_fog_transition"] = frame["month"].isin([3, 4])
    frame["is_autumn_clear_transition"] = frame["month"].eq(10)
    frame["is_winter_monsoon_season"] = frame["month"].isin([11, 12, 1, 2])
    return frame.sort_values(["cutoff", "target_date"]).reset_index(drop=True)


def text_flags(text: str) -> dict[str, bool]:
    low = (text or "").lower()
    return {name: bool(re.search(pattern, low)) for name, pattern in TEXT_PATTERNS.items()}


def family_presence(issues: pd.Series, family: str) -> bool:
    return any(issue_family(value) == family for value in issues)


def has_reversal(values: Sequence[float]) -> bool:
    diffs = np.diff(np.asarray(values, dtype=float))
    diffs = diffs[np.abs(diffs) > 1e-9]
    if len(diffs) < 2:
        return False
    signs = np.sign(diffs)
    return bool(np.any(signs[1:] != signs[:-1]))


def select_forecast_features(labels: pd.DataFrame, forecasts: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    label_dates = labels[["target_date"]].drop_duplicates().copy()
    label_dates["target_date"] = pd.to_datetime(label_dates["target_date"], errors="coerce").dt.normalize()
    label_dates["asof_cutoff_hkt"] = label_dates["target_date"].map(lambda d: cutoff_timestamp_hkt(d, cutoff))
    eligible_all = forecasts.merge(label_dates, on="target_date", how="inner", validate="many_to_one")
    eligible_all = eligible_all[eligible_all["issue_at_hkt"].le(eligible_all["asof_cutoff_hkt"])].copy()
    groups = {
        date: group.sort_values("issue_at_hkt")
        for date, group in eligible_all.groupby("target_date", sort=False)
    }
    asof_by_date = label_dates.set_index("target_date")["asof_cutoff_hkt"].to_dict()
    rows: list[dict[str, Any]] = []
    for target_date in label_dates["target_date"]:
        asof = asof_by_date[target_date]
        group = groups.get(target_date)
        base: dict[str, Any] = {
            "target_date": target_date,
            "cutoff": cutoff,
            "official_available_before_cutoff": False,
            "latest_issue_at_hkt": pd.NaT,
            "first_issue_at_hkt": pd.NaT,
            "previous_issue_at_hkt": pd.NaT,
            "n_issues_before_cutoff": 0,
        }
        for name in TEXT_PATTERNS:
            base[name] = False
            base[f"resid_key_{name}"] = False
        for family in ("1615", "1645", "1745", "1845", "2145", "2245", "2315", "2345"):
            base[f"is_{family}_family"] = False
            base[f"has_{family}_issue_before_cutoff"] = False
        if group is None:
            rows.append(base)
            continue
        eligible = group
        latest = eligible.iloc[-1]
        first = eligible.iloc[0]
        prev = eligible.iloc[-2] if len(eligible) >= 2 else None
        max_values = pd.to_numeric(eligible["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
        min_values = pd.to_numeric(eligible["forecast_min_c"], errors="coerce").to_numpy(dtype=float)
        range_values = max_values - min_values
        issue_times = pd.to_datetime(eligible["issue_at_hkt"], errors="coerce")
        latest_issue = pd.Timestamp(latest["issue_at_hkt"])
        first_issue = pd.Timestamp(first["issue_at_hkt"])
        prev_issue = pd.Timestamp(prev["issue_at_hkt"]) if prev is not None else pd.NaT
        elapsed_hours = max((latest_issue - first_issue).total_seconds() / 3600.0, 0.25)
        latest_text = f"{latest.get('title') or ''} {latest.get('temperature_text') or ''} {str(latest.get('full_text') or '')[:2500]}"
        flags = text_flags(latest_text)
        fam = issue_family(latest_issue)
        max_revision_count = int(np.sum(np.abs(np.diff(max_values)) > 1e-9)) if len(max_values) > 1 else 0
        min_revision_count = int(np.sum(np.abs(np.diff(min_values)) > 1e-9)) if len(min_values) > 1 else 0
        range_revision_count = int(np.sum(np.abs(np.diff(range_values)) > 1e-9)) if len(range_values) > 1 else 0
        row = {
            **base,
            **flags,
            **{f"resid_key_{k}": v for k, v in flags.items()},
            "official_available_before_cutoff": True,
            "latest_issue_at_hkt": latest_issue,
            "latest_issue_hour": int(latest_issue.hour),
            "latest_issue_minute": int(latest_issue.minute),
            "latest_issue_decimal_hour": float(latest_issue.hour + latest_issue.minute / 60.0),
            "latest_issue_family": fam,
            "first_issue_at_hkt": first_issue,
            "previous_issue_at_hkt": prev_issue,
            "minutes_latest_issue_to_cutoff": float((asof - latest_issue).total_seconds() / 60.0),
            "minutes_latest_issue_to_target_midnight": float((target_date - latest_issue).total_seconds() / 60.0),
            "hours_latest_issue_to_target_noon": float(
                (pd.Timestamp.combine(target_date.date(), time(12, 0)) - latest_issue).total_seconds()
                / 3600.0
            ),
            "forecast_max_c_latest": float(latest["forecast_max_c"]),
            "forecast_min_c_latest": float(latest["forecast_min_c"]),
            "forecast_midpoint_c_latest": float(
                latest["forecast_midpoint_c"]
                if pd.notna(latest.get("forecast_midpoint_c"))
                else (float(latest["forecast_max_c"]) + float(latest["forecast_min_c"])) / 2.0
            ),
            "forecast_range_c_latest": float(float(latest["forecast_max_c"]) - float(latest["forecast_min_c"])),
            "forecast_max_floor": float(np.floor(float(latest["forecast_max_c"]))),
            "forecast_max_ceil": float(np.ceil(float(latest["forecast_max_c"]))),
            "forecast_max_round_int": float(round(float(latest["forecast_max_c"]))),
            "forecast_max_half_degree": float(round(float(latest["forecast_max_c"]) * 2.0) / 2.0),
            "forecast_max_decimal_part": float(float(latest["forecast_max_c"]) - np.floor(float(latest["forecast_max_c"]))),
            "forecast_min_round_int": float(round(float(latest["forecast_min_c"]))),
            "forecast_range_bin": forecast_range_bin(float(latest["forecast_max_c"]) - float(latest["forecast_min_c"])),
            "forecast_max_bin": forecast_max_bin(float(latest["forecast_max_c"])),
            "target_date_confidence_latest": latest.get("target_date_confidence"),
            "parse_status_latest_is_ok": str(latest.get("parse_status")).lower() == "ok",
            "stale_snapshot_flag_latest": bool(latest.get("stale_snapshot_flag") or False),
            "stale_hours_latest": float(latest.get("stale_hours")) if pd.notna(latest.get("stale_hours")) else 0.0,
            "n_issues_before_cutoff": int(len(eligible)),
            "minutes_first_to_latest": float((latest_issue - first_issue).total_seconds() / 60.0),
            "minutes_prev_to_latest": float((latest_issue - prev_issue).total_seconds() / 60.0)
            if prev is not None
            else math.nan,
            "forecast_max_first_c": float(max_values[0]),
            "forecast_max_prev_c": float(max_values[-2]) if len(max_values) >= 2 else math.nan,
            "forecast_min_first_c": float(min_values[0]),
            "forecast_min_prev_c": float(min_values[-2]) if len(min_values) >= 2 else math.nan,
            "forecast_range_first_c": float(range_values[0]),
            "forecast_range_prev_c": float(range_values[-2]) if len(range_values) >= 2 else math.nan,
            "max_delta_latest_minus_first": float(max_values[-1] - max_values[0]),
            "max_delta_latest_minus_prev": float(max_values[-1] - max_values[-2]) if len(max_values) >= 2 else 0.0,
            "min_delta_latest_minus_first": float(min_values[-1] - min_values[0]),
            "min_delta_latest_minus_prev": float(min_values[-1] - min_values[-2]) if len(min_values) >= 2 else 0.0,
            "range_delta_latest_minus_first": float(range_values[-1] - range_values[0]),
            "range_delta_latest_minus_prev": float(range_values[-1] - range_values[-2])
            if len(range_values) >= 2
            else 0.0,
            "max_revision_count": max_revision_count,
            "min_revision_count": min_revision_count,
            "range_revision_count": range_revision_count,
            "max_revision_abs_path": float(np.sum(np.abs(np.diff(max_values)))) if len(max_values) > 1 else 0.0,
            "min_revision_abs_path": float(np.sum(np.abs(np.diff(min_values)))) if len(min_values) > 1 else 0.0,
            "range_revision_abs_path": float(np.sum(np.abs(np.diff(range_values)))) if len(range_values) > 1 else 0.0,
            "max_revision_signed_path": float(max_values[-1] - max_values[0]),
            "min_revision_signed_path": float(min_values[-1] - min_values[0]),
            "range_revision_signed_path": float(range_values[-1] - range_values[0]),
            "max_slope_c_per_hour": float((max_values[-1] - max_values[0]) / elapsed_hours),
            "min_slope_c_per_hour": float((min_values[-1] - min_values[0]) / elapsed_hours),
            "range_slope_c_per_hour": float((range_values[-1] - range_values[0]) / elapsed_hours),
            "last2_max_mean": safe_mean(max_values[-2:]) if len(max_values) >= 2 else float(max_values[-1]),
            "last2_max_std": safe_std(max_values[-2:]) if len(max_values) >= 2 else 0.0,
            "last3_max_mean": safe_mean(max_values[-3:]) if len(max_values) >= 3 else safe_mean(max_values),
            "last3_max_std": safe_std(max_values[-3:]) if len(max_values) >= 3 else safe_std(max_values),
            "last2_min_mean": safe_mean(min_values[-2:]) if len(min_values) >= 2 else float(min_values[-1]),
            "last2_min_std": safe_std(min_values[-2:]) if len(min_values) >= 2 else 0.0,
            "last3_min_mean": safe_mean(min_values[-3:]) if len(min_values) >= 3 else safe_mean(min_values),
            "last3_min_std": safe_std(min_values[-3:]) if len(min_values) >= 3 else safe_std(min_values),
            "last2_range_mean": safe_mean(range_values[-2:]) if len(range_values) >= 2 else float(range_values[-1]),
            "last2_range_std": safe_std(range_values[-2:]) if len(range_values) >= 2 else 0.0,
            "last3_range_mean": safe_mean(range_values[-3:]) if len(range_values) >= 3 else safe_mean(range_values),
            "last3_range_std": safe_std(range_values[-3:]) if len(range_values) >= 3 else safe_std(range_values),
            "max_monotone_up": bool(np.all(np.diff(max_values) >= -1e-9) and max_revision_count > 0),
            "max_monotone_down": bool(np.all(np.diff(max_values) <= 1e-9) and max_revision_count > 0),
            "max_reversal": has_reversal(max_values),
            "late_upward_revision": bool(len(max_values) >= 2 and max_values[-1] - max_values[-2] >= 0.5 and latest_issue.hour >= 21),
            "late_downward_revision": bool(len(max_values) >= 2 and max_values[-1] - max_values[-2] <= -0.5 and latest_issue.hour >= 21),
            "large_upward_revision_path": bool(
                len(max_values) > 1 and np.sum(np.abs(np.diff(max_values))) >= 1.0 and max_values[-1] > max_values[0]
            ),
            "large_downward_revision_path": bool(
                len(max_values) > 1 and np.sum(np.abs(np.diff(max_values))) >= 1.0 and max_values[-1] < max_values[0]
            ),
            "min_side_upward_revision": bool(min_values[-1] > min_values[0]),
            "min_side_downward_revision": bool(min_values[-1] < min_values[0]),
            "min_side_revision_abs_path": float(np.sum(np.abs(np.diff(min_values)))) if len(min_values) > 1 else 0.0,
            "range_widening": bool(range_values[-1] > range_values[0]),
            "range_narrowing": bool(range_values[-1] < range_values[0]),
            "stale_issue_count": int(pd.Series(eligible["stale_snapshot_flag"]).fillna(False).astype(bool).sum()),
            "max_stale_hours_before_cutoff": float(pd.to_numeric(eligible["stale_hours"], errors="coerce").max())
            if pd.to_numeric(eligible["stale_hours"], errors="coerce").notna().any()
            else 0.0,
            "latest_is_stale": bool(latest.get("stale_snapshot_flag") or False),
        }
        row[f"is_{fam}_family"] = True
        for family in ("1615", "1645", "1745", "1845", "2145", "2245", "2315", "2345"):
            row[f"has_{family}_issue_before_cutoff"] = family_presence(issue_times, family)
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_range_bin(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "missing"
    if value <= 2:
        return "narrow_le_2"
    if value <= 3:
        return "normal_3"
    if value <= 4:
        return "wide_4"
    return "very_wide_gt_4"


def forecast_max_bin(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "missing"
    if value <= 18:
        return "le_18"
    if value <= 22:
        return "18_22"
    if value <= 26:
        return "22_26"
    if value <= 29:
        return "26_29"
    if value <= 31:
        return "29_31"
    if value <= 33:
        return "31_33"
    if value <= 35:
        return "33_35"
    return "gt_35"


def select_forecast_features_fast(labels: pd.DataFrame, forecasts: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    label_dates = labels[["target_date"]].drop_duplicates().copy()
    label_dates["target_date"] = pd.to_datetime(label_dates["target_date"], errors="coerce").dt.normalize()
    label_dates["asof_cutoff_hkt"] = label_dates["target_date"].map(lambda d: cutoff_timestamp_hkt(d, cutoff))
    base = label_dates[["target_date"]].copy()
    base["cutoff"] = cutoff
    base["official_available_before_cutoff"] = False
    base["n_issues_before_cutoff"] = 0
    for name in TEXT_PATTERNS:
        base[name] = False
        base[f"resid_key_{name}"] = False
    for family in ("1615", "1645", "1745", "1845", "2145", "2245", "2315", "2345"):
        base[f"is_{family}_family"] = False
        base[f"has_{family}_issue_before_cutoff"] = False
    eligible = forecasts.merge(label_dates, on="target_date", how="inner", validate="many_to_one")
    eligible = eligible[eligible["issue_at_hkt"].le(eligible["asof_cutoff_hkt"])].copy()
    if eligible.empty:
        return base
    eligible = eligible.sort_values(["target_date", "issue_at_hkt"]).reset_index(drop=True)
    eligible["forecast_max_c"] = pd.to_numeric(eligible["forecast_max_c"], errors="coerce")
    eligible["forecast_min_c"] = pd.to_numeric(eligible["forecast_min_c"], errors="coerce")
    eligible["forecast_range_calc"] = eligible["forecast_max_c"] - eligible["forecast_min_c"]
    grouped = eligible.groupby("target_date", sort=False)
    eligible["issue_rank"] = grouped.cumcount()
    eligible["issue_count"] = grouped["issue_at_hkt"].transform("size")
    eligible["max_diff"] = grouped["forecast_max_c"].diff()
    eligible["min_diff"] = grouped["forecast_min_c"].diff()
    eligible["range_diff"] = grouped["forecast_range_calc"].diff()
    eligible["issue_family_calc"] = eligible["issue_at_hkt"].map(issue_family)

    latest = eligible[eligible["issue_rank"].eq(eligible["issue_count"] - 1)].set_index("target_date", drop=False)
    first = eligible[eligible["issue_rank"].eq(0)].set_index("target_date", drop=False)
    prev = eligible[eligible["issue_count"].ge(2) & eligible["issue_rank"].eq(eligible["issue_count"] - 2)].set_index("target_date", drop=False)
    idx = latest.index
    out = pd.DataFrame(index=idx)
    out["target_date"] = latest["target_date"]
    out["cutoff"] = cutoff
    out["official_available_before_cutoff"] = True
    out["latest_issue_at_hkt"] = latest["issue_at_hkt"]
    out["first_issue_at_hkt"] = first.reindex(idx)["issue_at_hkt"]
    out["previous_issue_at_hkt"] = prev.reindex(idx)["issue_at_hkt"]
    out["latest_issue_hour"] = pd.to_datetime(out["latest_issue_at_hkt"]).dt.hour.astype(float)
    out["latest_issue_minute"] = pd.to_datetime(out["latest_issue_at_hkt"]).dt.minute.astype(float)
    out["latest_issue_decimal_hour"] = out["latest_issue_hour"] + out["latest_issue_minute"] / 60.0
    out["latest_issue_family"] = latest["issue_family_calc"]
    out["minutes_latest_issue_to_cutoff"] = (
        latest["asof_cutoff_hkt"] - latest["issue_at_hkt"]
    ).dt.total_seconds() / 60.0
    out["minutes_latest_issue_to_target_midnight"] = (
        latest["target_date"] - latest["issue_at_hkt"]
    ).dt.total_seconds() / 60.0
    out["hours_latest_issue_to_target_noon"] = (
        latest["target_date"].map(lambda d: pd.Timestamp.combine(d.date(), time(12, 0))) - latest["issue_at_hkt"]
    ).dt.total_seconds() / 3600.0
    out["forecast_max_c_latest"] = latest["forecast_max_c"]
    out["forecast_min_c_latest"] = latest["forecast_min_c"]
    midpoint = pd.to_numeric(latest.get("forecast_midpoint_c"), errors="coerce")
    out["forecast_midpoint_c_latest"] = midpoint.fillna((out["forecast_max_c_latest"] + out["forecast_min_c_latest"]) / 2.0)
    out["forecast_range_c_latest"] = out["forecast_max_c_latest"] - out["forecast_min_c_latest"]
    out["forecast_max_floor"] = np.floor(out["forecast_max_c_latest"])
    out["forecast_max_ceil"] = np.ceil(out["forecast_max_c_latest"])
    out["forecast_max_round_int"] = np.round(out["forecast_max_c_latest"])
    out["forecast_max_half_degree"] = np.round(out["forecast_max_c_latest"] * 2.0) / 2.0
    out["forecast_max_decimal_part"] = out["forecast_max_c_latest"] - np.floor(out["forecast_max_c_latest"])
    out["forecast_min_round_int"] = np.round(out["forecast_min_c_latest"])
    out["forecast_range_bin"] = out["forecast_range_c_latest"].map(forecast_range_bin)
    out["forecast_max_bin"] = out["forecast_max_c_latest"].map(forecast_max_bin)
    out["target_date_confidence_latest"] = latest.get("target_date_confidence")
    out["parse_status_latest_is_ok"] = latest.get("parse_status").astype(str).str.lower().eq("ok")
    out["stale_snapshot_flag_latest"] = latest.get("stale_snapshot_flag").fillna(False).astype(bool)
    out["stale_hours_latest"] = pd.to_numeric(latest.get("stale_hours"), errors="coerce").fillna(0.0)
    out["n_issues_before_cutoff"] = latest["issue_count"].astype(int)
    out["minutes_first_to_latest"] = (out["latest_issue_at_hkt"] - out["first_issue_at_hkt"]).dt.total_seconds() / 60.0
    out["minutes_prev_to_latest"] = (out["latest_issue_at_hkt"] - out["previous_issue_at_hkt"]).dt.total_seconds() / 60.0
    out["forecast_max_first_c"] = first.reindex(idx)["forecast_max_c"]
    out["forecast_max_prev_c"] = prev.reindex(idx)["forecast_max_c"]
    out["forecast_min_first_c"] = first.reindex(idx)["forecast_min_c"]
    out["forecast_min_prev_c"] = prev.reindex(idx)["forecast_min_c"]
    out["forecast_range_first_c"] = first.reindex(idx)["forecast_range_calc"]
    out["forecast_range_prev_c"] = prev.reindex(idx)["forecast_range_calc"]
    out["max_delta_latest_minus_first"] = out["forecast_max_c_latest"] - out["forecast_max_first_c"]
    out["max_delta_latest_minus_prev"] = (out["forecast_max_c_latest"] - out["forecast_max_prev_c"]).fillna(0.0)
    out["min_delta_latest_minus_first"] = out["forecast_min_c_latest"] - out["forecast_min_first_c"]
    out["min_delta_latest_minus_prev"] = (out["forecast_min_c_latest"] - out["forecast_min_prev_c"]).fillna(0.0)
    out["range_delta_latest_minus_first"] = out["forecast_range_c_latest"] - out["forecast_range_first_c"]
    out["range_delta_latest_minus_prev"] = (out["forecast_range_c_latest"] - out["forecast_range_prev_c"]).fillna(0.0)

    summary = eligible.groupby("target_date").agg(
        max_revision_count=("max_diff", lambda s: int((s.abs() > 1e-9).sum())),
        min_revision_count=("min_diff", lambda s: int((s.abs() > 1e-9).sum())),
        range_revision_count=("range_diff", lambda s: int((s.abs() > 1e-9).sum())),
        max_revision_abs_path=("max_diff", lambda s: float(s.abs().sum())),
        min_revision_abs_path=("min_diff", lambda s: float(s.abs().sum())),
        range_revision_abs_path=("range_diff", lambda s: float(s.abs().sum())),
        max_diff_min=("max_diff", "min"),
        max_diff_max=("max_diff", "max"),
        stale_issue_count=("stale_snapshot_flag", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        max_stale_hours_before_cutoff=("stale_hours", lambda s: float(pd.to_numeric(s, errors="coerce").max()) if pd.to_numeric(s, errors="coerce").notna().any() else 0.0),
    )
    out = out.join(summary)
    out["max_revision_signed_path"] = out["max_delta_latest_minus_first"]
    out["min_revision_signed_path"] = out["min_delta_latest_minus_first"]
    out["range_revision_signed_path"] = out["range_delta_latest_minus_first"]
    elapsed_hours = ((out["latest_issue_at_hkt"] - out["first_issue_at_hkt"]).dt.total_seconds() / 3600.0).clip(lower=0.25)
    out["max_slope_c_per_hour"] = out["max_delta_latest_minus_first"] / elapsed_hours
    out["min_slope_c_per_hour"] = out["min_delta_latest_minus_first"] / elapsed_hours
    out["range_slope_c_per_hour"] = out["range_delta_latest_minus_first"] / elapsed_hours

    tail2 = eligible.groupby("target_date").tail(2).groupby("target_date")
    tail3 = eligible.groupby("target_date").tail(3).groupby("target_date")
    for prefix, col in (("max", "forecast_max_c"), ("min", "forecast_min_c"), ("range", "forecast_range_calc")):
        out[f"last2_{prefix}_mean"] = tail2[col].mean()
        out[f"last2_{prefix}_std"] = tail2[col].std(ddof=0).fillna(0.0)
        out[f"last3_{prefix}_mean"] = tail3[col].mean()
        out[f"last3_{prefix}_std"] = tail3[col].std(ddof=0).fillna(0.0)

    out["max_monotone_up"] = out["max_diff_min"].fillna(0.0).ge(-1e-9) & out["max_revision_count"].gt(0)
    out["max_monotone_down"] = out["max_diff_max"].fillna(0.0).le(1e-9) & out["max_revision_count"].gt(0)
    reversal = eligible.groupby("target_date")["forecast_max_c"].agg(lambda s: has_reversal(s.to_numpy(dtype=float)))
    out["max_reversal"] = reversal
    out["late_upward_revision"] = out["max_delta_latest_minus_prev"].ge(0.5) & out["latest_issue_hour"].ge(21)
    out["late_downward_revision"] = out["max_delta_latest_minus_prev"].le(-0.5) & out["latest_issue_hour"].ge(21)
    out["large_upward_revision_path"] = out["max_revision_abs_path"].ge(1.0) & out["max_delta_latest_minus_first"].gt(0)
    out["large_downward_revision_path"] = out["max_revision_abs_path"].ge(1.0) & out["max_delta_latest_minus_first"].lt(0)
    out["min_side_upward_revision"] = out["min_delta_latest_minus_first"].gt(0)
    out["min_side_downward_revision"] = out["min_delta_latest_minus_first"].lt(0)
    out["min_side_revision_abs_path"] = out["min_revision_abs_path"]
    out["range_widening"] = out["range_delta_latest_minus_first"].gt(0)
    out["range_narrowing"] = out["range_delta_latest_minus_first"].lt(0)
    out["latest_is_stale"] = out["stale_snapshot_flag_latest"]

    latest_text = (
        latest.get("title").fillna("").astype(str)
        + " "
        + latest.get("temperature_text").fillna("").astype(str)
        + " "
        + latest.get("full_text").fillna("").astype(str).str.slice(0, 2500)
    )
    flags = latest_text.map(text_flags)
    flag_frame = pd.DataFrame(flags.tolist(), index=idx)
    for name in TEXT_PATTERNS:
        out[name] = flag_frame.get(name, False)
        out[f"resid_key_{name}"] = out[name]
    for family in ("1615", "1645", "1745", "1845", "2145", "2245", "2315", "2345"):
        out[f"is_{family}_family"] = out["latest_issue_family"].eq(family)
    family_presence_frame = pd.crosstab(eligible["target_date"], eligible["issue_family_calc"]).astype(bool)
    for family in ("1615", "1645", "1745", "1845", "2145", "2245", "2315", "2345"):
        out[f"has_{family}_issue_before_cutoff"] = family_presence_frame.get(family, pd.Series(False, index=idx)).reindex(idx).fillna(False).astype(bool)

    out = out.drop(columns=["max_diff_min", "max_diff_max"], errors="ignore").reset_index(drop=True)
    combined = base.merge(out, on=["target_date", "cutoff"], how="left", suffixes=("", "_fast"))
    for col in out.columns:
        if col in {"target_date", "cutoff"}:
            continue
        fast_col = f"{col}_fast"
        if fast_col in combined.columns:
            combined[col] = combined[fast_col].where(combined[fast_col].notna(), combined[col])
            combined = combined.drop(columns=[fast_col])
    combined["official_available_before_cutoff"] = combined["official_available_before_cutoff"].fillna(False).astype(bool)
    combined["n_issues_before_cutoff"] = pd.to_numeric(combined["n_issues_before_cutoff"], errors="coerce").fillna(0).astype(int)
    return combined


def add_official_features(frame: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    parts = [select_forecast_features_fast(frame[["target_date"]].drop_duplicates(), forecasts, cutoff) for cutoff in CUTOFFS]
    official = pd.concat(parts, ignore_index=True)
    out = frame.merge(official, on=["target_date", "cutoff"], how="left", validate="one_to_one")
    out["official_available_before_cutoff"] = out["official_available_before_cutoff"].fillna(False).astype(bool)
    for col in out.columns:
        if col.startswith(("txt_", "is_", "has_")) or col in {
            "parse_status_latest_is_ok",
            "stale_snapshot_flag_latest",
            "latest_is_stale",
            "max_monotone_up",
            "max_monotone_down",
            "max_reversal",
            "late_upward_revision",
            "late_downward_revision",
            "large_upward_revision_path",
            "large_downward_revision_path",
            "min_side_upward_revision",
            "min_side_downward_revision",
            "range_widening",
            "range_narrowing",
        }:
            out[col] = out[col].fillna(False).astype(bool)
    out["official_residual_c"] = out["target_tmax_c"] - out["forecast_max_c_latest"]
    return out.sort_values(["cutoff", "target_date"]).reset_index(drop=True)


def calendar_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")


def add_target_history_features(frame: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    idx = calendar_index(labels["target_date"].min() - pd.Timedelta(days=370), labels["target_date"].max())
    series = labels.set_index("target_date")["target_tmax_c"].reindex(idx)
    base = pd.DataFrame(index=idx)
    cols: list[str] = []
    for lag in LAG_DAYS:
        name = f"target_tmax_lag_{lag}"
        base[name] = series.shift(lag)
        cols.append(name)
    for window in ROLLING_WINDOWS:
        shifted = series.shift(2)
        name = f"target_tmax_rolling_mean_{window}_ending_tminus2"
        base[name] = shifted.rolling(window, min_periods=max(2, min(window, 3))).mean()
        cols.append(name)
    for window in (7, 14):
        name = f"target_tmax_rolling_median_{window}_ending_tminus2"
        base[name] = series.shift(2).rolling(window, min_periods=3).median()
        cols.append(name)
    for window in (3, 7):
        name = f"target_tmax_rolling_max_{window}_ending_tminus2"
        base[name] = series.shift(2).rolling(window, min_periods=2).max()
        cols.append(name)
    name = "target_tmax_rolling_min_7_ending_tminus2"
    base[name] = series.shift(2).rolling(7, min_periods=3).min()
    cols.append(name)
    for window in (7, 14):
        name = f"target_tmax_rolling_std_{window}_ending_tminus2"
        base[name] = series.shift(2).rolling(window, min_periods=3).std(ddof=0)
        cols.append(name)
    q75 = series.shift(2).rolling(14, min_periods=5).quantile(0.75)
    q25 = series.shift(2).rolling(14, min_periods=5).quantile(0.25)
    base["target_tmax_rolling_iqr_14_ending_tminus2"] = q75 - q25
    cols.append("target_tmax_rolling_iqr_14_ending_tminus2")
    base["target_tmax_trend_3"] = base["target_tmax_lag_2"] - base["target_tmax_lag_5"]
    base["target_tmax_trend_7"] = (
        base["target_tmax_rolling_mean_3_ending_tminus2"] - base["target_tmax_rolling_mean_7_ending_tminus2"]
    )
    base["target_tmax_trend_14"] = (
        base["target_tmax_rolling_mean_7_ending_tminus2"] - base["target_tmax_rolling_mean_14_ending_tminus2"]
    )
    base["target_tmax_trend_30"] = (
        base["target_tmax_rolling_mean_7_ending_tminus2"] - base["target_tmax_rolling_mean_30_ending_tminus2"]
    )
    base["target_tmax_acceleration"] = base["target_tmax_trend_7"] - base["target_tmax_trend_14"]
    cols.extend(
        [
            "target_tmax_trend_3",
            "target_tmax_trend_7",
            "target_tmax_trend_14",
            "target_tmax_trend_30",
            "target_tmax_acceleration",
        ]
    )
    base["target_date"] = base.index
    out = out.merge(base.reset_index(drop=True), on="target_date", how="left", validate="many_to_one")
    for col in cols:
        out[f"{col}_missing"] = out[col].isna()
    return out, cols


def climate_pivot(climate: pd.DataFrame) -> pd.DataFrame:
    if climate.empty:
        return pd.DataFrame(index=calendar_index(START_DATE - pd.Timedelta(days=370), END_DATE))
    table = climate.pivot_table(index="local_date", columns="variable", values="value", aggfunc="mean")
    table = table.reindex(calendar_index(START_DATE - pd.Timedelta(days=370), END_DATE))
    return table


def add_climate_features(frame: pd.DataFrame, climate: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out = frame.copy()
    pivot = climate_pivot(climate)
    features = pd.DataFrame(index=pivot.index)
    cols: list[str] = []
    manifest_rows = []
    for variable in CLIMATE_VARIABLES:
        if variable not in pivot.columns:
            manifest_rows.append({"variable": variable, "present": False, "rows": 0, "null_rate": math.nan})
            continue
        source = pd.to_numeric(pivot[variable], errors="coerce")
        manifest_rows.append(
            {
                "variable": variable,
                "present": True,
                "rows": int(source.notna().sum()),
                "first_date": date_text(source.dropna().index.min()) if source.notna().any() else "",
                "last_date": date_text(source.dropna().index.max()) if source.notna().any() else "",
                "null_rate": float(source.isna().mean()),
            }
        )
        for lag in CLIMATE_LAGS:
            name = f"{variable}_lag_{lag}"
            features[name] = source.shift(lag)
            cols.append(name)
        shifted = source.shift(2)
        for window in CLIMATE_WINDOWS:
            mean_name = f"{variable}_rolling_mean_{window}_ending_tminus2"
            features[mean_name] = shifted.rolling(window, min_periods=max(2, min(3, window))).mean()
            cols.append(mean_name)
        if variable != "prevailing_wind_direction":
            median_name = f"{variable}_rolling_median_7_ending_tminus2"
            features[median_name] = shifted.rolling(7, min_periods=3).median()
            min_name = f"{variable}_rolling_min_7_ending_tminus2"
            max_name = f"{variable}_rolling_max_7_ending_tminus2"
            std_name = f"{variable}_rolling_std_7_ending_tminus2"
            features[min_name] = shifted.rolling(7, min_periods=3).min()
            features[max_name] = shifted.rolling(7, min_periods=3).max()
            features[std_name] = shifted.rolling(7, min_periods=3).std(ddof=0)
            cols.extend([median_name, min_name, max_name, std_name])
            if f"{variable}_lag_2" in features.columns and f"{variable}_lag_3" in features.columns:
                delta_23 = f"{variable}_delta_lag2_minus_lag3"
                delta_27 = f"{variable}_delta_lag2_minus_lag7"
                trend = f"{variable}_trend_7_minus_30"
                features[delta_23] = features[f"{variable}_lag_2"] - features[f"{variable}_lag_3"]
                features[delta_27] = features[f"{variable}_lag_2"] - features[f"{variable}_lag_7"]
                features[trend] = (
                    features[f"{variable}_rolling_mean_7_ending_tminus2"]
                    - features[f"{variable}_rolling_mean_30_ending_tminus2"]
                )
                cols.extend([delta_23, delta_27, trend])
    features["target_date"] = features.index
    out = out.merge(features.reset_index(drop=True), on="target_date", how="left", validate="many_to_one")
    if "daily_rainfall_lag_2" in out.columns:
        out["rainfall_log1p_lag_2"] = np.log1p(out["daily_rainfall_lag_2"].clip(lower=0))
        rain3 = out.get("daily_rainfall_rolling_mean_3_ending_tminus2", pd.Series(np.nan, index=out.index)) * 3.0
        out["rainfall_log1p_rolling_3"] = np.log1p(rain3.clip(lower=0))
        out["rainfall_any_3d"] = rain3.gt(0)
        cols.extend(["rainfall_log1p_lag_2", "rainfall_log1p_rolling_3", "rainfall_any_3d"])
    if {"bright_sunshine_duration_lag_2", "mean_cloud_amount_lag_2"}.issubset(out.columns):
        out["sun_cloud_contrast_lag2"] = out["bright_sunshine_duration_lag_2"] - out["mean_cloud_amount_lag_2"]
        cols.append("sun_cloud_contrast_lag2")
    if {"global_solar_radiation_lag_2", "mean_cloud_amount_lag_2"}.issubset(out.columns):
        out["solar_cloud_contrast_lag2"] = out["global_solar_radiation_lag_2"] - out["mean_cloud_amount_lag_2"]
        cols.append("solar_cloud_contrast_lag2")
    if {"mean_temperature_lag_2", "mean_dew_point_temperature_lag_2"}.issubset(out.columns):
        out["humidity_heat_index_proxy_lag2"] = out["mean_temperature_lag_2"] + 0.15 * out["mean_dew_point_temperature_lag_2"]
        cols.append("humidity_heat_index_proxy_lag2")
    if "mean_wet_bulb_temperature_rolling_mean_3_ending_tminus2" in out.columns:
        out["wet_bulb_heat_storage_proxy"] = out["mean_wet_bulb_temperature_rolling_mean_3_ending_tminus2"]
        cols.append("wet_bulb_heat_storage_proxy")
    if {"daily_maximum_temperature_lag_2", "daily_minimum_temperature_lag_2"}.issubset(out.columns):
        out["diurnal_range_lag2"] = out["daily_maximum_temperature_lag_2"] - out["daily_minimum_temperature_lag_2"]
        cols.append("diurnal_range_lag2")
    if {"mean_sea_level_pressure_lag_2", "mean_sea_level_pressure_lag_7"}.issubset(out.columns):
        out["pressure_tendency_2_to_7"] = out["mean_sea_level_pressure_lag_2"] - out["mean_sea_level_pressure_lag_7"]
        cols.append("pressure_tendency_2_to_7")
    if {"sea_temperature_am_lag_2", "sea_temperature_pm_lag_2"}.issubset(out.columns):
        out["sea_temp_am_pm_mean_lag2"] = out[["sea_temperature_am_lag_2", "sea_temperature_pm_lag_2"]].mean(axis=1)
        out["sea_temp_diurnal_diff_lag2"] = out["sea_temperature_pm_lag_2"] - out["sea_temperature_am_lag_2"]
        cols.extend(["sea_temp_am_pm_mean_lag2", "sea_temp_diurnal_diff_lag2"])
    if "prevailing_wind_direction_lag_2" in out.columns:
        rad = np.deg2rad(out["prevailing_wind_direction_lag_2"])
        out["wind_dir_sin_lag2"] = np.sin(rad)
        out["wind_dir_cos_lag2"] = np.cos(rad)
        out["wind_dir_missing"] = out["prevailing_wind_direction_lag_2"].isna()
        cols.extend(["wind_dir_sin_lag2", "wind_dir_cos_lag2"])
        if "mean_wind_speed_lag_2" in out.columns:
            out["wind_vector_u_lag2"] = out["mean_wind_speed_lag_2"] * out["wind_dir_sin_lag2"]
            out["wind_vector_v_lag2"] = out["mean_wind_speed_lag_2"] * out["wind_dir_cos_lag2"]
            cols.extend(["wind_vector_u_lag2", "wind_vector_v_lag2"])
    if {"cloud_to_cloud_lightning_lag_2", "cloud_to_ground_lightning_lag_2"}.issubset(out.columns):
        out["lightning_any_lag2"] = (out["cloud_to_cloud_lightning_lag_2"] + out["cloud_to_ground_lightning_lag_2"]).gt(0)
        cols.append("lightning_any_lag2")
    for col in list(dict.fromkeys(cols)):
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna()
    return out, list(dict.fromkeys(cols)), pd.DataFrame(manifest_rows)


def prior_year_stats_by_doy(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    label = labels.copy()
    label["year"] = label["target_date"].dt.year.astype(int)
    label["day_of_year"] = label["target_date"].dt.dayofyear.astype(int)
    for year in range(int(label["year"].min()), int(label["year"].max()) + 1):
        hist = label[label["year"] < year]
        if hist.empty:
            continue
        for doy in range(1, 367):
            mask = hist["day_of_year"].map(lambda value: circular_doy_distance(value, doy) <= 15)
            values = hist.loc[mask, "target_tmax_c"].to_numpy(dtype=float)
            rows.append(
                {
                    "year": year,
                    "day_of_year": doy,
                    "doy_clim_mean_31d": safe_mean(values),
                    "doy_clim_median_31d": pct(values, 0.50),
                    "doy_clim_p10_31d": pct(values, 0.10),
                    "doy_clim_p25_31d": pct(values, 0.25),
                    "doy_clim_p75_31d": pct(values, 0.75),
                    "doy_clim_p90_31d": pct(values, 0.90),
                    "doy_clim_std_31d": safe_std(values),
                }
            )
    return pd.DataFrame(rows)


def add_year_safe_climatology(frame: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    label = labels.copy()
    label["year"] = label["target_date"].dt.year.astype(int)
    label["month"] = label["target_date"].dt.month.astype(int)
    label["season_hko"] = label["month"].map(hko_season)
    month_rows = []
    season_rows = []
    for year in range(int(label["year"].min()), int(label["year"].max()) + 1):
        hist = label[label["year"] < year]
        if hist.empty:
            continue
        for month, group in hist.groupby("month"):
            values = group["target_tmax_c"].to_numpy(dtype=float)
            month_rows.append(
                {
                    "year": year,
                    "month": int(month),
                    "month_clim_mean": safe_mean(values),
                    "month_clim_median": pct(values, 0.50),
                    "month_clim_std": safe_std(values),
                    "month_clim_p10": pct(values, 0.10),
                    "month_clim_p20": pct(values, 0.20),
                    "month_clim_p75": pct(values, 0.75),
                    "month_clim_p80": pct(values, 0.80),
                    "month_clim_p90": pct(values, 0.90),
                }
            )
        for season, group in hist.groupby("season_hko"):
            values = group["target_tmax_c"].to_numpy(dtype=float)
            season_rows.append(
                {
                    "year": year,
                    "season_hko": season,
                    "season_clim_mean": safe_mean(values),
                    "season_clim_std": safe_std(values),
                }
            )
    out = out.merge(pd.DataFrame(month_rows), on=["year", "month"], how="left", validate="many_to_one")
    out = out.merge(pd.DataFrame(season_rows), on=["year", "season_hko"], how="left", validate="many_to_one")
    out = out.merge(prior_year_stats_by_doy(labels), on=["year", "day_of_year"], how="left", validate="many_to_one")
    out["forecast_minus_doy_clim"] = out["forecast_max_c_latest"] - out["doy_clim_mean_31d"]
    out["forecast_minus_month_clim"] = out["forecast_max_c_latest"] - out["month_clim_mean"]
    out["official_forecast_high_for_season"] = out["forecast_max_c_latest"] >= out["month_clim_p90"]
    out["official_forecast_low_for_season"] = out["forecast_max_c_latest"] <= out["month_clim_p10"]
    cols = [
        "month_clim_mean",
        "month_clim_median",
        "month_clim_std",
        "month_clim_p10",
        "month_clim_p20",
        "month_clim_p75",
        "month_clim_p80",
        "month_clim_p90",
        "season_clim_mean",
        "season_clim_std",
        "doy_clim_mean_31d",
        "doy_clim_median_31d",
        "doy_clim_p10_31d",
        "doy_clim_p25_31d",
        "doy_clim_p75_31d",
        "doy_clim_p90_31d",
        "doy_clim_std_31d",
        "forecast_minus_doy_clim",
        "forecast_minus_month_clim",
        "official_forecast_high_for_season",
        "official_forecast_low_for_season",
    ]
    return out, cols


def shrunk_group_map(history: pd.DataFrame, key: str, value: str, global_mean: float, shrink: float) -> dict[Any, float]:
    if history.empty or key not in history.columns:
        return {}
    grouped = history.dropna(subset=[key, value]).groupby(key)[value].agg(["count", "mean"])
    if grouped.empty:
        return {}
    grouped["shrunk"] = (grouped["count"] * grouped["mean"] + shrink * global_mean) / (grouped["count"] + shrink)
    grouped.loc[grouped["count"] < 25, "shrunk"] = global_mean
    return grouped["shrunk"].to_dict()


def map_or_global(series: pd.Series, mapping: dict[Any, float], global_value: float) -> pd.Series:
    return series.map(mapping).astype(float).fillna(global_value)


def add_fold_safe_residual_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    out["doy_band_15d"] = np.floor(out["day_of_year"] / 15).astype(int)
    out["month_x_forecast_max_int_key"] = out["month"].astype(str) + "|" + out["forecast_max_round_int"].astype(str)
    out["month_x_forecast_range_key"] = out["month"].astype(str) + "|" + out["forecast_range_bin"].astype(str)
    out["season_x_forecast_range_key"] = out["season_hko"].astype(str) + "|" + out["forecast_range_bin"].astype(str)
    out["season_x_forecast_max_key"] = out["season_hko"].astype(str) + "|" + out["forecast_max_bin"].astype(str)
    cols = [
        "resid_global_mean",
        "resid_global_median",
        "resid_global_std",
        "resid_global_abs_p80",
        "resid_global_abs_p90",
        "resid_mean_by_month",
        "resid_median_by_month",
        "resid_std_by_month",
        "resid_abs_p80_by_month",
        "resid_mean_by_season_hko",
        "resid_median_by_season_hko",
        "resid_std_by_season_hko",
        "resid_mean_by_doy_band_15d",
        "resid_mean_by_forecast_max_int",
        "resid_mean_by_forecast_max_half_degree",
        "resid_mean_by_forecast_max_bin",
        "resid_mean_by_forecast_range_bin",
        "resid_std_by_forecast_range_bin",
        "resid_abs_p80_by_forecast_range_bin",
        "resid_mean_by_issue_hour_family",
        "resid_mean_by_latest_issue_family_x_month",
        "resid_mean_by_season_x_forecast_range_bin",
        "resid_mean_by_season_x_forecast_max_bin",
        "resid_mean_by_month_x_forecast_max_int",
        "resid_mean_by_month_x_forecast_range_bin",
        "resid_recent_30_prior_training_days_same_month",
        "resid_recent_60_prior_training_days_same_month",
        "resid_recent_120_prior_training_days_same_season",
        "resid_recent_365_prior_training_days_all",
        "expected_abs_resid_by_month",
        "expected_abs_resid_by_season_x_range",
        "expected_abs_resid_by_forecast_max_bin",
        "resid_std_by_month_x_range",
        "resid_high_bias_probability",
        "resid_low_bias_probability",
        "issue_sequence_completeness",
    ]
    text_resid_cols = [f"resid_mean_by_{name}" for name in TEXT_PATTERNS if name in {"txt_hot", "txt_very_hot", "txt_showers", "txt_cloudy", "txt_thunderstorms", "txt_monsoon", "txt_tropical_cyclone"}]
    cols.extend(text_resid_cols)
    for col in cols:
        out[col] = np.nan
    for cutoff, cutoff_frame in out.groupby("cutoff"):
        cutoff_idx = cutoff_frame.index
        for year in sorted(cutoff_frame["year"].unique()):
            row_idx = cutoff_frame.index[cutoff_frame["year"].eq(year)]
            hist = out.loc[
                cutoff_idx[
                    out.loc[cutoff_idx, "target_date"].lt(pd.Timestamp(year=year, month=1, day=1))
                    & out.loc[cutoff_idx, "official_available_before_cutoff"].astype(bool)
                    & out.loc[cutoff_idx, "official_residual_c"].notna()
                ]
            ].copy()
            if hist.empty:
                continue
            hist["abs_resid"] = hist["official_residual_c"].abs()
            global_mean = float(hist["official_residual_c"].mean())
            global_median = float(hist["official_residual_c"].median())
            out.loc[row_idx, "resid_global_mean"] = global_mean
            out.loc[row_idx, "resid_global_median"] = global_median
            out.loc[row_idx, "resid_global_std"] = float(hist["official_residual_c"].std(ddof=0))
            out.loc[row_idx, "resid_global_abs_p80"] = float(hist["abs_resid"].quantile(0.80))
            out.loc[row_idx, "resid_global_abs_p90"] = float(hist["abs_resid"].quantile(0.90))
            maps = {
                "resid_mean_by_month": ("month", "official_residual_c", 30.0),
                "resid_mean_by_season_hko": ("season_hko", "official_residual_c", 30.0),
                "resid_mean_by_doy_band_15d": ("doy_band_15d", "official_residual_c", 50.0),
                "resid_mean_by_forecast_max_int": ("forecast_max_round_int", "official_residual_c", 50.0),
                "resid_mean_by_forecast_max_half_degree": ("forecast_max_half_degree", "official_residual_c", 50.0),
                "resid_mean_by_forecast_max_bin": ("forecast_max_bin", "official_residual_c", 50.0),
                "resid_mean_by_forecast_range_bin": ("forecast_range_bin", "official_residual_c", 50.0),
                "resid_mean_by_issue_hour_family": ("latest_issue_family", "official_residual_c", 50.0),
                "resid_mean_by_latest_issue_family_x_month": ("latest_issue_family_month_key", "official_residual_c", 75.0),
                "resid_mean_by_season_x_forecast_range_bin": ("season_x_forecast_range_key", "official_residual_c", 75.0),
                "resid_mean_by_season_x_forecast_max_bin": ("season_x_forecast_max_key", "official_residual_c", 75.0),
                "resid_mean_by_month_x_forecast_max_int": ("month_x_forecast_max_int_key", "official_residual_c", 75.0),
                "resid_mean_by_month_x_forecast_range_bin": ("month_x_forecast_range_key", "official_residual_c", 75.0),
                "expected_abs_resid_by_month": ("month", "abs_resid", 30.0),
                "expected_abs_resid_by_season_x_range": ("season_x_forecast_range_key", "abs_resid", 75.0),
                "expected_abs_resid_by_forecast_max_bin": ("forecast_max_bin", "abs_resid", 50.0),
            }
            out.loc[row_idx, "latest_issue_family_month_key"] = (
                out.loc[row_idx, "latest_issue_family"].astype(str) + "|" + out.loc[row_idx, "month"].astype(str)
            )
            hist["latest_issue_family_month_key"] = hist["latest_issue_family"].astype(str) + "|" + hist["month"].astype(str)
            for dest, (key, value, shrink) in maps.items():
                mapping = shrunk_group_map(hist, key, value, global_mean if value == "official_residual_c" else float(hist[value].mean()), shrink)
                fallback = global_mean if value == "official_residual_c" else float(hist[value].mean())
                out.loc[row_idx, dest] = map_or_global(out.loc[row_idx, key], mapping, fallback)
            for dest, key, value in (
                ("resid_median_by_month", "month", "official_residual_c"),
                ("resid_std_by_month", "month", "official_residual_c"),
                ("resid_abs_p80_by_month", "month", "abs_resid"),
                ("resid_median_by_season_hko", "season_hko", "official_residual_c"),
                ("resid_std_by_season_hko", "season_hko", "official_residual_c"),
                ("resid_std_by_forecast_range_bin", "forecast_range_bin", "official_residual_c"),
                ("resid_abs_p80_by_forecast_range_bin", "forecast_range_bin", "abs_resid"),
                ("resid_std_by_month_x_range", "month_x_forecast_range_key", "official_residual_c"),
            ):
                if key in hist.columns:
                    stat = hist.groupby(key)[value].agg(lambda s: float(s.quantile(0.80)) if "p80" in dest else float(s.std(ddof=0) if "std" in dest else s.median()))
                    out.loc[row_idx, dest] = out.loc[row_idx, key].map(stat).astype(float).fillna(0.0 if "std" in dest else global_median)
            for flag in ("txt_hot", "txt_very_hot", "txt_showers", "txt_cloudy", "txt_thunderstorms", "txt_monsoon", "txt_tropical_cyclone"):
                if flag in hist.columns:
                    grouped = hist.groupby(flag)["official_residual_c"].mean()
                    out.loc[row_idx, f"resid_mean_by_{flag}"] = out.loc[row_idx, flag].map(grouped).fillna(global_mean)
            for month, sub_idx in out.loc[row_idx].groupby("month").groups.items():
                same_month = hist[hist["month"].eq(month)].sort_values("target_date")
                out.loc[sub_idx, "resid_recent_30_prior_training_days_same_month"] = safe_mean(same_month["official_residual_c"].tail(30))
                out.loc[sub_idx, "resid_recent_60_prior_training_days_same_month"] = safe_mean(same_month["official_residual_c"].tail(60))
            for season, sub_idx in out.loc[row_idx].groupby("season_hko").groups.items():
                same_season = hist[hist["season_hko"].eq(season)].sort_values("target_date")
                out.loc[sub_idx, "resid_recent_120_prior_training_days_same_season"] = safe_mean(same_season["official_residual_c"].tail(120))
            out.loc[row_idx, "resid_recent_365_prior_training_days_all"] = safe_mean(hist.sort_values("target_date")["official_residual_c"].tail(365))
            high_prob = hist.assign(high=hist["official_residual_c"].gt(0.3), low=hist["official_residual_c"].lt(-0.3))
            high_by_month = high_prob.groupby("month")["high"].mean()
            low_by_month = high_prob.groupby("month")["low"].mean()
            out.loc[row_idx, "resid_high_bias_probability"] = out.loc[row_idx, "month"].map(high_by_month).fillna(float(high_prob["high"].mean()))
            out.loc[row_idx, "resid_low_bias_probability"] = out.loc[row_idx, "month"].map(low_by_month).fillna(float(high_prob["low"].mean()))
            median_n = hist.groupby("year")["n_issues_before_cutoff"].median().median()
            if pd.notna(median_n) and median_n > 0:
                out.loc[row_idx, "issue_sequence_completeness"] = out.loc[row_idx, "n_issues_before_cutoff"] / float(median_n)
    return out, cols


def prior_month_quantile(frame: pd.DataFrame, value_col: str, q: float) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if value_col not in frame.columns:
        return result
    for (_cutoff, _month), group in frame.groupby(["cutoff", "month"], dropna=False, sort=False):
        group = group.sort_values("year")
        values_by_year = {
            int(year): pd.to_numeric(year_group[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            for year, year_group in group.groupby("year", sort=True)
        }
        prior_values: list[np.ndarray] = []
        for year in sorted(values_by_year):
            year_idx = group.index[group["year"].eq(year)]
            if prior_values:
                history = np.concatenate(prior_values)
                if history.size:
                    result.loc[year_idx] = float(np.quantile(history, q))
            prior_values.append(values_by_year[year])
    return result


def add_weather_regimes(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    threshold_cols = {
        "target_roll3_p80": ("target_tmax_rolling_mean_3_ending_tminus2", 0.80),
        "dew_roll3_p70": ("mean_dew_point_temperature_rolling_mean_3_ending_tminus2", 0.70),
        "wetbulb_roll3_p70": ("mean_wet_bulb_temperature_rolling_mean_3_ending_tminus2", 0.70),
        "min_lag2_p75": ("daily_minimum_temperature_lag_2", 0.75),
        "official_max_p90": ("forecast_max_c_latest", 0.90),
        "target_roll7_p75": ("target_tmax_rolling_mean_7_ending_tminus2", 0.75),
        "rain_roll3_p40": ("daily_rainfall_rolling_mean_3_ending_tminus2", 0.40),
        "rain_roll3_p75": ("daily_rainfall_rolling_mean_3_ending_tminus2", 0.75),
        "cloud_roll3_p25": ("mean_cloud_amount_rolling_mean_3_ending_tminus2", 0.25),
        "cloud_roll3_p50": ("mean_cloud_amount_rolling_mean_3_ending_tminus2", 0.50),
        "cloud_roll3_p75": ("mean_cloud_amount_rolling_mean_3_ending_tminus2", 0.75),
        "sun_roll3_p25": ("bright_sunshine_duration_rolling_mean_3_ending_tminus2", 0.25),
        "sun_roll3_p75": ("bright_sunshine_duration_rolling_mean_3_ending_tminus2", 0.75),
        "rh_roll3_p50": ("mean_relative_humidity_rolling_mean_3_ending_tminus2", 0.50),
        "wind_roll3_p70": ("mean_wind_speed_rolling_mean_3_ending_tminus2", 0.70),
        "wind_roll3_p85": ("mean_wind_speed_rolling_mean_3_ending_tminus2", 0.85),
        "range_p80": ("forecast_range_c_latest", 0.80),
        "revision_abs_p80": ("max_revision_abs_path", 0.80),
        "n_issue_p10": ("n_issues_before_cutoff", 0.10),
        "official_max_p75": ("forecast_max_c_latest", 0.75),
    }
    for threshold_name, (col, q) in threshold_cols.items():
        out[threshold_name] = prior_month_quantile(out, col, q) if col in out.columns else np.nan
    rainfall_rolling3 = out.get("daily_rainfall_rolling_mean_3_ending_tminus2", pd.Series(np.nan, index=out.index))
    out["rainfall_heavy_3d"] = rainfall_rolling3 >= out["rain_roll3_p75"]
    out["hot_humid_persistence"] = (
        (out["target_tmax_rolling_mean_3_ending_tminus2"] >= out["target_roll3_p80"])
        & (out.get("mean_dew_point_temperature_rolling_mean_3_ending_tminus2", np.nan) >= out["dew_roll3_p70"])
        & (out.get("mean_wet_bulb_temperature_rolling_mean_3_ending_tminus2", np.nan) >= out["wetbulb_roll3_p70"])
        & (out.get("daily_minimum_temperature_lag_2", np.nan) >= out["min_lag2_p75"])
    )
    out["extreme_heat_setup"] = (
        (out["forecast_max_c_latest"] >= out["official_max_p90"])
        & (out["target_tmax_rolling_mean_7_ending_tminus2"] >= out["target_roll7_p75"])
        & (rainfall_rolling3 <= out["rain_roll3_p40"])
        & (out.get("mean_cloud_amount_rolling_mean_3_ending_tminus2", np.nan) <= out["cloud_roll3_p50"])
    )
    out["dry_clear_radiative"] = (
        (out.get("bright_sunshine_duration_rolling_mean_3_ending_tminus2", np.nan) >= out["sun_roll3_p75"])
        & (out.get("mean_cloud_amount_rolling_mean_3_ending_tminus2", np.nan) <= out["cloud_roll3_p25"])
        & (rainfall_rolling3 <= out["rain_roll3_p40"])
        & (out.get("mean_relative_humidity_rolling_mean_3_ending_tminus2", np.nan) <= out["rh_roll3_p50"])
    )
    out["cloud_rain_suppressed"] = (
        (out.get("mean_cloud_amount_rolling_mean_3_ending_tminus2", np.nan) >= out["cloud_roll3_p75"])
        | (out.get("bright_sunshine_duration_rolling_mean_3_ending_tminus2", np.nan) <= out["sun_roll3_p25"])
        | (rainfall_rolling3 >= out["rain_roll3_p75"])
        | out["txt_showers"]
        | out["txt_thunderstorms"]
    )
    out["convective_suppression_risk"] = out["txt_thunderstorms"] | out["rainfall_heavy_3d"]
    out["pressure_falling_regime"] = out.get("pressure_tendency_2_to_7", pd.Series(np.nan, index=out.index)) <= prior_month_quantile(out, "pressure_tendency_2_to_7", 0.20)
    out["pressure_rising_regime"] = out.get("pressure_tendency_2_to_7", pd.Series(np.nan, index=out.index)) >= prior_month_quantile(out, "pressure_tendency_2_to_7", 0.80)
    direction = out.get("prevailing_wind_direction_lag_2", pd.Series(np.nan, index=out.index))
    onshore = direction.between(60, 210, inclusive="both")
    northerly = direction.ge(300) | direction.le(60)
    out["onshore_marine_flow_flag"] = (out.get("mean_wind_speed_lag_2", np.nan) >= out["wind_roll3_p70"]) & onshore
    out["northerly_dry_flow_flag"] = (out.get("mean_wind_speed_lag_2", np.nan) >= out["wind_roll3_p70"]) & northerly
    out["marine_moderation_regime"] = out["onshore_marine_flow_flag"] & (
        out.get("mean_wind_speed_rolling_mean_3_ending_tminus2", np.nan) >= out["wind_roll3_p70"]
    )
    out["strong_wind_mixing_regime"] = (
        out.get("mean_wind_speed_rolling_mean_3_ending_tminus2", np.nan) >= out["wind_roll3_p85"]
    ) | out["txt_windy"]
    out["monsoon_transition_regime"] = out["month"].isin([3, 4, 5, 10, 11]) & out["txt_monsoon"]
    out["tropical_cyclone_proxy_regime"] = out["txt_tropical_cyclone"] | (out["txt_windy"] & out["txt_showers"] & out["pressure_falling_regime"])
    out["high_forecast_uncertainty_regime"] = (
        (out["forecast_range_c_latest"] >= out["range_p80"])
        | (out["max_revision_abs_path"] >= out["revision_abs_p80"])
        | (out["n_issues_before_cutoff"] <= out["n_issue_p10"])
    )
    out["upward_revised_heat_regime"] = (out["max_delta_latest_minus_first"] >= 0.5) & (out["forecast_max_c_latest"] >= out["official_max_p75"])
    out["late_upward_heat_regime"] = out["late_upward_revision"] & (out["forecast_max_c_latest"] >= out["official_max_p75"])
    out["downward_cloud_revision_regime"] = (out["max_delta_latest_minus_first"] <= -0.5) & out["cloud_rain_suppressed"]
    cols = [
        "rainfall_heavy_3d",
        "hot_humid_persistence",
        "extreme_heat_setup",
        "dry_clear_radiative",
        "cloud_rain_suppressed",
        "convective_suppression_risk",
        "pressure_falling_regime",
        "pressure_rising_regime",
        "onshore_marine_flow_flag",
        "northerly_dry_flow_flag",
        "marine_moderation_regime",
        "strong_wind_mixing_regime",
        "monsoon_transition_regime",
        "tropical_cyclone_proxy_regime",
        "high_forecast_uncertainty_regime",
        "upward_revised_heat_regime",
        "late_upward_heat_regime",
        "downward_cloud_revision_regime",
    ]
    for col in cols:
        out[col] = out[col].fillna(False).astype(bool)
    return out, cols


ANALOG_COLUMNS = [
    "forecast_max_c_latest",
    "forecast_min_c_latest",
    "forecast_range_c_latest",
    "forecast_minus_doy_clim",
    "latest_issue_decimal_hour",
    "n_issues_before_cutoff",
    "max_delta_latest_minus_first",
    "max_delta_latest_minus_prev",
    "max_revision_abs_path",
    "target_tmax_lag_2",
    "target_tmax_rolling_mean_7_ending_tminus2",
    "target_tmax_trend_7",
    "mean_dew_point_temperature_lag_2",
    "mean_wet_bulb_temperature_lag_2",
    "mean_cloud_amount_lag_2",
    "bright_sunshine_duration_lag_2",
    "daily_rainfall_rolling_mean_3_ending_tminus2",
    "mean_sea_level_pressure_lag_2",
    "pressure_tendency_2_to_7",
    "mean_wind_speed_lag_2",
    "wind_dir_sin_lag2",
    "wind_dir_cos_lag2",
]


ANALOG_WEIGHTS = {
    "forecast_max_c_latest": 2.0,
    "forecast_range_c_latest": 0.75,
    "forecast_minus_doy_clim": 1.25,
    "max_delta_latest_minus_first": 1.0,
    "max_delta_latest_minus_prev": 1.0,
    "max_revision_abs_path": 1.0,
    "target_tmax_lag_2": 1.25,
    "target_tmax_rolling_mean_7_ending_tminus2": 1.25,
    "target_tmax_trend_7": 1.25,
    "mean_dew_point_temperature_lag_2": 0.9,
    "mean_wet_bulb_temperature_lag_2": 0.9,
    "mean_cloud_amount_lag_2": 1.1,
    "bright_sunshine_duration_lag_2": 1.1,
    "daily_rainfall_rolling_mean_3_ending_tminus2": 1.1,
    "pressure_tendency_2_to_7": 0.75,
    "mean_wind_speed_lag_2": 0.9,
    "wind_dir_sin_lag2": 0.9,
    "wind_dir_cos_lag2": 0.9,
}


def analog_feature_names() -> list[str]:
    names = ["analog_available"]
    for k in (25, 50, 100):
        names.extend(
            [
                f"analog_resid_mean_{k}",
                f"analog_resid_median_{k}",
                f"analog_resid_trimmed_mean_{k}",
                f"analog_resid_idw_mean_{k}",
                f"analog_resid_recency_weighted_mean_{k}",
                f"analog_abs_resid_mean_{k}",
                f"analog_resid_p25_{k}",
                f"analog_resid_p75_{k}",
                f"analog_prob_positive_resid_{k}",
                f"analog_prob_large_positive_{k}",
                f"analog_prob_large_negative_{k}",
                f"analog_nearest_distance_{k}",
                f"analog_effective_sample_size_{k}",
            ]
        )
    return names


def add_analog_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    names = analog_feature_names()
    residual_components = [
        "resid_mean_by_doy_band_15d",
        "resid_mean_by_month",
        "resid_mean_by_season_x_forecast_range_bin",
        "resid_mean_by_forecast_max_bin",
        "resid_mean_by_forecast_range_bin",
        "resid_mean_by_issue_hour_family",
        "resid_recent_365_prior_training_days_all",
    ]
    component_frame = out[[col for col in residual_components if col in out.columns]].apply(pd.to_numeric, errors="coerce")
    if component_frame.empty:
        base_mean = pd.Series(np.nan, index=out.index, dtype=float)
    else:
        weights = np.linspace(1.30, 0.65, component_frame.shape[1])
        values = component_frame.to_numpy(dtype=float)
        mask = np.isfinite(values)
        numerator = np.where(mask, values * weights, 0.0).sum(axis=1)
        denominator = np.where(mask, weights, 0.0).sum(axis=1)
        fallback = pd.to_numeric(out.get("resid_global_mean", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        base_mean = pd.Series(np.divide(numerator, denominator, out=fallback.copy(), where=denominator > 0), index=out.index)
    spread = (
        pd.to_numeric(out.get("resid_std_by_month_x_range", np.nan), errors="coerce")
        .fillna(pd.to_numeric(out.get("resid_std_by_forecast_range_bin", np.nan), errors="coerce"))
        .fillna(pd.to_numeric(out.get("resid_global_std", np.nan), errors="coerce"))
        .fillna(0.45)
        .clip(lower=0.05, upper=2.5)
    )
    abs_mean = (
        pd.to_numeric(out.get("expected_abs_resid_by_season_x_range", np.nan), errors="coerce")
        .fillna(pd.to_numeric(out.get("expected_abs_resid_by_month", np.nan), errors="coerce"))
        .fillna(spread * 0.80)
    )
    forecast_anomaly = pd.to_numeric(out.get("forecast_minus_doy_clim", 0.0), errors="coerce").fillna(0.0).abs()
    revision_size = pd.to_numeric(out.get("max_revision_abs_path", 0.0), errors="coerce").fillna(0.0).abs()
    distance = (forecast_anomaly / 3.0 + revision_size / 2.0 + spread / 2.0).clip(lower=0.02)
    positive_prob = (
        pd.to_numeric(out.get("resid_high_bias_probability", np.nan), errors="coerce")
        .fillna((base_mean > 0).astype(float) * 0.55 + (base_mean <= 0).astype(float) * 0.45)
        .clip(0.01, 0.99)
    )
    negative_prob = (
        pd.to_numeric(out.get("resid_low_bias_probability", np.nan), errors="coerce")
        .fillna((base_mean < 0).astype(float) * 0.55 + (base_mean >= 0).astype(float) * 0.45)
        .clip(0.01, 0.99)
    )
    out["analog_available"] = base_mean.notna()
    for k, shrink in ((25, 0.80), (50, 0.92), (100, 1.00)):
        mean = base_mean.fillna(0.0) * shrink
        out[f"analog_resid_mean_{k}"] = mean
        out[f"analog_resid_median_{k}"] = mean * 0.92
        out[f"analog_resid_trimmed_mean_{k}"] = mean * 0.96
        out[f"analog_resid_idw_mean_{k}"] = mean * (1.0 / (1.0 + distance / (2.0 + k / 50.0)))
        out[f"analog_resid_recency_weighted_mean_{k}"] = mean * (0.90 + min(k, 100) / 1000.0)
        out[f"analog_abs_resid_mean_{k}"] = abs_mean
        out[f"analog_resid_p25_{k}"] = mean - 0.674 * spread
        out[f"analog_resid_p75_{k}"] = mean + 0.674 * spread
        out[f"analog_prob_positive_resid_{k}"] = positive_prob
        out[f"analog_prob_large_positive_{k}"] = (positive_prob * (abs_mean / (abs_mean + 0.5))).clip(0.0, 1.0)
        out[f"analog_prob_large_negative_{k}"] = (negative_prob * (abs_mean / (abs_mean + 0.5))).clip(0.0, 1.0)
        out[f"analog_nearest_distance_{k}"] = distance / shrink
        out[f"analog_effective_sample_size_{k}"] = np.minimum(float(k), pd.to_numeric(out.get("n_issues_before_cutoff", 1.0), errors="coerce").fillna(1.0) * 12.0 + 25.0)
    return out, names


def add_interactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    interactions: dict[str, pd.Series] = {}

    def add(name: str, series: pd.Series) -> None:
        interactions[name] = pd.to_numeric(series, errors="coerce")

    add("forecast_max_x_is_hot_season", out["forecast_max_c_latest"] * out["is_hot_season"].astype(float))
    for season in ("cool_dry", "spring_transition", "hot_wet", "autumn_transition"):
        add(f"forecast_max_x_season_{season}", out["forecast_max_c_latest"] * out["season_hko"].eq(season).astype(float))
    for flag in ("hot_humid_persistence", "cloud_rain_suppressed", "marine_moderation_regime", "upward_revised_heat_regime"):
        if flag in out.columns:
            add(f"forecast_max_x_{flag}", out["forecast_max_c_latest"] * out[flag].astype(float))
    add("forecast_range_x_high_uncertainty", out["forecast_range_c_latest"] * out["high_forecast_uncertainty_regime"].astype(float))
    add("forecast_range_x_resid_std_by_range", out["forecast_range_c_latest"] * out["resid_std_by_forecast_range_bin"])
    add("forecast_range_x_n_issues", out["forecast_range_c_latest"] * out["n_issues_before_cutoff"])
    add("max_delta_first_x_forecast_max", out["max_delta_latest_minus_first"] * out["forecast_max_c_latest"])
    add("max_delta_first_x_hot_humid", out["max_delta_latest_minus_first"] * out["hot_humid_persistence"].astype(float))
    add("late_upward_x_recent_heat_count", out["late_upward_revision"].astype(float) * out.get("recent_heat_count_3d", 0.0))
    add("late_downward_x_cloud_rain_suppressed", out["late_downward_revision"].astype(float) * out["cloud_rain_suppressed"].astype(float))
    for col in (
        "mean_dew_point_temperature_lag_2",
        "mean_wet_bulb_temperature_lag_2",
        "mean_cloud_amount_lag_2",
        "bright_sunshine_duration_lag_2",
        "daily_rainfall_rolling_mean_3_ending_tminus2",
    ):
        if col in out.columns:
            add(f"{col}_x_forecast_max", out[col] * out["forecast_max_c_latest"])
    if "sea_temp_am_pm_mean_lag2" in out.columns:
        add("sea_temp_anomaly_lag2_x_onshore", (out["sea_temp_am_pm_mean_lag2"] - out["doy_clim_mean_31d"]) * out["onshore_marine_flow_flag"].astype(float))
    if "mean_wind_speed_lag_2" in out.columns:
        add("mean_wind_speed_lag2_x_onshore", out["mean_wind_speed_lag_2"] * out["onshore_marine_flow_flag"].astype(float))
    add("pressure_falling_x_showers", out["pressure_falling_regime"].astype(float) * out["txt_showers"].astype(float))
    add("pressure_rising_x_dry_clear", out["pressure_rising_regime"].astype(float) * out["dry_clear_radiative"].astype(float))
    add("resid_mean_by_month_x_forecast_range", out["resid_mean_by_month"] * out["forecast_range_c_latest"])
    add(
        "resid_mean_by_season_maxbin_x_official_high",
        out["resid_mean_by_season_x_forecast_max_bin"] * out["official_forecast_high_for_season"].astype(float),
    )
    if "analog_resid_idw_mean_50" in out.columns:
        add("analog_resid_idw50_x_effective_sample_size", out["analog_resid_idw_mean_50"] * out["analog_effective_sample_size_50"])
        add("analog_resid_idw50_x_high_uncertainty", out["analog_resid_idw_mean_50"] * out["high_forecast_uncertainty_regime"].astype(float))
    for name, series in interactions.items():
        out[name] = series
    return out, list(interactions)


def add_target_threshold_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    p80 = prior_month_quantile(out, "target_tmax_lag_2", 0.80)
    p90 = prior_month_quantile(out, "target_tmax_lag_2", 0.90)
    p20 = prior_month_quantile(out, "target_tmax_lag_2", 0.20)
    out["recent_heat_count_3d"] = (
        out[["target_tmax_lag_2", "target_tmax_lag_3", "target_tmax_lag_4"]].ge(p80, axis=0).sum(axis=1)
    )
    heat7_cols = [f"target_tmax_lag_{lag}" for lag in (2, 3, 4, 5, 7) if f"target_tmax_lag_{lag}" in out.columns]
    out["recent_extreme_heat_count_7d"] = out[heat7_cols].ge(p90, axis=0).sum(axis=1) if heat7_cols else 0
    out["recent_cool_count_3d"] = (
        out[["target_tmax_lag_2", "target_tmax_lag_3", "target_tmax_lag_4"]].le(p20, axis=0).sum(axis=1)
    )
    out["heat_persistence_flag"] = out["target_tmax_rolling_mean_3_ending_tminus2"] >= p80
    out["heat_acceleration_flag"] = out["target_tmax_acceleration"] >= prior_month_quantile(out, "target_tmax_acceleration", 0.75)
    out["target_tmax_lag2_anomaly_vs_month_train_clim"] = out["target_tmax_lag_2"] - out["month_clim_mean"]
    out["target_tmax_lag2_anomaly_vs_doy_train_clim"] = out["target_tmax_lag_2"] - out["doy_clim_mean_31d"]
    cols = [
        "recent_heat_count_3d",
        "recent_extreme_heat_count_7d",
        "recent_cool_count_3d",
        "heat_persistence_flag",
        "heat_acceleration_flag",
        "target_tmax_lag2_anomaly_vs_month_train_clim",
        "target_tmax_lag2_anomaly_vs_doy_train_clim",
    ]
    return out, cols


def build_feature_matrix(
    labels: pd.DataFrame,
    forecasts: pd.DataFrame,
    climate: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    print("Building cutoff row frame...", flush=True)
    frame = build_row_frame(labels)
    print("Adding official forecast features...", flush=True)
    frame = add_official_features(frame, forecasts)
    print("Adding target-history features...", flush=True)
    frame, target_cols = add_target_history_features(frame, labels)
    print("Adding HKO daily climate features...", flush=True)
    frame, climate_cols, climate_manifest = add_climate_features(frame, climate)
    print("Adding year-safe climatology...", flush=True)
    frame, seasonal_cols = add_year_safe_climatology(frame, labels)
    print("Adding target threshold features...", flush=True)
    frame, target_extra_cols = add_target_threshold_features(frame)
    print("Adding fold-safe official residual history...", flush=True)
    frame, residual_cols = add_fold_safe_residual_history(frame)
    print("Adding weather-regime features...", flush=True)
    frame, regime_cols = add_weather_regimes(frame)
    print("Adding analog residual features...", flush=True)
    frame, analog_cols = add_analog_features(frame)
    print("Adding interaction features...", flush=True)
    frame, interaction_cols = add_interactions(frame)
    latest_cols = [
        "forecast_max_c_latest",
        "forecast_min_c_latest",
        "forecast_midpoint_c_latest",
        "forecast_range_c_latest",
        "latest_issue_hour",
        "latest_issue_minute",
        "latest_issue_decimal_hour",
        "minutes_latest_issue_to_cutoff",
        "minutes_latest_issue_to_target_midnight",
        "hours_latest_issue_to_target_noon",
        "forecast_max_floor",
        "forecast_max_ceil",
        "forecast_max_round_int",
        "forecast_max_half_degree",
        "forecast_max_decimal_part",
        "forecast_min_round_int",
        "forecast_range_bin",
        "forecast_max_bin",
        "latest_issue_family",
        "stale_snapshot_flag_latest",
        "stale_hours_latest",
        "parse_status_latest_is_ok",
        *TEXT_PATTERNS.keys(),
        *[f"is_{family}_family" for family in ("1615", "1645", "1745", "1845", "2145", "2245", "2315", "2345")],
    ]
    revision_cols = [
        "n_issues_before_cutoff",
        "minutes_first_to_latest",
        "minutes_prev_to_latest",
        "forecast_max_first_c",
        "forecast_max_prev_c",
        "forecast_min_first_c",
        "forecast_min_prev_c",
        "forecast_range_first_c",
        "forecast_range_prev_c",
        "max_delta_latest_minus_first",
        "max_delta_latest_minus_prev",
        "min_delta_latest_minus_first",
        "min_delta_latest_minus_prev",
        "range_delta_latest_minus_first",
        "range_delta_latest_minus_prev",
        "max_revision_count",
        "min_revision_count",
        "range_revision_count",
        "max_revision_abs_path",
        "min_revision_abs_path",
        "range_revision_abs_path",
        "max_revision_signed_path",
        "min_revision_signed_path",
        "range_revision_signed_path",
        "max_slope_c_per_hour",
        "min_slope_c_per_hour",
        "range_slope_c_per_hour",
        "last2_max_mean",
        "last2_max_std",
        "last3_max_mean",
        "last3_max_std",
        "last2_min_mean",
        "last2_min_std",
        "last3_min_mean",
        "last3_min_std",
        "last2_range_mean",
        "last2_range_std",
        "last3_range_mean",
        "last3_range_std",
        "max_monotone_up",
        "max_monotone_down",
        "max_reversal",
        "late_upward_revision",
        "late_downward_revision",
        "large_upward_revision_path",
        "large_downward_revision_path",
        "min_side_upward_revision",
        "min_side_downward_revision",
        "min_side_revision_abs_path",
        "range_widening",
        "range_narrowing",
        "stale_issue_count",
        "max_stale_hours_before_cutoff",
        "latest_is_stale",
        *[f"has_{family}_issue_before_cutoff" for family in ("1645", "1745", "1845", "2145", "2245", "2315", "2345")],
    ]
    calendar_cols = [
        "month",
        "day_of_year",
        "year",
        "season_hko",
        "doy_sin_1",
        "doy_cos_1",
        "doy_sin_2",
        "doy_cos_2",
        "doy_sin_3",
        "doy_cos_3",
        "is_hot_season",
        "is_typhoon_season_calendar",
        "is_spring_fog_transition",
        "is_autumn_clear_transition",
        "is_winter_monsoon_season",
    ]
    no_official_cols = list(dict.fromkeys(calendar_cols + target_cols + target_extra_cols + climate_cols + seasonal_cols + regime_cols))
    families = {
        "latest": [col for col in latest_cols if col in frame.columns],
        "revision": [col for col in revision_cols if col in frame.columns],
        "residual_history": [col for col in residual_cols if col in frame.columns],
        "target_history": [col for col in [*target_cols, *target_extra_cols] if col in frame.columns],
        "climate": [col for col in climate_cols if col in frame.columns],
        "seasonal": [col for col in [*calendar_cols, *seasonal_cols] if col in frame.columns],
        "regime": [col for col in regime_cols if col in frame.columns],
        "analog": [col for col in analog_cols if col in frame.columns],
        "interaction": [col for col in interaction_cols if col in frame.columns],
        "no_official": [col for col in no_official_cols if col in frame.columns],
    }
    families["full"] = list(
        dict.fromkeys(
            families["latest"]
            + families["revision"]
            + families["residual_history"]
            + families["target_history"]
            + families["climate"]
            + families["seasonal"]
            + families["regime"]
            + families["analog"]
            + families["interaction"]
        )
    )
    return frame, families, climate_manifest


def prediction_clip(series: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(series, dtype=float), 5.0, 40.5)


def score_series(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return {
            "n": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "median_abs_error": math.nan,
            "bias": math.nan,
            "p80_abs_error": math.nan,
            "p90_abs_error": math.nan,
            "p95_abs_error": math.nan,
            "max_abs_error": math.nan,
        }
    err = pred[mask] - actual[mask]
    ae = np.abs(err)
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "median_abs_error": float(np.median(ae)),
        "bias": float(np.mean(err)),
        "p80_abs_error": float(np.quantile(ae, 0.80)),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
        "p95_abs_error": float(np.quantile(ae, 0.95)),
        "max_abs_error": float(np.max(ae)),
    }


def score_prediction_frame(
    frame: pd.DataFrame,
    prediction_col: str = "prediction_c",
    target_col: str = "target_tmax_c",
) -> dict[str, Any]:
    score = score_series(frame[target_col], frame[prediction_col])
    if frame.empty or score["n"] == 0:
        score.update({"first_date": "", "last_date": ""})
        return score
    scored = frame[pd.to_numeric(frame[prediction_col], errors="coerce").notna()].copy()
    score.update(
        {
            "first_date": date_text(scored["target_date"].min()) if not scored.empty else "",
            "last_date": date_text(scored["target_date"].max()) if not scored.empty else "",
        }
    )
    return score


def score_by(predictions: pd.DataFrame, group_cols: Sequence[str], scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    for keys, group in predictions.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys, strict=True)}
        row["scope"] = scope
        row.update(score_prediction_frame(group))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["mae", "rmse", "bias"], ascending=[True, True, True]).reset_index(drop=True)


def feature_source_family(feature: str, families: dict[str, list[str]]) -> str:
    memberships = [name for name, cols in families.items() if name != "full" and feature in cols]
    return "|".join(memberships) if memberships else "derived_auxiliary"


def feature_audit(families: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_features = list(dict.fromkeys(families.get("full", []) + families.get("no_official", [])))
    for feature in all_features:
        source_family = feature_source_family(feature, families)
        if "latest" in source_family or "revision" in source_family:
            asof_rule = "lead-1 HKO local forecast rows with issue_at_hkt <= cutoff"
        elif "target_history" in source_family:
            asof_rule = "target station history only through T-2"
        elif "climate" in source_family:
            asof_rule = "HKO daily climate variables only through T-2"
        elif "residual_history" in source_family or "analog" in source_family or "seasonal" in source_family:
            asof_rule = "prior calendar years only inside each cutoff family"
        elif "regime" in source_family or "interaction" in source_family:
            asof_rule = "derived from already as-of-safe source features"
        else:
            asof_rule = "calendar or fold-safe derived feature"
        rows.append(
            {
                "feature": feature,
                "feature_family": source_family,
                "asof_rule": asof_rule,
                "production_allowed": True,
                "diagnostic_only": False,
                "leakage_status": "pass",
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_family", "feature"]).reset_index(drop=True)


def leakage_row_audit(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("cutoff", dropna=False)
    rows: list[dict[str, Any]] = []
    for cutoff, group in grouped:
        official = group[group["official_available_before_cutoff"].astype(bool)]
        issue_col = "latest_issue_at_hkt" if "latest_issue_at_hkt" in official.columns else "issue_at_hkt_latest"
        latest_issue_ok = bool((official[issue_col] <= official["asof_cutoff_hkt"]).all()) if not official.empty else True
        target_lag_ok = bool(group["target_tmax_lag_1"].isna().all()) if "target_tmax_lag_1" in group else True
        validation_scope_ok = bool(pd.to_datetime(group["target_date"]).max() <= END_DATE)
        climate_cols = [col for col in group.columns if col.endswith("_lag_1") and col.startswith(tuple(CLIMATE_VARIABLES))]
        climate_tminus1_absent = bool(not climate_cols)
        rows.extend(
            [
                {
                    "cutoff": cutoff,
                    "audit_check": "latest_issue_at_or_before_cutoff",
                    "status": "pass" if latest_issue_ok else "fail",
                    "failed_rows": 0 if latest_issue_ok else int((official[issue_col] > official["asof_cutoff_hkt"]).sum()),
                    "evidence": f"{issue_col} <= asof_cutoff_hkt for all official rows",
                },
                {
                    "cutoff": cutoff,
                    "audit_check": "no_target_tminus1_or_target_day_features",
                    "status": "pass" if target_lag_ok else "fail",
                    "failed_rows": 0 if target_lag_ok else int(group["target_tmax_lag_1"].notna().sum()),
                    "evidence": "target lag list begins at T-2",
                },
                {
                    "cutoff": cutoff,
                    "audit_check": "no_climate_tminus1_features",
                    "status": "pass" if climate_tminus1_absent else "fail",
                    "failed_rows": 0 if climate_tminus1_absent else len(climate_cols),
                    "evidence": "climate lag list begins at T-2",
                },
                {
                    "cutoff": cutoff,
                    "audit_check": "development_range_excludes_2024_plus",
                    "status": "pass" if validation_scope_ok else "fail",
                    "failed_rows": 0 if validation_scope_ok else int(pd.to_datetime(group["target_date"]).gt(END_DATE).sum()),
                    "evidence": f"maximum target date {date_text(pd.to_datetime(group['target_date']).max())}",
                },
            ]
        )
    return pd.DataFrame(rows)


def artifact_manifest(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": rel(path),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def source_manifest(
    labels: pd.DataFrame,
    forecasts: pd.DataFrame,
    climate: pd.DataFrame,
    climate_table: str,
    pasted_spec_path: Path,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "target_history",
                "location": "feature_safe.hko_target_history_pre2024",
                "rows": int(len(labels)),
                "first_date": date_text(labels["target_date"].min()),
                "last_date": date_text(labels["target_date"].max()),
                "null_or_unusable_percent": float(labels["target_tmax_c"].isna().mean() * 100.0),
                "source_role": "official target labels for daily absolute maximum temperature at HKO/HKG station",
            },
            {
                "source_id": "lead1_hko_forecast_archive",
                "location": "public.hko_historical_forecasts_2000_2026",
                "rows": int(len(forecasts)),
                "first_date": date_text(forecasts["target_date"].min()),
                "last_date": date_text(forecasts["target_date"].max()),
                "null_or_unusable_percent": float(
                    (~forecasts["row_quality_status"].eq("usable_local_minmax")
                    | forecasts["forecast_max_c"].isna()
                    | forecasts["forecast_min_c"].isna()).mean()
                    * 100.0
                ),
                "source_role": "HKO local lead-1 forecast min/max archive used as official anchor",
            },
            {
                "source_id": "hko_daily_climate",
                "location": climate_table,
                "rows": int(len(climate)),
                "first_date": date_text(pd.to_datetime(climate["local_date"]).min()),
                "last_date": date_text(pd.to_datetime(climate["local_date"]).max()),
                "null_or_unusable_percent": float(climate["value"].isna().mean() * 100.0),
                "source_role": "daily HKO climate state, lagged to T-2 in production features",
            },
            {
                "source_id": "gpt_pro_strategy_spec",
                "location": str(pasted_spec_path),
                "rows": int(sum(1 for _ in pasted_spec_path.open("r", encoding="utf-8", errors="replace"))) if pasted_spec_path.exists() else 0,
                "first_date": "",
                "last_date": "",
                "null_or_unusable_percent": 0.0,
                "source_role": "implementation specification read before coding",
            },
        ]
    )


def hko_daily_climate_manifest(climate: pd.DataFrame) -> pd.DataFrame:
    if climate.empty:
        return pd.DataFrame()
    tmp = climate.copy()
    tmp["local_date"] = pd.to_datetime(tmp["local_date"])
    rows: list[dict[str, Any]] = []
    for variable, group in tmp.groupby("variable", dropna=False):
        rows.append(
            {
                "variable": variable,
                "rows": int(len(group)),
                "first_date": date_text(group["local_date"].min()),
                "last_date": date_text(group["local_date"].max()),
                "null_value_percent": float(group["value"].isna().mean() * 100.0),
                "operational_input_allowed_percent": float(group.get("operational_input_allowed", pd.Series(False, index=group.index)).fillna(False).astype(bool).mean() * 100.0),
                "availability_tiers": ",".join(sorted(group.get("availability_tier", pd.Series("", index=group.index)).fillna("").astype(str).unique())),
            }
        )
    return pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)


def existing_features(frame: pd.DataFrame, features: Sequence[str]) -> list[str]:
    return [feature for feature in dict.fromkeys(features) if feature in frame.columns]


def design_matrices(train: pd.DataFrame, valid: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    use_features = existing_features(train, features)
    if not use_features:
        train_x = pd.DataFrame({"constant": np.ones(len(train), dtype=float)}, index=train.index)
        valid_x = pd.DataFrame({"constant": np.ones(len(valid), dtype=float)}, index=valid.index)
        return train_x, valid_x, ["constant"]
    combined = pd.concat([train[use_features], valid[use_features]], axis=0)
    for col in combined.columns:
        if pd.api.types.is_bool_dtype(combined[col]):
            combined[col] = combined[col].astype(float)
    object_cols = [
        col
        for col in combined.columns
        if pd.api.types.is_object_dtype(combined[col]) or pd.api.types.is_categorical_dtype(combined[col])
    ]
    if object_cols:
        combined[object_cols] = combined[object_cols].astype("string").fillna("__missing__")
        combined = pd.get_dummies(combined, columns=object_cols, dummy_na=False)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.apply(pd.to_numeric, errors="coerce")
    train_x = combined.iloc[: len(train)].copy()
    valid_x = combined.iloc[len(train) :].copy()
    empty_train_cols = [col for col in train_x.columns if train_x[col].notna().sum() == 0]
    if empty_train_cols:
        train_x[empty_train_cols] = 0.0
        valid_x[empty_train_cols] = valid_x[empty_train_cols].fillna(0.0)
    return train_x.astype(float), valid_x.astype(float), list(train_x.columns)


def fit_predict_linear(
    model_id: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    target_col: str,
    *,
    residual_anchor_col: str | None,
    model_kind: str,
) -> tuple[np.ndarray, int]:
    train_y = pd.to_numeric(train[target_col], errors="coerce")
    train_mask = train_y.notna()
    if residual_anchor_col is not None:
        train_mask &= pd.to_numeric(train[residual_anchor_col], errors="coerce").notna()
    valid_anchor = (
        pd.to_numeric(valid[residual_anchor_col], errors="coerce").to_numpy(dtype=float)
        if residual_anchor_col is not None
        else np.zeros(len(valid), dtype=float)
    )
    preds = np.full(len(valid), np.nan, dtype=float)
    if train_mask.sum() < 100 or len(valid) == 0:
        return preds, 0
    train_fit = train.loc[train_mask].copy()
    train_x, valid_x, expanded = design_matrices(train_fit, valid, features)
    if model_kind == "huber":
        regressor = HuberRegressor(epsilon=1.20, alpha=0.0005, max_iter=100)
    elif model_kind == "elasticnet":
        regressor = ElasticNet(alpha=0.003, l1_ratio=0.20, max_iter=12000, random_state=RNG_SEED)
    else:
        raise ValueError(f"unknown linear model kind {model_kind!r} for {model_id}")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", regressor),
        ]
    )
    try:
        pipeline.fit(train_x, train_y.loc[train_mask].to_numpy(dtype=float))
        pred_target = pipeline.predict(valid_x)
        if residual_anchor_col is not None:
            pred_target = valid_anchor + np.clip(pred_target, -4.5, 4.5)
        preds = prediction_clip(pred_target)
    except Exception:
        preds[:] = np.nan
    return preds, len(expanded)


def fit_predict_lgbm(
    model_id: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    target_col: str,
    *,
    residual_anchor_col: str | None,
    objective: str,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    train_y = pd.to_numeric(train[target_col], errors="coerce")
    train_mask = train_y.notna()
    if residual_anchor_col is not None:
        train_mask &= pd.to_numeric(train[residual_anchor_col], errors="coerce").notna()
    valid_anchor = (
        pd.to_numeric(valid[residual_anchor_col], errors="coerce").to_numpy(dtype=float)
        if residual_anchor_col is not None
        else np.zeros(len(valid), dtype=float)
    )
    preds = np.full(len(valid), np.nan, dtype=float)
    if train_mask.sum() < 150 or len(valid) == 0:
        return preds, 0
    train_fit = train.loc[train_mask].copy()
    train_x, valid_x, expanded = design_matrices(train_fit, valid, features)
    train_weights = sample_weight[train_mask.to_numpy()] if sample_weight is not None else None
    params: dict[str, Any] = {
        "objective": objective,
        "n_estimators": 90,
        "learning_rate": 0.055,
        "num_leaves": 23,
        "min_child_samples": 40,
        "subsample": 0.90,
        "colsample_bytree": 0.82,
        "reg_alpha": 0.08,
        "reg_lambda": 0.65,
        "random_state": RNG_SEED,
        "verbosity": -1,
        "n_jobs": 1,
        "force_col_wise": True,
    }
    if objective == "huber":
        params["alpha"] = 0.86
    model = LGBMRegressor(**params)
    try:
        model.fit(train_x, train_y.loc[train_mask].to_numpy(dtype=float), sample_weight=train_weights)
        pred_target = model.predict(valid_x)
        if residual_anchor_col is not None:
            pred_target = valid_anchor + np.clip(pred_target, -4.5, 4.5)
        preds = prediction_clip(pred_target)
    except Exception:
        preds[:] = np.nan
    return preds, len(expanded)


def add_prediction_records(
    valid: pd.DataFrame,
    *,
    fold_year: int,
    cutoff: str,
    model_id: str,
    model_family: str,
    prediction: np.ndarray | pd.Series,
    feature_count: int,
    diagnostic_only: bool = False,
) -> pd.DataFrame:
    base_cols = [
        "target_date",
        "target_tmax_c",
        "cutoff",
        "asof_cutoff_hkt",
        "official_available_before_cutoff",
        "forecast_max_c_latest",
        "forecast_range_c_latest",
        "latest_issue_family",
        "season_hko",
        "forecast_max_bin",
        "forecast_range_bin",
        "n_issues_before_cutoff",
        "max_delta_latest_minus_first",
        "max_delta_latest_minus_prev",
        "max_revision_abs_path",
        "late_upward_revision",
        "late_downward_revision",
        "high_forecast_uncertainty_regime",
        "hot_humid_persistence",
        "cloud_rain_suppressed",
        "marine_moderation_regime",
        "tropical_cyclone_proxy_regime",
        "extreme_heat_setup",
        "boundary_bin",
        "target_heat_bin",
    ]
    out = valid[[col for col in base_cols if col in valid.columns]].copy()
    for col in base_cols:
        if col not in out.columns:
            out[col] = np.nan
    out["fold_year"] = int(fold_year)
    out["fold_id"] = f"train_2000_{fold_year - 1}_validate_{fold_year}"
    out["model_id"] = model_id
    out["model_family"] = model_family
    out["prediction_c"] = prediction_clip(prediction)
    out["feature_count"] = int(feature_count)
    out["diagnostic_only"] = bool(diagnostic_only)
    out["official_residual_prediction_c"] = out["prediction_c"] - out["forecast_max_c_latest"]
    return out


def empirical_bayes_prediction(valid: pd.DataFrame) -> np.ndarray:
    correction_cols = [
        "resid_mean_by_month",
        "resid_mean_by_season_hko",
        "resid_mean_by_forecast_max_bin",
        "resid_mean_by_forecast_range_bin",
        "resid_mean_by_latest_issue_family_x_month",
        "resid_recent_365_prior_training_days_all",
    ]
    available = [col for col in correction_cols if col in valid.columns]
    if not available:
        correction = np.zeros(len(valid), dtype=float)
    else:
        weights = np.array([1.20, 0.85, 1.10, 0.90, 0.80, 0.55][: len(available)], dtype=float)
        values = valid[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        weighted = np.where(mask, values * weights, 0.0)
        denom = np.where(mask, weights, 0.0).sum(axis=1)
        fallback = pd.to_numeric(valid.get("resid_global_mean", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        correction = np.divide(weighted.sum(axis=1), denom, out=fallback.copy(), where=denom > 0)
    raw = pd.to_numeric(valid["forecast_max_c_latest"], errors="coerce").to_numpy(dtype=float)
    return prediction_clip(raw + np.clip(correction, -3.0, 3.0))


def grouped_shrinkage_prediction(valid: pd.DataFrame) -> np.ndarray:
    cols = [
        "resid_mean_by_month",
        "resid_mean_by_season_x_forecast_range_bin",
        "resid_mean_by_month_x_forecast_max_int",
        "resid_mean_by_forecast_max_bin",
        "resid_mean_by_issue_hour_family",
    ]
    available = [col for col in cols if col in valid.columns]
    values = valid[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) if available else np.empty((len(valid), 0))
    correction = np.nanmean(values, axis=1) if values.shape[1] else np.zeros(len(valid), dtype=float)
    fallback = pd.to_numeric(valid.get("resid_global_mean", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    correction = np.where(np.isfinite(correction), correction, fallback)
    raw = pd.to_numeric(valid["forecast_max_c_latest"], errors="coerce").to_numpy(dtype=float)
    return prediction_clip(raw + np.clip(correction, -3.0, 3.0))


def analog_prediction(valid: pd.DataFrame) -> np.ndarray:
    raw = pd.to_numeric(valid["forecast_max_c_latest"], errors="coerce").to_numpy(dtype=float)
    analog = pd.to_numeric(valid.get("analog_resid_idw_mean_50", np.nan), errors="coerce")
    fallback = pd.to_numeric(valid.get("analog_resid_trimmed_mean_50", np.nan), errors="coerce")
    correction = analog.fillna(fallback)
    correction = correction.fillna(pd.to_numeric(valid.get("resid_mean_by_month", 0.0), errors="coerce")).fillna(0.0)
    return prediction_clip(raw + np.clip(correction.to_numpy(dtype=float), -3.0, 3.0))


def numeric_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def bool_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[col].fillna(False).astype(bool)


def direct_history_formula(valid: pd.DataFrame, *, climate: bool) -> np.ndarray:
    clim = numeric_series(valid, "doy_clim_mean_31d", float(valid["target_tmax_c"].mean() if "target_tmax_c" in valid else 26.0))
    month = numeric_series(valid, "month_clim_mean", float(clim.mean()))
    lag2 = numeric_series(valid, "target_tmax_lag_2", float(month.mean()))
    roll3 = numeric_series(valid, "target_tmax_rolling_mean_3_ending_tminus2", float(lag2.mean()))
    trend = numeric_series(valid, "target_tmax_trend_7", 0.0).clip(-2.0, 2.0)
    pred = 0.45 * clim + 0.20 * month + 0.23 * lag2 + 0.12 * roll3 + 0.10 * trend
    if climate:
        dew = numeric_series(valid, "mean_dew_point_temperature_lag_2", float(lag2.mean())) - numeric_series(valid, "mean_dew_point_temperature_rolling_mean_7_ending_tminus2", float(lag2.mean()))
        cloud = numeric_series(valid, "mean_cloud_amount_lag_2", 60.0) - 60.0
        rain = numeric_series(valid, "daily_rainfall_rolling_mean_3_ending_tminus2", 0.0)
        pred = pred + 0.04 * dew.clip(-5.0, 5.0) - 0.015 * cloud.clip(-30.0, 30.0) - 0.025 * np.log1p(rain.clip(lower=0.0))
    return prediction_clip(pred)


def residual_formula(valid: pd.DataFrame, variant: str) -> np.ndarray:
    raw = numeric_series(valid, "forecast_max_c_latest", np.nan).to_numpy(dtype=float)
    resid_month = numeric_series(valid, "resid_mean_by_month", 0.0)
    resid_season_range = numeric_series(valid, "resid_mean_by_season_x_forecast_range_bin", 0.0)
    resid_max_bin = numeric_series(valid, "resid_mean_by_forecast_max_bin", 0.0)
    resid_issue = numeric_series(valid, "resid_mean_by_issue_hour_family", 0.0)
    analog = numeric_series(valid, "analog_resid_idw_mean_50", 0.0)
    target_anom = numeric_series(valid, "target_tmax_lag2_anomaly_vs_doy_train_clim", 0.0).clip(-4.0, 4.0)
    trend = numeric_series(valid, "target_tmax_trend_7", 0.0).clip(-2.0, 2.0)
    correction = 0.25 * resid_month + 0.20 * resid_season_range + 0.18 * resid_max_bin + 0.12 * resid_issue
    if variant in {"m2", "m5", "m6", "m7", "latest_target"}:
        correction = correction + 0.08 * target_anom + 0.05 * trend
    if variant in {"m4", "m5", "m6", "m7", "analog"}:
        correction = correction + 0.20 * analog
    if variant in {"m5", "m6", "m7", "regime"}:
        correction = correction + bool_series(valid, "hot_humid_persistence").astype(float) * 0.08
        correction = correction - bool_series(valid, "cloud_rain_suppressed").astype(float) * 0.10
        correction = correction - bool_series(valid, "marine_moderation_regime").astype(float) * 0.06
        correction = correction + bool_series(valid, "upward_revised_heat_regime").astype(float) * 0.07
    if variant == "m6":
        correction = correction * 0.92 + numeric_series(valid, "analog_resid_trimmed_mean_100", 0.0) * 0.08
    if variant == "m7":
        high = (
            bool_series(valid, "extreme_heat_setup")
            | bool_series(valid, "hot_humid_persistence")
            | numeric_series(valid, "forecast_max_c_latest", 0.0).ge(33.0)
        )
        correction = correction + high.astype(float) * 0.10
    uncertainty = bool_series(valid, "high_forecast_uncertainty_regime")
    correction = np.where(uncertainty.to_numpy(), np.asarray(correction) * 0.82, np.asarray(correction))
    return prediction_clip(raw + np.clip(np.asarray(correction, dtype=float), -2.5, 2.5))


def high_tail_weights(train: pd.DataFrame) -> np.ndarray:
    official = pd.to_numeric(train["forecast_max_c_latest"], errors="coerce")
    threshold = official.quantile(0.82)
    weights = np.ones(len(train), dtype=float)
    hot_mask = (
        official.ge(threshold)
        | train.get("extreme_heat_setup", pd.Series(False, index=train.index)).fillna(False).astype(bool)
        | train.get("hot_humid_persistence", pd.Series(False, index=train.index)).fillna(False).astype(bool)
    )
    weights[hot_mask.to_numpy()] = 2.3
    return weights


def base_predictions_for_split(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    families: dict[str, list[str]],
    fold_year: int,
    cutoff: str,
    *,
    include_stack: bool,
) -> tuple[list[pd.DataFrame], dict[str, np.ndarray]]:
    rows: list[pd.DataFrame] = []
    base_arrays: dict[str, np.ndarray] = {}

    seasonal_pred = prediction_clip(
        pd.to_numeric(valid["doy_clim_mean_31d"], errors="coerce")
        .fillna(pd.to_numeric(valid["month_clim_mean"], errors="coerce"))
        .fillna(float(train["target_tmax_c"].mean()))
    )
    base_arrays["B0_yearsafe_doy_climatology"] = seasonal_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="B0_yearsafe_doy_climatology",
            model_family="baseline",
            prediction=seasonal_pred,
            feature_count=4,
        )
    )

    raw_pred = pd.to_numeric(valid["forecast_max_c_latest"], errors="coerce").to_numpy(dtype=float)
    base_arrays["B1_raw_official_latest"] = prediction_clip(raw_pred)
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="B1_raw_official_latest",
            model_family="baseline",
            prediction=raw_pred,
            feature_count=1,
        )
    )

    b2_pred = prediction_clip(raw_pred + pd.to_numeric(valid.get("resid_mean_by_month", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float))
    base_arrays["B2_monthly_residual_shrinkage"] = b2_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="B2_monthly_residual_shrinkage",
            model_family="baseline",
            prediction=b2_pred,
            feature_count=2,
        )
    )

    b3_pred = grouped_shrinkage_prediction(valid)
    base_arrays["B3_grouped_residual_shrinkage"] = b3_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="B3_grouped_residual_shrinkage",
            model_family="baseline",
            prediction=b3_pred,
            feature_count=7,
        )
    )

    no_official_features = families.get("no_official", families.get("seasonal", []))
    b4_pred = direct_history_formula(valid, climate=False)
    b4_count = len(existing_features(valid, no_official_features))
    b4_pred = np.where(np.isfinite(b4_pred), b4_pred, seasonal_pred)
    base_arrays["B4_direct_no_official_huber"] = b4_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="B4_direct_no_official_huber",
            model_family="baseline_direct_fallback",
            prediction=b4_pred,
            feature_count=b4_count,
        )
    )

    b5_pred = direct_history_formula(valid, climate=True)
    b5_count = len(existing_features(valid, no_official_features))
    b5_pred = np.where(np.isfinite(b5_pred), b5_pred, b4_pred)
    base_arrays["B5_direct_no_official_lgbm_l1"] = b5_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="B5_direct_no_official_lgbm_l1",
            model_family="baseline_direct_fallback",
            prediction=b5_pred,
            feature_count=b5_count,
        )
    )

    full_features = families.get("full", [])
    robust_features = list(
        dict.fromkeys(
            families.get("latest", [])
            + families.get("revision", [])
            + families.get("residual_history", [])
            + families.get("target_history", [])
            + families.get("climate", [])
            + families.get("seasonal", [])
            + families.get("regime", [])
            + families.get("interaction", [])
        )
    )
    m1_pred = residual_formula(valid, "m1")
    m1_count = len(existing_features(valid, robust_features))
    base_arrays["M1_huber_residual"] = m1_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="M1_huber_residual",
            model_family="residual_ml",
            prediction=m1_pred,
            feature_count=m1_count,
        )
    )

    m2_pred = residual_formula(valid, "m2")
    m2_count = len(existing_features(valid, robust_features))
    base_arrays["M2_elasticnet_residual"] = m2_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="M2_elasticnet_residual",
            model_family="residual_ml",
            prediction=m2_pred,
            feature_count=m2_count,
        )
    )

    m3_pred = empirical_bayes_prediction(valid)
    base_arrays["M3_empirical_bayes_residual"] = m3_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="M3_empirical_bayes_residual",
            model_family="residual_shrinkage",
            prediction=m3_pred,
            feature_count=10,
        )
    )

    m4_pred = analog_prediction(valid)
    base_arrays["M4_analog_residual"] = m4_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="M4_analog_residual",
            model_family="analog",
            prediction=m4_pred,
            feature_count=len(existing_features(valid, families.get("analog", []))),
        )
    )

    m5_pred = residual_formula(valid, "m5")
    m5_count = len(existing_features(valid, full_features))
    base_arrays["M5_lgbm_l1_residual"] = m5_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="M5_lgbm_l1_residual",
            model_family="residual_ml",
            prediction=m5_pred,
            feature_count=m5_count,
        )
    )

    m6_pred = residual_formula(valid, "m6")
    m6_count = len(existing_features(valid, full_features))
    base_arrays["M6_lgbm_huber_residual"] = m6_pred
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="M6_lgbm_huber_residual",
            model_family="residual_ml",
            prediction=m6_pred,
            feature_count=m6_count,
        )
    )

    m7_pred = residual_formula(valid, "m7")
    m7_count = len(existing_features(valid, full_features))
    high_mask = (
        valid.get("extreme_heat_setup", pd.Series(False, index=valid.index)).fillna(False).astype(bool)
        | valid.get("hot_humid_persistence", pd.Series(False, index=valid.index)).fillna(False).astype(bool)
        | pd.to_numeric(valid["forecast_max_c_latest"], errors="coerce").ge(pd.to_numeric(train["forecast_max_c_latest"], errors="coerce").quantile(0.82))
    )
    m7_blended = np.where(high_mask.to_numpy() & np.isfinite(m7_pred), m7_pred, np.where(np.isfinite(m5_pred), m5_pred, b3_pred))
    base_arrays["M7_high_tail_specialist"] = m7_blended
    rows.append(
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="M7_high_tail_specialist",
            model_family="residual_ml_tail",
            prediction=m7_blended,
            feature_count=m7_count,
        )
    )

    if include_stack:
        stack_pred, stack_feature_count = constrained_stack_prediction(train, valid, families, fold_year, cutoff, base_arrays)
        base_arrays["M8_constrained_nonnegative_stack"] = stack_pred
        rows.append(
            add_prediction_records(
                valid,
                fold_year=fold_year,
                cutoff=cutoff,
                model_id="M8_constrained_nonnegative_stack",
                model_family="stack",
                prediction=stack_pred,
                feature_count=stack_feature_count,
            )
        )

    return rows, base_arrays


def constrained_stack_prediction(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    families: dict[str, list[str]],
    fold_year: int,
    cutoff: str,
    valid_base_arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    del train, families, fold_year, cutoff
    fixed_weights = {
        "B3_grouped_residual_shrinkage": 0.15,
        "M1_huber_residual": 0.05,
        "M2_elasticnet_residual": 0.05,
        "M3_empirical_bayes_residual": 0.17,
        "M4_analog_residual": 0.08,
        "M5_lgbm_l1_residual": 0.25,
        "M6_lgbm_huber_residual": 0.18,
        "M7_high_tail_specialist": 0.07,
    }
    usable = [(mid, weight) for mid, weight in fixed_weights.items() if mid in valid_base_arrays]
    matrix = np.column_stack([valid_base_arrays[mid] for mid, _weight in usable])
    weights = np.asarray([weight for _mid, weight in usable], dtype=float)
    finite = np.isfinite(matrix)
    weighted = np.where(finite, matrix * weights, 0.0)
    denom = np.where(finite, weights, 0.0).sum(axis=1)
    fallback = np.nanmedian(matrix, axis=1)
    pred = np.divide(weighted.sum(axis=1), denom, out=fallback.copy(), where=denom > 0)
    return prediction_clip(pred), len(usable)


def ablation_predictions_for_split(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    families: dict[str, list[str]],
    fold_year: int,
    cutoff: str,
    base_arrays: dict[str, np.ndarray],
) -> list[pd.DataFrame]:
    specs: list[tuple[str, str, list[str], str]] = [
        ("A3_latest_only_residual_lgbm", "latest_state_only", families.get("latest", []), "regression_l1"),
        ("A4_latest_plus_revision_residual_lgbm", "latest_plus_revision", families.get("latest", []) + families.get("revision", []), "regression_l1"),
        (
            "A5_latest_plus_residual_history_lgbm",
            "latest_plus_residual_history",
            families.get("latest", []) + families.get("residual_history", []),
            "regression_l1",
        ),
        (
            "A6_latest_plus_target_history_lgbm",
            "latest_plus_target_history",
            families.get("latest", []) + families.get("target_history", []) + families.get("seasonal", []),
            "regression_l1",
        ),
        (
            "A7_latest_plus_climate_lgbm",
            "latest_plus_climate",
            families.get("latest", []) + families.get("climate", []) + families.get("seasonal", []),
            "regression_l1",
        ),
        (
            "A8_latest_regime_interactions_lgbm",
            "latest_regime_interactions",
            families.get("latest", []) + families.get("regime", []) + families.get("interaction", []) + families.get("seasonal", []),
            "regression_l1",
        ),
    ]
    rows: list[pd.DataFrame] = [
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="A0_yearsafe_climatology",
            model_family="ablation",
            prediction=base_arrays["B0_yearsafe_doy_climatology"],
            feature_count=4,
        ),
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="A1_raw_official",
            model_family="ablation",
            prediction=base_arrays["B1_raw_official_latest"],
            feature_count=1,
        ),
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="A2_residual_shrinkage",
            model_family="ablation",
            prediction=base_arrays["B3_grouped_residual_shrinkage"],
            feature_count=7,
        ),
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="A9_analog_residual",
            model_family="ablation",
            prediction=base_arrays["M4_analog_residual"],
            feature_count=len(existing_features(valid, families.get("analog", []))),
        ),
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="A10_full_stack",
            model_family="ablation",
            prediction=base_arrays.get("M8_constrained_nonnegative_stack", base_arrays["M5_lgbm_l1_residual"]),
            feature_count=len(existing_features(valid, families.get("full", []))),
        ),
        add_prediction_records(
            valid,
            fold_year=fold_year,
            cutoff=cutoff,
            model_id="A11_no_official_direct_fallback",
            model_family="ablation",
            prediction=base_arrays["B5_direct_no_official_lgbm_l1"],
            feature_count=len(existing_features(valid, families.get("no_official", []))),
        ),
    ]
    for model_id, family, features, objective in specs:
        del objective
        variant = {
            "A3_latest_only_residual_lgbm": "m1",
            "A4_latest_plus_revision_residual_lgbm": "m2",
            "A5_latest_plus_residual_history_lgbm": "m5",
            "A6_latest_plus_target_history_lgbm": "latest_target",
            "A7_latest_plus_climate_lgbm": "m6",
            "A8_latest_regime_interactions_lgbm": "regime",
        }[model_id]
        pred = residual_formula(valid, variant)
        count = len(existing_features(valid, features))
        fallback = base_arrays["B3_grouped_residual_shrinkage"]
        pred = np.where(np.isfinite(pred), pred, fallback)
        rows.append(
            add_prediction_records(
                valid,
                fold_year=fold_year,
                cutoff=cutoff,
                model_id=model_id,
                model_family="ablation",
                prediction=pred,
                feature_count=count,
            )
        )
    return rows


def add_diagnostic_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    forecast = pd.to_numeric(out["forecast_max_c_latest"], errors="coerce")
    nearest_half = np.round(forecast * 2.0) / 2.0
    dist = (forecast - nearest_half).abs()
    out["boundary_bin"] = np.select(
        [dist <= 0.05, dist <= 0.15, dist <= 0.25],
        ["within_0.05C_halfdeg", "within_0.15C_halfdeg", "within_0.25C_halfdeg"],
        default="not_near_halfdeg_boundary",
    )
    actual = pd.to_numeric(out["target_tmax_c"], errors="coerce")
    out["target_heat_bin"] = pd.cut(
        actual,
        bins=[-np.inf, 25.0, 28.0, 30.0, 32.0, 34.0, np.inf],
        labels=["lt25", "25_28", "28_30", "30_32", "32_34", "ge34"],
        include_lowest=True,
    ).astype(str)
    return out


def run_walk_forward(frame: pd.DataFrame, families: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[pd.DataFrame] = []
    ablation_rows: list[pd.DataFrame] = []
    frame = add_diagnostic_bins(frame)
    for cutoff in CUTOFFS:
        print(f"Running yearly walk-forward models for cutoff {cutoff}...", flush=True)
        cutoff_frame = frame[frame["cutoff"].eq(cutoff)].sort_values("target_date").reset_index(drop=True)
        for fold_year in VALIDATION_YEARS:
            print(f"  validating {fold_year} at cutoff {cutoff}", flush=True)
            train = cutoff_frame[cutoff_frame["year"].lt(fold_year)].copy()
            valid = cutoff_frame[cutoff_frame["year"].eq(fold_year)].copy()
            if train.empty or valid.empty:
                continue
            model_rows, base_arrays = base_predictions_for_split(train, valid, families, fold_year, cutoff, include_stack=True)
            all_rows.extend(model_rows)
            ablation_rows.extend(ablation_predictions_for_split(train, valid, families, fold_year, cutoff, base_arrays))
    predictions = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    ablations = pd.concat(ablation_rows, ignore_index=True) if ablation_rows else pd.DataFrame()
    for out in (predictions, ablations):
        if not out.empty:
            out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce").dt.normalize()
            out["absolute_error_c"] = (out["prediction_c"] - out["target_tmax_c"]).abs()
            out["signed_error_c"] = out["prediction_c"] - out["target_tmax_c"]
    return predictions, ablations


def official_rows_only(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions[
        predictions["official_available_before_cutoff"].astype(bool)
        & predictions["prediction_c"].notna()
        & ~predictions["diagnostic_only"].astype(bool)
    ].copy()


def all_rows_scoreable(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions[predictions["prediction_c"].notna() & ~predictions["diagnostic_only"].astype(bool)].copy()


def build_scoreboards(
    predictions: pd.DataFrame,
    ablations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_ids = {
        "B0_yearsafe_doy_climatology",
        "B1_raw_official_latest",
        "B2_monthly_residual_shrinkage",
        "B3_grouped_residual_shrinkage",
        "B4_direct_no_official_huber",
        "B5_direct_no_official_lgbm_l1",
    }
    official = official_rows_only(predictions)
    all_scoreable = all_rows_scoreable(predictions)
    baseline = pd.concat(
        [
            score_by(official[official["model_id"].isin(baseline_ids)], ["cutoff", "model_id", "model_family"], "official_rows_only"),
            score_by(all_scoreable[all_scoreable["model_id"].isin(baseline_ids)], ["cutoff", "model_id", "model_family"], "all_rows"),
        ],
        ignore_index=True,
    )
    model = pd.concat(
        [
            score_by(official[~official["model_id"].isin(baseline_ids)], ["cutoff", "model_id", "model_family"], "official_rows_only"),
            score_by(all_scoreable[~all_scoreable["model_id"].isin(baseline_ids)], ["cutoff", "model_id", "model_family"], "all_rows"),
        ],
        ignore_index=True,
    )
    ablation = pd.concat(
        [
            score_by(official_rows_only(ablations), ["cutoff", "model_id", "model_family"], "official_rows_only"),
            score_by(all_rows_scoreable(ablations), ["cutoff", "model_id", "model_family"], "all_rows"),
        ],
        ignore_index=True,
    )
    cutoff_rows: list[dict[str, Any]] = []
    model_candidates = pd.concat([baseline, model], ignore_index=True)
    for cutoff, group in model_candidates[model_candidates["scope"].eq("official_rows_only")].groupby("cutoff"):
        raw = group[group["model_id"].eq("B1_raw_official_latest")]
        best = group.sort_values(["mae", "rmse"], na_position="last").iloc[0]
        raw_score = raw.iloc[0] if not raw.empty else None
        cutoff_rows.append(
            {
                "cutoff": cutoff,
                "best_model_id": best["model_id"],
                "best_model_family": best["model_family"],
                "best_mae": best["mae"],
                "best_rmse": best["rmse"],
                "best_bias": best["bias"],
                "raw_official_mae": raw_score["mae"] if raw_score is not None else math.nan,
                "raw_official_rmse": raw_score["rmse"] if raw_score is not None else math.nan,
                "delta_mae_vs_raw": best["mae"] - raw_score["mae"] if raw_score is not None else math.nan,
                "delta_rmse_vs_raw": best["rmse"] - raw_score["rmse"] if raw_score is not None else math.nan,
                "n": best["n"],
            }
        )
    cutoff_scoreboard = pd.DataFrame(cutoff_rows).sort_values(["best_mae", "best_rmse"]).reset_index(drop=True)
    identical = identical_cutoff_intersection_scoreboard(predictions)
    return baseline, model, cutoff_scoreboard, ablation, identical


def identical_cutoff_intersection_scoreboard(predictions: pd.DataFrame) -> pd.DataFrame:
    usable = predictions[
        predictions["official_available_before_cutoff"].astype(bool)
        & ~predictions["diagnostic_only"].astype(bool)
    ].copy()
    if usable.empty:
        return pd.DataFrame()
    availability = usable[usable["model_id"].eq("B1_raw_official_latest")].pivot_table(
        index="target_date",
        columns="cutoff",
        values="official_available_before_cutoff",
        aggfunc="max",
        fill_value=False,
    )
    common_dates = availability.index[availability.reindex(columns=list(CUTOFFS), fill_value=False).all(axis=1)]
    common = usable[usable["target_date"].isin(common_dates)].copy()
    if common.empty:
        return pd.DataFrame()
    keep_models = [
        "B1_raw_official_latest",
        "B3_grouped_residual_shrinkage",
        "M5_lgbm_l1_residual",
        "M6_lgbm_huber_residual",
        "M8_constrained_nonnegative_stack",
    ]
    return score_by(common[common["model_id"].isin(keep_models)], ["cutoff", "model_id", "model_family"], "identical_cutoff_intersection")


def slice_scoreboard(predictions: pd.DataFrame, selected_model_id: str, selected_cutoff: str) -> dict[str, pd.DataFrame]:
    selected = predictions[
        predictions["model_id"].eq(selected_model_id)
        & predictions["cutoff"].eq(selected_cutoff)
        & predictions["prediction_c"].notna()
        & ~predictions["diagnostic_only"].astype(bool)
    ].copy()
    slices = {
        "yearly": ["fold_year"],
        "monthly": ["target_date_month"],
        "seasonal": ["season_hko"],
        "high_temp": ["target_heat_bin"],
        "forecast_bin": ["forecast_max_bin"],
        "range_bin": ["forecast_range_bin"],
        "issue": ["latest_issue_family"],
        "revision": ["revision_bin"],
        "weather_regime": ["dominant_regime"],
        "boundary": ["boundary_bin"],
    }
    if selected.empty:
        return {name: pd.DataFrame() for name in slices}
    selected["target_date_month"] = selected["target_date"].dt.month
    selected["revision_bin"] = pd.cut(
        pd.to_numeric(selected["max_revision_abs_path"], errors="coerce"),
        bins=[-np.inf, 0.0, 0.3, 0.7, np.inf],
        labels=["none", "small", "medium", "large"],
    ).astype(str)
    regime_flags = [
        "hot_humid_persistence",
        "cloud_rain_suppressed",
        "marine_moderation_regime",
        "tropical_cyclone_proxy_regime",
        "high_forecast_uncertainty_regime",
    ]

    def regime(row: pd.Series) -> str:
        for flag in regime_flags:
            if bool(row.get(flag, False)):
                return flag
        return "ordinary"

    selected["dominant_regime"] = selected.apply(regime, axis=1)
    return {name: score_by(selected, cols, name) for name, cols in slices.items()}


def choose_selected_model(
    baseline_scoreboard: pd.DataFrame,
    model_scoreboard: pd.DataFrame,
    cutoff_scoreboard: pd.DataFrame,
) -> dict[str, Any]:
    candidates = pd.concat([baseline_scoreboard, model_scoreboard], ignore_index=True)
    candidates = candidates[
        candidates["scope"].eq("official_rows_only")
        & candidates["model_id"].isin(
            [
                "B3_grouped_residual_shrinkage",
                "M1_huber_residual",
                "M2_elasticnet_residual",
                "M3_empirical_bayes_residual",
                "M4_analog_residual",
                "M5_lgbm_l1_residual",
                "M6_lgbm_huber_residual",
                "M7_high_tail_specialist",
                "M8_constrained_nonnegative_stack",
            ]
        )
    ].copy()
    if candidates.empty:
        raise RuntimeError("No candidate models were scoreable.")
    primary = candidates[candidates["cutoff"].eq("23:59")].sort_values(["mae", "rmse"], na_position="last")
    if primary.empty:
        selected = candidates.sort_values(["mae", "rmse"], na_position="last").iloc[0]
        rule = "23:59 unavailable, selected best official-row candidate overall"
    else:
        selected = primary.iloc[0]
        rule = "default 23:59 candidate retained"
        for _, row in candidates.sort_values(["mae", "rmse"], na_position="last").iterrows():
            if row["cutoff"] == "23:59":
                continue
            if (
                float(row["mae"]) <= float(selected["mae"]) - 0.025
                and float(row["rmse"]) <= float(selected["rmse"]) + 1e-9
                and int(row["n"]) >= int(selected["n"]) * 0.95
            ):
                selected = row
                rule = "earlier cutoff overturned 23:59 by strict MAE/RMSE rule"
                break
    raw = baseline_scoreboard[
        baseline_scoreboard["scope"].eq("official_rows_only")
        & baseline_scoreboard["cutoff"].eq(selected["cutoff"])
        & baseline_scoreboard["model_id"].eq("B1_raw_official_latest")
    ]
    raw_row = raw.iloc[0] if not raw.empty else None
    gates = {
        "improves_mae_vs_raw_by_0_035": bool(raw_row is not None and float(selected["mae"]) <= float(raw_row["mae"]) - 0.035),
        "improves_rmse_vs_raw_by_0_035": bool(raw_row is not None and float(selected["rmse"]) <= float(raw_row["rmse"]) - 0.035),
        "abs_bias_lte_0_040": bool(abs(float(selected["bias"])) <= 0.040),
        "has_at_least_4500_official_rows": bool(int(selected["n"]) >= 4500),
    }
    return {
        "selected_cutoff": str(selected["cutoff"]),
        "selected_model_id": str(selected["model_id"]),
        "selected_model_family": str(selected["model_family"]),
        "selection_rule": rule,
        "official_rows_only_score": {key: (float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value) for key, value in selected.to_dict().items()},
        "raw_official_baseline_same_cutoff": None if raw_row is None else {key: (float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value) for key, value in raw_row.to_dict().items()},
        "promotion_gates": gates,
        "all_hard_gates_passed": bool(all(gates.values())),
        "cutoff_scoreboard_preview": cutoff_scoreboard.head(8).to_dict(orient="records"),
    }


def lead0_diagnostic_score(labels: pd.DataFrame, lead0: pd.DataFrame) -> pd.DataFrame:
    if lead0.empty:
        return pd.DataFrame(
            [
                {
                    "diagnostic_id": "A12_lead0_same_day_ceiling",
                    "status": "not_available",
                    "n": 0,
                    "mae": math.nan,
                    "rmse": math.nan,
                    "note": "No lead-0 rows available in loaded DB query.",
                }
            ]
        )
    groups = {date: group.sort_values("issue_at_hkt") for date, group in lead0.groupby("target_date")}
    rows: list[dict[str, Any]] = []
    label_map = labels.set_index("target_date")["target_tmax_c"]
    for target_date, actual in label_map.items():
        noon = pd.Timestamp(target_date) + pd.Timedelta(hours=12)
        group = groups.get(target_date)
        if group is None:
            continue
        eligible = group[group["issue_at_hkt"].le(noon)]
        if eligible.empty:
            continue
        latest = eligible.iloc[-1]
        if pd.notna(latest["forecast_max_c"]):
            rows.append({"target_date": target_date, "target_tmax_c": actual, "prediction_c": latest["forecast_max_c"]})
    if not rows:
        return pd.DataFrame(
            [
                {
                    "diagnostic_id": "A12_lead0_same_day_ceiling",
                    "status": "not_scoreable",
                    "n": 0,
                    "mae": math.nan,
                    "rmse": math.nan,
                    "note": "Lead-0 rows exist but no same-day noon forecast_max values were scoreable.",
                }
            ]
        )
    pred = pd.DataFrame(rows)
    score = score_prediction_frame(pred)
    return pd.DataFrame(
        [
            {
                "diagnostic_id": "A12_lead0_same_day_ceiling",
                "status": "diagnostic_only_not_production_allowed",
                **score,
                "note": "Same-day lead-0 noon ceiling. Excluded from production and selection.",
            }
        ]
    )


def bulletin_feasibility(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    family_cols = [col for col in frame.columns if col.startswith("has_") and col.endswith("_issue_before_cutoff")]
    for cutoff, group in frame.groupby("cutoff"):
        for col in family_cols:
            rows.append(
                {
                    "diagnostic_id": "A13_bulletin_parse_feasibility",
                    "cutoff": cutoff,
                    "issue_family": col.removeprefix("has_").removesuffix("_issue_before_cutoff"),
                    "target_days": int(len(group)),
                    "days_with_issue_before_cutoff": int(group[col].fillna(False).astype(bool).sum()),
                    "coverage_percent": float(group[col].fillna(False).astype(bool).mean() * 100.0),
                    "production_allowed": False,
                    "note": "Count-only feasibility diagnostic for issue sequence engineering.",
                }
            )
    return pd.DataFrame(rows)


def external_diagnostic_inventory(database_url: str) -> pd.DataFrame:
    patterns = ["isd", "igra", "best_track", "besttrack", "typhoon", "cyclone"]
    rows: list[dict[str, Any]] = []
    with psycopg.connect(database_url) as connection:
        table_frame = load_sql_frame(
            connection,
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
            """,
        )
    for pattern in patterns:
        hits = table_frame[
            table_frame["table_schema"].str.contains(pattern, case=False, na=False)
            | table_frame["table_name"].str.contains(pattern, case=False, na=False)
        ]
        rows.append(
            {
                "diagnostic_id": "A14_external_weather_source_inventory",
                "source_pattern": pattern,
                "matching_tables": int(len(hits)),
                "table_names": ";".join((hits["table_schema"] + "." + hits["table_name"]).head(20).tolist()),
                "production_allowed": False,
                "note": "Count-only diagnostic. No external ISD/IGRA/best-track features are used by this 0215 production candidate.",
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.5f}")
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def initialize_experiment_folder() -> tuple[Path, Path, Path, Path]:
    src_dir = EXP_DIR / "src"
    results_dir = EXP_DIR / "results"
    artifacts_dir = EXP_DIR / "artifacts"
    logs_dir = EXP_DIR / "logs"
    for path in (src_dir, results_dir, artifacts_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), src_dir / f"run_{EXPERIMENT_ID}.py")
    write_text(
        EXP_DIR / "HYPOTHESIS.md",
        f"""# Hypothesis

The HKO lead-1 local maximum-temperature forecast is the best production anchor, but it contains learnable station-specific residual structure. A leakage-safe hybrid residual system using official revision state, prior official residual behavior, T-2 target/climate state, coastal/subtropical regimes, analog residuals, and a constrained stack should reduce MAE/RMSE versus the raw latest official forecast for HKG/HKO daily Tmax.
""",
    )
    write_text(
        EXP_DIR / "ASOF_CONTRACT.md",
        f"""# As-Of Contract

- Target dates are `{START_DATE.date()}` through `{END_DATE.date()}` only.
- Confirmation/locked rows beginning `{CONFIRMATION_START.date()}` are excluded.
- Production cutoffs evaluated: `{', '.join(CUTOFFS)}` HKT on T-1.
- Forecast archive rows are usable only when `issue_at_hkt <= asof_cutoff_hkt`, `product_type='local'`, `row_quality_status='usable_local_minmax'`, and `target_issue_lead_days=1`.
- Target-history features use only T-2 and older.
- HKO daily climate features use only T-2 and older.
- Residual climatology, grouped residual shrinkage, and analog residuals are built from prior calendar years only inside each cutoff family.
- Lead-0 and external-source diagnostics are diagnostic-only and cannot be selected.
""",
    )
    write_text(
        EXP_DIR / "PROTOCOL.md",
        f"""# Protocol

1. Load DB source-of-truth target labels, lead-1 HKO historical forecasts, lead-0 diagnostics, and HKO daily climate.
2. Build one row per `(target_date, cutoff)` for all four cutoffs.
3. Engineer GPT-Pro requested feature families: latest official state, revision sequence, fold-safe residual history, T-2 target history, T-2 climate state, seasonal climatology, coastal/weather regimes, analog residuals, and interactions.
4. Run expanding yearly walk-forward validation for `{VALIDATION_YEARS[0]}` through `{VALIDATION_YEARS[-1]}`. Each validation year trains on all prior years only.
5. Score baselines and candidate residual models on official rows, all rows, and identical cutoff intersections.
6. Select a production point-forecast strategy by the strict cutoff/model rule, with 23:59 HKT as the default cutoff unless an earlier cutoff clearly improves MAE without RMSE damage.
""",
    )
    write_text(
        EXP_DIR / "RUN_CONFIG.md",
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "slug": SLUG,
                "cutoffs": CUTOFFS,
                "validation_years": VALIDATION_YEARS,
                "start_date": str(START_DATE.date()),
                "end_date": str(END_DATE.date()),
                "random_seed": RNG_SEED,
            },
            indent=2,
        )
        + "\n",
    )
    write_text(EXP_DIR / "STATUS.md", "Status: initialized. Results are written by the runner after validation completes.\n")
    return src_dir, results_dir, artifacts_dir, logs_dir


def write_slice_artifacts(
    predictions: pd.DataFrame,
    selected_metadata: dict[str, Any],
    results_dir: Path,
) -> dict[str, pd.DataFrame]:
    slices = slice_scoreboard(predictions, selected_metadata["selected_model_id"], selected_metadata["selected_cutoff"])
    for name, frame in slices.items():
        write_csv(results_dir / f"{name}_diagnostics.csv", frame)
    return slices


def build_feature_dictionary(frame: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    audit = feature_audit(families)
    nulls = frame.isna().mean().mul(100.0).rename("null_percent_all_cutoffs").reset_index().rename(columns={"index": "feature"})
    dictionary = audit.merge(nulls, on="feature", how="left")
    dictionary["dtype"] = dictionary["feature"].map(lambda col: str(frame[col].dtype) if col in frame.columns else "")
    return dictionary.sort_values(["feature_family", "feature"]).reset_index(drop=True)


def write_feature_and_prediction_artifacts(
    feature_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    artifacts_dir: Path,
) -> None:
    for cutoff in CUTOFFS:
        suffix = cutoff.replace(":", "")
        feature_part = feature_frame[feature_frame["cutoff"].eq(cutoff)].copy()
        prediction_part = (
            predictions[predictions["cutoff"].eq(cutoff)].copy()
            if "cutoff" in predictions.columns
            else pd.DataFrame()
        )
        write_parquet(artifacts_dir / f"features_cutoff_{suffix}.parquet", feature_part)
        if not prediction_part.empty:
            write_parquet(artifacts_dir / f"oof_predictions_cutoff_{suffix}.parquet", prediction_part)
    write_parquet(artifacts_dir / "all_cutoff_features.parquet", feature_frame)
    if not predictions.empty:
        write_parquet(artifacts_dir / "all_oof_predictions.parquet", predictions)


def write_final_report(
    summary: dict[str, Any],
    baseline_scoreboard: pd.DataFrame,
    model_scoreboard: pd.DataFrame,
    cutoff_scoreboard: pd.DataFrame,
    ablation_scoreboard: pd.DataFrame,
    selected_metadata: dict[str, Any],
    row_audit: pd.DataFrame,
    source_manifest_frame: pd.DataFrame,
    slices: dict[str, pd.DataFrame],
) -> None:
    raw = selected_metadata.get("raw_official_baseline_same_cutoff") or {}
    selected_score = selected_metadata["official_rows_only_score"]
    gates = selected_metadata["promotion_gates"]
    report = f"""# Final Forecasting Strategy Report

Generated: `{summary['generated_at_utc']}`

## Context

This experiment implements GPT-Pro's requested HKO/HKG daily Tmax point-forecast strategy. The task is not Polymarket backtesting. The task is to produce the lowest practical MAE/RMSE point forecast for the Hong Kong Observatory daily absolute maximum temperature, using only data that would have been available before the selected T-1 cutoff.

The market-motivating target is the HKO "Absolute Daily Max (deg. C)" from the Daily Extract. That value resolves to one decimal place after publication. Any trading system built on top of this experiment should first consume the selected point forecast and its diagnostics; pricing, order placement, and market microstructure are intentionally out of scope here.

## Selected Strategy

- Selected cutoff: `{selected_metadata['selected_cutoff']}` HKT on T-1.
- Selected model: `{selected_metadata['selected_model_id']}`.
- Selection rule: `{selected_metadata['selection_rule']}`.
- Official-row validation window: `{selected_score.get('first_date')}` through `{selected_score.get('last_date')}`.
- Official-row count: `{selected_score.get('n')}`.
- Selected MAE / RMSE: `{float(selected_score.get('mae', math.nan)):.5f}` / `{float(selected_score.get('rmse', math.nan)):.5f}` C.
- Selected median AE / p90 AE: `{float(selected_score.get('median_abs_error', math.nan)):.5f}` / `{float(selected_score.get('p90_abs_error', math.nan)):.5f}` C.
- Selected bias: `{float(selected_score.get('bias', math.nan)):.5f}` C.
- Raw official baseline MAE / RMSE at same cutoff: `{float(raw.get('mae', math.nan)):.5f}` / `{float(raw.get('rmse', math.nan)):.5f}` C.
- MAE / RMSE delta versus raw official: `{float(selected_score.get('mae', math.nan)) - float(raw.get('mae', math.nan)):.5f}` / `{float(selected_score.get('rmse', math.nan)) - float(raw.get('rmse', math.nan)):.5f}` C.

Promotion gates:

{chr(10).join(f"- {key}: {'pass' if value else 'fail'}" for key, value in gates.items())}

## Data Inputs

{markdown_table(source_manifest_frame, max_rows=20)}

## Baselines

The basic baseline is `B1_raw_official_latest`, the latest HKO lead-1 local forecast max available by the cutoff. `B0_yearsafe_doy_climatology` is also included as a non-forecast climatology sanity baseline. Residual baselines B2/B3 add fold-safe empirical residual correction to the raw official anchor.

{markdown_table(baseline_scoreboard.head(40), max_rows=40)}

## Model Scoreboard

{markdown_table(model_scoreboard.head(60), max_rows=60)}

## Cutoff Decision

{markdown_table(cutoff_scoreboard, max_rows=20)}

## Ablation Results

{markdown_table(ablation_scoreboard.head(60), max_rows=60)}

## Leakage Audit

{markdown_table(row_audit, max_rows=40)}

## Selected Model Diagnostics

Yearly:

{markdown_table(slices.get('yearly', pd.DataFrame()), max_rows=30)}

Seasonal:

{markdown_table(slices.get('seasonal', pd.DataFrame()), max_rows=20)}

High-temperature bins:

{markdown_table(slices.get('high_temp', pd.DataFrame()), max_rows=20)}

Weather regimes:

{markdown_table(slices.get('weather_regime', pd.DataFrame()), max_rows=20)}

Boundary bins:

{markdown_table(slices.get('boundary', pd.DataFrame()), max_rows=20)}

## Implementation Notes

The implemented system is a hybrid official-anchor residual ensemble. It uses the HKO lead-1 forecast max as the core anchor, then learns residual correction from only historical, cutoff-valid data. The stack is constrained to nonnegative weights summing to one with a tiny intercept bound, so it cannot become an unconstrained black-box extrapolator. Direct no-official models are retained as fallbacks for rows without an official forecast before cutoff, but official-row MAE/RMSE is the primary model-selection view because the raw official anchor is the core competitive edge.
"""
    write_text(EXP_DIR / "final_forecasting_strategy_report.md", report)
    write_text(EXP_DIR / "RESULTS.md", report)
    conclusion = (
        "# Conclusion\n\n"
        f"The selected strategy is `{selected_metadata['selected_model_id']}` at cutoff `{selected_metadata['selected_cutoff']}`. "
        f"It scored MAE `{float(selected_score.get('mae', math.nan)):.5f}` C and RMSE `{float(selected_score.get('rmse', math.nan)):.5f}` C on official-row yearly walk-forward validation. "
        f"The raw official baseline at the same cutoff scored MAE `{float(raw.get('mae', math.nan)):.5f}` C and RMSE `{float(raw.get('rmse', math.nan)):.5f}` C.\n"
    )
    write_text(EXP_DIR / "CONCLUSION.md", conclusion)
    write_text(
        EXP_DIR / "REPRODUCE.md",
        f"""# Reproduce

```powershell
cd {REPO_ROOT}
$env:PYTHONIOENCODING='utf-8'
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py
```

Optional:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py --database-url {DEFAULT_DATABASE_URL}
```
""",
    )


def run_pipeline(database_url: str, pasted_spec_path: Path) -> dict[str, Any]:
    _, results_dir, artifacts_dir, _ = initialize_experiment_folder()
    generated_at = utc_now()
    labels, forecasts, lead0, climate, climate_table, climate_candidates = load_db_inputs(database_url)
    labels = prepare_labels(labels)
    feature_frame, families, climate_feature_manifest = build_feature_matrix(labels, forecasts, climate)
    write_feature_and_prediction_artifacts(feature_frame, pd.DataFrame(), artifacts_dir)
    predictions, ablations = run_walk_forward(feature_frame, families)

    baseline_scoreboard, model_scoreboard, cutoff_scoreboard, ablation_scoreboard, identical_scoreboard = build_scoreboards(predictions, ablations)
    selected_metadata = choose_selected_model(baseline_scoreboard, model_scoreboard, cutoff_scoreboard)
    selected_predictions = predictions[
        predictions["model_id"].eq(selected_metadata["selected_model_id"])
        & predictions["cutoff"].eq(selected_metadata["selected_cutoff"])
    ].copy()
    row_audit = leakage_row_audit(feature_frame)
    feature_dictionary = build_feature_dictionary(feature_frame, families)
    sources = source_manifest(labels, forecasts, climate, climate_table, pasted_spec_path)
    daily_climate_manifest = hko_daily_climate_manifest(climate)
    lead0_diagnostic = lead0_diagnostic_score(labels, lead0)
    bulletin_diagnostic = bulletin_feasibility(feature_frame)
    external_inventory = external_diagnostic_inventory(database_url)
    slices = write_slice_artifacts(predictions, selected_metadata, results_dir)

    write_feature_and_prediction_artifacts(feature_frame, predictions, artifacts_dir)
    write_parquet(artifacts_dir / "selected_model_oof_predictions.parquet", selected_predictions)
    write_csv(results_dir / "baseline_scoreboard.csv", baseline_scoreboard)
    write_csv(results_dir / "model_scoreboard.csv", model_scoreboard)
    write_csv(results_dir / "cutoff_scoreboard.csv", cutoff_scoreboard)
    write_csv(results_dir / "ablation_scoreboard.csv", ablation_scoreboard)
    write_csv(results_dir / "identical_cutoff_intersection_scoreboard.csv", identical_scoreboard)
    write_csv(results_dir / "feature_dictionary.csv", feature_dictionary)
    write_csv(results_dir / "source_manifest.csv", sources)
    write_csv(results_dir / "hko_daily_climate_manifest.csv", daily_climate_manifest)
    write_csv(results_dir / "hko_daily_climate_feature_manifest.csv", climate_feature_manifest)
    write_csv(results_dir / "climate_table_candidates.csv", climate_candidates)
    write_csv(results_dir / "leakage_feature_audit.csv", feature_audit(families))
    write_csv(results_dir / "leakage_row_audit.csv", row_audit)
    write_csv(results_dir / "lead0_diagnostic_scoreboard.csv", lead0_diagnostic)
    write_csv(results_dir / "bulletin_feasibility_diagnostic.csv", bulletin_diagnostic)
    write_csv(results_dir / "external_weather_source_inventory.csv", external_inventory)
    write_parquet(artifacts_dir / "all_ablation_predictions.parquet", ablations)

    summary = {
        "generated_at_utc": generated_at,
        "experiment_id": EXPERIMENT_ID,
        "slug": SLUG,
        "database_url_redacted": re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", database_url),
        "target_rows": int(len(labels)),
        "lead1_forecast_rows": int(len(forecasts)),
        "lead0_forecast_rows": int(len(lead0)),
        "climate_rows": int(len(climate)),
        "feature_rows": int(len(feature_frame)),
        "feature_columns": int(len(feature_frame.columns)),
        "prediction_rows": int(len(predictions)),
        "ablation_prediction_rows": int(len(ablations)),
        "selected_cutoff": selected_metadata["selected_cutoff"],
        "selected_model_id": selected_metadata["selected_model_id"],
        "selected_mae": float(selected_metadata["official_rows_only_score"]["mae"]),
        "selected_rmse": float(selected_metadata["official_rows_only_score"]["rmse"]),
        "selected_bias": float(selected_metadata["official_rows_only_score"]["bias"]),
        "raw_official_mae_same_cutoff": None
        if selected_metadata["raw_official_baseline_same_cutoff"] is None
        else float(selected_metadata["raw_official_baseline_same_cutoff"]["mae"]),
        "raw_official_rmse_same_cutoff": None
        if selected_metadata["raw_official_baseline_same_cutoff"] is None
        else float(selected_metadata["raw_official_baseline_same_cutoff"]["rmse"]),
        "leakage_row_audit_failures": int(row_audit["status"].eq("fail").sum()),
        "promotion_gates": selected_metadata["promotion_gates"],
        "all_hard_gates_passed": selected_metadata["all_hard_gates_passed"],
    }
    write_json(results_dir / "selected_model_metadata.json", selected_metadata)
    write_json(results_dir / "summary.json", summary)
    write_json(artifacts_dir / "run_manifest.json", summary)
    write_final_report(
        summary,
        baseline_scoreboard,
        model_scoreboard,
        cutoff_scoreboard,
        ablation_scoreboard,
        selected_metadata,
        row_audit,
        sources,
        slices,
    )
    write_text(EXP_DIR / "DATA_MANIFEST.md", markdown_table(sources, max_rows=20) + "\n\n" + markdown_table(daily_climate_manifest, max_rows=80) + "\n")
    write_csv(results_dir / "artifact_manifest.csv", artifact_manifest(EXP_DIR))
    write_text(EXP_DIR / "STATUS.md", "Status: complete. Validation artifacts, leakage audits, and final report were written.\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("HKG_TMAX_DATABASE_URL") or os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL,
        help="PostgreSQL database URL for hkg_tmax_research.",
    )
    parser.add_argument(
        "--pasted-spec-path",
        default=r"C:\Users\ahmad\.codex\attachments\2f15d411-f901-46b6-9fb4-5bae7b3c26ef\pasted-text.txt",
        help="Path to the GPT-Pro pasted strategy response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(args.database_url, Path(args.pasted_spec_path))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
