"""Info.gov local-forecast anchor and revision-path feature builder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from hkg_tmax.features.feature_registry import FeatureRegistry
from hkg_tmax.features.text_regime_flags import forecast_text_flags

HKT = ZoneInfo("Asia/Hong_Kong")


@dataclass(frozen=True)
class CutoffProfile:
    name: str
    hkt_time: time


CUTOFF_PROFILES = (
    CutoffProfile("tminus1_1500", time(15, 0)),
    CutoffProfile("tminus1_1800", time(18, 0)),
    CutoffProfile("tminus1_2100", time(21, 0)),
    CutoffProfile("tminus1_2359", time(23, 59)),
)


TARGET_SQL = """
SELECT local_date::date AS target_date,
       target_tmax_c::double precision AS y_true_c,
       'label_core'::text AS label_source
FROM label_core.hko_daily_tmax
WHERE local_date BETWEEN %(start_date)s AND %(presealed_end_date)s
  AND quality_status = 'VALID'
  AND target_tmax_c IS NOT NULL
UNION ALL
SELECT local_date::date AS target_date,
       target_tmax_c::double precision AS y_true_c,
       'sealed_confirmation'::text AS label_source
FROM sealed_confirmation.hko_daily_tmax
WHERE local_date BETWEEN %(sealed_start_date)s AND %(sealed_end_date)s
  AND target_tmax_c IS NOT NULL
ORDER BY target_date
"""

STRICT_FORECAST_SQL = """
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
  target_date::date AS target_date,
  target_issue_lead_days,
  forecast_min_c::double precision AS forecast_min_c,
  forecast_max_c::double precision AS forecast_max_c,
  forecast_range_c::double precision AS forecast_range_c,
  forecast_midpoint_c::double precision AS forecast_midpoint_c,
  row_quality_status,
  temperature_text,
  parse_status,
  parse_notes,
  full_text,
  raw_sha256,
  raw_path,
  ingested_at_utc
FROM public.hko_historical_forecasts_2000_2026
WHERE source = 'info_gov'
  AND product_type = 'local'
  AND row_quality_status = 'usable_local_minmax'
  AND target_issue_lead_days = 1
  AND target_date BETWEEN %(start_date)s AND %(sealed_end_date)s
  AND forecast_max_c IS NOT NULL
  AND forecast_min_c IS NOT NULL
  AND issue_at_utc IS NOT NULL
  AND target_date IS NOT NULL
ORDER BY target_date, issue_at_utc, source_url
"""


def load_sql_frame(connection: Any, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        columns = [desc.name for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def load_targets(connection: Any, params: dict[str, Any]) -> pd.DataFrame:
    targets = load_sql_frame(connection, TARGET_SQL, params)
    targets["target_date"] = pd.to_datetime(targets["target_date"], errors="coerce").dt.normalize()
    return targets.sort_values("target_date").drop_duplicates("target_date", keep="last").reset_index(drop=True)


def load_strict_forecasts(connection: Any, params: dict[str, Any]) -> pd.DataFrame:
    forecasts = load_sql_frame(connection, STRICT_FORECAST_SQL, params)
    if forecasts.empty:
        return forecasts
    forecasts["target_date"] = pd.to_datetime(forecasts["target_date"], errors="coerce").dt.normalize()
    forecasts["issue_at_hkt"] = pd.to_datetime(forecasts["issue_at_hkt"], errors="coerce")
    forecasts["issue_at_utc"] = pd.to_datetime(forecasts["issue_at_utc"], errors="coerce", utc=True)
    forecasts["forecast_max_c"] = pd.to_numeric(forecasts["forecast_max_c"], errors="coerce")
    forecasts["forecast_min_c"] = pd.to_numeric(forecasts["forecast_min_c"], errors="coerce")
    forecasts["forecast_range_c"] = forecasts["forecast_max_c"] - forecasts["forecast_min_c"]
    forecasts["forecast_midpoint_c"] = (
        pd.to_numeric(forecasts["forecast_midpoint_c"], errors="coerce")
        .fillna((forecasts["forecast_max_c"] + forecasts["forecast_min_c"]) / 2.0)
    )
    return forecasts.dropna(subset=["target_date", "issue_at_utc"]).sort_values(
        ["target_date", "issue_at_utc", "source_url"]
    ).reset_index(drop=True)


def cutoff_timestamps(target_date: pd.Timestamp, profile: CutoffProfile) -> tuple[pd.Timestamp, pd.Timestamp]:
    local_date = (pd.Timestamp(target_date).normalize() - pd.Timedelta(days=1)).date()
    local = pd.Timestamp.combine(local_date, profile.hkt_time).tz_localize(HKT)
    return local, local.tz_convert("UTC")


def build_cutoff_frame(targets: pd.DataFrame, profiles: tuple[CutoffProfile, ...] = CUTOFF_PROFILES) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in targets.itertuples(index=False):
        target_date = pd.Timestamp(record.target_date).normalize()
        for profile in profiles:
            cutoff_hkt, cutoff_utc = cutoff_timestamps(target_date, profile)
            rows.append(
                {
                    "target_date": target_date,
                    "y_true_c": float(record.y_true_c),
                    "label_source": str(record.label_source),
                    "cutoff_profile": profile.name,
                    "cutoff_at_hkt": cutoff_hkt,
                    "cutoff_at_utc": cutoff_utc,
                    "split": split_for_date(target_date),
                    "month": int(target_date.month),
                    "quarter": int((target_date.month - 1) // 3 + 1),
                    "day_of_year": int(target_date.dayofyear),
                    "year": int(target_date.year),
                }
            )
    frame = pd.DataFrame(rows)
    frame["doy_sin"] = np.sin(2.0 * np.pi * frame["day_of_year"] / 366.0)
    frame["doy_cos"] = np.cos(2.0 * np.pi * frame["day_of_year"] / 366.0)
    frame["warm_season_flag"] = frame["month"].isin([4, 5, 6, 7, 8, 9, 10]).astype(int)
    frame["cool_season_flag"] = frame["month"].isin([11, 12, 1, 2, 3]).astype(int)
    frame["hot_season_flag"] = frame["month"].isin([6, 7, 8, 9]).astype(int)
    frame["typhoon_season_flag"] = frame["month"].isin([6, 7, 8, 9, 10]).astype(int)
    frame["shoulder_season_flag"] = frame["month"].isin([3, 4, 5, 10, 11]).astype(int)
    frame["trend_years_since_2000"] = frame["year"] + (frame["day_of_year"] - 1) / 366.0 - 2000.0
    frame["post_2010_flag"] = (frame["year"] >= 2010).astype(int)
    frame["post_2018_flag"] = (frame["year"] >= 2018).astype(int)
    frame["season_bucket"] = frame["month"].map(season_bucket)
    return frame.sort_values(["cutoff_profile", "target_date"]).reset_index(drop=True)


def split_for_date(target_date: pd.Timestamp) -> str:
    if target_date <= pd.Timestamp("2010-12-31"):
        return "train_base_2000_2010"
    if target_date <= pd.Timestamp("2013-12-31"):
        return "rolling_valid_2011_2013"
    if target_date <= pd.Timestamp("2016-12-31"):
        return "rolling_valid_2014_2016"
    if target_date <= pd.Timestamp("2019-12-31"):
        return "rolling_valid_2017_2019"
    if target_date <= pd.Timestamp("2021-12-31"):
        return "rolling_valid_2020_2021"
    if target_date <= pd.Timestamp("2023-12-31"):
        return "presealed_holdout_2022_2023"
    return "sealed_confirmation_2024_plus"


def season_bucket(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def issue_hour_bucket(hour: float | int | None) -> str:
    if hour is None or pd.isna(hour):
        return "missing"
    h = int(hour)
    if h < 16:
        return "before_16"
    if h <= 17:
        return "16_17"
    if h <= 20:
        return "18_20"
    if h <= 22:
        return "21_22"
    return "23_plus"


def forecast_range_bin(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if value <= 0:
        return "0"
    if value <= 1:
        return "1"
    if value <= 2:
        return "2"
    return "ge3"


def forecast_max_bin(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if value <= 20:
        return "le20"
    if value <= 24:
        return "21_24"
    if value <= 27:
        return "25_27"
    if value <= 30:
        return "28_30"
    if value <= 32:
        return "31_32"
    return "ge33"


def _safe_std(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.nanstd(values, ddof=0))


def _last_change_age_hours(issue_times: pd.Series, values: np.ndarray, cutoff: pd.Timestamp) -> float:
    if len(values) <= 1:
        return math.nan
    latest_value = values[-1]
    changed_positions = np.where(values != latest_value)[0]
    if len(changed_positions) == 0:
        return float((cutoff - issue_times.iloc[0]).total_seconds() / 3600.0)
    change_idx = int(changed_positions[-1] + 1)
    return float((cutoff - issue_times.iloc[change_idx]).total_seconds() / 3600.0)


def _tail_slope(issue_times: pd.Series, values: np.ndarray, n_tail: int = 3) -> float:
    if len(values) < 2:
        return 0.0
    tail_values = values[-n_tail:]
    tail_times = issue_times.iloc[-len(tail_values):]
    hours = (tail_times - tail_times.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    if np.nanmax(hours) <= 0:
        return 0.0
    slope, _ = np.polyfit(hours, tail_values, deg=1)
    return float(slope)


def _hkt_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(HKT)
    return timestamp.tz_convert(HKT)


def _hkt_series(values: pd.Series) -> pd.Series:
    series = pd.to_datetime(values, errors="coerce")
    if getattr(series.dtype, "tz", None) is None:
        return series.dt.tz_localize(HKT)
    return series.dt.tz_convert(HKT)


def select_forecast_anchors(rows: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        out = rows[["target_date", "cutoff_profile"]].copy()
        out["forecast_selector_status"] = "no_forecast_table_rows"
        return out
    grouped = {
        target_date: group.sort_values(["issue_at_utc", "source_url"]).reset_index(drop=True)
        for target_date, group in forecasts.groupby("target_date", sort=False)
    }
    feature_rows: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        target_date = pd.Timestamp(row.target_date).normalize()
        cutoff_utc = pd.Timestamp(row.cutoff_at_utc)
        group = grouped.get(target_date)
        base: dict[str, Any] = {
            "target_date": target_date,
            "cutoff_profile": row.cutoff_profile,
            "forecast_selector_status": "no_eligible_anchor",
            "eligible_forecast_count": 0,
            "forecast_anchor_tie_flag": 0,
        }
        if group is None or group.empty:
            feature_rows.append(base)
            continue
        eligible = group[group["issue_at_utc"].le(cutoff_utc)].copy()
        if eligible.empty:
            feature_rows.append(base)
            continue
        eligible = eligible.sort_values(["issue_at_utc", "raw_sha256", "source_url"], na_position="last")
        latest_issue = eligible["issue_at_utc"].iloc[-1]
        tied = eligible[eligible["issue_at_utc"].eq(latest_issue)]
        latest = tied.sort_values(["source_url"], na_position="last").iloc[-1]
        max_values = pd.to_numeric(eligible["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
        min_values = pd.to_numeric(eligible["forecast_min_c"], errors="coerce").to_numpy(dtype=float)
        range_values = max_values - min_values
        issue_hkt = _hkt_series(eligible["issue_at_hkt"])
        latest_issue_hkt = _hkt_timestamp(latest["issue_at_hkt"])
        latest_flags = forecast_text_flags(latest.get("full_text"))
        previous_flags = forecast_text_flags(eligible.iloc[-2].get("full_text")) if len(eligible) >= 2 else {}
        feature = {
            **base,
            "forecast_selector_status": "selected",
            "anchor_source_url": latest.get("source_url"),
            "anchor_raw_sha256": latest.get("raw_sha256"),
            "anchor_issue_at_hkt": latest.get("issue_at_hkt"),
            "anchor_issue_at_utc": latest.get("issue_at_utc"),
            "selected_forecast_issue_at_hkt": latest.get("issue_at_hkt"),
            "selected_forecast_issue_at_utc": latest.get("issue_at_utc"),
            "selected_forecast_source_url": latest.get("source_url"),
            "anchor_forecast_min_c": float(latest["forecast_min_c"]),
            "anchor_forecast_max_c": float(latest["forecast_max_c"]),
            "anchor_forecast_range_c": float(latest["forecast_max_c"] - latest["forecast_min_c"]),
            "anchor_forecast_midpoint_c": float(latest["forecast_midpoint_c"]),
            "official_max_c": float(latest["forecast_max_c"]),
            "official_min_c": float(latest["forecast_min_c"]),
            "official_range_c": float(latest["forecast_max_c"] - latest["forecast_min_c"]),
            "official_midpoint_c": float(latest["forecast_midpoint_c"]),
            "official_max_round_c": float(round(float(latest["forecast_max_c"]))),
            "official_range_bin": forecast_range_bin(float(latest["forecast_max_c"] - latest["forecast_min_c"])),
            "official_max_bin": forecast_max_bin(float(latest["forecast_max_c"])),
            "issue_hour_hkt": int(latest_issue_hkt.hour),
            "issue_minute_hkt": int(latest_issue_hkt.minute),
            "issue_hour_bucket": issue_hour_bucket(latest_issue_hkt.hour),
            "issue_age_minutes": float((cutoff_utc - pd.Timestamp(latest["issue_at_utc"])).total_seconds() / 60.0),
            "lead_seconds_to_target_start": float(
                (_hkt_timestamp(target_date) - latest_issue_hkt).total_seconds()
            ),
            "latest_issue_is_after_23_flag": int(latest_issue_hkt.hour >= 23),
            "forecast_anchor_tie_flag": int(len(tied) > 1),
            "eligible_forecast_count": int(len(eligible)),
            "max_forecast_revision_issue_at_utc": eligible["issue_at_utc"].max(),
            "rev_count": int(len(eligible)),
            "rev_first_max_c": float(max_values[0]),
            "rev_prev_max_c": float(max_values[-2]) if len(max_values) >= 2 else math.nan,
            "rev_latest_minus_prev_max_c": float(max_values[-1] - max_values[-2]) if len(max_values) >= 2 else math.nan,
            "rev_latest_minus_first_max_c": float(max_values[-1] - max_values[0]),
            "rev_path_max_c": float(np.nanmax(max_values)),
            "rev_path_min_c": float(np.nanmin(max_values)),
            "rev_path_range_c": float(np.nanmax(max_values) - np.nanmin(max_values)),
            "rev_path_std_c": _safe_std(max_values),
            "rev_num_up_moves": int(np.sum(np.diff(max_values) > 0)) if len(max_values) >= 2 else 0,
            "rev_num_down_moves": int(np.sum(np.diff(max_values) < 0)) if len(max_values) >= 2 else 0,
            "rev_num_same_moves": int(np.sum(np.diff(max_values) == 0)) if len(max_values) >= 2 else 0,
            "rev_last_change_age_hours": _last_change_age_hours(issue_hkt, max_values, pd.Timestamp(row.cutoff_at_hkt)),
            "rev_last3_slope_c_per_hour": _tail_slope(issue_hkt, max_values),
            "rev_latest3_all_same_flag": int(len(max_values) >= 3 and np.all(max_values[-3:] == max_values[-1])),
            "rev_first_issue_hour_hkt": int(issue_hkt.iloc[0].hour),
            "rev_last_issue_hour_hkt": int(issue_hkt.iloc[-1].hour),
            "rev_issue_span_hours": float(
                (issue_hkt.iloc[-1] - issue_hkt.iloc[0]).total_seconds() / 3600.0
            ),
            "rev_latest_minus_prev_min_c": float(min_values[-1] - min_values[-2]) if len(min_values) >= 2 else math.nan,
            "rev_latest_minus_first_min_c": float(min_values[-1] - min_values[0]),
            "rev_range_latest_minus_prev_c": float(range_values[-1] - range_values[-2]) if len(range_values) >= 2 else math.nan,
            "rev_range_path_std_c": _safe_std(range_values),
            "forecast_revision_missing_prev_flag": int(len(eligible) < 2),
            "forecast_single_issue_flag": int(len(eligible) == 1),
            "forecast_full_text_missing_flag": int(not bool(str(latest.get("full_text") or "").strip())),
        }
        feature.update(latest_flags)
        for name in (
            "fcst_flag_thunderstorm",
            "fcst_flag_showers",
            "fcst_flag_very_hot",
            "fcst_flag_cloudy",
            "fcst_flag_bright_periods",
        ):
            feature[f"rev_{name.removeprefix('fcst_flag_')}_added_latest"] = int(
                latest_flags.get(name, 0) > previous_flags.get(name, 0)
            )
        feature["rev_text_regime_changed_latest"] = int(
            any(latest_flags.get(name, 0) != previous_flags.get(name, 0) for name in latest_flags)
        )
        feature_rows.append(feature)
    return pd.DataFrame(feature_rows)


def add_forecast_features(
    rows: pd.DataFrame,
    forecasts: pd.DataFrame,
    registry: FeatureRegistry,
) -> pd.DataFrame:
    forecast_features = select_forecast_anchors(rows, forecasts)
    out = rows.merge(forecast_features, on=["target_date", "cutoff_profile"], how="left", validate="one_to_one")
    out["residual_y_c"] = out["y_true_c"] - out["anchor_forecast_max_c"]
    official_cols = [
        "official_max_c",
        "official_min_c",
        "official_range_c",
        "official_midpoint_c",
        "official_max_round_c",
        "official_range_bin",
        "official_max_bin",
        "issue_hour_hkt",
        "issue_minute_hkt",
        "issue_hour_bucket",
        "issue_age_minutes",
        "lead_seconds_to_target_start",
        "latest_issue_is_after_23_flag",
        "forecast_anchor_tie_flag",
    ]
    revision_cols = [col for col in out.columns if col.startswith("rev_") or col.startswith("fcst_flag_")]
    revision_cols += [
        "eligible_forecast_count",
        "forecast_revision_missing_prev_flag",
        "forecast_single_issue_flag",
        "forecast_full_text_missing_flag",
    ]
    calendar_cols = [
        "month",
        "quarter",
        "day_of_year",
        "doy_sin",
        "doy_cos",
        "warm_season_flag",
        "cool_season_flag",
        "hot_season_flag",
        "typhoon_season_flag",
        "shoulder_season_flag",
        "trend_years_since_2000",
        "post_2010_flag",
        "post_2018_flag",
    ]
    registry.add(
        [col for col in official_cols if col in out.columns],
        family="official_anchor",
        source_table="public.hko_historical_forecasts_2000_2026",
        source_time_column="issue_at_utc",
        eligibility_rule="source=info_gov local usable_local_minmax lead1 and issue_at_utc <= cutoff_at_utc",
    )
    registry.add(
        [col for col in revision_cols if col in out.columns],
        family="forecast_revision",
        source_table="public.hko_historical_forecasts_2000_2026",
        source_time_column="issue_at_utc",
        eligibility_rule="all lead1 Info.gov local forecast issues for target_date with issue_at_utc <= cutoff_at_utc",
    )
    registry.add(
        calendar_cols,
        family="calendar",
        source_table="derived_calendar",
        source_time_column="target_date",
        eligibility_rule="calendar known before cutoff",
    )
    return out
