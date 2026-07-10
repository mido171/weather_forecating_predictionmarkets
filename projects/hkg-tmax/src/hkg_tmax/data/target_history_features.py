"""Leakage-safe target-history and climatology features."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax.features.feature_registry import FeatureRegistry


TARGET_HISTORY_SQL = """
SELECT local_date::date AS local_date,
       target_tmax_c::double precision AS target_tmax_c
FROM feature_safe.hko_target_history_pre2024
WHERE target_tmax_c IS NOT NULL
ORDER BY local_date
"""


LAGS = (2, 3, 7, 14, 30, 60, 365)
ROLL_WINDOWS = (7, 14, 30, 60, 365)
ROLL_MIN_COUNTS = {7: 5, 14: 10, 30: 20, 60: 40, 365: 240}


def load_target_history(connection: Any) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(TARGET_HISTORY_SQL)
        rows = cursor.fetchall()
        columns = [desc.name for desc in cursor.description]
    frame = pd.DataFrame(rows, columns=columns)
    frame["local_date"] = pd.to_datetime(frame["local_date"], errors="coerce").dt.normalize()
    frame["target_tmax_c"] = pd.to_numeric(frame["target_tmax_c"], errors="coerce")
    return frame.dropna(subset=["local_date", "target_tmax_c"]).sort_values("local_date").reset_index(drop=True)


def add_target_history_features(
    rows: pd.DataFrame,
    history: pd.DataFrame,
    registry: FeatureRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = rows[["target_date"]].drop_duplicates().sort_values("target_date").reset_index(drop=True)
    features = build_target_history_for_dates(unique_dates["target_date"], history)
    out = rows.merge(features, on="target_date", how="left", validate="many_to_one")
    out["official_max_minus_doy_clim30_c"] = out["official_max_c"] - out["target_clim_doy_30yr_median_c"]
    out["official_max_minus_month_clim10_c"] = out["official_max_c"] - out["target_clim_month_10yr_mean_c"]
    out["official_midpoint_minus_doy_clim30_c"] = out["official_midpoint_c"] - out["target_clim_doy_30yr_median_c"]
    out["hko_latest_temp_minus_doy_hour_clim_c"] = out["hko_latest_temp_c"] - out["target_clim_doy_30yr_median_c"]
    out["hko_temp_mean_24h_minus_doy_clim_c"] = out["hko_temp_mean_24h_c"] - out["target_clim_doy_30yr_median_c"]
    out["hko_pre_target_afternoon_max_minus_official_max_c"] = (
        out["hko_pre_target_afternoon_max_sofar_c"] - out["official_max_c"]
    )
    out["hko_evening_temp_minus_official_min_c"] = out["hko_pre_target_evening_temp_c"] - out["official_min_c"]
    target_cols = [col for col in out.columns if col.startswith("target_")]
    normalized_cols = [
        "official_max_minus_doy_clim30_c",
        "official_max_minus_month_clim10_c",
        "official_midpoint_minus_doy_clim30_c",
        "hko_latest_temp_minus_doy_hour_clim_c",
        "hko_temp_mean_24h_minus_doy_clim_c",
        "hko_pre_target_afternoon_max_minus_official_max_c",
        "hko_evening_temp_minus_official_min_c",
    ]
    registry.add(
        [col for col in target_cols if col not in {"target_date"}],
        family="target_history",
        source_table="feature_safe.hko_target_history_pre2024",
        source_time_column="local_date",
        eligibility_rule="local_date <= target_date - 2 days; sealed-blind for confirmation rows",
        uses_target_label_boolean=True,
        minimum_lag_days=2,
    )
    registry.add(
        normalized_cols,
        family="target_history_normalized",
        source_table="feature_safe.hko_target_history_pre2024",
        source_time_column="local_date",
        eligibility_rule="past-only target climatology merged with eligible forecast/hourly fields",
        uses_target_label_boolean=True,
        minimum_lag_days=2,
    )
    audit = target_history_audit(features)
    return out, audit


def build_target_history_for_dates(target_dates: pd.Series, history: pd.DataFrame) -> pd.DataFrame:
    history = history.sort_values("local_date").reset_index(drop=True)
    date_index = pd.date_range(history["local_date"].min(), history["local_date"].max(), freq="D")
    series = history.set_index("local_date")["target_tmax_c"].reindex(date_index)
    hist_dates = history["local_date"].to_numpy(dtype="datetime64[ns]")
    hist_values = history["target_tmax_c"].to_numpy(dtype=float)
    hist_years = history["local_date"].dt.year.to_numpy(dtype=int)
    hist_months = history["local_date"].dt.month.to_numpy(dtype=int)
    hist_doys = history["local_date"].dt.dayofyear.to_numpy(dtype=int)
    doy_indices = {}
    for doy in range(1, 367):
        distance = np.abs(hist_doys - doy)
        doy_indices[doy] = np.flatnonzero(np.minimum(distance, 366 - distance) <= 7)
    month_indices = {month: np.flatnonzero(hist_months == month) for month in range(1, 13)}
    records: list[dict[str, Any]] = []
    for target in pd.to_datetime(target_dates).dt.normalize():
        max_source = target - pd.Timedelta(days=2)
        record: dict[str, Any] = {
            "target_date": target,
            "target_history_max_source_date": max_source,
        }
        for lag in LAGS:
            value = series.get(target - pd.Timedelta(days=lag), math.nan)
            record[f"target_lag{lag}_tmax_c"] = float(value) if pd.notna(value) else math.nan
            record[f"target_lag_{lag}_missing_flag"] = int(pd.isna(value))
        for window in ROLL_WINDOWS:
            start = max_source - pd.Timedelta(days=window - 1)
            values = series.loc[start:max_source].dropna().to_numpy(dtype=float)
            min_count = ROLL_MIN_COUNTS[window]
            if len(values) >= min_count:
                record[f"target_roll{window}_mean_lag2_c"] = float(np.mean(values))
                if window in {7, 14}:
                    record[f"target_roll{window}_max_lag2_c"] = float(np.max(values))
            else:
                record[f"target_roll{window}_mean_lag2_c"] = math.nan
                if window in {7, 14}:
                    record[f"target_roll{window}_max_lag2_c"] = math.nan
            record[f"target_roll_{window}_insufficient_count_flag"] = int(len(values) < min_count)
        record["target_roll30_anomaly_lag2_c"] = _diff(
            record["target_lag2_tmax_c"], record["target_roll30_mean_lag2_c"]
        )
        record["target_roll7_minus_roll30_c"] = _diff(
            record["target_roll7_mean_lag2_c"], record["target_roll30_mean_lag2_c"]
        )
        record["target_hot_spell_33_lag2_days"] = spell_length(series, max_source, threshold=33.0, direction="ge")
        record["target_very_hot_spell_34_lag2_days"] = spell_length(series, max_source, threshold=34.0, direction="ge")
        record["target_cool_spell_16_lag2_days"] = spell_length(series, max_source, threshold=16.0, direction="le")
        climatology = past_climatology(
            target,
            max_source,
            hist_dates=hist_dates,
            hist_values=hist_values,
            hist_years=hist_years,
            hist_months=hist_months,
            hist_doys=hist_doys,
            doy_indices=doy_indices,
            month_indices=month_indices,
        )
        record.update(climatology)
        record["target_modern_warming_signal_c"] = _diff(
            record["target_clim_doy_10yr_median_c"], record["target_clim_doy_30yr_median_c"]
        )
        record["target_lag2_minus_doy30_clim_c"] = _diff(
            record["target_lag2_tmax_c"], record["target_clim_doy_30yr_median_c"]
        )
        record["target_roll30_minus_doy30_clim_c"] = _diff(
            record["target_roll30_mean_lag2_c"], record["target_clim_doy_30yr_median_c"]
        )
        records.append(record)
    return pd.DataFrame(records)


def spell_length(series: pd.Series, max_source: pd.Timestamp, *, threshold: float, direction: str) -> int:
    count = 0
    cursor = max_source
    while cursor in series.index:
        value = series.get(cursor, math.nan)
        if pd.isna(value):
            break
        if direction == "ge" and float(value) >= threshold:
            count += 1
        elif direction == "le" and float(value) <= threshold:
            count += 1
        else:
            break
        cursor -= pd.Timedelta(days=1)
    return count


def past_climatology(
    target: pd.Timestamp,
    max_source: pd.Timestamp,
    *,
    hist_dates: np.ndarray,
    hist_values: np.ndarray,
    hist_years: np.ndarray,
    hist_months: np.ndarray,
    hist_doys: np.ndarray,
    doy_indices: dict[int, np.ndarray],
    month_indices: dict[int, np.ndarray],
) -> dict[str, float]:
    max_source_np = np.datetime64(max_source.to_datetime64())
    target_year = int(target.year)
    month = int(target.month)
    doy = int(target.dayofyear)
    doy_idx = doy_indices[doy]
    doy_idx = doy_idx[hist_dates[doy_idx] <= max_source_np]
    month_idx = month_indices[month]
    month_idx = month_idx[hist_dates[month_idx] <= max_source_np]
    values_all = hist_values[doy_idx]
    values_30 = hist_values[doy_idx[hist_years[doy_idx] >= target_year - 30]]
    values_10 = hist_values[doy_idx[hist_years[doy_idx] >= target_year - 10]]
    month_30 = hist_values[month_idx[hist_years[month_idx] >= target_year - 30]]
    month_10 = hist_values[month_idx[hist_years[month_idx] >= target_year - 10]]
    return {
        "target_clim_doy_all_past_median_c": _median(values_all),
        "target_clim_doy_30yr_median_c": _median(values_30),
        "target_clim_doy_10yr_median_c": _median(values_10),
        "target_clim_month_30yr_mean_c": _mean(month_30),
        "target_clim_month_10yr_mean_c": _mean(month_10),
        "target_clim_month_10yr_std_c": _std(month_10),
    }


def circular_doy_distance(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, 366 - diff)


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else math.nan


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else math.nan


def _diff(left: object, right: object) -> float:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return math.nan
    return float(left) - float(right)


def target_history_audit(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in features.columns:
        if column == "target_date":
            continue
        rows.append(
            {
                "feature": column,
                "missing_pct": float(features[column].isna().mean() * 100.0),
                "non_null_count": int(features[column].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
