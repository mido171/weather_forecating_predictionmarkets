"""Lag-safe official forecast residual memory features.

These features describe how the strict official forecast anchor has erred on
previous target dates under the same cutoff profile.  They are deliberately
date-based and enforce a lag-2 floor: for target date T, the newest residual
allowed as a predictor is T-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RESIDUAL_MEMORY_FEATURES: list[str] = [
    "residual_lag2_c",
    "residual_lag3_c",
    "residual_lag7_c",
    "residual_lag14_c",
    "residual_roll7_mean_lag2_c",
    "residual_roll14_mean_lag2_c",
    "residual_roll30_mean_lag2_c",
    "residual_roll60_mean_lag2_c",
    "residual_roll7_median_lag2_c",
    "residual_roll30_median_lag2_c",
    "residual_roll7_abs_mean_lag2_c",
    "residual_roll14_abs_mean_lag2_c",
    "residual_roll30_abs_mean_lag2_c",
    "residual_roll60_abs_mean_lag2_c",
    "residual_roll7_std_lag2_c",
    "residual_roll30_std_lag2_c",
    "residual_roll7_positive_rate_lag2",
    "residual_roll14_positive_rate_lag2",
    "residual_roll30_positive_rate_lag2",
    "residual_roll14_large_positive_rate_lag2",
    "residual_roll14_large_negative_rate_lag2",
    "residual_roll30_large_abs_rate_lag2",
    "residual_roll7_ewm_halflife3_lag2_c",
    "residual_roll30_ewm_halflife10_lag2_c",
    "residual_roll7_minus_roll30_lag2_c",
    "residual_memory_count_roll7",
    "residual_memory_count_roll14",
    "residual_memory_count_roll30",
    "residual_memory_count_roll60",
    "residual_memory_missing_lag2_flag",
    "residual_memory_missing_roll7_flag",
    "residual_memory_missing_roll30_flag",
]

RESIDUAL_MEMORY_LINEAGE_COLUMNS: list[str] = [
    "residual_memory_lag2_source_date",
    "residual_memory_max_source_date",
    "residual_memory_min_lag_days",
]


@dataclass(frozen=True)
class ResidualMemoryBuildResult:
    frame: pd.DataFrame
    feature_names: list[str]
    feature_audit: pd.DataFrame
    publication_safety_audit: dict[str, Any]
    lineage_rows: list[dict[str, Any]]


def _as_target_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _official_residual_series(group: pd.DataFrame) -> pd.Series:
    selected = group[
        group["forecast_selector_status"].eq("selected")
        & group["anchor_forecast_max_c"].notna()
        & group["y_true_c"].notna()
    ].copy()
    selected["target_date"] = _as_target_date(selected["target_date"])
    selected["official_residual_c"] = (
        pd.to_numeric(selected["y_true_c"], errors="coerce")
        - pd.to_numeric(selected["anchor_forecast_max_c"], errors="coerce")
    )
    selected = selected.sort_values(["target_date", "anchor_issue_at_utc"], na_position="last")
    selected = selected.drop_duplicates("target_date", keep="last")
    return selected.set_index("target_date")["official_residual_c"].sort_index()


def _window_values(residual_by_date: pd.Series, target_date: pd.Timestamp, *, window: int, min_lag_days: int) -> pd.Series:
    end = target_date - pd.Timedelta(days=min_lag_days)
    start = end - pd.Timedelta(days=window - 1)
    if pd.isna(start) or pd.isna(end):
        return pd.Series(dtype=float)
    dates = pd.date_range(start, end, freq="D")
    return residual_by_date.reindex(dates).dropna().astype(float)


def _ewm_last(values: pd.Series, *, halflife: float) -> float:
    if values.empty:
        return float("nan")
    return float(values.ewm(halflife=halflife, adjust=False).mean().iloc[-1])


def _row_features(
    residual_by_date: pd.Series,
    target_date: pd.Timestamp,
    *,
    min_lag_days: int,
    min_counts: dict[int, int],
    large_threshold_c: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    latest_source_date: pd.Timestamp | pd.NaT = pd.NaT
    for lag in (2, 3, 7, 14):
        source_date = target_date - pd.Timedelta(days=lag)
        value = residual_by_date.get(source_date, np.nan)
        row[f"residual_lag{lag}_c"] = float(value) if pd.notna(value) else np.nan
        if lag == 2:
            row["residual_memory_lag2_source_date"] = source_date if pd.notna(value) else pd.NaT

    windows: dict[int, pd.Series] = {}
    for window in (7, 14, 30, 60):
        values = _window_values(residual_by_date, target_date, window=window, min_lag_days=min_lag_days)
        windows[window] = values
        row[f"residual_memory_count_roll{window}"] = int(values.count())
        if not values.empty:
            latest_source_date = max(latest_source_date, values.index.max()) if pd.notna(latest_source_date) else values.index.max()

    for window, values in windows.items():
        enough = int(values.count()) >= int(min_counts.get(window, 1))
        prefix = f"residual_roll{window}"
        row[f"{prefix}_mean_lag2_c"] = float(values.mean()) if enough else np.nan
        row[f"{prefix}_abs_mean_lag2_c"] = float(values.abs().mean()) if enough else np.nan
        if window in {7, 30}:
            row[f"{prefix}_median_lag2_c"] = float(values.median()) if enough else np.nan
            row[f"{prefix}_std_lag2_c"] = float(values.std(ddof=0)) if enough else np.nan
        if window in {7, 14, 30}:
            row[f"{prefix}_positive_rate_lag2"] = float((values > 0.0).mean()) if enough else np.nan
        if window == 14:
            row["residual_roll14_large_positive_rate_lag2"] = float((values >= large_threshold_c).mean()) if enough else np.nan
            row["residual_roll14_large_negative_rate_lag2"] = float((values <= -large_threshold_c).mean()) if enough else np.nan
        if window == 30:
            row["residual_roll30_large_abs_rate_lag2"] = float((values.abs() >= large_threshold_c).mean()) if enough else np.nan

    row["residual_roll7_ewm_halflife3_lag2_c"] = (
        _ewm_last(windows[7], halflife=3.0) if int(windows[7].count()) >= int(min_counts.get(7, 1)) else np.nan
    )
    row["residual_roll30_ewm_halflife10_lag2_c"] = (
        _ewm_last(windows[30], halflife=10.0) if int(windows[30].count()) >= int(min_counts.get(30, 1)) else np.nan
    )
    if pd.notna(row.get("residual_roll7_mean_lag2_c")) and pd.notna(row.get("residual_roll30_mean_lag2_c")):
        row["residual_roll7_minus_roll30_lag2_c"] = float(row["residual_roll7_mean_lag2_c"] - row["residual_roll30_mean_lag2_c"])
    else:
        row["residual_roll7_minus_roll30_lag2_c"] = np.nan
    row["residual_memory_missing_lag2_flag"] = int(pd.isna(row["residual_lag2_c"]))
    row["residual_memory_missing_roll7_flag"] = int(pd.isna(row["residual_roll7_mean_lag2_c"]))
    row["residual_memory_missing_roll30_flag"] = int(pd.isna(row["residual_roll30_mean_lag2_c"]))
    row["residual_memory_max_source_date"] = latest_source_date
    row["residual_memory_min_lag_days"] = (
        int((target_date - latest_source_date).days) if pd.notna(latest_source_date) else np.nan
    )
    return row


def build_residual_memory_features(
    matrix: pd.DataFrame,
    *,
    cutoff_profiles: list[str] | None = None,
    min_lag_days: int = 2,
    min_counts: dict[int, int] | None = None,
    large_threshold_c: float = 1.5,
    lag1_enabled: bool = False,
) -> ResidualMemoryBuildResult:
    if lag1_enabled:
        raise ValueError("lag1 residual memory is disabled unless a publication-proof audit enables it")
    min_counts = min_counts or {7: 4, 14: 7, 30: 15, 60: 30}
    frame = matrix.copy()
    frame["target_date"] = _as_target_date(frame["target_date"])
    selected_cutoffs = cutoff_profiles or sorted(frame["cutoff_profile"].dropna().astype(str).unique().tolist())
    parts: list[pd.DataFrame] = []
    for cutoff in selected_cutoffs:
        group = frame[frame["cutoff_profile"].eq(cutoff)].copy()
        if group.empty:
            continue
        residual_by_date = _official_residual_series(group)
        records: list[dict[str, Any]] = []
        for target_date in group["target_date"]:
            records.append(
                _row_features(
                    residual_by_date,
                    pd.Timestamp(target_date),
                    min_lag_days=min_lag_days,
                    min_counts=min_counts,
                    large_threshold_c=large_threshold_c,
                )
            )
        feature_part = pd.DataFrame(records, index=group.index)
        parts.append(pd.concat([group, feature_part], axis=1))
    out = pd.concat(parts, axis=0).sort_index() if parts else frame
    audit = residual_memory_publication_safety_audit(out, min_lag_days=min_lag_days, lag1_enabled=lag1_enabled)
    return ResidualMemoryBuildResult(
        frame=out,
        feature_names=list(RESIDUAL_MEMORY_FEATURES),
        feature_audit=residual_memory_feature_audit(out, list(RESIDUAL_MEMORY_FEATURES)),
        publication_safety_audit=audit,
        lineage_rows=residual_memory_lineage_rows(list(RESIDUAL_MEMORY_FEATURES), min_lag_days=min_lag_days),
    )


def residual_memory_lineage_rows(feature_names: list[str], *, min_lag_days: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "feature_name": feature,
            "family": "official_residual_memory",
            "source_table": "label_core.hko_daily_tmax + public.hko_historical_forecasts_2000_2026",
            "source_time_column": "target_date and selected official forecast issue_at_utc",
            "eligibility_rule": (
                "official residual = settled target minus strict selected official max; "
                "same cutoff profile; source target_date <= prediction target_date - 2 days"
            ),
            "uses_target_label_boolean": True,
            "minimum_lag_days": min_lag_days,
        }
        for feature in feature_names
    ]


def residual_memory_feature_audit(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cutoff, group in frame.groupby("cutoff_profile", dropna=False):
        for feature in feature_names:
            if feature not in group:
                continue
            rows.append(
                {
                    "cutoff_profile": cutoff,
                    "feature": feature,
                    "family": "official_residual_memory",
                    "rows": int(len(group)),
                    "non_null_count": int(group[feature].notna().sum()),
                    "missing_pct": float(group[feature].isna().mean() * 100.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["cutoff_profile", "feature"]).reset_index(drop=True)


def residual_memory_publication_safety_audit(
    frame: pd.DataFrame,
    *,
    min_lag_days: int = 2,
    lag1_enabled: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    target_date = _as_target_date(frame["target_date"]) if "target_date" in frame else pd.Series(dtype="datetime64[ns]")
    latest_allowed = target_date - pd.Timedelta(days=min_lag_days)
    max_source = pd.to_datetime(frame.get("residual_memory_max_source_date"), errors="coerce")
    future_source = max_source.notna() & (max_source > latest_allowed)
    checks.append(
        {
            "check_name": "residual_memory_max_source_date_lag2_or_older",
            "status": "pass" if int(future_source.sum()) == 0 else "fail",
            "violation_count": int(future_source.sum()),
            "sample_target_dates": frame.loc[future_source, "target_date"].head(10).astype(str).tolist()
            if "target_date" in frame
            else [],
        }
    )
    min_lag = pd.to_numeric(frame.get("residual_memory_min_lag_days"), errors="coerce")
    min_lag_bad = min_lag.notna() & (min_lag < min_lag_days)
    checks.append(
        {
            "check_name": "residual_memory_min_lag_days_floor",
            "status": "pass" if int(min_lag_bad.sum()) == 0 else "fail",
            "violation_count": int(min_lag_bad.sum()),
            "minimum_lag_days": min_lag_days,
        }
    )
    lag1_columns = [
        column
        for column in frame.columns
        if str(column) == "residual_lag1_c" or str(column).startswith("residual_lag1_")
    ]
    lag1_violation = bool(lag1_columns and not lag1_enabled)
    checks.append(
        {
            "check_name": "lag1_residual_memory_disabled",
            "status": "pass" if not lag1_violation else "fail",
            "violation_count": int(lag1_violation),
            "columns": lag1_columns,
        }
    )
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "status": status,
        "total_violations": int(sum(int(check["violation_count"]) for check in checks)),
        "min_lag_days": min_lag_days,
        "lag1_enabled": bool(lag1_enabled),
        "checks": checks,
    }
