from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0083_expanded_frame_official_anchor_replay import date_text
from scripts.run_hkg_t24_0085_long_history_feature_station_residual_bridge import (
    FEATURE_MATRIX_PATH,
    numeric_feature_columns,
    safe_corr,
)
from scripts.run_hkg_t24_beastmode_signal_discovery import (
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (
    classify_feature_family,
    station_ids_in_feature,
    update_markdown_section,
)

FOLDER_NAME = "0089_remaining_error_regime_autopsy"
INPUT_0088_TOP_PATH = (
    RESEARCH_ROOT / "0088_prior_gated_specialist_stack" / "artifacts" / "top_predictions.csv"
)
HIGH_ERROR_QUANTILE = 0.80
LOW_ERROR_QUANTILE = 0.20
MIN_FEATURE_ROWS = 730
MIN_TAIL_ROWS = 60


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def rmse(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return math.nan
    return float(np.sqrt(np.mean(np.square(clean.to_numpy(dtype=float)))))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    missing = [path for path in (INPUT_0088_TOP_PATH, FEATURE_MATRIX_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0089 requires 0088 predictions and the feature matrix first: {missing}")

    predictions = pd.read_csv(INPUT_0088_TOP_PATH)
    required = {
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "forecast_max_c",
        "candidate_prediction_c",
        "season",
        "frame_segment",
        "era_bucket",
    }
    missing_cols = required.difference(predictions.columns)
    if missing_cols:
        raise RuntimeError(f"0088 top predictions missing required columns: {sorted(missing_cols)}")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions = predictions[predictions["target_date"].notna() & (predictions["target_date"] < CONFIRMATION_START)].copy()
    require_no_confirmation_dates(predictions["target_date"], context="0089 prediction input")
    for column in ("target_tmax_c", "forecast_max_c", "candidate_prediction_c"):
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    predictions = predictions[
        predictions[["target_tmax_c", "forecast_max_c", "candidate_prediction_c"]].notna().all(axis=1)
    ].copy()
    predictions["candidate_error_c"] = predictions["candidate_prediction_c"] - predictions["target_tmax_c"]
    predictions["candidate_abs_error_c"] = predictions["candidate_error_c"].abs()
    predictions["raw_error_c"] = predictions["forecast_max_c"] - predictions["target_tmax_c"]
    predictions["raw_abs_error_c"] = predictions["raw_error_c"].abs()
    predictions["remaining_improvement_vs_raw_c"] = predictions["raw_abs_error_c"] - predictions["candidate_abs_error_c"]
    for column in ("forecast_source_family", "season", "frame_segment", "era_bucket"):
        predictions[column] = predictions[column].astype(str)

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features = features.loc[:, ~features.columns.duplicated()].copy()
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].notna() & (features["target_date"] < CONFIRMATION_START)].copy()
    require_no_confirmation_dates(features["target_date"], context="0089 feature matrix")
    feature_cols = numeric_feature_columns(features)
    joined = predictions.merge(features[["target_date", *feature_cols]], on="target_date", how="left")
    require_no_confirmation_dates(joined["target_date"], context="0089 joined frame")
    return predictions.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True), joined, feature_cols


def score_group(group: pd.DataFrame, *, grouping: str, group_value: str, full_mae: float) -> dict[str, object]:
    dates = pd.to_datetime(group["target_date"], errors="coerce")
    abs_error = pd.to_numeric(group["candidate_abs_error_c"], errors="coerce")
    signed_error = pd.to_numeric(group["candidate_error_c"], errors="coerce")
    raw_abs_error = pd.to_numeric(group["raw_abs_error_c"], errors="coerce")
    return {
        "grouping": grouping,
        "group_value": group_value,
        "n": int(len(group)),
        "first_date": date_text(dates.min()),
        "last_date": date_text(dates.max()),
        "mae": float(abs_error.mean()),
        "rmse": rmse(signed_error),
        "bias": float(signed_error.mean()),
        "median_abs_error": float(abs_error.median()),
        "p90_abs_error": float(abs_error.quantile(0.90)),
        "over_2c_rate": float(abs_error.gt(2.0).mean()),
        "underforecast_1c_rate": float(signed_error.lt(-1.0).mean()),
        "overforecast_1c_rate": float(signed_error.gt(1.0).mean()),
        "raw_mae": float(raw_abs_error.mean()),
        "delta_mae_vs_raw": float(abs_error.mean() - raw_abs_error.mean()),
        "delta_mae_vs_full": float(abs_error.mean() - full_mae),
        "share_total_abs_error": float(abs_error.sum()),
    }


def error_regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    full_mae = float(frame["candidate_abs_error_c"].mean())
    rows: list[dict[str, object]] = [score_group(frame, grouping="all", group_value="all", full_mae=full_mae)]
    specs = [
        ("forecast_source_family", ["forecast_source_family"]),
        ("frame_segment", ["frame_segment"]),
        ("season", ["season"]),
        ("month", ["month"]),
        ("era_bucket", ["era_bucket"]),
        ("source_season", ["forecast_source_family", "season"]),
        ("source_frame", ["forecast_source_family", "frame_segment"]),
        ("frame_season", ["frame_segment", "season"]),
    ]
    working = frame.copy()
    working["month"] = pd.to_datetime(working["target_date"], errors="coerce").dt.month.astype(str)
    for grouping, columns in specs:
        for keys, group in working.groupby(columns, observed=True, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            group_value = " | ".join(str(value) for value in key_tuple)
            rows.append(score_group(group, grouping=grouping, group_value=group_value, full_mae=full_mae))
    out = pd.DataFrame(rows)
    out["share_total_abs_error_pct"] = out["share_total_abs_error"] / out.loc[out["grouping"].eq("all"), "share_total_abs_error"].iloc[0] * 100.0
    return out.sort_values(["mae", "n"], ascending=[False, False]).reset_index(drop=True)


def error_sign_table(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["error_direction"] = np.select(
        [
            working["candidate_error_c"].lt(-1.0),
            working["candidate_error_c"].gt(1.0),
            working["candidate_abs_error_c"].le(0.5),
        ],
        ["underforecast_gt_1c", "overforecast_gt_1c", "within_0p5c"],
        default="middle_error",
    )
    full_mae = float(working["candidate_abs_error_c"].mean())
    rows: list[dict[str, object]] = []
    for columns, grouping in (
        (["error_direction"], "error_direction"),
        (["forecast_source_family", "error_direction"], "source_error_direction"),
        (["season", "error_direction"], "season_error_direction"),
        (["frame_segment", "error_direction"], "frame_error_direction"),
    ):
        for keys, group in working.groupby(columns, observed=True, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            rows.append(
                score_group(group, grouping=grouping, group_value=" | ".join(map(str, key_tuple)), full_mae=full_mae)
            )
    return pd.DataFrame(rows).sort_values(["mae", "n"], ascending=[False, False]).reset_index(drop=True)


def high_low_feature_contrasts(
    joined: pd.DataFrame,
    feature_cols: list[str],
    *,
    high_quantile: float = HIGH_ERROR_QUANTILE,
    low_quantile: float = LOW_ERROR_QUANTILE,
    min_feature_rows: int = MIN_FEATURE_ROWS,
    min_tail_rows: int = MIN_TAIL_ROWS,
) -> pd.DataFrame:
    threshold_high = float(joined["candidate_abs_error_c"].quantile(high_quantile))
    threshold_low = float(joined["candidate_abs_error_c"].quantile(low_quantile))
    high = joined[joined["candidate_abs_error_c"].ge(threshold_high)].copy()
    low = joined[joined["candidate_abs_error_c"].le(threshold_low)].copy()
    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        series = pd.to_numeric(joined[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = series.notna() & joined["candidate_abs_error_c"].notna()
        if int(valid.sum()) < min_feature_rows or series[valid].nunique(dropna=True) < 5:
            continue
        high_values = pd.to_numeric(high[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        low_values = pd.to_numeric(low[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(high_values) < min_tail_rows or len(low_values) < min_tail_rows:
            continue
        full_std = float(series[valid].std(ddof=0))
        mean_diff = float(high_values.mean() - low_values.mean())
        standardized_diff = mean_diff / full_std if full_std > 0 else math.nan
        corr_abs_n, corr_abs = safe_corr(
            joined[feature],
            joined["candidate_abs_error_c"],
            min_rows=min_feature_rows,
        )
        corr_signed_n, corr_signed = safe_corr(
            joined[feature],
            joined["candidate_error_c"],
            min_rows=min_feature_rows,
        )
        _n, corr_improvement = safe_corr(
            joined[feature],
            joined["remaining_improvement_vs_raw_c"],
            min_rows=min_feature_rows,
        )
        priority = (
            abs(finite_float(standardized_diff, 0.0)) * 0.85
            + abs(finite_float(corr_abs, 0.0)) * 1.35
            + abs(finite_float(corr_signed, 0.0)) * 0.55
            + abs(finite_float(corr_improvement, 0.0)) * 0.45
        )
        rows.append(
            {
                "feature": feature,
                "family": classify_feature_family(feature),
                "station_ids": station_ids_in_feature(feature),
                "valid_rows": int(valid.sum()),
                "high_error_threshold_c": threshold_high,
                "low_error_threshold_c": threshold_low,
                "high_error_rows": int(len(high_values)),
                "low_error_rows": int(len(low_values)),
                "high_error_mean": float(high_values.mean()),
                "low_error_mean": float(low_values.mean()),
                "high_minus_low_mean": mean_diff,
                "standardized_high_low_diff": standardized_diff,
                "corr_abs_error_rows": corr_abs_n,
                "corr_abs_error": corr_abs,
                "corr_signed_error_rows": corr_signed_n,
                "corr_signed_error": corr_signed,
                "corr_improvement_vs_raw": corr_improvement,
                "contrast_priority": float(priority),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["contrast_priority", "valid_rows"], ascending=[False, False]).reset_index(drop=True)


def feature_bucket_error_profiles(joined: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in contrasts.head(25)["feature"].tolist():
        working = joined[[feature, "candidate_abs_error_c", "candidate_error_c", "remaining_improvement_vs_raw_c"]].copy()
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
        working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, "candidate_abs_error_c"])
        if len(working) < MIN_FEATURE_ROWS or working[feature].nunique(dropna=True) < 5:
            continue
        try:
            working["feature_bucket"] = pd.qcut(working[feature], q=5, duplicates="drop").astype(str)
        except ValueError:
            continue
        for bucket, group in working.groupby("feature_bucket", observed=True):
            if len(group) < MIN_TAIL_ROWS:
                continue
            rows.append(
                {
                    "feature": feature,
                    "family": classify_feature_family(feature),
                    "station_ids": station_ids_in_feature(feature),
                    "feature_bucket": str(bucket),
                    "n": int(len(group)),
                    "feature_min": float(group[feature].min()),
                    "feature_max": float(group[feature].max()),
                    "mae": float(group["candidate_abs_error_c"].mean()),
                    "rmse": rmse(group["candidate_error_c"]),
                    "bias": float(group["candidate_error_c"].mean()),
                    "p90_abs_error": float(group["candidate_abs_error_c"].quantile(0.90)),
                    "improvement_vs_raw_mean": float(group["remaining_improvement_vs_raw_c"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["feature", "mae"], ascending=[True, False]).reset_index(drop=True)


def family_summary(contrasts: pd.DataFrame) -> pd.DataFrame:
    if contrasts.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for family, group in contrasts.groupby("family", observed=True):
        best = group.sort_values("contrast_priority", ascending=False).iloc[0]
        rows.append(
            {
                "family": str(family),
                "feature_count": int(len(group)),
                "best_feature": str(best["feature"]),
                "best_station_ids": str(best["station_ids"]),
                "best_contrast_priority": float(best["contrast_priority"]),
                "best_standardized_high_low_diff": float(best["standardized_high_low_diff"]),
                "best_corr_abs_error": float(best["corr_abs_error"]),
                "top10_features_in_family": int(group.head(10).shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("best_contrast_priority", ascending=False).reset_index(drop=True)


def station_summary(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in contrasts.to_dict("records"):
        for station_id in [part for part in str(row["station_ids"]).split(",") if part]:
            rows.append(
                {
                    "station_id": station_id,
                    "feature": row["feature"],
                    "family": row["family"],
                    "contrast_priority": row["contrast_priority"],
                    "standardized_high_low_diff": row["standardized_high_low_diff"],
                    "corr_abs_error": row["corr_abs_error"],
                }
            )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        return pd.DataFrame(columns=["station_id", "feature_count", "best_feature"])
    out_rows: list[dict[str, object]] = []
    for station_id, group in expanded.groupby("station_id", observed=True):
        best = group.sort_values("contrast_priority", ascending=False).iloc[0]
        out_rows.append(
            {
                "station_id": str(station_id),
                "feature_count": int(len(group)),
                "best_feature": str(best["feature"]),
                "best_family": str(best["family"]),
                "best_contrast_priority": float(best["contrast_priority"]),
                "best_standardized_high_low_diff": float(best["standardized_high_low_diff"]),
                "best_corr_abs_error": float(best["corr_abs_error"]),
            }
        )
    return pd.DataFrame(out_rows).sort_values("best_contrast_priority", ascending=False).reset_index(drop=True)


def specialist_action(feature: str, family: str) -> str:
    name = feature.lower()
    if family == "isd_station_network":
        if "morning_to_midday" in name or "temp_rise" in name:
            return "Build a morning-warming-rate gate: separate rapid inland warm-up days from suppressed coastal/cloudy days."
        if "dewpoint" in name or "humidity" in name:
            return "Build a station-network moisture suppression expert with source-specific shrinkage."
        return "Build a station-network disagreement/local-gradient specialist using only prior residuals."
    if family == "upper_air":
        if "ceiling" in name or "mixed" in name:
            return "Build a boundary-layer ceiling specialist to detect when surface Tmax is capped below the official range."
        if "temp" in name or "theta" in name:
            return "Build an upper-air thermal-state residual specialist, guarded by season and forecast source."
        return "Build an upper-air regime gate for remaining high-error days."
    if family == "target_memory":
        return "Build a persistence/reversal specialist using lagged target memory and rolling heat-state features."
    if family == "hko_daily_climate":
        return "Build a marine/radiation/daily-climate specialist for sea-temperature and cloud/rain suppression regimes."
    if family == "calendar_climatology":
        return "Use only as a guardrail, not as a standalone specialist, because calendar effects can overfit seasonality."
    return "Review manually and only promote if it survives source, frame, and season gates."


def next_specialist_leads(contrasts: pd.DataFrame) -> pd.DataFrame:
    if contrasts.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    used_families: set[str] = set()
    for _, row in contrasts.iterrows():
        family = str(row["family"])
        include = len(rows) < 12 or family not in used_families
        if not include:
            continue
        used_families.add(family)
        rows.append(
            {
                "lead_rank": len(rows) + 1,
                "feature": row["feature"],
                "family": family,
                "station_ids": row["station_ids"],
                "contrast_priority": row["contrast_priority"],
                "standardized_high_low_diff": row["standardized_high_low_diff"],
                "corr_abs_error": row["corr_abs_error"],
                "corr_signed_error": row["corr_signed_error"],
                "recommended_action": specialist_action(str(row["feature"]), family),
            }
        )
        if len(rows) >= 20 and len(used_families) >= 5:
            break
    return pd.DataFrame(rows)


def worst_cases(joined: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    top_features = [feature for feature in contrasts.head(12)["feature"].tolist() if feature in joined.columns]
    keep = [
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "forecast_max_c",
        "candidate_prediction_c",
        "candidate_error_c",
        "candidate_abs_error_c",
        "raw_abs_error_c",
        "remaining_improvement_vs_raw_c",
        "season",
        "frame_segment",
        "era_bucket",
        *top_features,
    ]
    return joined.sort_values("candidate_abs_error_c", ascending=False)[keep].head(250).reset_index(drop=True)


def build_summary(
    *,
    generated_at: str,
    predictions: pd.DataFrame,
    joined: pd.DataFrame,
    contrasts: pd.DataFrame,
    regimes: pd.DataFrame,
    families: pd.DataFrame,
    stations: pd.DataFrame,
    leads: pd.DataFrame,
) -> dict[str, object]:
    dates = pd.to_datetime(predictions["target_date"], errors="coerce")
    top_regime = regimes[~regimes["grouping"].eq("all")].iloc[0].to_dict() if len(regimes) > 1 else {}
    top_feature = contrasts.iloc[0].to_dict() if not contrasts.empty else {}
    return {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "input_0088_top_path": str(INPUT_0088_TOP_PATH),
        "feature_matrix_path": str(FEATURE_MATRIX_PATH),
        "rows": int(len(predictions)),
        "joined_rows": int(len(joined)),
        "first_target_date": date_text(dates.min()),
        "last_target_date": date_text(dates.max()),
        "mae": float(predictions["candidate_abs_error_c"].mean()),
        "rmse": rmse(predictions["candidate_error_c"]),
        "raw_mae": float(predictions["raw_abs_error_c"].mean()),
        "ranked_contrast_feature_count": int(len(contrasts)),
        "regime_count": int(len(regimes)),
        "family_count": int(len(families)),
        "station_count": int(len(stations)),
        "lead_count": int(len(leads)),
        "top_error_regime": top_regime.get("group_value", ""),
        "top_error_regime_grouping": top_regime.get("grouping", ""),
        "top_error_regime_mae": top_regime.get("mae"),
        "top_contrast_feature": top_feature.get("feature", ""),
        "top_contrast_family": top_feature.get("family", ""),
        "top_contrast_station_ids": top_feature.get("station_ids", ""),
        "top_contrast_priority": top_feature.get("contrast_priority"),
        "top_contrast_corr_abs_error": top_feature.get("corr_abs_error"),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "remaining_error_regime_autopsy_complete",
        "next_recommended_task": (
            "Run 0090 to convert the highest 0089 specialist leads into guarded past-only residual specialists, "
            "starting with the top high-error contrast features and rejecting any source/frame/season regression."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    regimes: pd.DataFrame,
    signs: pd.DataFrame,
    contrasts: pd.DataFrame,
    families: pd.DataFrame,
    stations: pd.DataFrame,
    buckets: pd.DataFrame,
    leads: pd.DataFrame,
    worst: pd.DataFrame,
) -> str:
    return f"""# 0089 Remaining Error Regime Autopsy

Generated: `{generated_at}`

## Purpose

`0088` showed that a simple prior-only router does not improve on the `0087` interaction champion. `0089` therefore stops trying to blend the same predictions and instead asks a sharper question: where are the remaining misses, what weather/station features distinguish those misses, and which next specialists are justified by evidence?

This is a leakage-safe diagnostic run, not model training. It uses the current `0088`/`0087` champion predictions from `{summary['first_target_date']}` through `{summary['last_target_date']}`, joins only same-target-date feature values already present in the pre-2024 research matrix, and keeps all 2024+ confirmation rows sealed.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Champion MAE | `{summary['mae']}` |
| Champion RMSE | `{summary['rmse']}` |
| Raw official MAE | `{summary['raw_mae']}` |
| Ranked contrast features | `{summary['ranked_contrast_feature_count']}` |
| Regime rows | `{summary['regime_count']}` |
| Family count | `{summary['family_count']}` |
| Station count | `{summary['station_count']}` |
| Top error regime | `{summary['top_error_regime_grouping']}: {summary['top_error_regime']}` |
| Top error regime MAE | `{summary['top_error_regime_mae']}` |
| Top contrast feature | `{summary['top_contrast_feature']}` |
| Top contrast family | `{summary['top_contrast_family']}` |
| Top contrast station IDs | `{summary['top_contrast_station_ids']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Highest Error Regimes

{markdown_table(regimes.head(30), max_rows=30)}

## Error Direction Regimes

{markdown_table(signs.head(25), max_rows=25)}

## High-Vs-Low Error Feature Contrasts

{markdown_table(contrasts.head(30), max_rows=30)}

## Feature-Family Attribution

{markdown_table(families, max_rows=20)}

## Station Attribution

{markdown_table(stations.head(30), max_rows=30)}

## Top Feature Bucket Profiles

{markdown_table(buckets.head(50), max_rows=50)}

## Next Specialist Leads

{markdown_table(leads, max_rows=25)}

## Worst Remaining Cases With Top Features

{markdown_table(worst.head(30), max_rows=30)}

## Interpretation

The most actionable rows are not simply the highest correlations. They are features that separate the worst-error tail from the low-error tail and also show a relationship to signed or absolute residual error. Those features indicate where a small, guarded specialist may have a chance to reduce MAE without turning the system into an uncontrolled black box.

The next experiment should use these leads to build source-aware, frame-aware, season-aware residual specialists. Any candidate must learn only from earlier target dates and must be rejected if it worsens old-frame, newly available, press, RSS, or seasonal slices.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], regimes: pd.DataFrame, contrasts: pd.DataFrame, leads: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0089_remaining_error_regime_autopsy.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Remaining-error autopsy | `{summary['rows']}` rows, `{summary['first_target_date']}` to `{summary['last_target_date']}` | Pre-2024 only |
| Current champion score | MAE `{summary['mae']}`, RMSE `{summary['rmse']}` | 0087/0088 champion |
| Ranked feature contrasts | `{summary['ranked_contrast_feature_count']}` | High-error vs low-error |
| Top error regime | `{summary['top_error_regime_grouping']}: {summary['top_error_regime']}` | MAE `{summary['top_error_regime_mae']}` |
| Top contrast feature | `{summary['top_contrast_feature']}` | `{summary['top_contrast_family']}` |
| Next specialist leads | `{summary['lead_count']}` | Documented |
| Leakage | `0` 2024+ rows | PASS |

Highest error regimes:

{markdown_table(regimes.head(8), max_rows=8)}

Top high-vs-low error feature contrasts:

{markdown_table(contrasts.head(8), max_rows=8)}

Next specialist leads:

{markdown_table(leads.head(8), max_rows=8)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0089 Remaining Error Regime Autopsy",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=(
            "Implement `0090_guarded_specialists_from_error_autopsy`: convert the top 0089 high-error contrast "
            "features into guarded, past-only residual specialists with source/frame/season no-regression gates. "
            "Use the current RSS archive only until the forecast backfill catches up; keep 2024+ sealed."
        ),
    )


def run() -> dict[str, object]:
    generated_at = now_utc()
    predictions, joined, feature_cols = load_inputs()
    regimes = error_regime_summary(predictions)
    signs = error_sign_table(predictions)
    contrasts = high_low_feature_contrasts(joined, feature_cols)
    families = family_summary(contrasts)
    stations = station_summary(contrasts)
    buckets = feature_bucket_error_profiles(joined, contrasts)
    leads = next_specialist_leads(contrasts)
    worst = worst_cases(joined, contrasts)
    summary = build_summary(
        generated_at=generated_at,
        predictions=predictions,
        joined=joined,
        contrasts=contrasts,
        regimes=regimes,
        families=families,
        stations=stations,
        leads=leads,
    )

    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "error_regime_summary.csv", regimes)
    write_csv(artifacts / "error_direction_summary.csv", signs)
    write_csv(artifacts / "high_low_feature_contrasts.csv", contrasts)
    write_csv(artifacts / "family_error_contrast_summary.csv", families)
    write_csv(artifacts / "station_error_contrast_summary.csv", stations)
    write_csv(artifacts / "top_feature_bucket_error_profiles.csv", buckets)
    write_csv(artifacts / "next_specialist_leads.csv", leads)
    write_csv(artifacts / "worst_remaining_cases_with_top_features.csv", worst)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "remaining_error_regime_autopsy_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            regimes=regimes,
            signs=signs,
            contrasts=contrasts,
            families=families,
            stations=stations,
            buckets=buckets,
            leads=leads,
            worst=worst,
        ),
    )
    update_milestones(summary, regimes, contrasts, leads)
    require_no_confirmation_dates(predictions["target_date"], context="0089 predictions")
    require_no_confirmation_dates(joined["target_date"], context="0089 joined frame")
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Autopsy the remaining pre-2024 errors of the current HKG Tmax champion."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
