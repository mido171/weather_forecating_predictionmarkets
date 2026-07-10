from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0083_expanded_frame_official_anchor_replay import date_text
from scripts.run_hkg_t24_beastmode_signal_discovery import (
    CONFIRMATION_START,
    DATASETS_ROOT,
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

FOLDER_NAME = "0085_long_history_feature_station_residual_bridge"
FEATURE_MATRIX_PATH = (
    DATASETS_ROOT
    / "12_hkg_t24_robust_experiment_outputs"
    / "hkg_t24_exp0050_0099_feature_matrix.parquet"
)
INPUT_0084_TOP_PATH = (
    RESEARCH_ROOT / "0084_expanded_frame_hardened_official_specialists" / "artifacts" / "top_predictions.csv"
)
MIN_FULL_HISTORY_ROWS = 3650
MIN_OFFICIAL_ROWS = 730
MIN_SEGMENT_ROWS = 365
MIN_BUCKET_ROWS = 60

NON_FEATURE_EXACT = {
    "target_tmax_c",
    "target_date",
    "valid_at_utc",
    "valid_at_hkt",
    "raw_retrieved_at_utc",
    "content_sha256",
    "operational_input_allowed",
    "release_latency_proven",
}
NON_FEATURE_PREFIXES = ("official_", "forecast_")


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_feature_matrix() -> pd.DataFrame:
    if not FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Missing feature matrix: {FEATURE_MATRIX_PATH}")
    frame = pd.read_parquet(FEATURE_MATRIX_PATH)
    if "target_date" not in frame.columns or "target_tmax_c" not in frame.columns:
        raise RuntimeError("Feature matrix must include target_date and target_tmax_c")
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"].notna()].copy()
    frame = frame[frame["target_date"] < CONFIRMATION_START].copy()
    require_no_confirmation_dates(frame["target_date"], context="0085 feature matrix")
    return frame.sort_values("target_date").reset_index(drop=True)


def load_residual_frame() -> pd.DataFrame:
    if not INPUT_0084_TOP_PATH.exists():
        raise FileNotFoundError(f"Missing 0084 top predictions: {INPUT_0084_TOP_PATH}")
    frame = pd.read_csv(INPUT_0084_TOP_PATH)
    required = {
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "forecast_max_c",
        "candidate_prediction_c",
        "frame_segment",
        "season",
        "era_bucket",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"0084 top predictions missing required columns: {sorted(missing)}")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"].notna()].copy()
    frame = frame[frame["target_date"] < CONFIRMATION_START].copy()
    require_no_confirmation_dates(frame["target_date"], context="0085 residual frame")
    for col in ("target_tmax_c", "forecast_max_c", "candidate_prediction_c"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[frame[["target_tmax_c", "forecast_max_c", "candidate_prediction_c"]].notna().all(axis=1)].copy()
    frame["raw_error_c"] = frame["forecast_max_c"] - frame["target_tmax_c"]
    frame["best_error_c"] = frame["candidate_prediction_c"] - frame["target_tmax_c"]
    frame["raw_abs_error_c"] = frame["raw_error_c"].abs()
    frame["best_abs_error_c"] = frame["best_error_c"].abs()
    frame["correction_abs_improvement_c"] = frame["raw_abs_error_c"] - frame["best_abs_error_c"]
    frame["forecast_source_family"] = frame["forecast_source_family"].astype(str)
    frame["frame_segment"] = frame["frame_segment"].astype(str)
    frame["season"] = frame["season"].astype(str)
    frame["era_bucket"] = frame["era_bucket"].astype(str)
    return frame.sort_values("target_date").reset_index(drop=True)


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in NON_FEATURE_EXACT:
            continue
        if column.startswith(NON_FEATURE_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        columns.append(column)
    return columns


def safe_corr(x: pd.Series, y: pd.Series, *, min_rows: int) -> tuple[int, float]:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1)
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < min_rows:
        return int(len(pair)), math.nan
    if pair.iloc[:, 0].nunique(dropna=True) < 2 or pair.iloc[:, 1].nunique(dropna=True) < 2:
        return int(len(pair)), math.nan
    return int(len(pair)), float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def bucket_signal(feature: pd.Series, target: pd.Series, *, min_rows: int = MIN_BUCKET_ROWS) -> dict[str, object]:
    pair = pd.concat([pd.to_numeric(feature, errors="coerce"), pd.to_numeric(target, errors="coerce")], axis=1)
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < min_rows * 3 or pair.iloc[:, 0].nunique(dropna=True) < 5:
        return {
            "bucket_rows": int(len(pair)),
            "bucket_count": 0,
            "bucket_target_min": math.nan,
            "bucket_target_max": math.nan,
            "bucket_target_spread": math.nan,
        }
    try:
        buckets = pd.qcut(pair.iloc[:, 0], q=5, duplicates="drop")
    except ValueError:
        return {
            "bucket_rows": int(len(pair)),
            "bucket_count": 0,
            "bucket_target_min": math.nan,
            "bucket_target_max": math.nan,
            "bucket_target_spread": math.nan,
        }
    grouped = pair.iloc[:, 1].groupby(buckets, observed=True)
    stats = grouped.agg(["count", "mean"]).reset_index(drop=True)
    stats = stats[stats["count"].ge(min_rows)].copy()
    if len(stats) < 2:
        return {
            "bucket_rows": int(len(pair)),
            "bucket_count": int(len(stats)),
            "bucket_target_min": math.nan,
            "bucket_target_max": math.nan,
            "bucket_target_spread": math.nan,
        }
    low = float(stats["mean"].min())
    high = float(stats["mean"].max())
    return {
        "bucket_rows": int(len(pair)),
        "bucket_count": int(len(stats)),
        "bucket_target_min": low,
        "bucket_target_max": high,
        "bucket_target_spread": float(high - low),
    }


def non_null_date_range(frame: pd.DataFrame, column: str) -> tuple[int, str, str]:
    series = pd.to_numeric(frame[column], errors="coerce")
    valid = frame.loc[series.notna(), "target_date"]
    if valid.empty:
        return 0, "", ""
    return int(len(valid)), date_text(valid.min()), date_text(valid.max())


def stability_label(train_corr: float, eval_corr: float) -> str:
    if not np.isfinite(train_corr) or not np.isfinite(eval_corr):
        return "insufficient"
    if abs(train_corr) < 0.03 and abs(eval_corr) < 0.03:
        return "weak_both_periods"
    if math.copysign(1.0, train_corr) == math.copysign(1.0, eval_corr):
        return "same_sign"
    return "sign_flip"


def priority_score(row: dict[str, object]) -> float:
    components = [
        abs(float(row.get("corr_best_error", 0.0) or 0.0)) * 1.0,
        abs(float(row.get("corr_best_abs_error", 0.0) or 0.0)) * 1.25,
        abs(float(row.get("corr_correction_abs_improvement", 0.0) or 0.0)) * 1.25,
        min(abs(float(row.get("bucket_best_abs_error_spread", 0.0) or 0.0)) / 2.0, 1.0) * 0.75,
        min(abs(float(row.get("bucket_improvement_spread", 0.0) or 0.0)) / 2.0, 1.0) * 0.75,
    ]
    if row.get("target_stability") == "same_sign":
        components.append(0.05)
    return float(sum(value for value in components if math.isfinite(value)))


def build_joined_frame(features: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    residual_cols = [
        "target_date",
        "forecast_source_family",
        "forecast_max_c",
        "candidate_prediction_c",
        "raw_error_c",
        "best_error_c",
        "raw_abs_error_c",
        "best_abs_error_c",
        "correction_abs_improvement_c",
        "frame_segment",
        "season",
        "era_bucket",
    ]
    joined = residuals[residual_cols].merge(features, on="target_date", how="left", suffixes=("", "_feature"))
    if "target_tmax_c_feature" in joined.columns:
        joined = joined.drop(columns=["target_tmax_c_feature"])
    require_no_confirmation_dates(joined["target_date"], context="0085 joined official-feature frame")
    return joined


def rank_features(features: pd.DataFrame, joined: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    train = features[(features["target_date"] >= pd.Timestamp("1949-01-01")) & (features["target_date"] <= pd.Timestamp("1999-12-31"))]
    eval_frame = features[(features["target_date"] >= pd.Timestamp("2000-01-01")) & (features["target_date"] < CONFIRMATION_START)]
    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        full_rows, first_date, last_date = non_null_date_range(features, feature)
        if full_rows < MIN_FULL_HISTORY_ROWS:
            continue
        train_n, train_corr = safe_corr(train[feature], train["target_tmax_c"], min_rows=MIN_FULL_HISTORY_ROWS)
        eval_n, eval_corr = safe_corr(eval_frame[feature], eval_frame["target_tmax_c"], min_rows=MIN_SEGMENT_ROWS)
        official_n, corr_raw_error = safe_corr(joined[feature], joined["raw_error_c"], min_rows=MIN_OFFICIAL_ROWS)
        _n, corr_best_error = safe_corr(joined[feature], joined["best_error_c"], min_rows=MIN_OFFICIAL_ROWS)
        _n, corr_raw_abs = safe_corr(joined[feature], joined["raw_abs_error_c"], min_rows=MIN_OFFICIAL_ROWS)
        _n, corr_best_abs = safe_corr(joined[feature], joined["best_abs_error_c"], min_rows=MIN_OFFICIAL_ROWS)
        _n, corr_improvement = safe_corr(
            joined[feature],
            joined["correction_abs_improvement_c"],
            min_rows=MIN_OFFICIAL_ROWS,
        )
        best_abs_bucket = bucket_signal(joined[feature], joined["best_abs_error_c"])
        improvement_bucket = bucket_signal(joined[feature], joined["correction_abs_improvement_c"])
        row: dict[str, object] = {
            "feature": feature,
            "family": classify_feature_family(feature),
            "station_ids": station_ids_in_feature(feature),
            "full_history_rows": full_rows,
            "first_non_null_date": first_date,
            "last_non_null_date": last_date,
            "train_1949_1999_rows": train_n,
            "corr_target_1949_1999": train_corr,
            "eval_2000_2023_rows": eval_n,
            "corr_target_2000_2023": eval_corr,
            "target_stability": stability_label(train_corr, eval_corr),
            "official_rows": official_n,
            "corr_raw_error": corr_raw_error,
            "corr_best_error": corr_best_error,
            "corr_raw_abs_error": corr_raw_abs,
            "corr_best_abs_error": corr_best_abs,
            "corr_correction_abs_improvement": corr_improvement,
            "bucket_best_abs_error_spread": best_abs_bucket["bucket_target_spread"],
            "bucket_improvement_spread": improvement_bucket["bucket_target_spread"],
            "bucket_best_abs_error_rows": best_abs_bucket["bucket_rows"],
            "bucket_best_abs_error_bucket_count": best_abs_bucket["bucket_count"],
        }
        row["priority_score"] = priority_score(row)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["priority_score", "official_rows"], ascending=[False, False]).reset_index(drop=True)


def segment_rankings(joined: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    candidate_features = rankings.head(150)["feature"].tolist()
    rows: list[dict[str, object]] = []
    segments = [
        ("old_frame", joined["frame_segment"].eq("current_0081_frame")),
        ("newly_available", joined["frame_segment"].eq("newly_available_official_frame")),
        ("press", joined["forecast_source_family"].eq("press_archive")),
        ("rss", joined["forecast_source_family"].eq("rss_archive")),
    ]
    for segment_name, mask in segments:
        segment = joined[mask].copy()
        if len(segment) < MIN_SEGMENT_ROWS:
            continue
        for feature in candidate_features:
            n, corr_abs = safe_corr(segment[feature], segment["best_abs_error_c"], min_rows=MIN_SEGMENT_ROWS)
            _n, corr_improvement = safe_corr(
                segment[feature],
                segment["correction_abs_improvement_c"],
                min_rows=MIN_SEGMENT_ROWS,
            )
            if not np.isfinite(corr_abs) and not np.isfinite(corr_improvement):
                continue
            rows.append(
                {
                    "segment": segment_name,
                    "feature": feature,
                    "family": classify_feature_family(feature),
                    "station_ids": station_ids_in_feature(feature),
                    "rows": n,
                    "corr_best_abs_error": corr_abs,
                    "corr_correction_abs_improvement": corr_improvement,
                    "segment_priority": abs(corr_abs if np.isfinite(corr_abs) else 0.0)
                    + abs(corr_improvement if np.isfinite(corr_improvement) else 0.0),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.sort_values(["segment", "segment_priority"], ascending=[True, False])
        .groupby("segment", observed=True)
        .head(30)
        .reset_index(drop=True)
    )


def family_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    grouped = rankings.groupby("family", observed=True)
    rows: list[dict[str, object]] = []
    for family, group in grouped:
        best = group.sort_values("priority_score", ascending=False).iloc[0]
        rows.append(
            {
                "family": str(family),
                "feature_count": int(len(group)),
                "best_priority_score": float(best["priority_score"]),
                "best_feature": str(best["feature"]),
                "best_corr_best_abs_error": float(best["corr_best_abs_error"]),
                "best_corr_improvement": float(best["corr_correction_abs_improvement"]),
                "same_sign_target_features": int(group["target_stability"].eq("same_sign").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("best_priority_score", ascending=False).reset_index(drop=True)


def station_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in rankings.iterrows():
        station_ids = [station for station in str(row["station_ids"]).split(",") if station]
        for station_id in station_ids:
            rows.append(
                {
                    "station_id": station_id,
                    "feature": row["feature"],
                    "family": row["family"],
                    "priority_score": row["priority_score"],
                    "corr_best_abs_error": row["corr_best_abs_error"],
                    "corr_correction_abs_improvement": row["corr_correction_abs_improvement"],
                }
            )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        return pd.DataFrame(columns=["station_id", "feature_count", "best_feature"])
    out_rows: list[dict[str, object]] = []
    for station_id, group in expanded.groupby("station_id", observed=True):
        best = group.sort_values("priority_score", ascending=False).iloc[0]
        out_rows.append(
            {
                "station_id": str(station_id),
                "feature_count": int(len(group)),
                "best_priority_score": float(best["priority_score"]),
                "best_feature": str(best["feature"]),
                "best_family": str(best["family"]),
                "best_corr_best_abs_error": float(best["corr_best_abs_error"]),
                "best_corr_improvement": float(best["corr_correction_abs_improvement"]),
            }
        )
    return pd.DataFrame(out_rows).sort_values("best_priority_score", ascending=False).reset_index(drop=True)


def bucket_profiles(joined: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in rankings.head(20)["feature"].tolist():
        pair = joined[
            [
                feature,
                "best_abs_error_c",
                "raw_abs_error_c",
                "correction_abs_improvement_c",
                "forecast_source_family",
                "frame_segment",
            ]
        ].copy()
        pair[feature] = pd.to_numeric(pair[feature], errors="coerce")
        pair = pair.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, "best_abs_error_c"])
        if len(pair) < MIN_BUCKET_ROWS * 3 or pair[feature].nunique(dropna=True) < 5:
            continue
        try:
            pair["feature_bucket"] = pd.qcut(pair[feature], q=5, duplicates="drop").astype(str)
        except ValueError:
            continue
        grouped = pair.groupby("feature_bucket", observed=True)
        for bucket, group in grouped:
            if len(group) < MIN_BUCKET_ROWS:
                continue
            rows.append(
                {
                    "feature": feature,
                    "family": classify_feature_family(feature),
                    "station_ids": station_ids_in_feature(feature),
                    "feature_bucket": str(bucket),
                    "rows": int(len(group)),
                    "feature_min": float(group[feature].min()),
                    "feature_max": float(group[feature].max()),
                    "best_abs_error_mean": float(group["best_abs_error_c"].mean()),
                    "raw_abs_error_mean": float(group["raw_abs_error_c"].mean()),
                    "correction_abs_improvement_mean": float(group["correction_abs_improvement_c"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_summary(
    *,
    generated_at: str,
    features: pd.DataFrame,
    joined: pd.DataFrame,
    rankings: pd.DataFrame,
    families: pd.DataFrame,
    stations: pd.DataFrame,
) -> dict[str, object]:
    feature_dates = pd.to_datetime(features["target_date"], errors="coerce")
    official_dates = pd.to_datetime(joined["target_date"], errors="coerce")
    top = rankings.iloc[0].to_dict() if not rankings.empty else {}
    return {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "feature_matrix_path": str(FEATURE_MATRIX_PATH),
        "residual_input_path": str(INPUT_0084_TOP_PATH),
        "feature_matrix_rows": int(len(features)),
        "feature_matrix_first_date": date_text(feature_dates.min()),
        "feature_matrix_last_date": date_text(feature_dates.max()),
        "joined_official_rows": int(len(joined)),
        "joined_official_first_date": date_text(official_dates.min()),
        "joined_official_last_date": date_text(official_dates.max()),
        "ranked_feature_count": int(len(rankings)),
        "family_count": int(len(families)),
        "station_count": int(len(stations)),
        "top_feature": top.get("feature", ""),
        "top_feature_family": top.get("family", ""),
        "top_feature_station_ids": top.get("station_ids", ""),
        "top_priority_score": top.get("priority_score"),
        "top_corr_best_abs_error": top.get("corr_best_abs_error"),
        "top_corr_improvement": top.get("corr_correction_abs_improvement"),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "long_history_feature_station_residual_bridge_complete",
        "next_recommended_task": (
            "Run 0086 to turn the top 0085 feature/station residual signals into guarded past-only local "
            "specialists on the expanded official frame, starting with the highest-ranked upper-air, ISD station, "
            "marine/daily-climate, and target-memory failure regimes. Keep 2024+ sealed."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    rankings: pd.DataFrame,
    families: pd.DataFrame,
    stations: pd.DataFrame,
    segments: pd.DataFrame,
    buckets: pd.DataFrame,
) -> str:
    return f"""# 0085 Long-History Feature And Station Residual Bridge

Generated: `{generated_at}`

## Purpose

The current official-anchor system is not close to the desired 0.45 MAE. `0085` starts the deeper bridge the user requested: connect the long-history feature matrix to the expanded official forecast residual frame, then rank every usable station, upper-air, daily climate, target-memory, marine-adjacent, and calendar feature by how much information it appears to carry about remaining Tmax forecast error.

This is not a model-training run. It is a leakage-controlled discovery layer that tells us where to build the next guarded specialists. The critical distinction is that the feature matrix has long history, while official forecast residuals only exist on the current official forecast archive segments. Therefore the analysis reports both long-history target stability and official-frame residual signal.

## Inputs And Date Ranges

| Input | Rows | Date range |
|---|---:|---|
| Long-history feature matrix | `{summary['feature_matrix_rows']}` | `{summary['feature_matrix_first_date']}` to `{summary['feature_matrix_last_date']}` |
| Joined official residual frame | `{summary['joined_official_rows']}` | `{summary['joined_official_first_date']}` to `{summary['joined_official_last_date']}` |

The joined official residual frame uses only target dates before `{CONFIRMATION_START.date()}`. No 2024+ confirmation rows are used.

## What Was Measured

For every numeric feature with enough history, this run computes:

- full-history availability and first/last non-null dates;
- target Tmax correlation on `1949-1999`;
- target Tmax correlation on `2000-2023`;
- whether the target relationship keeps the same sign across those periods;
- correlation with raw official forecast error;
- correlation with the `0084` best residual error;
- correlation with raw and corrected absolute error;
- correlation with correction improvement, where positive means the official correction helped;
- quintile bucket spreads for corrected absolute error and correction improvement;
- station IDs embedded in feature names.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Ranked features | `{summary['ranked_feature_count']}` |
| Feature families | `{summary['family_count']}` |
| Stations with ranked features | `{summary['station_count']}` |
| Top feature | `{summary['top_feature']}` |
| Top feature family | `{summary['top_feature_family']}` |
| Top feature station IDs | `{summary['top_feature_station_ids']}` |
| Top priority score | `{summary['top_priority_score']}` |
| Top corr best abs error | `{summary['top_corr_best_abs_error']}` |
| Top corr correction improvement | `{summary['top_corr_improvement']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Top Residual Features

{markdown_table(rankings.head(25), max_rows=25)}

## Feature-Family Summary

{markdown_table(families, max_rows=20)}

## Station Summary

{markdown_table(stations.head(30), max_rows=30)}

## Segment-Specific Feature Rankings

{markdown_table(segments, max_rows=40)}

## Top Feature Bucket Profiles

{markdown_table(buckets.head(40), max_rows=40)}

## Interpretation

The most useful features are those that satisfy two conditions at once: they have long historical coverage, and they explain residual behavior on the official forecast rows. A feature that only correlates with target Tmax is useful for weather state, but not enough. A feature that correlates with absolute residual error or correction improvement tells us where the official forecast is likely weak, where the 0083/0084 correction helps, and where a specialist may be worth building.

The next step should not be a generic bigger model. The correct next step is to take the top ranked features by family and station, build guarded past-only specialists for the source/era regimes where they are strongest, and reject any specialist that does not survive source, era, season, old-frame, and newly-available-frame checks.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], rankings: pd.DataFrame, families: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0085_long_history_feature_station_residual_bridge.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Long-history matrix | `{summary['feature_matrix_rows']}` rows, `{summary['feature_matrix_first_date']}` to `{summary['feature_matrix_last_date']}` | Joined to residual frame |
| Official residual frame | `{summary['joined_official_rows']}` rows, `{summary['joined_official_first_date']}` to `{summary['joined_official_last_date']}` | Pre-2024 only |
| Ranked features | `{summary['ranked_feature_count']}` | Residual information screen |
| Top feature | `{summary['top_feature']}` | `{summary['top_feature_family']}` |
| Station count | `{summary['station_count']}` | Features with station IDs |
| Leakage | `0` 2024+ rows | PASS |

Top residual features:

{markdown_table(rankings.head(8), max_rows=8)}

Feature-family summary:

{markdown_table(families.head(8), max_rows=8)}

Interpretation: `0085` is the bridge from official forecast correction into the full long-history station/weather feature universe. The next work should convert the strongest residual signals into guarded local specialists, not a broad black-box model.
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0085 Long-History Feature And Station Residual Bridge",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=(
            "Implement `0086_guarded_long_history_residual_specialists`: use the top 0085 feature/station signals "
            "to build guarded past-only local specialists on the expanded official frame, with explicit source, "
            "era, season, old-frame, and newly-available-frame acceptance gates. Keep 2024+ sealed."
        ),
    )


def run() -> dict[str, object]:
    generated_at = now_utc()
    features = load_feature_matrix()
    residuals = load_residual_frame()
    joined = build_joined_frame(features, residuals)
    feature_cols = numeric_feature_columns(features)
    rankings = rank_features(features, joined, feature_cols)
    families = family_summary(rankings)
    stations = station_summary(rankings)
    segments = segment_rankings(joined, rankings)
    buckets = bucket_profiles(joined, rankings)
    summary = build_summary(
        generated_at=generated_at,
        features=features,
        joined=joined,
        rankings=rankings,
        families=families,
        stations=stations,
    )

    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "feature_residual_rankings.csv", rankings)
    write_csv(artifacts / "family_summary.csv", families)
    write_csv(artifacts / "station_summary.csv", stations)
    write_csv(artifacts / "segment_feature_rankings.csv", segments)
    write_csv(artifacts / "top_bucket_profiles.csv", buckets)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "long_history_feature_station_residual_bridge_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            rankings=rankings,
            families=families,
            stations=stations,
            segments=segments,
            buckets=buckets,
        ),
    )
    update_milestones(summary, rankings, families)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Rank long-history HKG Tmax features by official residual information gain."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
