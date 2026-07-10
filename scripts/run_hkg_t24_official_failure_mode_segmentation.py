from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    feature_family,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_multistation_attribute_information_gain import (  # noqa: E402
    analysis_column_allowed,
    build_official_feature_frame,
)
from scripts.run_hkg_t24_regime_expert_factory import (  # noqa: E402
    add_composite_features,
)

FOLDER_NAME = "0021_failure_modes"
MIN_SCAN_ROWS = 500
MIN_BUCKET_ROWS = 50
MIN_SEGMENT_ROWS = 45
TOP_PAIR_FEATURES = 24
TOP_WORST_EVENTS = 150


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 80) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def build_failure_frame() -> pd.DataFrame:
    frame = add_composite_features(build_official_feature_frame())
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(frame["target_date"], context="official failure-mode frame")
    if "season" not in frame.columns and "month" in frame.columns:
        frame["season"] = frame["month"].map(season_name)
    frame["official_error_c"] = pd.to_numeric(frame["forecast_max_c"], errors="coerce") - pd.to_numeric(
        frame["target_tmax_c"],
        errors="coerce",
    )
    frame["official_abs_error_c"] = frame["official_error_c"].abs()
    return assign_failure_flags(frame)


def season_name(month: object) -> str:
    value = int(month)
    if value in (12, 1, 2):
        return "DJF"
    if value in (3, 4, 5):
        return "MAM"
    if value in (6, 7, 8):
        return "JJA"
    return "SON"


def assign_failure_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    abs_error = pd.to_numeric(out["official_abs_error_c"], errors="coerce")
    error = pd.to_numeric(out["official_error_c"], errors="coerce")
    abs_p80 = float(abs_error.quantile(0.80))
    abs_p90 = float(abs_error.quantile(0.90))
    under_p10 = float(error.quantile(0.10))
    over_p90 = float(error.quantile(0.90))
    out["severe_abs_error"] = abs_error >= abs_p80
    out["extreme_abs_error"] = abs_error >= abs_p90
    out["severe_underprediction"] = error <= under_p10
    out["severe_overprediction"] = error >= over_p90
    out["error_direction"] = np.select(
        [error < 0, error > 0],
        ["underpredicted_hotter_actual", "overpredicted_cooler_actual"],
        default="exact",
    )
    out["failure_mode"] = np.select(
        [
            out["severe_underprediction"],
            out["severe_overprediction"],
            out["severe_abs_error"],
        ],
        ["severe_underprediction", "severe_overprediction", "large_abs_error_other"],
        default="routine",
    )
    out.attrs["failure_thresholds"] = {
        "abs_error_p80_c": abs_p80,
        "abs_error_p90_c": abs_p90,
        "underprediction_error_p10_c": under_p10,
        "overprediction_error_p90_c": over_p90,
    }
    return out


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    forbidden = {
        "official_error_c",
        "official_abs_error_c",
        "target_tmax_c",
        "forecast_max_c",
        "severe_abs_error",
        "extreme_abs_error",
        "severe_underprediction",
        "severe_overprediction",
    }
    return [
        column
        for column in frame.columns
        if column not in forbidden
        and analysis_column_allowed(column)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def standardized_mean_diff(feature: pd.Series, flag: pd.Series) -> tuple[int, float, float, float]:
    values = pd.to_numeric(feature, errors="coerce")
    flags = flag.astype(bool)
    pair = pd.DataFrame({"value": values, "flag": flags}).dropna()
    if pair.empty or pair["flag"].nunique() < 2:
        return 0, math.nan, math.nan, math.nan
    flagged = pair[pair["flag"]]["value"]
    baseline = pair[~pair["flag"]]["value"]
    if flagged.empty or baseline.empty:
        return 0, math.nan, math.nan, math.nan
    scale = float(pair["value"].std(ddof=0))
    diff = float(flagged.mean() - baseline.mean())
    std_diff = math.nan if scale == 0.0 or not np.isfinite(scale) else float(diff / scale)
    return int(len(pair)), diff, float(flagged.median() - baseline.median()), std_diff


def feature_contrast_scan(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in numeric_feature_columns(frame):
        values = pd.to_numeric(frame[feature], errors="coerce")
        n = int(values.notna().sum())
        if n < MIN_SCAN_ROWS or values.nunique(dropna=True) <= 2:
            continue
        for flag_column, label in [
            ("severe_abs_error", "top20_abs_error"),
            ("extreme_abs_error", "top10_abs_error"),
            ("severe_underprediction", "worst_underprediction"),
            ("severe_overprediction", "worst_overprediction"),
        ]:
            contrast_n, mean_diff, median_diff, std_diff = standardized_mean_diff(values, frame[flag_column])
            if contrast_n < MIN_SCAN_ROWS or not np.isfinite(std_diff):
                continue
            rows.append(
                {
                    "feature": feature,
                    "family": feature_family(feature),
                    "failure_slice": label,
                    "n": contrast_n,
                    "mean_diff_flag_minus_rest": mean_diff,
                    "median_diff_flag_minus_rest": median_diff,
                    "standardized_mean_diff": std_diff,
                    "priority": abs(std_diff),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["priority", "feature"], ascending=[False, True]).reset_index(drop=True)


def quantile_bucket(values: pd.Series, bins: int, *, prefix: str = "q") -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < bins * MIN_BUCKET_ROWS or numeric.nunique(dropna=True) < bins:
        return pd.Series(["insufficient"] * len(values), index=values.index)
    ranked = numeric.rank(method="first")
    labels = [f"{prefix}{index + 1}" for index in range(bins)]
    return pd.qcut(ranked, bins, labels=labels).astype(str).where(numeric.notna(), "missing")


def feature_bucket_failure_lifts(frame: pd.DataFrame) -> pd.DataFrame:
    global_mae = float(frame["official_abs_error_c"].mean())
    global_under = float(frame["severe_underprediction"].mean())
    global_over = float(frame["severe_overprediction"].mean())
    global_abs = float(frame["severe_abs_error"].mean())
    rows: list[dict[str, object]] = []
    for feature in numeric_feature_columns(frame):
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) < MIN_SCAN_ROWS or values.nunique(dropna=True) <= 2:
            continue
        buckets = quantile_bucket(values, 5)
        if buckets.eq("insufficient").all():
            continue
        work = frame[["target_date", "official_error_c", "official_abs_error_c", "severe_underprediction", "severe_overprediction", "severe_abs_error"]].copy()
        work["bucket"] = buckets
        work["feature_value"] = values
        for bucket, group in work.groupby("bucket", observed=True, dropna=False):
            if bucket in {"missing", "insufficient"} or len(group) < MIN_BUCKET_ROWS:
                continue
            mae = float(group["official_abs_error_c"].mean())
            under_rate = float(group["severe_underprediction"].mean())
            over_rate = float(group["severe_overprediction"].mean())
            severe_abs_rate = float(group["severe_abs_error"].mean())
            bias = float(group["official_error_c"].mean())
            rows.append(
                {
                    "feature": feature,
                    "family": feature_family(feature),
                    "bucket": bucket,
                    "n": int(len(group)),
                    "feature_min": float(group["feature_value"].min()),
                    "feature_max": float(group["feature_value"].max()),
                    "mae_c": mae,
                    "mae_lift_vs_global_c": mae - global_mae,
                    "bias_c": bias,
                    "severe_abs_rate": severe_abs_rate,
                    "severe_abs_rate_lift": severe_abs_rate - global_abs,
                    "severe_under_rate": under_rate,
                    "severe_under_rate_lift": under_rate - global_under,
                    "severe_over_rate": over_rate,
                    "severe_over_rate_lift": over_rate - global_over,
                    "priority": abs(mae - global_mae)
                    + abs(bias) * 0.25
                    + abs(severe_abs_rate - global_abs) * 2.0
                    + abs(under_rate - global_under)
                    + abs(over_rate - global_over),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("priority", ascending=False).reset_index(drop=True)


def aggregate_error_by(frame: pd.DataFrame, group_cols: list[str], *, min_rows: int = 1) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, dropna=False, observed=True):
        if len(group) < min_rows:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        error = pd.to_numeric(group["official_error_c"], errors="coerce")
        abs_error = error.abs()
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "n": int(len(group)),
                "mae_c": float(abs_error.mean()),
                "rmse_c": float(np.sqrt(np.mean(np.square(error)))),
                "bias_c": float(error.mean()),
                "p90_abs_error_c": float(abs_error.quantile(0.90)),
                "severe_abs_rate": float(group["severe_abs_error"].mean()),
                "severe_under_rate": float(group["severe_underprediction"].mean()),
                "severe_over_rate": float(group["severe_overprediction"].mean()),
                "first_date": str(group["target_date"].min().date()),
                "last_date": str(group["target_date"].max().date()),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["mae_c", "n"], ascending=[False, False]).reset_index(drop=True)


def choose_pair_features(contrast: pd.DataFrame, bucket_lifts: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for source in [contrast, bucket_lifts]:
        if source.empty:
            continue
        for feature in source["feature"].to_list():
            if feature not in features:
                features.append(feature)
            if len(features) >= TOP_PAIR_FEATURES:
                return features
    return features


def two_feature_segments(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    global_mae = float(frame["official_abs_error_c"].mean())
    bucket_cache = {feature: quantile_bucket(frame[feature], 3) for feature in features}
    for feature_a, feature_b in combinations(features, 2):
        buckets_a = bucket_cache[feature_a]
        buckets_b = bucket_cache[feature_b]
        if buckets_a.eq("insufficient").all() or buckets_b.eq("insufficient").all():
            continue
        work = frame[["target_date", "forecast_source_family", "official_error_c", "official_abs_error_c", "severe_abs_error", "severe_underprediction", "severe_overprediction"]].copy()
        work["bucket_a"] = buckets_a
        work["bucket_b"] = buckets_b
        work = work[~work["bucket_a"].isin(["missing", "insufficient"])]
        work = work[~work["bucket_b"].isin(["missing", "insufficient"])]
        for (bucket_a, bucket_b), group in work.groupby(["bucket_a", "bucket_b"], observed=True):
            if len(group) < MIN_SEGMENT_ROWS:
                continue
            mae = float(group["official_abs_error_c"].mean())
            bias = float(group["official_error_c"].mean())
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "family_a": feature_family(feature_a),
                    "family_b": feature_family(feature_b),
                    "bucket_a": bucket_a,
                    "bucket_b": bucket_b,
                    "n": int(len(group)),
                    "mae_c": mae,
                    "mae_lift_vs_global_c": mae - global_mae,
                    "bias_c": bias,
                    "severe_abs_rate": float(group["severe_abs_error"].mean()),
                    "severe_under_rate": float(group["severe_underprediction"].mean()),
                    "severe_over_rate": float(group["severe_overprediction"].mean()),
                    "first_date": str(group["target_date"].min().date()),
                    "last_date": str(group["target_date"].max().date()),
                    "priority": (mae - global_mae) + abs(bias) * 0.20,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["priority", "n"], ascending=[False, False]).reset_index(drop=True)


def diagnostic_feature_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    candidates = [
        "forecast_range_c",
        "forecast_midpoint_c",
        "daily_waglan_island_sea_temperature_lag7_roll7",
        "forecast_minus_waglan_sea_temp_roll7_c",
        "ua_mse_1000hpa_kj_kg",
        "ua_theta_e_1000hpa_k",
        "ua_dewpoint_1000hpa_c",
        "isd_morning_to_midday_temp_rise_c",
        "isd_temp_dewpoint_spread_mean_c",
        "daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7",
        "daily_hong_kong_observatory_daily_rainfall_lag7_roll7",
        "spread_air_592870_minus_596730_c",
    ]
    thresholds: dict[str, float] = {}
    for feature in candidates:
        if feature in frame.columns:
            values = pd.to_numeric(frame[feature], errors="coerce")
            if values.notna().sum() >= MIN_SCAN_ROWS:
                thresholds[f"{feature}__q20"] = float(values.quantile(0.20))
                thresholds[f"{feature}__q80"] = float(values.quantile(0.80))
    return thresholds


def row_feature_is_high(row: dict[str, object], thresholds: dict[str, float], feature: str) -> bool:
    value = row.get(feature, math.nan)
    return pd.notna(value) and f"{feature}__q80" in thresholds and float(value) >= thresholds[f"{feature}__q80"]


def row_feature_is_low(row: dict[str, object], thresholds: dict[str, float], feature: str) -> bool:
    value = row.get(feature, math.nan)
    return pd.notna(value) and f"{feature}__q20" in thresholds and float(value) <= thresholds[f"{feature}__q20"]


def build_archetype_labels(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    labels: list[str] = []
    for row in frame.to_dict("records"):
        parts: list[str] = []
        month_value = row.get("month")
        month = int(month_value) if pd.notna(month_value) else 0
        if month in (3, 4, 5):
            parts.append("spring_transition")
        if month in (6, 7, 8):
            parts.append("summer")
        if row_feature_is_high(row, thresholds, "forecast_range_c"):
            parts.append("wide_official_range")
        if row_feature_is_high(row, thresholds, "ua_mse_1000hpa_kj_kg") or row_feature_is_high(row, thresholds, "ua_theta_e_1000hpa_k"):
            parts.append("high_upper_heat")
        if row_feature_is_high(row, thresholds, "forecast_minus_waglan_sea_temp_roll7_c"):
            parts.append("forecast_far_above_sea")
        if row_feature_is_high(row, thresholds, "isd_morning_to_midday_temp_rise_c"):
            parts.append("strong_morning_warmup")
        if row_feature_is_low(row, thresholds, "isd_temp_dewpoint_spread_mean_c"):
            parts.append("humid_surface")
        if row_feature_is_high(row, thresholds, "daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7"):
            parts.append("cloudy_recent")
        if row_feature_is_high(row, thresholds, "daily_hong_kong_observatory_daily_rainfall_lag7_roll7"):
            parts.append("rainy_recent")
        if row_feature_is_high(row, thresholds, "spread_air_592870_minus_596730_c") or row_feature_is_low(row, thresholds, "spread_air_592870_minus_596730_c"):
            parts.append("station_spread_extreme")
        labels.append("+".join(parts[:4]) if parts else "unclassified")
    return pd.Series(labels, index=frame.index)


def archetype_summary(frame: pd.DataFrame) -> pd.DataFrame:
    thresholds = diagnostic_feature_thresholds(frame)
    work = frame.copy()
    work["archetype"] = build_archetype_labels(work, thresholds)
    summary = aggregate_error_by(work, ["archetype"], min_rows=30)
    if summary.empty:
        return summary
    return summary.sort_values(["mae_c", "severe_abs_rate"], ascending=[False, False]).reset_index(drop=True)


def worst_events(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "forecast_min_c",
        "forecast_max_c",
        "forecast_range_c",
        "forecast_midpoint_c",
        "official_error_c",
        "official_abs_error_c",
        "failure_mode",
        "season",
        "month",
        "daily_waglan_island_sea_temperature_lag7_roll7",
        "ua_mse_1000hpa_kj_kg",
        "ua_theta_e_1000hpa_k",
        "isd_morning_to_midday_temp_rise_c",
        "spread_air_592870_minus_596730_c",
        "daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7",
        "daily_hong_kong_observatory_daily_rainfall_lag7_roll7",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].sort_values("official_abs_error_c", ascending=False).head(TOP_WORST_EVENTS).reset_index(drop=True)


def write_outputs(
    *,
    frame: pd.DataFrame,
    contrasts: pd.DataFrame,
    bucket_lifts: pd.DataFrame,
    segments: pd.DataFrame,
    grouped: dict[str, pd.DataFrame],
    archetypes: pd.DataFrame,
    worst: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    write_csv(artifacts / "feature_contrasts.csv", contrasts)
    write_csv(artifacts / "bucket_lifts.csv", bucket_lifts)
    write_csv(artifacts / "two_feature_segments.csv", segments)
    write_csv(artifacts / "archetypes.csv", archetypes)
    write_csv(artifacts / "worst_events.csv", worst)
    for name, table in grouped.items():
        write_csv(artifacts / f"{name}.csv", table)

    best_contrast = contrasts.iloc[0] if not contrasts.empty else None
    best_bucket = bucket_lifts.iloc[0] if not bucket_lifts.empty else None
    worst_segment = segments.iloc[0] if not segments.empty else None
    worst_archetype = archetypes.iloc[0] if not archetypes.empty else None
    thresholds = frame.attrs.get("failure_thresholds", {})
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "official_mae": float(frame["official_abs_error_c"].mean()),
        "thresholds": thresholds,
        "feature_contrast_rows": int(len(contrasts)),
        "bucket_lift_rows": int(len(bucket_lifts)),
        "two_feature_segment_rows": int(len(segments)),
        "archetype_rows": int(len(archetypes)),
        "best_contrast_feature": "" if best_contrast is None else str(best_contrast["feature"]),
        "best_contrast_slice": "" if best_contrast is None else str(best_contrast["failure_slice"]),
        "best_contrast_std_diff": None if best_contrast is None else float(best_contrast["standardized_mean_diff"]),
        "best_bucket_feature": "" if best_bucket is None else str(best_bucket["feature"]),
        "best_bucket": "" if best_bucket is None else str(best_bucket["bucket"]),
        "best_bucket_mae": None if best_bucket is None else float(best_bucket["mae_c"]),
        "worst_segment_feature_a": "" if worst_segment is None else str(worst_segment["feature_a"]),
        "worst_segment_feature_b": "" if worst_segment is None else str(worst_segment["feature_b"]),
        "worst_segment_mae": None if worst_segment is None else float(worst_segment["mae_c"]),
        "worst_archetype": "" if worst_archetype is None else str(worst_archetype["archetype"]),
        "worst_archetype_mae": None if worst_archetype is None else float(worst_archetype["mae_c"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "official_failure_mode_segmentation_manifest.json", manifest)

    result_lines = []
    if best_contrast is not None:
        result_lines.append(
            f"Strongest single-feature contrast: `{best_contrast['feature']}` for `{best_contrast['failure_slice']}` "
            f"with standardized mean difference `{best_contrast['standardized_mean_diff']:.4f}`."
        )
    if best_bucket is not None:
        result_lines.append(
            f"Highest-priority feature bucket: `{best_bucket['feature']}` `{best_bucket['bucket']}` "
            f"with MAE `{best_bucket['mae_c']:.4f}` and bias `{best_bucket['bias_c']:.4f}`."
        )
    if worst_segment is not None:
        result_lines.append(
            f"Worst two-feature segment: `{worst_segment['feature_a']}` `{worst_segment['bucket_a']}` x "
            f"`{worst_segment['feature_b']}` `{worst_segment['bucket_b']}`, MAE `{worst_segment['mae_c']:.4f}`."
        )
    if worst_archetype is not None:
        result_lines.append(
            f"Worst labeled archetype: `{worst_archetype['archetype']}`, MAE `{worst_archetype['mae_c']:.4f}`."
        )

    readme = f"""# Official Forecast Failure-Mode Segmentation

Generated: `{manifest['generated_at_utc']}`

## What Was Tested

This insight studies where the official HKO Tmax forecast anchor fails hardest. It does not train a deployable model. It ranks single-feature contrasts, feature buckets, two-feature segments, labeled failure archetypes, and the worst individual events.

## Leakage Contract

- All rows are earlier than `{CONFIRMATION_START.date()}`.
- Official forecast rows are the same pre-cutoff rows used by the prior official-anchor screens.
- Failure thresholds are diagnostic and computed only within the pre-2024 research frame.
- No 2024+ labels are touched.
- Segment definitions here are research diagnostics; production use would require fold-local threshold selection.

## Main Results

{" ".join(result_lines)}

## Failure Thresholds

{markdown_table(pd.DataFrame([thresholds]), max_rows=1)}

## Source / Season / Direction Summary

{markdown_table(grouped["source_season_direction"].head(20), max_rows=20)}

## Top Single-Feature Contrasts

{markdown_table(contrasts.head(25), max_rows=25)}

## Top Feature Buckets

{markdown_table(bucket_lifts.head(25), max_rows=25)}

## Worst Two-Feature Segments

{markdown_table(segments.head(25), max_rows=25)}

## Worst Archetypes

{markdown_table(archetypes.head(20), max_rows=20)}

## Interpretation

This screen is intended to reveal sharper regime definitions for the next specialist model. The important question is not whether a bucket is already a model, but whether it identifies a repeatable failure mode: large official bias, high severe-error rate, or a clear under/overprediction direction.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    return manifest


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Official Failure-Mode Segmentation\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_official_failure_mode_segmentation.py`:

- `{FOLDER_NAME}`: official forecast residual failure-mode contrasts, feature buckets, two-feature segments, archetypes, and worst events.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Official MAE | {manifest['official_mae']} |
| Feature contrast rows | {manifest['feature_contrast_rows']} |
| Bucket lift rows | {manifest['bucket_lift_rows']} |
| Two-feature segment rows | {manifest['two_feature_segment_rows']} |
| Worst segment MAE | {manifest['worst_segment_mae']} |
| Worst archetype MAE | {manifest['worst_archetype_mae']} |

Leakage contract: all rows are before `{CONFIRMATION_START.date()}`; this is diagnostic segmentation, not a deployable model.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = build_failure_frame()
    contrasts = feature_contrast_scan(frame)
    bucket_lifts = feature_bucket_failure_lifts(frame)
    pair_features = choose_pair_features(contrasts, bucket_lifts)
    segments = two_feature_segments(frame, pair_features)
    grouped = {
        "source_season_direction": aggregate_error_by(frame, ["forecast_source_family", "season", "error_direction"], min_rows=20),
        "failure_mode_summary": aggregate_error_by(frame, ["failure_mode"], min_rows=1),
        "source_month_summary": aggregate_error_by(frame, ["forecast_source_family", "month"], min_rows=20),
    }
    archetypes = archetype_summary(frame)
    worst = worst_events(frame)
    return write_outputs(
        frame=frame,
        contrasts=contrasts,
        bucket_lifts=bucket_lifts,
        segments=segments,
        grouped=grouped,
        archetypes=archetypes,
        worst=worst,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 official failure-mode segmentation.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
