from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    EVAL_END,
    EVAL_START,
    TRAIN_END,
)
from scripts.run_hkg_t24_station_contribution_atlas import (  # noqa: E402
    MIN_EVAL_ROWS,
    MIN_OFFICIAL_ROWS,
    MIN_TERTILE_ROWS,
    MIN_TRAIN_ROWS,
    STATION_ATTRIBUTES,
    load_official_residuals,
    load_station_day_features,
    load_station_metadata,
    load_target,
    safe_corr,
    tertile_spread,
)

FOLDER_NAME = "0050_station_lag_slope_information_atlas"
LAG_DAYS = (1, 2, 3, 7, 14)
DELTA_DAYS = (1, 2, 3, 7, 14)
ROLLING_WINDOWS = (3, 7, 14)
SLOPE_WINDOWS = (3, 7, 14)
TOP_TIMESERIES_VARIANTS = 10


@dataclass(frozen=True)
class VariantSpec:
    transform: str
    source_attribute: str
    feature_name: str


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def rolling_slope(values: pd.Series, window: int) -> pd.Series:
    if window < 2:
        raise ValueError("rolling slope window must be at least 2")
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())

    def slope(raw: np.ndarray) -> float:
        if len(raw) != window or np.isnan(raw).any():
            return math.nan
        y = raw.astype(float)
        return float(np.dot(x_centered, y - y.mean()) / denominator)

    return values.rolling(window=window, min_periods=window).apply(slope, raw=True)


def station_attribute_variants(series: pd.Series, attribute: str) -> dict[str, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    variants: dict[str, pd.Series] = {"current": numeric}
    for lag in LAG_DAYS:
        variants[f"lag_{lag}d"] = numeric.shift(lag)
    for lag in DELTA_DAYS:
        variants[f"delta_{lag}d"] = numeric - numeric.shift(lag)
    for window in ROLLING_WINDOWS:
        min_periods = max(2, min(window, math.ceil(window / 2)))
        mean = numeric.rolling(window=window, min_periods=min_periods).mean()
        variants[f"rolling_mean_{window}d"] = mean
        variants[f"current_minus_rolling_mean_{window}d"] = numeric - mean
    for window in SLOPE_WINDOWS:
        variants[f"rolling_slope_{window}d"] = rolling_slope(numeric, window)
    return {f"{attribute}__{name}": values for name, values in variants.items()}


def variant_transform_name(feature_name: str) -> str:
    return feature_name.rsplit("__", 1)[-1]


def score_variant(
    *,
    station_rows: pd.DataFrame,
    official_station_rows: pd.DataFrame,
    metadata: dict[str, object],
    station_id: str,
    attribute: str,
    feature_name: str,
    values: pd.Series,
) -> dict[str, object]:
    train_mask = station_rows["target_date"] <= TRAIN_END
    eval_mask = (station_rows["target_date"] >= EVAL_START) & (station_rows["target_date"] <= EVAL_END)
    n_train, corr_train = safe_corr(
        values[train_mask],
        station_rows.loc[train_mask, "target_anomaly_vs_past_doy_c"],
        min_rows=MIN_TRAIN_ROWS,
    )
    n_eval, corr_eval = safe_corr(
        values[eval_mask],
        station_rows.loc[eval_mask, "target_anomaly_vs_past_doy_c"],
        min_rows=MIN_EVAL_ROWS,
    )
    spread = tertile_spread(
        values,
        station_rows["target_anomaly_vs_past_doy_c"],
        train_mask,
        eval_mask,
        min_rows=MIN_TERTILE_ROWS,
    )
    n_official_error = 0
    corr_official_error = math.nan
    n_official_abs = 0
    corr_official_abs = math.nan
    if not official_station_rows.empty and feature_name in official_station_rows.columns:
        n_official_error, corr_official_error = safe_corr(
            official_station_rows[feature_name],
            official_station_rows["official_error_c"],
            min_rows=MIN_OFFICIAL_ROWS,
        )
        n_official_abs, corr_official_abs = safe_corr(
            official_station_rows[feature_name],
            official_station_rows["official_abs_error_c"],
            min_rows=MIN_OFFICIAL_ROWS,
        )
    abs_train = abs(corr_train) if math.isfinite(corr_train) else math.nan
    abs_eval = abs(corr_eval) if math.isfinite(corr_eval) else math.nan
    abs_official = abs(corr_official_error) if math.isfinite(corr_official_error) else math.nan
    spread_value = float(spread["cell_spread"]) if isinstance(spread["cell_spread"], float) else math.nan
    stable_sign = bool(math.isfinite(corr_train) and math.isfinite(corr_eval) and corr_train * corr_eval > 0)
    priority_score = (
        (abs_eval if math.isfinite(abs_eval) else 0.0)
        + 0.35 * (abs_official if math.isfinite(abs_official) else 0.0)
        + 0.08 * min((spread_value if math.isfinite(spread_value) else 0.0) / 3.0, 1.0)
        + (0.04 if stable_sign else 0.0)
    )
    return {
        "station_id": station_id,
        "source_attribute": attribute,
        "transform": variant_transform_name(feature_name),
        "feature_name": feature_name,
        "robust_pre2000_testable": bool(n_train >= MIN_TRAIN_ROWS and n_eval >= MIN_EVAL_ROWS),
        "stable_train_eval_sign": stable_sign,
        "latitude": metadata.get("latitude", math.nan),
        "longitude": metadata.get("longitude", math.nan),
        "elevation_m": metadata.get("elevation_m", math.nan),
        "coordinate_sanity_status": metadata.get("coordinate_sanity_status", ""),
        "distance_to_hko_km": metadata.get("distance_to_hko_km", math.nan),
        "bearing_from_hko_deg": metadata.get("bearing_from_hko_deg", math.nan),
        "first_observed_at_hkt": metadata.get("first_observed_at_hkt", ""),
        "last_observed_at_hkt": metadata.get("last_observed_at_hkt", ""),
        "n_train_pre2000": n_train,
        "corr_train_pre2000_target_anomaly": corr_train,
        "abs_corr_train_pre2000_target_anomaly": abs_train,
        "n_eval_2000_2023": n_eval,
        "corr_eval_2000_2023_target_anomaly": corr_eval,
        "abs_corr_eval_2000_2023_target_anomaly": abs_eval,
        "eval_tertile_target_anomaly_spread_c": spread["cell_spread"],
        "eval_tertile_edges_pre2000": spread["tertile_edges"],
        "eval_tertile_high_cell": spread["high_cell"],
        "eval_tertile_low_cell": spread["low_cell"],
        "n_official_error_corr": n_official_error,
        "corr_official_error": corr_official_error,
        "abs_corr_official_error": abs_official,
        "n_official_abs_error_corr": n_official_abs,
        "corr_official_abs_error": corr_official_abs,
        "abs_corr_official_abs_error": abs(corr_official_abs)
        if math.isfinite(corr_official_abs)
        else math.nan,
        "priority_score": priority_score,
    }


def add_variant_columns(station_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_parts: list[pd.DataFrame] = []
    catalog_rows: list[dict[str, object]] = []
    base_cols = [
        "station_id",
        "local_date",
        "target_date",
        "target_tmax_c",
        "past_doy_count",
        "past_doy_mean_tmax_c",
        "target_anomaly_vs_past_doy_c",
    ]
    for station_id, station_rows in station_frame.groupby("station_id", observed=True):
        station_out = station_rows[base_cols].sort_values("target_date").reset_index(drop=True).copy()
        feature_columns: dict[str, pd.Series] = {}
        for attribute in STATION_ATTRIBUTES:
            variants = station_attribute_variants(station_out.merge(
                station_rows[["target_date", attribute]],
                on="target_date",
                how="left",
                validate="one_to_one",
            )[attribute], attribute)
            for feature_name, values in variants.items():
                feature_columns[feature_name] = values.reset_index(drop=True)
                catalog_rows.append(
                    {
                        "station_id": str(station_id),
                        "source_attribute": attribute,
                        "transform": variant_transform_name(feature_name),
                        "feature_name": feature_name,
                    }
                )
        if feature_columns:
            station_out = pd.concat([station_out, pd.DataFrame(feature_columns)], axis=1)
        out_parts.append(station_out)
    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    return out.sort_values(["target_date", "station_id"]).reset_index(drop=True), pd.DataFrame(catalog_rows)


def build_variant_atlas(
    variant_frame: pd.DataFrame,
    variant_catalog: pd.DataFrame,
    official: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    official_joined = variant_frame.merge(official, on="target_date", how="inner") if not official.empty else pd.DataFrame()
    official_by_station = {
        str(station_id): rows.copy()
        for station_id, rows in official_joined.groupby("station_id", observed=True)
    } if not official_joined.empty else {}
    meta_lookup = metadata.set_index("station_id").to_dict("index")
    records: list[dict[str, object]] = []
    for station_id, station_rows in variant_frame.groupby("station_id", observed=True):
        station_catalog = variant_catalog[variant_catalog["station_id"].astype(str).eq(str(station_id))]
        meta = meta_lookup.get(str(station_id), {})
        for row in station_catalog.itertuples(index=False):
            feature_name = str(row.feature_name)
            if feature_name not in station_rows.columns:
                continue
            records.append(
                score_variant(
                    station_rows=station_rows,
                    official_station_rows=official_by_station.get(str(station_id), pd.DataFrame()),
                    metadata=meta,
                    station_id=str(station_id),
                    attribute=str(row.source_attribute),
                    feature_name=feature_name,
                    values=pd.to_numeric(station_rows[feature_name], errors="coerce"),
                )
            )
    out = pd.DataFrame(records)
    return out.sort_values(
        [
            "priority_score",
            "stable_train_eval_sign",
            "abs_corr_eval_2000_2023_target_anomaly",
            "abs_corr_official_error",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def transformation_summary(variant_atlas: pd.DataFrame) -> pd.DataFrame:
    if variant_atlas.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for transform, group in variant_atlas.groupby("transform", observed=True):
        robust = group[group["robust_pre2000_testable"].astype(bool)]
        leader = robust.iloc[0] if not robust.empty else group.iloc[0]
        rows.append(
            {
                "transform": transform,
                "candidate_rows": int(len(group)),
                "robust_rows": int(len(robust)),
                "stable_sign_rows": int(group["stable_train_eval_sign"].astype(bool).sum()),
                "best_station_id": leader["station_id"],
                "best_source_attribute": leader["source_attribute"],
                "best_feature_name": leader["feature_name"],
                "best_priority_score": float(leader["priority_score"]),
                "best_eval_abs_corr": float(leader["abs_corr_eval_2000_2023_target_anomaly"])
                if math.isfinite(float(leader["abs_corr_eval_2000_2023_target_anomaly"]))
                else math.nan,
                "best_official_abs_corr": float(leader["abs_corr_official_error"])
                if math.isfinite(float(leader["abs_corr_official_error"]))
                else math.nan,
                "best_eval_tertile_spread_c": float(leader["eval_tertile_target_anomaly_spread_c"])
                if math.isfinite(float(leader["eval_tertile_target_anomaly_spread_c"]))
                else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["best_priority_score", "robust_rows"], ascending=[False, False])


def station_transform_rankings(variant_atlas: pd.DataFrame) -> pd.DataFrame:
    if variant_atlas.empty:
        return pd.DataFrame()
    rows = []
    for station_id, group in variant_atlas.groupby("station_id", observed=True):
        robust = group[group["robust_pre2000_testable"].astype(bool)]
        scored = robust if not robust.empty else group
        leader = scored.iloc[0]
        rows.append(
            {
                "station_id": station_id,
                "robust_variant_count": int(len(robust)),
                "stable_sign_count": int(group["stable_train_eval_sign"].astype(bool).sum()),
                "best_feature_name": leader["feature_name"],
                "best_source_attribute": leader["source_attribute"],
                "best_transform": leader["transform"],
                "best_priority_score": float(leader["priority_score"]),
                "best_eval_abs_corr": float(leader["abs_corr_eval_2000_2023_target_anomaly"])
                if math.isfinite(float(leader["abs_corr_eval_2000_2023_target_anomaly"]))
                else math.nan,
                "best_official_abs_corr": float(leader["abs_corr_official_error"])
                if math.isfinite(float(leader["abs_corr_official_error"]))
                else math.nan,
                "distance_to_hko_km": leader["distance_to_hko_km"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["robust_variant_count", "best_priority_score"],
        ascending=[False, False],
        na_position="last",
    )


def top_variant_timeseries(
    variant_frame: pd.DataFrame,
    variant_atlas: pd.DataFrame,
    official: pd.DataFrame,
) -> pd.DataFrame:
    if variant_atlas.empty:
        return pd.DataFrame()
    top = variant_atlas[variant_atlas["robust_pre2000_testable"].astype(bool)].head(TOP_TIMESERIES_VARIANTS)
    if top.empty:
        top = variant_atlas.head(TOP_TIMESERIES_VARIANTS)
    frames: list[pd.DataFrame] = []
    official_keep = official[["target_date", "forecast_source_family", "official_error_c", "official_abs_error_c"]]
    for row in top.itertuples(index=False):
        feature_name = str(row.feature_name)
        station_id = str(row.station_id)
        subset = variant_frame[variant_frame["station_id"].astype(str).eq(station_id)].copy()
        if feature_name not in subset.columns:
            continue
        out = subset[
            [
                "target_date",
                "station_id",
                "target_tmax_c",
                "target_anomaly_vs_past_doy_c",
                feature_name,
            ]
        ].copy()
        out = out.rename(columns={feature_name: "feature_value"})
        out["source_attribute"] = str(row.source_attribute)
        out["transform"] = str(row.transform)
        out["feature_name"] = feature_name
        if not official_keep.empty:
            out = out.merge(official_keep, on="target_date", how="left")
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_readme(
    *,
    summary: dict[str, Any],
    coverage: pd.DataFrame,
    transform_summary: pd.DataFrame,
    station_rankings: pd.DataFrame,
    variant_atlas: pd.DataFrame,
) -> str:
    top_variant = summary.get("top_variant", {})
    top_text = "No scoreable station lag/slope variant was produced."
    if top_variant:
        top_text = (
            f"Top variant: station `{top_variant['station_id']}`, feature "
            f"`{top_variant['feature_name']}`, eval anomaly corr "
            f"`{top_variant['corr_eval_2000_2023_target_anomaly']}`, eval tertile spread "
            f"`{top_variant['eval_tertile_target_anomaly_spread_c']}`, official-error corr "
            f"`{top_variant['corr_official_error']}`."
        )
    return f"""# Station Lag/Slope Information Atlas

Generated: `{summary['generated_at_utc']}`

## Purpose

This insight folder extends `0047_station_contribution_atlas`. The earlier atlas asked whether a station's latest T-1 pre-15:00 value, one-day change, network-relative value, or station-pair spread had signal. This atlas asks a deeper question: does the recent path of each station attribute carry extra information?

The analysis creates lagged values, multi-day deltas, rolling means, current-minus-rolling-mean departures, and rolling slopes for every currently usable station/attribute pair. It then checks whether those variants explain the HKG target Tmax anomaly and whether they also line up with official forecast residuals on the currently available scored forecast rows.

## Leakage Control

- Every station value for target date `T` comes from station `local_date = T-1` and the latest observation before 15:00 HKT, inherited from the `0047` loader.
- Lags, rolling means, and rolling slopes are computed only within each station's past ordered station series. They never use target date `T` observations after the operational cutoff.
- Target anomaly uses a past-only day-of-year climatology.
- Feature selection evidence is split: pre-2000 is measured separately from 2000-2023 evaluation.
- Rows on or after `{CONFIRMATION_START.date()}` are rejected.
- Forecast residual checks use only the currently available scored forecast rows; those rows remain non-contiguous while the backfill runs.

## Dataset Scope

| Item | Value |
|---|---:|
| Station-day rows | {summary['station_feature_rows']} |
| Stations | {summary['station_count']} |
| Source attributes | {summary['source_attribute_count']} |
| Transform types | {summary['transform_count']} |
| Variant rows scored | {summary['variant_rows']} |
| Robust pre-2000-testable rows | {summary['robust_variant_rows']} |
| Official overlap rows | {summary['official_overlap_rows']} |
| Uses 2024+ rows | {summary['uses_2024_plus_rows']} |

## Main Result

{top_text}

## Station Coverage

{markdown_table(coverage, max_rows=50)}

## Transform Summary

{markdown_table(transform_summary, max_rows=80)}

## Station Rankings

{markdown_table(station_rankings.head(80), max_rows=80)}

## Top Variant Atlas Rows

{markdown_table(variant_atlas.head(120), max_rows=120)}

## Interpretation

This atlas is meant to answer whether station history adds value beyond one-day station snapshots. The most important rows are the variants with enough pre-2000 history, stable train/eval sign, high 2000-2023 target-anomaly correlation, and non-trivial official-error correlation. Those are the candidate features that should feed the next residual/regime model tests after the forecast backfill becomes available.

The analysis is not a final model and does not claim production MAE. It is a feature-discovery map. A strong row here says "this station behavior is physically informative and deserves controlled OOF testing." A weak row says "do not waste model capacity on this transformation unless it appears in a stronger interaction later."

## Files

- `artifacts/station_lag_slope_variant_atlas.csv`
- `artifacts/transformation_summary.csv`
- `artifacts/station_transform_rankings.csv`
- `artifacts/top_variant_timeseries.csv`
- `artifacts/station_coverage.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Station Lag/Slope Information Atlas\n"
    base = existing.split(marker)[0].rstrip()
    top = summary.get("top_variant", {})
    top_feature = top.get("feature_name", "") if isinstance(top, dict) else ""
    text = f"""{base}
{marker}
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_lag_slope_information_atlas.py`:

- `{FOLDER_NAME}`: long-history station lag, delta, rolling mean, rolling departure, and rolling slope information-gain atlas.

| Metric | Value |
|---|---:|
| Stations | {summary['station_count']} |
| Variant rows scored | {summary['variant_rows']} |
| Robust variants | {summary['robust_variant_rows']} |
| Top feature | `{top_feature}` |
| Top eval abs corr | {summary['top_eval_abs_corr']} |
| Top official abs corr | {summary['top_official_abs_corr']} |

Leakage contract: station values use only T-1 latest-before-15:00 HKT or earlier station history; all scored rows are before `{CONFIRMATION_START.date()}`.
"""
    write_text(index_path, text)


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Milestones\n"
    section_marker = "\n## Station Lag/Slope Information Atlas\n"
    blockers_marker = "\n## Current Blockers And Gaps\n"
    next_marker = "\n## Exact Next Recommended Codex Task\n"
    if section_marker in existing:
        base, rest = existing.split(section_marker, 1)
        suffix = f"{blockers_marker}{rest.split(blockers_marker, 1)[1]}" if blockers_marker in rest else ""
    elif blockers_marker in existing:
        base, rest = existing.split(blockers_marker, 1)
        suffix = f"{blockers_marker}{rest}"
    else:
        base = existing.rstrip()
        suffix = ""
    top = summary.get("top_variant", {})
    top_feature = top.get("feature_name", "") if isinstance(top, dict) else ""
    section = f"""{base.rstrip()}
{section_marker}
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_lag_slope_information_atlas.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Station lag/slope variants | `{summary['variant_rows']}` rows across `{summary['station_count']}` stations and `{summary['transform_count']}` transform types | Audited |
| Robust variants | `{summary['robust_variant_rows']}` variants have enough pre-2000 and 2000-2023 rows | Diagnostic |
| Top variant | `{top_feature}` | Documented |
| Top eval abs corr | `{summary['top_eval_abs_corr']}` | Diagnostic |
| Top official abs corr | `{summary['top_official_abs_corr']}` | Current scored archive only |
| Leakage guard | T-1 latest-before-15:00 HKT station timing, station-local rolling history, zero 2024+ rows | Guarded |

Interpretation: `0050` extends the station work from static values and pair spreads into recent station trajectories. Strong rows should become candidate inputs for the next strict OOF residual/regime experiments after forecast backfill progress lands.
"""
    blocker = (
        f"34. Station lag/slope information screening scored `{summary['variant_rows']}` station trajectory variants. "
        f"The top feature is `{top_feature}` with eval abs corr `{summary['top_eval_abs_corr']}` and official-error "
        f"abs corr `{summary['top_official_abs_corr']}`. These are feature-discovery signals, not final MAE proof."
    )
    if blockers_marker in suffix and blocker not in suffix:
        before_next, after_next = suffix.split(next_marker, 1) if next_marker in suffix else (suffix, "")
        before_next = before_next.rstrip() + f"\n{blocker}\n"
        next_task = f"""{next_marker}

Continue local deep-dive work while the forecast archive backfill runs: build `0051` regime-specific station trajectory interactions, using the strongest `0047`/`0050` pressure, wind, temperature, and dew-spread signals under pre-2000 selection and 2000-2023 evaluation.
"""
        suffix = before_next + "\n" + next_task if after_next else before_next
    section += suffix
    write_text(path, section)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    target = load_target()
    metadata = load_station_metadata()
    station_frame = load_station_day_features(target)
    official = load_official_residuals()
    require_no_confirmation_dates(station_frame["target_date"], context="0050 station frame")
    if not official.empty:
        require_no_confirmation_dates(official["target_date"], context="0050 official frame")

    variant_frame, variant_catalog = add_variant_columns(station_frame)
    variant_atlas = build_variant_atlas(variant_frame, variant_catalog, official, metadata)
    transforms = transformation_summary(variant_atlas)
    rankings = station_transform_rankings(variant_atlas)
    timeseries = top_variant_timeseries(variant_frame, variant_atlas, official)
    coverage = metadata.merge(
        station_frame.groupby("station_id", observed=True)
        .agg(
            station_day_feature_rows=("target_date", "size"),
            first_target_date=("target_date", "min"),
            last_target_date=("target_date", "max"),
        )
        .reset_index(),
        on="station_id",
        how="left",
    ).sort_values(["first_target_date", "station_id"])
    top_variant = variant_atlas.iloc[0].to_dict() if not variant_atlas.empty else {}
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "station_feature_rows": int(len(station_frame)),
        "station_count": int(station_frame["station_id"].nunique()),
        "source_attribute_count": len(STATION_ATTRIBUTES),
        "transform_count": int(variant_catalog["transform"].nunique()) if not variant_catalog.empty else 0,
        "variant_rows": int(len(variant_atlas)),
        "robust_variant_rows": int(variant_atlas["robust_pre2000_testable"].sum()) if not variant_atlas.empty else 0,
        "stable_sign_variant_rows": int(variant_atlas["stable_train_eval_sign"].sum()) if not variant_atlas.empty else 0,
        "official_overlap_rows": int(len(official)),
        "uses_2024_plus_rows": False,
        "top_variant": top_variant,
        "top_eval_abs_corr": top_variant.get("abs_corr_eval_2000_2023_target_anomaly", math.nan)
        if top_variant
        else math.nan,
        "top_official_abs_corr": top_variant.get("abs_corr_official_error", math.nan)
        if top_variant
        else math.nan,
        "leakage_guard": {
            "feature_date_rule": "target_date = station local_date + 1 day",
            "station_cutoff_rule": "latest_before_1500_hkt from T-1 station local_date",
            "rolling_rule": "station-local ordered history ending at the T-1 cutoff-safe row",
            "feature_selection_period": f"target_date <= {TRAIN_END.date()}",
            "evaluation_period": f"{EVAL_START.date()} <= target_date <= {EVAL_END.date()}",
            "confirmation_start": str(CONFIRMATION_START.date()),
        },
    }

    write_csv(artifacts / "station_coverage.csv", coverage)
    write_csv(artifacts / "station_lag_slope_variant_atlas.csv", variant_atlas)
    write_csv(artifacts / "transformation_summary.csv", transforms)
    write_csv(artifacts / "station_transform_rankings.csv", rankings)
    write_csv(artifacts / "top_variant_timeseries.csv", timeseries)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_lag_slope_information_atlas_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            coverage=coverage,
            transform_summary=transforms,
            station_rankings=rankings,
            variant_atlas=variant_atlas,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 station lag/slope information atlas.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
