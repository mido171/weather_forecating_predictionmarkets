from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    HEADLINE_END,
    PRESS_FORECAST_EXPORT_PATH,
    RESEARCH_ROOT,
    RSS_FORECAST_PATH,
    build_analysis_frame,
    feature_family,
    hkt_cutoff_utc_for_target_dates,
    load_champion_predictions,
    load_features,
    markdown_table,
    require_no_confirmation_dates,
    rolling_bias_correction_scores,
    scan_features,
    score_prediction_frame,
    select_latest_pre_cutoff_forecast,
    write_csv,
    write_json,
    write_text,
)

TOP_FEATURES = 80


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def source_score(frame: pd.DataFrame, prediction_col: str = "forecast_max_c") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, group in frame.groupby("forecast_source_family", dropna=False, observed=True):
        rows.append({"forecast_source_family": source, **score_prediction_frame(group, prediction_col)})
    return pd.DataFrame(rows).sort_values("mae", ascending=True).reset_index(drop=True)


def load_official_forecasts(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    if PRESS_FORECAST_EXPORT_PATH.exists():
        press = pd.read_parquet(PRESS_FORECAST_EXPORT_PATH)
        selected = select_latest_pre_cutoff_forecast(
            press,
            target_col="target_date",
            issue_col="issue_at_hkt",
            max_col="forecast_max_c",
            min_col="forecast_min_c",
            source_name="hko_press_archive",
        )
        selected["forecast_source_family"] = "press_archive"
        rows.append(selected)

    if RSS_FORECAST_PATH.exists():
        rss = pd.read_parquet(RSS_FORECAST_PATH)
        selected = select_latest_pre_cutoff_forecast(
            rss,
            target_col="forecast_date",
            issue_col="available_at_hkt",
            max_col="forecast_max_temperature_c",
            min_col="forecast_min_temperature_c",
            source_name="hko_rss",
        )
        selected["forecast_source_family"] = "rss_archive"
        rows.append(selected)

    if not rows:
        return pd.DataFrame()

    official = pd.concat(rows, ignore_index=True)
    official["target_date"] = pd.to_datetime(official["target_date"], errors="coerce").dt.normalize()
    official = official[official["target_date"] <= HEADLINE_END].copy()
    require_no_confirmation_dates(official["target_date"], context="forecast anchor official forecast load")
    official["forecast_range_c"] = official["forecast_max_c"] - official["forecast_min_c"]
    official["forecast_midpoint_c"] = np.where(
        official["forecast_min_c"].notna(),
        (official["forecast_min_c"] + official["forecast_max_c"]) / 2.0,
        np.nan,
    )
    official["cutoff_utc"] = hkt_cutoff_utc_for_target_dates(official["target_date"])
    official["issue_to_cutoff_hours"] = (
        (official["cutoff_utc"] - official["issue_utc"]).dt.total_seconds() / 3600.0
    )
    official = official.drop_duplicates(["forecast_source_family", "target_date"], keep="last")

    feature_cols = [
        "target_date",
        "target_tmax_c",
        "season",
        "month",
        "point_forecast",
        "champion_error_c",
        "champion_abs_error_c",
    ]
    available_feature_cols = [col for col in feature_cols if col in features.columns]
    joined = official.merge(features[available_feature_cols], on="target_date", how="inner")
    joined["official_error_c"] = joined["forecast_max_c"] - joined["target_tmax_c"]
    joined["official_abs_error_c"] = joined["official_error_c"].abs()
    joined["official_midpoint_error_c"] = joined["forecast_midpoint_c"] - joined["target_tmax_c"]
    joined["official_underpredicted"] = joined["official_error_c"] < 0
    joined["official_overpredicted"] = joined["official_error_c"] > 0
    return joined.sort_values(["forecast_source_family", "target_date"]).reset_index(drop=True)


def aggregate_error_by(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, dropna=False, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        error = group["official_error_c"]
        row = {col: value for col, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "n": int(len(group)),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "bias": float(error.mean()),
                "p90_abs_error": float(error.abs().quantile(0.9)),
                "first_date": str(group["target_date"].min().date()),
                "last_date": str(group["target_date"].max().date()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mae"], ascending=False).reset_index(drop=True)


def bucket_series(values: pd.Series, bins: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < bins * 25 or numeric.nunique(dropna=True) < bins:
        return pd.Series(["insufficient"] * len(values), index=values.index)
    ranked = numeric.rank(method="first")
    return pd.qcut(ranked, bins, labels=[f"{prefix}{idx + 1}" for idx in range(bins)]).astype(str).where(
        numeric.notna(),
        "missing",
    )


def build_official_residual_anatomy(official: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    out["source_scoreboard"] = source_score(official)
    out["by_source_season"] = aggregate_error_by(official, ["forecast_source_family", "season"])
    out["by_source_month"] = aggregate_error_by(official, ["forecast_source_family", "month"])

    range_frames: list[pd.DataFrame] = []
    for source, group in official.groupby("forecast_source_family", observed=True):
        work = group.copy()
        work["forecast_range_bucket"] = bucket_series(work["forecast_range_c"], 5, "range_q")
        table = aggregate_error_by(work[~work["forecast_range_bucket"].eq("insufficient")], ["forecast_range_bucket"])
        if not table.empty:
            table["forecast_source_family"] = source
            range_frames.append(table)
    out["by_forecast_range_bucket"] = pd.concat(range_frames, ignore_index=True) if range_frames else pd.DataFrame()

    lead_frames: list[pd.DataFrame] = []
    for source, group in official.groupby("forecast_source_family", observed=True):
        work = group.copy()
        work["issue_to_cutoff_bucket"] = bucket_series(work["issue_to_cutoff_hours"], 5, "issue_age_q")
        table = aggregate_error_by(work[~work["issue_to_cutoff_bucket"].eq("insufficient")], ["issue_to_cutoff_bucket"])
        if not table.empty:
            table["forecast_source_family"] = source
            lead_frames.append(table)
    out["by_issue_age_bucket"] = pd.concat(lead_frames, ignore_index=True) if lead_frames else pd.DataFrame()
    return out


def choose_features(feature_scan: pd.DataFrame) -> list[str]:
    if feature_scan.empty:
        return []
    ordered = feature_scan.sort_values(
        ["max_abs_residual_corr", "max_abs_abs_error_corr", "max_abs_target_corr"],
        ascending=False,
    )
    features: list[str] = []
    for feature in ordered["feature"].to_list():
        if feature not in features:
            features.append(feature)
        if len(features) >= TOP_FEATURES:
            break
    return features


def safe_corr(left: pd.Series, right: pd.Series, *, min_rows: int = 250) -> float:
    pair = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(pair) < min_rows or pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def build_feature_conditioning(
    official: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = choose_features(feature_scan)
    enriched = official.merge(
        feature_frame[["target_date", *[feature for feature in features if feature in feature_frame.columns]]],
        on="target_date",
        how="left",
    )
    correlation_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    for source, group in enriched.groupby("forecast_source_family", observed=True):
        for feature in features:
            if feature not in group.columns:
                continue
            n = int(pd.to_numeric(group[feature], errors="coerce").notna().sum())
            if n < 250:
                continue
            correlation_rows.append(
                {
                    "forecast_source_family": source,
                    "feature": feature,
                    "family": feature_family(feature),
                    "n": n,
                    "official_error_corr": safe_corr(group[feature], group["official_error_c"]),
                    "official_abs_error_corr": safe_corr(group[feature], group["official_abs_error_c"]),
                    "official_underprediction_corr": safe_corr(group[feature], group["official_underpredicted"].astype(int)),
                }
            )
            work = group[["target_date", "official_error_c", "official_abs_error_c", feature]].copy()
            work["feature_bucket"] = bucket_series(work[feature], 5, "q")
            if work["feature_bucket"].eq("insufficient").all():
                continue
            table = (
                work.groupby("feature_bucket", dropna=False, observed=True)
                .agg(
                    n=("target_date", "count"),
                    feature_min=(feature, "min"),
                    feature_max=(feature, "max"),
                    official_mae=("official_abs_error_c", "mean"),
                    official_bias=("official_error_c", "mean"),
                )
                .reset_index()
            )
            table = table[~table["feature_bucket"].isin(["missing", "insufficient"])].copy()
            table = table[table["n"] >= 50].copy()
            if len(table) < 3:
                continue
            table["forecast_source_family"] = source
            table["feature"] = feature
            table["family"] = feature_family(feature)
            bucket_rows.extend(table.to_dict("records"))

    correlations = pd.DataFrame(correlation_rows)
    if not correlations.empty:
        correlations["priority"] = (
            correlations["official_error_corr"].abs().fillna(0.0) * 2.0
            + correlations["official_abs_error_corr"].abs().fillna(0.0)
            + correlations["official_underprediction_corr"].abs().fillna(0.0)
        )
        correlations = correlations.sort_values("priority", ascending=False).reset_index(drop=True)

    buckets = pd.DataFrame(bucket_rows)
    if not buckets.empty:
        summary = (
            buckets.groupby(["forecast_source_family", "feature", "family"], observed=True)
            .agg(
                bucket_count=("feature_bucket", "nunique"),
                min_bucket_n=("n", "min"),
                official_mae_spread=("official_mae", lambda s: float(s.max() - s.min())),
                official_bias_spread=("official_bias", lambda s: float(s.max() - s.min())),
            )
            .reset_index()
        )
        summary["threshold_priority"] = summary["official_bias_spread"].abs() + summary["official_mae_spread"].abs()
        summary = summary.sort_values("threshold_priority", ascending=False).reset_index(drop=True)
    else:
        summary = pd.DataFrame()
    return correlations, summary


def build_blend_sensitivity(official: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap = official[official["point_forecast"].notna()].copy()
    if overlap.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, object]] = []
    predictions = overlap[["forecast_source_family", "target_date", "target_tmax_c", "forecast_max_c", "point_forecast"]].copy()
    for source, group in overlap.groupby("forecast_source_family", observed=True):
        for weight in np.linspace(0, 1, 21):
            col = f"diagnostic_blend_official_weight_{weight:.2f}"
            work = group.copy()
            work[col] = weight * work["forecast_max_c"] + (1.0 - weight) * work["point_forecast"]
            rows.append(
                {
                    "forecast_source_family": source,
                    "official_weight": float(weight),
                    "diagnostic_not_deployable_without_fold_local_weight_selection": True,
                    **score_prediction_frame(work, col),
                }
            )
            if weight in (0.0, 0.5, 0.75, 1.0):
                predictions.loc[group.index, col] = work[col]
    grid = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    return grid, predictions


def build_past_only_source_bias_scores(official: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, group in official.groupby("forecast_source_family", observed=True):
        if len(group) < 365:
            continue
        _corrected, scores = rolling_bias_correction_scores(group, half_lives=[7, 14, 30, 60, 90, 180, 365])
        for row in scores.to_dict("records"):
            rows.append({"forecast_source_family": source, **row})
    return pd.DataFrame(rows).sort_values(["mae", "rmse"], na_position="last").reset_index(drop=True)


def write_residual_anatomy(official: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    folder = RESEARCH_ROOT / "0012_official_forecast_residual_anatomy"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "official_selected_forecast_rows.csv", official)
    for name, table in tables.items():
        write_csv(artifacts / f"{name}.csv", table)

    text = f"""# Official Forecast Residual Anatomy

Generated: `{now_utc()}`

## What Was Tested

This insight treats the official HKO forecast as the anchor signal and studies where that anchor is wrong. It uses only forecasts issued no later than `T-1 15:00 HKT`, and blocks target labels from `{CONFIRMATION_START.date()}` onward.

## Source Scoreboard

{markdown_table(tables['source_scoreboard'], max_rows=10)}

## Error By Source And Season

{markdown_table(tables['by_source_season'], max_rows=20)}

## Error By Forecast Range

{markdown_table(tables['by_forecast_range_bucket'], max_rows=20)}

## Interpretation

The official forecast archive is still partial, but this confirms the correct direction: the official forecast is the strongest anchor we have. The remaining problem is not finding a better standalone weather-only model; it is finding when and how to correct the official forecast.
"""
    write_text(folder / "README.md", text)


def write_feature_conditioning(correlations: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0013_official_forecast_feature_conditioning"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "feature_corr.csv", correlations)
    write_csv(artifacts / "feature_thresholds.csv", thresholds)
    text = f"""# Official Forecast Feature Conditioning

Generated: `{now_utc()}`

## What Was Tested

This insight asks which long-history features explain official forecast errors, not just target Tmax. That distinction matters: to beat a strong official forecast, we need features that explain the official residual.

## Top Official-Error Correlation Signals

{markdown_table(correlations.head(25), max_rows=25)}

## Top Threshold Signals

{markdown_table(thresholds.head(25), max_rows=25)}

## Interpretation

Features with strong official-error or official-underprediction correlation are candidate bias-correction inputs. Threshold tables are diagnostic only; any deployed threshold must be recreated inside rolling-origin folds.
"""
    write_text(folder / "README.md", text)


def write_blend_sensitivity(grid: pd.DataFrame, predictions: pd.DataFrame, bias_scores: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0014_official_anchor_blend_sensitivity"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "weight_grid.csv", grid)
    write_csv(artifacts / "blend_predictions.csv", predictions)
    write_csv(artifacts / "past_only_bias.csv", bias_scores)
    text = f"""# Official Anchor Blend Sensitivity

Generated: `{now_utc()}`

## What Was Tested

This folder tests how much signal sits in the official forecast versus the current long-history champion on overlapping rows. The fixed-weight grid is diagnostic and is not deployable by itself because choosing a best fixed weight after seeing the full window would be forward-looking. Past-only bias rows are safer because each correction uses only earlier target dates.

## Diagnostic Fixed-Weight Grid

{markdown_table(grid.head(25), max_rows=25)}

## Past-Only Official Bias Scores

{markdown_table(bias_scores.head(20), max_rows=20)}

## Interpretation

The official forecast should be the center of the next system. The long-history/station/upper-air stack should be used to correct official residuals, estimate uncertainty, and decide when to trust or fade the official anchor.
"""
    write_text(folder / "README.md", text)


def write_coverage_update(official: pd.DataFrame, export_summary_path: Path) -> None:
    folder = RESEARCH_ROOT / "0015_official_forecast_archive_coverage_gap"
    artifacts = folder / "artifacts"
    coverage = (
        official.assign(year=official["target_date"].dt.year)
        .groupby(["forecast_source_family", "year"], observed=True)
        .agg(
            selected_scoreable_days=("target_date", "nunique"),
            first_target_date=("target_date", "min"),
            last_target_date=("target_date", "max"),
        )
        .reset_index()
    )
    coverage["first_target_date"] = coverage["first_target_date"].dt.strftime("%Y-%m-%d")
    coverage["last_target_date"] = coverage["last_target_date"].dt.strftime("%Y-%m-%d")
    write_csv(artifacts / "selected_days_by_year.csv", coverage)

    export_summary = {}
    if export_summary_path.exists():
        export_summary = json.loads(export_summary_path.read_text(encoding="utf-8"))
    write_json(artifacts / "press_export_summary.json", export_summary)

    text = f"""# Official Forecast Archive Coverage Gap

Generated: `{now_utc()}`

## What This Documents

This folder records exactly how much official forecast history is currently scoreable after leakage-safe pre-cutoff selection.

## Selected Forecast Coverage

{markdown_table(coverage, max_rows=40)}

## Press Export Snapshot

```json
{json.dumps(export_summary, indent=2, sort_keys=True)}
```

## Current Gap

The press archive has a candidate index through 2026, but scoreable selected forecasts are still limited by which raw detail pages have actually been retrieved and parsed. This remains the main blocker to testing the official-forecast anchor across 2000-2023.
"""
    write_text(folder / "README.md", text)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Forecast-Anchor Forensics\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional forecast-anchor folders created by `scripts/run_hkg_t24_forecast_anchor_forensics.py`:

- `0012_official_forecast_residual_anatomy`: source/season/range anatomy of official forecast errors.
- `0013_official_forecast_feature_conditioning`: long-history feature conditioning of official forecast residuals.
- `0014_official_anchor_blend_sensitivity`: official/champion overlap and past-only official bias checks.
- `0015_official_forecast_archive_coverage_gap`: exact selected official forecast coverage by source/year.

Key counts:

| Metric | Value |
|---|---:|
| Official selected rows | {manifest['official_selected_rows']} |
| Press selected rows | {manifest['press_selected_rows']} |
| RSS selected rows | {manifest['rss_selected_rows']} |
| Official feature correlations | {manifest['feature_correlation_rows']} |
| Official feature threshold rows | {manifest['feature_threshold_rows']} |
| Diagnostic blend grid rows | {manifest['blend_grid_rows']} |

Leakage contract: all official forecasts are selected using the latest issue no later than `T-1 15:00 HKT`; labels from `{CONFIRMATION_START.date()}` onward remain blocked.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    features = load_features()
    champion = load_champion_predictions()
    frame = build_analysis_frame(features, champion)
    require_no_confirmation_dates(frame["target_date"], context="forecast anchor frame")
    feature_scan = scan_features(frame)
    official = load_official_forecasts(frame)
    if official.empty:
        raise RuntimeError("No official forecast rows available for forecast-anchor forensics.")
    require_no_confirmation_dates(official["target_date"], context="forecast anchor official selected rows")

    anatomy = build_official_residual_anatomy(official)
    correlations, thresholds = build_feature_conditioning(official, frame, feature_scan)
    blend_grid, blend_predictions = build_blend_sensitivity(official)
    bias_scores = build_past_only_source_bias_scores(official)

    write_residual_anatomy(official, anatomy)
    write_feature_conditioning(correlations, thresholds)
    write_blend_sensitivity(blend_grid, blend_predictions, bias_scores)
    write_coverage_update(
        official,
        PROJECT_PATHS.data_root
        / "datasets"
        / "05_hko_historical_rss_forecasts"
        / "hko_press_archive_offline_export_manifest.json",
    )

    source_counts = official["forecast_source_family"].value_counts().to_dict()
    manifest = {
        "generated_at_utc": now_utc(),
        "confirmation_start": str(CONFIRMATION_START.date()),
        "official_selected_rows": int(len(official)),
        "press_selected_rows": int(source_counts.get("press_archive", 0)),
        "rss_selected_rows": int(source_counts.get("rss_archive", 0)),
        "first_target_date": str(official["target_date"].min().date()),
        "last_target_date": str(official["target_date"].max().date()),
        "feature_correlation_rows": int(len(correlations)),
        "feature_threshold_rows": int(len(thresholds)),
        "blend_grid_rows": int(len(blend_grid)),
        "past_only_bias_score_rows": int(len(bias_scores)),
        "folders": [
            "0012_official_forecast_residual_anatomy",
            "0013_official_forecast_feature_conditioning",
            "0014_official_anchor_blend_sensitivity",
            "0015_official_forecast_archive_coverage_gap",
        ],
    }
    write_json(RESEARCH_ROOT / "forecast_anchor_forensics_manifest.json", manifest)
    update_master_index(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run official-forecast-anchor forensics.").parse_args()


def main() -> None:
    parse_args()
    manifest = run()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
