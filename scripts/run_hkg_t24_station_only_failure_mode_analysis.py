from __future__ import annotations

import argparse
import json
import math
import re
import sys
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
    safe_corr,
    update_markdown_section,
)
from scripts.run_hkg_t24_station_contribution_atlas import load_target  # noqa: E402

FOLDER_NAME = "0056_station_only_failure_mode_analysis"
ARTIFACT_0054 = RESEARCH_ROOT / "0054_station_only_walkforward_matrix_audit" / "artifacts"
ARTIFACT_0055 = RESEARCH_ROOT / "0055_station_only_walkforward_benchmark" / "artifacts"
PREDICTIONS_PATH = ARTIFACT_0055 / "predictions.parquet"
SUMMARY_0055_PATH = ARTIFACT_0055 / "summary.json"
FEATURE_MATRIX_PATH = ARTIFACT_0054 / "features.parquet"
COMPONENT_CATALOG_PATH = ARTIFACT_0054 / "components.csv"
TRAIN_THRESHOLD_END = pd.Timestamp("1999-12-31")
MIN_GROUP_ROWS = 120
MIN_CORR_ROWS = 365


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def quantile_edges(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < MIN_CORR_ROWS or clean.nunique(dropna=True) < 3:
        return math.nan, math.nan
    low, high = clean.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return math.nan, math.nan
    return float(low), float(high)


def apply_tertile(values: pd.Series, edges: tuple[float, float]) -> pd.Series:
    low, high = edges
    numeric = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return pd.Series("missing", index=values.index, dtype="object")
    labels = np.select(
        [numeric.isna(), numeric <= low, numeric <= high],
        ["missing", "low", "mid"],
        default="high",
    )
    return pd.Series(labels, index=values.index, dtype="object")


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def load_best_predictions() -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = load_json(SUMMARY_0055_PATH)
    best_model = str(summary["best_model_id"])
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions = predictions[predictions["model_id"].astype(str).eq(best_model)].copy()
    if predictions.empty:
        raise RuntimeError(f"0055 best model has no predictions: {best_model}")
    predictions = predictions[predictions["target_date"].between(EVAL_START, EVAL_END)].copy()
    require_no_confirmation_dates(predictions["target_date"], context="0056 best predictions")
    predictions["error_c"] = predictions["point_forecast_c"] - predictions["target_tmax_c"]
    predictions["abs_error_c"] = predictions["error_c"].abs()
    predictions["month"] = predictions["target_date"].dt.month
    predictions["year"] = predictions["target_date"].dt.year
    predictions["season"] = predictions["month"].map(month_to_season)
    return predictions, summary


def load_analysis_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions, summary = load_best_predictions()
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    station_cols = [
        col
        for col in features.columns
        if col not in {"target_date", "source_local_date_rule", "source_cutoff_hkt"}
    ]
    features["available_feature_fraction"] = features[station_cols].notna().mean(axis=1)
    target = load_target()[["target_date", "target_tmax_c"]].copy()
    train_target = target[target["target_date"].le(TRAIN_THRESHOLD_END)]
    heat_edges = quantile_edges(train_target["target_tmax_c"])
    frame = predictions.merge(features[["target_date", "available_feature_fraction", *station_cols]], on="target_date", how="left")
    frame["heat_bucket_pre2000_target"] = apply_tertile(frame["target_tmax_c"], heat_edges)
    require_no_confirmation_dates(frame["target_date"], context="0056 analysis frame")
    return frame, pd.read_csv(COMPONENT_CATALOG_PATH), summary


def choose_regime_features(catalog: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rules = [
        ("temp_14d_trajectory", "station_trajectory", "air_temperature_c_latest_before_1500__current_minus_rolling_mean_14d"),
        ("dew_14d_trajectory", "station_trajectory", "dew_point_c_latest_before_1500__current_minus_rolling_mean_14d"),
        ("pressure_pair_spread", "station_pair_spread", "sea_level_pressure_hpa_latest_before_1500"),
        ("wind_station_attribute", "station_attribute", "wind_speed_mps_latest_before_1500"),
    ]
    for regime_name, source_family, raw_feature_name in rules:
        candidates = catalog[
            catalog["source_family"].astype(str).eq(source_family)
            & catalog["raw_feature_name"].astype(str).eq(raw_feature_name)
        ].copy()
        if candidates.empty:
            continue
        rows.append(
            {
                "regime_name": regime_name,
                "feature": str(candidates.iloc[0]["feature_id"]),
                "source_family": source_family,
                "raw_feature_name": raw_feature_name,
            }
        )
    return rows


def attach_regime_buckets(frame: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    rows: list[dict[str, object]] = []
    train_like = load_target()[["target_date"]].copy()
    train_like = train_like[train_like["target_date"].le(TRAIN_THRESHOLD_END)]
    history = pd.read_parquet(FEATURE_MATRIX_PATH)
    history["target_date"] = pd.to_datetime(history["target_date"], errors="coerce").dt.normalize()
    history = history[history["target_date"].isin(set(train_like["target_date"]))].copy()
    history_station_cols = [
        col
        for col in history.columns
        if col not in {"target_date", "source_local_date_rule", "source_cutoff_hkt"}
    ]
    history["available_feature_fraction"] = history[history_station_cols].notna().mean(axis=1)

    availability_edges = quantile_edges(history["available_feature_fraction"])
    out["availability_bucket"] = apply_tertile(out["available_feature_fraction"], availability_edges)
    rows.append(
        {
            "bucket_column": "availability_bucket",
            "source_feature": "available_feature_fraction",
            "threshold_fit_window": f"<= {TRAIN_THRESHOLD_END.date()}",
            "low_edge": availability_edges[0],
            "high_edge": availability_edges[1],
            "deployability": "deployable_pre_cutoff",
        }
    )

    for spec in choose_regime_features(catalog):
        feature = spec["feature"]
        if feature not in out.columns or feature not in history.columns:
            continue
        edges = quantile_edges(history[feature])
        bucket_column = f"{spec['regime_name']}_bucket"
        out[bucket_column] = apply_tertile(out[feature], edges)
        rows.append(
            {
                "bucket_column": bucket_column,
                "source_feature": feature,
                "threshold_fit_window": f"<= {TRAIN_THRESHOLD_END.date()}",
                "low_edge": edges[0],
                "high_edge": edges[1],
                "deployability": "deployable_pre_cutoff",
            }
        )
    rows.append(
        {
            "bucket_column": "heat_bucket_pre2000_target",
            "source_feature": "target_tmax_c",
            "threshold_fit_window": f"<= {TRAIN_THRESHOLD_END.date()}",
            "low_edge": math.nan,
            "high_edge": math.nan,
            "deployability": "diagnostic_outcome_only_not_a_forecast_feature",
        }
    )
    return out, pd.DataFrame(rows)


def score_group(frame: pd.DataFrame, group_cols: list[str], *, min_rows: int = MIN_GROUP_ROWS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, dropna=False, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        if len(group) < min_rows:
            continue
        error = pd.to_numeric(group["error_c"], errors="coerce")
        abs_error = error.abs()
        row = {col: value for col, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "n": int(len(group)),
                "first_date": str(group["target_date"].min().date()),
                "last_date": str(group["target_date"].max().date()),
                "mae": float(abs_error.mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "bias": float(error.mean()),
                "median_abs_error": float(abs_error.median()),
                "share_abs_error_ge_2c": float((abs_error >= 2.0).mean()),
                "p90_abs_error": float(abs_error.quantile(0.90)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mae", "n"], ascending=[False, False]).reset_index(drop=True)


def build_group_analyses(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_sets = [
        ("fold", ["fold_id"]),
        ("year", ["year"]),
        ("month", ["month"]),
        ("season", ["season"]),
        ("heat_bucket", ["heat_bucket_pre2000_target"]),
        ("availability_bucket", ["availability_bucket"]),
        ("season_x_heat", ["season", "heat_bucket_pre2000_target"]),
    ]
    for column in frame.columns:
        if column.endswith("_bucket") and column not in {"heat_bucket_pre2000_target", "availability_bucket"}:
            group_sets.append((column, [column]))
    rows: list[pd.DataFrame] = []
    overall_mae = float(frame["abs_error_c"].mean())
    for analysis_name, cols in group_sets:
        summary = score_group(frame, cols)
        if summary.empty:
            continue
        summary.insert(0, "analysis_name", analysis_name)
        summary["mae_lift_vs_overall"] = summary["mae"] - overall_mae
        rows.append(summary)
    all_groups = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ranked = all_groups[all_groups["n"] >= MIN_GROUP_ROWS].copy()
    ranked = ranked.sort_values(["mae_lift_vs_overall", "share_abs_error_ge_2c", "n"], ascending=[False, False, False])
    return all_groups, ranked.reset_index(drop=True)


def feature_error_correlations(frame: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [feature for feature in catalog["feature_id"].astype(str).tolist() if feature in frame.columns]
    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        n_error, corr_error = safe_corr(frame[feature], frame["error_c"], min_rows=MIN_CORR_ROWS)
        n_abs, corr_abs = safe_corr(frame[feature], frame["abs_error_c"], min_rows=MIN_CORR_ROWS)
        meta = catalog[catalog["feature_id"].astype(str).eq(feature)].iloc[0]
        rows.append(
            {
                "feature": feature,
                "source_family": str(meta["source_family"]),
                "raw_feature_name": str(meta["raw_feature_name"]),
                "n_error_corr": n_error,
                "corr_error": corr_error,
                "abs_corr_error": abs(corr_error) if math.isfinite(corr_error) else math.nan,
                "n_abs_error_corr": n_abs,
                "corr_abs_error": corr_abs,
                "abs_corr_abs_error": abs(corr_abs) if math.isfinite(corr_abs) else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["abs_corr_abs_error", "abs_corr_error"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def worst_days(frame: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "target_date",
        "fold_id",
        "target_tmax_c",
        "point_forecast_c",
        "error_c",
        "abs_error_c",
        "month",
        "season",
        "heat_bucket_pre2000_target",
        "availability_bucket",
    ]
    bucket_cols = [col for col in frame.columns if col.endswith("_bucket") and col not in keep_cols]
    return frame[[*keep_cols, *bucket_cols]].sort_values("abs_error_c", ascending=False).head(120).reset_index(drop=True)


def leakage_audit(frame: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "regime_thresholds_pre2000",
            "passed": bool(thresholds["threshold_fit_window"].astype(str).str.contains(str(TRAIN_THRESHOLD_END.date())).all()),
            "evidence": f"{len(thresholds)} threshold rows document <= {TRAIN_THRESHOLD_END.date()} fit window",
        },
        {
            "check_id": "outcome_heat_bucket_marked_diagnostic",
            "passed": bool(
                thresholds[
                    thresholds["bucket_column"].eq("heat_bucket_pre2000_target")
                ]["deployability"].astype(str).str.contains("diagnostic").all()
            ),
            "evidence": "target-based heat bucket is not marked as a deployable forecast input",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    group_rank: pd.DataFrame,
    feature_corr: pd.DataFrame,
    worst: pd.DataFrame,
    thresholds: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    return f"""# Station-Only Failure Mode Analysis

Generated: `{summary['generated_at_utc']}`

## Purpose

`0055` showed that the best station-only benchmark is useful but still far behind an official-forecast anchor on overlapping rows. This folder asks where that station-only winner fails: which folds, months, heat levels, station-availability states, and station-regime buckets produce the largest errors?

No model is trained here. This is failure analysis for the `0055` best model `{summary['best_model_id']}`.

## Scope

| Item | Value |
|---|---:|
| Best model analyzed | {summary['best_model_id']} |
| Rows analyzed | {summary['rows_analyzed']} |
| First date | {summary['first_date']} |
| Last date | {summary['last_date']} |
| Overall MAE | {summary['overall_mae']} |
| Overall RMSE | {summary['overall_rmse']} |
| Overall bias | {summary['overall_bias']} |
| Uses 2024+ rows | {summary['uses_2024_plus_rows']} |

## Leakage Contract

- Only `0055` pre-2024 OOF predictions are analyzed.
- Deployable regime thresholds use feature history up to `1999-12-31`.
- The target heat bucket is explicitly diagnostic-only because it uses the realized target label.
- This folder does not alter forecasts, train a model, tune hyperparameters, or open 2024+ confirmation rows.

## Highest-MAE Failure Regimes

{markdown_table(group_rank.head(80), max_rows=80)}

## Feature Correlation With Absolute Error

{markdown_table(feature_corr.head(80), max_rows=80)}

## Worst Individual Days

{markdown_table(worst.head(80), max_rows=80)}

## Threshold Audit

{markdown_table(thresholds, max_rows=40)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

The most important use of this folder is not to promote station-only forecasts. It identifies where station-only structure is weak so the later official-anchor residual system can add targeted specialists: hot/cool level specialists, late-period bias repair, and pressure/temperature-regime adjustments. The high-bias 2018-2023 fold from `0055` is especially important because it suggests trend or station-domain drift that a static station-only Ridge model does not fully absorb.

## Files

- `artifacts/group_summary.csv`
- `artifacts/failure_regime_rank.csv`
- `artifacts/feature_error_correlations.csv`
- `artifacts/worst_days.csv`
- `artifacts/regime_thresholds.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/analysis_frame_sample.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_failure_mode_analysis.py`:

- `{FOLDER_NAME}`: failure-mode analysis for the `0055` best station-only benchmark.

| Metric | Value |
|---|---:|
| Rows analyzed | {summary['rows_analyzed']} |
| Overall MAE | {summary['overall_mae']} |
| Overall RMSE | {summary['overall_rmse']} |
| Worst ranked regime | {summary['top_failure_regime']} |
| Worst regime MAE | {summary['top_failure_mae']} |

Leakage contract: thresholded deployable regimes use pre-2000 feature history; target heat buckets are diagnostic-only.
"""
    update_markdown_section(RESEARCH_ROOT / "README.md", heading="Station-Only Failure Mode Analysis", section=section)


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_failure_mode_analysis.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Model analyzed | `{summary['best_model_id']}` from `0055` | Complete |
| Rows analyzed | `{summary['rows_analyzed']}` | Complete |
| Overall MAE / RMSE | `{summary['overall_mae']}` / `{summary['overall_rmse']}` | Diagnostic |
| Top failure regime | `{summary['top_failure_regime']}` | Documented |
| Top failure MAE | `{summary['top_failure_mae']}` | Diagnostic |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0056` turns the station-only benchmark miss pattern into concrete next specialist targets. It does not train or promote a new model.
"""
    update_markdown_section(
        path,
        heading="Station-Only Failure Mode Analysis",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"40. Station-only failure analysis found top regime `{summary['top_failure_regime']}` "
        f"with MAE `{summary['top_failure_mae']}`. This should guide residual specialists, not replace the official-anchor system."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Build a small station-only residual-specialist design queue from `0056`: late-period bias repair, hot/cool outcome diagnostics converted into deployable pre-cutoff proxies, and pressure/temperature regime interactions. Keep it as design/audit work until the continuous official forecast backfill is verified.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, catalog, source_summary = load_analysis_frame()
    frame, thresholds = attach_regime_buckets(frame, catalog)
    group_summary, failure_rank = build_group_analyses(frame)
    feature_corr = feature_error_correlations(frame, catalog)
    worst = worst_days(frame)
    leakage = leakage_audit(frame, thresholds)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0056 leakage audit failed: {failed}")

    top = failure_rank.iloc[0] if not failure_rank.empty else pd.Series(dtype=object)
    top_group_cols = [
        col
        for col in failure_rank.columns
        if col
        not in {
            "analysis_name",
            "n",
            "first_date",
            "last_date",
            "mae",
            "rmse",
            "bias",
            "median_abs_error",
            "share_abs_error_ge_2c",
            "p90_abs_error",
            "mae_lift_vs_overall",
        }
        and pd.notna(top.get(col, np.nan))
    ]
    top_regime = (
        f"{top.get('analysis_name', '')}: "
        + ", ".join(f"{col}={top.get(col)}" for col in top_group_cols)
        if not top.empty
        else ""
    )
    error = frame["error_c"]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "best_model_id": str(source_summary["best_model_id"]),
        "rows_analyzed": int(len(frame)),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "overall_mae": float(frame["abs_error_c"].mean()),
        "overall_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "overall_bias": float(error.mean()),
        "top_failure_regime": top_regime,
        "top_failure_mae": float(top.get("mae", math.nan)) if not top.empty else math.nan,
        "top_failure_lift_vs_overall": float(top.get("mae_lift_vs_overall", math.nan)) if not top.empty else math.nan,
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "group_summary.csv", group_summary)
    write_csv(artifacts / "failure_regime_rank.csv", failure_rank)
    write_csv(artifacts / "feature_error_correlations.csv", feature_corr)
    write_csv(artifacts / "worst_days.csv", worst)
    write_csv(artifacts / "regime_thresholds.csv", thresholds)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_csv(artifacts / "analysis_frame_sample.csv", frame.head(1000))
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_failure_mode_analysis_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            group_rank=failure_rank,
            feature_corr=feature_corr,
            worst=worst,
            thresholds=thresholds,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Analyze failure modes of the 0055 station-only winner.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
