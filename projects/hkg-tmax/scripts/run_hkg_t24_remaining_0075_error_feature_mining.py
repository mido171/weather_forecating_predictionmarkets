from __future__ import annotations

import argparse
import json
import math
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
    FEATURE_MATRIX_PATH,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_online_no_regret_trust_router import build_router_frame  # noqa: E402

FOLDER_NAME = "0077_remaining_0075_error_feature_mining"
MIN_FEATURE_ROWS = 300
MIN_SOURCE_FEATURE_ROWS = 150
MIN_BUCKET_ROWS = 25


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def feature_family(column: str) -> str:
    if column.startswith("igra_"):
        return "upper_air_igra"
    if column.startswith("isd_"):
        return "regional_isd_station"
    if column.startswith("daily_"):
        return "hko_daily_climate"
    if column.startswith("target_") or column in {"year", "month", "day_of_year", "doy_sin", "doy_cos"}:
        return "target_memory_calendar"
    if "station" in column or "spread" in column or "gradient" in column:
        return "station_network_derived"
    if "forecast" in column or "weather" in column or "wind" in column or "rh_" in column:
        return "official_forecast_state"
    return "other_numeric"


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "target_tmax_c",
        "target_date",
        "current_target_tmax_c",
        "m0069_prediction_c",
        "m0074_prediction_c",
        "m0075_prediction_c",
        "residual_0075_c",
        "abs_error_0075_c",
    }
    columns: list[str] = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        name = str(column)
        if name in excluded or name.startswith("target_tmax_c"):
            continue
        columns.append(name)
    return columns


def safe_corr(x: pd.Series, y: pd.Series, *, method: str = "pearson") -> float:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < MIN_BUCKET_ROWS or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return math.nan
    return float(pair["x"].corr(pair["y"], method=method))


def quantile_bucket_stats(values: pd.Series, target: pd.Series, *, bucket_count: int = 5) -> dict[str, object]:
    pair = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "target": target}).dropna()
    if len(pair) < MIN_FEATURE_ROWS or pair["value"].nunique() < bucket_count:
        return {
            "bucket_count": 0,
            "bucket_spread": math.nan,
            "best_bucket_mean": math.nan,
            "worst_bucket_mean": math.nan,
            "best_bucket": "",
            "worst_bucket": "",
        }
    try:
        pair["bucket"] = pd.qcut(pair["value"], q=bucket_count, duplicates="drop")
    except ValueError:
        return {
            "bucket_count": 0,
            "bucket_spread": math.nan,
            "best_bucket_mean": math.nan,
            "worst_bucket_mean": math.nan,
            "best_bucket": "",
            "worst_bucket": "",
        }
    grouped = pair.groupby("bucket", observed=True)["target"].agg(["count", "mean"]).reset_index()
    grouped = grouped[grouped["count"].ge(MIN_BUCKET_ROWS)].copy()
    if grouped.empty:
        return {
            "bucket_count": 0,
            "bucket_spread": math.nan,
            "best_bucket_mean": math.nan,
            "worst_bucket_mean": math.nan,
            "best_bucket": "",
            "worst_bucket": "",
        }
    best = grouped.loc[grouped["mean"].idxmin()]
    worst = grouped.loc[grouped["mean"].idxmax()]
    return {
        "bucket_count": int(len(grouped)),
        "bucket_spread": float(grouped["mean"].max() - grouped["mean"].min()),
        "best_bucket_mean": float(best["mean"]),
        "worst_bucket_mean": float(worst["mean"]),
        "best_bucket": str(best["bucket"]),
        "worst_bucket": str(worst["bucket"]),
    }


def feature_scoreboard(frame: pd.DataFrame, columns: list[str], *, min_rows: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        valid = frame[[column, "residual_0075_c", "abs_error_0075_c"]].dropna()
        if len(valid) < min_rows or valid[column].nunique() < 8:
            continue
        abs_bucket = quantile_bucket_stats(valid[column], valid["abs_error_0075_c"])
        residual_bucket = quantile_bucket_stats(valid[column], valid["residual_0075_c"])
        rows.append(
            {
                "feature": column,
                "family": feature_family(column),
                "n": int(len(valid)),
                "non_null_share": float(len(valid) / len(frame)),
                "corr_abs_error": safe_corr(valid[column], valid["abs_error_0075_c"]),
                "spearman_abs_error": safe_corr(valid[column], valid["abs_error_0075_c"], method="spearman"),
                "corr_signed_residual": safe_corr(valid[column], valid["residual_0075_c"]),
                "spearman_signed_residual": safe_corr(
                    valid[column], valid["residual_0075_c"], method="spearman"
                ),
                "abs_error_bucket_spread_c": abs_bucket["bucket_spread"],
                "abs_error_best_bucket": abs_bucket["best_bucket"],
                "abs_error_best_bucket_mae": abs_bucket["best_bucket_mean"],
                "abs_error_worst_bucket": abs_bucket["worst_bucket"],
                "abs_error_worst_bucket_mae": abs_bucket["worst_bucket_mean"],
                "signed_residual_bucket_spread_c": residual_bucket["bucket_spread"],
                "signed_residual_low_bucket": residual_bucket["best_bucket"],
                "signed_residual_low_mean_c": residual_bucket["best_bucket_mean"],
                "signed_residual_high_bucket": residual_bucket["worst_bucket"],
                "signed_residual_high_mean_c": residual_bucket["worst_bucket_mean"],
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["priority_score"] = (
        out["corr_abs_error"].abs().fillna(0.0)
        + out["corr_signed_residual"].abs().fillna(0.0)
        + (out["abs_error_bucket_spread_c"].fillna(0.0) / 2.0)
        + (out["signed_residual_bucket_spread_c"].abs().fillna(0.0) / 3.0)
    )
    return out.sort_values(["priority_score", "n"], ascending=[False, False]).reset_index(drop=True)


def source_feature_scoreboard(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source, group in frame.groupby("forecast_source_family", observed=True):
        board = feature_scoreboard(group.copy(), columns, min_rows=MIN_SOURCE_FEATURE_ROWS)
        if board.empty:
            continue
        board.insert(0, "forecast_source_family", source)
        rows.append(board.head(100))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["priority_score", "n"], ascending=[False, False]
    ).reset_index(drop=True)


def contrast_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    abs_threshold = float(frame["abs_error_0075_c"].quantile(0.90))
    under_threshold = float(frame["residual_0075_c"].quantile(0.90))
    over_threshold = float(frame["residual_0075_c"].quantile(0.10))
    masks = {
        "top10_abs_error": frame["abs_error_0075_c"].ge(abs_threshold),
        "top10_underprediction": frame["residual_0075_c"].ge(under_threshold),
        "top10_overprediction": frame["residual_0075_c"].le(over_threshold),
    }
    rows: list[dict[str, object]] = []
    for name, mask in masks.items():
        baseline = frame.loc[~mask].copy()
        cluster = frame.loc[mask].copy()
        for column in columns:
            feature = pd.to_numeric(frame[column], errors="coerce")
            valid_cluster = feature.loc[cluster.index].dropna()
            valid_baseline = feature.loc[baseline.index].dropna()
            if len(valid_cluster) < MIN_BUCKET_ROWS or len(valid_baseline) < MIN_FEATURE_ROWS:
                continue
            std = float(feature.std(skipna=True))
            if not math.isfinite(std) or std <= 0.0:
                continue
            cluster_mean = float(valid_cluster.mean())
            baseline_mean = float(valid_baseline.mean())
            rows.append(
                {
                    "cluster": name,
                    "feature": column,
                    "family": feature_family(column),
                    "cluster_rows": int(len(valid_cluster)),
                    "baseline_rows": int(len(valid_baseline)),
                    "cluster_mean": cluster_mean,
                    "baseline_mean": baseline_mean,
                    "standardized_mean_diff": (cluster_mean - baseline_mean) / std,
                    "absolute_mean_diff": cluster_mean - baseline_mean,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["cluster", "standardized_mean_diff"], ascending=[True, False]
    ).reset_index(drop=True)


def family_summary(scoreboard: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return pd.DataFrame()
    grouped = scoreboard.groupby("family", observed=True).agg(
        feature_count=("feature", "count"),
        median_abs_corr_error=("corr_abs_error", lambda values: float(values.abs().median())),
        max_abs_corr_error=("corr_abs_error", lambda values: float(values.abs().max())),
        median_abs_corr_residual=("corr_signed_residual", lambda values: float(values.abs().median())),
        max_abs_corr_residual=("corr_signed_residual", lambda values: float(values.abs().max())),
        max_abs_error_bucket_spread_c=("abs_error_bucket_spread_c", "max"),
        max_signed_residual_bucket_spread_c=("signed_residual_bucket_spread_c", "max"),
        top_priority_score=("priority_score", "max"),
    )
    return grouped.reset_index().sort_values("top_priority_score", ascending=False)


def build_joined_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    router_frame, _summary_0069, _summary_0074, summary_0075 = build_router_frame()
    feature_matrix = pd.read_parquet(FEATURE_MATRIX_PATH)
    feature_matrix["target_date"] = pd.to_datetime(feature_matrix["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(feature_matrix["target_date"], context="0077 feature matrix")
    feature_matrix = feature_matrix[feature_matrix["target_date"].lt(pd.Timestamp("2024-01-01"))].copy()
    router_subset = router_frame[
        [
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "m0075_prediction_c",
        ]
    ].copy()
    router_subset["target_date"] = pd.to_datetime(router_subset["target_date"], errors="coerce").dt.normalize()
    joined = router_subset.merge(feature_matrix, on="target_date", how="left", suffixes=("_current", "_feature"))
    joined["current_target_tmax_c"] = pd.to_numeric(joined["target_tmax_c_current"], errors="coerce")
    joined["residual_0075_c"] = joined["current_target_tmax_c"] - pd.to_numeric(
        joined["m0075_prediction_c"], errors="coerce"
    )
    joined["abs_error_0075_c"] = joined["residual_0075_c"].abs()
    if joined["abs_error_0075_c"].isna().any():
        raise RuntimeError("0077 joined frame has missing 0075 errors")
    return joined, summary_0075


def leakage_audit(joined: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(joined["target_date"], errors="coerce").max() < pd.Timestamp("2024-01-01")),
            "evidence": f"last target_date {pd.to_datetime(joined['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "current_target_excluded_from_feature_scan",
            "passed": bool(
                "target_tmax_c" not in feature_columns
                and "current_target_tmax_c" not in feature_columns
                and not any(column.startswith("target_tmax_c") for column in feature_columns)
            ),
            "evidence": f"{len(feature_columns)} scanned numeric feature columns",
        },
        {
            "check_id": "diagnostic_only_no_model_promotion",
            "passed": True,
            "evidence": "0077 ranks remaining-error associations only; no candidate predictions are produced",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    scoreboard: pd.DataFrame,
    source_scoreboard: pd.DataFrame,
    contrasts: pd.DataFrame,
    families: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    return f"""# Remaining 0075 Error Feature Mining

Generated: `{summary['generated_at_utc']}`

## Purpose

`0077` joins the long-history feature matrix to the current `0075` champion residuals and mines which station, upper-air, target-memory, daily-climate, and forecast-state attributes explain the remaining errors. This is a diagnostic feature-discovery pass, not a new predictive model.

## Data Contract

- Feature matrix range: `{summary['feature_matrix_first_date']}` to `{summary['feature_matrix_last_date']}`.
- Joined forecast-error range: `{summary['first_date']}` to `{summary['last_date']}`.
- Joined rows: `{summary['joined_rows']}`.
- Scanned numeric features: `{summary['numeric_feature_count']}`.
- No 2024+ confirmation rows are used.
- Current-day target Tmax is used only to compute the already-known research residual; it is excluded from feature ranking.

## Headline Findings

| Item | Value |
|---|---:|
| 0075 MAE | {summary['base_0075_mae']} |
| Top feature | {summary['top_feature']} |
| Top feature family | {summary['top_feature_family']} |
| Top feature priority score | {summary['top_feature_priority_score']} |
| Top abs-error corr feature | {summary['top_abs_error_corr_feature']} |
| Top signed-residual corr feature | {summary['top_signed_residual_corr_feature']} |
| Top high-error contrast feature | {summary['top_high_error_contrast_feature']} |

## Interpretation

The strongest rows here should drive the next specialist design. Features that are strong only in tiny source-specific slices remain diagnostic until they show enough pre-2024 support and timestamp eligibility.

## Overall Feature Scoreboard

{markdown_table(scoreboard, max_rows=120)}

## Source-Specific Feature Scoreboard

{markdown_table(source_scoreboard, max_rows=120)}

## Family Summary

{markdown_table(families, max_rows=40)}

## High-Error And Tail Contrasts

{markdown_table(contrasts, max_rows=120)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/feature_correlation_scoreboard.csv`
- `artifacts/source_feature_scoreboard.csv`
- `artifacts/family_summary.csv`
- `artifacts/high_error_contrasts.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_remaining_0075_error_feature_mining.py`:

- `{FOLDER_NAME}`: diagnostic feature mining for remaining `0075` residual and absolute-error structure.

| Metric | Value |
|---|---:|
| Joined rows | {summary['joined_rows']} |
| Scanned numeric features | {summary['numeric_feature_count']} |
| Top feature | {summary['top_feature']} |
| Top feature family | {summary['top_feature_family']} |
| Top high-error contrast feature | {summary['top_high_error_contrast_feature']} |

Leakage contract: no 2024+ rows; current target is excluded from feature scanning; diagnostic only.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Remaining 0075 Error Feature Mining",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_remaining_0075_error_feature_mining.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | long-history feature matrix joined to `0075` residuals | Diagnostic |
| Feature matrix range | `{summary['feature_matrix_first_date']}` to `{summary['feature_matrix_last_date']}` | Historical context |
| Joined residual rows | `{summary['joined_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Pre-2024 only |
| Scanned numeric features | `{summary['numeric_feature_count']}` | Tested |
| Top feature | `{summary['top_feature']}` | Highest combined priority |
| Top feature family | `{summary['top_feature_family']}` | Mechanism hint |
| Top abs-error corr feature | `{summary['top_abs_error_corr_feature']}` | Error-size hint |
| Top signed-residual corr feature | `{summary['top_signed_residual_corr_feature']}` | Bias-direction hint |
| Top high-error contrast feature | `{summary['top_high_error_contrast_feature']}` | Large-miss cluster hint |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0077` does not promote a model; it identifies the next residual-specialist targets for `0075` large-error clusters.
"""
    update_markdown_section(
        path,
        heading="Remaining 0075 Error Feature Mining",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"61. Remaining `0075` error mining scanned `{summary['numeric_feature_count']}` features; "
        f"top feature `{summary['top_feature']}` from `{summary['top_feature_family']}`, "
        f"top high-error contrast `{summary['top_high_error_contrast_feature']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Build `0078` as a strictly prior-only residual-specialist screen using the top `0077` families: target-memory/calendar transition state, IGRA lower-layer thermal/moisture state, and regional ISD station gradients. Require enough pre-2024 support and report both positive and negative specialist results.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    joined, summary_0075 = build_joined_frame()
    feature_matrix_dates = pd.to_datetime(pd.read_parquet(FEATURE_MATRIX_PATH, columns=["target_date"])["target_date"])
    feature_columns = numeric_feature_columns(joined)
    scoreboard = feature_scoreboard(joined, feature_columns, min_rows=MIN_FEATURE_ROWS)
    source_scoreboard = source_feature_scoreboard(joined, feature_columns)
    contrasts = contrast_table(joined, feature_columns)
    families = family_summary(scoreboard)
    leakage = leakage_audit(joined, feature_columns)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0077 leakage audit failed: {failed}")

    top_feature = scoreboard.iloc[0] if not scoreboard.empty else None
    top_abs = scoreboard.assign(abs_corr=scoreboard["corr_abs_error"].abs()).sort_values(
        "abs_corr", ascending=False
    ).iloc[0]
    top_signed = scoreboard.assign(abs_corr=scoreboard["corr_signed_residual"].abs()).sort_values(
        "abs_corr", ascending=False
    ).iloc[0]
    high_error = contrasts[contrasts["cluster"].eq("top10_abs_error")].copy()
    high_error["abs_std_diff"] = high_error["standardized_mean_diff"].abs()
    top_high_error = high_error.sort_values("abs_std_diff", ascending=False).iloc[0] if not high_error.empty else None

    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "feature_matrix_first_date": str(feature_matrix_dates.min().date()),
        "feature_matrix_last_date": str(feature_matrix_dates.max().date()),
        "joined_rows": int(len(joined)),
        "first_date": str(pd.to_datetime(joined["target_date"]).min().date()),
        "last_date": str(pd.to_datetime(joined["target_date"]).max().date()),
        "numeric_feature_count": int(len(feature_columns)),
        "base_0075_candidate": str(summary_0075["best_candidate"]),
        "base_0075_mae": float(summary_0075["best_mae"]),
        "top_feature": "" if top_feature is None else str(top_feature["feature"]),
        "top_feature_family": "" if top_feature is None else str(top_feature["family"]),
        "top_feature_priority_score": None if top_feature is None else float(top_feature["priority_score"]),
        "top_abs_error_corr_feature": str(top_abs["feature"]) if not scoreboard.empty else "",
        "top_abs_error_corr": float(top_abs["corr_abs_error"]) if not scoreboard.empty else None,
        "top_signed_residual_corr_feature": str(top_signed["feature"]) if not scoreboard.empty else "",
        "top_signed_residual_corr": float(top_signed["corr_signed_residual"]) if not scoreboard.empty else None,
        "top_high_error_contrast_feature": "" if top_high_error is None else str(top_high_error["feature"]),
        "top_high_error_contrast_std_diff": (
            None if top_high_error is None else float(top_high_error["standardized_mean_diff"])
        ),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "feature_correlation_scoreboard.csv", scoreboard)
    write_csv(artifacts / "source_feature_scoreboard.csv", source_scoreboard)
    write_csv(artifacts / "family_summary.csv", families)
    write_csv(artifacts / "high_error_contrasts.csv", contrasts)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "remaining_0075_error_feature_mining_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            source_scoreboard=source_scoreboard,
            contrasts=contrasts,
            families=families,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Mine remaining 0075 error feature associations.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
