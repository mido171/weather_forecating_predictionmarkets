from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import update_markdown_section

FOLDER_NAME = "0103_current_rss_safe_continuation"
INPUT_0100_ATLAS_PATH = RESEARCH_ROOT / "0100_stable_mam_cell_feature_atlas" / "artifacts" / "feature_atlas.csv"
INPUT_0101_TOP_PREDICTIONS_PATH = (
    RESEARCH_ROOT / "0101_stable_mam_cell_feature_specialists" / "artifacts" / "top_predictions.csv"
)
INPUT_0101_DIAGNOSTICS_PATH = (
    RESEARCH_ROOT / "0101_stable_mam_cell_feature_specialists" / "artifacts" / "best_candidate_diagnostics.csv"
)
INPUT_0101_SUMMARY_PATH = (
    RESEARCH_ROOT / "0101_stable_mam_cell_feature_specialists" / "artifacts" / "summary.json"
)
INPUT_0102_SUMMARY_PATH = RESEARCH_ROOT / "0102_timestamp_proof_unlock_queue" / "artifacts" / "summary.json"

SAFE_FEATURE_FAMILIES = ("calendar_climatology", "isd_station_network", "target_memory")
BLOCKED_FEATURE_FAMILIES = ("upper_air", "hko_daily_climate", "marine_proxy")


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def safe_mean_abs(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.abs().mean()) if not clean.empty else math.nan


def safe_rmse(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(math.sqrt(float((clean * clean).mean()))) if not clean.empty else math.nan


def score_subset(frame: pd.DataFrame, *, slice_type: str, slice_value: str) -> dict[str, object]:
    clean = frame.dropna(subset=["target_date", "target_tmax_c", "forecast_max_c", "candidate_prediction_c"]).copy()
    if clean.empty:
        return {
            "slice_type": slice_type,
            "slice_value": slice_value,
            "n": 0,
            "first_date": "",
            "last_date": "",
            "official_mae": math.nan,
            "candidate_mae": math.nan,
            "delta_mae_vs_official": math.nan,
            "official_rmse": math.nan,
            "candidate_rmse": math.nan,
            "candidate_bias": math.nan,
            "active_correction_rows": 0,
            "mean_active_correction_c": math.nan,
        }
    official_error = clean["forecast_max_c"] - clean["target_tmax_c"]
    candidate_error = clean["candidate_prediction_c"] - clean["target_tmax_c"]
    active = bool_series(clean["specialist_active"]) if "specialist_active" in clean.columns else pd.Series(False, index=clean.index)
    corrections = (
        pd.to_numeric(clean["specialist_correction_c"], errors="coerce")
        if "specialist_correction_c" in clean.columns
        else pd.Series(math.nan, index=clean.index)
    )
    official_mae = safe_mean_abs(official_error)
    candidate_mae = safe_mean_abs(candidate_error)
    return {
        "slice_type": slice_type,
        "slice_value": slice_value,
        "n": int(len(clean)),
        "first_date": str(clean["target_date"].min().date()),
        "last_date": str(clean["target_date"].max().date()),
        "official_mae": official_mae,
        "candidate_mae": candidate_mae,
        "delta_mae_vs_official": float(candidate_mae - official_mae),
        "official_rmse": safe_rmse(official_error),
        "candidate_rmse": safe_rmse(candidate_error),
        "candidate_bias": float(candidate_error.mean()),
        "active_correction_rows": int(active.sum()),
        "mean_active_correction_c": float(corrections[active].mean()) if active.any() else 0.0,
    }


def load_predictions() -> pd.DataFrame:
    missing = [
        path
        for path in (INPUT_0101_TOP_PREDICTIONS_PATH, INPUT_0101_DIAGNOSTICS_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"0103 requires 0101 prediction artifacts first: {missing}")
    predictions = pd.read_csv(INPUT_0101_TOP_PREDICTIONS_PATH)
    diagnostics = pd.read_csv(INPUT_0101_DIAGNOSTICS_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    diagnostics["target_date"] = pd.to_datetime(diagnostics["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(predictions["target_date"], context="0103 top predictions")
    require_no_confirmation_dates(diagnostics["target_date"], context="0103 diagnostics")
    diagnostic_cols = [
        "target_date",
        "agreement_row",
        "specialist_active_row",
        "feature_bucket",
        "gate_active_row",
        "prior_rows",
        "prior_mean_residual_c",
        "specialist_active",
        "specialist_correction_c",
    ]
    available_diagnostic_cols = [col for col in diagnostic_cols if col in diagnostics.columns]
    merged = predictions.merge(
        diagnostics[available_diagnostic_cols],
        on="target_date",
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values("target_date").reset_index(drop=True)


def build_slice_scoreboard(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [score_subset(predictions, slice_type="overall", slice_value="all_pre_2024")]
    for column in ("forecast_source_family", "season", "frame_segment", "era_bucket"):
        if column not in predictions.columns:
            continue
        for value, group in predictions.groupby(column, sort=True, observed=True):
            rows.append(score_subset(group, slice_type=column, slice_value=str(value)))
    if {"forecast_source_family", "season"}.issubset(predictions.columns):
        for keys, group in predictions.groupby(["forecast_source_family", "season"], sort=True, observed=True):
            rows.append(score_subset(group, slice_type="source_x_season", slice_value=f"{keys[0]}__{keys[1]}"))
    if "forecast_source_family" in predictions.columns:
        rss = predictions[predictions["forecast_source_family"].astype(str).str.contains("rss", case=False, na=False)].copy()
        for year, group in rss.groupby(rss["target_date"].dt.year, sort=True):
            rows.append(score_subset(group, slice_type="rss_year", slice_value=str(year)))
    return pd.DataFrame(rows).sort_values(["slice_type", "slice_value"]).reset_index(drop=True)


def archive_gap_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(predictions["target_date"], errors="coerce").dropna().drop_duplicates().sort_values()
    rows: list[dict[str, object]] = []
    previous = None
    for current in dates:
        if previous is not None:
            gap_days = int((current - previous).days) - 1
            if gap_days > 0:
                rows.append(
                    {
                        "gap_start": str((previous + pd.Timedelta(days=1)).date()),
                        "gap_end": str((current - pd.Timedelta(days=1)).date()),
                        "missing_days": gap_days,
                        "previous_scored_date": str(previous.date()),
                        "next_scored_date": str(current.date()),
                    }
                )
        previous = current
    return pd.DataFrame(rows).sort_values("missing_days", ascending=False).reset_index(drop=True)


def build_correction_activation_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    if "specialist_active" not in predictions.columns:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["specialist_active_bool"] = bool_series(frame["specialist_active"])
    rows: list[dict[str, object]] = []
    group_columns = ["forecast_source_family", "season", "era_bucket"]
    for column in group_columns:
        if column not in frame.columns:
            continue
        for value, group in frame.groupby(column, sort=True, observed=True):
            active = group["specialist_active_bool"]
            rows.append(
                {
                    "slice_type": column,
                    "slice_value": str(value),
                    "rows": int(len(group)),
                    "active_rows": int(active.sum()),
                    "active_rate": float(active.mean()) if len(group) else math.nan,
                    "mean_active_correction_c": float(
                        pd.to_numeric(group.loc[active, "specialist_correction_c"], errors="coerce").mean()
                    )
                    if active.any() and "specialist_correction_c" in group.columns
                    else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["active_rows", "rows"], ascending=[False, False]).reset_index(drop=True)


def build_safe_feature_backlog() -> pd.DataFrame:
    if not INPUT_0100_ATLAS_PATH.exists():
        raise FileNotFoundError(f"0103 requires 0100 feature atlas first: {INPUT_0100_ATLAS_PATH}")
    atlas = pd.read_csv(INPUT_0100_ATLAS_PATH)
    atlas["allowed_for_future_walkforward_bool"] = bool_series(atlas["allowed_for_future_walkforward"])
    atlas["diagnostic_score"] = pd.to_numeric(atlas["diagnostic_score"], errors="coerce")
    safe = atlas[
        atlas["allowed_for_future_walkforward_bool"]
        & atlas["family"].isin(SAFE_FEATURE_FAMILIES)
        & ~atlas["family"].isin(BLOCKED_FEATURE_FAMILIES)
    ].copy()
    safe["0103_role"] = "future_allowed_backlog_not_timestamp_blocked"
    return safe.sort_values(["diagnostic_score", "feature"], ascending=[False, True]).reset_index(drop=True)


def build_rss_coverage_snapshot(predictions: pd.DataFrame) -> pd.DataFrame:
    rss = predictions[predictions["forecast_source_family"].astype(str).str.contains("rss", case=False, na=False)].copy()
    if rss.empty:
        return pd.DataFrame(
            [
                {
                    "source": "rss_archive",
                    "rows": 0,
                    "first_date": "",
                    "last_date": "",
                    "calendar_days_spanned": 0,
                    "coverage_ratio_inside_span": math.nan,
                    "note": "no scoreable RSS rows in current 0101 predictions",
                }
            ]
        )
    first = rss["target_date"].min()
    last = rss["target_date"].max()
    span_days = int((last - first).days) + 1
    return pd.DataFrame(
        [
            {
                "source": "rss_archive",
                "rows": int(len(rss)),
                "first_date": str(first.date()),
                "last_date": str(last.date()),
                "calendar_days_spanned": span_days,
                "coverage_ratio_inside_span": float(rss["target_date"].nunique() / span_days),
                "note": "current scoreable RSS window only; 2024+ target outcomes remain sealed",
            }
        ]
    )


def load_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    predictions = load_predictions()
    slice_scoreboard = build_slice_scoreboard(predictions)
    gaps = archive_gap_audit(predictions)
    activation = build_correction_activation_audit(predictions)
    safe_backlog = build_safe_feature_backlog()
    rss_snapshot = build_rss_coverage_snapshot(predictions)
    overall = score_subset(predictions, slice_type="overall", slice_value="all_pre_2024")
    rss_rows = slice_scoreboard[slice_scoreboard["slice_type"].eq("forecast_source_family")]
    rss_rows = rss_rows[rss_rows["slice_value"].astype(str).str.contains("rss", case=False, na=False)]
    largest_gap = gaps.iloc[0].to_dict() if not gaps.empty else {}
    summary_0101 = load_summary(INPUT_0101_SUMMARY_PATH)
    summary_0102 = load_summary(INPUT_0102_SUMMARY_PATH)
    summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "input_0101_best_candidate": summary_0101.get("best_candidate", ""),
        "input_0101_best_mae": summary_0101.get("best_mae", math.nan),
        "input_0101_best_rmse": summary_0101.get("best_rmse", math.nan),
        "input_0102_unlockable_feature_count": summary_0102.get("unlockable_feature_count", math.nan),
        "rows": int(len(predictions)),
        "first_target_date": str(predictions["target_date"].min().date()),
        "last_target_date": str(predictions["target_date"].max().date()),
        "overall_official_mae": overall["official_mae"],
        "overall_candidate_mae": overall["candidate_mae"],
        "overall_delta_mae_vs_official": overall["delta_mae_vs_official"],
        "rss_rows": int(rss_snapshot.iloc[0]["rows"]),
        "rss_first_date": str(rss_snapshot.iloc[0]["first_date"]),
        "rss_last_date": str(rss_snapshot.iloc[0]["last_date"]),
        "rss_candidate_mae": float(rss_rows.iloc[0]["candidate_mae"]) if not rss_rows.empty else math.nan,
        "rss_delta_mae_vs_official": float(rss_rows.iloc[0]["delta_mae_vs_official"]) if not rss_rows.empty else math.nan,
        "safe_backlog_features": int(len(safe_backlog)),
        "largest_gap_start": str(largest_gap.get("gap_start", "")),
        "largest_gap_end": str(largest_gap.get("gap_end", "")),
        "largest_gap_missing_days": int(largest_gap.get("missing_days", 0)) if largest_gap else 0,
        "strict_deployable_model_promoted": False,
        "uses_2024_plus_target_rows": False,
        "status": "current_rss_continuation_analysis_only_complete",
        "next_recommended_task": (
            "Run 0104_safe_feature_interaction_stability_lab: use only future-allowed calendar, station, "
            "and target-memory features to test smoother interaction/stack stability by source and season; "
            "do not use upper-air, HKO daily climate, or marine proxies until 0102 is unlocked."
        ),
    }
    return slice_scoreboard, gaps, activation, safe_backlog, rss_snapshot, summary


def build_readme(
    *,
    slice_scoreboard: pd.DataFrame,
    gaps: pd.DataFrame,
    activation: pd.DataFrame,
    safe_backlog: pd.DataFrame,
    rss_snapshot: pd.DataFrame,
    summary: dict[str, object],
) -> str:
    slice_cols = [
        "slice_type",
        "slice_value",
        "n",
        "first_date",
        "last_date",
        "official_mae",
        "candidate_mae",
        "delta_mae_vs_official",
        "active_correction_rows",
    ]
    feature_cols = [
        "feature",
        "family",
        "diagnostic_score",
        "timestamp_audit_status",
        "allowed_for_future_walkforward",
        "first_non_null_date",
        "last_non_null_date",
        "0103_role",
    ]
    return f"""# 0103 Current RSS Continuation Without Blocked Sources

Generated: `{summary['generated_at_utc']}`

## Purpose

The forecast backfill is still in progress, so this analysis continues with the currently scoreable pre-2024 archive. It does not unlock or newly use upper-air, HKO daily climate, or marine proxy features. Those families remain diagnostic-only because `0102` found zero timestamp-proof unlocks.

`0103` answers three practical questions:

1. How does the current `0101` research candidate score by source, season, era, and current RSS slice?
2. How severe is the current scoreable archive gap while backfill continues?
3. Which future-allowed feature families remain available for the next safe interaction/stability lab?

## Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows scored | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Input 0101 candidate | `{summary['input_0101_best_candidate']}` |
| Input 0101 MAE/RMSE | `{summary['input_0101_best_mae']}` / `{summary['input_0101_best_rmse']}` |
| 0102 unlockable blocked features | `{summary['input_0102_unlockable_feature_count']}` |
| Overall official MAE | `{summary['overall_official_mae']}` |
| Overall candidate MAE | `{summary['overall_candidate_mae']}` |
| Overall delta vs official | `{summary['overall_delta_mae_vs_official']}` |
| Current RSS rows | `{summary['rss_rows']}` |
| Current RSS range | `{summary['rss_first_date']}` to `{summary['rss_last_date']}` |
| Current RSS candidate MAE | `{summary['rss_candidate_mae']}` |
| Current RSS delta vs official | `{summary['rss_delta_mae_vs_official']}` |
| Future-allowed safe backlog features | `{summary['safe_backlog_features']}` |
| Largest archive gap | `{summary['largest_gap_start']}` to `{summary['largest_gap_end']}` |
| Largest gap missing days | `{summary['largest_gap_missing_days']}` |
| Strict deployable model promoted | `{summary['strict_deployable_model_promoted']}` |
| 2024+ target rows used | `{summary['uses_2024_plus_target_rows']}` |

## Main Slice Scores

{markdown_table(slice_scoreboard[slice_cols].head(45), max_rows=45)}

## Current RSS Snapshot

{markdown_table(rss_snapshot, max_rows=10)}

## Archive Gap Audit

{markdown_table(gaps.head(20), max_rows=20)}

## Correction Activation Audit

{markdown_table(activation.head(40), max_rows=40)}

## Safe Feature Backlog

{markdown_table(safe_backlog[feature_cols].head(35), max_rows=35)}

## Leakage Controls

All prediction rows are before `{CONFIRMATION_START.date().isoformat()}`. This script trains no model and promotes no deployable candidate. It reads `0101` predictions and diagnostics only for slice scoring, then reads `0100` future-allowed rows only to define the safe next-feature backlog. Upper-air, HKO daily climate, and marine proxy families remain excluded from promotion until timestamp/publication evidence is attached.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(
    summary: dict[str, object],
    slice_scoreboard: pd.DataFrame,
    gaps: pd.DataFrame,
    safe_backlog: pd.DataFrame,
) -> None:
    slice_cols = [
        "slice_type",
        "slice_value",
        "n",
        "official_mae",
        "candidate_mae",
        "delta_mae_vs_official",
    ]
    feature_cols = ["feature", "family", "diagnostic_score", "timestamp_audit_status"]
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0103_current_rss_continuation_without_blocked_sources.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Rows scored | `{summary['rows']}` | Pre-2024 current archive |
| Overall candidate MAE | `{summary['overall_candidate_mae']}` | Existing 0101 research candidate |
| Current RSS rows/range | `{summary['rss_rows']}` / `{summary['rss_first_date']}` to `{summary['rss_last_date']}` | Current available scoreable RSS |
| Current RSS candidate MAE | `{summary['rss_candidate_mae']}` | Slice score |
| Largest archive gap | `{summary['largest_gap_start']}` to `{summary['largest_gap_end']}` | Backfill still needed |
| Safe backlog features | `{summary['safe_backlog_features']}` | Calendar + station + target memory |
| Strict deployable model promoted | `{summary['strict_deployable_model_promoted']}` | Analysis only |
| Leakage | `0` 2024+ target rows | PASS |

Top slice scores:

{markdown_table(slice_scoreboard[slice_cols].head(20), max_rows=20)}

Largest gaps:

{markdown_table(gaps.head(10), max_rows=10)}

Top safe backlog:

{markdown_table(safe_backlog[feature_cols].head(15), max_rows=15)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0103 Current RSS Continuation Without Blocked Sources",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    slice_scoreboard, gaps, activation, safe_backlog, rss_snapshot, summary = build_outputs()
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "slices.csv", slice_scoreboard)
    write_csv(artifacts / "gaps.csv", gaps)
    write_csv(artifacts / "activation.csv", activation)
    write_csv(artifacts / "safe_feature_backlog.csv", safe_backlog)
    write_csv(artifacts / "rss_snapshot.csv", rss_snapshot)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "current_rss_safe_continuation_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            slice_scoreboard=slice_scoreboard,
            gaps=gaps,
            activation=activation,
            safe_backlog=safe_backlog,
            rss_snapshot=rss_snapshot,
            summary=summary,
        ),
    )
    update_milestones(summary, slice_scoreboard, gaps, safe_backlog)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-summary", action="store_true", help="Print JSON summary after writing artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run()
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
