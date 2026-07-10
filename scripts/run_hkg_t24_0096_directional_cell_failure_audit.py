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

FOLDER_NAME = "0096_directional_cell_failure_audit"
INPUT_0095_SUMMARY_PATH = RESEARCH_ROOT / "0095_mam_error_direction_split_lab" / "artifacts" / "summary.json"
INPUT_0095_TOP_PATH = RESEARCH_ROOT / "0095_mam_error_direction_split_lab" / "artifacts" / "top_predictions.csv"
INPUT_0095_DIAGNOSTICS_PATH = (
    RESEARCH_ROOT / "0095_mam_error_direction_split_lab" / "artifacts" / "best_gate_diagnostics.csv"
)
INPUT_0094_TOP_PATH = RESEARCH_ROOT / "0094_expanded_high_error_interaction_lab" / "artifacts" / "top_predictions.csv"
MIN_CELL_ROWS = 20
MEANINGFUL_DELTA_C = 0.005


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def mam_submonth(target_date: pd.Timestamp) -> str:
    if target_date.month == 3:
        return "march"
    if target_date.month == 4:
        return "april"
    if target_date.month == 5:
        return "may"
    return "non_mam"


def cell_status(delta_mae_candidate_minus_base: float, n_rows: int) -> str:
    if n_rows < MIN_CELL_ROWS or not math.isfinite(delta_mae_candidate_minus_base):
        return "too_sparse"
    if delta_mae_candidate_minus_base <= -MEANINGFUL_DELTA_C:
        return "stable_improving"
    if delta_mae_candidate_minus_base >= MEANINGFUL_DELTA_C:
        return "damaging"
    return "neutral"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    missing = [
        path
        for path in (INPUT_0095_SUMMARY_PATH, INPUT_0095_TOP_PATH, INPUT_0095_DIAGNOSTICS_PATH, INPUT_0094_TOP_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"0096 requires 0094 and 0095 artifacts first: {missing}")
    summary_0095 = json.loads(INPUT_0095_SUMMARY_PATH.read_text(encoding="utf-8"))
    top_0095 = pd.read_csv(INPUT_0095_TOP_PATH)
    top_0094 = pd.read_csv(INPUT_0094_TOP_PATH)
    diagnostics = pd.read_csv(INPUT_0095_DIAGNOSTICS_PATH)
    for frame in (top_0095, top_0094, diagnostics):
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
        frame.dropna(subset=["target_date"], inplace=True)
        frame.sort_values(["target_date", "forecast_source_family"], inplace=True)
        require_no_confirmation_dates(frame["target_date"], context="0096 input")
    return top_0095, top_0094, diagnostics, summary_0095


def build_analysis_frame(top_0095: pd.DataFrame, top_0094: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    key = ["target_date", "forecast_source_family"]
    base = top_0094[
        [
            *key,
            "candidate_prediction_c",
            "candidate_error_c",
        ]
    ].rename(
        columns={
            "candidate_prediction_c": "base_0094_prediction_c",
            "candidate_error_c": "base_0094_error_c",
        }
    )
    candidate = top_0095[
        [
            *key,
            "target_tmax_c",
            "forecast_max_c",
            "season",
            "frame_segment",
            "era_bucket",
            "candidate_id",
            "candidate_prediction_c",
            "candidate_error_c",
        ]
    ].rename(
        columns={
            "candidate_prediction_c": "candidate_0095_prediction_c",
            "candidate_error_c": "candidate_0095_error_c",
        }
    )
    diag_cols = [
        *key,
        "pair_name",
        "active_gate",
        "direction_mode",
        "pair_bucket",
        "gate_active_row",
        "prior_rows",
        "prior_mean_residual_c",
        "prior_direction",
        "specialist_active",
        "specialist_correction_c",
    ]
    frame = candidate.merge(base, on=key, how="inner").merge(diagnostics[diag_cols], on=key, how="inner")
    frame["base_abs_error_c"] = frame["base_0094_error_c"].abs()
    frame["candidate_abs_error_c"] = frame["candidate_0095_error_c"].abs()
    frame["abs_error_improvement_c"] = frame["base_abs_error_c"] - frame["candidate_abs_error_c"]
    frame["delta_abs_error_candidate_minus_base_c"] = frame["candidate_abs_error_c"] - frame["base_abs_error_c"]
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame["month"] = frame["target_date"].dt.month
    frame["mam_submonth"] = frame["target_date"].map(mam_submonth)
    frame["pair_bucket"] = pd.to_numeric(frame["pair_bucket"], errors="coerce")
    frame["pair_bucket_label"] = frame["pair_bucket"].map(lambda value: "missing" if pd.isna(value) else f"bucket_{int(value)}")
    frame["gate_active_row"] = frame["gate_active_row"].astype(bool)
    frame["specialist_active"] = frame["specialist_active"].astype(bool)
    require_no_confirmation_dates(frame["target_date"], context="0096 analysis frame")
    return frame.sort_values(key).reset_index(drop=True)


def summarize_groups(frame: pd.DataFrame, group_columns: list[str], *, active_only: bool = True) -> pd.DataFrame:
    scoped = frame[frame["gate_active_row"]].copy() if active_only else frame.copy()
    rows: list[dict[str, object]] = []
    for values, group in scoped.groupby(group_columns, observed=True, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        base_mae = float(group["base_abs_error_c"].mean())
        candidate_mae = float(group["candidate_abs_error_c"].mean())
        delta = candidate_mae - base_mae
        row = {column: value for column, value in zip(group_columns, values, strict=True)}
        row.update(
            {
                "n": int(len(group)),
                "first_date": group["target_date"].min().date().isoformat(),
                "last_date": group["target_date"].max().date().isoformat(),
                "specialist_active_rows": int(group["specialist_active"].sum()),
                "base_0094_mae": base_mae,
                "candidate_0095_mae": candidate_mae,
                "delta_mae_candidate_minus_base": delta,
                "mean_abs_error_improvement_c": float(group["abs_error_improvement_c"].mean()),
                "median_abs_error_improvement_c": float(group["abs_error_improvement_c"].median()),
                "mean_specialist_correction_c": float(pd.to_numeric(group["specialist_correction_c"], errors="coerce").mean()),
                "median_prior_rows": float(pd.to_numeric(group["prior_rows"], errors="coerce").median()),
                "status": cell_status(delta, len(group)),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["delta_mae_candidate_minus_base", "n"], ascending=[True, False])


def worst_cases(frame: pd.DataFrame, *, limit: int = 80) -> pd.DataFrame:
    cols = [
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "base_0094_prediction_c",
        "candidate_0095_prediction_c",
        "base_abs_error_c",
        "candidate_abs_error_c",
        "delta_abs_error_candidate_minus_base_c",
        "prior_direction",
        "pair_bucket_label",
        "mam_submonth",
        "specialist_correction_c",
        "prior_rows",
    ]
    return frame[frame["gate_active_row"]].sort_values("delta_abs_error_candidate_minus_base_c", ascending=False)[
        cols
    ].head(limit)


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    top_0095, top_0094, diagnostics, summary_0095 = load_inputs()
    frame = build_analysis_frame(top_0095, top_0094, diagnostics)
    direction_summary = summarize_groups(frame, ["prior_direction"])
    bucket_summary = summarize_groups(frame, ["pair_bucket_label", "prior_direction"])
    source_submonth_summary = summarize_groups(frame, ["forecast_source_family", "mam_submonth", "prior_direction"])
    cases = worst_cases(frame)
    stable_cells = bucket_summary[bucket_summary["status"].eq("stable_improving")].copy()
    damaging_cells = bucket_summary[bucket_summary["status"].eq("damaging")].copy()
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "rows": int(len(frame)),
        "gate_active_rows": int(frame["gate_active_row"].sum()),
        "specialist_active_rows": int(frame["specialist_active"].sum()),
        "first_target_date": dates.min().date().isoformat(),
        "last_target_date": dates.max().date().isoformat(),
        "input_0095_best_candidate": summary_0095["best_candidate"],
        "input_0095_best_mae": float(summary_0095["best_mae"]),
        "input_0095_best_rmse": float(summary_0095["best_rmse"]),
        "input_0095_delta_vs_0094": float(summary_0095["best_delta_mae_vs_0094_base"]),
        "stable_improving_bucket_direction_cells": int(len(stable_cells)),
        "damaging_bucket_direction_cells": int(len(damaging_cells)),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "directional_cell_failure_audit_complete",
        "next_recommended_task": (
            "Run 0097_stable_directional_cell_specialist: build a bounded specialist from only the stable "
            "0096 improving bucket-direction cells, with explicit guards for the damaging cells and no 2024+ rows."
        ),
    }
    return frame, direction_summary, bucket_summary, source_submonth_summary, cases, summary


def build_readme(
    *,
    summary: dict[str, object],
    direction_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    source_submonth_summary: pd.DataFrame,
    cases: pd.DataFrame,
) -> str:
    return f"""# 0096 Directional Cell Failure Audit

Generated: `{summary['generated_at_utc']}`

## Purpose

`0095` improved the current pre-2024 champion by splitting MAM corrections by prior residual direction. This audit does not create a new forecast. It explains where that gain came from and where it still caused damage, so the next specialist can target stable cells instead of widening the rule blindly.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Gate-active rows | `{summary['gate_active_rows']}` |
| Specialist-active rows | `{summary['specialist_active_rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Input 0095 best | `{summary['input_0095_best_candidate']}` |
| Input 0095 MAE | `{summary['input_0095_best_mae']}` |
| Input 0095 RMSE | `{summary['input_0095_best_rmse']}` |
| Delta vs 0094 | `{summary['input_0095_delta_vs_0094']}` |
| Stable bucket-direction cells | `{summary['stable_improving_bucket_direction_cells']}` |
| Damaging bucket-direction cells | `{summary['damaging_bucket_direction_cells']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Plain-English Finding

The audit compares the 0095 candidate against the 0094 baseline row by row. Positive `mean_abs_error_improvement_c` means 0095 reduced absolute error. Negative values mean the direction split made the row worse.

The next step should not simply increase correction strength. It should keep the stable bucket-direction cells and explicitly guard out the damaging cells. This is especially important because the total MAE gain is real but small, so a few unstable spring cells can erase it.

## Direction Summary

{markdown_table(direction_summary, max_rows=30)}

## Bucket-Direction Summary

{markdown_table(bucket_summary.head(60), max_rows=60)}

## Source/Submonth/Direction Summary

{markdown_table(source_submonth_summary.head(60), max_rows=60)}

## Worst Damaging Rows

{markdown_table(cases.head(60), max_rows=60)}

## Leakage Controls

This is an audit of existing 0094 and 0095 pre-2024 predictions only. It does not compute a new prediction, does not tune on 2024+ rows, and does not use target outcomes to decide a deployable action. The output is a design queue for the next leakage-safe candidate.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(
    summary: dict[str, object],
    direction_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    source_submonth_summary: pd.DataFrame,
) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0096_directional_cell_failure_audit.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Audited 0095 gate-active rows | `{summary['gate_active_rows']}` | Pre-2024 only |
| Specialist-active rows | `{summary['specialist_active_rows']}` | Existing 0095 best |
| Stable bucket-direction cells | `{summary['stable_improving_bucket_direction_cells']}` | Candidate queue |
| Damaging bucket-direction cells | `{summary['damaging_bucket_direction_cells']}` | Guard queue |
| Input 0095 MAE | `{summary['input_0095_best_mae']}` | Current champion |
| Leakage | `0` 2024+ rows | PASS |

Direction summary:

{markdown_table(direction_summary.head(12), max_rows=12)}

Top bucket-direction cells:

{markdown_table(bucket_summary.head(12), max_rows=12)}

Source/submonth summary:

{markdown_table(source_submonth_summary.head(12), max_rows=12)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0096 Directional Cell Failure Audit",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    frame, direction_summary, bucket_summary, source_submonth_summary, cases, summary = build_outputs()
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "analysis_frame.csv", frame)
    write_csv(artifacts / "direction_summary.csv", direction_summary)
    write_csv(artifacts / "bucket_direction_summary.csv", bucket_summary)
    write_csv(artifacts / "source_submonth_direction_summary.csv", source_submonth_summary)
    write_csv(artifacts / "worst_damaging_rows.csv", cases)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "directional_cell_failure_audit_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            direction_summary=direction_summary,
            bucket_summary=bucket_summary,
            source_submonth_summary=source_submonth_summary,
            cases=cases,
        ),
    )
    update_milestones(summary, direction_summary, bucket_summary, source_submonth_summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Audit 0095 direction-split gains and failures by prior direction and cell."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
