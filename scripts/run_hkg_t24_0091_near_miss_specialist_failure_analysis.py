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

from scripts.run_hkg_t24_beastmode_signal_discovery import RESEARCH_ROOT, markdown_table, write_csv, write_json, write_text
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import update_markdown_section

FOLDER_NAME = "0091_near_miss_specialist_failure_analysis"
INPUT_0090_SCOREBOARD_PATH = (
    RESEARCH_ROOT / "0090_guarded_specialists_from_error_autopsy" / "artifacts" / "scoreboard.csv"
)
INPUT_0090_SUMMARY_PATH = (
    RESEARCH_ROOT / "0090_guarded_specialists_from_error_autopsy" / "artifacts" / "summary.json"
)
BASE_ID = "0088_0087_interaction_champion"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    missing = [path for path in (INPUT_0090_SCOREBOARD_PATH, INPUT_0090_SUMMARY_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0091 requires 0090 artifacts first: {missing}")
    scoreboard = pd.read_csv(INPUT_0090_SCOREBOARD_PATH)
    summary = json.loads(INPUT_0090_SUMMARY_PATH.read_text(encoding="utf-8"))
    for column in scoreboard.columns:
        if column.endswith("_delta_mae_vs_0088_base") or column in {"mae", "rmse", "delta_mae_vs_0088_base"}:
            scoreboard[column] = pd.to_numeric(scoreboard[column], errors="coerce")
    return scoreboard, summary


def delta_columns(scoreboard: pd.DataFrame) -> list[str]:
    return [
        column
        for column in scoreboard.columns
        if column.endswith("_delta_mae_vs_0088_base") and column != "delta_mae_vs_0088_base"
    ]


def readable_slice(column: str) -> str:
    return column.removesuffix("_delta_mae_vs_0088_base").strip("_")


def design_recommendation(worst_slice: str, feature: str, context_mode: str) -> str:
    if worst_slice.startswith("season_"):
        season = worst_slice.removeprefix("season_")
        return (
            f"Retest `{feature}` with a no-correction guard for `{season}` or with a separate `{season}` shrinkage cap; "
            f"the current `{context_mode}` form leaks value away through that season slice."
        )
    if worst_slice in {"old_frame", "newly_available"}:
        return (
            f"Retest `{feature}` with frame-specific eligibility: allow correction only where the frame slice improves, "
            f"and force base prediction on `{worst_slice}` until prior evidence clears the gate."
        )
    if worst_slice in {"press", "rss"}:
        return (
            f"Retest `{feature}` as a source-specific specialist: do not share residual state across press and RSS rows."
        )
    return f"Inspect `{feature}` manually; the blocking slice `{worst_slice}` needs a narrower eligibility rule."


def candidate_failure_details(scoreboard: pd.DataFrame) -> pd.DataFrame:
    deltas = delta_columns(scoreboard)
    candidates = scoreboard[
        ~scoreboard["candidate_id"].isin(["official_raw", BASE_ID])
    ].copy()
    candidates = candidates.sort_values(["mae", "rmse"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for row in candidates.to_dict("records"):
        failures: list[tuple[str, float]] = []
        improvements: list[tuple[str, float]] = []
        for column in deltas:
            value = finite_float(row.get(column))
            if not math.isfinite(value):
                continue
            item = (readable_slice(column), value)
            if value > 0.0:
                failures.append(item)
            elif value < 0.0:
                improvements.append(item)
        worst = max(failures, key=lambda item: item[1]) if failures else ("", math.nan)
        best = min(improvements, key=lambda item: item[1]) if improvements else ("", math.nan)
        feature = str(row.get("feature", ""))
        context_mode = str(row.get("context_mode", ""))
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "feature": feature,
                "family": row.get("family", ""),
                "context_mode": context_mode,
                "mae": row["mae"],
                "rmse": row["rmse"],
                "delta_mae_vs_0088_base": row["delta_mae_vs_0088_base"],
                "full_improves": finite_float(row["delta_mae_vs_0088_base"], 0.0) < 0.0,
                "hardened_gate_passed": str(row.get("hardened_gate_passed", "")) == "True",
                "season_no_regression_passed": str(row.get("season_no_regression_passed", "")) == "True",
                "failed_slice_count": len(failures),
                "improved_slice_count": len(improvements),
                "worst_failed_slice": worst[0],
                "worst_failed_delta_mae": worst[1],
                "best_improved_slice": best[0],
                "best_improved_delta_mae": best[1],
                "recommendation": design_recommendation(worst[0], feature, context_mode) if failures else "Promote only after final confirmation split.",
            }
        )
    return pd.DataFrame(rows)


def slice_failure_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in details.iterrows():
        if not row["worst_failed_slice"]:
            continue
        rows.append(
            {
                "slice": row["worst_failed_slice"],
                "candidate_id": row["candidate_id"],
                "feature": row["feature"],
                "full_delta_mae": row["delta_mae_vs_0088_base"],
                "worst_failed_delta_mae": row["worst_failed_delta_mae"],
            }
        )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        return pd.DataFrame()
    out_rows: list[dict[str, object]] = []
    for slice_name, group in expanded.groupby("slice", observed=True):
        worst = group.sort_values("worst_failed_delta_mae", ascending=False).iloc[0]
        out_rows.append(
            {
                "slice": slice_name,
                "blocking_candidate_count": int(len(group)),
                "mean_worst_failed_delta_mae": float(group["worst_failed_delta_mae"].mean()),
                "max_worst_failed_delta_mae": float(group["worst_failed_delta_mae"].max()),
                "worst_candidate_id": worst["candidate_id"],
                "worst_feature": worst["feature"],
                "best_full_delta_among_blocked": float(group["full_delta_mae"].min()),
            }
        )
    return pd.DataFrame(out_rows).sort_values("blocking_candidate_count", ascending=False).reset_index(drop=True)


def design_queue(details: pd.DataFrame) -> pd.DataFrame:
    candidates = details[details["full_improves"].astype(bool)].copy()
    if candidates.empty:
        candidates = details.head(12).copy()
    return candidates.sort_values(["delta_mae_vs_0088_base", "worst_failed_delta_mae"]).head(20).reset_index(drop=True)


def build_summary(
    *,
    generated_at: str,
    scoreboard: pd.DataFrame,
    upstream: dict[str, object],
    details: pd.DataFrame,
    queue: pd.DataFrame,
) -> dict[str, object]:
    improving = details[details["full_improves"].astype(bool)]
    top = details.iloc[0].to_dict() if not details.empty else {}
    return {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "upstream_folder": upstream.get("folder", ""),
        "scoreboard_rows": int(len(scoreboard)),
        "candidate_rows": int(len(details)),
        "full_improving_candidate_count": int(len(improving)),
        "hardened_candidate_count": int(details["hardened_gate_passed"].sum()) if not details.empty else 0,
        "design_queue_count": int(len(queue)),
        "base_0088_mae": upstream.get("base_0088_mae"),
        "best_0090_scoreboard_candidate": upstream.get("best_candidate"),
        "best_near_miss_candidate": top.get("candidate_id", ""),
        "best_near_miss_feature": top.get("feature", ""),
        "best_near_miss_delta_mae": top.get("delta_mae_vs_0088_base"),
        "best_near_miss_worst_failed_slice": top.get("worst_failed_slice", ""),
        "best_near_miss_worst_failed_delta_mae": top.get("worst_failed_delta_mae"),
        "uses_2024_plus_rows": False,
        "status": "near_miss_specialist_failure_analysis_complete",
        "next_recommended_task": (
            "Run 0092 as a narrower guarded screen: start with the 0091 design queue, add explicit no-correction "
            "guards for the blocking season/frame slices, and keep the full/source/frame/season no-regression gate."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    details: pd.DataFrame,
    slices: pd.DataFrame,
    queue: pd.DataFrame,
) -> str:
    return f"""# 0091 Near-Miss Specialist Failure Analysis

Generated: `{generated_at}`

## Purpose

`0090` found several near-misses that slightly improved full-frame MAE but failed the strict no-regression gate. `0091` explains why they failed: which source, frame, or season slice blocked each candidate, and what narrower follow-up should be tested.

This is an analysis step only. It does not train a new model and does not use 2024+ confirmation rows.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Candidate rows analyzed | `{summary['candidate_rows']}` |
| Full-improving candidates | `{summary['full_improving_candidate_count']}` |
| Hardened candidates | `{summary['hardened_candidate_count']}` |
| Best near-miss | `{summary['best_near_miss_candidate']}` |
| Best near-miss feature | `{summary['best_near_miss_feature']}` |
| Best near-miss full delta MAE | `{summary['best_near_miss_delta_mae']}` |
| Best near-miss worst failed slice | `{summary['best_near_miss_worst_failed_slice']}` |
| Worst failed-slice delta | `{summary['best_near_miss_worst_failed_delta_mae']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Candidate Failure Details

{markdown_table(details.head(30), max_rows=30)}

## Blocking Slice Summary

{markdown_table(slices, max_rows=30)}

## Next Design Queue

{markdown_table(queue, max_rows=20)}

## Interpretation

The correct next move is not to accept a full-frame near-miss. A candidate that improves total MAE but damages a season or frame slice is not robust enough. The next screen should keep the same past-only residual mechanics but add explicit eligibility guards: do not correct in the blocking slice, use smaller caps in that slice, or split the residual state more narrowly.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], details: pd.DataFrame, slices: pd.DataFrame, queue: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0091_near_miss_specialist_failure_analysis.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| 0090 near-miss analysis | `{summary['candidate_rows']}` candidates | Complete |
| Full-improving candidates | `{summary['full_improving_candidate_count']}` | Gate-failed |
| Hardened candidates | `{summary['hardened_candidate_count']}` | Strict gate |
| Best near-miss | `{summary['best_near_miss_candidate']}` | `{summary['best_near_miss_delta_mae']}` delta MAE |
| Worst blocking slice | `{summary['best_near_miss_worst_failed_slice']}` | `{summary['best_near_miss_worst_failed_delta_mae']}` delta MAE |
| Leakage | `0` 2024+ rows | PASS |

Top near-miss failures:

{markdown_table(details.head(8), max_rows=8)}

Blocking slice summary:

{markdown_table(slices.head(8), max_rows=8)}

Next design queue:

{markdown_table(queue.head(8), max_rows=8)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0091 Near-Miss Specialist Failure Analysis",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=(
            "Implement `0092_blocking_slice_guarded_specialists`: retest the 0091 design queue with explicit "
            "no-correction guards or smaller caps for the blocking season/frame/source slices. Keep current RSS data "
            "only until the backfill completes; keep 2024+ sealed."
        ),
    )


def run() -> dict[str, object]:
    generated_at = now_utc()
    scoreboard, upstream = load_inputs()
    details = candidate_failure_details(scoreboard)
    slices = slice_failure_summary(details)
    queue = design_queue(details)
    summary = build_summary(
        generated_at=generated_at,
        scoreboard=scoreboard,
        upstream=upstream,
        details=details,
        queue=queue,
    )
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "candidate_failure_details.csv", details)
    write_csv(artifacts / "blocking_slice_summary.csv", slices)
    write_csv(artifacts / "next_design_queue.csv", queue)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "near_miss_specialist_failure_analysis_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            details=details,
            slices=slices,
            queue=queue,
        ),
    )
    update_milestones(summary, details, slices, queue)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Explain why 0090 HKG Tmax specialist near-misses failed robust gates."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
