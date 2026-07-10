from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import apply_specialist
from scripts.run_hkg_t24_0090_guarded_specialists_from_error_autopsy import (
    BASE_ID,
    build_working_frame,
    evaluation_masks,
    load_inputs,
    make_specs,
    score_candidate,
)
from scripts.run_hkg_t24_0091_near_miss_specialist_failure_analysis import readable_slice
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

FOLDER_NAME = "0092_blocking_slice_guarded_specialists"
INPUT_0090_SCOREBOARD_PATH = (
    RESEARCH_ROOT / "0090_guarded_specialists_from_error_autopsy" / "artifacts" / "scoreboard.csv"
)
INPUT_0091_QUEUE_PATH = (
    RESEARCH_ROOT / "0091_near_miss_specialist_failure_analysis" / "artifacts" / "next_design_queue.csv"
)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_queue_and_scoreboard() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [path for path in (INPUT_0090_SCOREBOARD_PATH, INPUT_0091_QUEUE_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0092 requires 0090 and 0091 artifacts first: {missing}")
    scoreboard = pd.read_csv(INPUT_0090_SCOREBOARD_PATH)
    queue = pd.read_csv(INPUT_0091_QUEUE_PATH)
    for column in scoreboard.columns:
        if column.endswith("_delta_mae_vs_0088_base") or column in {"mae", "rmse", "delta_mae_vs_0088_base"}:
            scoreboard[column] = pd.to_numeric(scoreboard[column], errors="coerce")
    return queue, scoreboard


def failed_slices_for_candidate(scoreboard: pd.DataFrame, candidate_id: str) -> list[str]:
    row = scoreboard[scoreboard["candidate_id"].eq(candidate_id)]
    if row.empty:
        return []
    record = row.iloc[0]
    slices: list[str] = []
    for column in scoreboard.columns:
        if not column.endswith("_delta_mae_vs_0088_base") or column == "delta_mae_vs_0088_base":
            continue
        value = pd.to_numeric(pd.Series([record[column]]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 0.0:
            slices.append(readable_slice(column))
    return slices


def mask_for_slice(frame: pd.DataFrame, slice_name: str) -> np.ndarray:
    source = frame["forecast_source_family"].astype(str)
    segment = frame["frame_segment"].astype(str)
    season = frame["season"].astype(str)
    if slice_name == "press":
        return source.eq("press_archive").to_numpy(dtype=bool)
    if slice_name == "rss":
        return source.eq("rss_archive").to_numpy(dtype=bool)
    if slice_name == "old_frame":
        return segment.eq("current_0081_frame").to_numpy(dtype=bool)
    if slice_name == "newly_available":
        return segment.eq("newly_available_official_frame").to_numpy(dtype=bool)
    if slice_name.startswith("season_"):
        return season.eq(slice_name.removeprefix("season_")).to_numpy(dtype=bool)
    return np.zeros(len(frame), dtype=bool)


def apply_no_correction_guards(
    *,
    frame: pd.DataFrame,
    prediction: np.ndarray,
    failed_slices: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    guarded = prediction.copy()
    rows: list[dict[str, object]] = []
    for slice_name in failed_slices:
        mask = mask_for_slice(frame, slice_name)
        guarded[mask] = base[mask]
        rows.append({"slice": slice_name, "guarded_rows": int(mask.sum())})
    return guarded, pd.DataFrame(rows)


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features, base, leads = load_inputs()
    queue, previous_scoreboard = load_queue_and_scoreboard()
    frame, thresholds = build_working_frame(features, base, leads)
    spec_by_id = {spec.candidate_id: spec for spec in make_specs(leads, thresholds)}
    mask_map = evaluation_masks(frame)
    base_prediction = frame["candidate_prediction_c"].to_numpy(dtype=float)
    raw_prediction = frame["forecast_max_c"].to_numpy(dtype=float)
    rows = [
        score_candidate(
            frame,
            candidate_id="official_raw",
            candidate_class="official_raw",
            prediction=raw_prediction,
            mask_map=mask_map,
        ),
        score_candidate(
            frame,
            candidate_id=BASE_ID,
            candidate_class="0088_base",
            prediction=base_prediction,
            mask_map=mask_map,
        ),
    ]
    guard_rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {BASE_ID: base_prediction, "official_raw": raw_prediction}
    for queued in queue.to_dict("records"):
        original_id = str(queued["candidate_id"])
        spec = spec_by_id.get(original_id)
        if spec is None:
            continue
        prediction, _diagnostics = apply_specialist(frame, spec)
        failed_slices = failed_slices_for_candidate(previous_scoreboard, original_id)
        guarded, guard_table = apply_no_correction_guards(
            frame=frame,
            prediction=prediction,
            failed_slices=failed_slices,
        )
        guarded_id = f"guarded_failed_slices__{original_id}"
        predictions[guarded_id] = guarded
        if not guard_table.empty:
            guard_table["candidate_id"] = guarded_id
            guard_table["original_candidate_id"] = original_id
            guard_rows.extend(guard_table.to_dict("records"))
        rows.append(
            score_candidate(
                frame,
                candidate_id=guarded_id,
                candidate_class="0092_blocking_slice_guarded_specialist",
                prediction=guarded,
                mask_map=mask_map,
                extra={
                    "original_candidate_id": original_id,
                    "feature": spec.feature,
                    "context_mode": spec.context_mode,
                    "failed_slices_guarded": ";".join(failed_slices),
                    "guarded_slice_count": len(failed_slices),
                },
            )
        )
    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    if hardened.empty:
        best_id = BASE_ID
    else:
        best_id = str(hardened.sort_values(["mae", "rmse"]).iloc[0]["candidate_id"])
    top_predictions = frame[
        [
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "forecast_max_c",
            "season",
            "frame_segment",
            "era_bucket",
        ]
    ].copy()
    top_predictions["candidate_id"] = best_id
    top_predictions["candidate_prediction_c"] = predictions[best_id]
    top_predictions["candidate_error_c"] = top_predictions["candidate_prediction_c"] - top_predictions["target_tmax_c"]
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    base_row = scoreboard[scoreboard["candidate_id"].eq(BASE_ID)].iloc[0]
    best_row = scoreboard[scoreboard["candidate_id"].eq(best_id)].iloc[0]
    summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "rows": int(len(frame)),
        "first_target_date": dates.min().date().isoformat(),
        "last_target_date": dates.max().date().isoformat(),
        "queued_candidate_count": int(len(queue)),
        "candidate_count": int(len(scoreboard)),
        "hardened_candidate_count": int(len(hardened)),
        "base_0088_mae": float(base_row["mae"]),
        "base_0088_rmse": float(base_row["rmse"]),
        "best_candidate": str(best_id),
        "best_mae": float(best_row["mae"]),
        "best_rmse": float(best_row["rmse"]),
        "best_delta_mae_vs_0088_base": float(best_row["delta_mae_vs_0088_base"]),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "blocking_slice_guarded_specialists_complete",
        "next_recommended_task": (
            "Run 0093_guarded_champion_sensitivity_check: stress-test the 0092 guarded champion with alternative "
            "guard definitions, guard subsets, correction caps, and min-history settings before treating it as the "
            "new pre-2024 research champion."
        ),
    }
    require_no_confirmation_dates(top_predictions["target_date"], context="0092 top predictions")
    return scoreboard, pd.DataFrame(guard_rows), top_predictions, queue, summary


def build_readme(
    *,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    guards: pd.DataFrame,
    queue: pd.DataFrame,
) -> str:
    return f"""# 0092 Blocking-Slice Guarded Specialists

Generated: `{summary['generated_at_utc']}`

## Purpose

`0091` showed that the best 0090 near-misses failed only specific source, frame, or season slices. `0092` retests those near-misses with explicit no-correction guards on every previously failing slice. When a guard is active, the prediction falls back to the current 0088/0087 champion for that slice.

This is still leakage-safe and past-only: it reuses the same residual specialist predictions and only applies deterministic guards derived from the pre-2024 0091 failure analysis.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Queued candidates | `{summary['queued_candidate_count']}` |
| Candidate count | `{summary['candidate_count']}` |
| Hardened candidates | `{summary['hardened_candidate_count']}` |
| 0088 base MAE | `{summary['base_0088_mae']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Delta vs 0088 base | `{summary['best_delta_mae_vs_0088_base']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Input Design Queue

{markdown_table(queue, max_rows=20)}

## Guarded Slices

{markdown_table(guards, max_rows=30)}

## Scoreboard

{markdown_table(scoreboard, max_rows=20)}

## Interpretation

This screen answers whether the near-miss signals can be made deployable by refusing to correct their known bad slices. A hardened pass here would be a real candidate for a sensitivity check. No pass means the signal is still more useful diagnostically than operationally in this simple specialist form.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], scoreboard: pd.DataFrame, guards: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0092_blocking_slice_guarded_specialists.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Blocking-slice guards | `{summary['queued_candidate_count']}` near-misses retested | Pre-2024 only |
| Hardened candidates | `{summary['hardened_candidate_count']}` | Strict gate |
| Best candidate | `{summary['best_candidate']}` | `{summary['best_mae']}` MAE |
| Delta vs 0088 base | `{summary['best_delta_mae_vs_0088_base']}` | Guarded value |
| Leakage | `0` 2024+ rows | PASS |

Guarded slices:

{markdown_table(guards.head(12), max_rows=12)}

Scoreboard:

{markdown_table(scoreboard.head(8), max_rows=8)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0092 Blocking-Slice Guarded Specialists",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    scoreboard, guards, top_predictions, queue, summary = build_outputs()
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "guarded_slices.csv", guards)
    write_csv(artifacts / "input_design_queue.csv", queue)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "blocking_slice_guarded_specialists_manifest.json", summary)
    write_text(folder / "README.md", build_readme(summary=summary, scoreboard=scoreboard, guards=guards, queue=queue))
    update_milestones(summary, scoreboard, guards)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Retest 0091 near-miss HKG Tmax specialists with blocking-slice no-correction guards."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
