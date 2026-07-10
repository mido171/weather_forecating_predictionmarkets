from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0090_guarded_specialists_from_error_autopsy import score_candidate
from scripts.run_hkg_t24_0094_expanded_high_error_interaction_lab import active_mask_for_gate
from scripts.run_hkg_t24_0095_mam_error_direction_split_lab import (
    BASE_ID,
    DirectionSplitSpec,
    build_working_frame,
    evaluation_masks_0095,
    load_inputs as load_0095_base_inputs,
    mode_allows_direction,
    prior_direction,
    safe_token,
    select_strong_pairs,
    short_hash,
    spec_definition,
)
from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import ResidualState
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

FOLDER_NAME = "0097_stable_directional_cell_specialist"
INPUT_0095_SUMMARY_PATH = RESEARCH_ROOT / "0095_mam_error_direction_split_lab" / "artifacts" / "summary.json"
INPUT_0095_SCOREBOARD_PATH = RESEARCH_ROOT / "0095_mam_error_direction_split_lab" / "artifacts" / "scoreboard.csv"
INPUT_0095_TOP_PATH = RESEARCH_ROOT / "0095_mam_error_direction_split_lab" / "artifacts" / "top_predictions.csv"
INPUT_0096_SUMMARY_PATH = RESEARCH_ROOT / "0096_directional_cell_failure_audit" / "artifacts" / "summary.json"
INPUT_0096_BUCKET_SUMMARY_PATH = (
    RESEARCH_ROOT / "0096_directional_cell_failure_audit" / "artifacts" / "bucket_direction_summary.csv"
)
CANDIDATE_CLASS = "0097_stable_directional_cell_specialist"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def bucket_label(value: float) -> str:
    if not math.isfinite(value):
        return "missing"
    return f"bucket_{int(value)}"


def parse_bucket_label(label: object) -> str | None:
    text = str(label)
    if not text.startswith("bucket_"):
        return None
    suffix = text.removeprefix("bucket_")
    if not suffix.isdigit():
        return None
    return f"bucket_{int(suffix)}"


def load_cell_sets(bucket_summary: pd.DataFrame) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    stable_cells: set[tuple[str, str]] = set()
    damaging_cells: set[tuple[str, str]] = set()
    for row in bucket_summary.itertuples(index=False):
        label = parse_bucket_label(row.pair_bucket_label)
        direction = str(row.prior_direction)
        status = str(row.status)
        if label is None or direction == "inactive":
            continue
        cell = (label, direction)
        if status == "stable_improving":
            stable_cells.add(cell)
        elif status == "damaging":
            damaging_cells.add(cell)
    return stable_cells, damaging_cells


def build_spec_from_scoreboard_row(row: pd.Series) -> DirectionSplitSpec:
    base_pair = str(row["pair_name"])
    candidate_id = (
        f"stablecell_{short_hash(base_pair, str(row['active_gate']), str(row['direction_mode']), str(row['min_history']), str(row['direction_threshold_c']))}_"
        f"{safe_token(base_pair, max_len=48)}_{row['active_gate']}_{row['direction_mode']}"
    )
    return DirectionSplitSpec(
        candidate_id=candidate_id,
        pair_name=base_pair,
        feature_a=str(row["feature_a"]),
        feature_b=str(row["feature_b"]),
        group_a=str(row["group_a"]),
        group_b=str(row["group_b"]),
        active_gate=str(row["active_gate"]),
        direction_mode=str(row["direction_mode"]),
        min_history=int(float(row["min_history"])),
        direction_threshold_c=float(row["direction_threshold_c"]),
        shrink_rows=float(row["shrink_rows"]),
        correction_cap_c=float(row["correction_cap_c"]),
    )


def align_prediction(frame: pd.DataFrame, predictions: pd.DataFrame, prediction_column: str) -> np.ndarray:
    key = ["target_date", "forecast_source_family"]
    scoped = predictions[[*key, prediction_column]].copy()
    scoped["target_date"] = pd.to_datetime(scoped["target_date"], errors="coerce").dt.normalize()
    aligned = frame[key].merge(scoped, on=key, how="left")
    values = pd.to_numeric(aligned[prediction_column], errors="coerce")
    if values.isna().any():
        missing = int(values.isna().sum())
        raise RuntimeError(f"0097 could not align {missing} prediction rows from {prediction_column}")
    return values.to_numpy(dtype=float)


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    DirectionSplitSpec,
    set[tuple[str, str]],
    set[tuple[str, str]],
]:
    missing = [
        path
        for path in (
            INPUT_0095_SUMMARY_PATH,
            INPUT_0095_SCOREBOARD_PATH,
            INPUT_0095_TOP_PATH,
            INPUT_0096_SUMMARY_PATH,
            INPUT_0096_BUCKET_SUMMARY_PATH,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"0097 requires 0095 and 0096 artifacts first: {missing}")

    features, base, pairs, previous_scoreboard, _summary_0094 = load_0095_base_inputs()
    selected_pairs = select_strong_pairs(pairs, previous_scoreboard)
    frame, thresholds = build_working_frame(features, base, selected_pairs)
    summary_0095 = json.loads(INPUT_0095_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary_0096 = json.loads(INPUT_0096_SUMMARY_PATH.read_text(encoding="utf-8"))
    scoreboard_0095 = pd.read_csv(INPUT_0095_SCOREBOARD_PATH)
    top_0095 = pd.read_csv(INPUT_0095_TOP_PATH)
    top_0095["target_date"] = pd.to_datetime(top_0095["target_date"], errors="coerce").dt.normalize()
    top_0095 = top_0095[top_0095["target_date"].notna() & (top_0095["target_date"] < CONFIRMATION_START)].copy()
    bucket_summary = pd.read_csv(INPUT_0096_BUCKET_SUMMARY_PATH)
    stable_cells, damaging_cells = load_cell_sets(bucket_summary)

    best_id = str(summary_0095["best_candidate"])
    best_rows = scoreboard_0095[scoreboard_0095["candidate_id"].astype(str).eq(best_id)].copy()
    if best_rows.empty:
        raise RuntimeError(f"0097 could not find 0095 best candidate in scoreboard: {best_id}")
    best_spec = build_spec_from_scoreboard_row(best_rows.iloc[0])
    require_no_confirmation_dates(frame["target_date"], context="0097 working frame")
    require_no_confirmation_dates(top_0095["target_date"], context="0097 0095 top predictions")
    return frame, thresholds, summary_0095, summary_0096, bucket_summary, top_0095, best_spec, stable_cells, damaging_cells


def apply_stable_cell_specialist(
    frame: pd.DataFrame,
    spec: DirectionSplitSpec,
    *,
    stable_cells: set[tuple[str, str]],
    damaging_cells: set[tuple[str, str]],
    include_diagnostics: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    residual = frame["base_residual_c"].to_numpy(dtype=float)
    pair_bucket = frame[f"{spec.pair_name}__bucket"].to_numpy(dtype=float)
    gate_active = active_mask_for_gate(frame, spec.active_gate)
    predictions = base.copy()
    active = np.zeros(len(frame), dtype=bool)
    prior_rows = np.zeros(len(frame), dtype=int)
    prior_mean = np.full(len(frame), np.nan, dtype=float)
    corrections = np.zeros(len(frame), dtype=float)
    direction_codes = np.full(len(frame), "inactive", dtype=object)
    pair_labels = np.full(len(frame), "missing", dtype=object)
    stable_allowed = np.zeros(len(frame), dtype=bool)
    damaging_blocked = np.zeros(len(frame), dtype=bool)
    states: dict[tuple[str, int], ResidualState] = {}

    for _, date_group in frame.groupby("target_date", sort=True, observed=True):
        pending_updates: list[tuple[tuple[str, int], float]] = []
        for idx in date_group.index:
            row_idx = int(idx)
            if not gate_active[row_idx] or not math.isfinite(pair_bucket[row_idx]):
                continue
            bucket = int(pair_bucket[row_idx])
            label = bucket_label(pair_bucket[row_idx])
            pair_labels[row_idx] = label
            key = (spec.pair_name, bucket)
            state = states.setdefault(key, ResidualState())
            prior_rows[row_idx] = state.count
            mean_residual = state.mean()
            prior_mean[row_idx] = mean_residual if state.count > 0 else math.nan
            direction = prior_direction(mean_residual, spec.direction_threshold_c) if state.count > 0 else "neutral"
            direction_codes[row_idx] = direction
            cell = (label, direction)
            stable_allowed[row_idx] = cell in stable_cells
            damaging_blocked[row_idx] = cell in damaging_cells
            if (
                state.count >= spec.min_history
                and mode_allows_direction(spec.direction_mode, direction)
                and stable_allowed[row_idx]
                and not damaging_blocked[row_idx]
            ):
                shrink = state.count / (state.count + spec.shrink_rows)
                correction = float(np.clip(mean_residual * shrink, -spec.correction_cap_c, spec.correction_cap_c))
                predictions[row_idx] = base[row_idx] - correction
                corrections[row_idx] = correction
                active[row_idx] = abs(correction) > 1e-12
            pending_updates.append((key, residual[row_idx]))
        for key, residual_value in pending_updates:
            states[key].update(residual_value)

    if not include_diagnostics:
        return predictions, pd.DataFrame()
    diagnostics = frame[["target_date", "forecast_source_family", "season", "frame_segment", "era_bucket"]].copy()
    diagnostics["candidate_id"] = spec.candidate_id
    diagnostics["pair_name"] = spec.pair_name
    diagnostics["active_gate"] = spec.active_gate
    diagnostics["direction_mode"] = spec.direction_mode
    diagnostics["pair_bucket"] = pair_bucket
    diagnostics["pair_bucket_label"] = pair_labels
    diagnostics["gate_active_row"] = gate_active
    diagnostics["prior_rows"] = prior_rows
    diagnostics["prior_mean_residual_c"] = prior_mean
    diagnostics["prior_direction"] = direction_codes
    diagnostics["stable_cell_allowed"] = stable_allowed
    diagnostics["damaging_cell_blocked"] = damaging_blocked
    diagnostics["specialist_active"] = active
    diagnostics["specialist_correction_c"] = corrections
    return predictions, diagnostics


def score_with_0094_delta(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    candidate_class: str,
    prediction: np.ndarray,
    mask_map: dict[str, np.ndarray],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = score_candidate(
        frame,
        candidate_id=candidate_id,
        candidate_class=candidate_class,
        prediction=prediction,
        mask_map=mask_map,
        extra=extra,
    )
    row["delta_mae_vs_0094_base"] = float(row["delta_mae_vs_0088_base"])
    return row


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    (
        frame,
        thresholds,
        summary_0095,
        summary_0096,
        bucket_summary,
        top_0095,
        spec,
        stable_cells,
        damaging_cells,
    ) = load_inputs()
    mask_map = evaluation_masks_0095(frame)
    raw_prediction = frame["forecast_max_c"].to_numpy(dtype=float)
    base_prediction = frame["candidate_prediction_c"].to_numpy(dtype=float)
    input_0095_prediction = align_prediction(frame, top_0095, "candidate_prediction_c")
    stable_prediction, diagnostics = apply_stable_cell_specialist(
        frame,
        spec,
        stable_cells=stable_cells,
        damaging_cells=damaging_cells,
        include_diagnostics=True,
    )
    stable_cells_frame = bucket_summary[bucket_summary["status"].eq("stable_improving")].copy()
    guard_cells_frame = bucket_summary[bucket_summary["status"].eq("damaging")].copy()
    definition = pd.DataFrame(
        [
            {
                **spec_definition(spec),
                "candidate_id": spec.candidate_id,
                "candidate_class": CANDIDATE_CLASS,
                "input_0095_candidate": summary_0095["best_candidate"],
                "stable_cell_count": len(stable_cells),
                "damaging_guard_cell_count": len(damaging_cells),
                "cell_policy": "allow_only_0096_stable_improving_bucket_direction_cells",
            }
        ]
    )
    rows = [
        score_with_0094_delta(
            frame,
            candidate_id="official_raw",
            candidate_class="official_raw",
            prediction=raw_prediction,
            mask_map=mask_map,
        ),
        score_with_0094_delta(
            frame,
            candidate_id=BASE_ID,
            candidate_class="0094_base",
            prediction=base_prediction,
            mask_map=mask_map,
        ),
        score_with_0094_delta(
            frame,
            candidate_id=str(summary_0095["best_candidate"]),
            candidate_class="0095_mam_error_direction_split",
            prediction=input_0095_prediction,
            mask_map=mask_map,
            extra={"source_experiment": "0095"},
        ),
        score_with_0094_delta(
            frame,
            candidate_id=spec.candidate_id,
            candidate_class=CANDIDATE_CLASS,
            prediction=stable_prediction,
            mask_map=mask_map,
            extra={
                **spec_definition(spec),
                "stable_cell_count": len(stable_cells),
                "damaging_guard_cell_count": len(damaging_cells),
                "specialist_active_rows": int(diagnostics["specialist_active"].sum()),
                "stable_allowed_rows": int(diagnostics["stable_cell_allowed"].sum()),
                "damaging_blocked_rows": int(diagnostics["damaging_cell_blocked"].sum()),
            },
        ),
    ]
    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    eligible = scoreboard[
        scoreboard["candidate_class"].isin(["0095_mam_error_direction_split", CANDIDATE_CLASS])
        & scoreboard["hardened_gate_passed"].astype(bool)
        & (pd.to_numeric(scoreboard["delta_mae_vs_0094_base"], errors="coerce") < 0.0)
    ].copy()
    if eligible.empty:
        best_row = scoreboard[scoreboard["candidate_id"].eq(BASE_ID)].iloc[0]
        best_prediction = base_prediction
    else:
        best_row = eligible.sort_values(["mae", "rmse"]).iloc[0]
        if str(best_row["candidate_class"]) == CANDIDATE_CLASS:
            best_prediction = stable_prediction
        else:
            best_prediction = input_0095_prediction

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
    top_predictions["candidate_id"] = str(best_row["candidate_id"])
    top_predictions["candidate_prediction_c"] = best_prediction
    top_predictions["candidate_error_c"] = top_predictions["candidate_prediction_c"] - top_predictions["target_tmax_c"]
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    stable_row = scoreboard[scoreboard["candidate_class"].eq(CANDIDATE_CLASS)].iloc[0]
    summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "rows": int(len(frame)),
        "first_target_date": dates.min().date().isoformat(),
        "last_target_date": dates.max().date().isoformat(),
        "input_0095_best_candidate": summary_0095["best_candidate"],
        "input_0095_best_mae": float(summary_0095["best_mae"]),
        "input_0095_best_rmse": float(summary_0095["best_rmse"]),
        "input_0096_stable_improving_bucket_direction_cells": int(summary_0096["stable_improving_bucket_direction_cells"]),
        "input_0096_damaging_bucket_direction_cells": int(summary_0096["damaging_bucket_direction_cells"]),
        "stable_cell_count_used": int(len(stable_cells)),
        "damaging_guard_cell_count_used": int(len(damaging_cells)),
        "stable_specialist_candidate": spec.candidate_id,
        "stable_specialist_mae": float(stable_row["mae"]),
        "stable_specialist_rmse": float(stable_row["rmse"]),
        "stable_specialist_delta_mae_vs_0094_base": float(stable_row["delta_mae_vs_0094_base"]),
        "stable_specialist_delta_mae_vs_0095_best": float(stable_row["mae"]) - float(summary_0095["best_mae"]),
        "stable_specialist_active_rows": int(diagnostics["specialist_active"].sum()),
        "stable_allowed_rows": int(diagnostics["stable_cell_allowed"].sum()),
        "damaging_blocked_rows": int(diagnostics["damaging_cell_blocked"].sum()),
        "best_candidate": str(best_row["candidate_id"]),
        "best_candidate_class": str(best_row["candidate_class"]),
        "best_mae": float(best_row["mae"]),
        "best_rmse": float(best_row["rmse"]),
        "best_delta_mae_vs_0094_base": float(best_row["delta_mae_vs_0094_base"]),
        "stable_specialist_is_new_champion": bool(str(best_row["candidate_class"]) == CANDIDATE_CLASS),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "stable_directional_cell_specialist_complete",
        "next_recommended_task": (
            "Run 0098_source_submonth_stable_cell_specialist: test whether the 0096 source/submonth stable "
            "cells can recover the small robust MAM gain lost by bucket-only guarding, still using no 2024+ rows "
            "and only the currently available forecast archive."
        ),
    }
    require_no_confirmation_dates(top_predictions["target_date"], context="0097 top predictions")
    return scoreboard, definition, stable_cells_frame, guard_cells_frame, diagnostics, top_predictions, summary


def build_readme(
    *,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    definition: pd.DataFrame,
    stable_cells_frame: pd.DataFrame,
    guard_cells_frame: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    return f"""# 0097 Stable Directional Cell Specialist

Generated: `{summary['generated_at_utc']}`

## Purpose

`0096` audited the `0095` MAM direction split and found that only a narrow bucket-direction cell was clearly stable. `0097` turns that audit into a bounded specialist: it keeps the exact `0095` prior-error machinery, but the correction can fire only when the row is inside a `0096` stable-improving bucket-direction cell. Any `0096` damaging cell is explicitly blocked.

This is intentionally conservative. The goal is not to widen the model. The goal is to test whether a smaller rule is more robust while preserving as much of the `0095` gain as possible.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Input 0095 best | `{summary['input_0095_best_candidate']}` |
| Input 0095 MAE | `{summary['input_0095_best_mae']}` |
| Input 0095 RMSE | `{summary['input_0095_best_rmse']}` |
| Stable cells used | `{summary['stable_cell_count_used']}` |
| Damaging guard cells used | `{summary['damaging_guard_cell_count_used']}` |
| Stable specialist candidate | `{summary['stable_specialist_candidate']}` |
| Stable specialist MAE | `{summary['stable_specialist_mae']}` |
| Stable specialist RMSE | `{summary['stable_specialist_rmse']}` |
| Stable specialist delta vs 0094 | `{summary['stable_specialist_delta_mae_vs_0094_base']}` |
| Stable specialist delta vs 0095 | `{summary['stable_specialist_delta_mae_vs_0095_best']}` |
| Stable specialist active rows | `{summary['stable_specialist_active_rows']}` |
| Best candidate after 0097 | `{summary['best_candidate']}` |
| Best candidate class | `{summary['best_candidate_class']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| 0097 is new champion | `{summary['stable_specialist_is_new_champion']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Plain-English Finding

The original `0095` rule corrected any MAM row whose matching pair bucket had enough prior overforecast history. `0097` narrows that: it only corrects rows where the pair bucket and prior direction were already proven stable by `0096`.

If `0097` is slightly worse than `0095`, that is not automatically bad. It means the broad `0095` rule earned a tiny amount of extra MAE from cells that were positive but not stable enough under the `0096` materiality screen. The tradeoff is simple: `0095` is better on raw MAE, while `0097` is more guarded.

## Candidate Definition

{markdown_table(definition, max_rows=20)}

## Stable Cells Used

{markdown_table(stable_cells_frame, max_rows=30)}

## Damaging Cells Guarded

{markdown_table(guard_cells_frame, max_rows=30)}

## Scoreboard

{markdown_table(scoreboard, max_rows=20)}

## Specialist Diagnostics Sample

{markdown_table(diagnostics[diagnostics['gate_active_row']].head(40), max_rows=40)}

## Leakage Controls

All rows are before `{summary['confirmation_start']}`. The current row's target error is never used to choose a correction. Pair residual states are updated only after all rows for the current target date are scored. The allowed/blocked cells are imported from the already documented `0096` audit and no 2024+ confirmation rows are opened.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    stable_cells_frame: pd.DataFrame,
    guard_cells_frame: pd.DataFrame,
) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0097_stable_directional_cell_specialist.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Stable 0096 cells used | `{summary['stable_cell_count_used']}` | Bounded candidate |
| Damaging cells guarded | `{summary['damaging_guard_cell_count_used']}` | Explicit guard |
| Stable specialist active rows | `{summary['stable_specialist_active_rows']}` | Pre-2024 only |
| Stable specialist MAE | `{summary['stable_specialist_mae']}` | Candidate |
| Delta vs 0094 | `{summary['stable_specialist_delta_mae_vs_0094_base']}` | Candidate value |
| Delta vs 0095 | `{summary['stable_specialist_delta_mae_vs_0095_best']}` | Robustness tradeoff |
| Best after 0097 | `{summary['best_candidate']}` | `{summary['best_candidate_class']}` |
| Leakage | `0` 2024+ rows | PASS |

Scoreboard:

{markdown_table(scoreboard, max_rows=10)}

Stable cells:

{markdown_table(stable_cells_frame, max_rows=10)}

Guarded damaging cells:

{markdown_table(guard_cells_frame, max_rows=10)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0097 Stable Directional Cell Specialist",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    scoreboard, definition, stable_cells_frame, guard_cells_frame, diagnostics, top_predictions, summary = build_outputs()
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "candidate_definition.csv", definition)
    write_csv(artifacts / "selected_stable_cells.csv", stable_cells_frame)
    write_csv(artifacts / "selected_guard_cells.csv", guard_cells_frame)
    write_csv(artifacts / "stable_specialist_diagnostics.csv", diagnostics)
    write_csv(artifacts / "best_gate_diagnostics.csv", diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "stable_directional_cell_specialist_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            definition=definition,
            stable_cells_frame=stable_cells_frame,
            guard_cells_frame=guard_cells_frame,
            diagnostics=diagnostics,
        ),
    )
    update_milestones(summary, scoreboard, stable_cells_frame, guard_cells_frame)
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
