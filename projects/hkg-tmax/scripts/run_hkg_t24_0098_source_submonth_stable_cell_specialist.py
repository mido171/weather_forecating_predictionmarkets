from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import ResidualState
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
from scripts.run_hkg_t24_0096_directional_cell_failure_audit import mam_submonth
from scripts.run_hkg_t24_0097_stable_directional_cell_specialist import (
    align_prediction,
    bucket_label,
    build_spec_from_scoreboard_row,
    load_cell_sets,
    score_with_0094_delta,
)
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

FOLDER_NAME = "0098_source_submonth_stable_cell_specialist"
INPUT_0095_SUMMARY_PATH = RESEARCH_ROOT / "0095_mam_error_direction_split_lab" / "artifacts" / "summary.json"
INPUT_0095_SCOREBOARD_PATH = RESEARCH_ROOT / "0095_mam_error_direction_split_lab" / "artifacts" / "scoreboard.csv"
INPUT_0095_TOP_PATH = RESEARCH_ROOT / "0095_mam_error_direction_split_lab" / "artifacts" / "top_predictions.csv"
INPUT_0096_SUMMARY_PATH = RESEARCH_ROOT / "0096_directional_cell_failure_audit" / "artifacts" / "summary.json"
INPUT_0096_BUCKET_SUMMARY_PATH = (
    RESEARCH_ROOT / "0096_directional_cell_failure_audit" / "artifacts" / "bucket_direction_summary.csv"
)
INPUT_0096_SOURCE_SUBMONTH_SUMMARY_PATH = (
    RESEARCH_ROOT / "0096_directional_cell_failure_audit" / "artifacts" / "source_submonth_direction_summary.csv"
)
CANDIDATE_CLASS = "0098_source_submonth_stable_cell_specialist"
CELL_POLICIES = ("bucket_only", "source_submonth_only", "bucket_or_source_submonth", "bucket_and_source_submonth")


@dataclass(frozen=True)
class SourceSubmonthCellSets:
    stable: set[tuple[str, str, str]]
    damaging: set[tuple[str, str, str]]


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def source_submonth_key(source: object, target_date: object, direction: str) -> tuple[str, str, str]:
    timestamp = pd.Timestamp(target_date)
    return str(source), mam_submonth(timestamp), direction


def load_source_submonth_cell_sets(summary: pd.DataFrame) -> SourceSubmonthCellSets:
    stable: set[tuple[str, str, str]] = set()
    damaging: set[tuple[str, str, str]] = set()
    for row in summary.itertuples(index=False):
        source = str(row.forecast_source_family)
        submonth = str(row.mam_submonth)
        direction = str(row.prior_direction)
        status = str(row.status)
        if direction == "inactive" or submonth == "non_mam":
            continue
        cell = (source, submonth, direction)
        if status == "stable_improving":
            stable.add(cell)
        elif status == "damaging":
            damaging.add(cell)
    return SourceSubmonthCellSets(stable=stable, damaging=damaging)


def cell_policy_allows(policy: str, *, bucket_allowed: bool, source_submonth_allowed: bool) -> bool:
    if policy == "bucket_only":
        return bucket_allowed
    if policy == "source_submonth_only":
        return source_submonth_allowed
    if policy == "bucket_or_source_submonth":
        return bucket_allowed or source_submonth_allowed
    if policy == "bucket_and_source_submonth":
        return bucket_allowed and source_submonth_allowed
    raise ValueError(f"Unsupported 0098 cell policy: {policy}")


def spec_for_policy(base_spec: DirectionSplitSpec, policy: str) -> DirectionSplitSpec:
    candidate_id = (
        f"srcsub_{short_hash(base_spec.pair_name, policy, base_spec.active_gate, base_spec.direction_mode)}_"
        f"{safe_token(policy)}_{safe_token(base_spec.pair_name, max_len=34)}"
    )
    return DirectionSplitSpec(
        candidate_id=candidate_id,
        pair_name=base_spec.pair_name,
        feature_a=base_spec.feature_a,
        feature_b=base_spec.feature_b,
        group_a=base_spec.group_a,
        group_b=base_spec.group_b,
        active_gate=base_spec.active_gate,
        direction_mode=base_spec.direction_mode,
        min_history=base_spec.min_history,
        direction_threshold_c=base_spec.direction_threshold_c,
        shrink_rows=base_spec.shrink_rows,
        correction_cap_c=base_spec.correction_cap_c,
    )


def load_inputs() -> tuple[
    pd.DataFrame,
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    DirectionSplitSpec,
    set[tuple[str, str]],
    set[tuple[str, str]],
    SourceSubmonthCellSets,
]:
    missing = [
        path
        for path in (
            INPUT_0095_SUMMARY_PATH,
            INPUT_0095_SCOREBOARD_PATH,
            INPUT_0095_TOP_PATH,
            INPUT_0096_SUMMARY_PATH,
            INPUT_0096_BUCKET_SUMMARY_PATH,
            INPUT_0096_SOURCE_SUBMONTH_SUMMARY_PATH,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"0098 requires 0095 and 0096 artifacts first: {missing}")

    features, base, pairs, previous_scoreboard, _summary_0094 = load_0095_base_inputs()
    selected_pairs = select_strong_pairs(pairs, previous_scoreboard)
    frame, _thresholds = build_working_frame(features, base, selected_pairs)
    summary_0095 = json.loads(INPUT_0095_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary_0096 = json.loads(INPUT_0096_SUMMARY_PATH.read_text(encoding="utf-8"))
    scoreboard_0095 = pd.read_csv(INPUT_0095_SCOREBOARD_PATH)
    top_0095 = pd.read_csv(INPUT_0095_TOP_PATH)
    top_0095["target_date"] = pd.to_datetime(top_0095["target_date"], errors="coerce").dt.normalize()
    top_0095 = top_0095[top_0095["target_date"].notna() & (top_0095["target_date"] < CONFIRMATION_START)].copy()
    bucket_summary = pd.read_csv(INPUT_0096_BUCKET_SUMMARY_PATH)
    source_submonth_summary = pd.read_csv(INPUT_0096_SOURCE_SUBMONTH_SUMMARY_PATH)
    stable_bucket_cells, damaging_bucket_cells = load_cell_sets(bucket_summary)
    source_submonth_cells = load_source_submonth_cell_sets(source_submonth_summary)

    best_id = str(summary_0095["best_candidate"])
    best_rows = scoreboard_0095[scoreboard_0095["candidate_id"].astype(str).eq(best_id)].copy()
    if best_rows.empty:
        raise RuntimeError(f"0098 could not find 0095 best candidate in scoreboard: {best_id}")
    best_spec = build_spec_from_scoreboard_row(best_rows.iloc[0])
    require_no_confirmation_dates(frame["target_date"], context="0098 working frame")
    require_no_confirmation_dates(top_0095["target_date"], context="0098 0095 top predictions")
    return (
        frame,
        summary_0095,
        summary_0096,
        bucket_summary,
        source_submonth_summary,
        top_0095,
        best_spec,
        stable_bucket_cells,
        damaging_bucket_cells,
        source_submonth_cells,
    )


def apply_source_submonth_specialist(
    frame: pd.DataFrame,
    spec: DirectionSplitSpec,
    *,
    cell_policy: str,
    stable_bucket_cells: set[tuple[str, str]],
    damaging_bucket_cells: set[tuple[str, str]],
    source_submonth_cells: SourceSubmonthCellSets,
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
    submonth_labels = np.full(len(frame), "inactive", dtype=object)
    bucket_allowed = np.zeros(len(frame), dtype=bool)
    source_submonth_allowed = np.zeros(len(frame), dtype=bool)
    policy_allowed = np.zeros(len(frame), dtype=bool)
    bucket_blocked = np.zeros(len(frame), dtype=bool)
    source_submonth_blocked = np.zeros(len(frame), dtype=bool)
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
            submonth = mam_submonth(pd.Timestamp(frame.at[row_idx, "target_date"]))
            submonth_labels[row_idx] = submonth
            key = (spec.pair_name, bucket)
            state = states.setdefault(key, ResidualState())
            prior_rows[row_idx] = state.count
            mean_residual = state.mean()
            prior_mean[row_idx] = mean_residual if state.count > 0 else math.nan
            direction = prior_direction(mean_residual, spec.direction_threshold_c) if state.count > 0 else "neutral"
            direction_codes[row_idx] = direction

            bucket_cell = (label, direction)
            source_cell = (str(frame.at[row_idx, "forecast_source_family"]), submonth, direction)
            bucket_allowed[row_idx] = bucket_cell in stable_bucket_cells
            source_submonth_allowed[row_idx] = source_cell in source_submonth_cells.stable
            policy_allowed[row_idx] = cell_policy_allows(
                cell_policy,
                bucket_allowed=bool(bucket_allowed[row_idx]),
                source_submonth_allowed=bool(source_submonth_allowed[row_idx]),
            )
            bucket_blocked[row_idx] = bucket_cell in damaging_bucket_cells
            source_submonth_blocked[row_idx] = source_cell in source_submonth_cells.damaging
            if (
                state.count >= spec.min_history
                and mode_allows_direction(spec.direction_mode, direction)
                and policy_allowed[row_idx]
                and not bucket_blocked[row_idx]
                and not source_submonth_blocked[row_idx]
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
    diagnostics["cell_policy"] = cell_policy
    diagnostics["pair_name"] = spec.pair_name
    diagnostics["active_gate"] = spec.active_gate
    diagnostics["direction_mode"] = spec.direction_mode
    diagnostics["pair_bucket"] = pair_bucket
    diagnostics["pair_bucket_label"] = pair_labels
    diagnostics["mam_submonth"] = submonth_labels
    diagnostics["gate_active_row"] = gate_active
    diagnostics["prior_rows"] = prior_rows
    diagnostics["prior_mean_residual_c"] = prior_mean
    diagnostics["prior_direction"] = direction_codes
    diagnostics["bucket_stable_allowed"] = bucket_allowed
    diagnostics["source_submonth_stable_allowed"] = source_submonth_allowed
    diagnostics["cell_policy_allowed"] = policy_allowed
    diagnostics["bucket_damaging_blocked"] = bucket_blocked
    diagnostics["source_submonth_damaging_blocked"] = source_submonth_blocked
    diagnostics["specialist_active"] = active
    diagnostics["specialist_correction_c"] = corrections
    return predictions, diagnostics


def build_outputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    (
        frame,
        summary_0095,
        summary_0096,
        bucket_summary,
        source_submonth_summary,
        top_0095,
        base_spec,
        stable_bucket_cells,
        damaging_bucket_cells,
        source_submonth_cells,
    ) = load_inputs()
    mask_map = evaluation_masks_0095(frame)
    raw_prediction = frame["forecast_max_c"].to_numpy(dtype=float)
    base_prediction = frame["candidate_prediction_c"].to_numpy(dtype=float)
    input_0095_prediction = align_prediction(frame, top_0095, "candidate_prediction_c")
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
    ]
    definitions: list[dict[str, object]] = []
    predictions_by_id: dict[str, np.ndarray] = {str(summary_0095["best_candidate"]): input_0095_prediction}
    diagnostics_frames: list[pd.DataFrame] = []
    for policy in CELL_POLICIES:
        spec = spec_for_policy(base_spec, policy)
        prediction, diagnostics = apply_source_submonth_specialist(
            frame,
            spec,
            cell_policy=policy,
            stable_bucket_cells=stable_bucket_cells,
            damaging_bucket_cells=damaging_bucket_cells,
            source_submonth_cells=source_submonth_cells,
            include_diagnostics=True,
        )
        predictions_by_id[spec.candidate_id] = prediction
        diagnostics_frames.append(diagnostics)
        definitions.append(
            {
                **spec_definition(spec),
                "candidate_id": spec.candidate_id,
                "candidate_class": CANDIDATE_CLASS,
                "cell_policy": policy,
                "input_0095_candidate": summary_0095["best_candidate"],
                "stable_bucket_cell_count": len(stable_bucket_cells),
                "damaging_bucket_cell_count": len(damaging_bucket_cells),
                "stable_source_submonth_cell_count": len(source_submonth_cells.stable),
                "damaging_source_submonth_cell_count": len(source_submonth_cells.damaging),
            }
        )
        rows.append(
            score_with_0094_delta(
                frame,
                candidate_id=spec.candidate_id,
                candidate_class=CANDIDATE_CLASS,
                prediction=prediction,
                mask_map=mask_map,
                extra={
                    **spec_definition(spec),
                    "cell_policy": policy,
                    "specialist_active_rows": int(diagnostics["specialist_active"].sum()),
                    "bucket_stable_allowed_rows": int(diagnostics["bucket_stable_allowed"].sum()),
                    "source_submonth_stable_allowed_rows": int(diagnostics["source_submonth_stable_allowed"].sum()),
                    "cell_policy_allowed_rows": int(diagnostics["cell_policy_allowed"].sum()),
                    "bucket_damaging_blocked_rows": int(diagnostics["bucket_damaging_blocked"].sum()),
                    "source_submonth_damaging_blocked_rows": int(
                        diagnostics["source_submonth_damaging_blocked"].sum()
                    ),
                },
            )
        )

    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    candidate_definitions = pd.DataFrame(definitions)
    diagnostics_all = pd.concat(diagnostics_frames, ignore_index=True)
    candidates = scoreboard[scoreboard["candidate_class"].eq(CANDIDATE_CLASS)].copy()
    best_0098 = candidates.sort_values(["mae", "rmse"]).iloc[0]
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
        best_prediction = predictions_by_id.get(str(best_row["candidate_id"]), input_0095_prediction)

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
    best_diagnostics = diagnostics_all[diagnostics_all["candidate_id"].astype(str).eq(str(best_0098["candidate_id"]))].copy()
    stable_bucket_frame = bucket_summary[bucket_summary["status"].eq("stable_improving")].copy()
    guard_bucket_frame = bucket_summary[bucket_summary["status"].eq("damaging")].copy()
    stable_source_frame = source_submonth_summary[source_submonth_summary["status"].eq("stable_improving")].copy()
    guard_source_frame = source_submonth_summary[source_submonth_summary["status"].eq("damaging")].copy()
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
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
        "stable_bucket_cell_count_used": int(len(stable_bucket_cells)),
        "damaging_bucket_cell_count_used": int(len(damaging_bucket_cells)),
        "stable_source_submonth_cell_count_used": int(len(source_submonth_cells.stable)),
        "damaging_source_submonth_cell_count_used": int(len(source_submonth_cells.damaging)),
        "candidate_policy_count": int(len(CELL_POLICIES)),
        "best_0098_candidate": str(best_0098["candidate_id"]),
        "best_0098_policy": str(best_0098["cell_policy"]),
        "best_0098_mae": float(best_0098["mae"]),
        "best_0098_rmse": float(best_0098["rmse"]),
        "best_0098_delta_mae_vs_0094_base": float(best_0098["delta_mae_vs_0094_base"]),
        "best_0098_delta_mae_vs_0095_best": float(best_0098["mae"]) - float(summary_0095["best_mae"]),
        "best_0098_specialist_active_rows": int(best_0098["specialist_active_rows"]),
        "best_candidate": str(best_row["candidate_id"]),
        "best_candidate_class": str(best_row["candidate_class"]),
        "best_mae": float(best_row["mae"]),
        "best_rmse": float(best_row["rmse"]),
        "best_delta_mae_vs_0094_base": float(best_row["delta_mae_vs_0094_base"]),
        "new_0098_champion": bool(str(best_row["candidate_class"]) == CANDIDATE_CLASS),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "source_submonth_stable_cell_specialist_complete",
        "next_recommended_task": (
            "Run 0099_mam_cell_policy_sensitivity: stress-test the best 0098 source/submonth policy across "
            "adjacent min-history, direction-threshold, and cap settings, while keeping 2024+ rows sealed."
        ),
    }
    require_no_confirmation_dates(top_predictions["target_date"], context="0098 top predictions")
    return (
        scoreboard,
        candidate_definitions,
        stable_bucket_frame,
        guard_bucket_frame,
        stable_source_frame,
        guard_source_frame,
        diagnostics_all,
        best_diagnostics,
        top_predictions,
        summary,
    )


def build_readme(
    *,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    candidate_definitions: pd.DataFrame,
    stable_bucket_frame: pd.DataFrame,
    stable_source_frame: pd.DataFrame,
    guard_source_frame: pd.DataFrame,
    best_diagnostics: pd.DataFrame,
) -> str:
    return f"""# 0098 Source/Submonth Stable Cell Specialist

Generated: `{summary['generated_at_utc']}`

## Purpose

`0097` proved that the strict bucket-only stable-cell guard is safer but gives back a tiny part of the `0095` MAE gain. `0098` tests whether source family and MAM sub-month are the missing stability dimensions. It keeps the same `0095` prior residual state, but compares four allowed-cell policies:

- `bucket_only`: reproduce the `0097` guard.
- `source_submonth_only`: allow only source/submonth/direction cells marked stable by `0096`.
- `bucket_or_source_submonth`: allow a row if either the bucket-direction cell or source/submonth/direction cell was stable.
- `bucket_and_source_submonth`: require both dimensions to be stable.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Input 0095 MAE | `{summary['input_0095_best_mae']}` |
| Input 0095 RMSE | `{summary['input_0095_best_rmse']}` |
| Stable bucket cells used | `{summary['stable_bucket_cell_count_used']}` |
| Stable source/submonth cells used | `{summary['stable_source_submonth_cell_count_used']}` |
| Best 0098 policy | `{summary['best_0098_policy']}` |
| Best 0098 candidate | `{summary['best_0098_candidate']}` |
| Best 0098 MAE | `{summary['best_0098_mae']}` |
| Best 0098 RMSE | `{summary['best_0098_rmse']}` |
| Best 0098 delta vs 0094 | `{summary['best_0098_delta_mae_vs_0094_base']}` |
| Best 0098 delta vs 0095 | `{summary['best_0098_delta_mae_vs_0095_best']}` |
| Best 0098 active rows | `{summary['best_0098_specialist_active_rows']}` |
| Overall best candidate | `{summary['best_candidate']}` |
| Overall best class | `{summary['best_candidate_class']}` |
| New 0098 champion | `{summary['new_0098_champion']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Plain-English Finding

This is a stability test, not a broad new model. The question is whether the direction split should trust a source/submonth context, a bucket context, or both. A source/submonth context says, for example, "RSS March overforecast rows have historically benefited from this correction."

If the best 0098 policy does not beat `0095`, the interpretation is that source/submonth stability explains part of the 0095 gain but does not fully replace the broader 0095 rule. If it beats `0097`, then source/submonth context is useful for avoiding over-narrow bucket guards.

## Candidate Definitions

{markdown_table(candidate_definitions, max_rows=20)}

## Scoreboard

{markdown_table(scoreboard, max_rows=20)}

## Stable Bucket Cells

{markdown_table(stable_bucket_frame, max_rows=20)}

## Stable Source/Submonth Cells

{markdown_table(stable_source_frame, max_rows=20)}

## Damaging Source/Submonth Cells

{markdown_table(guard_source_frame, max_rows=20)}

## Best 0098 Diagnostics Sample

{markdown_table(best_diagnostics[best_diagnostics['gate_active_row']].head(50), max_rows=50)}

## Leakage Controls

All rows are before `{summary['confirmation_start']}`. The current row's target error is never used to choose a correction. Pair residual states are updated after the current target date is scored. Stable and damaging source/submonth cells are imported from the documented `0096` pre-2024 audit; this experiment opens no 2024+ confirmation rows.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    stable_source_frame: pd.DataFrame,
    guard_source_frame: pd.DataFrame,
) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0098_source_submonth_stable_cell_specialist.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Candidate policies | `{summary['candidate_policy_count']}` | Bounded screen |
| Stable source/submonth cells | `{summary['stable_source_submonth_cell_count_used']}` | From 0096 |
| Damaging source/submonth cells | `{summary['damaging_source_submonth_cell_count_used']}` | Explicit guard |
| Best 0098 policy | `{summary['best_0098_policy']}` | Candidate |
| Best 0098 MAE | `{summary['best_0098_mae']}` | Candidate |
| Best 0098 RMSE | `{summary['best_0098_rmse']}` | Candidate |
| Delta vs 0095 | `{summary['best_0098_delta_mae_vs_0095_best']}` | Promotion value |
| Overall best after 0098 | `{summary['best_candidate']}` | `{summary['best_candidate_class']}` |
| Leakage | `0` 2024+ rows | PASS |

Scoreboard:

{markdown_table(scoreboard, max_rows=10)}

Stable source/submonth cells:

{markdown_table(stable_source_frame, max_rows=10)}

Damaging source/submonth cells:

{markdown_table(guard_source_frame, max_rows=10)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0098 Source/Submonth Stable Cell Specialist",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    (
        scoreboard,
        candidate_definitions,
        stable_bucket_frame,
        guard_bucket_frame,
        stable_source_frame,
        guard_source_frame,
        diagnostics_all,
        best_diagnostics,
        top_predictions,
        summary,
    ) = build_outputs()
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "candidate_definitions.csv", candidate_definitions)
    write_csv(artifacts / "bucket_stable_cells.csv", stable_bucket_frame)
    write_csv(artifacts / "bucket_guard_cells.csv", guard_bucket_frame)
    write_csv(artifacts / "srcsub_stable_cells.csv", stable_source_frame)
    write_csv(artifacts / "srcsub_guard_cells.csv", guard_source_frame)
    write_csv(artifacts / "candidate_diagnostics.csv", diagnostics_all)
    write_csv(artifacts / "best_gate_diagnostics.csv", best_diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "source_submonth_stable_cell_specialist_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            candidate_definitions=candidate_definitions,
            stable_bucket_frame=stable_bucket_frame,
            stable_source_frame=stable_source_frame,
            guard_source_frame=guard_source_frame,
            best_diagnostics=best_diagnostics,
        ),
    )
    update_milestones(summary, scoreboard, stable_source_frame, guard_source_frame)
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
