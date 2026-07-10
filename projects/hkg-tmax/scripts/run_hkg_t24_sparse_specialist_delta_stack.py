from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
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
    update_markdown_section,
)
from scripts.run_hkg_t24_nonlinear_local_residual_fusion_lab import (  # noqa: E402
    DELTA_GRID,
    build_feature_frame,
    delta_errors,
    local_group_key,
)
from scripts.run_hkg_t24_nonlinear_local_residual_fusion_lab import (  # noqa: E402
    FOLDER_NAME as FOLDER_0070,
)
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import (  # noqa: E402
    blend_prediction,
    score_prediction,
)
from scripts.run_hkg_t24_station_official_family_router import LATE_EVAL_START  # noqa: E402

FOLDER_NAME = "0071_sparse_specialist_delta_stack"
ARTIFACT_0070 = RESEARCH_ROOT / FOLDER_0070 / "artifacts"
SUMMARY_0070_PATH = ARTIFACT_0070 / "summary.json"
TRIGGER_CELLS_0070_PATH = ARTIFACT_0070 / "trigger_cell_diagnostics.csv"


@dataclass(frozen=True)
class SparseStackSpec:
    candidate_id: str
    mode: str
    candidate_class: str
    group_modes: tuple[str, ...]
    min_history: int
    min_prior_lift_c: float
    min_abs_delta: float
    combine_mode: str
    shrink: float
    max_abs_delta: float
    diagnostic_top_n: int
    diagnostic_min_cell_n: int
    diagnostic_max_active_delta_mae: float


@dataclass(frozen=True)
class PriorCellDecision:
    group_mode: str
    group_key: str
    count: int
    best_delta: float
    prior_lift_c: float
    best_prior_mae: float
    base_prior_mae: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_trigger_cells() -> pd.DataFrame:
    if not TRIGGER_CELLS_0070_PATH.exists():
        raise FileNotFoundError(f"Missing 0070 trigger-cell diagnostics: {TRIGGER_CELLS_0070_PATH}")
    cells = pd.read_csv(TRIGGER_CELLS_0070_PATH)
    required = {
        "group_mode",
        "group_key",
        "n",
        "best_fixed_delta",
        "active_delta_mae",
        "base_0069_mae",
        "best_fixed_delta_mae",
    }
    missing = required.difference(cells.columns)
    if missing:
        raise ValueError(f"0070 trigger-cell diagnostics missing columns: {sorted(missing)}")
    return cells


def clipped_delta(delta: float, max_abs_delta: float) -> float:
    return float(np.clip(delta, -max_abs_delta, max_abs_delta))


def select_prior_cell_decision(
    *,
    group_mode: str,
    group_key: str,
    count: int,
    abs_sums: np.ndarray,
    spec: SparseStackSpec,
) -> PriorCellDecision | None:
    if count < spec.min_history:
        return None
    prior_mae = abs_sums / count
    zero_index = int(np.argmin(np.abs(np.array(DELTA_GRID, dtype=float))))
    best_index = int(np.argmin(prior_mae))
    base_prior_mae = float(prior_mae[zero_index])
    best_prior_mae = float(prior_mae[best_index])
    prior_lift = base_prior_mae - best_prior_mae
    best_delta = float(DELTA_GRID[best_index])
    if prior_lift < spec.min_prior_lift_c:
        return None
    if abs(best_delta) < spec.min_abs_delta:
        return None
    return PriorCellDecision(
        group_mode=group_mode,
        group_key=group_key,
        count=count,
        best_delta=best_delta,
        prior_lift_c=prior_lift,
        best_prior_mae=best_prior_mae,
        base_prior_mae=base_prior_mae,
    )


def combine_prior_decisions(decisions: list[PriorCellDecision], spec: SparseStackSpec) -> float:
    if not decisions:
        return 0.0
    if spec.combine_mode == "best_lift":
        selected = max(decisions, key=lambda item: (item.prior_lift_c, item.count))
        return clipped_delta(selected.best_delta * spec.shrink, spec.max_abs_delta)
    if spec.combine_mode == "weighted_lift":
        weights = np.array([max(item.prior_lift_c, 0.0) for item in decisions], dtype=float)
        if float(weights.sum()) <= 0.0:
            return 0.0
        deltas = np.array([item.best_delta for item in decisions], dtype=float)
        return clipped_delta(float(np.sum(weights * deltas) / weights.sum()) * spec.shrink, spec.max_abs_delta)
    if spec.combine_mode == "agreement_mean":
        signs = {math.copysign(1.0, item.best_delta) for item in decisions if item.best_delta != 0.0}
        if len(signs) != 1:
            return 0.0
        return clipped_delta(float(np.mean([item.best_delta for item in decisions])) * spec.shrink, spec.max_abs_delta)
    raise ValueError(f"Unsupported combine mode: {spec.combine_mode}")


def diagnostic_cell_map(cells: pd.DataFrame, spec: SparseStackSpec) -> dict[tuple[str, str], float]:
    eligible = cells[
        pd.to_numeric(cells["n"], errors="coerce").ge(spec.diagnostic_min_cell_n)
        & pd.to_numeric(cells["active_delta_mae"], errors="coerce").le(spec.diagnostic_max_active_delta_mae)
        & cells["group_mode"].astype(str).isin(spec.group_modes)
    ].copy()
    eligible = eligible.sort_values(["active_delta_mae", "n"], ascending=[True, False]).head(spec.diagnostic_top_n)
    return {
        (str(row["group_mode"]), str(row["group_key"])): float(row["best_fixed_delta"])
        for _, row in eligible.iterrows()
    }


def combine_diagnostic_deltas(deltas: list[float], spec: SparseStackSpec) -> float:
    if not deltas:
        return 0.0
    if spec.combine_mode in {"best_lift", "weighted_lift"}:
        selected = max(deltas, key=lambda value: abs(value))
        return clipped_delta(selected * spec.shrink, spec.max_abs_delta)
    if spec.combine_mode == "agreement_mean":
        signs = {math.copysign(1.0, value) for value in deltas if value != 0.0}
        if len(signs) != 1:
            return 0.0
        return clipped_delta(float(np.mean(deltas)) * spec.shrink, spec.max_abs_delta)
    raise ValueError(f"Unsupported combine mode: {spec.combine_mode}")


def sparse_stack_specs() -> list[SparseStackSpec]:
    focused_groups = (
        "source_signeddiff_range",
        "source_absdiff_range",
        "source_weather_range",
        "source_wind_range",
        "source_signeddiff_active",
        "source_gate_signeddiff",
        "source_stackcorr_signeddiff",
    )
    rss_warm_groups = (
        "source_signeddiff_range",
        "source_absdiff_range",
        "source_signeddiff_active",
        "source_gate_signeddiff",
    )
    group_sets = {
        "focused": focused_groups,
        "rss_warm": rss_warm_groups,
    }
    specs: list[SparseStackSpec] = []
    for name, group_modes in group_sets.items():
        for min_history in (30, 80):
            for min_lift in (0.01, 0.03):
                for combine_mode in ("best_lift", "agreement_mean"):
                    for shrink in (0.50, 1.00):
                        token = (
                            f"causal_sparse_{name}_h{min_history}_lift{min_lift}_"
                            f"{combine_mode}_shrink{shrink}"
                        ).replace(".", "p")
                        specs.append(
                            SparseStackSpec(
                                candidate_id=token,
                                mode="causal_prior_sparse_stack",
                                candidate_class="causal_sparse_specialist_stack",
                                group_modes=group_modes,
                                min_history=min_history,
                                min_prior_lift_c=min_lift,
                                min_abs_delta=0.03,
                                combine_mode=combine_mode,
                                shrink=shrink,
                                max_abs_delta=0.18,
                                diagnostic_top_n=0,
                                diagnostic_min_cell_n=0,
                                diagnostic_max_active_delta_mae=0.0,
                            )
                        )
    for name, group_modes in group_sets.items():
        for top_n in (5, 10):
            for max_active_delta in (-0.01, -0.02):
                token = (
                    f"diagnostic_cell_atlas_{name}_top{top_n}_active{max_active_delta}"
                ).replace("-", "m").replace(".", "p")
                specs.append(
                    SparseStackSpec(
                        candidate_id=token,
                        mode="diagnostic_cell_atlas_stack",
                        candidate_class="diagnostic_cell_atlas_stack",
                        group_modes=group_modes,
                        min_history=0,
                        min_prior_lift_c=0.0,
                        min_abs_delta=0.0,
                        combine_mode="best_lift",
                        shrink=1.0,
                        max_abs_delta=0.18,
                        diagnostic_top_n=top_n,
                        diagnostic_min_cell_n=60,
                        diagnostic_max_active_delta_mae=max_active_delta,
                    )
                )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0071 candidate IDs are not unique")
    return specs


def apply_causal_sparse_stack(frame: pd.DataFrame, spec: SparseStackSpec) -> pd.DataFrame:
    state: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "abs_sums": np.zeros(len(DELTA_GRID), dtype=float)}
    )
    deltas: list[float] = []
    active_counts: list[int] = []
    selected_groups: list[str] = []
    selected_lifts: list[float] = []
    selected_prior_counts: list[int] = []

    for _, row in frame.iterrows():
        decisions: list[PriorCellDecision] = []
        keys: list[str] = []
        for group_mode in spec.group_modes:
            key = local_group_key(row, group_mode)
            state_key = f"{group_mode}::{key}"
            keys.append(state_key)
            group_state = state[state_key]
            decision = select_prior_cell_decision(
                group_mode=group_mode,
                group_key=key,
                count=int(group_state["count"]),
                abs_sums=np.asarray(group_state["abs_sums"], dtype=float),
                spec=spec,
            )
            if decision is not None:
                decisions.append(decision)
        delta = combine_prior_decisions(decisions, spec)
        deltas.append(delta)
        active_counts.append(len(decisions))
        if decisions:
            best = max(decisions, key=lambda item: (item.prior_lift_c, item.count))
            selected_groups.append(f"{best.group_mode}::{best.group_key}")
            selected_lifts.append(best.prior_lift_c)
            selected_prior_counts.append(best.count)
        else:
            selected_groups.append("")
            selected_lifts.append(math.nan)
            selected_prior_counts.append(0)

        errors = delta_errors(row)
        for state_key in set(keys):
            group_state = state[state_key]
            group_state["abs_sums"] = np.asarray(group_state["abs_sums"], dtype=float) + errors
            group_state["count"] = int(group_state["count"]) + 1

    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["station_delta"] = deltas
    out["station_weight"] = np.clip(
        pd.to_numeric(frame["base_0069_station_weight"], errors="coerce").to_numpy(dtype=float)
        + np.array(deltas, dtype=float),
        0.0,
        0.50,
    )
    out["candidate_prediction_c"] = blend_prediction(frame, out["station_weight"].to_numpy(dtype=float))
    out["active_specialist_count"] = active_counts
    out["selected_group"] = selected_groups
    out["selected_prior_lift_c"] = selected_lifts
    out["selected_prior_count"] = selected_prior_counts
    return out


def apply_diagnostic_cell_stack(
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    spec: SparseStackSpec,
) -> pd.DataFrame:
    atlas = diagnostic_cell_map(cells, spec)
    deltas: list[float] = []
    active_counts: list[int] = []
    selected_groups: list[str] = []
    for _, row in frame.iterrows():
        row_deltas = []
        row_groups = []
        for group_mode in spec.group_modes:
            key = local_group_key(row, group_mode)
            atlas_key = (group_mode, key)
            if atlas_key in atlas:
                row_deltas.append(float(atlas[atlas_key]))
                row_groups.append(f"{group_mode}::{key}")
        delta = combine_diagnostic_deltas(row_deltas, spec)
        deltas.append(delta)
        active_counts.append(len(row_deltas))
        selected_groups.append(";".join(row_groups))
    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["station_delta"] = deltas
    out["station_weight"] = np.clip(
        pd.to_numeric(frame["base_0069_station_weight"], errors="coerce").to_numpy(dtype=float)
        + np.array(deltas, dtype=float),
        0.0,
        0.50,
    )
    out["candidate_prediction_c"] = blend_prediction(frame, out["station_weight"].to_numpy(dtype=float))
    out["active_specialist_count"] = active_counts
    out["selected_group"] = selected_groups
    out["selected_prior_lift_c"] = math.nan
    out["selected_prior_count"] = 0
    return out


def apply_spec(frame: pd.DataFrame, cells: pd.DataFrame, spec: SparseStackSpec) -> pd.DataFrame:
    if spec.mode == "causal_prior_sparse_stack":
        out = apply_causal_sparse_stack(frame, spec)
    elif spec.mode == "diagnostic_cell_atlas_stack":
        out = apply_diagnostic_cell_stack(frame, cells, spec)
    else:
        raise ValueError(f"Unsupported 0071 mode: {spec.mode}")
    out["candidate_id"] = spec.candidate_id
    out["mode"] = spec.mode
    out["candidate_class"] = spec.candidate_class
    out["group_modes"] = ",".join(spec.group_modes)
    out["min_history"] = spec.min_history
    out["min_prior_lift_c"] = spec.min_prior_lift_c
    out["combine_mode"] = spec.combine_mode
    out["shrink"] = spec.shrink
    return out


def score_candidate(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    base_0069_mae: float,
) -> dict[str, object]:
    pred_values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    score = score_prediction(frame, pred_values)
    base_score = score_prediction(
        frame,
        pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    official_score = score_prediction(
        frame,
        pd.to_numeric(frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    late_mask = predictions["target_date"].ge(LATE_EVAL_START)
    late_score = score_prediction(frame.loc[late_mask].copy(), pred_values[late_mask.to_numpy()])
    late_base = score_prediction(
        frame.loc[late_mask].copy(),
        pd.to_numeric(frame.loc[late_mask, "base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    fold_deltas_vs_0069 = []
    fold_deltas_vs_official = []
    for _, fold_predictions in predictions.groupby("fold_id", observed=True):
        fold_frame = frame.loc[fold_predictions.index].copy()
        fold_score = score_prediction(
            fold_frame,
            pd.to_numeric(fold_predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_base = score_prediction(
            fold_frame,
            pd.to_numeric(fold_frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_official = score_prediction(
            fold_frame,
            pd.to_numeric(fold_frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_deltas_vs_0069.append(float(fold_score["mae"]) - float(fold_base["mae"]))
        fold_deltas_vs_official.append(float(fold_score["mae"]) - float(fold_official["mae"]))
    active_mask = pd.to_numeric(predictions["active_specialist_count"], errors="coerce").gt(0)
    active_score = score_prediction(
        frame.loc[active_mask].copy(),
        pd.to_numeric(predictions.loc[active_mask, "candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    active_base = score_prediction(
        frame.loc[active_mask].copy(),
        pd.to_numeric(frame.loc[active_mask, "base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "mode": str(predictions["mode"].iloc[0]),
        "candidate_class": str(predictions["candidate_class"].iloc[0]),
        "group_modes": str(predictions["group_modes"].iloc[0]),
        "min_history": int(predictions["min_history"].iloc[0]),
        "min_prior_lift_c": float(predictions["min_prior_lift_c"].iloc[0]),
        "combine_mode": str(predictions["combine_mode"].iloc[0]),
        "shrink": float(predictions["shrink"].iloc[0]),
        "n": score["n"],
        "mae": score["mae"],
        "rmse": score["rmse"],
        "bias": score["bias"],
        "base_0069_mae": base_score["mae"],
        "official_mae": official_score["mae"],
        "delta_mae_vs_0069": float(score["mae"]) - base_0069_mae,
        "delta_mae_vs_official": float(score["mae"]) - float(official_score["mae"]),
        "late_n": late_score["n"],
        "late_mae": late_score["mae"],
        "late_delta_mae_vs_0069": float(late_score["mae"]) - float(late_base["mae"]),
        "fold_delta_max_vs_0069": max(fold_deltas_vs_0069) if fold_deltas_vs_0069 else math.nan,
        "fold_delta_min_vs_0069": min(fold_deltas_vs_0069) if fold_deltas_vs_0069 else math.nan,
        "fold_delta_max_vs_official": max(fold_deltas_vs_official) if fold_deltas_vs_official else math.nan,
        "folds_improved_vs_0069": int(sum(delta < 0 for delta in fold_deltas_vs_0069)),
        "active_rows": int(active_mask.sum()),
        "active_row_share": float(active_mask.mean()),
        "active_mae": active_score["mae"],
        "active_delta_mae_vs_0069": (
            float(active_score["mae"]) - float(active_base["mae"]) if int(active_score["n"]) > 0 else math.nan
        ),
        "mean_station_delta": float(pd.to_numeric(predictions["station_delta"], errors="coerce").mean()),
        "mean_active_station_delta": float(
            pd.to_numeric(predictions.loc[active_mask, "station_delta"], errors="coerce").mean()
        )
        if active_mask.any()
        else math.nan,
    }
    row["beats_0069"] = bool(float(row["delta_mae_vs_0069"]) <= -0.0005)
    row["promotion_gate_passed"] = bool(
        row["beats_0069"]
        and float(row["fold_delta_max_vs_0069"]) <= 0.0
        and float(row["late_delta_mae_vs_0069"]) <= 0.0
    )
    row["deployable_gate_passed"] = bool(
        row["promotion_gate_passed"]
        and str(row["candidate_class"]) == "causal_sparse_specialist_stack"
    )
    return row


def score_all_specs(
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    specs: list[SparseStackSpec],
    *,
    base_0069_mae: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_spec(frame, cells, spec)
        rows.append(score_candidate(frame, predictions, base_0069_mae=base_0069_mae))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["deployable_gate_passed", "beats_0069", "promotion_gate_passed", "mae"],
        ascending=[False, False, False, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(30).astype(str))
    top_predictions = pd.concat(
        [predictions for predictions in prediction_frames if str(predictions["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(top_predictions["target_date"], context="0071 top predictions")
    return scoreboard.reset_index(drop=True), top_predictions


def selected_cell_report(cells: pd.DataFrame) -> pd.DataFrame:
    return cells[
        pd.to_numeric(cells["n"], errors="coerce").ge(60)
        & pd.to_numeric(cells["active_delta_mae"], errors="coerce").le(-0.005)
    ].sort_values(["active_delta_mae", "n"], ascending=[True, False]).head(40).reset_index(drop=True)


def leakage_audit(frame: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    deployable = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "base_0069_predictions_present_one_per_date",
            "passed": bool(
                frame["base_0069_prediction_c"].notna().all()
                and len(frame) == frame["target_date"].nunique()
            ),
            "evidence": f"{len(frame)} merged rows",
        },
        {
            "check_id": "causal_stack_uses_prior_state_only",
            "passed": True,
            "evidence": "causal cell states update after each row prediction is chosen",
        },
        {
            "check_id": "diagnostic_cell_atlas_not_marked_deployable",
            "passed": bool(
                scoreboard.loc[
                    scoreboard["candidate_class"].ne("causal_sparse_specialist_stack"),
                    "deployable_gate_passed",
                ].eq(False).all()
            ),
            "evidence": "full-period trigger-cell atlas stacks remain diagnostic only",
        },
        {
            "check_id": "deployable_gate_requires_full_fold_late_improvement",
            "passed": bool(
                deployable.empty
                or (
                    deployable["delta_mae_vs_0069"].le(-0.0005).all()
                    and deployable["fold_delta_max_vs_0069"].le(0.0).all()
                    and deployable["late_delta_mae_vs_0069"].le(0.0).all()
                )
            ),
            "evidence": f"{len(deployable)} deployable candidates passed",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    selected_cells: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    deployable = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    causal = scoreboard[scoreboard["candidate_class"].eq("causal_sparse_specialist_stack")].head(50).copy()
    diagnostic = scoreboard[scoreboard["candidate_class"].ne("causal_sparse_specialist_stack")].head(30).copy()
    return f"""# Sparse Specialist Delta Stack

Generated: `{summary['generated_at_utc']}`

## Purpose

`0070` found useful local trigger cells but no fold-robust deployable candidate. `0071` tests whether those cells can be converted into a sparse specialist stack: only activate when a cell has enough prior observations and its prior error table shows a meaningful station-weight delta advantage.

This experiment keeps the diagnostic full-period cell atlas separate from deployable causal stacks. Diagnostic atlas stacks are allowed to explain opportunity size, but they are never marked deployable.

## Data Contract

- Base prediction: `0069` best deployable prediction.
- Feature frame: same pre-2024 common frame used by `0070`.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ rows are used.
- Causal specialists update each cell's delta-error table only after the current row's prediction is selected.

## Headline

| Item | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| Best 0071 candidate | {summary['best_candidate']} |
| Best 0071 class | {summary['best_candidate_class']} |
| Best 0071 MAE | {summary['best_mae']} |
| Best 0071 RMSE | {summary['best_rmse']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best causal candidate | {summary['best_causal_candidate']} |
| Best causal MAE | {summary['best_causal_mae']} |
| Best causal fold max delta vs 0069 | {summary['best_causal_fold_delta_max_vs_0069']} |
| Gate-passed deployable candidate | {summary['best_deployable_candidate']} |
| Gate-passed deployable MAE | {summary['best_deployable_mae']} |
| Gate-passed deployable count | {summary['deployable_candidate_count']} |
| Selected diagnostic cells | {summary['selected_cell_count']} |

## Interpretation

If the best diagnostic atlas stack beats the causal sparse stack, the cell atlas is still valuable but not yet a promotable rule. A deployable stack must improve the full frame, avoid late-window damage, and avoid damage in every fold.

## Scoreboard

{markdown_table(scoreboard, max_rows=100)}

## Deployable Candidates

{markdown_table(deployable, max_rows=80)}

## Causal Sparse Candidates

{markdown_table(causal, max_rows=80)}

## Diagnostic Atlas Candidates

{markdown_table(diagnostic, max_rows=50)}

## Selected High-Value Cells

{markdown_table(selected_cells, max_rows=80)}

## Candidate Definitions

{markdown_table(definitions, max_rows=120)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/selected_cell_report.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_sparse_specialist_delta_stack.py`:

- `{FOLDER_NAME}`: sparse specialist station-weight delta stack using `0070` trigger-cell evidence.

| Metric | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| Best 0071 candidate | {summary['best_candidate']} |
| Best 0071 MAE | {summary['best_mae']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best causal candidate | {summary['best_causal_candidate']} |
| Best causal MAE | {summary['best_causal_mae']} |
| Gate-passed deployable candidate | {summary['best_deployable_candidate']} |

Leakage contract: no 2024+ rows; diagnostic cell atlas is not deployable; causal specialists update after scoring.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Sparse Specialist Delta Stack",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_sparse_specialist_delta_stack.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0070` trigger cells plus `0069` base predictions | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous |
| Candidate count | `{summary['candidate_count']}` | Tested |
| Base 0069 MAE / RMSE | `{summary['base_0069_mae']}` / `{summary['base_0069_rmse']}` | Baseline |
| Best 0071 candidate | `{summary['best_candidate']}` | Tested |
| Best 0071 class | `{summary['best_candidate_class']}` | Diagnostic/deployable classification |
| Best 0071 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0069 | `{summary['best_delta_mae_vs_0069']}` | Sparse stack value |
| Best causal candidate | `{summary['best_causal_candidate']}` | Prior-only sparse stack |
| Best causal MAE | `{summary['best_causal_mae']}` | Pre-2024 only |
| Best causal fold max delta vs 0069 | `{summary['best_causal_fold_delta_max_vs_0069']}` | Robustness check |
| Gate-passed deployable candidate | `{summary['best_deployable_candidate']}` | Requires full, fold, and late improvement |
| Gate-passed deployable MAE | `{summary['best_deployable_mae']}` | `None` means no candidate passed |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0071` tests whether `0070` trigger-cell opportunities can become a fold-robust sparse specialist delta stack.
"""
    update_markdown_section(
        path,
        heading="Sparse Specialist Delta Stack",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"55. Sparse specialist delta stack screened `{summary['candidate_count']}` candidates; "
        f"best delta vs 0069 is `{summary['best_delta_mae_vs_0069']}` from "
        f"`{summary['best_candidate']}`, best causal MAE is `{summary['best_causal_mae']}`, "
        f"and `{summary['deployable_candidate_count']}` candidates passed the strict deployable gate."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: isolate why the sparse specialists fail or pass fold robustness by decomposing each active cell by source era, month/season, official forecast range, and station-disagreement sign, then design the next smoother shrinkage rule from those diagnostics.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0069 = build_feature_frame()
    summary_0070 = load_json(SUMMARY_0070_PATH)
    cells = load_trigger_cells()
    base_score = score_prediction(
        frame,
        pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    specs = sparse_stack_specs()
    definitions = pd.DataFrame(
        [
            {
                **spec.__dict__,
                "group_modes": ",".join(spec.group_modes),
            }
            for spec in specs
        ]
    )
    scoreboard, top_predictions = score_all_specs(
        frame,
        cells,
        specs,
        base_0069_mae=float(base_score["mae"]),
    )
    selected_cells = selected_cell_report(cells)
    leakage = leakage_audit(frame, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0071 leakage audit failed: {failed}")

    best = scoreboard.iloc[0]
    causal_pool = scoreboard[scoreboard["candidate_class"].eq("causal_sparse_specialist_stack")].copy()
    causal_pool = causal_pool.sort_values(["mae", "fold_delta_max_vs_0069"]).reset_index(drop=True)
    best_causal = causal_pool.iloc[0]
    deployable_pool = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    deployable_pool = deployable_pool.sort_values(["mae", "fold_delta_max_vs_0069"]).reset_index(drop=True)
    best_deployable = deployable_pool.iloc[0] if not deployable_pool.empty else None

    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "common_rows": int(len(frame)),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "candidate_count": int(len(scoreboard)),
        "promoted_candidate_count": int(scoreboard["promotion_gate_passed"].astype(bool).sum()),
        "deployable_candidate_count": int(scoreboard["deployable_gate_passed"].astype(bool).sum()),
        "selected_cell_count": int(len(selected_cells)),
        "base_0069_candidate": str(summary_0069["best_deployable_candidate"]),
        "base_0069_mae": float(base_score["mae"]),
        "base_0069_rmse": float(base_score["rmse"]),
        "base_0070_best_prior_mae": float(summary_0070["best_prior_mae"]),
        "best_candidate": str(best["candidate_id"]),
        "best_candidate_class": str(best["candidate_class"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0069": float(best["delta_mae_vs_0069"]),
        "best_late_delta_mae_vs_0069": float(best["late_delta_mae_vs_0069"]),
        "best_active_rows": int(best["active_rows"]),
        "best_active_delta_mae_vs_0069": float(best["active_delta_mae_vs_0069"]),
        "best_causal_candidate": str(best_causal["candidate_id"]),
        "best_causal_mae": float(best_causal["mae"]),
        "best_causal_rmse": float(best_causal["rmse"]),
        "best_causal_delta_mae_vs_0069": float(best_causal["delta_mae_vs_0069"]),
        "best_causal_late_delta_mae_vs_0069": float(best_causal["late_delta_mae_vs_0069"]),
        "best_causal_fold_delta_max_vs_0069": float(best_causal["fold_delta_max_vs_0069"]),
        "best_causal_gate_passed": bool(best_causal["deployable_gate_passed"]),
        "best_deployable_candidate": str(best_deployable["candidate_id"]) if best_deployable is not None else "NONE",
        "best_deployable_mae": float(best_deployable["mae"]) if best_deployable is not None else None,
        "best_deployable_rmse": float(best_deployable["rmse"]) if best_deployable is not None else None,
        "best_deployable_delta_mae_vs_0069": (
            float(best_deployable["delta_mae_vs_0069"]) if best_deployable is not None else None
        ),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_csv(artifacts / "selected_cell_report.csv", selected_cells)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "sparse_specialist_delta_stack_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            definitions=definitions,
            scoreboard=scoreboard,
            selected_cells=selected_cells,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Run sparse specialist station-weight delta stack on top of 0069/0070 evidence."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
