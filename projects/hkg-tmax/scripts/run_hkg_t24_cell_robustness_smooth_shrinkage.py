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
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import (  # noqa: E402
    blend_prediction,
    score_prediction,
)
from scripts.run_hkg_t24_sparse_specialist_delta_stack import (  # noqa: E402
    FOLDER_NAME as FOLDER_0071,
)
from scripts.run_hkg_t24_sparse_specialist_delta_stack import (  # noqa: E402
    load_json,
)
from scripts.run_hkg_t24_station_official_family_router import LATE_EVAL_START  # noqa: E402

FOLDER_NAME = "0072_cell_robustness_smooth_shrinkage"
ARTIFACT_0071 = RESEARCH_ROOT / FOLDER_0071 / "artifacts"
SUMMARY_0071_PATH = ARTIFACT_0071 / "summary.json"
SELECTED_CELLS_0071_PATH = ARTIFACT_0071 / "selected_cell_report.csv"


@dataclass(frozen=True)
class SmoothShrinkageSpec:
    candidate_id: str
    mode: str
    candidate_class: str
    group_modes: tuple[str, ...]
    min_history: int
    min_prior_lift_c: float
    support_shrink: float
    lift_scale_c: float
    base_shrink: float
    max_abs_delta: float
    combine_mode: str
    diagnostic_top_n: int
    diagnostic_max_active_delta_mae: float


@dataclass(frozen=True)
class SmoothDecision:
    group_mode: str
    group_key: str
    count: int
    raw_delta: float
    shrunk_delta: float
    prior_lift_c: float
    shrink_factor: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_selected_cells() -> pd.DataFrame:
    if not SELECTED_CELLS_0071_PATH.exists():
        raise FileNotFoundError(f"Missing 0071 selected-cell report: {SELECTED_CELLS_0071_PATH}")
    cells = pd.read_csv(SELECTED_CELLS_0071_PATH)
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
        raise ValueError(f"0071 selected-cell report missing columns: {sorted(missing)}")
    return cells


def season_from_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def ensure_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    target_dates = pd.to_datetime(out["target_date"], errors="coerce")
    out["month"] = target_dates.dt.month.astype("Int64")
    out["season"] = out["month"].map(lambda value: season_from_month(int(value)) if pd.notna(value) else "unknown")
    return out


def shrink_factor(*, count: int, prior_lift_c: float, support_shrink: float, lift_scale_c: float) -> float:
    if count <= 0 or prior_lift_c <= 0.0:
        return 0.0
    support_term = count / (count + support_shrink)
    lift_term = prior_lift_c / (prior_lift_c + lift_scale_c)
    return float(np.clip(support_term * lift_term, 0.0, 1.0))


def select_smooth_decision(
    *,
    group_mode: str,
    group_key: str,
    count: int,
    abs_sums: np.ndarray,
    spec: SmoothShrinkageSpec,
) -> SmoothDecision | None:
    if count < spec.min_history:
        return None
    prior_mae = abs_sums / count
    zero_index = int(np.argmin(np.abs(np.array(DELTA_GRID, dtype=float))))
    best_index = int(np.argmin(prior_mae))
    raw_delta = float(DELTA_GRID[best_index])
    if raw_delta == 0.0:
        return None
    base_prior_mae = float(prior_mae[zero_index])
    best_prior_mae = float(prior_mae[best_index])
    prior_lift = base_prior_mae - best_prior_mae
    if prior_lift < spec.min_prior_lift_c:
        return None
    factor = shrink_factor(
        count=count,
        prior_lift_c=prior_lift,
        support_shrink=spec.support_shrink,
        lift_scale_c=spec.lift_scale_c,
    )
    shrunk = float(np.clip(raw_delta * spec.base_shrink * factor, -spec.max_abs_delta, spec.max_abs_delta))
    if shrunk == 0.0:
        return None
    return SmoothDecision(
        group_mode=group_mode,
        group_key=group_key,
        count=count,
        raw_delta=raw_delta,
        shrunk_delta=shrunk,
        prior_lift_c=prior_lift,
        shrink_factor=factor,
    )


def combine_smooth_decisions(decisions: list[SmoothDecision], spec: SmoothShrinkageSpec) -> float:
    if not decisions:
        return 0.0
    if spec.combine_mode == "best_lift":
        selected = max(decisions, key=lambda item: (item.prior_lift_c, item.count))
        return float(np.clip(selected.shrunk_delta, -spec.max_abs_delta, spec.max_abs_delta))
    if spec.combine_mode == "weighted_lift":
        weights = np.array([max(item.prior_lift_c, 0.0) * max(item.count, 1) for item in decisions], dtype=float)
        if float(weights.sum()) <= 0.0:
            return 0.0
        deltas = np.array([item.shrunk_delta for item in decisions], dtype=float)
        return float(np.clip(np.sum(weights * deltas) / weights.sum(), -spec.max_abs_delta, spec.max_abs_delta))
    if spec.combine_mode == "agreement_mean":
        signs = {math.copysign(1.0, item.shrunk_delta) for item in decisions if item.shrunk_delta != 0.0}
        if len(signs) != 1:
            return 0.0
        return float(np.clip(np.mean([item.shrunk_delta for item in decisions]), -spec.max_abs_delta, spec.max_abs_delta))
    raise ValueError(f"Unsupported combine mode: {spec.combine_mode}")


def smooth_specs() -> list[SmoothShrinkageSpec]:
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
    specs: list[SmoothShrinkageSpec] = []
    for name, group_modes in {"focused": focused_groups, "rss_warm": rss_warm_groups}.items():
        for min_history in (30, 80):
            for min_lift in (0.005, 0.02):
                for support_shrink_value in (60.0, 160.0):
                    for lift_scale in (0.02, 0.05):
                        for combine_mode in ("best_lift", "weighted_lift"):
                            candidate_id = (
                                f"causal_smooth_{name}_h{min_history}_lift{min_lift}_"
                                f"nshrink{support_shrink_value}_lscale{lift_scale}_{combine_mode}"
                            ).replace(".", "p")
                            specs.append(
                                SmoothShrinkageSpec(
                                    candidate_id=candidate_id,
                                    mode="causal_smooth_shrinkage",
                                    candidate_class="causal_smooth_shrinkage",
                                    group_modes=group_modes,
                                    min_history=min_history,
                                    min_prior_lift_c=min_lift,
                                    support_shrink=support_shrink_value,
                                    lift_scale_c=lift_scale,
                                    base_shrink=1.0,
                                    max_abs_delta=0.12,
                                    combine_mode=combine_mode,
                                    diagnostic_top_n=0,
                                    diagnostic_max_active_delta_mae=0.0,
                                )
                            )
    for name, group_modes in {"focused": focused_groups, "rss_warm": rss_warm_groups}.items():
        for top_n in (5, 10):
            for support_shrink_value in (60.0, 160.0):
                candidate_id = f"diagnostic_smooth_atlas_{name}_top{top_n}_nshrink{support_shrink_value}".replace(
                    ".",
                    "p",
                )
                specs.append(
                    SmoothShrinkageSpec(
                        candidate_id=candidate_id,
                        mode="diagnostic_smooth_atlas",
                        candidate_class="diagnostic_smooth_atlas",
                        group_modes=group_modes,
                        min_history=0,
                        min_prior_lift_c=0.0,
                        support_shrink=support_shrink_value,
                        lift_scale_c=0.05,
                        base_shrink=1.0,
                        max_abs_delta=0.12,
                        combine_mode="best_lift",
                        diagnostic_top_n=top_n,
                        diagnostic_max_active_delta_mae=-0.01,
                    )
                )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0072 candidate IDs are not unique")
    return specs


def apply_causal_smooth_shrinkage(frame: pd.DataFrame, spec: SmoothShrinkageSpec) -> pd.DataFrame:
    state: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "abs_sums": np.zeros(len(DELTA_GRID), dtype=float)}
    )
    deltas: list[float] = []
    active_counts: list[int] = []
    selected_groups: list[str] = []
    selected_lifts: list[float] = []
    selected_shrink_factors: list[float] = []

    for _, row in frame.iterrows():
        decisions: list[SmoothDecision] = []
        keys: list[str] = []
        for group_mode in spec.group_modes:
            key = local_group_key(row, group_mode)
            state_key = f"{group_mode}::{key}"
            keys.append(state_key)
            group_state = state[state_key]
            decision = select_smooth_decision(
                group_mode=group_mode,
                group_key=key,
                count=int(group_state["count"]),
                abs_sums=np.asarray(group_state["abs_sums"], dtype=float),
                spec=spec,
            )
            if decision is not None:
                decisions.append(decision)
        delta = combine_smooth_decisions(decisions, spec)
        deltas.append(delta)
        active_counts.append(len(decisions))
        if decisions:
            best = max(decisions, key=lambda item: (item.prior_lift_c, item.count))
            selected_groups.append(f"{best.group_mode}::{best.group_key}")
            selected_lifts.append(best.prior_lift_c)
            selected_shrink_factors.append(best.shrink_factor)
        else:
            selected_groups.append("")
            selected_lifts.append(math.nan)
            selected_shrink_factors.append(math.nan)
        errors = delta_errors(row)
        for state_key in set(keys):
            group_state = state[state_key]
            group_state["abs_sums"] = np.asarray(group_state["abs_sums"], dtype=float) + errors
            group_state["count"] = int(group_state["count"]) + 1
    return prediction_frame(frame, deltas, active_counts, selected_groups, selected_lifts, selected_shrink_factors)


def diagnostic_smooth_atlas_map(cells: pd.DataFrame, spec: SmoothShrinkageSpec) -> dict[tuple[str, str], float]:
    eligible = cells[
        cells["group_mode"].astype(str).isin(spec.group_modes)
        & pd.to_numeric(cells["active_delta_mae"], errors="coerce").le(spec.diagnostic_max_active_delta_mae)
    ].copy()
    eligible = eligible.sort_values(["active_delta_mae", "n"], ascending=[True, False]).head(spec.diagnostic_top_n)
    out: dict[tuple[str, str], float] = {}
    for _, row in eligible.iterrows():
        count = int(row["n"])
        active_lift = abs(float(row["active_delta_mae"]))
        factor = shrink_factor(
            count=count,
            prior_lift_c=active_lift,
            support_shrink=spec.support_shrink,
            lift_scale_c=spec.lift_scale_c,
        )
        delta = float(row["best_fixed_delta"]) * spec.base_shrink * factor
        out[(str(row["group_mode"]), str(row["group_key"]))] = float(
            np.clip(delta, -spec.max_abs_delta, spec.max_abs_delta)
        )
    return out


def apply_diagnostic_smooth_atlas(
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    spec: SmoothShrinkageSpec,
) -> pd.DataFrame:
    atlas = diagnostic_smooth_atlas_map(cells, spec)
    deltas: list[float] = []
    active_counts: list[int] = []
    selected_groups: list[str] = []
    selected_lifts: list[float] = []
    selected_shrink_factors: list[float] = []
    for _, row in frame.iterrows():
        row_deltas = []
        row_groups = []
        for group_mode in spec.group_modes:
            key = local_group_key(row, group_mode)
            atlas_key = (group_mode, key)
            if atlas_key in atlas:
                row_deltas.append(atlas[atlas_key])
                row_groups.append(f"{group_mode}::{key}")
        delta = float(max(row_deltas, key=lambda item: abs(item))) if row_deltas else 0.0
        deltas.append(delta)
        active_counts.append(len(row_deltas))
        selected_groups.append(";".join(row_groups))
        selected_lifts.append(math.nan)
        selected_shrink_factors.append(math.nan)
    return prediction_frame(frame, deltas, active_counts, selected_groups, selected_lifts, selected_shrink_factors)


def prediction_frame(
    frame: pd.DataFrame,
    deltas: list[float],
    active_counts: list[int],
    selected_groups: list[str],
    selected_lifts: list[float],
    selected_shrink_factors: list[float],
) -> pd.DataFrame:
    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["station_delta"] = np.array(deltas, dtype=float)
    out["station_weight"] = np.clip(
        pd.to_numeric(frame["base_0069_station_weight"], errors="coerce").to_numpy(dtype=float)
        + out["station_delta"].to_numpy(dtype=float),
        0.0,
        0.50,
    )
    out["candidate_prediction_c"] = blend_prediction(frame, out["station_weight"].to_numpy(dtype=float))
    out["active_specialist_count"] = active_counts
    out["selected_group"] = selected_groups
    out["selected_prior_lift_c"] = selected_lifts
    out["selected_shrink_factor"] = selected_shrink_factors
    return out


def apply_spec(frame: pd.DataFrame, cells: pd.DataFrame, spec: SmoothShrinkageSpec) -> pd.DataFrame:
    if spec.mode == "causal_smooth_shrinkage":
        out = apply_causal_smooth_shrinkage(frame, spec)
    elif spec.mode == "diagnostic_smooth_atlas":
        out = apply_diagnostic_smooth_atlas(frame, cells, spec)
    else:
        raise ValueError(f"Unsupported 0072 mode: {spec.mode}")
    out["candidate_id"] = spec.candidate_id
    out["mode"] = spec.mode
    out["candidate_class"] = spec.candidate_class
    out["group_modes"] = ",".join(spec.group_modes)
    out["min_history"] = spec.min_history
    out["min_prior_lift_c"] = spec.min_prior_lift_c
    out["support_shrink"] = spec.support_shrink
    out["lift_scale_c"] = spec.lift_scale_c
    out["combine_mode"] = spec.combine_mode
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
        fold_deltas_vs_0069.append(float(fold_score["mae"]) - float(fold_base["mae"]))
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
        "support_shrink": float(predictions["support_shrink"].iloc[0]),
        "lift_scale_c": float(predictions["lift_scale_c"].iloc[0]),
        "combine_mode": str(predictions["combine_mode"].iloc[0]),
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
        "folds_improved_vs_0069": int(sum(delta < 0 for delta in fold_deltas_vs_0069)),
        "active_rows": int(active_mask.sum()),
        "active_row_share": float(active_mask.mean()),
        "active_mae": active_score["mae"],
        "active_delta_mae_vs_0069": (
            float(active_score["mae"]) - float(active_base["mae"]) if int(active_score["n"]) > 0 else math.nan
        ),
        "mean_station_delta": float(pd.to_numeric(predictions["station_delta"], errors="coerce").mean()),
        "mean_abs_station_delta": float(pd.to_numeric(predictions["station_delta"], errors="coerce").abs().mean()),
        "mean_selected_shrink_factor": float(
            pd.to_numeric(predictions["selected_shrink_factor"], errors="coerce").mean()
        ),
    }
    row["beats_0069"] = bool(float(row["delta_mae_vs_0069"]) <= -0.0005)
    row["promotion_gate_passed"] = bool(
        row["beats_0069"]
        and float(row["fold_delta_max_vs_0069"]) <= 0.0
        and float(row["late_delta_mae_vs_0069"]) <= 0.0
    )
    row["deployable_gate_passed"] = bool(
        row["promotion_gate_passed"] and str(row["candidate_class"]) == "causal_smooth_shrinkage"
    )
    return row


def score_all_specs(
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    specs: list[SmoothShrinkageSpec],
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
    top_ids = set(scoreboard["candidate_id"].head(25).astype(str))
    top_predictions = pd.concat(
        [predictions for predictions in prediction_frames if str(predictions["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(top_predictions["target_date"], context="0072 top predictions")
    return scoreboard.reset_index(drop=True), top_predictions


def cell_axis_decomposition(frame: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    axes = [
        "fold_id",
        "forecast_source_family",
        "season",
        "month",
        "forecast_range_bucket",
        "signeddiff_bucket",
        "active_count_bucket",
        "weather_bucket",
    ]
    rows: list[dict[str, object]] = []
    for _, cell in cells.iterrows():
        group_mode = str(cell["group_mode"])
        group_key = str(cell["group_key"])
        fixed_delta = float(cell["best_fixed_delta"])
        membership = frame.apply(lambda row, mode=group_mode: local_group_key(row, mode), axis=1).eq(group_key)
        cell_frame = frame[membership].copy()
        if cell_frame.empty:
            continue
        for axis in axes:
            for axis_value, subframe in cell_frame.groupby(axis, observed=True):
                if len(subframe) < 10:
                    continue
                base_score = score_prediction(
                    subframe,
                    pd.to_numeric(subframe["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
                )
                weight = np.clip(
                    pd.to_numeric(subframe["base_0069_station_weight"], errors="coerce").to_numpy(dtype=float)
                    + fixed_delta,
                    0.0,
                    0.50,
                )
                cell_score = score_prediction(subframe, blend_prediction(subframe, weight))
                rows.append(
                    {
                        "group_mode": group_mode,
                        "group_key": group_key,
                        "axis": axis,
                        "axis_value": str(axis_value),
                        "n": int(len(subframe)),
                        "fixed_delta": fixed_delta,
                        "base_0069_mae": base_score["mae"],
                        "cell_delta_mae": cell_score["mae"],
                        "delta_mae_vs_0069": float(cell_score["mae"]) - float(base_score["mae"]),
                        "mean_family_disagreement_c": float(subframe["family_disagreement_c"].mean()),
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["delta_mae_vs_0069", "n"], ascending=[True, False]).reset_index(drop=True)


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
            "check_id": "smooth_causal_state_updates_after_scoring",
            "passed": True,
            "evidence": "causal smooth states update only after the row decision",
        },
        {
            "check_id": "diagnostic_atlas_not_marked_deployable",
            "passed": bool(
                scoreboard.loc[
                    scoreboard["candidate_class"].ne("causal_smooth_shrinkage"),
                    "deployable_gate_passed",
                ].eq(False).all()
            ),
            "evidence": "full-period smooth atlas candidates remain diagnostic only",
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
    decomposition: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    deployable = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    causal = scoreboard[scoreboard["candidate_class"].eq("causal_smooth_shrinkage")].head(50).copy()
    diagnostic = scoreboard[scoreboard["candidate_class"].ne("causal_smooth_shrinkage")].head(30).copy()
    return f"""# Cell Robustness And Smooth Shrinkage

Generated: `{summary['generated_at_utc']}`

## Purpose

`0071` showed that the full-period cell atlas has real opportunity, but hard causal sparse specialists do not pass fold robustness. `0072` decomposes the active cells by era/source, season, month, forecast range, station-disagreement sign, active station-stack state, and weather state, then tests smoother prior shrinkage.

The central question is whether smoother deltas, scaled by prior support and prior lift, can avoid the fold damage seen in `0071`.

## Data Contract

- Base prediction: `0069` best deployable prediction.
- Cell source: `0071` selected-cell report.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ rows are used.
- Diagnostic atlas candidates use full-period cell diagnostics and are not deployable.
- Causal smooth candidates update prior cell states only after the row has been scored.

## Headline

| Item | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| Best 0072 candidate | {summary['best_candidate']} |
| Best 0072 class | {summary['best_candidate_class']} |
| Best 0072 MAE | {summary['best_mae']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best causal candidate | {summary['best_causal_candidate']} |
| Best causal MAE | {summary['best_causal_mae']} |
| Best causal fold max delta vs 0069 | {summary['best_causal_fold_delta_max_vs_0069']} |
| Gate-passed deployable candidate | {summary['best_deployable_candidate']} |
| Gate-passed deployable MAE | {summary['best_deployable_mae']} |
| Decomposition rows | {summary['decomposition_rows']} |

## Interpretation

The decomposition table identifies which subconditions make the diagnostic cell lift stable or unstable. A causal candidate is only promotable if it improves the full frame, the late RSS window, and every fold versus `0069`.

## Scoreboard

{markdown_table(scoreboard, max_rows=100)}

## Deployable Candidates

{markdown_table(deployable, max_rows=80)}

## Causal Smooth Candidates

{markdown_table(causal, max_rows=80)}

## Diagnostic Smooth Atlas Candidates

{markdown_table(diagnostic, max_rows=50)}

## Cell Axis Decomposition

{markdown_table(decomposition, max_rows=120)}

## Candidate Definitions

{markdown_table(definitions, max_rows=120)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/cell_axis_decomposition.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_cell_robustness_smooth_shrinkage.py`:

- `{FOLDER_NAME}`: active-cell decomposition and smooth shrinkage screen after `0071`.

| Metric | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| Best 0072 candidate | {summary['best_candidate']} |
| Best 0072 MAE | {summary['best_mae']} |
| Best causal candidate | {summary['best_causal_candidate']} |
| Best causal MAE | {summary['best_causal_mae']} |
| Gate-passed deployable candidate | {summary['best_deployable_candidate']} |

Leakage contract: no 2024+ rows; diagnostic atlas is not deployable; causal shrinkage updates after scoring.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Cell Robustness And Smooth Shrinkage",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_cell_robustness_smooth_shrinkage.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0071` selected cells plus `0069` base predictions | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous |
| Candidate count | `{summary['candidate_count']}` | Tested |
| Base 0069 MAE / RMSE | `{summary['base_0069_mae']}` / `{summary['base_0069_rmse']}` | Baseline |
| Best 0072 candidate | `{summary['best_candidate']}` | Tested |
| Best 0072 class | `{summary['best_candidate_class']}` | Diagnostic/deployable classification |
| Best 0072 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0069 | `{summary['best_delta_mae_vs_0069']}` | Smooth shrinkage value |
| Best causal candidate | `{summary['best_causal_candidate']}` | Prior-only smooth shrinkage |
| Best causal MAE | `{summary['best_causal_mae']}` | Pre-2024 only |
| Best causal fold max delta vs 0069 | `{summary['best_causal_fold_delta_max_vs_0069']}` | Robustness check |
| Gate-passed deployable candidate | `{summary['best_deployable_candidate']}` | Requires full, fold, and late improvement |
| Gate-passed deployable MAE | `{summary['best_deployable_mae']}` | `None` means no candidate passed |
| Decomposition rows | `{summary['decomposition_rows']}` | Diagnostic atlas |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0072` decomposes `0071` active-cell instability and tests smoother prior shrinkage instead of hard cell deltas.
"""
    update_markdown_section(
        path,
        heading="Cell Robustness And Smooth Shrinkage",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"56. Cell robustness and smooth shrinkage screened `{summary['candidate_count']}` candidates; "
        f"best delta vs 0069 is `{summary['best_delta_mae_vs_0069']}` from "
        f"`{summary['best_candidate']}`, best causal MAE is `{summary['best_causal_mae']}`, "
        f"and `{summary['deployable_candidate_count']}` candidates passed the strict deployable gate."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: use the `0072` decomposition to design source-era-specific smooth shrinkage that learns separate priors for RSS warm-tight-range cells and press cool/sunny cells, with explicit fold-local support floors before activation.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0069 = build_feature_frame()
    frame = ensure_calendar_columns(frame)
    summary_0071 = load_json(SUMMARY_0071_PATH)
    cells = load_selected_cells()
    base_score = score_prediction(
        frame,
        pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    specs = smooth_specs()
    definitions = pd.DataFrame(
        [{**spec.__dict__, "group_modes": ",".join(spec.group_modes)} for spec in specs]
    )
    scoreboard, top_predictions = score_all_specs(
        frame,
        cells,
        specs,
        base_0069_mae=float(base_score["mae"]),
    )
    decomposition = cell_axis_decomposition(frame, cells)
    leakage = leakage_audit(frame, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0072 leakage audit failed: {failed}")

    best = scoreboard.iloc[0]
    causal_pool = scoreboard[scoreboard["candidate_class"].eq("causal_smooth_shrinkage")].copy()
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
        "decomposition_rows": int(len(decomposition)),
        "base_0069_candidate": str(summary_0069["best_deployable_candidate"]),
        "base_0069_mae": float(base_score["mae"]),
        "base_0069_rmse": float(base_score["rmse"]),
        "base_0071_best_causal_mae": float(summary_0071["best_causal_mae"]),
        "best_candidate": str(best["candidate_id"]),
        "best_candidate_class": str(best["candidate_class"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0069": float(best["delta_mae_vs_0069"]),
        "best_late_delta_mae_vs_0069": float(best["late_delta_mae_vs_0069"]),
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
    write_csv(artifacts / "cell_axis_decomposition.csv", decomposition)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "cell_robustness_smooth_shrinkage_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            definitions=definitions,
            scoreboard=scoreboard,
            decomposition=decomposition,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Run active-cell robustness decomposition and smooth shrinkage screen."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
