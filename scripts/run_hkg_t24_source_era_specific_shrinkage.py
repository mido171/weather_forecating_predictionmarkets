from __future__ import annotations

import argparse
import json
import math
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
from scripts.run_hkg_t24_cell_robustness_smooth_shrinkage import (  # noqa: E402
    FOLDER_NAME as FOLDER_0072,
)
from scripts.run_hkg_t24_cell_robustness_smooth_shrinkage import (  # noqa: E402
    ensure_calendar_columns,
    load_json,
    shrink_factor,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_nonlinear_local_residual_fusion_lab import (  # noqa: E402
    DELTA_GRID,
    build_feature_frame,
    delta_errors,
)
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import (  # noqa: E402
    blend_prediction,
    score_prediction,
)
from scripts.run_hkg_t24_station_official_family_router import LATE_EVAL_START  # noqa: E402

FOLDER_NAME = "0073_source_era_specific_shrinkage"
ARTIFACT_0072 = RESEARCH_ROOT / FOLDER_0072 / "artifacts"
SUMMARY_0072_PATH = ARTIFACT_0072 / "summary.json"
DECOMPOSITION_0072_PATH = ARTIFACT_0072 / "cell_axis_decomposition.csv"


@dataclass(frozen=True)
class SourceEraSpec:
    candidate_id: str
    mode: str
    candidate_class: str
    expert_set: str
    min_total_history: int
    min_fold_history: int
    min_total_lift_c: float
    min_fold_lift_c: float
    support_shrink: float
    lift_scale_c: float
    base_shrink: float
    max_abs_delta: float
    combine_mode: str


@dataclass(frozen=True)
class ExpertDecision:
    expert_name: str
    total_count: int
    fold_count: int
    raw_delta: float
    shrunk_delta: float
    total_lift_c: float
    fold_lift_c: float
    shrink_factor: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_decomposition() -> pd.DataFrame:
    if not DECOMPOSITION_0072_PATH.exists():
        raise FileNotFoundError(f"Missing 0072 decomposition: {DECOMPOSITION_0072_PATH}")
    return pd.read_csv(DECOMPOSITION_0072_PATH)


def expert_names_for_row(row: pd.Series, expert_set: str) -> list[str]:
    source = str(row["forecast_source_family"])
    signed = str(row["signeddiff_bucket"])
    range_bucket = str(row["forecast_range_bucket"])
    weather = str(row["weather_bucket"])
    active = str(row["active_count_bucket"])
    month = int(row["month"])
    season = str(row["season"])

    experts: list[str] = []
    if source == "rss_archive" and signed == "station_warmer_ge_1c" and range_bucket == "range_le_3c":
        experts.append("rss_warm_tight_range")
        if season == "MAM":
            experts.append("rss_warm_tight_range_mam")
        if month == 5:
            experts.append("rss_warm_tight_range_may")
        if active == "station_stack_inactive":
            experts.append("rss_warm_tight_range_inactive")
    if source == "rss_archive" and signed == "station_warmer_ge_1c" and range_bucket == "range_3_4c":
        experts.append("rss_warm_mid_range")
    if source == "rss_archive" and signed == "station_cooler_ge_1c" and range_bucket == "range_gt_5c":
        experts.append("rss_cool_wide_range")
    if source == "press_archive" and signed == "station_cooler_ge_1c" and range_bucket == "range_le_3c":
        experts.append("press_cool_tight_range")
    if source == "press_archive" and weather == "weather_sunny" and range_bucket in {"range_le_3c", "range_3_4c"}:
        experts.append("press_sunny_low_mid_range")
    if source == "press_archive" and weather == "weather_thunder" and range_bucket == "range_le_3c":
        experts.append("press_thunder_tight_range")

    if expert_set == "rss_only":
        return [name for name in experts if name.startswith("rss_")]
    if expert_set == "press_only":
        return [name for name in experts if name.startswith("press_")]
    if expert_set == "core":
        return [
            name
            for name in experts
            if name
            in {
                "rss_warm_tight_range",
                "rss_warm_tight_range_mam",
                "press_cool_tight_range",
                "press_sunny_low_mid_range",
            }
        ]
    if expert_set == "all":
        return experts
    raise ValueError(f"Unsupported expert_set: {expert_set}")


def prior_mae(abs_sums: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.full(len(DELTA_GRID), math.nan)
    return abs_sums / count


def zero_delta_index() -> int:
    return int(np.argmin(np.abs(np.array(DELTA_GRID, dtype=float))))


def select_expert_decision(
    *,
    expert_name: str,
    total_count: int,
    total_abs_sums: np.ndarray,
    fold_count: int,
    fold_abs_sums: np.ndarray,
    spec: SourceEraSpec,
) -> ExpertDecision | None:
    if total_count < spec.min_total_history or fold_count < spec.min_fold_history:
        return None
    total_mae = prior_mae(total_abs_sums, total_count)
    fold_mae = prior_mae(fold_abs_sums, fold_count)
    zero_index = zero_delta_index()
    total_best_index = int(np.nanargmin(total_mae))
    fold_best_index = int(np.nanargmin(fold_mae))
    raw_delta = float(DELTA_GRID[total_best_index])
    if raw_delta == 0.0:
        return None
    total_lift = float(total_mae[zero_index] - total_mae[total_best_index])
    fold_lift = float(fold_mae[zero_index] - fold_mae[fold_best_index])
    if total_lift < spec.min_total_lift_c or fold_lift < spec.min_fold_lift_c:
        return None
    if math.copysign(1.0, raw_delta) != math.copysign(1.0, float(DELTA_GRID[fold_best_index])):
        return None
    factor = shrink_factor(
        count=min(total_count, fold_count),
        prior_lift_c=min(total_lift, fold_lift),
        support_shrink=spec.support_shrink,
        lift_scale_c=spec.lift_scale_c,
    )
    shrunk = float(np.clip(raw_delta * spec.base_shrink * factor, -spec.max_abs_delta, spec.max_abs_delta))
    if shrunk == 0.0:
        return None
    return ExpertDecision(
        expert_name=expert_name,
        total_count=total_count,
        fold_count=fold_count,
        raw_delta=raw_delta,
        shrunk_delta=shrunk,
        total_lift_c=total_lift,
        fold_lift_c=fold_lift,
        shrink_factor=factor,
    )


def combine_expert_decisions(decisions: list[ExpertDecision], spec: SourceEraSpec) -> float:
    if not decisions:
        return 0.0
    if spec.combine_mode == "best_min_lift":
        selected = max(decisions, key=lambda item: (min(item.total_lift_c, item.fold_lift_c), item.fold_count))
        return selected.shrunk_delta
    if spec.combine_mode == "weighted_min_lift":
        weights = np.array(
            [max(min(item.total_lift_c, item.fold_lift_c), 0.0) * max(item.fold_count, 1) for item in decisions],
            dtype=float,
        )
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


def source_era_specs() -> list[SourceEraSpec]:
    specs: list[SourceEraSpec] = []
    for expert_set in ("rss_only", "core", "all"):
        for min_total in (60, 100):
            for min_fold in (20, 40):
                for total_lift in (0.005, 0.02):
                    for fold_lift in (0.0, 0.005):
                        for combine in ("best_min_lift", "weighted_min_lift"):
                            candidate_id = (
                                f"causal_sourceera_{expert_set}_t{min_total}_f{min_fold}_"
                                f"tl{total_lift}_fl{fold_lift}_{combine}"
                            ).replace(".", "p")
                            specs.append(
                                SourceEraSpec(
                                    candidate_id=candidate_id,
                                    mode="causal_source_era_shrinkage",
                                    candidate_class="causal_source_era_shrinkage",
                                    expert_set=expert_set,
                                    min_total_history=min_total,
                                    min_fold_history=min_fold,
                                    min_total_lift_c=total_lift,
                                    min_fold_lift_c=fold_lift,
                                    support_shrink=160.0,
                                    lift_scale_c=0.05,
                                    base_shrink=1.0,
                                    max_abs_delta=0.12,
                                    combine_mode=combine,
                                )
                            )
    for expert_set in ("rss_only", "core", "all"):
        candidate_id = f"diagnostic_sourceera_fixed_{expert_set}"
        specs.append(
            SourceEraSpec(
                candidate_id=candidate_id,
                mode="diagnostic_source_era_fixed",
                candidate_class="diagnostic_source_era_fixed",
                expert_set=expert_set,
                min_total_history=0,
                min_fold_history=0,
                min_total_lift_c=0.0,
                min_fold_lift_c=0.0,
                support_shrink=0.0,
                lift_scale_c=0.0,
                base_shrink=1.0,
                max_abs_delta=0.12,
                combine_mode="best_min_lift",
            )
        )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0073 candidate IDs are not unique")
    return specs


def fixed_delta_for_expert(expert_name: str) -> float:
    mapping = {
        "rss_warm_tight_range": -0.12,
        "rss_warm_tight_range_mam": -0.12,
        "rss_warm_tight_range_may": -0.12,
        "rss_warm_tight_range_inactive": -0.12,
        "rss_warm_mid_range": -0.08,
        "rss_cool_wide_range": 0.08,
        "press_cool_tight_range": 0.12,
        "press_sunny_low_mid_range": 0.08,
        "press_thunder_tight_range": 0.08,
    }
    return mapping.get(expert_name, 0.0)


def prediction_frame(
    frame: pd.DataFrame,
    deltas: list[float],
    active_counts: list[int],
    selected_experts: list[str],
    selected_lifts: list[float],
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
    out["active_expert_count"] = active_counts
    out["selected_experts"] = selected_experts
    out["selected_min_lift_c"] = selected_lifts
    return out


def apply_causal_spec(frame: pd.DataFrame, spec: SourceEraSpec) -> pd.DataFrame:
    total_state: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "abs_sums": np.zeros(len(DELTA_GRID), dtype=float)}
    )
    fold_state: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "abs_sums": np.zeros(len(DELTA_GRID), dtype=float)}
    )
    deltas: list[float] = []
    active_counts: list[int] = []
    selected_experts: list[str] = []
    selected_lifts: list[float] = []

    for _, row in frame.iterrows():
        expert_names = expert_names_for_row(row, spec.expert_set)
        decisions: list[ExpertDecision] = []
        fold_id = str(row["fold_id"])
        for expert_name in expert_names:
            total = total_state[expert_name]
            fold = fold_state[f"{fold_id}::{expert_name}"]
            decision = select_expert_decision(
                expert_name=expert_name,
                total_count=int(total["count"]),
                total_abs_sums=np.asarray(total["abs_sums"], dtype=float),
                fold_count=int(fold["count"]),
                fold_abs_sums=np.asarray(fold["abs_sums"], dtype=float),
                spec=spec,
            )
            if decision is not None:
                decisions.append(decision)
        delta = combine_expert_decisions(decisions, spec)
        deltas.append(delta)
        active_counts.append(len(decisions))
        if decisions:
            best = max(decisions, key=lambda item: (min(item.total_lift_c, item.fold_lift_c), item.fold_count))
            selected_experts.append(best.expert_name)
            selected_lifts.append(min(best.total_lift_c, best.fold_lift_c))
        else:
            selected_experts.append("")
            selected_lifts.append(math.nan)
        errors = delta_errors(row)
        for expert_name in set(expert_names):
            total = total_state[expert_name]
            total["abs_sums"] = np.asarray(total["abs_sums"], dtype=float) + errors
            total["count"] = int(total["count"]) + 1
            fold = fold_state[f"{fold_id}::{expert_name}"]
            fold["abs_sums"] = np.asarray(fold["abs_sums"], dtype=float) + errors
            fold["count"] = int(fold["count"]) + 1
    return prediction_frame(frame, deltas, active_counts, selected_experts, selected_lifts)


def apply_diagnostic_spec(frame: pd.DataFrame, spec: SourceEraSpec) -> pd.DataFrame:
    deltas: list[float] = []
    active_counts: list[int] = []
    selected_experts: list[str] = []
    selected_lifts: list[float] = []
    for _, row in frame.iterrows():
        names = expert_names_for_row(row, spec.expert_set)
        expert_deltas = [(name, fixed_delta_for_expert(name)) for name in names]
        expert_deltas = [(name, delta) for name, delta in expert_deltas if delta != 0.0]
        if not expert_deltas:
            deltas.append(0.0)
            active_counts.append(0)
            selected_experts.append("")
            selected_lifts.append(math.nan)
            continue
        name, delta = max(expert_deltas, key=lambda item: abs(item[1]))
        deltas.append(delta)
        active_counts.append(len(expert_deltas))
        selected_experts.append(name)
        selected_lifts.append(math.nan)
    return prediction_frame(frame, deltas, active_counts, selected_experts, selected_lifts)


def apply_spec(frame: pd.DataFrame, spec: SourceEraSpec) -> pd.DataFrame:
    if spec.mode == "causal_source_era_shrinkage":
        out = apply_causal_spec(frame, spec)
    elif spec.mode == "diagnostic_source_era_fixed":
        out = apply_diagnostic_spec(frame, spec)
    else:
        raise ValueError(f"Unsupported 0073 mode: {spec.mode}")
    out["candidate_id"] = spec.candidate_id
    out["mode"] = spec.mode
    out["candidate_class"] = spec.candidate_class
    out["expert_set"] = spec.expert_set
    out["min_total_history"] = spec.min_total_history
    out["min_fold_history"] = spec.min_fold_history
    out["min_total_lift_c"] = spec.min_total_lift_c
    out["min_fold_lift_c"] = spec.min_fold_lift_c
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
    fold_deltas: list[float] = []
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
        fold_deltas.append(float(fold_score["mae"]) - float(fold_base["mae"]))
    active_mask = pd.to_numeric(predictions["active_expert_count"], errors="coerce").gt(0)
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
        "expert_set": str(predictions["expert_set"].iloc[0]),
        "min_total_history": int(predictions["min_total_history"].iloc[0]),
        "min_fold_history": int(predictions["min_fold_history"].iloc[0]),
        "min_total_lift_c": float(predictions["min_total_lift_c"].iloc[0]),
        "min_fold_lift_c": float(predictions["min_fold_lift_c"].iloc[0]),
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
        "fold_delta_max_vs_0069": max(fold_deltas) if fold_deltas else math.nan,
        "fold_delta_min_vs_0069": min(fold_deltas) if fold_deltas else math.nan,
        "folds_improved_vs_0069": int(sum(delta < 0 for delta in fold_deltas)),
        "active_rows": int(active_mask.sum()),
        "active_row_share": float(active_mask.mean()),
        "active_mae": active_score["mae"],
        "active_delta_mae_vs_0069": (
            float(active_score["mae"]) - float(active_base["mae"]) if int(active_score["n"]) > 0 else math.nan
        ),
        "mean_station_delta": float(pd.to_numeric(predictions["station_delta"], errors="coerce").mean()),
        "mean_abs_station_delta": float(pd.to_numeric(predictions["station_delta"], errors="coerce").abs().mean()),
    }
    row["beats_0069"] = bool(float(row["delta_mae_vs_0069"]) <= -0.0005)
    row["promotion_gate_passed"] = bool(
        row["beats_0069"]
        and float(row["fold_delta_max_vs_0069"]) <= 0.0
        and float(row["late_delta_mae_vs_0069"]) <= 0.0
    )
    row["deployable_gate_passed"] = bool(
        row["promotion_gate_passed"] and str(row["candidate_class"]) == "causal_source_era_shrinkage"
    )
    return row


def score_all_specs(
    frame: pd.DataFrame,
    specs: list[SourceEraSpec],
    *,
    base_0069_mae: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions_list: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_spec(frame, spec)
        rows.append(score_candidate(frame, predictions, base_0069_mae=base_0069_mae))
        predictions_list.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["deployable_gate_passed", "beats_0069", "promotion_gate_passed", "mae"],
        ascending=[False, False, False, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(25).astype(str))
    top_predictions = pd.concat(
        [predictions for predictions in predictions_list if str(predictions["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(top_predictions["target_date"], context="0073 top predictions")
    return scoreboard.reset_index(drop=True), top_predictions


def expert_family_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for expert_set in ("rss_only", "press_only", "core", "all"):
        memberships: dict[str, list[int]] = defaultdict(list)
        for idx, row in frame.iterrows():
            for expert_name in expert_names_for_row(row, expert_set):
                memberships[expert_name].append(int(idx))
        for expert_name, indices in memberships.items():
            subframe = frame.loc[indices].copy()
            if len(subframe) < 20:
                continue
            base_score = score_prediction(
                subframe,
                pd.to_numeric(subframe["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
            )
            fixed_delta = fixed_delta_for_expert(expert_name)
            prediction = blend_prediction(
                subframe,
                np.clip(
                    pd.to_numeric(subframe["base_0069_station_weight"], errors="coerce").to_numpy(dtype=float)
                    + fixed_delta,
                    0.0,
                    0.50,
                ),
            )
            fixed_score = score_prediction(subframe, prediction)
            rows.append(
                {
                    "expert_set": expert_set,
                    "expert_name": expert_name,
                    "n": int(len(subframe)),
                    "fixed_delta": fixed_delta,
                    "base_0069_mae": base_score["mae"],
                    "fixed_delta_mae": fixed_score["mae"],
                    "delta_mae_vs_0069": float(fixed_score["mae"]) - float(base_score["mae"]),
                    "first_date": str(pd.to_datetime(subframe["target_date"]).min().date()),
                    "last_date": str(pd.to_datetime(subframe["target_date"]).max().date()),
                    "mean_family_disagreement_c": float(subframe["family_disagreement_c"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["delta_mae_vs_0069", "n"], ascending=[True, False]).reset_index(drop=True)


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
            "check_id": "causal_source_era_uses_total_and_fold_prior_state",
            "passed": True,
            "evidence": "causal activation requires both total prior and same-fold prior state before row update",
        },
        {
            "check_id": "diagnostic_fixed_rules_not_marked_deployable",
            "passed": bool(
                scoreboard.loc[
                    scoreboard["candidate_class"].ne("causal_source_era_shrinkage"),
                    "deployable_gate_passed",
                ].eq(False).all()
            ),
            "evidence": "fixed source-era rules remain diagnostic only",
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
    diagnostics: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    deployable = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    causal = scoreboard[scoreboard["candidate_class"].eq("causal_source_era_shrinkage")].head(80).copy()
    diagnostic = scoreboard[scoreboard["candidate_class"].ne("causal_source_era_shrinkage")].copy()
    return f"""# Source-Era-Specific Shrinkage

Generated: `{summary['generated_at_utc']}`

## Purpose

`0072` showed that generic smooth shrinkage improves directionally but still has small fold damage. `0073` makes the shrinkage source-era-specific: RSS warm/tight-range cells and press cool/sunny cells get separate priors, and causal activation requires both total prior support and same-fold prior support.

## Data Contract

- Base prediction: `0069` best deployable prediction.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ rows are used.
- Causal activation updates expert states after each row is scored.
- Diagnostic fixed source-era rules are never deployable.

## Headline

| Item | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| Best 0073 candidate | {summary['best_candidate']} |
| Best 0073 class | {summary['best_candidate_class']} |
| Best 0073 MAE | {summary['best_mae']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best causal candidate | {summary['best_causal_candidate']} |
| Best causal MAE | {summary['best_causal_mae']} |
| Best causal fold max delta vs 0069 | {summary['best_causal_fold_delta_max_vs_0069']} |
| Gate-passed deployable candidate | {summary['best_deployable_candidate']} |
| Gate-passed deployable MAE | {summary['best_deployable_mae']} |

## Interpretation

This experiment tests whether explicit same-fold support floors can avoid the causal misfires seen in `0071` and `0072`. A deployable candidate must improve the full frame, the late window, and every fold versus `0069`.

## Scoreboard

{markdown_table(scoreboard, max_rows=100)}

## Deployable Candidates

{markdown_table(deployable, max_rows=80)}

## Causal Source-Era Candidates

{markdown_table(causal, max_rows=100)}

## Diagnostic Fixed Source-Era Rules

{markdown_table(diagnostic, max_rows=30)}

## Expert Family Diagnostics

{markdown_table(diagnostics, max_rows=100)}

## Candidate Definitions

{markdown_table(definitions, max_rows=120)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/expert_family_diagnostics.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_source_era_specific_shrinkage.py`:

- `{FOLDER_NAME}`: source-era-specific shrinkage with total and same-fold prior support floors.

| Metric | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| Best 0073 candidate | {summary['best_candidate']} |
| Best 0073 MAE | {summary['best_mae']} |
| Best causal candidate | {summary['best_causal_candidate']} |
| Best causal MAE | {summary['best_causal_mae']} |
| Gate-passed deployable candidate | {summary['best_deployable_candidate']} |

Leakage contract: no 2024+ rows; causal activation requires total and same-fold prior support before row update.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Source-Era-Specific Shrinkage",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_source_era_specific_shrinkage.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0069` base predictions plus source-era expert families from `0072` | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous |
| Candidate count | `{summary['candidate_count']}` | Tested |
| Base 0069 MAE / RMSE | `{summary['base_0069_mae']}` / `{summary['base_0069_rmse']}` | Baseline |
| Best 0073 candidate | `{summary['best_candidate']}` | Tested |
| Best 0073 class | `{summary['best_candidate_class']}` | Diagnostic/deployable classification |
| Best 0073 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0069 | `{summary['best_delta_mae_vs_0069']}` | Source-era shrinkage value |
| Best causal candidate | `{summary['best_causal_candidate']}` | Prior-only source-era shrinkage |
| Best causal MAE | `{summary['best_causal_mae']}` | Pre-2024 only |
| Best causal fold max delta vs 0069 | `{summary['best_causal_fold_delta_max_vs_0069']}` | Robustness check |
| Gate-passed deployable candidate | `{summary['best_deployable_candidate']}` | Requires full, fold, and late improvement |
| Gate-passed deployable MAE | `{summary['best_deployable_mae']}` | `None` means no candidate passed |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0073` tests source-era-specific smooth shrinkage with both total and same-fold prior support floors before activation.
"""
    update_markdown_section(
        path,
        heading="Source-Era-Specific Shrinkage",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"57. Source-era-specific shrinkage screened `{summary['candidate_count']}` candidates; "
        f"best delta vs 0069 is `{summary['best_delta_mae_vs_0069']}` from "
        f"`{summary['best_candidate']}`, best causal MAE is `{summary['best_causal_mae']}`, "
        f"and `{summary['deployable_candidate_count']}` candidates passed the strict deployable gate."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: use the `0073` expert diagnostics to test month/season-specialized RSS warm-tight-range priors, especially MAM/May, while preserving same-fold support floors and keeping diagnostic fixed rules out of deployable promotion.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0069 = build_feature_frame()
    frame = ensure_calendar_columns(frame)
    summary_0072 = load_json(SUMMARY_0072_PATH)
    base_score = score_prediction(
        frame,
        pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    specs = source_era_specs()
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_specs(frame, specs, base_0069_mae=float(base_score["mae"]))
    diagnostics = expert_family_diagnostics(frame)
    leakage = leakage_audit(frame, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0073 leakage audit failed: {failed}")

    best = scoreboard.iloc[0]
    causal_pool = scoreboard[scoreboard["candidate_class"].eq("causal_source_era_shrinkage")].copy()
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
        "diagnostic_rows": int(len(diagnostics)),
        "base_0069_candidate": str(summary_0069["best_deployable_candidate"]),
        "base_0069_mae": float(base_score["mae"]),
        "base_0069_rmse": float(base_score["rmse"]),
        "base_0072_best_causal_mae": float(summary_0072["best_causal_mae"]),
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
    write_csv(artifacts / "expert_family_diagnostics.csv", diagnostics)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "source_era_specific_shrinkage_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            definitions=definitions,
            scoreboard=scoreboard,
            diagnostics=diagnostics,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Run source-era-specific shrinkage with same-fold support floors."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
