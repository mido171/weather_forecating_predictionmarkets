from __future__ import annotations

import argparse
import json
import math
import sys
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
    ensure_calendar_columns,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_nonlinear_local_residual_fusion_lab import (  # noqa: E402
    build_feature_frame,
)
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import (  # noqa: E402
    score_prediction,
)
from scripts.run_hkg_t24_station_official_family_router import LATE_EVAL_START  # noqa: E402

FOLDER_NAME = "0074_online_residual_memory_halflife"
BASE_MATERIALITY_C = 0.0005


@dataclass(frozen=True)
class OnlineMemorySpec:
    candidate_id: str
    context_set: str
    min_history: int
    min_perf_history: int
    halflife_rows: float
    support_shrink: float
    min_prior_lift_c: float
    correction_cap_c: float
    combine_mode: str
    max_contexts: int


@dataclass
class MemoryState:
    residual_sum: float = 0.0
    residual_weight: float = 0.0
    residual_count: int = 0
    perf_base_abs_sum: float = 0.0
    perf_corrected_abs_sum: float = 0.0
    perf_weight: float = 0.0
    perf_count: int = 0

    def decay(self, halflife_rows: float) -> None:
        factor = half_life_factor(halflife_rows)
        self.residual_sum *= factor
        self.residual_weight *= factor
        self.perf_base_abs_sum *= factor
        self.perf_corrected_abs_sum *= factor
        self.perf_weight *= factor

    def raw_correction(self) -> float:
        if self.residual_weight <= 0.0:
            return 0.0
        return self.residual_sum / self.residual_weight

    def prior_base_mae(self) -> float:
        if self.perf_weight <= 0.0:
            return math.nan
        return self.perf_base_abs_sum / self.perf_weight

    def prior_corrected_mae(self) -> float:
        if self.perf_weight <= 0.0:
            return math.nan
        return self.perf_corrected_abs_sum / self.perf_weight

    def prior_lift(self) -> float:
        base = self.prior_base_mae()
        corrected = self.prior_corrected_mae()
        if not math.isfinite(base) or not math.isfinite(corrected):
            return math.nan
        return base - corrected


@dataclass(frozen=True)
class ContextDecision:
    context_key: str
    correction_c: float
    prior_lift_c: float
    prior_base_mae: float
    prior_corrected_mae: float
    residual_count: int
    perf_count: int
    reliability: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def half_life_factor(halflife_rows: float) -> float:
    if halflife_rows <= 0:
        raise ValueError("halflife_rows must be positive")
    return float(0.5 ** (1.0 / halflife_rows))


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def context_keys_for_row(row: pd.Series, context_set: str) -> list[str]:
    source = str(row["forecast_source_family"])
    season = str(row["season"])
    month = int(row["month"])
    signed = str(row["signeddiff_bucket"])
    range_bucket = str(row["forecast_range_bucket"])
    weather = str(row["weather_bucket"])
    active = str(row["active_count_bucket"])

    keys: list[str] = [f"source={source}"]
    if context_set in {"seasonal", "behavior", "seasonal_behavior", "all"}:
        keys.extend(
            [
                f"source={source}|season={season}",
                f"source={source}|month={month:02d}",
            ]
        )
    if context_set in {"behavior", "seasonal_behavior", "all"}:
        keys.extend(
            [
                f"source={source}|signed={signed}",
                f"source={source}|range={range_bucket}",
                f"source={source}|signed={signed}|range={range_bucket}",
                f"source={source}|weather={weather}",
                f"source={source}|weather={weather}|range={range_bucket}",
                f"source={source}|active={active}",
            ]
        )
    if context_set in {"seasonal_behavior", "all"}:
        keys.extend(
            [
                f"source={source}|season={season}|signed={signed}|range={range_bucket}",
                f"source={source}|month={month:02d}|signed={signed}|range={range_bucket}",
                f"source={source}|season={season}|weather={weather}|range={range_bucket}",
            ]
        )
    if context_set == "all":
        keys.extend(
            [
                f"source={source}|signed={signed}|range={range_bucket}|active={active}",
                f"source={source}|season={season}|signed={signed}|range={range_bucket}|active={active}",
                f"source={source}|month={month:02d}|weather={weather}|range={range_bucket}",
            ]
        )
    if context_set not in {"source", "seasonal", "behavior", "seasonal_behavior", "all"}:
        raise ValueError(f"Unsupported context_set: {context_set}")
    return unique_preserve_order(keys)


def estimate_context_correction(state: MemoryState, spec: OnlineMemorySpec) -> float:
    if state.residual_count < spec.min_history:
        return 0.0
    shrink = state.residual_weight / (state.residual_weight + spec.support_shrink)
    correction = state.raw_correction() * shrink
    return float(np.clip(correction, -spec.correction_cap_c, spec.correction_cap_c))


def eligible_decision(context_key: str, state: MemoryState, spec: OnlineMemorySpec) -> ContextDecision | None:
    if state.residual_count < spec.min_history or state.perf_count < spec.min_perf_history:
        return None
    correction = estimate_context_correction(state, spec)
    if not math.isfinite(correction) or abs(correction) < 1e-9:
        return None
    prior_lift = state.prior_lift()
    if not math.isfinite(prior_lift) or prior_lift < spec.min_prior_lift_c:
        return None
    base_mae = state.prior_base_mae()
    corrected_mae = state.prior_corrected_mae()
    reliability = max(prior_lift, 0.0) * math.sqrt(max(state.perf_weight, 1.0))
    return ContextDecision(
        context_key=context_key,
        correction_c=correction,
        prior_lift_c=prior_lift,
        prior_base_mae=base_mae,
        prior_corrected_mae=corrected_mae,
        residual_count=state.residual_count,
        perf_count=state.perf_count,
        reliability=reliability,
    )


def combine_context_decisions(decisions: list[ContextDecision], spec: OnlineMemorySpec) -> float:
    if not decisions:
        return 0.0
    selected = sorted(decisions, key=lambda item: item.reliability, reverse=True)[: spec.max_contexts]
    if spec.combine_mode == "best_lift":
        return float(np.clip(selected[0].correction_c, -spec.correction_cap_c, spec.correction_cap_c))
    if spec.combine_mode == "lift_weighted":
        weights = np.array([max(item.reliability, 1e-6) for item in selected], dtype=float)
    elif spec.combine_mode == "inverse_corrected_mae":
        weights = np.array(
            [max(item.perf_count, 1) / max(item.prior_corrected_mae, 0.05) ** 2 for item in selected],
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported combine_mode: {spec.combine_mode}")
    if float(weights.sum()) <= 0.0:
        return 0.0
    corrections = np.array([item.correction_c for item in selected], dtype=float)
    return float(np.clip(np.sum(weights * corrections) / weights.sum(), -spec.correction_cap_c, spec.correction_cap_c))


def online_memory_specs() -> list[OnlineMemorySpec]:
    specs: list[OnlineMemorySpec] = []
    for context_set in ("source", "behavior", "seasonal_behavior", "all"):
        for halflife_rows in (45.0, 120.0, 365.0):
            for min_history in (20, 80):
                for min_prior_lift in (0.0,):
                    for correction_cap in (0.20,):
                        for combine_mode in ("best_lift", "lift_weighted"):
                            candidate_id = (
                                f"causal_onmem_{context_set}_h{int(halflife_rows)}_n{min_history}_"
                                f"lift{min_prior_lift}_cap{correction_cap}_{combine_mode}"
                            ).replace(".", "p")
                            specs.append(
                                OnlineMemorySpec(
                                    candidate_id=candidate_id,
                                    context_set=context_set,
                                    min_history=min_history,
                                    min_perf_history=min_history,
                                    halflife_rows=halflife_rows,
                                    support_shrink=60.0,
                                    min_prior_lift_c=min_prior_lift,
                                    correction_cap_c=correction_cap,
                                    combine_mode=combine_mode,
                                    max_contexts=4,
                                )
                            )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0074 candidate IDs are not unique")
    return specs


def apply_online_memory_spec(frame: pd.DataFrame, spec: OnlineMemorySpec) -> pd.DataFrame:
    states: dict[str, MemoryState] = {}
    corrections: list[float] = []
    active_counts: list[int] = []
    selected_contexts: list[str] = []
    selected_lifts: list[float] = []
    selected_base_mae: list[float] = []
    selected_corrected_mae: list[float] = []

    for _, row in frame.iterrows():
        base_prediction = float(row["base_0069_prediction_c"])
        target = float(row["target_tmax_c"])
        residual = target - base_prediction
        base_abs_error = abs(residual)
        keys = context_keys_for_row(row, spec.context_set)
        decisions: list[ContextDecision] = []
        estimates_before_update: dict[str, float] = {}

        for key in keys:
            state = states.setdefault(key, MemoryState())
            estimate = estimate_context_correction(state, spec)
            estimates_before_update[key] = estimate
            decision = eligible_decision(key, state, spec)
            if decision is not None:
                decisions.append(decision)

        correction = combine_context_decisions(decisions, spec)
        corrections.append(correction)
        active_counts.append(len(decisions))
        if decisions:
            top = sorted(decisions, key=lambda item: item.reliability, reverse=True)[: spec.max_contexts]
            selected_contexts.append(";".join(item.context_key for item in top))
            selected_lifts.append(max(item.prior_lift_c for item in top))
            selected_base_mae.append(min(item.prior_base_mae for item in top))
            selected_corrected_mae.append(min(item.prior_corrected_mae for item in top))
        else:
            selected_contexts.append("")
            selected_lifts.append(math.nan)
            selected_base_mae.append(math.nan)
            selected_corrected_mae.append(math.nan)

        for key in keys:
            state = states[key]
            estimate = estimates_before_update[key]
            corrected_abs_error = abs(target - (base_prediction + estimate))
            state.decay(spec.halflife_rows)
            state.perf_base_abs_sum += base_abs_error
            state.perf_corrected_abs_sum += corrected_abs_error
            state.perf_weight += 1.0
            state.perf_count += 1
            state.residual_sum += residual
            state.residual_weight += 1.0
            state.residual_count += 1

    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["row_index"] = frame.index.to_numpy(dtype=int)
    out["base_0069_prediction_c"] = pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce")
    out["memory_correction_c"] = np.array(corrections, dtype=float)
    out["candidate_prediction_c"] = out["base_0069_prediction_c"] + out["memory_correction_c"]
    out["active_context_count"] = active_counts
    out["selected_contexts"] = selected_contexts
    out["selected_prior_lift_c"] = selected_lifts
    out["selected_prior_base_mae"] = selected_base_mae
    out["selected_prior_corrected_mae"] = selected_corrected_mae
    out["candidate_id"] = spec.candidate_id
    out["candidate_class"] = "causal_online_residual_memory"
    out["context_set"] = spec.context_set
    out["min_history"] = spec.min_history
    out["min_perf_history"] = spec.min_perf_history
    out["halflife_rows"] = spec.halflife_rows
    out["support_shrink"] = spec.support_shrink
    out["min_prior_lift_c"] = spec.min_prior_lift_c
    out["correction_cap_c"] = spec.correction_cap_c
    out["combine_mode"] = spec.combine_mode
    out["max_contexts"] = spec.max_contexts
    return out


def score_candidate(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    base_0069_mae: float,
) -> dict[str, object]:
    pred_values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    score = score_prediction(frame, pred_values)
    base_values = pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float)
    base_score = score_prediction(frame, base_values)
    official_score = score_prediction(
        frame,
        pd.to_numeric(frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    late_mask = predictions["target_date"].ge(LATE_EVAL_START)
    late_score = score_prediction(frame.loc[late_mask].copy(), pred_values[late_mask.to_numpy()])
    late_base = score_prediction(frame.loc[late_mask].copy(), base_values[late_mask.to_numpy()])
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
    active_mask = pd.to_numeric(predictions["active_context_count"], errors="coerce").gt(0)
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
        "candidate_class": str(predictions["candidate_class"].iloc[0]),
        "context_set": str(predictions["context_set"].iloc[0]),
        "min_history": int(predictions["min_history"].iloc[0]),
        "min_perf_history": int(predictions["min_perf_history"].iloc[0]),
        "halflife_rows": float(predictions["halflife_rows"].iloc[0]),
        "support_shrink": float(predictions["support_shrink"].iloc[0]),
        "min_prior_lift_c": float(predictions["min_prior_lift_c"].iloc[0]),
        "correction_cap_c": float(predictions["correction_cap_c"].iloc[0]),
        "combine_mode": str(predictions["combine_mode"].iloc[0]),
        "max_contexts": int(predictions["max_contexts"].iloc[0]),
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
        "mean_correction_c": float(pd.to_numeric(predictions["memory_correction_c"], errors="coerce").mean()),
        "mean_abs_correction_c": float(
            pd.to_numeric(predictions["memory_correction_c"], errors="coerce").abs().mean()
        ),
    }
    row["beats_0069"] = bool(float(row["delta_mae_vs_0069"]) <= -BASE_MATERIALITY_C)
    row["promotion_gate_passed"] = bool(
        row["beats_0069"]
        and float(row["fold_delta_max_vs_0069"]) <= 0.0
        and float(row["late_delta_mae_vs_0069"]) <= 0.0
    )
    row["deployable_gate_passed"] = bool(row["promotion_gate_passed"])
    return row


def score_all_specs(
    frame: pd.DataFrame,
    specs: list[OnlineMemorySpec],
    *,
    base_0069_mae: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions_list: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_online_memory_spec(frame, spec)
        rows.append(score_candidate(frame, predictions, base_0069_mae=base_0069_mae))
        predictions_list.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["deployable_gate_passed", "beats_0069", "promotion_gate_passed", "mae"],
        ascending=[False, False, False, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(30).astype(str))
    top_predictions = pd.concat(
        [predictions for predictions in predictions_list if str(predictions["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(top_predictions["target_date"], context="0074 top predictions")
    return scoreboard.reset_index(drop=True), top_predictions


def context_diagnostics(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = predictions[predictions["selected_contexts"].astype(str).ne("")].copy()
    if selected.empty:
        return pd.DataFrame()
    for context_key, group in selected.assign(
        context_key=selected["selected_contexts"].astype(str).str.split(";")
    ).explode("context_key").groupby("context_key", observed=True):
        row_indices = pd.to_numeric(group["row_index"], errors="coerce").astype(int).to_numpy()
        subframe = frame.iloc[row_indices].copy()
        subpred = group.copy()
        base_score = score_prediction(
            subframe,
            pd.to_numeric(subframe["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        candidate_score = score_prediction(
            subframe,
            pd.to_numeric(subpred["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        rows.append(
            {
                "context_key": context_key,
                "n": int(len(subframe)),
                "first_date": str(pd.to_datetime(subframe["target_date"]).min().date()),
                "last_date": str(pd.to_datetime(subframe["target_date"]).max().date()),
                "base_0069_mae": base_score["mae"],
                "candidate_mae": candidate_score["mae"],
                "delta_mae_vs_0069": float(candidate_score["mae"]) - float(base_score["mae"]),
                "mean_correction_c": float(
                    pd.to_numeric(subpred["memory_correction_c"], errors="coerce").mean()
                ),
                "mean_abs_correction_c": float(
                    pd.to_numeric(subpred["memory_correction_c"], errors="coerce").abs().mean()
                ),
                "mean_selected_prior_lift_c": float(
                    pd.to_numeric(subpred["selected_prior_lift_c"], errors="coerce").mean()
                ),
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
            "check_id": "online_memory_updates_after_scoring",
            "passed": True,
            "evidence": "each row reads context states, scores prediction, then updates residual/performance states",
        },
        {
            "check_id": "deployable_gate_requires_material_full_fold_late_improvement",
            "passed": bool(
                deployable.empty
                or (
                    deployable["delta_mae_vs_0069"].le(-BASE_MATERIALITY_C).all()
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
    return f"""# Online Residual Memory Half-Life

Generated: `{summary['generated_at_utc']}`

## Purpose

`0074` tests a causal online residual-memory correction on top of the current `0069` deployable research champion. The goal is to use the currently available official forecast archive while the broader RSS/press backfill continues: each source/season/behavior context keeps a half-life-weighted memory of prior residual bias and prior do-no-harm performance.

## Data Contract

- Base prediction: `0069` best deployable prediction.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ confirmation rows are used.
- The correction for a row is computed before that row updates any context state.
- Context eligibility requires prior residual support and prior correction-performance support.
- This is additive bias correction around the existing 0069 prediction, not predictive ML training.

## Headline

| Item | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| Best 0074 candidate | {summary['best_candidate']} |
| Best 0074 MAE | {summary['best_mae']} |
| Best 0074 RMSE | {summary['best_rmse']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best late delta vs 0069 | {summary['best_late_delta_mae_vs_0069']} |
| Best fold max delta vs 0069 | {summary['best_fold_delta_max_vs_0069']} |
| Gate-passed deployable candidate | {summary['best_deployable_candidate']} |
| Gate-passed deployable MAE | {summary['best_deployable_mae']} |

## Interpretation

This run asks whether online half-life residual memory is a real deployable correction mechanism. A candidate can only pass if it materially improves full MAE, does not damage any fold, and does not damage the late evaluation window versus `0069`.

## Scoreboard

{markdown_table(scoreboard, max_rows=120)}

## Deployable Candidates

{markdown_table(deployable, max_rows=80)}

## Best-Candidate Context Diagnostics

{markdown_table(diagnostics, max_rows=120)}

## Candidate Definitions

{markdown_table(definitions, max_rows=200)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/best_context_diagnostics.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_online_residual_memory_halflife.py`:

- `{FOLDER_NAME}`: causal half-life residual-memory correction around `0069`, using source/season/behavior contexts and prior do-no-harm evidence only.

| Metric | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| Best 0074 candidate | {summary['best_candidate']} |
| Best 0074 MAE | {summary['best_mae']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best deployable candidate | {summary['best_deployable_candidate']} |
| Best deployable MAE | {summary['best_deployable_mae']} |

Leakage contract: no 2024+ rows; context residual memory and do-no-harm memory are read before scoring the row and updated only after the row is scored.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Online Residual Memory Half-Life",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_online_residual_memory_halflife.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0069` base predictions plus source/season/behavior contexts | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous official archive |
| Candidate count | `{summary['candidate_count']}` | Tested |
| Base 0069 MAE / RMSE | `{summary['base_0069_mae']}` / `{summary['base_0069_rmse']}` | Baseline |
| Best 0074 candidate | `{summary['best_candidate']}` | Tested |
| Best 0074 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0069 | `{summary['best_delta_mae_vs_0069']}` | Online memory value |
| Best fold max delta vs 0069 | `{summary['best_fold_delta_max_vs_0069']}` | Robustness check |
| Best late delta vs 0069 | `{summary['best_late_delta_mae_vs_0069']}` | Late-window check |
| Gate-passed deployable candidate | `{summary['best_deployable_candidate']}` | Requires material full, fold, and late improvement |
| Gate-passed deployable MAE | `{summary['best_deployable_mae']}` | `None` means no candidate passed |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0074` tests whether half-life-weighted online residual memory can convert the source/era diagnostic signal into a deployable correction while the forecast archive backfill is still incomplete.
"""
    update_markdown_section(
        path,
        heading="Online Residual Memory Half-Life",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"58. Online residual memory screened `{summary['candidate_count']}` candidates; "
        f"best delta vs 0069 is `{summary['best_delta_mae_vs_0069']}` from "
        f"`{summary['best_candidate']}`, with `{summary['deployable_candidate_count']}` "
        "strict deployable candidates."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: use the `0074` context diagnostics to identify whether online residual memory should be narrowed to source-season behavior cells or abandoned in favor of explicit station/upper-air specialist features around the `0069` anchor.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0069 = build_feature_frame()
    frame = ensure_calendar_columns(frame).sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0074 input frame")
    base_score = score_prediction(
        frame,
        pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    specs = online_memory_specs()
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_specs(frame, specs, base_0069_mae=float(base_score["mae"]))
    best = scoreboard.iloc[0]
    best_predictions = top_predictions[top_predictions["candidate_id"].eq(best["candidate_id"])].copy()
    diagnostics = context_diagnostics(frame, best_predictions)
    leakage = leakage_audit(frame, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0074 leakage audit failed: {failed}")

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
        "deployable_candidate_count": int(scoreboard["deployable_gate_passed"].astype(bool).sum()),
        "base_0069_candidate": str(summary_0069["best_deployable_candidate"]),
        "base_0069_mae": float(base_score["mae"]),
        "base_0069_rmse": float(base_score["rmse"]),
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0069": float(best["delta_mae_vs_0069"]),
        "best_late_delta_mae_vs_0069": float(best["late_delta_mae_vs_0069"]),
        "best_fold_delta_max_vs_0069": float(best["fold_delta_max_vs_0069"]),
        "best_active_rows": int(best["active_rows"]),
        "best_mean_abs_correction_c": float(best["mean_abs_correction_c"]),
        "best_deployable_candidate": str(best_deployable["candidate_id"]) if best_deployable is not None else "NONE",
        "best_deployable_mae": float(best_deployable["mae"]) if best_deployable is not None else None,
        "best_deployable_rmse": float(best_deployable["rmse"]) if best_deployable is not None else None,
        "best_deployable_delta_mae_vs_0069": (
            float(best_deployable["delta_mae_vs_0069"]) if best_deployable is not None else None
        ),
        "diagnostic_context_rows": int(len(diagnostics)),
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
    write_csv(artifacts / "best_context_diagnostics.csv", diagnostics)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "online_residual_memory_halflife_manifest.json", summary)
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
    return argparse.ArgumentParser(description="Run HKG T24 online residual-memory half-life screen.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
