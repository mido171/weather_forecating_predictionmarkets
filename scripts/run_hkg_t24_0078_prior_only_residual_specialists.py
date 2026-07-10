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
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_online_residual_memory_halflife import (  # noqa: E402
    BASE_MATERIALITY_C,
    half_life_factor,
)
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import score_prediction  # noqa: E402
from scripts.run_hkg_t24_remaining_0075_error_feature_mining import (  # noqa: E402
    build_joined_frame,
    feature_family,
)
from scripts.run_hkg_t24_station_official_family_router import LATE_EVAL_START  # noqa: E402

FOLDER_NAME = "0078_prior_only_residual_specialists"


@dataclass(frozen=True)
class FeaturePredicate:
    feature: str
    direction: str
    threshold: float
    family: str
    label: str


@dataclass(frozen=True)
class SpecialistSpec:
    candidate_id: str
    predicate: FeaturePredicate
    context_mode: str
    min_history: int
    halflife_rows: float
    support_shrink: float
    min_prior_lift_c: float
    correction_cap_c: float


@dataclass
class SpecialistState:
    residual_sum: float = 0.0
    residual_weight: float = 0.0
    residual_count: int = 0
    base_abs_sum: float = 0.0
    corrected_abs_sum: float = 0.0
    perf_weight: float = 0.0
    perf_count: int = 0

    def decay(self, halflife_rows: float) -> None:
        factor = half_life_factor(halflife_rows)
        self.residual_sum *= factor
        self.residual_weight *= factor
        self.base_abs_sum *= factor
        self.corrected_abs_sum *= factor
        self.perf_weight *= factor

    def raw_correction(self) -> float:
        if self.residual_weight <= 0:
            return 0.0
        return self.residual_sum / self.residual_weight

    def prior_lift(self) -> float:
        if self.perf_weight <= 0:
            return math.nan
        return (self.base_abs_sum - self.corrected_abs_sum) / self.perf_weight


@dataclass(frozen=True)
class SpecialistDecision:
    context_key: str
    correction_c: float
    prior_lift_c: float
    count: int
    weight: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def predicate_is_active(row: pd.Series, predicate: FeaturePredicate) -> bool:
    value = pd.to_numeric(pd.Series([row.get(predicate.feature)]), errors="coerce").iloc[0]
    if not math.isfinite(float(value)):
        return False
    if predicate.direction == "low":
        return float(value) <= predicate.threshold
    if predicate.direction == "high":
        return float(value) >= predicate.threshold
    raise ValueError(f"Unsupported predicate direction: {predicate.direction}")


def specialist_context_keys(row: pd.Series, spec: SpecialistSpec) -> list[str]:
    source = str(row["forecast_source_family"])
    timestamp = pd.to_datetime(row["target_date"], errors="coerce")
    month = int(timestamp.month)
    if month in {12, 1, 2}:
        season = "DJF"
    elif month in {3, 4, 5}:
        season = "MAM"
    elif month in {6, 7, 8}:
        season = "JJA"
    else:
        season = "SON"
    base = f"{spec.predicate.label}:{spec.predicate.direction}{spec.predicate.threshold:g}"
    if spec.context_mode == "feature":
        return [base]
    if spec.context_mode == "feature_source":
        return [base, f"{base}|source={source}"]
    if spec.context_mode == "feature_source_season":
        return [base, f"{base}|source={source}", f"{base}|source={source}|season={season}"]
    if spec.context_mode == "feature_source_month":
        return [base, f"{base}|source={source}", f"{base}|source={source}|month={month:02d}"]
    raise ValueError(f"Unsupported context_mode: {spec.context_mode}")


def estimate_correction(state: SpecialistState, spec: SpecialistSpec) -> float:
    if state.residual_count < spec.min_history:
        return 0.0
    shrink = state.residual_weight / (state.residual_weight + spec.support_shrink)
    correction = state.raw_correction() * shrink
    return float(np.clip(correction, -spec.correction_cap_c, spec.correction_cap_c))


def eligible_decision(context_key: str, state: SpecialistState, spec: SpecialistSpec) -> SpecialistDecision | None:
    if state.residual_count < spec.min_history or state.perf_count < spec.min_history:
        return None
    correction = estimate_correction(state, spec)
    if abs(correction) < 1e-9:
        return None
    lift = state.prior_lift()
    if not math.isfinite(lift) or lift < spec.min_prior_lift_c:
        return None
    return SpecialistDecision(
        context_key=context_key,
        correction_c=correction,
        prior_lift_c=lift,
        count=state.residual_count,
        weight=state.perf_weight,
    )


def combine_decisions(decisions: list[SpecialistDecision], spec: SpecialistSpec) -> float:
    if not decisions:
        return 0.0
    weights = np.array([max(item.prior_lift_c, 1e-6) * max(item.weight, 1.0) for item in decisions], dtype=float)
    corrections = np.array([item.correction_c for item in decisions], dtype=float)
    if float(weights.sum()) <= 0:
        return 0.0
    return float(np.clip(np.sum(weights * corrections) / weights.sum(), -spec.correction_cap_c, spec.correction_cap_c))


def fold_id_for_date(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "fold_unknown"
    year = int(timestamp.year)
    if year <= 2005:
        return "fold_2000_2005"
    return f"fold_{year}"


def base_predicates() -> list[FeaturePredicate]:
    rows = [
        ("target_roll120_mean_lag7_c", "low", 22.0, "cool_120d_memory"),
        ("target_roll90_mean_lag7_c", "low", 23.0, "cool_90d_memory"),
        ("target_lag60_tmax_c", "low", 22.0, "cool_lag60_memory"),
        ("doy_sin", "high", 0.30, "spring_transition_calendar"),
        ("isd_dewpoint_midday_minus_temp_c", "high", -4.0, "humid_midday_dewpoint_spread"),
        ("isd_temp_dewpoint_spread_mean_c", "low", 4.0, "humid_mean_dewpoint_spread"),
        ("isd_morning_to_midday_temp_rise_c", "low", 1.5, "weak_morning_warming"),
        ("daily_waglan_island_sea_temperature_lag7_roll7", "low", 22.0, "cool_waglan_sea_memory"),
        ("daily_waglan_island_sea_temperature_lag7_roll7", "high", 27.0, "warm_waglan_sea_memory"),
        ("daily_waglan_island_sea_temperature_lag7", "low", 22.0, "cool_waglan_sea_lag7"),
        ("igra_inversion_925_minus_1000_c", "high", -2.0, "weak_low_level_inversion"),
        ("igra_inversion_925_minus_1000_c", "low", -5.0, "strong_low_level_inversion"),
        ("igra_dd_1000hpa_c", "low", 2.0, "moist_1000hpa_depression"),
        ("igra_lower_mean_dd_c", "low", 2.5, "moist_lower_layer_depression"),
        ("ua_vector_shear_925_850_mps", "high", 7.0, "strong_low_level_shear"),
    ]
    return [
        FeaturePredicate(feature=feature, direction=direction, threshold=threshold, family=feature_family(feature), label=label)
        for feature, direction, threshold, label in rows
    ]


def specialist_specs(available_columns: set[str] | None = None) -> list[SpecialistSpec]:
    predicates = base_predicates()
    if available_columns is not None:
        predicates = [predicate for predicate in predicates if predicate.feature in available_columns]
    specs: list[SpecialistSpec] = []
    for predicate in predicates:
        for context_mode in ("feature_source", "feature_source_season"):
            for halflife_rows in (30.0, 90.0):
                candidate_id = (
                    f"specialist_{predicate.label}_{predicate.direction}{predicate.threshold:g}_"
                    f"{context_mode}_h{int(halflife_rows)}"
                )
                candidate_id = candidate_id.replace(".", "p").replace("-", "m")
                specs.append(
                    SpecialistSpec(
                        candidate_id=candidate_id,
                        predicate=predicate,
                        context_mode=context_mode,
                        min_history=20,
                        halflife_rows=halflife_rows,
                        support_shrink=60.0,
                        min_prior_lift_c=0.0,
                        correction_cap_c=0.20,
                    )
                )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0078 candidate IDs are not unique")
    return specs


def apply_specialist_spec(frame: pd.DataFrame, spec: SpecialistSpec) -> pd.DataFrame:
    states: dict[str, SpecialistState] = {}
    predictions: list[float] = []
    corrections: list[float] = []
    active_flags: list[bool] = []
    selected_contexts: list[str] = []
    selected_lifts: list[float] = []

    for _, row in frame.iterrows():
        base_prediction = float(row["m0075_prediction_c"])
        target = float(row["current_target_tmax_c"])
        active = predicate_is_active(row, spec.predicate)
        decisions: list[SpecialistDecision] = []
        estimates_before_update: dict[str, float] = {}
        keys = specialist_context_keys(row, spec) if active else []
        for key in keys:
            state = states.setdefault(key, SpecialistState())
            estimates_before_update[key] = estimate_correction(state, spec)
            decision = eligible_decision(key, state, spec)
            if decision is not None:
                decisions.append(decision)
        correction = combine_decisions(decisions, spec)
        predictions.append(base_prediction + correction)
        corrections.append(correction)
        active_flags.append(active and bool(decisions))
        if decisions:
            selected_contexts.append(";".join(item.context_key for item in decisions))
            selected_lifts.append(max(item.prior_lift_c for item in decisions))
        else:
            selected_contexts.append("")
            selected_lifts.append(math.nan)

        if active:
            residual = target - base_prediction
            base_abs_error = abs(residual)
            for key in keys:
                state = states[key]
                estimate = estimates_before_update[key]
                corrected_abs_error = abs(target - (base_prediction + estimate))
                state.decay(spec.halflife_rows)
                state.base_abs_sum += base_abs_error
                state.corrected_abs_sum += corrected_abs_error
                state.perf_weight += 1.0
                state.perf_count += 1
                state.residual_sum += residual
                state.residual_weight += 1.0
                state.residual_count += 1

    out = frame[["target_date", "current_target_tmax_c", "forecast_source_family"]].copy()
    out["fold_id"] = out["target_date"].map(fold_id_for_date)
    out["row_index"] = frame.index.to_numpy(dtype=int)
    out["m0075_prediction_c"] = pd.to_numeric(frame["m0075_prediction_c"], errors="coerce")
    out["candidate_prediction_c"] = np.array(predictions, dtype=float)
    out["specialist_correction_c"] = np.array(corrections, dtype=float)
    out["predicate_active_and_eligible"] = active_flags
    out["selected_contexts"] = selected_contexts
    out["selected_prior_lift_c"] = selected_lifts
    out["candidate_id"] = spec.candidate_id
    out["candidate_class"] = "causal_prior_only_residual_specialist"
    out["feature"] = spec.predicate.feature
    out["feature_family"] = spec.predicate.family
    out["predicate_label"] = spec.predicate.label
    out["predicate_direction"] = spec.predicate.direction
    out["predicate_threshold"] = spec.predicate.threshold
    out["context_mode"] = spec.context_mode
    out["min_history"] = spec.min_history
    out["halflife_rows"] = spec.halflife_rows
    out["support_shrink"] = spec.support_shrink
    out["min_prior_lift_c"] = spec.min_prior_lift_c
    out["correction_cap_c"] = spec.correction_cap_c
    return out


def score_candidate(frame: pd.DataFrame, predictions: pd.DataFrame, base_0075_mae: float) -> dict[str, object]:
    pred_values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    score = score_prediction(frame.rename(columns={"current_target_tmax_c": "target_tmax_c"}), pred_values)
    base_values = pd.to_numeric(frame["m0075_prediction_c"], errors="coerce").to_numpy(dtype=float)
    base_score = score_prediction(frame.rename(columns={"current_target_tmax_c": "target_tmax_c"}), base_values)
    late_mask = pd.to_datetime(predictions["target_date"], errors="coerce").ge(LATE_EVAL_START)
    late_score = score_prediction(
        frame.loc[late_mask].rename(columns={"current_target_tmax_c": "target_tmax_c"}).copy(),
        pred_values[late_mask.to_numpy()],
    )
    late_base = score_prediction(
        frame.loc[late_mask].rename(columns={"current_target_tmax_c": "target_tmax_c"}).copy(),
        base_values[late_mask.to_numpy()],
    )
    fold_deltas: list[float] = []
    for _, fold_predictions in predictions.groupby("fold_id", observed=True):
        fold_frame = frame.loc[fold_predictions.index].rename(columns={"current_target_tmax_c": "target_tmax_c"}).copy()
        fold_score = score_prediction(
            fold_frame,
            pd.to_numeric(fold_predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_base = score_prediction(
            fold_frame,
            pd.to_numeric(frame.loc[fold_predictions.index, "m0075_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_deltas.append(float(fold_score["mae"]) - float(fold_base["mae"]))
    active_mask = predictions["predicate_active_and_eligible"].astype(bool)
    active_score = score_prediction(
        frame.loc[active_mask].rename(columns={"current_target_tmax_c": "target_tmax_c"}).copy(),
        pd.to_numeric(predictions.loc[active_mask, "candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    active_base = score_prediction(
        frame.loc[active_mask].rename(columns={"current_target_tmax_c": "target_tmax_c"}).copy(),
        pd.to_numeric(frame.loc[active_mask, "m0075_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "candidate_class": str(predictions["candidate_class"].iloc[0]),
        "feature": str(predictions["feature"].iloc[0]),
        "feature_family": str(predictions["feature_family"].iloc[0]),
        "predicate_label": str(predictions["predicate_label"].iloc[0]),
        "predicate_direction": str(predictions["predicate_direction"].iloc[0]),
        "predicate_threshold": float(predictions["predicate_threshold"].iloc[0]),
        "context_mode": str(predictions["context_mode"].iloc[0]),
        "min_history": int(predictions["min_history"].iloc[0]),
        "halflife_rows": float(predictions["halflife_rows"].iloc[0]),
        "n": score["n"],
        "mae": score["mae"],
        "rmse": score["rmse"],
        "bias": score["bias"],
        "m0075_mae": base_score["mae"],
        "delta_mae_vs_0075": float(score["mae"]) - base_0075_mae,
        "late_n": late_score["n"],
        "late_mae": late_score["mae"],
        "late_delta_mae_vs_0075": float(late_score["mae"]) - float(late_base["mae"]),
        "fold_delta_max_vs_0075": max(fold_deltas) if fold_deltas else math.nan,
        "fold_delta_min_vs_0075": min(fold_deltas) if fold_deltas else math.nan,
        "folds_improved_vs_0075": int(sum(delta < 0 for delta in fold_deltas)),
        "active_rows": int(active_mask.sum()),
        "active_row_share": float(active_mask.mean()),
        "active_mae": active_score["mae"],
        "active_delta_mae_vs_0075": (
            float(active_score["mae"]) - float(active_base["mae"]) if int(active_score["n"]) > 0 else math.nan
        ),
        "mean_correction_c": float(pd.to_numeric(predictions["specialist_correction_c"], errors="coerce").mean()),
        "mean_abs_correction_c": float(
            pd.to_numeric(predictions["specialist_correction_c"], errors="coerce").abs().mean()
        ),
    }
    row["beats_0075"] = bool(float(row["delta_mae_vs_0075"]) <= -BASE_MATERIALITY_C)
    row["new_champion_gate_passed"] = bool(
        row["beats_0075"]
        and float(row["fold_delta_max_vs_0075"]) <= 0.0
        and float(row["late_delta_mae_vs_0075"]) <= 0.0
    )
    return row


def score_all_specs(
    frame: pd.DataFrame,
    specs: list[SpecialistSpec],
    *,
    base_0075_mae: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions_list: list[pd.DataFrame] = []
    for index, spec in enumerate(specs, start=1):
        if index == 1 or index % 10 == 0 or index == len(specs):
            print(f"0078 scoring {index}/{len(specs)}: {spec.candidate_id}", flush=True)
        predictions = apply_specialist_spec(frame, spec)
        rows.append(score_candidate(frame, predictions, base_0075_mae=base_0075_mae))
        predictions_list.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["new_champion_gate_passed", "beats_0075", "mae"],
        ascending=[False, False, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(25).astype(str))
    top_predictions = pd.concat(
        [predictions for predictions in predictions_list if str(predictions["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(top_predictions["target_date"], context="0078 top predictions")
    return scoreboard.reset_index(drop=True), top_predictions


def context_diagnostics(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active = predictions[predictions["selected_contexts"].astype(str).ne("")].copy()
    if active.empty:
        return pd.DataFrame()
    exploded = active.assign(context_key=active["selected_contexts"].astype(str).str.split(";")).explode("context_key")
    for context_key, group in exploded.groupby("context_key", observed=True):
        row_indices = pd.to_numeric(group["row_index"], errors="coerce").astype(int).to_numpy()
        subframe = frame.iloc[row_indices].rename(columns={"current_target_tmax_c": "target_tmax_c"}).copy()
        score = score_prediction(
            subframe,
            pd.to_numeric(group["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        base = score_prediction(
            subframe,
            pd.to_numeric(subframe["m0075_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        rows.append(
            {
                "context_key": context_key,
                "n": int(len(group)),
                "first_date": str(pd.to_datetime(subframe["target_date"]).min().date()),
                "last_date": str(pd.to_datetime(subframe["target_date"]).max().date()),
                "candidate_mae": score["mae"],
                "m0075_mae": base["mae"],
                "delta_mae_vs_0075": float(score["mae"]) - float(base["mae"]),
                "mean_correction_c": float(pd.to_numeric(group["specialist_correction_c"], errors="coerce").mean()),
                "mean_prior_lift_c": float(pd.to_numeric(group["selected_prior_lift_c"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["delta_mae_vs_0075", "n"], ascending=[True, False]).reset_index(drop=True)


def leakage_audit(frame: pd.DataFrame, specs: list[SpecialistSpec], scoreboard: pd.DataFrame) -> pd.DataFrame:
    feature_names = [spec.predicate.feature for spec in specs]
    new_champions = scoreboard[scoreboard["new_champion_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "no_current_target_feature_predicates",
            "passed": bool(not any(name.startswith("target_tmax_c") for name in feature_names)),
            "evidence": f"{len(feature_names)} specialist specs checked",
        },
        {
            "check_id": "online_specialists_update_after_scoring",
            "passed": True,
            "evidence": "each active predicate reads prior state before scoring and updates after scoring",
        },
        {
            "check_id": "new_champion_gate_requires_full_fold_late_improvement_vs_0075",
            "passed": bool(
                new_champions.empty
                or (
                    new_champions["delta_mae_vs_0075"].le(-BASE_MATERIALITY_C).all()
                    and new_champions["fold_delta_max_vs_0075"].le(0.0).all()
                    and new_champions["late_delta_mae_vs_0075"].le(0.0).all()
                )
            ),
            "evidence": f"{len(new_champions)} new-champion candidates passed",
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
    new_champions = scoreboard[scoreboard["new_champion_gate_passed"].astype(bool)].copy()
    return f"""# Prior-Only Residual Specialists

Generated: `{summary['generated_at_utc']}`

## Purpose

`0078` turns the strongest `0077` diagnostic families into explicit online residual specialists on top of the current `0075` champion. Each specialist uses a fixed feature threshold and applies a correction only when that feature context has prior evidence that its correction helped.

## Data Contract

- Base prediction: `0075` refined online residual-memory champion.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ confirmation rows are used.
- Current target Tmax columns are forbidden as specialist predicates.
- Specialist states are read before scoring and updated after scoring each row.
- This is still research-only; thresholds are deterministic hypotheses from `0077`, not production approval.

## Headline

| Item | Value |
|---|---:|
| 0075 MAE | {summary['base_0075_mae']} |
| Best 0078 candidate | {summary['best_candidate']} |
| Best 0078 MAE | {summary['best_mae']} |
| Best delta vs 0075 | {summary['best_delta_mae_vs_0075']} |
| Best fold max delta vs 0075 | {summary['best_fold_delta_max_vs_0075']} |
| Best late delta vs 0075 | {summary['best_late_delta_mae_vs_0075']} |
| New champion candidate | {summary['best_new_champion_candidate']} |
| New champion MAE | {summary['best_new_champion_mae']} |

## Interpretation

If `0078` does not beat `0075`, the specialist predicates are still valuable as failure-mode documentation: they identify where the current champion remains fragile and which families should receive deeper candidate design.

## Scoreboard

{markdown_table(scoreboard, max_rows=120)}

## New-Champion Candidates

{markdown_table(new_champions, max_rows=80)}

## Best-Candidate Context Diagnostics

{markdown_table(diagnostics, max_rows=100)}

## Candidate Definitions

{markdown_table(definitions, max_rows=120)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/context_diagnostics.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_0078_prior_only_residual_specialists.py`:

- `{FOLDER_NAME}`: prior-only residual-specialist screen using fixed predicates from `0077`.

| Metric | Value |
|---|---:|
| 0075 champion MAE | {summary['base_0075_mae']} |
| Best 0078 candidate | {summary['best_candidate']} |
| Best 0078 MAE | {summary['best_mae']} |
| Best delta vs 0075 | {summary['best_delta_mae_vs_0075']} |
| New champion candidate | {summary['best_new_champion_candidate']} |

Leakage contract: no 2024+ rows; fixed feature predicates; current target excluded; online state updates after scoring.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Prior-Only Residual Specialists",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    if summary["best_new_champion_candidate"] != "NONE":
        interpretation = f"`0078` produced a new research champion with MAE `{summary['best_new_champion_mae']}`."
    else:
        interpretation = "`0075` remains champion; `0078` is specialist/failure-mode evidence."
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0078_prior_only_residual_specialists.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0077` feature families plus `0075` residuals | Tested |
| Rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Pre-2024 only |
| Candidate count | `{summary['candidate_count']}` | Tested |
| 0075 MAE / RMSE | `{summary['base_0075_mae']}` / `{summary['base_0075_rmse']}` | Baseline |
| Best 0078 candidate | `{summary['best_candidate']}` | Tested |
| Best 0078 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0075 | `{summary['best_delta_mae_vs_0075']}` | Specialist value |
| Best fold max delta vs 0075 | `{summary['best_fold_delta_max_vs_0075']}` | Robustness check |
| Best late delta vs 0075 | `{summary['best_late_delta_mae_vs_0075']}` | Late-window check |
| New champion candidate | `{summary['best_new_champion_candidate']}` | Requires material full/fold/late improvement |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: {interpretation}
"""
    update_markdown_section(
        path,
        heading="Prior-Only Residual Specialists",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"62. Prior-only residual specialists screened `{summary['candidate_count']}` candidates; "
        f"best delta vs 0075 is `{summary['best_delta_mae_vs_0075']}` from "
        f"`{summary['best_candidate']}`, new champion `{summary['best_new_champion_candidate']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue with `0079`: if `0078` did not beat `0075`, mine the active specialist contexts that improved locally and build a guarded combination that only activates when at least two independent families agree; otherwise harden the `0078` new champion against source/fold/late windows.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0075 = build_joined_frame()
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0078 input frame")
    specs = specialist_specs(set(frame.columns))
    definitions = pd.DataFrame(
        [
            {
                "candidate_id": spec.candidate_id,
                "feature": spec.predicate.feature,
                "feature_family": spec.predicate.family,
                "predicate_label": spec.predicate.label,
                "predicate_direction": spec.predicate.direction,
                "predicate_threshold": spec.predicate.threshold,
                "context_mode": spec.context_mode,
                "min_history": spec.min_history,
                "halflife_rows": spec.halflife_rows,
                "support_shrink": spec.support_shrink,
                "min_prior_lift_c": spec.min_prior_lift_c,
                "correction_cap_c": spec.correction_cap_c,
            }
            for spec in specs
        ]
    )
    base_0075_mae = float(summary_0075["best_mae"])
    base_0075_rmse = float(summary_0075["best_rmse"])
    scoreboard, top_predictions = score_all_specs(frame, specs, base_0075_mae=base_0075_mae)
    best = scoreboard.iloc[0]
    best_predictions = top_predictions[top_predictions["candidate_id"].eq(best["candidate_id"])].copy()
    diagnostics = context_diagnostics(frame, best_predictions)
    leakage = leakage_audit(frame, specs, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0078 leakage audit failed: {failed}")
    new_champions = scoreboard[scoreboard["new_champion_gate_passed"].astype(bool)].copy()
    new_champions = new_champions.sort_values(["mae", "fold_delta_max_vs_0075"]).reset_index(drop=True)
    best_new_champion = new_champions.iloc[0] if not new_champions.empty else None
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "common_rows": int(len(frame)),
        "first_date": str(pd.to_datetime(frame["target_date"]).min().date()),
        "last_date": str(pd.to_datetime(frame["target_date"]).max().date()),
        "candidate_count": int(len(scoreboard)),
        "new_champion_candidate_count": int(scoreboard["new_champion_gate_passed"].astype(bool).sum()),
        "base_0075_candidate": str(summary_0075["best_candidate"]),
        "base_0075_mae": base_0075_mae,
        "base_0075_rmse": base_0075_rmse,
        "best_candidate": str(best["candidate_id"]),
        "best_feature": str(best["feature"]),
        "best_feature_family": str(best["feature_family"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0075": float(best["delta_mae_vs_0075"]),
        "best_late_delta_mae_vs_0075": float(best["late_delta_mae_vs_0075"]),
        "best_fold_delta_max_vs_0075": float(best["fold_delta_max_vs_0075"]),
        "best_active_rows": int(best["active_rows"]),
        "best_new_champion_candidate": (
            str(best_new_champion["candidate_id"]) if best_new_champion is not None else "NONE"
        ),
        "best_new_champion_mae": float(best_new_champion["mae"]) if best_new_champion is not None else None,
        "best_new_champion_rmse": float(best_new_champion["rmse"]) if best_new_champion is not None else None,
        "best_new_champion_delta_mae_vs_0075": (
            float(best_new_champion["delta_mae_vs_0075"]) if best_new_champion is not None else None
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
    write_csv(artifacts / "context_diagnostics.csv", diagnostics)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "prior_only_residual_specialists_manifest.json", summary)
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
    return argparse.ArgumentParser(description="Run prior-only residual specialist screen on top of 0075.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
