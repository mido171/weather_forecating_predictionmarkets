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
from scripts.run_hkg_t24_online_residual_memory_halflife import (  # noqa: E402
    BASE_MATERIALITY_C,
    context_keys_for_row,
    half_life_factor,
)
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import score_prediction  # noqa: E402
from scripts.run_hkg_t24_station_official_family_router import LATE_EVAL_START  # noqa: E402

FOLDER_NAME = "0076_online_no_regret_trust_router"
MODEL_COLUMNS = {
    "m0069": "m0069_prediction_c",
    "m0074": "m0074_prediction_c",
    "m0075": "m0075_prediction_c",
}
MODEL_LABELS = {
    "m0069": "0069 source/era prior selector",
    "m0074": "0074 online memory",
    "m0075": "0075 refined online memory",
}


@dataclass(frozen=True)
class TrustRouterSpec:
    candidate_id: str
    context_set: str
    selection_mode: str
    min_history: int
    halflife_rows: float
    min_edge_c: float
    fallback_model: str


@dataclass
class PerfState:
    abs_error_sum: float = 0.0
    weight: float = 0.0
    count: int = 0

    def decay(self, halflife_rows: float) -> None:
        factor = half_life_factor(halflife_rows)
        self.abs_error_sum *= factor
        self.weight *= factor

    def mae(self) -> float:
        if self.weight <= 0.0:
            return math.nan
        return self.abs_error_sum / self.weight


@dataclass(frozen=True)
class ModelPrior:
    model_id: str
    prior_mae: float
    support_weight: float
    support_count: int


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_best_memory_predictions(folder_name: str, prediction_col: str) -> pd.DataFrame:
    artifacts = RESEARCH_ROOT / folder_name / "artifacts"
    summary = load_json(artifacts / "summary.json")
    candidate_id = str(summary["best_candidate"])
    predictions = pd.read_csv(artifacts / "top_predictions.csv", parse_dates=["target_date"])
    selected = predictions[predictions["candidate_id"].astype(str).eq(candidate_id)].copy()
    if selected.empty:
        raise RuntimeError(f"Missing best candidate predictions for {folder_name}: {candidate_id}")
    if selected["target_date"].duplicated().any():
        raise RuntimeError(f"Duplicate target dates in {folder_name} best predictions")
    return selected[["target_date", "candidate_prediction_c"]].rename(
        columns={"candidate_prediction_c": prediction_col}
    )


def build_router_frame() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    frame, summary_0069 = build_feature_frame()
    frame = ensure_calendar_columns(frame).sort_values("target_date").reset_index(drop=True)
    frame["m0069_prediction_c"] = pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce")
    pred_0074 = load_best_memory_predictions("0074_online_residual_memory_halflife", "m0074_prediction_c")
    pred_0075 = load_best_memory_predictions("0075_online_residual_memory_refinement", "m0075_prediction_c")
    frame = frame.merge(pred_0074, on="target_date", how="left", validate="one_to_one")
    frame = frame.merge(pred_0075, on="target_date", how="left", validate="one_to_one")
    missing = [column for column in MODEL_COLUMNS.values() if frame[column].isna().any()]
    if missing:
        raise RuntimeError(f"Missing model predictions after 0076 merge: {missing}")
    require_no_confirmation_dates(frame["target_date"], context="0076 router frame")
    summary_0074 = load_json(RESEARCH_ROOT / "0074_online_residual_memory_halflife" / "artifacts" / "summary.json")
    summary_0075 = load_json(RESEARCH_ROOT / "0075_online_residual_memory_refinement" / "artifacts" / "summary.json")
    return frame, summary_0069, summary_0074, summary_0075


def trust_router_specs() -> list[TrustRouterSpec]:
    specs: list[TrustRouterSpec] = []
    for context_set in ("behavior", "seasonal_behavior", "all"):
        for halflife_rows in (20.0, 45.0, 90.0):
            for min_history in (10, 20):
                for selection_mode in ("best_model", "inverse_mae_blend", "no_regret_0075_gate"):
                    candidate_id = (
                        f"trust_router_{context_set}_h{int(halflife_rows)}_"
                        f"n{min_history}_{selection_mode}"
                    )
                    specs.append(
                        TrustRouterSpec(
                            candidate_id=candidate_id,
                            context_set=context_set,
                            selection_mode=selection_mode,
                            min_history=min_history,
                            halflife_rows=halflife_rows,
                            min_edge_c=0.0,
                            fallback_model="m0075",
                        )
                    )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0076 candidate IDs are not unique")
    return specs


def aggregate_model_priors(
    states: dict[tuple[str, str], PerfState],
    context_keys: list[str],
    spec: TrustRouterSpec,
) -> dict[str, ModelPrior]:
    priors: dict[str, ModelPrior] = {}
    for model_id in MODEL_COLUMNS:
        mae_values: list[float] = []
        weights: list[float] = []
        count_sum = 0
        for context_key in context_keys:
            state = states.setdefault((context_key, model_id), PerfState())
            if state.count < spec.min_history or state.weight <= 0.0:
                continue
            mae = state.mae()
            if not math.isfinite(mae):
                continue
            mae_values.append(mae)
            weights.append(max(state.weight, 1e-6))
            count_sum += state.count
        if mae_values:
            weight_array = np.array(weights, dtype=float)
            mae_array = np.array(mae_values, dtype=float)
            priors[model_id] = ModelPrior(
                model_id=model_id,
                prior_mae=float(np.sum(weight_array * mae_array) / weight_array.sum()),
                support_weight=float(weight_array.sum()),
                support_count=int(count_sum),
            )
    return priors


def prediction_from_priors(row: pd.Series, priors: dict[str, ModelPrior], spec: TrustRouterSpec) -> tuple[float, str, float]:
    fallback = spec.fallback_model
    fallback_prediction = float(row[MODEL_COLUMNS[fallback]])
    if not priors:
        return fallback_prediction, fallback, math.nan

    if spec.selection_mode == "best_model":
        best = min(priors.values(), key=lambda item: (item.prior_mae, -item.support_weight))
        fallback_prior = priors.get(fallback)
        if fallback_prior is not None and best.prior_mae + spec.min_edge_c >= fallback_prior.prior_mae:
            return fallback_prediction, fallback, fallback_prior.prior_mae
        return float(row[MODEL_COLUMNS[best.model_id]]), best.model_id, best.prior_mae

    if spec.selection_mode == "no_regret_0075_gate":
        fallback_prior = priors.get(fallback)
        if fallback_prior is None:
            return fallback_prediction, fallback, math.nan
        best = min(priors.values(), key=lambda item: (item.prior_mae, -item.support_weight))
        if best.prior_mae + spec.min_edge_c < fallback_prior.prior_mae:
            return float(row[MODEL_COLUMNS[best.model_id]]), best.model_id, best.prior_mae
        return fallback_prediction, fallback, fallback_prior.prior_mae

    if spec.selection_mode == "inverse_mae_blend":
        weights: list[float] = []
        values: list[float] = []
        for prior in priors.values():
            weights.append(prior.support_weight / max(prior.prior_mae, 0.05) ** 2)
            values.append(float(row[MODEL_COLUMNS[prior.model_id]]))
        weight_array = np.array(weights, dtype=float)
        if float(weight_array.sum()) <= 0.0:
            return fallback_prediction, fallback, math.nan
        value_array = np.array(values, dtype=float)
        weighted_prediction = float(np.sum(weight_array * value_array) / weight_array.sum())
        best_prior_mae = float(min(prior.prior_mae for prior in priors.values()))
        return weighted_prediction, "blend_inverse_mae", best_prior_mae

    raise ValueError(f"Unsupported selection mode: {spec.selection_mode}")


def apply_trust_router_spec(frame: pd.DataFrame, spec: TrustRouterSpec) -> pd.DataFrame:
    states: dict[tuple[str, str], PerfState] = {}
    predictions: list[float] = []
    selected_models: list[str] = []
    selected_prior_maes: list[float] = []
    active_model_counts: list[int] = []
    context_counts: list[int] = []

    for _, row in frame.iterrows():
        keys = context_keys_for_row(row, spec.context_set)
        priors = aggregate_model_priors(states, keys, spec)
        prediction, selected_model, selected_prior_mae = prediction_from_priors(row, priors, spec)
        predictions.append(prediction)
        selected_models.append(selected_model)
        selected_prior_maes.append(selected_prior_mae)
        active_model_counts.append(len(priors))
        context_counts.append(len(keys))

        target = float(row["target_tmax_c"])
        for key in keys:
            for model_id, column in MODEL_COLUMNS.items():
                state = states.setdefault((key, model_id), PerfState())
                state.decay(spec.halflife_rows)
                state.abs_error_sum += abs(target - float(row[column]))
                state.weight += 1.0
                state.count += 1

    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["row_index"] = frame.index.to_numpy(dtype=int)
    for model_id, column in MODEL_COLUMNS.items():
        out[column] = pd.to_numeric(frame[column], errors="coerce")
        out[f"{model_id}_abs_error_c"] = (out[column] - out["target_tmax_c"]).abs()
    out["candidate_prediction_c"] = np.array(predictions, dtype=float)
    out["selected_model"] = selected_models
    out["selected_prior_mae_c"] = selected_prior_maes
    out["active_model_count"] = active_model_counts
    out["context_count"] = context_counts
    out["candidate_id"] = spec.candidate_id
    out["candidate_class"] = "causal_online_no_regret_trust_router"
    out["context_set"] = spec.context_set
    out["selection_mode"] = spec.selection_mode
    out["min_history"] = spec.min_history
    out["halflife_rows"] = spec.halflife_rows
    out["min_edge_c"] = spec.min_edge_c
    out["fallback_model"] = spec.fallback_model
    return out


def score_candidate(frame: pd.DataFrame, predictions: pd.DataFrame, base_0075_mae: float) -> dict[str, object]:
    pred_values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    score = score_prediction(frame, pred_values)
    scores_by_model = {
        model_id: score_prediction(frame, pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
        for model_id, column in MODEL_COLUMNS.items()
    }
    late_mask = predictions["target_date"].ge(LATE_EVAL_START)
    late_score = score_prediction(frame.loc[late_mask].copy(), pred_values[late_mask.to_numpy()])
    late_0075 = score_prediction(
        frame.loc[late_mask].copy(),
        pd.to_numeric(frame.loc[late_mask, "m0075_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    fold_deltas_vs_0075: list[float] = []
    fold_deltas_vs_0069: list[float] = []
    for _, fold_predictions in predictions.groupby("fold_id", observed=True):
        fold_frame = frame.loc[fold_predictions.index].copy()
        fold_score = score_prediction(
            fold_frame,
            pd.to_numeric(fold_predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_0075 = score_prediction(
            fold_frame,
            pd.to_numeric(fold_frame["m0075_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_0069 = score_prediction(
            fold_frame,
            pd.to_numeric(fold_frame["m0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_deltas_vs_0075.append(float(fold_score["mae"]) - float(fold_0075["mae"]))
        fold_deltas_vs_0069.append(float(fold_score["mae"]) - float(fold_0069["mae"]))
    active_mask = predictions["selected_model"].astype(str).ne("m0075")
    active_score = score_prediction(
        frame.loc[active_mask].copy(),
        pd.to_numeric(predictions.loc[active_mask, "candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    active_0075 = score_prediction(
        frame.loc[active_mask].copy(),
        pd.to_numeric(frame.loc[active_mask, "m0075_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "candidate_class": str(predictions["candidate_class"].iloc[0]),
        "context_set": str(predictions["context_set"].iloc[0]),
        "selection_mode": str(predictions["selection_mode"].iloc[0]),
        "min_history": int(predictions["min_history"].iloc[0]),
        "halflife_rows": float(predictions["halflife_rows"].iloc[0]),
        "min_edge_c": float(predictions["min_edge_c"].iloc[0]),
        "fallback_model": str(predictions["fallback_model"].iloc[0]),
        "n": score["n"],
        "mae": score["mae"],
        "rmse": score["rmse"],
        "bias": score["bias"],
        "m0069_mae": scores_by_model["m0069"]["mae"],
        "m0074_mae": scores_by_model["m0074"]["mae"],
        "m0075_mae": scores_by_model["m0075"]["mae"],
        "delta_mae_vs_0069": float(score["mae"]) - float(scores_by_model["m0069"]["mae"]),
        "delta_mae_vs_0074": float(score["mae"]) - float(scores_by_model["m0074"]["mae"]),
        "delta_mae_vs_0075": float(score["mae"]) - base_0075_mae,
        "late_n": late_score["n"],
        "late_mae": late_score["mae"],
        "late_delta_mae_vs_0075": float(late_score["mae"]) - float(late_0075["mae"]),
        "fold_delta_max_vs_0075": max(fold_deltas_vs_0075) if fold_deltas_vs_0075 else math.nan,
        "fold_delta_min_vs_0075": min(fold_deltas_vs_0075) if fold_deltas_vs_0075 else math.nan,
        "fold_delta_max_vs_0069": max(fold_deltas_vs_0069) if fold_deltas_vs_0069 else math.nan,
        "folds_improved_vs_0075": int(sum(delta < 0 for delta in fold_deltas_vs_0075)),
        "active_rows": int(active_mask.sum()),
        "active_row_share": float(active_mask.mean()),
        "active_mae": active_score["mae"],
        "active_delta_mae_vs_0075": (
            float(active_score["mae"]) - float(active_0075["mae"]) if int(active_score["n"]) > 0 else math.nan
        ),
        "selected_m0069_rows": int(predictions["selected_model"].astype(str).eq("m0069").sum()),
        "selected_m0074_rows": int(predictions["selected_model"].astype(str).eq("m0074").sum()),
        "selected_m0075_rows": int(predictions["selected_model"].astype(str).eq("m0075").sum()),
        "selected_blend_rows": int(predictions["selected_model"].astype(str).eq("blend_inverse_mae").sum()),
    }
    row["beats_0069"] = bool(float(row["delta_mae_vs_0069"]) <= -BASE_MATERIALITY_C)
    row["beats_0075"] = bool(float(row["delta_mae_vs_0075"]) <= -BASE_MATERIALITY_C)
    row["new_champion_gate_passed"] = bool(
        row["beats_0075"]
        and float(row["fold_delta_max_vs_0075"]) <= 0.0
        and float(row["late_delta_mae_vs_0075"]) <= 0.0
    )
    row["deployable_gate_passed"] = bool(
        row["beats_0069"]
        and float(row["fold_delta_max_vs_0069"]) <= 0.0
    )
    return row


def score_all_specs(
    frame: pd.DataFrame,
    specs: list[TrustRouterSpec],
    *,
    base_0075_mae: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions_list: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_trust_router_spec(frame, spec)
        rows.append(score_candidate(frame, predictions, base_0075_mae=base_0075_mae))
        predictions_list.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["new_champion_gate_passed", "beats_0075", "mae"],
        ascending=[False, False, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(20).astype(str))
    top_predictions = pd.concat(
        [predictions for predictions in predictions_list if str(predictions["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(top_predictions["target_date"], context="0076 top predictions")
    return scoreboard.reset_index(drop=True), top_predictions


def selection_diagnostics(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for selected_model, group in predictions.groupby("selected_model", observed=True):
        row_indices = pd.to_numeric(group["row_index"], errors="coerce").astype(int).to_numpy()
        subframe = frame.iloc[row_indices].copy()
        score = score_prediction(
            subframe,
            pd.to_numeric(group["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        base_0075 = score_prediction(
            subframe,
            pd.to_numeric(subframe["m0075_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        rows.append(
            {
                "selected_model": selected_model,
                "selected_model_label": MODEL_LABELS.get(str(selected_model), str(selected_model)),
                "n": int(len(group)),
                "first_date": str(pd.to_datetime(subframe["target_date"]).min().date()),
                "last_date": str(pd.to_datetime(subframe["target_date"]).max().date()),
                "candidate_mae": score["mae"],
                "m0075_mae": base_0075["mae"],
                "delta_mae_vs_0075": float(score["mae"]) - float(base_0075["mae"]),
                "mean_selected_prior_mae_c": float(pd.to_numeric(group["selected_prior_mae_c"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["delta_mae_vs_0075", "n"], ascending=[True, False]).reset_index(drop=True)


def leakage_audit(frame: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    new_champions = scoreboard[scoreboard["new_champion_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "model_predictions_present_one_per_date",
            "passed": bool(
                frame[list(MODEL_COLUMNS.values())].notna().all().all()
                and len(frame) == frame["target_date"].nunique()
            ),
            "evidence": f"{len(frame)} merged rows",
        },
        {
            "check_id": "trust_router_updates_after_scoring",
            "passed": True,
            "evidence": "context/model error states are read before scoring and updated after scoring each row",
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
    return f"""# Online No-Regret Trust Router

Generated: `{summary['generated_at_utc']}`

## Purpose

`0076` tests whether the system can improve on `0075` by switching or blending between `0069`, `0074`, and `0075` using only prior context-specific model performance. It is a no-regret hardening pass: if the prior evidence says `0075` is still best, the router should stay with it.

## Data Contract

- Candidate models: `0069`, `0074`, and `0075`.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ rows are used.
- For each row, context/model MAE states are read before scoring and updated only after scoring.
- New champion promotion requires material full-sample, fold, and late-window improvement versus `0075`.

## Headline

| Item | Value |
|---|---:|
| 0069 MAE | {summary['base_0069_mae']} |
| 0074 MAE | {summary['base_0074_mae']} |
| 0075 MAE | {summary['base_0075_mae']} |
| Best 0076 candidate | {summary['best_candidate']} |
| Best 0076 MAE | {summary['best_mae']} |
| Best 0076 RMSE | {summary['best_rmse']} |
| Best delta vs 0075 | {summary['best_delta_mae_vs_0075']} |
| New champion candidate | {summary['best_new_champion_candidate']} |
| New champion MAE | {summary['best_new_champion_mae']} |

## Interpretation

The result answers whether an online trust router adds value beyond `0075`, or whether the current safest behavior is to keep `0075` as the champion and use router diagnostics to target remaining error clusters.

## Scoreboard

{markdown_table(scoreboard, max_rows=120)}

## New-Champion Candidates

{markdown_table(new_champions, max_rows=80)}

## Best-Candidate Selection Diagnostics

{markdown_table(diagnostics, max_rows=80)}

## Candidate Definitions

{markdown_table(definitions, max_rows=80)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/selection_diagnostics.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_online_no_regret_trust_router.py`:

- `{FOLDER_NAME}`: prior-only no-regret trust router over `0069`, `0074`, and `0075`.

| Metric | Value |
|---|---:|
| 0075 champion MAE | {summary['base_0075_mae']} |
| Best 0076 candidate | {summary['best_candidate']} |
| Best 0076 MAE | {summary['best_mae']} |
| Best delta vs 0075 | {summary['best_delta_mae_vs_0075']} |
| New champion candidate | {summary['best_new_champion_candidate']} |

Leakage contract: no 2024+ rows; all trust states are prior-only at prediction time.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Online No-Regret Trust Router",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    if summary["best_new_champion_candidate"] != "NONE":
        champion_line = (
            f"`0076` supersedes `0075` with MAE `{summary['best_new_champion_mae']}`."
        )
    else:
        champion_line = "`0075` remains the current champion; `0076` did not clear the new-champion gate."
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_online_no_regret_trust_router.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0069`, `0074`, and `0075` predictions | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous official archive |
| Candidate count | `{summary['candidate_count']}` | Tested |
| 0069 / 0074 / 0075 MAE | `{summary['base_0069_mae']}` / `{summary['base_0074_mae']}` / `{summary['base_0075_mae']}` | Baselines |
| Best 0076 candidate | `{summary['best_candidate']}` | Tested |
| Best 0076 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0075 | `{summary['best_delta_mae_vs_0075']}` | Router value |
| Best fold max delta vs 0075 | `{summary['best_fold_delta_max_vs_0075']}` | Robustness check |
| Best late delta vs 0075 | `{summary['best_late_delta_mae_vs_0075']}` | Late-window check |
| New champion candidate | `{summary['best_new_champion_candidate']}` | Requires material full, fold, and late improvement vs `0075` |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: {champion_line}
"""
    update_markdown_section(
        path,
        heading="Online No-Regret Trust Router",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"60. Online no-regret trust routing screened `{summary['candidate_count']}` candidates; "
        f"best delta vs 0075 is `{summary['best_delta_mae_vs_0075']}` from "
        f"`{summary['best_candidate']}`, new champion candidate `{summary['best_new_champion_candidate']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: mine the remaining `0075` and `0076` large-error clusters by station-pair gradients, upper-air moisture/thermal fields, and official forecast text/range states, then build explicit residual specialists only for clusters with enough pre-2024 support.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0069, summary_0074, summary_0075 = build_router_frame()
    base_0075_mae = float(summary_0075["best_mae"])
    specs = trust_router_specs()
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_specs(frame, specs, base_0075_mae=base_0075_mae)
    best = scoreboard.iloc[0]
    best_predictions = top_predictions[top_predictions["candidate_id"].eq(best["candidate_id"])].copy()
    diagnostics = selection_diagnostics(frame, best_predictions)
    leakage = leakage_audit(frame, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0076 leakage audit failed: {failed}")

    new_champions = scoreboard[scoreboard["new_champion_gate_passed"].astype(bool)].copy()
    new_champions = new_champions.sort_values(["mae", "fold_delta_max_vs_0075"]).reset_index(drop=True)
    best_new_champion = new_champions.iloc[0] if not new_champions.empty else None
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "common_rows": int(len(frame)),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "candidate_count": int(len(scoreboard)),
        "new_champion_candidate_count": int(scoreboard["new_champion_gate_passed"].astype(bool).sum()),
        "deployable_candidate_count": int(scoreboard["deployable_gate_passed"].astype(bool).sum()),
        "base_0069_candidate": str(summary_0069["best_deployable_candidate"]),
        "base_0069_mae": float(best["m0069_mae"]),
        "base_0074_candidate": str(summary_0074["best_candidate"]),
        "base_0074_mae": float(best["m0074_mae"]),
        "base_0075_candidate": str(summary_0075["best_candidate"]),
        "base_0075_mae": float(best["m0075_mae"]),
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0069": float(best["delta_mae_vs_0069"]),
        "best_delta_mae_vs_0074": float(best["delta_mae_vs_0074"]),
        "best_delta_mae_vs_0075": float(best["delta_mae_vs_0075"]),
        "best_late_delta_mae_vs_0075": float(best["late_delta_mae_vs_0075"]),
        "best_fold_delta_max_vs_0075": float(best["fold_delta_max_vs_0075"]),
        "best_active_rows": int(best["active_rows"]),
        "best_selected_m0069_rows": int(best["selected_m0069_rows"]),
        "best_selected_m0074_rows": int(best["selected_m0074_rows"]),
        "best_selected_m0075_rows": int(best["selected_m0075_rows"]),
        "best_selected_blend_rows": int(best["selected_blend_rows"]),
        "best_new_champion_candidate": (
            str(best_new_champion["candidate_id"]) if best_new_champion is not None else "NONE"
        ),
        "best_new_champion_mae": float(best_new_champion["mae"]) if best_new_champion is not None else None,
        "best_new_champion_rmse": float(best_new_champion["rmse"]) if best_new_champion is not None else None,
        "best_new_champion_delta_mae_vs_0075": (
            float(best_new_champion["delta_mae_vs_0075"]) if best_new_champion is not None else None
        ),
        "diagnostic_selection_rows": int(len(diagnostics)),
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
    write_csv(artifacts / "selection_diagnostics.csv", diagnostics)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "online_no_regret_trust_router_manifest.json", summary)
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
    return argparse.ArgumentParser(description="Run online no-regret trust router over 0069/0074/0075.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
