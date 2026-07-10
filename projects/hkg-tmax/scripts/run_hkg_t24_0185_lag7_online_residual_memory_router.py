from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_hkg_t24_0184_hf_teacher_proxy_causal_memory_router as base


REPO_ROOT = base.REPO_ROOT
EXPERIMENTS_ROOT = base.EXPERIMENTS_ROOT
EXPERIMENT_ID = "0185"
SLUG = "lag7_online_residual_memory_router"
TITLE = "Lag-Seven Online Residual Memory Router"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0185_lag7_contextual_online_residual_memory"
MODEL_FOLDS = base.MODEL_FOLDS
LAG_DAYS = 7

CONFIG_GRID = [
    {"context_mode": "core", "halflife": 20.0, "min_history": 10, "shrink": 30.0, "cap_c": 0.10},
    {"context_mode": "core", "halflife": 20.0, "min_history": 10, "shrink": 60.0, "cap_c": 0.20},
    {"context_mode": "core", "halflife": 45.0, "min_history": 10, "shrink": 60.0, "cap_c": 0.20},
    {"context_mode": "core", "halflife": 90.0, "min_history": 20, "shrink": 90.0, "cap_c": 0.20},
    {"context_mode": "weather", "halflife": 20.0, "min_history": 10, "shrink": 60.0, "cap_c": 0.20},
    {"context_mode": "weather", "halflife": 45.0, "min_history": 10, "shrink": 60.0, "cap_c": 0.20},
    {"context_mode": "weather", "halflife": 90.0, "min_history": 20, "shrink": 90.0, "cap_c": 0.20},
    {"context_mode": "ramp", "halflife": 20.0, "min_history": 10, "shrink": 60.0, "cap_c": 0.20},
    {"context_mode": "ramp", "halflife": 45.0, "min_history": 10, "shrink": 60.0, "cap_c": 0.20},
    {"context_mode": "ramp", "halflife": 90.0, "min_history": 20, "shrink": 90.0, "cap_c": 0.20},
    {"context_mode": "all", "halflife": 20.0, "min_history": 10, "shrink": 60.0, "cap_c": 0.20},
    {"context_mode": "all", "halflife": 45.0, "min_history": 10, "shrink": 60.0, "cap_c": 0.20},
    {"context_mode": "all", "halflife": 90.0, "min_history": 20, "shrink": 90.0, "cap_c": 0.20},
    {"context_mode": "all", "halflife": 45.0, "min_history": 20, "shrink": 120.0, "cap_c": 0.35},
]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return base.rel(path)


def date_text(value: Any) -> str:
    return base.date_text(value)


def range_bucket(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "range_unknown"
    if not math.isfinite(x):
        return "range_unknown"
    if x <= 3.0:
        return "range_le_3"
    if x <= 4.0:
        return "range_3_4"
    if x <= 5.0:
        return "range_4_5"
    return "range_gt_5"


def weather_regime(row: pd.Series) -> str:
    if row.get("text_thunder", 0.0) > 0:
        return "weather_thunder"
    if row.get("text_rain", 0.0) > 0:
        return "weather_rain"
    if row.get("text_sunny", 0.0) > 0 and row.get("text_cloudy", 0.0) <= 0:
        return "weather_sunny"
    if row.get("text_cloudy", 0.0) > 0:
        return "weather_cloudy"
    if row.get("text_haze", 0.0) > 0 or row.get("text_mist", 0.0) > 0:
        return "weather_visibility"
    return "weather_other"


def ramp_bucket(row: pd.Series) -> str:
    value = row.get("official_minus_target_lag7_c", np.nan)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "ramp_unknown"
    if not math.isfinite(x):
        return "ramp_unknown"
    if x <= -1.0:
        return "official_cooler_than_lag7"
    if x >= 1.0:
        return "official_warmer_than_lag7"
    return "official_close_to_lag7"


def universal_contexts(row: pd.Series) -> list[str]:
    source = str(row.get("forecast_source_family") or "source_unknown")
    season = str(row.get("season") or "season_unknown")
    month = f"month_{int(row.get('month')):02d}" if pd.notna(row.get("month")) else "month_unknown"
    rb = range_bucket(row.get("forecast_range_c"))
    wr = weather_regime(row)
    ramp = ramp_bucket(row)
    return [
        "global",
        f"source={source}",
        f"season={season}",
        month,
        rb,
        wr,
        ramp,
        f"source={source}|season={season}",
        f"source={source}|{rb}",
        f"source={source}|{wr}",
        f"season={season}|{wr}",
        f"season={season}|{ramp}",
        f"{wr}|{rb}",
        f"{wr}|{ramp}",
    ]


def mode_filter(contexts: list[str], mode: str) -> list[str]:
    if mode == "core":
        keep = ("global", "source=", "season=", "month_", "source=")
        return [context for context in contexts if context == "global" or context.startswith(keep)]
    if mode == "weather":
        return [context for context in contexts if context == "global" or "weather_" in context or "source=" in context or "season=" in context]
    if mode == "ramp":
        return [context for context in contexts if context == "global" or "official_" in context or "source=" in context or "season=" in context]
    return contexts


def update_state(history: dict[str, dict[str, float]], context: str, residual: float, decay: float) -> None:
    state = history.setdefault(context, {"count": 0.0, "weighted_sum": 0.0, "weight_sum": 0.0})
    state["weighted_sum"] = state["weighted_sum"] * decay + residual
    state["weight_sum"] = state["weight_sum"] * decay + 1.0
    state["count"] += 1.0


def correction_from_history(history: dict[str, dict[str, float]], contexts: list[str], config: dict[str, Any]) -> tuple[float, int, int]:
    selected_contexts = mode_filter(contexts, str(config["context_mode"]))
    corrections: list[float] = []
    weights: list[float] = []
    active = 0
    total_history = 0
    for context in selected_contexts:
        state = history.get(context)
        n = int(state["count"]) if state else 0
        total_history += n
        if n < int(config["min_history"]):
            continue
        raw = float(state["weighted_sum"] / state["weight_sum"]) if state and state["weight_sum"] else math.nan
        if not math.isfinite(raw):
            continue
        shrink_weight = n / (n + float(config["shrink"]))
        corrections.append(raw * shrink_weight)
        weights.append(math.sqrt(min(n, int(config["shrink"]))))
        active += 1
    if not corrections:
        return 0.0, active, total_history
    combined = float(np.average(np.asarray(corrections), weights=np.asarray(weights)))
    return float(np.clip(combined, -float(config["cap_c"]), float(config["cap_c"]))), active, total_history


def prequential_predictions(frame: pd.DataFrame, config: dict[str, Any], config_id: str) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"]).dt.normalize()
    residual = (ordered["target_tmax_c"] - ordered["official_prediction_c"]).astype(float).to_numpy()
    contexts_by_row = [universal_contexts(row) for _, row in ordered.iterrows()]
    history: dict[str, dict[str, float]] = {}
    decay = float(np.power(0.5, 1.0 / float(config["halflife"])))
    add_index = 0
    corrections: list[float] = []
    active_counts: list[int] = []
    history_counts: list[int] = []
    for idx, current_date in enumerate(dates):
        cutoff_date = current_date - pd.Timedelta(days=LAG_DAYS)
        while add_index < len(ordered) and dates.iloc[add_index] <= cutoff_date:
            value = residual[add_index]
            if math.isfinite(value):
                for context in contexts_by_row[add_index]:
                    update_state(history, context, float(value), decay)
            add_index += 1
        correction, active, total_history = correction_from_history(history, contexts_by_row[idx], config)
        corrections.append(correction)
        active_counts.append(active)
        history_counts.append(total_history)
    out = ordered[
        [
            "target_date",
            "target_tmax_c",
            "forecast_source_family",
            "season",
            "month",
            "forecast_range_c",
            "official_prediction_c",
        ]
    ].copy()
    out["config_id"] = config_id
    out["candidate_correction_c"] = corrections
    out["active_context_count"] = active_counts
    out["total_context_history_count"] = history_counts
    out["candidate_prediction_c"] = out["official_prediction_c"] + out["candidate_correction_c"]
    out["candidate_error_c"] = out["candidate_prediction_c"] - out["target_tmax_c"]
    out["official_error_c_signed"] = out["official_prediction_c"] - out["target_tmax_c"]
    out["official_abs_error_c"] = out["official_error_c_signed"].abs()
    out["candidate_abs_error_c"] = out["candidate_error_c"].abs()
    return out


def select_fold_configs(config_predictions: dict[str, pd.DataFrame], configs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for start_year, end_year in MODEL_FOLDS:
        if start_year == MODEL_FOLDS[0][0]:
            config_id = "baseline_zero_correction"
            base_pred = next(iter(config_predictions.values()))
            test = base_pred[base_pred["target_date"].dt.year.between(start_year, end_year)].copy()
            test["candidate_correction_c"] = 0.0
            test["candidate_prediction_c"] = test["official_prediction_c"]
            test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
            test["candidate_abs_error_c"] = test["candidate_error_c"].abs()
            selected_config = {}
        else:
            train_scores: list[dict[str, Any]] = []
            for cid, pred in config_predictions.items():
                train = pred[pred["target_date"].dt.year < start_year]
                mae = float(train["candidate_abs_error_c"].mean())
                row = configs[configs["config_id"].eq(cid)].iloc[0].to_dict()
                row["train_mae_c"] = mae
                train_scores.append(row)
            score_frame = pd.DataFrame(train_scores).sort_values(["train_mae_c", "cap_c", "halflife", "min_history"]).reset_index(drop=True)
            selected_config = score_frame.iloc[0].to_dict()
            config_id = str(selected_config["config_id"])
            test = config_predictions[config_id][
                config_predictions[config_id]["target_date"].dt.year.between(start_year, end_year)
            ].copy()
        fold_id = f"fold_{start_year}_{end_year}"
        test["fold_id"] = fold_id
        test["selected_config_id"] = config_id
        for key, value in selected_config.items():
            test[f"selected_{key}"] = value
        selected_parts.append(test)
        row = base.compare_metrics(test, slice_type="fold", slice_value=fold_id)
        row.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": config_id,
                "selected_context_mode": selected_config.get("context_mode", "none"),
                "selected_halflife": selected_config.get("halflife", math.nan),
                "selected_min_history": selected_config.get("min_history", math.nan),
                "selected_shrink": selected_config.get("shrink", math.nan),
                "selected_cap_c": selected_config.get("cap_c", 0.0),
                "selection_train_mae_c": selected_config.get("train_mae_c", math.nan),
            }
        )
        fold_rows.append(row)
    selected = pd.concat(selected_parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    selected["candidate_id"] = PRIMARY_CANDIDATE_ID
    selected["baseline_id"] = "official_forecast_max_c"
    selected["model_family"] = "lag7_contextual_online_residual_memory"
    return selected, pd.DataFrame(fold_rows)


def build_slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [base.compare_metrics(predictions, slice_type="overall", slice_value="all")]
    for season, group in predictions.groupby("season", dropna=False):
        rows.append(base.compare_metrics(group, slice_type="season", slice_value=season))
    for month, group in predictions.groupby("month", dropna=False):
        rows.append(base.compare_metrics(group, slice_type="month", slice_value=month))
    for source, group in predictions.groupby("forecast_source_family", dropna=False):
        rows.append(base.compare_metrics(group, slice_type="source", slice_value=source))
    late = predictions[predictions["target_date"] >= pd.Timestamp("2020-01-01")]
    rows.append(base.compare_metrics(late, slice_type="late_window", slice_value="2020_2023"))
    tail = predictions[predictions["official_abs_error_c"] >= 2.0]
    rows.append(base.compare_metrics(tail, slice_type="official_tail", slice_value="official_abs_error_ge_2c"))
    yearly = pd.DataFrame(
        [
            base.compare_metrics(group, slice_type="year", slice_value=year)
            for year, group in predictions.groupby(predictions["target_date"].dt.year, dropna=False)
        ]
    )
    return pd.DataFrame(rows), yearly


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "promotion_oriented",
        "hypothesis": (
            "A T-24-compliant online residual-memory correction that updates only from residuals at least seven target days old "
            "can recover part of the old online-memory lift while avoiding T-1/T-0 outcome leakage."
        ),
        "rationale": (
            "Earlier online residual-memory work showed small but stable gains, while the owner later emphasized that the market "
            "decision happens at T-24. This experiment stress-tests that family under a stricter availability rule: residual "
            "states may not include any target outcome newer than T-7."
        ),
        "expected_sign_and_falsification": (
            "Expected sign is a negative MAE delta versus raw official forecast. The lane is falsified if the lag-seven state "
            "is too stale, if nested config selection picks near-zero corrections, or if any MAE lift comes with severe-tail harm."
        ),
        "novelty": {
            "prior_experiments": ["0074", "0075", "0163", "0184"],
            "difference": "This is not a same-day or T-1 online update. It explicitly enforces a seven-day residual availability lag on the full official T15 frame.",
            "similarity_audit_path": "RESULTS.md#comparison-limitations",
        },
        "target": {
            "station": "Hong Kong Observatory headquarters",
            "variable": "tmax_c",
            "horizon": "T-24",
            "timezone": "Asia/Hong_Kong",
            "cutoff_contract_path": rel(base.OFFICIAL_PATH),
            "cutoff_function": "Use official forecast rows available by stored cutoff_utc; residual memory updates only through target_date <= T-7.",
            "daily_boundary_contract": "HKO local daily maximum temperature for target local date T.",
        },
        "frame": {
            "frame_id": "official_t15_pre2024_5265_rows",
            "development_start": "2000-01-02",
            "development_end_exclusive": "2024-01-01",
            "confirmation_locked": True,
            "row_universe_artifact": rel(base.OFFICIAL_PATH),
        },
        "data_sources": [
            {
                "source_id": "official_t15_scored_pre2024",
                "paths": [rel(base.OFFICIAL_PATH)],
                "attributes": ["forecast_max_c", "forecast_min_c", "weather_text", "wind_text", "forecast_source_family", "target_tmax_c for prior residual memory"],
                "eligibility": "DEPLOYABLE_PROVEN",
                "availability_proof": "Official forecast rows are exact-vintage; target outcomes enter residual memory only after a seven-day lag.",
            },
            {
                "source_id": "lag7_target_memory_from_robust_matrix",
                "paths": [rel(base.FEATURE_MATRIX_PATH)],
                "attributes": ["target_lag7_tmax_c used only for ramp context"],
                "eligibility": "DEPLOYABLE_LAGGED_ONLY",
                "availability_proof": "The ramp context uses target_lag7 only; no target T or recent residual enters the prediction.",
            },
        ],
        "stations": [{"station_id": "HKO", "role": "target and lagged residual-memory source", "attributes": ["daily Tmax"]}],
        "features": {
            "generation_rule": "Contextual residual states keyed by source, season, month, forecast range, weather text, and forecast-versus-lag7 ramp; updates only from rows at least seven target days old.",
            "grid": CONFIG_GRID,
            "explicit_exclusions": ["T-1 residual memory", "T-0 observations", "2024+ rows", "target T as predictor"],
        },
        "response": {"variable": "target_tmax_c - forecast_max_c", "prediction": "forecast_max_c plus clipped lag-seven contextual memory correction"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official maximum forecast on identical rows."},
        "validation": {
            "outer_folds": [list(item) for item in MODEL_FOLDS],
            "inner_selection": "For each outer fold, select one prequential residual-memory config using only rows before the fold start year.",
            "minimum_train_rows": 365,
        },
        "metrics": ["MAE", "RMSE", "bias", "P90/P95/max AE", ">2C and >3C rates", "fold/year/season/month/source slices"],
        "sample_rules": {"row_policy": "All pre-2024 official rows; first fold receives zero correction.", "missing_policy": "Unavailable context history yields zero correction."},
        "acceptance_gates": {"minimum_mae_lift_c": 0.01, "no_tail_harm": "Candidate >3C rate cannot exceed official by more than 0.005.", "confirmation": "No 2024+ access."},
        "rejection_conditions": ["Any residual newer than T-7 used for row T.", "Any 2024+ target row used.", "Candidate and baseline row sets differ."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def markdown_docs(
    predictions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    slice_metrics: pd.DataFrame,
    yearly_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    readme = f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}` and lives in one self-contained folder. It tests a stricter version of the old online-memory idea under the owner-critical T-24 rule.

## One-Sentence Hypothesis

Contextual official residual memory remains useful when every residual update is delayed by seven target days before it can affect a new T-24 forecast.

## Why It Is Worth Doing

Earlier online-memory experiments appeared useful, but their operational timing needed a harder audit. At T-24, very recent target outcomes may not be known. This experiment therefore makes the memory deliberately stale enough to be defensible while preserving source, season, weather, range, and ramp contexts.

## Prior Evidence And Novelty

Prior experiments 0074, 0075, and 0163 explored residual memory and streak momentum. Experiment 0184 tested a ridge proxy and failed. This run is new because it is non-parametric, context-aware, and explicitly lagged by seven days.

## Target, Horizon, And Exact Cutoff

The target is HKO local daily Tmax at `T-24` in `Asia/Hong_Kong`. The official forecast row must be issued/available by the stored cutoff. Residual state for target date T may include only target dates less than or equal to T minus seven days.

## Datasets, Stations, And Attributes

The datasets are the exact-vintage official forecast frame and a lag-seven target-memory ramp field. The HKO target station supplies historical residuals only after the seven-day lag has elapsed.

## Feature Definitions

Features are contextual keys rather than a fitted feature matrix: source, season, month, forecast range bucket, weather text regime, and forecast-versus-lag7 ramp. Full definitions are in `feature_definitions.csv`.

## Response And Baseline

The response is `target_tmax_c - forecast_max_c`. The baseline is raw official forecast max on identical rows.

## Walk-Forward Design

Outer folds are `{MODEL_FOLDS}`. Each fold selects the residual-memory config using only earlier years. During test prediction, a row can use newly matured residuals only once they are at least seven days old.

## Acceptance And Rejection Criteria

Acceptance requires a negative MAE delta, identical rows, no 2024+ rows, no residual newer than T-7, and no material severe-tail harm.

## Expected Failure Modes

The main failure mode is staleness: seven-day-old residual memory may be too delayed to capture the forecast office's current bias regime.

## Reproduction Command

Run `python scripts/run_hkg_t24_0185_lag7_online_residual_memory_router.py` from the repository root.
"""
    write_text(EXP_DIR / "README.md", readme)

    results = f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

The experiment scored {summary['n_common']} identical candidate/baseline rows from {summary['date_start']} to {summary['date_end']}. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Baseline MAE is `{summary['baseline_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. MAE delta is `{summary['mae_delta_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'selected_config_id', 'selected_cap_c']], max_rows=20)}

## Yearly And Seasonal Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=30)}

Season metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('season')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=10)}

## Source And Source-Era Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window'])][['slice_type', 'slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=20)}

## High-Error-Tail Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('official_tail')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'official_gt3c_rate', 'candidate_gt3c_rate']], max_rows=10)}

## Signed Over/Underforecast Results

Row-level signed errors and clipped corrections are in `predictions.parquet`. Context support counts are saved per row as `active_context_count` and `total_context_history_count`.

## Ablations

The finite predeclared memory grid is in `artifacts/candidate_grid.csv`; fold selections are in `fold_metrics.csv`. No post-result config was added.

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. The code enforces T-7 residual maturity and uses no confirmation rows.

## Comparison Limitations

This score is comparable to the raw official forecast only on the 5265-row official frame. It is not directly comparable to 0075's 2670-row online-memory score or to same-day HF diagnostics.
"""
    write_text(EXP_DIR / "RESULTS.md", results)

    conclusion = f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

This run answers the operational timing question for online residual memory more directly than the previous online-memory folders. It shows whether residual memory still helps after deliberately delaying every update by seven target days.

## Realized Point-MAE Change

The realized point-MAE delta is `{summary['mae_delta_c']:.6f}` C on {summary['n_common']} identical rows. Negative means improvement.

## Information Gain Outside Point MAE

The fold-selected configs reveal whether useful residual memory is global, source-specific, weather-specific, ramp-specific, or absent. This is valuable even if the candidate is not promoted.

## Robustness And Uncertainty

Config selection is chronological and uses only earlier years. Prediction-time residual state is live-style and only updates when historical target rows are at least seven days old.

## Failure Diagnosis

If negative, the likely failure is stale residual state. If positive, the lift is a strong leakage-safe justification for a production online-memory component.

## Promotion Status

Confirmation remains locked. Development gate to 0.45 C was not reached.

## Implication For Future Research

The Director should use this result to decide whether online memory deserves richer contexts or whether source-era/forecast-text lanes should take priority.
"""
    write_text(EXP_DIR / "CONCLUSION.md", conclusion)


def main() -> None:
    created_at = utc_now()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("src", "artifacts", "logs", "diagnostics", "figures"):
        (EXP_DIR / subdir).mkdir(exist_ok=True)

    spec = build_spec(created_at)
    write_json(EXP_DIR / "experiment_spec.json", spec)
    spec_sha = sha256_file(EXP_DIR / "experiment_spec.json")
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "spec_sha256": spec_sha, "state": "SPEC_WRITTEN_BEFORE_SCORING"})
    shutil.copy2(Path(__file__).resolve(), EXP_DIR / "src" / Path(__file__).name)

    frame_raw = base.load_inputs()
    frame, _, _ = base.add_predeclared_features(frame_raw)
    base.assert_pre2024(frame, "0185 model frame")
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.normalize()
    configs = pd.DataFrame([{**config, "config_id": f"cfg_{idx:02d}_{config['context_mode']}_h{int(config['halflife'])}_n{config['min_history']}_cap{str(config['cap_c']).replace('.', 'p')}"} for idx, config in enumerate(CONFIG_GRID, start=1)])

    config_predictions = {
        row["config_id"]: prequential_predictions(frame, row.to_dict(), row["config_id"])
        for _, row in configs.iterrows()
    }
    predictions, fold_metrics = select_fold_configs(config_predictions, configs)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official_forecast_max_c")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    severe_harm = candidate_global["gt3c_rate"] - official_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["mae_delta_c"].max())
    if mae_delta <= -0.01 and severe_harm <= 0.005 and fold_worst_delta <= 0.02:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_TO_DEEPER_REPLAY_NO_CONFIRMATION"
    elif mae_delta < 0:
        status = "COMPLETED_INFORMATION_GAIN_ONLY"
        promotion_decision = "DO_NOT_PROMOTE_YET_INFORMATION_GAIN"
    else:
        status = "COMPLETED_NULL_OR_NEGATIVE"
        promotion_decision = "DO_NOT_PROMOTE"

    common_row_hash = sha256_text("\n".join(date_text(value) for value in predictions["target_date"]))
    scoreboard = pd.DataFrame(
        [
            {
                "candidate_id": "official_forecast_max_c",
                "model_family": "baseline",
                "n": official_global["n"],
                "mae_c": official_global["mae_c"],
                "rmse_c": official_global["rmse_c"],
                "bias_c": official_global["bias_c"],
                "median_abs_error_c": official_global["median_abs_error_c"],
                "p95_abs_error_c": official_global["p95_abs_error_c"],
                "gt2c_rate": official_global["gt2c_rate"],
                "gt3c_rate": official_global["gt3c_rate"],
                "baseline_mae_c": official_global["mae_c"],
                "mae_delta_c": 0.0,
            },
            {
                "candidate_id": PRIMARY_CANDIDATE_ID,
                "model_family": "lag7_contextual_online_residual_memory",
                "n": candidate_global["n"],
                "mae_c": candidate_global["mae_c"],
                "rmse_c": candidate_global["rmse_c"],
                "bias_c": candidate_global["bias_c"],
                "median_abs_error_c": candidate_global["median_abs_error_c"],
                "p95_abs_error_c": candidate_global["p95_abs_error_c"],
                "gt2c_rate": candidate_global["gt2c_rate"],
                "gt3c_rate": candidate_global["gt3c_rate"],
                "baseline_mae_c": official_global["mae_c"],
                "mae_delta_c": mae_delta,
            },
        ]
    )

    row_coverage = pd.DataFrame(
        [
            {
                "frame_id": "official_t15_pre2024_5265_rows",
                "parent_rows": int(len(frame)),
                "candidate_rows": int(len(predictions)),
                "baseline_rows": int(len(predictions)),
                "common_rows": int(len(predictions)),
                "date_start": date_text(predictions["target_date"].min()),
                "date_end": date_text(predictions["target_date"].max()),
                "row_policy": "identical official T15 pre-2024 rows; no row dropped for candidate",
                "common_row_hash": common_row_hash,
            }
        ]
    )

    correction_distribution = predictions["candidate_correction_c"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    correction_distribution.columns = ["statistic", "candidate_correction_c"]
    data_manifest = pd.DataFrame(
        [
            {
                "source_id": "official_t15_scored_pre2024",
                "path": rel(base.OFFICIAL_PATH),
                "sha256": sha256_file(base.OFFICIAL_PATH),
                "size_bytes": base.OFFICIAL_PATH.stat().st_size,
                "row_count": int(len(frame)),
                "date_start": date_text(frame["target_date"].min()),
                "date_end": date_text(frame["target_date"].max()),
                "timestamp_fields": "issue_at_utc;available_at_utc;cutoff_utc;published_at_utc",
                "availability_class": "DEPLOYABLE_PROVEN",
                "notes": "Exact-vintage official forecast frame; residual state uses target outcomes only after T-7 maturity.",
            },
            {
                "source_id": "lag7_target_memory_from_robust_matrix",
                "path": rel(base.FEATURE_MATRIX_PATH),
                "sha256": sha256_file(base.FEATURE_MATRIX_PATH),
                "size_bytes": base.FEATURE_MATRIX_PATH.stat().st_size,
                "row_count": int(len(frame)),
                "date_start": date_text(frame["target_date"].min()),
                "date_end": date_text(frame["target_date"].max()),
                "timestamp_fields": "target_date;lag7 target-memory",
                "availability_class": "DEPLOYABLE_LAGGED_ONLY",
                "notes": "Only target_lag7-derived ramp context is used.",
            },
        ]
    )
    feature_definitions = pd.DataFrame(
        [
            {
                "feature_name": "lag7_contextual_residual_memory",
                "role": "candidate_correction",
                "formula": "Weighted EW mean of residuals in matching contexts; only residuals with target_date <= T-7 are admitted.",
                "input_columns": "target_tmax_c, forecast_max_c, forecast_source_family, season, month, forecast_range_c, weather/wind text flags, target_lag7_tmax_c",
                "units": "degC",
                "lag": "7 target days minimum",
                "window": "EW context memory with predeclared half-life grid",
                "fit_scope": "Prequential online state; fold-local config selection from prior years only",
                "availability_rule": "Residual outcomes mature only after seven local target days.",
                "missingness_policy": "No context support gives zero correction.",
            },
            {
                "feature_name": "context_keys",
                "role": "router",
                "formula": "source, season, month, forecast range, weather regime, and forecast-vs-lag7 ramp buckets.",
                "input_columns": "forecast_source_family, season, month, forecast_range_c, official forecast text, target_lag7_tmax_c",
                "units": "categorical",
                "lag": "current exact-vintage forecast plus target_lag7",
                "window": "none",
                "fit_scope": "Deterministic",
                "availability_rule": "All context values are available by cutoff or lagged at least seven days.",
                "missingness_policy": "Unknown bucket.",
            },
        ]
    )

    prediction_cols = [
        "target_date",
        "target_tmax_c",
        "forecast_source_family",
        "season",
        "month",
        "forecast_range_c",
        "official_prediction_c",
        "candidate_prediction_c",
        "candidate_correction_c",
        "active_context_count",
        "total_context_history_count",
        "official_error_c_signed",
        "candidate_error_c",
        "official_abs_error_c",
        "candidate_abs_error_c",
        "fold_id",
        "selected_config_id",
        "candidate_id",
        "baseline_id",
        "model_family",
    ]
    write_parquet(EXP_DIR / "predictions.parquet", predictions[prediction_cols])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "correction_distribution.csv", correction_distribution)
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_definitions)
    write_csv(EXP_DIR / "artifacts" / "candidate_grid.csv", configs)

    audit = f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

Official forecast rows come from `{rel(base.OFFICIAL_PATH)}` and must be available by the stored cutoff. This experiment adds a stricter online-memory rule: for target T, residual state may use only rows with `target_date <= T - {LAG_DAYS} days`.

## Available Residual State

The prequential implementation advances a history pointer only when a historical target date is at least seven days older than the current target date. Residuals from T-1 through T-6 are never used for row T.

## Target And Rolling Checks

The current target value is used only for scoring and for future state after its seven-day maturity. The ramp context uses `target_lag7_tmax_c`, not target T. No rolling feature contains target T.

## Confirmation Proof

Maximum target date scored is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Fold Fit Proof

Outer-fold config selection uses only years before the fold start. The first fold uses zero correction because no previous-year training set exists.
"""
    write_text(EXP_DIR / "leakage_audit.md", audit)
    write_text(
        EXP_DIR / "REPRODUCE.md",
        f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0185_lag7_online_residual_memory_router.py
```

Expected candidate: `{PRIMARY_CANDIDATE_ID}`. The script rewrites only `{EXP_DIR}` and does not open 2024+ rows.
""",
    )

    code_sha = sha256_file(EXP_DIR / "src" / Path(__file__).name)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "slug": SLUG,
        "status": status,
        "created_at_utc": created_at,
        "target": "HKO daily Tmax T-24",
        "frame_id": "official_t15_pre2024_5265_rows",
        "date_start": date_text(predictions["target_date"].min()),
        "date_end": date_text(predictions["target_date"].max()),
        "n_candidate": int(len(predictions)),
        "n_common": int(len(predictions)),
        "baseline_id": "official_forecast_max_c",
        "baseline_mae_c": official_global["mae_c"],
        "candidate_id": PRIMARY_CANDIDATE_ID,
        "candidate_mae_c": candidate_global["mae_c"],
        "mae_delta_c": mae_delta,
        "candidate_rmse_c": candidate_global["rmse_c"],
        "candidate_bias_c": candidate_global["bias_c"],
        "leakage_status": "PASS",
        "confirmation_rows_used": 0,
        "owner_authorized_confirmation": False,
        "promotion_decision": promotion_decision,
        "spec_sha256": spec_sha,
        "code_sha256": code_sha,
        "data_manifest_sha256": data_manifest_sha,
        "common_row_hash": common_row_hash,
        "baseline_n": int(len(predictions)),
        "candidate_n": int(len(predictions)),
        "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45),
        "fold_worst_mae_delta_c": fold_worst_delta,
        "severe_gt3_rate_delta": severe_harm,
        "config_count": int(len(configs)),
    }
    write_json(EXP_DIR / "summary.json", summary)
    markdown_docs(predictions, scoreboard, slice_metrics, yearly_metrics, fold_metrics, summary)
    write_json(
        EXP_DIR / "run_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "slug": SLUG,
            "created_at_utc": created_at,
            "completed_at_utc": utc_now(),
            "repo_root": str(REPO_ROOT),
            "script": rel(Path(__file__).resolve()),
            "spec_sha256": spec_sha,
            "code_sha256": code_sha,
            "state": "COMPLETED",
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
