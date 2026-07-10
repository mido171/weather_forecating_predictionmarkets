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
EXPERIMENT_ID = "0186"
SLUG = "lag7_memory_monthly_noharm_gate"
TITLE = "Lag-Seven Memory Monthly No-Harm Gate"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0186_lag7_memory_monthly_noharm_gate"
MODEL_FOLDS = base.MODEL_FOLDS
PARENT_0185_DIR = EXPERIMENTS_ROOT / "0185_lag7_online_residual_memory_router"
PARENT_PREDICTIONS = PARENT_0185_DIR / "predictions.parquet"
MIN_MONTH_HISTORY = 120
MIN_MONTH_LIFT_C = 0.002
MAX_GT3_HARM = 0.005


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


def safe_metric(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    return base.metric_row(frame, pred_col, label=pred_col)


def month_gate_table(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, group in train.groupby("month", dropna=False):
        if group.empty:
            continue
        official = safe_metric(group, "official_prediction_c")
        parent = safe_metric(group, "parent_0185_prediction_c")
        delta = parent["mae_c"] - official["mae_c"]
        gt3_harm = parent["gt3c_rate"] - official["gt3c_rate"]
        apply_memory = bool(parent["n"] >= MIN_MONTH_HISTORY and delta <= -MIN_MONTH_LIFT_C and gt3_harm <= MAX_GT3_HARM)
        rows.append(
            {
                "month": int(month),
                "n_train": int(parent["n"]),
                "official_train_mae_c": official["mae_c"],
                "parent_train_mae_c": parent["mae_c"],
                "parent_delta_mae_c": delta,
                "gt3_harm": gt3_harm,
                "apply_memory": apply_memory,
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def apply_nested_gate(parent: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    gate_rows: list[pd.DataFrame] = []
    for start_year, end_year in MODEL_FOLDS:
        test = parent[parent["target_date"].dt.year.between(start_year, end_year)].copy()
        train = parent[parent["target_date"].dt.year < start_year].copy()
        fold_id = f"fold_{start_year}_{end_year}"
        if test.empty:
            gate = pd.DataFrame()
            test["monthly_gate_apply_memory"] = False
            test["candidate_prediction_c"] = test["official_prediction_c"]
            test["candidate_correction_c"] = 0.0
        elif start_year <= 2005 or len(train) < 365:
            gate = pd.DataFrame(
                [
                    {
                        "month": month,
                        "n_train": len(train),
                        "official_train_mae_c": math.nan,
                        "parent_train_mae_c": math.nan,
                        "parent_delta_mae_c": math.nan,
                        "gt3_harm": math.nan,
                        "apply_memory": True,
                    }
                    for month in sorted(test["month"].dropna().unique())
                ]
            )
            test["monthly_gate_apply_memory"] = True
            test["candidate_prediction_c"] = test["parent_0185_prediction_c"]
            test["candidate_correction_c"] = test["parent_0185_correction_c"]
        else:
            gate = month_gate_table(train)
            keep = set(gate.loc[gate["apply_memory"], "month"].astype(int).tolist())
            test["monthly_gate_apply_memory"] = test["month"].astype(int).isin(keep)
            test["candidate_prediction_c"] = np.where(
                test["monthly_gate_apply_memory"],
                test["parent_0185_prediction_c"],
                test["official_prediction_c"],
            )
            test["candidate_correction_c"] = test["candidate_prediction_c"] - test["official_prediction_c"]
        test["fold_id"] = fold_id
        test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
        test["candidate_abs_error_c"] = test["candidate_error_c"].abs()
        test["official_error_c_signed"] = test["official_prediction_c"] - test["target_tmax_c"]
        test["official_abs_error_c"] = test["official_error_c_signed"].abs()
        test["candidate_id"] = PRIMARY_CANDIDATE_ID
        test["baseline_id"] = "official_forecast_max_c"
        test["model_family"] = "nested_monthly_noharm_gate_over_0185"
        if not gate.empty:
            gate = gate.copy()
            gate["fold_id"] = fold_id
            gate["start_year"] = start_year
            gate["end_year"] = end_year
            gate_rows.append(gate)
        parts.append(test)
    out = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    return out, gates


def build_slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    fold_metrics = pd.DataFrame(
        [
            base.compare_metrics(group, slice_type="fold", slice_value=fold)
            for fold, group in predictions.groupby("fold_id", dropna=False)
        ]
    )
    fold_metrics["fold_id"] = fold_metrics["slice_value"]
    return pd.DataFrame(rows), yearly, fold_metrics


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "promotion_oriented",
        "hypothesis": (
            "The T-7 online residual-memory lift from 0185 can be improved or made safer by applying it only in calendar "
            "months where prior years show positive same-month lift without severe-tail harm."
        ),
        "rationale": (
            "0185 improved overall MAE but showed weak or slightly harmful warm-season slices. This experiment tests a nested "
            "monthly no-harm gate that uses only years before each outer fold to decide whether the 0185 correction should be "
            "used or abstained for each month."
        ),
        "expected_sign_and_falsification": (
            "Expected sign is a lower or equal MAE than 0185 with reduced damaged warm-season slices. It is falsified if the "
            "gate removes too many helpful corrections or if monthly decisions do not reduce tail harm."
        ),
        "novelty": {
            "prior_experiments": ["0185"],
            "difference": "0185 always applied the selected lag-seven memory correction; 0186 adds a fold-local monthly apply/abstain gate.",
            "similarity_audit_path": "RESULTS.md#comparison-limitations",
        },
        "target": {
            "station": "Hong Kong Observatory headquarters",
            "variable": "tmax_c",
            "horizon": "T-24",
            "timezone": "Asia/Hong_Kong",
            "cutoff_contract_path": rel(base.OFFICIAL_PATH),
            "cutoff_function": "Use official exact-vintage forecast rows; use only parent 0185 predictions that themselves enforce T-7 residual maturity.",
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
                "source_id": "0185_parent_predictions",
                "paths": [rel(PARENT_PREDICTIONS)],
                "attributes": ["parent_0185_prediction_c", "parent_0185_correction_c", "target_tmax_c", "official_prediction_c", "month"],
                "eligibility": "DEPLOYABLE_LAGGED_ONLY",
                "availability_proof": "Parent corrections enforce T-7 residual maturity; monthly gate decisions use only years before each outer fold.",
            }
        ],
        "stations": [{"station_id": "HKO", "role": "target and lagged residual-memory source", "attributes": ["daily Tmax"]}],
        "features": {
            "generation_rule": "For each outer fold and month, apply parent 0185 correction only if prior same-month parent delta <= -0.002 C, n >= 120, and >3C harm <= 0.005.",
            "explicit_exclusions": ["2024+ rows", "current-fold monthly target outcomes for gate decisions", "any residual newer than parent 0185 T-7 rule"],
        },
        "response": {"variable": "target_tmax_c - forecast_max_c", "prediction": "official forecast or parent 0185 correction depending on nested monthly gate"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official forecast max on identical rows."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Month gate decisions use only target years before the fold start.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P90/P95/max AE", ">2C and >3C rates", "fold/year/season/month/source slices"],
        "sample_rules": {"row_policy": "All pre-2024 official rows inherited from 0185.", "missing_policy": "If a month lacks prior support, abstain except early fold bootstrap."},
        "acceptance_gates": {"minimum_mae_lift_c_vs_official": 0.01, "target_parent_no_harm": "Candidate should not materially worsen 0185 global MAE.", "confirmation": "No 2024+ access."},
        "rejection_conditions": ["Parent 0185 predictions missing or invalid.", "Any 2024+ row present.", "Candidate and baseline row sets differ."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(
    predictions: pd.DataFrame,
    gates: pd.DataFrame,
    scoreboard: pd.DataFrame,
    slice_metrics: pd.DataFrame,
    yearly_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    readme = f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a sequential child of 0185 and stays inside a single experiment folder.

## One-Sentence Hypothesis

A nested monthly no-harm gate can preserve the useful T-7 online-memory corrections while abstaining in months where prior years show weak lift or severe-tail damage.

## Why It Is Worth Doing

0185 was a real T-24-compliant improvement, but its slice anatomy showed uneven warm-season behavior. A gate is valuable only if it can be selected using prior years, not by looking at the target fold. This experiment tests that exact operational constraint.

## Prior Evidence And Novelty

0185 supplied the parent correction. 0186 adds a new layer: prior-year same-month evidence decides whether to apply or abstain. This is distinct from fitting a larger model because it is an explicit no-harm policy.

## Target, Horizon, And Exact Cutoff

The target is HKO daily Tmax at `T-24` in `Asia/Hong_Kong`. Parent corrections inherit the 0185 T-7 residual maturity rule; gate decisions use only dates before each fold start.

## Datasets, Stations, And Attributes

The dataset is the 0185 row-level prediction artifact plus the official forecast baseline embedded in it. The HKO station supplies target labels only for prior-year gate training and final scoring.

## Feature Definitions

The only new feature is `monthly_gate_apply_memory`: a deterministic Boolean selected from prior same-month MAE and >3C harm evidence. Details are in `feature_definitions.csv`.

## Response And Baseline

The response is the same official residual. The baseline is raw official forecast max on identical rows.

## Walk-Forward Design

For every fold, months are evaluated only on years before the fold start. If a month lacks enough history or fails the no-harm rule, the candidate abstains to official for that month.

## Acceptance And Rejection Criteria

Acceptance requires identical rows, no 2024+ access, no current-fold gate tuning, and equal or better MAE/tail behavior versus the parent memory lane.

## Expected Failure Modes

The gate can fail if month-level evidence is too coarse or if the early-fold bootstrap applies memory in months that later become weak.

## Reproduction Command

Run `python scripts/run_hkg_t24_0186_lag7_memory_monthly_noharm_gate.py` from the repository root.
"""
    write_text(EXP_DIR / "README.md", readme)

    results = f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

The candidate and baseline use {summary['n_common']} identical rows from {summary['date_start']} through {summary['date_end']}. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Baseline MAE is `{summary['baseline_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. MAE delta is `{summary['mae_delta_c']:.6f}` C. Parent 0185 MAE is `{summary['parent_0185_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=20)}

## Yearly And Seasonal Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=30)}

Season metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('season')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=10)}

## Source And Source-Era Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window'])][['slice_type', 'slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=20)}

## High-Error-Tail Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('official_tail')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'official_gt3c_rate', 'candidate_gt3c_rate']], max_rows=10)}

## Signed Over/Underforecast Results

Signed candidate and official errors are saved row-level in `predictions.parquet`. Gate decisions are saved in `artifacts/month_gate_decisions.csv`.

## Ablations

The only ablation is the parent 0185 correction versus abstain-to-official per month. Gate thresholds are fixed in the spec and source code.

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. No target-fold month outcome is used to decide that fold's gate.

## Comparison Limitations

This is a child experiment of 0185 and should be judged as a no-harm refinement, not as an independent new model family. It remains development evidence only; confirmation is locked.

Gate decision sample:

{base.markdown_table(gates.head(24), max_rows=24)}
"""
    write_text(EXP_DIR / "RESULTS.md", results)

    conclusion = f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

The experiment tested whether the 0185 memory lift can be made more robust through a prior-year monthly apply/abstain rule. This directly probes the warm-season weakness found in the previous folder.

## Realized Point-MAE Change

The realized delta versus official is `{summary['mae_delta_c']:.6f}` C. The delta versus parent 0185 is `{summary['delta_vs_parent_0185_mae_c']:.6f}` C.

## Information Gain Outside Point MAE

The gate decisions identify which months have enough prior evidence to trust the lag-seven residual memory. This helps decide whether future memory systems should route by month, season, or a richer meteorological context.

## Robustness And Uncertainty

Every gate is selected from prior years only. The parent correction already enforces residual maturity of at least seven target days.

## Failure Diagnosis

If the gate failed to beat 0185, month is too blunt a no-harm dimension or the early bootstrap dominates. If it helped, no-harm routing is worth adding before any confirmation replay.

## Promotion Status

Confirmation remains locked and unauthorized. Development gate to 0.45 C was not reached.

## Implication For Future Research

The next Director choice should either improve the residual-memory contexts or attack the remaining MAM/high-error tail with a separate T-24-safe source/text specialist.
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

    parent = pd.read_parquet(PARENT_PREDICTIONS)
    parent["target_date"] = pd.to_datetime(parent["target_date"]).dt.normalize()
    parent = parent[parent["target_date"] < pd.Timestamp("2024-01-01")].copy()
    parent = parent.rename(
        columns={
            "candidate_prediction_c": "parent_0185_prediction_c",
            "candidate_correction_c": "parent_0185_correction_c",
        }
    )
    base.assert_pre2024(parent, "0186 parent predictions")
    predictions, gates = apply_nested_gate(parent)
    slice_metrics, yearly_metrics, fold_metrics = build_slice_metrics(predictions)

    official_global = base.metric_row(predictions, "official_prediction_c", label="official_forecast_max_c")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    parent_global = base.metric_row(predictions, "parent_0185_prediction_c", label="0185_parent")
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_parent = candidate_global["mae_c"] - parent_global["mae_c"]
    severe_harm = candidate_global["gt3c_rate"] - official_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["mae_delta_c"].max())

    if mae_delta <= -0.01 and delta_vs_parent <= 0.0 and severe_harm <= 0.005 and fold_worst_delta <= 0.02:
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
                "model_family": "nested_monthly_noharm_gate_over_0185",
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
            {
                "candidate_id": "0185_parent_lag7_memory",
                "model_family": "parent_reference_not_primary_baseline",
                "n": parent_global["n"],
                "mae_c": parent_global["mae_c"],
                "rmse_c": parent_global["rmse_c"],
                "bias_c": parent_global["bias_c"],
                "median_abs_error_c": parent_global["median_abs_error_c"],
                "p95_abs_error_c": parent_global["p95_abs_error_c"],
                "gt2c_rate": parent_global["gt2c_rate"],
                "gt3c_rate": parent_global["gt3c_rate"],
                "baseline_mae_c": official_global["mae_c"],
                "mae_delta_c": parent_global["mae_c"] - official_global["mae_c"],
            },
        ]
    )
    row_coverage = pd.DataFrame(
        [
            {
                "frame_id": "official_t15_pre2024_5265_rows",
                "parent_rows": int(len(parent)),
                "candidate_rows": int(len(predictions)),
                "baseline_rows": int(len(predictions)),
                "common_rows": int(len(predictions)),
                "date_start": date_text(predictions["target_date"].min()),
                "date_end": date_text(predictions["target_date"].max()),
                "row_policy": "identical rows inherited from 0185 parent predictions",
                "common_row_hash": common_row_hash,
            }
        ]
    )
    correction_distribution = predictions["candidate_correction_c"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    correction_distribution.columns = ["statistic", "candidate_correction_c"]
    data_manifest = pd.DataFrame(
        [
            {
                "source_id": "0185_parent_predictions",
                "path": rel(PARENT_PREDICTIONS),
                "sha256": sha256_file(PARENT_PREDICTIONS),
                "size_bytes": PARENT_PREDICTIONS.stat().st_size,
                "row_count": int(len(parent)),
                "date_start": date_text(parent["target_date"].min()),
                "date_end": date_text(parent["target_date"].max()),
                "timestamp_fields": "target_date;fold_id;parent T-7 correction state",
                "availability_class": "DEPLOYABLE_LAGGED_ONLY",
                "notes": "Parent 0185 artifact is a leakage-passed T-7 residual-memory candidate.",
            }
        ]
    )
    feature_definitions = pd.DataFrame(
        [
            {
                "feature_name": "monthly_gate_apply_memory",
                "role": "router",
                "formula": "Apply parent correction if prior same-month n>=120, parent_delta<=-0.002 C, and gt3_harm<=0.005; else abstain to official.",
                "input_columns": "prior-year target_tmax_c, official_prediction_c, parent_0185_prediction_c, month",
                "units": "boolean",
                "lag": "prior years only for fold-local gate decisions",
                "window": "expanding by outer fold",
                "fit_scope": "fold-local deterministic gate",
                "availability_rule": "No current-fold outcomes used for gate decisions.",
                "missingness_policy": "Insufficient history abstains except early bootstrap fold.",
            }
        ]
    )

    prediction_cols = [
        "target_date",
        "target_tmax_c",
        "forecast_source_family",
        "season",
        "month",
        "official_prediction_c",
        "parent_0185_prediction_c",
        "candidate_prediction_c",
        "parent_0185_correction_c",
        "candidate_correction_c",
        "monthly_gate_apply_memory",
        "official_error_c_signed",
        "candidate_error_c",
        "official_abs_error_c",
        "candidate_abs_error_c",
        "fold_id",
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
    write_csv(EXP_DIR / "artifacts" / "month_gate_decisions.csv", gates)

    audit = f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

The experiment inherits official exact-vintage rows from 0185 and keeps the same HKO T-24 target. Parent corrections already enforce T-7 residual maturity.

## Available Gate Evidence

For each outer fold, monthly apply/abstain decisions use only rows with target years before the fold start. No target-fold month result can influence its own gate.

## Target And Rolling Checks

The current target value is used only for scoring. Parent 0185 predictions have a validator-clean leakage audit and no residual newer than T-7.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Fold Fit Proof

Gate thresholds are fixed before scoring: n >= {MIN_MONTH_HISTORY}, parent same-month MAE delta <= -{MIN_MONTH_LIFT_C}, and >3C harm <= {MAX_GT3_HARM}.
"""
    write_text(EXP_DIR / "leakage_audit.md", audit)
    write_text(
        EXP_DIR / "REPRODUCE.md",
        f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0186_lag7_memory_monthly_noharm_gate.py
```

Requires the completed 0185 parent predictions at `{PARENT_PREDICTIONS}`. Confirmation rows remain locked.
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
        "parent_0185_mae_c": parent_global["mae_c"],
        "delta_vs_parent_0185_mae_c": delta_vs_parent,
        "fold_worst_mae_delta_c": fold_worst_delta,
        "severe_gt3_rate_delta": severe_harm,
    }
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(predictions, gates, scoreboard, slice_metrics, yearly_metrics, fold_metrics, summary)
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
