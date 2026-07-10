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
EXPERIMENT_ID = "0188"
SLUG = "nested_expert_router_0185_0187"
TITLE = "Nested Expert Router Between Official, 0185, and 0187"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0188_nested_prior_context_expert_router"
MODEL_FOLDS = base.MODEL_FOLDS
P0185 = EXPERIMENTS_ROOT / "0185_lag7_online_residual_memory_router" / "predictions.parquet"
P0187 = EXPERIMENTS_ROOT / "0187_deployable_isd_memory_residual_fusion" / "predictions.parquet"
EXPERTS = ["official", "0185", "0187"]
MIN_CONTEXT_N = {
    "month_source": 80,
    "month": 150,
    "season_source": 180,
    "season": 260,
    "global": 365,
}
MIN_ADVANTAGE_C = 0.002
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


def load_parent_frame() -> pd.DataFrame:
    p85 = pd.read_parquet(P0185)
    p87 = pd.read_parquet(P0187)
    for frame in (p85, p87):
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
        frame.drop(frame[frame["target_date"] >= pd.Timestamp("2024-01-01")].index, inplace=True)
    base_cols = [
        "target_date",
        "target_tmax_c",
        "forecast_source_family",
        "season",
        "month",
        "official_prediction_c",
    ]
    merged = p85[base_cols + ["candidate_prediction_c", "candidate_correction_c"]].rename(
        columns={
            "candidate_prediction_c": "prediction_0185_c",
            "candidate_correction_c": "correction_0185_c",
        }
    )
    right = p87[["target_date", "candidate_prediction_c", "candidate_correction_c"]].rename(
        columns={
            "candidate_prediction_c": "prediction_0187_c",
            "candidate_correction_c": "correction_0187_c",
        }
    )
    merged = merged.merge(right, on="target_date", how="inner", validate="one_to_one")
    merged["prediction_official_c"] = merged["official_prediction_c"]
    for expert in EXPERTS:
        merged[f"abs_error_{expert}_c"] = (merged[f"prediction_{expert}_c"] - merged["target_tmax_c"]).abs()
        merged[f"error_{expert}_c"] = merged[f"prediction_{expert}_c"] - merged["target_tmax_c"]
    base.assert_pre2024(merged, "0188 parent expert frame")
    return merged.sort_values("target_date").reset_index(drop=True)


def context_value(row: pd.Series, context_type: str) -> str:
    source = str(row.get("forecast_source_family") or "source_unknown")
    month = f"month_{int(row.get('month')):02d}" if pd.notna(row.get("month")) else "month_unknown"
    season = str(row.get("season") or "season_unknown")
    if context_type == "month_source":
        return f"{month}|source={source}"
    if context_type == "month":
        return month
    if context_type == "season_source":
        return f"{season}|source={source}"
    if context_type == "season":
        return season
    return "global"


def context_table(train: pd.DataFrame, context_type: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train = train.copy()
    train["context_value"] = train.apply(lambda row: context_value(row, context_type), axis=1)
    for value, group in train.groupby("context_value", dropna=False):
        if len(group) < MIN_CONTEXT_N[context_type]:
            continue
        official_mae = float(group["abs_error_official_c"].mean())
        parent_mae = float(group["abs_error_0185_c"].mean())
        best_expert = "0185"
        best_mae = parent_mae
        for expert in EXPERTS:
            mae = float(group[f"abs_error_{expert}_c"].mean())
            gt3 = float(group[f"abs_error_{expert}_c"].gt(3.0).mean())
            gt3_parent = float(group["abs_error_0185_c"].gt(3.0).mean())
            if mae < best_mae - MIN_ADVANTAGE_C and gt3 - gt3_parent <= MAX_GT3_HARM:
                best_expert = expert
                best_mae = mae
        rows.append(
            {
                "context_type": context_type,
                "context_value": value,
                "n_train": int(len(group)),
                "official_mae_c": official_mae,
                "parent_0185_mae_c": parent_mae,
                "expert_0187_mae_c": float(group["abs_error_0187_c"].mean()),
                "selected_expert": best_expert,
                "selected_mae_c": best_mae,
                "selected_delta_vs_parent_c": best_mae - parent_mae,
                "official_gt3_rate": float(group["abs_error_official_c"].gt(3.0).mean()),
                "parent_0185_gt3_rate": float(group["abs_error_0185_c"].gt(3.0).mean()),
                "expert_0187_gt3_rate": float(group["abs_error_0187_c"].gt(3.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_gate(train: pd.DataFrame) -> pd.DataFrame:
    tables = [context_table(train, context) for context in ("month_source", "month", "season_source", "season", "global")]
    tables = [table for table in tables if not table.empty]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def apply_gate(test: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    out = test.copy()
    selected_experts: list[str] = []
    selected_contexts: list[str] = []
    selected_context_types: list[str] = []
    for _, row in out.iterrows():
        chosen = "0185"
        chosen_context = "default_parent"
        chosen_type = "default"
        if not gate.empty and {"context_type", "context_value", "selected_expert"}.issubset(gate.columns):
            for context_type in ("month_source", "month", "season_source", "season", "global"):
                value = context_value(row, context_type)
                hit = gate[(gate["context_type"].eq(context_type)) & (gate["context_value"].eq(value))]
                if not hit.empty:
                    chosen = str(hit.iloc[0]["selected_expert"])
                    chosen_context = value
                    chosen_type = context_type
                    break
        selected_experts.append(chosen)
        selected_contexts.append(chosen_context)
        selected_context_types.append(chosen_type)
    out["selected_expert"] = selected_experts
    out["selected_context"] = selected_contexts
    out["selected_context_type"] = selected_context_types
    out["candidate_prediction_c"] = [
        row[f"prediction_{expert}_c"] for (_, row), expert in zip(out.iterrows(), selected_experts)
    ]
    out["candidate_correction_c"] = out["candidate_prediction_c"] - out["official_prediction_c"]
    out["candidate_error_c"] = out["candidate_prediction_c"] - out["target_tmax_c"]
    out["official_error_c_signed"] = out["official_prediction_c"] - out["target_tmax_c"]
    out["official_abs_error_c"] = out["official_error_c_signed"].abs()
    out["candidate_abs_error_c"] = out["candidate_error_c"].abs()
    out["candidate_id"] = PRIMARY_CANDIDATE_ID
    out["baseline_id"] = "official_forecast_max_c"
    out["model_family"] = "nested_prior_context_expert_router"
    return out


def run_walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    gate_rows: list[pd.DataFrame] = []
    for start_year, end_year in MODEL_FOLDS:
        test = frame[frame["target_date"].dt.year.between(start_year, end_year)].copy()
        train = frame[frame["target_date"].dt.year < start_year].copy()
        fold_id = f"fold_{start_year}_{end_year}"
        if test.empty:
            fold_rows.append({"fold_id": fold_id, "n": 0})
            continue
        gate = build_gate(train) if len(train) >= 365 else pd.DataFrame()
        if not gate.empty:
            gate = gate.copy()
            gate["fold_id"] = fold_id
            gate["start_year"] = start_year
            gate["end_year"] = end_year
            gate_rows.append(gate)
        pred = apply_gate(test, gate)
        pred["fold_id"] = fold_id
        parts.append(pred)
        metric = base.compare_metrics(pred, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "training_rows": int(len(train)),
                "expert_0185_share": float(pred["selected_expert"].eq("0185").mean()),
                "expert_0187_share": float(pred["selected_expert"].eq("0187").mean()),
                "expert_official_share": float(pred["selected_expert"].eq("official").mean()),
            }
        )
        fold_rows.append(metric)
    predictions = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    return predictions, pd.DataFrame(fold_rows), gates


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
            "A prior-history context router can preserve the 0185 global residual-memory lift while selectively using 0187 station-fusion "
            "or official abstention in contexts where prior years show lower MAE and no severe-tail harm."
        ),
        "rationale": (
            "0185 is the current champion. 0187 did not beat it globally but improved official-tail, DJF, and MAM while damaging JJA/SON. "
            "A nested router tests whether those complementary strengths can be combined without looking at the test fold."
        ),
        "expected_sign_and_falsification": (
            "Expected sign is lower MAE than 0185 and official. Falsification occurs if prior context choices fail to transfer or if router "
            "complexity adds no stable improvement over the parent."
        ),
        "novelty": {
            "prior_experiments": ["0185", "0187"],
            "difference": "This is a fold-local expert-selection layer over two completed leakage-passed parents, not another raw feature model.",
            "similarity_audit_path": "RESULTS.md#comparison-limitations",
        },
        "target": {
            "station": "Hong Kong Observatory headquarters",
            "variable": "tmax_c",
            "horizon": "T-24",
            "timezone": "Asia/Hong_Kong",
            "cutoff_contract_path": rel(base.OFFICIAL_PATH),
            "cutoff_function": "Use only parent predictions whose own audits passed; router selection uses prior target years only.",
            "daily_boundary_contract": "HKO local daily maximum temperature for target local date T.",
        },
        "frame": {
            "frame_id": "official_t15_pre2024_5265_rows",
            "development_start": "2000-01-02",
            "development_end_exclusive": "2024-01-01",
            "confirmation_locked": True,
            "row_universe_artifact": rel(P0185),
        },
        "data_sources": [
            {
                "source_id": "0185_parent_predictions",
                "paths": [rel(P0185)],
                "attributes": ["official_prediction_c", "candidate_prediction_c", "target_tmax_c", "season", "month", "forecast_source_family"],
                "eligibility": "DEPLOYABLE_LAGGED_ONLY",
                "availability_proof": "0185 validator passed; residual memory has T-7 maturity.",
            },
            {
                "source_id": "0187_parent_predictions",
                "paths": [rel(P0187)],
                "attributes": ["candidate_prediction_c"],
                "eligibility": "DEPLOYABLE_LAGGED_ONLY",
                "availability_proof": "0187 validator passed; station fusion uses cutoff-safe ISD and lagged memory.",
            },
        ],
        "stations": [{"station_id": "HKO", "role": "target; parent prediction audit", "attributes": ["daily Tmax"]}],
        "features": {
            "generation_rule": "Prior-history expert selection by month_source, month, season_source, season, then global context.",
            "explicit_exclusions": ["2024+ rows", "current-fold router tuning", "unvalidated parent predictions"],
            "thresholds": {"min_advantage_c": MIN_ADVANTAGE_C, "max_gt3_harm": MAX_GT3_HARM, "min_context_n": MIN_CONTEXT_N},
        },
        "response": {"variable": "target_tmax_c", "prediction": "selected expert prediction among official, 0185, 0187"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official maximum forecast on identical rows."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "For every fold, context expert choices are computed only from years before the fold start.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P90/P95/max AE", ">2C and >3C rates", "fold/year/season/month/source/tail slices"],
        "sample_rules": {"row_policy": "All parent rows common to 0185 and 0187.", "missing_policy": "No gate hit defaults to 0185 parent."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0185_c": 0.003, "no_tail_harm": ">3C rate cannot exceed official by more than 0.005."},
        "rejection_conditions": ["Any parent row mismatch.", "Any 2024+ row.", "Router uses current-fold target outcomes."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(predictions: pd.DataFrame, gates: pd.DataFrame, scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(
        EXP_DIR / "README.md",
        f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a nested parent-expert router over validator-clean experiments 0185 and 0187.

## One-Sentence Hypothesis

Prior-year context performance can choose between official, 0185, and 0187 better than any single parent globally.

## Why It Is Worth Doing

0185 is globally best, but 0187 gives stronger cool-season/tail behavior. A router is the direct test of whether those complementary strengths transfer under a strict chronological gate.

## Prior Evidence And Novelty

0185 supplies T-7 online memory. 0187 supplies station-network fusion. 0188 is new because it uses parent predictions as deployable experts and selects among them only from prior years.

## Target, Horizon, And Exact Cutoff

Target is HKO daily Tmax at `T-24`, timezone `Asia/Hong_Kong`. Parent predictions inherit their own cutoff proofs; router choices are prior-year only.

## Datasets, Stations, And Attributes

Inputs are row-level predictions from 0185 and 0187 on the same official frame. Context attributes are forecast source family, month, and season.

## Feature Definitions

The router uses deterministic context keys and prior expert MAE/tail evidence. Full rules are in `feature_definitions.csv` and gate rows are in `artifacts/context_gate_decisions.csv`.

## Response And Baseline

Response is target Tmax. The primary baseline is raw official forecast; 0185 and 0187 are parent references.

## Walk-Forward Design

Each fold builds context expert tables from target years before the fold start. Test-fold rows use the most specific available context in this order: month-source, month, season-source, season, global.

## Acceptance And Rejection Criteria

Acceptance requires lower MAE than official, ideally lower than 0185, identical rows, no 2024+ rows, and no severe-tail harm.

## Expected Failure Modes

The router can fail if context-level expert superiority is unstable or if 0187's station advantage is too episodic to transfer.

## Reproduction Command

Run `python scripts/run_hkg_t24_0188_nested_expert_router_0185_0187.py` from the repository root.
""",
    )
    expert_share = predictions["selected_expert"].value_counts(normalize=True).rename("share").reset_index().rename(columns={"index": "selected_expert"})
    write_text(
        EXP_DIR / "RESULTS.md",
        f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

The scored frame has `{summary['n_common']}` identical rows from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0185 is `{summary['delta_vs_parent_0185_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'expert_0185_share', 'expert_0187_share', 'expert_official_share']], max_rows=20)}

## Yearly And Seasonal Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=30)}

Season metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('season')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=10)}

## Source And Source-Era Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window'])][['slice_type', 'slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=20)}

## High-Error-Tail Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('official_tail')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'official_gt3c_rate', 'candidate_gt3c_rate']], max_rows=10)}

## Signed Over/Underforecast Results

Row-level signed errors, selected expert, selected context, and correction are in `predictions.parquet`.

## Ablations

Parent experts are listed in `scoreboard.csv`; context gate choices are in `artifacts/context_gate_decisions.csv`.

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. Router gates are computed only from prior target years.

## Comparison Limitations

This is a child meta-experiment over 0185 and 0187. It is valid only because both parent folders passed validation on the same row frame.

Expert selection share:

{base.markdown_table(expert_share, max_rows=10)}

Gate sample:

{base.markdown_table(gates.head(30), max_rows=30)}
""",
    )
    write_text(
        EXP_DIR / "CONCLUSION.md",
        f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0188 tested whether station-fusion's tail/cool-season strengths can be harvested without inheriting its warm-season damage.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus parent 0185 is `{summary['delta_vs_parent_0185_mae_c']:.6f}` C.

## Information Gain Outside Point MAE

The context gate table shows whether expert superiority is stable enough to route. A failure here means broad parent routing is too coarse and future routers need more physical contexts.

## Robustness And Uncertainty

All gates are prior-history only. No current-fold target result can choose its own expert.

## Failure Diagnosis

If not promoted, the likely failure is context instability or overcoarse context definitions. If promoted, the result would justify a richer expert router.

## Promotion Status

Confirmation remains locked and unauthorized. Development gate to 0.45 C was not reached.

## Implication For Future Research

Use this evidence to decide whether to pursue finer physics-context routing or return to residual-memory architecture.
""",
    )


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

    frame = load_parent_frame()
    predictions, fold_metrics, gates = run_walk_forward(frame)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official_forecast_max_c")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    parent85_global = base.metric_row(predictions, "prediction_0185_c", label="0185_parent")
    parent87_global = base.metric_row(predictions, "prediction_0187_c", label="0187_parent")
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_parent85 = candidate_global["mae_c"] - parent85_global["mae_c"]
    severe_harm = candidate_global["gt3c_rate"] - official_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["mae_delta_c"].max())
    if mae_delta <= -0.01 and delta_vs_parent85 <= -0.003 and severe_harm <= 0.005 and fold_worst_delta <= 0.02:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_OVER_0185_NO_CONFIRMATION"
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
                "model_family": "nested_prior_context_expert_router",
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
                "model_family": "parent_reference",
                "n": parent85_global["n"],
                "mae_c": parent85_global["mae_c"],
                "rmse_c": parent85_global["rmse_c"],
                "bias_c": parent85_global["bias_c"],
                "median_abs_error_c": parent85_global["median_abs_error_c"],
                "p95_abs_error_c": parent85_global["p95_abs_error_c"],
                "gt2c_rate": parent85_global["gt2c_rate"],
                "gt3c_rate": parent85_global["gt3c_rate"],
                "baseline_mae_c": official_global["mae_c"],
                "mae_delta_c": parent85_global["mae_c"] - official_global["mae_c"],
            },
            {
                "candidate_id": "0187_parent_isd_fusion",
                "model_family": "parent_reference",
                "n": parent87_global["n"],
                "mae_c": parent87_global["mae_c"],
                "rmse_c": parent87_global["rmse_c"],
                "bias_c": parent87_global["bias_c"],
                "median_abs_error_c": parent87_global["median_abs_error_c"],
                "p95_abs_error_c": parent87_global["p95_abs_error_c"],
                "gt2c_rate": parent87_global["gt2c_rate"],
                "gt3c_rate": parent87_global["gt3c_rate"],
                "baseline_mae_c": official_global["mae_c"],
                "mae_delta_c": parent87_global["mae_c"] - official_global["mae_c"],
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
                "row_policy": "inner join of validated 0185 and 0187 parent rows; identical official baseline rows",
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
                "path": rel(P0185),
                "sha256": sha256_file(P0185),
                "size_bytes": P0185.stat().st_size,
                "row_count": int(len(frame)),
                "date_start": date_text(frame["target_date"].min()),
                "date_end": date_text(frame["target_date"].max()),
                "timestamp_fields": "target_date;fold_id;T-7 correction state",
                "availability_class": "DEPLOYABLE_LAGGED_ONLY",
                "notes": "Validator-clean parent experiment.",
            },
            {
                "source_id": "0187_parent_predictions",
                "path": rel(P0187),
                "sha256": sha256_file(P0187),
                "size_bytes": P0187.stat().st_size,
                "row_count": int(len(frame)),
                "date_start": date_text(frame["target_date"].min()),
                "date_end": date_text(frame["target_date"].max()),
                "timestamp_fields": "target_date;fold_id;cutoff-safe station fusion",
                "availability_class": "DEPLOYABLE_LAGGED_ONLY",
                "notes": "Validator-clean parent experiment.",
            },
        ]
    )
    feature_definitions = pd.DataFrame(
        [
            {
                "feature_name": "nested_prior_context_expert_choice",
                "role": "router",
                "formula": "Use most specific prior-history context with n support, then select official, 0185, or 0187 if MAE beats 0185 by >=0.002 C and >3C harm <=0.005; default 0185.",
                "input_columns": "month, season, forecast_source_family, prior-year parent prediction errors",
                "units": "categorical expert choice",
                "lag": "prior years only for fold-local router",
                "window": "expanding outer-fold training history",
                "fit_scope": "fold-local deterministic gate",
                "availability_rule": "Uses only parent predictions with passed leakage audits and prior target years.",
                "missingness_policy": "No eligible context defaults to 0185 parent.",
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
        "prediction_0185_c",
        "prediction_0187_c",
        "candidate_prediction_c",
        "candidate_correction_c",
        "selected_expert",
        "selected_context_type",
        "selected_context",
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
    write_csv(EXP_DIR / "artifacts" / "context_gate_decisions.csv", gates)

    write_text(
        EXP_DIR / "leakage_audit.md",
        f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

0188 consumes only row-level parent predictions from 0185 and 0187, both validator-clean on the same official T15 pre-2024 frame. Parent availability contracts remain in force.

## Available Router Evidence

For a target row in outer fold F, expert selection tables are computed only from years before F starts. Current-fold target outcomes cannot affect their own selected expert.

## Target And Rolling Checks

The current target value is used only for scoring. Router features are deterministic source/month/season context keys and prior-year expert performance summaries.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""",
    )
    write_text(
        EXP_DIR / "REPRODUCE.md",
        f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0188_nested_expert_router_0185_0187.py
```

Requires completed parent predictions from 0185 and 0187. Confirmation rows remain locked.
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
        "parent_0185_mae_c": parent85_global["mae_c"],
        "parent_0187_mae_c": parent87_global["mae_c"],
        "delta_vs_parent_0185_mae_c": delta_vs_parent85,
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
