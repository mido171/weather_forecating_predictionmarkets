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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_hkg_t24_0184_hf_teacher_proxy_causal_memory_router as base
import run_hkg_t24_0194_isd_role_compressed_regime_proxy as role0194


REPO_ROOT = base.REPO_ROOT
EXPERIMENTS_ROOT = base.EXPERIMENTS_ROOT
EXPERIMENT_ID = "0196"
SLUG = "station_network_tail_conditioned_residual_expert"
TITLE = "Station-Network Tail-Conditioned Residual Expert Over 0194"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0196_station_network_tail_conditioned_residual_expert_over_0194"
P0194 = EXPERIMENTS_ROOT / "0194_isd_role_compressed_regime_proxy" / "predictions.parquet"
SRC_COPY_NAME = "run_0196.py"
MODEL_FOLDS = base.MODEL_FOLDS
INNER_MIN_LIFT_C = 0.001

MODEL_GRID = [
    {"model": "zero", "cap_c": 0.0, "scale": 0.0, "tail_threshold_c": 0.0, "tail_weight": 0.0},
    {"model": "ridge", "alpha": 100.0, "cap_c": 0.08, "scale": 0.50, "tail_threshold_c": 1.5, "tail_weight": 2.0},
    {"model": "ridge", "alpha": 300.0, "cap_c": 0.12, "scale": 0.75, "tail_threshold_c": 1.5, "tail_weight": 4.0},
    {"model": "huber", "alpha": 0.0010, "epsilon": 1.35, "cap_c": 0.10, "scale": 0.50, "tail_threshold_c": 1.5, "tail_weight": 3.0},
    {"model": "huber", "alpha": 0.0020, "epsilon": 1.50, "cap_c": 0.15, "scale": 0.75, "tail_threshold_c": 2.0, "tail_weight": 5.0},
    {"model": "hgb", "learning_rate": 0.025, "max_leaf_nodes": 7, "l2_regularization": 5.0, "max_iter": 80, "cap_c": 0.10, "scale": 0.50, "tail_threshold_c": 1.5, "tail_weight": 3.0},
    {"model": "hgb", "learning_rate": 0.025, "max_leaf_nodes": 15, "l2_regularization": 8.0, "max_iter": 100, "cap_c": 0.15, "scale": 0.75, "tail_threshold_c": 2.0, "tail_weight": 6.0},
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


def make_estimator(config: dict[str, Any]) -> Any:
    if config["model"] == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(config["alpha"]))),
            ]
        )
    if config["model"] == "huber":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(alpha=float(config["alpha"]), epsilon=float(config["epsilon"]), max_iter=500)),
            ]
        )
    if config["model"] == "hgb":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=float(config["learning_rate"]),
                        max_leaf_nodes=int(config["max_leaf_nodes"]),
                        l2_regularization=float(config["l2_regularization"]),
                        max_iter=int(config["max_iter"]),
                        random_state=196,
                        loss="absolute_error",
                    ),
                ),
            ]
        )
    return None


def load_frame() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    role_frame, feature_cols, feature_defs = role0194.load_feature_frame()
    parent = pd.read_parquet(P0194)
    parent["target_date"] = pd.to_datetime(parent["target_date"], errors="coerce").dt.normalize()
    parent = parent[parent["target_date"] < pd.Timestamp("2024-01-01")].copy()
    parent = parent[
        [
            "target_date",
            "candidate_prediction_c",
            "candidate_correction_c",
            "isd_role_correction_c",
            "selected_config_id",
            "selected_model",
        ]
    ].rename(
        columns={
            "candidate_prediction_c": "parent_0194_prediction_c",
            "candidate_correction_c": "parent_0194_total_correction_c",
            "isd_role_correction_c": "parent_0194_isd_role_correction_c",
            "selected_config_id": "parent_0194_selected_config_id",
            "selected_model": "parent_0194_selected_model",
        }
    )
    frame = role_frame.merge(parent, on="target_date", how="left", validate="one_to_one")
    frame["parent_0194_error_c"] = frame["parent_0194_prediction_c"] - frame["target_tmax_c"]
    frame["parent_0194_abs_error_c"] = frame["parent_0194_error_c"].abs()
    frame["parent_0194_residual_c"] = frame["target_tmax_c"] - frame["parent_0194_prediction_c"]
    frame["parent_0194_isd_abs_c"] = frame["parent_0194_isd_role_correction_c"].abs()
    frame["parent_0194_isd_positive"] = frame["parent_0194_isd_role_correction_c"].gt(0.0).astype(float)
    frame["parent_0194_isd_at_cap"] = frame["parent_0194_isd_abs_c"].ge(0.149).astype(float)
    frame["parent_0194_delta_from_0190_c"] = frame["parent_0194_prediction_c"] - frame["parent_0190_prediction_c"]
    extra_cols = [
        "parent_0194_isd_role_correction_c",
        "parent_0194_isd_abs_c",
        "parent_0194_isd_positive",
        "parent_0194_isd_at_cap",
        "parent_0194_delta_from_0190_c",
    ]
    for column in extra_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    feature_cols = [*feature_cols, *extra_cols]
    feature_defs = pd.concat(
        [
            feature_defs,
            pd.DataFrame(
                [
                    {
                        "feature_name": column,
                        "role": "candidate_predictor",
                        "family": "known_0194_parent_correction_context",
                        "formula": "Known parent 0194 correction/cap context at decision time.",
                        "input_columns": column,
                        "units": "degC/indicator",
                        "lag": "known from parent prediction",
                        "window": "row-level parent prediction",
                        "fit_scope": "Fold-local imputation, scaling, fitting, and config selection",
                        "availability_rule": "Parent 0194 prediction is available before target outcome.",
                        "missingness_policy": "Median imputation inside each fitted training fold.",
                    }
                    for column in extra_cols
                ]
            ),
        ],
        ignore_index=True,
    )
    base.assert_pre2024(frame, "0196 model frame")
    return frame.sort_values("target_date").reset_index(drop=True), feature_cols, feature_defs


def tail_sample_weight(train: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    threshold = float(config["tail_threshold_c"])
    weight = float(config["tail_weight"])
    if threshold <= 0 or weight <= 0:
        return np.ones(len(train), dtype=float)
    return 1.0 + weight * train["parent_0194_abs_error_c"].ge(threshold).to_numpy(dtype=float)


def predict_config(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], config: dict[str, Any]) -> np.ndarray:
    if config["model"] == "zero":
        return np.zeros(len(test), dtype=float)
    estimator = make_estimator(config)
    y_train = train["target_tmax_c"].to_numpy(dtype=float) - train["parent_0194_prediction_c"].to_numpy(dtype=float)
    weights = tail_sample_weight(train, config)
    estimator.fit(train[feature_cols], y_train, model__sample_weight=weights)
    raw = estimator.predict(test[feature_cols])
    clipped = np.clip(raw, -float(config["cap_c"]), float(config["cap_c"]))
    return float(config["scale"]) * clipped


def inner_select(train: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    max_year = int(train["target_date"].dt.year.max())
    split_year = max(int(train["target_date"].dt.year.min()) + 2, max_year - 2)
    inner_fit = train[train["target_date"].dt.year < split_year].copy()
    inner_val = train[train["target_date"].dt.year >= split_year].copy()
    zero = {"config_id": "cfg_00_zero", **MODEL_GRID[0], "inner_mae_c": math.nan, "inner_parent_mae_c": math.nan, "inner_delta_vs_0194_mae_c": 0.0, "selection_reason": "bootstrap_or_no_lift"}
    if len(inner_fit) < 365 or len(inner_val) < 120:
        return zero, pd.DataFrame([zero])
    parent_mae = float(np.mean(np.abs(inner_val["parent_0194_prediction_c"] - inner_val["target_tmax_c"])))
    rows = []
    for idx, config in enumerate(MODEL_GRID):
        cfg = {"config_id": f"cfg_{idx:02d}_{config['model']}", **config}
        try:
            correction = predict_config(inner_fit, inner_val, feature_cols, cfg)
            pred = inner_val["parent_0194_prediction_c"].to_numpy(dtype=float) + correction
            mae = float(np.mean(np.abs(pred - inner_val["target_tmax_c"].to_numpy(dtype=float))))
        except Exception as exc:
            mae = math.inf
            cfg["fit_error"] = str(exc)
        cfg["inner_parent_mae_c"] = parent_mae
        cfg["inner_mae_c"] = mae
        cfg["inner_delta_vs_0194_mae_c"] = mae - parent_mae
        rows.append(cfg)
    scores = pd.DataFrame(rows).sort_values(["inner_delta_vs_0194_mae_c", "cap_c", "scale", "config_id"]).reset_index(drop=True)
    best = scores.iloc[0].to_dict()
    if float(best["inner_delta_vs_0194_mae_c"]) <= -INNER_MIN_LIFT_C:
        best["selection_reason"] = "prior_inner_lift"
        return best, scores
    zero["inner_parent_mae_c"] = parent_mae
    zero["inner_mae_c"] = parent_mae
    zero["inner_delta_vs_0194_mae_c"] = 0.0
    zero["selection_reason"] = "zero_fallback_inner_lift_below_threshold"
    return zero, scores


def compare_four(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = base.metric_row(frame, "official_prediction_c", label="official")
    p0190 = base.metric_row(frame, "parent_0190_prediction_c", label="0190_parent")
    p0194 = base.metric_row(frame, "parent_0194_prediction_c", label="0194_parent")
    candidate = base.metric_row(frame, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    return {
        "slice_type": slice_type,
        "slice_value": str(slice_value),
        "n": candidate["n"],
        "official_mae_c": official["mae_c"],
        "parent_0190_mae_c": p0190["mae_c"],
        "parent_0194_mae_c": p0194["mae_c"],
        "candidate_mae_c": candidate["mae_c"],
        "mae_delta_c": candidate["mae_c"] - official["mae_c"],
        "delta_vs_0190_mae_c": candidate["mae_c"] - p0190["mae_c"],
        "delta_vs_0194_mae_c": candidate["mae_c"] - p0194["mae_c"],
        "candidate_bias_c": candidate["bias_c"],
        "official_gt3c_rate": official["gt3c_rate"],
        "parent_0194_gt3c_rate": p0194["gt3c_rate"],
        "candidate_gt3c_rate": candidate["gt3c_rate"],
        "parent_0194_p95_abs_error_c": p0194["p95_abs_error_c"],
        "candidate_p95_abs_error_c": candidate["p95_abs_error_c"],
    }


def run_walk_forward(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    fold_rows = []
    selection_rows = []
    score_rows = []
    for start_year, end_year in MODEL_FOLDS:
        fold_id = f"fold_{start_year}_{end_year}"
        test_mask = frame["target_date"].dt.year.between(start_year, end_year)
        test = frame[test_mask].copy()
        if test.empty:
            fold_rows.append({"fold_id": fold_id, "n": 0})
            continue
        train = frame[frame["target_date"].dt.year < start_year].copy()
        if len(train) < 365:
            selected = {"config_id": "cfg_00_zero_bootstrap", **MODEL_GRID[0], "inner_mae_c": math.nan, "inner_parent_mae_c": math.nan, "inner_delta_vs_0194_mae_c": 0.0, "selection_reason": "first_fold_no_prior_history"}
            scores = pd.DataFrame([selected])
            correction = np.zeros(len(test), dtype=float)
        else:
            selected, scores = inner_select(train, feature_cols)
            correction = predict_config(train, test, feature_cols, selected)
        test["tail_expert_correction_c"] = correction
        test["candidate_prediction_c"] = test["parent_0194_prediction_c"] + test["tail_expert_correction_c"]
        test["candidate_correction_c"] = test["candidate_prediction_c"] - test["official_prediction_c"]
        test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
        test["candidate_abs_error_c"] = test["candidate_error_c"].abs()
        test["official_error_c_signed"] = test["official_prediction_c"] - test["target_tmax_c"]
        test["official_abs_error_c"] = test["official_error_c_signed"].abs()
        test["fold_id"] = fold_id
        test["selected_config_id"] = selected["config_id"]
        test["selected_model"] = selected["model"]
        test["selected_cap_c"] = selected.get("cap_c", 0.0)
        test["selected_scale"] = selected.get("scale", 0.0)
        test["selected_tail_threshold_c"] = selected.get("tail_threshold_c", 0.0)
        test["selected_tail_weight"] = selected.get("tail_weight", 0.0)
        test["selected_inner_delta_vs_0194_mae_c"] = selected.get("inner_delta_vs_0194_mae_c", math.nan)
        metric = compare_four(test, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected["config_id"],
                "selected_model": selected["model"],
                "selected_cap_c": selected.get("cap_c", 0.0),
                "selected_scale": selected.get("scale", 0.0),
                "selected_tail_threshold_c": selected.get("tail_threshold_c", 0.0),
                "selected_tail_weight": selected.get("tail_weight", 0.0),
                "selected_inner_delta_vs_0194_mae_c": selected.get("inner_delta_vs_0194_mae_c", math.nan),
                "selection_reason": selected.get("selection_reason", ""),
                "mean_tail_expert_correction_c": float(test["tail_expert_correction_c"].mean()),
                "mean_abs_tail_expert_correction_c": float(test["tail_expert_correction_c"].abs().mean()),
            }
        )
        fold_rows.append(metric)
        selection_rows.append({"fold_id": fold_id, **selected})
        scores = scores.copy()
        scores["fold_id"] = fold_id
        score_rows.append(scores)
        parts.append(test)
    predictions = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    predictions["candidate_id"] = PRIMARY_CANDIDATE_ID
    predictions["baseline_id"] = "official_forecast_max_c"
    predictions["model_family"] = "station_network_tail_conditioned_residual_expert"
    predictions["feature_count"] = len(feature_cols)
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(selection_rows), pd.concat(score_rows, ignore_index=True)


def build_slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [compare_four(predictions, slice_type="overall", slice_value="all")]
    for season, group in predictions.groupby("season", dropna=False):
        rows.append(compare_four(group, slice_type="season", slice_value=season))
    for month, group in predictions.groupby("month", dropna=False):
        rows.append(compare_four(group, slice_type="month", slice_value=month))
    for source, group in predictions.groupby("forecast_source_family", dropna=False):
        rows.append(compare_four(group, slice_type="source", slice_value=source))
    late = predictions[predictions["target_date"] >= pd.Timestamp("2020-01-01")]
    rows.append(compare_four(late, slice_type="late_window", slice_value="2020_2023"))
    tail = predictions[predictions["parent_0194_abs_error_c"] >= 2.0]
    rows.append(compare_four(tail, slice_type="parent_tail", slice_value="parent_0194_abs_error_ge_2c"))
    yearly = pd.DataFrame([compare_four(group, slice_type="year", slice_value=year) for year, group in predictions.groupby(predictions["target_date"].dt.year, dropna=False)])
    return pd.DataFrame(rows), yearly


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "promotion_oriented",
        "hypothesis": "After 0194, remaining large-error days still contain station-network residual structure that can be learned by a tail-weighted model without using target-current outcomes.",
        "rationale": "0194 delivered the largest recent lift and improved tails, but parent-tail MAE remains high. 0195 showed month-only abstention is too weak, so this tests a direct tail-conditioned residual expert with zero fallback.",
        "expected_sign_and_falsification": "Expected sign is lower MAE and parent-tail MAE than 0194. Falsified if nested selection falls back to zero or selected corrections worsen 0194.",
        "novelty": {"prior_experiments": ["0194", "0195"], "difference": "Tail-weighted residual model over 0194 using the 0194 station-role feature set; not month-only governance.", "similarity_audit_path": "RESULTS.md#comparison-limitations"},
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "Features are the same cutoff-safe ISD role features used by 0194 plus known 0194 correction context.", "daily_boundary_contract": "HKO local daily maximum temperature for target local date T."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(P0194)},
        "data_sources": [
            {"source_id": "0194_parent_predictions", "paths": [rel(P0194)], "attributes": ["parent prediction", "known correction context"], "eligibility": "DEPLOYABLE_LAGGED_ONLY", "availability_proof": "0194 validator passed."},
            {"source_id": "robust_feature_matrix_isd", "paths": [rel(base.FEATURE_MATRIX_PATH)], "attributes": ["isd_* role features"], "eligibility": "DEPLOYABLE_PROVEN", "availability_proof": "Same cutoff-safe ISD feature family as 0194."},
        ],
        "stations": [{"station_id": "regional_isd_network", "role": "deployable surface-regime proxy", "attributes": ["temperature", "dew point", "pressure", "wind", "station spread", "graph modes"]}],
        "features": {"generation_rule": "Reuse 0194 role-compressed ISD features plus known 0194 correction/cap context. Fit residual target_tmax_c - parent_0194_prediction_c with tail-weighted training only inside prior windows.", "grid": MODEL_GRID, "explicit_exclusions": ["2024+ rows", "current target outcome as predictor", "current residual/error as predictor", "IGRA", "daily HKO climate predictors"]},
        "response": {"variable": "target_tmax_c - parent_0194_prediction_c", "prediction": "parent_0194_prediction_c plus clipped tail-weighted station residual correction"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official forecast on identical rows; 0194 reported as parent champion reference."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Prior-window validation must beat 0194 parent by at least 0.001 C or zero correction is selected.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P95 AE", ">3C rate", "fold/year/season/month/source/parent-tail slices", "delta_vs_0194"],
        "sample_rules": {"row_policy": "All 0194 parent rows.", "missing_policy": "No row drops; fold-local median imputation."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0194_c": 0.001, "max_fold_harm_vs_0194_c": 0.001, "no_parent_tail_harm": ">3C rate cannot exceed 0194 by more than 0.005."},
        "rejection_conditions": ["Any 2024+ row.", "Any target residual/error column admitted as predictor.", "Config selected by scored fold outcomes."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any], feature_count: int) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a tail-weighted station-network residual expert over 0194.

## One-Sentence Hypothesis

Remaining 0194 tail errors retain station-network structure that a small clipped residual expert can correct.

## Why It Is Worth Doing

0194 is the current champion and proved ISD role features matter. Parent-tail MAE remains high, and 0195 showed month-only governance is too weak. This run tests the tail lane directly while keeping zero correction as the default fallback.

## Prior Evidence And Novelty

0194 trained a general station-role correction over 0190. 0196 instead trains a second residual over 0194 with tail-weighted prior samples and stricter caps.

## Target, Horizon, And Exact Cutoff

Target is HKO Tmax at `T-24`, timezone `Asia/Hong_Kong`. Feature availability inherits 0194's cutoff-safe ISD role contract.

## Datasets, Stations, And Attributes

Inputs are validator-clean 0194 predictions and the same robust ISD role feature matrix used by 0194.

## Feature Definitions

The candidate uses `{feature_count}` predictors. Full formulas and availability rules are in `feature_definitions.csv`.

## Response And Baseline

Response is parent 0194 residual. Official raw forecast is the primary baseline; 0194 is the parent champion reference.

## Walk-Forward Design

Each fold selects one model using only prior years. If no candidate beats 0194 by `{INNER_MIN_LIFT_C}` C in prior validation, zero correction is selected.

## Acceptance And Rejection Criteria

Acceptance requires at least 0.001 C global MAE lift versus 0194 without fold or severe-tail harm.

## Reproduction Command

Run `python scripts/run_hkg_t24_0196_station_network_tail_conditioned_residual_expert.py` from the repository root.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

Rows: `{summary['n_common']}` from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Parent 0194 MAE is `{summary['parent_0194_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus 0194 is `{summary['delta_vs_0194_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'parent_0194_mae_c', 'candidate_mae_c', 'delta_vs_0194_mae_c', 'selected_config_id', 'selected_model', 'selection_reason', 'mean_abs_tail_expert_correction_c']], max_rows=20)}

## Yearly And Monthly Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'parent_0194_mae_c', 'candidate_mae_c', 'delta_vs_0194_mae_c']], max_rows=30)}

Month metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('month')][['slice_value', 'n', 'parent_0194_mae_c', 'candidate_mae_c', 'delta_vs_0194_mae_c']], max_rows=20)}

## Tail And Source Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['season', 'source', 'late_window', 'parent_tail'])][['slice_type', 'slice_value', 'n', 'parent_0194_mae_c', 'candidate_mae_c', 'delta_vs_0194_mae_c', 'parent_0194_gt3c_rate', 'candidate_gt3c_rate']], max_rows=25)}

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. Tail weights are computed only inside chronological training windows; current target error is never a predictor.

## Comparison Limitations

This is a child experiment over 0194. If it does not beat 0194 globally, it should not replace the parent even if parent-tail slices improve.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0196 tested whether station-network features can still correct the largest remaining 0194 errors when training is explicitly weighted toward historical parent-tail cases.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0194 is `{summary['delta_vs_0194_mae_c']:.6f}` C.

## Information Gain Outside Point MAE

Selected folds and tail slices show whether the 0194 residual has remaining deployable station-regime structure or whether the role-compressed parent already consumed it.

## Robustness And Uncertainty

Robustness comes from the zero-correction fallback, prior-window model selection, fold-local imputation and scaling, and small clipped residual corrections. The remaining uncertainty is that the selected tail expert is still another station-network model over the same development frame, so it must remain locked away from 2024+ confirmation data.

## Failure Diagnosis

If zero correction is selected or global MAE worsens, the remaining 0194 tail errors are not stable enough for a second station model under prior-window selection. In that case, future work should shift toward trust/shrinkage around saturated 0194 corrections rather than more additive residual modeling.

## Promotion Status

Confirmation remains locked. Development gate to 0.45 C was not reached.
""")


def main() -> None:
    created_at = utc_now()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("src", "artifacts", "logs", "diagnostics", "figures"):
        (EXP_DIR / subdir).mkdir(parents=True, exist_ok=True)
    spec = build_spec(created_at)
    write_json(EXP_DIR / "experiment_spec.json", spec)
    spec_sha = sha256_file(EXP_DIR / "experiment_spec.json")
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "spec_sha256": spec_sha, "state": "SPEC_WRITTEN_BEFORE_SCORING"})
    src_copy_path = EXP_DIR / "src" / SRC_COPY_NAME
    shutil.copy2(Path(__file__).resolve(), src_copy_path)
    frame, feature_cols, feature_defs = load_frame()
    forbidden = [c for c in feature_cols if c in {"target_tmax_c", "official_residual_c", "official_abs_error_c", "parent_0194_residual_c", "parent_0194_abs_error_c"} or "residual" in c.lower() or "abs_error" in c.lower()]
    if forbidden:
        raise RuntimeError(f"Forbidden predictors selected: {forbidden}")
    predictions, fold_metrics, selections, inner_scores = run_walk_forward(frame, feature_cols)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official")
    p0190_global = base.metric_row(predictions, "parent_0190_prediction_c", label="0190_parent")
    p0194_global = base.metric_row(predictions, "parent_0194_prediction_c", label="0194_parent")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_0190 = candidate_global["mae_c"] - p0190_global["mae_c"]
    delta_vs_0194 = candidate_global["mae_c"] - p0194_global["mae_c"]
    severe_harm_0194 = candidate_global["gt3c_rate"] - p0194_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["delta_vs_0194_mae_c"].max())
    if mae_delta <= -0.01 and delta_vs_0194 <= -0.001 and severe_harm_0194 <= 0.005 and fold_worst_delta <= 0.001:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_OVER_0194_NO_CONFIRMATION"
    elif mae_delta < 0:
        status = "COMPLETED_INFORMATION_GAIN_ONLY"
        promotion_decision = "DO_NOT_PROMOTE_YET_INFORMATION_GAIN"
    else:
        status = "COMPLETED_NULL_OR_NEGATIVE"
        promotion_decision = "DO_NOT_PROMOTE"
    common_row_hash = sha256_text("\n".join(date_text(value) for value in predictions["target_date"]))
    scoreboard = pd.DataFrame(
        [
            {"candidate_id": "official_forecast_max_c", "model_family": "baseline", "n": official_global["n"], "mae_c": official_global["mae_c"], "rmse_c": official_global["rmse_c"], "bias_c": official_global["bias_c"], "median_abs_error_c": official_global["median_abs_error_c"], "p95_abs_error_c": official_global["p95_abs_error_c"], "gt2c_rate": official_global["gt2c_rate"], "gt3c_rate": official_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": 0.0},
            {"candidate_id": "0194_parent_isd_role_proxy", "model_family": "parent_reference", "n": p0194_global["n"], "mae_c": p0194_global["mae_c"], "rmse_c": p0194_global["rmse_c"], "bias_c": p0194_global["bias_c"], "median_abs_error_c": p0194_global["median_abs_error_c"], "p95_abs_error_c": p0194_global["p95_abs_error_c"], "gt2c_rate": p0194_global["gt2c_rate"], "gt3c_rate": p0194_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": p0194_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "station_network_tail_conditioned_residual_expert", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0194 parent rows", "common_row_hash": common_row_hash}])
    correction_distribution = predictions["tail_expert_correction_c"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    correction_distribution.columns = ["statistic", "tail_expert_correction_c"]
    data_manifest = pd.DataFrame(
        [
            {"source_id": "0194_parent_predictions", "path": rel(P0194), "sha256": sha256_file(P0194), "size_bytes": P0194.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;parent prediction;known correction context", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean 0194 predictions."},
            {"source_id": "robust_feature_matrix_isd", "path": rel(base.FEATURE_MATRIX_PATH), "sha256": sha256_file(base.FEATURE_MATRIX_PATH), "size_bytes": base.FEATURE_MATRIX_PATH.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;cutoff-safe ISD station summaries", "availability_class": "DEPLOYABLE_PROVEN", "notes": "Same role-compressed ISD feature family as 0194."},
        ]
    )
    pred_cols = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c", "parent_0190_prediction_c", "parent_0194_prediction_c", "candidate_prediction_c", "parent_0194_isd_role_correction_c", "tail_expert_correction_c", "candidate_correction_c", "parent_0194_error_c", "candidate_error_c", "official_abs_error_c", "parent_0194_abs_error_c", "candidate_abs_error_c", "fold_id", "selected_config_id", "selected_model", "selected_cap_c", "selected_scale", "selected_tail_threshold_c", "selected_tail_weight", "selected_inner_delta_vs_0194_mae_c", "candidate_id", "baseline_id", "model_family", "feature_count"]
    write_parquet(EXP_DIR / "predictions.parquet", predictions[pred_cols])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "correction_distribution.csv", correction_distribution)
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_defs)
    write_csv(EXP_DIR / "artifacts" / "model_grid.csv", pd.DataFrame(MODEL_GRID))
    write_csv(EXP_DIR / "artifacts" / "fold_model_selections.csv", selections)
    write_csv(EXP_DIR / "artifacts" / "inner_selection_scores.csv", inner_scores)
    write_text(EXP_DIR / "diagnostics" / "feature_columns.txt", "\n".join(feature_cols) + "\n")
    write_text(EXP_DIR / "leakage_audit.md", f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

0196 consumes validator-clean 0194 parent predictions and cutoff-safe ISD role features. Tail weights are derived only for rows inside chronological training windows after their outcomes are known.

## Available Feature Eligibility

Allowed predictors are the 0194 role-compressed ISD features and known 0194 correction/cap context. Current-row target residuals and current-row absolute errors are explicitly rejected as predictors.

## Target And Rolling Checks

Each outer fold fits models, imputers, scalers, and config selection only on earlier target years. Scored fold outcomes are used only for scoring.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate, parent 0194, parent 0190, and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0196_station_network_tail_conditioned_residual_expert.py
```

Requires completed parent predictions from 0194. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "status": status, "created_at_utc": created_at, "target": "HKO daily Tmax T-24", "frame_id": "official_t15_pre2024_5265_rows", "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "n_candidate": int(len(predictions)), "n_common": int(len(predictions)), "baseline_id": "official_forecast_max_c", "baseline_mae_c": official_global["mae_c"], "candidate_id": PRIMARY_CANDIDATE_ID, "candidate_mae_c": candidate_global["mae_c"], "mae_delta_c": mae_delta, "candidate_rmse_c": candidate_global["rmse_c"], "candidate_bias_c": candidate_global["bias_c"], "leakage_status": "PASS", "confirmation_rows_used": 0, "owner_authorized_confirmation": False, "promotion_decision": promotion_decision, "spec_sha256": spec_sha, "code_sha256": code_sha, "data_manifest_sha256": data_manifest_sha, "common_row_hash": common_row_hash, "baseline_n": int(len(predictions)), "candidate_n": int(len(predictions)), "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45), "parent_0190_mae_c": p0190_global["mae_c"], "delta_vs_0190_mae_c": delta_vs_0190, "parent_0194_mae_c": p0194_global["mae_c"], "delta_vs_0194_mae_c": delta_vs_0194, "fold_worst_delta_vs_0194_mae_c": fold_worst_delta, "severe_gt3_rate_delta_vs_0194": severe_harm_0194, "feature_count": len(feature_cols)}
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary, len(feature_cols))
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
