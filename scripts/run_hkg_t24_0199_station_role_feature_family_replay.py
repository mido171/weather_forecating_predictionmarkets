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
import run_hkg_t24_0196_station_network_tail_conditioned_residual_expert as exp0196


REPO_ROOT = base.REPO_ROOT
EXPERIMENTS_ROOT = base.EXPERIMENTS_ROOT
EXPERIMENT_ID = "0199"
SLUG = "station_role_feature_family_replay"
TITLE = "Station-Role Feature Family Replay Over 0196"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0199_station_role_feature_family_replay_over_0196"
P0196 = EXPERIMENTS_ROOT / "0196_station_network_tail_conditioned_residual_expert" / "predictions.parquet"
SRC_COPY_NAME = "run_0199.py"
MODEL_FOLDS = base.MODEL_FOLDS
INNER_MIN_LIFT_C = 0.001

GROUPS = ["thermal", "moisture", "pressure", "wind", "graph", "context", "thermal_moisture", "thermal_pressure", "thermal_wind"]
MODEL_GRID = [{"config_id": "cfg_00_parent_0196", "model": "parent_0196", "feature_group": "parent_0196", "cap_c": 0.0, "scale": 0.0, "tail_threshold_c": 0.0, "tail_weight": 0.0}]
for group in GROUPS:
    MODEL_GRID.extend(
        [
            {"model": "ridge", "feature_group": group, "alpha": 200.0, "cap_c": 0.08, "scale": 0.50, "tail_threshold_c": 1.5, "tail_weight": 2.0},
            {"model": "huber", "feature_group": group, "alpha": 0.0015, "epsilon": 1.35, "cap_c": 0.10, "scale": 0.50, "tail_threshold_c": 1.5, "tail_weight": 3.0},
            {"model": "hgb", "feature_group": group, "learning_rate": 0.025, "max_leaf_nodes": 7, "l2_regularization": 8.0, "max_iter": 80, "cap_c": 0.12, "scale": 0.60, "tail_threshold_c": 2.0, "tail_weight": 5.0},
        ]
    )
for idx, cfg in enumerate(MODEL_GRID):
    cfg.setdefault("config_id", f"cfg_{idx:02d}_{cfg['feature_group']}_{cfg['model']}")


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


def load_frame() -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict[str, list[str]]]:
    frame, all_features, feature_defs = exp0196.load_frame()
    p0196 = pd.read_parquet(P0196)
    p0196["target_date"] = pd.to_datetime(p0196["target_date"], errors="coerce").dt.normalize()
    p0196 = p0196[p0196["target_date"] < pd.Timestamp("2024-01-01")].copy()
    p0196 = p0196[
        [
            "target_date",
            "candidate_prediction_c",
            "candidate_correction_c",
            "tail_expert_correction_c",
            "selected_config_id",
            "selected_model",
        ]
    ].rename(
        columns={
            "candidate_prediction_c": "parent_0196_prediction_c",
            "candidate_correction_c": "parent_0196_total_correction_c",
            "tail_expert_correction_c": "parent_0196_tail_expert_correction_c",
            "selected_config_id": "parent_0196_selected_config_id",
            "selected_model": "parent_0196_selected_model",
        }
    )
    frame = frame.merge(p0196, on="target_date", how="left", validate="one_to_one")
    frame["parent_0196_error_c"] = frame["parent_0196_prediction_c"] - frame["target_tmax_c"]
    frame["parent_0196_abs_error_c"] = frame["parent_0196_error_c"].abs()
    frame["parent_0196_residual_c"] = frame["target_tmax_c"] - frame["parent_0196_prediction_c"]

    context = [c for c in all_features if c in {"month", "day_of_year", "doy_sin", "doy_cos", "source_press_archive", "source_rss_archive"} or "parent_0194" in c or "parent_0190" in c]
    groups: dict[str, list[str]] = {name: list(context) for name in GROUPS}
    for col in all_features:
        low = col.lower()
        if "temp" in low or "thermal" in low or "gradient" in low or "warm" in low:
            groups["thermal"].append(col)
            groups["thermal_moisture"].append(col)
            groups["thermal_pressure"].append(col)
            groups["thermal_wind"].append(col)
        if "dew" in low or "moist" in low or "spread" in low:
            groups["moisture"].append(col)
            groups["thermal_moisture"].append(col)
        if "pressure" in low or "hpa" in low:
            groups["pressure"].append(col)
            groups["thermal_pressure"].append(col)
        if "wind" in low or "mps" in low:
            groups["wind"].append(col)
            groups["thermal_wind"].append(col)
        if "graph" in low or "coverage" in low or "laplacian" in low:
            groups["graph"].append(col)
    for group, cols in groups.items():
        uniq = []
        for col in cols:
            if col in frame.columns and col not in uniq:
                uniq.append(col)
        groups[group] = uniq
    for col in all_features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    base.assert_pre2024(frame, "0199 model frame")
    return frame.sort_values("target_date").reset_index(drop=True), all_features, feature_defs, groups


def predict_config(train: pd.DataFrame, test: pd.DataFrame, groups: dict[str, list[str]], config: dict[str, Any]) -> np.ndarray:
    if config["model"] == "parent_0196":
        return test["parent_0196_prediction_c"].to_numpy(dtype=float)
    features = groups.get(str(config["feature_group"]), [])
    if not features:
        raise RuntimeError(f"No features for group {config['feature_group']}")
    estimator = exp0196.make_estimator(config)
    y_train = train["target_tmax_c"].to_numpy(dtype=float) - train["parent_0194_prediction_c"].to_numpy(dtype=float)
    weights = exp0196.tail_sample_weight(train, config)
    estimator.fit(train[features], y_train, model__sample_weight=weights)
    raw = estimator.predict(test[features])
    correction = float(config["scale"]) * np.clip(raw, -float(config["cap_c"]), float(config["cap_c"]))
    return test["parent_0194_prediction_c"].to_numpy(dtype=float) + correction


def inner_select(train: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[dict[str, Any], pd.DataFrame]:
    max_year = int(train["target_date"].dt.year.max())
    split_year = max(int(train["target_date"].dt.year.min()) + 2, max_year - 2)
    inner_fit = train[train["target_date"].dt.year < split_year].copy()
    inner_val = train[train["target_date"].dt.year >= split_year].copy()
    parent_cfg = MODEL_GRID[0].copy()
    parent_cfg.update({"inner_mae_c": math.nan, "inner_parent_0196_mae_c": math.nan, "inner_delta_vs_0196_mae_c": 0.0, "selection_reason": "bootstrap_or_no_lift"})
    if len(inner_fit) < 365 or len(inner_val) < 120:
        return parent_cfg, pd.DataFrame([parent_cfg])
    parent_mae = float(np.mean(np.abs(inner_val["parent_0196_prediction_c"] - inner_val["target_tmax_c"])))
    rows = []
    for cfg in MODEL_GRID:
        row = cfg.copy()
        try:
            pred = predict_config(inner_fit, inner_val, groups, row)
            mae = float(np.mean(np.abs(pred - inner_val["target_tmax_c"].to_numpy(dtype=float))))
        except Exception as exc:
            mae = math.inf
            row["fit_error"] = str(exc)
        row["inner_parent_0196_mae_c"] = parent_mae
        row["inner_mae_c"] = mae
        row["inner_delta_vs_0196_mae_c"] = mae - parent_mae
        rows.append(row)
    scores = pd.DataFrame(rows).sort_values(["inner_delta_vs_0196_mae_c", "cap_c", "scale", "config_id"]).reset_index(drop=True)
    best = scores.iloc[0].to_dict()
    if float(best["inner_delta_vs_0196_mae_c"]) <= -INNER_MIN_LIFT_C:
        best["selection_reason"] = "prior_inner_lift"
        return best, scores
    parent_cfg["inner_parent_0196_mae_c"] = parent_mae
    parent_cfg["inner_mae_c"] = parent_mae
    parent_cfg["inner_delta_vs_0196_mae_c"] = 0.0
    parent_cfg["selection_reason"] = "parent_0196_fallback_inner_lift_below_threshold"
    return parent_cfg, scores


def compare(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = base.metric_row(frame, "official_prediction_c", label="official")
    p0194 = base.metric_row(frame, "parent_0194_prediction_c", label="p0194")
    p0196 = base.metric_row(frame, "parent_0196_prediction_c", label="p0196")
    candidate = base.metric_row(frame, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    return {
        "slice_type": slice_type,
        "slice_value": str(slice_value),
        "n": candidate["n"],
        "official_mae_c": official["mae_c"],
        "parent_0194_mae_c": p0194["mae_c"],
        "parent_0196_mae_c": p0196["mae_c"],
        "candidate_mae_c": candidate["mae_c"],
        "mae_delta_c": candidate["mae_c"] - official["mae_c"],
        "delta_vs_0196_mae_c": candidate["mae_c"] - p0196["mae_c"],
        "candidate_bias_c": candidate["bias_c"],
        "parent_0196_gt3c_rate": p0196["gt3c_rate"],
        "candidate_gt3c_rate": candidate["gt3c_rate"],
        "parent_0196_p95_abs_error_c": p0196["p95_abs_error_c"],
        "candidate_p95_abs_error_c": candidate["p95_abs_error_c"],
    }


def run_walk_forward(frame: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            selected = MODEL_GRID[0].copy()
            selected.update({"inner_mae_c": math.nan, "inner_parent_0196_mae_c": math.nan, "inner_delta_vs_0196_mae_c": 0.0, "selection_reason": "first_fold_no_prior_history"})
            scores = pd.DataFrame([selected])
        else:
            selected, scores = inner_select(train, groups)
        test["candidate_prediction_c"] = predict_config(train, test, groups, selected) if len(train) >= 365 else test["parent_0196_prediction_c"].to_numpy(dtype=float)
        test["candidate_correction_c"] = test["candidate_prediction_c"] - test["official_prediction_c"]
        test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
        test["candidate_abs_error_c"] = test["candidate_error_c"].abs()
        test["official_abs_error_c"] = (test["official_prediction_c"] - test["target_tmax_c"]).abs()
        test["fold_id"] = fold_id
        test["selected_config_id"] = selected["config_id"]
        test["selected_model"] = selected["model"]
        test["selected_feature_group"] = selected["feature_group"]
        test["selected_feature_count"] = len(groups.get(str(selected["feature_group"]), []))
        metric = compare(test, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected["config_id"],
                "selected_model": selected["model"],
                "selected_feature_group": selected["feature_group"],
                "selected_feature_count": len(groups.get(str(selected["feature_group"]), [])),
                "selection_reason": selected.get("selection_reason", ""),
                "selected_inner_delta_vs_0196_mae_c": selected.get("inner_delta_vs_0196_mae_c", math.nan),
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
    predictions["model_family"] = "station_role_feature_family_replay"
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(selection_rows), pd.concat(score_rows, ignore_index=True)


def build_slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [compare(predictions, slice_type="overall", slice_value="all")]
    for season, group in predictions.groupby("season", dropna=False):
        rows.append(compare(group, slice_type="season", slice_value=season))
    for month, group in predictions.groupby("month", dropna=False):
        rows.append(compare(group, slice_type="month", slice_value=month))
    for source, group in predictions.groupby("forecast_source_family", dropna=False):
        rows.append(compare(group, slice_type="source", slice_value=source))
    late = predictions[predictions["target_date"] >= pd.Timestamp("2020-01-01")]
    rows.append(compare(late, slice_type="late_window", slice_value="2020_2023"))
    tail = predictions[predictions["parent_0196_abs_error_c"] >= 2.0]
    rows.append(compare(tail, slice_type="parent_tail", slice_value="parent_0196_abs_error_ge_2c"))
    yearly = pd.DataFrame([compare(group, slice_type="year", slice_value=year) for year, group in predictions.groupby(predictions["target_date"].dt.year, dropna=False)])
    return pd.DataFrame(rows), yearly


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "promotion_oriented",
        "hypothesis": "A restricted station-role feature family can beat the broad 0196 station-tail champion by reducing overfit while preserving the durable physical signal.",
        "rationale": "0196 is strong but broad. 0197 and 0198 showed post-hoc trust/routing layers do not improve it, so the next question is which physical feature family carries the lift and whether a cleaner restricted expert is better.",
        "expected_sign_and_falsification": "Expected sign is a lower MAE than 0196 or strong mechanism dissection. Falsified for promotion if 0196 fallback remains selected in every fold.",
        "novelty": {"prior_experiments": ["0196", "0197", "0198"], "difference": "Restricted physical feature-family replay with 0196 as selectable fallback, not another trust or routing layer.", "similarity_audit_path": "RESULTS.md#comparison-limitations"},
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "Uses the same cutoff-safe ISD role features as 0196; model selection is prior-year only.", "daily_boundary_contract": "HKO local daily maximum temperature for target local date T."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(P0196)},
        "data_sources": [{"source_id": "0196_parent_predictions", "paths": [rel(P0196)], "attributes": ["0196 champion prediction"], "eligibility": "DEPLOYABLE_LAGGED_ONLY", "availability_proof": "0196 validator passed."}, {"source_id": "robust_feature_matrix_isd", "paths": [rel(base.FEATURE_MATRIX_PATH)], "attributes": ["role-compressed ISD feature families"], "eligibility": "DEPLOYABLE_PROVEN", "availability_proof": "Same cutoff-safe ISD role feature contract as 0196."}],
        "stations": [{"station_id": "regional_isd_network", "role": "deployable surface-regime proxy", "attributes": ["temperature", "dew point", "pressure", "wind", "graph modes"]}],
        "features": {"generation_rule": "Replay tail-weighted residual experts over 0194 using restricted thermal, moisture, pressure, wind, graph, context, and paired physical feature families. Include 0196 parent as explicit fallback.", "grid": MODEL_GRID, "explicit_exclusions": ["2024+ rows", "current target outcome as predictor", "current residual/error as predictor", "IGRA", "daily HKO climate predictors"]},
        "response": {"variable": "target_tmax_c - parent_0194_prediction_c for model configs; 0196 frozen parent for fallback", "prediction": "selected restricted family expert or 0196 parent"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official forecast on identical rows; 0196 is parent champion reference."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Prior-window validation must beat 0196 by at least 0.001 C or 0196 fallback is selected.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P95 AE", ">3C rate", "fold/year/season/month/source/tail slices", "selected feature family"],
        "sample_rules": {"row_policy": "All 0196 parent rows.", "missing_policy": "No row drops; fold-local median imputation."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0196_c": 0.001, "max_fold_harm_vs_0196_c": 0.001, "no_parent_tail_harm": ">3C rate cannot exceed 0196 by more than 0.005."},
        "rejection_conditions": ["Any 2024+ row.", "Any target residual/error column admitted as predictor.", "Config selected by scored fold outcomes."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a station-role feature-family replay over 0196.

## One-Sentence Hypothesis

A restricted physical station-role family can beat or explain the broad 0196 station-tail champion.

## Why It Is Worth Doing

0196 is the current champion, while 0197 and 0198 show trust/routing layers over it do not help. The next high-value question is whether the station-network signal can be made cleaner by isolating thermal, moisture, pressure, wind, graph, or context families.

## Prior Evidence And Novelty

0196 used a broad role feature set. 0199 tests restricted feature families with 0196 as an explicit fallback, so a family must beat the champion on prior history before being used.

## Target, Horizon, And Exact Cutoff

Target is HKO Tmax at `T-24`, timezone `Asia/Hong_Kong`. Feature availability inherits the cutoff-safe ISD role contract from 0196.

## Datasets, Stations, And Attributes

Inputs are validator-clean 0196 predictions and the robust ISD role feature matrix used by 0196.

## Feature Definitions

Feature families and exact columns are recorded in `diagnostics/feature_groups.json`; column definitions are in `feature_definitions.csv`.

## Response And Baseline

Model configs predict parent 0194 residual; the fallback config uses the frozen 0196 champion. Official raw forecast is the primary baseline and 0196 is the parent reference.

## Walk-Forward Design

Each fold selects one config using only prior years. If no restricted family beats 0196 by `{INNER_MIN_LIFT_C}` C in prior validation, 0196 is selected.

## Acceptance And Rejection Criteria

Acceptance requires at least 0.001 C global MAE lift versus 0196 without fold or severe-tail harm.

## Reproduction Command

Run `python scripts/run_hkg_t24_0199_station_role_feature_family_replay.py` from the repository root.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

Rows: `{summary['n_common']}` from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Parent 0196 MAE is `{summary['parent_0196_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'selected_config_id', 'selected_model', 'selected_feature_group', 'selected_feature_count', 'selection_reason']], max_rows=20)}

## Yearly And Monthly Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=30)}

Month metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('month')][['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=20)}

## Tail And Source Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['season', 'source', 'late_window', 'parent_tail'])][['slice_type', 'slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'parent_0196_gt3c_rate', 'candidate_gt3c_rate']], max_rows=25)}

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. All fitting and feature-family selection are chronological.

## Comparison Limitations

This is a child replay over the 0196 development frame. If it does not beat 0196 globally, it should not replace the parent, but selected-family evidence remains useful.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0199 tested whether the broad 0196 station-tail signal is better captured by a restricted physical feature family.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Information Gain Outside Point MAE

The selected feature families and inner-validation tables identify which station-role mechanisms can compete with the broad champion and which are redundant.

## Robustness And Uncertainty

Robustness comes from explicit 0196 fallback, prior-year selection, fold-local preprocessing, small caps, and restricted feature sets. The uncertainty is that this is still adaptive replay on the same development corpus and must not be treated as confirmation.

## Failure Diagnosis

If 0196 fallback dominates, the broad station-tail interaction is not easily decomposed into one restricted family. If a restricted family wins one fold but fails globally, it becomes a mechanism clue rather than a promoted system.

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
    frame, all_features, feature_defs, groups = load_frame()
    forbidden = [c for c in all_features if c in {"target_tmax_c", "official_residual_c", "official_abs_error_c", "parent_0196_residual_c", "parent_0196_abs_error_c"} or "residual" in c.lower() or "abs_error" in c.lower()]
    if forbidden:
        raise RuntimeError(f"Forbidden predictors selected: {forbidden}")
    predictions, fold_metrics, selections, inner_scores = run_walk_forward(frame, groups)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official")
    p0196_global = base.metric_row(predictions, "parent_0196_prediction_c", label="p0196")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_0196 = candidate_global["mae_c"] - p0196_global["mae_c"]
    severe_harm_0196 = candidate_global["gt3c_rate"] - p0196_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["delta_vs_0196_mae_c"].max())
    if mae_delta <= -0.01 and delta_vs_0196 <= -0.001 and severe_harm_0196 <= 0.005 and fold_worst_delta <= 0.001:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_OVER_0196_NO_CONFIRMATION"
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
            {"candidate_id": "0196_parent_tail_expert", "model_family": "parent_reference", "n": p0196_global["n"], "mae_c": p0196_global["mae_c"], "rmse_c": p0196_global["rmse_c"], "bias_c": p0196_global["bias_c"], "median_abs_error_c": p0196_global["median_abs_error_c"], "p95_abs_error_c": p0196_global["p95_abs_error_c"], "gt2c_rate": p0196_global["gt2c_rate"], "gt3c_rate": p0196_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": p0196_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "station_role_feature_family_replay", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0196 parent rows", "common_row_hash": common_row_hash}])
    data_manifest = pd.DataFrame([{"source_id": "0196_parent_predictions", "path": rel(P0196), "sha256": sha256_file(P0196), "size_bytes": P0196.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;0196 prediction", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean 0196 predictions."}, {"source_id": "robust_feature_matrix_isd", "path": rel(base.FEATURE_MATRIX_PATH), "sha256": sha256_file(base.FEATURE_MATRIX_PATH), "size_bytes": base.FEATURE_MATRIX_PATH.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;cutoff-safe ISD station summaries", "availability_class": "DEPLOYABLE_PROVEN", "notes": "Same role-compressed ISD feature family as 0196, partitioned into physical groups."}])
    pred_cols = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c", "parent_0194_prediction_c", "parent_0196_prediction_c", "candidate_prediction_c", "candidate_correction_c", "parent_0196_error_c", "candidate_error_c", "official_abs_error_c", "parent_0196_abs_error_c", "candidate_abs_error_c", "fold_id", "selected_config_id", "selected_model", "selected_feature_group", "selected_feature_count", "candidate_id", "baseline_id", "model_family"]
    write_parquet(EXP_DIR / "predictions.parquet", predictions[pred_cols])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "correction_distribution.csv", predictions["candidate_correction_c"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index())
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_defs)
    write_csv(EXP_DIR / "artifacts" / "model_grid.csv", pd.DataFrame(MODEL_GRID))
    write_csv(EXP_DIR / "artifacts" / "fold_model_selections.csv", selections)
    write_csv(EXP_DIR / "artifacts" / "inner_selection_scores.csv", inner_scores)
    write_json(EXP_DIR / "diagnostics" / "feature_groups.json", groups)
    write_text(EXP_DIR / "leakage_audit.md", f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

0199 consumes validator-clean 0196 parent predictions and the cutoff-safe ISD role feature family used by 0196. All model fitting, imputation, scaling, and feature-family selection are chronological.

## Available Feature Eligibility

Allowed predictors are predeclared station-role family columns. Current target residuals and absolute errors are rejected as predictors. The 0196 parent is a frozen fallback candidate.

## Target And Rolling Checks

Each outer fold uses only earlier target years for model and config selection. Scored fold outcomes are used only for scoring.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate, parent 0196, and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0199_station_role_feature_family_replay.py
```

Requires completed parent predictions from 0196. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "status": status, "created_at_utc": created_at, "target": "HKO daily Tmax T-24", "frame_id": "official_t15_pre2024_5265_rows", "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "n_candidate": int(len(predictions)), "n_common": int(len(predictions)), "baseline_id": "official_forecast_max_c", "baseline_mae_c": official_global["mae_c"], "candidate_id": PRIMARY_CANDIDATE_ID, "candidate_mae_c": candidate_global["mae_c"], "mae_delta_c": mae_delta, "candidate_rmse_c": candidate_global["rmse_c"], "candidate_bias_c": candidate_global["bias_c"], "leakage_status": "PASS", "confirmation_rows_used": 0, "owner_authorized_confirmation": False, "promotion_decision": promotion_decision, "spec_sha256": spec_sha, "code_sha256": code_sha, "data_manifest_sha256": data_manifest_sha, "common_row_hash": common_row_hash, "baseline_n": int(len(predictions)), "candidate_n": int(len(predictions)), "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45), "parent_0196_mae_c": p0196_global["mae_c"], "delta_vs_0196_mae_c": delta_vs_0196, "fold_worst_delta_vs_0196_mae_c": fold_worst_delta, "severe_gt3_rate_delta_vs_0196": severe_harm_0196}
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
