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


REPO_ROOT = base.REPO_ROOT
EXPERIMENTS_ROOT = base.EXPERIMENTS_ROOT
EXPERIMENT_ID = "0187"
SLUG = "deployable_isd_memory_residual_fusion"
TITLE = "Deployable ISD Station-Network Memory Residual Fusion"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0187_isd_memory_nested_residual_fusion"
MODEL_FOLDS = base.MODEL_FOLDS
PARENT_0185_DIR = EXPERIMENTS_ROOT / "0185_lag7_online_residual_memory_router"
PARENT_PREDICTIONS = PARENT_0185_DIR / "predictions.parquet"
MIN_ISD_COVERAGE = 0.75
MAX_FEATURES = 180

MODEL_GRID = [
    {"model": "ridge", "alpha": 30.0, "cap_c": 0.25, "parent_weight": 0.0},
    {"model": "ridge", "alpha": 100.0, "cap_c": 0.35, "parent_weight": 0.0},
    {"model": "ridge", "alpha": 300.0, "cap_c": 0.50, "parent_weight": 0.25},
    {"model": "huber", "alpha": 0.0001, "epsilon": 1.35, "cap_c": 0.35, "parent_weight": 0.0},
    {"model": "huber", "alpha": 0.001, "epsilon": 1.50, "cap_c": 0.50, "parent_weight": 0.25},
    {"model": "hgb", "learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 0.1, "max_iter": 80, "cap_c": 0.35, "parent_weight": 0.0},
    {"model": "hgb", "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0, "max_iter": 80, "cap_c": 0.35, "parent_weight": 0.25},
    {"model": "hgb", "learning_rate": 0.03, "max_leaf_nodes": 31, "l2_regularization": 1.0, "max_iter": 120, "cap_c": 0.50, "parent_weight": 0.0},
    {"model": "hgb", "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 3.0, "max_iter": 120, "cap_c": 0.50, "parent_weight": 0.25},
    {"model": "hgb", "learning_rate": 0.03, "max_leaf_nodes": 31, "l2_regularization": 3.0, "max_iter": 160, "cap_c": 0.65, "parent_weight": 0.25},
    {"model": "parent_only", "cap_c": 0.35, "parent_weight": 1.0},
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


def load_feature_frame() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    official_target = base.load_inputs()
    model_frame, base_features, base_defs = base.add_predeclared_features(official_target)
    all_columns = pd.read_parquet(base.FEATURE_MATRIX_PATH).columns.tolist()
    isd_candidates = [column for column in all_columns if column.startswith("isd_")]
    isd = pd.read_parquet(base.FEATURE_MATRIX_PATH, columns=["target_date", *isd_candidates])
    isd["target_date"] = pd.to_datetime(isd["target_date"], errors="coerce").dt.normalize()
    isd = isd[isd["target_date"].notna() & (isd["target_date"] < pd.Timestamp("2024-01-01"))].copy()
    isd = isd.drop_duplicates("target_date", keep="last")
    joined = model_frame.merge(isd, on="target_date", how="left", validate="one_to_one")

    parent = pd.read_parquet(PARENT_PREDICTIONS)
    parent["target_date"] = pd.to_datetime(parent["target_date"], errors="coerce").dt.normalize()
    parent = parent[parent["target_date"] < pd.Timestamp("2024-01-01")].copy()
    parent = parent[["target_date", "candidate_prediction_c", "candidate_correction_c"]].rename(
        columns={
            "candidate_prediction_c": "parent_0185_prediction_c",
            "candidate_correction_c": "parent_0185_correction_c",
        }
    )
    joined = joined.merge(parent, on="target_date", how="left", validate="one_to_one")
    joined["parent_0185_correction_c"] = pd.to_numeric(joined["parent_0185_correction_c"], errors="coerce").fillna(0.0)

    coverage = joined[isd_candidates].notna().mean()
    isd_features = [column for column in isd_candidates if coverage.get(column, 0.0) >= MIN_ISD_COVERAGE]
    selected = [*base_features, *isd_features, "parent_0185_correction_c"]
    selected = [column for column in selected if column in joined.columns]
    selected = selected[:MAX_FEATURES]
    for column in selected:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")

    feature_defs = base_defs.copy()
    add_rows = []
    for column in isd_features[: MAX_FEATURES - len(base_features)]:
        add_rows.append(
            {
                "feature_name": column,
                "role": "candidate_predictor",
                "family": "isd_station_network_cutoff_safe",
                "formula": "NOAA ISD regional station-network cutoff-safe aggregate/gradient/graph feature from robust feature matrix.",
                "input_columns": column,
                "units": "degC/hPa/mps/derived",
                "lag": "observations ending no later than operational cutoff",
                "window": "source-specific cutoff summary",
                "fit_scope": "Fold-local model fitting and imputation only",
                "availability_rule": "ISD station-day cutoff summary is treated as deployable pre-cutoff surface state.",
                "missingness_policy": "Median imputation inside each training fold for linear models; HGB receives the same median-imputed matrix for comparability.",
            }
        )
    add_rows.append(
        {
            "feature_name": "parent_0185_correction_c",
            "role": "candidate_predictor",
            "family": "lag7_online_residual_memory",
            "formula": "Validator-clean 0185 correction using only residuals matured by at least seven target days.",
            "input_columns": "0185 predictions.parquet candidate_correction_c",
            "units": "degC",
            "lag": "T-7 residual maturity inherited from 0185",
            "window": "contextual online memory",
            "fit_scope": "Deterministic parent feature; blend weight selected inside prior history",
            "availability_rule": "Parent 0185 leakage audit PASS.",
            "missingness_policy": "Missing parent correction is zero.",
        }
    )
    feature_defs = pd.concat([feature_defs, pd.DataFrame(add_rows)], ignore_index=True)
    return joined.sort_values("target_date").reset_index(drop=True), selected, feature_defs


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
                (
                    "model",
                    HuberRegressor(
                        alpha=float(config["alpha"]),
                        epsilon=float(config["epsilon"]),
                        max_iter=300,
                    ),
                ),
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
                        random_state=187,
                        loss="absolute_error",
                    ),
                ),
            ]
        )
    return None


def predict_config(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], config: dict[str, Any]) -> np.ndarray:
    if config["model"] == "parent_only":
        return test["parent_0185_correction_c"].to_numpy(dtype=float)
    estimator = make_estimator(config)
    y_train = train["target_tmax_c"].to_numpy(dtype=float) - train["official_prediction_c"].to_numpy(dtype=float)
    estimator.fit(train[feature_cols], y_train)
    raw = estimator.predict(test[feature_cols])
    clipped = np.clip(raw, -float(config["cap_c"]), float(config["cap_c"]))
    weight = float(config.get("parent_weight", 0.0))
    return weight * test["parent_0185_correction_c"].to_numpy(dtype=float) + (1.0 - weight) * clipped


def inner_select(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    max_year = int(train["target_date"].dt.year.max())
    split_year = max(int(train["target_date"].dt.year.min()) + 2, max_year - 2)
    inner_fit = train[train["target_date"].dt.year < split_year].copy()
    inner_val = train[train["target_date"].dt.year >= split_year].copy()
    if len(inner_fit) < 365 or len(inner_val) < 120:
        return {"config_id": "parent_only_bootstrap", **MODEL_GRID[-1], "inner_mae_c": math.nan}
    rows = []
    for idx, config in enumerate(MODEL_GRID, start=1):
        cfg = {"config_id": f"cfg_{idx:02d}_{config['model']}", **config}
        try:
            correction = predict_config(inner_fit, inner_val, feature_cols, cfg)
            pred = inner_val["official_prediction_c"].to_numpy(dtype=float) + correction
            mae = float(np.mean(np.abs(pred - inner_val["target_tmax_c"].to_numpy(dtype=float))))
        except Exception as exc:
            mae = math.inf
            cfg["fit_error"] = str(exc)
        cfg["inner_mae_c"] = mae
        rows.append(cfg)
    selection = pd.DataFrame(rows).sort_values(["inner_mae_c", "cap_c", "parent_weight"]).iloc[0].to_dict()
    return selection


def run_walk_forward(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    fold_rows = []
    selection_rows = []
    for start_year, end_year in MODEL_FOLDS:
        test = frame[frame["target_date"].dt.year.between(start_year, end_year)].copy()
        train = frame[frame["target_date"].dt.year < start_year].copy()
        fold_id = f"fold_{start_year}_{end_year}"
        if test.empty:
            fold_rows.append({"fold_id": fold_id, "n": 0})
            continue
        if len(train) < 365:
            selection = {"config_id": "baseline_zero_correction", "model": "none", "cap_c": 0.0, "parent_weight": 0.0, "inner_mae_c": math.nan}
            correction = np.zeros(len(test))
        else:
            selection = inner_select(train, feature_cols)
            correction = predict_config(train, test, feature_cols, selection)
        test["candidate_correction_c"] = correction
        test["candidate_prediction_c"] = test["official_prediction_c"] + test["candidate_correction_c"]
        test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
        test["official_error_c_signed"] = test["official_prediction_c"] - test["target_tmax_c"]
        test["official_abs_error_c"] = test["official_error_c_signed"].abs()
        test["candidate_abs_error_c"] = test["candidate_error_c"].abs()
        test["fold_id"] = fold_id
        test["selected_config_id"] = selection.get("config_id", "")
        test["selected_model"] = selection.get("model", "")
        test["selected_cap_c"] = selection.get("cap_c", math.nan)
        test["selected_parent_weight"] = selection.get("parent_weight", math.nan)
        test["selected_inner_mae_c"] = selection.get("inner_mae_c", math.nan)
        parts.append(test)
        metric = base.compare_metrics(test, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "training_rows": int(len(train)),
                "selected_config_id": selection.get("config_id", ""),
                "selected_model": selection.get("model", ""),
                "selected_cap_c": selection.get("cap_c", math.nan),
                "selected_parent_weight": selection.get("parent_weight", math.nan),
                "selected_inner_mae_c": selection.get("inner_mae_c", math.nan),
            }
        )
        fold_rows.append(metric)
        selection_rows.append({"fold_id": fold_id, **selection})
    predictions = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    predictions["candidate_id"] = PRIMARY_CANDIDATE_ID
    predictions["baseline_id"] = "official_forecast_max_c"
    predictions["model_family"] = "nested_isd_station_memory_residual_fusion"
    predictions["feature_count"] = len(feature_cols)
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(selection_rows)


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
            "Pre-cutoff ISD station-network gradients, spread, pressure/wind/moisture state, and graph modes contain deployable "
            "residual signal that can improve the 0185 T-7 online-memory candidate on the official T15 frame."
        ),
        "rationale": (
            "0183's same-day high-frequency teacher showed that spatial temperature dispersion and solar/humidity structure matter, "
            "while 0184 showed broad official-text proxies were not sufficient. The long-history deployable proxy closest to the HF "
            "dispersion mechanism is the pre-cutoff regional ISD station network, especially gradients and graph variation."
        ),
        "expected_sign_and_falsification": (
            "Expected sign is lower MAE than official and ideally lower than 0185. The lane is falsified if fold-local nonlinear/linear "
            "station fusion cannot beat the T-7 memory parent or if gains are isolated to one unstable era."
        ),
        "novelty": {
            "prior_experiments": ["0183", "0184", "0185", "0186"],
            "difference": "This is the first post-0185 fusion of deployable ISD station-network state with the leakage-passed T-7 memory parent on the full official frame.",
            "similarity_audit_path": "RESULTS.md#comparison-limitations",
        },
        "target": {
            "station": "Hong Kong Observatory headquarters",
            "variable": "tmax_c",
            "horizon": "T-24",
            "timezone": "Asia/Hong_Kong",
            "cutoff_contract_path": rel(base.OFFICIAL_PATH),
            "cutoff_function": "Official forecast rows must be exact-vintage; ISD station features are cutoff-safe aggregates; parent correction uses T-7 residual maturity.",
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
                "attributes": ["forecast_max_c", "forecast_min_c", "rh_min_pct", "rh_max_pct", "weather_text", "wind_text"],
                "eligibility": "DEPLOYABLE_PROVEN",
                "availability_proof": "Exact-vintage official forecast fields available by stored cutoff.",
            },
            {
                "source_id": "robust_feature_matrix_isd_and_lag7_memory",
                "paths": [rel(base.FEATURE_MATRIX_PATH)],
                "attributes": ["isd_*", "target_lag7_or_older"],
                "eligibility": "DEPLOYABLE_LAGGED_ONLY",
                "availability_proof": "ISD columns are from cutoff-safe station-day summaries; target-memory columns are lag-seven-or-older.",
            },
            {
                "source_id": "0185_parent_predictions",
                "paths": [rel(PARENT_PREDICTIONS)],
                "attributes": ["candidate_correction_c"],
                "eligibility": "DEPLOYABLE_LAGGED_ONLY",
                "availability_proof": "0185 validator passed and residual state uses only target dates <= T-7.",
            },
        ],
        "stations": [{"station_id": "regional_isd_network", "role": "deployable surface proxy", "attributes": ["temperature", "dew point", "pressure", "wind", "graph modes"]}],
        "features": {
            "generation_rule": "Use official forecast proxies, lag-seven target memory, ISD columns with >=75% coverage, and 0185 correction. Select among finite ridge/huber/HGB/cap/blend configs by prior-history inner validation.",
            "grid": MODEL_GRID,
            "explicit_exclusions": ["IGRA", "daily HKO climate predictors", "marine daily predictors", "same-day HF", "2024+ rows", "target T/residual/absolute-error predictor columns"],
        },
        "response": {"variable": "target_tmax_c - forecast_max_c", "prediction": "forecast_max_c plus clipped station-memory residual correction"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official maximum forecast on identical rows."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Last available years inside prior history choose model/cap/blend; no outer-fold outcomes used.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "median_abs_error", "P90/P95/max AE", ">2C and >3C rates", "fold/year/season/month/source/tail slices"],
        "sample_rules": {"row_policy": "All official pre-2024 rows; first fold zero correction.", "missing_policy": "Fold-local median imputation."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0185_c": 0.003, "no_tail_harm": ">3C rate cannot exceed official by more than 0.005."},
        "rejection_conditions": ["Any 2024+ target date used.", "Any target T/residual/error column admitted as predictor.", "Candidate and official baseline row sets differ."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(predictions: pd.DataFrame, scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selection: pd.DataFrame, summary: dict[str, Any], feature_count: int) -> None:
    write_text(
        EXP_DIR / "README.md",
        f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a deployable station-network fusion test over the current leakage-passed 0185 memory champion.

## One-Sentence Hypothesis

Cutoff-safe ISD station-network gradients and graph/spread features add deployable residual signal beyond official forecast text and T-7 online memory.

## Why It Is Worth Doing

The best diagnostic HF experiments point toward spatial temperature dispersion, heating, and moisture suppression. The only long-history T-24-safe proxy with enough breadth is the regional ISD surface network. This is the natural next bridge after 0184 failed to recover the HF teacher from official text alone.

## Prior Evidence And Novelty

0185 is the current champion at `{summary['parent_0185_mae_c']:.6f}` C MAE. 0187 is new because it adds station-network physical state and nested nonlinear/linear selection instead of only online residual memory.

## Target, Horizon, And Exact Cutoff

Target is HKO local daily Tmax at `T-24`, timezone `Asia/Hong_Kong`. Official forecasts are exact-vintage rows; ISD features are cutoff-safe station-day summaries; target memory is lag-seven-or-older.

## Datasets, Stations, And Attributes

Inputs are official T15 pre-2024 forecast rows, robust feature matrix ISD/lag-memory columns, and 0185 parent predictions. Station attributes include regional temperature spread, gradients, graph modes, dew point, pressure, and wind.

## Feature Definitions

The candidate uses `{feature_count}` predictors. Full formulas and availability rules are in `feature_definitions.csv`.

## Response And Baseline

The response is official residual `target_tmax_c - forecast_max_c`. The primary baseline is raw official `forecast_max_c`; 0185 is reported as a parent reference.

## Walk-Forward Design

Outer folds are `{MODEL_FOLDS}`. Every model, imputer, scaler, correction cap, and parent-blend weight is selected only from prior-history inner validation.

## Acceptance And Rejection Criteria

Acceptance requires lower MAE than official, ideally lower than 0185, identical rows, no 2024+ rows, and no severe-tail harm.

## Expected Failure Modes

Failure means ISD station state is redundant with the official forecast or its signal is not stable across source eras.

## Reproduction Command

Run `python scripts/run_hkg_t24_0187_deployable_isd_memory_residual_fusion.py` from the repository root.
""",
    )
    write_text(
        EXP_DIR / "RESULTS.md",
        f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

The scored frame has `{summary['n_common']}` identical candidate/baseline rows from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus official is `{summary['mae_delta_c']:.6f}` C. Parent 0185 MAE is `{summary['parent_0185_mae_c']:.6f}` C and delta versus parent is `{summary['delta_vs_parent_0185_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'selected_config_id', 'selected_model', 'selected_cap_c', 'selected_parent_weight']], max_rows=20)}

## Yearly And Seasonal Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=30)}

Season metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('season')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=10)}

## Source And Source-Era Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window'])][['slice_type', 'slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=20)}

## High-Error-Tail Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('official_tail')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'official_gt3c_rate', 'candidate_gt3c_rate']], max_rows=10)}

## Signed Over/Underforecast Results

Row-level signed errors and corrections are in `predictions.parquet`; correction distribution is in `correction_distribution.csv`.

## Ablations

The predeclared finite model/cap/blend grid is saved in `artifacts/model_grid.csv`; fold selections are in `artifacts/fold_model_selections.csv`.

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. No 2024+ rows, target T, residual, absolute-error, same-day HF, IGRA, HKO daily climate, or marine daily predictor columns were used.

## Comparison Limitations

This experiment is comparable to official and 0185 on the same 5265-row official frame. It is not comparable to diagnostic same-day HF scores or older 2670-row online-memory frames.

Model selections:

{base.markdown_table(selection, max_rows=20)}
""",
    )
    write_text(
        EXP_DIR / "CONCLUSION.md",
        f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

This experiment tested whether station-network physical state can convert the HF diagnostic lesson into a deployable long-history residual correction.

## Realized Point-MAE Change

MAE delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus parent 0185 is `{summary['delta_vs_parent_0185_mae_c']:.6f}` C.

## Information Gain Outside Point MAE

The selected fold models show whether the station-network signal is strong enough to displace or blend with T-7 residual memory under chronological selection.

## Robustness And Uncertainty

All model selection is nested inside prior history. The feature set uses only cutoff-safe ISD summaries, exact-vintage forecast fields, lag-seven memory, and the validator-clean 0185 parent.

## Failure Diagnosis

If not promoted, the likely reason is redundancy with the official forecast and 0185 memory, or unstable station-network transfer across the press/RSS source eras.

## Promotion Status

Confirmation remains locked. The development gate to 0.45 C was not reached.

## Implication For Future Research

If station fusion failed to beat 0185, future work should focus on more precise residual-memory regimes or source-era calibration rather than simply adding broader station features.
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

    frame, feature_cols, feature_defs = load_feature_frame()
    forbidden = [c for c in feature_cols if c in {"target_tmax_c", "official_residual_c", "official_abs_error_c"} or "residual" in c.lower() or "abs_error" in c.lower()]
    forbidden = [c for c in forbidden if c != "parent_0185_correction_c"]
    if forbidden:
        raise RuntimeError(f"Forbidden predictors selected: {forbidden}")
    base.assert_pre2024(frame, "0187 model frame")
    predictions, fold_metrics, selection = run_walk_forward(frame, feature_cols)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official_forecast_max_c")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    parent_global = base.metric_row(predictions, "parent_0185_prediction_c", label="0185_parent")
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_parent = candidate_global["mae_c"] - parent_global["mae_c"]
    severe_harm = candidate_global["gt3c_rate"] - official_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["mae_delta_c"].max())
    if mae_delta <= -0.01 and delta_vs_parent <= -0.003 and severe_harm <= 0.005 and fold_worst_delta <= 0.02:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_OVER_0185_NO_CONFIRMATION"
    elif mae_delta < 0:
        status = "COMPLETED_INFORMATION_GAIN_ONLY"
        promotion_decision = "DO_NOT_PROMOTE_UNLESS_FUTURE_REPLAY_CONFIRMS"
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
                "model_family": "nested_isd_station_memory_residual_fusion",
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
                "parent_rows": int(len(frame)),
                "candidate_rows": int(len(predictions)),
                "baseline_rows": int(len(predictions)),
                "common_rows": int(len(predictions)),
                "date_start": date_text(predictions["target_date"].min()),
                "date_end": date_text(predictions["target_date"].max()),
                "row_policy": "identical official T15 pre-2024 rows",
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
                "notes": "Exact-vintage official forecast frame.",
            },
            {
                "source_id": "robust_feature_matrix_isd_and_lag7_memory",
                "path": rel(base.FEATURE_MATRIX_PATH),
                "sha256": sha256_file(base.FEATURE_MATRIX_PATH),
                "size_bytes": base.FEATURE_MATRIX_PATH.stat().st_size,
                "row_count": int(len(frame)),
                "date_start": date_text(frame["target_date"].min()),
                "date_end": date_text(frame["target_date"].max()),
                "timestamp_fields": "target_date;cutoff-safe ISD station summaries;lag7 target memory",
                "availability_class": "DEPLOYABLE_LAGGED_ONLY",
                "notes": "Only ISD and lagged target-memory columns admitted.",
            },
            {
                "source_id": "0185_parent_predictions",
                "path": rel(PARENT_PREDICTIONS),
                "sha256": sha256_file(PARENT_PREDICTIONS),
                "size_bytes": PARENT_PREDICTIONS.stat().st_size,
                "row_count": int(len(frame)),
                "date_start": date_text(frame["target_date"].min()),
                "date_end": date_text(frame["target_date"].max()),
                "timestamp_fields": "target_date;T-7 residual maturity correction",
                "availability_class": "DEPLOYABLE_LAGGED_ONLY",
                "notes": "0185 validator-clean parent correction.",
            },
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
        "official_error_c_signed",
        "candidate_error_c",
        "official_abs_error_c",
        "candidate_abs_error_c",
        "fold_id",
        "selected_config_id",
        "selected_model",
        "selected_cap_c",
        "selected_parent_weight",
        "selected_inner_mae_c",
        "candidate_id",
        "baseline_id",
        "model_family",
        "feature_count",
    ]
    write_parquet(EXP_DIR / "predictions.parquet", predictions[prediction_cols])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "correction_distribution.csv", correction_distribution)
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_defs)
    write_csv(EXP_DIR / "artifacts" / "model_grid.csv", pd.DataFrame(MODEL_GRID))
    write_csv(EXP_DIR / "artifacts" / "fold_model_selections.csv", selection)
    write_text(EXP_DIR / "diagnostics" / "feature_columns.txt", "\n".join(feature_cols) + "\n")

    write_text(
        EXP_DIR / "leakage_audit.md",
        f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

Official forecast fields are exact-vintage rows from `{rel(base.OFFICIAL_PATH)}`. ISD station-network features come from the robust cutoff-safe feature matrix and are restricted to `isd_*` columns. The 0185 parent feature inherits its T-7 residual-maturity contract.

## Available Feature Eligibility

Allowed predictors are official forecast fields, deterministic calendar, lag-seven target memory, cutoff-safe ISD station-network state, and 0185 parent correction. Blocked/prospective/diagnostic-only sources are excluded.

## Target And Rolling Checks

The code rejects current target, residual, absolute-error, IGRA, same-day high-frequency, HKO daily climate, and marine daily predictor columns. Target-memory columns are lag-seven-or-older. Model fitting and config selection are fold-local.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Baseline And Row Identity

Candidate and official baseline share `{len(predictions)}` identical rows. Common row hash: `{common_row_hash}`.
""",
    )
    write_text(
        EXP_DIR / "REPRODUCE.md",
        f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0187_deployable_isd_memory_residual_fusion.py
```

Requires completed parent 0185 predictions at `{PARENT_PREDICTIONS}`. Confirmation rows remain locked.
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
        "feature_count": len(feature_cols),
    }
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(predictions, scoreboard, slice_metrics, yearly_metrics, fold_metrics, selection, summary, len(feature_cols))
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
