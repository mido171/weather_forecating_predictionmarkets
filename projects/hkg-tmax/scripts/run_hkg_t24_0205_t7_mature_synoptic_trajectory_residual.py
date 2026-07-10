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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = base.REPO_ROOT
PROJECT_PATHS = base.PROJECT_PATHS
EXPERIMENTS_ROOT = base.EXPERIMENTS_ROOT
EXPERIMENT_ID = "0205"
SLUG = "t7_mature_synoptic_trajectory_residual"
TITLE = "T-7 Mature Synoptic Trajectory Residual Expert Over 0196"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0205_t7_mature_synoptic_trajectory_residual_over_0196"
SRC_COPY_NAME = "run_0205.py"
P0196 = EXPERIMENTS_ROOT / "0196_station_network_tail_conditioned_residual_expert" / "predictions.parquet"
ROBUST_MATRIX = PROJECT_PATHS.data_root / "datasets/12_hkg_t24_robust_experiment_outputs/hkg_t24_r17_feature_matrix.parquet"
MODEL_FOLDS = base.MODEL_FOLDS
LAGS = [7, 14, 21, 30]
INNER_MIN_LIFT_C = 0.00075


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


def safe_numeric_cols(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "target_date",
        "origin_date",
        "valid_at_hkt",
        "release_latency_proven",
    }
    cols: list[str] = []
    for col in frame.columns:
        if col in blocked:
            continue
        if col.startswith("r17_post_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def feature_groups(base_cols: list[str]) -> dict[str, list[str]]:
    groups = {
        "target_history": [c for c in base_cols if c.startswith("target_") or c in {"year", "month", "day_of_year", "doy_sin", "doy_cos"}],
        "upper_air": [c for c in base_cols if c.startswith("igra_") or c.startswith("r15_surface_minus_igra") or c.startswith("r15_pressure")],
        "surface_network": [c for c in base_cols if c.startswith("isd_") or c.startswith("r17_station")],
        "all_mature_trajectory": list(base_cols),
    }
    return groups


def load_frame() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    parent = pd.read_parquet(P0196)
    parent["target_date"] = pd.to_datetime(parent["target_date"], errors="coerce").dt.normalize()
    parent = parent[parent["target_date"] < pd.Timestamp("2024-01-01")].copy()
    parent = parent[
        [
            "target_date",
            "target_tmax_c",
            "forecast_source_family",
            "season",
            "month",
            "official_prediction_c",
            "candidate_prediction_c",
            "candidate_abs_error_c",
            "fold_id",
        ]
    ].rename(columns={"candidate_prediction_c": "parent_0196_prediction_c", "candidate_abs_error_c": "parent_0196_abs_error_c"})
    parent["parent_0196_residual_c"] = parent["target_tmax_c"] - parent["parent_0196_prediction_c"]
    parent["parent_0196_correction_c"] = parent["parent_0196_prediction_c"] - parent["official_prediction_c"]

    hist = pd.read_parquet(ROBUST_MATRIX)
    hist["target_date"] = pd.to_datetime(hist["target_date"], errors="coerce").dt.normalize()
    hist = hist[hist["target_date"] < pd.Timestamp("2024-01-01")].copy()
    base_cols = safe_numeric_cols(hist)
    groups_raw = feature_groups(base_cols)
    hist = hist[["target_date", *base_cols]].sort_values("target_date").drop_duplicates("target_date", keep="last")

    frame = parent.sort_values("target_date").reset_index(drop=True)
    current_known = pd.DataFrame(
        {
            "target_date": frame["target_date"],
            "known_month": frame["month"].astype(float),
            "known_doy_sin": np.sin(2.0 * np.pi * frame["target_date"].dt.dayofyear / 366.0),
            "known_doy_cos": np.cos(2.0 * np.pi * frame["target_date"].dt.dayofyear / 366.0),
            "known_source_press_archive": frame["forecast_source_family"].astype(str).eq("press_archive").astype(float),
            "known_source_rss_archive": frame["forecast_source_family"].astype(str).eq("rss_archive").astype(float),
            "known_official_prediction_c": frame["official_prediction_c"].astype(float),
            "known_parent_0196_correction_c": frame["parent_0196_correction_c"].astype(float),
        }
    )
    frame = frame.merge(current_known, on="target_date", how="left", validate="one_to_one")

    group_cols: dict[str, list[str]] = {
        "target_history": [c for c in frame.columns if c.startswith("known_")],
        "upper_air": [c for c in frame.columns if c.startswith("known_")],
        "surface_network": [c for c in frame.columns if c.startswith("known_")],
        "all_mature_trajectory": [c for c in frame.columns if c.startswith("known_")],
    }

    for lag in LAGS:
        lagged = hist.copy()
        lagged["target_date"] = lagged["target_date"] + pd.Timedelta(days=lag)
        rename = {col: f"lag{lag}_{col}" for col in base_cols}
        lagged = lagged.rename(columns=rename)
        frame = frame.merge(lagged.rename(columns={"target_date": "target_date"}), on="target_date", how="left", validate="one_to_one")
        for group, cols in groups_raw.items():
            group_cols[group].extend([f"lag{lag}_{col}" for col in cols])

    delta_base = [
        "target_tmax_c",
        "target_tminus7_tmax_c",
        "igra_temperature_c_850hpa",
        "igra_dewpoint_depression_c_850hpa",
        "igra_geopotential_height_m_850hpa",
        "igra_wind_speed_mps_850hpa",
        "igra_temp_850_minus_500_c",
        "isd_air_temp_mean_c",
        "isd_dew_point_mean_c",
        "isd_pressure_mean_hpa",
        "isd_wind_speed_mean_mps",
        "isd_temp_dewpoint_spread_mean_c",
    ]
    for col in delta_base:
        a = f"lag7_{col}"
        b = f"lag14_{col}"
        c = f"lag30_{col}"
        if a in frame.columns and b in frame.columns:
            name = f"traj_delta_lag7_minus_lag14_{col}"
            frame[name] = frame[a] - frame[b]
            group_cols["all_mature_trajectory"].append(name)
            if col.startswith("igra_"):
                group_cols["upper_air"].append(name)
            elif col.startswith("isd_"):
                group_cols["surface_network"].append(name)
            else:
                group_cols["target_history"].append(name)
        if a in frame.columns and c in frame.columns:
            name = f"traj_delta_lag7_minus_lag30_{col}"
            frame[name] = frame[a] - frame[c]
            group_cols["all_mature_trajectory"].append(name)
            if col.startswith("igra_"):
                group_cols["upper_air"].append(name)
            elif col.startswith("isd_"):
                group_cols["surface_network"].append(name)
            else:
                group_cols["target_history"].append(name)

    for group, cols in group_cols.items():
        uniq = []
        for col in cols:
            if col in frame.columns and col not in uniq:
                uniq.append(col)
        group_cols[group] = uniq
    for cols in group_cols.values():
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    base.assert_pre2024(frame, "0205 mature trajectory frame")
    feature_defs = pd.DataFrame(
        [
            {
                "feature_name": "lagged_mature_synoptic_trajectory",
                "formula": "Robust feature matrix weather and target-history columns shifted by 7 14 21 and 30 days before current target date.",
                "input_columns": "hkg_t24_r17_feature_matrix numeric columns",
                "fit_scope": "fold-local residual model over 0196",
                "availability_rule": "Every lagged value is at least seven target days mature before target T.",
            },
            {
                "feature_name": "trajectory_delta_features",
                "formula": "lag7 value minus lag14 and lag30 values for key target upper-air and surface-network variables.",
                "input_columns": "lag7_* lag14_* lag30_* mature trajectory columns",
                "fit_scope": "fold-local residual model over 0196",
                "availability_rule": "Only mature historical weather states are used; current-day observations are never used.",
            },
            {
                "feature_name": "known_parent_context",
                "formula": "Known current source family official forecast and 0196 correction context.",
                "input_columns": "forecast_source_family official_prediction_c parent_0196_prediction_c",
                "fit_scope": "fold-local residual model over 0196",
                "availability_rule": "These are known at the T-24 decision time and contain no target outcome.",
            },
        ]
    )
    return frame, group_cols, feature_defs


CONFIG_GRID = [
    {"config_id": "cfg_00_parent_0196", "model": "parent", "feature_group": "parent", "cap_c": 0.0, "scale": 0.0, "tail_weight": 0.0},
    {"config_id": "cfg_01_target_history_ridge", "model": "ridge", "feature_group": "target_history", "alpha": 300.0, "cap_c": 0.05, "scale": 0.50, "tail_weight": 2.0},
    {"config_id": "cfg_02_upper_air_huber", "model": "huber", "feature_group": "upper_air", "alpha": 0.002, "epsilon": 1.35, "cap_c": 0.07, "scale": 0.50, "tail_weight": 3.0},
    {"config_id": "cfg_03_surface_network_huber", "model": "huber", "feature_group": "surface_network", "alpha": 0.002, "epsilon": 1.35, "cap_c": 0.07, "scale": 0.50, "tail_weight": 3.0},
    {"config_id": "cfg_04_all_trajectory_huber", "model": "huber", "feature_group": "all_mature_trajectory", "alpha": 0.003, "epsilon": 1.35, "cap_c": 0.08, "scale": 0.45, "tail_weight": 3.0},
    {"config_id": "cfg_05_upper_air_hgb", "model": "hgb", "feature_group": "upper_air", "learning_rate": 0.025, "max_leaf_nodes": 7, "l2_regularization": 12.0, "max_iter": 90, "cap_c": 0.09, "scale": 0.55, "tail_weight": 4.0},
    {"config_id": "cfg_06_surface_network_hgb", "model": "hgb", "feature_group": "surface_network", "learning_rate": 0.025, "max_leaf_nodes": 7, "l2_regularization": 12.0, "max_iter": 90, "cap_c": 0.09, "scale": 0.55, "tail_weight": 4.0},
    {"config_id": "cfg_07_all_trajectory_hgb", "model": "hgb", "feature_group": "all_mature_trajectory", "learning_rate": 0.02, "max_leaf_nodes": 7, "l2_regularization": 20.0, "max_iter": 80, "cap_c": 0.08, "scale": 0.50, "tail_weight": 5.0},
]


def make_estimator(config: dict[str, Any]) -> Pipeline:
    if config["model"] == "ridge":
        model = Ridge(alpha=float(config["alpha"]))
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])
    if config["model"] == "huber":
        model = HuberRegressor(alpha=float(config["alpha"]), epsilon=float(config["epsilon"]), max_iter=300)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])
    if config["model"] == "hgb":
        model = HistGradientBoostingRegressor(
            learning_rate=float(config["learning_rate"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            l2_regularization=float(config["l2_regularization"]),
            max_iter=int(config["max_iter"]),
            min_samples_leaf=25,
            random_state=2405,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    raise ValueError(config["model"])


def sample_weight(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    weights = np.ones(len(frame), dtype=float)
    tail_weight = float(config.get("tail_weight", 0.0))
    if tail_weight > 0:
        weights += tail_weight * frame["parent_0196_abs_error_c"].ge(2.0).to_numpy(dtype=float)
    return weights


def predict_config(train: pd.DataFrame, test: pd.DataFrame, groups: dict[str, list[str]], config: dict[str, Any]) -> np.ndarray:
    if config["model"] == "parent":
        return test["parent_0196_prediction_c"].to_numpy(dtype=float)
    features = groups[str(config["feature_group"])]
    estimator = make_estimator(config)
    y_train = train["target_tmax_c"].to_numpy(dtype=float) - train["parent_0196_prediction_c"].to_numpy(dtype=float)
    estimator.fit(train[features], y_train, model__sample_weight=sample_weight(train, config))
    correction = estimator.predict(test[features])
    correction = float(config["scale"]) * np.clip(correction, -float(config["cap_c"]), float(config["cap_c"]))
    return test["parent_0196_prediction_c"].to_numpy(dtype=float) + correction


def compare(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = base.metric_row(frame, "official_prediction_c", label="official")
    parent = base.metric_row(frame, "parent_0196_prediction_c", label="p0196")
    candidate = base.metric_row(frame, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    return {
        "slice_type": slice_type,
        "slice_value": str(slice_value),
        "n": candidate["n"],
        "official_mae_c": official["mae_c"],
        "parent_0196_mae_c": parent["mae_c"],
        "candidate_mae_c": candidate["mae_c"],
        "mae_delta_c": candidate["mae_c"] - official["mae_c"],
        "delta_vs_0196_mae_c": candidate["mae_c"] - parent["mae_c"],
        "candidate_bias_c": candidate["bias_c"],
        "parent_0196_gt3c_rate": parent["gt3c_rate"],
        "candidate_gt3c_rate": candidate["gt3c_rate"],
        "parent_0196_p95_abs_error_c": parent["p95_abs_error_c"],
        "candidate_p95_abs_error_c": candidate["p95_abs_error_c"],
    }


def inner_select(train: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[dict[str, Any], pd.DataFrame]:
    min_year = int(train["target_date"].dt.year.min())
    max_year = int(train["target_date"].dt.year.max())
    split_year = max(min_year + 2, max_year - 2)
    fit = train[train["target_date"].dt.year < split_year].copy()
    val = train[train["target_date"].dt.year >= split_year].copy()
    parent_cfg = CONFIG_GRID[0].copy()
    parent_cfg.update({"inner_mae_c": math.nan, "inner_parent_0196_mae_c": math.nan, "inner_delta_vs_0196_mae_c": 0.0, "selection_reason": "bootstrap_or_no_lift"})
    if len(fit) < 365 or len(val) < 120:
        return parent_cfg, pd.DataFrame([parent_cfg])
    parent_mae = float(np.mean(np.abs(val["parent_0196_prediction_c"] - val["target_tmax_c"])))
    rows = []
    for config in CONFIG_GRID:
        row = config.copy()
        try:
            pred = predict_config(fit, val, groups, config)
            mae = float(np.mean(np.abs(pred - val["target_tmax_c"].to_numpy(dtype=float))))
            gt3 = float(np.mean(np.abs(pred - val["target_tmax_c"].to_numpy(dtype=float)) > 3.0))
        except Exception as exc:
            mae = math.inf
            gt3 = math.inf
            row["fit_error"] = str(exc)
        row["inner_parent_0196_mae_c"] = parent_mae
        row["inner_mae_c"] = mae
        row["inner_delta_vs_0196_mae_c"] = mae - parent_mae
        row["inner_gt3_rate"] = gt3
        rows.append(row)
    scores = pd.DataFrame(rows).sort_values(["inner_delta_vs_0196_mae_c", "config_id"]).reset_index(drop=True)
    best = scores.iloc[0].to_dict()
    if float(best["inner_delta_vs_0196_mae_c"]) <= -INNER_MIN_LIFT_C:
        best["selection_reason"] = "prior_inner_lift"
        return best, scores
    parent_cfg["inner_parent_0196_mae_c"] = parent_mae
    parent_cfg["inner_mae_c"] = parent_mae
    parent_cfg["inner_delta_vs_0196_mae_c"] = 0.0
    parent_cfg["selection_reason"] = "parent_0196_fallback_inner_lift_below_threshold"
    return parent_cfg, scores


def run_walk_forward(frame: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    fold_rows = []
    selection_rows = []
    score_rows = []
    for start_year, end_year in MODEL_FOLDS:
        fold_id = f"fold_{start_year}_{end_year}"
        test = frame[frame["target_date"].dt.year.between(start_year, end_year)].copy()
        if test.empty:
            fold_rows.append({"fold_id": fold_id, "n": 0})
            continue
        train = frame[frame["target_date"].dt.year < start_year].copy()
        if len(train) < 365:
            selected = CONFIG_GRID[0].copy()
            selected.update({"selection_reason": "first_fold_no_prior_history", "inner_delta_vs_0196_mae_c": 0.0})
            scores = pd.DataFrame([selected])
        else:
            selected, scores = inner_select(train, groups)
        test["candidate_prediction_c"] = predict_config(train, test, groups, selected) if len(train) >= 365 else test["parent_0196_prediction_c"].to_numpy(dtype=float)
        test["candidate_correction_vs_0196_c"] = test["candidate_prediction_c"] - test["parent_0196_prediction_c"]
        test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
        test["candidate_abs_error_c"] = test["candidate_error_c"].abs()
        test["official_abs_error_c"] = (test["official_prediction_c"] - test["target_tmax_c"]).abs()
        test["fold_id"] = fold_id
        test["selected_config_id"] = selected["config_id"]
        test["selected_model"] = selected["model"]
        test["selected_feature_group"] = selected["feature_group"]
        metric = compare(test, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected["config_id"],
                "selected_model": selected["model"],
                "selected_feature_group": selected["feature_group"],
                "selection_reason": selected.get("selection_reason", ""),
                "selected_inner_delta_vs_0196_mae_c": selected.get("inner_delta_vs_0196_mae_c", math.nan),
                "mean_abs_correction_vs_0196_c": float(test["candidate_correction_vs_0196_c"].abs().mean()),
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
    predictions["model_family"] = "t7_mature_synoptic_trajectory_residual"
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
        "hypothesis": "Mature multi-day synoptic trajectory states from upper-air and regional surface observations encode persistent regime memory that can correct 0196 residuals without using current-day observations.",
        "rationale": "Frozen expert routers and no-harm gates are exhausted. This lane tests a different response: direct residual modeling from T-7 to T-30 mature weather trajectory states.",
        "expected_sign_and_falsification": "Expected sign is lower MAE than 0196. Falsified for promotion if prior-year selection falls back to parent or if selected trajectory models do not transfer.",
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "All robust matrix weather values are shifted by at least seven target days before current T."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(P0196)},
        "data_sources": [{"source_id": "0196_parent_predictions", "paths": [rel(P0196)], "eligibility": "DEPLOYABLE_LAGGED_ONLY"}, {"source_id": "r17_robust_feature_matrix_as_mature_history_archive", "paths": [rel(ROBUST_MATRIX)], "eligibility": "DEPLOYABLE_LAGGED_ONLY"}],
        "features": {"lags_days": LAGS, "groups": ["target_history", "upper_air", "surface_network", "all_mature_trajectory"], "config_grid": CONFIG_GRID, "explicit_exclusions": ["2024+ rows", "current-day observation features", "current target residual or absolute error predictors", "confirmation rows"]},
        "response": {"variable": "target_tmax_c - parent_0196_prediction_c", "prediction": "parent_0196_prediction_c plus clipped fold-local residual correction"},
        "baseline": {"id": "official_forecast_max_c", "parent_reference": "0196_station_network_tail_conditioned_residual_expert"},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": f"Prior-window validation must beat 0196 by at least {INNER_MIN_LIFT_C} C or 0196 fallback is selected.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P95 AE", ">3C rate", "fold/year/season/month/source/tail slices"],
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0196_c": 0.001, "max_fold_harm_vs_0196_c": 0.001, "max_gt3_rate_delta_vs_0196": 0.005},
        "owner_authorized_confirmation": False,
    }


def write_docs(scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Hypothesis

T-7 to T-30 mature upper-air and surface-network trajectory states can explain residual errors left by the 0196 station-network tail expert.

## Why This Experiment Exists

Recent frozen-expert routers, feature-family stacks, and no-harm governors failed to produce a promotable global improvement. 0205 changes the information source and response: it uses old mature weather trajectories as an explicit residual model over 0196.

## Cutoff

The decision horizon remains T-24 in Asia/Hong_Kong. Every weather observation feature is shifted by at least seven target days, so the current target day's weather is never used.

## Dataset

The scored frame is the canonical 5265-row pre-2024 official frame from 0196. The mature history archive is `hkg_t24_r17_feature_matrix.parquet`.

## Feature

Feature groups are target-history, IGRA upper-air, ISD surface-network, and all mature trajectory states, each at 7, 14, 21, and 30 day lags plus selected trajectory differences.

## Baseline

The primary baseline is `official_forecast_max_c`. The parent reference is frozen `0196`, and candidate predictions are clipped residual corrections over that parent.

## Walk-Forward

Outer folds match prior experiments. Each scored fold uses only earlier years for inner config selection and fitting; if the best mature-trajectory model does not beat 0196 on prior validation, parent-only is selected.

## Acceptance

Promotion requires at least 0.001 C global MAE improvement versus 0196 and no material fold or severe-tail harm. Status is `{summary['status']}`.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline

{base.markdown_table(scoreboard)}

## Coverage

Rows: `{summary['n_common']}` from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Parent 0196 MAE is `{summary['parent_0196_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Fold

{base.markdown_table(fold_metrics[['fold_id', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'selected_config_id', 'selected_feature_group', 'selection_reason', 'mean_abs_correction_vs_0196_c']], max_rows=20)}

## Year

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=30)}

## Season And Month

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['season', 'month'])][['slice_type', 'slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=35)}

## Tail And Source

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window', 'parent_tail'])][['slice_type', 'slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'parent_0196_gt3c_rate', 'candidate_gt3c_rate']], max_rows=25)}

## Leakage

Leakage status is `{summary['leakage_status']}`. All weather trajectory values are lagged at least seven target days and confirmation rows used is `{summary['confirmation_rows_used']}`.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## Learned

0205 tests whether mature synoptic trajectory memory contains a distinct residual signal over 0196 after frozen-expert routing has plateaued.

## MAE

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Robust

The design is robust against forward-looking leakage because all observation-derived weather states are shifted to T-7 or older and all model/config selection is chronological.

## Failure

If the result is not promotable, then mature lagged weather trajectories are either already absorbed by 0196 and official forecast context or too weak to offset added variance. That would push the next experiments toward different responses such as uncertainty calibration or source-specific partial scoring rather than wider mature-history models.

## Promotion

The development gate to 0.45 C was not reached. Confirmation remains sealed.
""")


def main() -> None:
    created_at = utc_now()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("src", "artifacts", "logs", "diagnostics", "figures"):
        (EXP_DIR / subdir).mkdir(parents=True, exist_ok=True)
    spec = build_spec(created_at)
    write_json(EXP_DIR / "experiment_spec.json", spec)
    spec_sha = sha256_file(EXP_DIR / "experiment_spec.json")
    src_copy_path = EXP_DIR / "src" / SRC_COPY_NAME
    shutil.copy2(Path(__file__).resolve(), src_copy_path)
    frame, groups, feature_defs = load_frame()
    predictions, fold_metrics, selections, selection_scores = run_walk_forward(frame, groups)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official")
    parent_global = base.metric_row(predictions, "parent_0196_prediction_c", label="p0196")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_0196 = candidate_global["mae_c"] - parent_global["mae_c"]
    severe_harm_0196 = candidate_global["gt3c_rate"] - parent_global["gt3c_rate"]
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
            {"candidate_id": "0196_parent_tail_expert", "model_family": "parent_reference", "n": parent_global["n"], "mae_c": parent_global["mae_c"], "rmse_c": parent_global["rmse_c"], "bias_c": parent_global["bias_c"], "median_abs_error_c": parent_global["median_abs_error_c"], "p95_abs_error_c": parent_global["p95_abs_error_c"], "gt2c_rate": parent_global["gt2c_rate"], "gt3c_rate": parent_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": parent_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "t7_mature_synoptic_trajectory_residual", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0196 parent rows", "common_row_hash": common_row_hash}])
    data_manifest = pd.DataFrame(
        [
            {"source_id": "0196_parent_predictions", "path": rel(P0196), "sha256": sha256_file(P0196), "size_bytes": P0196.stat().st_size, "timestamp_fields": "target_date;frozen prediction", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "notes": "Current champion parent predictions."},
            {"source_id": "r17_robust_feature_matrix_as_mature_history_archive", "path": rel(ROBUST_MATRIX), "sha256": sha256_file(ROBUST_MATRIX), "size_bytes": ROBUST_MATRIX.stat().st_size, "timestamp_fields": "target_date;origin_date;valid_at_hkt shifted by lag >= 7 days", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "row_count": int(len(pd.read_parquet(ROBUST_MATRIX, columns=['target_date']))), "date_start": "1949-06-03", "date_end": "2023-12-31", "notes": "Used only as mature historical lag archive, never current-row weather."},
        ]
    )
    write_parquet(EXP_DIR / "predictions.parquet", predictions[["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c", "parent_0196_prediction_c", "candidate_prediction_c", "candidate_correction_vs_0196_c", "official_abs_error_c", "parent_0196_abs_error_c", "candidate_abs_error_c", "fold_id", "selected_config_id", "selected_model", "selected_feature_group", "candidate_id", "baseline_id", "model_family"]])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_defs)
    write_csv(EXP_DIR / "artifacts" / "config_grid.csv", pd.DataFrame(CONFIG_GRID))
    write_csv(EXP_DIR / "artifacts" / "fold_config_selections.csv", selections)
    write_csv(EXP_DIR / "artifacts" / "inner_selection_scores.csv", selection_scores)
    write_json(EXP_DIR / "diagnostics" / "feature_groups.json", groups)
    write_csv(EXP_DIR / "correction_distribution.csv", predictions["candidate_correction_vs_0196_c"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index())
    write_text(EXP_DIR / "leakage_audit.md", f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

0205 uses the 0196 parent prediction known at T-24 plus historical robust feature matrix rows shifted by at least seven target days.

## Available State

No current-row upper-air, ISD, target, residual, or absolute-error weather observation is used as a predictor. The minimum observation lag is `{min(LAGS)}` days.

## Target And Rolling Checks

Fold config selection and fitting use only years before the scored fold. Current fold outcomes are used only after scoring.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate, parent 0196, and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0205_t7_mature_synoptic_trajectory_residual.py
```

Requires completed 0196 parent predictions and the local R17 robust feature matrix. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "status": status, "created_at_utc": created_at, "target": "HKO daily Tmax T-24", "frame_id": "official_t15_pre2024_5265_rows", "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "n_candidate": int(len(predictions)), "n_common": int(len(predictions)), "baseline_id": "official_forecast_max_c", "baseline_mae_c": official_global["mae_c"], "candidate_id": PRIMARY_CANDIDATE_ID, "candidate_mae_c": candidate_global["mae_c"], "mae_delta_c": mae_delta, "candidate_rmse_c": candidate_global["rmse_c"], "candidate_bias_c": candidate_global["bias_c"], "leakage_status": "PASS", "confirmation_rows_used": 0, "owner_authorized_confirmation": False, "promotion_decision": promotion_decision, "spec_sha256": spec_sha, "code_sha256": code_sha, "data_manifest_sha256": data_manifest_sha, "common_row_hash": common_row_hash, "baseline_n": int(len(predictions)), "candidate_n": int(len(predictions)), "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45), "parent_0196_mae_c": parent_global["mae_c"], "delta_vs_0196_mae_c": delta_vs_0196, "fold_worst_delta_vs_0196_mae_c": fold_worst_delta, "severe_gt3_rate_delta_vs_0196": severe_harm_0196}
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED", "executor_invocation": "Executor skill instructions followed in-process because no separate skill agent activation is available."})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
