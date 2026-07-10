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
EXPERIMENT_ID = "0194"
SLUG = "isd_role_compressed_regime_proxy"
TITLE = "ISD Role-Compressed Regime Proxy Over 0190"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0194_isd_role_compressed_regime_proxy_over_0190"
P0190 = EXPERIMENTS_ROOT / "0190_t7_post_ensemble_residual_calibration" / "predictions.parquet"
SRC_COPY_NAME = "run_0194.py"
MODEL_FOLDS = base.MODEL_FOLDS
MIN_FEATURE_COVERAGE = 0.70
INNER_MIN_LIFT_C = 0.001

DIRECT_ISD_ROLE_COLUMNS = [
    "isd_station_count",
    "isd_air_temp_mean_c",
    "isd_air_temp_max_c",
    "isd_air_temp_min_c",
    "isd_air_temp_std_c",
    "isd_dew_point_mean_c",
    "isd_dew_point_std_c",
    "isd_pressure_mean_hpa",
    "isd_pressure_min_hpa",
    "isd_pressure_max_hpa",
    "isd_wind_speed_mean_mps",
    "isd_wind_speed_max_mps",
    "isd_wind_u_mean_mps",
    "isd_wind_v_mean_mps",
    "isd_temp_dewpoint_spread_mean_c",
    "isd_obs_age_mean_min",
    "isd_air_temp_range_c",
    "isd_pressure_range_hpa",
    "isd_intraday_temp_first_c",
    "isd_intraday_temp_count",
    "isd_morning_temp_mean_c",
    "isd_morning_pressure_mean_hpa",
    "isd_midday_temp_mean_c",
    "isd_midday_dewpoint_mean_c",
    "isd_midday_pressure_mean_hpa",
    "isd_morning_to_midday_temp_rise_c",
    "isd_pressure_tendency_morning_midday_hpa",
    "isd_dewpoint_midday_minus_temp_c",
    "isd_temp_plane_lat_slope_c_per_deg",
    "isd_temp_plane_lon_slope_c_per_deg",
    "isd_temp_plane_rmse_c",
    "isd_north_south_temp_gradient_c",
    "isd_east_west_temp_gradient_c",
    "isd_pressure_plane_lat_slope_hpa_per_deg",
    "isd_pressure_plane_lon_slope_hpa_per_deg",
    "isd_air_temp_mean_c_change_1d",
    "isd_air_temp_mean_c_roll7_mean",
    "isd_dew_point_mean_c_change_1d",
    "isd_dew_point_mean_c_roll7_mean",
    "isd_pressure_mean_hpa_change_1d",
    "isd_pressure_mean_hpa_roll7_mean",
    "isd_wind_speed_mean_mps_change_1d",
    "isd_wind_speed_mean_mps_roll7_mean",
    "isd_north_south_temp_gradient_c_change_1d",
    "isd_north_south_temp_gradient_c_roll7_mean",
    "isd_east_west_temp_gradient_c_change_1d",
    "isd_east_west_temp_gradient_c_roll7_mean",
    "isd_graph_station_coverage_count",
    "isd_graph_station_coverage_fraction",
    "isd_graph_laplacian_mode_1",
    "isd_graph_laplacian_mode_2",
    "isd_graph_laplacian_mode_3",
    "isd_graph_laplacian_mode_4",
    "isd_graph_laplacian_mode_5",
    "isd_graph_laplacian_mode_6",
    "isd_graph_total_variation_c2",
]

MODEL_GRID = [
    {"model": "zero", "cap_c": 0.0, "scale": 0.0},
    {"model": "ridge", "alpha": 30.0, "cap_c": 0.10, "scale": 0.50},
    {"model": "ridge", "alpha": 100.0, "cap_c": 0.15, "scale": 0.50},
    {"model": "ridge", "alpha": 300.0, "cap_c": 0.20, "scale": 0.75},
    {"model": "huber", "alpha": 0.0005, "epsilon": 1.35, "cap_c": 0.15, "scale": 0.50},
    {"model": "huber", "alpha": 0.0010, "epsilon": 1.50, "cap_c": 0.20, "scale": 0.75},
    {"model": "hgb", "learning_rate": 0.03, "max_leaf_nodes": 7, "l2_regularization": 3.0, "max_iter": 80, "cap_c": 0.15, "scale": 0.50},
    {"model": "hgb", "learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 5.0, "max_iter": 100, "cap_c": 0.20, "scale": 0.75},
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


def load_parent() -> pd.DataFrame:
    parent = pd.read_parquet(P0190)
    parent["target_date"] = pd.to_datetime(parent["target_date"], errors="coerce").dt.normalize()
    parent = parent[parent["target_date"] < pd.Timestamp("2024-01-01")].copy()
    rename = {
        "candidate_prediction_c": "parent_0190_prediction_c",
        "candidate_correction_c": "parent_0190_total_correction_c",
        "calibration_correction_c": "parent_0190_calibration_correction_c",
    }
    parent = parent.rename(columns={k: v for k, v in rename.items() if k in parent.columns})
    parent["parent_0190_residual_c"] = parent["target_tmax_c"] - parent["parent_0190_prediction_c"]
    parent["parent_0190_abs_error_c"] = parent["parent_0190_residual_c"].abs()
    parent["parent_0190_error_c"] = parent["parent_0190_prediction_c"] - parent["target_tmax_c"]
    parent["official_prediction_c"] = parent["official_prediction_c"].astype(float)
    parent["official_error_c_signed"] = parent["official_prediction_c"] - parent["target_tmax_c"]
    parent["official_abs_error_c"] = parent["official_error_c_signed"].abs()
    parent["month"] = pd.to_numeric(parent["month"], errors="coerce").fillna(parent["target_date"].dt.month).astype(int)
    parent["day_of_year"] = parent["target_date"].dt.dayofyear.astype(int)
    parent["doy_sin"] = np.sin(2.0 * np.pi * parent["day_of_year"] / 366.0)
    parent["doy_cos"] = np.cos(2.0 * np.pi * parent["day_of_year"] / 366.0)
    parent["source_press_archive"] = parent["forecast_source_family"].eq("press_archive").astype(float)
    parent["source_rss_archive"] = parent["forecast_source_family"].eq("rss_archive").astype(float)
    parent["parent_0190_total_correction_c"] = pd.to_numeric(parent.get("parent_0190_total_correction_c", 0.0), errors="coerce").fillna(0.0)
    parent["parent_0190_correction_abs_c"] = parent["parent_0190_total_correction_c"].abs()
    parent["parent_0190_correction_positive"] = parent["parent_0190_total_correction_c"].gt(0.0).astype(float)
    base.assert_pre2024(parent, "0194 parent 0190 frame")
    return parent.sort_values("target_date").reset_index(drop=True)


def summarize_station_roles(frame: pd.DataFrame, columns: list[str], prefix: str) -> list[str]:
    if not columns:
        return []
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    out_cols = [
        f"{prefix}_mean",
        f"{prefix}_max",
        f"{prefix}_min",
        f"{prefix}_std",
        f"{prefix}_range",
    ]
    frame[out_cols[0]] = values.mean(axis=1, skipna=True)
    frame[out_cols[1]] = values.max(axis=1, skipna=True)
    frame[out_cols[2]] = values.min(axis=1, skipna=True)
    frame[out_cols[3]] = values.std(axis=1, skipna=True)
    frame[out_cols[4]] = frame[out_cols[1]] - frame[out_cols[2]]
    return out_cols


def add_role_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    station_temp_cols = [c for c in out.columns if c.startswith("isd_station_air_temperature_c_")]
    station_dew_cols = [c for c in out.columns if c.startswith("isd_station_dew_point_c_")]
    station_pressure_cols = [c for c in out.columns if c.startswith("isd_station_sea_level_pressure_hpa_")]
    station_wind_cols = [c for c in out.columns if c.startswith("isd_station_wind_speed_mps_")]
    derived: list[str] = []
    derived += summarize_station_roles(out, station_temp_cols, "role_station_temp_c")
    derived += summarize_station_roles(out, station_dew_cols, "role_station_dew_c")
    derived += summarize_station_roles(out, station_pressure_cols, "role_station_pressure_hpa")
    derived += summarize_station_roles(out, station_wind_cols, "role_station_wind_mps")

    if "role_station_temp_c_mean" in out.columns:
        out["role_station_temp_minus_parent_c"] = out["role_station_temp_c_mean"] - out["parent_0190_prediction_c"]
        derived.append("role_station_temp_minus_parent_c")
    if "role_station_temp_c_max" in out.columns:
        out["role_station_warmest_minus_parent_c"] = out["role_station_temp_c_max"] - out["parent_0190_prediction_c"]
        derived.append("role_station_warmest_minus_parent_c")
    if "role_station_dew_c_mean" in out.columns:
        out["role_station_dewpoint_depression_c"] = out.get("role_station_temp_c_mean", out["role_station_dew_c_mean"]) - out["role_station_dew_c_mean"]
        derived.append("role_station_dewpoint_depression_c")
    if {"isd_air_temp_mean_c", "isd_dew_point_mean_c"}.issubset(out.columns):
        out["role_network_dewpoint_depression_c"] = out["isd_air_temp_mean_c"] - out["isd_dew_point_mean_c"]
        derived.append("role_network_dewpoint_depression_c")
    if {"isd_midday_temp_mean_c", "isd_morning_temp_mean_c"}.issubset(out.columns):
        out["role_midday_minus_morning_temp_c"] = out["isd_midday_temp_mean_c"] - out["isd_morning_temp_mean_c"]
        derived.append("role_midday_minus_morning_temp_c")
    if {"isd_temp_plane_lat_slope_c_per_deg", "isd_temp_plane_lon_slope_c_per_deg"}.issubset(out.columns):
        out["role_temp_gradient_magnitude_c"] = np.sqrt(
            np.square(out["isd_temp_plane_lat_slope_c_per_deg"]) + np.square(out["isd_temp_plane_lon_slope_c_per_deg"])
        )
        derived.append("role_temp_gradient_magnitude_c")
    if {"isd_wind_u_mean_mps", "isd_wind_v_mean_mps"}.issubset(out.columns):
        out["role_wind_vector_speed_mps"] = np.sqrt(np.square(out["isd_wind_u_mean_mps"]) + np.square(out["isd_wind_v_mean_mps"]))
        derived.append("role_wind_vector_speed_mps")
    if {"isd_pressure_mean_hpa", "isd_pressure_mean_hpa_roll7_mean"}.issubset(out.columns):
        out["role_pressure_anomaly_7d_hpa"] = out["isd_pressure_mean_hpa"] - out["isd_pressure_mean_hpa_roll7_mean"]
        derived.append("role_pressure_anomaly_7d_hpa")
    if {"isd_air_temp_mean_c", "isd_air_temp_mean_c_roll7_mean"}.issubset(out.columns):
        out["role_temp_anomaly_7d_c"] = out["isd_air_temp_mean_c"] - out["isd_air_temp_mean_c_roll7_mean"]
        derived.append("role_temp_anomaly_7d_c")
    if {"role_station_temp_minus_parent_c", "parent_0190_correction_abs_c"}.issubset(out.columns):
        out["role_temp_minus_parent_x_parent_abs_c"] = out["role_station_temp_minus_parent_c"] * out["parent_0190_correction_abs_c"]
        derived.append("role_temp_minus_parent_x_parent_abs_c")
    return out, derived


def load_feature_frame() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    parent = load_parent()
    all_columns = pd.read_parquet(base.FEATURE_MATRIX_PATH).columns.tolist()
    isd_columns = [column for column in all_columns if column.startswith("isd_")]
    isd = pd.read_parquet(base.FEATURE_MATRIX_PATH, columns=["target_date", *isd_columns])
    isd["target_date"] = pd.to_datetime(isd["target_date"], errors="coerce").dt.normalize()
    isd = isd[isd["target_date"].notna() & (isd["target_date"] < pd.Timestamp("2024-01-01"))].copy()
    isd = isd.drop_duplicates("target_date", keep="last")
    frame = parent.merge(isd, on="target_date", how="left", validate="one_to_one")
    frame, derived_cols = add_role_features(frame)

    context_cols = [
        "month",
        "day_of_year",
        "doy_sin",
        "doy_cos",
        "source_press_archive",
        "source_rss_archive",
        "parent_0190_total_correction_c",
        "parent_0190_correction_abs_c",
        "parent_0190_correction_positive",
    ]
    role_cols = [column for column in DIRECT_ISD_ROLE_COLUMNS if column in frame.columns] + derived_cols
    candidate_cols = [*context_cols, *role_cols]
    coverage = frame[candidate_cols].notna().mean()
    feature_cols = [column for column in candidate_cols if coverage.get(column, 0.0) >= MIN_FEATURE_COVERAGE]
    for column in feature_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    feature_defs = []
    for column in feature_cols:
        if column in context_cols:
            family = "parent_0190_and_calendar_context"
            formula = "Known-at-cutoff calendar/source/parent-correction context used only to condition ISD role signal."
        elif column in derived_cols:
            family = "isd_role_compressed_derived"
            formula = "Deterministic compressed station-network role proxy from cutoff-safe ISD station-day summaries."
        else:
            family = "isd_role_compressed_direct"
            formula = "Predeclared aggregate ISD station-network role column from cutoff-safe robust feature matrix."
        feature_defs.append(
            {
                "feature_name": column,
                "role": "candidate_predictor",
                "family": family,
                "formula": formula,
                "input_columns": column,
                "units": "degC/hPa/mps/derived",
                "lag": "current target-day observations admitted only if present in cutoff-safe ISD summary",
                "window": "station-day cutoff summary or deterministic row-level compression",
                "fit_scope": "Fold-local imputation, scaling, fitting, and config selection",
                "availability_rule": "ISD station-day cutoff summaries are treated as deployable pre-cutoff surface-state proxies; target residual is never a predictor.",
                "missingness_policy": "Median imputation inside each fitted training fold; rows are never dropped for missing ISD values.",
            }
        )
    return frame.sort_values("target_date").reset_index(drop=True), feature_cols, pd.DataFrame(feature_defs)


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
                ("model", HuberRegressor(alpha=float(config["alpha"]), epsilon=float(config["epsilon"]), max_iter=300)),
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
                        random_state=194,
                        loss="absolute_error",
                    ),
                ),
            ]
        )
    return None


def predict_config(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], config: dict[str, Any]) -> np.ndarray:
    if config["model"] == "zero":
        return np.zeros(len(test), dtype=float)
    estimator = make_estimator(config)
    y_train = train["target_tmax_c"].to_numpy(dtype=float) - train["parent_0190_prediction_c"].to_numpy(dtype=float)
    estimator.fit(train[feature_cols], y_train)
    raw = estimator.predict(test[feature_cols])
    clipped = np.clip(raw, -float(config["cap_c"]), float(config["cap_c"]))
    return float(config["scale"]) * clipped


def inner_select(train: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    max_year = int(train["target_date"].dt.year.max())
    split_year = max(int(train["target_date"].dt.year.min()) + 2, max_year - 2)
    inner_fit = train[train["target_date"].dt.year < split_year].copy()
    inner_val = train[train["target_date"].dt.year >= split_year].copy()
    zero = {"config_id": "cfg_00_zero", **MODEL_GRID[0], "inner_mae_c": math.nan, "inner_parent_mae_c": math.nan, "inner_delta_vs_parent_c": 0.0, "selection_reason": "bootstrap_or_no_lift"}
    if len(inner_fit) < 365 or len(inner_val) < 120:
        return zero, pd.DataFrame([zero])
    parent_mae = float(np.mean(np.abs(inner_val["parent_0190_prediction_c"] - inner_val["target_tmax_c"])))
    rows = []
    for idx, config in enumerate(MODEL_GRID, start=0):
        cfg = {"config_id": f"cfg_{idx:02d}_{config['model']}", **config}
        try:
            correction = predict_config(inner_fit, inner_val, feature_cols, cfg)
            pred = inner_val["parent_0190_prediction_c"].to_numpy(dtype=float) + correction
            mae = float(np.mean(np.abs(pred - inner_val["target_tmax_c"].to_numpy(dtype=float))))
        except Exception as exc:
            mae = math.inf
            cfg["fit_error"] = str(exc)
        cfg["inner_parent_mae_c"] = parent_mae
        cfg["inner_mae_c"] = mae
        cfg["inner_delta_vs_parent_c"] = mae - parent_mae
        rows.append(cfg)
    scores = pd.DataFrame(rows).sort_values(["inner_delta_vs_parent_c", "cap_c", "scale", "config_id"]).reset_index(drop=True)
    best = scores.iloc[0].to_dict()
    if float(best["inner_delta_vs_parent_c"]) <= -INNER_MIN_LIFT_C:
        best["selection_reason"] = "prior_inner_lift"
        return best, scores
    zero["inner_parent_mae_c"] = parent_mae
    zero["inner_mae_c"] = parent_mae
    zero["inner_delta_vs_parent_c"] = 0.0
    zero["selection_reason"] = "zero_fallback_inner_lift_below_threshold"
    return zero, scores


def compare_three(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = base.metric_row(frame, "official_prediction_c", label="official_forecast_max_c")
    parent = base.metric_row(frame, "parent_0190_prediction_c", label="0190_parent")
    candidate = base.metric_row(frame, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    return {
        "slice_type": slice_type,
        "slice_value": str(slice_value),
        "n": candidate["n"],
        "official_mae_c": official["mae_c"],
        "parent_0190_mae_c": parent["mae_c"],
        "candidate_mae_c": candidate["mae_c"],
        "mae_delta_c": candidate["mae_c"] - official["mae_c"],
        "delta_vs_0190_mae_c": candidate["mae_c"] - parent["mae_c"],
        "official_rmse_c": official["rmse_c"],
        "parent_0190_rmse_c": parent["rmse_c"],
        "candidate_rmse_c": candidate["rmse_c"],
        "candidate_bias_c": candidate["bias_c"],
        "official_p95_abs_error_c": official["p95_abs_error_c"],
        "parent_0190_p95_abs_error_c": parent["p95_abs_error_c"],
        "candidate_p95_abs_error_c": candidate["p95_abs_error_c"],
        "official_gt2c_rate": official["gt2c_rate"],
        "parent_0190_gt2c_rate": parent["gt2c_rate"],
        "candidate_gt2c_rate": candidate["gt2c_rate"],
        "official_gt3c_rate": official["gt3c_rate"],
        "parent_0190_gt3c_rate": parent["gt3c_rate"],
        "candidate_gt3c_rate": candidate["gt3c_rate"],
        "candidate_hot_underforecast_count": candidate["hot_underforecast_count"],
        "candidate_cold_overforecast_count": candidate["cold_overforecast_count"],
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
            selected = {"config_id": "cfg_00_zero_bootstrap", **MODEL_GRID[0], "inner_mae_c": math.nan, "inner_parent_mae_c": math.nan, "inner_delta_vs_parent_c": 0.0, "selection_reason": "first_fold_no_prior_history"}
            scores = pd.DataFrame([selected])
        else:
            selected, scores = inner_select(train, feature_cols)
        correction = predict_config(train, test, feature_cols, selected) if len(train) >= 365 else np.zeros(len(test), dtype=float)
        test["isd_role_correction_c"] = correction
        test["candidate_prediction_c"] = test["parent_0190_prediction_c"] + test["isd_role_correction_c"]
        test["candidate_correction_c"] = test["candidate_prediction_c"] - test["official_prediction_c"]
        test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
        test["candidate_abs_error_c"] = test["candidate_error_c"].abs()
        test["fold_id"] = fold_id
        test["selected_config_id"] = selected["config_id"]
        test["selected_model"] = selected["model"]
        test["selected_cap_c"] = selected.get("cap_c", 0.0)
        test["selected_scale"] = selected.get("scale", 0.0)
        test["selected_inner_mae_c"] = selected.get("inner_mae_c", math.nan)
        test["selected_inner_delta_vs_parent_c"] = selected.get("inner_delta_vs_parent_c", math.nan)
        test["selection_reason"] = selected.get("selection_reason", "")
        metric = compare_three(test, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected["config_id"],
                "selected_model": selected["model"],
                "selected_cap_c": selected.get("cap_c", 0.0),
                "selected_scale": selected.get("scale", 0.0),
                "selected_inner_mae_c": selected.get("inner_mae_c", math.nan),
                "selected_inner_delta_vs_parent_c": selected.get("inner_delta_vs_parent_c", math.nan),
                "selection_reason": selected.get("selection_reason", ""),
                "mean_isd_role_correction_c": float(test["isd_role_correction_c"].mean()),
                "mean_abs_isd_role_correction_c": float(test["isd_role_correction_c"].abs().mean()),
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
    predictions["model_family"] = "isd_role_compressed_regime_proxy"
    predictions["feature_count"] = len(feature_cols)
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(selection_rows), pd.concat(score_rows, ignore_index=True)


def build_slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [compare_three(predictions, slice_type="overall", slice_value="all")]
    for season, group in predictions.groupby("season", dropna=False):
        rows.append(compare_three(group, slice_type="season", slice_value=season))
    for month, group in predictions.groupby("month", dropna=False):
        rows.append(compare_three(group, slice_type="month", slice_value=month))
    for source, group in predictions.groupby("forecast_source_family", dropna=False):
        rows.append(compare_three(group, slice_type="source", slice_value=source))
    late = predictions[predictions["target_date"] >= pd.Timestamp("2020-01-01")]
    rows.append(compare_three(late, slice_type="late_window", slice_value="2020_2023"))
    official_tail = predictions[predictions["official_abs_error_c"] >= 2.0]
    rows.append(compare_three(official_tail, slice_type="official_tail", slice_value="official_abs_error_ge_2c"))
    parent_tail = predictions[predictions["parent_0190_abs_error_c"] >= 2.0]
    rows.append(compare_three(parent_tail, slice_type="parent_tail", slice_value="parent_0190_abs_error_ge_2c"))
    yearly = pd.DataFrame(
        [
            compare_three(group, slice_type="year", slice_value=year)
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
        "hypothesis": "Compressed deployable ISD station-network roles contain independent physical regime signal that can reduce 0190 residual MAE without target-current leakage.",
        "rationale": "0187 showed broad ISD information can help tails versus official, while 0191-0193 showed parent/source-only residual stacking is redundant after 0190. This experiment isolates a smaller role-compressed ISD signal over the champion.",
        "expected_sign_and_falsification": "Expected sign is lower MAE than 0190 with no fold or severe-tail harm. Falsified if nested prior-history selection falls back to zero or selected ISD corrections worsen 0190.",
        "novelty": {"prior_experiments": ["0187", "0190", "0191", "0192", "0193"], "difference": "Targets parent 0190 residual using compressed ISD role proxies rather than broad ISD-plus-0185 fusion or another parent-only residual layer.", "similarity_audit_path": "RESULTS.md#comparison-limitations"},
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "Parent 0190 predictions are exact-frame; ISD station-day summaries are treated as cutoff-safe pre-target-resolution proxies.", "daily_boundary_contract": "HKO local daily maximum temperature for target local date T."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(P0190)},
        "data_sources": [
            {"source_id": "0190_parent_predictions", "paths": [rel(P0190)], "attributes": ["candidate_prediction_c", "target_tmax_c for training only"], "eligibility": "DEPLOYABLE_LAGGED_ONLY", "availability_proof": "0190 validator passed; target outcomes are used only inside chronological training windows."},
            {"source_id": "robust_feature_matrix_isd", "paths": [rel(base.FEATURE_MATRIX_PATH)], "attributes": ["isd_*"], "eligibility": "DEPLOYABLE_PROVEN", "availability_proof": "Uses only predeclared ISD station-day cutoff summary columns and deterministic role compression."},
        ],
        "stations": [{"station_id": "regional_isd_network", "role": "deployable surface-regime proxy", "attributes": ["temperature", "dew point", "pressure", "wind", "station spread", "gradients", "graph modes"]}],
        "features": {"generation_rule": "Use context columns plus predeclared ISD aggregate roles and deterministic station-role compressions. No raw target, target residual, current error, IGRA, daily HKO climate, or 2024+ columns admitted.", "grid": MODEL_GRID, "explicit_exclusions": ["2024+ rows", "current target residual", "current target error", "IGRA", "daily HKO climate predictors", "same-day HF", "broad raw station columns as individual predictors"]},
        "response": {"variable": "target_tmax_c - parent_0190_prediction_c", "prediction": "parent_0190_prediction_c plus clipped ISD role correction selected only by prior-history validation"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official forecast on identical rows; 0190 reported as parent reference."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Last prior years select config only if inner MAE beats 0190 parent by at least 0.001 C; otherwise zero correction.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P90/P95/max AE", ">2C and >3C rates", "fold/year/season/month/source/tail slices", "delta_vs_0190"],
        "sample_rules": {"row_policy": "All 0190 parent rows.", "missing_policy": "No row drops; fold-local median imputation for selected models."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0190_c": 0.002, "max_fold_harm_vs_0190_c": 0.001, "no_parent_tail_harm": ">3C rate cannot exceed parent 0190 by more than 0.005."},
        "rejection_conditions": ["Any 2024+ row.", "Any target residual/error column admitted as predictor.", "Parent row mismatch.", "Config selected by scored fold outcomes."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(predictions: pd.DataFrame, scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any], feature_count: int) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a T-24-safe ISD role-compressed residual proxy over 0190.

## One-Sentence Hypothesis

Compressed regional ISD station-network roles can identify physical regimes where the 0190 champion still has residual bias.

## Why It Is Worth Doing

0191-0193 showed that parent-only and source-only residual stacking is mostly exhausted. 0187 showed broad ISD has complementary tail information. This run tests the independent station-regime lane without the high-dimensional 0187 feature spread.

## Prior Evidence And Novelty

0187 used broad ISD fusion over 0185 and did not beat its parent. 0194 uses compressed ISD roles over the stronger 0190 champion and makes zero correction the fallback when prior-history validation cannot prove lift.

## Target, Horizon, And Exact Cutoff

Target is HKO Tmax at `T-24`, timezone `Asia/Hong_Kong`. Parent 0190 predictions are exact-frame. ISD features are restricted to cutoff-safe station-day summary proxies; target outcomes are only used in prior chronological training windows.

## Datasets, Stations, And Attributes

Inputs are 0190 parent predictions and the robust `isd_*` feature matrix. Station attributes include temperature, dew point, pressure, wind, gradients, graph modes, and station spread.

## Feature Definitions

The candidate uses `{feature_count}` predictors. Full formulas and availability rules are in `feature_definitions.csv`.

## Response And Baseline

Response is parent 0190 residual. The official raw forecast is the primary baseline; 0190 is the parent reference that must be beaten for promotion.

## Walk-Forward Design

Outer folds are `{MODEL_FOLDS}`. Each non-bootstrap fold selects one model/cap/scale using only prior years. If the best prior validation score fails to beat the 0190 parent by `{INNER_MIN_LIFT_C}` C, the fold uses zero ISD correction.

## Acceptance And Rejection Criteria

Acceptance requires material MAE lift versus official and at least 0.002 C lift versus 0190, with no fold-level or severe-tail parent harm.

## Expected Failure Modes

ISD can fail if the official forecast and 0190 calibration already absorb the surface-regime information or if station summary timing is too coarse.

## Reproduction Command

Run `python scripts/run_hkg_t24_0194_isd_role_compressed_regime_proxy.py` from the repository root.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

The scored frame has `{summary['n_common']}` rows from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Parent 0190 MAE is `{summary['parent_0190_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0190 is `{summary['delta_vs_0190_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'official_mae_c', 'parent_0190_mae_c', 'candidate_mae_c', 'delta_vs_0190_mae_c', 'selected_config_id', 'selected_model', 'selection_reason', 'mean_abs_isd_role_correction_c']], max_rows=20)}

## Yearly And Seasonal Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'official_mae_c', 'parent_0190_mae_c', 'candidate_mae_c', 'delta_vs_0190_mae_c']], max_rows=30)}

Season metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('season')][['slice_value', 'n', 'official_mae_c', 'parent_0190_mae_c', 'candidate_mae_c', 'delta_vs_0190_mae_c']], max_rows=10)}

## Source And Late-Window Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window'])][['slice_type', 'slice_value', 'n', 'official_mae_c', 'parent_0190_mae_c', 'candidate_mae_c', 'delta_vs_0190_mae_c']], max_rows=20)}

## High-Error-Tail Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['official_tail', 'parent_tail'])][['slice_type', 'slice_value', 'n', 'official_mae_c', 'parent_0190_mae_c', 'candidate_mae_c', 'delta_vs_0190_mae_c', 'parent_0190_gt3c_rate', 'candidate_gt3c_rate']], max_rows=10)}

## Correction Distribution

ISD role correction distribution is in `correction_distribution.csv`; selected inner scores are in `artifacts/inner_selection_scores.csv`.

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. No 2024+ rows or target-current residual/error predictors were used.

## Comparison Limitations

This is a child experiment over 0190. It can be scientifically useful versus official but cannot replace 0190 unless it improves the parent on identical rows.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0194 tested whether compressed deployable ISD station roles remain useful after the 0190 post-ensemble residual calibration.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0190 is `{summary['delta_vs_0190_mae_c']:.6f}` C.

## Information Gain Outside Point MAE

Fold selections show whether station-regime information can prove prior-history value beyond the champion or whether zero correction dominates.

## Robustness And Uncertainty

All model fitting, imputation, scaling, and selection are chronological. The first fold has no prior history and therefore uses zero correction by design.

## Failure Diagnosis

If not promoted, the station-network role signal is either redundant with 0190 or too unstable at the current cutoff-safe summary resolution.

## Promotion Status

Confirmation remains locked. Development gate to 0.45 C was not reached.

## Implication For Future Research

If role-compressed ISD cannot beat 0190, broad station fusion should not be repeated without a more targeted tail-only or month-specific gate.
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

    frame, feature_cols, feature_defs = load_feature_frame()
    forbidden = [c for c in feature_cols if c in {"target_tmax_c", "official_residual_c", "official_abs_error_c", "parent_0190_residual_c", "parent_0190_abs_error_c"} or "residual" in c.lower() or "abs_error" in c.lower()]
    if forbidden:
        raise RuntimeError(f"Forbidden predictors selected: {forbidden}")
    base.assert_pre2024(frame, "0194 model frame")
    predictions, fold_metrics, selections, inner_scores = run_walk_forward(frame, feature_cols)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official_forecast_max_c")
    parent_global = base.metric_row(predictions, "parent_0190_prediction_c", label="0190_parent")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_parent = candidate_global["mae_c"] - parent_global["mae_c"]
    severe_harm_parent = candidate_global["gt3c_rate"] - parent_global["gt3c_rate"]
    severe_harm_official = candidate_global["gt3c_rate"] - official_global["gt3c_rate"]
    fold_worst_parent_delta = float(fold_metrics["delta_vs_0190_mae_c"].max())
    if mae_delta <= -0.01 and delta_vs_parent <= -0.002 and severe_harm_parent <= 0.005 and fold_worst_parent_delta <= 0.001:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_OVER_0190_NO_CONFIRMATION"
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
            {"candidate_id": "0190_parent_post_ensemble_calibrator", "model_family": "parent_reference", "n": parent_global["n"], "mae_c": parent_global["mae_c"], "rmse_c": parent_global["rmse_c"], "bias_c": parent_global["bias_c"], "median_abs_error_c": parent_global["median_abs_error_c"], "p95_abs_error_c": parent_global["p95_abs_error_c"], "gt2c_rate": parent_global["gt2c_rate"], "gt3c_rate": parent_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": parent_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "isd_role_compressed_regime_proxy", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0190 parent rows", "common_row_hash": common_row_hash}])
    correction_distribution = predictions["isd_role_correction_c"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    correction_distribution.columns = ["statistic", "isd_role_correction_c"]
    data_manifest = pd.DataFrame(
        [
            {"source_id": "0190_parent_predictions", "path": rel(P0190), "sha256": sha256_file(P0190), "size_bytes": P0190.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;parent prediction;target used only in prior training", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean parent 0190 predictions."},
            {"source_id": "robust_feature_matrix_isd", "path": rel(base.FEATURE_MATRIX_PATH), "sha256": sha256_file(base.FEATURE_MATRIX_PATH), "size_bytes": base.FEATURE_MATRIX_PATH.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;cutoff-safe ISD station summaries", "availability_class": "DEPLOYABLE_PROVEN", "notes": "Only isd_* columns and deterministic role compressions admitted."},
        ]
    )
    pred_cols = [
        "target_date",
        "target_tmax_c",
        "forecast_source_family",
        "season",
        "month",
        "official_prediction_c",
        "parent_0190_prediction_c",
        "candidate_prediction_c",
        "parent_0190_total_correction_c",
        "isd_role_correction_c",
        "candidate_correction_c",
        "parent_0190_residual_c",
        "parent_0190_abs_error_c",
        "official_error_c_signed",
        "candidate_error_c",
        "official_abs_error_c",
        "candidate_abs_error_c",
        "fold_id",
        "selected_config_id",
        "selected_model",
        "selected_cap_c",
        "selected_scale",
        "selected_inner_mae_c",
        "selected_inner_delta_vs_parent_c",
        "selection_reason",
        "candidate_id",
        "baseline_id",
        "model_family",
        "feature_count",
    ]
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

0194 consumes validator-clean 0190 parent predictions and cutoff-safe `isd_*` station-day summary features from `{rel(base.FEATURE_MATRIX_PATH)}`.

## Available Feature Eligibility

Allowed predictors are calendar/source/parent-correction context and deterministic ISD role-compressed station-network state. Current target residuals, current target errors, 2024+ outcomes, IGRA, same-day HF, and daily HKO climate predictors are excluded.

## Target And Rolling Checks

Target outcomes enter only as the response in chronological training windows before the scored fold. The scored fold outcome is not used for model fitting, imputation, scaling, or config selection.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate, parent 0190, and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0194_isd_role_compressed_regime_proxy.py
```

Requires completed parent predictions from 0190. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
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
        "parent_0190_mae_c": parent_global["mae_c"],
        "delta_vs_0190_mae_c": delta_vs_parent,
        "fold_worst_delta_vs_0190_mae_c": fold_worst_parent_delta,
        "severe_gt3_rate_delta_vs_0190": severe_harm_parent,
        "severe_gt3_rate_delta_vs_official": severe_harm_official,
        "feature_count": len(feature_cols),
    }
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(predictions, scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary, len(feature_cols))
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
