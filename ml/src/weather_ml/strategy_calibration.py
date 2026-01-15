"""Calibration helper for feature sweep strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import json
import numpy as np
import pandas as pd

from weather_ml import artifacts
from weather_ml import config as config_module
from weather_ml import dataset
from weather_ml import derived_features
from weather_ml import global_normal_calibration
from weather_ml import splits

SPREAD_COL = "gefsatmos_tmp_spread_f"


@dataclass(frozen=True)
class StrategyMeta:
    strategy_id: str
    raw_model_cols_used: list[str]
    uses_spread_feature: bool
    calendar_enabled: bool
    derived_features: dict
    final_feature_columns: list[str]


@dataclass(frozen=True)
class CalibrationArtifacts:
    calibration_path: Path
    residuals_path: Path


def load_strategy_meta(run_dir: Path) -> StrategyMeta:
    meta_path = run_dir / "strategy_meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return StrategyMeta(
        strategy_id=str(payload["strategy_id"]),
        raw_model_cols_used=[str(col) for col in payload["raw_model_cols_used"]],
        uses_spread_feature=bool(payload.get("uses_spread_feature", False)),
        calendar_enabled=bool(payload.get("calendar_enabled", False)),
        derived_features=dict(payload.get("derived_features", {})),
        final_feature_columns=[str(col) for col in payload["final_feature_columns"]],
    )


def load_config_from_run(run_dir: Path):
    config_path = run_dir / "config_resolved.yaml"
    config = config_module.load_config(config_path)
    repo_root = Path(__file__).resolve().parents[3]
    return config_module.resolve_paths(config, repo_root=repo_root)


def load_sweep_strategy_run(sweep_root: Path, strategy_id: str) -> Path:
    sweep_path = sweep_root / "feature_strategy_sweep.json"
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    for entry in payload.get("strategies", []):
        if str(entry.get("strategy_id")) == strategy_id:
            return Path(entry["run_dir"])
    raise ValueError(f"Strategy {strategy_id} not found in sweep JSON.")


def build_strategy_features(
    *,
    strategy_meta: StrategyMeta,
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> pd.DataFrame:
    required = list(strategy_meta.final_feature_columns)
    base_cols = list(strategy_meta.raw_model_cols_used)
    if strategy_meta.uses_spread_feature and SPREAD_COL in target_df.columns:
        base_cols.append(SPREAD_COL)

    _ensure_columns_exist(target_df, base_cols, "calibration")
    features = target_df[base_cols].astype(float).copy()

    rowwise_needed = _filter_needed(
        required, [entry["name"] for entry in strategy_meta.derived_features.get("rowwise", [])]
    )
    if rowwise_needed:
        rowwise = derived_features.compute_rowwise_features(
            target_df, strategy_meta.raw_model_cols_used, include=rowwise_needed
        )
        features = pd.concat([features, rowwise], axis=1)

    train_fitted = strategy_meta.derived_features.get("train_fitted", [])
    if _has_fitted(train_fitted, "bias_correction"):
        bias = derived_features.fit_bias_correction(
            train_df, strategy_meta.raw_model_cols_used, y_col="actual_tmax_f"
        )
        target_bc = derived_features.apply_bias_correction(
            target_df, strategy_meta.raw_model_cols_used, bias
        )
        bc_cols = [f"{col}_bc" for col in strategy_meta.raw_model_cols_used]
        bc_needed = _filter_needed(required, bc_cols)
        if bc_needed:
            features[bc_cols] = target_bc[bc_cols].astype(float)

        bc_rowwise_needed = [
            col for col in required if col.startswith("ens_") and "_bc_" in col
        ]
        if bc_rowwise_needed:
            include = [_strip_bc_marker(col) for col in bc_rowwise_needed]
            bc_rowwise = derived_features.compute_rowwise_features(
                target_bc,
                bc_cols,
                prefix="bc",
                include=include,
            )
            features = pd.concat([features, bc_rowwise], axis=1)

    if _has_fitted(train_fitted, "reliability_weights"):
        weights = derived_features.fit_reliability_weights(
            train_df, strategy_meta.raw_model_cols_used, y_col="actual_tmax_f"
        )
        features = derived_features.apply_reliability_features(
            features, strategy_meta.raw_model_cols_used, weights
        )

    if _has_fitted(train_fitted, "stack_ridge_pred"):
        if not cv_splits:
            raise ValueError("Stacking requires CV splits; enable split.cv.")
        _, ridge_model = derived_features.fit_stack_ridge_oof(
            train_df,
            strategy_meta.raw_model_cols_used,
            y_col="actual_tmax_f",
            cv_splits=cv_splits,
            seed=seed,
        )
        features = derived_features.apply_stack_feature(
            features, strategy_meta.raw_model_cols_used, ridge_model
        )
        if "ens_wmean_6" in features.columns and "ens_wmean_minus_stack" in required:
            features["ens_wmean_minus_stack"] = (
                features["ens_wmean_6"] - features["stack_ridge_pred"]
            )
        median_name = f"ens_median_{len(strategy_meta.raw_model_cols_used)}"
        if median_name in features.columns and "stack_minus_median" in required:
            features["stack_minus_median"] = (
                features["stack_ridge_pred"] - features[median_name]
            )

    if strategy_meta.calendar_enabled:
        features = _add_calendar_features(features, target_df)

    missing = [col for col in required if col not in features.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    output = features[required].copy()
    if output.isna().any().any():
        missing_cols = output.columns[output.isna().any()].tolist()
        raise ValueError(f"NaNs detected in calibration features: {missing_cols}")
    return output


def run_strategy_calibration(
    *,
    sweep_root: Path,
    strategy_id: str,
    cal_start: date,
    cal_end: date,
    ddof: int,
    output_dir: Path | None = None,
) -> CalibrationArtifacts:
    run_dir = load_sweep_strategy_run(sweep_root, strategy_id)
    strategy_meta = load_strategy_meta(run_dir)
    config = load_config_from_run(run_dir)

    global_normal_calibration.check_calibration_window(
        cal_start=cal_start,
        cal_end=cal_end,
        train_end=config.split.train_end,
        val_end=config.split.validation.val_end,
    )

    df = dataset.load_csv(Path(config.data.csv_path))
    split = splits.filter_date_ranges(
        df,
        train_start=config.split.train_start,
        train_end=config.split.train_end,
        test_start=config.split.test_start,
        test_end=config.split.test_end,
        gap_dates=config.split.gap_dates,
        val_start=config.split.validation.val_start,
        val_end=config.split.validation.val_end,
        validation_enabled=config.split.validation.enabled,
    )
    train_df = split.train_df
    if train_df.empty:
        raise ValueError("Training split is empty; cannot compute fitted features.")

    required_cols = (
        ["station_id", "target_date_local", "asof_utc", "actual_tmax_f"]
        + list(strategy_meta.raw_model_cols_used)
    )
    if strategy_meta.uses_spread_feature and SPREAD_COL in df.columns:
        required_cols.append(SPREAD_COL)
    cal_df = global_normal_calibration.select_calibration_rows(
        df,
        cal_start=cal_start,
        cal_end=cal_end,
        station_scope=None,
        required_columns=required_cols,
    )

    cv_splits = splits.make_time_cv_splits(
        train_df,
        n_splits=config.split.cv.n_splits,
        gap_days=config.split.cv.gap_days,
    )

    X_cal = build_strategy_features(
        strategy_meta=strategy_meta,
        train_df=train_df,
        target_df=cal_df,
        cv_splits=cv_splits,
        seed=config.seeds.global_seed,
    )

    mean_model = _load_mean_model(run_dir)
    mu_hat = mean_model.predict(X_cal.to_numpy(dtype=float))
    residuals = cal_df["actual_tmax_f"].to_numpy(dtype=float) - mu_hat

    residual_stats = global_normal_calibration.compute_residual_stats(
        residuals, ddof=ddof
    )
    dataset_hash = artifacts.sha256_file(Path(config.data.csv_path))
    model_hash = artifacts.sha256_file(run_dir / "mean_model.joblib")
    dataset_id = _read_dataset_id(run_dir)

    payload = {
        "method": "strategy_normal_residual",
        "strategy_id": strategy_meta.strategy_id,
        "error_definition": "e = actual_tmax_f - mu_hat",
        "ddof": ddof,
        "n": residual_stats["n"],
        "bias_mean_error_f": residual_stats["bias_mean_error_f"],
        "sigma_std_error_f": residual_stats["sigma_std_error_f"],
        "mae_f": residual_stats["mae_f"],
        "rmse_f": residual_stats["rmse_f"],
        "residual_quantiles_f": residual_stats["residual_quantiles_f"],
        "calibration_window": {"start": cal_start.isoformat(), "end": cal_end.isoformat()},
        "model_ref": {
            "run_dir": str(run_dir),
            "mean_model_artifact": "mean_model.joblib",
            "model_hash": model_hash,
        },
        "dataset_ref": {
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "rows_used": int(residual_stats["n"]),
        },
        "created_utc": artifacts.utc_now_iso(),
    }

    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "strategy_normal_calibration.json"
    residuals_path = output_dir / "strategy_normal_residuals.csv"

    calibration_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    global_normal_calibration.write_residuals_csv(
        residuals_path,
        cal_df=cal_df,
        mu_hat=mu_hat,
        residuals=residuals,
        include_abs=True,
    )
    return CalibrationArtifacts(
        calibration_path=calibration_path, residuals_path=residuals_path
    )


def _load_mean_model(run_dir: Path):
    import joblib

    return joblib.load(run_dir / "mean_model.joblib")


def _read_dataset_id(run_dir: Path) -> str:
    dataset_id_path = run_dir / "dataset_id.txt"
    if dataset_id_path.exists():
        return dataset_id_path.read_text(encoding="utf-8").strip()
    return ""


def _strip_bc_marker(name: str) -> str:
    cleaned = name.replace("_bc_", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _filter_needed(required: list[str], candidates: Iterable[str]) -> list[str]:
    required_set = set(required)
    return [item for item in candidates if item in required_set]


def _has_fitted(train_fitted: Iterable[dict], name: str) -> bool:
    for entry in train_fitted:
        if entry.get("name") == name:
            return True
    return False


def _ensure_columns_exist(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def _add_calendar_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    dates = pd.to_datetime(df["target_date_local"])
    features["month"] = dates.dt.month.astype(int)
    features["day_of_year"] = dates.dt.dayofyear.astype(int)
    radians = 2 * np.pi * features["day_of_year"] / 365.25
    features["sin_doy"] = np.sin(radians)
    features["cos_doy"] = np.cos(radians)
    features["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    return features
