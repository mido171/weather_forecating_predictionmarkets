"""Calibration helper for time feature sweep experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import json
import numpy as np
import pandas as pd

from weather_ml import artifacts
from weather_ml import config as config_module
from weather_ml import dataset
from weather_ml import global_normal_calibration
from weather_ml import splits
from weather_ml import time_feature_library as tfl
from weather_ml import time_feature_sweep as tfs
from weather_ml import validate


@dataclass(frozen=True)
class CalibrationArtifacts:
    calibration_path: Path
    residuals_path: Path


def load_experiment_run(sweep_root: Path, experiment_id: str) -> Path:
    sweep_path = sweep_root / "time_feature_sweep.json"
    if not sweep_path.exists():
        fallback = sweep_root / experiment_id
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Sweep JSON not found: {sweep_path}")
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    for entry in payload.get("experiments", []):
        if str(entry.get("experiment_id")) == experiment_id:
            return Path(entry["run_dir"])
    raise ValueError(f"Experiment {experiment_id} not found in sweep JSON.")


def load_config_from_run(run_dir: Path):
    config_path = run_dir / "config_resolved.yaml"
    config = config_module.load_config(config_path)
    repo_root = Path(__file__).resolve().parents[3]
    return config_module.resolve_paths(config, repo_root=repo_root)


def _load_mean_model(run_dir: Path):
    import joblib

    return joblib.load(run_dir / "mean_model.joblib")


def _find_experiment(experiment_id: str) -> tfs.ExperimentDefinition:
    for exp in tfs._build_experiments():
        if exp.experiment_id == experiment_id:
            return exp
    raise ValueError(f"Experiment {experiment_id} not found in registry.")


def run_experiment_calibration(
    *,
    sweep_root: Path,
    experiment_id: str,
    run_dir_override: Path | None,
    cal_start: date,
    cal_end: date,
    ddof: int,
    truth_lag: int,
    allow_overlap: bool = False,
    output_dir: Path | None = None,
) -> CalibrationArtifacts:
    run_dir = run_dir_override or load_experiment_run(sweep_root, experiment_id)
    config = load_config_from_run(run_dir)

    if not allow_overlap:
        global_normal_calibration.check_calibration_window(
            cal_start=cal_start,
            cal_end=cal_end,
            train_end=config.split.train_end,
            val_end=config.split.validation.val_end,
        )

    df = dataset.load_csv(Path(config.data.csv_path))
    rules = validate.build_rules_from_config(config)
    validate.run_all_validations(df, rules)

    df = tfl.prepare_frame(df)
    df = tfs._add_base_columns(df)

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
        raise ValueError("Training split is empty; cannot compute calibration features.")

    experiment = _find_experiment(experiment_id)
    context = tfs.ExperimentContext(
        df=df,
        train_df=train_df,
        val_df=split.val_df,
        test_df=split.test_df,
        group_key=df["station_id"],
        truth_lag=truth_lag,
        seed=config.seeds.global_seed,
    )
    derived = experiment.build_features(context)

    base_cols = tfs._base_feature_columns(df)
    feature_df = pd.concat([df[base_cols].astype(float), derived.features], axis=1)
    imputed, _ = tfs._impute_features(feature_df, train_df.index)

    required_cols = ["station_id", "target_date_local", "asof_utc", "actual_tmax_f"]
    cal_df = global_normal_calibration.select_calibration_rows(
        df,
        cal_start=cal_start,
        cal_end=cal_end,
        station_scope=None,
        required_columns=required_cols,
    )
    feature_columns = base_cols + list(derived.features.columns)
    X_cal = imputed.loc[cal_df.index, feature_columns].to_numpy(dtype=float)

    mean_model = _load_mean_model(run_dir)
    mu_hat = mean_model.predict(X_cal)
    residuals = cal_df["actual_tmax_f"].to_numpy(dtype=float) - mu_hat

    residual_stats = global_normal_calibration.compute_residual_stats(
        residuals, ddof=ddof
    )
    dataset_hash = artifacts.sha256_file(Path(config.data.csv_path))
    model_hash = artifacts.sha256_file(run_dir / "mean_model.joblib")
    dataset_id = (run_dir / "dataset_id.txt").read_text(encoding="utf-8").strip()

    payload = global_normal_calibration.build_calibration_payload(
        residual_stats=residual_stats,
        ddof=ddof,
        cal_start=cal_start,
        cal_end=cal_end,
        station_scope=None,
        run_dir=run_dir,
        mean_model_path=run_dir / "mean_model.joblib",
        model_hash=model_hash,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        rows_used=int(residual_stats["n"]),
    )
    payload["experiment_id"] = experiment_id
    payload["truth_lag_days"] = truth_lag

    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "experiment_normal_calibration.json"
    residuals_path = output_dir / "experiment_normal_residuals.csv"

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
