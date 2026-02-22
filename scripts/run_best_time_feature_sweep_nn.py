from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from weather_ml import config as config_module
from weather_ml import dataset
from weather_ml import splits
from weather_ml import time_feature_library as tfl
from weather_ml import time_feature_sweep as tfs
from weather_ml import utils_seed
from weather_ml import validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the best time feature sweep experiment with an MLP regressor."
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/time_feature_sweep",
        help="Root directory containing time_feature_sweep runs.",
    )
    parser.add_argument(
        "--truth-lag",
        type=int,
        default=2,
        help="Truth lag in days for truth-based features (default: 2).",
    )
    return parser


def _iter_sweep_paths(root: Path) -> list[Path]:
    if root.is_file() and root.name == "time_feature_sweep.json":
        return [root]
    sweep_json = root / "time_feature_sweep.json"
    if sweep_json.exists():
        return [sweep_json]
    return sorted(root.glob("*/time_feature_sweep.json"))


def _best_entry(payload: dict) -> dict | None:
    leaderboard = payload.get("leaderboard_test_mae") or []
    for entry in leaderboard:
        if str(entry.get("experiment_id")) != "BASE":
            return entry
    best = None
    for exp in payload.get("experiments", []):
        exp_id = str(exp.get("experiment_id"))
        if exp_id == "BASE":
            continue
        metrics = (exp.get("metrics") or {}).get("test") or {}
        value = metrics.get("mae")
        if value is None:
            continue
        if best is None or value < best["value"]:
            best = {"experiment_id": exp_id, "value": float(value)}
    return best


def _resolve_run_dir(payload: dict, sweep_root: Path, experiment_id: str) -> Path:
    for exp in payload.get("experiments", []):
        if str(exp.get("experiment_id")) == experiment_id:
            run_dir = Path(exp["run_dir"])
            return run_dir if run_dir.is_absolute() else sweep_root / run_dir
    fallback = sweep_root / experiment_id
    if fallback.exists():
        return fallback
    raise ValueError(f"Run dir not found for experiment {experiment_id}")


def _load_config(run_dir: Path):
    config_path = run_dir / "config_resolved.yaml"
    config = config_module.load_config(config_path)
    repo_root = tfs._resolve_repo_root()
    return config_module.resolve_paths(config, repo_root=repo_root)


def _find_experiment(experiment_id: str) -> tfs.ExperimentDefinition:
    for exp in tfs._build_experiments():
        if exp.experiment_id == experiment_id:
            return exp
    raise ValueError(f"Experiment {experiment_id} not found in registry.")


def _build_mlp(seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        random_state=seed,
    )


def main() -> int:
    args = build_parser().parse_args()
    artifacts_root = Path(args.artifacts_root)
    sweep_paths = _iter_sweep_paths(artifacts_root)
    if not sweep_paths:
        raise FileNotFoundError(f"No sweep JSON found under {artifacts_root}")

    best = None
    best_payload = None
    best_sweep_root = None
    for sweep_path in sweep_paths:
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        entry = _best_entry(payload)
        if entry is None:
            continue
        if best is None or entry["value"] < best["value"]:
            best = entry
            best_payload = payload
            best_sweep_root = sweep_path.parent

    if best is None or best_payload is None or best_sweep_root is None:
        raise ValueError("No valid experiments found in sweep payloads.")

    experiment_id = str(best["experiment_id"])
    run_dir = _resolve_run_dir(best_payload, best_sweep_root, experiment_id)
    config = _load_config(run_dir)
    tfs._apply_model_cols(config)

    utils_seed.set_global_determinism(
        config.seeds.global_seed, single_thread=config.seeds.force_single_thread
    )

    df = dataset.load_csv(Path(config.data.csv_path))
    rules = validate.build_rules_from_config(config)
    try:
        validate.run_all_validations(df, rules)
    except ValueError as exc:
        message = str(exc)
        if "Unexpected columns present" in message:
            print(f"Validation warning (ignored): {message}")
        else:
            raise

    df = tfl.prepare_frame(df)
    pre_split = splits.filter_date_ranges(
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
    df = tfs._impute_base_columns(df, pre_split.train_df)
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
    val_df = split.val_df
    test_df = split.test_df

    experiment = _find_experiment(experiment_id)
    context = tfs.ExperimentContext(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        group_key=df["station_id"],
        truth_lag=int(args.truth_lag),
        seed=config.seeds.global_seed,
    )
    derived = experiment.build_features(context)

    base_cols = tfs._base_feature_columns(df)
    base_features = df[base_cols].astype(float)
    feature_df = pd.concat([base_features, derived.features], axis=1)
    imputed, _ = tfs._impute_features(feature_df, train_df.index)

    train_features = imputed.loc[train_df.index].astype(float)
    val_features = imputed.loc[val_df.index].astype(float)
    test_features = imputed.loc[test_df.index].astype(float)

    X_train = train_features.to_numpy(dtype=float)
    y_train = train_df["actual_tmax_f"].to_numpy(dtype=float)
    X_val = val_features.to_numpy(dtype=float)
    y_val = val_df["actual_tmax_f"].to_numpy(dtype=float)
    X_test = test_features.to_numpy(dtype=float)
    y_test = test_df["actual_tmax_f"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) else np.empty((0, X_train.shape[1]))

    model_train = _build_mlp(config.seeds.global_seed)
    model_train.fit(X_train_scaled, y_train)
    mu_train = model_train.predict(X_train_scaled)
    mu_val = model_train.predict(X_val_scaled) if len(X_val) else np.array([])

    train_full_df = (
        pd.concat([train_df, val_df], ignore_index=True)
        if not val_df.empty
        else train_df.copy()
    )
    full_features = pd.concat([train_features, val_features], ignore_index=True)
    y_train_full = train_full_df["actual_tmax_f"].to_numpy(dtype=float)

    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(full_features.to_numpy(dtype=float))
    X_test_scaled = scaler_full.transform(X_test)

    model_full = _build_mlp(config.seeds.global_seed)
    model_full.fit(X_full_scaled, y_train_full)
    mu_test = model_full.predict(X_test_scaled)

    train_metrics = tfs._regression_summary(y_train, mu_train)
    val_metrics = tfs._regression_summary(y_val, mu_val) if len(y_val) else {}
    test_metrics = tfs._regression_summary(y_test, mu_test)

    print("Best sweep:", best_sweep_root)
    print("Best experiment:", experiment_id)
    print("Best sweep test MAE:", best["value"])
    print("NN metrics (train):", train_metrics)
    print("NN metrics (val):", val_metrics)
    print("NN metrics (test):", test_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
