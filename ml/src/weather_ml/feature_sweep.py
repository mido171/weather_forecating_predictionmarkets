"""Feature strategy sweep runner."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from weather_ml import artifacts
from weather_ml import config as config_module
from weather_ml import dataset
from weather_ml import derived_features
from weather_ml import distribution
from weather_ml import models_mean
from weather_ml import report
from weather_ml import splits
from weather_ml import utils_seed
from weather_ml import validate

LOGGER = logging.getLogger(__name__)

POINT_MODEL_COLS = [
    "nbm_tmax_f",
    "gfs_tmax_f",
    "gefsatmosmean_tmax_f",
    "nam_tmax_f",
    "hrrr_tmax_f",
    "rap_tmax_f",
]
SPREAD_COL = "gefsatmos_tmp_spread_f"


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    description: str
    model_cols: list[str]
    include_spread: bool
    calendar: bool
    rowwise_features: list[str]
    bias_corrected_features: list[str]
    use_bias_correction: bool
    use_reliability_weights: bool
    use_stacking: bool


@dataclass
class StrategyArtifacts:
    run_dir: Path
    hashes: dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature strategy sweep runner.")
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    parser.add_argument("--sweep-id", help="Optional sweep id override.")
    parser.add_argument("--sweep-root", help="Optional sweep output root.")
    parser.add_argument(
        "--allow-tuning",
        action="store_true",
        help="Enable hyperparameter tuning (default: fixed params).",
    )
    parser.add_argument(
        "--strategy-ids",
        nargs="*",
        help="Optional list of strategy ids to run (default: all).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = build_parser().parse_args(argv)
    config = config_module.load_config(args.config)
    repo_root = _resolve_repo_root()
    config = config_module.resolve_paths(config, repo_root=repo_root)

    utils_seed.set_global_determinism(
        config.seeds.global_seed, single_thread=config.seeds.force_single_thread
    )

    sweep_id = args.sweep_id or _default_sweep_id()
    sweep_root = (
        Path(args.sweep_root)
        if args.sweep_root
        else Path(config.artifacts.root_dir) / "feature_sweep" / sweep_id
    )
    sweep_root.mkdir(parents=True, exist_ok=True)

    df = dataset.load_csv(Path(config.data.csv_path))
    rules = validate.build_rules_from_config(config)
    validate.run_all_validations(df, rules)

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
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty.")

    cv_splits = splits.make_time_cv_splits(
        train_df,
        n_splits=config.split.cv.n_splits,
        gap_days=config.split.cv.gap_days,
    )

    strategies = _build_strategies()
    if args.strategy_ids:
        strategy_set = {sid.upper() for sid in args.strategy_ids}
        strategies = [s for s in strategies if s.strategy_id.upper() in strategy_set]
        if not strategies:
            raise ValueError("No strategies matched requested ids.")

    results = []
    for strategy in strategies:
        LOGGER.info("Running strategy %s", strategy.strategy_id)
        strategy_run_dir = sweep_root / strategy.strategy_id
        strategy_run_dir.mkdir(parents=True, exist_ok=True)
        strategy_result = _run_strategy(
            strategy,
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            cv_splits=cv_splits,
            sweep_root=sweep_root,
            run_dir=strategy_run_dir,
            allow_tuning=args.allow_tuning,
        )
        results.append(strategy_result)

    baseline_id = "S02"
    baseline = next(
        (entry for entry in results if entry["strategy_id"] == baseline_id), None
    )
    if baseline is None:
        baseline_id = results[0]["strategy_id"]
        baseline = results[0]
    _apply_deltas(results, baseline, baseline_id)

    csv_hash = artifacts.sha256_file(Path(config.data.csv_path))
    sweep_payload = {
        "sweep_id": sweep_id,
        "created_utc": artifacts.utc_now_iso(),
        "dataset_ref": {
            "csv_path": config.data.csv_path,
            "csv_hash": csv_hash,
            "schema_version": config.data.dataset_schema_version,
        },
        "split_ref": _split_ref(config),
        "model_ref": _model_ref(config, allow_tuning=args.allow_tuning),
        "baseline_strategy_id": baseline_id,
        "strategies": results,
        "leaderboard_test_mae": _leaderboard(results, split="test", metric="mae"),
        "leaderboard_val_mae": _leaderboard(results, split="val", metric="mae"),
    }
    sweep_path = sweep_root / "feature_strategy_sweep.json"
    sweep_path.write_text(
        json.dumps(sweep_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    LOGGER.info("Sweep complete. Output: %s", sweep_path)
    return 0


def _run_strategy(
    strategy: StrategyDefinition,
    *,
    config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    sweep_root: Path,
    run_dir: Path,
    allow_tuning: bool,
) -> dict:
    uses_spread = strategy.include_spread and SPREAD_COL in train_df.columns
    base_cols = list(strategy.model_cols)
    if uses_spread:
        base_cols.append(SPREAD_COL)
    _ensure_columns_exist(train_df, base_cols, strategy.strategy_id)

    feature_frames = _build_feature_frames(
        strategy=strategy,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        base_cols=base_cols,
        cv_splits=cv_splits,
        seed=config.seeds.global_seed,
    )

    X_train = feature_frames.train_features.to_numpy(dtype=float)
    y_train = train_df["actual_tmax_f"].to_numpy(dtype=float)
    X_val = feature_frames.val_features.to_numpy(dtype=float)
    y_val = val_df["actual_tmax_f"].to_numpy(dtype=float)
    X_test = feature_frames.test_features.to_numpy(dtype=float)
    y_test = test_df["actual_tmax_f"].to_numpy(dtype=float)

    _ensure_no_nan(feature_frames, strategy.strategy_id)

    model_name = config.models.mean.primary
    base_model = models_mean.get_mean_model(model_name, seed=config.seeds.global_seed)
    fixed_params = _fixed_params(config.models.mean.param_grid.get(model_name, {}))
    if allow_tuning:
        tuned = models_mean.tune_model_timecv(
            base_model,
            X_train,
            y_train,
            cv_splits,
            config.models.mean.param_grid.get(model_name, {}),
        )
        model_train = tuned.estimator
        best_params = tuned.best_params
    else:
        model_train = base_model
        if fixed_params:
            model_train.set_params(**fixed_params)
        model_train.fit(X_train, y_train)
        best_params = fixed_params

    mu_train = model_train.predict(X_train)
    mu_val = model_train.predict(X_val) if len(X_val) else np.array([])

    train_full_df = (
        pd.concat([train_df, val_df], ignore_index=True)
        if not val_df.empty
        else train_df.copy()
    )
    full_features = pd.concat(
        [feature_frames.train_features, feature_frames.val_features], ignore_index=True
    )
    y_train_full = train_full_df["actual_tmax_f"].to_numpy(dtype=float)

    model_full = models_mean.get_mean_model(model_name, seed=config.seeds.global_seed)
    if best_params:
        model_full.set_params(**best_params)
    model_full.fit(full_features.to_numpy(dtype=float), y_train_full)
    mu_test = model_full.predict(X_test)

    train_metrics = _regression_summary(y_train, mu_train)
    val_metrics = _regression_summary(y_val, mu_val) if len(y_val) else None
    test_metrics = _regression_summary(y_test, mu_test)

    metrics_summary = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    feature_list_path = run_dir / "feature_list.json"
    feature_list_path.write_text(
        json.dumps(feature_frames.feature_columns, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    config_snapshot = _config_snapshot(config, strategy, feature_frames.feature_columns)
    config_path = run_dir / "config_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    dataset_id = artifacts.compute_dataset_id(
        Path(config.data.csv_path),
        config.data.dataset_schema_version,
        {
            "strategy_id": strategy.strategy_id,
            "feature_columns": feature_frames.feature_columns,
        },
    )
    dataset_dir = sweep_root / "datasets" / dataset_id
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    data_parquet = artifacts.snapshot_to_parquet(all_df, dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    metadata = _build_dataset_metadata(train_df, val_df, test_df, dataset_id, config)
    artifacts.write_metadata(metadata_path, metadata)
    artifacts.write_hash_manifest(
        [data_parquet, metadata_path], dataset_dir / "hashes.json"
    )
    (run_dir / "dataset_id.txt").write_text(dataset_id, encoding="utf-8")

    mean_model_path = run_dir / "mean_model.joblib"
    sigma_model_path = run_dir / "sigma_model.joblib"
    feature_state_path = run_dir / "feature_state.joblib"
    import joblib

    joblib.dump(model_full, mean_model_path)
    joblib.dump(None, sigma_model_path)
    joblib.dump({"feature_columns": feature_frames.feature_columns}, feature_state_path)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    feature_importance = _feature_importance(
        model_full, feature_frames.feature_columns, full_features.to_numpy(dtype=float)
    )
    report_path = run_dir / "report.md"
    report.write_report(
        report_path,
        dataset_summary=_build_dataset_summary(train_df, val_df, test_df),
        metrics=metrics_summary,
        model_summary={
            "model": model_name,
            "params": best_params,
            "allow_tuning": allow_tuning,
        },
        feature_importance=feature_importance,
        global_calibration=None,
        baseline_calibration=None,
        config=config_snapshot,
    )

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_test - mu_test
    report.plot_residual_hist(plots_dir / "residual_hist.png", residuals)
    report.plot_residual_vs_pred(plots_dir / "residual_vs_pred.png", mu_test, residuals)

    sigma_fallback = (
        float(np.std(y_train - mu_train, ddof=1)) if len(y_train) > 1 else 1.0
    )
    sigma_test = np.full_like(mu_test, sigma_fallback, dtype=float)
    pmf = _build_pmf(mu_test, sigma_test, config)
    bin_probs = _build_bin_probs(pmf, config)
    predictions_path = run_dir / "predictions_test.parquet"
    _write_predictions(
        predictions_path, test_df, mu_test, sigma_test, pmf, bin_probs, config
    )

    strategy_meta = _strategy_meta(strategy, uses_spread, feature_frames)
    (run_dir / "strategy_meta.json").write_text(
        json.dumps(strategy_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "strategy_feature_columns.json").write_text(
        json.dumps(feature_frames.feature_columns, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    hash_paths = [
        config_path,
        feature_list_path,
        mean_model_path,
        sigma_model_path,
        feature_state_path,
        metrics_path,
        report_path,
        predictions_path,
        run_dir / "strategy_meta.json",
        run_dir / "strategy_feature_columns.json",
    ]
    hash_paths.extend(plots_dir.glob("*.png"))
    hashes_path = run_dir / "hashes.json"
    artifacts.write_hash_manifest(hash_paths, hashes_path)
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))

    worst_days = _worst_test_days(test_df, y_test, mu_test)

    return {
        "strategy_id": strategy.strategy_id,
        "description": strategy.description,
        "raw_model_cols_used": list(strategy.model_cols),
        "uses_spread_feature": uses_spread,
        "calendar_enabled": strategy.calendar,
        "derived_features": feature_frames.derived_metadata,
        "final_feature_columns": feature_frames.feature_columns,
        "num_features": int(len(feature_frames.feature_columns)),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "deltas_vs_baseline": {},
        "worst_test_days": worst_days,
        "run_dir": str(run_dir),
        "artifact_hashes": hashes,
    }


@dataclass
class FeatureFrames:
    train_features: pd.DataFrame
    val_features: pd.DataFrame
    test_features: pd.DataFrame
    feature_columns: list[str]
    derived_metadata: dict


def _build_feature_frames(
    *,
    strategy: StrategyDefinition,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_cols: list[str],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> FeatureFrames:
    feature_columns: list[str] = []
    derived_meta = {"rowwise": [], "train_fitted": []}

    train_features = train_df[base_cols].astype(float).copy()
    val_features = val_df[base_cols].astype(float).copy()
    test_features = test_df[base_cols].astype(float).copy()
    feature_columns.extend(base_cols)

    if strategy.rowwise_features:
        rowwise_train = derived_features.compute_rowwise_features(
            train_df,
            strategy.model_cols,
            include=strategy.rowwise_features,
        )
        if val_df.empty:
            rowwise_val = rowwise_train.iloc[:0].copy()
        else:
            rowwise_val = derived_features.compute_rowwise_features(
                val_df,
                strategy.model_cols,
                include=strategy.rowwise_features,
            )
        rowwise_test = derived_features.compute_rowwise_features(
            test_df,
            strategy.model_cols,
            include=strategy.rowwise_features,
        )
        train_features = pd.concat([train_features, rowwise_train], axis=1)
        val_features = pd.concat([val_features, rowwise_val], axis=1)
        test_features = pd.concat([test_features, rowwise_test], axis=1)
        feature_columns.extend(list(rowwise_train.columns))
        derived_meta["rowwise"].extend(
            _rowwise_formula_entries(rowwise_train.columns, strategy.model_cols)
        )

    if strategy.use_bias_correction:
        bias = derived_features.fit_bias_correction(
            train_df, strategy.model_cols, y_col="actual_tmax_f"
        )
        train_bc = derived_features.apply_bias_correction(
            train_df, strategy.model_cols, bias
        )
        val_bc = derived_features.apply_bias_correction(val_df, strategy.model_cols, bias)
        test_bc = derived_features.apply_bias_correction(
            test_df, strategy.model_cols, bias
        )
        bc_cols = [f"{col}_bc" for col in strategy.model_cols]
        train_features[bc_cols] = train_bc[bc_cols].astype(float)
        val_features[bc_cols] = val_bc[bc_cols].astype(float)
        test_features[bc_cols] = test_bc[bc_cols].astype(float)
        feature_columns.extend(bc_cols)
        bias_features = list(bc_cols)

        if strategy.bias_corrected_features:
            bc_rowwise_train = derived_features.compute_rowwise_features(
                train_bc,
                bc_cols,
                prefix="bc",
                include=strategy.bias_corrected_features,
            )
            if val_df.empty:
                bc_rowwise_val = bc_rowwise_train.iloc[:0].copy()
            else:
                bc_rowwise_val = derived_features.compute_rowwise_features(
                    val_bc,
                    bc_cols,
                    prefix="bc",
                    include=strategy.bias_corrected_features,
                )
            bc_rowwise_test = derived_features.compute_rowwise_features(
                test_bc,
                bc_cols,
                prefix="bc",
                include=strategy.bias_corrected_features,
            )
            train_features = pd.concat([train_features, bc_rowwise_train], axis=1)
            val_features = pd.concat([val_features, bc_rowwise_val], axis=1)
            test_features = pd.concat([test_features, bc_rowwise_test], axis=1)
            feature_columns.extend(list(bc_rowwise_train.columns))
            derived_meta["rowwise"].extend(
                _rowwise_formula_entries(bc_rowwise_train.columns, strategy.model_cols)
            )
            bias_features.extend(list(bc_rowwise_train.columns))

        derived_meta["train_fitted"].append(
            {
                "name": "bias_correction",
                "fit_on": "train",
                "description": "bias_m_train = mean(y - m); m_bc = m + bias_m_train",
                "models": list(strategy.model_cols),
                "features_added": bias_features,
            }
        )

    if strategy.use_reliability_weights:
        weights = derived_features.fit_reliability_weights(
            train_df, strategy.model_cols, y_col="actual_tmax_f"
        )
        train_features = derived_features.apply_reliability_features(
            train_features, strategy.model_cols, weights
        )
        val_features = derived_features.apply_reliability_features(
            val_features, strategy.model_cols, weights
        )
        test_features = derived_features.apply_reliability_features(
            test_features, strategy.model_cols, weights
        )
        new_cols = [col for col in train_features.columns if col not in feature_columns]
        feature_columns.extend(new_cols)
        derived_meta["train_fitted"].append(
            {
                "name": "reliability_weights",
                "fit_on": "train",
                "description": "mae_m_train; w_m=1/(mae+1e-6) normalized; ens_wmean",
                "weights_norm": weights.weights_norm,
                "features_added": new_cols,
            }
        )

    if strategy.use_stacking:
        if not cv_splits:
            raise ValueError(
                f"{strategy.strategy_id}: stacking requires CV splits; enable split.cv."
            )
        oof_pred, ridge_model = derived_features.fit_stack_ridge_oof(
            train_df,
            strategy.model_cols,
            y_col="actual_tmax_f",
            cv_splits=cv_splits,
            seed=seed,
        )
        fallback_mask = np.isnan(oof_pred)
        if fallback_mask.any():
            fallback = np.median(
                train_df[strategy.model_cols].to_numpy(dtype=float), axis=1
            )
            oof_pred[fallback_mask] = fallback[fallback_mask]
        train_features["stack_ridge_pred"] = oof_pred
        val_features = derived_features.apply_stack_feature(
            val_features, strategy.model_cols, ridge_model
        )
        test_features = derived_features.apply_stack_feature(
            test_features, strategy.model_cols, ridge_model
        )
        feature_columns.append("stack_ridge_pred")
        stack_features = ["stack_ridge_pred"]
        derived_meta["train_fitted"].append(
            {
                "name": "stack_ridge_pred",
                "fit_on": "train",
                "description": "Ridge meta-model OOF on train; full-train for val/test",
                "fallback_rows": int(fallback_mask.sum()),
            }
        )

        if "ens_wmean_6" in train_features.columns:
            train_features["ens_wmean_minus_stack"] = (
                train_features["ens_wmean_6"] - train_features["stack_ridge_pred"]
            )
            val_features["ens_wmean_minus_stack"] = (
                val_features["ens_wmean_6"] - val_features["stack_ridge_pred"]
            )
            test_features["ens_wmean_minus_stack"] = (
                test_features["ens_wmean_6"] - test_features["stack_ridge_pred"]
            )
            feature_columns.append("ens_wmean_minus_stack")
            stack_features.append("ens_wmean_minus_stack")
        if f"ens_median_{len(strategy.model_cols)}" in train_features.columns:
            train_features["stack_minus_median"] = (
                train_features["stack_ridge_pred"]
                - train_features[f"ens_median_{len(strategy.model_cols)}"]
            )
            val_features["stack_minus_median"] = (
                val_features["stack_ridge_pred"]
                - val_features[f"ens_median_{len(strategy.model_cols)}"]
            )
            test_features["stack_minus_median"] = (
                test_features["stack_ridge_pred"]
                - test_features[f"ens_median_{len(strategy.model_cols)}"]
            )
            feature_columns.append("stack_minus_median")
            stack_features.append("stack_minus_median")

        derived_meta["train_fitted"][-1]["features_added"] = stack_features

    if strategy.calendar:
        train_features = _add_calendar_features(train_features, train_df)
        val_features = _add_calendar_features(val_features, val_df)
        test_features = _add_calendar_features(test_features, test_df)
        new_cols = [col for col in train_features.columns if col not in feature_columns]
        feature_columns.extend(new_cols)

    return FeatureFrames(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        feature_columns=feature_columns,
        derived_metadata=derived_meta,
    )


def _build_strategies() -> list[StrategyDefinition]:
    strategies = []
    f6 = list(POINT_MODEL_COLS)
    f5 = [col for col in f6 if col != "rap_tmax_f"]
    f3 = ["nbm_tmax_f", "gfs_tmax_f", "gefsatmosmean_tmax_f"]

    strategies.append(
        StrategyDefinition(
            strategy_id="S01",
            description="raw_all_6",
            model_cols=f6,
            include_spread=True,
            calendar=False,
            rowwise_features=[],
            bias_corrected_features=[],
            use_bias_correction=False,
            use_reliability_weights=False,
            use_stacking=False,
        )
    )
    strategies.append(
        StrategyDefinition(
            strategy_id="S02",
            description="raw_all_6_plus_calendar",
            model_cols=f6,
            include_spread=True,
            calendar=True,
            rowwise_features=[],
            bias_corrected_features=[],
            use_bias_correction=False,
            use_reliability_weights=False,
            use_stacking=False,
        )
    )
    strategies.append(
        StrategyDefinition(
            strategy_id="S03",
            description="add_blends_center",
            model_cols=f6,
            include_spread=True,
            calendar=True,
            rowwise_features=_blend_features(len(f6)),
            bias_corrected_features=[],
            use_bias_correction=False,
            use_reliability_weights=False,
            use_stacking=False,
        )
    )
    strategies.append(
        StrategyDefinition(
            strategy_id="S04",
            description="add_robust_blends",
            model_cols=f6,
            include_spread=True,
            calendar=True,
            rowwise_features=_blend_features(len(f6)) + _robust_blend_features(len(f6)),
            bias_corrected_features=[],
            use_bias_correction=False,
            use_reliability_weights=False,
            use_stacking=False,
        )
    )
    strategies.append(
        StrategyDefinition(
            strategy_id="S05",
            description="add_dispersion_robust_spread",
            model_cols=f6,
            include_spread=True,
            calendar=True,
            rowwise_features=_dispersion_features(len(f6)),
            bias_corrected_features=[],
            use_bias_correction=False,
            use_reliability_weights=False,
            use_stacking=False,
        )
    )
    strategies.append(
        StrategyDefinition(
            strategy_id="S06",
            description="add_model_vs_consensus_deltas",
            model_cols=f6,
            include_spread=True,
            calendar=True,
            rowwise_features=_consensus_deltas(f6, include_mean=True, include_median=True),
            bias_corrected_features=[],
            use_bias_correction=False,
            use_reliability_weights=False,
            use_stacking=False,
        )
    )
    strategies.append(
        StrategyDefinition(
            strategy_id="S07",
            description="drop_worst_rap_plus_derived",
            model_cols=f5,
            include_spread=True,
            calendar=True,
            rowwise_features=_blend_features(len(f5))
            + _robust_blend_features(len(f5))
            + _limited_dispersion_features(len(f5))
            + _consensus_deltas(f5, include_mean=False, include_median=True),
            bias_corrected_features=[],
            use_bias_correction=False,
            use_reliability_weights=False,
            use_stacking=False,
        )
    )
    strategies.append(
        StrategyDefinition(
            strategy_id="S08",
            description="top3_only_plus_robust (ens_closest3_mean omitted)",
            model_cols=f3,
            include_spread=True,
            calendar=True,
            rowwise_features=_blend_features(len(f3))
            + _dispersion_core_features(len(f3))
            + _consensus_deltas(f3, include_mean=False, include_median=True),
            bias_corrected_features=[],
            use_bias_correction=False,
            use_reliability_weights=False,
            use_stacking=False,
        )
    )
    strategies.append(
        StrategyDefinition(
            strategy_id="S09",
            description="bias_corrected_inputs_and_blends",
            model_cols=f6,
            include_spread=True,
            calendar=True,
            rowwise_features=[f"ens_mean_{len(f6)}", f"ens_median_{len(f6)}"],
            bias_corrected_features=[
                f"ens_mean_{len(f6)}",
                f"ens_median_{len(f6)}",
                f"ens_trimmed_mean_{len(f6)}_1",
            ],
            use_bias_correction=True,
            use_reliability_weights=False,
            use_stacking=False,
        )
    )
    strategies.append(
        StrategyDefinition(
            strategy_id="S10",
            description="reliability_weighted_and_stacking_meta",
            model_cols=f6,
            include_spread=True,
            calendar=True,
            rowwise_features=[f"ens_median_{len(f6)}"],
            bias_corrected_features=[],
            use_bias_correction=False,
            use_reliability_weights=True,
            use_stacking=True,
        )
    )
    return strategies


def _blend_features(n: int) -> list[str]:
    return [
        f"ens_mean_{n}",
        f"ens_median_{n}",
        f"ens_min_{n}",
        f"ens_max_{n}",
    ]


def _robust_blend_features(n: int) -> list[str]:
    return [f"ens_trimmed_mean_{n}_1", f"ens_winsor_mean_{n}_1"]


def _dispersion_features(n: int) -> list[str]:
    return [
        f"ens_std_{n}",
        f"ens_range_{n}",
        f"ens_p25_{n}",
        f"ens_p75_{n}",
        f"ens_iqr_{n}",
        f"ens_mad_{n}",
        f"ens_outlier_gap_{n}",
    ]


def _dispersion_core_features(n: int) -> list[str]:
    return [f"ens_std_{n}", f"ens_range_{n}"]


def _limited_dispersion_features(n: int) -> list[str]:
    return [f"ens_std_{n}", f"ens_iqr_{n}"]


def _consensus_deltas(
    model_cols: list[str], *, include_mean: bool, include_median: bool
) -> list[str]:
    features = []
    n = len(model_cols)
    if include_mean:
        features.extend([f"ens_mean_{n}", f"ens_median_{n}"])
    elif include_median:
        features.append(f"ens_median_{n}")
    for col in model_cols:
        if include_mean:
            features.append(f"{col}_minus_ens_mean")
            features.append(f"{col}_minus_ens_mean_abs")
        if include_median:
            features.append(f"{col}_minus_ens_median")
            features.append(f"{col}_minus_ens_median_abs")
    return features


def _rowwise_formula_entries(columns: Iterable[str], model_cols: list[str]) -> list[dict]:
    n = len(model_cols)
    formulas = _formula_map(model_cols, n)
    entries = []
    for col in columns:
        formula_key = col
        suffix = ""
        if "_bc_" in col:
            formula_key = col.replace("_bc_", "_")
            while "__" in formula_key:
                formula_key = formula_key.replace("__", "_")
            suffix = " on bias-corrected inputs"
        formula = formulas.get(formula_key, "derived")
        if suffix and formula != "derived":
            formula = f"{formula}{suffix}"
        entries.append({"name": col, "formula": formula})
    return entries


def _formula_map(model_cols: list[str], n: int) -> dict[str, str]:
    formula = {
        f"ens_mean_{n}": "mean(F)",
        f"ens_median_{n}": "median(F)",
        f"ens_min_{n}": "min(F)",
        f"ens_max_{n}": "max(F)",
        f"ens_trimmed_mean_{n}_1": "mean(sorted(F)[1:-1])",
        f"ens_winsor_mean_{n}_1": "winsorize 1 each tail then mean",
        f"ens_p25_{n}": "p25(F)",
        f"ens_p75_{n}": "p75(F)",
        f"ens_iqr_{n}": "p75(F) - p25(F)",
        f"ens_range_{n}": "max(F) - min(F)",
        f"ens_std_{n}": "std(F, ddof=0)",
        f"ens_mad_{n}": "median(|F - median(F)|)",
        f"ens_outlier_gap_{n}": "max(|F - median(F)|)",
        f"ens_second_min_{n}": "sorted(F)[1]",
        f"ens_second_max_{n}": "sorted(F)[-2]",
        "ens_closest3_mean": "mean(3 closest to median(F))",
    }
    for col in model_cols:
        formula[f"{col}_minus_ens_mean"] = f"{col} - ens_mean_{n}"
        formula[f"{col}_minus_ens_mean_abs"] = f"|{col} - ens_mean_{n}|"
        formula[f"{col}_minus_ens_median"] = f"{col} - ens_median_{n}"
        formula[f"{col}_minus_ens_median_abs"] = f"|{col} - ens_median_{n}|"
        formula[f"rank_{col}_in_ens"] = (
            "rank in F with tie-break order "
            + ", ".join(derived_features.RAW_MODEL_ORDER)
        )
    return formula


def _ensure_columns_exist(df: pd.DataFrame, cols: list[str], strategy_id: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{strategy_id}: missing required columns: {missing}")


def _ensure_no_nan(feature_frames: FeatureFrames, strategy_id: str) -> None:
    for name, frame in [
        ("train", feature_frames.train_features),
        ("val", feature_frames.val_features),
        ("test", feature_frames.test_features),
    ]:
        if frame.isna().any().any():
            missing_cols = frame.columns[frame.isna().any()].tolist()
            raise ValueError(
                f"{strategy_id}: NaNs detected in {name} features: {missing_cols}"
            )


def _regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {}
    error = y_pred - y_true
    abs_error = np.abs(error)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "medianAE": float(np.median(abs_error)),
        "maxAE": float(np.max(abs_error)),
        "corr": corr,
        "n": int(len(y_true)),
    }


def _worst_test_days(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    error = y_pred - y_true
    abs_error = np.abs(error)
    worst_idx = np.argsort(-abs_error)[:10]
    records = []
    for idx in worst_idx:
        row = df.iloc[idx]
        records.append(
            {
                "station_id": row["station_id"],
                "target_date_local": str(row["target_date_local"]),
                "asof_utc": str(row["asof_utc"]),
                "y_true": float(y_true[idx]),
                "y_pred": float(y_pred[idx]),
                "error": float(error[idx]),
                "abs_error": float(abs_error[idx]),
            }
        )
    return records


def _write_predictions(
    path: Path,
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    pmf: np.ndarray,
    bin_probs: dict[str, np.ndarray],
    config,
) -> None:
    records = df[["station_id", "target_date_local", "asof_utc"]].copy()
    records["mu_hat_f"] = mu
    records["sigma_hat_f"] = sigma
    records["p_temp_json"] = [
        json.dumps(row.tolist(), separators=(",", ":"), ensure_ascii=True) for row in pmf
    ]
    records["p_bins_json"] = [
        json.dumps(
            {name: float(prob[idx]) for name, prob in bin_probs.items()},
            separators=(",", ":"),
            ensure_ascii=True,
        )
        for idx in range(len(records))
    ]
    records["support_min_f"] = config.distribution.support_min_f
    records["support_max_f"] = config.distribution.support_max_f
    records.to_parquet(path, index=False, engine="pyarrow")


def _build_pmf(mu: np.ndarray, sigma: np.ndarray, config) -> np.ndarray:
    pmf_rows = []
    for mu_i, sigma_i in zip(mu, sigma):
        pmf = distribution.normal_integer_pmf(
            float(mu_i),
            float(sigma_i),
            support_min=config.distribution.support_min_f,
            support_max=config.distribution.support_max_f,
        )
        pmf_rows.append(pmf)
    return np.vstack(pmf_rows)


def _build_bin_probs(pmf: np.ndarray, config) -> dict[str, np.ndarray]:
    bin_probs: dict[str, list[float]] = {
        spec.get("name", ""): [] for spec in config.calibration.bins_to_calibrate
    }
    for row in pmf:
        for spec in config.calibration.bins_to_calibrate:
            name = spec.get("name", "")
            prob = distribution.bins_from_pmf(
                row,
                support_min=config.distribution.support_min_f,
                bin_specs=[spec],
            ).get(name, 0.0)
            bin_probs.setdefault(name, []).append(prob)
    return {name: np.array(values) for name, values in bin_probs.items()}


def _feature_importance(
    model: object, feature_names: list[str], x: np.ndarray
) -> dict | None:
    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_")).reshape(-1)
        std = x.std(axis=0)
        importance = coef * std
        pairs = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, importance)
        ]
        pairs = sorted(pairs, key=lambda item: abs(item["importance"]), reverse=True)
        return {"type": "linear", "top_features": pairs[:50]}
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(getattr(model, "feature_importances_")).reshape(-1)
        pairs = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, importances)
        ]
        pairs = sorted(pairs, key=lambda item: abs(item["importance"]), reverse=True)
        return {"type": "tree", "top_features": pairs[:50]}
    return None


def _build_dataset_summary(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return {
        "row_count": int(len(all_df)),
        "station_counts": all_df["station_id"].value_counts().to_dict(),
        "date_coverage": {
            "min": str(pd.to_datetime(all_df["target_date_local"]).min().date()),
            "max": str(pd.to_datetime(all_df["target_date_local"]).max().date()),
        },
        "missing_by_column": all_df.isna().sum().to_dict(),
        "split_counts": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }


def _build_dataset_metadata(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, dataset_id: str, config
) -> dict:
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return {
        "dataset_id": dataset_id,
        "csv_path": config.data.csv_path,
        "schema_version": config.data.dataset_schema_version,
        "created_at": artifacts.utc_now_iso(),
        "row_count_raw": int(len(all_df)),
        "row_count": int(len(all_df)),
        "missing_by_column": all_df.isna().sum().to_dict(),
        "station_counts": all_df["station_id"].value_counts().to_dict(),
    }


def _config_snapshot(config, strategy: StrategyDefinition, feature_columns: list[str]) -> dict:
    payload = asdict(config)
    payload["strategy"] = {
        "strategy_id": strategy.strategy_id,
        "description": strategy.description,
        "model_cols": strategy.model_cols,
        "include_spread": strategy.include_spread,
        "calendar": strategy.calendar,
        "feature_columns": feature_columns,
    }
    return payload


def _split_ref(config) -> dict:
    return {
        "train_start": str(config.split.train_start),
        "train_end": str(config.split.train_end),
        "val_start": str(config.split.validation.val_start),
        "val_end": str(config.split.validation.val_end),
        "test_start": str(config.split.test_start),
        "test_end": str(config.split.test_end),
        "gap_dates": [str(d) for d in config.split.gap_dates],
    }


def _model_ref(config, *, allow_tuning: bool) -> dict:
    model_name = config.models.mean.primary
    fixed_params = _fixed_params(config.models.mean.param_grid.get(model_name, {}))
    return {
        "model": model_name,
        "allow_tuning": allow_tuning,
        "fixed_params": fixed_params,
        "seed": config.seeds.global_seed,
    }


def _fixed_params(param_grid: dict[str, list]) -> dict[str, float]:
    params: dict[str, float] = {}
    for key, values in param_grid.items():
        if isinstance(values, list) and values:
            params[key] = values[0]
    return params


def _strategy_meta(
    strategy: StrategyDefinition, uses_spread: bool, feature_frames: FeatureFrames
) -> dict:
    return {
        "strategy_id": strategy.strategy_id,
        "description": strategy.description,
        "raw_model_cols_used": strategy.model_cols,
        "uses_spread_feature": uses_spread,
        "calendar_enabled": strategy.calendar,
        "derived_features": feature_frames.derived_metadata,
        "final_feature_columns": feature_frames.feature_columns,
    }


def _leaderboard(results: list[dict], *, split: str, metric: str) -> list[dict]:
    entries = []
    for entry in results:
        metrics_block = entry["metrics"].get(split) or {}
        value = metrics_block.get(metric)
        if value is None:
            continue
        entries.append({"strategy_id": entry["strategy_id"], metric: value})
    return sorted(entries, key=lambda item: item[metric])


def _apply_deltas(results: list[dict], baseline: dict, baseline_id: str) -> None:
    baseline_metrics = baseline["metrics"]["test"]
    for entry in results:
        test_metrics = entry["metrics"]["test"]
        entry["deltas_vs_baseline"] = {
            "baseline_strategy_id": baseline_id,
            "delta_test_mae": test_metrics["mae"] - baseline_metrics["mae"],
            "delta_test_rmse": test_metrics["rmse"] - baseline_metrics["rmse"],
            "delta_test_bias": test_metrics["bias"] - baseline_metrics["bias"],
            "delta_test_corr": test_metrics["corr"] - baseline_metrics["corr"],
        }


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


def _default_sweep_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


if __name__ == "__main__":
    raise SystemExit(main())
