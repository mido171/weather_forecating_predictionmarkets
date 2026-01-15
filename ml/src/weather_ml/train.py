"""Training CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from weather_ml import artifacts
from weather_ml import baselines
from weather_ml import calibration
from weather_ml import config as config_module
from weather_ml import dataset
from weather_ml import distribution
from weather_ml import features
from weather_ml import global_normal_calibration
from weather_ml import metrics
from weather_ml import models_mean
from weather_ml import models_sigma
from weather_ml import report
from weather_ml import splits
from weather_ml import utils_seed
from weather_ml import validate

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weather ML training pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-id", help="Optional run id override.")
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

    csv_path = Path(config.data.csv_path)
    df = dataset.load_csv(csv_path)
    rules = validate.build_rules_from_config(config)
    validate.run_all_validations(df, rules)

    dataset_id = artifacts.compute_dataset_id(
        csv_path, config.data.dataset_schema_version, asdict(config.features)
    )
    artifacts_root = Path(config.artifacts.root_dir)
    dataset_dir = artifacts_root / "datasets" / dataset_id
    data_parquet = artifacts.snapshot_to_parquet(df, dataset_dir)
    dataset_metadata = _build_dataset_metadata(df, dataset_id, csv_path, config)
    metadata_path = dataset_dir / "metadata.json"
    artifacts.write_metadata(metadata_path, dataset_metadata)
    artifacts.write_hash_manifest(
        [data_parquet, metadata_path], dataset_dir / "hashes.json"
    )

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

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if test_df.empty:
        raise ValueError("Test split is empty.")

    LOGGER.info(
        "Split sizes: train=%s val=%s test=%s",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    X_train_df, y_train, state_inner = features.build_features(
        train_df, config=config, training=True
    )
    X_train = X_train_df.to_numpy(dtype=float)

    X_val = None
    y_val = None
    if not val_df.empty:
        X_val_df, y_val, _ = features.build_features(
            val_df, config=config, fit_state=state_inner
        )
        X_val = X_val_df.to_numpy(dtype=float)

    cv_splits = []
    if config.split.cv.enabled:
        cv_splits = splits.make_time_cv_splits(
            train_df, n_splits=config.split.cv.n_splits, gap_days=config.split.cv.gap_days
        )

    candidate_results = []
    tuned_models = {}
    candidates = config.models.mean.candidates or [config.models.mean.primary]
    for name in candidates:
        try:
            base_model = models_mean.get_mean_model(name, seed=config.seeds.global_seed)
        except ImportError as exc:
            LOGGER.warning("Skipping %s: %s", name, exc)
            continue
        param_grid = config.models.mean.param_grid.get(name, {})
        tuned = models_mean.tune_model_timecv(
            base_model, X_train, y_train, cv_splits, param_grid
        )
        tuned_models[name] = tuned
        val_mae = None
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = tuned.estimator.predict(X_val)
            val_metrics = metrics.regression_metrics(y_val, val_pred)
            val_mae = val_metrics["mae"]
        candidate_results.append(
            {
                "model": name,
                "best_params": tuned.best_params,
                "cv_score": tuned.best_score,
                "val_mae": val_mae,
            }
        )

    if not candidate_results:
        raise RuntimeError("No mean model candidates could be trained.")

    best_candidate = _select_best_candidate(
        candidate_results, config.models.mean.primary
    )
    best_model_name = best_candidate["model"]
    best_params = best_candidate["best_params"]
    mean_model_inner = tuned_models[best_model_name].estimator

    train_pred_inner = mean_model_inner.predict(X_train)
    train_metrics_inner = metrics.regression_metrics(y_train, train_pred_inner)

    val_pred_inner = None
    val_metrics_inner = None
    if X_val is not None and y_val is not None:
        val_pred_inner = mean_model_inner.predict(X_val)
        val_metrics_inner = metrics.regression_metrics(y_val, val_pred_inner)

    sigma_model_inner = None
    sigma_val = None
    if config.models.sigma.method == "two_stage":
        sigma_model_inner = _train_sigma_model(
            model_name=config.models.sigma.primary,
            mean_model_name=best_model_name,
            best_mean_params=best_params,
            X=X_train,
            y=y_train,
            splits=cv_splits,
            seed=config.seeds.global_seed,
            param_grid=config.models.sigma.param_grid.get(
                config.models.sigma.primary, {}
            ),
            eps=config.models.sigma.eps,
        )
        if sigma_model_inner is not None and X_val is not None:
            sigma_val = models_sigma.predict_sigma(
                sigma_model_inner,
                X=X_val,
                eps=config.models.sigma.eps,
                sigma_floor=config.models.sigma.sigma_floor,
            )

    calibrators = {}
    val_prob_metrics = {}
    val_probabilities = None
    if X_val is not None and y_val is not None and val_pred_inner is not None and sigma_val is not None:
        val_probabilities = _build_probabilities(
            mu=val_pred_inner,
            sigma=sigma_val,
            support_min=config.distribution.support_min_f,
            support_max=config.distribution.support_max_f,
            bin_specs=config.calibration.bins_to_calibrate,
        )
        val_prob_metrics = _probabilistic_metrics(
            y_true=y_val,
            pmf=val_probabilities["pmf"],
            support_min=config.distribution.support_min_f,
            bin_probs=val_probabilities["bin_probs"],
            bin_specs=config.calibration.bins_to_calibrate,
        )
        if config.calibration.enabled:
            y_true_by_bin = {
                spec["name"]: metrics.event_indicator(y_val, spec).astype(int)
                for spec in config.calibration.bins_to_calibrate
            }
            calibrators = calibration.fit_isotonic_calibrators(
                val_probabilities["bin_probs"], y_true_by_bin
            )

    train_full_df = (
        pd.concat([train_df, val_df], ignore_index=True)
        if not val_df.empty
        else train_df.copy()
    )
    cv_splits_full = []
    if config.split.cv.enabled:
        cv_splits_full = splits.make_time_cv_splits(
            train_full_df,
            n_splits=config.split.cv.n_splits,
            gap_days=config.split.cv.gap_days,
        )
    X_train_full_df, y_train_full, state_full = features.build_features(
        train_full_df, config=config, training=True
    )
    X_train_full = X_train_full_df.to_numpy(dtype=float)

    X_test_df, y_test, _ = features.build_features(
        test_df, config=config, fit_state=state_full
    )
    X_test = X_test_df.to_numpy(dtype=float)

    mean_model_full = models_mean.get_mean_model(
        best_model_name, seed=config.seeds.global_seed
    )
    if best_params:
        mean_model_full.set_params(**best_params)
    mean_model_full.fit(X_train_full, y_train_full)

    mu_test = mean_model_full.predict(X_test)
    sigma_test = None
    sigma_model_full = None
    if config.models.sigma.method == "two_stage":
        sigma_model_full = _train_sigma_model(
            model_name=config.models.sigma.primary,
            mean_model_name=best_model_name,
            best_mean_params=best_params,
            X=X_train_full,
            y=y_train_full,
            splits=cv_splits_full,
            seed=config.seeds.global_seed,
            param_grid=config.models.sigma.param_grid.get(
                config.models.sigma.primary, {}
            ),
            eps=config.models.sigma.eps,
        )
        if sigma_model_full is not None:
            sigma_test = models_sigma.predict_sigma(
                sigma_model_full,
                X=X_test,
                eps=config.models.sigma.eps,
                sigma_floor=config.models.sigma.sigma_floor,
            )

    if sigma_test is None:
        raise RuntimeError("Sigma model training failed; sigma predictions missing.")

    test_probabilities = _build_probabilities(
        mu=mu_test,
        sigma=sigma_test,
        support_min=config.distribution.support_min_f,
        support_max=config.distribution.support_max_f,
        bin_specs=config.calibration.bins_to_calibrate,
    )
    bin_probs_test = test_probabilities["bin_probs"]
    if calibrators:
        bin_probs_test = calibration.apply_calibrators(bin_probs_test, calibrators)

    test_prob_metrics = _probabilistic_metrics(
        y_true=y_test,
        pmf=test_probabilities["pmf"],
        support_min=config.distribution.support_min_f,
        bin_probs=bin_probs_test,
        bin_specs=config.calibration.bins_to_calibrate,
    )

    test_metrics = metrics.regression_metrics(y_test, mu_test)
    test_station_metrics = metrics.per_station_metrics(
        test_df, y_true=y_test, y_pred=mu_test
    )

    baseline_metrics = _baseline_metrics(train_df, train_full_df, val_df, test_df)

    model_summary = {
        "selected_model": best_model_name,
        "best_params": best_params,
        "candidates": candidate_results,
        "sigma_model": config.models.sigma.primary,
    }

    metrics_summary = {
        "train": train_metrics_inner,
        "validation": val_metrics_inner,
        "test": test_metrics,
        "probabilistic_validation": val_prob_metrics,
        "probabilistic_test": test_prob_metrics,
        "baseline": baseline_metrics,
        "per_station_test": test_station_metrics,
    }

    feature_importance = _feature_importance(
        mean_model_full, state_full.feature_columns, X_train_full
    )

    run_id = args.run_id or config.artifacts.run_id or _default_run_id()
    run_dir = _prepare_run_dir(
        artifacts_root, run_id, overwrite=config.artifacts.overwrite
    )
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config_resolved.yaml"
    _write_yaml(config_path, asdict(config))
    (run_dir / "dataset_id.txt").write_text(dataset_id, encoding="utf-8")
    (run_dir / "feature_list.json").write_text(
        json.dumps(state_full.feature_columns, indent=2, sort_keys=True), encoding="utf-8"
    )

    mean_model_path = run_dir / "mean_model.joblib"
    joblib.dump(mean_model_full, mean_model_path)
    sigma_model_path = run_dir / "sigma_model.joblib"
    joblib.dump(sigma_model_full, sigma_model_path)
    feature_state_path = run_dir / "feature_state.joblib"
    joblib.dump(state_full, feature_state_path)

    calibrator_path = None
    if calibrators:
        calibrator_path = run_dir / "calibrators.joblib"
        joblib.dump(calibrators, calibrator_path)

    metrics_path = run_dir / "metrics.json"
    _write_json(metrics_path, metrics_summary)

    global_calibration = None
    global_residuals_path = None
    global_calibration_path = None
    baseline_calibration = None
    baseline_residuals_path = None
    baseline_calibration_path = None
    if config.postprocess.global_normal_calibration.enabled:
        gnc = config.postprocess.global_normal_calibration
        global_normal_calibration.check_calibration_window(
            cal_start=gnc.cal_start,
            cal_end=gnc.cal_end,
            train_end=config.split.train_end,
            val_end=config.split.validation.val_end,
        )
        required_cols = (
            ["station_id", "target_date_local", "asof_utc", "actual_tmax_f"]
            + list(config.features.base_features)
        )
        cal_df = global_normal_calibration.select_calibration_rows(
            df,
            cal_start=gnc.cal_start,
            cal_end=gnc.cal_end,
            station_scope=gnc.station_scope,
            required_columns=required_cols,
        )
        X_cal_df, _, _ = features.build_features(
            cal_df, config=config, fit_state=state_full
        )
        mu_cal = mean_model_full.predict(X_cal_df.to_numpy(dtype=float))
        residuals = cal_df["actual_tmax_f"].to_numpy() - mu_cal
        residual_stats = global_normal_calibration.compute_residual_stats(
            residuals, ddof=gnc.ddof
        )
        dataset_hash = artifacts.sha256_file(data_parquet)
        model_hash = artifacts.sha256_file(mean_model_path)
        global_calibration = global_normal_calibration.build_calibration_payload(
            residual_stats=residual_stats,
            ddof=gnc.ddof,
            cal_start=gnc.cal_start,
            cal_end=gnc.cal_end,
            station_scope=gnc.station_scope,
            run_dir=run_dir,
            mean_model_path=mean_model_path,
            model_hash=model_hash,
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            rows_used=int(residual_stats["n"]),
        )
        global_calibration_path = run_dir / "global_normal_calibration.json"
        _write_json(global_calibration_path, global_calibration)
        global_residuals_path = run_dir / "global_normal_residuals.csv"
        global_normal_calibration.write_residuals_csv(
            global_residuals_path,
            cal_df=cal_df,
            mu_hat=mu_cal,
            residuals=residuals,
            include_abs=True,
        )

    if config.postprocess.baseline_median_calibration.enabled:
        bmc = config.postprocess.baseline_median_calibration
        if bmc.cal_start is None or bmc.cal_end is None:
            raise ValueError("Baseline median calibration window must be set.")
        forecast_columns = global_normal_calibration.resolve_forecast_columns(
            bmc.forecast_columns, config.features.base_features
        )
        required_cols = (
            ["station_id", "target_date_local", "asof_utc", "actual_tmax_f"]
            + list(forecast_columns)
        )
        baseline_df = global_normal_calibration.select_calibration_rows(
            df,
            cal_start=bmc.cal_start,
            cal_end=bmc.cal_end,
            station_scope=bmc.station_scope,
            required_columns=required_cols,
        )
        median_forecast = global_normal_calibration.compute_median_forecast(
            baseline_df, forecast_columns
        )
        baseline_residuals = (
            baseline_df["actual_tmax_f"].to_numpy() - median_forecast
        )
        baseline_stats = global_normal_calibration.compute_residual_stats(
            baseline_residuals, ddof=bmc.ddof
        )
        dataset_hash = artifacts.sha256_file(data_parquet)
        baseline_calibration = global_normal_calibration.build_baseline_payload(
            residual_stats=baseline_stats,
            ddof=bmc.ddof,
            cal_start=bmc.cal_start,
            cal_end=bmc.cal_end,
            station_scope=bmc.station_scope,
            forecast_columns=forecast_columns,
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            rows_used=int(baseline_stats["n"]),
        )
        baseline_calibration_path = (
            run_dir / "baseline_median_normal_calibration.json"
        )
        _write_json(baseline_calibration_path, baseline_calibration)
        baseline_residuals_path = run_dir / "baseline_median_normal_residuals.csv"
        global_normal_calibration.write_baseline_residuals_csv(
            baseline_residuals_path,
            cal_df=baseline_df,
            median_forecast=median_forecast,
            residuals=baseline_residuals,
            include_abs=True,
        )

    report_path = run_dir / "report.md"
    report.write_report(
        report_path,
        dataset_summary=_build_dataset_summary(df, train_df, val_df, test_df),
        metrics=metrics_summary,
        model_summary=model_summary,
        feature_importance=feature_importance,
        global_calibration=global_calibration,
        baseline_calibration=baseline_calibration,
        config=asdict(config),
    )

    residuals = y_test - mu_test
    report.plot_residual_hist(plots_dir / "residual_hist.png", residuals)
    report.plot_residual_vs_pred(plots_dir / "residual_vs_pred.png", mu_test, residuals)

    if calibrators and val_probabilities is not None:
        for spec in config.calibration.bins_to_calibrate:
            name = spec["name"]
            y_true_bin = metrics.event_indicator(y_val, spec).astype(int)
            y_prob_bin = val_probabilities["bin_probs"][name]
            report.plot_calibration_curve(
                plots_dir / f"calibration_{name}.png",
                y_true_bin,
                y_prob_bin,
                title=f"Calibration - {name}",
            )

    predictions_path = run_dir / "predictions_test.parquet"
    _write_predictions(
        predictions_path,
        test_df,
        mu_test,
        sigma_test,
        test_probabilities["pmf"],
        bin_probs_test,
        config.distribution.support_min_f,
    )

    hash_paths = [
        config_path,
        mean_model_path,
        sigma_model_path,
        feature_state_path,
        metrics_path,
        report_path,
        predictions_path,
    ]
    if calibrator_path is not None:
        hash_paths.append(calibrator_path)
    if global_calibration_path is not None:
        hash_paths.append(global_calibration_path)
    if global_residuals_path is not None:
        hash_paths.append(global_residuals_path)
    if baseline_calibration_path is not None:
        hash_paths.append(baseline_calibration_path)
    if baseline_residuals_path is not None:
        hash_paths.append(baseline_residuals_path)
    hash_paths.extend(plots_dir.glob("*.png"))
    artifacts.write_hash_manifest(hash_paths, run_dir / "hashes.json")

    LOGGER.info("Training complete. Run dir: %s", run_dir)
    return 0


def _train_sigma_model(
    *,
    model_name: str,
    mean_model_name: str,
    best_mean_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    param_grid: dict[str, list],
    eps: float,
) -> object | None:
    try:
        mu_oof, oof_mask = models_sigma.build_oof_predictions(
            mean_model_name,
            X=X,
            y=y,
            splits=splits,
            seed=seed,
            param_grid={},
            fixed_params=best_mean_params,
        )
    except Exception as exc:
        LOGGER.warning("Sigma OOF predictions failed: %s", exc)
        return None
    if not oof_mask.any():
        LOGGER.warning("No OOF predictions available for sigma training.")
        return None
    z_target = models_sigma.build_sigma_targets(y[oof_mask], mu_oof[oof_mask], eps=eps)
    sigma_splits = _filter_splits_for_mask(splits, oof_mask)
    try:
        sigma_model = models_sigma.fit_sigma_model(
            model_name,
            X=X[oof_mask],
            z_target=z_target,
            splits=sigma_splits,
            seed=seed,
            param_grid=param_grid,
        )
    except ImportError as exc:
        LOGGER.warning("Skipping sigma model %s: %s", model_name, exc)
        return None
    return sigma_model


def _filter_splits_for_mask(
    splits: list[tuple[np.ndarray, np.ndarray]], mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not splits:
        return []
    mask_indices = np.where(mask)[0]
    index_map = {int(orig): idx for idx, orig in enumerate(mask_indices)}
    filtered = []
    for train_idx, val_idx in splits:
        train_filtered = [index_map[i] for i in train_idx if i in index_map]
        val_filtered = [index_map[i] for i in val_idx if i in index_map]
        if train_filtered and val_filtered:
            filtered.append((np.array(train_filtered), np.array(val_filtered)))
    return filtered


def _build_probabilities(
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    support_min: int,
    support_max: int,
    bin_specs: list[dict],
) -> dict[str, object]:
    pmf_rows = [
        distribution.normal_integer_pmf(
            float(mu_val),
            float(sigma_val),
            support_min=support_min,
            support_max=support_max,
        )
        for mu_val, sigma_val in zip(mu, sigma)
    ]
    pmf = np.vstack(pmf_rows)
    bin_probs = _bin_probabilities(pmf, support_min, bin_specs)
    return {"pmf": pmf, "bin_probs": bin_probs}


def _bin_probabilities(
    pmf: np.ndarray, support_min: int, bin_specs: list[dict]
) -> dict[str, np.ndarray]:
    probs: dict[str, np.ndarray] = {}
    for spec in bin_specs:
        name = spec.get("name")
        if not name:
            continue
        if spec.get("type") == "threshold":
            if "lt" in spec:
                cutoff = int(spec["lt"])
                idx = cutoff - support_min
                probs[name] = pmf[:, :idx].sum(axis=1)
            elif "ge" in spec:
                cutoff = int(spec["ge"])
                idx = cutoff - support_min
                probs[name] = pmf[:, idx:].sum(axis=1)
        elif spec.get("type") == "range":
            start = int(spec["start"])
            end = int(spec["end"])
            start_idx = max(start - support_min, 0)
            end_idx = min(end - support_min + 1, pmf.shape[1])
            probs[name] = pmf[:, start_idx:end_idx].sum(axis=1)
    return probs


def _probabilistic_metrics(
    *,
    y_true: np.ndarray,
    pmf: np.ndarray,
    support_min: int,
    bin_probs: dict[str, np.ndarray],
    bin_specs: list[dict],
) -> dict[str, object]:
    results = {"log_loss": metrics.log_loss_from_pmf(y_true, pmf, support_min=support_min)}
    brier: dict[str, float] = {}
    for spec in bin_specs:
        name = spec["name"]
        y_event = metrics.event_indicator(y_true, spec).astype(int)
        probs = bin_probs[name]
        if np.any(probs < -1e-6) or np.any(probs > 1 + 1e-6):
            LOGGER.warning(
                "Bin probabilities out of bounds for %s: min=%s max=%s",
                name,
                float(np.min(probs)),
                float(np.max(probs)),
            )
        brier[name] = metrics.brier_score(y_event, probs)
    results["brier_scores"] = brier
    return results


def _baseline_metrics(
    train_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    model_cols = [
        "gfs_tmax_f",
        "nam_tmax_f",
        "gefsatmosmean_tmax_f",
        "rap_tmax_f",
        "hrrr_tmax_f",
        "nbm_tmax_f",
    ]
    results: dict[str, dict[str, dict[str, float]]] = {}
    if not val_df.empty:
        ens_val = baselines.predict_ensemble_mean(val_df, model_cols)
        results["ensemble_mean_val"] = metrics.regression_metrics(
            val_df["actual_tmax_f"].to_numpy(), ens_val
        )
        climo_val = baselines.predict_climatology(
            train_df,
            val_df,
            label_col="actual_tmax_f",
            station_col="station_id",
            date_col="target_date_local",
            label_lag_days=2,
        )
        results["climatology_val"] = metrics.regression_metrics(
            val_df["actual_tmax_f"].to_numpy(), climo_val
        )
    if not test_df.empty:
        ens_test = baselines.predict_ensemble_mean(test_df, model_cols)
        results["ensemble_mean_test"] = metrics.regression_metrics(
            test_df["actual_tmax_f"].to_numpy(), ens_test
        )
        climo_test = baselines.predict_climatology(
            train_full_df,
            test_df,
            label_col="actual_tmax_f",
            station_col="station_id",
            date_col="target_date_local",
            label_lag_days=2,
        )
        results["climatology_test"] = metrics.regression_metrics(
            test_df["actual_tmax_f"].to_numpy(), climo_test
        )
    return results


def _build_dataset_summary(
    df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    return {
        "row_count": int(len(df)),
        "station_counts": df["station_id"].value_counts().to_dict(),
        "date_coverage": {
            "min": str(pd.to_datetime(df["target_date_local"]).min().date()),
            "max": str(pd.to_datetime(df["target_date_local"]).max().date()),
        },
        "missing_by_column": df.isna().sum().to_dict(),
        "split_counts": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }


def _build_dataset_metadata(
    df: pd.DataFrame, dataset_id: str, csv_path: Path, config
) -> dict:
    return {
        "dataset_id": dataset_id,
        "csv_path": str(csv_path),
        "schema_version": config.data.dataset_schema_version,
        "created_at": artifacts.utc_now_iso(),
        "row_count_raw": int(len(df)),
        "row_count": int(len(df)),
        "missing_by_column": df.isna().sum().to_dict(),
        "station_counts": df["station_id"].value_counts().to_dict(),
    }


def _feature_importance(
    model: object, feature_names: list[str], X: np.ndarray
) -> dict | None:
    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_")).reshape(-1)
        std = X.std(axis=0)
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


def _select_best_candidate(candidate_results: list[dict], primary: str) -> dict:
    best = None
    for candidate in candidate_results:
        if candidate.get("val_mae") is not None:
            if best is None or candidate["val_mae"] < best["val_mae"]:
                best = candidate
    if best is not None:
        return best
    for candidate in candidate_results:
        if candidate["model"] == primary:
            return candidate
    return candidate_results[0]


def _write_predictions(
    path: Path,
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    pmf: np.ndarray,
    bin_probs: dict[str, np.ndarray],
    support_min: int,
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
    records["support_min_f"] = support_min
    records["support_max_f"] = support_min + pmf.shape[1] - 1
    records.to_parquet(path, index=False, engine="pyarrow")


def _prepare_run_dir(root: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = root / "runs" / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run dir already exists: {run_dir} (set overwrite=true to reuse)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


if __name__ == "__main__":
    raise SystemExit(main())
