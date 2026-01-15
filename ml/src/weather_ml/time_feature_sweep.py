"""Time-structured feature sweep (E01-E100)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import yaml

from weather_ml import artifacts
from weather_ml import config as config_module
from weather_ml import dataset
from weather_ml import distribution
from weather_ml import derived_features
from weather_ml import hmm_utils
from weather_ml import models_mean
from weather_ml import report
from weather_ml import splits
from weather_ml import utils_seed
from weather_ml import validate
from weather_ml import time_feature_library as tfl

LOGGER = logging.getLogger(__name__)

MODEL_COLS = [
    "nbm_tmax_f",
    "gfs_tmax_f",
    "gefsatmosmean_tmax_f",
    "nam_tmax_f",
    "hrrr_tmax_f",
    "rap_tmax_f",
]
SPREAD_COL = "gefsatmos_tmp_spread_f"
GEFS_SPREAD_ALIAS = "gefs_spread"
CALENDAR_COLS = ["month", "day_of_year", "sin_doy", "cos_doy", "is_weekend"]
EPS = 1e-6


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    description: str
    build_features: Callable[["ExperimentContext"], "DerivedFeatureSet"]


@dataclass
class DerivedFeatureSet:
    features: pd.DataFrame
    formulas: list[dict]
    train_fitted: list[dict]


@dataclass
class ExperimentContext:
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    group_key: pd.Series
    truth_lag: int
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time feature sweep (E01-E100).")
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    parser.add_argument("--sweep-id", help="Optional sweep id override.")
    parser.add_argument("--sweep-root", help="Optional sweep output root.")
    parser.add_argument(
        "--allow-tuning",
        action="store_true",
        help="Enable hyperparameter tuning (default: fixed params).",
    )
    parser.add_argument(
        "--experiment-ids",
        nargs="*",
        help="Optional list of experiment ids to run (default: all).",
    )
    parser.add_argument(
        "--truth-lag",
        type=int,
        default=2,
        help="Truth lag in days for truth-based features (default: 2).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Bootstrap resamples for MAE delta CI (default: 10000).",
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
        else Path(config.artifacts.root_dir) / "time_feature_sweep" / sweep_id
    )
    sweep_root.mkdir(parents=True, exist_ok=True)

    df = dataset.load_csv(Path(config.data.csv_path))
    rules = validate.build_rules_from_config(config)
    validate.run_all_validations(df, rules)

    df = tfl.prepare_frame(df)
    df = _add_base_columns(df)

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

    experiments = _build_experiments()
    if args.experiment_ids:
        exp_set = {eid.upper() for eid in args.experiment_ids}
        experiments = [e for e in experiments if e.experiment_id.upper() in exp_set]
        if not experiments:
            raise ValueError("No experiments matched requested ids.")

    context = ExperimentContext(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        group_key=df["station_id"],
        truth_lag=int(args.truth_lag),
        seed=config.seeds.global_seed,
    )

    results = []
    baseline_id = "BASE"
    baseline_result = _run_baseline(
        config=config,
        context=context,
        sweep_root=sweep_root,
        allow_tuning=args.allow_tuning,
        cv_splits=cv_splits,
    )
    baseline_result["experiment_id"] = baseline_id
    baseline_result["description"] = "raw forecasts + spread + calendar (baseline)"
    results.append(baseline_result)

    for experiment in experiments:
        LOGGER.info("Running %s", experiment.experiment_id)
        run_dir = sweep_root / experiment.experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result = _run_experiment(
            experiment=experiment,
            config=config,
            context=context,
            run_dir=run_dir,
            allow_tuning=args.allow_tuning,
            cv_splits=cv_splits,
        )
        results.append(result)

    baseline_entry = next(entry for entry in results if entry["experiment_id"] == baseline_id)
    _apply_deltas_and_bootstrap(
        results,
        baseline_entry,
        test_rows=context.test_df,
        bootstrap_samples=args.bootstrap_samples,
        seed=config.seeds.global_seed,
    )

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
        "baseline_experiment_id": baseline_id,
        "experiments": results,
        "leaderboard_test_mae": _leaderboard(results, split="test", metric="mae"),
        "leaderboard_val_mae": _leaderboard(results, split="val", metric="mae"),
    }
    sweep_path = sweep_root / "time_feature_sweep.json"
    sweep_path.write_text(
        json.dumps(sweep_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    LOGGER.info("Sweep complete. Output: %s", sweep_path)
    return 0


def _add_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [col for col in MODEL_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")
    if SPREAD_COL in df.columns:
        df[GEFS_SPREAD_ALIAS] = df[SPREAD_COL].astype(float)
    df = tfl.add_calendar_features(df)
    df = tfl.add_ensemble_stats(df, MODEL_COLS)
    if "actual_tmax_f" in df.columns:
        df["resid_ens_mean"] = df["actual_tmax_f"] - df["ens_mean"]
        df["resid_ens_median"] = df["actual_tmax_f"] - df["ens_median"]
    return df


def _base_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = list(MODEL_COLS)
    if SPREAD_COL in df.columns:
        cols.append(SPREAD_COL)
    cols.extend(CALENDAR_COLS)
    return cols


def _min_periods(window: int) -> int:
    return int(np.ceil(window * 0.7))


def _formula_entry(name: str, formula: str, params: dict | None = None) -> dict:
    payload = {"name": name, "formula": formula}
    if params:
        payload["params"] = params
    return payload


def _add_feature(
    features: pd.DataFrame,
    formulas: list[dict],
    name: str,
    series: pd.Series | np.ndarray,
    formula: str,
    params: dict | None = None,
) -> None:
    features[name] = series
    formulas.append(_formula_entry(name, formula, params))


def _station_quantile(
    train_df: pd.DataFrame, column: str, q: float
) -> tuple[dict, float]:
    grouped = train_df.groupby("station_id")[column].quantile(q)
    default = float(train_df[column].quantile(q))
    return grouped.to_dict(), default


def _map_station_threshold(
    df: pd.DataFrame, thresholds: dict, default: float
) -> pd.Series:
    return df["station_id"].map(thresholds).fillna(default).astype(float)


def _seasonal_bias_maps(
    train_df: pd.DataFrame, cols: list[str], label_col: str
) -> dict[str, dict]:
    maps: dict[str, dict] = {}
    for col in cols:
        resid = train_df[label_col] - train_df[col]
        grouped = resid.groupby([train_df["station_id"], train_df["month"]]).mean()
        maps[col] = {
            "map": grouped.to_dict(),
            "default": float(resid.mean()),
        }
    return maps


def _apply_seasonal_bias(
    df: pd.DataFrame, bias_map: dict
) -> pd.Series:
    keys = list(zip(df["station_id"], df["month"]))
    series = pd.Series(keys, index=df.index).map(bias_map["map"])
    return series.fillna(bias_map["default"]).astype(float)


def _lagged_diff(series: pd.Series, group_key: pd.Series, lag: int) -> pd.Series:
    diff = series.groupby(group_key).diff()
    return diff.groupby(group_key).shift(lag)


def _ensure_columns_exist(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def _impute_features(
    features: pd.DataFrame, train_index: pd.Index
) -> tuple[pd.DataFrame, dict]:
    cleaned = features.replace([np.inf, -np.inf], np.nan)
    medians = cleaned.loc[train_index].median(numeric_only=True)
    medians = medians.fillna(0.0)
    filled = cleaned.fillna(medians)
    return filled, {"method": "train_median", "fill_values": medians.to_dict()}


def _run_baseline(
    *,
    config,
    context: ExperimentContext,
    sweep_root: Path,
    allow_tuning: bool,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    run_dir = sweep_root / "BASE"
    run_dir.mkdir(parents=True, exist_ok=True)
    derived = DerivedFeatureSet(
        features=pd.DataFrame(index=context.df.index),
        formulas=[],
        train_fitted=[],
    )
    return _run_with_features(
        experiment_id="BASE",
        description="raw forecasts + spread + calendar (baseline)",
        derived=derived,
        config=config,
        context=context,
        sweep_root=sweep_root,
        run_dir=run_dir,
        allow_tuning=allow_tuning,
        cv_splits=cv_splits,
    )


def _run_experiment(
    *,
    experiment: ExperimentDefinition,
    config,
    context: ExperimentContext,
    run_dir: Path,
    allow_tuning: bool,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    derived = experiment.build_features(context)
    return _run_with_features(
        experiment_id=experiment.experiment_id,
        description=experiment.description,
        derived=derived,
        config=config,
        context=context,
        sweep_root=run_dir.parent,
        run_dir=run_dir,
        allow_tuning=allow_tuning,
        cv_splits=cv_splits,
    )


def _run_with_features(
    *,
    experiment_id: str,
    description: str,
    derived: DerivedFeatureSet,
    config,
    context: ExperimentContext,
    sweep_root: Path,
    run_dir: Path,
    allow_tuning: bool,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    base_cols = _base_feature_columns(context.df)
    _ensure_columns_exist(context.df, base_cols, experiment_id)

    base_features = context.df[base_cols].astype(float).copy()
    feature_df = pd.concat([base_features, derived.features], axis=1)
    feature_columns = base_cols + list(derived.features.columns)

    imputed, impute_meta = _impute_features(feature_df, context.train_df.index)
    train_features = imputed.loc[context.train_df.index].astype(float)
    val_features = imputed.loc[context.val_df.index].astype(float)
    test_features = imputed.loc[context.test_df.index].astype(float)

    X_train = train_features.to_numpy(dtype=float)
    y_train = context.train_df["actual_tmax_f"].to_numpy(dtype=float)
    X_val = val_features.to_numpy(dtype=float)
    y_val = context.val_df["actual_tmax_f"].to_numpy(dtype=float)
    X_test = test_features.to_numpy(dtype=float)
    y_test = context.test_df["actual_tmax_f"].to_numpy(dtype=float)

    model_name = config.models.mean.primary
    base_model = models_mean.get_mean_model(model_name, seed=context.seed)
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
        pd.concat([context.train_df, context.val_df], ignore_index=True)
        if not context.val_df.empty
        else context.train_df.copy()
    )
    full_features = pd.concat([train_features, val_features], ignore_index=True)
    y_train_full = train_full_df["actual_tmax_f"].to_numpy(dtype=float)

    model_full = models_mean.get_mean_model(model_name, seed=context.seed)
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
        json.dumps(feature_columns, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    derived_meta = {
        "formulas": list(derived.formulas),
        "train_fitted": list(derived.train_fitted),
        "imputation": impute_meta,
    }

    config_snapshot = _config_snapshot(
        config,
        experiment_id=experiment_id,
        description=description,
        feature_columns=feature_columns,
        derived_meta=derived_meta,
    )
    config_path = run_dir / "config_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    dataset_id = artifacts.compute_dataset_id(
        Path(config.data.csv_path),
        config.data.dataset_schema_version,
        {"experiment_id": experiment_id, "feature_columns": feature_columns},
    )
    dataset_dir = sweep_root / "datasets" / dataset_id
    all_df = pd.concat(
        [context.train_df, context.val_df, context.test_df], ignore_index=True
    )
    data_parquet = artifacts.snapshot_to_parquet(all_df, dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    metadata = _build_dataset_metadata(
        context.train_df, context.val_df, context.test_df, dataset_id, config
    )
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
    joblib.dump(
        {"feature_columns": feature_columns, "impute_values": impute_meta},
        feature_state_path,
    )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    feature_importance = _feature_importance(
        model_full, feature_columns, full_features.to_numpy(dtype=float)
    )
    report_path = run_dir / "report.md"
    report.write_report(
        report_path,
        dataset_summary=_build_dataset_summary(
            context.train_df, context.val_df, context.test_df
        ),
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
        predictions_path, context.test_df, mu_test, sigma_test, pmf, bin_probs, config
    )

    experiment_meta = {
        "experiment_id": experiment_id,
        "description": description,
        "base_features": base_cols,
        "derived_features": derived_meta,
    }
    (run_dir / "experiment_meta.json").write_text(
        json.dumps(experiment_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "experiment_feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2, sort_keys=True),
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
        run_dir / "experiment_meta.json",
        run_dir / "experiment_feature_columns.json",
    ]
    hash_paths.extend(plots_dir.glob("*.png"))
    hashes_path = run_dir / "hashes.json"
    artifacts.write_hash_manifest(hash_paths, hashes_path)
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))

    worst_days = _worst_test_days(context.test_df, y_test, mu_test)
    uses_spread = SPREAD_COL in base_cols
    calendar_enabled = all(col in feature_columns for col in CALENDAR_COLS)

    return {
        "experiment_id": experiment_id,
        "description": description,
        "raw_model_cols_used": list(MODEL_COLS),
        "uses_spread_feature": uses_spread,
        "calendar_enabled": calendar_enabled,
        "derived_features": derived_meta,
        "final_feature_columns": feature_columns,
        "num_features": int(len(feature_columns)),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "deltas_vs_baseline": {},
        "worst_test_days": worst_days,
        "run_dir": str(run_dir),
        "artifact_hashes": hashes,
        "_y_test": y_test,
        "_mu_test": mu_test,
    }


def _apply_deltas_and_bootstrap(
    results: list[dict],
    baseline_entry: dict,
    *,
    test_rows: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> None:
    baseline_metrics = baseline_entry["metrics"]["test"]
    baseline_y = baseline_entry.get("_y_test")
    baseline_mu = baseline_entry.get("_mu_test")
    if baseline_y is None or baseline_mu is None:
        return
    rng = np.random.default_rng(seed)
    n = len(baseline_y)

    for entry in results:
        y_true = entry.get("_y_test")
        y_pred = entry.get("_mu_test")
        if y_true is None or y_pred is None:
            continue
        metrics = entry["metrics"]["test"]
        deltas = {
            "delta_test_mae": float(metrics["mae"] - baseline_metrics["mae"]),
            "delta_test_rmse": float(metrics["rmse"] - baseline_metrics["rmse"]),
            "delta_test_bias": float(metrics["bias"] - baseline_metrics["bias"]),
            "delta_test_corr": float(metrics["corr"] - baseline_metrics["corr"]),
            "delta_test_medianAE": float(
                metrics["medianAE"] - baseline_metrics["medianAE"]
            ),
            "delta_test_maxAE": float(metrics["maxAE"] - baseline_metrics["maxAE"]),
        }

        if entry["experiment_id"] != baseline_entry["experiment_id"] and n > 0:
            deltas_boot = np.zeros(bootstrap_samples, dtype=float)
            for idx in range(bootstrap_samples):
                sample_idx = rng.integers(0, n, size=n)
                mae_exp = float(
                    np.mean(np.abs(y_pred[sample_idx] - y_true[sample_idx]))
                )
                mae_base = float(
                    np.mean(np.abs(baseline_mu[sample_idx] - baseline_y[sample_idx]))
                )
                deltas_boot[idx] = mae_exp - mae_base
            deltas["mae_delta_bootstrap"] = {
                "mean": float(np.mean(deltas_boot)),
                "p025": float(np.percentile(deltas_boot, 2.5)),
                "p975": float(np.percentile(deltas_boot, 97.5)),
                "samples": int(bootstrap_samples),
            }
        else:
            deltas["mae_delta_bootstrap"] = {
                "mean": 0.0,
                "p025": 0.0,
                "p975": 0.0,
                "samples": int(bootstrap_samples),
            }

        entry["deltas_vs_baseline"] = deltas

    for entry in results:
        entry.pop("_y_test", None)
        entry.pop("_mu_test", None)


def _regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {}
    error = y_pred - y_true
    abs_error = np.abs(error)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "medianAE": float(np.median(abs_error)),
        "maxAE": float(np.max(abs_error)),
        "corr": corr,
        "n": int(len(y_true)),
    }


def _worst_test_days(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> list[dict]:
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
            if "lt" in spec:
                total = float(np.sum(row[: spec["lt"] - config.distribution.support_min_f]))
            elif "ge" in spec:
                total = float(
                    np.sum(row[spec["ge"] - config.distribution.support_min_f :])
                )
            else:
                total = float(np.sum(row))
            bin_probs[name].append(total)
    return {name: np.array(vals, dtype=float) for name, vals in bin_probs.items()}


def _feature_importance(
    model: object, feature_names: list[str], x: np.ndarray
) -> dict | None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pairs = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, importances)
        ]
        pairs = sorted(pairs, key=lambda item: abs(item["importance"]), reverse=True)
        return {"type": "tree", "top_features": pairs[:50]}
    if hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        pairs = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, coef)
        ]
        pairs = sorted(pairs, key=lambda item: abs(item["importance"]), reverse=True)
        return {"type": "linear", "top_features": pairs[:50]}
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
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_id: str,
    config,
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


def _config_snapshot(
    config,
    *,
    experiment_id: str,
    description: str,
    feature_columns: list[str],
    derived_meta: dict,
) -> dict:
    payload = asdict(config)
    payload["experiment"] = {
        "experiment_id": experiment_id,
        "description": description,
        "feature_columns": feature_columns,
        "derived_features": derived_meta,
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
    }


def _fixed_params(param_grid: dict[str, list]) -> dict[str, float]:
    params: dict[str, float] = {}
    for key, values in param_grid.items():
        if isinstance(values, list) and values:
            params[key] = values[0]
        elif not isinstance(values, list):
            params[key] = values
    return params


def _leaderboard(
    results: list[dict], *, split: str, metric: str
) -> list[dict]:
    entries = []
    for entry in results:
        metrics = entry["metrics"].get(split)
        if metrics is None:
            continue
        value = metrics.get(metric)
        if value is None:
            continue
        entries.append(
            {
                "experiment_id": entry["experiment_id"],
                "metric": metric,
                "value": float(value),
            }
        )
    return sorted(entries, key=lambda item: item["value"])


def _argmin_model(values: pd.DataFrame, model_cols: list[str]) -> pd.Series:
    priority = {name: idx for idx, name in enumerate(derived_features.RAW_MODEL_ORDER)}
    arr = values[model_cols].to_numpy(dtype=float)
    output = []
    for row in arr:
        best = None
        for idx, val in enumerate(row):
            col = model_cols[idx]
            if np.isnan(val):
                continue
            key = (val, priority.get(col, idx))
            if best is None or key < best[0]:
                best = (key, col)
        output.append(best[1] if best else None)
    return pd.Series(output, index=values.index)


def _dominant_outlier_id(df: pd.DataFrame, model_cols: list[str]) -> pd.Series:
    diffs = (df[model_cols].sub(df["ens_median"], axis=0)).abs().to_numpy(dtype=float)
    priority = {name: idx for idx, name in enumerate(derived_features.RAW_MODEL_ORDER)}
    output = []
    for row in diffs:
        best = None
        for idx, val in enumerate(row):
            col = model_cols[idx]
            key = (-val, priority.get(col, idx))
            if best is None or key < best[0]:
                best = (key, col)
        output.append(best[1])
    return pd.Series(output, index=df.index)


def _standardize_features(
    train_df: pd.DataFrame, df: pd.DataFrame, cols: list[str]
) -> tuple[np.ndarray, dict, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(train_df[cols].to_numpy(dtype=float))
    all_scaled = scaler.transform(df[cols].to_numpy(dtype=float))
    meta = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "cols": cols,
    }
    return all_scaled, meta, scaler


def _knn_neighbors(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    *,
    group_key: pd.Series,
    truth_lag: int,
    lookback_days: int,
    k: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = len(df)
    neighbors: list[np.ndarray] = [np.array([], dtype=int) for _ in range(n)]
    distances: list[np.ndarray] = [np.array([], dtype=float) for _ in range(n)]
    dates = pd.to_datetime(df["target_date_local"]).values.astype("datetime64[D]")
    stations = group_key.to_numpy()

    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        station_dates = dates[idx]
        station_features = features_scaled[idx]
        for pos, row_idx in enumerate(idx):
            current = station_features[pos]
            if not np.isfinite(current).all():
                continue
            cutoff = station_dates[pos] - np.timedelta64(truth_lag, "D")
            start_date = station_dates[pos] - np.timedelta64(lookback_days, "D")
            start = np.searchsorted(station_dates, start_date, side="left")
            end = np.searchsorted(station_dates, cutoff, side="right")
            if end <= start:
                continue
            cand_idx = idx[start:end]
            cand_features = station_features[start:end]
            finite_mask = np.isfinite(cand_features).all(axis=1)
            if not finite_mask.any():
                continue
            cand_idx = cand_idx[finite_mask]
            cand_features = cand_features[finite_mask]
            dists = np.linalg.norm(cand_features - current, axis=1)
            order = np.argsort(dists)[:k]
            neighbors[row_idx] = cand_idx[order]
            distances[row_idx] = dists[order]
    return neighbors, distances


def _knn_mean_std(
    neighbors: list[np.ndarray], distances: list[np.ndarray], values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(values)
    mean = np.full(n, np.nan, dtype=float)
    std = np.full(n, np.nan, dtype=float)
    mean_dist = np.full(n, np.nan, dtype=float)
    for idx, neigh in enumerate(neighbors):
        if neigh.size == 0:
            continue
        vals = values[neigh]
        mean[idx] = float(np.mean(vals))
        std[idx] = float(np.std(vals, ddof=0))
        mean_dist[idx] = float(np.mean(distances[idx]))
    return mean, std, mean_dist


def _knn_quantiles(
    neighbors: list[np.ndarray], values: np.ndarray, quantiles: Iterable[float]
) -> dict[float, np.ndarray]:
    outputs = {q: np.full(len(values), np.nan, dtype=float) for q in quantiles}
    for idx, neigh in enumerate(neighbors):
        if neigh.size == 0:
            continue
        vals = values[neigh]
        for q in outputs:
            outputs[q][idx] = float(np.quantile(vals, q))
    return outputs


def _knn_prob(
    neighbors: list[np.ndarray], flags: np.ndarray
) -> np.ndarray:
    probs = np.full(len(flags), np.nan, dtype=float)
    for idx, neigh in enumerate(neighbors):
        if neigh.size == 0:
            continue
        probs[idx] = float(np.mean(flags[neigh]))
    return probs


def _knn_kernel_resid(
    neighbors: list[np.ndarray],
    distances: list[np.ndarray],
    resid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(resid)
    weighted = np.full(n, np.nan, dtype=float)
    nearest = np.full(n, np.nan, dtype=float)
    eff_n = np.full(n, np.nan, dtype=float)
    for idx, neigh in enumerate(neighbors):
        if neigh.size == 0:
            continue
        dists = distances[idx]
        if dists.size == 0:
            continue
        h = float(np.median(dists))
        if not np.isfinite(h) or h <= 0:
            h = float(np.mean(dists)) if np.mean(dists) > 0 else 1e-6
        weights = np.exp(-(dists**2) / (h**2))
        sum_w = float(np.sum(weights))
        if sum_w <= 0:
            continue
        weighted[idx] = float(np.sum(weights * resid[neigh]) / sum_w)
        nearest[idx] = float(np.min(dists))
        sum_w2 = float(np.sum(weights**2))
        eff_n[idx] = float((sum_w**2) / sum_w2) if sum_w2 > 0 else float(neigh.size)
    return weighted, nearest, eff_n


def _rolling_abs_mean(
    series: pd.Series, *, window: int, lag: int, group_key: pd.Series
) -> pd.Series:
    return tfl.rolling_mean(
        series.abs(), window=window, min_periods=_min_periods(window), lag=lag, group_key=group_key
    )


def _rolling_rmse(
    series: pd.Series, *, window: int, lag: int, group_key: pd.Series
) -> pd.Series:
    mean_sq = tfl.rolling_mean(
        series**2, window=window, min_periods=_min_periods(window), lag=lag, group_key=group_key
    )
    return np.sqrt(np.maximum(mean_sq, 0.0))


def _safe_divide(numer: np.ndarray, denom: np.ndarray, default: float = 0.0) -> np.ndarray:
    output = np.full_like(numer, default, dtype=float)
    mask = np.isfinite(numer) & np.isfinite(denom) & (denom != 0)
    output[mask] = numer[mask] / denom[mask]
    return output


def _rowwise_inverse_weights(values: np.ndarray, eps: float) -> np.ndarray:
    weights = 1.0 / (values + eps)
    weights[~np.isfinite(weights)] = 0.0
    sum_w = np.sum(weights, axis=1)
    uniform = np.full_like(weights, 1.0 / weights.shape[1])
    return np.divide(weights, sum_w[:, None], out=uniform, where=sum_w[:, None] > 0)


def _rowwise_entropy(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.clip(weights, eps, 1.0)
    return -np.sum(w * np.log(w), axis=1)


def _fit_ridge_coeffs(
    X: np.ndarray, y: np.ndarray, *, l2: float
) -> tuple[float, np.ndarray]:
    n_samples, n_features = X.shape
    if n_samples == 0:
        return 0.0, np.zeros(n_features, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    Xs = (X - mean) / scale
    design = np.concatenate([np.ones((n_samples, 1)), Xs], axis=1)
    penalty = np.eye(n_features + 1)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + l2 * penalty
    rhs = design.T @ y
    try:
        coeff = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    b0_std = coeff[0]
    b_std = coeff[1:]
    b = b_std / scale
    b0 = b0_std - np.sum(b_std * mean / scale)
    return float(b0), b


def _solve_simplex_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    init: np.ndarray | None = None,
) -> np.ndarray:
    n_models = X.shape[1]
    if init is None:
        init = np.full(n_models, 1.0 / n_models, dtype=float)

    def _obj(w: np.ndarray) -> float:
        resid = y - X @ w
        return 0.5 * float(np.sum(resid**2)) + 0.5 * float(l2) * float(np.sum(w**2))

    def _grad(w: np.ndarray) -> np.ndarray:
        resid = y - X @ w
        return -X.T @ resid + l2 * w

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    res = minimize(
        _obj,
        init,
        jac=_grad,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-9, "disp": False},
    )
    if not res.success or not np.isfinite(res.x).all():
        return init
    weights = np.clip(res.x, 0.0, 1.0)
    total = float(weights.sum())
    if total > 0:
        weights /= total
    else:
        weights = init
    return weights


def _knn_neighbors_mahalanobis(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    *,
    inv_cov: np.ndarray,
    group_key: pd.Series,
    truth_lag: int,
    lookback_days: int,
    k: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = len(df)
    neighbors: list[np.ndarray] = [np.array([], dtype=int) for _ in range(n)]
    distances: list[np.ndarray] = [np.array([], dtype=float) for _ in range(n)]
    dates = pd.to_datetime(df["target_date_local"]).values.astype("datetime64[D]")
    stations = group_key.to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        station_dates = dates[idx]
        station_features = features_scaled[idx]
        for pos, row_idx in enumerate(idx):
            current = station_features[pos]
            if not np.isfinite(current).all():
                continue
            cutoff = station_dates[pos] - np.timedelta64(truth_lag, "D")
            start_date = station_dates[pos] - np.timedelta64(lookback_days, "D")
            start = np.searchsorted(station_dates, start_date, side="left")
            end = np.searchsorted(station_dates, cutoff, side="right")
            if end <= start:
                continue
            cand_idx = idx[start:end]
            cand_features = station_features[start:end]
            finite_mask = np.isfinite(cand_features).all(axis=1)
            if not finite_mask.any():
                continue
            cand_idx = cand_idx[finite_mask]
            cand_features = cand_features[finite_mask]
            diffs = cand_features - current
            dists = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))
            order = np.argsort(dists)[:k]
            neighbors[row_idx] = cand_idx[order]
            distances[row_idx] = dists[order]
    return neighbors, distances


def _exp_e01(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    for window in (7, 30, 60):
        min_p = _min_periods(window)
        name = f"ens_mean_roll_mean_{window}_l1"
        series = tfl.rolling_mean(
            df["ens_mean"], window=window, min_periods=min_p, lag=1, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            name,
            series,
            "roll_mean(ens_mean)",
            {"window": window, "lag": 1, "min_periods": min_p},
        )
    for window in (7, 30, 60):
        min_p = _min_periods(window)
        name = f"ens_median_roll_mean_{window}_l1"
        series = tfl.rolling_mean(
            df["ens_median"], window=window, min_periods=min_p, lag=1, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            name,
            series,
            "roll_mean(ens_median)",
            {"window": window, "lag": 1, "min_periods": min_p},
        )
    _add_feature(
        features,
        formulas,
        "ens_mean_dev_from_rm30",
        df["ens_mean"] - features["ens_mean_roll_mean_30_l1"],
        "ens_mean - ens_mean_roll_mean_30_l1",
    )
    _add_feature(
        features,
        formulas,
        "ens_median_dev_from_rm30",
        df["ens_median"] - features["ens_median_roll_mean_30_l1"],
        "ens_median - ens_median_roll_mean_30_l1",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e02(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    ewm14 = tfl.ewm_mean(
        df["ens_mean"], halflife=14, min_periods=10, lag=1, group_key=gk
    )
    ewm30 = tfl.ewm_mean(
        df["ens_mean"], halflife=30, min_periods=10, lag=1, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_ewm_mean_hl14_l1",
        ewm14,
        "ewm_mean(ens_mean)",
        {"halflife": 14, "lag": 1, "min_periods": 10},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_ewm_mean_hl30_l1",
        ewm30,
        "ewm_mean(ens_mean)",
        {"halflife": 30, "lag": 1, "min_periods": 10},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_shock_hl30",
        df["ens_mean"] - ewm30,
        "ens_mean - ens_mean_ewm_mean_hl30_l1",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_shock_hl14",
        df["ens_mean"] - ewm14,
        "ens_mean - ens_mean_ewm_mean_hl14_l1",
    )
    ewm_med14 = tfl.ewm_mean(
        df["ens_median"], halflife=14, min_periods=10, lag=1, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        "ens_median_ewm_mean_hl14_l1",
        ewm_med14,
        "ewm_mean(ens_median)",
        {"halflife": 14, "lag": 1, "min_periods": 10},
    )
    _add_feature(
        features,
        formulas,
        "ens_median_shock_hl14",
        df["ens_median"] - ewm_med14,
        "ens_median - ens_median_ewm_mean_hl14_l1",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e03(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    slope15 = tfl.rolling_slope(
        df["ens_mean"], window=15, min_periods=_min_periods(15), lag=1, group_key=gk
    )
    slope60 = tfl.rolling_slope(
        df["ens_mean"], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_slope_15_l1",
        slope15,
        "rolling_slope(ens_mean)",
        {"window": 15, "lag": 1, "min_periods": _min_periods(15)},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_slope_60_l1",
        slope60,
        "rolling_slope(ens_mean)",
        {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_accel_proxy",
        slope15 - slope60,
        "ens_mean_slope_15_l1 - ens_mean_slope_60_l1",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e04(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    min_p = _min_periods(60)
    q10 = tfl.rolling_quantile(
        df["ens_mean"], window=60, min_periods=min_p, lag=1, q=0.10, group_key=gk
    )
    q50 = tfl.rolling_quantile(
        df["ens_mean"], window=60, min_periods=min_p, lag=1, q=0.50, group_key=gk
    )
    q90 = tfl.rolling_quantile(
        df["ens_mean"], window=60, min_periods=min_p, lag=1, q=0.90, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_q10_60_l1",
        q10,
        "rolling_quantile(ens_mean, q=0.10)",
        {"window": 60, "lag": 1, "min_periods": min_p},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_q50_60_l1",
        q50,
        "rolling_quantile(ens_mean, q=0.50)",
        {"window": 60, "lag": 1, "min_periods": min_p},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_q90_60_l1",
        q90,
        "rolling_quantile(ens_mean, q=0.90)",
        {"window": 60, "lag": 1, "min_periods": min_p},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_envelope_width_60",
        q90 - q10,
        "ens_mean_q90_60_l1 - ens_mean_q10_60_l1",
    )
    prank = tfl.percent_rank(
        df["ens_mean"], window=60, min_periods=min_p, lag=1, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_prank_60_l1",
        prank,
        "percent_rank(ens_mean)",
        {"window": 60, "lag": 1, "min_periods": min_p},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e05(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    metrics = ["ens_std", "ens_range", "ens_iqr", "ens_mad"]
    if GEFS_SPREAD_ALIAS in df.columns:
        metrics.append(GEFS_SPREAD_ALIAS)
    for metric in metrics:
        for window in (7, 30):
            min_p = _min_periods(window)
            name = f"{metric}_roll_mean_{window}_l1"
            series = tfl.rolling_mean(
                df[metric], window=window, min_periods=min_p, lag=1, group_key=gk
            )
            _add_feature(
                features,
                formulas,
                name,
                series,
                f"roll_mean({metric})",
                {"window": window, "lag": 1, "min_periods": min_p},
            )
        name = f"{metric}_roll_std_30_l1"
        series = tfl.rolling_std(
            df[metric], window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            name,
            series,
            f"roll_std({metric})",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e06(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    ewm7 = tfl.ewm_mean(
        df["ens_std"], halflife=7, min_periods=10, lag=1, group_key=gk
    )
    ewm30 = tfl.ewm_mean(
        df["ens_std"], halflife=30, min_periods=10, lag=1, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        "ens_std_ewm_hl7_l1",
        ewm7,
        "ewm_mean(ens_std)",
        {"halflife": 7, "lag": 1, "min_periods": 10},
    )
    _add_feature(
        features,
        formulas,
        "ens_std_ewm_hl30_l1",
        ewm30,
        "ewm_mean(ens_std)",
        {"halflife": 30, "lag": 1, "min_periods": 10},
    )
    _add_feature(
        features,
        formulas,
        "ens_std_shock_hl30",
        df["ens_std"] - ewm30,
        "ens_std - ens_std_ewm_hl30_l1",
    )
    if GEFS_SPREAD_ALIAS in df.columns:
        ewm14 = tfl.ewm_mean(
            df[GEFS_SPREAD_ALIAS], halflife=14, min_periods=10, lag=1, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            "gefs_spread_ewm_hl14_l1",
            ewm14,
            "ewm_mean(gefs_spread)",
            {"halflife": 14, "lag": 1, "min_periods": 10},
        )
        _add_feature(
            features,
            formulas,
            "gefs_spread_shock_hl14",
            df[GEFS_SPREAD_ALIAS] - ewm14,
            "gefs_spread - gefs_spread_ewm_hl14_l1",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e07(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    metrics = ["ens_std", "ens_range"]
    if GEFS_SPREAD_ALIAS in df.columns:
        metrics.append(GEFS_SPREAD_ALIAS)
    for metric in metrics:
        rm = tfl.rolling_mean(
            df[metric], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
        )
        rs = tfl.rolling_std(
            df[metric], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            f"{metric}_rm60_l1",
            rm,
            f"roll_mean({metric})",
            {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"{metric}_rs60_l1",
            rs,
            f"roll_std({metric})",
            {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
        )
        z = (df[metric] - rm) / (rs + 0.1)
        _add_feature(
            features,
            formulas,
            f"{metric}_z60_l1",
            z,
            f"({metric} - {metric}_rm60_l1) / ({metric}_rs60_l1 + eps)",
            {"eps": 0.1},
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e08(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thresholds, default = _station_quantile(ctx.train_df, "ens_std", 0.90)
    thr = _map_station_threshold(df, thresholds, default)
    hi_spread = (df["ens_std"] > thr).astype(int)
    train_fitted.append(
        {
            "name": "thr_spread_hi",
            "fit_on": "train",
            "description": "q90_train(ens_std) per station",
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    _add_feature(
        features,
        formulas,
        "hi_spread_count_7_l1",
        tfl.rolling_event_count(
            hi_spread, window=7, min_periods=_min_periods(7), lag=1, group_key=gk
        ),
        "roll_sum(hi_spread)",
        {"window": 7, "lag": 1, "min_periods": _min_periods(7)},
    )
    _add_feature(
        features,
        formulas,
        "hi_spread_frac_30_l1",
        tfl.rolling_event_mean(
            hi_spread, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        ),
        "roll_mean(hi_spread)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    _add_feature(
        features,
        formulas,
        "hi_spread_streak_len_l1",
        tfl.streak_length(hi_spread, lag=1, cap=60, group_key=gk),
        "streak_length(hi_spread)",
        {"lag": 1, "cap": 60},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e09(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    std_cols = []
    for col in MODEL_COLS:
        std15 = tfl.rolling_std(
            df[col], window=15, min_periods=_min_periods(15), lag=1, group_key=gk
        )
        std60 = tfl.rolling_std(
            df[col], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
        )
        name15 = f"{col}_roll_std_15_l1"
        name60 = f"{col}_roll_std_60_l1"
        _add_feature(
            features,
            formulas,
            name15,
            std15,
            f"roll_std({col})",
            {"window": 15, "lag": 1, "min_periods": _min_periods(15)},
        )
        _add_feature(
            features,
            formulas,
            name60,
            std60,
            f"roll_std({col})",
            {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
        )
        std_cols.append(name15)
    std_vals = features[std_cols].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "model_vol_mean_15",
        np.nanmean(std_vals, axis=1),
        "mean({model}_roll_std_15_l1)",
    )
    _add_feature(
        features,
        formulas,
        "model_vol_max_15",
        np.nanmax(std_vals, axis=1),
        "max({model}_roll_std_15_l1)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e10(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    drift_cols = []
    for col in MODEL_COLS:
        drift = df[col] - df["ens_mean"]
        mean30 = tfl.rolling_mean(
            drift, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        )
        std30 = tfl.rolling_std(
            drift, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        )
        name_mean = f"drift_{col}_roll_mean_30_l1"
        name_std = f"drift_{col}_roll_std_30_l1"
        _add_feature(
            features,
            formulas,
            name_mean,
            mean30,
            f"roll_mean({col} - ens_mean)",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
        _add_feature(
            features,
            formulas,
            name_std,
            std30,
            f"roll_std({col} - ens_mean)",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
        drift_cols.append(name_mean)
    drift_vals = features[drift_cols].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "drift_abs_max_rm30",
        np.nanmax(np.abs(drift_vals), axis=1),
        "max(|drift_*_roll_mean_30_l1|)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e11(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    ranks = tfl.ranks_with_tie_break(df, MODEL_COLS)
    for col in ranks.columns:
        mean30 = tfl.rolling_mean(
            ranks[col], window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        )
        std30 = tfl.rolling_std(
            ranks[col], window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            f"{col}_roll_mean_30_l1",
            mean30,
            f"roll_mean({col})",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
        _add_feature(
            features,
            formulas,
            f"{col}_roll_std_30_l1",
            std30,
            f"roll_std({col})",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
    top_id = tfl.argmax_with_tie_break(df, MODEL_COLS)
    freq_cols = []
    for model in MODEL_COLS:
        indicator = (top_id == model).astype(int)
        freq = tfl.rolling_event_mean(
            indicator, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        )
        freq_cols.append(freq)
    freq_stack = np.vstack([col.to_numpy(dtype=float) for col in freq_cols]).T
    entropy = -np.sum(freq_stack * np.log(freq_stack + 1e-9), axis=1)
    _add_feature(
        features,
        formulas,
        "top_identity_entropy_30_l1",
        entropy,
        "entropy(top_model_freq_30_l1)",
        {"window": 30, "lag": 1, "eps": 1e-9},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e12(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    last_deltas = []
    for col in MODEL_COLS:
        diff = df[col].groupby(gk).diff()
        delta_last = diff.groupby(gk).shift(1)
        last_name = f"delta_{col}_last_l1"
        _add_feature(
            features,
            formulas,
            last_name,
            delta_last,
            f"diff({col}) shifted by 1",
        )
        absmean = tfl.rolling_mean(
            diff.abs(), window=15, min_periods=_min_periods(15), lag=1, group_key=gk
        )
        std15 = tfl.rolling_std(
            diff, window=15, min_periods=_min_periods(15), lag=1, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            f"delta_{col}_absmean_15_l1",
            absmean,
            f"roll_mean(|diff({col})|)",
            {"window": 15, "lag": 1, "min_periods": _min_periods(15)},
        )
        _add_feature(
            features,
            formulas,
            f"delta_{col}_std_15_l1",
            std15,
            f"roll_std(diff({col}))",
            {"window": 15, "lag": 1, "min_periods": _min_periods(15)},
        )
        last_deltas.append(delta_last)
    last_vals = np.vstack([col.to_numpy(dtype=float) for col in last_deltas]).T
    _add_feature(
        features,
        formulas,
        "delta_std_across_models_last",
        np.nanstd(last_vals, axis=1),
        "std(delta_model_last_l1)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e13(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    diff = df["ens_mean"].groupby(gk).diff()
    delta_last = diff.groupby(gk).shift(1)
    _add_feature(
        features,
        formulas,
        "ens_mean_delta1_l1",
        delta_last,
        "diff(ens_mean) shifted by 1",
    )
    mean7 = tfl.rolling_mean(
        diff, window=7, min_periods=_min_periods(7), lag=1, group_key=gk
    )
    std7 = tfl.rolling_std(
        diff, window=7, min_periods=_min_periods(7), lag=1, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_delta1_mean_7_l1",
        mean7,
        "roll_mean(diff(ens_mean))",
        {"window": 7, "lag": 1, "min_periods": _min_periods(7)},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_delta1_std_7_l1",
        std7,
        "roll_std(diff(ens_mean))",
        {"window": 7, "lag": 1, "min_periods": _min_periods(7)},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_delta1_sign_7",
        np.sign(mean7),
        "sign(ens_mean_delta1_mean_7_l1)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e14(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    delta_last = []
    for col in MODEL_COLS:
        diff = df[col].groupby(gk).diff()
        delta_last.append(diff.groupby(gk).shift(1))
    delta_vals = np.vstack([col.to_numpy(dtype=float) for col in delta_last]).T
    _add_feature(
        features,
        formulas,
        "delta1_std_models_l1",
        np.nanstd(delta_vals, axis=1),
        "std(delta_model_last_l1)",
    )
    _add_feature(
        features,
        formulas,
        "delta1_range_models_l1",
        np.nanmax(delta_vals, axis=1) - np.nanmin(delta_vals, axis=1),
        "max(delta_model_last_l1) - min(delta_model_last_l1)",
    )
    median_sign = np.sign(np.nanmedian(delta_vals, axis=1))
    signs = np.sign(delta_vals)
    agreement = np.mean(signs == median_sign[:, None], axis=1)
    _add_feature(
        features,
        formulas,
        "trend_agreement_frac_l1",
        agreement,
        "mean(sign(delta_model_last_l1) == sign(median(delta_model_last_l1)))",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e15(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    diff = df["ens_mean"].groupby(gk).diff()
    mom_sign = np.sign(diff).fillna(0)
    _add_feature(
        features,
        formulas,
        "mom_sign_changes_7_l1",
        tfl.switch_count(
            mom_sign, window=7, min_periods=_min_periods(7), lag=1, group_key=gk
        ),
        "switch_count(mom_sign)",
        {"window": 7, "lag": 1, "min_periods": _min_periods(7)},
    )
    dd2 = diff - diff.groupby(gk).shift(1)
    dd2_abs = dd2.abs().groupby(gk).shift(1)
    _add_feature(
        features,
        formulas,
        "ens_mean_dd2_abs_l1",
        dd2_abs,
        "|diff(ens_mean) - diff(ens_mean).lag1| shifted by 1",
    )
    reversal = (diff * diff.groupby(gk).shift(1) < 0).astype(int)
    _add_feature(
        features,
        formulas,
        "reversal_last_l1",
        reversal.groupby(gk).shift(1),
        "1[diff(ens_mean)*diff(ens_mean).lag1 < 0] shifted by 1",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e16(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    diff = df["ens_mean"].groupby(gk).diff()
    delta_last = diff.groupby(gk).shift(1)
    std60 = tfl.rolling_std(
        df["ens_mean"], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
    )
    rm60 = tfl.rolling_mean(
        df["ens_mean"], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_delta1_l1",
        delta_last,
        "diff(ens_mean) shifted by 1",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_std_60_l1",
        std60,
        "roll_std(ens_mean)",
        {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
    )
    _add_feature(
        features,
        formulas,
        "scaled_momentum_60",
        delta_last / (std60 + 0.1),
        "ens_mean_delta1_l1 / (ens_mean_std_60_l1 + eps)",
        {"eps": 0.1},
    )
    _add_feature(
        features,
        formulas,
        "momentum_to_level",
        delta_last / (rm60.abs() + 0.1),
        "ens_mean_delta1_l1 / (|ens_mean_roll_mean_60_l1| + eps)",
        {"eps": 0.1},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e17(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    rm7 = tfl.rolling_mean(
        df["ens_mean"], window=7, min_periods=_min_periods(7), lag=1, group_key=gk
    )
    rm60 = tfl.rolling_mean(
        df["ens_mean"], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
    )
    std60 = tfl.rolling_std(
        df["ens_mean"], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
    )
    level_shift = rm7 - rm60
    _add_feature(
        features,
        formulas,
        "level_shift_7v60",
        level_shift,
        "ens_mean_roll_mean_7_l1 - ens_mean_roll_mean_60_l1",
    )
    _add_feature(
        features,
        formulas,
        "level_shift_z_7v60",
        level_shift / (std60 + 0.1),
        "level_shift_7v60 / (ens_mean_std_60_l1 + eps)",
        {"eps": 0.1},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e18(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    rm7 = tfl.rolling_mean(
        df["ens_std"], window=7, min_periods=_min_periods(7), lag=1, group_key=gk
    )
    rm60 = tfl.rolling_mean(
        df["ens_std"], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
    )
    rs60 = tfl.rolling_std(
        df["ens_std"], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
    )
    shift = rm7 - rm60
    _add_feature(
        features,
        formulas,
        "spread_shift_7v60",
        shift,
        "ens_std_rm7_l1 - ens_std_rm60_l1",
    )
    _add_feature(
        features,
        formulas,
        "spread_shift_z_7v60",
        shift / (rs60 + 0.1),
        "spread_shift_7v60 / (ens_std_rs60_l1 + eps)",
        {"eps": 0.1},
    )
    if GEFS_SPREAD_ALIAS in df.columns:
        rm7_g = tfl.rolling_mean(
            df[GEFS_SPREAD_ALIAS],
            window=7,
            min_periods=_min_periods(7),
            lag=1,
            group_key=gk,
        )
        rm60_g = tfl.rolling_mean(
            df[GEFS_SPREAD_ALIAS],
            window=60,
            min_periods=_min_periods(60),
            lag=1,
            group_key=gk,
        )
        rs60_g = tfl.rolling_std(
            df[GEFS_SPREAD_ALIAS],
            window=60,
            min_periods=_min_periods(60),
            lag=1,
            group_key=gk,
        )
        shift_g = rm7_g - rm60_g
        _add_feature(
            features,
            formulas,
            "gefs_spread_shift_7v60",
            shift_g,
            "gefs_spread_rm7_l1 - gefs_spread_rm60_l1",
        )
        _add_feature(
            features,
            formulas,
            "gefs_spread_shift_z_7v60",
            shift_g / (rs60_g + 0.1),
            "gefs_spread_shift_7v60 / (gefs_spread_rs60_l1 + eps)",
            {"eps": 0.1},
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e19(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    drift_shift_cols = []
    for col in MODEL_COLS:
        drift = df[col] - df["ens_mean"]
        rm7 = tfl.rolling_mean(
            drift, window=7, min_periods=_min_periods(7), lag=1, group_key=gk
        )
        rm60 = tfl.rolling_mean(
            drift, window=60, min_periods=_min_periods(60), lag=1, group_key=gk
        )
        shift = rm7 - rm60
        name = f"drift_shift_{col}_7v60"
        _add_feature(
            features,
            formulas,
            name,
            shift,
            f"roll_mean({col}-ens_mean,7) - roll_mean({col}-ens_mean,60)",
        )
        drift_shift_cols.append(name)
    shift_vals = features[drift_shift_cols].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "drift_shift_absmax",
        np.nanmax(np.abs(shift_vals), axis=1),
        "max(|drift_shift_*|)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e20(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    shape_ratio = df["ens_range"] / (df["ens_iqr"] + 0.1)
    rm30 = tfl.rolling_mean(
        shape_ratio, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
    )
    rm7 = tfl.rolling_mean(
        shape_ratio, window=7, min_periods=_min_periods(7), lag=1, group_key=gk
    )
    rm60 = tfl.rolling_mean(
        shape_ratio, window=60, min_periods=_min_periods(60), lag=1, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        "shape_ratio_rm30_l1",
        rm30,
        "roll_mean(shape_ratio)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    _add_feature(
        features,
        formulas,
        "shape_ratio_shift_7v60",
        rm7 - rm60,
        "roll_mean(shape_ratio,7) - roll_mean(shape_ratio,60)",
    )
    outlier_rm30 = tfl.rolling_mean(
        df["ens_outlier_gap"],
        window=30,
        min_periods=_min_periods(30),
        lag=1,
        group_key=gk,
    )
    _add_feature(
        features,
        formulas,
        "outlier_gap_rm30_l1",
        outlier_rm30,
        "roll_mean(ens_outlier_gap)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e21(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    for window in (30, 60):
        min_p = _min_periods(window)
        mean_bias = tfl.rolling_mean(
            df["resid_ens_mean"],
            window=window,
            min_periods=min_p,
            lag=lag,
            group_key=gk,
        )
        median_bias = tfl.rolling_mean(
            df["resid_ens_median"],
            window=window,
            min_periods=min_p,
            lag=lag,
            group_key=gk,
        )
        _add_feature(
            features,
            formulas,
            f"bias_ensmean_rm{window}_l{lag}",
            mean_bias,
            "roll_mean(resid_ens_mean)",
            {"window": window, "lag": lag, "min_periods": min_p},
        )
        _add_feature(
            features,
            formulas,
            f"bias_ensmedian_rm{window}_l{lag}",
            median_bias,
            "roll_mean(resid_ens_median)",
            {"window": window, "lag": lag, "min_periods": min_p},
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e22(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    abs_mean = df["resid_ens_mean"].abs()
    abs_median = df["resid_ens_median"].abs()
    sq_mean = df["resid_ens_mean"] ** 2
    sq_median = df["resid_ens_median"] ** 2
    min_p = _min_periods(30)
    mae_mean = tfl.rolling_mean(
        abs_mean, window=30, min_periods=min_p, lag=lag, group_key=gk
    )
    mae_median = tfl.rolling_mean(
        abs_median, window=30, min_periods=min_p, lag=lag, group_key=gk
    )
    rmse_mean = np.sqrt(
        tfl.rolling_mean(sq_mean, window=30, min_periods=min_p, lag=lag, group_key=gk)
    )
    rmse_median = np.sqrt(
        tfl.rolling_mean(
            sq_median, window=30, min_periods=min_p, lag=lag, group_key=gk
        )
    )
    _add_feature(
        features,
        formulas,
        f"mae_ensmean_rm30_l{lag}",
        mae_mean,
        "roll_mean(|resid_ens_mean|)",
        {"window": 30, "lag": lag, "min_periods": min_p},
    )
    _add_feature(
        features,
        formulas,
        f"rmse_ensmean_rm30_l{lag}",
        rmse_mean,
        "sqrt(roll_mean(resid_ens_mean^2))",
        {"window": 30, "lag": lag, "min_periods": min_p},
    )
    _add_feature(
        features,
        formulas,
        f"mae_ensmedian_rm30_l{lag}",
        mae_median,
        "roll_mean(|resid_ens_median|)",
        {"window": 30, "lag": lag, "min_periods": min_p},
    )
    _add_feature(
        features,
        formulas,
        f"rmse_ensmedian_rm30_l{lag}",
        rmse_median,
        "sqrt(roll_mean(resid_ens_median^2))",
        {"window": 30, "lag": lag, "min_periods": min_p},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e23(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        bias = tfl.rolling_mean(
            resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        mae = tfl.rolling_mean(
            resid.abs(),
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        _add_feature(
            features,
            formulas,
            f"bias_{col}_rm60_l{lag}",
            bias,
            f"roll_mean(resid_{col})",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"mae_{col}_rm60_l{lag}",
            mae,
            f"roll_mean(|resid_{col}|)",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e24(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    mae_cols = []
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae = tfl.rolling_mean(
            resid.abs(),
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        name = f"mae_{col}_rm60_l{lag}"
        _add_feature(
            features,
            formulas,
            name,
            mae,
            f"roll_mean(|resid_{col}|)",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        mae_cols.append(name)
    mae_ens = tfl.rolling_mean(
        df["resid_ens_mean"].abs(),
        window=60,
        min_periods=_min_periods(60),
        lag=lag,
        group_key=gk,
    )
    _add_feature(
        features,
        formulas,
        f"mae_ensmean_rm60_l{lag}",
        mae_ens,
        "roll_mean(|resid_ens_mean|)",
        {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
    )
    for col, mae_name in zip(MODEL_COLS, mae_cols):
        rel = features[mae_name] - mae_ens
        _add_feature(
            features,
            formulas,
            f"rel_mae_{col}_vs_ens_rm60_l{lag}",
            rel,
            f"{mae_name} - mae_ensmean_rm60_l{lag}",
        )
    mae_frame = pd.DataFrame(
        {model: features[name] for model, name in zip(MODEL_COLS, mae_cols)},
        index=df.index,
    )
    best_id = _argmin_model(mae_frame, MODEL_COLS)
    for col in MODEL_COLS:
        _add_feature(
            features,
            formulas,
            f"best_is_{col}",
            (best_id == col).astype(int),
            f"1[best_model_id == {col}]",
        )
    best_forecast = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        mask = best_id == col
        best_forecast[mask.to_numpy()] = df.loc[mask, col].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "best_model_forecast_today",
        best_forecast,
        "forecast(best_model_id)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e25(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    min_p = _min_periods(60)
    r = df["resid_ens_mean"]
    q10 = tfl.rolling_quantile(
        r, window=60, min_periods=min_p, lag=lag, q=0.10, group_key=gk
    )
    q50 = tfl.rolling_quantile(
        r, window=60, min_periods=min_p, lag=lag, q=0.50, group_key=gk
    )
    q90 = tfl.rolling_quantile(
        r, window=60, min_periods=min_p, lag=lag, q=0.90, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        f"r_q10_rm60_l{lag}",
        q10,
        "rolling_quantile(resid_ens_mean, q=0.10)",
        {"window": 60, "lag": lag, "min_periods": min_p},
    )
    _add_feature(
        features,
        formulas,
        f"r_q50_rm60_l{lag}",
        q50,
        "rolling_quantile(resid_ens_mean, q=0.50)",
        {"window": 60, "lag": lag, "min_periods": min_p},
    )
    _add_feature(
        features,
        formulas,
        f"r_q90_rm60_l{lag}",
        q90,
        "rolling_quantile(resid_ens_mean, q=0.90)",
        {"window": 60, "lag": lag, "min_periods": min_p},
    )
    _add_feature(
        features,
        formulas,
        f"r_iqr_rm60_l{lag}",
        q90 - q10,
        f"r_q90_rm60_l{lag} - r_q10_rm60_l{lag}",
    )
    _add_feature(
        features,
        formulas,
        f"r_asym_rm60_l{lag}",
        np.abs(q10) - np.abs(q90),
        f"|r_q10_rm60_l{lag}| - |r_q90_rm60_l{lag}|",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e26(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thresholds, default = _station_quantile(ctx.train_df, "ens_mean", 0.50)
    thr = _map_station_threshold(df, thresholds, default)
    warm = (df["ens_mean"] >= thr).astype(int)
    train_fitted.append(
        {
            "name": "thr_temp",
            "fit_on": "train",
            "description": "median_train(ens_mean) per station",
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    resid = df["resid_ens_mean"]
    num_warm = tfl.rolling_sum(
        resid * warm, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    den_warm = tfl.rolling_sum(
        warm, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    num_cold = tfl.rolling_sum(
        resid * (1 - warm),
        window=60,
        min_periods=_min_periods(60),
        lag=lag,
        group_key=gk,
    )
    den_cold = tfl.rolling_sum(
        (1 - warm),
        window=60,
        min_periods=_min_periods(60),
        lag=lag,
        group_key=gk,
    )
    bias_warm = num_warm / den_warm
    bias_cold = num_cold / den_cold
    bias_uncond = tfl.rolling_mean(
        resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    selected = np.where(
        (warm == 1) & (den_warm >= 15),
        bias_warm,
        np.where((warm == 0) & (den_cold >= 15), bias_cold, bias_uncond),
    )
    _add_feature(
        features,
        formulas,
        f"bias_warm_rm60_l{lag}",
        bias_warm,
        "mean(resid_ens_mean | warm_regime)",
    )
    _add_feature(
        features,
        formulas,
        f"bias_cold_rm60_l{lag}",
        bias_cold,
        "mean(resid_ens_mean | cold_regime)",
    )
    _add_feature(
        features,
        formulas,
        f"bias_selected_rm60_l{lag}",
        selected,
        "bias_warm/bias_cold selected by current ens_mean",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e27(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thresholds, default = _station_quantile(ctx.train_df, "ens_std", 0.75)
    thr = _map_station_threshold(df, thresholds, default)
    hi = (df["ens_std"] > thr).astype(int)
    train_fitted.append(
        {
            "name": "thr_spread",
            "fit_on": "train",
            "description": "q75_train(ens_std) per station",
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    resid = df["resid_ens_mean"]
    num_hi = tfl.rolling_sum(
        resid * hi, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    den_hi = tfl.rolling_sum(
        hi, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    num_lo = tfl.rolling_sum(
        resid * (1 - hi),
        window=60,
        min_periods=_min_periods(60),
        lag=lag,
        group_key=gk,
    )
    den_lo = tfl.rolling_sum(
        (1 - hi),
        window=60,
        min_periods=_min_periods(60),
        lag=lag,
        group_key=gk,
    )
    bias_hi = num_hi / den_hi
    bias_lo = num_lo / den_lo
    bias_uncond = tfl.rolling_mean(
        resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    selected = np.where(
        (hi == 1) & (den_hi >= 15),
        bias_hi,
        np.where((hi == 0) & (den_lo >= 15), bias_lo, bias_uncond),
    )
    _add_feature(
        features,
        formulas,
        f"bias_hispread_rm60_l{lag}",
        bias_hi,
        "mean(resid_ens_mean | hi_spread)",
    )
    _add_feature(
        features,
        formulas,
        f"bias_lospread_rm60_l{lag}",
        bias_lo,
        "mean(resid_ens_mean | lo_spread)",
    )
    _add_feature(
        features,
        formulas,
        "bias_by_spread_selected",
        selected,
        "bias_hispread/bias_lospread selected by current ens_std",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e28(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    ewm14 = tfl.ewm_mean(
        df["resid_ens_mean"], halflife=14, min_periods=10, lag=lag, group_key=gk
    )
    ewm30 = tfl.ewm_mean(
        df["resid_ens_mean"], halflife=30, min_periods=10, lag=lag, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        f"r_ewm_mean_hl14_l{lag}",
        ewm14,
        "ewm_mean(resid_ens_mean)",
        {"halflife": 14, "lag": lag, "min_periods": 10},
    )
    _add_feature(
        features,
        formulas,
        f"r_ewm_mean_hl30_l{lag}",
        ewm30,
        "ewm_mean(resid_ens_mean)",
        {"halflife": 30, "lag": lag, "min_periods": 10},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_biascorr_hl14",
        df["ens_mean"] + ewm14,
        "ens_mean + r_ewm_mean_hl14",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e29(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    min_p = _min_periods(60)
    mean_x = tfl.rolling_mean(
        df["ens_mean"], window=60, min_periods=min_p, lag=lag, group_key=gk
    )
    mean_y = tfl.rolling_mean(
        df["actual_tmax_f"], window=60, min_periods=min_p, lag=lag, group_key=gk
    )
    mean_xy = tfl.rolling_mean(
        df["ens_mean"] * df["actual_tmax_f"],
        window=60,
        min_periods=min_p,
        lag=lag,
        group_key=gk,
    )
    mean_x2 = tfl.rolling_mean(
        df["ens_mean"] ** 2, window=60, min_periods=min_p, lag=lag, group_key=gk
    )
    cov = mean_xy - mean_x * mean_y
    var = mean_x2 - mean_x**2
    b = cov / (var + 1e-6)
    a = mean_y - b * mean_x
    _add_feature(
        features,
        formulas,
        f"a_rm60_l{lag}",
        a,
        "mean(y) - b * mean(ens_mean)",
    )
    _add_feature(
        features,
        formulas,
        f"b_rm60_l{lag}",
        b,
        "cov(ens_mean,y)/var(ens_mean)",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_lin_calib_rm60",
        a + b * df["ens_mean"],
        "a_rm60 + b_rm60 * ens_mean",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e30(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    mae_cols = []
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae = tfl.rolling_mean(
            resid.abs(),
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        name = f"mae_{col}_rm60_l{lag}"
        _add_feature(
            features,
            formulas,
            name,
            mae,
            f"roll_mean(|resid_{col}|)",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        mae_cols.append(name)
    mae_frame = pd.DataFrame(
        {model: features[name] for model, name in zip(MODEL_COLS, mae_cols)},
        index=df.index,
    )
    winner_id = _argmin_model(mae_frame, MODEL_COLS)
    winner_forecast = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        mask = winner_id == col
        winner_forecast[mask.to_numpy()] = df.loc[mask, col].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "winner_forecast_today",
        winner_forecast,
        "forecast(winner_id)",
    )
    _add_feature(
        features,
        formulas,
        "winner_switch_count_30_l1",
        tfl.switch_count(
            winner_id, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        ),
        "switch_count(winner_id)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    _add_feature(
        features,
        formulas,
        "winner_streak_len_l1",
        tfl.streak_length(winner_id, lag=1, cap=60, group_key=gk),
        "streak_length(winner_id)",
        {"lag": 1, "cap": 60},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e31(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    resid_ens = df["resid_ens_mean"]
    corr_cols = []
    min_p = _min_periods(60)
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mean_r = tfl.rolling_mean(
            resid, window=60, min_periods=min_p, lag=lag, group_key=gk
        )
        mean_e = tfl.rolling_mean(
            resid_ens, window=60, min_periods=min_p, lag=lag, group_key=gk
        )
        mean_re = tfl.rolling_mean(
            resid * resid_ens, window=60, min_periods=min_p, lag=lag, group_key=gk
        )
        mean_r2 = tfl.rolling_mean(
            resid**2, window=60, min_periods=min_p, lag=lag, group_key=gk
        )
        mean_e2 = tfl.rolling_mean(
            resid_ens**2, window=60, min_periods=min_p, lag=lag, group_key=gk
        )
        cov = mean_re - mean_r * mean_e
        var_r = mean_r2 - mean_r**2
        var_e = mean_e2 - mean_e**2
        corr = cov / np.sqrt(np.maximum(var_r * var_e, 0.0) + 1e-6)
        name = f"corr_resid_{col}_ens_rm60_l{lag}"
        _add_feature(
            features,
            formulas,
            name,
            corr,
            "corr(resid_model, resid_ens_mean)",
            {"window": 60, "lag": lag, "min_periods": min_p},
        )
        corr_cols.append(name)
    corr_vals = features[corr_cols].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        f"mean_corr_models_rm60_l{lag}",
        np.nanmean(corr_vals, axis=1),
        "mean(corr_resid_model_ens)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e32(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    min_p = _min_periods(60)
    u = df["resid_ens_mean"].abs()
    s = df["ens_std"]
    mean_s = tfl.rolling_mean(s, window=60, min_periods=min_p, lag=lag, group_key=gk)
    mean_u = tfl.rolling_mean(u, window=60, min_periods=min_p, lag=lag, group_key=gk)
    mean_su = tfl.rolling_mean(
        s * u, window=60, min_periods=min_p, lag=lag, group_key=gk
    )
    mean_s2 = tfl.rolling_mean(
        s**2, window=60, min_periods=min_p, lag=lag, group_key=gk
    )
    cov = mean_su - mean_s * mean_u
    var = mean_s2 - mean_s**2
    b = cov / (var + 1e-6)
    a = mean_u - b * mean_s
    _add_feature(
        features,
        formulas,
        f"b_u_s_rm60_l{lag}",
        b,
        "|resid| ~ a + b * ens_std",
    )
    _add_feature(
        features,
        formulas,
        f"a_u_s_rm60_l{lag}",
        a,
        "|resid| ~ a + b * ens_std",
    )
    _add_feature(
        features,
        formulas,
        "pred_abs_err_from_spread",
        a + b * s,
        "a_u_s_rm60 + b_u_s_rm60 * ens_std",
    )
    if GEFS_SPREAD_ALIAS in df.columns:
        gs = df[GEFS_SPREAD_ALIAS]
        mean_g = tfl.rolling_mean(
            gs, window=60, min_periods=min_p, lag=lag, group_key=gk
        )
        mean_gu = tfl.rolling_mean(
            gs * u, window=60, min_periods=min_p, lag=lag, group_key=gk
        )
        mean_g2 = tfl.rolling_mean(
            gs**2, window=60, min_periods=min_p, lag=lag, group_key=gk
        )
        cov_g = mean_gu - mean_g * mean_u
        var_g = mean_g2 - mean_g**2
        b_g = cov_g / (var_g + 1e-6)
        a_g = mean_u - b_g * mean_g
        _add_feature(
            features,
            formulas,
            f"b_u_gefs_rm60_l{lag}",
            b_g,
            "|resid| ~ a + b * gefs_spread",
        )
        _add_feature(
            features,
            formulas,
            "pred_abs_err_from_gefs_spread",
            a_g + b_g * gs,
            "a_u_gefs_rm60 + b_u_gefs_rm60 * gefs_spread",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e33(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, scaler = _standardize_features(ctx.train_df, df, feature_cols)
    kmeans = KMeans(n_clusters=6, random_state=ctx.seed, n_init=10)
    kmeans.fit(scaler.transform(ctx.train_df[feature_cols].to_numpy(dtype=float)))
    clusters = kmeans.predict(scaled)
    centers = kmeans.cluster_centers_
    dist = np.linalg.norm(scaled - centers[clusters], axis=1)
    _add_feature(
        features,
        formulas,
        "kmeans_cluster_id",
        clusters,
        "kmeans_cluster_id",
        {"k": 6},
    )
    _add_feature(
        features,
        formulas,
        "kmeans_dist_to_centroid",
        dist,
        "euclidean_distance_to_centroid",
    )
    freq = np.full(len(df), np.nan, dtype=float)
    for cid in range(6):
        indicator = pd.Series((clusters == cid).astype(int), index=df.index)
        freq_c = tfl.rolling_event_mean(
            indicator,
            window=30,
            min_periods=_min_periods(30),
            lag=1,
            group_key=ctx.group_key,
        )
        freq[clusters == cid] = freq_c[clusters == cid]
    _add_feature(
        features,
        formulas,
        "kmeans_cluster_freq_30_l1",
        freq,
        "rolling_mean(cluster_id == current)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    train_fitted.append(
        {
            "name": "kmeans_clusters",
            "fit_on": "train",
            "description": "StandardScaler + KMeans(K=6) on forecast vector",
            "features": feature_cols,
            "scaler": scaler_meta,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e34(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    disagree = df[MODEL_COLS].sub(df["ens_mean"], axis=0)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(disagree.loc[ctx.train_df.index].to_numpy(dtype=float))
    scores = pca.transform(disagree.to_numpy(dtype=float))
    pc1 = scores[:, 0]
    pc2 = scores[:, 1]
    _add_feature(features, formulas, "pc1_score", pc1, "PCA1(disagreement)")
    _add_feature(features, formulas, "pc2_score", pc2, "PCA2(disagreement)")
    pc1_rm30 = tfl.rolling_mean(
        pd.Series(pc1, index=df.index),
        window=30,
        min_periods=_min_periods(30),
        lag=1,
        group_key=gk,
    )
    pc1_rs30 = tfl.rolling_std(
        pd.Series(pc1, index=df.index),
        window=30,
        min_periods=_min_periods(30),
        lag=1,
        group_key=gk,
    )
    pc2_rm30 = tfl.rolling_mean(
        pd.Series(pc2, index=df.index),
        window=30,
        min_periods=_min_periods(30),
        lag=1,
        group_key=gk,
    )
    pc2_rs30 = tfl.rolling_std(
        pd.Series(pc2, index=df.index),
        window=30,
        min_periods=_min_periods(30),
        lag=1,
        group_key=gk,
    )
    _add_feature(
        features,
        formulas,
        "pc1_rm30_l1",
        pc1_rm30,
        "roll_mean(pc1)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    _add_feature(
        features,
        formulas,
        "pc1_rs30_l1",
        pc1_rs30,
        "roll_std(pc1)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    _add_feature(
        features,
        formulas,
        "pc2_rm30_l1",
        pc2_rm30,
        "roll_mean(pc2)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    _add_feature(
        features,
        formulas,
        "pc2_rs30_l1",
        pc2_rs30,
        "roll_std(pc2)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    train_fitted.append(
        {
            "name": "pca_disagreement",
            "fit_on": "train",
            "description": "PCA on disagreement vector",
            "components": pca.components_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e35(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    train_abs_err = ctx.train_df["resid_ens_mean"].abs()
    thr = float(train_abs_err.quantile(0.90))
    hard_train = (train_abs_err > thr).astype(int)
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy", "month"]
    X_train = ctx.train_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y_train = hard_train.to_numpy(dtype=int)
    if len(np.unique(y_train)) < 2:
        probs = np.full(len(df), float(y_train[0]), dtype=float)
        train_fitted.append(
            {
                "name": "hard_day_classifier",
                "fit_on": "train",
                "description": "single-class fallback",
                "threshold": thr,
                "features": feature_cols,
                "probability": float(y_train[0]),
            }
        )
    else:
        clf = LogisticRegression(
            solver="liblinear", random_state=ctx.seed, max_iter=200
        )
        clf.fit(X_train_scaled, y_train)
        X_all = scaler.transform(df[feature_cols].to_numpy(dtype=float))
        probs = clf.predict_proba(X_all)[:, 1]
        train_fitted.append(
            {
                "name": "hard_day_classifier",
                "fit_on": "train",
                "description": "logistic regression on forecast features",
                "threshold": thr,
                "features": feature_cols,
                "coef": clf.coef_.tolist(),
                "intercept": clf.intercept_.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            }
        )
    _add_feature(
        features,
        formulas,
        "p_hard_day",
        probs,
        "P(|resid_ens_mean| > q90_train)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e36(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thresholds, default = _station_quantile(ctx.train_df, "ens_std", 0.90)
    thr = _map_station_threshold(df, thresholds, default)
    hi = (df["ens_std"] > thr).astype(int)
    prev = hi.groupby(gk).shift(1)
    trans_11 = ((prev == 1) & (hi == 1)).astype(int)
    trans_10 = ((prev == 1) & (hi == 0)).astype(int)
    trans_01 = ((prev == 0) & (hi == 1)).astype(int)
    trans_00 = ((prev == 0) & (hi == 0)).astype(int)
    min_p = _min_periods(60)
    n11 = tfl.rolling_sum(trans_11, window=60, min_periods=min_p, lag=1, group_key=gk)
    n10 = tfl.rolling_sum(trans_10, window=60, min_periods=min_p, lag=1, group_key=gk)
    n01 = tfl.rolling_sum(trans_01, window=60, min_periods=min_p, lag=1, group_key=gk)
    n00 = tfl.rolling_sum(trans_00, window=60, min_periods=min_p, lag=1, group_key=gk)
    p11 = n11 / (n10 + n11 + 1e-6)
    p01 = n01 / (n00 + n01 + 1e-6)
    _add_feature(
        features,
        formulas,
        "p11_rm60_l1",
        p11,
        "P(hi->hi) over 60d",
    )
    _add_feature(
        features,
        formulas,
        "p01_rm60_l1",
        p01,
        "P(lo->hi) over 60d",
    )
    _add_feature(
        features,
        formulas,
        "regime_stickiness",
        p11 - p01,
        "p11_rm60_l1 - p01_rm60_l1",
    )
    train_fitted.append(
        {
            "name": "thr_spread_hi",
            "fit_on": "train",
            "description": "q90_train(ens_std) per station",
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e37(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    top_id = tfl.argmax_with_tie_break(df, MODEL_COLS)
    bot_id = tfl.argmin_with_tie_break(df, MODEL_COLS)
    for model in MODEL_COLS:
        top_ind = (top_id == model).astype(int)
        bot_ind = (bot_id == model).astype(int)
        _add_feature(
            features,
            formulas,
            f"top_freq_{model}_rm30_l1",
            tfl.rolling_event_mean(
                top_ind, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
            ),
            "roll_mean(is_top)",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
        _add_feature(
            features,
            formulas,
            f"bot_freq_{model}_rm30_l1",
            tfl.rolling_event_mean(
                bot_ind, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
            ),
            "roll_mean(is_bottom)",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e38(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    outmag_cols = []
    for col in MODEL_COLS:
        outmag = (df[col] - df["ens_median"]).abs()
        rm30 = tfl.rolling_mean(
            outmag, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        )
        q90 = tfl.rolling_quantile(
            outmag, window=60, min_periods=_min_periods(60), lag=1, q=0.90, group_key=gk
        )
        name_rm = f"outmag_{col}_rm30_l1"
        name_q = f"outmag_{col}_q90_rm60_l1"
        _add_feature(
            features,
            formulas,
            name_rm,
            rm30,
            "roll_mean(|model - ens_median|)",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
        _add_feature(
            features,
            formulas,
            name_q,
            q90,
            "rolling_quantile(|model - ens_median|, q=0.90)",
            {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
        )
        outmag_cols.append(name_rm)
    outmag_vals = features[outmag_cols].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "outmag_max_rm30_l1",
        np.nanmax(outmag_vals, axis=1),
        "max(outmag_*_rm30_l1)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e39(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    dom_id = _dominant_outlier_id(df, MODEL_COLS)
    for model in MODEL_COLS:
        indicator = (dom_id == model).astype(int)
        _add_feature(
            features,
            formulas,
            f"dom_outlier_is_{model}_freq_30_l1",
            tfl.rolling_event_mean(
                indicator, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
            ),
            "roll_mean(dom_outlier_id == model)",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e40(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    dom_id = _dominant_outlier_id(df, MODEL_COLS)
    _add_feature(
        features,
        formulas,
        "outlier_streak_len_l1",
        tfl.streak_length(dom_id, lag=1, cap=60, group_key=gk),
        "streak_length(dom_outlier_id)",
        {"lag": 1, "cap": 60},
    )
    _add_feature(
        features,
        formulas,
        "outlier_switch_count_30_l1",
        tfl.switch_count(
            dom_id, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        ),
        "switch_count(dom_outlier_id)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e41(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    neighbors, distances = _knn_neighbors(
        df,
        scaled,
        group_key=ctx.group_key,
        truth_lag=ctx.truth_lag,
        lookback_days=365,
        k=10,
    )
    y_vals = df["actual_tmax_f"].to_numpy(dtype=float)
    mean_y, std_y, mean_dist = _knn_mean_std(neighbors, distances, y_vals)
    _add_feature(
        features,
        formulas,
        f"knn_y_mean_k10_l{ctx.truth_lag}",
        mean_y,
        "KNN mean(y)",
        {"k": 10, "lookback_days": 365, "truth_lag": ctx.truth_lag},
    )
    _add_feature(
        features,
        formulas,
        f"knn_y_std_k10_l{ctx.truth_lag}",
        std_y,
        "KNN std(y)",
        {"k": 10, "lookback_days": 365, "truth_lag": ctx.truth_lag},
    )
    _add_feature(
        features,
        formulas,
        f"knn_dist_mean_k10_l{ctx.truth_lag}",
        mean_dist,
        "KNN mean distance",
        {"k": 10, "lookback_days": 365, "truth_lag": ctx.truth_lag},
    )
    train_fitted.append(
        {
            "name": "knn_analog_y",
            "fit_on": "train",
            "description": "KNN analogs on standardized forecast vector",
            "features": feature_cols,
            "scaler": scaler_meta,
            "k": 10,
            "lookback_days": 365,
            "truth_lag": ctx.truth_lag,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e42(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    neighbors, distances = _knn_neighbors(
        df,
        scaled,
        group_key=ctx.group_key,
        truth_lag=ctx.truth_lag,
        lookback_days=365,
        k=10,
    )
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    mean_resid, _, _ = _knn_mean_std(neighbors, distances, resid)
    quant = _knn_quantiles(neighbors, resid, [0.10, 0.90])
    _add_feature(
        features,
        formulas,
        f"knn_resid_mean_k10_l{ctx.truth_lag}",
        mean_resid,
        "KNN mean(resid)",
        {"k": 10, "lookback_days": 365, "truth_lag": ctx.truth_lag},
    )
    _add_feature(
        features,
        formulas,
        f"knn_resid_q10_k10_l{ctx.truth_lag}",
        quant[0.10],
        "KNN q10(resid)",
        {"k": 10, "lookback_days": 365, "truth_lag": ctx.truth_lag},
    )
    _add_feature(
        features,
        formulas,
        f"knn_resid_q90_k10_l{ctx.truth_lag}",
        quant[0.90],
        "KNN q90(resid)",
        {"k": 10, "lookback_days": 365, "truth_lag": ctx.truth_lag},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_knn_corr",
        df["ens_mean"] + mean_resid,
        "ens_mean + knn_resid_mean",
    )
    train_fitted.append(
        {
            "name": "knn_residual",
            "fit_on": "train",
            "description": "KNN residual analogs on standardized forecast vector",
            "features": feature_cols,
            "scaler": scaler_meta,
            "k": 10,
            "lookback_days": 365,
            "truth_lag": ctx.truth_lag,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e43(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    train_abs_err = ctx.train_df["resid_ens_mean"].abs()
    thr_err = float(train_abs_err.quantile(0.90))
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    neighbors, _ = _knn_neighbors(
        df,
        scaled,
        group_key=ctx.group_key,
        truth_lag=ctx.truth_lag,
        lookback_days=730,
        k=20,
    )
    hard_flags = (df["resid_ens_mean"].abs() > thr_err).to_numpy(dtype=int)
    prob = _knn_prob(neighbors, hard_flags)
    _add_feature(
        features,
        formulas,
        f"knn_hard_prob_k20_l{ctx.truth_lag}",
        prob,
        "KNN mean(hard_day)",
        {"k": 20, "lookback_days": 730, "truth_lag": ctx.truth_lag},
    )
    ens_std = df["ens_std"].to_numpy(dtype=float)
    std_mean = float(ctx.train_df["ens_std"].mean())
    std_std = float(ctx.train_df["ens_std"].std(ddof=0)) if len(ctx.train_df) else 1.0
    std_std = std_std if std_std > 0 else 1.0
    knn_minus_spread = prob - (ens_std - std_mean) / std_std
    _add_feature(
        features,
        formulas,
        "knn_hard_prob_minus_spread",
        knn_minus_spread,
        "knn_hard_prob - z(ens_std)",
    )
    train_fitted.append(
        {
            "name": "knn_hard_prob",
            "fit_on": "train",
            "description": "KNN hard-day probability",
            "features": feature_cols,
            "scaler": scaler_meta,
            "k": 20,
            "lookback_days": 730,
            "truth_lag": ctx.truth_lag,
            "thr_err": thr_err,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e44(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    neighbors, distances = _knn_neighbors(
        df,
        scaled,
        group_key=ctx.group_key,
        truth_lag=ctx.truth_lag,
        lookback_days=365,
        k=20,
    )
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    weighted, nearest, eff_n = _knn_kernel_resid(neighbors, distances, resid)
    _add_feature(
        features,
        formulas,
        f"kern_resid_mean_k20_l{ctx.truth_lag}",
        weighted,
        "kernel_weighted_mean(resid)",
        {"k": 20, "lookback_days": 365, "truth_lag": ctx.truth_lag},
    )
    _add_feature(
        features,
        formulas,
        f"nearest_dist_l{ctx.truth_lag}",
        nearest,
        "nearest_distance",
    )
    _add_feature(
        features,
        formulas,
        f"effective_n_l{ctx.truth_lag}",
        eff_n,
        "effective_sample_size",
    )
    train_fitted.append(
        {
            "name": "knn_kernel_resid",
            "fit_on": "train",
            "description": "Kernel-weighted residual analogs",
            "features": feature_cols,
            "scaler": scaler_meta,
            "k": 20,
            "lookback_days": 365,
            "truth_lag": ctx.truth_lag,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e45(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thresholds, default = _station_quantile(ctx.train_df, "ens_std", 0.90)
    thr = _map_station_threshold(df, thresholds, default)
    ext = (df["ens_std"] > thr).astype(int)
    _add_feature(
        features,
        formulas,
        "days_since_ext_spread_l1",
        tfl.days_since_event(ext, lag=1, cap=365, group_key=gk),
        "days_since(ext_spread)",
        {"lag": 1, "cap": 365},
    )
    _add_feature(
        features,
        formulas,
        "ext_spread_count_30_l1",
        tfl.rolling_event_count(
            ext, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        ),
        "roll_sum(ext_spread)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    _add_feature(
        features,
        formulas,
        "ext_spread_count_60_l1",
        tfl.rolling_event_count(
            ext, window=60, min_periods=_min_periods(60), lag=1, group_key=gk
        ),
        "roll_sum(ext_spread)",
        {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
    )
    train_fitted.append(
        {
            "name": "thr_spread_ext",
            "fit_on": "train",
            "description": "q90_train(ens_std) per station",
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e46(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thr_hi_map, thr_hi_default = _station_quantile(ctx.train_df, "ens_mean", 0.95)
    thr_lo_map, thr_lo_default = _station_quantile(ctx.train_df, "ens_mean", 0.05)
    thr_hi = _map_station_threshold(df, thr_hi_map, thr_hi_default)
    thr_lo = _map_station_threshold(df, thr_lo_map, thr_lo_default)
    hi = (df["ens_mean"] > thr_hi).astype(int)
    lo = (df["ens_mean"] < thr_lo).astype(int)
    _add_feature(
        features,
        formulas,
        "days_since_hi_fcst_l1",
        tfl.days_since_event(hi, lag=1, cap=365, group_key=gk),
        "days_since(hi_fcst)",
        {"lag": 1, "cap": 365},
    )
    _add_feature(
        features,
        formulas,
        "days_since_lo_fcst_l1",
        tfl.days_since_event(lo, lag=1, cap=365, group_key=gk),
        "days_since(lo_fcst)",
        {"lag": 1, "cap": 365},
    )
    _add_feature(
        features,
        formulas,
        "hi_fcst_count_60_l1",
        tfl.rolling_event_count(
            hi, window=60, min_periods=_min_periods(60), lag=1, group_key=gk
        ),
        "roll_sum(hi_fcst)",
        {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
    )
    _add_feature(
        features,
        formulas,
        "lo_fcst_count_60_l1",
        tfl.rolling_event_count(
            lo, window=60, min_periods=_min_periods(60), lag=1, group_key=gk
        ),
        "roll_sum(lo_fcst)",
        {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
    )
    train_fitted.append(
        {
            "name": "thr_hi_lo_fcst",
            "fit_on": "train",
            "description": "q95/q05_train(ens_mean) per station",
            "thr_hi": thr_hi_map,
            "thr_lo": thr_lo_map,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e47(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thresholds, default = _station_quantile(ctx.train_df, "ens_outlier_gap", 0.90)
    thr = _map_station_threshold(df, thresholds, default)
    ext = (df["ens_outlier_gap"] > thr).astype(int)
    _add_feature(
        features,
        formulas,
        "days_since_gap_ext_l1",
        tfl.days_since_event(ext, lag=1, cap=365, group_key=gk),
        "days_since(outlier_gap_ext)",
        {"lag": 1, "cap": 365},
    )
    _add_feature(
        features,
        formulas,
        "gap_ext_count_30_l1",
        tfl.rolling_event_count(
            ext, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        ),
        "roll_sum(outlier_gap_ext)",
        {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
    )
    _add_feature(
        features,
        formulas,
        "gap_ext_count_60_l1",
        tfl.rolling_event_count(
            ext, window=60, min_periods=_min_periods(60), lag=1, group_key=gk
        ),
        "roll_sum(outlier_gap_ext)",
        {"window": 60, "lag": 1, "min_periods": _min_periods(60)},
    )
    train_fitted.append(
        {
            "name": "thr_outlier_gap",
            "fit_on": "train",
            "description": "q90_train(outlier_gap) per station",
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e48(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    train_abs = ctx.train_df["resid_ens_mean"].abs()
    thresholds = train_abs.groupby(ctx.train_df["station_id"]).quantile(0.90).to_dict()
    default = float(train_abs.quantile(0.90))
    thr = _map_station_threshold(df, thresholds, default)
    ext = (df["resid_ens_mean"].abs() > thr).astype(int)
    _add_feature(
        features,
        formulas,
        f"days_since_err_ext_l{lag}",
        tfl.days_since_event(ext, lag=lag, cap=365, group_key=gk),
        "days_since(err_ext)",
        {"lag": lag, "cap": 365},
    )
    _add_feature(
        features,
        formulas,
        f"err_ext_count_60_l{lag}",
        tfl.rolling_event_count(
            ext, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        ),
        "roll_sum(err_ext)",
        {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
    )
    train_fitted.append(
        {
            "name": "thr_err_ext",
            "fit_on": "train",
            "description": "q90_train(|resid_ens_mean|) per station",
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e49(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thr_cold_map, thr_cold_default = _station_quantile(
        ctx.train_df, "resid_ens_mean", 0.05
    )
    thr_warm_map, thr_warm_default = _station_quantile(
        ctx.train_df, "resid_ens_mean", 0.95
    )
    thr_cold = _map_station_threshold(df, thr_cold_map, thr_cold_default)
    thr_warm = _map_station_threshold(df, thr_warm_map, thr_warm_default)
    resid = df["resid_ens_mean"]
    cold = (resid < thr_cold).astype(int)
    warm = (resid > thr_warm).astype(int)
    _add_feature(
        features,
        formulas,
        f"days_since_cold_bust_l{lag}",
        tfl.days_since_event(cold, lag=lag, cap=365, group_key=gk),
        "days_since(cold_bust)",
        {"lag": lag, "cap": 365},
    )
    _add_feature(
        features,
        formulas,
        f"days_since_warm_bust_l{lag}",
        tfl.days_since_event(warm, lag=lag, cap=365, group_key=gk),
        "days_since(warm_bust)",
        {"lag": lag, "cap": 365},
    )
    cold_count = tfl.rolling_event_count(
        cold, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    warm_count = tfl.rolling_event_count(
        warm, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        f"cold_bust_count_60_l{lag}",
        cold_count,
        "roll_sum(cold_bust)",
        {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
    )
    _add_feature(
        features,
        formulas,
        f"warm_bust_count_60_l{lag}",
        warm_count,
        "roll_sum(warm_bust)",
        {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
    )
    _add_feature(
        features,
        formulas,
        f"bust_balance_60_l{lag}",
        cold_count - warm_count,
        "cold_bust_count_60_l - warm_bust_count_60_l",
    )
    train_fitted.append(
        {
            "name": "thr_bust",
            "fit_on": "train",
            "description": "q05/q95_train(resid_ens_mean) per station",
            "thr_cold": thr_cold_map,
            "thr_warm": thr_warm_map,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e50(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    resid = df["resid_ens_mean"]
    sign = np.sign(resid).fillna(0)
    _add_feature(
        features,
        formulas,
        f"resid_sign_streak_l{lag}",
        tfl.streak_length(sign, lag=lag, cap=60, group_key=gk),
        "streak_length(sign(resid))",
        {"lag": lag, "cap": 60},
    )
    _add_feature(
        features,
        formulas,
        f"resid_sign_changes_30_l{lag}",
        tfl.switch_count(
            sign, window=30, min_periods=_min_periods(30), lag=lag, group_key=gk
        ),
        "switch_count(sign(resid))",
        {"window": 30, "lag": lag, "min_periods": _min_periods(30)},
    )
    _add_feature(
        features,
        formulas,
        f"resid_mean_15_l{lag}",
        tfl.rolling_mean(
            resid, window=15, min_periods=_min_periods(15), lag=lag, group_key=gk
        ),
        "roll_mean(resid_ens_mean)",
        {"window": 15, "lag": lag, "min_periods": _min_periods(15)},
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e51(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        bias = tfl.rolling_mean(
            resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            f"bias_{col}_rm60_l{lag}",
            bias,
            f"roll_mean(resid_{col})",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"abs_bias_{col}_rm60_l{lag}",
            bias.abs(),
            f"|bias_{col}_rm60_l{lag}|",
        )
    bias_ens = tfl.rolling_mean(
        df["resid_ens_mean"],
        window=60,
        min_periods=_min_periods(60),
        lag=lag,
        group_key=gk,
    )
    _add_feature(
        features,
        formulas,
        f"bias_ensmean_rm60_l{lag}",
        bias_ens,
        "roll_mean(resid_ens_mean)",
        {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
    )
    _add_feature(
        features,
        formulas,
        f"abs_bias_ensmean_rm60_l{lag}",
        bias_ens.abs(),
        f"|bias_ensmean_rm60_l{lag}|",
    )
    for col in MODEL_COLS:
        rel = features[f"abs_bias_{col}_rm60_l{lag}"] - features[
            f"abs_bias_ensmean_rm60_l{lag}"
        ]
        _add_feature(
            features,
            formulas,
            f"rel_abs_bias_{col}_vs_ens_rm60_l{lag}",
            rel,
            f"abs_bias_{col}_rm60_l - abs_bias_ensmean_rm60_l",
        )
    abs_bias_frame = pd.DataFrame(
        {col: features[f"abs_bias_{col}_rm60_l{lag}"] for col in MODEL_COLS},
        index=df.index,
    )
    best_id = _argmin_model(abs_bias_frame, MODEL_COLS)
    for col in MODEL_COLS:
        _add_feature(
            features,
            formulas,
            f"best_bias_is_{col}",
            (best_id == col).astype(int),
            f"1[best_bias_model_id == {col}]",
        )
    best_forecast = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        mask = best_id == col
        best_forecast[mask.to_numpy()] = df.loc[mask, col].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "best_bias_model_forecast_today",
        best_forecast,
        "forecast(best_bias_model_id)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e52(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    rmse_vals = {}
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        rmse = _rolling_rmse(resid, window=60, lag=lag, group_key=gk)
        rmse_vals[col] = rmse
        _add_feature(
            features,
            formulas,
            f"rmse_{col}_rm60_l{lag}",
            rmse,
            f"rmse(resid_{col})",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
    rmse_ens = _rolling_rmse(df["resid_ens_mean"], window=60, lag=lag, group_key=gk)
    _add_feature(
        features,
        formulas,
        f"rmse_ensmean_rm60_l{lag}",
        rmse_ens,
        "rmse(resid_ens_mean)",
        {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
    )
    for col in MODEL_COLS:
        rel = rmse_vals[col] - rmse_ens
        _add_feature(
            features,
            formulas,
            f"rel_rmse_{col}_vs_ens_rm60_l{lag}",
            rel,
            f"rmse_{col}_rm60_l - rmse_ensmean_rm60_l",
        )
    rmse_frame = pd.DataFrame(rmse_vals, index=df.index)
    best_id = _argmin_model(rmse_frame, MODEL_COLS)
    for col in MODEL_COLS:
        _add_feature(
            features,
            formulas,
            f"best_rmse_is_{col}",
            (best_id == col).astype(int),
            f"1[best_rmse_model_id == {col}]",
        )
    best_forecast = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        mask = best_id == col
        best_forecast[mask.to_numpy()] = df.loc[mask, col].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "best_rmse_model_forecast_today",
        best_forecast,
        "forecast(best_rmse_model_id)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e53(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    lambda_bias = 0.5
    mu_rmse = 0.2
    score_frame = pd.DataFrame(index=df.index)
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
        bias = tfl.rolling_mean(
            resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        rmse = _rolling_rmse(resid, window=60, lag=lag, group_key=gk)
        score = mae + lambda_bias * bias.abs() + mu_rmse * rmse
        score_frame[col] = score
        _add_feature(
            features,
            formulas,
            f"mae_{col}_rm60_l{lag}",
            mae,
            f"mae(resid_{col})",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"bias_{col}_rm60_l{lag}",
            bias,
            f"bias(resid_{col})",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"rmse_{col}_rm60_l{lag}",
            rmse,
            f"rmse(resid_{col})",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"score_{col}_rm60_l{lag}",
            score,
            f"mae + {lambda_bias}*|bias| + {mu_rmse}*rmse",
        )
    best_id = _argmin_model(score_frame, MODEL_COLS)
    for col in MODEL_COLS:
        _add_feature(
            features,
            formulas,
            f"best_score_is_{col}",
            (best_id == col).astype(int),
            f"1[best_score_model_id == {col}]",
        )
    best_forecast = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        mask = best_id == col
        best_forecast[mask.to_numpy()] = df.loc[mask, col].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "best_score_model_forecast_today",
        best_forecast,
        "forecast(best_score_model_id)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e54(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    worst_shift = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae15 = _rolling_abs_mean(resid, window=15, lag=lag, group_key=gk)
        mae60 = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
        std60 = tfl.rolling_std(
            resid.abs(),
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        shift = mae15 - mae60
        z = _safe_divide(shift.to_numpy(dtype=float), std60.to_numpy(dtype=float) + 0.1)
        _add_feature(
            features,
            formulas,
            f"mae_{col}_rm15_l{lag}",
            mae15,
            "roll_mean(|resid|)",
            {"window": 15, "lag": lag, "min_periods": _min_periods(15)},
        )
        _add_feature(
            features,
            formulas,
            f"mae_{col}_rm60_l{lag}",
            mae60,
            "roll_mean(|resid|)",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"skill_shift_mae_{col}_15v60_l{lag}",
            shift,
            "mae_rm15 - mae_rm60",
        )
        _add_feature(
            features,
            formulas,
            f"skill_shift_z_{col}_15v60_l{lag}",
            z,
            "skill_shift / std(|resid|)",
        )
        worst_shift = np.nanmax(
            np.vstack([worst_shift, shift.to_numpy(dtype=float)]), axis=0
        )
    _add_feature(
        features,
        formulas,
        f"worst_skill_shift_15v60_l{lag}",
        worst_shift,
        "max(skill_shift_mae_15v60)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e55(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cv_values = []
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mean_abs = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
        std_abs = tfl.rolling_std(
            resid.abs(),
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        cv = _safe_divide(std_abs.to_numpy(dtype=float), mean_abs.to_numpy(dtype=float) + 0.1)
        cv_values.append(cv)
        _add_feature(
            features,
            formulas,
            f"abs_err_mean_{col}_rm60_l{lag}",
            mean_abs,
            "roll_mean(|resid|)",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"abs_err_std_{col}_rm60_l{lag}",
            std_abs,
            "roll_std(|resid|)",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"abs_err_cv_{col}_rm60_l{lag}",
            cv,
            "abs_err_std / (abs_err_mean+eps)",
        )
    cv_matrix = np.vstack(cv_values)
    _add_feature(
        features,
        formulas,
        f"min_cv_model_rm60_l{lag}",
        np.nanmin(cv_matrix, axis=0),
        "min(abs_err_cv)",
    )
    _add_feature(
        features,
        formulas,
        f"mean_cv_models_rm60_l{lag}",
        np.nanmean(cv_matrix, axis=0),
        "mean(abs_err_cv)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e56(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    ewm_mae = {}
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        abs_err = resid.abs()
        ewm14 = tfl.ewm_mean(abs_err, halflife=14, min_periods=10, lag=lag, group_key=gk)
        ewm30 = tfl.ewm_mean(abs_err, halflife=30, min_periods=10, lag=lag, group_key=gk)
        ewm_mae[col] = ewm14
        _add_feature(
            features,
            formulas,
            f"ewm_mae_{col}_hl14_l{lag}",
            ewm14,
            "ewm_mean(|resid|)",
            {"halflife": 14, "lag": lag, "min_periods": 10},
        )
        _add_feature(
            features,
            formulas,
            f"ewm_mae_{col}_hl30_l{lag}",
            ewm30,
            "ewm_mean(|resid|)",
            {"halflife": 30, "lag": lag, "min_periods": 10},
        )
    ewm_ens = tfl.ewm_mean(
        df["resid_ens_mean"].abs(), halflife=14, min_periods=10, lag=lag, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        f"ewm_mae_ensmean_hl14_l{lag}",
        ewm_ens,
        "ewm_mean(|resid_ens_mean|)",
        {"halflife": 14, "lag": lag, "min_periods": 10},
    )
    for col in MODEL_COLS:
        rel = ewm_mae[col] - ewm_ens
        _add_feature(
            features,
            formulas,
            f"rel_ewm_mae_{col}_vs_ens_hl14_l{lag}",
            rel,
            "ewm_mae_model - ewm_mae_ensmean",
        )
    ewm_frame = pd.DataFrame(ewm_mae, index=df.index)
    best_id = _argmin_model(ewm_frame, MODEL_COLS)
    for col in MODEL_COLS:
        _add_feature(
            features,
            formulas,
            f"best_ewm_is_{col}",
            (best_id == col).astype(int),
            f"1[best_ewm_model_id == {col}]",
        )
    best_forecast = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        mask = best_id == col
        best_forecast[mask.to_numpy()] = df.loc[mask, col].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "best_ewm_model_forecast_today",
        best_forecast,
        "forecast(best_ewm_model_id)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e57(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    mae_frame = pd.DataFrame(index=df.index)
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
        mae_frame[col] = mae
        _add_feature(
            features,
            formulas,
            f"mae_{col}_rm60_l{lag}",
            mae,
            "roll_mean(|resid|)",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
    weights = _rowwise_inverse_weights(mae_frame.to_numpy(dtype=float), eps=0.05)
    fcst = np.sum(weights * df[MODEL_COLS].to_numpy(dtype=float), axis=1)
    entropy = _rowwise_entropy(weights, eps=1e-9)
    w_max = np.max(weights, axis=1)
    w_sorted = np.sort(weights, axis=1)
    w_top2 = np.sum(w_sorted[:, -2:], axis=1)
    _add_feature(
        features,
        formulas,
        f"fcst_invmae_wmean_rm60_l{lag}",
        fcst,
        "sum(w_m * model)",
    )
    _add_feature(
        features,
        formulas,
        f"w_entropy_rm60_l{lag}",
        entropy,
        "weight_entropy",
    )
    _add_feature(
        features,
        formulas,
        f"w_max_rm60_l{lag}",
        w_max,
        "max(weight)",
    )
    _add_feature(
        features,
        formulas,
        f"w_top2_sum_rm60_l{lag}",
        w_top2,
        "sum(top2 weights)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e58(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    alpha = 0.98
    sigma0_map = {}
    for station, group in ctx.train_df.groupby("station_id"):
        rmse = []
        for col in MODEL_COLS:
            resid = group["actual_tmax_f"] - group[col]
            val = float(np.sqrt(np.mean(resid**2))) if len(resid) else 1.0
            rmse.append(val if val > 0 else 1.0)
        sigma0_map[station] = np.array(rmse, dtype=float)
    weights_history = np.full((len(df), len(MODEL_COLS)), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        if idx.size == 0:
            continue
        sigma0 = sigma0_map.get(station)
        if sigma0 is None:
            sigma0 = np.full(len(MODEL_COLS), 1.0, dtype=float)
        weights = np.full(len(MODEL_COLS), 1.0 / len(MODEL_COLS), dtype=float)
        for pos, row_idx in enumerate(idx):
            resid_vec = (
                df.loc[row_idx, "actual_tmax_f"] - df.loc[row_idx, MODEL_COLS]
            ).to_numpy(dtype=float)
            sigma = np.where(sigma0 <= 0, 1.0, sigma0)
            likelihood = np.exp(-(resid_vec**2) / (2.0 * sigma**2))
            weights = (weights**alpha) * likelihood
            total = float(np.sum(weights))
            if total > 0:
                weights = weights / total
            else:
                weights = np.full(len(MODEL_COLS), 1.0 / len(MODEL_COLS), dtype=float)
            weights_history[row_idx] = weights
    weights_df = pd.DataFrame(weights_history, columns=[f"w_dma_{c}" for c in MODEL_COLS], index=df.index)
    weights_lag = weights_df.groupby(df["station_id"]).shift(lag)
    weights_lag = weights_lag.to_numpy(dtype=float)
    fcst = np.sum(weights_lag * df[MODEL_COLS].to_numpy(dtype=float), axis=1)
    entropy = _rowwise_entropy(weights_lag, eps=1e-9)
    w_max = np.nanmax(weights_lag, axis=1)
    _add_feature(
        features,
        formulas,
        f"fcst_dma_wmean_l{lag}",
        fcst,
        "sum(w_dma * model)",
    )
    _add_feature(
        features,
        formulas,
        f"w_dma_entropy_l{lag}",
        entropy,
        "entropy(w_dma)",
    )
    _add_feature(
        features,
        formulas,
        f"w_dma_max_l{lag}",
        w_max,
        "max(w_dma)",
    )
    train_fitted.append(
        {
            "name": "dma_weights",
            "fit_on": "train",
            "alpha": alpha,
            "sigma0_map": {k: v.tolist() for k, v in sigma0_map.items()},
            "truth_lag": lag,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e59(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    window = 60
    min_p = _min_periods(window)
    coeffs = np.full((len(df), len(MODEL_COLS)), np.nan, dtype=float)
    intercepts = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    dates = pd.to_datetime(df["target_date_local"]).to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            start = max(0, pos - window)
            end = pos - lag + 1
            window_idx = idx[start:end]
            if len(window_idx) < min_p:
                continue
            X = df.loc[window_idx, MODEL_COLS].to_numpy(dtype=float)
            y = df.loc[window_idx, "actual_tmax_f"].to_numpy(dtype=float)
            b0, b = _fit_ridge_coeffs(X, y, l2=1.0)
            intercepts[row_idx] = b0
            coeffs[row_idx] = b
    for i, col in enumerate(MODEL_COLS):
        _add_feature(
            features,
            formulas,
            f"ridge_b_{col}_rm60_l{lag}",
            coeffs[:, i],
            "ridge_coeff",
            {"window": window, "lag": lag, "min_periods": min_p, "l2": 1.0},
        )
    _add_feature(
        features,
        formulas,
        f"ridge_b0_rm60_l{lag}",
        intercepts,
        "ridge_intercept",
        {"window": window, "lag": lag, "min_periods": min_p, "l2": 1.0},
    )
    fcst = intercepts + np.sum(coeffs * df[MODEL_COLS].to_numpy(dtype=float), axis=1)
    _add_feature(
        features,
        formulas,
        f"fcst_ridge_rm60_l{lag}",
        fcst,
        "ridge_b0 + sum(b_m * model)",
    )
    _add_feature(
        features,
        formulas,
        f"ridge_weight_l1norm_rm60_l{lag}",
        np.sum(np.abs(coeffs), axis=1),
        "sum(|ridge_b_m|)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e60(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    window = 60
    min_p = _min_periods(window)
    weights = np.full((len(df), len(MODEL_COLS)), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            start = max(0, pos - window)
            end = pos - lag + 1
            window_idx = idx[start:end]
            if len(window_idx) < min_p:
                continue
            X = df.loc[window_idx, MODEL_COLS].to_numpy(dtype=float)
            y = df.loc[window_idx, "actual_tmax_f"].to_numpy(dtype=float)
            w = _solve_simplex_weights(X, y, l2=0.1)
            weights[row_idx] = w
    fcst = np.sum(weights * df[MODEL_COLS].to_numpy(dtype=float), axis=1)
    entropy = _rowwise_entropy(weights, eps=1e-9)
    w_max = np.nanmax(weights, axis=1)
    for i, col in enumerate(MODEL_COLS):
        _add_feature(
            features,
            formulas,
            f"w_simplex_{col}_rm60_l{lag}",
            weights[:, i],
            "simplex_weight",
            {"window": window, "lag": lag, "min_periods": min_p, "l2": 0.1},
        )
    _add_feature(
        features,
        formulas,
        f"fcst_simplex_rm60_l{lag}",
        fcst,
        "sum(w_simplex * model)",
    )
    _add_feature(
        features,
        formulas,
        f"w_simplex_entropy_rm60_l{lag}",
        entropy,
        "entropy(w_simplex)",
    )
    _add_feature(
        features,
        formulas,
        f"w_simplex_max_rm60_l{lag}",
        w_max,
        "max(w_simplex)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e61(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thresholds, default = _station_quantile(ctx.train_df, "ens_std", 0.75)
    thr = _map_station_threshold(df, thresholds, default)
    hi = (df["ens_std"] > thr).astype(int)
    lo = 1 - hi
    hi_count = tfl.rolling_event_count(
        hi, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    lo_count = tfl.rolling_event_count(
        lo, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    min_samples = 15
    selected_mae = pd.DataFrame(index=df.index)
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae_uncond = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
        mae_hi = tfl.rolling_conditional_mean(
            resid.abs(),
            hi,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        mae_lo = tfl.rolling_conditional_mean(
            resid.abs(),
            lo,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        selected = np.where(hi.to_numpy(dtype=int) == 1, mae_hi, mae_lo)
        fallback = np.where(hi.to_numpy(dtype=int) == 1, hi_count, lo_count)
        selected = np.where(fallback >= min_samples, selected, mae_uncond)
        selected_mae[col] = selected
        _add_feature(
            features,
            formulas,
            f"mae_{col}_selected_rm60_l{lag}",
            selected,
            "conditional_mae_by_spread",
        )
    best_id = _argmin_model(selected_mae, MODEL_COLS)
    for col in MODEL_COLS:
        _add_feature(
            features,
            formulas,
            f"best_model_by_spread_is_{col}",
            (best_id == col).astype(int),
            f"1[best_model_by_spread == {col}]",
        )
    best_forecast = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        mask = best_id == col
        best_forecast[mask.to_numpy()] = df.loc[mask, col].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "best_model_by_spread_forecast_today",
        best_forecast,
        "forecast(best_model_by_spread)",
    )
    train_fitted.append(
        {
            "name": "thr_spread_q75",
            "fit_on": "train",
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e62(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thresholds, default = _station_quantile(ctx.train_df, "ens_mean", 0.50)
    thr = _map_station_threshold(df, thresholds, default)
    warm = (df["ens_mean"] >= thr).astype(int)
    cold = 1 - warm
    warm_count = tfl.rolling_event_count(
        warm, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    cold_count = tfl.rolling_event_count(
        cold, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    min_samples = 15
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        bias_uncond = tfl.rolling_mean(
            resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        bias_warm = tfl.rolling_conditional_mean(
            resid,
            warm,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        bias_cold = tfl.rolling_conditional_mean(
            resid,
            cold,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        selected = np.where(warm.to_numpy(dtype=int) == 1, bias_warm, bias_cold)
        counts = np.where(warm.to_numpy(dtype=int) == 1, warm_count, cold_count)
        selected = np.where(counts >= min_samples, selected, bias_uncond)
        _add_feature(
            features,
            formulas,
            f"bias_{col}_selected_rm60_l{lag}",
            selected,
            "conditional_bias_by_temp",
        )
        _add_feature(
            features,
            formulas,
            f"{col}_biascorr_selected_rm60_l{lag}",
            df[col] + selected,
            "forecast + bias_selected",
        )
    train_fitted.append(
        {
            "name": "thr_temp_median",
            "fit_on": "train",
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e63(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    delta = df["ens_mean"].groupby(gk).diff()
    sign = np.sign(delta).fillna(0)
    sign_l1 = sign.groupby(gk).shift(1).fillna(0)
    warmup = (sign == 1).astype(int)
    cooldown = (sign == -1).astype(int)
    warm_count = tfl.rolling_event_count(
        warmup, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    cool_count = tfl.rolling_event_count(
        cooldown, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    min_samples = 15
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae_warm = tfl.rolling_conditional_mean(
            resid.abs(),
            warmup,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        mae_cool = tfl.rolling_conditional_mean(
            resid.abs(),
            cooldown,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        bias_warm = tfl.rolling_conditional_mean(
            resid,
            warmup,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        bias_cool = tfl.rolling_conditional_mean(
            resid,
            cooldown,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        selected_mae = np.where(sign_l1.to_numpy(dtype=int) == 1, mae_warm, mae_cool)
        selected_bias = np.where(sign_l1.to_numpy(dtype=int) == 1, bias_warm, bias_cool)
        count_sel = np.where(sign_l1.to_numpy(dtype=int) == 1, warm_count, cool_count)
        fallback_mae = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
        fallback_bias = tfl.rolling_mean(
            resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        selected_mae = np.where(count_sel >= min_samples, selected_mae, fallback_mae)
        selected_bias = np.where(count_sel >= min_samples, selected_bias, fallback_bias)
        _add_feature(
            features,
            formulas,
            f"mae_{col}_mom_selected_rm60_l{lag}",
            selected_mae,
            "mae_selected_by_momentum",
        )
        _add_feature(
            features,
            formulas,
            f"bias_{col}_mom_selected_rm60_l{lag}",
            selected_bias,
            "bias_selected_by_momentum",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e64(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    dates = pd.to_datetime(df["target_date_local"]).values.astype("datetime64[D]")
    doy = pd.to_datetime(df["target_date_local"]).dt.dayofyear.to_numpy()
    stations = df["station_id"].to_numpy()
    lookback_days = 730
    radius = 15
    min_samples = 30
    mae_vals = {col: np.full(len(df), np.nan, dtype=float) for col in MODEL_COLS}
    bias_vals = {col: np.full(len(df), np.nan, dtype=float) for col in MODEL_COLS}
    sample_counts = np.full(len(df), np.nan, dtype=float)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            cutoff_date = dates[pos] - np.timedelta64(lag, "D")
            start_date = dates[pos] - np.timedelta64(lookback_days, "D")
            cand_mask = (dates[idx] >= start_date) & (dates[idx] <= cutoff_date)
            if not cand_mask.any():
                continue
            cand_idx = idx[cand_mask]
            doy_diff = np.abs(doy[cand_idx] - doy[row_idx])
            doy_dist = np.minimum(doy_diff, 366 - doy_diff)
            cand_idx = cand_idx[doy_dist <= radius]
            if cand_idx.size == 0:
                continue
            sample_counts[row_idx] = float(cand_idx.size)
            for col in MODEL_COLS:
                resid = df.loc[cand_idx, "actual_tmax_f"] - df.loc[cand_idx, col]
                mae_vals[col][row_idx] = float(np.mean(np.abs(resid)))
                bias_vals[col][row_idx] = float(np.mean(resid))
    _add_feature(
        features,
        formulas,
        f"n_doy_samples_l{lag}",
        sample_counts,
        "count_doy_neighbors",
    )
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        fallback_mae = _rolling_abs_mean(resid, window=60, lag=lag, group_key=df["station_id"])
        fallback_bias = tfl.rolling_mean(
            resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=df["station_id"]
        )
        mae_final = np.where(sample_counts >= min_samples, mae_vals[col], fallback_mae)
        bias_final = np.where(sample_counts >= min_samples, bias_vals[col], fallback_bias)
        _add_feature(
            features,
            formulas,
            f"mae_{col}_doy15_l{lag}",
            mae_final,
            "mae_doy_neighborhood",
        )
        _add_feature(
            features,
            formulas,
            f"bias_{col}_doy15_l{lag}",
            bias_final,
            "bias_doy_neighborhood",
        )
    train_fitted.append(
        {
            "name": "doy_skill",
            "fit_on": "train",
            "radius": radius,
            "lookback_days": lookback_days,
            "min_samples": min_samples,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e65(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thr_temp_map, temp_default = _station_quantile(ctx.train_df, "ens_mean", 0.50)
    thr_spread_map, spread_default = _station_quantile(ctx.train_df, "ens_std", 0.75)
    thr_temp = _map_station_threshold(df, thr_temp_map, temp_default)
    thr_spread = _map_station_threshold(df, thr_spread_map, spread_default)
    temp_bin = (df["ens_mean"] >= thr_temp).astype(int)
    spread_bin = (df["ens_std"] >= thr_spread).astype(int)
    regime = temp_bin * 2 + spread_bin
    min_samples = 15
    bias_uncond = tfl.rolling_mean(
        df["resid_ens_mean"], window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    mae_uncond = _rolling_abs_mean(df["resid_ens_mean"], window=60, lag=lag, group_key=gk)
    selected_bias = np.full(len(df), np.nan, dtype=float)
    selected_mae = np.full(len(df), np.nan, dtype=float)
    selected_count = np.full(len(df), np.nan, dtype=float)
    for reg in range(4):
        reg_flag = (regime == reg).astype(int)
        count = tfl.rolling_event_count(
            reg_flag, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        bias = tfl.rolling_conditional_mean(
            df["resid_ens_mean"],
            reg_flag,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        mae = tfl.rolling_conditional_mean(
            df["resid_ens_mean"].abs(),
            reg_flag,
            window=60,
            min_periods=_min_periods(60),
            lag=lag,
            group_key=gk,
        )
        _add_feature(
            features,
            formulas,
            f"bias_reg{reg}_rm60_l{lag}",
            bias,
            "conditional_bias_regime",
        )
        _add_feature(
            features,
            formulas,
            f"mae_reg{reg}_rm60_l{lag}",
            mae,
            "conditional_mae_regime",
        )
        mask = regime.to_numpy() == reg
        selected_bias[mask] = bias.to_numpy()[mask]
        selected_mae[mask] = mae.to_numpy()[mask]
        selected_count[mask] = count.to_numpy()[mask]
    selected_bias = np.where(selected_count >= min_samples, selected_bias, bias_uncond)
    selected_mae = np.where(selected_count >= min_samples, selected_mae, mae_uncond)
    _add_feature(
        features,
        formulas,
        f"bias_reg_selected_rm60_l{lag}",
        selected_bias,
        "bias_selected_by_regime",
    )
    _add_feature(
        features,
        formulas,
        f"mae_reg_selected_rm60_l{lag}",
        selected_mae,
        "mae_selected_by_regime",
    )
    train_fitted.append(
        {
            "name": "regime_thresholds",
            "fit_on": "train",
            "thr_temp": thr_temp_map,
            "thr_spread": thr_spread_map,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e66(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thr_temp_map, temp_default = _station_quantile(ctx.train_df, "ens_mean", 0.50)
    thr_spread_map, spread_default = _station_quantile(ctx.train_df, "ens_std", 0.75)
    thr_temp = _map_station_threshold(df, thr_temp_map, temp_default)
    thr_spread = _map_station_threshold(df, thr_spread_map, spread_default)
    temp_bin = (df["ens_mean"] >= thr_temp).astype(int)
    spread_bin = (df["ens_std"] >= thr_spread).astype(int)
    regime = temp_bin * 2 + spread_bin
    min_samples = 20
    mae_uncond = pd.DataFrame(index=df.index)
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae_uncond[col] = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
    mae_reg = {reg: pd.DataFrame(index=df.index) for reg in range(4)}
    count_reg = {}
    for reg in range(4):
        reg_flag = (regime == reg).astype(int)
        count = tfl.rolling_event_count(
            reg_flag, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        count_reg[reg] = count
        for col in MODEL_COLS:
            resid = df["actual_tmax_f"] - df[col]
            mae = tfl.rolling_conditional_mean(
                resid.abs(),
                reg_flag,
                window=60,
                min_periods=_min_periods(60),
                lag=lag,
                group_key=gk,
            )
            mae_reg[reg][col] = mae
    selected_mae = np.full((len(df), len(MODEL_COLS)), np.nan, dtype=float)
    selected_count = np.full(len(df), np.nan, dtype=float)
    for reg in range(4):
        mask = regime.to_numpy() == reg
        selected_mae[mask] = mae_reg[reg].to_numpy(dtype=float)[mask]
        selected_count[mask] = count_reg[reg].to_numpy(dtype=float)[mask]
    fallback_mask = selected_count < min_samples
    if fallback_mask.any():
        selected_mae[fallback_mask] = mae_uncond.to_numpy(dtype=float)[fallback_mask]
    selected_df = pd.DataFrame(selected_mae, columns=MODEL_COLS, index=df.index)
    best_id = _argmin_model(selected_df, MODEL_COLS)
    for col in MODEL_COLS:
        _add_feature(
            features,
            formulas,
            f"best_reg_is_{col}",
            (best_id == col).astype(int),
            f"1[best_model_regime == {col}]",
        )
    best_forecast = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        mask = best_id == col
        best_forecast[mask.to_numpy()] = df.loc[mask, col].to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        f"best_regime_forecast_today",
        best_forecast,
        "forecast(best_model_regime)",
    )
    _add_feature(
        features,
        formulas,
        f"n_regime_samples_selected_l{lag}",
        selected_count,
        "regime_sample_count",
    )
    train_fitted.append(
        {
            "name": "regime_thresholds",
            "fit_on": "train",
            "thr_temp": thr_temp_map,
            "thr_spread": thr_spread_map,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e67(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    stations = df["station_id"].to_numpy()
    cusum_pos = np.full(len(df), np.nan, dtype=float)
    cusum_neg = np.full(len(df), np.nan, dtype=float)
    alarm = np.full(len(df), np.nan, dtype=float)
    params = {}
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        train_mask = ctx.train_df["station_id"] == station
        train_resid = ctx.train_df.loc[train_mask, "resid_ens_mean"].to_numpy(dtype=float)
        sigma = float(np.std(train_resid, ddof=0)) if len(train_resid) else 1.0
        sigma = sigma if sigma > 0 else 1.0
        k = 0.5 * sigma
        h = 5.0 * sigma
        params[station] = {"sigma": sigma, "k": k, "h": h}
        cplus = 0.0
        cminus = 0.0
        cplus_hist = np.full(len(idx), np.nan, dtype=float)
        cminus_hist = np.full(len(idx), np.nan, dtype=float)
        for pos, row_idx in enumerate(idx):
            r = resid[row_idx]
            cplus = max(0.0, cplus + r - k)
            cminus = max(0.0, cminus - r - k)
            cplus_hist[pos] = cplus
            cminus_hist[pos] = cminus
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            cusum_pos[row_idx] = cplus_hist[pos - lag]
            cusum_neg[row_idx] = cminus_hist[pos - lag]
            alarm[row_idx] = float(
                (cplus_hist[pos - lag] > h) or (cminus_hist[pos - lag] > h)
            )
    _add_feature(
        features,
        formulas,
        f"cusum_pos_l{lag}",
        cusum_pos,
        "cusum_positive",
    )
    _add_feature(
        features,
        formulas,
        f"cusum_neg_l{lag}",
        cusum_neg,
        "cusum_negative",
    )
    _add_feature(
        features,
        formulas,
        f"cusum_alarm_l{lag}",
        alarm,
        "cusum_alarm",
    )
    train_fitted.append(
        {
            "name": "cusum_params",
            "fit_on": "train",
            "params": params,
            "truth_lag": lag,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e68(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    stations = df["station_id"].to_numpy()
    ph_stat = np.full(len(df), np.nan, dtype=float)
    alarm = np.full(len(df), np.nan, dtype=float)
    thresholds = {}
    delta = 0.05
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        mean = 0.0
        ph = 0.0
        ph_min = 0.0
        ph_hist = np.full(len(idx), np.nan, dtype=float)
        for pos, row_idx in enumerate(idx):
            r = resid[row_idx]
            mean = mean + (r - mean) / max(1, pos + 1)
            ph = ph + (r - mean - delta)
            ph_min = min(ph_min, ph)
            ph_hist[pos] = ph - ph_min
        train_idx = ctx.train_df.index.intersection(df.index[idx])
        train_vals = ph_hist[[np.where(idx == i)[0][0] for i in train_idx]] if len(train_idx) else ph_hist
        thr = float(np.quantile(train_vals[np.isfinite(train_vals)], 0.99)) if len(train_vals) else 0.0
        thresholds[station] = thr
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            ph_val = ph_hist[pos - lag]
            ph_stat[row_idx] = ph_val
            alarm[row_idx] = float(ph_val > thr)
    _add_feature(
        features,
        formulas,
        f"ph_stat_l{lag}",
        ph_stat,
        "page_hinkley_stat",
    )
    _add_feature(
        features,
        formulas,
        f"ph_alarm_l{lag}",
        alarm,
        "page_hinkley_alarm",
    )
    train_fitted.append(
        {
            "name": "page_hinkley_thresholds",
            "fit_on": "train",
            "thresholds": thresholds,
            "delta": delta,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e69(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    r = df["resid_ens_mean"]
    q10_15 = tfl.rolling_quantile(
        r, window=15, min_periods=_min_periods(15), lag=lag, q=0.10, group_key=gk
    )
    q50_15 = tfl.rolling_quantile(
        r, window=15, min_periods=_min_periods(15), lag=lag, q=0.50, group_key=gk
    )
    q90_15 = tfl.rolling_quantile(
        r, window=15, min_periods=_min_periods(15), lag=lag, q=0.90, group_key=gk
    )
    q10_60 = tfl.rolling_quantile(
        r, window=60, min_periods=_min_periods(60), lag=lag, q=0.10, group_key=gk
    )
    q50_60 = tfl.rolling_quantile(
        r, window=60, min_periods=_min_periods(60), lag=lag, q=0.50, group_key=gk
    )
    q90_60 = tfl.rolling_quantile(
        r, window=60, min_periods=_min_periods(60), lag=lag, q=0.90, group_key=gk
    )
    iqr_15 = q90_15 - q10_15
    iqr_60 = q90_60 - q10_60
    _add_feature(features, formulas, f"res_q50_shift_15v60_l{lag}", q50_15 - q50_60, "q50_15 - q50_60")
    _add_feature(features, formulas, f"res_iqr_15_l{lag}", iqr_15, "iqr_15")
    _add_feature(features, formulas, f"res_iqr_60_l{lag}", iqr_60, "iqr_60")
    _add_feature(features, formulas, f"res_iqr_shift_15v60_l{lag}", iqr_15 - iqr_60, "iqr_15 - iqr_60")
    skew_proxy = _safe_divide((q90_60 + q10_60 - 2 * q50_60).to_numpy(), iqr_60.to_numpy() + 0.1)
    _add_feature(features, formulas, f"res_skew_proxy_60_l{lag}", skew_proxy, "skew_proxy")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e70(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    r = df["resid_ens_mean"]
    ewm7 = tfl.ewm_mean(r, halflife=7, min_periods=10, lag=lag, group_key=gk)
    ewm30 = tfl.ewm_mean(r, halflife=30, min_periods=10, lag=lag, group_key=gk)
    _add_feature(
        features,
        formulas,
        f"bias_ewm_hl7_l{lag}",
        ewm7,
        "ewm_mean(resid)",
        {"halflife": 7, "lag": lag, "min_periods": 10},
    )
    _add_feature(
        features,
        formulas,
        f"bias_ewm_hl30_l{lag}",
        ewm30,
        "ewm_mean(resid)",
        {"halflife": 30, "lag": lag, "min_periods": 10},
    )
    _add_feature(
        features,
        formulas,
        f"bias_drift_ewm_7m30_l{lag}",
        ewm7 - ewm30,
        "ewm7 - ewm30",
    )
    _add_feature(
        features,
        formulas,
        f"ens_mean_biascorr_ewm7_l{lag}",
        df["ens_mean"] + ewm7,
        "ens_mean + bias_ewm7",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e71(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    abs_resid = np.abs(resid)
    stations = df["station_id"].to_numpy()
    p0 = np.full(len(df), np.nan, dtype=float)
    p1 = np.full(len(df), np.nan, dtype=float)
    bias = np.full(len(df), np.nan, dtype=float)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        train_idx = np.intersect1d(idx, ctx.train_df.index)
        if train_idx.size < 5:
            continue
        obs_train = np.column_stack([resid[train_idx], abs_resid[train_idx]])
        obs_train = obs_train[np.isfinite(obs_train).all(axis=1)]
        if obs_train.shape[0] < 5:
            continue
        params = hmm_utils.fit_gaussian_hmm(
            obs_train, n_states=2, n_iters=10, seed=ctx.seed
        )
        obs_full = np.column_stack([resid[idx], abs_resid[idx]])
        med = np.nanmedian(obs_train, axis=0)
        obs_full = np.where(np.isfinite(obs_full), obs_full, med)
        probs = hmm_utils.forward_filter(obs_full, params)
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            p0_val = probs[pos - lag, 0]
            p1_val = probs[pos - lag, 1]
            p0[row_idx] = p0_val
            p1[row_idx] = p1_val
            bias[row_idx] = p0_val * params.means[0][0] + p1_val * params.means[1][0]
        train_fitted.append(
            {
                "name": "hmm_residual_state",
                "fit_on": "train",
                "station_id": station,
                "pi": params.pi.tolist(),
                "A": params.A.tolist(),
                "means": params.means.tolist(),
                "covars": params.covars.tolist(),
            }
        )
    _add_feature(
        features,
        formulas,
        f"p_state0_l{lag}",
        p0,
        "HMM filtered P(state0)",
    )
    _add_feature(
        features,
        formulas,
        f"p_state1_l{lag}",
        p1,
        "HMM filtered P(state1)",
    )
    _add_feature(
        features,
        formulas,
        f"hmm_bias_l{lag}",
        bias,
        "p_state * mean_resid",
    )
    _add_feature(
        features,
        formulas,
        f"ens_mean_hmm_biascorr_l{lag}",
        df["ens_mean"] + bias,
        "ens_mean + hmm_bias",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e72(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    y = df["actual_tmax_f"]
    _add_feature(
        features,
        formulas,
        f"y_lag{lag}",
        y.groupby(gk).shift(lag),
        "y lag",
    )
    _add_feature(
        features,
        formulas,
        f"y_lag{lag+1}",
        y.groupby(gk).shift(lag + 1),
        "y lag",
    )
    _add_feature(
        features,
        formulas,
        f"y_roll_mean_7_l{lag}",
        tfl.rolling_mean(y, window=7, min_periods=_min_periods(7), lag=lag, group_key=gk),
        "roll_mean(y)",
        {"window": 7, "lag": lag, "min_periods": _min_periods(7)},
    )
    _add_feature(
        features,
        formulas,
        f"y_roll_mean_30_l{lag}",
        tfl.rolling_mean(y, window=30, min_periods=_min_periods(30), lag=lag, group_key=gk),
        "roll_mean(y)",
        {"window": 30, "lag": lag, "min_periods": _min_periods(30)},
    )
    y_lag2 = y.groupby(gk).shift(lag)
    y_lag3 = y.groupby(gk).shift(lag + 1)
    _add_feature(
        features,
        formulas,
        f"y_delta_l{lag}",
        y_lag2 - y_lag3,
        "y(T-2)-y(T-3)",
    )
    _add_feature(
        features,
        formulas,
        f"y_anom_vs_recent7_l{lag}",
        y_lag2 - features[f"y_roll_mean_7_l{lag}"],
        "y(T-2)-roll_mean_7",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e73(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    doy = pd.to_datetime(df["target_date_local"]).dt.dayofyear
    clim_vals = np.full(len(df), np.nan, dtype=float)
    clim_maps = {}
    for station, group in ctx.train_df.groupby("station_id"):
        doy_mean = group.groupby(pd.to_datetime(group["target_date_local"]).dt.dayofyear)[
            "actual_tmax_f"
        ].mean()
        overall = float(group["actual_tmax_f"].mean()) if len(group) else 0.0
        clim_maps[station] = {"overall": overall, "doy": doy_mean.to_dict()}
        idx = df["station_id"] == station
        mapped = doy[idx].map(doy_mean).fillna(overall)
        clim_vals[idx.to_numpy()] = mapped.to_numpy(dtype=float)
    anom = df["actual_tmax_f"] - clim_vals
    _add_feature(
        features,
        formulas,
        f"anom_lag{lag}",
        anom.groupby(gk).shift(lag),
        "anom lag",
    )
    _add_feature(
        features,
        formulas,
        f"anom_roll_mean_30_l{lag}",
        tfl.rolling_mean(anom, window=30, min_periods=_min_periods(30), lag=lag, group_key=gk),
        "roll_mean(anom)",
    )
    _add_feature(
        features,
        formulas,
        f"anom_roll_std_30_l{lag}",
        tfl.rolling_std(anom, window=30, min_periods=_min_periods(30), lag=lag, group_key=gk),
        "roll_std(anom)",
    )
    _add_feature(
        features,
        formulas,
        f"anom_roll_mean_7_l{lag}",
        tfl.rolling_mean(anom, window=7, min_periods=_min_periods(7), lag=lag, group_key=gk),
        "roll_mean(anom)",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_anom_today",
        df["ens_mean"] - clim_vals,
        "ens_mean - clim_doy",
    )
    train_fitted.append(
        {
            "name": "climatology_doy",
            "fit_on": "train",
            "maps": clim_maps,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e74(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    y = df["actual_tmax_f"]
    slope15 = tfl.rolling_slope(
        y, window=15, min_periods=_min_periods(15), lag=lag, group_key=gk
    )
    slope60 = tfl.rolling_slope(
        y, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    _add_feature(
        features,
        formulas,
        f"y_slope_15_l{lag}",
        slope15,
        "rolling_slope(y)",
    )
    _add_feature(
        features,
        formulas,
        f"y_slope_60_l{lag}",
        slope60,
        "rolling_slope(y)",
    )
    _add_feature(
        features,
        formulas,
        f"y_accel_proxy_l{lag}",
        slope15 - slope60,
        "slope15 - slope60",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e75(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    dy = df["actual_tmax_f"].groupby(gk).diff()
    _add_feature(
        features,
        formulas,
        f"dy_absmean_30_l{lag}",
        tfl.rolling_mean(dy.abs(), window=30, min_periods=_min_periods(30), lag=lag, group_key=gk),
        "roll_mean(|dy|)",
    )
    _add_feature(
        features,
        formulas,
        f"dy_std_30_l{lag}",
        tfl.rolling_std(dy, window=30, min_periods=_min_periods(30), lag=lag, group_key=gk),
        "roll_std(dy)",
    )
    sign = np.sign(dy).fillna(0)
    _add_feature(
        features,
        formulas,
        f"dy_sign_changes_30_l{lag}",
        tfl.switch_count(sign, window=30, min_periods=_min_periods(30), lag=lag, group_key=gk),
        "switch_count(sign(dy))",
    )
    dy_lag2 = dy.groupby(gk).shift(lag)
    dy_lag3 = dy.groupby(gk).shift(lag + 1)
    _add_feature(
        features,
        formulas,
        f"dy_turning_point_l{lag}",
        ((dy_lag2 * dy_lag3) < 0).astype(int),
        "1[dy(T-2)*dy(T-3)<0]",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e76(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    doy = pd.to_datetime(df["target_date_local"]).dt.dayofyear
    clim_vals = np.full(len(df), np.nan, dtype=float)
    for station, group in ctx.train_df.groupby("station_id"):
        doy_mean = group.groupby(pd.to_datetime(group["target_date_local"]).dt.dayofyear)[
            "actual_tmax_f"
        ].mean()
        overall = float(group["actual_tmax_f"].mean()) if len(group) else 0.0
        idx = df["station_id"] == station
        mapped = doy[idx].map(doy_mean).fillna(overall)
        clim_vals[idx.to_numpy()] = mapped.to_numpy(dtype=float)
    anom = df["actual_tmax_f"] - clim_vals
    anom_mean7 = tfl.rolling_mean(
        anom, window=7, min_periods=_min_periods(7), lag=lag, group_key=gk
    )
    regime = (anom_mean7 >= 0).astype(int)
    pos = (anom >= 0).astype(int)
    neg = 1 - pos
    pos_count = tfl.rolling_event_count(
        pos, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    neg_count = tfl.rolling_event_count(
        neg, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
    )
    min_samples = 15
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae_pos = tfl.rolling_conditional_mean(
            resid.abs(), pos, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        mae_neg = tfl.rolling_conditional_mean(
            resid.abs(), neg, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        selected = np.where(regime.to_numpy(dtype=int) == 1, mae_pos, mae_neg)
        counts = np.where(regime.to_numpy(dtype=int) == 1, pos_count, neg_count)
        fallback = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
        selected = np.where(counts >= min_samples, selected, fallback)
        _add_feature(
            features,
            formulas,
            f"mae_{col}_anom_selected_rm60_l{lag}",
            selected,
            "mae_selected_by_anom_regime",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e77(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    ens_std = df["ens_std"]
    ens_mean = df["ens_mean"]
    std_lag = ens_std.groupby(gk).shift(1)
    mean_lag = ens_mean.groupby(gk).shift(1)
    acf_std_60 = tfl.rolling_corr(
        ens_std,
        std_lag,
        window=60,
        min_periods=_min_periods(60),
        lag=1,
        group_key=gk,
    )
    acf_std_30 = tfl.rolling_corr(
        ens_std,
        std_lag,
        window=30,
        min_periods=_min_periods(30),
        lag=1,
        group_key=gk,
    )
    acf_mean_60 = tfl.rolling_corr(
        ens_mean,
        mean_lag,
        window=60,
        min_periods=_min_periods(60),
        lag=1,
        group_key=gk,
    )
    _add_feature(features, formulas, "acf1_ens_std_rm60_l1", acf_std_60, "acf1(ens_std)")
    _add_feature(features, formulas, "acf1_ens_std_rm30_l1", acf_std_30, "acf1(ens_std)")
    _add_feature(features, formulas, "acf1_ens_mean_rm60_l1", acf_mean_60, "acf1(ens_mean)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e78(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    acf_vals = []
    for col in MODEL_COLS:
        drift = df[col] - df["ens_mean"]
        drift_lag = drift.groupby(gk).shift(1)
        acf = tfl.rolling_corr(
            drift,
            drift_lag,
            window=60,
            min_periods=_min_periods(60),
            lag=1,
            group_key=gk,
        )
        acf_vals.append(acf.to_numpy(dtype=float))
        _add_feature(
            features,
            formulas,
            f"acf1_drift_{col}_rm60_l1",
            acf,
            "acf1(drift)",
        )
    acf_mat = np.vstack(acf_vals)
    _add_feature(
        features,
        formulas,
        "acf1_drift_mean_rm60_l1",
        np.nanmean(acf_mat, axis=0),
        "mean(acf1_drift)",
    )
    _add_feature(
        features,
        formulas,
        "acf1_drift_min_rm60_l1",
        np.nanmin(acf_mat, axis=0),
        "min(acf1_drift)",
    )
    _add_feature(
        features,
        formulas,
        "acf1_drift_max_rm60_l1",
        np.nanmax(acf_mat, axis=0),
        "max(acf1_drift)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e79(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    pairs = []
    for i, col_i in enumerate(MODEL_COLS):
        for col_j in MODEL_COLS[i + 1 :]:
            pairs.append((col_i, col_j))
    corr_vals = []
    for col_i, col_j in pairs:
        corr = tfl.rolling_corr(
            df[col_i],
            df[col_j],
            window=60,
            min_periods=_min_periods(60),
            lag=1,
            group_key=gk,
        )
        corr_vals.append(corr.to_numpy(dtype=float))
        _add_feature(
            features,
            formulas,
            f"corr_{col_i}_{col_j}_rm60_l1",
            corr,
            "rolling_corr",
        )
    corr_mat = np.vstack(corr_vals)
    g1 = {"gfs_tmax_f", "nam_tmax_f", "gefsatmosmean_tmax_f"}
    g2 = {"hrrr_tmax_f", "rap_tmax_f", "nbm_tmax_f"}
    syn_mask = []
    meso_mask = []
    cross_mask = []
    for i, (col_i, col_j) in enumerate(pairs):
        if col_i in g1 and col_j in g1:
            syn_mask.append(i)
        elif col_i in g2 and col_j in g2:
            meso_mask.append(i)
        else:
            cross_mask.append(i)
    _add_feature(
        features,
        formulas,
        "corr_synoptic_mean_rm60_l1",
        np.nanmean(corr_mat[syn_mask], axis=0) if syn_mask else np.nan,
        "mean(corr_synoptic)",
    )
    _add_feature(
        features,
        formulas,
        "corr_meso_mean_rm60_l1",
        np.nanmean(corr_mat[meso_mask], axis=0) if meso_mask else np.nan,
        "mean(corr_meso)",
    )
    _add_feature(
        features,
        formulas,
        "corr_cross_mean_rm60_l1",
        np.nanmean(corr_mat[cross_mask], axis=0) if cross_mask else np.nan,
        "mean(corr_cross)",
    )
    _add_feature(
        features,
        formulas,
        "corr_pairwise_std_rm60_l1",
        np.nanstd(corr_mat, axis=0),
        "std(pairwise_corr)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e80(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    deltas = []
    for col in MODEL_COLS:
        deltas.append(df[col].groupby(gk).diff())
    delta_df = pd.concat(deltas, axis=1)
    delta_df.columns = [f"delta_{c}" for c in MODEL_COLS]
    train_delta = delta_df.loc[ctx.train_df.index].dropna()
    if len(train_delta) == 0:
        return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)
    pca = PCA(n_components=2, random_state=ctx.seed)
    pca.fit(train_delta.to_numpy(dtype=float))
    scores = pca.transform(delta_df.fillna(0.0).to_numpy(dtype=float))
    pc1 = pd.Series(scores[:, 0], index=df.index)
    pc2 = pd.Series(scores[:, 1], index=df.index)
    pc1_l1 = pc1.groupby(gk).shift(1)
    pc2_l1 = pc2.groupby(gk).shift(1)
    _add_feature(features, formulas, "pc_delta1_l1", pc1_l1, "pc1(delta)")
    _add_feature(features, formulas, "pc_delta2_l1", pc2_l1, "pc2(delta)")
    _add_feature(
        features,
        formulas,
        "pc_delta1_rm30_l1",
        tfl.rolling_mean(pc1_l1, window=30, min_periods=_min_periods(30), lag=0, group_key=gk),
        "roll_mean(pc_delta1_l1)",
    )
    _add_feature(
        features,
        formulas,
        "pc_delta1_rs30_l1",
        tfl.rolling_std(pc1_l1, window=30, min_periods=_min_periods(30), lag=0, group_key=gk),
        "roll_std(pc_delta1_l1)",
    )
    _add_feature(
        features,
        formulas,
        "pc_delta2_rm30_l1",
        tfl.rolling_mean(pc2_l1, window=30, min_periods=_min_periods(30), lag=0, group_key=gk),
        "roll_mean(pc_delta2_l1)",
    )
    train_fitted.append(
        {
            "name": "pca_delta",
            "fit_on": "train",
            "components": pca.components_.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e81(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    train_scaled = scaled[ctx.train_df.index]
    cov = np.cov(train_scaled, rowvar=False)
    shrink = 0.1
    cov = (1.0 - shrink) * cov + shrink * np.eye(cov.shape[0])
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(cov.shape[0])
    neighbors, distances = _knn_neighbors_mahalanobis(
        df,
        scaled,
        inv_cov=inv_cov,
        group_key=ctx.group_key,
        truth_lag=ctx.truth_lag,
        lookback_days=365,
        k=20,
    )
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    mean_resid, std_resid, mean_dist = _knn_mean_std(neighbors, distances, resid)
    _add_feature(
        features,
        formulas,
        f"knnM_resid_mean_k20_l{ctx.truth_lag}",
        mean_resid,
        "KNN Mahalanobis mean(resid)",
    )
    _add_feature(
        features,
        formulas,
        f"knnM_resid_std_k20_l{ctx.truth_lag}",
        std_resid,
        "KNN Mahalanobis std(resid)",
    )
    _add_feature(
        features,
        formulas,
        f"knnM_dist_mean_k20_l{ctx.truth_lag}",
        mean_dist,
        "KNN Mahalanobis mean distance",
    )
    train_fitted.append(
        {
            "name": "knn_mahalanobis",
            "fit_on": "train",
            "features": feature_cols,
            "scaler": scaler_meta,
            "shrinkage": shrink,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e82(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    neighbors, distances = _knn_neighbors(
        df,
        scaled,
        group_key=ctx.group_key,
        truth_lag=ctx.truth_lag,
        lookback_days=365,
        k=50,
    )
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["target_date_local"]).values.astype("datetime64[D]")
    weighted = np.full(len(df), np.nan, dtype=float)
    std = np.full(len(df), np.nan, dtype=float)
    eff_n = np.full(len(df), np.nan, dtype=float)
    nearest = np.full(len(df), np.nan, dtype=float)
    tau = 30.0
    for idx, neigh in enumerate(neighbors):
        if neigh.size == 0:
            continue
        dists = distances[idx]
        if dists.size == 0:
            continue
        h = float(np.median(dists))
        if not np.isfinite(h) or h <= 0:
            h = float(np.mean(dists)) if np.mean(dists) > 0 else 1e-6
        age_days = (dates[idx] - dates[neigh]).astype("timedelta64[D]").astype(float)
        weights = np.exp(-(dists**2) / (h**2)) * np.exp(-age_days / tau)
        sum_w = float(np.sum(weights))
        if sum_w <= 0:
            continue
        vals = resid[neigh]
        mean_val = float(np.sum(weights * vals) / sum_w)
        weighted[idx] = mean_val
        std[idx] = float(np.sqrt(np.sum(weights * (vals - mean_val) ** 2) / sum_w))
        nearest[idx] = float(np.min(dists))
        sum_w2 = float(np.sum(weights**2))
        eff_n[idx] = float((sum_w**2) / sum_w2) if sum_w2 > 0 else float(neigh.size)
    _add_feature(
        features,
        formulas,
        f"kern_td_resid_mean_l{ctx.truth_lag}",
        weighted,
        "kernel_time_decay_mean(resid)",
    )
    _add_feature(
        features,
        formulas,
        f"kern_td_resid_std_l{ctx.truth_lag}",
        std,
        "kernel_time_decay_std(resid)",
    )
    _add_feature(
        features,
        formulas,
        f"kern_td_ess_l{ctx.truth_lag}",
        eff_n,
        "kernel_time_decay_ess",
    )
    _add_feature(
        features,
        formulas,
        f"kern_td_nearest_dist_l{ctx.truth_lag}",
        nearest,
        "kernel_time_decay_nearest",
    )
    train_fitted.append(
        {
            "name": "knn_time_decay",
            "fit_on": "train",
            "features": feature_cols,
            "scaler": scaler_meta,
            "k": 50,
            "lookback_days": 365,
            "tau_days": tau,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e83(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    neighbors, _ = _knn_neighbors(
        df,
        scaled,
        group_key=ctx.group_key,
        truth_lag=ctx.truth_lag,
        lookback_days=365,
        k=50,
    )
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    a = np.full(len(df), np.nan, dtype=float)
    b = np.full(len(df), np.nan, dtype=float)
    c = np.full(len(df), np.nan, dtype=float)
    pred = np.full(len(df), np.nan, dtype=float)
    for idx, neigh in enumerate(neighbors):
        if neigh.size < 10:
            continue
        X = df.loc[neigh, ["ens_std", "ens_mean"]].to_numpy(dtype=float)
        y = resid[neigh]
        b0, coeff = _fit_ridge_coeffs(X, y, l2=0.1)
        a[idx] = b0
        b[idx] = coeff[0]
        c[idx] = coeff[1]
        pred[idx] = b0 + coeff[0] * df.loc[idx, "ens_std"] + coeff[1] * df.loc[idx, "ens_mean"]
    _add_feature(features, formulas, f"analog_loc_a_l{ctx.truth_lag}", a, "local_reg_a")
    _add_feature(features, formulas, f"analog_loc_b_l{ctx.truth_lag}", b, "local_reg_b")
    _add_feature(features, formulas, f"analog_loc_c_l{ctx.truth_lag}", c, "local_reg_c")
    _add_feature(
        features,
        formulas,
        f"analog_loc_resid_pred_l{ctx.truth_lag}",
        pred,
        "local_reg_resid_pred",
    )
    _add_feature(
        features,
        formulas,
        f"ens_mean_analogloc_corr_l{ctx.truth_lag}",
        df["ens_mean"] + pred,
        "ens_mean + analog_loc_resid_pred",
    )
    train_fitted.append(
        {
            "name": "knn_local_regression",
            "fit_on": "train",
            "features": feature_cols,
            "scaler": scaler_meta,
            "k": 50,
            "lookback_days": 365,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e84(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    neighbors, _ = _knn_neighbors(
        df,
        scaled,
        group_key=ctx.group_key,
        truth_lag=ctx.truth_lag,
        lookback_days=365,
        k=30,
    )
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    q = _knn_quantiles(neighbors, resid, [0.10, 0.50, 0.90])
    q10 = q[0.10]
    q50 = q[0.50]
    q90 = q[0.90]
    _add_feature(
        features,
        formulas,
        f"analog_resid_q10_k30_l{ctx.truth_lag}",
        q10,
        "analog_resid_q10",
    )
    _add_feature(
        features,
        formulas,
        f"analog_resid_q50_k30_l{ctx.truth_lag}",
        q50,
        "analog_resid_q50",
    )
    _add_feature(
        features,
        formulas,
        f"analog_resid_q90_k30_l{ctx.truth_lag}",
        q90,
        "analog_resid_q90",
    )
    _add_feature(
        features,
        formulas,
        f"analog_resid_iqr_k30_l{ctx.truth_lag}",
        q90 - q10,
        "analog_resid_iqr",
    )
    _add_feature(
        features,
        formulas,
        f"ens_mean_analog_q50_corr_l{ctx.truth_lag}",
        df["ens_mean"] + q50,
        "ens_mean + analog_resid_q50",
    )
    train_fitted.append(
        {
            "name": "knn_analog_quantiles",
            "fit_on": "train",
            "features": feature_cols,
            "scaler": scaler_meta,
            "k": 30,
            "lookback_days": 365,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e85(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    thresholds, default = _station_quantile(ctx.train_df, "ens_std", 0.75)
    thr = _map_station_threshold(df, thresholds, default)
    hi = (df["ens_std"] > thr).to_numpy(dtype=bool)
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["target_date_local"]).values.astype("datetime64[D]")
    stations = df["station_id"].to_numpy()
    neighbors = [np.array([], dtype=int) for _ in range(len(df))]
    distances = [np.array([], dtype=float) for _ in range(len(df))]
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        station_dates = dates[idx]
        station_feat = scaled[idx]
        for pos, row_idx in enumerate(idx):
            cutoff = station_dates[pos] - np.timedelta64(ctx.truth_lag, "D")
            start_date = station_dates[pos] - np.timedelta64(365, "D")
            start = np.searchsorted(station_dates, start_date, side="left")
            end = np.searchsorted(station_dates, cutoff, side="right")
            if end <= start:
                continue
            cand_idx = idx[start:end]
            cand_idx = cand_idx[hi[cand_idx] == hi[row_idx]]
            if cand_idx.size == 0:
                continue
            cand_feat = station_feat[np.searchsorted(idx, cand_idx)]
            dists = np.linalg.norm(cand_feat - station_feat[pos], axis=1)
            order = np.argsort(dists)[:20]
            neighbors[row_idx] = cand_idx[order]
            distances[row_idx] = dists[order]
    mean_resid, _, mean_dist = _knn_mean_std(neighbors, distances, resid)
    _add_feature(
        features,
        formulas,
        f"knn_reg_resid_mean_k20_l{ctx.truth_lag}",
        mean_resid,
        "knn_regime_resid_mean",
    )
    _add_feature(
        features,
        formulas,
        f"knn_reg_dist_mean_k20_l{ctx.truth_lag}",
        mean_dist,
        "knn_regime_dist_mean",
    )
    _add_feature(
        features,
        formulas,
        f"knn_reg_neighbor_count_l{ctx.truth_lag}",
        np.array([len(n) for n in neighbors], dtype=float),
        "knn_regime_neighbor_count",
    )
    train_fitted.append(
        {
            "name": "knn_regime_restricted",
            "fit_on": "train",
            "features": feature_cols,
            "scaler": scaler_meta,
            "thresholds": thresholds,
            "default": float(default),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e86(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    feature_cols = MODEL_COLS + ["ens_std", "sin_doy", "cos_doy"]
    scaled, scaler_meta, _ = _standardize_features(ctx.train_df, df, feature_cols)
    train_scaled = scaled[ctx.train_df.index]
    if train_scaled.shape[0] < 21:
        return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)
    d20_vals = []
    for i in range(train_scaled.shape[0]):
        dists = np.linalg.norm(train_scaled - train_scaled[i], axis=1)
        order = np.sort(dists)
        if len(order) > 20:
            d20_vals.append(float(order[20]))
    d_thresh = float(np.median(d20_vals)) if d20_vals else float(np.median(np.linalg.norm(train_scaled, axis=1)))
    resid = df["resid_ens_mean"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["target_date_local"]).values.astype("datetime64[D]")
    stations = df["station_id"].to_numpy()
    mean_resid = np.full(len(df), np.nan, dtype=float)
    std_resid = np.full(len(df), np.nan, dtype=float)
    mean_dist = np.full(len(df), np.nan, dtype=float)
    count = np.zeros(len(df), dtype=float)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        station_dates = dates[idx]
        station_feat = scaled[idx]
        for pos, row_idx in enumerate(idx):
            cutoff = station_dates[pos] - np.timedelta64(ctx.truth_lag, "D")
            start_date = station_dates[pos] - np.timedelta64(365, "D")
            start = np.searchsorted(station_dates, start_date, side="left")
            end = np.searchsorted(station_dates, cutoff, side="right")
            if end <= start:
                continue
            cand_idx = idx[start:end]
            cand_feat = station_feat[start:end]
            dists = np.linalg.norm(cand_feat - station_feat[pos], axis=1)
            mask = dists <= d_thresh
            if not mask.any():
                continue
            vals = resid[cand_idx[mask]]
            mean_resid[row_idx] = float(np.mean(vals))
            std_resid[row_idx] = float(np.std(vals, ddof=0))
            mean_dist[row_idx] = float(np.mean(dists[mask]))
            count[row_idx] = float(np.sum(mask))
    _add_feature(features, formulas, f"knn_adapt_k_l{ctx.truth_lag}", count, "knn_adapt_k")
    _add_feature(features, formulas, f"knn_adapt_resid_mean_l{ctx.truth_lag}", mean_resid, "knn_adapt_resid_mean")
    _add_feature(features, formulas, f"knn_adapt_resid_std_l{ctx.truth_lag}", std_resid, "knn_adapt_resid_std")
    _add_feature(features, formulas, f"knn_adapt_dist_mean_l{ctx.truth_lag}", mean_dist, "knn_adapt_dist_mean")
    train_fitted.append(
        {
            "name": "knn_adaptive_threshold",
            "fit_on": "train",
            "features": feature_cols,
            "scaler": scaler_meta,
            "d_thresh": d_thresh,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e87(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    slope15 = tfl.rolling_slope(
        df["ens_std"], window=15, min_periods=_min_periods(15), lag=1, group_key=gk
    )
    slope60 = tfl.rolling_slope(
        df["ens_std"], window=60, min_periods=_min_periods(60), lag=1, group_key=gk
    )
    _add_feature(features, formulas, "ens_std_slope_15_l1", slope15, "slope(ens_std)")
    _add_feature(features, formulas, "ens_std_slope_60_l1", slope60, "slope(ens_std)")
    _add_feature(
        features,
        formulas,
        "ens_std_accel_15v60_l1",
        slope15 - slope60,
        "slope15 - slope60",
    )
    delta_last = df["ens_std"].groupby(gk).diff().groupby(gk).shift(1)
    _add_feature(
        features,
        formulas,
        "ens_std_delta_last_l1",
        delta_last,
        "ens_std(T-1)-ens_std(T-2)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e88(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    mean_syn = (df["gfs_tmax_f"] + df["nam_tmax_f"] + df["gefsatmosmean_tmax_f"]) / 3.0
    mean_meso = (df["hrrr_tmax_f"] + df["rap_tmax_f"] + df["nbm_tmax_f"]) / 3.0
    split = mean_meso - mean_syn
    split_rm30 = tfl.rolling_mean(
        split, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
    )
    split_rs30 = tfl.rolling_std(
        split, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
    )
    _add_feature(features, formulas, "split_meso_minus_syn", split, "mean_meso-mean_syn")
    _add_feature(features, formulas, "split_rm30_l1", split_rm30, "roll_mean(split)")
    _add_feature(features, formulas, "split_rs30_l1", split_rs30, "roll_std(split)")
    _add_feature(features, formulas, "split_dev_rm30", split - split_rm30, "split - split_rm30")
    split_sign = np.sign(split).fillna(0)
    _add_feature(
        features,
        formulas,
        "split_sign_streak_l1",
        tfl.streak_length(split_sign, lag=1, cap=60, group_key=gk),
        "streak_length(sign(split))",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e89(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    streaks = []
    for col in MODEL_COLS:
        drift = df[col] - df["ens_mean"]
        sign = np.where(drift.abs() < 0.1, 0.0, np.sign(drift))
        sign_series = pd.Series(sign, index=df.index)
        streak = tfl.streak_length(sign_series, lag=1, cap=60, group_key=gk)
        streaks.append(streak.to_numpy(dtype=float))
        _add_feature(
            features,
            formulas,
            f"drift_sign_streak_{col}_l1",
            streak,
            "streak_length(sign(drift))",
        )
        _add_feature(
            features,
            formulas,
            f"drift_sign_changes_{col}_rm30_l1",
            tfl.switch_count(
                sign_series, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
            ),
            "switch_count(sign(drift))",
        )
    streak_mat = np.vstack(streaks)
    _add_feature(
        features,
        formulas,
        "max_drift_streak_l1",
        np.nanmax(streak_mat, axis=0),
        "max(drift_sign_streak)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e90(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    ranks = tfl.rank_data(df, MODEL_COLS)
    rank_vals = ranks.to_numpy(dtype=float)
    kendall = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        for pos in range(1, len(idx)):
            prev = rank_vals[idx[pos - 1]]
            curr = rank_vals[idx[pos]]
            discord = 0
            for i in range(len(prev)):
                for j in range(i + 1, len(prev)):
                    if (prev[i] - prev[j]) * (curr[i] - curr[j]) < 0:
                        discord += 1
            kendall[idx[pos]] = float(discord)
    kendall_series = pd.Series(kendall, index=df.index)
    _add_feature(
        features,
        formulas,
        "kendall_dist_last_l1",
        kendall_series.groupby(gk).shift(1),
        "kendall_dist(T-1)",
    )
    _add_feature(
        features,
        formulas,
        "kendall_dist_mean_30_l1",
        tfl.rolling_mean(kendall_series, window=30, min_periods=_min_periods(30), lag=1, group_key=gk),
        "roll_mean(kendall_dist)",
    )
    _add_feature(
        features,
        formulas,
        "kendall_dist_std_30_l1",
        tfl.rolling_std(kendall_series, window=30, min_periods=_min_periods(30), lag=1, group_key=gk),
        "roll_std(kendall_dist)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e91(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    pairs = [
        ("rap_tmax_f", "nbm_tmax_f"),
        ("hrrr_tmax_f", "nbm_tmax_f"),
        ("nam_tmax_f", "nbm_tmax_f"),
        ("gfs_tmax_f", "nbm_tmax_f"),
        ("gefsatmosmean_tmax_f", "nbm_tmax_f"),
    ]
    for left, right in pairs:
        delta = df[left] - df[right]
        left_name = left.replace("_tmax_f", "")
        right_name = right.replace("_tmax_f", "")
        prefix = f"delta_{left_name}_minus_{right_name}"
        rm30 = tfl.rolling_mean(
            delta, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        )
        rs30 = tfl.rolling_std(
            delta, window=30, min_periods=_min_periods(30), lag=1, group_key=gk
        )
        slope15 = tfl.rolling_slope(
            delta, window=15, min_periods=_min_periods(15), lag=1, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            f"{prefix}_rm30_l1",
            rm30,
            "roll_mean(delta)",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
        _add_feature(
            features,
            formulas,
            f"{prefix}_rs30_l1",
            rs30,
            "roll_std(delta)",
            {"window": 30, "lag": 1, "min_periods": _min_periods(30)},
        )
        _add_feature(
            features,
            formulas,
            f"{prefix}_slope15_l1",
            slope15,
            "roll_slope(delta)",
            {"window": 15, "lag": 1, "min_periods": _min_periods(15)},
        )
        _add_feature(
            features,
            formulas,
            f"{prefix}_dev_rm30",
            delta - rm30,
            "delta - delta_rm30",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e92(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        bias = tfl.rolling_mean(
            resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        _add_feature(
            features,
            formulas,
            f"bias_{col}_rm60_l{lag}",
            bias,
            "roll_mean(resid)",
            {"window": 60, "lag": lag, "min_periods": _min_periods(60)},
        )
        _add_feature(
            features,
            formulas,
            f"{col}_corr_rm60_l{lag}",
            df[col] + bias,
            "forecast + bias_rm60",
        )
    bias_ens = tfl.rolling_mean(
        df["resid_ens_mean"],
        window=60,
        min_periods=_min_periods(60),
        lag=lag,
        group_key=gk,
    )
    _add_feature(
        features,
        formulas,
        f"bias_ensmean_rm60_l{lag}",
        bias_ens,
        "roll_mean(resid_ens_mean)",
    )
    _add_feature(
        features,
        formulas,
        f"ensmean_corr_rm60_l{lag}",
        df["ens_mean"] + bias_ens,
        "ens_mean + bias_ensmean",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e93(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    train_abs_err = ctx.train_df["resid_ens_mean"].abs()
    thr = float(train_abs_err.quantile(0.85))
    hard_train = (train_abs_err > thr).astype(int)
    feature_cols = MODEL_COLS + ["ens_std", "ens_range", "sin_doy", "cos_doy", "month"]
    X_train = ctx.train_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y_train = hard_train.to_numpy(dtype=int)
    if len(np.unique(y_train)) < 2:
        probs = np.full(len(df), float(y_train[0]), dtype=float)
        train_fitted.append(
            {
                "name": "hard_day_classifier_q85",
                "fit_on": "train",
                "description": "single-class fallback",
                "threshold": thr,
                "features": feature_cols,
                "probability": float(y_train[0]),
            }
        )
    else:
        clf = LogisticRegression(
            solver="liblinear", random_state=ctx.seed, max_iter=200
        )
        clf.fit(X_train_scaled, y_train)
        X_all = scaler.transform(df[feature_cols].to_numpy(dtype=float))
        probs = clf.predict_proba(X_all)[:, 1]
        train_fitted.append(
            {
                "name": "hard_day_classifier_q85",
                "fit_on": "train",
                "description": "logistic regression on forecast features",
                "threshold": thr,
                "features": feature_cols,
                "coef": clf.coef_.tolist(),
                "intercept": clf.intercept_.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            }
        )
    shrink = np.clip(probs, 0.0, 1.0)
    _add_feature(
        features,
        formulas,
        "p_hard_day_q85",
        probs,
        "P(|resid_ens_mean| > q85_train)",
    )
    _add_feature(
        features,
        formulas,
        "ens_shrink_mean_to_median",
        (1.0 - shrink) * df["ens_mean"] + shrink * df["ens_median"],
        "(1-p_hard)*ens_mean + p_hard*ens_median",
    )
    _add_feature(
        features,
        formulas,
        "spread_times_phard",
        df["ens_std"] * probs,
        "ens_std * p_hard",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e94(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    mae_frame = pd.DataFrame(index=df.index)
    bias_frame = pd.DataFrame(index=df.index)
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
        bias = tfl.rolling_mean(
            resid, window=60, min_periods=_min_periods(60), lag=lag, group_key=gk
        )
        mae_frame[col] = mae
        bias_frame[col] = bias
        _add_feature(
            features,
            formulas,
            f"mae_{col}_rm60_l{lag}",
            mae,
            "roll_mean(|resid|)",
        )
        _add_feature(
            features,
            formulas,
            f"bias_{col}_rm60_l{lag}",
            bias,
            "roll_mean(resid)",
        )
    weights = _rowwise_inverse_weights(mae_frame.to_numpy(dtype=float), eps=0.05)
    fcst = np.sum(weights * df[MODEL_COLS].to_numpy(dtype=float), axis=1)
    bias_weighted = np.sum(weights * bias_frame.to_numpy(dtype=float), axis=1)
    corrected = fcst + bias_weighted
    entropy = _rowwise_entropy(weights, eps=1e-9)
    w_max = np.max(weights, axis=1)
    _add_feature(
        features,
        formulas,
        f"fcst_wmean_invmae_rm60_l{lag}",
        fcst,
        "sum(w * model)",
    )
    _add_feature(
        features,
        formulas,
        f"bias_wmean_invmae_rm60_l{lag}",
        bias_weighted,
        "sum(w * bias)",
    )
    _add_feature(
        features,
        formulas,
        f"fcst_wmean_invmae_biascorr_rm60_l{lag}",
        corrected,
        "fcst_wmean + bias_wmean",
    )
    _add_feature(
        features,
        formulas,
        f"w_entropy_rm60_l{lag}",
        entropy,
        "entropy(weights)",
    )
    _add_feature(
        features,
        formulas,
        f"w_max_rm60_l{lag}",
        w_max,
        "max(weights)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e95(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        q10 = tfl.rolling_quantile(
            resid, window=60, min_periods=_min_periods(60), lag=lag, q=0.10, group_key=gk
        )
        q50 = tfl.rolling_quantile(
            resid, window=60, min_periods=_min_periods(60), lag=lag, q=0.50, group_key=gk
        )
        q90 = tfl.rolling_quantile(
            resid, window=60, min_periods=_min_periods(60), lag=lag, q=0.90, group_key=gk
        )
        iqr = q90 - q10
        asym = q10.abs() - q90.abs()
        _add_feature(
            features,
            formulas,
            f"resid_{col}_q10_rm60_l{lag}",
            q10,
            "rolling_quantile(resid, q=0.10)",
        )
        _add_feature(
            features,
            formulas,
            f"resid_{col}_q50_rm60_l{lag}",
            q50,
            "rolling_quantile(resid, q=0.50)",
        )
        _add_feature(
            features,
            formulas,
            f"resid_{col}_q90_rm60_l{lag}",
            q90,
            "rolling_quantile(resid, q=0.90)",
        )
        _add_feature(
            features,
            formulas,
            f"resid_{col}_iqr_rm60_l{lag}",
            iqr,
            "resid_q90 - resid_q10",
        )
        _add_feature(
            features,
            formulas,
            f"resid_{col}_asym_rm60_l{lag}",
            asym,
            "|q10| - |q90|",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e96(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    window = 60
    min_p = _min_periods(window)
    a_vals = np.full(len(df), np.nan, dtype=float)
    b_vals = np.full(len(df), np.nan, dtype=float)
    c_vals = np.full(len(df), np.nan, dtype=float)
    pred_vals = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            start = max(0, pos - window)
            end = pos - lag + 1
            window_idx = idx[start:end]
            if len(window_idx) < min_p:
                continue
            X = df.loc[window_idx, ["ens_std", "ens_mean"]].to_numpy(dtype=float)
            y = df.loc[window_idx, "resid_ens_mean"].to_numpy(dtype=float)
            b0, coeff = _fit_ridge_coeffs(X, y, l2=0.1)
            a_vals[row_idx] = b0
            b_vals[row_idx] = coeff[0]
            c_vals[row_idx] = coeff[1]
            pred_vals[row_idx] = b0 + coeff[0] * df.loc[row_idx, "ens_std"] + coeff[1] * df.loc[
                row_idx, "ens_mean"
            ]
    _add_feature(
        features,
        formulas,
        f"resid_reg_a_rm60_l{lag}",
        a_vals,
        "ridge_intercept",
        {"window": window, "lag": lag, "min_periods": min_p, "l2": 0.1},
    )
    _add_feature(
        features,
        formulas,
        f"resid_reg_b_rm60_l{lag}",
        b_vals,
        "ridge_coef_ens_std",
    )
    _add_feature(
        features,
        formulas,
        f"resid_reg_c_rm60_l{lag}",
        c_vals,
        "ridge_coef_ens_mean",
    )
    _add_feature(
        features,
        formulas,
        f"resid_pred_rm60_l{lag}",
        pred_vals,
        "a + b*ens_std + c*ens_mean",
    )
    _add_feature(
        features,
        formulas,
        f"ens_mean_residreg_corr_rm60_l{lag}",
        df["ens_mean"] + pred_vals,
        "ens_mean + resid_pred",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e97(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    delta = df["ens_mean"].groupby(gk).diff().fillna(0.0)
    obs_all = np.column_stack([df["ens_std"].to_numpy(dtype=float), delta.to_numpy(dtype=float)])
    p0 = np.full(len(df), np.nan, dtype=float)
    p1 = np.full(len(df), np.nan, dtype=float)
    state_argmax = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        train_mask = ctx.train_df["station_id"] == station
        train_index = ctx.train_df.loc[train_mask].index
        train_idx = df.index.get_indexer(train_index)
        train_idx = train_idx[train_idx >= 0]
        obs_train = obs_all[train_idx]
        if len(obs_train) < 10:
            params = None
            train_fitted.append(
                {
                    "name": "hmm_forecast_only",
                    "fit_on": "train",
                    "station_id": station,
                    "description": "insufficient data fallback",
                }
            )
        else:
            params = hmm_utils.fit_gaussian_hmm(
                obs_train, n_states=2, n_iters=10, seed=ctx.seed
            )
            train_fitted.append(
                {
                    "name": "hmm_forecast_only",
                    "fit_on": "train",
                    "station_id": station,
                    "pi": params.pi.tolist(),
                    "A": params.A.tolist(),
                    "means": params.means.tolist(),
                    "covars": params.covars.tolist(),
                }
            )
        if params is None:
            p0[idx[1:]] = 0.5
            p1[idx[1:]] = 0.5
            continue
        alpha = hmm_utils.forward_filter(obs_all[idx], params)
        if len(idx) > 1:
            p0[idx[1:]] = alpha[:-1, 0]
            p1[idx[1:]] = alpha[:-1, 1]
    state_argmax = np.where(np.isnan(p0), np.nan, (p1 > p0).astype(float))
    _add_feature(
        features,
        formulas,
        "p_state0_l1",
        p0,
        "P(state0 | obs<=T-1)",
    )
    _add_feature(
        features,
        formulas,
        "p_state1_l1",
        p1,
        "P(state1 | obs<=T-1)",
    )
    _add_feature(
        features,
        formulas,
        "hmm_state_argmax_l1",
        state_argmax,
        "argmax(p_state)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e98(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    abs_err = (ctx.train_df[MODEL_COLS].sub(ctx.train_df["actual_tmax_f"], axis=0)).abs()
    best_id = _argmin_model(abs_err, MODEL_COLS)
    label_map = {col: idx for idx, col in enumerate(MODEL_COLS)}
    y_train = best_id.map(label_map).to_numpy(dtype=int)
    feature_cols = MODEL_COLS + ["ens_std", "ens_range", "sin_doy", "cos_doy", "month"]
    X_train = ctx.train_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    probs_full = np.zeros((len(df), len(MODEL_COLS)), dtype=float)
    if len(np.unique(y_train)) < 2:
        cls = int(y_train[0]) if len(y_train) else 0
        probs_full[:, cls] = 1.0
        train_fitted.append(
            {
                "name": "best_model_classifier",
                "fit_on": "train",
                "description": "single-class fallback",
                "features": feature_cols,
                "class_index": cls,
            }
        )
    else:
        clf = LogisticRegression(
            solver="lbfgs", multi_class="multinomial", max_iter=400, random_state=ctx.seed
        )
        clf.fit(X_train_scaled, y_train)
        X_all = scaler.transform(df[feature_cols].to_numpy(dtype=float))
        probs = clf.predict_proba(X_all)
        for cls_idx, cls in enumerate(clf.classes_):
            if cls < len(MODEL_COLS):
                probs_full[:, cls] = probs[:, cls_idx]
        train_fitted.append(
            {
                "name": "best_model_classifier",
                "fit_on": "train",
                "description": "multinomial logistic regression",
                "features": feature_cols,
                "coef": clf.coef_.tolist(),
                "intercept": clf.intercept_.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            }
        )
    for i, col in enumerate(MODEL_COLS):
        _add_feature(
            features,
            formulas,
            f"p_best_{col}",
            probs_full[:, i],
            f"P(best_model == {col})",
        )
    expected = np.sum(probs_full * df[MODEL_COLS].to_numpy(dtype=float), axis=1)
    entropy = _rowwise_entropy(probs_full, eps=1e-9)
    _add_feature(
        features,
        formulas,
        "fcst_expected_bestprob",
        expected,
        "sum(p_best * model)",
    )
    _add_feature(
        features,
        formulas,
        "bestprob_entropy",
        entropy,
        "entropy(p_best)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e99(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    bias_maps = _seasonal_bias_maps(ctx.train_df, MODEL_COLS, "actual_tmax_f")
    ens_bias_map = _seasonal_bias_maps(ctx.train_df, ["ens_mean"], "actual_tmax_f")["ens_mean"]
    for col in MODEL_COLS:
        bias = _apply_seasonal_bias(df, bias_maps[col])
        _add_feature(
            features,
            formulas,
            f"bias_{col}_month",
            bias,
            "seasonal_bias_by_month",
        )
        _add_feature(
            features,
            formulas,
            f"{col}_seas_corr",
            df[col] + bias,
            "forecast + seasonal_bias",
        )
        nested = {}
        for (station, month), value in bias_maps[col]["map"].items():
            nested.setdefault(station, {})[int(month)] = float(value)
        train_fitted.append(
            {
                "name": f"bias_{col}_month",
                "fit_on": "train",
                "by_station": nested,
                "default": float(bias_maps[col]["default"]),
            }
        )
    bias_ens = _apply_seasonal_bias(df, ens_bias_map)
    _add_feature(
        features,
        formulas,
        "bias_ensmean_month",
        bias_ens,
        "seasonal_bias_by_month",
    )
    _add_feature(
        features,
        formulas,
        "ensmean_seas_corr",
        df["ens_mean"] + bias_ens,
        "ens_mean + seasonal_bias",
    )
    nested_ens = {}
    for (station, month), value in ens_bias_map["map"].items():
        nested_ens.setdefault(station, {})[int(month)] = float(value)
    train_fitted.append(
        {
            "name": "bias_ensmean_month",
            "fit_on": "train",
            "by_station": nested_ens,
            "default": float(ens_bias_map["default"]),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_e100(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    bias_maps = _seasonal_bias_maps(ctx.train_df, MODEL_COLS, "actual_tmax_f")
    seas_corr = []
    for col in MODEL_COLS:
        bias = _apply_seasonal_bias(df, bias_maps[col])
        corr = df[col] + bias
        seas_corr.append(corr.to_numpy(dtype=float))
        _add_feature(
            features,
            formulas,
            f"{col}_seas_corr",
            corr,
            "forecast + seasonal_bias",
        )
        nested = {}
        for (station, month), value in bias_maps[col]["map"].items():
            nested.setdefault(station, {})[int(month)] = float(value)
        train_fitted.append(
            {
                "name": f"bias_{col}_month",
                "fit_on": "train",
                "by_station": nested,
                "default": float(bias_maps[col]["default"]),
            }
        )
    seas_corr_mat = np.vstack(seas_corr).T
    mae_frame = pd.DataFrame(index=df.index)
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae = _rolling_abs_mean(resid, window=60, lag=lag, group_key=gk)
        mae_frame[col] = mae
        _add_feature(
            features,
            formulas,
            f"mae_{col}_rm60_l{lag}",
            mae,
            "roll_mean(|resid|)",
        )
    weights = _rowwise_inverse_weights(mae_frame.to_numpy(dtype=float), eps=0.05)
    fcst = np.sum(weights * seas_corr_mat, axis=1)
    entropy = _rowwise_entropy(weights, eps=1e-9)
    w_max = np.max(weights, axis=1)
    _add_feature(
        features,
        formulas,
        f"fcst_hybrid_seas_invmae_rm60_l{lag}",
        fcst,
        "sum(w * seasonal_corrected_forecast)",
    )
    _add_feature(
        features,
        formulas,
        f"w_entropy_rm60_l{lag}",
        entropy,
        "entropy(weights)",
    )
    _add_feature(
        features,
        formulas,
        f"w_max_rm60_l{lag}",
        w_max,
        "max(weights)",
    )
    _add_feature(
        features,
        formulas,
        "hybrid_minus_ensmean",
        fcst - df["ens_mean"].to_numpy(dtype=float),
        "hybrid_forecast - ens_mean",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _build_experiments_04() -> list[ExperimentDefinition]:
    return [
        ExperimentDefinition("E51", "Signed bias state + least-biased model identity", _exp_e51),
        ExperimentDefinition("E52", "RMSE skill state + best-RMSE selector", _exp_e52),
        ExperimentDefinition("E53", "Composite skill score (MAE + bias + RMSE)", _exp_e53),
        ExperimentDefinition("E54", "Skill momentum (rm15 vs rm60) per model", _exp_e54),
        ExperimentDefinition("E55", "Error volatility state (std/CV)", _exp_e55),
        ExperimentDefinition("E56", "EWMA(|resid|) skill state + best EWMA model", _exp_e56),
        ExperimentDefinition("E57", "Inverse-MAE weights + weight diagnostics", _exp_e57),
        ExperimentDefinition("E58", "DMA-style sequential weights", _exp_e58),
        ExperimentDefinition("E59", "Rolling ridge combo weights + forecast", _exp_e59),
        ExperimentDefinition("E60", "Rolling simplex weights + forecast", _exp_e60),
        ExperimentDefinition("E61", "Spread-regime conditional MAE + winner", _exp_e61),
        ExperimentDefinition("E62", "Temp-regime conditional bias + corrected forecasts", _exp_e62),
        ExperimentDefinition("E63", "Momentum-conditioned MAE/bias", _exp_e63),
        ExperimentDefinition("E64", "DOY-localized skill state", _exp_e64),
        ExperimentDefinition("E65", "4-regime residual bias/MAE for ensemble mean", _exp_e65),
        ExperimentDefinition("E66", "Regime-specific best model selector", _exp_e66),
        ExperimentDefinition("E67", "CUSUM drift stats on residuals", _exp_e67),
        ExperimentDefinition("E68", "Page-Hinkley drift statistic", _exp_e68),
        ExperimentDefinition("E69", "Residual quantile shift (15 vs 60)", _exp_e69),
        ExperimentDefinition("E70", "EWMA bias drift index", _exp_e70),
        ExperimentDefinition("E71", "Truth-based HMM state probabilities", _exp_e71),
        ExperimentDefinition("E72", "Lagged actual Tmax state", _exp_e72),
        ExperimentDefinition("E73", "Observed anomaly features (climatology)", _exp_e73),
        ExperimentDefinition("E74", "Realized trend slopes", _exp_e74),
        ExperimentDefinition("E75", "Realized volatility regime", _exp_e75),
        ExperimentDefinition("E76", "Anomaly-regime conditional MAE", _exp_e76),
        ExperimentDefinition("E77", "ACF1 of ens_std/ens_mean", _exp_e77),
        ExperimentDefinition("E78", "ACF1 of model drift series", _exp_e78),
        ExperimentDefinition("E79", "Pairwise forecast correlation structure", _exp_e79),
        ExperimentDefinition("E80", "PCA on forecast-change vectors", _exp_e80),
        ExperimentDefinition("E81", "Mahalanobis analog residual correction", _exp_e81),
        ExperimentDefinition("E82", "Time-decayed kernel analog residual", _exp_e82),
        ExperimentDefinition("E83", "Analog local regression residual correction", _exp_e83),
        ExperimentDefinition("E84", "Analog residual quantiles", _exp_e84),
        ExperimentDefinition("E85", "Regime-restricted analogs", _exp_e85),
        ExperimentDefinition("E86", "Adaptive-K analog residuals", _exp_e86),
        ExperimentDefinition("E87", "Spread trend/acceleration", _exp_e87),
        ExperimentDefinition("E88", "Synoptic vs mesoscale split persistence", _exp_e88),
        ExperimentDefinition("E89", "Per-model drift sign persistence", _exp_e89),
        ExperimentDefinition("E90", "Kendall rank-order instability", _exp_e90),
        ExperimentDefinition("E91", "Pairwise delta time structure", _exp_e91),
        ExperimentDefinition("E92", "Bias-corrected model forecasts", _exp_e92),
        ExperimentDefinition("E93", "Hard-day shrinkage to median", _exp_e93),
        ExperimentDefinition("E94", "Inverse-MAE weights + bias correction", _exp_e94),
        ExperimentDefinition("E95", "Residual quantiles per model", _exp_e95),
        ExperimentDefinition("E96", "Rolling residual regression on spread/level", _exp_e96),
        ExperimentDefinition("E97", "Forecast-only HMM state probabilities", _exp_e97),
        ExperimentDefinition("E98", "Best-model probability classifier", _exp_e98),
        ExperimentDefinition("E99", "Seasonal bias lookup by month", _exp_e99),
        ExperimentDefinition("E100", "Seasonal-corrected inverse-MAE blend", _exp_e100),
    ]


def _build_experiments() -> list[ExperimentDefinition]:
    return [
        ExperimentDefinition(
            "E01",
            "Rolling mean/median level (7/30/60) with deviation",
            _exp_e01,
        ),
        ExperimentDefinition(
            "E02",
            "EWMA mean/median baseline and shock features",
            _exp_e02,
        ),
        ExperimentDefinition(
            "E03",
            "Rolling slope of ens_mean (15/60) + acceleration",
            _exp_e03,
        ),
        ExperimentDefinition(
            "E04",
            "Trailing quantiles of ens_mean + percentile rank",
            _exp_e04,
        ),
        ExperimentDefinition(
            "E05",
            "Rolling level/volatility of disagreement metrics",
            _exp_e05,
        ),
        ExperimentDefinition(
            "E06",
            "EWMA spread baseline + shock features",
            _exp_e06,
        ),
        ExperimentDefinition(
            "E07",
            "Spread z-score anomalies vs 60d baseline",
            _exp_e07,
        ),
        ExperimentDefinition(
            "E08",
            "High-spread burstiness counts + streak length",
            _exp_e08,
        ),
        ExperimentDefinition(
            "E09",
            "Per-model forecast volatility (15/60)",
            _exp_e09,
        ),
        ExperimentDefinition(
            "E10",
            "Rolling mean/std of model drift vs ensemble",
            _exp_e10,
        ),
        ExperimentDefinition(
            "E11",
            "Rank stability stats + top identity entropy",
            _exp_e11,
        ),
        ExperimentDefinition(
            "E12",
            "Rolling mean/std of model day-to-day changes",
            _exp_e12,
        ),
        ExperimentDefinition(
            "E13",
            "Ensemble momentum and persistence features",
            _exp_e13,
        ),
        ExperimentDefinition(
            "E14",
            "Dispersion of model trend signals",
            _exp_e14,
        ),
        ExperimentDefinition(
            "E15",
            "Reversal/turning point structure",
            _exp_e15,
        ),
        ExperimentDefinition(
            "E16",
            "Scaled momentum normalized by variability",
            _exp_e16,
        ),
        ExperimentDefinition(
            "E17",
            "Short-minus-long level shift (7 vs 60)",
            _exp_e17,
        ),
        ExperimentDefinition(
            "E18",
            "Short-minus-long spread shift (7 vs 60)",
            _exp_e18,
        ),
        ExperimentDefinition(
            "E19",
            "Per-model drift shift vs long baseline",
            _exp_e19,
        ),
        ExperimentDefinition(
            "E20",
            "Disagreement shape ratio evolution",
            _exp_e20,
        ),
        ExperimentDefinition(
            "E21",
            "Rolling bias of ensemble mean/median (truth lag)",
            _exp_e21,
        ),
        ExperimentDefinition(
            "E22",
            "Rolling MAE/RMSE of ensemble mean/median",
            _exp_e22,
        ),
        ExperimentDefinition(
            "E23",
            "Per-model rolling bias/MAE (dynamic reliability)",
            _exp_e23,
        ),
        ExperimentDefinition(
            "E24",
            "Relative skill vs ensemble + best model flag",
            _exp_e24,
        ),
        ExperimentDefinition(
            "E25",
            "Rolling residual quantiles + asymmetry",
            _exp_e25,
        ),
        ExperimentDefinition(
            "E26",
            "Conditional bias by temperature regime",
            _exp_e26,
        ),
        ExperimentDefinition(
            "E27",
            "Conditional bias by spread regime",
            _exp_e27,
        ),
        ExperimentDefinition(
            "E28",
            "EWMA residual bias correction",
            _exp_e28,
        ),
        ExperimentDefinition(
            "E29",
            "Rolling linear correction y ~ a + b*ens_mean",
            _exp_e29,
        ),
        ExperimentDefinition(
            "E30",
            "Dynamic model switching + switch frequency",
            _exp_e30,
        ),
        ExperimentDefinition(
            "E31",
            "Rolling corr of model residuals vs ensemble",
            _exp_e31,
        ),
        ExperimentDefinition(
            "E32",
            "Rolling regression |resid| vs spread",
            _exp_e32,
        ),
        ExperimentDefinition(
            "E33",
            "KMeans regime cluster id + stability",
            _exp_e33,
        ),
        ExperimentDefinition(
            "E34",
            "PCA disagreement regime + rolling stats",
            _exp_e34,
        ),
        ExperimentDefinition(
            "E35",
            "Predicted hard-day probability",
            _exp_e35,
        ),
        ExperimentDefinition(
            "E36",
            "Spread regime transition rates",
            _exp_e36,
        ),
        ExperimentDefinition(
            "E37",
            "Frequency of being top/bottom model",
            _exp_e37,
        ),
        ExperimentDefinition(
            "E38",
            "Outlier magnitude persistence",
            _exp_e38,
        ),
        ExperimentDefinition(
            "E39",
            "Dominant outlier identity counts",
            _exp_e39,
        ),
        ExperimentDefinition(
            "E40",
            "Outlier stickiness + switch rate",
            _exp_e40,
        ),
        ExperimentDefinition(
            "E41",
            "KNN analog mean/std of y",
            _exp_e41,
        ),
        ExperimentDefinition(
            "E42",
            "KNN analog residual correction",
            _exp_e42,
        ),
        ExperimentDefinition(
            "E43",
            "KNN analog hard-day probability",
            _exp_e43,
        ),
        ExperimentDefinition(
            "E44",
            "Kernel-weighted analog residual",
            _exp_e44,
        ),
        ExperimentDefinition(
            "E45",
            "Extreme spread memory (days since + counts)",
            _exp_e45,
        ),
        ExperimentDefinition(
            "E46",
            "Forecast-extreme memory (hi/lo)",
            _exp_e46,
        ),
        ExperimentDefinition(
            "E47",
            "Extreme outlierness memory",
            _exp_e47,
        ),
        ExperimentDefinition(
            "E48",
            "Extreme realized error memory",
            _exp_e48,
        ),
        ExperimentDefinition(
            "E49",
            "Cold/warm bust memory",
            _exp_e49,
        ),
        ExperimentDefinition(
            "E50",
            "Residual sign streak and change rate",
            _exp_e50,
        ),
    ] + _build_experiments_04()


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
