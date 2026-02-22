"""Time-structured feature sweep (E01-E100)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone, date
from pathlib import Path
from itertools import permutations
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge, QuantileRegressor, LinearRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize, nnls
from scipy.stats import genpareto, t as student_t
from scipy.special import logsumexp
import yaml

from weather_ml import artifacts
from weather_ml import config as config_module
from weather_ml import dataset
from weather_ml import distribution
from weather_ml import derived_features
from weather_ml import hmm_utils
from weather_ml import models_mean
from weather_ml import rs_moe
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
EXTRA_BASE_COLS: list[str] = []
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
    artifact_writers: list[Callable[[Path], list[Path]]] = field(default_factory=list)


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
    _apply_model_cols(config)

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
    if pre_split.train_df.empty or pre_split.test_df.empty:
        raise ValueError("Train/test split is empty.")
    df = _impute_base_columns(df, pre_split.train_df)
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
        json.dumps(_sanitize_for_yaml(sweep_payload), indent=2, sort_keys=True),
        encoding="utf-8",
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
    # When sweeping across multiple stations in a single run, include explicit station identity
    # features so tree models can learn station-specific offsets/interactions.
    #
    # This is only added when there is more than one station present; for single-station sweeps,
    # station onehots would be constant and add noise.
    stations = sorted(df["station_id"].dropna().unique().tolist())
    if len(stations) > 1:
        for station_id in stations:
            col = f"station_is_{station_id}"
            df[col] = (df["station_id"] == station_id).astype(int)
    if "actual_tmax_f" in df.columns:
        df["resid_ens_mean"] = df["actual_tmax_f"] - df["ens_mean"]
        df["resid_ens_median"] = df["actual_tmax_f"] - df["ens_median"]
    return df


def _base_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = list(MODEL_COLS)
    if SPREAD_COL in df.columns:
        cols.append(SPREAD_COL)
    cols.extend(EXTRA_BASE_COLS)
    cols.extend(CALENDAR_COLS)
    station_cols = sorted([col for col in df.columns if col.startswith("station_is_")])
    cols.extend(station_cols)
    return cols


def _min_periods(window: int) -> int:
    return int(np.ceil(window * 0.7))


def _formula_entry(name: str, formula: str, params: dict | None = None) -> dict:
    payload = {"name": name, "formula": formula}
    if params:
        payload["params"] = params
    return payload


def _short_dataset_dir_name(dataset_id: str) -> str:
    return dataset_id[:12]


def _impute_base_columns(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [col for col in MODEL_COLS if col in df.columns]
    if SPREAD_COL in df.columns:
        cols.append(SPREAD_COL)
    cols.extend([col for col in EXTRA_BASE_COLS if col in df.columns])
    if not cols:
        return df
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    medians = train_df[cols].median(numeric_only=True).fillna(0.0)
    missing_before = df[cols].isna().sum().to_dict()
    for col in cols:
        df[col] = df[col].fillna(float(medians[col]))
    missing_after = df[cols].isna().sum().to_dict()
    if any(count > 0 for count in missing_before.values()):
        LOGGER.info(
            "Imputed missing base columns using train medians. before=%s after=%s",
            missing_before,
            missing_after,
        )
    return df


def _apply_model_cols(config) -> None:
    global MODEL_COLS, EXTRA_BASE_COLS
    base_features = list(config.features.base_features)
    model_cols = [col for col in base_features if col.endswith("_tmax_f")]
    mos_tmax_cols = [
        col for col in base_features if col in ("gfs_n_x_max", "nam_n_x_max")
    ]
    model_cols.extend([col for col in mos_tmax_cols if col not in model_cols])
    if not model_cols:
        raise ValueError("features.base_features must include at least one *_tmax_f column.")
    MODEL_COLS = model_cols
    EXTRA_BASE_COLS = [
        col for col in base_features if col not in MODEL_COLS and col != SPREAD_COL
    ]
    LOGGER.info("Time feature sweep model columns: %s", MODEL_COLS)
    if EXTRA_BASE_COLS:
        LOGGER.info("Time feature sweep extra base columns: %s", EXTRA_BASE_COLS)


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

    run_config = config
    if experiment_id.upper() == "EX201" and config.mean_model.type != "rs_moe":
        run_config = replace(
            config,
            mean_model=config_module.MeanModelStrategyConfig(type="rs_moe"),
        )

    rs_moe_fit: rs_moe.RsMoeFitResult | None = None
    rs_moe_train_out: dict[str, np.ndarray] | None = None
    rs_moe_val_out: dict[str, np.ndarray] | None = None
    rs_moe_test_out: dict[str, np.ndarray] | None = None

    if run_config.mean_model.type == "rs_moe":
        model_name = "rs_moe"
        best_params: dict[str, object] = {}
        model_cols_for_labeler = [col for col in MODEL_COLS if col in context.df.columns]
        if not model_cols_for_labeler:
            raise ValueError("RS-MoE requires at least one forecast model column for regime labeling.")

        labeler_cfg = rs_moe.BustRegimeLabelerConfig(
            type=run_config.rs_moe.regime_labeler.type,
            residual_threshold_f=run_config.rs_moe.regime_labeler.residual_threshold_f,
            baseline_pred_source=run_config.rs_moe.regime_labeler.baseline_pred_source,
        )
        oof_cfg = rs_moe.OofGatingConfig(
            enabled=run_config.rs_moe.oof_gating.enabled,
            method=run_config.rs_moe.oof_gating.method,
            n_folds=run_config.rs_moe.oof_gating.n_folds,
            burnin_fraction=run_config.rs_moe.oof_gating.burnin_fraction,
            min_rows_per_fold=run_config.rs_moe.oof_gating.min_rows_per_fold,
            weight_floor=run_config.rs_moe.oof_gating.weight_floor,
            random_seed=run_config.rs_moe.oof_gating.random_seed,
        )
        bounds = tuple(run_config.rs_moe.gate_calibration.temperature_bounds)
        if len(bounds) != 2:
            raise ValueError("rs_moe.gate_calibration.temperature_bounds must have 2 values.")
        gate_cal_cfg = rs_moe.GateCalibrationConfig(
            method=run_config.rs_moe.gate_calibration.method,
            temperature_init=run_config.rs_moe.gate_calibration.temperature_init,
            temperature_bounds=(float(bounds[0]), float(bounds[1])),
            optimizer=run_config.rs_moe.gate_calibration.optimizer,
            max_iter=run_config.rs_moe.gate_calibration.max_iter,
            tol=run_config.rs_moe.gate_calibration.tol,
        )
        experts_cfg = rs_moe.ExpertsConfig(
            library=run_config.rs_moe.experts.library,
            objective_variant=run_config.rs_moe.experts.objective_variant,
            absoluteerror_params=run_config.rs_moe.experts.absoluteerror_params,
            quantile_median_params=run_config.rs_moe.experts.quantile_median_params,
        )

        rs_moe_fit = rs_moe.train_rs_moe(
            train_df=context.train_df,
            X_train=X_train,
            feature_names=feature_columns,
            target_col="actual_tmax_f",
            model_cols_for_labeler=model_cols_for_labeler,
            regimes=run_config.rs_moe.regimes,
            regime_labeler=labeler_cfg,
            oof_gating=oof_cfg,
            gate_model_library=run_config.rs_moe.gate_model.library,
            gate_model_params=run_config.rs_moe.gate_model.params,
            gate_calibration=gate_cal_cfg,
            experts=experts_cfg,
        )
        model_full = rs_moe_fit.model

        rs_moe_train_out = model_full.predict_components(X_train)
        mu_train = rs_moe_train_out["mu_hat"]
        if len(X_val):
            rs_moe_val_out = model_full.predict_components(X_val)
            mu_val = rs_moe_val_out["mu_hat"]
        else:
            mu_val = np.array([])
        rs_moe_test_out = model_full.predict_components(X_test)
        mu_test = rs_moe_test_out["mu_hat"]
    else:
        model_name = run_config.models.mean.primary
        base_model = models_mean.get_mean_model(model_name, seed=context.seed)
        fixed_params = _fixed_params(run_config.models.mean.param_grid.get(model_name, {}))
        if allow_tuning:
            tuned = models_mean.tune_model_timecv(
                base_model,
                X_train,
                y_train,
                cv_splits,
                run_config.models.mean.param_grid.get(model_name, {}),
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

    if run_config.mean_model.type == "rs_moe":
        train_full_df = (
            pd.concat([context.train_df, context.val_df], ignore_index=True)
            if not context.val_df.empty
            else context.train_df.copy()
        )
        full_features = pd.concat([train_features, val_features], ignore_index=True)

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
    derived_meta = _sanitize_for_yaml(derived_meta)

    config_snapshot = _sanitize_for_yaml(
        _config_snapshot(
            run_config,
            experiment_id=experiment_id,
            description=description,
            feature_columns=feature_columns,
            derived_meta=derived_meta,
        )
    )
    config_path = run_dir / "config_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    dataset_id = artifacts.compute_dataset_id(
        Path(run_config.data.csv_path),
        run_config.data.dataset_schema_version,
        {"experiment_id": experiment_id, "feature_columns": feature_columns},
    )
    dataset_dir = sweep_root / "datasets" / _short_dataset_dir_name(dataset_id)
    all_df = pd.concat(
        [context.train_df, context.val_df, context.test_df], ignore_index=True
    )
    data_parquet = artifacts.snapshot_to_parquet(all_df, dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    metadata = _build_dataset_metadata(
        context.train_df, context.val_df, context.test_df, dataset_id, run_config
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

    rs_moe_extra_paths: list[Path] = []
    rs_moe_sections: dict[str, object] | None = None
    if run_config.mean_model.type == "rs_moe":
        if rs_moe_fit is None:
            raise RuntimeError("RS-MoE enabled but fit result missing.")

        gate_calibration_payload = {
            "method": "temperature_scaling",
            "temperature": float(rs_moe_fit.temperature),
            "config": asdict(run_config.rs_moe.gate_calibration),
        }
        model_full.save(run_dir, gate_calibration_payload=gate_calibration_payload)

        gate_model_path = run_dir / "gate_model.cbm"
        gate_calibration_path = run_dir / "gate_calibration.json"
        expert_cool_path = run_dir / "expert_cool_model.joblib"
        expert_normal_path = run_dir / "expert_normal_model.joblib"
        expert_warm_path = run_dir / "expert_warm_model.joblib"
        rs_moe_extra_paths.extend(
            [
                gate_model_path,
                gate_calibration_path,
                expert_cool_path,
                expert_normal_path,
                expert_warm_path,
            ]
        )

        oof_logits_path, oof_probs_path = rs_moe.write_oof_gate_artifacts(
            run_dir,
            train_df=context.train_df,
            oof_logits=rs_moe_fit.oof_logits,
            oof_is_model_based=rs_moe_fit.oof_is_model_based,
            y_regime=rs_moe_fit.y_regime_train,
            oof_probs=rs_moe_fit.oof_probs,
            oof_probs_smoothed=rs_moe_fit.oof_probs_smoothed,
        )
        rs_moe_extra_paths.extend([oof_logits_path, oof_probs_path])

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    feature_importance = _feature_importance(
        model_full, feature_columns, full_features.to_numpy(dtype=float)
    )
    report_path = run_dir / "report.md"
    if run_config.mean_model.type == "rs_moe":
        model_cols_for_labeler = [col for col in MODEL_COLS if col in context.df.columns]
        labeler_cfg = rs_moe.BustRegimeLabelerConfig(
            type=run_config.rs_moe.regime_labeler.type,
            residual_threshold_f=run_config.rs_moe.regime_labeler.residual_threshold_f,
            baseline_pred_source=run_config.rs_moe.regime_labeler.baseline_pred_source,
        )
        labeler = rs_moe.BustRegimeLabeler(
            labeler_cfg,
            model_cols=model_cols_for_labeler,
            target_col="actual_tmax_f",
        ).fit(context.train_df)
        y_regime_train = labeler.transform(context.train_df)
        y_regime_val = (
            labeler.transform(context.val_df) if not context.val_df.empty else np.array([], dtype=int)
        )
        y_regime_test = labeler.transform(context.test_df)

        def _prevalence(y_int: np.ndarray) -> dict:
            counts = np.bincount(y_int.astype(int), minlength=3)
            total = int(np.sum(counts))
            fracs = counts / total if total > 0 else np.array([0.0, 0.0, 0.0], dtype=float)
            return {
                "n": total,
                "counts": {
                    "cool": int(counts[0]),
                    "normal": int(counts[1]),
                    "warm": int(counts[2]),
                },
                "fractions": {
                    "cool": float(fracs[0]),
                    "normal": float(fracs[1]),
                    "warm": float(fracs[2]),
                },
            }

        gate_diag: dict[str, object]
        if rs_moe_val_out is not None and len(y_regime_val):
            p_val = rs_moe_val_out["p"]
            y_pred = np.argmax(p_val, axis=1).astype(int)
            cm = rs_moe.confusion_matrix_3(y_regime_val, y_pred)
            gate_diag = {
                "multiclass_logloss_nll": rs_moe.multiclass_nll(p_val, y_regime_val),
                "accuracy": float(np.mean(y_pred == y_regime_val)) if len(y_regime_val) else 0.0,
                "per_class_precision_recall": {
                    "cool": rs_moe.precision_recall_from_cm(cm)["0"],
                    "normal": rs_moe.precision_recall_from_cm(cm)["1"],
                    "warm": rs_moe.precision_recall_from_cm(cm)["2"],
                },
                "confusion_matrix": cm.tolist(),
                "class_prevalence": {
                    "train": _prevalence(y_regime_train),
                    "validation": _prevalence(y_regime_val),
                    "test": _prevalence(y_regime_test),
                },
                "temperature": float(rs_moe_fit.temperature) if rs_moe_fit else None,
                "avg_max_prob": rs_moe.average_max_prob(p_val),
                "avg_entropy": rs_moe.average_entropy(p_val),
            }
        else:
            gate_diag = {
                "note": "validation split empty; gate diagnostics unavailable",
                "class_prevalence": {
                    "train": _prevalence(y_regime_train),
                    "validation": _prevalence(y_regime_val),
                    "test": _prevalence(y_regime_test),
                },
                "temperature": float(rs_moe_fit.temperature) if rs_moe_fit else None,
            }

        expert_diag: dict[str, object]
        if rs_moe_val_out is not None and len(y_val):
            p_val = rs_moe_val_out["p"]
            expert_diag = {
                "cool": {
                    "weighted_mae": rs_moe.weighted_mae(y_val, rs_moe_val_out["mu_cool"], p_val[:, 0]),
                    "unweighted_mae": float(np.mean(np.abs(y_val - rs_moe_val_out["mu_cool"]))),
                    "top_features": rs_moe.feature_importance_top_k_xgb(
                        model_full.expert_cool, feature_columns, k=20
                    ),
                },
                "normal": {
                    "weighted_mae": rs_moe.weighted_mae(y_val, rs_moe_val_out["mu_normal"], p_val[:, 1]),
                    "unweighted_mae": float(np.mean(np.abs(y_val - rs_moe_val_out["mu_normal"]))),
                    "top_features": rs_moe.feature_importance_top_k_xgb(
                        model_full.expert_normal, feature_columns, k=20
                    ),
                },
                "warm": {
                    "weighted_mae": rs_moe.weighted_mae(y_val, rs_moe_val_out["mu_warm"], p_val[:, 2]),
                    "unweighted_mae": float(np.mean(np.abs(y_val - rs_moe_val_out["mu_warm"]))),
                    "top_features": rs_moe.feature_importance_top_k_xgb(
                        model_full.expert_warm, feature_columns, k=20
                    ),
                },
            }
        else:
            expert_diag = {"note": "validation split empty; expert diagnostics unavailable"}

        regime_mae: dict[str, object] = {}
        if len(y_test):
            for name, cls in (("cool", 0), ("normal", 1), ("warm", 2)):
                mask = y_regime_test == cls
                regime_mae[f"test_mae_given_true_{name}"] = float(
                    np.mean(np.abs(mu_test[mask] - y_test[mask]))
                ) if int(np.sum(mask)) > 0 else None
                regime_mae[f"test_n_true_{name}"] = int(np.sum(mask))

        rs_moe_sections = {
            "summary": {
                "regimes": list(run_config.rs_moe.regimes),
                "gate": {
                    "library": run_config.rs_moe.gate_model.library,
                    "params": run_config.rs_moe.gate_model.params,
                },
                "experts": {
                    "library": run_config.rs_moe.experts.library,
                    "objective_variant": run_config.rs_moe.experts.objective_variant,
                    "absoluteerror_params": run_config.rs_moe.experts.absoluteerror_params,
                    "quantile_median_params": run_config.rs_moe.experts.quantile_median_params,
                },
                "oof_gating": {
                    "method": run_config.rs_moe.oof_gating.method,
                    "n_folds": run_config.rs_moe.oof_gating.n_folds,
                    "burnin_fraction": run_config.rs_moe.oof_gating.burnin_fraction,
                    "weight_floor": run_config.rs_moe.oof_gating.weight_floor,
                },
                "calibrated_temperature": float(rs_moe_fit.temperature) if rs_moe_fit else None,
            },
            "gate_diagnostics": gate_diag,
            "expert_diagnostics": expert_diag,
            "mixture_metrics_summary": metrics_summary,
            "regime_stratified_mae": regime_mae,
        }

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
        rs_moe_sections=rs_moe_sections,
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
    pmf = _build_pmf(mu_test, sigma_test, run_config)
    bin_probs = _build_bin_probs(pmf, run_config)
    predictions_path = run_dir / "predictions_test.parquet"
    extra_cols: dict[str, object] = {}
    if run_config.mean_model.type == "rs_moe":
        if rs_moe_test_out is None or rs_moe_fit is None:
            raise RuntimeError("RS-MoE enabled but test components missing.")
        extra_cols.update(
            {
                "p_cool": rs_moe_test_out["p"][:, 0],
                "p_normal": rs_moe_test_out["p"][:, 1],
                "p_warm": rs_moe_test_out["p"][:, 2],
                "mu_cool": rs_moe_test_out["mu_cool"],
                "mu_normal": rs_moe_test_out["mu_normal"],
                "mu_warm": rs_moe_test_out["mu_warm"],
                "gate_temperature": float(rs_moe_fit.temperature),
                "model_type": "rs_moe",
            }
        )
    if experiment_id.upper() == "EX207" and "y0_ridge" in test_features.columns:
        extra_cols["y0_ridge"] = test_features["y0_ridge"].to_numpy(dtype=float)
    if experiment_id.upper() == "EX209" and "p_bust" in test_features.columns:
        extra_cols["p_bust"] = test_features["p_bust"].to_numpy(dtype=float)
    _write_predictions(
        predictions_path,
        context.test_df,
        mu_test,
        sigma_test,
        pmf,
        bin_probs,
        run_config,
        extra_cols=extra_cols or None,
    )

    experiment_meta = {
        "experiment_id": experiment_id,
        "description": description,
        "base_features": base_cols,
        "derived_features": derived_meta,
    }
    if run_config.mean_model.type == "rs_moe":
        experiment_meta["mean_model"] = {"type": "rs_moe"}
        experiment_meta["regime_labeler"] = {"type": run_config.rs_moe.regime_labeler.type}
        experiment_meta["rs_moe"] = asdict(run_config.rs_moe)
        experiment_meta["feature_columns"] = feature_columns
    experiment_meta = _sanitize_for_yaml(experiment_meta)
    (run_dir / "experiment_meta.json").write_text(
        json.dumps(experiment_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "experiment_feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    extra_artifact_paths: list[Path] = []
    for writer in derived.artifact_writers:
        extra_artifact_paths.extend(writer(run_dir))

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
    hash_paths.extend(rs_moe_extra_paths)
    hash_paths.extend(extra_artifact_paths)
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
    extra_cols: dict[str, object] | None = None,
) -> None:
    records = df[["station_id", "target_date_local", "asof_utc"]].copy()
    records["mu_hat_f"] = mu
    records["sigma_hat_f"] = sigma
    if extra_cols:
        for key, value in extra_cols.items():
            records[key] = value
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


def _sanitize_for_yaml(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_sanitize_for_yaml(item) for item in value.tolist()]
    if isinstance(value, (datetime, date)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            safe_key = _sanitize_for_yaml(key)
            if isinstance(safe_key, (list, dict)):
                safe_key = json.dumps(safe_key, sort_keys=True)
            elif safe_key is not None and not isinstance(
                safe_key, (str, int, float, bool)
            ):
                safe_key = str(safe_key)
            sanitized[safe_key] = _sanitize_for_yaml(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_yaml(item) for item in value]
    return str(value)


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
    numer_arr = np.asarray(numer, dtype=float)
    denom_arr = np.asarray(denom, dtype=float)
    output = np.full_like(numer_arr, default, dtype=float)
    mask = np.isfinite(numer_arr) & np.isfinite(denom_arr) & (denom_arr != 0)
    np.divide(numer_arr, denom_arr, out=output, where=mask)
    return output


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals, axis=axis, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return exp_vals / denom


def _spread_from_stats(df: pd.DataFrame, stats: dict[str, np.ndarray]) -> np.ndarray:
    if SPREAD_COL in df.columns:
        return df[SPREAD_COL].to_numpy(dtype=float)
    return stats["std"]


def _mean_available(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arrays = [df[col].to_numpy(dtype=float) for col in cols if col in df.columns]
    if not arrays:
        return np.full(len(df), np.nan, dtype=float)
    stacked = np.column_stack(arrays)
    return np.nanmean(stacked, axis=1)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler.mean_.copy(), scaler.scale_.copy()


def _standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale_safe = np.where(scale == 0.0, 1.0, scale)
    return (X - mean) / scale_safe


def _fit_ridge_linear(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    Xs, mean, scale = _standardize_fit(X)
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    return float(model.intercept_), model.coef_.copy(), mean, scale


def _fit_quantile_linear(
    X: np.ndarray, y: np.ndarray, quantile: float, alpha: float = 0.0
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    Xs, mean, scale = _standardize_fit(X)
    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
    model.fit(Xs, y)
    return float(model.intercept_), model.coef_.copy(), mean, scale


def _predict_linear(
    X: np.ndarray, intercept: float, coef: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    Xs = _standardize_apply(X, mean, scale)
    return intercept + Xs @ coef


def _fit_nnls_scale(spread: np.ndarray, abs_err: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(spread), spread])
    coeff, _ = nnls(X, abs_err)
    return float(coeff[0]), float(coeff[1])


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    w_sorted = np.where(np.isfinite(w_sorted), w_sorted, 0.0)
    total = float(np.sum(w_sorted))
    if total <= 0:
        return float(np.median(v_sorted))
    cum = np.cumsum(w_sorted) / total
    idx = int(np.searchsorted(cum, 0.5))
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])


def _rowwise_weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.full(values.shape[0], np.nan, dtype=float)
    for idx in range(values.shape[0]):
        output[idx] = _weighted_median(values[idx], weights[idx])
    return output


def _gap2(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    sorted_vals = np.sort(values)
    gaps = np.diff(sorted_vals)
    if gaps.size == 0:
        return 0.0
    if gaps.size == 1:
        return float(gaps[0])
    return float(np.partition(gaps, -2)[-2])


def _theil_sen_slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            denom = j - i
            if denom == 0:
                continue
            slopes.append((values[j] - values[i]) / denom)
    if not slopes:
        return 0.0
    return float(np.median(slopes))


def _fit_simplex_lad(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    sin_doy: np.ndarray | None = None,
    cos_doy: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float]:
    n_models = X.shape[1]
    w0 = np.full(n_models, 1.0 / n_models, dtype=float)
    x0 = np.concatenate([w0, np.array([0.0, 0.0, 0.0], dtype=float)])

    def _objective(params: np.ndarray) -> float:
        w = params[:n_models]
        k0, k1, k2 = params[n_models:]
        pred = k0
        if sin_doy is not None:
            pred = pred + k1 * sin_doy
        if cos_doy is not None:
            pred = pred + k2 * cos_doy
        pred = pred + X @ w
        err = np.abs(y - pred)
        penalty = ridge * np.sum((w - 1.0 / n_models) ** 2)
        return float(np.sum(err) + penalty)

    constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n_models]) - 1.0}]
    bounds = [(0.0, 1.0)] * n_models + [(None, None), (None, None), (None, None)]
    result = minimize(_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    params = result.x if result.success else x0
    weights = params[:n_models]
    k0, k1, k2 = params[n_models:]
    return weights, float(k0), float(k1), float(k2)


def _bma_fit_weights(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    n_models = mu.shape[1]
    weights = np.full(n_models, 1.0 / n_models, dtype=float)
    for _ in range(50):
        log_pdf = -0.5 * (
            ((y[:, None] - mu) ** 2) / (sigma**2) + np.log(2.0 * np.pi * sigma**2)
        )
        log_resp = np.log(weights + 1e-12) + log_pdf
        log_resp = log_resp - logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp)
        weights_new = np.mean(resp, axis=0)
        weights_new = np.where(np.isfinite(weights_new), weights_new, 0.0)
        if np.sum(weights_new) <= 0:
            break
        weights_new = weights_new / np.sum(weights_new)
        if np.max(np.abs(weights_new - weights)) < 1e-5:
            weights = weights_new
            break
        weights = weights_new
    return weights


def _bma_mixture_median(mu: np.ndarray, sigma: np.ndarray, weights: np.ndarray) -> float:
    low = float(np.min(mu) - 6.0)
    high = float(np.max(mu) + 6.0)
    for _ in range(25):
        mid = 0.5 * (low + high)
        cdf = np.sum(weights * student_t.cdf((mid - mu) / sigma, df=100))
        if cdf < 0.5:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _bocpd_student_t(
    series: np.ndarray,
    *,
    hazard: float,
    max_run: int = 400,
    mu0: float = 0.0,
    kappa0: float = 1e-2,
    alpha0: float = 2.0,
    beta0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(series)
    log_r = np.full(max_run + 1, -np.inf, dtype=float)
    log_r[0] = 0.0
    mu = np.full(max_run + 1, mu0, dtype=float)
    kappa = np.full(max_run + 1, kappa0, dtype=float)
    alpha = np.full(max_run + 1, alpha0, dtype=float)
    beta = np.full(max_run + 1, beta0, dtype=float)
    cp_prob = np.full(n, np.nan, dtype=float)
    exp_run = np.full(n, np.nan, dtype=float)

    log_h = np.log(hazard)
    log_1mh = np.log(1.0 - hazard)
    run_idx = np.arange(max_run + 1)

    for t, x in enumerate(series):
        if not np.isfinite(x):
            probs = np.exp(log_r - logsumexp(log_r))
            cp_prob[t] = probs[0]
            exp_run[t] = float(np.sum(run_idx * probs))
            continue

        nu = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        log_pred = student_t.logpdf(x, df=nu, loc=mu, scale=scale)

        log_growth = log_r + log_1mh + log_pred
        log_cp = logsumexp(log_r + log_h + log_pred)

        log_r_new = np.full_like(log_r, -np.inf)
        log_r_new[0] = log_cp
        log_r_new[1:] = log_growth[:-1]
        log_r_new = log_r_new - logsumexp(log_r_new)

        probs = np.exp(log_r_new)
        cp_prob[t] = probs[0]
        exp_run[t] = float(np.sum(run_idx * probs))

        mu_new = np.full_like(mu, mu0)
        kappa_new = np.full_like(kappa, kappa0)
        alpha_new = np.full_like(alpha, alpha0)
        beta_new = np.full_like(beta, beta0)
        mu_prev = mu[:-1]
        kappa_prev = kappa[:-1]
        alpha_prev = alpha[:-1]
        beta_prev = beta[:-1]
        kappa_up = kappa_prev + 1.0
        mu_up = (kappa_prev * mu_prev + x) / kappa_up
        alpha_up = alpha_prev + 0.5
        beta_up = beta_prev + 0.5 * kappa_prev * (x - mu_prev) ** 2 / kappa_up
        mu_new[1:] = mu_up
        kappa_new[1:] = kappa_up
        alpha_new[1:] = alpha_up
        beta_new[1:] = beta_up

        log_r = log_r_new
        mu = mu_new
        kappa = kappa_new
        alpha = alpha_new
        beta = beta_new

    return cp_prob, exp_run

def _ewm_alpha(halflife: float) -> float:
    return 1.0 - float(np.exp(np.log(0.5) / halflife))


def _core_model_cols(df: pd.DataFrame) -> list[str]:
    core = ["gefsatmosmean_tmax_f", "rap_tmax_f", "hrrr_tmax_f", "nbm_tmax_f"]
    missing = [col for col in core if col not in df.columns]
    if not missing:
        return core
    fallback = [col for col in MODEL_COLS if col in df.columns]
    if len(fallback) >= 4:
        LOGGER.warning("Core model columns missing %s; falling back to %s", missing, fallback[:4])
        return fallback[:4]
    if fallback:
        LOGGER.warning("Core model columns missing %s; using available MODEL_COLS=%s", missing, fallback)
        return fallback
    raise ValueError(f"Missing required core model columns: {missing}")


def _core_ensemble_stats(df: pd.DataFrame, core_cols: list[str]) -> dict[str, np.ndarray]:
    values = df[core_cols].to_numpy(dtype=float)
    mean = np.mean(values, axis=1)
    median = np.median(values, axis=1)
    min_vals = np.min(values, axis=1)
    max_vals = np.max(values, axis=1)
    return {
        "mean": mean,
        "median": median,
        "min": min_vals,
        "max": max_vals,
        "range": max_vals - min_vals,
        "std": np.std(values, axis=1, ddof=0),
    }


def _season_id(month: pd.Series) -> pd.Series:
    mapping = {
        12: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
        10: 3,
        11: 3,
    }
    return month.map(mapping).astype(int)


def _quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1:
        return np.array([], dtype=float)
    qs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    return np.quantile(values, qs)


def _bin_ids(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros_like(values, dtype=int)
    ids = np.searchsorted(edges, values, side="right")
    return np.clip(ids, 0, len(edges)).astype(int)


def _interp_percentile(
    values: np.ndarray, q_values: np.ndarray, p_grid: np.ndarray
) -> np.ndarray:
    q_values = np.maximum.accumulate(q_values)
    return np.interp(values, q_values, p_grid, left=p_grid[0], right=p_grid[-1])


def _apply_quantile_map(
    values: np.ndarray, q_f: np.ndarray, q_y: np.ndarray, p_grid: np.ndarray
) -> np.ndarray:
    p_hat = _interp_percentile(values, q_f, p_grid)
    return np.interp(p_hat, p_grid, q_y, left=q_y[0], right=q_y[-1])


def _circular_smooth(values: np.ndarray, bandwidth: float) -> np.ndarray:
    n = len(values)
    output = np.zeros_like(values, dtype=float)
    positions = np.arange(n, dtype=float)
    for idx in range(n):
        dist = np.minimum(np.abs(positions - idx), n - np.abs(positions - idx))
        weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
        weight_sum = np.sum(weights)
        output[idx] = float(np.sum(weights * values) / weight_sum) if weight_sum > 0 else float(values[idx])
    return output


def _rolling_median(
    series: pd.Series, *, window: int, lag: int, group_key: pd.Series
) -> pd.Series:
    return tfl.rolling_apply(
        series,
        window=window,
        min_periods=_min_periods(window),
        lag=lag,
        func=np.median,
        group_key=group_key,
    )


def _rolling_mad(
    series: pd.Series, *, window: int, lag: int, group_key: pd.Series
) -> pd.Series:
    def _mad(values: np.ndarray) -> float:
        med = np.median(values)
        return float(np.median(np.abs(values - med)))

    return tfl.rolling_apply(
        series,
        window=window,
        min_periods=_min_periods(window),
        lag=lag,
        func=_mad,
        group_key=group_key,
    )


def _fit_ar1(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return 0.0, 0.0
    lagged = values[:-1]
    current = values[1:]
    mask = np.isfinite(lagged) & np.isfinite(current)
    if mask.sum() < 2:
        return 0.0, 0.0
    x = lagged[mask]
    y = current[mask]
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return 0.0, 0.0
    phi = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    c = y_mean - phi * x_mean
    return c, phi


def _build_stage1_features(
    df: pd.DataFrame,
    core_cols: list[str],
    stats: dict[str, np.ndarray],
) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for col in core_cols:
        data[col] = df[col].to_numpy(dtype=float)
    data["ens_std_core"] = stats["std"]
    data["ens_range_core"] = stats["range"]
    for i in range(len(core_cols)):
        for j in range(i + 1, len(core_cols)):
            name = f"{core_cols[i]}_minus_{core_cols[j]}"
            data[name] = df[core_cols[i]].to_numpy(dtype=float) - df[core_cols[j]].to_numpy(dtype=float)
    data["month"] = df["month"].to_numpy(dtype=float)
    data["sin_doy"] = df["sin_doy"].to_numpy(dtype=float)
    data["cos_doy"] = df["cos_doy"].to_numpy(dtype=float)
    return pd.DataFrame(data, index=df.index)


def _cv_position_splits(
    train_df: pd.DataFrame, *, n_splits: int, gap_days: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    cv_idx = splits.make_time_cv_splits(train_df, n_splits=n_splits, gap_days=gap_days)
    if not cv_idx:
        return []
    indexer = {idx: pos for pos, idx in enumerate(train_df.index)}
    cv_pos: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in cv_idx:
        train_pos = np.array([indexer[i] for i in train_idx if i in indexer], dtype=int)
        val_pos = np.array([indexer[i] for i in val_idx if i in indexer], dtype=int)
        if len(train_pos) and len(val_pos):
            cv_pos.append((train_pos, val_pos))
    return cv_pos


def _time_oof_regression(
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    build_model: Callable[[], object],
    n_splits: int,
    gap_days: int,
) -> tuple[np.ndarray, object]:
    splits_pos = _cv_position_splits(train_df, n_splits=n_splits, gap_days=gap_days)
    oof = np.full(len(train_df), np.nan, dtype=float)
    for train_pos, val_pos in splits_pos:
        model = build_model()
        model.fit(X_train[train_pos], y_train[train_pos])
        oof[val_pos] = model.predict(X_train[val_pos])
    model_full = build_model()
    model_full.fit(X_train, y_train)
    return oof, model_full


def _time_oof_classifier(
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    build_model: Callable[[], object],
    n_splits: int,
    gap_days: int,
) -> tuple[np.ndarray, object]:
    splits_pos = _cv_position_splits(train_df, n_splits=n_splits, gap_days=gap_days)
    oof = np.full(len(train_df), np.nan, dtype=float)
    for train_pos, val_pos in splits_pos:
        model = build_model()
        model.fit(X_train[train_pos], y_train[train_pos])
        probs = model.predict_proba(X_train[val_pos])
        oof[val_pos] = probs[:, 1]
    model_full = build_model()
    model_full.fit(X_train, y_train)
    return oof, model_full


def _time_oof_classifier_multiclass(
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    build_model: Callable[[], object],
    n_splits: int,
    gap_days: int,
    n_classes: int,
) -> tuple[np.ndarray, object]:
    splits_pos = _cv_position_splits(train_df, n_splits=n_splits, gap_days=gap_days)
    oof = np.full((len(train_df), n_classes), np.nan, dtype=float)
    for train_pos, val_pos in splits_pos:
        model = build_model()
        model.fit(X_train[train_pos], y_train[train_pos])
        probs = model.predict_proba(X_train[val_pos])
        probs_full = np.zeros((len(val_pos), n_classes), dtype=float)
        for cls_idx, cls in enumerate(model.classes_):
            probs_full[:, int(cls)] = probs[:, cls_idx]
        oof[val_pos] = probs_full
    model_full = build_model()
    model_full.fit(X_train, y_train)
    return oof, model_full


def _kalman_loglik(resid: np.ndarray, q: float, r: float) -> float:
    if r <= 0:
        return float("-inf")
    b = 0.0
    p = r
    loglik = 0.0
    for value in resid:
        if not np.isfinite(value):
            continue
        p_pred = p + q
        s = p_pred + r
        if s <= 0:
            continue
        innov = value - b
        loglik += -0.5 * (np.log(2.0 * np.pi * s) + (innov**2) / s)
        k = p_pred / s
        b = b + k * innov
        p = (1.0 - k) * p_pred
    return float(loglik)


def _kalman_filter(resid: np.ndarray, q: float, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(resid)
    b_hat = np.full(n, np.nan, dtype=float)
    p_var = np.full(n, np.nan, dtype=float)
    innov = np.full(n, np.nan, dtype=float)
    b = 0.0
    p = r if r > 0 else 1.0
    for idx in range(n):
        value = resid[idx]
        if np.isfinite(value):
            p_pred = p + q
            s = p_pred + r
            if s > 0:
                k = p_pred / s
                innov_val = value - b
                b = b + k * innov_val
                p = (1.0 - k) * p_pred
                innov[idx] = innov_val
        b_hat[idx] = b
        p_var[idx] = p
    return b_hat, p_var, innov


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
    feature_df = df[feature_cols].astype(float).copy()
    imputed, impute_meta = _impute_features(feature_df, ctx.train_df.index)
    train_imputed = imputed.loc[ctx.train_df.index]
    scaled, scaler_meta, scaler = _standardize_features(
        train_imputed, imputed, feature_cols
    )
    kmeans = KMeans(n_clusters=6, random_state=ctx.seed, n_init=10)
    kmeans.fit(scaler.transform(train_imputed.to_numpy(dtype=float)))
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
            "imputation": impute_meta,
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
    syn_cols = [col for col in ("gfs_tmax_f", "nam_tmax_f", "gefsatmosmean_tmax_f") if col in df.columns]
    meso_cols = [col for col in ("hrrr_tmax_f", "rap_tmax_f", "nbm_tmax_f") if col in df.columns]
    if not syn_cols or not meso_cols:
        return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)
    mean_syn = df[syn_cols].mean(axis=1)
    mean_meso = df[meso_cols].mean(axis=1)
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
    pairs = [(left, right) for left, right in pairs if left in df.columns and right in df.columns]
    if not pairs:
        return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)
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


def _exp_ex01(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    train_df = ctx.train_df
    train_stats = _core_ensemble_stats(train_df, core_cols)
    train_resid = train_df["actual_tmax_f"].to_numpy(dtype=float) - train_stats["mean"]
    train_resid_series = pd.Series(train_resid, index=train_df.index)
    r_base = train_resid_series - train_resid_series.rolling(30, min_periods=20).mean()
    global_r = float(np.nanvar(r_base.to_numpy(dtype=float)))
    if not np.isfinite(global_r) or global_r <= 0:
        global_r = float(np.nanvar(train_resid)) if len(train_resid) else 1.0
    global_r = global_r if global_r > 0 else 1.0

    q_map: dict[str, float] = {}
    r_map: dict[str, float] = {}
    ratio_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        mask = train_df["station_id"] == station
        r_station = train_resid[mask.to_numpy()]
        if len(r_station) < 60:
            q_map[station] = 0.01 * global_r
            r_map[station] = global_r
            continue
        r_series = pd.Series(r_station)
        r_base_station = r_series - r_series.rolling(30, min_periods=20).mean()
        r_val = float(np.nanvar(r_base_station.to_numpy(dtype=float)))
        if not np.isfinite(r_val) or r_val <= 0:
            r_val = float(np.nanvar(r_station))
        if not np.isfinite(r_val) or r_val <= 0:
            r_val = global_r
        best_ratio = ratio_grid[0]
        best_ll = float("-inf")
        for ratio in ratio_grid:
            q_val = ratio * r_val
            ll = _kalman_loglik(r_station, q_val, r_val)
            if ll > best_ll:
                best_ll = ll
                best_ratio = ratio
        q_map[station] = best_ratio * r_val
        r_map[station] = r_val

    b_hat = np.full(len(df), np.nan, dtype=float)
    p_var = np.full(len(df), np.nan, dtype=float)
    innov = np.full(len(df), np.nan, dtype=float)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        q_val = q_map.get(station, 0.01 * global_r)
        r_val = r_map.get(station, global_r)
        b_station, p_station, innov_station = _kalman_filter(resid[idx], q_val, r_val)
        b_hat[idx] = b_station
        p_var[idx] = p_station
        innov[idx] = innov_station

    bias_l2 = pd.Series(b_hat, index=df.index).groupby(gk).shift(lag)
    sd_l2 = pd.Series(np.sqrt(np.maximum(p_var, 0.0)), index=df.index).groupby(gk).shift(lag)
    innov_l2 = pd.Series(innov, index=df.index).groupby(gk).shift(lag)
    r_vals = df["station_id"].map(r_map).fillna(global_r).to_numpy(dtype=float)
    innov_z = innov_l2.to_numpy(dtype=float) / (np.sqrt(r_vals) + EPS)

    _add_feature(
        features,
        formulas,
        f"kalman_bias_l{lag}",
        bias_l2,
        "kalman_bias_state",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"kalman_bias_sd_l{lag}",
        sd_l2,
        "kalman_state_sd",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"kalman_innov_l{lag}",
        innov_z,
        "(resid - bias) / sqrt(R)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_kalman_corr",
        ens_mean + bias_l2.to_numpy(dtype=float),
        "ens_mean + kalman_bias",
    )
    train_fitted.append(
        {
            "name": "kalman_qr",
            "fit_on": "train",
            "ratio_grid": ratio_grid,
            "per_station_q": q_map,
            "per_station_r": r_map,
            "global_r": global_r,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex02(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_spread = train_stats["std"]
    global_med = float(np.nanmedian(train_spread)) if len(train_spread) else 0.0
    global_iqr = float(np.nanquantile(train_spread, 0.75) - np.nanquantile(train_spread, 0.25)) if len(train_spread) else 1.0
    global_iqr = global_iqr if global_iqr > 0 else 1.0
    global_s = float(np.nanmedian(np.abs(ctx.train_df["actual_tmax_f"].to_numpy(dtype=float) - train_stats["mean"])))
    global_s = global_s if np.isfinite(global_s) and global_s > 0 else 1.0

    med_map = {}
    iqr_map = {}
    init_s_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        spread_vals = train_spread[mask.to_numpy()]
        if len(spread_vals) >= 30:
            med_map[station] = float(np.nanmedian(spread_vals))
            iqr_val = float(np.nanquantile(spread_vals, 0.75) - np.nanquantile(spread_vals, 0.25))
            iqr_map[station] = iqr_val if iqr_val > 0 else global_iqr
        else:
            med_map[station] = global_med
            iqr_map[station] = global_iqr
        resid_vals = (ctx.train_df.loc[mask, "actual_tmax_f"].to_numpy(dtype=float) - train_stats["mean"][mask.to_numpy()])
        if len(resid_vals) >= 10:
            init_s = float(np.nanmedian(np.abs(resid_vals)))
            init_s_map[station] = init_s if init_s > 0 else global_s
        else:
            init_s_map[station] = global_s

    alpha_s = _ewm_alpha(30)
    alpha_min = 0.01
    alpha_max = 0.08
    c = 2.5
    b_hist = np.full(len(df), np.nan, dtype=float)
    s_hist = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        b = 0.0
        s = init_s_map.get(station, global_s)
        count = 0
        med_spread = med_map.get(station, global_med)
        iqr_spread = iqr_map.get(station, global_iqr)
        for pos, row_idx in enumerate(idx):
            r_val = resid[row_idx]
            if np.isfinite(r_val):
                e_val = r_val - b
                s = s + alpha_s * (abs(e_val) - s)
                count += 1
                spread_z = (ens_std[row_idx] - med_spread) / (iqr_spread + EPS)
                alpha = alpha_min + (alpha_max - alpha_min) * _sigmoid(np.array([spread_z]))[0]
                clip_thr = c * s if np.isfinite(s) else c * global_s
                if not np.isfinite(clip_thr) or clip_thr <= 0:
                    clip_thr = c * global_s
                clipped = float(np.clip(e_val, -clip_thr, clip_thr))
                b = b + alpha * clipped
            b_hist[row_idx] = b
            s_hist[row_idx] = s if count >= 20 else np.nan

    bias_l2 = pd.Series(b_hist, index=df.index).groupby(gk).shift(lag)
    scale_l2 = pd.Series(s_hist, index=df.index).groupby(gk).shift(lag)

    _add_feature(
        features,
        formulas,
        f"huber_bias_l{lag}",
        bias_l2,
        "huber_bias_state",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"huber_scale_l{lag}",
        scale_l2,
        "ewm_mean(|innovation|)",
        {"lag": lag, "halflife": 30},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_huber_corr",
        ens_mean + bias_l2.to_numpy(dtype=float),
        "ens_mean + huber_bias",
    )
    train_fitted.append(
        {
            "name": "huber_bias_params",
            "fit_on": "train",
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "c": c,
            "med_spread": med_map,
            "iqr_spread": iqr_map,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex03(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    m_series = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=60,
        min_periods=30,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    delta = 0.05
    lambda_map: dict[str, float] = {}
    stations = df["station_id"].to_numpy()
    global_std = float(np.nanstd(resid)) if len(resid) else 1.0
    target_rate = max(1.0, len(ctx.train_df) / 90.0)
    for station in np.unique(stations):
        mask = ctx.train_df["station_id"] == station
        train_idx = ctx.train_df.index[mask].to_numpy()
        full_idx = df.index.get_indexer(train_idx)
        full_idx = full_idx[full_idx >= 0]
        r_train = resid[full_idx]
        m_train = m_series[full_idx]
        if len(r_train) < 30:
            lambda_map[station] = 2.0 * global_std
            continue
        std_val = float(np.nanstd(r_train)) if np.isfinite(np.nanstd(r_train)) else global_std
        grid = [1.0, 2.0, 3.0, 4.0]
        best_lambda = grid[0] * std_val
        best_score = float("inf")
        for mult in grid:
            thr = mult * std_val
            ph = 0.0
            min_ph = 0.0
            alarms = 0.0
            for pos in range(len(r_train)):
                m_prev = m_train[pos - 1] if pos > 0 else m_train[pos]
                ph = ph + (r_train[pos] - m_prev - delta)
                min_ph = min(min_ph, ph)
                if (ph - min_ph) > thr:
                    alarms += 1.0
            score = abs(alarms - target_rate)
            if score < best_score:
                best_score = score
                best_lambda = thr
        lambda_map[station] = best_lambda

    ph_stat = np.full(len(df), np.nan, dtype=float)
    alarm = np.full(len(df), np.nan, dtype=float)
    days_since = np.full(len(df), np.nan, dtype=float)
    b_reset = np.full(len(df), np.nan, dtype=float)
    alpha_reset = _ewm_alpha(14)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        lam = lambda_map.get(station, 2.0 * global_std)
        ph = 0.0
        min_ph = 0.0
        since = 0.0
        b_val = 0.0
        for pos, row_idx in enumerate(idx):
            m_prev = m_series[idx[pos - 1]] if pos > 0 else m_series[row_idx]
            r_val = resid[row_idx]
            ph = ph + (r_val - m_prev - delta)
            min_ph = min(min_ph, ph)
            alarm_flag = (ph - min_ph) > lam
            if alarm_flag:
                since = 0.0
                b_val = 0.0
            else:
                since = min(since + 1.0, 365.0)
                if np.isfinite(r_val):
                    b_val = b_val + alpha_reset * (r_val - b_val)
            ph_stat[row_idx] = ph - min_ph
            alarm[row_idx] = float(alarm_flag)
            days_since[row_idx] = since
            b_reset[row_idx] = b_val

    ph_alarm_l2 = pd.Series(alarm, index=df.index).groupby(gk).shift(lag)
    ph_days_l2 = pd.Series(days_since, index=df.index).groupby(gk).shift(lag)
    bias_reset_l2 = pd.Series(b_reset, index=df.index).groupby(gk).shift(lag)
    _add_feature(
        features,
        formulas,
        f"ph_alarm_l{lag}",
        ph_alarm_l2,
        "page_hinkley_alarm",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"ph_days_since_l{lag}",
        ph_days_l2,
        "page_hinkley_days_since",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"bias_reset_l{lag}",
        bias_reset_l2,
        "resettable_bias_state",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_reset_corr",
        ens_mean + bias_reset_l2.to_numpy(dtype=float),
        "ens_mean + reset_bias",
    )
    train_fitted.append(
        {
            "name": "page_hinkley_lambda",
            "fit_on": "train",
            "delta": delta,
            "lambda_per_station": lambda_map,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex04(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    slow = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=60,
        min_periods=30,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    fast = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=7,
        min_periods=10,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    drift = fast - slow
    drift_scale = tfl.ewm_mean(
        pd.Series(np.abs(drift), index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    gate = _sigmoid(1.5 * (np.abs(drift) / (drift_scale + EPS)) - 1.0)
    blend = slow + gate * drift
    blend_l2 = pd.Series(blend, index=df.index).groupby(gk).shift(lag)
    gate_l2 = pd.Series(gate, index=df.index).groupby(gk).shift(lag)

    _add_feature(
        features,
        formulas,
        f"blend_bias_l{lag}",
        blend_l2,
        "slow_bias + gate*drift",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"drift_gate_l{lag}",
        gate_l2,
        "sigmoid(|drift|/scale)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_blend_corr",
        ens_mean + blend_l2.to_numpy(dtype=float),
        "ens_mean + blend_bias",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex05(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    doy = df["day_of_year"].to_numpy(dtype=int)
    doy = np.where(doy > 365, 365, doy)

    train_df = ctx.train_df
    train_stats = _core_ensemble_stats(train_df, core_cols)
    train_resid = train_df["actual_tmax_f"].to_numpy(dtype=float) - train_stats["mean"]
    train_doy = train_df["day_of_year"].to_numpy(dtype=int)
    train_doy = np.where(train_doy > 365, 365, train_doy)
    global_mu = float(np.nanmean(train_resid)) if len(train_resid) else 0.0
    global_raw = np.full(365, global_mu, dtype=float)
    for d in range(1, 366):
        vals = train_resid[train_doy == d]
        if len(vals):
            global_raw[d - 1] = float(np.nanmean(vals))
    global_smooth = _circular_smooth(global_raw, bandwidth=15)

    station_maps: dict[str, np.ndarray] = {}
    for station in df["station_id"].unique():
        mask = train_df["station_id"] == station
        if mask.sum() < 200:
            continue
        r_station = train_resid[mask.to_numpy()]
        doy_station = train_doy[mask.to_numpy()]
        raw = np.full(365, float(np.nanmean(r_station)), dtype=float)
        for d in range(1, 366):
            vals = r_station[doy_station == d]
            if len(vals):
                raw[d - 1] = float(np.nanmean(vals))
        station_maps[station] = _circular_smooth(raw, bandwidth=15)

    bias_prior = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for idx, station in enumerate(stations):
        mu_vec = station_maps.get(station)
        if mu_vec is None:
            mu_vec = global_smooth
        bias_prior[idx] = mu_vec[doy[idx] - 1]

    delta_r = resid - bias_prior
    delta_ewm = tfl.ewm_mean(
        pd.Series(delta_r, index=df.index),
        halflife=14,
        min_periods=10,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    bias_total = bias_prior + delta_ewm
    bias_l2 = pd.Series(bias_total, index=df.index).groupby(gk).shift(lag)

    _add_feature(
        features,
        formulas,
        "bias_doy_prior",
        bias_prior,
        "smoothed_doy_bias",
    )
    _add_feature(
        features,
        formulas,
        f"bias_total_l{lag}",
        bias_l2,
        "doy_bias + delta_ewm",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_doycorr",
        ens_mean + bias_l2.to_numpy(dtype=float),
        "ens_mean + bias_total",
    )
    train_fitted.append(
        {
            "name": "doy_bias_prior",
            "fit_on": "train",
            "global": global_smooth.tolist(),
            "per_station": {k: v.tolist() for k, v in station_maps.items()},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex06(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    station_bias = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    )
    daily_mean = pd.Series(resid, index=df.index).groupby(df["target_date_local"]).mean()
    global_bias_series = daily_mean.ewm(halflife=30, min_periods=200, adjust=False).mean()
    global_bias = df["target_date_local"].map(global_bias_series).to_numpy(dtype=float)

    indicator = pd.Series(np.isfinite(resid).astype(float), index=df.index)
    n_obs = tfl.rolling_sum(indicator, window=120, min_periods=1, lag=0, group_key=gk)
    shrink_w = n_obs.to_numpy(dtype=float) / (n_obs.to_numpy(dtype=float) + 60.0)
    pooled = station_bias.to_numpy(dtype=float) * shrink_w + global_bias * (1.0 - shrink_w)
    pooled_l2 = pd.Series(pooled, index=df.index).groupby(gk).shift(lag)

    _add_feature(
        features,
        formulas,
        f"pooled_bias_l{lag}",
        pooled_l2,
        "station_bias*shrink + global_bias*(1-shrink)",
        {"lag": lag, "kappa": 60},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_pooled_corr",
        ens_mean + pooled_l2.to_numpy(dtype=float),
        "ens_mean + pooled_bias",
    )
    train_fitted.append(
        {
            "name": "hierarchical_bias_state",
            "fit_on": "train",
            "kappa": 60,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex07(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_resid = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float) - train_stats["mean"]
    train_df = ctx.train_df.copy()
    train_df["resid_core"] = train_resid
    mu_sm = train_df.groupby(["station_id", "month"])["resid_core"].mean()
    n_sm = train_df.groupby(["station_id", "month"])["resid_core"].count()
    mu_m = train_df.groupby("month")["resid_core"].mean()
    global_mu = float(np.nanmean(train_resid)) if len(train_resid) else 0.0
    kappa = 30.0

    mu_sm_map = mu_sm.to_dict()
    n_sm_map = n_sm.to_dict()
    mu_m_map = mu_m.to_dict()

    bias_vals = np.full(len(df), np.nan, dtype=float)
    shrink_vals = np.full(len(df), np.nan, dtype=float)
    for idx, row in df.iterrows():
        key = (row["station_id"], row["month"])
        mu_m_val = mu_m_map.get(row["month"], global_mu)
        n_val = float(n_sm_map.get(key, 0.0))
        shrink = n_val / (n_val + kappa) if n_val > 0 else 0.0
        mu_sm_val = mu_sm_map.get(key, mu_m_val)
        bias_vals[df.index.get_loc(idx)] = mu_m_val + shrink * (mu_sm_val - mu_m_val)
        shrink_vals[df.index.get_loc(idx)] = shrink

    core_stats = _core_ensemble_stats(df, core_cols)
    ens_mean = core_stats["mean"]
    _add_feature(
        features,
        formulas,
        "bias_station_month_prior",
        bias_vals,
        "month_bias_shrunk",
        {"kappa": kappa},
    )
    _add_feature(
        features,
        formulas,
        "shrink_station_month",
        shrink_vals,
        "n_sm/(n_sm+kappa)",
        {"kappa": kappa},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_station_month_corr",
        ens_mean + bias_vals,
        "ens_mean + station_month_bias",
    )
    train_fitted.append(
        {
            "name": "station_month_bias",
            "fit_on": "train",
            "mu_sm": {str(k): float(v) for k, v in mu_sm_map.items()},
            "n_sm": {str(k): int(v) for k, v in n_sm_map.items()},
            "mu_m": {int(k): float(v) for k, v in mu_m_map.items()},
            "kappa": kappa,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex08(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_mean = train_stats["mean"]
    train_y = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float)

    global_iso = IsotonicRegression(out_of_bounds="clip")
    global_iso.fit(train_mean, train_y)
    iso_models = {}
    station_counts = ctx.train_df["station_id"].value_counts()
    for station, count in station_counts.items():
        if count < 300:
            continue
        mask = ctx.train_df["station_id"] == station
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_mean[mask.to_numpy()], train_y[mask.to_numpy()])
        iso_models[station] = iso

    iso_vals = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        model = iso_models.get(station, global_iso)
        iso_vals[idx] = model.predict(ens_mean[idx])

    edges = _quantile_edges(train_mean, 10)
    seg_id = _bin_ids(ens_mean, edges)
    _add_feature(
        features,
        formulas,
        "ens_mean_iso_cal",
        iso_vals,
        "isotonic(ens_mean)",
    )
    _add_feature(
        features,
        formulas,
        "iso_delta",
        iso_vals - ens_mean,
        "ens_mean_iso - ens_mean",
    )
    _add_feature(
        features,
        formulas,
        "iso_segment_id",
        seg_id,
        "decile_bin(ens_mean)",
    )
    train_fitted.append(
        {
            "name": "isotonic_ens_mean",
            "fit_on": "train",
            "global_x": global_iso.X_thresholds_.tolist(),
            "global_y": global_iso.y_thresholds_.tolist(),
            "per_station": {
                station: {
                    "x": iso.X_thresholds_.tolist(),
                    "y": iso.y_thresholds_.tolist(),
                }
                for station, iso in iso_models.items()
            },
            "decile_edges": edges.tolist(),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex09(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_mean = train_stats["mean"]
    train_std = train_stats["std"]
    train_resid = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float) - train_mean
    temp_edges = _quantile_edges(train_mean, 10)
    spread_edges = _quantile_edges(train_std, 5)
    temp_bin = _bin_ids(train_mean, temp_edges)
    spread_bin = _bin_ids(train_std, spread_edges)
    n_temp = len(temp_edges) + 1
    n_spread = len(spread_edges) + 1
    cell_mean = np.full((n_temp, n_spread), 0.0, dtype=float)
    cell_n = np.zeros((n_temp, n_spread), dtype=float)
    for t_bin, s_bin, r_val in zip(temp_bin, spread_bin, train_resid, strict=False):
        if np.isfinite(r_val):
            cell_mean[t_bin, s_bin] += r_val
            cell_n[t_bin, s_bin] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_mean = np.divide(cell_mean, cell_n, out=np.zeros_like(cell_mean), where=cell_n > 0)
    marginal = np.full(n_temp, 0.0, dtype=float)
    for t_bin in range(n_temp):
        mask = temp_bin == t_bin
        vals = train_resid[mask]
        marginal[t_bin] = float(np.nanmean(vals)) if len(vals) else 0.0
    kappa = 50.0
    shrink = cell_n / (cell_n + kappa)
    bias2d = marginal[:, None] + shrink * (cell_mean - marginal[:, None])

    temp_all = _bin_ids(ens_mean, temp_edges)
    spread_all = _bin_ids(ens_std, spread_edges)
    bias_vals = bias2d[temp_all, spread_all]
    n_vals = cell_n[temp_all, spread_all]

    _add_feature(
        features,
        formulas,
        "bias_2d",
        bias_vals,
        "bias_surface(temp_bin, spread_bin)",
    )
    _add_feature(
        features,
        formulas,
        "bias2d_n",
        n_vals,
        "bias_surface_count",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_2d_corr",
        ens_mean + bias_vals,
        "ens_mean + bias_2d",
    )
    train_fitted.append(
        {
            "name": "bias_surface_2d",
            "fit_on": "train",
            "temp_edges": temp_edges.tolist(),
            "spread_edges": spread_edges.tolist(),
            "cell_mean": cell_mean.tolist(),
            "cell_n": cell_n.tolist(),
            "kappa": kappa,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex10(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    season_id = _season_id(df["month"])
    p_grid = np.arange(0.01, 1.0, 0.01)

    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_mean = train_stats["mean"]
    train_y = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float)
    train_season = _season_id(ctx.train_df["month"]).to_numpy(dtype=int)

    q_f = {}
    q_y = {}
    for season in np.unique(train_season):
        mask = train_season == season
        if mask.sum() < 10:
            continue
        q_f[season] = np.quantile(train_mean[mask], p_grid).tolist()
        q_y[season] = np.quantile(train_y[mask], p_grid).tolist()

    mapped = np.full(len(df), np.nan, dtype=float)
    for season in np.unique(season_id):
        mask = season_id.to_numpy(dtype=int) == season
        q_f_vals = np.array(q_f.get(season, np.quantile(train_mean, p_grid)))
        q_y_vals = np.array(q_y.get(season, np.quantile(train_y, p_grid)))
        mapped[mask] = _apply_quantile_map(ens_mean[mask], q_f_vals, q_y_vals, p_grid)

    _add_feature(
        features,
        formulas,
        "ens_mean_qmap",
        mapped,
        "quantile_map(ens_mean)",
    )
    _add_feature(
        features,
        formulas,
        "qmap_delta",
        mapped - ens_mean,
        "ens_mean_qmap - ens_mean",
    )
    train_fitted.append(
        {
            "name": "season_quantile_map",
            "fit_on": "train",
            "p_grid": p_grid.tolist(),
            "q_f": q_f,
            "q_y": q_y,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex11(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    season_id = _season_id(df["month"]).to_numpy(dtype=int)
    p_grid = np.arange(0.01, 1.0, 0.01)

    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_season = _season_id(ctx.train_df["month"]).to_numpy(dtype=int)
    train_y = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float)
    q_y = {}
    for season in np.unique(train_season):
        mask = train_season == season
        if mask.sum() < 10:
            continue
        q_y[season] = np.quantile(train_y[mask], p_grid).tolist()

    q_f = {col: {} for col in core_cols}
    for season in np.unique(train_season):
        mask = train_season == season
        for col in core_cols:
            vals = ctx.train_df.loc[mask, col].to_numpy(dtype=float)
            if len(vals) < 10:
                continue
            q_f[col][season] = np.quantile(vals, p_grid).tolist()

    mapped_vals = {}
    for col in core_cols:
        mapped = np.full(len(df), np.nan, dtype=float)
        for season in np.unique(season_id):
            mask = season_id == season
            q_f_vals = np.array(q_f.get(col, {}).get(season, np.quantile(ctx.train_df[col].to_numpy(dtype=float), p_grid)))
            q_y_vals = np.array(q_y.get(season, np.quantile(train_y, p_grid)))
            mapped[mask] = _apply_quantile_map(df[col].to_numpy(dtype=float)[mask], q_f_vals, q_y_vals, p_grid)
        mapped_vals[col] = mapped

    mapped_stack = np.column_stack([mapped_vals[col] for col in core_cols])
    mapped_mean = np.nanmean(mapped_stack, axis=1)
    mapped_std = np.nanstd(mapped_stack, axis=1)
    n_models = np.sum(np.isfinite(mapped_stack), axis=1)

    _add_feature(
        features,
        formulas,
        "mapped_ens_mean",
        mapped_mean,
        "mean(mapped_models)",
    )
    _add_feature(
        features,
        formulas,
        "mapped_spread_std",
        mapped_std,
        "std(mapped_models)",
    )
    _add_feature(
        features,
        formulas,
        "n_models_mapped",
        n_models,
        "count(mapped_models)",
    )
    train_fitted.append(
        {
            "name": "per_model_quantile_map",
            "fit_on": "train",
            "p_grid": p_grid.tolist(),
            "q_y": q_y,
            "q_f": q_f,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex12(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    bias = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=60,
        min_periods=30,
        lag=0,
        group_key=gk,
    )
    innov = resid - bias.to_numpy(dtype=float)
    var_state = tfl.ewm_mean(
        pd.Series(innov**2, index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    sigma = np.sqrt(np.maximum(var_state, 0.0) + EPS)

    bias_l2 = bias.groupby(gk).shift(lag)
    sigma_l2 = pd.Series(sigma, index=df.index).groupby(gk).shift(lag)

    lambda_grid = [0.0, 0.05, 0.1, 0.2]
    best_lambda = 0.0
    best_mae = float("inf")
    for lam in lambda_grid:
        damp = 1.0 / (1.0 + lam * sigma_l2.to_numpy(dtype=float))
        pred = ens_mean + damp * bias_l2.to_numpy(dtype=float)
        train_pred = pred[df.index.get_indexer(ctx.train_df.index)]
        train_y = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float)
        mask = np.isfinite(train_pred) & np.isfinite(train_y)
        if mask.sum() == 0:
            continue
        mae = float(np.mean(np.abs(train_pred[mask] - train_y[mask])))
        if mae < best_mae:
            best_mae = mae
            best_lambda = lam

    damp = 1.0 / (1.0 + best_lambda * sigma_l2.to_numpy(dtype=float))
    _add_feature(
        features,
        formulas,
        f"sigma_l{lag}",
        sigma_l2,
        "ewm_mean(innovation^2)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"bias_slow_l{lag}",
        bias_l2,
        "ewm_mean(resid)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_damped_corr",
        ens_mean + damp * bias_l2.to_numpy(dtype=float),
        "ens_mean + damp*bias",
    )
    _add_feature(
        features,
        formulas,
        "sigma_over_biasmag",
        sigma_l2.to_numpy(dtype=float) / (np.abs(bias_l2.to_numpy(dtype=float)) + EPS),
        "sigma_l2/|bias_l2|",
    )
    train_fitted.append(
        {
            "name": "volatility_damp_lambda",
            "fit_on": "train",
            "lambda_grid": lambda_grid,
            "best_lambda": best_lambda,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex13(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    abs_resid = np.abs(resid)

    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_std = train_stats["std"]
    global_med = float(np.nanmedian(train_std)) if len(train_std) else 0.0
    global_iqr = float(np.nanquantile(train_std, 0.75) - np.nanquantile(train_std, 0.25)) if len(train_std) else 1.0
    global_iqr = global_iqr if global_iqr > 0 else 1.0

    med_map = {}
    iqr_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        vals = train_std[mask.to_numpy()]
        if len(vals) >= 30:
            med_map[station] = float(np.nanmedian(vals))
            iqr_val = float(np.nanquantile(vals, 0.75) - np.nanquantile(vals, 0.25))
            iqr_map[station] = iqr_val if iqr_val > 0 else global_iqr
        else:
            med_map[station] = global_med
            iqr_map[station] = global_iqr

    spread_hi = np.zeros(len(df), dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        spread_hi[idx] = float(ens_std[idx] > med_map.get(station, global_med))

    scale_hi_num = tfl.ewm_mean(
        pd.Series(abs_resid * spread_hi, index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    )
    scale_hi_den = tfl.ewm_mean(
        pd.Series(spread_hi, index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    )
    scale_lo_num = tfl.ewm_mean(
        pd.Series(abs_resid * (1.0 - spread_hi), index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    )
    scale_lo_den = tfl.ewm_mean(
        pd.Series(1.0 - spread_hi, index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    )
    scale_hi = scale_hi_num.to_numpy(dtype=float) / (scale_hi_den.to_numpy(dtype=float) + EPS)
    scale_lo = scale_lo_num.to_numpy(dtype=float) / (scale_lo_den.to_numpy(dtype=float) + EPS)

    scale_hi_l2 = pd.Series(scale_hi, index=df.index).groupby(gk).shift(lag)
    scale_lo_l2 = pd.Series(scale_lo, index=df.index).groupby(gk).shift(lag)

    p_hi = np.zeros(len(df), dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        med_val = med_map.get(station, global_med)
        iqr_val = iqr_map.get(station, global_iqr)
        p_hi[idx] = _sigmoid(np.array([(ens_std[idx] - med_val) / (iqr_val + EPS)]))[0]

    pred_scale = p_hi * scale_hi_l2.to_numpy(dtype=float) + (1.0 - p_hi) * scale_lo_l2.to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        f"scale_hi_l{lag}",
        scale_hi_l2,
        "ewm_mean(|resid|*high_spread) / ewm_mean(high_spread)",
    )
    _add_feature(
        features,
        formulas,
        f"scale_lo_l{lag}",
        scale_lo_l2,
        "ewm_mean(|resid|*low_spread) / ewm_mean(low_spread)",
    )
    _add_feature(
        features,
        formulas,
        "pred_scale",
        pred_scale,
        "p_hi*scale_hi + (1-p_hi)*scale_lo",
    )
    _add_feature(
        features,
        formulas,
        "p_hi",
        p_hi,
        "sigmoid(spread_z)",
    )
    train_fitted.append(
        {
            "name": "spread_regime_scales",
            "fit_on": "train",
            "med_spread": med_map,
            "iqr_spread": iqr_map,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex14(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    abs_err = np.abs(y - ens_mean)

    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_std = train_stats["std"]
    train_abs_err = np.abs(ctx.train_df["actual_tmax_f"].to_numpy(dtype=float) - train_stats["mean"])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_std, train_abs_err)

    spread_edges = _quantile_edges(train_std, 10)
    train_bins = _bin_ids(train_std, spread_edges)
    q90_by_bin = {}
    for b in range(len(spread_edges) + 1):
        vals = train_abs_err[train_bins == b]
        q90_by_bin[b] = float(np.quantile(vals, 0.90)) if len(vals) else float(np.quantile(train_abs_err, 0.90))

    pred_iso = iso.predict(ens_std)
    bins_all = _bin_ids(ens_std, spread_edges)
    pred_q90 = np.array([q90_by_bin.get(b, q90_by_bin.get(0, 0.0)) for b in bins_all], dtype=float)

    _add_feature(
        features,
        formulas,
        "pred_abs_err_iso",
        pred_iso,
        "isotonic(abs_err|spread)",
    )
    _add_feature(
        features,
        formulas,
        "pred_abs_err_q90",
        pred_q90,
        "q90_abs_err(spread_bin)",
    )
    _add_feature(
        features,
        formulas,
        "spread_reliability_ratio",
        pred_iso / (ens_std + EPS),
        "pred_abs_err_iso / spread",
    )
    train_fitted.append(
        {
            "name": "spread_reliability_iso",
            "fit_on": "train",
            "iso_x": iso.X_thresholds_.tolist(),
            "iso_y": iso.y_thresholds_.tolist(),
            "spread_edges": spread_edges.tolist(),
            "q90_by_bin": q90_by_bin,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex15(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_median = stats["median"]
    ens_std = stats["std"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    pos_mask = (resid > 0).astype(float)
    neg_mask = (resid < 0).astype(float)
    pos_num = tfl.ewm_mean(
        pd.Series(resid * pos_mask, index=df.index),
        halflife=45,
        min_periods=10,
        lag=0,
        group_key=gk,
    )
    pos_den = tfl.ewm_mean(
        pd.Series(pos_mask, index=df.index),
        halflife=45,
        min_periods=10,
        lag=0,
        group_key=gk,
    )
    neg_num = tfl.ewm_mean(
        pd.Series(resid * neg_mask, index=df.index),
        halflife=45,
        min_periods=10,
        lag=0,
        group_key=gk,
    )
    neg_den = tfl.ewm_mean(
        pd.Series(neg_mask, index=df.index),
        halflife=45,
        min_periods=10,
        lag=0,
        group_key=gk,
    )
    pos_mean = pos_num.to_numpy(dtype=float) / (pos_den.to_numpy(dtype=float) + EPS)
    neg_mean = neg_num.to_numpy(dtype=float) / (neg_den.to_numpy(dtype=float) + EPS)

    pos_l2 = pd.Series(pos_mean, index=df.index).groupby(gk).shift(lag)
    neg_l2 = pd.Series(neg_mean, index=df.index).groupby(gk).shift(lag)
    skew_l2 = pos_l2.to_numpy(dtype=float) + neg_l2.to_numpy(dtype=float)

    asym_proxy = (ens_mean - ens_median) / (ens_std + EPS)
    expected_bias = np.where(asym_proxy >= 0, pos_l2.to_numpy(dtype=float), neg_l2.to_numpy(dtype=float))

    _add_feature(
        features,
        formulas,
        f"pos_bias_l{lag}",
        pos_l2,
        "ewm_mean(resid>0) / ewm_mean(indicator)",
    )
    _add_feature(
        features,
        formulas,
        f"neg_bias_l{lag}",
        neg_l2,
        "ewm_mean(resid<0) / ewm_mean(indicator)",
    )
    _add_feature(
        features,
        formulas,
        f"skew_state_l{lag}",
        skew_l2,
        "pos_bias + neg_bias",
    )
    _add_feature(
        features,
        formulas,
        "asym_proxy",
        asym_proxy,
        "(ens_mean - ens_median) / ens_std",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_asymcorr_simple",
        ens_mean + expected_bias,
        "ens_mean + expected_bias",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex16(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_median = stats["median"]
    ens_std = stats["std"]
    ens_range = stats["range"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    pos_mask = (resid > 0).astype(float)
    neg_mask = (resid < 0).astype(float)
    pos_num = tfl.ewm_mean(
        pd.Series(resid * pos_mask, index=df.index),
        halflife=45,
        min_periods=10,
        lag=0,
        group_key=gk,
    )
    pos_den = tfl.ewm_mean(
        pd.Series(pos_mask, index=df.index),
        halflife=45,
        min_periods=10,
        lag=0,
        group_key=gk,
    )
    neg_num = tfl.ewm_mean(
        pd.Series(resid * neg_mask, index=df.index),
        halflife=45,
        min_periods=10,
        lag=0,
        group_key=gk,
    )
    neg_den = tfl.ewm_mean(
        pd.Series(neg_mask, index=df.index),
        halflife=45,
        min_periods=10,
        lag=0,
        group_key=gk,
    )
    pos_mean = pos_num.to_numpy(dtype=float) / (pos_den.to_numpy(dtype=float) + EPS)
    neg_mean = neg_num.to_numpy(dtype=float) / (neg_den.to_numpy(dtype=float) + EPS)
    pos_l2 = pd.Series(pos_mean, index=df.index).groupby(gk).shift(lag)
    neg_l2 = pd.Series(neg_mean, index=df.index).groupby(gk).shift(lag)

    feat_df = _build_stage1_features(df, core_cols, stats)
    feat_df["ens_range_core"] = ens_range
    X_all = feat_df.to_numpy(dtype=float)
    train_feat = feat_df.loc[ctx.train_df.index]
    X_train = train_feat.to_numpy(dtype=float)
    train_resid = resid[df.index.get_indexer(ctx.train_df.index)]
    y_train = (train_resid > 0).astype(int)

    if len(np.unique(y_train)) < 2:
        p_full = np.full(len(df), float(np.mean(y_train)) if len(y_train) else 0.5, dtype=float)
        train_fitted.append(
            {
                "name": "resid_sign_classifier",
                "fit_on": "train",
                "description": "single-class fallback",
            }
        )
    else:
        def _build() -> LogisticRegression:
            return LogisticRegression(max_iter=500, random_state=ctx.seed)

        oof, model_full = _time_oof_classifier(
            ctx.train_df,
            X_train,
            y_train,
            build_model=_build,
            n_splits=6,
            gap_days=2,
        )
        p_full = model_full.predict_proba(X_all)[:, 1]
        train_pos = df.index.get_indexer(ctx.train_df.index)
        oof_mask = np.isfinite(oof)
        p_full[train_pos[oof_mask]] = oof[oof_mask]
        train_fitted.append(
            {
                "name": "resid_sign_classifier",
                "fit_on": "train",
                "coef": model_full.coef_.tolist(),
                "intercept": model_full.intercept_.tolist(),
                "features": list(feat_df.columns),
            }
        )

    expected_bias = p_full * pos_l2.to_numpy(dtype=float) + (1.0 - p_full) * neg_l2.to_numpy(dtype=float)
    _add_feature(
        features,
        formulas,
        "p_pos_hat",
        p_full,
        "P(resid>0 | X)",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_asymcorr",
        ens_mean + expected_bias,
        "ens_mean + blended_bias",
    )
    _add_feature(
        features,
        formulas,
        f"pos_bias_l{lag}",
        pos_l2,
        "pos_bias_state",
    )
    _add_feature(
        features,
        formulas,
        f"neg_bias_l{lag}",
        neg_l2,
        "neg_bias_state",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex17(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    values = df[core_cols].to_numpy(dtype=float)
    priority = {col: idx for idx, col in enumerate(derived_features.RAW_MODEL_ORDER)}
    perm_list = list(permutations(range(len(core_cols))))
    perm_to_id = {perm: idx for idx, perm in enumerate(perm_list)}
    perm_id = np.full(len(df), 0, dtype=int)
    for row_idx, row_vals in enumerate(values):
        tuples = []
        for idx, val in enumerate(row_vals):
            col = core_cols[idx]
            tuples.append((val, priority.get(col, idx), idx))
        tuples.sort()
        order = tuple(item[2] for item in tuples)
        perm_id[row_idx] = perm_to_id[order]

    bias_matrix = np.zeros((len(df), len(perm_list)), dtype=float)
    count_matrix = np.zeros_like(bias_matrix)
    for k in range(len(perm_list)):
        indicator = (perm_id == k).astype(float)
        bias_k = tfl.rolling_conditional_mean(
            pd.Series(resid, index=df.index),
            pd.Series(indicator, index=df.index),
            window=180,
            min_periods=60,
            lag=lag,
            group_key=gk,
        )
        count_k = tfl.rolling_sum(
            pd.Series(indicator, index=df.index),
            window=180,
            min_periods=60,
            lag=lag,
            group_key=gk,
        )
        bias_matrix[:, k] = bias_k.to_numpy(dtype=float)
        count_matrix[:, k] = count_k.to_numpy(dtype=float)

    station_bias_l2 = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=30,
        min_periods=20,
        lag=lag,
        group_key=gk,
    ).to_numpy(dtype=float)
    selected_bias = bias_matrix[np.arange(len(df)), perm_id]
    selected_count = count_matrix[np.arange(len(df)), perm_id]
    selected_bias = np.where(selected_count < 10, station_bias_l2, selected_bias)

    for k in range(len(perm_list)):
        _add_feature(
            features,
            formulas,
            f"perm_id_{k}",
            (perm_id == k).astype(float),
            f"1[perm_id == {k}]",
        )
    _add_feature(
        features,
        formulas,
        "bias_perm_selected",
        selected_bias,
        "conditional_bias_by_perm",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_perm_corr",
        ens_mean + selected_bias,
        "ens_mean + bias_perm",
    )
    train_fitted.append(
        {
            "name": "perm_bias_state",
            "fit_on": "train",
            "perm_map": {str(k): list(perm) for perm, k in perm_to_id.items()},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex18(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_median = stats["median"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    values = df[core_cols].to_numpy(dtype=float)
    sorted_vals = np.sort(values, axis=1)
    out_gap_hi = sorted_vals[:, -1] - sorted_vals[:, 1]
    out_gap_lo = sorted_vals[:, 2] - sorted_vals[:, 0]

    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_values = ctx.train_df[core_cols].to_numpy(dtype=float)
    train_sorted = np.sort(train_values, axis=1)
    train_out_hi = train_sorted[:, -1] - train_sorted[:, 1]
    train_out_lo = train_sorted[:, 2] - train_sorted[:, 0]
    thr_hi_map, thr_hi_default = _station_quantile(ctx.train_df.assign(out_hi=train_out_hi), "out_hi", 0.75)
    thr_lo_map, thr_lo_default = _station_quantile(ctx.train_df.assign(out_lo=train_out_lo), "out_lo", 0.75)
    thr_hi = _map_station_threshold(df, thr_hi_map, thr_hi_default).to_numpy(dtype=float)
    thr_lo = _map_station_threshold(df, thr_lo_map, thr_lo_default).to_numpy(dtype=float)
    is_hi = (out_gap_hi > thr_hi).astype(float)
    is_lo = (out_gap_lo > thr_lo).astype(float)

    hi_id = tfl.argmax_with_tie_break(df, core_cols)
    lo_id = tfl.argmin_with_tie_break(df, core_cols)
    id_map = {col: idx for idx, col in enumerate(core_cols)}
    hi_idx = hi_id.map(id_map).fillna(0).astype(int).to_numpy()
    lo_idx = lo_id.map(id_map).fillna(0).astype(int).to_numpy()

    kappa = 50.0
    bias_hi = np.zeros((len(df), len(core_cols)), dtype=float)
    bias_lo = np.zeros_like(bias_hi)
    count_hi = np.zeros_like(bias_hi)
    count_lo = np.zeros_like(bias_hi)
    for i, col in enumerate(core_cols):
        hi_flag = ((hi_idx == i).astype(float) * is_hi)
        lo_flag = ((lo_idx == i).astype(float) * is_lo)
        bias_hi[:, i] = tfl.rolling_conditional_mean(
            pd.Series(resid, index=df.index),
            pd.Series(hi_flag, index=df.index),
            window=365,
            min_periods=80,
            lag=lag,
            group_key=gk,
        ).to_numpy(dtype=float)
        bias_lo[:, i] = tfl.rolling_conditional_mean(
            pd.Series(resid, index=df.index),
            pd.Series(lo_flag, index=df.index),
            window=365,
            min_periods=80,
            lag=lag,
            group_key=gk,
        ).to_numpy(dtype=float)
        count_hi[:, i] = tfl.rolling_sum(
            pd.Series(hi_flag, index=df.index),
            window=365,
            min_periods=80,
            lag=lag,
            group_key=gk,
        ).to_numpy(dtype=float)
        count_lo[:, i] = tfl.rolling_sum(
            pd.Series(lo_flag, index=df.index),
            window=365,
            min_periods=80,
            lag=lag,
            group_key=gk,
        ).to_numpy(dtype=float)

    selected_bias = np.zeros(len(df), dtype=float)
    outlier_type = np.full(len(df), "none", dtype=object)
    for idx in range(len(df)):
        if is_hi[idx] > 0.0:
            n_val = count_hi[idx, hi_idx[idx]]
            shrink = n_val / (n_val + kappa) if n_val > 0 else 0.0
            selected_bias[idx] = shrink * bias_hi[idx, hi_idx[idx]]
            outlier_type[idx] = "hi"
        elif is_lo[idx] > 0.0:
            n_val = count_lo[idx, lo_idx[idx]]
            shrink = n_val / (n_val + kappa) if n_val > 0 else 0.0
            selected_bias[idx] = shrink * bias_lo[idx, lo_idx[idx]]
            outlier_type[idx] = "lo"
        else:
            selected_bias[idx] = 0.0

    _add_feature(
        features,
        formulas,
        "outlier_selected_bias",
        selected_bias,
        "conditional_outlier_bias",
    )
    _add_feature(
        features,
        formulas,
        "outlier_type_hi",
        (outlier_type == "hi").astype(float),
        "1[hi_outlier]",
    )
    _add_feature(
        features,
        formulas,
        "outlier_type_lo",
        (outlier_type == "lo").astype(float),
        "1[lo_outlier]",
    )
    _add_feature(
        features,
        formulas,
        "outlier_type_none",
        (outlier_type == "none").astype(float),
        "1[no_outlier]",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_outlier_corr",
        ens_mean + selected_bias,
        "ens_mean + outlier_bias",
    )
    train_fitted.append(
        {
            "name": "outlier_thresholds",
            "fit_on": "train",
            "thr_hi": thr_hi_map,
            "thr_lo": thr_lo_map,
            "kappa": kappa,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex19(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_median = stats["median"]
    values = df[core_cols].to_numpy(dtype=float)
    sum_all = np.sum(values, axis=1)
    loo_means = {}
    for i, col in enumerate(core_cols):
        loo_means[col] = (sum_all - values[:, i]) / max(len(core_cols) - 1, 1)
        _add_feature(
            features,
            formulas,
            f"loo_mean_{col}",
            loo_means[col],
            f"mean(models != {col})",
        )
        _add_feature(
            features,
            formulas,
            f"influence_{col}",
            ens_mean - loo_means[col],
            "ens_mean - loo_mean",
        )

    abs_dev = np.abs(values - ens_median[:, None])
    out_id = np.argmax(abs_dev, axis=1)
    outlier_excluded_mean = np.array([loo_means[core_cols[i]][idx] for idx, i in enumerate(out_id)], dtype=float)
    delta_excl = outlier_excluded_mean - ens_mean
    _add_feature(
        features,
        formulas,
        "outlier_excluded_mean",
        outlier_excluded_mean,
        "loo_mean(outlier_id)",
    )
    _add_feature(
        features,
        formulas,
        "delta_excl",
        delta_excl,
        "outlier_excluded_mean - ens_mean",
    )

    y = df["actual_tmax_f"].to_numpy(dtype=float)
    r_excl = y - outlier_excluded_mean
    bias_excl = tfl.ewm_mean(
        pd.Series(r_excl, index=df.index),
        halflife=30,
        min_periods=20,
        lag=lag,
        group_key=gk,
    )
    _add_feature(
        features,
        formulas,
        f"bias_excl_l{lag}",
        bias_excl,
        "ewm_mean(resid_excl)",
    )
    _add_feature(
        features,
        formulas,
        "excl_corr",
        outlier_excluded_mean + bias_excl.to_numpy(dtype=float),
        "outlier_excluded_mean + bias_excl",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex20(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_median = stats["median"]
    values = df[core_cols].to_numpy(dtype=float)
    deviations = np.abs(values - ens_median[:, None])
    tau = 2.0
    weights = np.exp(-deviations / tau)
    weights = _safe_divide(weights, np.sum(weights, axis=1, keepdims=True), default=0.25)
    core_mean = np.sum(weights * values, axis=1)
    entropy = -np.sum(weights * np.log(weights + 1e-12), axis=1) / np.log(len(core_cols))
    concentration = np.max(weights, axis=1)
    eff_n = 1.0 / np.sum(weights**2, axis=1)

    _add_feature(
        features,
        formulas,
        "core_mean",
        core_mean,
        "sum(w * model)",
    )
    _add_feature(
        features,
        formulas,
        "dev_entropy",
        entropy,
        "entropy(weights)",
    )
    _add_feature(
        features,
        formulas,
        "dev_concentration",
        concentration,
        "max(weights)",
    )
    _add_feature(
        features,
        formulas,
        "eff_n",
        eff_n,
        "1/sum(w^2)",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex21(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_ens = y - ens_mean
    resid_by_model = {col: y - df[col].to_numpy(dtype=float) for col in core_cols}

    scale = tfl.ewm_mean(
        pd.Series(np.abs(resid_ens), index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    gamma = 0.05
    weights = np.full((len(df), len(core_cols)), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        log_w = np.full(len(core_cols), np.log(1.0 / len(core_cols)))
        for row_idx in idx:
            b_val = scale[row_idx]
            if np.isfinite(b_val) and b_val > 0:
                for i, col in enumerate(core_cols):
                    r_val = resid_by_model[col][row_idx]
                    if np.isfinite(r_val):
                        log_w[i] = (1.0 - gamma) * log_w[i] + gamma * (-abs(r_val) / b_val)
            weights[row_idx] = _softmax(log_w[None, :])[0]

    weights_df = pd.DataFrame(weights, columns=core_cols, index=df.index)
    weights_l2 = weights_df.groupby(gk).shift(lag).to_numpy(dtype=float)
    f_bma = np.sum(weights_l2 * df[core_cols].to_numpy(dtype=float), axis=1)
    entropy = _rowwise_entropy(weights_l2, eps=1e-12) / np.log(len(core_cols))

    for i, col in enumerate(core_cols):
        _add_feature(
            features,
            formulas,
            f"w_{col}_l{lag}",
            weights_l2[:, i],
            "bma_weight_lagged",
        )
    _add_feature(
        features,
        formulas,
        "f_bma",
        f_bma,
        "sum(w_l2 * model)",
    )
    _add_feature(
        features,
        formulas,
        "w_entropy",
        entropy,
        "entropy(weights_l2)",
    )
    train_fitted.append(
        {
            "name": "bma_laplace",
            "fit_on": "train",
            "gamma": gamma,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex22(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_ens = y - ens_mean
    resid_by_model = {col: y - df[col].to_numpy(dtype=float) for col in core_cols}

    season_id = _season_id(df["month"]).to_numpy(dtype=int)
    train_season = _season_id(ctx.train_df["month"]).to_numpy(dtype=int)
    tau = 1.0
    prior_w = {}
    for season in np.unique(train_season):
        mask = train_season == season
        if mask.sum() == 0:
            continue
        mae = []
        for col in core_cols:
            vals = ctx.train_df.loc[mask, col].to_numpy(dtype=float)
            err = np.abs(ctx.train_df.loc[mask, "actual_tmax_f"].to_numpy(dtype=float) - vals)
            mae.append(float(np.mean(err)) if len(err) else 0.0)
        mae_arr = np.array(mae, dtype=float)
        logits = -mae_arr / tau
        prior = _softmax(logits[None, :])[0]
        prior_w[season] = prior

    scale = tfl.ewm_mean(
        pd.Series(np.abs(resid_ens), index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    gamma = 0.05
    weights = np.full((len(df), len(core_cols)), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        log_w = np.full(len(core_cols), np.log(1.0 / len(core_cols)))
        last_season = None
        for row_idx in idx:
            season = season_id[row_idx]
            if last_season is None or season != last_season:
                prior = prior_w.get(season, np.full(len(core_cols), 1.0 / len(core_cols)))
                log_w = np.log(prior)
                last_season = season
            b_val = scale[row_idx]
            if np.isfinite(b_val) and b_val > 0:
                for i, col in enumerate(core_cols):
                    r_val = resid_by_model[col][row_idx]
                    if np.isfinite(r_val):
                        log_w[i] = (1.0 - gamma) * log_w[i] + gamma * (-abs(r_val) / b_val)
            weights[row_idx] = _softmax(log_w[None, :])[0]

    weights_df = pd.DataFrame(weights, columns=core_cols, index=df.index)
    weights_l2 = weights_df.groupby(gk).shift(lag).to_numpy(dtype=float)
    f_bma = np.sum(weights_l2 * df[core_cols].to_numpy(dtype=float), axis=1)
    for i, col in enumerate(core_cols):
        _add_feature(
            features,
            formulas,
            f"season_prior_w_{col}",
            np.array([prior_w.get(season, np.full(len(core_cols), 1.0 / len(core_cols)))[i] for season in season_id], dtype=float),
            "season_prior_weight",
        )
        _add_feature(
            features,
            formulas,
            f"w_{col}_l{lag}",
            weights_l2[:, i],
            "bma_weight_lagged",
        )
    _add_feature(
        features,
        formulas,
        "f_bma_season",
        f_bma,
        "sum(w_l2 * model)",
    )
    train_fitted.append(
        {
            "name": "season_prior_bma",
            "fit_on": "train",
            "tau": tau,
            "gamma": gamma,
            "prior_weights": {int(k): v.tolist() for k, v in prior_w.items()},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex23(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    ens_range = stats["range"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    deviations = df[core_cols].to_numpy(dtype=float) - ens_mean[:, None]
    obs = np.column_stack([deviations, ens_std, ens_range])

    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_devs = ctx.train_df[core_cols].to_numpy(dtype=float) - train_stats["mean"][:, None]
    train_obs = np.column_stack([train_devs, train_stats["std"], train_stats["range"]])
    scaler = StandardScaler()
    scaler.fit(train_obs)
    obs_scaled = scaler.transform(obs)
    train_scaled = scaler.transform(train_obs)

    p_states = np.full((len(df), 5), np.nan, dtype=float)
    mu_resid = np.zeros(5, dtype=float)
    params = None
    if len(train_scaled) >= 5:
        params = hmm_utils.fit_gaussian_hmm(train_scaled, n_states=5, n_iters=10, seed=ctx.seed)
        stations = df["station_id"].to_numpy()
        for station in np.unique(stations):
            idx = np.where(stations == station)[0]
            alpha = hmm_utils.forward_filter(obs_scaled[idx], params)
            p_states[idx] = alpha

        train_mask = df.index.get_indexer(ctx.train_df.index)
        train_mask = train_mask[train_mask >= 0]
        for k in range(5):
            weights = p_states[train_mask, k]
            r_vals = resid[train_mask]
            mask = np.isfinite(weights) & np.isfinite(r_vals)
            if mask.sum() > 0:
                mu_resid[k] = float(np.sum(weights[mask] * r_vals[mask]) / np.sum(weights[mask]))
            else:
                mu_resid[k] = 0.0
    else:
        p_states[:, :] = 1.0 / 5.0

    bias_hmm = np.sum(p_states * mu_resid[None, :], axis=1)
    entropy = -np.sum(p_states * np.log(p_states + 1e-12), axis=1)

    for k in range(5):
        _add_feature(
            features,
            formulas,
            f"p_state_{k+1}",
            p_states[:, k],
            f"P(state{k})",
        )
    _add_feature(
        features,
        formulas,
        "hmm_entropy",
        entropy,
        "entropy(p_state)",
    )
    _add_feature(
        features,
        formulas,
        "bias_hmm",
        bias_hmm,
        "sum(p_state * mu_resid)",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_hmm_corr",
        ens_mean + bias_hmm,
        "ens_mean + hmm_bias",
    )
    if params is not None:
        train_fitted.append(
            {
                "name": "hmm_disagreement",
                "fit_on": "train",
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "pi": params.pi.tolist(),
                "A": params.A.tolist(),
                "means": params.means.tolist(),
                "covars": params.covars.tolist(),
                "mu_resid": mu_resid.tolist(),
            }
        )
    else:
        train_fitted.append(
            {
                "name": "hmm_disagreement",
                "fit_on": "train",
                "description": "insufficient train data; uniform states",
                "mu_resid": mu_resid.tolist(),
            }
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex24(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    stations = df["station_id"].to_numpy()

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    global_params = None
    if len(train_idx) >= 10:
        global_params = hmm_utils.fit_gaussian_hmm(
            resid[train_idx, None],
            n_states=2,
            n_iters=10,
            seed=ctx.seed,
        )
    p_warm = np.full(len(df), np.nan, dtype=float)
    mu_map: dict[str, tuple[float, float]] = {}
    for station in np.unique(stations):
        mask = ctx.train_df["station_id"] == station
        train_idx = ctx.train_df.index[mask].to_numpy()
        full_idx = df.index.get_indexer(train_idx)
        full_idx = full_idx[full_idx >= 0]
        params = None
        if len(full_idx) >= 300:
            params = hmm_utils.fit_gaussian_hmm(resid[full_idx, None], n_states=2, n_iters=10, seed=ctx.seed)
        elif global_params is not None:
            params = global_params
        if params is None:
            idx = np.where(stations == station)[0]
            p_warm[idx] = 0.5
            mu_map[station] = (0.0, 0.0)
            continue
        alpha = hmm_utils.forward_filter(resid[np.where(stations == station)[0], None], params)
        means = params.means[:, 0]
        warm_state = int(np.argmax(means))
        cold_state = int(1 - warm_state)
        idx = np.where(stations == station)[0]
        p_warm[idx] = alpha[:, warm_state]
        mu_map[station] = (float(means[warm_state]), float(means[cold_state]))

    global_warm = float(global_params.means[:, 0].max()) if global_params is not None else 0.0
    global_cold = float(global_params.means[:, 0].min()) if global_params is not None else 0.0
    mu_warm = np.array([mu_map.get(st, (global_warm, global_cold))[0] for st in stations], dtype=float)
    mu_cold = np.array([mu_map.get(st, (global_warm, global_cold))[1] for st in stations], dtype=float)
    p_warm_l2 = pd.Series(p_warm, index=df.index).groupby(gk).shift(lag)
    bias_l2 = p_warm_l2.to_numpy(dtype=float) * mu_warm + (1.0 - p_warm_l2.to_numpy(dtype=float)) * mu_cold

    _add_feature(
        features,
        formulas,
        f"p_warm_l{lag}",
        p_warm_l2,
        "P(warm_bias_state)",
    )
    _add_feature(
        features,
        formulas,
        f"bias_hmm_l{lag}",
        bias_l2,
        "p_warm*mu_warm + p_cold*mu_cold",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_residhmm_corr",
        ens_mean + bias_l2,
        "ens_mean + hmm_bias",
    )
    if global_params is not None:
        train_fitted.append(
            {
                "name": "resid_hmm",
                "fit_on": "train",
                "global_params": {
                    "pi": global_params.pi.tolist(),
                    "A": global_params.A.tolist(),
                    "means": global_params.means.tolist(),
                    "covars": global_params.covars.tolist(),
                },
            }
        )
    else:
        train_fitted.append(
            {
                "name": "resid_hmm",
                "fit_on": "train",
                "description": "insufficient data fallback",
            }
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex25(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    feature_df = _build_stage1_features(df, core_cols, stats)
    feature_df["ens_std_core"] = ens_std
    X_train = feature_df.loc[ctx.train_df.index].to_numpy(dtype=float)
    y_train = resid[df.index.get_indexer(ctx.train_df.index)]

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=ctx.seed)
    tree.fit(X_train, y_train)
    leaf_id = tree.apply(feature_df.to_numpy(dtype=float))
    train_leaf = leaf_id[df.index.get_indexer(ctx.train_df.index)]
    leaf_mean = {}
    leaf_mae = {}
    for leaf in np.unique(train_leaf):
        mask = train_leaf == leaf
        vals = y_train[mask]
        leaf_mean[int(leaf)] = float(np.mean(vals)) if len(vals) else 0.0
        leaf_mae[int(leaf)] = float(np.mean(np.abs(vals))) if len(vals) else 0.0

    leaf_mean_vals = np.array([leaf_mean.get(int(l), 0.0) for l in leaf_id], dtype=float)
    leaf_mae_vals = np.array([leaf_mae.get(int(l), 0.0) for l in leaf_id], dtype=float)
    _add_feature(
        features,
        formulas,
        "leaf_mean_resid",
        leaf_mean_vals,
        "leaf_mean_resid",
    )
    _add_feature(
        features,
        formulas,
        "leaf_mae",
        leaf_mae_vals,
        "leaf_mae",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_tree_corr",
        ens_mean + leaf_mean_vals,
        "ens_mean + leaf_mean_resid",
    )
    for leaf in sorted(leaf_mean.keys()):
        _add_feature(
            features,
            formulas,
            f"leaf_id_{leaf}",
            (leaf_id == leaf).astype(float),
            f"1[leaf_id == {leaf}]",
        )
    train_fitted.append(
        {
            "name": "regime_tree",
            "fit_on": "train",
            "leaf_mean": leaf_mean,
            "leaf_mae": leaf_mae,
            "features": list(feature_df.columns),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex26(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    feature_df = _build_stage1_features(df, core_cols, stats)
    X_train = feature_df.loc[ctx.train_df.index].to_numpy(dtype=float)
    y_train = resid[df.index.get_indexer(ctx.train_df.index)]

    def _build_model():
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=ctx.seed,
            n_jobs=1,
            verbose=-1,
        )

    oof, model_full = _time_oof_regression(
        ctx.train_df,
        X_train,
        y_train,
        build_model=_build_model,
        n_splits=6,
        gap_days=2,
    )
    pred_full = model_full.predict(feature_df.to_numpy(dtype=float))
    train_pos = df.index.get_indexer(ctx.train_df.index)
    oof_mask = np.isfinite(oof)
    pred_full[train_pos[oof_mask]] = oof[oof_mask]

    _add_feature(
        features,
        formulas,
        "pred_resid_mean_hat",
        pred_full,
        "stage1_resid_hat",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_stage1_corr",
        ens_mean + pred_full,
        "ens_mean + pred_resid",
    )
    _add_feature(
        features,
        formulas,
        "resid_hat_clip",
        np.clip(pred_full, -8.0, 8.0),
        "clip(pred_resid, -8, 8)",
    )
    train_fitted.append(
        {
            "name": "stage1_resid_model",
            "fit_on": "train",
            "features": list(feature_df.columns),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex27(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    target = np.log(np.abs(resid) + 0.5)
    feature_df = _build_stage1_features(df, core_cols, stats)
    X_train = feature_df.loc[ctx.train_df.index].to_numpy(dtype=float)
    y_train = target[df.index.get_indexer(ctx.train_df.index)]

    def _build_model():
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=ctx.seed,
            n_jobs=1,
            verbose=-1,
        )

    oof, model_full = _time_oof_regression(
        ctx.train_df,
        X_train,
        y_train,
        build_model=_build_model,
        n_splits=6,
        gap_days=2,
    )
    pred_full = model_full.predict(feature_df.to_numpy(dtype=float))
    train_pos = df.index.get_indexer(ctx.train_df.index)
    oof_mask = np.isfinite(oof)
    pred_full[train_pos[oof_mask]] = oof[oof_mask]

    pred_abs_err = np.exp(pred_full) - 0.5
    pred_abs_err = np.clip(pred_abs_err, 0.0, 15.0)
    train_pred = pred_abs_err[train_pos]
    train_pred = train_pred[np.isfinite(train_pred)]
    if len(train_pred):
        sorted_train = np.sort(train_pred)
        ranks = np.searchsorted(sorted_train, pred_abs_err, side="right") / len(sorted_train)
    else:
        ranks = np.full(len(df), 0.5, dtype=float)

    _add_feature(
        features,
        formulas,
        "pred_abs_err",
        pred_abs_err,
        "exp(pred_log_abs_err)-0.5",
    )
    _add_feature(
        features,
        formulas,
        "pred_abs_err_rank",
        ranks,
        "percentile(pred_abs_err)",
    )
    _add_feature(
        features,
        formulas,
        "uncertainty_damp",
        1.0 / (1.0 + 0.1 * pred_abs_err),
        "1/(1+0.1*pred_abs_err)",
    )
    train_fitted.append(
        {
            "name": "stage1_scale_model",
            "fit_on": "train",
            "features": list(feature_df.columns),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex28(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    feature_df = _build_stage1_features(df, core_cols, stats)
    X_train = feature_df.loc[ctx.train_df.index].to_numpy(dtype=float)
    y_train = y[df.index.get_indexer(ctx.train_df.index)]

    def _build_quantile(alpha: float):
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=ctx.seed,
            n_jobs=1,
            verbose=-1,
        )

    q_preds = {}
    for alpha in [0.25, 0.5, 0.75]:
        oof, model_full = _time_oof_regression(
            ctx.train_df,
            X_train,
            y_train,
            build_model=lambda a=alpha: _build_quantile(a),
            n_splits=6,
            gap_days=2,
        )
        pred_full = model_full.predict(feature_df.to_numpy(dtype=float))
        train_pos = df.index.get_indexer(ctx.train_df.index)
        oof_mask = np.isfinite(oof)
        pred_full[train_pos[oof_mask]] = oof[oof_mask]
        q_preds[alpha] = pred_full

    q25 = q_preds[0.25]
    q50 = q_preds[0.5]
    q75 = q_preds[0.75]
    q25 = np.minimum(q25, q50)
    q75 = np.maximum(q75, q50)

    _add_feature(
        features,
        formulas,
        "y_q25_hat",
        q25,
        "quantile_regression_q25",
    )
    _add_feature(
        features,
        formulas,
        "y_q50_hat",
        q50,
        "quantile_regression_q50",
    )
    _add_feature(
        features,
        formulas,
        "y_q75_hat",
        q75,
        "quantile_regression_q75",
    )
    _add_feature(
        features,
        formulas,
        "width_50",
        q75 - q25,
        "q75 - q25",
    )
    _add_feature(
        features,
        formulas,
        "skew_mid",
        (q75 + q25) / 2.0 - q50,
        "midpoint - q50",
    )
    _add_feature(
        features,
        formulas,
        "y_q50_minus_ensmean",
        q50 - ens_mean,
        "q50 - ens_mean",
    )
    train_fitted.append(
        {
            "name": "quantile_models",
            "fit_on": "train",
            "features": list(feature_df.columns),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex29(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    feature_df = _build_stage1_features(df, core_cols, stats)
    X_train = feature_df.loc[ctx.train_df.index].to_numpy(dtype=float)
    y_train = resid[df.index.get_indexer(ctx.train_df.index)]

    def _build_model():
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=ctx.seed,
            n_jobs=1,
            verbose=-1,
        )

    oof, model_full = _time_oof_regression(
        ctx.train_df,
        X_train,
        y_train,
        build_model=_build_model,
        n_splits=6,
        gap_days=2,
    )
    pred_full = model_full.predict(feature_df.to_numpy(dtype=float))
    train_pos = df.index.get_indexer(ctx.train_df.index)
    oof_mask = np.isfinite(oof)
    pred_full[train_pos[oof_mask]] = oof[oof_mask]
    a_train = np.abs(y_train - pred_full[train_pos])

    train_std = stats["std"][train_pos]
    spread_edges = _quantile_edges(train_std, 5)
    train_bins = _bin_ids(train_std, spread_edges)
    q90_by_bin = {}
    for b in range(len(spread_edges) + 1):
        vals = a_train[train_bins == b]
        q90_by_bin[b] = float(np.quantile(vals, 0.90)) if len(vals) else float(np.quantile(a_train, 0.90))

    bins_all = _bin_ids(ens_std, spread_edges)
    q90 = np.array([q90_by_bin.get(b, q90_by_bin.get(0, 0.0)) for b in bins_all], dtype=float)
    _add_feature(
        features,
        formulas,
        "pi90_halfwidth",
        q90,
        "q90(|r - r_hat|)",
    )
    _add_feature(
        features,
        formulas,
        "pi90_width",
        2.0 * q90,
        "2*pi90_halfwidth",
    )
    _add_feature(
        features,
        formulas,
        "pi90_over_spread",
        q90 / (ens_std + EPS),
        "pi90_halfwidth/spread",
    )
    train_fitted.append(
        {
            "name": "conformal_q90",
            "fit_on": "train",
            "spread_edges": spread_edges.tolist(),
            "q90_by_bin": q90_by_bin,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex30(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    doy = df["day_of_year"].to_numpy(dtype=int)
    doy = np.where(doy > 365, 365, doy)

    train_df = ctx.train_df
    train_y = train_df["actual_tmax_f"].to_numpy(dtype=float)
    train_doy = train_df["day_of_year"].to_numpy(dtype=int)
    train_doy = np.where(train_doy > 365, 365, train_doy)
    global_mean = float(np.nanmean(train_y)) if len(train_y) else 0.0
    global_p10 = float(np.nanquantile(train_y, 0.10)) if len(train_y) else 0.0
    global_p90 = float(np.nanquantile(train_y, 0.90)) if len(train_y) else 0.0
    global_mean_raw = np.full(365, global_mean, dtype=float)
    global_p10_raw = np.full(365, global_p10, dtype=float)
    global_p90_raw = np.full(365, global_p90, dtype=float)
    for d in range(1, 366):
        vals = train_y[train_doy == d]
        if len(vals):
            global_mean_raw[d - 1] = float(np.nanmean(vals))
            global_p10_raw[d - 1] = float(np.nanquantile(vals, 0.10))
            global_p90_raw[d - 1] = float(np.nanquantile(vals, 0.90))
    global_mean_sm = _circular_smooth(global_mean_raw, bandwidth=15)
    global_p10_sm = _circular_smooth(global_p10_raw, bandwidth=15)
    global_p90_sm = _circular_smooth(global_p90_raw, bandwidth=15)

    station_maps = {}
    for station in df["station_id"].unique():
        mask = train_df["station_id"] == station
        if mask.sum() < 200:
            continue
        vals = train_y[mask.to_numpy()]
        doy_vals = train_doy[mask.to_numpy()]
        mean_raw = np.full(365, float(np.nanmean(vals)), dtype=float)
        p10_raw = np.full(365, float(np.nanquantile(vals, 0.10)), dtype=float)
        p90_raw = np.full(365, float(np.nanquantile(vals, 0.90)), dtype=float)
        for d in range(1, 366):
            day_vals = vals[doy_vals == d]
            if len(day_vals):
                mean_raw[d - 1] = float(np.nanmean(day_vals))
                p10_raw[d - 1] = float(np.nanquantile(day_vals, 0.10))
                p90_raw[d - 1] = float(np.nanquantile(day_vals, 0.90))
        station_maps[station] = (
            _circular_smooth(mean_raw, bandwidth=15),
            _circular_smooth(p10_raw, bandwidth=15),
            _circular_smooth(p90_raw, bandwidth=15),
        )

    clim_mean = np.full(len(df), np.nan, dtype=float)
    clim_p10 = np.full(len(df), np.nan, dtype=float)
    clim_p90 = np.full(len(df), np.nan, dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        if station in station_maps:
            mean_vec, p10_vec, p90_vec = station_maps[station]
        else:
            mean_vec, p10_vec, p90_vec = global_mean_sm, global_p10_sm, global_p90_sm
        clim_mean[idx] = mean_vec[doy[idx] - 1]
        clim_p10[idx] = p10_vec[doy[idx] - 1]
        clim_p90[idx] = p90_vec[doy[idx] - 1]

    denom = np.maximum(clim_p90 - clim_p10, 2.0)
    ens_mean_anom = ens_mean - clim_mean
    ens_mean_anom_pctl = (ens_mean - clim_p10) / (denom + EPS)
    model_anom_std = np.std(df[core_cols].to_numpy(dtype=float) - clim_mean[:, None], axis=1)

    _add_feature(
        features,
        formulas,
        "clim_mean",
        clim_mean,
        "climatology_mean",
    )
    _add_feature(
        features,
        formulas,
        "clim_p10",
        clim_p10,
        "climatology_p10",
    )
    _add_feature(
        features,
        formulas,
        "clim_p90",
        clim_p90,
        "climatology_p90",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_anom",
        ens_mean_anom,
        "ens_mean - clim_mean",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_anom_pctl",
        ens_mean_anom_pctl,
        "(ens_mean - clim_p10)/(clim_p90-clim_p10)",
    )
    _add_feature(
        features,
        formulas,
        "model_anom_std",
        model_anom_std,
        "std(model - clim_mean)",
    )
    train_fitted.append(
        {
            "name": "doy_climatology",
            "fit_on": "train",
            "global_mean": global_mean_sm.tolist(),
            "global_p10": global_p10_sm.tolist(),
            "global_p90": global_p90_sm.tolist(),
            "per_station": {k: [v[0].tolist(), v[1].tolist(), v[2].tolist()] for k, v in station_maps.items()},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex31(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"]

    y_l2 = y.groupby(gk).shift(lag)
    y_l3 = y.groupby(gk).shift(lag + 1)
    y_mean7 = tfl.rolling_mean(y, window=7, min_periods=5, lag=lag, group_key=gk)
    y_std7 = tfl.rolling_std(y, window=7, min_periods=5, lag=lag, group_key=gk)
    y_slope7 = tfl.rolling_slope(y, window=7, min_periods=5, lag=lag, group_key=gk)

    persistence_gap = ens_mean - y_l2.to_numpy(dtype=float)
    trend_extrap = y_l2.to_numpy(dtype=float) + 2.0 * y_slope7.to_numpy(dtype=float)
    trend_gap = ens_mean - trend_extrap

    _add_feature(
        features,
        formulas,
        f"y_l{lag}",
        y_l2,
        "y(t-2)",
    )
    _add_feature(
        features,
        formulas,
        f"y_l{lag+1}",
        y_l3,
        "y(t-3)",
    )
    _add_feature(
        features,
        formulas,
        f"y_mean7_l{lag}",
        y_mean7,
        "roll_mean(y,7)",
    )
    _add_feature(
        features,
        formulas,
        f"y_std7_l{lag}",
        y_std7,
        "roll_std(y,7)",
    )
    _add_feature(
        features,
        formulas,
        f"y_slope7_l{lag}",
        y_slope7,
        "roll_slope(y,7)",
    )
    _add_feature(
        features,
        formulas,
        "persistence_gap",
        persistence_gap,
        "ens_mean - y_l2",
    )
    _add_feature(
        features,
        formulas,
        "trend_extrap",
        trend_extrap,
        "y_l2 + 2*y_slope7",
    )
    _add_feature(
        features,
        formulas,
        "trend_gap",
        trend_gap,
        "ens_mean - trend_extrap",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex32(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    doy = df["day_of_year"].to_numpy(dtype=int)
    doy = np.where(doy > 365, 365, doy)

    train_df = ctx.train_df
    train_y = train_df["actual_tmax_f"].to_numpy(dtype=float)
    train_doy = train_df["day_of_year"].to_numpy(dtype=int)
    train_doy = np.where(train_doy > 365, 365, train_doy)
    global_mean = float(np.nanmean(train_y)) if len(train_y) else 0.0
    global_raw = np.full(365, global_mean, dtype=float)
    for d in range(1, 366):
        vals = train_y[train_doy == d]
        if len(vals):
            global_raw[d - 1] = float(np.nanmean(vals))
    global_mean_sm = _circular_smooth(global_raw, bandwidth=15)

    station_maps = {}
    for station in df["station_id"].unique():
        mask = train_df["station_id"] == station
        if mask.sum() < 200:
            continue
        vals = train_y[mask.to_numpy()]
        doy_vals = train_doy[mask.to_numpy()]
        mean_raw = np.full(365, float(np.nanmean(vals)), dtype=float)
        for d in range(1, 366):
            day_vals = vals[doy_vals == d]
            if len(day_vals):
                mean_raw[d - 1] = float(np.nanmean(day_vals))
        station_maps[station] = _circular_smooth(mean_raw, bandwidth=15)

    clim_mean = np.full(len(df), np.nan, dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        mean_vec = station_maps.get(station, global_mean_sm)
        clim_mean[idx] = mean_vec[doy[idx] - 1]

    ens_mean_anom = ens_mean - clim_mean
    train_anom = ens_mean_anom[df.index.get_indexer(ctx.train_df.index)]
    edges = np.quantile(train_anom, [0.2, 0.4, 0.6, 0.8]) if len(train_anom) else np.array([])
    bins = _bin_ids(ens_mean_anom, edges)
    kappa = 50.0
    mu_global = float(np.nanmean(resid)) if len(resid) else 0.0
    bias_vals = np.zeros(len(df), dtype=float)
    for b in range(len(edges) + 1):
        mask = bins == b
        r_vals = resid[mask]
        n = np.isfinite(r_vals).sum()
        mu_b = float(np.nanmean(r_vals)) if n else mu_global
        shrink = n / (n + kappa) if n > 0 else 0.0
        bias_vals[mask] = mu_global + shrink * (mu_b - mu_global)

    _add_feature(
        features,
        formulas,
        "bias_anom_bin",
        bias_vals,
        "bias_by_anomaly_bin",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_anomcorr",
        ens_mean + bias_vals,
        "ens_mean + bias_anom",
    )
    train_fitted.append(
        {
            "name": "anom_bin_bias",
            "fit_on": "train",
            "edges": edges.tolist(),
            "kappa": kappa,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex33(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    y_l2 = df["actual_tmax_f"].groupby(gk).shift(lag).to_numpy(dtype=float)
    delta_pred = ens_mean - y_l2
    delta_pred = np.clip(delta_pred, -20.0, 20.0)

    edges = np.array([-np.inf, -8, -4, -2, 2, 4, 8, np.inf], dtype=float)
    bin_id = np.digitize(delta_pred, edges) - 1
    mu_global = float(np.nanmean(resid)) if len(resid) else 0.0
    bias_vals = np.zeros(len(df), dtype=float)
    for b in range(len(edges) - 1):
        mask = bin_id == b
        train_mask = mask & df.index.isin(ctx.train_df.index)
        r_vals = resid[train_mask]
        mu_b = float(np.nanmean(r_vals)) if len(r_vals) else mu_global
        bias_vals[mask] = mu_b

    _add_feature(
        features,
        formulas,
        "delta_pred",
        delta_pred,
        "ens_mean - y_l2",
    )
    _add_feature(
        features,
        formulas,
        "abs_delta_pred",
        np.abs(delta_pred),
        "|delta_pred|",
    )
    _add_feature(
        features,
        formulas,
        "bias_delta_bin",
        bias_vals,
        "bias_by_delta_bin",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_deltacorr",
        ens_mean + bias_vals,
        "ens_mean + bias_delta_bin",
    )
    train_fitted.append(
        {
            "name": "delta_pred_bins",
            "fit_on": "train",
            "edges": edges.tolist(),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex34(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    season_id = _season_id(df["month"]).to_numpy(dtype=int)
    p_grid = np.arange(0.05, 1.0, 0.05)

    train_df = ctx.train_df
    train_season = _season_id(train_df["month"]).to_numpy(dtype=int)
    train_y = train_df["actual_tmax_f"].to_numpy(dtype=float)
    qy_station = {}
    for station in train_df["station_id"].unique():
        mask_station = train_df["station_id"] == station
        if mask_station.sum() < 200:
            continue
        q_by_season = {}
        for season in np.unique(train_season):
            mask = mask_station & (train_season == season)
            vals = train_y[mask.to_numpy()]
            if len(vals) >= 20:
                q_by_season[int(season)] = np.quantile(vals, p_grid).tolist()
        if q_by_season:
            qy_station[station] = q_by_season

    qy_global = {}
    for season in np.unique(train_season):
        mask = train_season == season
        vals = train_y[mask]
        if len(vals):
            qy_global[int(season)] = np.quantile(vals, p_grid).tolist()

    forecast_pctl = np.full(len(df), 0.5, dtype=float)
    for idx, row in df.iterrows():
        station = row["station_id"]
        season = int(season_id[df.index.get_loc(idx)])
        q_vals = None
        if station in qy_station and season in qy_station[station]:
            q_vals = np.array(qy_station[station][season], dtype=float)
        elif season in qy_global:
            q_vals = np.array(qy_global[season], dtype=float)
        if q_vals is not None:
            forecast_pctl[df.index.get_loc(idx)] = _interp_percentile(
                np.array([ens_mean[df.index.get_loc(idx)]]),
                q_vals,
                p_grid,
            )[0]

    bins = np.clip((forecast_pctl * 10).astype(int), 0, 9)
    bias_vals = np.zeros(len(df), dtype=float)
    for b in range(10):
        mask = bins == b
        train_mask = mask & df.index.isin(ctx.train_df.index)
        vals = resid[train_mask]
        bias_vals[mask] = float(np.nanmean(vals)) if len(vals) else 0.0

    _add_feature(
        features,
        formulas,
        "forecast_pctl",
        forecast_pctl,
        "percentile_of_ens_mean",
    )
    _add_feature(
        features,
        formulas,
        "bias_pctl",
        bias_vals,
        "bias_by_forecast_pctl",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_pctlcorr",
        ens_mean + bias_vals,
        "ens_mean + bias_pctl",
    )
    train_fitted.append(
        {
            "name": "forecast_pctl_bias",
            "fit_on": "train",
            "p_grid": p_grid.tolist(),
            "qy_station": qy_station,
            "qy_global": qy_global,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex35(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    abs_err = np.abs(y - ens_mean)

    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_std = train_stats["std"]
    train_abs_err = np.abs(ctx.train_df["actual_tmax_f"].to_numpy(dtype=float) - train_stats["mean"])

    global_iso = IsotonicRegression(out_of_bounds="clip")
    global_iso.fit(train_std, train_abs_err)
    iso_models = {}
    station_counts = ctx.train_df["station_id"].value_counts()
    for station, count in station_counts.items():
        if count < 400:
            continue
        mask = ctx.train_df["station_id"] == station
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_std[mask.to_numpy()], train_abs_err[mask.to_numpy()])
        iso_models[station] = iso

    pred_station = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        model = iso_models.get(station)
        if model is None:
            pred_station[idx] = global_iso.predict(ens_std[idx])
        else:
            pred_station[idx] = model.predict(ens_std[idx])

    pred_global = global_iso.predict(ens_std)
    n_station = df["station_id"].map(station_counts).fillna(0).to_numpy(dtype=float)
    kappa = 400.0
    w = n_station / (n_station + kappa)
    pred_abs_err_station = w * pred_station + (1.0 - w) * pred_global

    _add_feature(
        features,
        formulas,
        "pred_abs_err_station",
        pred_abs_err_station,
        "station_shrunk_iso",
    )
    _add_feature(
        features,
        formulas,
        "station_reliability_ratio",
        pred_abs_err_station / (ens_std + EPS),
        "pred_abs_err_station / spread",
    )
    train_fitted.append(
        {
            "name": "station_spread_reliability",
            "fit_on": "train",
            "global_x": global_iso.X_thresholds_.tolist(),
            "global_y": global_iso.y_thresholds_.tolist(),
            "per_station": {
                station: {
                    "x": iso.X_thresholds_.tolist(),
                    "y": iso.y_thresholds_.tolist(),
                }
                for station, iso in iso_models.items()
            },
            "kappa": kappa,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex36(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_mat = np.column_stack([y - df[col].to_numpy(dtype=float) for col in core_cols])

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    train_resid = resid_mat[train_idx]
    mean = np.nanmean(train_resid, axis=0)
    std = np.nanstd(train_resid, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    resid_std = (resid_mat - mean[None, :]) / std[None, :]

    pca = PCA(n_components=2, random_state=ctx.seed)
    pca.fit(resid_std[train_idx])
    scores = pca.transform(resid_std)
    s1 = scores[:, 0]
    s2 = scores[:, 1]
    s1_state = tfl.ewm_mean(
        pd.Series(s1, index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    s2_state = tfl.ewm_mean(
        pd.Series(s2, index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    s1_l2 = pd.Series(s1_state, index=df.index).groupby(gk).shift(lag)
    s2_l2 = pd.Series(s2_state, index=df.index).groupby(gk).shift(lag)

    _add_feature(
        features,
        formulas,
        f"s1_state_l{lag}",
        s1_l2,
        "ewm_mean(pca1)",
    )
    _add_feature(
        features,
        formulas,
        f"s2_state_l{lag}",
        s2_l2,
        "ewm_mean(pca2)",
    )
    loadings = pca.components_
    for idx, col in enumerate(core_cols):
        proj = loadings[0, idx] * s1_l2.to_numpy(dtype=float) + loadings[1, idx] * s2_l2.to_numpy(dtype=float)
        _add_feature(
            features,
            formulas,
            f"proj_bias_{col}",
            proj,
            "loading1*s1 + loading2*s2",
        )
    train_fitted.append(
        {
            "name": "resid_pca",
            "fit_on": "train",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "components": pca.components_.tolist(),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex37(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_mat = np.column_stack([y - df[col].to_numpy(dtype=float) for col in core_cols])
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    train_resid = resid_mat[train_idx]
    global_cov = np.cov(train_resid, rowvar=False, ddof=0) if len(train_resid) else np.eye(len(core_cols))
    if global_cov.shape != (len(core_cols), len(core_cols)):
        global_cov = np.eye(len(core_cols))

    rho = 0.3
    weights = np.full((len(df), len(core_cols)), np.nan, dtype=float)
    eff_n = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        for pos, row_idx in enumerate(idx):
            end = pos - lag
            start = end - 180 + 1
            if end < 0:
                continue
            window_idx = idx[max(start, 0) : end + 1]
            window_resid = resid_mat[window_idx]
            mask = np.isfinite(window_resid).all(axis=1)
            if mask.sum() < 120:
                cov = global_cov
            else:
                cov = np.cov(window_resid[mask], rowvar=False, ddof=0)
                if cov.shape != (len(core_cols), len(core_cols)):
                    cov = global_cov
            cov_tilde = rho * global_cov + (1.0 - rho) * cov
            cov_tilde = cov_tilde + np.eye(len(core_cols)) * 1e-3
            try:
                inv = np.linalg.inv(cov_tilde)
                ones = np.ones(len(core_cols), dtype=float)
                denom = float(ones.T @ inv @ ones)
                if denom <= 0:
                    w = np.full(len(core_cols), 1.0 / len(core_cols))
                else:
                    w = (inv @ ones) / denom
            except np.linalg.LinAlgError:
                w = np.full(len(core_cols), 1.0 / len(core_cols))
            weights[row_idx] = w
            eff_n[row_idx] = 1.0 / np.sum(w**2)

    f_gls = np.sum(weights * df[core_cols].to_numpy(dtype=float), axis=1)
    for i, col in enumerate(core_cols):
        _add_feature(
            features,
            formulas,
            f"w_gls_{col}",
            weights[:, i],
            "gls_weight",
        )
    _add_feature(
        features,
        formulas,
        "f_gls",
        f_gls,
        "sum(w_gls * model)",
    )
    _add_feature(
        features,
        formulas,
        "eff_n_gls",
        eff_n,
        "1/sum(w_gls^2)",
    )
    train_fitted.append(
        {
            "name": "gls_covariance",
            "fit_on": "train",
            "rho": rho,
            "global_cov": global_cov.tolist(),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex38(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    fallback = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=30,
        min_periods=20,
        lag=lag,
        group_key=gk,
    ).to_numpy(dtype=float)
    mom_bias = np.full(len(df), np.nan, dtype=float)
    mom_spread = np.full(len(df), np.nan, dtype=float)
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        for pos, row_idx in enumerate(idx):
            end = pos - lag
            start = end - 60 + 1
            if end < 0 or start < 0:
                continue
            window_idx = idx[start : end + 1]
            window_vals = resid[window_idx]
            valid = window_vals[np.isfinite(window_vals)]
            if len(valid) < 40:
                mom_bias[row_idx] = fallback[row_idx]
                continue
            blocks = []
            for b in range(6):
                block_vals = window_vals[b * 10 : (b + 1) * 10]
                block_vals = block_vals[np.isfinite(block_vals)]
                if len(block_vals):
                    blocks.append(float(np.mean(block_vals)))
            if blocks:
                mom_bias[row_idx] = float(np.median(blocks))
                mom_spread[row_idx] = float(np.quantile(blocks, 0.75) - np.quantile(blocks, 0.25))
            else:
                mom_bias[row_idx] = fallback[row_idx]
    _add_feature(
        features,
        formulas,
        f"mom_bias_l{lag}",
        mom_bias,
        "median(block_means)",
    )
    _add_feature(
        features,
        formulas,
        f"mom_spread_l{lag}",
        mom_spread,
        "iqr(block_means)",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_mom_corr",
        ens_mean + mom_bias,
        "ens_mean + mom_bias",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex39(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"]
    resid = y - ens_mean

    med_bias = _rolling_median(resid, window=90, lag=lag, group_key=gk)
    mad = 1.4826 * _rolling_mad(resid, window=90, lag=lag, group_key=gk)
    mad_vals = mad.to_numpy(dtype=float)
    mad_vals = np.where(mad_vals < 0.5, 0.5, mad_vals)
    win_bias = np.clip(med_bias.to_numpy(dtype=float), -3.0 * mad_vals, 3.0 * mad_vals)

    _add_feature(
        features,
        formulas,
        f"med_bias_l{lag}",
        med_bias,
        "roll_median(resid)",
    )
    _add_feature(
        features,
        formulas,
        f"mad_l{lag}",
        mad,
        "1.4826*roll_median(|resid-median|)",
    )
    _add_feature(
        features,
        formulas,
        f"win_bias_l{lag}",
        win_bias,
        "clip(med_bias, +-3*mad)",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_medmad_corr",
        ens_mean + win_bias,
        "ens_mean + win_bias",
    )
    _add_feature(
        features,
        formulas,
        "bias_z_medmad",
        win_bias / (mad_vals + EPS),
        "win_bias/mad",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex40(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    season_id = _season_id(df["month"]).to_numpy(dtype=int)
    train_stats = _core_ensemble_stats(ctx.train_df, core_cols)
    train_mean = train_stats["mean"]
    train_season = _season_id(ctx.train_df["month"]).to_numpy(dtype=int)
    train_y = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float)
    train_resid = train_y - train_mean
    p_grid = np.arange(0.01, 1.0, 0.01)

    q_f = {}
    for season in np.unique(train_season):
        mask = train_season == season
        if mask.sum() > 0:
            q_f[season] = np.quantile(train_mean[mask], p_grid)

    tail_bias = np.zeros(len(df), dtype=float)
    tail_flag = np.full(len(df), "mid", dtype=object)
    for season in np.unique(season_id):
        mask_all = season_id == season
        q_vals = q_f.get(season, np.quantile(train_mean, p_grid))
        p_hat = _interp_percentile(ens_mean[mask_all], q_vals, p_grid)
        cold = p_hat < 0.10
        warm = p_hat > 0.90
        tail_flag[mask_all] = np.where(cold, "cold", np.where(warm, "warm", "mid"))
        train_mask = (train_season == season)
        if train_mask.sum() > 0:
            p_train = _interp_percentile(train_mean[train_mask], q_vals, p_grid)
            mu_cold = float(np.mean(train_resid[train_mask][p_train < 0.10])) if np.any(p_train < 0.10) else 0.0
            mu_warm = float(np.mean(train_resid[train_mask][p_train > 0.90])) if np.any(p_train > 0.90) else 0.0
            mu_mid = float(np.mean(train_resid[train_mask][(p_train >= 0.10) & (p_train <= 0.90)])) if np.any((p_train >= 0.10) & (p_train <= 0.90)) else 0.0
        else:
            mu_cold = mu_warm = mu_mid = 0.0
        tail_bias[mask_all & (tail_flag == "cold")] = mu_cold
        tail_bias[mask_all & (tail_flag == "warm")] = mu_warm
        tail_bias[mask_all & (tail_flag == "mid")] = mu_mid

    _add_feature(
        features,
        formulas,
        "tail_bias",
        tail_bias,
        "bias_by_forecast_tail",
    )
    _add_feature(
        features,
        formulas,
        "tail_flag_cold",
        (tail_flag == "cold").astype(float),
        "1[cold_tail]",
    )
    _add_feature(
        features,
        formulas,
        "tail_flag_warm",
        (tail_flag == "warm").astype(float),
        "1[warm_tail]",
    )
    _add_feature(
        features,
        formulas,
        "tail_flag_mid",
        (tail_flag == "mid").astype(float),
        "1[mid]",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_tailcorr",
        ens_mean + tail_bias,
        "ens_mean + tail_bias",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex41(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    abs_err = np.abs(y - ens_mean)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    train_std = ens_std[train_idx]
    train_abs = abs_err[train_idx]
    spread_edges = np.quantile(train_std, [0.33, 0.66]) if len(train_std) else np.array([0.0, 0.0])
    train_bins = _bin_ids(train_std, spread_edges)

    params = {}
    for b in range(3):
        vals = train_abs[train_bins == b]
        if len(vals) == 0:
            params[b] = {"u": 0.0, "xi": 0.0, "beta": 1.0, "emp_tail": 0.0}
            continue
        u = float(np.quantile(vals, 0.90))
        exceed = vals[vals > u] - u
        if len(exceed) < 20:
            emp_tail = float(np.mean(vals > 5.0))
            params[b] = {"u": u, "xi": 0.0, "beta": float(np.mean(exceed)) if len(exceed) else 1.0, "emp_tail": emp_tail}
            continue
        try:
            xi, loc, beta = genpareto.fit(exceed, floc=0.0)
            if xi >= 0.9 or beta <= 0:
                raise ValueError("unstable gpd")
            emp_tail = float(np.mean(vals > 5.0))
            params[b] = {"u": u, "xi": float(xi), "beta": float(beta), "emp_tail": emp_tail}
        except Exception:
            emp_tail = float(np.mean(vals > 5.0))
            params[b] = {"u": u, "xi": 0.0, "beta": float(np.mean(exceed)) if len(exceed) else 1.0, "emp_tail": emp_tail}

    bins_all = _bin_ids(ens_std, spread_edges)
    tail_prob = np.zeros(len(df), dtype=float)
    expected_excess = np.zeros(len(df), dtype=float)
    u_vals = np.zeros(len(df), dtype=float)
    for idx, b in enumerate(bins_all):
        p = params.get(b, params.get(0, {"u": 0.0, "xi": 0.0, "beta": 1.0, "emp_tail": 0.0}))
        u = p["u"]
        xi = p["xi"]
        beta = p["beta"]
        u_vals[idx] = u
        if 5.0 <= u:
            tail_prob[idx] = 1.0
        else:
            if xi == 0:
                tail_prob[idx] = float(np.exp(-(5.0 - u) / max(beta, EPS)))
            else:
                tail_prob[idx] = float((1.0 + xi * (5.0 - u) / max(beta, EPS)) ** (-1.0 / xi))
        expected_excess[idx] = float(beta / (1.0 - xi)) if xi < 1.0 else 0.0

    _add_feature(
        features,
        formulas,
        "tail_prob_5f",
        tail_prob,
        "P(|err| > 5)",
    )
    _add_feature(
        features,
        formulas,
        "expected_excess",
        expected_excess,
        "E[abs_err-u | abs_err>u]",
    )
    _add_feature(
        features,
        formulas,
        "u_b",
        u_vals,
        "gpd_threshold",
    )
    train_fitted.append(
        {
            "name": "gpd_tail",
            "fit_on": "train",
            "spread_edges": spread_edges.tolist(),
            "params": params,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex42(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    d = df[core_cols].to_numpy(dtype=float) - ens_mean[:, None]
    norm = np.linalg.norm(d, axis=1)
    d_norm = d / (norm[:, None] + EPS)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    kmeans = KMeans(n_clusters=8, random_state=ctx.seed, n_init=10)
    kmeans.fit(d_norm[train_idx])
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + EPS)
    labels = kmeans.predict(d_norm[train_idx])
    mu_k = np.zeros(8, dtype=float)
    for k in range(8):
        vals = resid[train_idx][labels == k]
        mu_k[k] = float(np.mean(vals)) if len(vals) else 0.0

    sim = d_norm @ centroids.T
    p_k = _softmax(sim / 0.2, axis=1)
    entropy = -np.sum(p_k * np.log(p_k + 1e-12), axis=1)
    bias_arch = np.sum(p_k * mu_k[None, :], axis=1)

    for k in range(8):
        _add_feature(
            features,
            formulas,
            f"p_arch_{k+1}",
            p_k[:, k],
            f"P(arch{k})",
        )
    _add_feature(
        features,
        formulas,
        "arche_entropy",
        entropy,
        "entropy(p_arch)",
    )
    _add_feature(
        features,
        formulas,
        "bias_arch",
        bias_arch,
        "sum(p_arch * mu_k)",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_arch_corr",
        ens_mean + bias_arch,
        "ens_mean + bias_arch",
    )
    train_fitted.append(
        {
            "name": "archetype_kmeans",
            "fit_on": "train",
            "centroids": centroids.tolist(),
            "mu_k": mu_k.tolist(),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex43(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_std = stats["std"]
    x = np.column_stack([df[col].to_numpy(dtype=float) for col in core_cols] + [ens_std])

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    scaler = StandardScaler()
    scaler.fit(x[train_idx])
    x_std = scaler.transform(x)
    pca = PCA(n_components=2, random_state=ctx.seed)
    pca.fit(x_std[train_idx])
    x_hat = pca.inverse_transform(pca.transform(x_std))
    recon_err = np.linalg.norm(x_std - x_hat, axis=1)
    recon_mean = float(np.mean(recon_err[train_idx])) if len(train_idx) else 0.0
    recon_std = float(np.std(recon_err[train_idx], ddof=0)) if len(train_idx) else 1.0
    recon_z = (recon_err - recon_mean) / (recon_std + EPS)

    _add_feature(
        features,
        formulas,
        "recon_err",
        recon_err,
        "pca_recon_error",
    )
    _add_feature(
        features,
        formulas,
        "recon_err_z",
        recon_z,
        "(recon_err - mean)/std",
    )
    for i, col in enumerate(core_cols):
        _add_feature(
            features,
            formulas,
            f"recon_resid_{col}",
            x_std[:, i] - x_hat[:, i],
            "x_std - x_hat",
        )
    _add_feature(
        features,
        formulas,
        "recon_resid_spread",
        x_std[:, -1] - x_hat[:, -1],
        "spread_std_resid",
    )
    train_fitted.append(
        {
            "name": "pca_recon",
            "fit_on": "train",
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "components": pca.components_.tolist(),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex44(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    values = df[core_cols].to_numpy(dtype=float)
    di = np.abs(values - ens_mean[:, None])
    denom = np.sum(di, axis=1) + EPS
    pi = di / denom[:, None]
    pi = np.where(np.sum(di, axis=1)[:, None] > 0, pi, 1.0 / len(core_cols))
    entropy = -np.sum(pi * np.log(pi + 1e-12), axis=1) / np.log(len(core_cols))
    concentration = np.max(pi, axis=1)
    gini = 1.0 - np.sum(pi**2, axis=1)
    one_bad = (concentration > 0.6).astype(float)

    _add_feature(
        features,
        formulas,
        "disagree_entropy",
        entropy,
        "entropy(pi)",
    )
    _add_feature(
        features,
        formulas,
        "disagree_concentration",
        concentration,
        "max(pi)",
    )
    _add_feature(
        features,
        formulas,
        "disagree_gini",
        gini,
        "1 - sum(pi^2)",
    )
    _add_feature(
        features,
        formulas,
        "one_bad_model_flag",
        one_bad,
        "1[concentration>0.6]",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex45(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = {col: y - df[col].to_numpy(dtype=float) for col in core_cols}

    def _rd_bias(col_i: str, col_j: str) -> np.ndarray:
        rd = resid[col_i] - resid[col_j]
        return tfl.ewm_mean(
            pd.Series(rd, index=df.index),
            halflife=60,
            min_periods=30,
            lag=lag,
            group_key=gk,
        ).to_numpy(dtype=float)

    pairs = [
        ("hrrr_tmax_f", "nbm_tmax_f"),
        ("rap_tmax_f", "gefsatmosmean_tmax_f"),
    ]
    for col_i, col_j in pairs:
        if col_i not in df.columns or col_j not in df.columns:
            continue
        bias_ij = _rd_bias(col_i, col_j)
        gap = df[col_i].to_numpy(dtype=float) - df[col_j].to_numpy(dtype=float)
        corrected = gap - bias_ij
        _add_feature(
            features,
            formulas,
            f"rd_{col_i}_{col_j}_bias_l{lag}",
            bias_ij,
            "ewm_mean(resid_i - resid_j)",
        )
        _add_feature(
            features,
            formulas,
            f"corrected_gap_{col_i}_{col_j}",
            corrected,
            "gap - rd_bias",
        )
        _add_feature(
            features,
            formulas,
            f"gap_sign_flip_{col_i}_{col_j}",
            (np.sign(gap) != np.sign(corrected)).astype(float),
            "sign(gap) != sign(corrected)",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex46(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    if "nbm_tmax_f" in df.columns and "hrrr_tmax_f" in df.columns:
        col_a = "nbm_tmax_f"
        col_b = "hrrr_tmax_f"
    else:
        col_a, col_b = core_cols[:2]
    A = df[col_a].to_numpy(dtype=float)
    B = df[col_b].to_numpy(dtype=float)
    y = df["actual_tmax_f"].to_numpy(dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    w_star = np.full(len(df), np.nan, dtype=float)
    for idx in train_idx:
        if not (np.isfinite(A[idx]) and np.isfinite(B[idx]) and np.isfinite(y[idx])):
            continue
        if abs(A[idx] - B[idx]) < 1e-6:
            w_star[idx] = 0.5
        else:
            w = (y[idx] - B[idx]) / (A[idx] - B[idx])
            w_star[idx] = float(np.clip(w, 0.0, 1.0))

    w_train = w_star[train_idx]
    w_train = w_train[np.isfinite(w_train)]
    spread_train = ens_std[train_idx]
    spread_train = spread_train[np.isfinite(w_star[train_idx])]
    if len(w_train) == 0:
        w_train = np.array([0.5], dtype=float)
        spread_train = np.array([0.0], dtype=float)
    corr = np.corrcoef(spread_train, w_train)[0, 1] if len(w_train) > 1 else 0.0
    increasing = corr >= 0
    iso = IsotonicRegression(out_of_bounds="clip", increasing=increasing)
    iso.fit(spread_train, w_train)
    w_hat = iso.predict(ens_std)
    w_hat = np.clip(w_hat, 0.0, 1.0)
    close_mask = np.abs(A - B) < 0.5
    w_hat = np.where(close_mask, 0.5, w_hat)
    blend = w_hat * A + (1.0 - w_hat) * B
    blend = np.where(close_mask, ens_mean, blend)

    _add_feature(
        features,
        formulas,
        "w_spread_iso",
        w_hat,
        "isotonic(w* | spread)",
    )
    _add_feature(
        features,
        formulas,
        "blend_ab",
        blend,
        "w*A + (1-w)*B",
    )
    _add_feature(
        features,
        formulas,
        "blend_minus_ensmean",
        blend - ens_mean,
        "blend - ens_mean",
    )
    train_fitted.append(
        {
            "name": "spread_blend_weight",
            "fit_on": "train",
            "col_a": col_a,
            "col_b": col_b,
            "increasing": increasing,
            "iso_x": iso.X_thresholds_.tolist(),
            "iso_y": iso.y_thresholds_.tolist(),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex47(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    train_df = ctx.train_df
    lambda_grid = [0.1, 1.0, 10.0]
    w0 = np.full(len(core_cols), 1.0 / len(core_cols), dtype=float)

    def _solve_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        n = X.shape[1]
        A = X.T @ X + lam * np.eye(n)
        b = X.T @ y + lam * w0
        ones = np.ones(n, dtype=float)
        K = np.block([[A, ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
        rhs = np.concatenate([b, [1.0]])
        try:
            sol = np.linalg.solve(K, rhs)
            return sol[:n]
        except np.linalg.LinAlgError:
            return w0

    def _fit_weights(sub_df: pd.DataFrame) -> np.ndarray:
        X = sub_df[core_cols].to_numpy(dtype=float)
        y = sub_df["actual_tmax_f"].to_numpy(dtype=float)
        best_w = w0
        best_mae = float("inf")
        for lam in lambda_grid:
            w = _solve_ridge(X, y, lam)
            preds = X @ w
            mae = float(np.mean(np.abs(preds - y))) if len(y) else float("inf")
            if mae < best_mae:
                best_mae = mae
                best_w = w
        return best_w

    weights_by_month = {}
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    season_weights = {}
    for season in range(4):
        season_df = train_df[train_df["month"].map(season_map) == season]
        if len(season_df) >= 200:
            season_weights[season] = _fit_weights(season_df)

    for month in range(1, 13):
        sub_df = train_df[train_df["month"] == month]
        if len(sub_df) >= 200:
            weights_by_month[month] = _fit_weights(sub_df)
        else:
            season = season_map[month]
            weights_by_month[month] = season_weights.get(season, w0)

    weights = np.zeros((len(df), len(core_cols)), dtype=float)
    for idx, month in enumerate(df["month"].to_numpy(dtype=int)):
        weights[idx] = weights_by_month.get(int(month), w0)
    seasonal_blend = np.sum(weights * df[core_cols].to_numpy(dtype=float), axis=1)

    for i, col in enumerate(core_cols):
        _add_feature(
            features,
            formulas,
            f"seasonal_w_{col}",
            weights[:, i],
            "month_weight",
        )
    _add_feature(
        features,
        formulas,
        "seasonal_blend",
        seasonal_blend,
        "sum(w_month * model)",
    )
    train_fitted.append(
        {
            "name": "seasonal_ridge_blend",
            "fit_on": "train",
            "lambda_grid": lambda_grid,
            "weights_by_month": {int(k): v.tolist() for k, v in weights_by_month.items()},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex48(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    global_c, global_phi = _fit_ar1(resid[train_idx])
    phi_map = {}
    c_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 200:
            phi_map[station] = float(np.clip(global_phi, -0.8, 0.95))
            c_map[station] = 0.0
            continue
        c_val, phi_val = _fit_ar1(resid[idx])
        phi_map[station] = float(np.clip(phi_val, -0.8, 0.95))
        c_map[station] = float(c_val)

    r_last_l2 = pd.Series(resid, index=df.index).groupby(gk).shift(lag)
    bias_long = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=60,
        min_periods=30,
        lag=lag,
        group_key=gk,
    )
    r_ar1 = np.full(len(df), np.nan, dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        phi = phi_map.get(station, float(np.clip(global_phi, -0.8, 0.95)))
        c_val = c_map.get(station, 0.0)
        r_ar1[idx] = c_val + phi * r_last_l2.to_numpy(dtype=float)[idx] + (1.0 - phi) * bias_long.to_numpy(dtype=float)[idx]

    _add_feature(
        features,
        formulas,
        "r_last_l2",
        r_last_l2,
        "resid(t-2)",
    )
    _add_feature(
        features,
        formulas,
        "r_ar1_pred",
        r_ar1,
        "c + phi*r_last_l2 + (1-phi)*bias_long",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_ar1_corr",
        ens_mean + r_ar1,
        "ens_mean + r_ar1_pred",
    )
    train_fitted.append(
        {
            "name": "ar1_params",
            "fit_on": "train",
            "phi": phi_map,
            "c": c_map,
            "global_phi": float(global_phi),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex49(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    month = df["month"].to_numpy(dtype=int)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    train_resid = resid[train_idx]
    global_r = float(np.nanvar(train_resid)) if len(train_resid) else 1.0
    global_r = global_r if global_r > 0 else 1.0
    ratio_grid = [1e-4, 1e-3, 1e-2, 1e-1]

    qr_by_month = {}
    for m in range(1, 13):
        mask = (month[train_idx] == m)
        r_vals = train_resid[mask]
        if len(r_vals) < 200:
            qr_by_month[m] = (0.01 * global_r, global_r)
            continue
        r_series = pd.Series(r_vals)
        r_base = r_series - r_series.rolling(30, min_periods=20).mean()
        r_val = float(np.nanvar(r_base.to_numpy(dtype=float)))
        if not np.isfinite(r_val) or r_val <= 0:
            r_val = float(np.nanvar(r_vals))
        if not np.isfinite(r_val) or r_val <= 0:
            r_val = global_r
        best_ratio = ratio_grid[0]
        best_ll = float("-inf")
        for ratio in ratio_grid:
            q_val = ratio * r_val
            ll = _kalman_loglik(r_vals, q_val, r_val)
            if ll > best_ll:
                best_ll = ll
                best_ratio = ratio
        qr_by_month[m] = (best_ratio * r_val, r_val)

    b_hat = np.full(len(df), np.nan, dtype=float)
    p_var = np.full(len(df), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx = np.where(df["station_id"].to_numpy() == station)[0]
        b = 0.0
        p = global_r
        for row_idx in idx:
            q_val, r_val = qr_by_month.get(int(month[row_idx]), (0.01 * global_r, global_r))
            val = resid[row_idx]
            if np.isfinite(val):
                p_pred = p + q_val
                s = p_pred + r_val
                if s > 0:
                    k = p_pred / s
                    b = b + k * (val - b)
                    p = (1.0 - k) * p_pred
            b_hat[row_idx] = b
            p_var[row_idx] = p

    bias_l2 = pd.Series(b_hat, index=df.index).groupby(gk).shift(lag)
    sd_l2 = pd.Series(np.sqrt(np.maximum(p_var, 0.0)), index=df.index).groupby(gk).shift(lag)
    _add_feature(
        features,
        formulas,
        f"kalman_bias_month_l{lag}",
        bias_l2,
        "kalman_bias_state_month",
    )
    _add_feature(
        features,
        formulas,
        f"kalman_sd_month_l{lag}",
        sd_l2,
        "kalman_sd_month",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_kf_month_corr",
        ens_mean + bias_l2.to_numpy(dtype=float),
        "ens_mean + kalman_bias_month",
    )
    train_fitted.append(
        {
            "name": "kalman_qr_by_month",
            "fit_on": "train",
            "qr_by_month": {int(k): [float(v[0]), float(v[1])] for k, v in qr_by_month.items()},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex50(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    bias = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=45,
        min_periods=25,
        lag=0,
        group_key=gk,
    )
    e = resid - bias.to_numpy(dtype=float)
    log_var = tfl.ewm_mean(
        pd.Series(np.log(e**2 + 1.0), index=df.index),
        halflife=30,
        min_periods=20,
        lag=0,
        group_key=gk,
    ).to_numpy(dtype=float)
    log_var = np.clip(log_var, -2.0, 5.0)
    sigma = np.sqrt(np.exp(log_var) - 1.0 + EPS)

    b_l2 = bias.groupby(gk).shift(lag)
    sigma_l2 = pd.Series(sigma, index=df.index).groupby(gk).shift(lag)
    e_l2 = pd.Series(e, index=df.index).groupby(gk).shift(lag)
    z_innov = e_l2.to_numpy(dtype=float) / (sigma_l2.to_numpy(dtype=float) + EPS)

    lambda_grid = [0.0, 0.05, 0.1, 0.2]
    best_lambda = 0.1
    best_mae = float("inf")
    for lam in lambda_grid:
        damp = 1.0 / (1.0 + lam * sigma_l2.to_numpy(dtype=float))
        pred = ens_mean + damp * b_l2.to_numpy(dtype=float)
        train_pred = pred[df.index.get_indexer(ctx.train_df.index)]
        train_y = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float)
        mask = np.isfinite(train_pred) & np.isfinite(train_y)
        if mask.sum() == 0:
            continue
        mae = float(np.mean(np.abs(train_pred[mask] - train_y[mask])))
        if mae < best_mae:
            best_mae = mae
            best_lambda = lam

    damp = 1.0 / (1.0 + best_lambda * sigma_l2.to_numpy(dtype=float))
    _add_feature(
        features,
        formulas,
        f"bias_l{lag}",
        b_l2,
        "ewm_mean(resid)",
    )
    _add_feature(
        features,
        formulas,
        f"sigma_l{lag}",
        sigma_l2,
        "ewm_mean(log(e^2+1))",
    )
    _add_feature(
        features,
        formulas,
        f"z_innov_l{lag}",
        z_innov,
        "e_l2 / sigma_l2",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_joint_corr",
        ens_mean + damp * b_l2.to_numpy(dtype=float),
        "ens_mean + damp*bias",
    )
    train_fitted.append(
        {
            "name": "joint_bias_vol",
            "fit_on": "train",
            "lambda_grid": lambda_grid,
            "best_lambda": best_lambda,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex51(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]

    ar1 = _exp_ex48(ctx)
    kalman = _exp_ex01(ctx)
    r_ar1 = ar1.features["r_ar1_pred"].to_numpy(dtype=float)
    kal_bias = kalman.features[f"kalman_bias_l{lag}"].to_numpy(dtype=float)
    kal_sd = kalman.features[f"kalman_bias_sd_l{lag}"].to_numpy(dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    train_sd = kal_sd[train_idx]
    med_sd = float(np.nanmedian(train_sd)) if len(train_sd) else 0.0
    iqr_sd = float(np.nanquantile(train_sd, 0.75) - np.nanquantile(train_sd, 0.25)) if len(train_sd) else 1.0
    iqr_sd = iqr_sd if iqr_sd > 0 else 1.0

    w = _sigmoid((kal_sd - med_sd) / (iqr_sd + EPS))
    blend_bias = w * r_ar1 + (1.0 - w) * kal_bias

    _add_feature(
        features,
        formulas,
        "r_ar1_pred",
        r_ar1,
        "ar1_resid_pred",
    )
    _add_feature(
        features,
        formulas,
        f"kalman_bias_l{lag}",
        kal_bias,
        "kalman_bias_state",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"kalman_bias_sd_l{lag}",
        kal_sd,
        "kalman_state_sd",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "ar1_kalman_blend_w",
        w,
        "sigmoid((kal_sd-med)/iqr)",
    )
    _add_feature(
        features,
        formulas,
        "ar1_kalman_blend_bias",
        blend_bias,
        "w*r_ar1 + (1-w)*kalman_bias",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_ar1_kalman_corr",
        ens_mean + blend_bias,
        "ens_mean + ar1_kalman_blend_bias",
    )
    train_fitted.extend(ar1.train_fitted)
    train_fitted.extend(kalman.train_fitted)
    train_fitted.append(
        {
            "name": "ar1_kalman_blend",
            "fit_on": "train",
            "median_sd": med_sd,
            "iqr_sd": iqr_sd,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex52(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    ar1 = _exp_ex48(ctx)
    ph = _exp_ex03(ctx)
    r_ar1 = ar1.features["r_ar1_pred"].to_numpy(dtype=float)
    ph_days = ph.features[f"ph_days_since_l{lag}"].to_numpy(dtype=float)
    ph_alarm = ph.features[f"ph_alarm_l{lag}"].to_numpy(dtype=float)

    fast_bias = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=7,
        min_periods=10,
        lag=lag,
        group_key=gk,
    ).to_numpy(dtype=float)

    days = np.nan_to_num(ph_days, nan=365.0)
    gate = _sigmoid((7.0 - days) / 2.0)
    blend_bias = gate * fast_bias + (1.0 - gate) * r_ar1

    _add_feature(
        features,
        formulas,
        "r_ar1_pred",
        r_ar1,
        "ar1_resid_pred",
    )
    _add_feature(
        features,
        formulas,
        f"ph_alarm_l{lag}",
        ph_alarm,
        "page_hinkley_alarm",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"ph_days_since_l{lag}",
        ph_days,
        "page_hinkley_days_since",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"fast_bias_l{lag}",
        fast_bias,
        "ewm_mean(resid, HL=7)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "ph_fast_gate",
        gate,
        "sigmoid((7-days_since)/2)",
    )
    _add_feature(
        features,
        formulas,
        "ph_fast_blend_bias",
        blend_bias,
        "gate*fast_bias + (1-gate)*r_ar1",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_ph_fast_corr",
        ens_mean + blend_bias,
        "ens_mean + ph_fast_blend_bias",
    )
    train_fitted.extend(ar1.train_fitted)
    train_fitted.extend(ph.train_fitted)
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex53(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]

    ar1 = _exp_ex48(ctx)
    asym = _exp_ex16(ctx)
    r_ar1 = ar1.features["r_ar1_pred"].to_numpy(dtype=float)
    p_pos = asym.features["p_pos_hat"].to_numpy(dtype=float)
    pos_bias = asym.features[f"pos_bias_l{lag}"].to_numpy(dtype=float)
    neg_bias = asym.features[f"neg_bias_l{lag}"].to_numpy(dtype=float)

    expected_bias = p_pos * pos_bias + (1.0 - p_pos) * neg_bias
    asym_strength = np.abs(pos_bias - neg_bias)
    gate = _sigmoid((asym_strength - 0.5) / 0.5)
    blend_bias = gate * expected_bias + (1.0 - gate) * r_ar1

    _add_feature(
        features,
        formulas,
        "p_pos_hat",
        p_pos,
        "P(resid>0 | X)",
    )
    _add_feature(
        features,
        formulas,
        f"pos_bias_l{lag}",
        pos_bias,
        "pos_bias_state",
    )
    _add_feature(
        features,
        formulas,
        f"neg_bias_l{lag}",
        neg_bias,
        "neg_bias_state",
    )
    _add_feature(
        features,
        formulas,
        "asym_strength",
        asym_strength,
        "|pos_bias - neg_bias|",
    )
    _add_feature(
        features,
        formulas,
        "asym_gate",
        gate,
        "sigmoid((asym_strength-0.5)/0.5)",
    )
    _add_feature(
        features,
        formulas,
        "asym_ar1_blend_bias",
        blend_bias,
        "asym_gate*asym_bias + (1-gate)*r_ar1",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_asym_ar1_corr",
        ens_mean + blend_bias,
        "ens_mean + asym_ar1_blend_bias",
    )
    train_fitted.extend(ar1.train_fitted)
    train_fitted.extend(asym.train_fitted)
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex54(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    ens_mean = _core_ensemble_stats(df, core_cols)["mean"]

    excl = _exp_ex19(ctx)
    outlier_excl = excl.features["outlier_excluded_mean"].to_numpy(dtype=float)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_excl = y - outlier_excl

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    train_resid = resid_excl[train_idx]
    r_series = pd.Series(train_resid)
    r_base = r_series - r_series.rolling(30, min_periods=20).mean()
    global_r = float(np.nanvar(r_base.to_numpy(dtype=float)))
    if not np.isfinite(global_r) or global_r <= 0:
        global_r = float(np.nanvar(train_resid)) if len(train_resid) else 1.0
    global_r = global_r if global_r > 0 else 1.0

    q_map: dict[str, float] = {}
    r_map: dict[str, float] = {}
    ratio_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        mask = ctx.train_df["station_id"] == station
        r_station = train_resid[mask.to_numpy()]
        if len(r_station) < 60:
            q_map[station] = 0.01 * global_r
            r_map[station] = global_r
            continue
        r_series = pd.Series(r_station)
        r_base_station = r_series - r_series.rolling(30, min_periods=20).mean()
        r_val = float(np.nanvar(r_base_station.to_numpy(dtype=float)))
        if not np.isfinite(r_val) or r_val <= 0:
            r_val = float(np.nanvar(r_station))
        if not np.isfinite(r_val) or r_val <= 0:
            r_val = global_r
        best_ratio = ratio_grid[0]
        best_ll = float("-inf")
        for ratio in ratio_grid:
            q_val = ratio * r_val
            ll = _kalman_loglik(r_station, q_val, r_val)
            if ll > best_ll:
                best_ll = ll
                best_ratio = ratio
        q_map[station] = best_ratio * r_val
        r_map[station] = r_val

    b_hat = np.full(len(df), np.nan, dtype=float)
    p_var = np.full(len(df), np.nan, dtype=float)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        q_val = q_map.get(station, 0.01 * global_r)
        r_val = r_map.get(station, global_r)
        b_station, p_station, _ = _kalman_filter(resid_excl[idx], q_val, r_val)
        b_hat[idx] = b_station
        p_var[idx] = p_station

    bias_l2 = pd.Series(b_hat, index=df.index).groupby(gk).shift(lag)
    sd_l2 = pd.Series(np.sqrt(np.maximum(p_var, 0.0)), index=df.index).groupby(gk).shift(lag)

    _add_feature(
        features,
        formulas,
        "outlier_excluded_mean",
        outlier_excl,
        "loo_mean(outlier_id)",
    )
    _add_feature(
        features,
        formulas,
        f"excl_kalman_bias_l{lag}",
        bias_l2,
        "kalman_bias_state(resid_excl)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"excl_kalman_sd_l{lag}",
        sd_l2,
        "kalman_state_sd(resid_excl)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "excl_kalman_corr",
        outlier_excl + bias_l2.to_numpy(dtype=float),
        "outlier_excluded_mean + excl_kalman_bias",
    )
    _add_feature(
        features,
        formulas,
        "delta_excl_from_ens",
        outlier_excl - ens_mean,
        "outlier_excluded_mean - ens_mean",
    )
    train_fitted.extend(excl.train_fitted)
    train_fitted.append(
        {
            "name": "kalman_qr_excl",
            "fit_on": "train",
            "ratio_grid": ratio_grid,
            "per_station_q": q_map,
            "per_station_r": r_map,
            "global_r": global_r,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex55(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []

    core = _exp_ex20(ctx)
    core_mean = core.features["core_mean"].to_numpy(dtype=float)
    dev_entropy = core.features["dev_entropy"].to_numpy(dtype=float)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - core_mean

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    global_c, global_phi = _fit_ar1(resid[train_idx])
    phi_map = {}
    c_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 200:
            phi_map[station] = float(np.clip(global_phi, -0.8, 0.95))
            c_map[station] = 0.0
            continue
        c_val, phi_val = _fit_ar1(resid[idx])
        phi_map[station] = float(np.clip(phi_val, -0.8, 0.95))
        c_map[station] = float(c_val)

    r_last_l2 = pd.Series(resid, index=df.index).groupby(gk).shift(lag)
    bias_long = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=60,
        min_periods=30,
        lag=lag,
        group_key=gk,
    )
    r_ar1 = np.full(len(df), np.nan, dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        phi = phi_map.get(station, float(np.clip(global_phi, -0.8, 0.95)))
        c_val = c_map.get(station, 0.0)
        r_ar1[idx] = c_val + phi * r_last_l2.to_numpy(dtype=float)[idx] + (1.0 - phi) * bias_long.to_numpy(dtype=float)[idx]

    _add_feature(
        features,
        formulas,
        "core_mean",
        core_mean,
        "entropy_weighted_mean",
    )
    _add_feature(
        features,
        formulas,
        "dev_entropy",
        dev_entropy,
        "entropy(weights)",
    )
    _add_feature(
        features,
        formulas,
        "r_ar1_core_pred",
        r_ar1,
        "ar1_resid_pred(core_mean)",
    )
    _add_feature(
        features,
        formulas,
        "core_mean_ar1_corr",
        core_mean + r_ar1,
        "core_mean + r_ar1_core_pred",
    )
    train_fitted.extend(core.train_fitted)
    train_fitted.append(
        {
            "name": "core_ar1_params",
            "fit_on": "train",
            "phi": phi_map,
            "c": c_map,
            "global_phi": float(global_phi),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex56(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    stats = _core_ensemble_stats(df, core_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    scale = _exp_ex27(ctx)
    pred_abs = scale.features["pred_abs_err"].to_numpy(dtype=float)
    pred_rank = scale.features["pred_abs_err_rank"].to_numpy(dtype=float)

    bias_slow = tfl.ewm_mean(
        pd.Series(resid, index=df.index),
        halflife=60,
        min_periods=30,
        lag=lag,
        group_key=gk,
    ).to_numpy(dtype=float)
    damp = 1.0 / (1.0 + 0.1 * pred_abs)
    corr = ens_mean + damp * bias_slow

    _add_feature(
        features,
        formulas,
        "pred_abs_err",
        pred_abs,
        "exp(pred_log_abs_err) - 0.5",
    )
    _add_feature(
        features,
        formulas,
        "pred_abs_err_rank",
        pred_rank,
        "rank(pred_abs_err)",
    )
    _add_feature(
        features,
        formulas,
        f"bias_slow_l{lag}",
        bias_slow,
        "ewm_mean(resid, HL=60)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "predabs_damp",
        damp,
        "1/(1+0.1*pred_abs_err)",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_predabs_corr",
        corr,
        "ens_mean + predabs_damp*bias_slow",
    )
    train_fitted.extend(scale.train_fitted)
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex57(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []

    ar1 = _exp_ex48(ctx)
    ph = _exp_ex03(ctx)
    huber = _exp_ex02(ctx)
    kalman = _exp_ex01(ctx)
    excl = _exp_ex19(ctx)
    core = _exp_ex20(ctx)
    damp = _exp_ex12(ctx)

    _add_feature(
        features,
        formulas,
        "ens_mean_ar1_corr",
        ar1.features["ens_mean_ar1_corr"].to_numpy(dtype=float),
        "ens_mean + r_ar1_pred",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_reset_corr",
        ph.features["ens_mean_reset_corr"].to_numpy(dtype=float),
        "ens_mean + reset_bias",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_huber_corr",
        huber.features["ens_mean_huber_corr"].to_numpy(dtype=float),
        "ens_mean + huber_bias",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_kalman_corr",
        kalman.features["ens_mean_kalman_corr"].to_numpy(dtype=float),
        "ens_mean + kalman_bias",
    )
    _add_feature(
        features,
        formulas,
        "outlier_excluded_mean",
        excl.features["outlier_excluded_mean"].to_numpy(dtype=float),
        "loo_mean(outlier_id)",
    )
    _add_feature(
        features,
        formulas,
        "core_mean",
        core.features["core_mean"].to_numpy(dtype=float),
        "entropy_weighted_mean",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_damped_corr",
        damp.features["ens_mean_damped_corr"].to_numpy(dtype=float),
        "ens_mean + damp*bias",
    )
    _add_feature(
        features,
        formulas,
        f"ph_alarm_l{lag}",
        ph.features[f"ph_alarm_l{lag}"].to_numpy(dtype=float),
        "page_hinkley_alarm",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "dev_entropy",
        core.features["dev_entropy"].to_numpy(dtype=float),
        "entropy(weights)",
    )
    train_fitted.extend(ar1.train_fitted)
    train_fitted.extend(ph.train_fitted)
    train_fitted.extend(huber.train_fitted)
    train_fitted.extend(kalman.train_fitted)
    train_fitted.extend(excl.train_fitted)
    train_fitted.extend(core.train_fitted)
    train_fitted.extend(damp.train_fitted)
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex58(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    ens_mean = _core_ensemble_stats(df, core_cols)["mean"]

    excl = _exp_ex19(ctx)
    core = _exp_ex20(ctx)
    outlier_excl = excl.features["outlier_excluded_mean"].to_numpy(dtype=float)
    concentration = core.features["dev_concentration"].to_numpy(dtype=float)
    entropy = core.features["dev_entropy"].to_numpy(dtype=float)

    w = np.clip(concentration, 0.0, 1.0)
    blend = w * outlier_excl + (1.0 - w) * ens_mean

    _add_feature(
        features,
        formulas,
        "outlier_excluded_mean",
        outlier_excl,
        "loo_mean(outlier_id)",
    )
    _add_feature(
        features,
        formulas,
        "dev_concentration",
        concentration,
        "max(weights)",
    )
    _add_feature(
        features,
        formulas,
        "dev_entropy",
        entropy,
        "entropy(weights)",
    )
    _add_feature(
        features,
        formulas,
        "excl_entropy_blend_w",
        w,
        "dev_concentration",
    )
    _add_feature(
        features,
        formulas,
        "excl_entropy_blend",
        blend,
        "w*outlier_excluded_mean + (1-w)*ens_mean",
    )
    train_fitted.extend(excl.train_fitted)
    train_fitted.extend(core.train_fitted)
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex59(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    ens_mean = _core_ensemble_stats(df, core_cols)["mean"]

    ar1 = _exp_ex48(ctx)
    spread = _exp_ex13(ctx)
    r_ar1 = ar1.features["r_ar1_pred"].to_numpy(dtype=float)
    p_hi = spread.features["p_hi"].to_numpy(dtype=float)
    damp = np.clip(1.0 - 0.5 * p_hi, 0.5, 1.0)
    corr = ens_mean + damp * r_ar1

    _add_feature(
        features,
        formulas,
        "r_ar1_pred",
        r_ar1,
        "ar1_resid_pred",
    )
    _add_feature(
        features,
        formulas,
        "p_hi",
        p_hi,
        "sigmoid(spread_z)",
    )
    _add_feature(
        features,
        formulas,
        "ar1_spread_damp",
        damp,
        "1-0.5*p_hi",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_ar1_spread_corr",
        corr,
        "ens_mean + ar1_spread_damp*r_ar1",
    )
    train_fitted.extend(ar1.train_fitted)
    train_fitted.extend(spread.train_fitted)
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex60(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    core_cols = _core_model_cols(df)
    ens_mean = _core_ensemble_stats(df, core_cols)["mean"]

    kalman = _exp_ex01(ctx)
    mom = _exp_ex38(ctx)
    vol = _exp_ex12(ctx)

    kal_bias = kalman.features[f"kalman_bias_l{lag}"].to_numpy(dtype=float)
    mom_bias = mom.features[f"mom_bias_l{lag}"].to_numpy(dtype=float)
    sigma = vol.features[f"sigma_l{lag}"].to_numpy(dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    train_sigma = sigma[train_idx]
    med_sigma = float(np.nanmedian(train_sigma)) if len(train_sigma) else 0.0
    iqr_sigma = float(np.nanquantile(train_sigma, 0.75) - np.nanquantile(train_sigma, 0.25)) if len(train_sigma) else 1.0
    iqr_sigma = iqr_sigma if iqr_sigma > 0 else 1.0

    w = _sigmoid((sigma - med_sigma) / (iqr_sigma + EPS))
    blend_bias = w * mom_bias + (1.0 - w) * kal_bias

    _add_feature(
        features,
        formulas,
        f"kalman_bias_l{lag}",
        kal_bias,
        "kalman_bias_state",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"mom_bias_l{lag}",
        mom_bias,
        "median(block_means)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        f"sigma_l{lag}",
        sigma,
        "ewm_mean(innovation^2)",
        {"lag": lag},
    )
    _add_feature(
        features,
        formulas,
        "mom_kalman_gate",
        w,
        "sigmoid((sigma-med)/iqr)",
    )
    _add_feature(
        features,
        formulas,
        "mom_kalman_blend_bias",
        blend_bias,
        "gate*mom_bias + (1-gate)*kalman_bias",
    )
    _add_feature(
        features,
        formulas,
        "ens_mean_mom_kalman_corr",
        ens_mean + blend_bias,
        "ens_mean + mom_kalman_blend_bias",
    )
    train_fitted.extend(kalman.train_fitted)
    train_fitted.extend(mom.train_fitted)
    train_fitted.extend(vol.train_fitted)
    train_fitted.append(
        {
            "name": "mom_kalman_gate",
            "fit_on": "train",
            "median_sigma": med_sigma,
            "iqr_sigma": iqr_sigma,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex61(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    X_all = np.column_stack([ens_mean, spread, sin_doy, cos_doy])
    y = df["actual_tmax_f"].to_numpy(dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    X_train = X_all[train_idx]
    y_train = y[train_idx]

    global_intercept, global_coef, global_mean, global_scale = _fit_ridge_linear(
        X_train, y_train, alpha=10.0
    )

    coef_map = {}
    scaler_map = {}
    intercept_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 200:
            intercept_map[station] = global_intercept
            coef_map[station] = global_coef
            scaler_map[station] = (global_mean, global_scale)
            continue
        X_s = X_all[idx]
        y_s = y[idx]
        intercept, coef, mean, scale = _fit_ridge_linear(X_s, y_s, alpha=10.0)
        intercept_map[station] = intercept
        coef_map[station] = coef
        scaler_map[station] = (mean, scale)

    emos_mu = np.full(len(df), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx = np.where(df["station_id"].to_numpy() == station)[0]
        intercept = intercept_map.get(station, global_intercept)
        coef = coef_map.get(station, global_coef)
        mean, scale = scaler_map.get(station, (global_mean, global_scale))
        emos_mu[idx] = _predict_linear(X_all[idx], intercept, coef, mean, scale)

    abs_err = np.abs(y - emos_mu)
    global_c0, global_c1 = _fit_nnls_scale(spread[train_idx], abs_err[train_idx])
    c0_map = {}
    c1_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 200:
            c0_map[station] = global_c0
            c1_map[station] = global_c1
            continue
        c0, c1 = _fit_nnls_scale(spread[idx], abs_err[idx])
        c0_map[station] = c0
        c1_map[station] = c1

    emos_sigma = np.full(len(df), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx = np.where(df["station_id"].to_numpy() == station)[0]
        c0 = c0_map.get(station, global_c0)
        c1 = c1_map.get(station, global_c1)
        emos_sigma[idx] = np.maximum(0.25, c0 + c1 * spread[idx])

    _add_feature(features, formulas, "emos_mu", emos_mu, "ridge(ens_mean, spread, sin_doy, cos_doy)")
    _add_feature(features, formulas, "emos_sigma", emos_sigma, "nnls(|e| ~ spread)")
    _add_feature(features, formulas, "emos_mu_plus", emos_mu + emos_sigma, "emos_mu + emos_sigma")
    _add_feature(features, formulas, "emos_mu_minus", emos_mu - emos_sigma, "emos_mu - emos_sigma")
    train_fitted.append(
        {
            "name": "emos_ridge",
            "fit_on": "train",
            "alpha": 10.0,
            "intercept": {str(k): float(v) for k, v in intercept_map.items()},
            "coef": {str(k): v.tolist() for k, v in coef_map.items()},
            "scaler": {str(k): [scaler_map[k][0].tolist(), scaler_map[k][1].tolist()] for k in scaler_map},
            "scale_nnls": {str(k): [float(c0_map[k]), float(c1_map[k])] for k in c0_map},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex62(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    X_all = np.column_stack([ens_mean, spread, sin_doy, cos_doy])
    y = df["actual_tmax_f"].to_numpy(dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    X_train = X_all[train_idx]
    y_train = y[train_idx]

    global_intercept, global_coef, global_mean, global_scale = _fit_quantile_linear(
        X_train, y_train, quantile=0.5, alpha=0.0
    )

    coef_map = {}
    scaler_map = {}
    intercept_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 200:
            intercept_map[station] = global_intercept
            coef_map[station] = global_coef
            scaler_map[station] = (global_mean, global_scale)
            continue
        X_s = X_all[idx]
        y_s = y[idx]
        intercept, coef, mean, scale = _fit_quantile_linear(
            X_s, y_s, quantile=0.5, alpha=0.0
        )
        coef = np.clip(coef, -2.0, 2.0)
        intercept_map[station] = intercept
        coef_map[station] = coef
        scaler_map[station] = (mean, scale)

    emos_q50 = np.full(len(df), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx = np.where(df["station_id"].to_numpy() == station)[0]
        intercept = intercept_map.get(station, global_intercept)
        coef = coef_map.get(station, global_coef)
        mean, scale = scaler_map.get(station, (global_mean, global_scale))
        emos_q50[idx] = _predict_linear(X_all[idx], intercept, coef, mean, scale)

    abs_err = np.abs(y - emos_q50)
    global_d0, global_d1 = _fit_nnls_scale(spread[train_idx], abs_err[train_idx])
    d0_map = {}
    d1_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 200:
            d0_map[station] = global_d0
            d1_map[station] = global_d1
            continue
        d0, d1 = _fit_nnls_scale(spread[idx], abs_err[idx])
        d0_map[station] = d0
        d1_map[station] = d1

    emos_mad = np.full(len(df), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx = np.where(df["station_id"].to_numpy() == station)[0]
        d0 = d0_map.get(station, global_d0)
        d1 = d1_map.get(station, global_d1)
        emos_mad[idx] = np.maximum(0.25, d0 + d1 * spread[idx])

    _add_feature(features, formulas, "emos_q50", emos_q50, "quantile_regression_tau_0.5")
    _add_feature(features, formulas, "emos_mad", emos_mad, "nnls(|e| ~ spread)")
    _add_feature(features, formulas, "emos_q50_plus", emos_q50 + 1.4826 * emos_mad, "q50 + 1.4826*mad")
    _add_feature(features, formulas, "emos_q50_minus", emos_q50 - 1.4826 * emos_mad, "q50 - 1.4826*mad")
    train_fitted.append(
        {
            "name": "emos_q50",
            "fit_on": "train",
            "intercept": {str(k): float(v) for k, v in intercept_map.items()},
            "coef": {str(k): v.tolist() for k, v in coef_map.items()},
            "scaler": {str(k): [scaler_map[k][0].tolist(), scaler_map[k][1].tolist()] for k in scaler_map},
            "mad_nnls": {str(k): [float(d0_map[k]), float(d1_map[k])] for k in d0_map},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex63(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    X_all = df[model_cols].to_numpy(dtype=float)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)

    weights_map = {}
    intercept_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        train_idx = df.index.get_indexer(ctx.train_df.index[mask])
        train_idx = train_idx[train_idx >= 0]
        if len(train_idx) < 100:
            weights_map[station] = np.full(len(model_cols), 1.0 / len(model_cols), dtype=float)
            intercept_map[station] = (0.0, 0.0, 0.0)
            continue
        X_train = X_all[train_idx]
        y_train = y[train_idx]
        w, k0, k1, k2 = _fit_simplex_lad(
            X_train, y_train, ridge=0.5, sin_doy=sin_doy[train_idx], cos_doy=cos_doy[train_idx]
        )
        weights_map[station] = w
        intercept_map[station] = (k0, k1, k2)

    stack_q50 = np.full(len(df), np.nan, dtype=float)
    weight_matrix = np.zeros((len(df), len(model_cols)), dtype=float)
    for station in df["station_id"].unique():
        idx = np.where(df["station_id"].to_numpy() == station)[0]
        w = weights_map.get(station)
        k0, k1, k2 = intercept_map.get(station, (0.0, 0.0, 0.0))
        weight_matrix[idx] = w
        stack_q50[idx] = k0 + k1 * sin_doy[idx] + k2 * cos_doy[idx] + X_all[idx] @ w

    w_entropy = _rowwise_entropy(weight_matrix)
    _add_feature(features, formulas, "stack_q50", stack_q50, "simplex_LAD_stack")
    for i, col in enumerate(model_cols):
        _add_feature(features, formulas, f"stack_w_{col}", weight_matrix[:, i], "simplex_weight")
    _add_feature(features, formulas, "stack_w_entropy", w_entropy, "entropy(weights)")
    train_fitted.append(
        {
            "name": "simplex_lad",
            "fit_on": "train",
            "weights": {str(k): v.tolist() for k, v in weights_map.items()},
            "intercepts": {str(k): list(intercept_map[k]) for k in intercept_map},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex64(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    X_all = np.column_stack([ens_mean, spread, sin_doy, cos_doy])
    y = df["actual_tmax_f"].to_numpy(dtype=float)

    t1_map = {}
    t2_map = {}
    coef_map = {}
    scaler_map = {}
    intercept_map = {}

    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 80:
            continue
        ens_train = ens_mean[idx]
        t1 = float(np.quantile(ens_train, 0.33))
        t2 = float(np.quantile(ens_train, 0.66))
        t1_map[station] = t1
        t2_map[station] = t2
        for label, reg_mask in {
            "low": ens_train <= t1,
            "mid": (ens_train > t1) & (ens_train <= t2),
            "high": ens_train > t2,
        }.items():
            reg_idx = idx[reg_mask]
            if len(reg_idx) < 80:
                continue
            intercept, coef, mean, scale = _fit_quantile_linear(
                X_all[reg_idx], y[reg_idx], quantile=0.5, alpha=0.0
            )
            coef_map[(station, label)] = coef
            intercept_map[(station, label)] = intercept
            scaler_map[(station, label)] = (mean, scale)

    global_t1 = float(np.quantile(ens_mean[ctx.train_df.index], 0.33))
    global_t2 = float(np.quantile(ens_mean[ctx.train_df.index], 0.66))
    global_params = {}
    for label, reg_mask in {
        "low": ens_mean[ctx.train_df.index] <= global_t1,
        "mid": (ens_mean[ctx.train_df.index] > global_t1) & (ens_mean[ctx.train_df.index] <= global_t2),
        "high": ens_mean[ctx.train_df.index] > global_t2,
    }.items():
        reg_idx = df.index.get_indexer(ctx.train_df.index[reg_mask])
        reg_idx = reg_idx[reg_idx >= 0]
        if len(reg_idx) < 80:
            continue
        intercept, coef, mean, scale = _fit_quantile_linear(
            X_all[reg_idx], y[reg_idx], quantile=0.5, alpha=0.0
        )
        global_params[label] = (intercept, coef, mean, scale)

    emos_piece = np.full(len(df), np.nan, dtype=float)
    g_low = np.zeros(len(df), dtype=float)
    g_mid = np.zeros(len(df), dtype=float)
    g_high = np.zeros(len(df), dtype=float)
    for station in df["station_id"].unique():
        idx = np.where(df["station_id"].to_numpy() == station)[0]
        t1 = t1_map.get(station, global_t1)
        t2 = t2_map.get(station, global_t2)
        g_low[idx] = _sigmoid((t1 - ens_mean[idx]) / 2.0)
        g_high[idx] = _sigmoid((ens_mean[idx] - t2) / 2.0)
        g_mid[idx] = np.clip(1.0 - g_low[idx] - g_high[idx], 0.0, 1.0)
        denom = g_low[idx] + g_mid[idx] + g_high[idx]
        denom = np.where(denom == 0.0, 1.0, denom)
        g_low[idx] /= denom
        g_mid[idx] /= denom
        g_high[idx] /= denom

        preds = {}
        for label in ["low", "mid", "high"]:
            key = (station, label)
            if key in coef_map:
                intercept = intercept_map[key]
                coef = coef_map[key]
                mean, scale = scaler_map[key]
            else:
                intercept, coef, mean, scale = global_params.get(label, (0.0, np.zeros(4), np.zeros(4), np.ones(4)))
            preds[label] = _predict_linear(X_all[idx], intercept, coef, mean, scale)
        emos_piece[idx] = g_low[idx] * preds["low"] + g_mid[idx] * preds["mid"] + g_high[idx] * preds["high"]

    _add_feature(features, formulas, "emos_q50_piece", emos_piece, "piecewise_q50_blend")
    _add_feature(features, formulas, "g_low", g_low, "sigmoid((T1-ens_mean)/2)")
    _add_feature(features, formulas, "g_mid", g_mid, "1-g_low-g_high")
    _add_feature(features, formulas, "g_high", g_high, "sigmoid((ens_mean-T2)/2)")
    train_fitted.append(
        {
            "name": "piecewise_q50",
            "fit_on": "train",
            "t1": {str(k): float(v) for k, v in t1_map.items()},
            "t2": {str(k): float(v) for k, v in t2_map.items()},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex65(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    X_all = np.column_stack([ens_mean, spread, sin_doy, cos_doy])
    y = df["actual_tmax_f"].to_numpy(dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    X_train = X_all[train_idx]
    y_train = y[train_idx]

    intercept, coef, mean, scale = _fit_quantile_linear(
        X_train, y_train, quantile=0.5, alpha=0.0
    )
    emos_q50 = _predict_linear(X_all, intercept, coef, mean, scale)

    resid = y - emos_q50
    scale_proxy = np.maximum(0.25, 1.4826 * np.abs(resid))
    z = resid / (scale_proxy + EPS)
    z_l2 = pd.Series(z, index=df.index).groupby(gk).shift(lag)

    phi_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 50:
            phi_map[station] = 0.0
            continue
        z_train = z[idx]
        z_l2_train = z_l2.to_numpy(dtype=float)[idx]
        mask_valid = np.isfinite(z_train) & np.isfinite(z_l2_train)
        if mask_valid.sum() < 30:
            phi_map[station] = 0.0
            continue
        num = float(np.sum(z_train[mask_valid] * z_l2_train[mask_valid]))
        den = float(np.sum(z_l2_train[mask_valid] ** 2))
        phi = num / den if den > 0 else 0.0
        phi_map[station] = float(np.clip(phi, -0.8, 0.8))

    z_hat = np.full(len(df), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx = np.where(df["station_id"].to_numpy() == station)[0]
        phi = phi_map.get(station, 0.0)
        z_hat[idx] = phi * z_l2.to_numpy(dtype=float)[idx]
    u_hat = z_hat * scale_proxy
    emos_q50_ar = emos_q50 + u_hat

    _add_feature(features, formulas, "emos_q50", emos_q50, "quantile_regression_tau_0.5")
    _add_feature(features, formulas, f"z_l{lag}", z_l2, "standardized_resid_l2")
    _add_feature(features, formulas, "z_hat", z_hat, "phi2 * z_l2")
    _add_feature(features, formulas, "emos_q50_ar", emos_q50_ar, "emos_q50 + u_hat")
    train_fitted.append(
        {
            "name": "ar_emos_phi2",
            "fit_on": "train",
            "phi2": {str(k): float(v) for k, v in phi_map.items()},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex66(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    kappa = 40.0

    bias = np.full(len(df), np.nan, dtype=float)
    bias_n = np.full(len(df), np.nan, dtype=float)
    shrink = np.full(len(df), np.nan, dtype=float)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    global_med = float(np.nanmedian(resid[train_idx])) if len(train_idx) else 0.0

    for station in df["station_id"].unique():
        train_mask = ctx.train_df["station_id"] == station
        idx_train = df.index.get_indexer(ctx.train_df.index[train_mask])
        idx_train = idx_train[idx_train >= 0]
        if len(idx_train) < 80:
            idx_all = np.where(df["station_id"].to_numpy() == station)[0]
            bias[idx_all] = global_med
            bias_n[idx_all] = 0.0
            shrink[idx_all] = 0.0
            continue
        ens_train = ens_mean[idx_train]
        spread_train = spread[idx_train]
        ens_edges = _quantile_edges(ens_train, 10)
        spread_edges = _quantile_edges(spread_train, 4)
        ens_bins = _bin_ids(ens_train, ens_edges)
        spr_bins = _bin_ids(spread_train, spread_edges)
        cell_stats = {}
        for b in range(10):
            for s in range(5):
                mask = (ens_bins == b) & (spr_bins == s)
                if not np.any(mask):
                    continue
                cell_vals = resid[idx_train][mask]
                cell_stats[(b, s)] = (float(np.median(cell_vals)), int(mask.sum()))
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        ens_bins_all = _bin_ids(ens_mean[idx_all], ens_edges)
        spr_bins_all = _bin_ids(spread[idx_all], spread_edges)
        for pos, row_idx in enumerate(idx_all):
            key = (int(ens_bins_all[pos]), int(spr_bins_all[pos]))
            if key in cell_stats:
                med_val, n_val = cell_stats[key]
            else:
                med_val, n_val = global_med, 0
            w = n_val / (n_val + kappa) if n_val > 0 else 0.0
            bias[row_idx] = np.clip(w * med_val + (1.0 - w) * global_med, -3.0, 3.0)
            bias_n[row_idx] = n_val
            shrink[row_idx] = w

    _add_feature(features, formulas, "bias_2d_med", bias, "median(resid) by ens_bin,spread_bin")
    _add_feature(features, formulas, "bias_2d_n", bias_n, "cell_count")
    _add_feature(features, formulas, "bias_2d_shrink", shrink, "n/(n+kappa)")
    _add_feature(features, formulas, "ens_mean_2d_corr", ens_mean + bias, "ens_mean + bias_2d_med")
    train_fitted.append(
        {
            "name": "bias_2d_med_map",
            "fit_on": "train",
            "kappa": kappa,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex67(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    kappa = 30.0

    pc1 = np.full(len(df), np.nan, dtype=float)
    bias = np.full(len(df), np.nan, dtype=float)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    global_med = float(np.nanmedian(resid[train_idx])) if len(train_idx) else 0.0

    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        if len(idx_train) < 80:
            pc1[idx_all] = 0.0
            bias[idx_all] = global_med
            continue
        d_train = df.loc[idx_train, model_cols].to_numpy(dtype=float) - ens_mean[idx_train][:, None]
        pca = PCA(n_components=1, random_state=ctx.seed)
        pca.fit(d_train)
        d_all = df.loc[idx_all, model_cols].to_numpy(dtype=float) - ens_mean[idx_all][:, None]
        pc_all = pca.transform(d_all).ravel()
        pc1[idx_all] = pc_all

        pc_edges = _quantile_edges(pc_all, 10)
        ens_edges = _quantile_edges(ens_mean[idx_train], 5)
        pc_bins_train = _bin_ids(pc_all[np.isin(idx_all, idx_train)], pc_edges)
        ens_bins_train = _bin_ids(ens_mean[idx_train], ens_edges)
        cell_stats = {}
        cell_n = {}
        for b in range(10):
            for ebin in range(6):
                mask = (pc_bins_train == b) & (ens_bins_train == ebin)
                if not np.any(mask):
                    continue
                cell_vals = resid[idx_train][mask]
                cell_stats[(b, ebin)] = float(np.median(cell_vals))
                cell_n[(b, ebin)] = int(mask.sum())

        pc_bins_all = _bin_ids(pc_all, pc_edges)
        ens_bins_all = _bin_ids(ens_mean[idx_all], ens_edges)
        for pos, row_idx in enumerate(idx_all):
            key = (int(pc_bins_all[pos]), int(ens_bins_all[pos]))
            n_val = cell_n.get(key, 0)
            med_val = cell_stats.get(key, global_med)
            w = n_val / (n_val + kappa) if n_val > 0 else 0.0
            bias[row_idx] = np.clip(w * med_val + (1.0 - w) * global_med, -3.0, 3.0)

    _add_feature(features, formulas, "pc1_disagreement", pc1, "PCA1(f_i - ens_mean)")
    _add_feature(features, formulas, "bias_pc1_ens_med", bias, "median(resid) by pc1_bin,ens_bin")
    _add_feature(features, formulas, "ens_mean_pc1_corr", ens_mean + bias, "ens_mean + bias_pc1")
    train_fitted.append(
        {
            "name": "pc1_bias_map",
            "fit_on": "train",
            "kappa": kappa,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex68(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    warm = df["month"].isin([5, 6, 7, 8, 9]).astype(int).to_numpy(dtype=int)

    bias = np.full(len(df), np.nan, dtype=float)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]

    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        if len(idx_train) < 100:
            bias[idx_all] = float(np.nanmedian(resid[train_idx])) if len(train_idx) else 0.0
            continue
        ens_edges = _quantile_edges(ens_mean[idx_train], 5)
        spr_edges = _quantile_edges(spread[idx_train], 3)
        ens_bins_train = _bin_ids(ens_mean[idx_train], ens_edges)
        spr_bins_train = _bin_ids(spread[idx_train], spr_edges)
        warm_train = warm[idx_train]

        cell_med = {}
        cell_n = {}
        med_2d = {}
        n_2d = {}
        med_1d = {}
        n_1d = {}
        for ebin in range(6):
            for sbin in range(4):
                for season in [0, 1]:
                    mask = (ens_bins_train == ebin) & (spr_bins_train == sbin) & (warm_train == season)
                    if not np.any(mask):
                        continue
                    cell_vals = resid[idx_train][mask]
                    cell_med[(ebin, sbin, season)] = float(np.median(cell_vals))
                    cell_n[(ebin, sbin, season)] = int(mask.sum())
                    med_2d[(ebin, sbin, season)] = float(np.median(cell_vals))
                    n_2d[(ebin, sbin, season)] = int(mask.sum())
            for season in [0, 1]:
                mask = (ens_bins_train == ebin) & (warm_train == season)
                if np.any(mask):
                    med_1d[(ebin, season)] = float(np.median(resid[idx_train][mask]))
                    n_1d[(ebin, season)] = int(mask.sum())

        global_med = {}
        for season in [0, 1]:
            mask = warm_train == season
            global_med[season] = float(np.median(resid[idx_train][mask])) if np.any(mask) else 0.0

        ens_bins_all = _bin_ids(ens_mean[idx_all], ens_edges)
        spr_bins_all = _bin_ids(spread[idx_all], spr_edges)
        warm_all = warm[idx_all]
        for pos, row_idx in enumerate(idx_all):
            ebin = int(ens_bins_all[pos])
            sbin = int(spr_bins_all[pos])
            season = int(warm_all[pos])
            n3 = cell_n.get((ebin, sbin, season), 0)
            n2 = n_2d.get((ebin, sbin, season), 0)
            n1 = n_1d.get((ebin, season), 0)
            w3 = n3 / (n3 + 20.0) if n3 > 0 else 0.0
            w2 = (1.0 - w3) * (n2 / (n2 + 40.0)) if n2 > 0 else 0.0
            w1 = (1.0 - w3 - w2) * (n1 / (n1 + 80.0)) if n1 > 0 else 0.0
            base = global_med.get(season, 0.0)
            med3 = cell_med.get((ebin, sbin, season), base)
            med2 = med_2d.get((ebin, sbin, season), base)
            med1 = med_1d.get((ebin, season), base)
            bias_val = w3 * med3 + w2 * med2 + w1 * med1 + (1.0 - w3 - w2 - w1) * base
            bias[row_idx] = float(np.clip(bias_val, -3.0, 3.0))

    _add_feature(features, formulas, "bias_3d_med", bias, "hierarchical_3d_bias")
    _add_feature(features, formulas, "ens_mean_3d_corr", ens_mean + bias, "ens_mean + bias_3d")
    _add_feature(features, formulas, "warm_season_flag", warm, "month in [5..9]")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex69(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    bias = np.full(len(df), np.nan, dtype=float)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]

    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        if len(idx_train) < 120:
            bias[idx_all] = float(np.nanmedian(resid[train_idx])) if len(train_idx) else 0.0
            continue
        X_train = np.column_stack([ens_mean[idx_train], spread[idx_train]])
        spline_t = SplineTransformer(n_knots=5, degree=3, include_bias=False)
        spline_s = SplineTransformer(n_knots=4, degree=3, include_bias=False)
        X_t = spline_t.fit_transform(X_train[:, [0]])
        X_s = spline_s.fit_transform(X_train[:, [1]])
        X_design = np.column_stack([X_t, X_s, sin_doy[idx_train], cos_doy[idx_train]])
        intercept, coef, mean, scale = _fit_ridge_linear(X_design, resid[idx_train], alpha=50.0)

        X_all = np.column_stack([ens_mean[idx_all], spread[idx_all]])
        X_t_all = spline_t.transform(X_all[:, [0]])
        X_s_all = spline_s.transform(X_all[:, [1]])
        X_design_all = np.column_stack([X_t_all, X_s_all, sin_doy[idx_all], cos_doy[idx_all]])
        bias[idx_all] = _predict_linear(X_design_all, intercept, coef, mean, scale)

    _add_feature(features, formulas, "bias_spline", bias, "spline_bias")
    _add_feature(features, formulas, "ens_mean_spline_corr", ens_mean + bias, "ens_mean + bias_spline")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex70(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    kappa = 40.0

    bias_q50 = np.full(len(df), np.nan, dtype=float)
    resid_iqr = np.full(len(df), np.nan, dtype=float)
    resid_skew = np.full(len(df), np.nan, dtype=float)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    global_q25 = float(np.nanquantile(resid[train_idx], 0.25)) if len(train_idx) else 0.0
    global_q50 = float(np.nanquantile(resid[train_idx], 0.5)) if len(train_idx) else 0.0
    global_q75 = float(np.nanquantile(resid[train_idx], 0.75)) if len(train_idx) else 0.0

    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        if len(idx_train) < 80:
            bias_q50[idx_all] = global_q50
            resid_iqr[idx_all] = global_q75 - global_q25
            resid_skew[idx_all] = global_q75 + global_q25 - 2.0 * global_q50
            continue
        ens_edges = _quantile_edges(ens_mean[idx_train], 10)
        spr_edges = _quantile_edges(spread[idx_train], 4)
        ens_bins_train = _bin_ids(ens_mean[idx_train], ens_edges)
        spr_bins_train = _bin_ids(spread[idx_train], spr_edges)
        cell_stats = {}
        cell_n = {}
        for b in range(10):
            for s in range(5):
                mask = (ens_bins_train == b) & (spr_bins_train == s)
                if not np.any(mask):
                    continue
                vals = resid[idx_train][mask]
                cell_stats[(b, s)] = (
                    float(np.quantile(vals, 0.25)),
                    float(np.quantile(vals, 0.5)),
                    float(np.quantile(vals, 0.75)),
                )
                cell_n[(b, s)] = int(mask.sum())
        ens_bins_all = _bin_ids(ens_mean[idx_all], ens_edges)
        spr_bins_all = _bin_ids(spread[idx_all], spr_edges)
        for pos, row_idx in enumerate(idx_all):
            key = (int(ens_bins_all[pos]), int(spr_bins_all[pos]))
            if key in cell_stats:
                q25, q50, q75 = cell_stats[key]
                n_val = cell_n.get(key, 0)
            else:
                q25, q50, q75 = global_q25, global_q50, global_q75
                n_val = 0
            w = n_val / (n_val + kappa) if n_val > 0 else 0.0
            q25 = w * q25 + (1.0 - w) * global_q25
            q50 = w * q50 + (1.0 - w) * global_q50
            q75 = w * q75 + (1.0 - w) * global_q75
            bias_q50[row_idx] = np.clip(q50, -3.0, 3.0)
            resid_iqr[row_idx] = np.clip(q75 - q25, 0.3, 5.0)
            resid_skew[row_idx] = q75 + q25 - 2.0 * q50

    _add_feature(features, formulas, "bias_q50_2d", bias_q50, "q50(resid) by ens_bin,spread_bin")
    _add_feature(features, formulas, "resid_iqr_2d", resid_iqr, "q75-q25 per cell")
    _add_feature(features, formulas, "resid_skew_2d", resid_skew, "q75+q25-2*q50")
    _add_feature(features, formulas, "ens_mean_q50corr", ens_mean + bias_q50, "ens_mean + bias_q50_2d")
    train_fitted.append(
        {
            "name": "resid_quantile_map",
            "fit_on": "train",
            "kappa": kappa,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex71(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]

    b_map = {}
    s_map = {}
    w_map = {}
    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        if len(idx_train) < 80:
            idx_train = train_idx
        mu = []
        sigma = []
        for col in model_cols:
            r = y[idx_train] - df[col].to_numpy(dtype=float)[idx_train]
            med = float(np.median(r))
            mad = float(np.median(np.abs(r - med)))
            b_map[(station, col)] = med
            s_val = max(0.25, 1.4826 * mad)
            s_map[(station, col)] = s_val
            mu.append(df[col].to_numpy(dtype=float)[idx_train] + med)
            sigma.append(np.full(len(idx_train), s_val))
        mu = np.column_stack(mu)
        sigma = np.column_stack(sigma)
        weights = _bma_fit_weights(y[idx_train], mu, sigma)
        if not np.isfinite(weights).all():
            inv = np.array([1.0 / s_map[(station, col)] for col in model_cols], dtype=float)
            weights = inv / np.sum(inv)
        w_map[station] = weights

    bma_mean = np.full(len(df), np.nan, dtype=float)
    bma_sigma = np.full(len(df), np.nan, dtype=float)
    weight_matrix = np.zeros((len(df), len(model_cols)), dtype=float)
    for station in df["station_id"].unique():
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        weights = w_map.get(station)
        weight_matrix[idx_all] = weights
        mu = np.column_stack(
            [df[col].to_numpy(dtype=float)[idx_all] + b_map[(station, col)] for col in model_cols]
        )
        sigma = np.column_stack([np.full(len(idx_all), s_map[(station, col)]) for col in model_cols])
        mean = mu @ weights
        var = np.sum(weights * (sigma**2 + (mu - mean[:, None]) ** 2), axis=1)
        bma_mean[idx_all] = mean
        bma_sigma[idx_all] = np.sqrt(np.maximum(var, 0.0))

    _add_feature(features, formulas, "bma_mean", bma_mean, "sum(w_i*(f_i+b_i))")
    _add_feature(features, formulas, "bma_sigma", bma_sigma, "sqrt(sum(w_i*(s_i^2+(mu_i-mean)^2)))")
    for i, col in enumerate(model_cols):
        _add_feature(features, formulas, f"bma_w_{col}", weight_matrix[:, i], "bma_weight")
    train_fitted.append(
        {
            "name": "bma_static",
            "fit_on": "train",
            "weights": {str(k): v.tolist() for k, v in w_map.items()},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex72(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_by_model = {col: y - df[col].to_numpy(dtype=float) for col in model_cols}
    resid_l2 = {col: pd.Series(resid_by_model[col], index=df.index).groupby(gk).shift(lag).to_numpy(dtype=float) for col in model_cols}

    weights = np.full((len(df), len(model_cols)), np.nan, dtype=float)
    bma_online = np.full(len(df), np.nan, dtype=float)
    eta = 0.25
    for station in df["station_id"].unique():
        idx = df.sort_values(["station_id", "target_date_local"]).index
        idx = [i for i in idx if df.loc[i, "station_id"] == station]
        logw = np.full(len(model_cols), 0.0, dtype=float)
        b_state = np.zeros(len(model_cols), dtype=float)
        a_state = np.ones(len(model_cols), dtype=float)
        for row_idx in idx:
            r_vals = np.array([resid_l2[col][row_idx] for col in model_cols], dtype=float)
            for i, r_val in enumerate(r_vals):
                if not np.isfinite(r_val):
                    continue
                r_clip = float(np.clip(r_val, -6.0, 6.0))
                b_state[i] = b_state[i] + _ewm_alpha(21) * (r_clip - b_state[i])
                a_state[i] = a_state[i] + _ewm_alpha(21) * (abs(r_clip - b_state[i]) - a_state[i])
                s_val = max(0.5, 1.25 * a_state[i])
                ll = float(student_t.logpdf(r_clip, df=4, loc=b_state[i], scale=s_val))
                logw[i] = logw[i] + eta * ll
            w = _softmax(logw[None, :], axis=1)[0]
            weights[row_idx] = w
            preds = np.array([df[col].to_numpy(dtype=float)[row_idx] + b_state[i] for i, col in enumerate(model_cols)], dtype=float)
            bma_online[row_idx] = float(np.sum(w * preds))

    _add_feature(features, formulas, "bma_online", bma_online, "online_student_t_weights")
    for i, col in enumerate(model_cols):
        _add_feature(features, formulas, f"bma_online_w_{col}", weights[:, i], "online_weight")
    w_eff = 1.0 / np.sum(weights**2, axis=1)
    _add_feature(features, formulas, "bma_online_eff_n", w_eff, "1/sum(w^2)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex73(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_by_model = {col: y - df[col].to_numpy(dtype=float) for col in model_cols}
    resid_l2 = {col: pd.Series(resid_by_model[col], index=df.index).groupby(gk).shift(lag).to_numpy(dtype=float) for col in model_cols}

    weights = np.full((len(df), len(model_cols)), np.nan, dtype=float)
    hedge_mean = np.full(len(df), np.nan, dtype=float)
    eta = 0.15
    for station in df["station_id"].unique():
        idx = df.sort_values(["station_id", "target_date_local"]).index
        idx = [i for i in idx if df.loc[i, "station_id"] == station]
        w = np.full(len(model_cols), 1.0 / len(model_cols), dtype=float)
        for row_idx in idx:
            losses = np.array([abs(resid_l2[col][row_idx]) for col in model_cols], dtype=float)
            losses = np.where(np.isfinite(losses), losses, 0.0)
            w = w * np.exp(-eta * losses)
            w = np.maximum(w, 1e-6)
            w = w / np.sum(w)
            weights[row_idx] = w
            preds = np.array([df[col].to_numpy(dtype=float)[row_idx] for col in model_cols], dtype=float)
            hedge_mean[row_idx] = float(np.sum(w * preds))

    _add_feature(features, formulas, "hedge_mean", hedge_mean, "hedge_exp_weights")
    for i, col in enumerate(model_cols):
        _add_feature(features, formulas, f"hedge_w_{col}", weights[:, i], "hedge_weight")
    w_entropy = _rowwise_entropy(weights)
    _add_feature(features, formulas, "hedge_w_entropy", w_entropy, "entropy(weights)")
    w_eff = 1.0 / np.sum(weights**2, axis=1)
    _add_feature(features, formulas, "hedge_eff_n", w_eff, "1/sum(w^2)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex74(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    spread = _spread_from_stats(df, stats)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_by_model = {col: y - df[col].to_numpy(dtype=float) for col in model_cols}
    resid_l2 = {col: pd.Series(resid_by_model[col], index=df.index).groupby(gk).shift(lag).to_numpy(dtype=float) for col in model_cols}

    med_map = {}
    iqr_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        train_spread = spread[df.index.get_indexer(ctx.train_df.index[mask])]
        if len(train_spread) < 30:
            train_spread = spread[df.index.get_indexer(ctx.train_df.index)]
        med_map[station] = float(np.nanmedian(train_spread))
        iqr_val = float(np.nanquantile(train_spread, 0.75) - np.nanquantile(train_spread, 0.25))
        iqr_map[station] = iqr_val if iqr_val > 0 else 1.0

    weights = np.full((len(df), len(model_cols)), np.nan, dtype=float)
    hedge_mean = np.full(len(df), np.nan, dtype=float)
    eta0 = 0.2
    lam = 0.25
    for station in df["station_id"].unique():
        idx = df.sort_values(["station_id", "target_date_local"]).index
        idx = [i for i in idx if df.loc[i, "station_id"] == station]
        w = np.full(len(model_cols), 1.0 / len(model_cols), dtype=float)
        med_val = med_map.get(station, float(np.nanmedian(spread)))
        iqr_val = iqr_map.get(station, 1.0)
        for row_idx in idx:
            spread_z = np.clip((spread[row_idx] - med_val) / (iqr_val + EPS), -3.0, 3.0)
            eta = np.clip(eta0 / (1.0 + spread_z), 0.05, 0.25)
            losses = np.array([abs(resid_l2[col][row_idx]) for col in model_cols], dtype=float)
            losses = np.where(np.isfinite(losses), losses, 0.0)
            w = w * np.exp(-eta * losses)
            w = np.maximum(w, 1e-6)
            w = w / np.sum(w)
            w = (1.0 - lam) * w + lam * (1.0 / len(model_cols))
            weights[row_idx] = w
            preds = np.array([df[col].to_numpy(dtype=float)[row_idx] for col in model_cols], dtype=float)
            hedge_mean[row_idx] = float(np.sum(w * preds))

    _add_feature(features, formulas, "hedge_shrunk_mean", hedge_mean, "shrunk_hedge")
    for i, col in enumerate(model_cols):
        _add_feature(features, formulas, f"hedge_shrunk_w_{col}", weights[:, i], "shrunk_weight")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex75(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_by_model = {col: y - df[col].to_numpy(dtype=float) for col in model_cols}
    resid_l2 = {
        col: pd.Series(resid_by_model[col], index=df.index).groupby(gk).shift(lag).to_numpy(dtype=float)
        for col in model_cols
    }

    med_mean = {}
    med_spread = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 30:
            idx = df.index.get_indexer(ctx.train_df.index)
        med_mean[station] = float(np.nanmedian(ens_mean[idx]))
        med_spread[station] = float(np.nanmedian(spread[idx]))

    weights = np.full((len(df), len(model_cols)), np.nan, dtype=float)
    regime_mean = np.full(len(df), np.nan, dtype=float)
    eta = 0.15
    regime_id = np.zeros(len(df), dtype=int)
    for idx in range(len(df)):
        t_mean = med_mean.get(df["station_id"].iloc[idx], float(np.nanmedian(ens_mean)))
        t_spread = med_spread.get(df["station_id"].iloc[idx], float(np.nanmedian(spread)))
        regime_id[idx] = int((ens_mean[idx] > t_mean) * 2 + (spread[idx] > t_spread))
    regime_l2 = pd.Series(regime_id, index=df.index).groupby(gk).shift(lag).fillna(0).to_numpy(dtype=int)

    for station in df["station_id"].unique():
        idx_sorted = df.sort_values(["station_id", "target_date_local"]).index
        idx_sorted = [i for i in idx_sorted if df.loc[i, "station_id"] == station]
        w_regimes = np.full((4, len(model_cols)), 1.0 / len(model_cols), dtype=float)
        t_mean = med_mean.get(station, float(np.nanmedian(ens_mean)))
        t_spread = med_spread.get(station, float(np.nanmedian(spread)))
        for row_idx in idx_sorted:
            reg_l2 = int(regime_l2[row_idx])
            losses = np.array([abs(resid_l2[col][row_idx]) for col in model_cols], dtype=float)
            losses = np.where(np.isfinite(losses), losses, 0.0)
            w_regimes[reg_l2] = w_regimes[reg_l2] * np.exp(-eta * losses)
            w_regimes[reg_l2] = np.maximum(w_regimes[reg_l2], 1e-6)
            w_regimes[reg_l2] = w_regimes[reg_l2] / np.sum(w_regimes[reg_l2])
            ens_val = ens_mean[row_idx]
            spr_val = spread[row_idx]
            p_hot = _sigmoid((ens_val - t_mean) / 2.0)
            p_spread = _sigmoid((spr_val - t_spread) / 0.5)
            p = np.array([
                (1 - p_hot) * (1 - p_spread),
                (1 - p_hot) * p_spread,
                p_hot * (1 - p_spread),
                p_hot * p_spread,
            ])
            p = p / np.sum(p)
            w_mix = p @ w_regimes
            weights[row_idx] = w_mix
            preds = np.array([df[col].to_numpy(dtype=float)[row_idx] for col in model_cols], dtype=float)
            regime_mean[row_idx] = float(np.sum(w_mix * preds))

    _add_feature(features, formulas, "regime_hedge_mean", regime_mean, "regime_hedge")
    for i, col in enumerate(model_cols):
        _add_feature(features, formulas, f"regime_hedge_w_{col}", weights[:, i], "regime_weight")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex76(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    resid_l2 = pd.Series(resid, index=df.index).groupby(gk).shift(lag).to_numpy(dtype=float)

    p_cp = np.full(len(df), np.nan, dtype=float)
    exp_run = np.full(len(df), np.nan, dtype=float)
    hazard = 1.0 / 120.0
    for station in df["station_id"].unique():
        idx = df.sort_values(["station_id", "target_date_local"]).index
        idx = [i for i in idx if df.loc[i, "station_id"] == station]
        series = np.array([resid_l2[i] for i in idx], dtype=float)
        cp_prob, exp_rl = _bocpd_student_t(series, hazard=hazard, max_run=400)
        for pos, row_idx in enumerate(idx):
            p_cp[row_idx] = cp_prob[pos]
            exp_run[row_idx] = exp_rl[pos]

    _add_feature(features, formulas, f"p_cp_l{lag}", p_cp, "BOCPD change prob (lag2)")
    _add_feature(features, formulas, f"exp_run_l{lag}", exp_run, "BOCPD expected run length")
    _add_feature(features, formulas, f"log_run_l{lag}", np.log1p(exp_run), "log(1+run_len)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex77(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]

    pc1 = np.full(len(df), np.nan, dtype=float)
    p_cp = np.full(len(df), np.nan, dtype=float)
    exp_run = np.full(len(df), np.nan, dtype=float)
    hazard = 1.0 / 120.0

    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        if len(idx_train) < 80:
            pc1[idx_all] = 0.0
            p_cp[idx_all] = 0.0
            exp_run[idx_all] = np.nan
            continue
        d_train = df.loc[idx_train, model_cols].to_numpy(dtype=float) - ens_mean[idx_train][:, None]
        pca = PCA(n_components=1, random_state=ctx.seed)
        pca.fit(d_train)
        d_all = df.loc[idx_all, model_cols].to_numpy(dtype=float) - ens_mean[idx_all][:, None]
        pc_all = pca.transform(d_all).ravel()
        pc1[idx_all] = pc_all

        idx_sorted = df.sort_values(["station_id", "target_date_local"]).index
        idx_sorted = [i for i in idx_sorted if df.loc[i, "station_id"] == station]
        pc_series = np.array([pc1[i] for i in idx_sorted], dtype=float)
        pc_l1 = pd.Series(pc_series).shift(1).to_numpy(dtype=float)
        cp_prob, exp_rl = _bocpd_student_t(pc_l1, hazard=hazard, max_run=400)
        for pos, row_idx in enumerate(idx_sorted):
            p_cp[row_idx] = cp_prob[pos]
            exp_run[row_idx] = exp_rl[pos]

    _add_feature(features, formulas, "pc1_disagreement", pc1, "PCA1(f_i-ens_mean)")
    _add_feature(features, formulas, "p_cp_pc1_l1", p_cp, "BOCPD change prob (pc1 lag1)")
    _add_feature(features, formulas, "exp_run_pc1_l1", exp_run, "BOCPD expected run length")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex78(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    resid_l2 = pd.Series(resid, index=df.index).groupby(gk).shift(lag)

    cp = _exp_ex76(ctx)
    p_cp = cp.features[f"p_cp_l{lag}"].to_numpy(dtype=float)

    bias_fast = tfl.ewm_mean(resid_l2, halflife=7, min_periods=10, lag=0, group_key=gk)
    bias_slow = tfl.ewm_mean(resid_l2, halflife=45, min_periods=20, lag=0, group_key=gk)
    gate = np.clip(p_cp, 0.0, 1.0)
    bias_cp = (1.0 - gate) * bias_slow.to_numpy(dtype=float) + gate * bias_fast.to_numpy(dtype=float)

    _add_feature(features, formulas, "bias_fast", bias_fast, "ewm_median_fast")
    _add_feature(features, formulas, "bias_slow", bias_slow, "ewm_median_slow")
    _add_feature(features, formulas, f"p_cp_l{lag}", p_cp, "BOCPD change prob")
    _add_feature(features, formulas, "bias_cp", bias_cp, "gate*bias_fast+(1-gate)*bias_slow")
    _add_feature(features, formulas, "ens_mean_cp_corr", ens_mean + bias_cp, "ens_mean + bias_cp")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex79(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid_mat = np.column_stack([y - df[col].to_numpy(dtype=float) for col in model_cols])
    resid_l2 = np.array([pd.Series(resid_mat[:, i], index=df.index).groupby(gk).shift(lag).to_numpy(dtype=float) for i in range(resid_mat.shape[1])]).T

    t2 = np.full(len(df), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx_sorted = df.sort_values(["station_id", "target_date_local"]).index
        idx_sorted = [i for i in idx_sorted if df.loc[i, "station_id"] == station]
        window = 90
        for pos, row_idx in enumerate(idx_sorted):
            start = max(0, pos - window + 1)
            hist_idx = idx_sorted[start:pos]
            if len(hist_idx) < 30:
                continue
            X = resid_l2[hist_idx]
            mu = np.nanmean(X, axis=0)
            cov = np.cov(X, rowvar=False)
            if cov.ndim == 0:
                cov = np.eye(len(model_cols)) * float(cov)
            cov = cov + 0.1 * np.eye(cov.shape[0])
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(cov)
            r = resid_l2[row_idx]
            if not np.all(np.isfinite(r)):
                continue
            diff = r - mu
            t2[row_idx] = float(diff.T @ inv @ diff)

    t2_ewm = (
        pd.Series(t2, index=df.index)
        .groupby(gk)
        .apply(lambda s: s.ewm(halflife=14, min_periods=10).mean())
        .to_numpy(dtype=float)
    )
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    thresh = float(np.nanquantile(t2_ewm[train_idx], 0.95)) if len(train_idx) else 0.0
    flag = (t2_ewm > thresh).astype(float)

    _add_feature(features, formulas, f"t2_drift_l{lag}", t2, "Hotelling_T2")
    _add_feature(features, formulas, "t2_ewm", t2_ewm, "ewm_mean(T2)")
    _add_feature(features, formulas, "t2_flag", flag, "T2_ewm > q95_train")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex80(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    deviations = df[model_cols].to_numpy(dtype=float) - ens_mean[:, None]
    x = np.column_stack([ens_mean, spread, deviations])

    mdist = np.full(len(df), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx_sorted = df.sort_values(["station_id", "target_date_local"]).index
        idx_sorted = [i for i in idx_sorted if df.loc[i, "station_id"] == station]
        window = 60
        for pos, row_idx in enumerate(idx_sorted):
            start = max(0, pos - window)
            hist_idx = idx_sorted[start:pos]
            if len(hist_idx) < 30:
                continue
            X = x[hist_idx]
            mu = np.nanmean(X, axis=0)
            cov = np.cov(X, rowvar=False)
            cov = cov + 0.3 * np.eye(cov.shape[0])
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(cov)
            diff = x[row_idx] - mu
            mdist[row_idx] = float(np.sqrt(np.maximum(diff.T @ inv @ diff, 0.0)))

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    med = float(np.nanmedian(mdist[train_idx])) if len(train_idx) else 0.0
    iqr = float(np.nanquantile(mdist[train_idx], 0.75) - np.nanquantile(mdist[train_idx], 0.25)) if len(train_idx) else 1.0
    iqr = iqr if iqr > 0 else 1.0
    mdist_z = (mdist - med) / (iqr + EPS)
    ood_flag = (mdist_z > 2.0).astype(float)

    _add_feature(features, formulas, "mdist_ood", mdist, "Mahalanobis distance")
    _add_feature(features, formulas, "mdist_ood_z", mdist_z, "(mdist-med)/iqr")
    _add_feature(features, formulas, "ood_flag", ood_flag, "mdist_ood_z > 2")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex81(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    deviations = df[model_cols].to_numpy(dtype=float) - ens_mean[:, None]
    X_all = np.column_stack([deviations, spread, sin_doy, cos_doy])

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    X_train = X_all[train_idx]

    gmm_global = GaussianMixture(n_components=5, covariance_type="diag", reg_covar=0.25**2, random_state=ctx.seed, max_iter=200)
    scaler_global = StandardScaler()
    Xs_global = scaler_global.fit_transform(X_train)
    gmm_global.fit(Xs_global)

    probs = np.full((len(df), 5), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        if len(idx_train) < 200:
            Xs = scaler_global.transform(X_all[idx_all])
            probs[idx_all] = gmm_global.predict_proba(Xs)
            continue
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_all[idx_train])
        gmm = GaussianMixture(n_components=5, covariance_type="diag", reg_covar=0.25**2, random_state=ctx.seed, max_iter=200)
        gmm.fit(Xs_train)
        Xs = scaler.transform(X_all[idx_all])
        probs[idx_all] = gmm.predict_proba(Xs)

    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    k_max = np.argmax(probs, axis=1)
    for k in range(5):
        _add_feature(features, formulas, f"gmm_p{k+1}", probs[:, k], "GMM posterior")
    _add_feature(features, formulas, "gmm_entropy", entropy, "entropy(p_k)")
    _add_feature(features, formulas, "gmm_k_max", k_max, "argmax_k p_k")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex82(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    gmm = _exp_ex81(ctx)
    probs = np.column_stack([gmm.features[f"gmm_p{k+1}"].to_numpy(dtype=float) for k in range(5)])
    k_max = np.argmax(probs, axis=1)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    kappa = 50.0

    m_k = np.zeros(5, dtype=float)
    n_k = np.zeros(5, dtype=float)
    for k in range(5):
        mask = (k_max[train_idx] == k)
        if np.any(mask):
            m_k[k] = float(np.median(resid[train_idx][mask]))
            n_k[k] = float(mask.sum())
        else:
            m_k[k] = float(np.median(resid[train_idx])) if len(train_idx) else 0.0
            n_k[k] = 0.0
    global_med = float(np.median(resid[train_idx])) if len(train_idx) else 0.0
    m_k = (n_k / (n_k + kappa)) * m_k + (kappa / (n_k + kappa)) * global_med
    bias_gmm = probs @ m_k

    _add_feature(features, formulas, "bias_gmm", bias_gmm, "sum(p_k * m_k)")
    _add_feature(features, formulas, "ens_mean_gmm_corr", ens_mean + bias_gmm, "ens_mean + bias_gmm")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex83(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    deviations = df[model_cols].to_numpy(dtype=float) - ens_mean[:, None]
    X_all = np.column_stack([deviations, spread, sin_doy, cos_doy])

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    X_train = X_all[train_idx]
    scaler_global = StandardScaler()
    Xs_global = scaler_global.fit_transform(X_train)
    params_global = hmm_utils.fit_gaussian_hmm(Xs_global, n_states=4, n_iters=10, seed=ctx.seed)

    probs = np.full((len(df), 4), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        if len(idx_train) < 200:
            Xs = scaler_global.transform(X_all[idx_all])
            probs[idx_all] = hmm_utils.forward_filter(Xs, params_global)
            continue
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_all[idx_train])
        params = hmm_utils.fit_gaussian_hmm(Xs_train, n_states=4, n_iters=10, seed=ctx.seed)
        Xs = scaler.transform(X_all[idx_all])
        probs[idx_all] = hmm_utils.forward_filter(Xs, params)

    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    switch_prob = np.full(len(df), np.nan, dtype=float)
    for idx in range(1, len(df)):
        switch_prob[idx] = 1.0 - np.sum(probs[idx - 1] * probs[idx])

    for k in range(4):
        _add_feature(features, formulas, f"hmm_p{k+1}", probs[:, k], "HMM posterior")
    _add_feature(features, formulas, "hmm_entropy", entropy, "entropy(p_k)")
    _add_feature(features, formulas, "hmm_switch_prob", switch_prob, "1-sum(p_t-1*p_t)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex84(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    X_all = df[model_cols].to_numpy(dtype=float)

    hmm = _exp_ex83(ctx)
    probs = np.column_stack([hmm.features[f"hmm_p{k+1}"].to_numpy(dtype=float) for k in range(4)])
    k_max = np.argmax(probs, axis=1)

    weights_per_state = {}
    global_weights, gk0, gk1, gk2 = _fit_simplex_lad(
        X_all[df.index.get_indexer(ctx.train_df.index)],
        y[df.index.get_indexer(ctx.train_df.index)],
        ridge=0.5,
        sin_doy=df.loc[ctx.train_df.index, "sin_doy"].to_numpy(dtype=float),
        cos_doy=df.loc[ctx.train_df.index, "cos_doy"].to_numpy(dtype=float),
    )
    for state in range(4):
        mask = (k_max[df.index.get_indexer(ctx.train_df.index)] == state)
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 100:
            weights_per_state[state] = global_weights
            continue
        w, _, _, _ = _fit_simplex_lad(
            X_all[idx],
            y[idx],
            ridge=0.5,
            sin_doy=df["sin_doy"].to_numpy(dtype=float)[idx],
            cos_doy=df["cos_doy"].to_numpy(dtype=float)[idx],
        )
        weights_per_state[state] = w

    weights = np.full((len(df), len(model_cols)), np.nan, dtype=float)
    hmm_moe = np.full(len(df), np.nan, dtype=float)
    for row_idx in range(len(df)):
        w_mix = np.zeros(len(model_cols), dtype=float)
        for state in range(4):
            w_mix += probs[row_idx, state] * weights_per_state.get(state, global_weights)
        w_mix = w_mix / np.sum(w_mix) if np.sum(w_mix) > 0 else global_weights
        weights[row_idx] = w_mix
        hmm_moe[row_idx] = float(np.sum(w_mix * X_all[row_idx]))

    _add_feature(features, formulas, "hmm_moe", hmm_moe, "sum(w_i(t)*f_i)")
    for i, col in enumerate(model_cols):
        _add_feature(features, formulas, f"hmm_w_{col}", weights[:, i], "hmm_weight")
    _add_feature(features, formulas, "hmm_entropy", hmm.features["hmm_entropy"].to_numpy(dtype=float), "entropy(p_k)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex85(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    ens_range = stats["range"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    range_edges = _quantile_edges(ens_range[train_idx], 3)
    range_bins = _bin_ids(ens_range, range_edges)
    values = df[model_cols].to_numpy(dtype=float)
    min_id = np.argmin(values, axis=1)
    max_id = np.argmax(values, axis=1)

    state_bias = np.full(len(df), np.nan, dtype=float)
    global_med = float(np.median(resid[train_idx])) if len(train_idx) else 0.0
    kappa = 60.0
    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        if len(idx_train) < 80:
            state_bias[idx_all] = global_med
            continue
        stats_map = {}
        for row_idx in idx_train:
            key = (min_id[row_idx], max_id[row_idx], range_bins[row_idx])
            stats_map.setdefault(key, []).append(resid[row_idx])
        med_map = {k: float(np.median(v)) for k, v in stats_map.items()}
        n_map = {k: len(v) for k, v in stats_map.items()}
        for row_idx in idx_all:
            key = (min_id[row_idx], max_id[row_idx], range_bins[row_idx])
            n_val = n_map.get(key, 0)
            med_val = med_map.get(key, global_med)
            w = n_val / (n_val + kappa) if n_val > 0 else 0.0
            state_bias[row_idx] = np.clip(w * med_val + (1.0 - w) * global_med, -3.0, 3.0)

    _add_feature(features, formulas, "state_bias_med", state_bias, "median(resid) by min/max/range_bin")
    _add_feature(features, formulas, "ens_mean_state_corr", ens_mean + state_bias, "ens_mean + state_bias")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex86(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean

    syn_cols = [col for col in ["gfs_n_x_max", "nam_n_x_max", "gefsatmosmean_tmax_f"] if col in df.columns]
    hires_cols = [col for col in ["hrrr_tmax_f", "rap_tmax_f"] if col in df.columns]
    syn_mean = _mean_available(df, syn_cols)
    hires_mean = _mean_available(df, hires_cols)
    sb_idx = np.clip(syn_mean - hires_mean, -10.0, 10.0)
    warm = df["month"].isin([5, 6, 7, 8, 9]).astype(int).to_numpy(dtype=int)

    bias = np.full(len(df), np.nan, dtype=float)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    global_med = float(np.median(resid[train_idx])) if len(train_idx) else 0.0
    kappa = 50.0

    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        if len(idx_train) < 60:
            bias[idx_all] = global_med
            continue
        warm_mask = warm[idx_train] == 1
        if warm_mask.sum() < 30:
            bias[idx_all] = global_med
            continue
        sb_train = sb_idx[idx_train][warm_mask]
        edges = _quantile_edges(sb_train, 10)
        bins_train = _bin_ids(sb_train, edges)
        cell_med = {}
        cell_n = {}
        resid_train = resid[idx_train][warm_mask]
        for b in range(11):
            mask = bins_train == b
            if not np.any(mask):
                continue
            cell_med[b] = float(np.median(resid_train[mask]))
            cell_n[b] = int(mask.sum())
        for row_idx in idx_all:
            if warm[row_idx] == 0:
                bias[row_idx] = global_med
                continue
            bin_id = _bin_ids(np.array([sb_idx[row_idx]]), edges)[0]
            n_val = cell_n.get(bin_id, 0)
            med_val = cell_med.get(bin_id, global_med)
            w = n_val / (n_val + kappa) if n_val > 0 else 0.0
            bias[row_idx] = np.clip(w * med_val + (1.0 - w) * global_med, -3.0, 3.0)

    _add_feature(features, formulas, "sb_idx", sb_idx, "syn_mean - hires_mean")
    _add_feature(features, formulas, "bias_sb_med", bias, "median(resid) by sb_bin")
    _add_feature(features, formulas, "ens_mean_sb_corr", ens_mean + bias, "ens_mean + sb_bias")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex87(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    ens_median = stats["median"]
    ens_min = stats["min"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    cool_tail = np.maximum(0.0, ens_median - ens_min)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    edges = _quantile_edges(cool_tail[train_idx], 5)
    bins = _bin_ids(cool_tail, edges)
    kappa = 50.0
    global_med = float(np.median(resid[train_idx])) if len(train_idx) else 0.0
    bias = np.full(len(df), np.nan, dtype=float)
    for b in range(6):
        mask = bins[train_idx] == b
        if not np.any(mask):
            continue
        med_val = float(np.median(resid[train_idx][mask]))
        n_val = int(mask.sum())
        w = n_val / (n_val + kappa)
        bias[bins == b] = np.clip(w * med_val + (1.0 - w) * global_med, -3.0, 3.0)
    bias = np.where(np.isnan(bias), global_med, bias)

    _add_feature(features, formulas, "cool_tail", cool_tail, "ens_median - ens_min")
    _add_feature(features, formulas, "bias_cooltail_med", bias, "median(resid) by cool_tail bin")
    _add_feature(features, formulas, "ens_mean_cooltail_corr", ens_mean + bias, "ens_mean + cooltail_bias")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex88(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    syn_cols = [col for col in ["gfs_n_x_max", "nam_n_x_max", "gefsatmosmean_tmax_f"] if col in df.columns]
    hires_cols = [col for col in ["hrrr_tmax_f", "rap_tmax_f"] if col in df.columns]
    syn_mean = _mean_available(df, syn_cols)
    hires_mean = _mean_available(df, hires_cols)
    sb_idx = np.clip(syn_mean - hires_mean, -10.0, 10.0)
    lag1 = pd.Series(sb_idx, index=df.index).groupby(gk).shift(1)
    sb_ewm7 = tfl.ewm_mean(lag1, halflife=3, min_periods=5, lag=0, group_key=gk).to_numpy(dtype=float)
    stats = _core_ensemble_stats(df, [col for col in MODEL_COLS if col in df.columns])
    spread = _spread_from_stats(df, stats)
    sb_spread = sb_ewm7 * spread

    _add_feature(features, formulas, "sb_idx", sb_idx, "syn_mean - hires_mean")
    _add_feature(features, formulas, "sb_ewm7", sb_ewm7, "ewm_mean(sb_idx_l1, HL=3)")
    _add_feature(features, formulas, "sb_spread", sb_spread, "sb_ewm7 * spread")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex89(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    values = df[model_cols].to_numpy(dtype=float)
    gap2 = np.array([_gap2(values[i]) for i in range(len(df))], dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    edges = _quantile_edges(gap2[train_idx], 5)
    bins = _bin_ids(gap2, edges)
    kappa = 50.0
    global_med = float(np.median(resid[train_idx])) if len(train_idx) else 0.0
    bias = np.full(len(df), np.nan, dtype=float)
    for b in range(6):
        mask = bins[train_idx] == b
        if not np.any(mask):
            continue
        med_val = float(np.median(resid[train_idx][mask]))
        n_val = int(mask.sum())
        w = n_val / (n_val + kappa)
        bias[bins == b] = np.clip(w * med_val + (1.0 - w) * global_med, -3.0, 3.0)
    bias = np.where(np.isnan(bias), global_med, bias)

    _add_feature(features, formulas, "gap2", gap2, "second_largest_gap")
    _add_feature(features, formulas, "bias_gap2_med", bias, "median(resid) by gap2 bin")
    _add_feature(features, formulas, "ens_mean_gap2_corr", ens_mean + bias, "ens_mean + gap2_bias")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex90(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    hires_cols = [col for col in ["hrrr_tmax_f", "rap_tmax_f"] if col in df.columns]
    hires_mean = _mean_available(df, hires_cols)
    nbm = df["nbm_tmax_f"].to_numpy(dtype=float) if "nbm_tmax_f" in df.columns else ens_mean
    tilt = np.clip(hires_mean - nbm, -8.0, 8.0)
    warm = df["month"].isin([5, 6, 7, 8, 9]).astype(int).to_numpy(dtype=int)

    bias = np.full(len(df), np.nan, dtype=float)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    global_med = float(np.median(resid[train_idx])) if len(train_idx) else 0.0
    kappa = 50.0
    edges = _quantile_edges(tilt[train_idx], 10)
    bins = _bin_ids(tilt, edges)

    for season in [0, 1]:
        mask_train = (warm[train_idx] == season)
        if mask_train.sum() < 30:
            bias[warm == season] = global_med
            continue
        for b in range(11):
            mask = (bins[train_idx] == b) & mask_train
            if not np.any(mask):
                continue
            med_val = float(np.median(resid[train_idx][mask]))
            n_val = int(mask.sum())
            w = n_val / (n_val + kappa)
            bias[(bins == b) & (warm == season)] = np.clip(w * med_val + (1.0 - w) * global_med, -3.0, 3.0)
    bias = np.where(np.isnan(bias), global_med, bias)

    _add_feature(features, formulas, "tilt", tilt, "hires_mean - nbm")
    _add_feature(features, formulas, "bias_tilt_med", bias, "median(resid) by tilt bin/season")
    _add_feature(features, formulas, "ens_mean_tilt_corr", ens_mean + bias, "ens_mean + tilt_bias")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex91(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    doy = df["day_of_year"].to_numpy(dtype=int)

    y_l2 = pd.Series(y, index=df.index).groupby(gk).shift(2)
    y_l3 = pd.Series(y, index=df.index).groupby(gk).shift(3)
    doy_l2 = pd.Series(doy, index=df.index).groupby(gk).shift(2)

    clim_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        train = ctx.train_df.loc[mask]
        if train.empty:
            continue
        train_doy = train["day_of_year"].to_numpy(dtype=int)
        train_y = train["actual_tmax_f"].to_numpy(dtype=float)
        climatology = np.full(366, np.nan, dtype=float)
        for d in range(1, 367):
            vals = train_y[train_doy == d]
            climatology[d - 1] = float(np.nanmean(vals)) if len(vals) else np.nan
        global_mean = float(np.nanmean(train_y)) if len(train_y) else 0.0
        climatology = np.where(np.isfinite(climatology), climatology, global_mean)
        climatology = _circular_smooth(climatology, bandwidth=7)
        clim_map[station] = climatology

    clim_t = np.full(len(df), np.nan, dtype=float)
    clim_l2 = np.full(len(df), np.nan, dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        clim = clim_map.get(station)
        if clim is None:
            continue
        d = int(doy[idx])
        d2 = int(doy_l2.to_numpy(dtype=float)[idx]) if np.isfinite(doy_l2.to_numpy(dtype=float)[idx]) else d
        d = max(1, min(d, 366))
        d2 = max(1, min(d2, 366))
        clim_t[idx] = clim[d - 1]
        clim_l2[idx] = clim[d2 - 1]

    anomaly_l2 = y_l2.to_numpy(dtype=float) - clim_l2
    _add_feature(features, formulas, "y_l2", y_l2, "y(t-2)")
    _add_feature(features, formulas, "y_l3", y_l3, "y(t-3)")
    _add_feature(features, formulas, "clim_l2", clim_l2, "climatology(doy_l2)")
    _add_feature(features, formulas, "anomaly_l2", anomaly_l2, "y_l2 - clim_l2")
    _add_feature(features, formulas, "abs_anomaly_l2", np.abs(anomaly_l2), "|anomaly_l2|")
    train_fitted.append(
        {
            "name": "climatology",
            "fit_on": "train",
            "stations": list(clim_map.keys()),
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex92(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    doy = df["day_of_year"].to_numpy(dtype=int)

    y_l2 = pd.Series(y, index=df.index).groupby(gk).shift(2).to_numpy(dtype=float)
    doy_l2 = pd.Series(doy, index=df.index).groupby(gk).shift(2).to_numpy(dtype=float)

    clim_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        train = ctx.train_df.loc[mask]
        if train.empty:
            continue
        train_doy = train["day_of_year"].to_numpy(dtype=int)
        train_y = train["actual_tmax_f"].to_numpy(dtype=float)
        climatology = np.full(366, np.nan, dtype=float)
        for d in range(1, 367):
            vals = train_y[train_doy == d]
            climatology[d - 1] = float(np.nanmean(vals)) if len(vals) else np.nan
        global_mean = float(np.nanmean(train_y)) if len(train_y) else 0.0
        climatology = np.where(np.isfinite(climatology), climatology, global_mean)
        climatology = _circular_smooth(climatology, bandwidth=7)
        clim_map[station] = climatology

    clim_t = np.full(len(df), np.nan, dtype=float)
    clim_t2 = np.full(len(df), np.nan, dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        clim = clim_map.get(station)
        if clim is None:
            continue
        d = int(doy[idx])
        d2 = int(doy_l2[idx]) if np.isfinite(doy_l2[idx]) else d
        d = max(1, min(d, 366))
        d2 = max(1, min(d2, 366))
        clim_t[idx] = clim[d - 1]
        clim_t2[idx] = clim[d2 - 1]

    delta = np.clip(clim_t - clim_t2, -5.0, 5.0)
    y_persist2 = y_l2 + delta
    _add_feature(features, formulas, "y_persist2", y_persist2, "y_l2 + (clim_t-clim_t2)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex93(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    persist = _exp_ex92(ctx)
    y_persist2 = persist.features["y_persist2"].to_numpy(dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    med = float(np.nanmedian(spread[train_idx])) if len(train_idx) else 0.0
    w = _sigmoid((med - spread) / 0.5)
    w = np.clip(w, 0.05, 0.95)
    blend = w * ens_mean + (1.0 - w) * y_persist2

    _add_feature(features, formulas, "persist_gate", w, "sigmoid((S-spread)/0.5)")
    _add_feature(features, formulas, "blend_persist", blend, "w*ens_mean + (1-w)*y_persist2")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex94(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    doy = df["day_of_year"].to_numpy(dtype=int)

    clim_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        train = ctx.train_df.loc[mask]
        if train.empty:
            continue
        train_doy = train["day_of_year"].to_numpy(dtype=int)
        train_y = train["actual_tmax_f"].to_numpy(dtype=float)
        climatology = np.full(366, np.nan, dtype=float)
        for d in range(1, 367):
            vals = train_y[train_doy == d]
            climatology[d - 1] = float(np.nanmean(vals)) if len(vals) else np.nan
        global_mean = float(np.nanmean(train_y)) if len(train_y) else 0.0
        climatology = np.where(np.isfinite(climatology), climatology, global_mean)
        climatology = _circular_smooth(climatology, bandwidth=7)
        clim_map[station] = climatology

    clim_t = np.full(len(df), np.nan, dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        clim = clim_map.get(station)
        if clim is None:
            continue
        d = int(doy[idx])
        d = max(1, min(d, 366))
        clim_t[idx] = clim[d - 1]

    anom = y - clim_t
    anom_l2 = pd.Series(anom, index=df.index).groupby(gk).shift(2)

    phi_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) < 50:
            phi_map[station] = 0.0
            continue
        anom_train = anom[idx]
        anom_l2_train = anom_l2.to_numpy(dtype=float)[idx]
        mask_valid = np.isfinite(anom_train) & np.isfinite(anom_l2_train)
        if mask_valid.sum() < 30:
            phi_map[station] = 0.0
            continue
        num = float(np.sum(anom_train[mask_valid] * anom_l2_train[mask_valid]))
        den = float(np.sum(anom_l2_train[mask_valid] ** 2))
        phi = num / den if den > 0 else 0.0
        phi_map[station] = float(np.clip(phi, 0.0, 0.9))

    y_ar2 = np.full(len(df), np.nan, dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        phi = phi_map.get(station, 0.0)
        anom_val = anom_l2.to_numpy(dtype=float)[idx]
        y_ar2[idx] = clim_t[idx] + phi * anom_val if np.isfinite(anom_val) else np.nan

    _add_feature(features, formulas, "y_ar2", y_ar2, "clim_t + phi*anom_l2")
    _add_feature(features, formulas, "phi_ar2", np.array([phi_map.get(s, 0.0) for s in df["station_id"]]), "phi per station")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex95(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    y_l2 = pd.Series(y, index=df.index).groupby(gk).shift(2).to_numpy(dtype=float)
    slope = np.full(len(df), np.nan, dtype=float)
    trend = np.full(len(df), np.nan, dtype=float)

    for station in df["station_id"].unique():
        idx_sorted = df.sort_values(["station_id", "target_date_local"]).index
        idx_sorted = [i for i in idx_sorted if df.loc[i, "station_id"] == station]
        for pos, row_idx in enumerate(idx_sorted):
            end = pos
            start = max(0, end - 8)
            window_idx = idx_sorted[start : end + 1]
            vals = y_l2[window_idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 6:
                slope[row_idx] = 0.0
                trend[row_idx] = y_l2[row_idx]
                continue
            s = _theil_sen_slope(vals)
            s = float(np.clip(s, -3.0, 3.0))
            slope[row_idx] = s
            trend[row_idx] = y_l2[row_idx] + 2.0 * s

    _add_feature(features, formulas, "slope_ts_9", slope, "Theil-Sen slope over y_l2..y_l10")
    _add_feature(features, formulas, "trend_extrap", trend, "y_l2 + 2*slope_ts_9")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex96(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    warm = df["month"].isin([5, 6, 7, 8, 9]).astype(int).to_numpy(dtype=int)
    kappa = 60.0

    b_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        train = ctx.train_df.loc[mask]
        if train.empty:
            continue
        for col in model_cols:
            vals = train["actual_tmax_f"].to_numpy(dtype=float) - train[col].to_numpy(dtype=float)
            b_all = float(np.median(vals)) if len(vals) else 0.0
            for season in [0, 1]:
                season_mask = train["month"].isin([5, 6, 7, 8, 9]) if season == 1 else ~train["month"].isin([5, 6, 7, 8, 9])
                res = vals[season_mask.to_numpy()]
                if len(res) == 0:
                    b_map[(station, col, season)] = b_all
                    continue
                b_m = float(np.median(res))
                n_m = len(res)
                b_shrunk = (n_m / (n_m + kappa)) * b_m + (kappa / (n_m + kappa)) * b_all
                b_map[(station, col, season)] = b_shrunk

    corrected = np.zeros((len(df), len(model_cols)), dtype=float)
    for i, col in enumerate(model_cols):
        col_vals = df[col].to_numpy(dtype=float)
        for idx, station in enumerate(df["station_id"].to_numpy()):
            season = warm[idx]
            bias = b_map.get((station, col, season), 0.0)
            corrected[idx, i] = col_vals[idx] + bias
        _add_feature(features, formulas, f"corr_{col}", corrected[:, i], "model + seasonal_bias")

    median_corr = np.median(corrected, axis=1)
    spread_corr = np.std(corrected, axis=1, ddof=0)

    _add_feature(features, formulas, "median_corr", median_corr, "median(corrected_models)")
    _add_feature(features, formulas, "spread_corr", spread_corr, "std(corrected_models)")
    train_fitted.append(
        {
            "name": "seasonal_bias_per_model",
            "fit_on": "train",
            "kappa": kappa,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex97(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    y = df["actual_tmax_f"].to_numpy(dtype=float)

    corr = _exp_ex96(ctx)
    corrected = np.column_stack(
        [corr.features[f"corr_{col}"].to_numpy(dtype=float) for col in model_cols]
    )

    mae_state = np.full((len(df), len(model_cols)), np.nan, dtype=float)
    for i, col in enumerate(model_cols):
        resid = y - df[col].to_numpy(dtype=float)
        resid_l2 = pd.Series(resid, index=df.index).groupby(gk).shift(lag)
        mae_state[:, i] = tfl.ewm_mean(resid_l2.abs(), halflife=21, min_periods=10, lag=0, group_key=gk).to_numpy(dtype=float)
    weights = 1.0 / np.maximum(0.3, mae_state)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    wmedian = _rowwise_weighted_median(corrected, weights)

    _add_feature(features, formulas, "wmedian_corr", wmedian, "weighted_median(corrected_models)")
    for i, col in enumerate(model_cols):
        _add_feature(features, formulas, f"wmedian_w_{col}", weights[:, i], "1/mae_state")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex98(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    corr = _exp_ex96(ctx)
    corrected = np.column_stack(
        [corr.features[f"corr_{col}"].to_numpy(dtype=float) for col in model_cols]
    )
    huber_loc = np.full(len(df), np.nan, dtype=float)
    huber_scale = np.full(len(df), np.nan, dtype=float)

    for idx in range(len(df)):
        vals = corrected[idx]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = max(0.5, 1.4826 * mad)
        delta = 1.5 * scale
        m = med
        for _ in range(10):
            r = vals - m
            w = np.where(np.abs(r) <= delta, 1.0, delta / (np.abs(r) + EPS))
            m = float(np.sum(w * vals) / np.sum(w))
        huber_loc[idx] = m
        huber_scale[idx] = scale

    _add_feature(features, formulas, "huber_loc", huber_loc, "Huber M-estimator")
    _add_feature(features, formulas, "huber_scale", huber_scale, "1.4826*MAD")
    _add_feature(features, formulas, "huber_loc_plus", huber_loc + huber_scale, "huber_loc + scale")
    _add_feature(features, formulas, "huber_loc_minus", huber_loc - huber_scale, "huber_loc - scale")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex99(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]

    b_map = {}
    s_map = {}
    w_map = {}
    for station in df["station_id"].unique():
        idx_train = df.index.get_indexer(ctx.train_df.index[ctx.train_df["station_id"] == station])
        idx_train = idx_train[idx_train >= 0]
        if len(idx_train) < 80:
            idx_train = train_idx
        mu = []
        sigma = []
        for col in model_cols:
            r = y[idx_train] - df[col].to_numpy(dtype=float)[idx_train]
            med = float(np.median(r))
            mad = float(np.median(np.abs(r - med)))
            b_map[(station, col)] = med
            s_val = max(0.25, 1.4826 * mad)
            s_map[(station, col)] = s_val
            mu.append(df[col].to_numpy(dtype=float)[idx_train] + med)
            sigma.append(np.full(len(idx_train), s_val))
        mu = np.column_stack(mu)
        sigma = np.column_stack(sigma)
        weights = _bma_fit_weights(y[idx_train], mu, sigma)
        if not np.isfinite(weights).all():
            inv = np.array([1.0 / s_map[(station, col)] for col in model_cols], dtype=float)
            weights = inv / np.sum(inv)
        w_map[station] = weights

    bma_median = np.full(len(df), np.nan, dtype=float)
    for station in df["station_id"].unique():
        idx_all = np.where(df["station_id"].to_numpy() == station)[0]
        weights = w_map.get(station)
        mu = np.column_stack(
            [df[col].to_numpy(dtype=float)[idx_all] + b_map[(station, col)] for col in model_cols]
        )
        sigma = np.column_stack([np.full(len(idx_all), s_map[(station, col)]) for col in model_cols])
        for pos, row_idx in enumerate(idx_all):
            bma_median[row_idx] = _bma_mixture_median(mu[pos], sigma[pos], weights)

    _add_feature(features, formulas, "bma_median", bma_median, "mixture_median")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex100(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    quant = _exp_ex70(ctx)
    resid_skew = quant.features["resid_skew_2d"].to_numpy(dtype=float)
    skew_sign = np.sign(resid_skew)
    skew_mag = np.clip(np.abs(resid_skew), 0.0, 2.0)

    _add_feature(features, formulas, "resid_skew_2d", resid_skew, "q75+q25-2*q50")
    _add_feature(features, formulas, "skew_sign", skew_sign, "sign(resid_skew)")
    _add_feature(features, formulas, "skew_mag", skew_mag, "|resid_skew|")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex101(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    month = df["month"].to_numpy(dtype=int)
    kappa = 50.0

    bias_map = {}
    for col in model_cols:
        train_vals = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float) - ctx.train_df[col].to_numpy(dtype=float)
        b_all = float(np.median(train_vals)) if len(train_vals) else 0.0
        for m in range(1, 13):
            mask = ctx.train_df["month"] == m
            vals = train_vals[mask.to_numpy()]
            if len(vals) == 0:
                continue
            b_m = float(np.median(vals))
            n_m = len(vals)
            b_shrunk = (n_m / (n_m + kappa)) * b_m + (kappa / (n_m + kappa)) * b_all
            bias_map[(col, m)] = b_shrunk

    corrected = np.zeros((len(df), len(model_cols)), dtype=float)
    for i, col in enumerate(model_cols):
        col_vals = df[col].to_numpy(dtype=float)
        for idx in range(len(df)):
            m = int(month[idx])
            bias = bias_map.get((col, m), 0.0)
            corrected[idx, i] = col_vals[idx] + bias

    month_corr_median = np.median(corrected, axis=1)
    _add_feature(features, formulas, "month_corr_median", month_corr_median, "median(corrected_models_month)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex102(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    sin = df["sin_doy"].to_numpy(dtype=float)
    cos = df["cos_doy"].to_numpy(dtype=float)
    radians = 2 * np.pi * df["day_of_year"].to_numpy(dtype=float) / 365.25
    sin2 = np.sin(2 * radians)
    cos2 = np.cos(2 * radians)
    y = df["actual_tmax_f"].to_numpy(dtype=float)

    bias_season = np.zeros((len(df), len(model_cols)), dtype=float)
    anom_ewm = np.zeros_like(bias_season)
    for i, col in enumerate(model_cols):
        train = ctx.train_df
        radians_t = 2 * np.pi * train["day_of_year"].to_numpy(dtype=float) / 365.25
        X = np.column_stack(
            [
                np.ones(len(train)),
                train["sin_doy"].to_numpy(dtype=float),
                train["cos_doy"].to_numpy(dtype=float),
                np.sin(2 * radians_t),
                np.cos(2 * radians_t),
            ]
        )
        r = train["actual_tmax_f"].to_numpy(dtype=float) - train[col].to_numpy(dtype=float)
        coef, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
        X_all = np.column_stack([np.ones(len(df)), sin, cos, sin2, cos2])
        bias_season[:, i] = X_all @ coef
        r_full = y - df[col].to_numpy(dtype=float) - bias_season[:, i]
        r_l2 = pd.Series(r_full, index=df.index).groupby(gk).shift(lag)
        anom_ewm[:, i] = tfl.ewm_mean(r_l2, halflife=21, min_periods=10, lag=0, group_key=gk).to_numpy(dtype=float)
        anom_ewm[:, i] = np.clip(anom_ewm[:, i], -2.0, 2.0)

    corrected = df[model_cols].to_numpy(dtype=float) + bias_season + anom_ewm
    corr_median = np.median(corrected, axis=1)
    corr_mean = np.mean(corrected, axis=1)

    _add_feature(features, formulas, "corr_median", corr_median, "median(f_i + bias_season + anom_ewm)")
    _add_feature(features, formulas, "corr_mean", corr_mean, "mean(f_i + bias_season + anom_ewm)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex103(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = ctx.truth_lag
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    resid_l2 = pd.Series(resid, index=df.index).groupby(gk).shift(lag).to_numpy(dtype=float)

    b_bandit = np.full(len(df), np.nan, dtype=float)
    w_store = np.full((len(df), 3), np.nan, dtype=float)
    eta = 0.4
    lam = 0.2
    for station in df["station_id"].unique():
        idx = df.sort_values(["station_id", "target_date_local"]).index
        idx = [i for i in idx if df.loc[i, "station_id"] == station]
        b = np.zeros(3, dtype=float)
        w = np.full(3, 1.0 / 3.0, dtype=float)
        for row_idx in idx:
            r_val = resid_l2[row_idx]
            if np.isfinite(r_val):
                b[0] = b[0] + _ewm_alpha(7) * (r_val - b[0])
                b[1] = b[1] + _ewm_alpha(21) * (r_val - b[1])
                b[2] = b[2] + _ewm_alpha(60) * (r_val - b[2])
                loss = np.abs(r_val - b)
                w = w * np.exp(-eta * loss)
                w = w / np.sum(w)
                w = (1.0 - lam) * w + lam * (1.0 / 3.0)
            w_store[row_idx] = w
            b_bandit[row_idx] = float(np.sum(w * b))

    _add_feature(features, formulas, "b_bandit", b_bandit, "bandit-weighted bias")
    _add_feature(features, formulas, "w7", w_store[:, 0], "bandit_weight_7")
    _add_feature(features, formulas, "w21", w_store[:, 1], "bandit_weight_21")
    _add_feature(features, formulas, "w60", w_store[:, 2], "bandit_weight_60")
    _add_feature(features, formulas, "ens_mean_bandit_corr", ens_mean + b_bandit, "ens_mean + b_bandit")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex104(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    abs_err = np.abs(y - ens_mean)
    kappa = 80.0

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    global_c0, global_c1 = _fit_nnls_scale(spread[train_idx], abs_err[train_idx])

    c0_map = {}
    c1_map = {}
    for m in range(1, 13):
        mask = ctx.train_df["month"] == m
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) == 0:
            c0_map[m] = global_c0
            c1_map[m] = global_c1
            continue
        c0, c1 = _fit_nnls_scale(spread[idx], abs_err[idx])
        n_m = len(idx)
        c0 = (n_m / (n_m + kappa)) * c0 + (kappa / (n_m + kappa)) * global_c0
        c1 = (n_m / (n_m + kappa)) * c1 + (kappa / (n_m + kappa)) * global_c1
        c0_map[m] = c0
        c1_map[m] = c1

    abs_err_hat = np.full(len(df), np.nan, dtype=float)
    for idx, m in enumerate(df["month"].to_numpy(dtype=int)):
        c0 = c0_map.get(int(m), global_c0)
        c1 = c1_map.get(int(m), global_c1)
        abs_err_hat[idx] = c0 + c1 * spread[idx]

    _add_feature(features, formulas, "abs_err_hat", abs_err_hat, "month_shrunk_scale")
    _add_feature(features, formulas, "spread", spread, "spread")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex105(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    X_all = np.column_stack([ens_mean, spread, sin_doy, cos_doy])
    y = df["actual_tmax_f"].to_numpy(dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    intercept, coef, mean, scale = _fit_ridge_linear(X_all[train_idx], y[train_idx], alpha=10.0)
    base_pred = _predict_linear(X_all, intercept, coef, mean, scale)
    resid = y - base_pred
    kappa = 100.0
    offset_map = {}
    for station in df["station_id"].unique():
        mask = ctx.train_df["station_id"] == station
        idx = df.index.get_indexer(ctx.train_df.index[mask])
        idx = idx[idx >= 0]
        if len(idx) == 0:
            offset_map[station] = 0.0
            continue
        offset = float(np.mean(resid[idx]))
        n = len(idx)
        offset_map[station] = (n / (n + kappa)) * offset

    hier_mu = np.full(len(df), np.nan, dtype=float)
    for idx, station in enumerate(df["station_id"].to_numpy()):
        hier_mu[idx] = base_pred[idx] + offset_map.get(station, 0.0)

    _add_feature(features, formulas, "hier_emos_mu", hier_mu, "global_ridge + station_offset")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex106(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    ens_std = stats["std"]
    ens_range = stats["range"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    values = df[model_cols].to_numpy(dtype=float)
    gap2 = np.array([_gap2(values[i]) for i in range(len(df))], dtype=float)
    sb = _exp_ex86(ctx)
    sb_idx = sb.features["sb_idx"].to_numpy(dtype=float)
    pc = _exp_ex67(ctx)
    pc1 = pc.features["pc1_disagreement"].to_numpy(dtype=float)

    X = np.column_stack([ens_mean, spread, ens_std, ens_range, gap2, sb_idx, pc1, sin_doy, cos_doy])
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    abs_err = np.abs(y - ens_mean)
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    thr = float(np.quantile(abs_err[train_idx], 0.90)) if len(train_idx) else 1.0
    y_train = (abs_err[train_idx] > thr).astype(int)
    X_train = X[train_idx]

    def _build() -> LogisticRegression:
        return LogisticRegression(max_iter=500, random_state=ctx.seed)

    oof, model_full = _time_oof_classifier(
        ctx.train_df,
        X_train,
        y_train,
        build_model=_build,
        n_splits=5,
        gap_days=2,
    )
    p_all = model_full.predict_proba(X)[:, 1]
    train_pos = df.index.get_indexer(ctx.train_df.index)
    oof_mask = np.isfinite(oof)
    p_all[train_pos[oof_mask]] = oof[oof_mask]

    _add_feature(features, formulas, "p_large_err", p_all, "P(|err|>q90)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex107(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    p_large = _exp_ex106(ctx).features["p_large_err"].to_numpy(dtype=float)
    wmedian = _exp_ex97(ctx).features["wmedian_corr"].to_numpy(dtype=float)
    g = np.clip(p_large, 0.05, 0.8)
    y_hedged = (1.0 - g) * ens_mean + g * wmedian

    _add_feature(features, formulas, "p_large_err", p_large, "P(|err|>q90)")
    _add_feature(features, formulas, "y_hedged", y_hedged, "(1-g)*ens_mean + g*wmedian")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex108(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    ens_range = stats["range"]
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    values = df[model_cols].to_numpy(dtype=float)
    gap2 = np.array([_gap2(values[i]) for i in range(len(df))], dtype=float)
    sb = _exp_ex86(ctx)
    sb_idx = sb.features["sb_idx"].to_numpy(dtype=float)
    pc = _exp_ex67(ctx)
    pc1 = pc.features["pc1_disagreement"].to_numpy(dtype=float)
    cool_tail = _exp_ex87(ctx).features["cool_tail"].to_numpy(dtype=float)

    X = np.column_stack([ens_mean, spread, ens_range, gap2, sb_idx, pc1, cool_tail, sin_doy, cos_doy])
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_mask = train_idx >= 0
    train_idx = train_idx[train_mask]
    labeler = rs_moe.BustRegimeLabeler(
        rs_moe.BustRegimeLabelerConfig(type="ex108_compat"),
        model_cols=model_cols,
        target_col="actual_tmax_f",
    ).fit(ctx.train_df)
    y_train = labeler.transform(ctx.train_df)[train_mask]
    X_train = X[train_idx]

    def _build() -> LogisticRegression:
        return LogisticRegression(max_iter=500, random_state=ctx.seed, multi_class="multinomial")

    oof, model_full = _time_oof_classifier_multiclass(
        ctx.train_df,
        X_train,
        y_train,
        build_model=_build,
        n_splits=5,
        gap_days=2,
        n_classes=3,
    )
    probs = model_full.predict_proba(X)
    probs_full = np.zeros((len(df), 3), dtype=float)
    for cls_idx, cls in enumerate(model_full.classes_):
        probs_full[:, int(cls)] = probs[:, cls_idx]
    train_pos = df.index.get_indexer(ctx.train_df.index)
    oof_mask = np.isfinite(oof).all(axis=1)
    probs_full[train_pos[oof_mask]] = oof[oof_mask]

    _add_feature(features, formulas, "p_cool", probs_full[:, 0], "P(cool bust)")
    _add_feature(features, formulas, "p_norm", probs_full[:, 1], "P(normal)")
    _add_feature(features, formulas, "p_warm", probs_full[:, 2], "P(warm bust)")
    _add_feature(features, formulas, "p_bust", 1.0 - probs_full[:, 1], "1-P(normal)")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex109(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    probs = _exp_ex108(ctx)
    p_cool = probs.features["p_cool"].to_numpy(dtype=float)
    p_norm = probs.features["p_norm"].to_numpy(dtype=float)
    p_warm = probs.features["p_warm"].to_numpy(dtype=float)

    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    a = float(np.quantile(np.abs(resid[train_idx]), 0.60)) if len(train_idx) else 1.0
    m_cool = float(np.median(resid[train_idx][resid[train_idx] < -a])) if len(train_idx) else 0.0
    m_norm = float(np.median(resid[train_idx][np.abs(resid[train_idx]) <= a])) if len(train_idx) else 0.0
    m_warm = float(np.median(resid[train_idx][resid[train_idx] > a])) if len(train_idx) else 0.0
    kappa = 100.0
    m_cool = (len(resid[train_idx]) / (len(resid[train_idx]) + kappa)) * m_cool
    m_warm = (len(resid[train_idx]) / (len(resid[train_idx]) + kappa)) * m_warm
    corr = p_cool * m_cool + p_norm * m_norm + p_warm * m_warm
    corr = np.clip(corr, -2.0, 2.0)

    _add_feature(features, formulas, "ens_mean_dir_corr", ens_mean + corr, "ens_mean + directional_correction")
    _add_feature(features, formulas, "dir_corr", corr, "p_cool*m_cool + p_norm*m_norm + p_warm*m_warm")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex110(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = [col for col in MODEL_COLS if col in df.columns]
    stats = _core_ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = _spread_from_stats(df, stats)
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    pc = _exp_ex67(ctx)
    pc1 = pc.features["pc1_disagreement"].to_numpy(dtype=float)
    sb = _exp_ex86(ctx)
    sb_idx = sb.features["sb_idx"].to_numpy(dtype=float)
    tilt = _exp_ex90(ctx).features["tilt"].to_numpy(dtype=float)
    cool_tail = _exp_ex87(ctx).features["cool_tail"].to_numpy(dtype=float)

    X = np.column_stack([ens_mean, spread, pc1, sb_idx, tilt, cool_tail, sin_doy, cos_doy])
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - ens_mean
    train_idx = df.index.get_indexer(ctx.train_df.index)
    train_idx = train_idx[train_idx >= 0]
    X_train = X[train_idx]
    y_train = resid[train_idx]

    def _build_q(q: float) -> QuantileRegressor:
        return QuantileRegressor(quantile=q, alpha=1.0, solver="highs")

    oof_q50, model_q50 = _time_oof_regression(
        ctx.train_df,
        X_train,
        y_train,
        build_model=lambda: _build_q(0.5),
        n_splits=5,
        gap_days=2,
    )
    oof_q25, model_q25 = _time_oof_regression(
        ctx.train_df,
        X_train,
        y_train,
        build_model=lambda: _build_q(0.25),
        n_splits=5,
        gap_days=2,
    )
    oof_q75, model_q75 = _time_oof_regression(
        ctx.train_df,
        X_train,
        y_train,
        build_model=lambda: _build_q(0.75),
        n_splits=5,
        gap_days=2,
    )

    pred_q50 = model_q50.predict(X)
    pred_q25 = model_q25.predict(X)
    pred_q75 = model_q75.predict(X)
    train_pos = df.index.get_indexer(ctx.train_df.index)
    pred_q50[train_pos] = oof_q50
    pred_q25[train_pos] = oof_q25
    pred_q75[train_pos] = oof_q75

    iqr_hat = np.clip(pred_q75 - pred_q25, 0.3, 6.0)
    pred_q50 = np.clip(pred_q50, -2.0, 2.0)
    ens_mean_corr = ens_mean + pred_q50

    _add_feature(features, formulas, "e_q50_hat", pred_q50, "quantile_regression_tau_0.5")
    _add_feature(features, formulas, "iqr_hat", iqr_hat, "q75_hat - q25_hat")
    _add_feature(features, formulas, "ens_mean_meta_corr", ens_mean_corr, "ens_mean + e_q50_hat")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex201(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_required_forecast_cols_6(df: pd.DataFrame) -> list[str]:
    cols = [
        "gefsatmosmean_tmax_f",
        "rap_tmax_f",
        "hrrr_tmax_f",
        "nbm_tmax_f",
        "gfs_n_x_max",
        "nam_n_x_max",
    ]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            "EX202-EX211 require core forecast columns "
            f"{cols} but missing {missing}. "
            "Add them to features.base_features in your config."
        )
    return cols


def _exp_ex202(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = _exp_required_forecast_cols_6(df)
    include = [
        "ens_mean_6",
        "ens_median_6",
        "ens_trimmed_mean_6_1",
        "ens_winsor_mean_6_1",
        "ens_std_6",
        "ens_range_6",
        "ens_iqr_6",
        "ens_mad_6",
    ]
    rowwise = derived_features.compute_rowwise_features(df, model_cols, include=include)
    rowwise = rowwise.rename(
        columns={
            "ens_trimmed_mean_6_1": "ens_trimmed_mean_k1",
            "ens_winsor_mean_6_1": "ens_winsor_mean_k1",
        }
    )

    _add_feature(features, formulas, "ens_mean_6", rowwise["ens_mean_6"], "mean(models_6)")
    _add_feature(features, formulas, "ens_median_6", rowwise["ens_median_6"], "median(models_6)")
    _add_feature(
        features, formulas, "ens_trimmed_mean_k1", rowwise["ens_trimmed_mean_k1"], "trimmed_mean(k=1)"
    )
    _add_feature(
        features, formulas, "ens_winsor_mean_k1", rowwise["ens_winsor_mean_k1"], "winsor_mean(k=1)"
    )
    _add_feature(features, formulas, "ens_std_6", rowwise["ens_std_6"], "std(models_6)")
    _add_feature(features, formulas, "ens_range_6", rowwise["ens_range_6"], "max(models_6)-min(models_6)")
    _add_feature(features, formulas, "ens_iqr_6", rowwise["ens_iqr_6"], "q75-q25(models_6)")
    _add_feature(features, formulas, "ens_mad_6", rowwise["ens_mad_6"], "median(|x-median|)")
    ens_skew_proxy = rowwise["ens_mean_6"].to_numpy(dtype=float) - rowwise["ens_median_6"].to_numpy(dtype=float)
    _add_feature(features, formulas, "ens_skew_proxy", ens_skew_proxy, "ens_mean_6 - ens_median_6")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex203(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = _exp_required_forecast_cols_6(df)
    values = df[model_cols].to_numpy(dtype=float)
    n = len(model_cols)
    ens_mean = np.mean(values, axis=1)
    ens_median = np.median(values, axis=1)

    _add_feature(features, formulas, "ens_mean_6", ens_mean, "mean(models_6)")
    _add_feature(features, formulas, "ens_median_6", ens_median, "median(models_6)")

    for idx, col in enumerate(model_cols):
        dev_mean = values[:, idx] - ens_mean
        dev_median = values[:, idx] - ens_median
        _add_feature(features, formulas, f"dev_mean_{col}", dev_mean, f"{col} - ens_mean_6")
        _add_feature(features, formulas, f"dev_median_{col}", dev_median, f"{col} - ens_median_6")
        _add_feature(features, formulas, f"abs_dev_mean_{col}", np.abs(dev_mean), f"|{col} - ens_mean_6|")
        _add_feature(features, formulas, f"abs_dev_median_{col}", np.abs(dev_median), f"|{col} - ens_median_6|")

    ranks = derived_features.rank_with_tie_break(
        values,
        model_cols,
        tie_breaker="column_name_lexicographic",
        zero_based=True,
    )
    denom = float(max(n - 1, 1))
    for idx, col in enumerate(model_cols):
        _add_feature(features, formulas, f"rank_{col}", ranks[:, idx], "rank_in_ens_0_based")
        _add_feature(features, formulas, f"rank_{col}_norm", ranks[:, idx] / denom, "rank/(n-1)")

    train_fitted.append(
        {
            "name": "ensemble_rank_deviation",
            "tie_breaker": "column_name_lexicographic",
            "rank_zero_based": True,
            "n_models": n,
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _sign_with_epsilon(values: np.ndarray, eps: float) -> np.ndarray:
    return np.where(values > eps, 1.0, np.where(values < -eps, -1.0, 0.0))


def _exp_ex204(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []

    required = [
        "nbm_tmax_f",
        "gefsatmosmean_tmax_f",
        "hrrr_tmax_f",
        "rap_tmax_f",
    ]
    _ensure_columns_exist(df, required, "EX204")

    gfs_col = "gfs_n_x_max" if "gfs_n_x_max" in df.columns else "gfs_tmax_f"
    nam_col = "nam_n_x_max" if "nam_n_x_max" in df.columns else "nam_tmax_f"
    _ensure_columns_exist(df, [gfs_col, nam_col], "EX204")

    eps = 0.1
    diff_nbm_gefs = df["nbm_tmax_f"].to_numpy(dtype=float) - df["gefsatmosmean_tmax_f"].to_numpy(dtype=float)
    diff_hrrr_rap = df["hrrr_tmax_f"].to_numpy(dtype=float) - df["rap_tmax_f"].to_numpy(dtype=float)
    diff_gfs_nam = df[gfs_col].to_numpy(dtype=float) - df[nam_col].to_numpy(dtype=float)

    for name, diff, formula in (
        ("diff_nbm_gefs", diff_nbm_gefs, "nbm - gefs"),
        ("diff_hrrr_rap", diff_hrrr_rap, "hrrr - rap"),
        ("diff_gfs_nam", diff_gfs_nam, f"{gfs_col} - {nam_col}"),
    ):
        _add_feature(features, formulas, name, diff, formula)
        _add_feature(features, formulas, f"abs_{name}", np.abs(diff), f"|{formula}|")
        _add_feature(features, formulas, f"sign_{name}", _sign_with_epsilon(diff, eps), f"sign_eps({eps})")

    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex205(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = _exp_required_forecast_cols_6(df)
    sin = df["sin_doy"].to_numpy(dtype=float)
    cos = df["cos_doy"].to_numpy(dtype=float)
    for col in model_cols:
        vals = df[col].to_numpy(dtype=float)
        _add_feature(features, formulas, f"{col}_x_sin_doy", vals * sin, f"{col} * sin_doy")
        _add_feature(features, formulas, f"{col}_x_cos_doy", vals * cos, f"{col} * cos_doy")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex206(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = _exp_required_forecast_cols_6(df)

    q_lo = 0.01
    q_hi = 0.99

    train_values = ctx.train_df[model_cols].to_numpy(dtype=float)
    train_mean = np.mean(train_values, axis=1)
    clamp: dict[str, dict[str, float]] = {}
    center_scale: dict[str, float] = {}
    for idx, col in enumerate(model_cols):
        lo = float(np.quantile(ctx.train_df[col].to_numpy(dtype=float), q_lo))
        hi = float(np.quantile(ctx.train_df[col].to_numpy(dtype=float), q_hi))
        clamp[col] = {"lo": lo, "hi": hi}
        centered_train = train_values[:, idx] - train_mean
        scale = float(np.std(centered_train, ddof=0))
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = 1.0
        center_scale[col] = scale

    values = df[model_cols].to_numpy(dtype=float)
    ens_mean = np.mean(values, axis=1)
    for idx, col in enumerate(model_cols):
        lo = clamp[col]["lo"]
        hi = clamp[col]["hi"]
        vals = values[:, idx]
        clamped = np.clip(vals, lo, hi)
        centered = vals - ens_mean
        tanh_center = np.tanh(centered / center_scale[col])
        _add_feature(features, formulas, f"{col}_clamp_q01_q99", clamped, "clip(q01,q99)")
        _add_feature(features, formulas, f"{col}_tanh_centered", tanh_center, "tanh((x-ens_mean)/scale)")

    train_fitted.append(
        {
            "name": "clamp_transform",
            "clamp_quantiles": [q_lo, q_hi],
            "n_train": int(len(ctx.train_df)),
            "params_by_col": {col: {"clamp": clamp[col], "center_scale": center_scale[col]} for col in model_cols},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex207(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = _exp_required_forecast_cols_6(df)
    ridge_cols = model_cols + ["sin_doy", "cos_doy"]
    _ensure_columns_exist(df, ridge_cols + ["actual_tmax_f"], "EX207")

    X_train = ctx.train_df[ridge_cols].to_numpy(dtype=float)
    y_train = ctx.train_df["actual_tmax_f"].to_numpy(dtype=float)
    ridge = Ridge(alpha=1.0, random_state=ctx.seed)
    ridge.fit(X_train, y_train)

    y0 = ridge.predict(df[ridge_cols].to_numpy(dtype=float))
    _add_feature(features, formulas, "y0_ridge", y0, "ridge(models_6 + sin/cos)")

    values = df[model_cols].to_numpy(dtype=float)
    ens_mean = np.mean(values, axis=1)
    _add_feature(features, formulas, "y0_minus_ensmean", y0 - ens_mean, "y0_ridge - ens_mean_6")

    for idx, col in enumerate(model_cols):
        delta = values[:, idx] - y0
        _add_feature(features, formulas, f"{col}_minus_y0_ridge", delta, f"{col} - y0_ridge")
        _add_feature(features, formulas, f"{col}_minus_y0_ridge_abs", np.abs(delta), f"|{col} - y0_ridge|")

    train_fitted.append(
        {
            "name": "ridge_baseline",
            "alpha": 1.0,
            "feature_cols": list(ridge_cols),
            "intercept": float(ridge.intercept_),
            "coef": {col: float(val) for col, val in zip(ridge_cols, ridge.coef_)},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex208(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = int(ctx.truth_lag)
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = _exp_required_forecast_cols_6(df)
    values = df[model_cols].to_numpy(dtype=float)
    y_base = np.mean(values, axis=1)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = y - y_base

    r_series = pd.Series(resid, index=df.index)
    ewm_7 = tfl.ewm_mean(r_series, halflife=7, min_periods=30, lag=lag, group_key=gk).to_numpy(dtype=float)
    ewm_30 = tfl.ewm_mean(r_series, halflife=30, min_periods=30, lag=lag, group_key=gk).to_numpy(dtype=float)
    drift = ewm_7 - ewm_30

    _add_feature(features, formulas, f"ewm_hl7_l{lag}", ewm_7, "ewm_mean(resid,hl=7)")
    _add_feature(features, formulas, f"ewm_hl30_l{lag}", ewm_30, "ewm_mean(resid,hl=30)")
    _add_feature(features, formulas, f"drift_7m30_l{lag}", drift, "ewm7 - ewm30")
    _add_feature(features, formulas, f"y_base_biascorr_l{lag}", y_base + ewm_7, "y_base + ewm7")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex209(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = _exp_required_forecast_cols_6(df)
    base_cols = _base_feature_columns(df)
    _ensure_columns_exist(df, base_cols + ["actual_tmax_f"], "EX209")

    values = df[model_cols].to_numpy(dtype=float)
    y0 = np.mean(values, axis=1)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    abs_err = np.abs(y - y0)

    train_pos = df.index.get_indexer(ctx.train_df.index)
    train_pos = train_pos[train_pos >= 0]
    tau = float(np.quantile(abs_err[train_pos], 0.70)) if len(train_pos) else 1.0
    y_train = (abs_err[train_pos] > tau).astype(int)
    X_train = df[base_cols].to_numpy(dtype=float)[train_pos]
    X_all = df[base_cols].to_numpy(dtype=float)

    from catboost import CatBoostClassifier

    params = {
        "loss_function": "Logloss",
        "iterations": 800,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 12345,
        "allow_writing_files": False,
        "verbose": False,
    }

    def _build() -> CatBoostClassifier:
        return CatBoostClassifier(**params)

    oof, model_full = _time_oof_classifier(
        ctx.train_df,
        X_train,
        y_train,
        build_model=_build,
        n_splits=5,
        gap_days=2,
    )

    p_all = model_full.predict_proba(X_all)[:, 1]
    oof_mask = np.isfinite(oof)
    if np.any(oof_mask):
        p_all[train_pos[oof_mask]] = oof[oof_mask]
    p_all = np.clip(p_all, 1e-6, 1.0 - 1e-6)
    logit = np.log(p_all / (1.0 - p_all))

    if np.any(oof_mask):
        y_oof = y_train[oof_mask]
        p_oof = oof[oof_mask]
        oof_logloss = float(log_loss(y_oof, p_oof))
        oof_auc = float(roc_auc_score(y_oof, p_oof)) if len(np.unique(y_oof)) >= 2 else None
    else:
        oof_logloss = None
        oof_auc = None

    _add_feature(features, formulas, "p_bust", p_all, "P(|y-ens_mean| > tau)")
    _add_feature(features, formulas, "logit_bust", logit, "logit(p_bust)")

    train_fitted.append(
        {
            "name": "bust_prob_feature",
            "baseline_pred": "ens_mean_6",
            "bust_tau_quantile_train": 0.70,
            "tau": tau,
            "train_bust_rate": float(np.mean(y_train)) if len(y_train) else None,
            "oof_logloss": oof_logloss,
            "oof_auc": oof_auc,
            "model": {"library": "catboost", "params": params},
            "oof": {"method": "time_blocked_cv", "n_splits": 5, "gap_days": 2},
        }
    )

    train_artifacts = ctx.train_df[["station_id", "target_date_local", "asof_utc"]].copy()
    train_artifacts["p_bust_oof"] = oof
    train_artifacts["y_bust"] = y_train

    def _write_artifacts(run_dir: Path) -> list[Path]:
        out_paths: list[Path] = []
        probs_path = run_dir / "oof_bust_probs_train.parquet"
        train_artifacts.to_parquet(probs_path, index=False, engine="pyarrow")
        out_paths.append(probs_path)

        model_path = run_dir / "bust_model.cbm"
        model_full.save_model(str(model_path))
        out_paths.append(model_path)

        meta_path = run_dir / "bust_model_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "tau": tau,
                    "bust_tau_quantile_train": 0.70,
                    "baseline_pred": "ens_mean_6",
                    "oof_logloss": oof_logloss,
                    "oof_auc": oof_auc,
                    "params": params,
                },
                indent=2,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        out_paths.append(meta_path)
        return out_paths

    return DerivedFeatureSet(
        features=features,
        formulas=formulas,
        train_fitted=train_fitted,
        artifact_writers=[_write_artifacts],
    )


def _exp_ex210(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    gk = ctx.group_key
    lag = int(ctx.truth_lag)
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = _exp_required_forecast_cols_6(df)
    sin = df["sin_doy"].to_numpy(dtype=float)
    cos = df["cos_doy"].to_numpy(dtype=float)
    radians = 2 * np.pi * df["day_of_year"].to_numpy(dtype=float) / 365.25
    sin2 = np.sin(2 * radians)
    cos2 = np.cos(2 * radians)
    y = df["actual_tmax_f"].to_numpy(dtype=float)

    bias_season = np.zeros((len(df), len(model_cols)), dtype=float)
    anom_ewm = np.zeros_like(bias_season)
    for i, col in enumerate(model_cols):
        train = ctx.train_df
        radians_t = 2 * np.pi * train["day_of_year"].to_numpy(dtype=float) / 365.25
        X = np.column_stack(
            [
                np.ones(len(train)),
                train["sin_doy"].to_numpy(dtype=float),
                train["cos_doy"].to_numpy(dtype=float),
                np.sin(2 * radians_t),
                np.cos(2 * radians_t),
            ]
        )
        r = train["actual_tmax_f"].to_numpy(dtype=float) - train[col].to_numpy(dtype=float)
        coef, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
        X_all = np.column_stack([np.ones(len(df)), sin, cos, sin2, cos2])
        bias_season[:, i] = X_all @ coef

        r_full = y - df[col].to_numpy(dtype=float) - bias_season[:, i]
        r_l2 = pd.Series(r_full, index=df.index).groupby(gk).shift(lag)
        anom = tfl.ewm_mean(
            r_l2,
            halflife=14,
            min_periods=30,
            lag=0,
            group_key=gk,
        ).to_numpy(dtype=float)
        anom_ewm[:, i] = np.clip(anom, -2.0, 2.0)

    corrected = df[model_cols].to_numpy(dtype=float) + bias_season + anom_ewm
    corr_mean = np.mean(corrected, axis=1)
    corr_median = np.median(corrected, axis=1)

    _add_feature(features, formulas, "corr_mean", corr_mean, "mean(f_i + bias_season + anom_ewm)")
    _add_feature(features, formulas, "corr_median", corr_median, "median(f_i + bias_season + anom_ewm)")
    train_fitted.append(
        {
            "name": "corrected_consensus",
            "seasonal_bias": {"method": "fourier", "order": 2, "fit_split": "train"},
            "anomaly_ewma": {"halflife": 14, "lag_days": lag, "min_history": 30, "clip": [-2.0, 2.0]},
        }
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def _exp_ex211(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    model_cols = _exp_required_forecast_cols_6(df)
    values = df[model_cols].to_numpy(dtype=float)
    n = len(model_cols)
    if n < 2:
        raise ValueError("EX211 requires at least 2 forecast columns.")

    sum_all = np.sum(values, axis=1)
    loo_means: dict[str, np.ndarray] = {}
    influences: dict[str, np.ndarray] = {}
    for i, col in enumerate(model_cols):
        loo = (sum_all - values[:, i]) / float(n - 1)
        loo_means[col] = loo
        influence = values[:, i] - loo
        influences[col] = influence
        _add_feature(features, formulas, f"loo_mean_{col}", loo, f"mean(models != {col})")
        _add_feature(features, formulas, f"influence_{col}", influence, f"{col} - loo_mean_{col}")

    ens_median = np.median(values, axis=1)
    abs_dev = np.abs(values - ens_median[:, None])
    out_id = np.argmax(abs_dev, axis=1)
    outlier_excluded_mean = np.array(
        [loo_means[model_cols[i]][row_idx] for row_idx, i in enumerate(out_id)],
        dtype=float,
    )
    max_abs_influence = np.max(np.abs(np.column_stack([influences[c] for c in model_cols])), axis=1)

    _add_feature(features, formulas, "outlier_excluded_mean", outlier_excluded_mean, "mean(excluding_outlier)")
    _add_feature(features, formulas, "max_abs_influence", max_abs_influence, "max(|m - loo_mean_m|)")
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


def _build_experiments_ex() -> list[ExperimentDefinition]:
    return [
        ExperimentDefinition("EX01", "Kalman bias state for ens_mean", _exp_ex01),
        ExperimentDefinition("EX02", "Spread-adaptive huberized bias EWMA", _exp_ex02),
        ExperimentDefinition("EX03", "Page-Hinkley resettable bias", _exp_ex03),
        ExperimentDefinition("EX04", "Fast-slow bias drift gating", _exp_ex04),
        ExperimentDefinition("EX05", "DOY-smoothed bias prior + delta", _exp_ex05),
        ExperimentDefinition("EX06", "Hierarchical station-shrunk bias state", _exp_ex06),
        ExperimentDefinition("EX07", "Station x month bias table with shrinkage", _exp_ex07),
        ExperimentDefinition("EX08", "Monotone isotonic calibration of ens_mean", _exp_ex08),
        ExperimentDefinition("EX09", "2D bias surface on (ens_mean, ens_std)", _exp_ex09),
        ExperimentDefinition("EX10", "Season-stratified quantile mapping", _exp_ex10),
        ExperimentDefinition("EX11", "Per-model quantile mapping then re-aggregate", _exp_ex11),
        ExperimentDefinition("EX12", "Residual variance state + damping", _exp_ex12),
        ExperimentDefinition("EX13", "Two-regime residual scale states", _exp_ex13),
        ExperimentDefinition("EX14", "Monotone spread to expected |error|", _exp_ex14),
        ExperimentDefinition("EX15", "Asymmetric bias states", _exp_ex15),
        ExperimentDefinition("EX16", "Residual sign probability bias blend", _exp_ex16),
        ExperimentDefinition("EX17", "Rank-permutation regime bias", _exp_ex17),
        ExperimentDefinition("EX18", "Outlier identity conditional bias", _exp_ex18),
        ExperimentDefinition("EX19", "Leave-one-out outlier-excluded consensus", _exp_ex19),
        ExperimentDefinition("EX20", "Deviation-entropy core mean", _exp_ex20),
        ExperimentDefinition("EX21", "Online Bayesian model averaging", _exp_ex21),
        ExperimentDefinition("EX22", "Season-prior Bayesian model averaging", _exp_ex22),
        ExperimentDefinition("EX23", "HMM disagreement regimes", _exp_ex23),
        ExperimentDefinition("EX24", "Residual HMM warm/cold bias", _exp_ex24),
        ExperimentDefinition("EX25", "Supervised regime tree residual mean", _exp_ex25),
        ExperimentDefinition("EX26", "Two-stage residual mean model", _exp_ex26),
        ExperimentDefinition("EX27", "Two-stage residual scale model", _exp_ex27),
        ExperimentDefinition("EX28", "Cross-fit quantile models", _exp_ex28),
        ExperimentDefinition("EX29", "Conformal nonconformity width feature", _exp_ex29),
        ExperimentDefinition("EX30", "Station DOY climatology + anomaly features", _exp_ex30),
        ExperimentDefinition("EX31", "Lagged truth persistence + trend", _exp_ex31),
        ExperimentDefinition("EX32", "Bias by forecast anomaly bin", _exp_ex32),
        ExperimentDefinition("EX33", "Forecast vs recent truth delta bias", _exp_ex33),
        ExperimentDefinition("EX34", "Forecast percentile bias curve", _exp_ex34),
        ExperimentDefinition("EX35", "Station-specific spread reliability", _exp_ex35),
        ExperimentDefinition("EX36", "Residual factor PCA states", _exp_ex36),
        ExperimentDefinition("EX37", "Residual covariance GLS weights", _exp_ex37),
        ExperimentDefinition("EX38", "Median-of-means bias estimator", _exp_ex38),
        ExperimentDefinition("EX39", "Median bias + MAD winsorization", _exp_ex39),
        ExperimentDefinition("EX40", "Tail-only bias prior", _exp_ex40),
        ExperimentDefinition("EX41", "GPD tail-risk score", _exp_ex41),
        ExperimentDefinition("EX42", "Disagreement archetype similarity", _exp_ex42),
        ExperimentDefinition("EX43", "PCA reconstruction error", _exp_ex43),
        ExperimentDefinition("EX44", "Disagreement entropy + concentration", _exp_ex44),
        ExperimentDefinition("EX45", "Pairwise bias-crossover features", _exp_ex45),
        ExperimentDefinition("EX46", "Monotone spread blend weights", _exp_ex46),
        ExperimentDefinition("EX47", "Seasonal constrained ridge blend", _exp_ex47),
        ExperimentDefinition("EX48", "Residual AR(1) predictor", _exp_ex48),
        ExperimentDefinition("EX49", "Seasonal Kalman bias by month", _exp_ex49),
        ExperimentDefinition("EX50", "Joint bias + volatility state", _exp_ex50),
    ]


def _build_experiments_ex2() -> list[ExperimentDefinition]:
    return [
        ExperimentDefinition("EX51", "AR1-Kalman uncertainty blend", _exp_ex51),
        ExperimentDefinition("EX52", "PH-gated fast bias + AR1", _exp_ex52),
        ExperimentDefinition("EX53", "Asymmetry blend with AR1 fallback", _exp_ex53),
        ExperimentDefinition("EX54", "Kalman bias on outlier-excluded mean", _exp_ex54),
        ExperimentDefinition("EX55", "AR1 on entropy core mean", _exp_ex55),
        ExperimentDefinition("EX56", "Pred-abs-error damped slow bias", _exp_ex56),
        ExperimentDefinition("EX57", "Stacked corrected forecasts", _exp_ex57),
        ExperimentDefinition("EX58", "Entropy-gated outlier-excluded blend", _exp_ex58),
        ExperimentDefinition("EX59", "Spread-damped AR1 correction", _exp_ex59),
        ExperimentDefinition("EX60", "Volatility-gated Kalman vs MOM", _exp_ex60),
    ]


def _build_experiments_ex3() -> list[ExperimentDefinition]:
    return [
        ExperimentDefinition("EX61", "EMOS ridge mean + spread-skill sigma", _exp_ex61),
        ExperimentDefinition("EX62", "EMOS median + spread-skill MAD", _exp_ex62),
        ExperimentDefinition("EX63", "Simplex median stacker (static)", _exp_ex63),
        ExperimentDefinition("EX64", "Piecewise EMOS median with smooth gates", _exp_ex64),
        ExperimentDefinition("EX65", "AR-EMOS on standardized errors", _exp_ex65),
        ExperimentDefinition("EX66", "2D residual median map (ens_mean x spread)", _exp_ex66),
        ExperimentDefinition("EX67", "2D residual median map (pc1 x ens_mean)", _exp_ex67),
        ExperimentDefinition("EX68", "3D residual median map (ens x spread x season)", _exp_ex68),
        ExperimentDefinition("EX69", "Additive spline bias model", _exp_ex69),
        ExperimentDefinition("EX70", "Conditional residual quantile map (ens x spread)", _exp_ex70),
        ExperimentDefinition("EX71", "Static BMA with component bias/scale", _exp_ex71),
        ExperimentDefinition("EX72", "Online Student-t likelihood weights", _exp_ex72),
        ExperimentDefinition("EX73", "Hedge exponential weights (MAE)", _exp_ex73),
        ExperimentDefinition("EX74", "Shrunk hedge with spread-dependent eta", _exp_ex74),
        ExperimentDefinition("EX75", "Regime-partitioned hedge weights", _exp_ex75),
        ExperimentDefinition("EX76", "BOCPD change probability on residuals", _exp_ex76),
        ExperimentDefinition("EX77", "BOCPD change probability on PC1", _exp_ex77),
        ExperimentDefinition("EX78", "Change-point gated bias state", _exp_ex78),
        ExperimentDefinition("EX79", "Hotelling T2 drift on residual vector", _exp_ex79),
        ExperimentDefinition("EX80", "Forecast-vector Mahalanobis OOD score", _exp_ex80),
        ExperimentDefinition("EX81", "GMM soft regimes on disagreement", _exp_ex81),
        ExperimentDefinition("EX82", "GMM regime median correction", _exp_ex82),
        ExperimentDefinition("EX83", "HMM sequential regimes on forecasts", _exp_ex83),
        ExperimentDefinition("EX84", "HMM regime-specific model weights", _exp_ex84),
        ExperimentDefinition("EX85", "Forecast ordering state calibration", _exp_ex85),
        ExperimentDefinition("EX86", "Sea-breeze proxy bias map", _exp_ex86),
        ExperimentDefinition("EX87", "Cool-tail depth bias map", _exp_ex87),
        ExperimentDefinition("EX88", "Sea-breeze persistence + spread interaction", _exp_ex88),
        ExperimentDefinition("EX89", "Disagreement gap2 bias map", _exp_ex89),
        ExperimentDefinition("EX90", "High-res vs NBM tilt bias map", _exp_ex90),
        ExperimentDefinition("EX91", "Lagged truth + anomaly features", _exp_ex91),
        ExperimentDefinition("EX92", "Climatology-adjusted persistence", _exp_ex92),
        ExperimentDefinition("EX93", "Spread-gated persistence blend", _exp_ex93),
        ExperimentDefinition("EX94", "Anomaly AR(2-day) forecast", _exp_ex94),
        ExperimentDefinition("EX95", "Theil-Sen trend extrapolation", _exp_ex95),
        ExperimentDefinition("EX96", "Seasonal per-model median bias correction", _exp_ex96),
        ExperimentDefinition("EX97", "Reliability-weighted median of corrected models", _exp_ex97),
        ExperimentDefinition("EX98", "Huber M-estimator consensus", _exp_ex98),
        ExperimentDefinition("EX99", "BMA mixture median", _exp_ex99),
        ExperimentDefinition("EX100", "Residual skew proxy features", _exp_ex100),
        ExperimentDefinition("EX101", "Hierarchical month bias per model", _exp_ex101),
        ExperimentDefinition("EX102", "Fourier seasonal bias + anomaly EWMA", _exp_ex102),
        ExperimentDefinition("EX103", "Bandit-weighted bias halflife", _exp_ex103),
        ExperimentDefinition("EX104", "Month-varying spread->error scale", _exp_ex104),
        ExperimentDefinition("EX105", "Hierarchical EMOS with station offsets", _exp_ex105),
        ExperimentDefinition("EX106", "Large-error probability (cross-fit)", _exp_ex106),
        ExperimentDefinition("EX107", "Large-error hedged forecast", _exp_ex107),
        ExperimentDefinition("EX108", "3-class bust classifier (cross-fit)", _exp_ex108),
        ExperimentDefinition("EX109", "Directional median correction", _exp_ex109),
        ExperimentDefinition("EX110", "Joint location+scale meta-model", _exp_ex110),
        ExperimentDefinition(
            "EX201",
            "RS-MoE: 3-class bust gate + 3 expert regressors, MAE objective, OOF weights + temperature scaling",
            _exp_ex201,
        ),
    ]


def _build_experiments_ex4() -> list[ExperimentDefinition]:
    return [
        ExperimentDefinition(
            "EX202",
            "Robust consensus + spread statistics feature pack",
            _exp_ex202,
        ),
        ExperimentDefinition(
            "EX203",
            "Rank + deviation-from-consensus feature pack",
            _exp_ex203,
        ),
        ExperimentDefinition(
            "EX204",
            "Pairwise diffs + disagreement direction signals",
            _exp_ex204,
        ),
        ExperimentDefinition(
            "EX205",
            "Seasonal interaction features (forecast × sin/cos DOY)",
            _exp_ex205,
        ),
        ExperimentDefinition(
            "EX206",
            "Nonlinear transforms + train-percentile clamping (variance control)",
            _exp_ex206,
        ),
        ExperimentDefinition(
            "EX207",
            "Ridge baseline y0 + residual-predictive feature expansion",
            _exp_ex207,
        ),
        ExperimentDefinition(
            "EX208",
            "EWMA residual drift features (strict lag, train-safe)",
            _exp_ex208,
        ),
        ExperimentDefinition(
            "EX209",
            "Bust classifier probabilities as features (cross-fit, balanced)",
            _exp_ex209,
        ),
        ExperimentDefinition(
            "EX210",
            "Corrected consensus: seasonal bias + anomaly EWMA (leakage-safe lag)",
            _exp_ex210,
        ),
        ExperimentDefinition(
            "EX211",
            "Leave-one-out + outlier-excluded consensus (6-model variant)",
            _exp_ex211,
        ),
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
    ] + _build_experiments_04() + _build_experiments_ex() + _build_experiments_ex2() + _build_experiments_ex3() + _build_experiments_ex4()


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
