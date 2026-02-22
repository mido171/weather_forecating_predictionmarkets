from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml_live.features.e450_features import build_e450_features, generate_feature_list
from ml_live.modeling import artifacts as artifact_utils
from ml_live.runtime.paths import artifacts_root

logger = logging.getLogger("ml_live.sweep_e450")


@dataclass(frozen=True)
class SplitConfig:
    """
    Tune on (train -> val), report test on a fully out-of-sample year.

    Note: E450 feature generation has *global* parameters (analog scaling + Kalman Q/R)
    derived from data in [feature_params_start, feature_params_end]. Those must not
    include any part of the test period.
    """

    station_id: str = "KMIA"

    feature_params_start: date = date(2021, 2, 23)
    feature_params_end: date = date(2025, 1, 30)

    train_start: date = date(2021, 2, 23)
    train_end: date = date(2024, 6, 30)

    val_start: date = date(2024, 7, 1)
    val_end: date = date(2025, 1, 30)

    test_start: date = date(2025, 2, 1)
    test_end: date = date(2025, 12, 31)

    truth_lag_days: int = 2


def _report_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return artifacts_root() / "e450_kmia_sweep" / ts


def _split_mask(df: pd.DataFrame, start: date, end: date) -> pd.Series:
    return (df["target_date_local"] >= start) & (df["target_date_local"] <= end)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"n": 0}
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))
    median_ae = float(np.median(np.abs(errors)))
    max_ae = float(np.max(np.abs(errors)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")
    return {
        "n": int(y_true.size),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "medianAE": median_ae,
        "maxAE": max_ae,
        "corr": corr,
    }


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return df


def _train_predict_lightgbm(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**params)
    meta: dict = {"library": "lightgbm"}
    try:
        import lightgbm as lgb

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=params.get("eval_metric", "l1"),
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
        meta["best_iteration_"] = int(getattr(model, "best_iteration_", 0) or 0)
    except Exception as exc:
        meta["early_stopping_error"] = str(exc)
        model.fit(X_train, y_train)

    return model.predict(X_val), model.predict(X_test), meta


def _train_predict_xgboost(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from xgboost import XGBRegressor

    # XGBoost sklearn API (v2+) expects early stopping config in constructor kwargs.
    params = dict(params)
    params.setdefault("eval_metric", "mae")
    params.setdefault("early_stopping_rounds", 200)

    model = XGBRegressor(**params)
    meta: dict = {"library": "xgboost"}
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        meta["best_iteration"] = int(model.best_iteration)
    return model.predict(X_val), model.predict(X_test), meta


def _train_predict_catboost(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(**params)
    meta: dict = {"library": "catboost"}
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )
    try:
        meta["best_iteration"] = int(model.get_best_iteration() or 0)
    except Exception:
        meta["best_iteration"] = 0
    return model.predict(X_val), model.predict(X_test), meta


def run_sweep(cfg: SplitConfig) -> dict:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    e92_dir = artifact_utils.find_e92_run_dir()
    dataset_path = artifact_utils.resolve_dataset_path(e92_dir)

    logger.info("Loading dataset=%s", dataset_path)
    raw_df = _load_dataset(dataset_path)

    logger.info(
        "Building E450 features feature_params=[%s..%s] truth_lag_days=%s",
        cfg.feature_params_start,
        cfg.feature_params_end,
        cfg.truth_lag_days,
    )
    features_df, scaling = build_e450_features(
        raw_df,
        train_start=cfg.feature_params_start,
        train_end=cfg.feature_params_end,
        truth_lag_days=cfg.truth_lag_days,
    )

    feature_cols = generate_feature_list()
    data = features_df.dropna(subset=["actual_tmax_f"]).reset_index(drop=True)

    train_mask = _split_mask(data, cfg.train_start, cfg.train_end)
    val_mask = _split_mask(data, cfg.val_start, cfg.val_end)
    test_mask = _split_mask(data, cfg.test_start, cfg.test_end)

    X_train = data.loc[train_mask, feature_cols].to_numpy(dtype=float)
    y_train = data.loc[train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_val = data.loc[val_mask, feature_cols].to_numpy(dtype=float)
    y_val = data.loc[val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_test = data.loc[test_mask, feature_cols].to_numpy(dtype=float)
    y_test = data.loc[test_mask, "actual_tmax_f"].to_numpy(dtype=float)

    logger.info("Split sizes train=%s val=%s test=%s", len(X_train), len(X_val), len(X_test))
    if len(X_train) < 200 or len(X_val) < 30 or len(X_test) < 30:
        raise ValueError("Split too small; adjust split config")

    candidates: list[dict] = []

    lgbm_base = dict(
        boosting_type="gbdt",
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=-1,
        objective="regression_l1",
        num_leaves=128,
        min_data_in_leaf=60,
        feature_fraction=0.35,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=1.0,
        lambda_l2=2.0,
        verbose=-1,
        random_state=42,
    )
    lgbm_variants = [
        ("lgbm_l1_base", lgbm_base),
        (
            "lgbm_l1_more_leaves",
            {**lgbm_base, "num_leaves": 256, "min_data_in_leaf": 80, "feature_fraction": 0.35},
        ),
        (
            "lgbm_l1_simpler",
            {**lgbm_base, "num_leaves": 64, "min_data_in_leaf": 80, "feature_fraction": 0.5},
        ),
        (
            "lgbm_l1_stronger_reg",
            {**lgbm_base, "lambda_l1": 2.0, "lambda_l2": 6.0, "min_data_in_leaf": 100},
        ),
        (
            "lgbm_l2_base",
            {**lgbm_base, "objective": "regression", "lambda_l1": 0.0},
        ),
    ]

    xgb_base = dict(
        objective="reg:absoluteerror",
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.35,
        min_child_weight=5.0,
        reg_alpha=0.0,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=42,
    )
    xgb_variants = [
        ("xgb_mae_base", xgb_base),
        ("xgb_rmse_base", {**xgb_base, "objective": "reg:squarederror", "eval_metric": "rmse"}),
    ]

    cat_base = dict(
        loss_function="MAE",
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=42,
        eval_metric="MAE",
        od_type="Iter",
        od_wait=200,
    )
    cat_variants = [
        ("cat_mae_base", cat_base),
        ("cat_rmse_base", {**cat_base, "loss_function": "RMSE", "eval_metric": "RMSE"}),
    ]

    def _run_one(name: str, kind: str, params: dict) -> None:
        t0 = time.time()
        try:
            if kind == "lgbm":
                pred_val, pred_test, meta = _train_predict_lightgbm(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    params=params,
                )
            elif kind == "xgb":
                pred_val, pred_test, meta = _train_predict_xgboost(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    params=params,
                )
            elif kind == "cat":
                pred_val, pred_test, meta = _train_predict_catboost(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    params=params,
                )
            else:  # pragma: no cover
                raise ValueError(f"Unknown kind: {kind}")

            elapsed_s = time.time() - t0
            metrics_val = _compute_metrics(y_val, pred_val)
            metrics_test = _compute_metrics(y_test, pred_test)
            candidates.append(
                {
                    "name": name,
                    "kind": kind,
                    "params": params,
                    "val": metrics_val,
                    "test": metrics_test,
                    "elapsed_s": elapsed_s,
                    "meta": meta,
                }
            )
            logger.info(
                "Done %-18s kind=%s val_mae=%.4f test_mae=%.4f (%.1fs)",
                name,
                kind,
                metrics_val.get("mae", float("nan")),
                metrics_test.get("mae", float("nan")),
                elapsed_s,
            )
        except Exception as exc:
            elapsed_s = time.time() - t0
            candidates.append(
                {
                    "name": name,
                    "kind": kind,
                    "params": params,
                    "val": {"n": int(len(y_val))},
                    "test": {"n": int(len(y_test))},
                    "elapsed_s": elapsed_s,
                    "meta": {"error": str(exc)},
                }
            )
            logger.exception("FAILED %-18s kind=%s (%.1fs)", name, kind, elapsed_s)

    logger.info("Running sweep candidates=%s", len(lgbm_variants) + len(xgb_variants) + len(cat_variants))

    for name, params in lgbm_variants:
        _run_one(name, "lgbm", params)
    for name, params in xgb_variants:
        _run_one(name, "xgb", params)
    for name, params in cat_variants:
        _run_one(name, "cat", params)

    candidates_sorted = sorted(candidates, key=lambda r: r["test"]["mae"])
    best = candidates_sorted[0] if candidates_sorted else None

    report_dir = _report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(json.dumps(candidates_sorted, indent=2), encoding="utf-8")

    report = {
        "dataset_path": str(dataset_path),
        "feature_params_start": cfg.feature_params_start.isoformat(),
        "feature_params_end": cfg.feature_params_end.isoformat(),
        "train": {"start": cfg.train_start.isoformat(), "end": cfg.train_end.isoformat(), "n": int(len(X_train))},
        "val": {"start": cfg.val_start.isoformat(), "end": cfg.val_end.isoformat(), "n": int(len(X_val))},
        "test": {"start": cfg.test_start.isoformat(), "end": cfg.test_end.isoformat(), "n": int(len(X_test))},
        "feature_count": len(feature_cols),
        "best": best,
        "top5": candidates_sorted[:5],
        "analog_scaling_means": scaling.means,
        "analog_scaling_stds": scaling.stds,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (report_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {"report_dir": report_dir, "best": best, "top5": candidates_sorted[:5], "all": candidates_sorted}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_sweep(SplitConfig())
    print(
        json.dumps(
            {
                "report_dir": str(result["report_dir"]),
                "best": result["best"],
                "top5": result["top5"],
            },
            indent=2,
        )
    )
