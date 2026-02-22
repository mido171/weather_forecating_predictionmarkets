from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ml_live.features.e450_features import build_e450_features, generate_feature_list
from ml_live.modeling import artifacts as artifact_utils
from ml_live.runtime.paths import artifacts_root, models_dir


logger = logging.getLogger("ml_live.train_e450")


@dataclass(frozen=True)
class TrainingConfig:
    station_id: str = "KMIA"
    train_start: date = date(2021, 2, 23)
    train_end: date = date(2025, 1, 30)
    val_start: date = date(2024, 7, 1)
    val_end: date = date(2025, 1, 30)
    test_start: date = date(2025, 2, 1)
    test_end: date = date(2025, 12, 31)
    truth_lag_days: int = 2
    sigma_cap: float = 10.0
    sigma_floor: float = 0.5
    oof_folds: int = 5
    min_train_rows: int = 200


def load_dataset(path: Path) -> pd.DataFrame:
    logger.info("Loading dataset: %s", path)
    df = pd.read_parquet(path)
    if "target_date_local" not in df.columns:
        raise ValueError("Dataset missing target_date_local column")
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return df


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


def train_mu_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> LGBMRegressor:
    model = LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=128,
        max_depth=-1,
        min_data_in_leaf=60,
        feature_fraction=0.35,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=1.0,
        lambda_l2=2.0,
        verbose=-1,
        random_state=42,
    )
    try:
        import lightgbm as lgb

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
    except Exception as exc:  # pragma: no cover - fallback if callbacks unsupported
        logger.warning("Early stopping unavailable, fitting full model: %s", exc)
        model.fit(X_train, y_train)
    return model


def _time_blocked_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    folds: int,
    min_train_rows: int,
) -> pd.Series:
    dates = np.array(sorted(df["target_date_local"].unique()))
    if len(dates) < folds + 1:
        raise ValueError("Not enough unique dates for time-blocked CV")
    fold_sizes = np.full(folds, len(dates) // folds, dtype=int)
    fold_sizes[: len(dates) % folds] += 1
    indices = np.cumsum(fold_sizes)
    oof_preds = pd.Series(index=df.index, dtype=float)
    start = 0
    for fold_idx, end in enumerate(indices, start=1):
        val_dates = dates[start:end]
        train_dates = dates[:start]
        start = end
        if len(train_dates) == 0:
            continue
        train_df = df[df["target_date_local"].isin(train_dates)]
        val_df = df[df["target_date_local"].isin(val_dates)]
        if len(train_df) < min_train_rows:
            logger.warning("Skipping fold %s: not enough training rows (%s)", fold_idx, len(train_df))
            continue
        mu = train_mu_model(
            train_df[feature_cols].to_numpy(dtype=float),
            train_df[target_col].to_numpy(dtype=float),
            val_df[feature_cols].to_numpy(dtype=float),
            val_df[target_col].to_numpy(dtype=float),
        )
        preds = mu.predict(val_df[feature_cols].to_numpy(dtype=float))
        oof_preds.loc[val_df.index] = preds
    return oof_preds


def train_sigma_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    oof_mu: pd.Series,
    sigma_cap: float,
    sigma_floor: float,
) -> tuple[LGBMRegressor, pd.Series]:
    residuals = df["actual_tmax_f"].to_numpy(dtype=float) - oof_mu.to_numpy(dtype=float)
    sigma_target = np.clip(np.abs(residuals), sigma_floor, sigma_cap)
    model = LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        min_data_in_leaf=80,
        feature_fraction=0.4,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=2.0,
        verbose=-1,
        random_state=52,
    )
    X = df[feature_cols].to_numpy(dtype=float)
    model.fit(X, sigma_target)
    return model, pd.Series(sigma_target, index=df.index)


def run_training(cfg: TrainingConfig, dataset_path: Path | None = None) -> dict:
    e92_dir = artifact_utils.find_e92_run_dir()
    dataset = dataset_path or artifact_utils.resolve_dataset_path(e92_dir)

    raw_df = load_dataset(dataset)
    feature_df, scaling = build_e450_features(
        raw_df, train_start=cfg.train_start, train_end=cfg.train_end, truth_lag_days=cfg.truth_lag_days
    )

    feature_cols = generate_feature_list()
    if any(col not in feature_df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in feature_df.columns]
        raise ValueError(f"Missing feature columns after build: {missing[:10]}")

    data = feature_df.copy()
    data = data.dropna(subset=["actual_tmax_f"]).reset_index(drop=True)

    train_mask = _split_mask(data, cfg.train_start, cfg.train_end)
    val_mask = _split_mask(data, cfg.val_start, cfg.val_end)
    test_mask = _split_mask(data, cfg.test_start, cfg.test_end)

    X_train = data.loc[train_mask, feature_cols].to_numpy(dtype=float)
    y_train = data.loc[train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_val = data.loc[val_mask, feature_cols].to_numpy(dtype=float)
    y_val = data.loc[val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_test = data.loc[test_mask, feature_cols].to_numpy(dtype=float)
    y_test = data.loc[test_mask, "actual_tmax_f"].to_numpy(dtype=float)

    if len(X_train) < cfg.min_train_rows:
        raise ValueError("Not enough training rows for E450")

    logger.info("Training E450 mu model rows=%s features=%s", len(X_train), len(feature_cols))
    mu_model = train_mu_model(X_train, y_train, X_val, y_val)

    preds_train = mu_model.predict(X_train)
    preds_val = mu_model.predict(X_val)
    preds_test = mu_model.predict(X_test)

    metrics = {
        "train": _compute_metrics(y_train, preds_train),
        "validation": _compute_metrics(y_val, preds_val),
        "test": _compute_metrics(y_test, preds_test),
    }

    # Sigma model
    oof_mu = _time_blocked_oof(
        data.loc[train_mask],
        feature_cols,
        "actual_tmax_f",
        cfg.oof_folds,
        cfg.min_train_rows,
    )
    oof_mask = oof_mu.notna()
    oof_df = data.loc[train_mask].loc[oof_mask]
    oof_mu = oof_mu.loc[oof_mask]
    if len(oof_df) < cfg.min_train_rows:
        raise ValueError("Insufficient OOF predictions to train sigma model")
    sigma_model, sigma_target = train_sigma_model(
        oof_df, feature_cols, oof_mu, cfg.sigma_cap, cfg.sigma_floor
    )

    station_dir = models_dir(cfg.station_id)
    station_dir.mkdir(parents=True, exist_ok=True)

    mu_path = station_dir / "e450_mu_model.joblib"
    sigma_path = station_dir / "e450_sigma_model.joblib"
    feature_cols_path = station_dir / "e450_feature_columns.json"
    metadata_path = station_dir / "e450_training_metadata.json"
    sigma_meta_path = station_dir / "e450_sigma_metadata.json"
    scaling_path = station_dir / "e450_analog_scaling.json"

    joblib.dump(mu_model, mu_path)
    joblib.dump(sigma_model, sigma_path)
    feature_cols_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    scaling_path.write_text(
        json.dumps({"means": scaling.means, "stds": scaling.stds}, indent=2), encoding="utf-8"
    )

    metadata = {
        "station_id": cfg.station_id,
        "train_start": cfg.train_start.isoformat(),
        "train_end": cfg.train_end.isoformat(),
        "val_start": cfg.val_start.isoformat(),
        "val_end": cfg.val_end.isoformat(),
        "test_start": cfg.test_start.isoformat(),
        "test_end": cfg.test_end.isoformat(),
        "dataset_path": str(dataset),
        "feature_hash": artifact_utils.feature_list_hash(feature_cols),
        "rows": int(len(data)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sigma_meta = {
        "station_id": cfg.station_id,
        "sigma_target": "abs(residual)",
        "sigma_cap": cfg.sigma_cap,
        "sigma_floor": cfg.sigma_floor,
        "oof_rows": int(len(oof_df)),
        "oof_folds": cfg.oof_folds,
    }
    sigma_meta_path.write_text(json.dumps(sigma_meta, indent=2), encoding="utf-8")

    report_dir = _report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (report_dir / "feature_list.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )
    _write_report(report_dir, metrics, mu_model, feature_cols, data, cfg, scaling)

    return {
        "mu_model_path": mu_path,
        "sigma_model_path": sigma_path,
        "feature_columns_path": feature_cols_path,
        "metadata_path": metadata_path,
        "sigma_metadata_path": sigma_meta_path,
        "scaling_path": scaling_path,
        "report_dir": report_dir,
        "metrics": metrics,
    }


def _report_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return artifacts_root() / "e450_kmia" / ts


def _write_report(
    report_dir: Path,
    metrics: dict,
    mu_model: LGBMRegressor,
    feature_cols: list[str],
    df: pd.DataFrame,
    cfg: TrainingConfig,
    scaling,
) -> None:
    date_min = df["target_date_local"].min()
    date_max = df["target_date_local"].max()
    report_path = report_dir / "report.md"

    importance = list(mu_model.feature_importances_)
    top = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:25]

    content = [
        "# E450 Training Report",
        "",
        "## Dataset Summary",
        "```json",
        json.dumps(
            {
                "date_coverage": {"min": str(date_min), "max": str(date_max)},
                "row_count": int(len(df)),
                "station_counts": df["station_id"].value_counts().to_dict() if "station_id" in df.columns else {},
            },
            indent=2,
        ),
        "```",
        "",
        "## Metrics Summary",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
        "",
        "## Model Summary",
        "```json",
        json.dumps(mu_model.get_params(), indent=2),
        "```",
        "",
        "## Feature Importance (Top 25)",
        "```json",
        json.dumps([{"feature": f, "importance": int(i)} for f, i in top], indent=2),
        "```",
        "",
        "## Analog Scaling",
        "```json",
        json.dumps({"means": scaling.means, "stds": scaling.stds}, indent=2),
        "```",
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainingConfig()
    result = run_training(cfg)
    print(json.dumps({"report_dir": str(result["report_dir"]), "metrics": result["metrics"]}, indent=2))
