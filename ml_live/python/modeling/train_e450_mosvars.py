from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml_live.db.mysql import MysqlConfig, MySqlStore
from ml_live.features.e450_features import build_e450_features, generate_feature_list
from ml_live.modeling import artifacts as artifact_utils
from ml_live.runtime.paths import artifacts_root, models_dir

logger = logging.getLogger("ml_live.train_e450_mosvars")


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Match the E450 sweep split so we can compare apples-to-apples.
    """

    station_id: str = "KMIA"

    # Parameters used by E450 feature generation (must not include any part of the test period).
    feature_params_start: date = date(2021, 2, 23)
    feature_params_end: date = date(2025, 1, 30)

    train_start: date = date(2021, 2, 23)
    train_end: date = date(2024, 6, 30)

    val_start: date = date(2024, 7, 1)
    val_end: date = date(2025, 1, 30)

    test_start: date = date(2025, 2, 1)
    test_end: date = date(2025, 12, 31)

    truth_lag_days: int = 2


MOS_VARIABLE_CODES = [
    # Note: n_x is already included via gfs_n_x_max / nam_n_x_max in the base dataset/features.
    "p06",
    "p12",
    "q06",
    "q12",
    "t06",
    "t06_1",
    "t06_2",
    "tmp",
    "vis",
    "wdr",
    "wsp",
    "cig",
    "dpt",
]

MOS_MODELS = ["gfs", "nam"]
MOS_SUFFIXES = ["min", "max", "mean", "median", "count"]


def _report_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return artifacts_root() / "e450_mosvars_kmia" / ts


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
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    return df


def _mysql_from_env() -> MysqlConfig:
    return MysqlConfig(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DB", "weather_predictionmarkets"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root"),
    )


def _expected_mos_feature_cols() -> list[str]:
    cols: list[str] = []
    for model in MOS_MODELS:
        for var in MOS_VARIABLE_CODES:
            for suffix in MOS_SUFFIXES:
                cols.append(f"mos_{model}_{var}_{suffix}")
    return cols


def _train_xgb_mae_deeper(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> "XGBRegressor":
    from xgboost import XGBRegressor

    # Match the best-performing config from artifacts/e450_kmia_sweep/*/results.json (xgb_mae_deeper).
    params = dict(
        objective="reg:absoluteerror",
        n_estimators=8000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.35,
        min_child_weight=8.0,
        reg_alpha=0.0,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=42,
        eval_metric="mae",
        early_stopping_rounds=200,
    )

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_experiment(cfg: ExperimentConfig) -> dict:
    e92_dir = artifact_utils.find_e92_run_dir()
    dataset_path = artifact_utils.resolve_dataset_path(e92_dir)
    logger.info("Loading base dataset=%s", dataset_path)
    raw_df = _load_dataset(dataset_path)
    raw_df = raw_df[raw_df["station_id"].str.upper() == cfg.station_id.upper()].copy()
    if raw_df.empty:
        raise ValueError(f"No dataset rows for station_id={cfg.station_id}")

    logger.info(
        "Building E450 engineered features feature_params=[%s..%s] truth_lag_days=%s",
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

    base_feature_cols = generate_feature_list()
    data = features_df.dropna(subset=["actual_tmax_f"]).reset_index(drop=True)

    train_mask = _split_mask(data, cfg.train_start, cfg.train_end)
    val_mask = _split_mask(data, cfg.val_start, cfg.val_end)
    test_mask = _split_mask(data, cfg.test_start, cfg.test_end)

    logger.info(
        "Split sizes train=%s val=%s test=%s",
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
    )

    X_train_base = data.loc[train_mask, base_feature_cols].to_numpy(dtype=float)
    y_train = data.loc[train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_val_base = data.loc[val_mask, base_feature_cols].to_numpy(dtype=float)
    y_val = data.loc[val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_test_base = data.loc[test_mask, base_feature_cols].to_numpy(dtype=float)
    y_test = data.loc[test_mask, "actual_tmax_f"].to_numpy(dtype=float)

    logger.info("Training baseline (E450 features only) xgb_mae_deeper features=%s", len(base_feature_cols))
    model_base = _train_xgb_mae_deeper(X_train_base, y_train, X_val_base, y_val)
    preds_base_train = model_base.predict(X_train_base)
    preds_base_val = model_base.predict(X_val_base)
    preds_base_test = model_base.predict(X_test_base)
    metrics_base = {
        "train": _compute_metrics(y_train, preds_base_train),
        "validation": _compute_metrics(y_val, preds_base_val),
        "test": _compute_metrics(y_test, preds_base_test),
    }

    # --- MOS extra features ---
    mysql_cfg = _mysql_from_env()
    store = MySqlStore(mysql_cfg)
    date_min = min(data["target_date_local"])
    date_max = max(data["target_date_local"])

    logger.info(
        "Fetching MOS extra variables station=%s dateRange=%s..%s vars=%s",
        cfg.station_id,
        date_min,
        date_max,
        MOS_VARIABLE_CODES,
    )
    mos_df = store.fetch_mos_variable_history(
        station_id=cfg.station_id,
        start_date=date_min,
        end_date=date_max,
        variable_codes=MOS_VARIABLE_CODES,
    )
    if mos_df.empty:
        raise ValueError(
            f"No MOS rows returned for station={cfg.station_id} dateRange={date_min}..{date_max} "
            f"vars={MOS_VARIABLE_CODES}"
        )
    mos_df["asof_utc"] = pd.to_datetime(mos_df["asof_utc"], utc=True)
    mos_df["target_date_local"] = pd.to_datetime(mos_df["target_date_local"]).dt.date

    data_aug = data.merge(mos_df, on=["station_id", "target_date_local", "asof_utc"], how="left")

    mos_feature_cols = _expected_mos_feature_cols()
    for col in mos_feature_cols:
        if col not in data_aug.columns:
            data_aug[col] = np.nan

    full_feature_cols = base_feature_cols + mos_feature_cols

    X_train_aug = data_aug.loc[train_mask, full_feature_cols].to_numpy(dtype=float)
    X_val_aug = data_aug.loc[val_mask, full_feature_cols].to_numpy(dtype=float)
    X_test_aug = data_aug.loc[test_mask, full_feature_cols].to_numpy(dtype=float)

    logger.info(
        "Training augmented (E450 + MOS vars) xgb_mae_deeper base_features=%s mos_features=%s total=%s",
        len(base_feature_cols),
        len(mos_feature_cols),
        len(full_feature_cols),
    )
    model_aug = _train_xgb_mae_deeper(X_train_aug, y_train, X_val_aug, y_val)
    preds_aug_train = model_aug.predict(X_train_aug)
    preds_aug_val = model_aug.predict(X_val_aug)
    preds_aug_test = model_aug.predict(X_test_aug)
    metrics_aug = {
        "train": _compute_metrics(y_train, preds_aug_train),
        "validation": _compute_metrics(y_val, preds_aug_val),
        "test": _compute_metrics(y_test, preds_aug_test),
    }

    # Save artifacts + report
    report_dir = _report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics_baseline.json").write_text(json.dumps(metrics_base, indent=2), encoding="utf-8")
    (report_dir / "metrics_mosvars.json").write_text(json.dumps(metrics_aug, indent=2), encoding="utf-8")
    (report_dir / "feature_list_baseline.json").write_text(
        json.dumps(base_feature_cols, indent=2), encoding="utf-8"
    )
    (report_dir / "feature_list_mosvars.json").write_text(
        json.dumps(full_feature_cols, indent=2), encoding="utf-8"
    )

    station_dir = models_dir(cfg.station_id)
    station_dir.mkdir(parents=True, exist_ok=True)
    # Save the augmented model (this is the one you'd likely use if it helps).
    model_path = station_dir / "e450_mosvars_mu_xgb.json"
    model_aug.save_model(model_path)

    # Importance quicklook (aug model).
    importance = list(getattr(model_aug, "feature_importances_", []))
    top = []
    if importance and len(importance) == len(full_feature_cols):
        top = sorted(zip(full_feature_cols, importance), key=lambda x: x[1], reverse=True)[:40]

    diff = {}
    for split in ["train", "validation", "test"]:
        if split in metrics_base and split in metrics_aug:
            diff[split] = {
                k: (metrics_aug[split].get(k) if k == "n" else float(metrics_aug[split].get(k, np.nan)) - float(metrics_base[split].get(k, np.nan)))
                for k in ["n", "mae", "rmse", "bias", "medianAE", "maxAE", "corr"]
                if k in metrics_base[split] and k in metrics_aug[split]
            }

    report_lines = [
        "# E450 + MOS Variables Experiment",
        "",
        "This run compares the baseline E450 feature set vs adding MOS daily variable_code-derived features",
        "(GFS/NAM: p06/p12/q06/q12/t06/t06_1/t06_2/tmp/vis/wdr/wsp/cig/dpt).",
        "",
        "## Config",
        "```json",
        json.dumps(
            {
                "station_id": cfg.station_id,
                "dataset_path": str(dataset_path),
                "feature_params": {
                    "start": cfg.feature_params_start.isoformat(),
                    "end": cfg.feature_params_end.isoformat(),
                },
                "split": {
                    "train": [cfg.train_start.isoformat(), cfg.train_end.isoformat()],
                    "val": [cfg.val_start.isoformat(), cfg.val_end.isoformat()],
                    "test": [cfg.test_start.isoformat(), cfg.test_end.isoformat()],
                },
                "truth_lag_days": cfg.truth_lag_days,
                "mos_variable_codes": MOS_VARIABLE_CODES,
                "mos_feature_suffixes": MOS_SUFFIXES,
            },
            indent=2,
        ),
        "```",
        "",
        "## Feature Counts",
        "```json",
        json.dumps(
            {
                "e450_engineered_features": len(base_feature_cols),
                "mos_extra_features": len(mos_feature_cols),
                "total_features_augmented": len(full_feature_cols),
            },
            indent=2,
        ),
        "```",
        "",
        "## Metrics (Baseline: E450 features only)",
        "```json",
        json.dumps(metrics_base, indent=2),
        "```",
        "",
        "## Metrics (Augmented: E450 + MOS vars)",
        "```json",
        json.dumps(metrics_aug, indent=2),
        "```",
        "",
        "## Delta (Augmented - Baseline)",
        "```json",
        json.dumps(diff, indent=2),
        "```",
        "",
        "## Top Feature Importances (Augmented, top 40 by gain/weight proxy)",
        "```json",
        json.dumps(
            [{"feature": f, "importance": float(i)} for f, i in top],
            indent=2,
        ),
        "```",
        "",
        f"Saved augmented model: {model_path}",
    ]
    (report_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "report_dir": report_dir,
        "metrics_baseline": metrics_base,
        "metrics_mosvars": metrics_aug,
        "delta": diff,
        "model_path": model_path,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ExperimentConfig()
    result = run_experiment(cfg)
    print(
        json.dumps(
            {
                "report_dir": str(result["report_dir"]),
                "metrics_baseline": result["metrics_baseline"],
                "metrics_mosvars": result["metrics_mosvars"],
                "delta": result["delta"],
            },
            indent=2,
        )
    )

