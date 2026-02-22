from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ml_live.features.e902_features import build_e902_features, generate_feature_list
from ml_live.modeling import artifacts as artifact_utils
from ml_live.runtime.paths import artifacts_root, models_dir

logger = logging.getLogger("ml_live.train_e902")


@dataclass(frozen=True)
class TrainingConfig:
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
    bias_window: int = 90
    clip_base: float = 1.0
    clip_spread_scale: float = 0.5
    decay_tau_days: float = 365.0
    blend_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
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


def _time_decay_weights(dates: np.ndarray, end_date: date, tau: float) -> np.ndarray:
    delta = np.array([(end_date - d).days for d in dates], dtype=float)
    return np.exp(-delta / tau)


def _train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    sample_weight: np.ndarray | None = None,
):
    model = XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        sample_weight=sample_weight,
    )
    return model


def _apply_bias_and_clip(
    *,
    history_dates: np.ndarray,
    history_raw_pred: np.ndarray,
    history_actual: np.ndarray,
    eval_dates: np.ndarray,
    eval_raw_pred: np.ndarray,
    eval_actual: np.ndarray,
    ens_min: np.ndarray,
    ens_max: np.ndarray,
    spread: np.ndarray,
    bias_window: int,
    clip_base: float,
    clip_spread_scale: float,
) -> np.ndarray:
    errors: list[tuple[date, float]] = []
    for d, pred, act in zip(history_dates, history_raw_pred, history_actual):
        if np.isfinite(pred) and np.isfinite(act):
            errors.append((d, float(pred - act)))

    corrected = np.full_like(eval_raw_pred, np.nan, dtype=float)
    for i, (d, pred, act) in enumerate(zip(eval_dates, eval_raw_pred, eval_actual)):
        if not np.isfinite(pred):
            corrected[i] = np.nan
            continue
        cutoff = d - timedelta(days=2)
        eligible = [err for (dt, err) in errors if dt <= cutoff]
        if eligible:
            bias = float(np.mean(eligible[-bias_window:]))
        else:
            bias = 0.0
        pred_corr = pred - bias
        lo = ens_min[i] - (clip_base + clip_spread_scale * spread[i])
        hi = ens_max[i] + (clip_base + clip_spread_scale * spread[i])
        pred_corr = float(np.clip(pred_corr, lo, hi))
        corrected[i] = pred_corr
        if np.isfinite(act):
            errors.append((d, float(pred - act)))
    return corrected


def _report_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return artifacts_root() / "e902_kmia" / ts


def run_training(cfg: TrainingConfig) -> dict:
    e92_dir = artifact_utils.find_e92_run_dir()
    dataset_path = artifact_utils.resolve_dataset_path(e92_dir)
    raw_df = load_dataset(dataset_path)

    feature_df, metadata = build_e902_features(
        raw_df,
        train_start=cfg.feature_params_start,
        train_end=cfg.feature_params_end,
        truth_lag_days=cfg.truth_lag_days,
    )

    feature_cols = generate_feature_list()
    if any(col not in feature_df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in feature_df.columns]
        raise ValueError(f"Missing feature columns after build: {missing[:10]}")

    data = feature_df.dropna(subset=["actual_tmax_f"]).reset_index(drop=True)
    train_mask = _split_mask(data, cfg.train_start, cfg.train_end)
    val_mask = _split_mask(data, cfg.val_start, cfg.val_end)
    test_mask = _split_mask(data, cfg.test_start, cfg.test_end)

    X_train = data.loc[train_mask, feature_cols].to_numpy(dtype=float)
    y_train = data.loc[train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_val = data.loc[val_mask, feature_cols].to_numpy(dtype=float)
    y_val = data.loc[val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_test = data.loc[test_mask, feature_cols].to_numpy(dtype=float)
    y_test = data.loc[test_mask, "actual_tmax_f"].to_numpy(dtype=float)

    train_dates = data.loc[train_mask, "target_date_local"].to_numpy()
    val_dates = data.loc[val_mask, "target_date_local"].to_numpy()
    test_dates = data.loc[test_mask, "target_date_local"].to_numpy()

    ens_min_val = data.loc[val_mask, "ens_raw_min"].to_numpy(dtype=float)
    ens_max_val = data.loc[val_mask, "ens_raw_max"].to_numpy(dtype=float)
    spread_val = data.loc[val_mask, "gefsatmos_tmp_spread_f"].to_numpy(dtype=float)
    ens_min_test = data.loc[test_mask, "ens_raw_min"].to_numpy(dtype=float)
    ens_max_test = data.loc[test_mask, "ens_raw_max"].to_numpy(dtype=float)
    spread_test = data.loc[test_mask, "gefsatmos_tmp_spread_f"].to_numpy(dtype=float)

    params_mae = dict(
        objective="reg:absoluteerror",
        tree_method="hist",
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=12,
        subsample=0.85,
        colsample_bytree=0.25,
        colsample_bynode=0.25,
        reg_lambda=6.0,
        reg_alpha=0.5,
        max_delta_step=0.0,
        n_estimators=8000,
        eval_metric="mae",
        early_stopping_rounds=200,
        random_state=42,
    )

    params_huber = dict(
        objective="reg:pseudohubererror",
        tree_method="hist",
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=14,
        subsample=0.8,
        colsample_bytree=0.25,
        colsample_bynode=0.25,
        reg_lambda=8.0,
        reg_alpha=0.7,
        n_estimators=8000,
        eval_metric="mae",
        early_stopping_rounds=200,
        random_state=52,
    )

    # Time-decay weights for training
    weights_train = _time_decay_weights(train_dates, cfg.train_end, cfg.decay_tau_days)

    # CV folds
    folds = [
        (date(2021, 2, 23), date(2023, 12, 31), date(2024, 1, 1), date(2024, 6, 30)),
        (date(2021, 2, 23), date(2024, 6, 30), date(2024, 7, 1), date(2024, 12, 31)),
        (date(2021, 2, 23), date(2024, 12, 31), date(2025, 1, 1), date(2025, 1, 30)),
    ]

    cv_results = {w: [] for w in cfg.blend_grid}
    for fold_idx, (tr_start, tr_end, va_start, va_end) in enumerate(folds, start=1):
        tr_mask = _split_mask(data, tr_start, tr_end)
        va_mask = _split_mask(data, va_start, va_end)
        if tr_mask.sum() < 200 or va_mask.sum() < 30:
            continue
        X_tr = data.loc[tr_mask, feature_cols].to_numpy(dtype=float)
        y_tr = data.loc[tr_mask, "actual_tmax_f"].to_numpy(dtype=float)
        X_va = data.loc[va_mask, feature_cols].to_numpy(dtype=float)
        y_va = data.loc[va_mask, "actual_tmax_f"].to_numpy(dtype=float)
        tr_dates = data.loc[tr_mask, "target_date_local"].to_numpy()
        va_dates = data.loc[va_mask, "target_date_local"].to_numpy()
        ens_min_va = data.loc[va_mask, "ens_raw_min"].to_numpy(dtype=float)
        ens_max_va = data.loc[va_mask, "ens_raw_max"].to_numpy(dtype=float)
        spread_va = data.loc[va_mask, "gefsatmos_tmp_spread_f"].to_numpy(dtype=float)

        w_tr = _time_decay_weights(tr_dates, tr_end, cfg.decay_tau_days)

        model_mae = _train_xgb(X_tr, y_tr, X_va, y_va, params_mae, sample_weight=w_tr)
        model_huber = _train_xgb(X_tr, y_tr, X_va, y_va, params_huber, sample_weight=w_tr)

        pred_tr_mae = model_mae.predict(X_tr)
        pred_tr_huber = model_huber.predict(X_tr)
        pred_va_mae = model_mae.predict(X_va)
        pred_va_huber = model_huber.predict(X_va)

        for w in cfg.blend_grid:
            pred_tr = w * pred_tr_mae + (1 - w) * pred_tr_huber
            pred_va = w * pred_va_mae + (1 - w) * pred_va_huber
            pred_va_corr = _apply_bias_and_clip(
                history_dates=tr_dates,
                history_raw_pred=pred_tr,
                history_actual=y_tr,
                eval_dates=va_dates,
                eval_raw_pred=pred_va,
                eval_actual=y_va,
                ens_min=ens_min_va,
                ens_max=ens_max_va,
                spread=spread_va,
                bias_window=cfg.bias_window,
                clip_base=cfg.clip_base,
                clip_spread_scale=cfg.clip_spread_scale,
            )
            metrics = _compute_metrics(y_va, pred_va_corr)
            cv_results[w].append(metrics)

    # Select best blend weight
    blend_scores = {}
    for w, metrics_list in cv_results.items():
        if not metrics_list:
            continue
        mae_avg = float(np.mean([m["mae"] for m in metrics_list]))
        maxae_avg = float(np.mean([m["maxAE"] for m in metrics_list]))
        blend_scores[w] = (mae_avg, maxae_avg)
    best_w = min(blend_scores.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]

    # Train final models on train+val
    full_mask = _split_mask(data, cfg.train_start, cfg.val_end)
    X_full = data.loc[full_mask, feature_cols].to_numpy(dtype=float)
    y_full = data.loc[full_mask, "actual_tmax_f"].to_numpy(dtype=float)
    full_dates = data.loc[full_mask, "target_date_local"].to_numpy()
    weights_full = _time_decay_weights(full_dates, cfg.val_end, cfg.decay_tau_days)

    model_mae = _train_xgb(X_full, y_full, X_val, y_val, params_mae, sample_weight=weights_full)
    model_huber = _train_xgb(X_full, y_full, X_val, y_val, params_huber, sample_weight=weights_full)

    pred_full_mae = model_mae.predict(X_full)
    pred_full_huber = model_huber.predict(X_full)
    pred_test_mae = model_mae.predict(X_test)
    pred_test_huber = model_huber.predict(X_test)

    pred_full = best_w * pred_full_mae + (1 - best_w) * pred_full_huber
    pred_test = best_w * pred_test_mae + (1 - best_w) * pred_test_huber

    pred_test_corr = _apply_bias_and_clip(
        history_dates=full_dates,
        history_raw_pred=pred_full,
        history_actual=y_full,
        eval_dates=test_dates,
        eval_raw_pred=pred_test,
        eval_actual=y_test,
        ens_min=ens_min_test,
        ens_max=ens_max_test,
        spread=spread_test,
        bias_window=cfg.bias_window,
        clip_base=cfg.clip_base,
        clip_spread_scale=cfg.clip_spread_scale,
    )

    metrics_test = _compute_metrics(y_test, pred_test_corr)

    # Metrics by month and regimes
    test_df = data.loc[test_mask].copy()
    test_df["pred"] = pred_test_corr
    test_df["abs_err"] = np.abs(test_df["pred"] - test_df["actual_tmax_f"])
    by_month = (
        test_df.groupby(test_df["target_date_local"].apply(lambda d: d.month))["abs_err"]
        .mean()
        .to_dict()
    )
    by_spread = {
        "low": float(test_df.loc[test_df["spread_bin_low"] == 1, "abs_err"].mean()),
        "mid": float(test_df.loc[test_df["spread_bin_mid"] == 1, "abs_err"].mean()),
        "high": float(test_df.loc[test_df["spread_bin_high"] == 1, "abs_err"].mean()),
    }
    by_disagree = {
        "low": float(test_df.loc[test_df["disagree_bin_low"] == 1, "abs_err"].mean()),
        "mid": float(test_df.loc[test_df["disagree_bin_mid"] == 1, "abs_err"].mean()),
        "high": float(test_df.loc[test_df["disagree_bin_high"] == 1, "abs_err"].mean()),
    }

    station_dir = models_dir(cfg.station_id)
    station_dir.mkdir(parents=True, exist_ok=True)
    mae_path = station_dir / "e902_mu_xgb_mae.json"
    huber_path = station_dir / "e902_mu_xgb_huber.json"
    feature_path = station_dir / "e902_feature_columns.json"
    metadata_path = station_dir / "e902_metadata.json"

    model_mae.save_model(str(mae_path))
    model_huber.save_model(str(huber_path))
    feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "station_id": cfg.station_id,
                "feature_params_start": cfg.feature_params_start.isoformat(),
                "feature_params_end": cfg.feature_params_end.isoformat(),
                "train_start": cfg.train_start.isoformat(),
                "train_end": cfg.train_end.isoformat(),
                "val_start": cfg.val_start.isoformat(),
                "val_end": cfg.val_end.isoformat(),
                "test_start": cfg.test_start.isoformat(),
                "test_end": cfg.test_end.isoformat(),
                "thresholds": metadata.thresholds.__dict__,
                "analog_scaling": {
                    "means": metadata.analog_scaling.means,
                    "stds": metadata.analog_scaling.stds,
                },
                "blend_weight": best_w,
                "bias_window": cfg.bias_window,
                "clip_base": cfg.clip_base,
                "clip_spread_scale": cfg.clip_spread_scale,
                "decay_tau_days": cfg.decay_tau_days,
                "params_mae": params_mae,
                "params_huber": params_huber,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_dir = _report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(metrics_test, indent=2), encoding="utf-8")
    (report_dir / "cv_summary.json").write_text(json.dumps(blend_scores, indent=2), encoding="utf-8")
    report_lines = [
        "# E902 Training Report",
        "",
        "## Test Metrics",
        "```json",
        json.dumps(metrics_test, indent=2),
        "```",
        "",
        "## Blend Weight",
        f"Selected w = {best_w}",
        "",
        "## MAE by Month",
        "```json",
        json.dumps(by_month, indent=2),
        "```",
        "",
        "## MAE by Spread Bin",
        "```json",
        json.dumps(by_spread, indent=2),
        "```",
        "",
        "## MAE by Disagreement Bin",
        "```json",
        json.dumps(by_disagree, indent=2),
        "```",
    ]
    (report_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "report_dir": report_dir,
        "metrics_test": metrics_test,
        "blend_weight": best_w,
        "by_month": by_month,
        "by_spread": by_spread,
        "by_disagree": by_disagree,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_training(TrainingConfig())
    print(json.dumps(result, indent=2))
