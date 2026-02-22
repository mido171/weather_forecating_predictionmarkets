"""Train CatBoost on a deliberately small, KNN-centered feature set."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from weather_ml import metrics
from weather_ml.mos_minimal_features import build_minimal_feature_frame


LOGGER = logging.getLogger(__name__)


DEFAULT_BASELINES = [
    "base_tmax_gfs",
    "base_tmax_nam",
    "base_tmax_blend",
    "obs_persist_tmax",
    "knn_v0_analog_mu",
    "knn_views_analog_mu_weighted",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CatBoost on minimal KNN feature set.")
    parser.add_argument("--csv", required=True, help="Path to MOS feature CSV (full).")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k neighbor slots to summarize.")
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--output", required=True, help="Output report JSON path.")
    parser.add_argument("--features-out", help="Optional path to write minimal feature CSV used for training.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def _parse_date(value: str) -> date:
    return pd.to_datetime(value).date()


def _split(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (df["target_date_local"] >= start) & (df["target_date_local"] <= end)
    return df[mask].copy()


def _tail_mae(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float | None:
    mask = y_true >= threshold
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def _baseline_metrics(df: pd.DataFrame, baselines: list[str]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    y = df["y_actual_tmax_f"].to_numpy(dtype=float)
    for col in baselines:
        if col not in df.columns:
            continue
        preds = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(preds)):
            continue
        mask = np.isfinite(y) & np.isfinite(preds)
        if not np.any(mask):
            continue
        results[col] = metrics.regression_metrics(y[mask], preds[mask])
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date

    features = build_minimal_feature_frame(df, top_k=int(args.top_k))
    feature_cols = list(features.columns)

    work = pd.concat(
        [
            df[["target_date_local", "y_actual_tmax_f"]].copy(),
            features,
        ],
        axis=1,
    )

    # Optional: write the actual feature matrix used.
    if args.features_out:
        out_path = Path(args.features_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        work.to_csv(out_path, index=False, na_rep="")

    train_start = _parse_date(args.train_start)
    train_end = _parse_date(args.train_end)
    test_start = _parse_date(args.test_start)
    test_end = _parse_date(args.test_end)

    train_df = _split(work, train_start, train_end)
    test_df = _split(work, test_start, test_end)

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if test_df.empty:
        raise ValueError("Test split is empty.")

    y_train = pd.to_numeric(train_df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test_df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float)

    # Drop rows with missing targets (features can be NaN; CatBoost handles that).
    train_mask = np.isfinite(y_train)
    test_mask = np.isfinite(y_test)
    train_df = train_df.loc[train_mask].copy()
    test_df = test_df.loc[test_mask].copy()
    y_train = y_train[train_mask]
    y_test = y_test[test_mask]

    # Simple time-based validation: last calendar year within train range.
    val_start = date(train_end.year, 1, 1)
    val_end = train_end
    val_df = train_df[(train_df["target_date_local"] >= val_start) & (train_df["target_date_local"] <= val_end)]
    fit_df = train_df[(train_df["target_date_local"] < val_start) | (train_df["target_date_local"] > val_end)]

    X_fit = fit_df[feature_cols].to_numpy(dtype=float)
    y_fit = pd.to_numeric(fit_df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float) if not val_df.empty else None
    y_val = pd.to_numeric(val_df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float) if not val_df.empty else None

    X_train_full = train_df[feature_cols].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("catboost is not installed.") from exc

    base_params = {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "iterations": 4000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 10.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "rsm": 0.9,
        "random_seed": int(args.seed),
        "thread_count": int(args.threads),
        "allow_writing_files": False,
        "verbose": False,
    }

    best_iterations = base_params["iterations"]
    if X_val is not None and y_val is not None and len(val_df) > 0 and len(fit_df) > 0:
        model = CatBoostRegressor(**base_params)
        model.fit(X_fit, y_fit, eval_set=(X_val, y_val), use_best_model=True)
        best_iter = model.get_best_iteration()
        if best_iter is not None and best_iter > 0:
            best_iterations = int(best_iter)
            LOGGER.info("Best iteration from val: %s", best_iterations)

    final_params = dict(base_params)
    final_params["iterations"] = best_iterations
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(X_train_full, y_train)

    train_pred = final_model.predict(X_train_full)
    test_pred = final_model.predict(X_test)

    train_metrics = metrics.regression_metrics(y_train, train_pred)
    test_metrics = metrics.regression_metrics(y_test, test_pred)
    test_metrics["mae_ge_90"] = _tail_mae(y_test, test_pred, 90.0)
    test_metrics["mae_ge_95"] = _tail_mae(y_test, test_pred, 95.0)

    baseline_test = _baseline_metrics(
        # baseline metrics computed on the original df slice for comparability
        _split(df, test_start, test_end),
        DEFAULT_BASELINES,
    )

    # Also include the raw KNN-topK neighbor-mean baseline (direct neighbor y mean).
    baseline_test["knn30_neighbor_tmax_mean"] = metrics.regression_metrics(
        y_test,
        pd.to_numeric(test_df["knn30_neighbor_tmax_mean"], errors="coerce").to_numpy(dtype=float),
    )

    importances = final_model.get_feature_importance()
    pairs = [
        {"feature": name, "importance": float(value)}
        for name, value in zip(feature_cols, np.asarray(importances).reshape(-1))
    ]
    pairs = sorted(pairs, key=lambda item: abs(item["importance"]), reverse=True)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path),
        "feature_spec": {
            "top_k": int(args.top_k),
            "feature_count": int(len(feature_cols)),
            "feature_columns": feature_cols,
        },
        "splits": {
            "train": {"start": str(train_start), "end": str(train_end), "rows": int(len(train_df))},
            "test": {"start": str(test_start), "end": str(test_end), "rows": int(len(test_df))},
            "validation_used": {
                "enabled": bool(X_val is not None and len(val_df) > 0 and len(fit_df) > 0),
                "start": str(val_start),
                "end": str(val_end),
                "rows": int(len(val_df)),
            },
        },
        "model": {"name": "CatBoostRegressor", "params": final_params},
        "metrics": {"train": train_metrics, "test": test_metrics, "baseline_test": baseline_test},
        "feature_importance": {"type": "catboost", "top_features": pairs[:50]},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Report written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

