"""Train CatBoost on MOS feature CSV and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from weather_ml import metrics


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
    parser = argparse.ArgumentParser(description="Train CatBoost model with fixed date split.")
    parser.add_argument("--csv", required=True, help="Path to MOS feature CSV.")
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument(
        "--val-start",
        help="Optional validation start date (defaults to Jan 1 of train_end year).",
    )
    parser.add_argument(
        "--val-end",
        help="Optional validation end date (defaults to train_end).",
    )
    parser.add_argument("--output", help="Output report JSON path.")
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
        preds = df[col].to_numpy(dtype=float)
        if np.all(np.isnan(preds)):
            continue
        mask = np.isfinite(y) & np.isfinite(preds)
        if not np.any(mask):
            continue
        results[col] = metrics.regression_metrics(y[mask], preds[mask])
    return results


def _default_val_range(train_end: date) -> tuple[date, date]:
    return date(train_end.year, 1, 1), train_end


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date

    train_start = _parse_date(args.train_start)
    train_end = _parse_date(args.train_end)
    test_start = _parse_date(args.test_start)
    test_end = _parse_date(args.test_end)

    train_full = _split(df, train_start, train_end)
    test_df = _split(df, test_start, test_end)

    if train_full.empty:
        raise ValueError("Training split is empty for the requested date range.")
    if test_df.empty:
        raise ValueError("Test split is empty for the requested date range.")

    # Default validation: last calendar year within the training range.
    if args.val_start and args.val_end:
        val_start = _parse_date(args.val_start)
        val_end = _parse_date(args.val_end)
    else:
        val_start, val_end = _default_val_range(train_end)

    val_df = _split(train_full, val_start, val_end)
    train_df = train_full[(train_full["target_date_local"] < val_start) | (train_full["target_date_local"] > val_end)]

    if train_df.empty or val_df.empty:
        # Fall back to training on the full range without early stopping.
        train_df = train_full
        val_df = pd.DataFrame()

    y_train = train_df["y_actual_tmax_f"].to_numpy(dtype=float)
    y_val = val_df["y_actual_tmax_f"].to_numpy(dtype=float) if not val_df.empty else None
    y_train_full = train_full["y_actual_tmax_f"].to_numpy(dtype=float)
    y_test = test_df["y_actual_tmax_f"].to_numpy(dtype=float)

    numeric_cols = train_full.select_dtypes(include=[np.number]).columns.tolist()
    if "y_actual_tmax_f" in numeric_cols:
        numeric_cols.remove("y_actual_tmax_f")

    X_train = train_df[numeric_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    X_train_full = train_full[numeric_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    X_test = test_df[numeric_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    X_val = None
    if not val_df.empty:
        X_val = val_df[numeric_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("catboost is not installed.") from exc

    base_params = {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "iterations": 1200,
        "learning_rate": 0.07,
        "depth": 5,
        "l2_leaf_reg": 8.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "rsm": 0.6,
        "boosting_type": "Plain",
        "border_count": 64,
        "random_seed": int(args.seed),
        "thread_count": int(args.threads),
        "allow_writing_files": False,
        "verbose": False,
    }

    best_iterations = base_params["iterations"]
    if X_val is not None and y_val is not None and len(val_df) > 0:
        model = CatBoostRegressor(**base_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        best_iter = model.get_best_iteration()
        if best_iter is not None and best_iter > 0:
            best_iterations = int(best_iter)
            LOGGER.info("Best iteration from val: %s", best_iterations)

    final_params = dict(base_params)
    final_params["iterations"] = best_iterations
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(X_train_full, y_train_full)

    train_pred = final_model.predict(X_train_full)
    test_pred = final_model.predict(X_test)

    train_metrics = metrics.regression_metrics(y_train_full, train_pred)
    test_metrics = metrics.regression_metrics(y_test, test_pred)
    test_metrics["mae_ge_90"] = _tail_mae(y_test, test_pred, 90.0)
    test_metrics["mae_ge_95"] = _tail_mae(y_test, test_pred, 95.0)

    baseline_test = _baseline_metrics(test_df, DEFAULT_BASELINES)

    importances = final_model.get_feature_importance()
    pairs = [
        {"feature": name, "importance": float(value)}
        for name, value in zip(numeric_cols, np.asarray(importances).reshape(-1))
    ]
    pairs = sorted(pairs, key=lambda item: abs(item["importance"]), reverse=True)

    report_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path),
        "splits": {
            "train": {"start": str(train_start), "end": str(train_end), "rows": int(len(train_full))},
            "test": {"start": str(test_start), "end": str(test_end), "rows": int(len(test_df))},
            "validation_used": {
                "enabled": bool(not val_df.empty),
                "start": str(val_start),
                "end": str(val_end),
                "rows": int(len(val_df)),
            },
        },
        "model": {
            "name": "CatBoostRegressor",
            "params": final_params,
        },
        "feature_count": len(numeric_cols),
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
            "baseline_test": baseline_test,
        },
        "feature_importance": {"type": "catboost", "top_features": pairs[:50]},
    }

    output_path = Path(args.output) if args.output else csv_path.parent / "mos_catboost_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    LOGGER.info("Report written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
