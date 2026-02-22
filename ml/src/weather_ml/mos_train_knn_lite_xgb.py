"""Train an XGBoost model on a lightweight KNN-neighbor-focused feature set."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from weather_ml import metrics
from weather_ml.mos_knn_lite import KnnLiteSpec, compute_knn_lite_features_from_v0_slots


LOGGER = logging.getLogger(__name__)


DEFAULT_BASELINES = [
    "base_tmax_gfs",
    "base_tmax_nam",
    "base_tmax_blend",
    "obs_persist_tmax",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train KNN-lite MOS model with XGBoost.")
    parser.add_argument("--csv", required=True, help="Path to MOS feature CSV (must include knn_v0_nn* slots).")
    parser.add_argument("--k", type=int, default=30, help="Top-k nearest neighbors to use.")
    parser.add_argument(
        "--label-lag-days",
        type=int,
        default=1,
        help="Exclude neighbor slots with age_days <= this value (truth not known yet).",
    )
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument(
        "--target-mode",
        choices=["direct", "residual_vs_knn_analog"],
        default="residual_vs_knn_analog",
        help="Whether to predict y directly or learn a residual on top of KNN analog mean.",
    )
    parser.add_argument(
        "--slot-mode",
        choices=["all", "top2", "none"],
        default="top2",
        help="Which per-neighbor slot features to include in XGB features.",
    )
    parser.add_argument("--output", help="Output report JSON path.")
    parser.add_argument("--seed", type=int, default=1337)
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date

    # Build the lightweight KNN feature set from the existing v0 neighbor slots.
    spec = KnnLiteSpec(
        k=int(args.k),
        label_lag_days=int(args.label_lag_days),
        prefix=f"knn{int(args.k)}_",
    )
    knn_lite = compute_knn_lite_features_from_v0_slots(df, spec=spec)
    df = pd.concat([df.reset_index(drop=True), knn_lite.reset_index(drop=True)], axis=1)

    # Core non-KNN features (kept intentionally small).
    core_cols = [
        "base_tmax_gfs",
        "base_tmax_nam",
        "base_tmax_blend",
        "base_tmax_abs_spread",
        "cal_d_doy_sin",
        "cal_d_doy_cos",
        "obs_tmax_last",
        "obs_tmax_roll_mean_7",
        "obs_tmax_roll_mean_30",
        "obs_tmax_z_30",
    ]
    core_cols = [c for c in core_cols if c in df.columns]

    knn_cols = [c for c in df.columns if c.startswith(spec.prefix)]
    slot_pattern = re.compile(rf"^{re.escape(spec.prefix)}nn\d+_")
    knn_agg_cols = [c for c in knn_cols if not slot_pattern.match(c)]

    if args.slot_mode == "all":
        knn_used_cols = knn_cols
    elif args.slot_mode == "none":
        knn_used_cols = knn_agg_cols
    else:  # top2
        keep_slots = []
        for i in (1, 2):
            for suffix in ["dist", "age_days", "w_norm", "residual", "y", "analog"]:
                col = f"{spec.prefix}nn{i}_{suffix}"
                if col in df.columns:
                    keep_slots.append(col)
        knn_used_cols = knn_agg_cols + keep_slots

    feature_cols = core_cols + knn_used_cols
    feature_cols = list(dict.fromkeys(feature_cols))  # stable de-dupe

    train_start = _parse_date(args.train_start)
    train_end = _parse_date(args.train_end)
    test_start = _parse_date(args.test_start)
    test_end = _parse_date(args.test_end)

    train_df = _split(df, train_start, train_end)
    test_df = _split(df, test_start, test_end)

    if train_df.empty:
        raise ValueError("Training split is empty for the requested date range.")
    if test_df.empty:
        raise ValueError("Test split is empty for the requested date range.")

    y_train_true = train_df["y_actual_tmax_f"].to_numpy(dtype=float)
    y_test_true = test_df["y_actual_tmax_f"].to_numpy(dtype=float)

    # Drop any rows where target is missing.
    train_mask = np.isfinite(y_train_true)
    test_mask = np.isfinite(y_test_true)
    train_df = train_df.loc[train_mask].copy()
    test_df = test_df.loc[test_mask].copy()
    y_train_true = y_train_true[train_mask]
    y_test_true = y_test_true[test_mask]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    y_train_model = y_train_true.copy()

    baseline_col = f"{spec.prefix}analog_mu"
    baseline_train = train_df[baseline_col].to_numpy(dtype=float) if baseline_col in train_df.columns else None
    baseline_test_vals = test_df[baseline_col].to_numpy(dtype=float) if baseline_col in test_df.columns else None
    if args.target_mode == "residual_vs_knn_analog":
        if baseline_train is None or baseline_test_vals is None:
            raise ValueError(f"target_mode={args.target_mode} requires baseline column {baseline_col}")
        train_ok = np.isfinite(baseline_train)
        test_ok = np.isfinite(baseline_test_vals)
        train_df = train_df.loc[train_ok].copy()
        test_df = test_df.loc[test_ok].copy()
        X_train = X_train[train_ok]
        y_train_model = y_train_true[train_ok] - baseline_train[train_ok]
        y_train_true = y_train_true[train_ok]
        baseline_train = baseline_train[train_ok]

        X_test = X_test[test_ok]
        y_test_true = y_test_true[test_ok]
        baseline_test_vals = baseline_test_vals[test_ok]

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is not installed.") from exc

    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=10.0,
        min_child_weight=20.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=args.seed,
        nthread=1,
        verbosity=0,
    )
    model.fit(X_train, y_train_model)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    if args.target_mode == "residual_vs_knn_analog":
        train_pred = train_pred + baseline_train
        test_pred = test_pred + baseline_test_vals

    train_metrics = metrics.regression_metrics(y_train_true, train_pred)
    test_metrics = metrics.regression_metrics(y_test_true, test_pred)
    test_metrics["mae_ge_90"] = _tail_mae(y_test_true, test_pred, 90.0)
    test_metrics["mae_ge_95"] = _tail_mae(y_test_true, test_pred, 95.0)

    baselines = list(DEFAULT_BASELINES) + [f"{spec.prefix}analog_mu"]
    baseline_test = _baseline_metrics(test_df, baselines)

    importances = getattr(model, "feature_importances_", None)
    top_features = None
    if importances is not None:
        pairs = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_cols, np.asarray(importances).reshape(-1))
        ]
        pairs = sorted(pairs, key=lambda item: abs(item["importance"]), reverse=True)
        top_features = pairs[:50]

    report_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path),
        "knn_lite": {"k": int(args.k), "label_lag_days": int(args.label_lag_days), "prefix": spec.prefix},
        "feature_mode": {"slot_mode": str(args.slot_mode)},
        "target_mode": str(args.target_mode),
        "splits": {
            "train": {"start": str(train_start), "end": str(train_end), "rows": int(len(train_df))},
            "test": {"start": str(test_start), "end": str(test_end), "rows": int(len(test_df))},
        },
        "model": {
            "name": "XGBRegressor",
            "params": {
                "n_estimators": int(model.n_estimators),
                "learning_rate": float(model.learning_rate),
                "max_depth": int(model.max_depth),
                "subsample": float(model.subsample),
                "colsample_bytree": float(model.colsample_bytree),
                "reg_lambda": float(model.reg_lambda),
                "min_child_weight": float(model.min_child_weight),
                "tree_method": str(model.tree_method),
                "random_state": int(args.seed),
            },
        },
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
            "baseline_test": baseline_test,
        },
        "feature_importance": {"type": "xgb_gain_proxy", "top_features": top_features}
        if top_features
        else None,
    }

    output_path = (
        Path(args.output)
        if args.output
        else csv_path.parent / f"mos_knn_lite_xgb_report_k{int(args.k)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    LOGGER.info("Report written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
