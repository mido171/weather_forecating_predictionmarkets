"""Train meta-models from OOF features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train meta models from OOF features.")
    parser.add_argument("--train-oof", required=True, help="Path to meta_features_train_oof.csv.")
    parser.add_argument("--output-dir", help="Output directory (defaults to alongside OOF).")
    parser.add_argument("--quantiles", nargs="*", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[30])
    return parser


def _sigma_feature_columns(windows: list[int]) -> list[str]:
    cols = [
        "sigma_gs",
        "sigma_mos",
        "width80_gs",
        "width80_mos",
        "abs_d_mu",
        "abs_d_sigma",
    ]
    for window in windows:
        cols.append(f"mae_gs_{window}")
        cols.append(f"mae_mos_{window}")
    return cols


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    oof_path = Path(args.train_oof)
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF features not found: {oof_path}")
    out_dir = Path(args.output_dir) if args.output_dir else oof_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(oof_path)
    windows = [int(w) for w in args.rolling_windows]
    feature_cols = common.meta_feature_columns(windows)

    y = df["y_true_f"].to_numpy(dtype=float)
    X, medians = common.impute_medians(df, feature_cols)

    point_model = Ridge(alpha=1.0)
    point_model.fit(X, y)
    joblib.dump(point_model, out_dir / "meta_model_point.pkl")

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required for meta quantile stacking.") from exc

    quantile_dir = out_dir / "meta_models_quantiles"
    quantile_dir.mkdir(parents=True, exist_ok=True)
    quantile_models = {}
    for q in args.quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(q),
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=300,
            min_data_in_leaf=20,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            random_state=1337,
            verbose=-1,
        )
        model.fit(X, y)
        key = common._quantile_key(float(q))
        quantile_models[key] = model
        joblib.dump(model, quantile_dir / f"{key}.pkl")

    sigma_cols = _sigma_feature_columns(windows)
    X_sigma, sigma_medians = common.impute_medians(df, sigma_cols)
    mu_blend = 0.5 * (df["mu_gs"].to_numpy(dtype=float) + df["mu_mos"].to_numpy(dtype=float))
    sigma_target = np.abs(y - mu_blend)
    sigma_model = Ridge(alpha=1.0)
    sigma_model.fit(X_sigma, np.log1p(sigma_target))
    joblib.dump(sigma_model, out_dir / "meta_model_sigma.pkl")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_cols,
        "feature_medians": medians,
        "sigma_feature_columns": sigma_cols,
        "sigma_feature_medians": sigma_medians,
        "quantiles": [float(q) for q in args.quantiles],
    }
    common.write_json(out_dir / "meta_model_metadata.json", metadata)
    print(f"Meta models written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
