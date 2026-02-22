"""Generate OOF base predictions and meta training features."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from . import common
from weather_ml import config as config_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build OOF base predictions for meta stacking.")
    parser.add_argument("--start", required=True, help="Meta-train start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Meta-train end date (YYYY-MM-DD).")
    parser.add_argument("--folds", type=int, default=5, help="Number of time-series folds.")
    parser.add_argument("--gap-days", type=int, default=2, help="Gap days between train/val folds.")
    parser.add_argument("--truth-lag", type=int, default=2, help="Truth lag days for leakage-safe features.")
    parser.add_argument("--grib-config", help="Path to EX210 config_resolved.yaml.")
    parser.add_argument("--mos-features", help="Path to MOS features.csv.")
    parser.add_argument("--mos-train-config", help="Path to MOS training config YAML.")
    parser.add_argument("--quantiles", nargs="*", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[30])
    parser.add_argument("--output-root", help="Optional output root under artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = common.resolve_repo_root()

    start = common.parse_date(args.start)
    end = common.parse_date(args.end)

    grib_config_path = (
        Path(args.grib_config) if args.grib_config else common.default_grib_config_path(repo_root)
    )
    mos_features_path = (
        Path(args.mos_features) if args.mos_features else common.default_mos_features_path(repo_root)
    )
    mos_train_cfg_path = (
        Path(args.mos_train_config)
        if args.mos_train_config
        else common.default_mos_train_config_path(repo_root)
    )

    grib_cfg = config_module.load_config(grib_config_path)
    grib_cfg = config_module.resolve_paths(grib_cfg, repo_root=repo_root)
    mos_train_cfg = common.load_yaml(mos_train_cfg_path)

    grib_df = common.load_gribstream_df(Path(grib_cfg.data.csv_path))
    mos_df = common.load_mos_df(mos_features_path)

    folds = common.make_oof_folds(
        grib_df,
        start=start,
        end=end,
        n_splits=int(args.folds),
        gap_days=int(args.gap_days),
    )
    if not folds:
        raise ValueError("No folds generated; check date range and dataset coverage.")

    gs_preds = []
    mos_preds = []

    for fold_idx, (train_dates, val_dates) in enumerate(folds, start=1):
        train_end = max(train_dates)
        grib_train_idx = grib_df.index[grib_df["target_date_local"] <= train_end]
        grib_pred_idx = grib_df.index[grib_df["target_date_local"].isin(val_dates)]

        mos_train_idx = mos_df.index[mos_df["target_date_local"] <= train_end]
        mos_pred_idx = mos_df.index[mos_df["target_date_local"].isin(val_dates)]

        gs = common.predict_gribstream(
            grib_df,
            config=grib_cfg,
            train_index=grib_train_idx,
            predict_index=grib_pred_idx,
            truth_lag=int(args.truth_lag),
            quantiles=args.quantiles,
        )
        gs.frame["fold"] = fold_idx
        gs_preds.append(gs.frame)

        mos = common.train_mos_predict(
            mos_df,
            train_index=mos_train_idx,
            predict_index=mos_pred_idx,
            quantiles=args.quantiles,
            lgbm_params=mos_train_cfg.get("lgbm_params", {}),
            quantile_params=mos_train_cfg.get("quantile_params", {}),
            recency_lambda=float(mos_train_cfg.get("recency_lambda", 0.0)),
        )
        mos.frame["fold"] = fold_idx
        mos_preds.append(mos.frame)

    gs_all = pd.concat(gs_preds, ignore_index=True)
    mos_all = pd.concat(mos_preds, ignore_index=True)

    merged = common.merge_base_predictions(gs_all, mos_all)
    merged = common.add_meta_features(
        merged,
        windows=[int(w) for w in args.rolling_windows],
        lag_days=int(args.truth_lag),
    )
    feature_cols = common.meta_feature_columns([int(w) for w in args.rolling_windows])

    output_root = (
        Path(args.output_root)
        if args.output_root
        else repo_root / "artifacts" / "meta_stack_eval"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    oof_path = run_dir / "meta_features_train_oof.csv"
    merged.to_csv(oof_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "meta_train": {"start": str(start), "end": str(end)},
        "folds": len(folds),
        "gap_days": int(args.gap_days),
        "truth_lag_days": int(args.truth_lag),
        "grib_config": str(grib_config_path),
        "mos_features": str(mos_features_path),
        "mos_train_config": str(mos_train_cfg_path),
        "quantiles": args.quantiles,
        "rolling_windows": [int(w) for w in args.rolling_windows],
        "rows": int(len(merged)),
        "feature_columns": feature_cols,
    }
    common.write_json(run_dir / "meta_oof_manifest.json", manifest)
    print(f"Wrote OOF meta features: {oof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
