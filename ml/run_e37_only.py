from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_mos_45_suite import (
    SuiteContext,
    default_suite_id,
    load_csv,
    regression_metrics,
    run_moe_gate,
    setup_logging,
    split_by_date,
    write_json,
    build_feature_store,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E37 (MoE onshore vs offshore) only.")
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument("--run-id", default=default_suite_id())
    parser.add_argument("--out-root", default="artifacts/MOS/experiments")
    parser.add_argument("--train-start", default="2002-01-22")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--val-start", default="2020-01-01")
    parser.add_argument("--val-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()

    df_raw = load_csv(args.features)
    feature_store = build_feature_store(df_raw)
    feature_store = feature_store[feature_store["y_actual_tmax_f"].notna()].copy()

    split = split_by_date(
        feature_store,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    ctx = SuiteContext(
        df=feature_store,
        y=feature_store["y_actual_tmax_f"].to_numpy(dtype=float),
        train_mask=split.pop("train_mask"),
        val_mask=split.pop("val_mask"),
        test_mask=split.pop("test_mask"),
        seed=args.seed,
        cache={},
    )

    DOY = ["cal_d_doy_sin", "cal_d_doy_cos"]
    gate_features = ["feat_u", "feat_v", "feat_wsp_mean", *DOY]
    expert_features = [
        "feat_dd_models",
        "feat_tmp_range_mean_models",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
        *DOY,
    ]
    pred_train, pred_val, pred_test = run_moe_gate(
        ctx,
        gate_features=gate_features,
        expert_features=expert_features,
        gate_target="feat_onshore",
        base_series="feat_le_median_biascorr",
    )

    metrics_payload = {
        "train": regression_metrics(ctx.y[ctx.train_mask], pred_train),
        "validation": regression_metrics(ctx.y[ctx.val_mask], pred_val),
        "test": regression_metrics(ctx.y[ctx.test_mask], pred_test),
    }
    result = {
        "experiment_id": "E37",
        "name": "MoE onshore vs offshore",
        "features": [*gate_features, *expert_features],
        "metrics": metrics_payload,
        "extras": {},
    }

    output_root = Path(args.out_root) / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "experiments_summary.json", {"experiments": [result]})
    write_json(output_root / "split_info.json", split)

    flat = {
        "experiment_id": "E37",
        "name": result["name"],
        "train_mae": metrics_payload["train"].get("mae"),
        "val_mae": metrics_payload["validation"].get("mae"),
        "test_mae": metrics_payload["test"].get("mae"),
        "train_bias": metrics_payload["train"].get("bias"),
        "val_bias": metrics_payload["validation"].get("bias"),
        "test_bias": metrics_payload["test"].get("bias"),
    }
    pd.DataFrame([flat]).to_csv(output_root / "experiments_summary.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
