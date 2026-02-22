from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y_true[mask] - y_pred[mask])))


def _bias(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.nanmean(y_pred[mask] - y_true[mask]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E37 minute-condensed A/B suite.")
    parser.add_argument("--feature-store", required=True, help="Merged feature store parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory for report and preds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import sys

    sys.path.append("ml")
    import run_mos_45_suite as base

    df = pd.read_parquet(args.feature_store)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])

    split = base.split_by_date(
        df,
        train_start="2002-01-22",
        train_end="2019-12-31",
        val_start="2020-01-01",
        val_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2025-12-31",
    )
    train_mask = split.pop("train_mask")
    val_mask = split.pop("val_mask")
    test_mask = split.pop("test_mask")

    ctx = base.SuiteContext(
        df=df,
        y=df["y_actual_tmax_f"].to_numpy(dtype=float),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        seed=args.seed,
        cache={},
    )

    DOY = ["cal_d_doy_sin", "cal_d_doy_cos"]
    gate_features = ["feat_u", "feat_v", "feat_wsp_mean", *DOY]
    expert_features_base = [
        "feat_dd_models",
        "feat_tmp_range_mean_models",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
        *DOY,
    ]
    base_series = "feat_le_median_biascorr"
    gate_label = (df["feat_onshore"] > 0.5).astype(int).to_numpy(dtype=int)

    def run_variant(expert_features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_vals = pd.to_numeric(df.get(base_series), errors="coerce").to_numpy(dtype=float)
        base_mean = float(np.nanmean(ctx.y[ctx.train_mask]))
        base_vals = np.where(np.isnan(base_vals), base_mean, base_vals)

        gate_df = base.ensure_columns(df, gate_features)
        gate_X, _ = base.impute_features(gate_df[gate_features], ctx.train_mask)
        X_gate = gate_X.to_numpy(dtype=float)
        gate_model = base.train_lgbm_classifier(
            X_gate[ctx.train_mask],
            gate_label[ctx.train_mask],
            X_gate[ctx.val_mask],
            gate_label[ctx.val_mask],
            seed=ctx.seed,
        )
        p_gate = gate_model.predict_proba(X_gate)[:, 1]

        expert_df = base.ensure_columns(df, expert_features)
        expert_X, _ = base.impute_features(expert_df[expert_features], ctx.train_mask)
        X_exp = expert_X.to_numpy(dtype=float)

        def fit_expert(mask: np.ndarray):
            if not mask.any():
                return None
            return base.train_lgbm_regressor(
                X_exp[mask],
                (ctx.y[mask] - base_vals[mask]),
                X_exp[ctx.val_mask & mask],
                (ctx.y[ctx.val_mask & mask] - base_vals[ctx.val_mask & mask]),
                seed=ctx.seed,
            )

        expert_on = fit_expert(gate_label == 1)
        expert_off = fit_expert(gate_label == 0)

        def predict(model, X_sub):
            if model is None:
                return np.full(len(X_sub), np.nan)
            return model.predict(X_sub)

        resid_on = predict(expert_on, X_exp)
        resid_off = predict(expert_off, X_exp)
        pred_all = base_vals + p_gate * resid_on + (1 - p_gate) * resid_off
        return pred_all, p_gate, base_vals

    # V0 baseline
    pred_v0, p_gate, base_vals = run_variant(expert_features_base)

    metrics = {}
    metrics["V0"] = {
        "train": base.regression_metrics(ctx.y[ctx.train_mask], pred_v0[ctx.train_mask]),
        "validation": base.regression_metrics(ctx.y[ctx.val_mask], pred_v0[ctx.val_mask]),
        "test": base.regression_metrics(ctx.y[ctx.test_mask], pred_v0[ctx.test_mask]),
    }

    if abs(metrics["V0"]["test"]["mae"] - 0.81) > 0.1:
        raise RuntimeError(
            f"Baseline V0 MAE not close to expected 0.81 (got {metrics['V0']['test']['mae']:.4f})."
        )

    minute_all = [
        "iem_tmax_t1",
        "iem_tmin_t1",
        "iem_range_t1",
        "tmax_time_min_t1",
        "plateau_05_t1",
        "heat_12_15_t1",
        "heat_15_18_t1",
        "cool_18_21_t1",
        "max_drop_30_t1",
        "drop_cnt_15_19_t1",
        "T00",
        "T03",
        "T06",
        "night_drop_00_06",
        "slope_last180",
        "std_last180",
        "T06_adj",
        "diff_lag1",
        "diff_ewma_30",
        "diff_std_30",
        "MRI_suppress",
        "MRI_late",
    ]
    minute_night = ["T06_adj", "night_drop_00_06", "slope_last180", "std_last180"]
    minute_t1 = [
        "iem_range_t1",
        "tmax_time_min_t1",
        "plateau_05_t1",
        "max_drop_30_t1",
        "drop_cnt_15_19_t1",
        "heat_12_15_t1",
        "heat_15_18_t1",
    ]
    minute_translator = ["diff_lag1", "diff_ewma_30", "diff_std_30"]
    minute_indices = ["MRI_suppress", "MRI_late"]

    variants = {
        "V1": expert_features_base + minute_night,
        "V2": expert_features_base + minute_t1,
        "V3": expert_features_base + minute_translator,
        "V4": expert_features_base + minute_indices,
        "V5": expert_features_base + minute_all,
    }

    preds = {"V0": pred_v0}
    for name, feats in variants.items():
        pred_all, _, _ = run_variant(feats)
        preds[name] = pred_all
        metrics[name] = {
            "train": base.regression_metrics(ctx.y[ctx.train_mask], pred_all[ctx.train_mask]),
            "validation": base.regression_metrics(ctx.y[ctx.val_mask], pred_all[ctx.val_mask]),
            "test": base.regression_metrics(ctx.y[ctx.test_mask], pred_all[ctx.test_mask]),
        }

    # V6: second-stage corrector on residuals
    minute_only = base.ensure_columns(df, minute_all)
    minute_X, _ = base.impute_features(minute_only[minute_all], ctx.train_mask)
    X_minute = minute_X.to_numpy(dtype=float)
    residual = ctx.y - pred_v0
    model_v6 = base.train_lgbm_regressor(
        X_minute[ctx.train_mask],
        residual[ctx.train_mask],
        X_minute[ctx.val_mask],
        residual[ctx.val_mask],
        seed=ctx.seed,
    )
    e_hat = model_v6.predict(X_minute)
    pred_v6 = pred_v0 + e_hat
    preds["V6"] = pred_v6
    metrics["V6"] = {
        "train": base.regression_metrics(ctx.y[ctx.train_mask], pred_v6[ctx.train_mask]),
        "validation": base.regression_metrics(ctx.y[ctx.val_mask], pred_v6[ctx.val_mask]),
        "test": base.regression_metrics(ctx.y[ctx.test_mask], pred_v6[ctx.test_mask]),
    }

    # Slice metrics
    onshore_mask = (df["feat_onshore"] > 0.5).to_numpy() & ctx.test_mask
    offshore_mask = (df["feat_onshore"] <= 0.5).to_numpy() & ctx.test_mask
    suppress = df["MRI_suppress"].to_numpy(dtype=float)
    suppress_test = suppress[ctx.test_mask]
    high_cut = np.nanpercentile(suppress_test, 70)
    low_cut = np.nanpercentile(suppress_test, 30)
    high_mask = (suppress >= high_cut) & ctx.test_mask
    low_mask = (suppress <= low_cut) & ctx.test_mask

    slice_metrics = {}
    for name, pred in preds.items():
        slice_metrics[name] = {
            "test_onshore_mae": _mae(ctx.y, pred, onshore_mask),
            "test_offshore_mae": _mae(ctx.y, pred, offshore_mask),
            "test_high_suppress_mae": _mae(ctx.y, pred, high_mask),
            "test_low_suppress_mae": _mae(ctx.y, pred, low_mask),
        }

    # Leakage assertions
    leak_violation = df.get("leak_violation")
    leak_count = int(leak_violation.sum()) if leak_violation is not None else 0
    max_delta = None
    if "max_minute_ts_used_utc" in df.columns and "decision_utc" in df.columns:
        delta = pd.to_datetime(df["max_minute_ts_used_utc"]) - pd.to_datetime(df["decision_utc"])
        max_delta = delta.max()

    # Save preds
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame(
        {
            "target_date_local": df["target_date_local"].dt.date,
            "y": ctx.y,
            "base": base_vals,
            "p_gate": p_gate,
            "pred_v0": pred_v0,
            "pred_v1": preds.get("V1"),
            "pred_v2": preds.get("V2"),
            "pred_v3": preds.get("V3"),
            "pred_v4": preds.get("V4"),
            "pred_v5": preds.get("V5"),
            "pred_v6": preds.get("V6"),
        }
    )
    preds_df.to_parquet(out_dir / "preds.parquet", index=False)

    # Report
    def fmt(x: float) -> str:
        return f"{x:.4f}"

    rows = []
    for name in ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]:
        m = metrics[name]
        rows.append(
            {
                "variant": name,
                "train_mae": m["train"]["mae"],
                "val_mae": m["validation"]["mae"],
                "test_mae": m["test"]["mae"],
                "test_delta_vs_v0": m["test"]["mae"] - metrics["V0"]["test"]["mae"],
                "test_bias": m["test"]["bias"],
            }
        )
    table = pd.DataFrame(rows)

    report_lines = []
    report_lines.append("# E37_MINUTE_CONDENSED_V1 Report")
    report_lines.append("")
    report_lines.append("## Baseline reproduction")
    report_lines.append(
        f"V0 test MAE = {fmt(metrics['V0']['test']['mae'])} (expected ~0.81)."
    )
    report_lines.append("")
    report_lines.append("## MAE summary")
    report_lines.append("| Variant | Train MAE | Val MAE | Test MAE | ΔTest vs V0 | Test Bias |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in table.iterrows():
        report_lines.append(
            f"| {r['variant']} | {fmt(r['train_mae'])} | {fmt(r['val_mae'])} | {fmt(r['test_mae'])} | {fmt(r['test_delta_vs_v0'])} | {fmt(r['test_bias'])} |"
        )
    report_lines.append("")
    report_lines.append("## Slice metrics (test)")
    report_lines.append("| Variant | Onshore MAE | Offshore MAE | High Suppress MAE | Low Suppress MAE |")
    report_lines.append("|---|---:|---:|---:|---:|")
    for name in ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]:
        sm = slice_metrics[name]
        report_lines.append(
            f"| {name} | {fmt(sm['test_onshore_mae'])} | {fmt(sm['test_offshore_mae'])} | {fmt(sm['test_high_suppress_mae'])} | {fmt(sm['test_low_suppress_mae'])} |"
        )
    report_lines.append("")
    report_lines.append("## Leakage assertions")
    report_lines.append(f"Leak violations: {leak_count}")
    if max_delta is not None:
        report_lines.append(f"Max(max_minute_ts_used_utc - decision_utc): {max_delta}")
    report_lines.append("")
    report_lines.append("## Worst 20 errors per variant (test)")
    for name in ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]:
        pred = preds[name]
        abs_err = np.abs(ctx.y - pred)
        test_idx = np.where(ctx.test_mask)[0]
        worst = test_idx[np.argsort(abs_err[test_idx])[-20:]][::-1]
        report_lines.append(f"### {name}")
        report_lines.append("| target_date_local | abs_error |")
        report_lines.append("|---|---:|")
        for idx in worst:
            report_lines.append(f"| {df['target_date_local'].iloc[idx].date()} | {fmt(abs_err[idx])} |")
        report_lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
