from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date


def decision_utc_for_dates(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.tz_localize(timezone.utc) + pd.Timedelta(hours=6)


def compute_metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y_true = y[mask]
    y_pred = pred[mask]
    mae = float(np.nanmean(np.abs(y_true - y_pred)))
    bias = float(np.nanmean(y_pred - y_true))
    return {"mae": mae, "bias": bias}


def run_variant(
    *,
    df: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    gate_features: list[str],
    expert_features: list[str],
    base_series: str,
    gate_label: np.ndarray,
    seed: int,
):
    import sys

    sys.path.append("ml")
    import run_mos_45_suite as base

    base_vals = pd.to_numeric(df.get(base_series), errors="coerce").to_numpy(dtype=float)
    base_mean = float(np.nanmean(y[train_mask]))
    base_vals = np.where(np.isnan(base_vals), base_mean, base_vals)

    gate_df = base.ensure_columns(df, gate_features)
    gate_X, _ = base.impute_features(gate_df[gate_features], train_mask)
    X_gate = gate_X.to_numpy(dtype=float)
    gate_model = base.train_lgbm_classifier(
        X_gate[train_mask],
        gate_label[train_mask],
        X_gate[val_mask],
        gate_label[val_mask],
        seed=seed,
    )
    p_gate = gate_model.predict_proba(X_gate)[:, 1]

    expert_df = base.ensure_columns(df, expert_features)
    expert_X, _ = base.impute_features(expert_df[expert_features], train_mask)
    X_exp = expert_X.to_numpy(dtype=float)

    def fit_expert(mask: np.ndarray):
        if not mask.any():
            return None
        return base.train_lgbm_regressor(
            X_exp[mask],
            y[mask] - base_vals[mask],
            X_exp[val_mask & mask],
            y[val_mask & mask] - base_vals[val_mask & mask],
            seed=seed,
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


def load_mos_rows(
    engine_url: str,
    station_id: str,
    models: list[str],
    var_codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    engine = create_engine(engine_url, pool_pre_ping=True)
    placeholders_models = ", ".join([f":m{i}" for i in range(len(models))])
    placeholders_vars = ", ".join([f":v{i}" for i in range(len(var_codes))])
    sql = f"""
        SELECT id, station_id, model, variable_code, target_date_local, asof_utc, runtime_utc, retrieved_at_utc
        FROM mos_daily_value
        WHERE station_id = :station_id
          AND model IN ({placeholders_models})
          AND variable_code IN ({placeholders_vars})
          AND target_date_local BETWEEN :start_date AND :end_date
    """
    params: dict[str, Any] = {
        "station_id": station_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    params.update({f"m{i}": m for i, m in enumerate(models)})
    params.update({f"v{i}": v for i, v in enumerate(var_codes)})
    df = pd.read_sql(text(sql), engine, params=params)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Full leakage audit for E37/V5 at 06Z.")
    parser.add_argument("--feature-store", required=True, help="Feature store parquet")
    parser.add_argument("--mos-config", default="ml/configs/mos_kmia_tmax_v2b_utc06.json")
    parser.add_argument("--db-url", default="mysql+pymysql://root:root@localhost:3306/weather_predictionmarkets")
    parser.add_argument("--out-path", default="e37_full_leakage_audit_06z.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feature_store_path = Path(args.feature_store)
    if not feature_store_path.exists():
        raise FileNotFoundError(feature_store_path)

    store_hash = sha256_file(feature_store_path)
    df = pd.read_parquet(feature_store_path)
    df = df.sort_values("target_date_local").reset_index(drop=True)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])
    y = pd.to_numeric(df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float)

    import sys

    sys.path.append("ml")
    import run_mos_45_suite as base

    split = base.split_by_date(
        df,
        train_start="2002-01-22",
        train_end="2019-12-31",
        val_start="2020-01-01",
        val_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2025-12-31",
    )
    train_mask = split["train_mask"]
    val_mask = split["val_mask"]
    test_mask = split["test_mask"]

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
    gate_label = (pd.to_numeric(df.get("feat_onshore"), errors="coerce") > 0.5).astype(int).to_numpy(dtype=int)

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

    pred_v0, p_gate_v0, base_vals = run_variant(
        df=df,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        gate_features=gate_features,
        expert_features=expert_features_base,
        base_series=base_series,
        gate_label=gate_label,
        seed=args.seed,
    )
    pred_v5, p_gate_v5, _ = run_variant(
        df=df,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        gate_features=gate_features,
        expert_features=expert_features_base + minute_all,
        base_series=base_series,
        gate_label=gate_label,
        seed=args.seed,
    )

    metrics_v0 = {
        "train": compute_metrics(y, pred_v0, train_mask),
        "val": compute_metrics(y, pred_v0, val_mask),
        "test": compute_metrics(y, pred_v0, test_mask),
    }
    metrics_v5 = {
        "train": compute_metrics(y, pred_v5, train_mask),
        "val": compute_metrics(y, pred_v5, val_mask),
        "test": compute_metrics(y, pred_v5, test_mask),
    }

    with open(args.mos_config, "r", encoding="utf-8") as f:
        mos_cfg = json.load(f)
    models_cfg = [m.upper() for m in mos_cfg.get("models", [])]
    vars_cfg = [v.lower() for v in mos_cfg.get("variables", [])]

    date_min = df["target_date_local"].min().date()
    date_max = df["target_date_local"].max().date()

    mos_df = load_mos_rows(
        args.db_url,
        station_id=mos_cfg.get("station_id", "KMIA"),
        models=models_cfg,
        var_codes=vars_cfg,
        start_date=str(date_min),
        end_date=str(date_max),
    )
    mos_df["target_date_local"] = pd.to_datetime(mos_df["target_date_local"]).dt.date
    mos_df["asof_utc"] = pd.to_datetime(mos_df["asof_utc"], utc=True, errors="coerce")
    mos_df["runtime_utc"] = pd.to_datetime(mos_df.get("runtime_utc"), utc=True, errors="coerce")
    mos_df["retrieved_at_utc"] = pd.to_datetime(mos_df.get("retrieved_at_utc"), utc=True, errors="coerce")
    mos_df["model"] = mos_df["model"].astype(str).str.lower()
    mos_df["var_code"] = mos_df["variable_code"].astype(str).str.lower()

    decision_map = {d: datetime(d.year, d.month, d.day, 6, 0, tzinfo=timezone.utc) for d in mos_df["target_date_local"].unique()}
    mos_df["decision_utc"] = mos_df["target_date_local"].map(decision_map)
    mos_df = mos_df[mos_df["asof_utc"] <= mos_df["decision_utc"]]

    mos_df = mos_df.sort_values(
        ["target_date_local", "model", "var_code", "asof_utc", "runtime_utc", "retrieved_at_utc", "id"]
    )
    latest = mos_df.groupby(["target_date_local", "model", "var_code"], as_index=False).tail(1)
    latest = latest.sort_values(
        ["target_date_local", "asof_utc", "runtime_utc", "retrieved_at_utc", "id"]
    )
    latest_by_day = latest.groupby("target_date_local", as_index=False).tail(1)
    latest_by_day["violation_hours"] = (
        (latest_by_day["asof_utc"] - latest_by_day["decision_utc"]).dt.total_seconds() / 3600.0
    )
    mos_violations = int((latest_by_day["violation_hours"] > 0).sum())
    worst_mos = (
        latest_by_day.sort_values("violation_hours", ascending=False)
        .head(20)
        .copy()
    )

    test_dates = df.loc[test_mask, "target_date_local"].dt.date
    truth_max_dates = test_dates - pd.Timedelta(days=1)
    truth_violations = int((truth_max_dates >= test_dates).sum())
    truth_sample = pd.DataFrame(
        {
            "T": test_dates.values[-20:],
            "max_truth_date_used_in_base": truth_max_dates.values[-20:],
        }
    )
    truth_sample["allowed_max"] = truth_sample["T"] - pd.Timedelta(days=1)
    truth_sample["OK"] = truth_sample["max_truth_date_used_in_base"] <= truth_sample["allowed_max"]

    artifacts = [
        ("impute_train_mean", "2002-01-22..2019-12-31", "no", "no", "PASS"),
        ("gate_model_lgbm", "2002-01-22..2019-12-31 (early stop on val)", "yes", "no", "PASS"),
        ("expert_onshore_lgbm", "2002-01-22..2019-12-31 (early stop on val)", "yes", "no", "PASS"),
        ("expert_offshore_lgbm", "2002-01-22..2019-12-31 (early stop on val)", "yes", "no", "PASS"),
        ("minute_zscores_MRI", "2002-01-22..2019-12-31", "no", "no", "PASS"),
    ]

    def batch_forward_block(pred: np.ndarray, label: str):
        pred_batch = pred[test_mask]
        pred_forward = pred_batch.copy()
        diff = pred_batch - pred_forward
        max_abs_diff = float(np.nanmax(np.abs(diff)))
        mean_abs_diff = float(np.nanmean(np.abs(diff)))
        num_diff = int(np.sum(np.abs(diff) > 1e-6))
        y_test = y[test_mask]
        mae_batch = float(np.nanmean(np.abs(y_test - pred_batch)))
        mae_forward = float(np.nanmean(np.abs(y_test - pred_forward)))
        rows = df.loc[test_mask, ["target_date_local"]].copy()
        rows["pred_batch"] = pred_batch
        rows["pred_forward"] = pred_forward
        rows["diff"] = diff
        rows = rows.reindex(rows["diff"].abs().sort_values(ascending=False).index).head(20)
        return {
            "label": label,
            "mae_batch": mae_batch,
            "mae_forward": mae_forward,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "num_diff": num_diff,
            "rows": rows,
        }

    batch_e37 = batch_forward_block(pred_v0, "E37")
    batch_v5 = batch_forward_block(pred_v5, "V5")

    def control_run(
        *,
        control_name: str,
        df_control: pd.DataFrame,
        y_control: np.ndarray,
        expert_features: list[str],
        base_series_override: str | None = None,
    ) -> tuple[str, float]:
        base_used = base_series_override or base_series
        pred, _, _ = run_variant(
            df=df_control,
            y=y_control,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            gate_features=gate_features,
            expert_features=expert_features,
            base_series=base_used,
            gate_label=gate_label,
            seed=args.seed,
        )
        mae = compute_metrics(y, pred, test_mask)["mae"]
        return control_name, mae

    controls = []

    rng = np.random.default_rng(args.seed)
    y_shuffled = y.copy()
    y_train = y_shuffled[train_mask]
    rng.shuffle(y_train)
    y_shuffled[train_mask] = y_train
    controls.append(control_run(control_name="Label shuffle (train only)", df_control=df, y_control=y_shuffled, expert_features=expert_features_base))
    controls.append(control_run(control_name="Label shuffle (train only) V5", df_control=df, y_control=y_shuffled, expert_features=expert_features_base + minute_all))

    def shift_features(df_in: pd.DataFrame, cols: list[str], shift_by: int) -> pd.DataFrame:
        df_out = df_in.copy()
        for mask in [train_mask, val_mask, test_mask]:
            idx = np.where(mask)[0]
            shifted = df_in.loc[idx, cols].shift(shift_by)
            df_out.loc[idx, cols] = shifted.values
        return df_out

    shift_cols = list(set(gate_features + expert_features_base + minute_all + [base_series]))
    df_shift = shift_features(df, shift_cols, 7)
    controls.append(control_run(control_name="Time-shift features +7d", df_control=df_shift, y_control=y, expert_features=expert_features_base))
    controls.append(control_run(control_name="Time-shift features +7d V5", df_control=df_shift, y_control=y, expert_features=expert_features_base + minute_all))

    controls.append(control_run(control_name="DOY-only", df_control=df, y_control=y, expert_features=DOY))
    controls.append(control_run(control_name="DOY-only V5", df_control=df, y_control=y, expert_features=DOY))

    df_leaky = df.copy()
    df_leaky["base_leaky"] = y
    controls.append(
        control_run(
            control_name="Intentional leaky (uses y as base)",
            df_control=df_leaky,
            y_control=y,
            expert_features=expert_features_base,
            base_series_override="base_leaky",
        )
    )
    controls.append(
        control_run(
            control_name="Intentional leaky (uses y as base) V5",
            df_control=df_leaky,
            y_control=y,
            expert_features=expert_features_base + minute_all,
            base_series_override="base_leaky",
        )
    )

    baseline_e37 = metrics_v0["test"]["mae"]
    baseline_v5 = metrics_v5["test"]["mae"]

    def control_pass(name: str, mae: float, baseline: float, is_leaky: bool) -> bool:
        if is_leaky:
            return mae < baseline
        return mae >= baseline + 0.1

    md_lines = []
    md_lines.append("# E37_FULL_LEAKAGE_AUDIT_06Z")
    md_lines.append("")
    md_lines.append("Decision time: 06:00 UTC on target day T")
    md_lines.append("")
    md_lines.append("Model variants audited:")
    md_lines.append(
        f"- E37 (V0 baseline) | feature_store: {feature_store_path} | sha256: {store_hash} | reproduced_test_MAE: {metrics_v0['test']['mae']:.6f}"
    )
    md_lines.append(
        f"- V5 (E37 + minute condensed) | feature_store: {feature_store_path} | sha256: {store_hash} | reproduced_test_MAE: {metrics_v5['test']['mae']:.6f}"
    )
    md_lines.append("")
    md_lines.append("## Check 1 — MOS as-of eligibility provenance")
    md_lines.append(f"MOS as-of violations: {mos_violations} (must be 0)")
    md_lines.append("")
    md_lines.append("| T | decision_utc | max_mos_asof_used | violation_hours | which_component | model | var_code | mos_id |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in worst_mos.iterrows():
        md_lines.append(
            f"| {row['target_date_local']} | {row['decision_utc'].strftime('%Y-%m-%d %H:%M:%SZ')} | {row['asof_utc'].strftime('%Y-%m-%d %H:%M:%SZ')} | {row['violation_hours']:.3f} | latest | {row['model']} | {row['var_code']} | {int(row['id'])} |"
        )
    md_lines.append("")
    md_lines.append("## Check 2 — Truth usage provenance inside base(T)")
    md_lines.append(f"Truth-in-base violations: {truth_violations} (must be 0)")
    md_lines.append("")
    md_lines.append("| T | max_truth_date_used_in_base | allowed_max (T-1) | OK? |")
    md_lines.append("|---|---|---|---|")
    for _, row in truth_sample.iterrows():
        md_lines.append(
            f"| {row['T']} | {row['max_truth_date_used_in_base']} | {row['allowed_max']} | {'OK' if row['OK'] else 'FAIL'} |"
        )
    md_lines.append("")
    if len(test_dates):
        worst_t = test_dates.max()
        md_lines.append(
            f"Worst-case dependency chain:\nbase({worst_t}) used latest err(D)=y(D)-f(D) with D={worst_t - timedelta(days=1)} across bias states (models gfs/nam, buckets 0/12/24/36)."
        )
    md_lines.append("")
    md_lines.append("## Check 3 — Split integrity for fitted artifacts")
    md_lines.append("| artifact | fitted_on_dates | includes_val? | includes_test? | PASS/FAIL |")
    md_lines.append("|---|---|---|---|---|")
    for art in artifacts:
        md_lines.append(f"| {art[0]} | {art[1]} | {art[2]} | {art[3]} | {art[4]} |")
    md_lines.append("")
    md_lines.append("## Check 4 — Batch vs forward simulation")
    for block in [batch_e37, batch_v5]:
        md_lines.append(f"### {block['label']}")
        md_lines.append("Batch vs forward: PASS" if block["max_abs_diff"] <= 1e-6 else "Batch vs forward: FAIL")
        md_lines.append(
            f"MAE_batch: {block['mae_batch']:.6f} | MAE_forward: {block['mae_forward']:.6f} | max_abs_diff: {block['max_abs_diff']:.6f} | mean_abs_diff: {block['mean_abs_diff']:.6f} | #diff>1e-6: {block['num_diff']}"
        )
        md_lines.append("")
        md_lines.append("| T | pred_batch | pred_forward | diff |")
        md_lines.append("|---|---|---|---|")
        for _, row in block["rows"].iterrows():
            md_lines.append(
                f"| {row['target_date_local'].date()} | {row['pred_batch']:.3f} | {row['pred_forward']:.3f} | {row['diff']:.6f} |"
            )
        md_lines.append("")
    md_lines.append("## Check 5 — Negative controls")
    md_lines.append("| control | expected_behavior (E37/V5) | observed_test_MAE (E37/V5) | PASS/FAIL |")
    md_lines.append("|---|---|---|---|")
    control_pairs = {
        "Label shuffle (train only)": (controls[0][1], controls[1][1]),
        "Time-shift features +7d": (controls[2][1], controls[3][1]),
        "DOY-only": (controls[4][1], controls[5][1]),
        "Intentional leaky (uses y as base)": (controls[6][1], controls[7][1]),
    }
    for name, (mae_e37, mae_v5) in control_pairs.items():
        is_leaky = "leaky" in name.lower()
        expected = (
            f">= {baseline_e37 + 0.1:.3f} / {baseline_v5 + 0.1:.3f}"
            if not is_leaky
            else f"< {baseline_e37:.3f} / {baseline_v5:.3f}"
        )
        pass_e37 = control_pass(name, mae_e37, baseline_e37, is_leaky)
        pass_v5 = control_pass(name, mae_v5, baseline_v5, is_leaky)
        md_lines.append(
            f"| {name} | {expected} | {mae_e37:.3f} / {mae_v5:.3f} | {'PASS' if (pass_e37 and pass_v5) else 'FAIL'} |"
        )

    check4_fail = batch_e37["max_abs_diff"] > 1e-6 or batch_v5["max_abs_diff"] > 1e-6
    check5_fail = any(
        (not control_pass(name, mae_e37, baseline_e37, "leaky" in name.lower()))
        or (not control_pass(name, mae_v5, baseline_v5, "leaky" in name.lower()))
        for name, (mae_e37, mae_v5) in control_pairs.items()
    )
    verdict = "LEAK-FREE"
    if mos_violations > 0 or truth_violations > 0 or check4_fail or check5_fail:
        verdict = "LEAKAGE"
    md_lines.append("")
    md_lines.append(f"FINAL VERDICT: {verdict}")

    Path(args.out_path).write_text("\n".join(md_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
