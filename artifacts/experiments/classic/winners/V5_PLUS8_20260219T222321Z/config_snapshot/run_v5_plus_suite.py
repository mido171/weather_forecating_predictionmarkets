from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, text


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y_true - y_pred)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(y_pred - y_true))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        SELECT id, station_id, model, variable_code, target_date_local, asof_utc, runtime_utc, retrieved_at_utc,
               value_mean, value_max, value_min
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
    return pd.read_sql(text(sql), engine, params=params)


def compute_revision_features(
    mos_df: pd.DataFrame,
    *,
    max_vars: set[str],
    models: list[str],
    var_codes: list[str],
) -> pd.DataFrame:
    df = mos_df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True, errors="coerce")
    df["runtime_utc"] = pd.to_datetime(df["runtime_utc"], utc=True, errors="coerce")
    df["retrieved_at_utc"] = pd.to_datetime(df["retrieved_at_utc"], utc=True, errors="coerce")
    df["model"] = df["model"].astype(str).str.lower()
    df["variable_code"] = df["variable_code"].astype(str).str.lower()

    # choose value based on variable
    use_max = df["variable_code"].isin(list(max_vars))
    val = df["value_mean"].copy()
    val = val.where(~use_max, df["value_max"])
    val = val.where(~val.isna(), df["value_max"])
    val = val.where(~val.isna(), df["value_min"])
    df["value"] = val

    decision_utc = pd.to_datetime(df["target_date_local"]).dt.tz_localize(timezone.utc) + pd.Timedelta(hours=6)
    df["decision_utc"] = decision_utc

    buckets = [0, 12, 24]
    wide = None
    for bucket in buckets:
        threshold = df["decision_utc"] - pd.Timedelta(hours=bucket)
        df_b = df[df["asof_utc"] <= threshold]
        df_b = df_b.sort_values(
            ["target_date_local", "model", "variable_code", "asof_utc", "runtime_utc", "retrieved_at_utc", "id"]
        )
        latest = df_b.groupby(["target_date_local", "model", "variable_code"], as_index=False).tail(1)
        pivot = latest.pivot_table(
            index="target_date_local",
            columns=["model", "variable_code"],
            values="value",
            aggfunc="first",
        )
        pivot.columns = [f"mos_{var}_{model}_b{bucket}" for (model, var) in pivot.columns]
        pivot = pivot.reset_index()
        wide = pivot if wide is None else wide.merge(pivot, on="target_date_local", how="outer")

    if wide is None:
        return pd.DataFrame()

    # revisions and disagreements
    for model in models:
        model_l = model.lower()
        for var in var_codes:
            col0 = f"mos_{var}_{model_l}_b0"
            col12 = f"mos_{var}_{model_l}_b12"
            col24 = f"mos_{var}_{model_l}_b24"
            if col0 in wide.columns and col12 in wide.columns:
                wide[f"rev12_{var}_{model_l}"] = wide[col0] - wide[col12]
                wide[f"abs_rev12_{var}_{model_l}"] = wide[f"rev12_{var}_{model_l}"].abs()
            if col0 in wide.columns and col24 in wide.columns:
                wide[f"rev24_{var}_{model_l}"] = wide[col0] - wide[col24]
                wide[f"abs_rev24_{var}_{model_l}"] = wide[f"rev24_{var}_{model_l}"].abs()

    # cross-model disagreement at b0
    for var in var_codes:
        gfs = f"mos_{var}_gfs_b0"
        nam = f"mos_{var}_nam_b0"
        if gfs in wide.columns and nam in wide.columns:
            wide[f"disc0_{var}"] = wide[gfs] - wide[nam]
            wide[f"abs_disc0_{var}"] = wide[f"disc0_{var}"].abs()

    return wide


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V5+ suite for KMIA.")
    parser.add_argument(
        "--feature-store",
        default="artifacts/experiments/winners/E37_V5_MINUTE_CONDENSED_V1/feature_store_e37_minute_condensed.parquet",
    )
    parser.add_argument("--mos-config", default="ml/configs/mos_kmia_tmax_v2b_utc06.json")
    parser.add_argument("--db-url", default="mysql+pymysql://root:root@localhost:3306/weather_predictionmarkets")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("artifacts/experiments") / f"V5_PLUS_{utc_now_tag()}"
    ensure_dir(out_dir)

    feature_store_path = Path(args.feature_store)
    df = pd.read_parquet(feature_store_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])

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

    y = pd.to_numeric(df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float)

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

    def train_gate() -> np.ndarray:
        gate_df = base.ensure_columns(df, gate_features)
        gate_X, _ = base.impute_features(gate_df[gate_features], train_mask)
        X_gate = gate_X.to_numpy(dtype=float)
        gate_model = base.train_lgbm_classifier(
            X_gate[train_mask],
            gate_label[train_mask],
            X_gate[val_mask],
            gate_label[val_mask],
            seed=args.seed,
        )
        return gate_model.predict_proba(X_gate)[:, 1]

    def train_experts(expert_features: list[str], *, objective: str = "l1", alpha: float = 0.5) -> np.ndarray:
        base_vals = pd.to_numeric(df.get(base_series), errors="coerce").to_numpy(dtype=float)
        base_mean = float(np.nanmean(y[train_mask]))
        base_vals = np.where(np.isnan(base_vals), base_mean, base_vals)

        X_exp_df = base.ensure_columns(df, expert_features)
        X_exp, _ = base.impute_features(X_exp_df[expert_features], train_mask)
        X = X_exp.to_numpy(dtype=float)

        p_gate = train_gate()

        def fit(mask: np.ndarray):
            if not mask.any():
                return None
            if objective == "quantile":
                return base.train_lgbm_quantile(
                    X[mask],
                    y[mask] - base_vals[mask],
                    X[val_mask & mask],
                    y[val_mask & mask] - base_vals[val_mask & mask],
                    seed=args.seed,
                    alpha=alpha,
                )
            return base.train_lgbm_regressor(
                X[mask],
                y[mask] - base_vals[mask],
                X[val_mask & mask],
                y[val_mask & mask] - base_vals[val_mask & mask],
                seed=args.seed,
            )

        expert_on = fit(gate_label == 1)
        expert_off = fit(gate_label == 0)

        def predict(model):
            if model is None:
                return np.zeros(len(X), dtype=float)
            return model.predict(X)

        resid_on = predict(expert_on)
        resid_off = predict(expert_off)
        pred_all = base_vals + p_gate * resid_on + (1 - p_gate) * resid_off
        return pred_all

    # Baseline V5
    pred_v5 = train_experts(expert_features_base + minute_all)
    p_gate = train_gate()
    base_vals = pd.to_numeric(df.get(base_series), errors="coerce").to_numpy(dtype=float)
    base_mean = float(np.nanmean(y[train_mask]))
    base_vals = np.where(np.isnan(base_vals), base_mean, base_vals)

    # MOS revision features
    with open(args.mos_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    models = [m.upper() for m in cfg.get("models", ["GFS", "NAM"])]
    var_codes = ["tmp", "dpt", "wdr", "wsp", "q12", "p12", "cig", "vis"]
    max_vars = {"p12", "q12", "cig"}

    mos_df = load_mos_rows(
        args.db_url,
        station_id=cfg.get("station_id", "KMIA"),
        models=models,
        var_codes=var_codes,
        start_date=str(df["target_date_local"].min().date()),
        end_date=str(df["target_date_local"].max().date()),
    )
    rev_wide = compute_revision_features(
        mos_df, max_vars=max_vars, models=models, var_codes=var_codes
    )
    df = df.merge(rev_wide, left_on=df["target_date_local"].dt.date, right_on="target_date_local", how="left")
    if "key_0" in df.columns:
        df = df.drop(columns=["key_0"])

    # Derived revision features
    def avg_abs_rev(var: str) -> pd.Series:
        g = df.get(f"abs_rev24_{var}_gfs")
        n = df.get(f"abs_rev24_{var}_nam")
        return np.nanmean(np.vstack([g, n]), axis=0)

    df["abs_rev_tmp_24"] = avg_abs_rev("tmp")
    df["abs_rev_q12_24"] = avg_abs_rev("q12")
    df["abs_rev_cig_24"] = avg_abs_rev("cig")
    df["abs_disc_tmp_0"] = df.get("abs_disc0_tmp")
    df["rev_wdr_24"] = np.nanmean(
        np.vstack([df.get("rev24_wdr_gfs"), df.get("rev24_wdr_nam")]), axis=0
    )

    # Error persistence features (safe: use T-1)
    err = y - pred_v5
    df["err_lag1"] = pd.Series(err).shift(1)
    df["err_ewma7"] = pd.Series(err).shift(1).ewm(span=7, adjust=False).mean()
    df["err_std14"] = pd.Series(err).shift(1).rolling(14).std()

    results: dict[str, dict[str, float]] = {}
    preds: dict[str, np.ndarray] = {"V5": pred_v5}

    def record(name: str, pred: np.ndarray):
        results[name] = {
            "train": _mae(y[train_mask], pred[train_mask]),
            "val": _mae(y[val_mask], pred[val_mask]),
            "test": _mae(y[test_mask], pred[test_mask]),
            "bias": _bias(y[test_mask], pred[test_mask]),
        }
        preds[name] = pred

    record("V5", pred_v5)

    # V5+1 Regime-calibrated bias fix
    suppress = df["MRI_suppress"].to_numpy(dtype=float)
    sup_thresh = np.nanmedian(suppress[val_mask])
    onshore = p_gate >= 0.5
    high_sup = suppress >= sup_thresh
    err_val = pred_v5[val_mask] - y[val_mask]
    regimes = [
        (onshore & high_sup, "onshore_high"),
        (onshore & (~high_sup), "onshore_low"),
        ((~onshore) & high_sup, "offshore_high"),
        ((~onshore) & (~high_sup), "offshore_low"),
    ]
    bias_map = {}
    for mask, key in regimes:
        bias_map[key] = float(np.nanmedian(err_val[mask[val_mask]]))
    bias_adj = np.zeros(len(df), dtype=float)
    for mask, key in regimes:
        bias_adj[mask] = bias_map[key]
    bias_adj = np.clip(bias_adj, -0.5, 0.5)
    pred_v5p1 = pred_v5 - bias_adj
    record("V5+1", pred_v5p1)

    # V5+2 MOS revision add-on corrector
    rev_features = []
    for var in var_codes:
        for model in ["gfs", "nam"]:
            for suf in ["rev12", "rev24", "abs_rev12", "abs_rev24"]:
                col = f"{suf}_{var}_{model}"
                if col in df.columns:
                    rev_features.append(col)
        for suf in ["disc0", "abs_disc0"]:
            col = f"{suf}_{var}"
            if col in df.columns:
                rev_features.append(col)

    add_feats_v5p2 = rev_features + minute_all + [
        "feat_u",
        "feat_v",
        "feat_dd_models",
        "feat_q12_max",
        "feat_cig_min",
    ]
    X2_df = base.ensure_columns(df, add_feats_v5p2)
    X2, _ = base.impute_features(X2_df[add_feats_v5p2], train_mask)
    model_v5p2 = base.train_lgbm_regressor(
        X2[train_mask],
        err[train_mask],
        X2[val_mask],
        err[val_mask],
        seed=args.seed,
    )
    e_hat2 = model_v5p2.predict(X2)
    pred_v5p2 = pred_v5 + np.clip(e_hat2, -2.5, 2.5)
    record("V5+2", pred_v5p2)

    # V5+3 properly wired second-stage corrector
    v5p3_feats = minute_all + [
        "abs_rev_tmp_24",
        "abs_rev_q12_24",
        "abs_rev_cig_24",
        "abs_disc_tmp_0",
        "err_lag1",
        "err_ewma7",
        "err_std14",
    ]
    X3_df = base.ensure_columns(df, v5p3_feats)
    X3, _ = base.impute_features(X3_df[v5p3_feats], train_mask)
    model_v5p3 = base.train_lgbm_regressor(
        X3[train_mask],
        err[train_mask],
        X3[val_mask],
        err[val_mask],
        seed=args.seed,
    )
    e_hat3 = model_v5p3.predict(X3)
    w3 = np.exp(-0.2 * np.abs(df["abs_rev_tmp_24"].to_numpy(dtype=float)))
    pred_v5p3 = pred_v5 + w3 * np.clip(e_hat3, -2.0, 2.0)
    record("V5+3", pred_v5p3)

    # V5+4 uncertainty-weighted shrinkage between base and V5
    better = (np.abs(y - pred_v5) < np.abs(y - base_vals)).astype(int)
    v5p4_feats = [
        "abs_rev_tmp_24",
        "abs_disc_tmp_0",
        "abs_rev_q12_24",
        "std_last180",
        "MRI_suppress",
        "drop_cnt_15_19_t1",
        "feat_dd_models",
    ]
    X4_df = base.ensure_columns(df, v5p4_feats)
    X4, _ = base.impute_features(X4_df[v5p4_feats], train_mask)
    clf_v5p4 = base.train_lgbm_classifier(
        X4[train_mask],
        better[train_mask],
        X4[val_mask],
        better[val_mask],
        seed=args.seed,
    )
    p_better = clf_v5p4.predict_proba(X4)[:, 1]
    pred_v5p4 = p_better * pred_v5 + (1 - p_better) * base_vals
    record("V5+4", pred_v5p4)

    # V5+5 front-day specialist expert
    q80_drop = np.nanquantile(df["night_drop_00_06"].to_numpy(dtype=float)[train_mask], 0.8)
    q70_dd = np.nanquantile(df["feat_dd_models"].to_numpy(dtype=float)[train_mask], 0.7)
    q70_wsp = np.nanquantile(df["feat_wsp_mean"].to_numpy(dtype=float)[train_mask], 0.7)
    front_label = (
        (df["night_drop_00_06"] > q80_drop)
        & (df["feat_dd_models"] > q70_dd)
        & (df["feat_wsp_mean"] > q70_wsp)
    ).astype(int)
    df["t06_minus_tmax_t1"] = df["T06_adj"] - df["iem_tmax_t1"]
    v5p5_gate_feats = [
        "night_drop_00_06",
        "slope_last180",
        "t06_minus_tmax_t1",
        "feat_dd_models",
        "feat_wsp_mean",
        "rev_wdr_24",
    ]
    X5g_df = base.ensure_columns(df, v5p5_gate_feats)
    X5g, _ = base.impute_features(X5g_df[v5p5_gate_feats], train_mask)
    clf_front = base.train_lgbm_classifier(
        X5g[train_mask],
        front_label[train_mask],
        X5g[val_mask],
        front_label[val_mask],
        seed=args.seed,
    )
    p_front = clf_front.predict_proba(X5g)[:, 1]
    # expert on front days
    mask_front = front_label.astype(bool)
    X5e_df = base.ensure_columns(df, v5p5_gate_feats)
    X5e, _ = base.impute_features(X5e_df[v5p5_gate_feats], train_mask)
    model_front = base.train_lgbm_regressor(
        X5e[train_mask & mask_front],
        err[train_mask & mask_front],
        X5e[val_mask & mask_front],
        err[val_mask & mask_front],
        seed=args.seed,
    )
    e_front = model_front.predict(X5e)
    pred_v5p5 = pred_v5 + (p_front > 0.7) * np.clip(e_front, -3.0, 3.0)
    record("V5+5", pred_v5p5)

    # V5+6 overcast/tropical specialist expert
    q70_q12 = np.nanquantile(df["feat_q12_max"].to_numpy(dtype=float)[train_mask], 0.7)
    q30_cig = np.nanquantile(df["feat_cig_min"].to_numpy(dtype=float)[train_mask], 0.3)
    overcast_label = ((df["feat_q12_max"] > q70_q12) & (df["feat_cig_min"] < q30_cig)).astype(int)
    v5p6_feats = [
        "feat_q12_max",
        "feat_p12_max",
        "feat_cig_min",
        "feat_vis_min",
        "abs_rev_q12_24",
        "abs_rev_cig_24",
        "night_drop_00_06",
        "std_last180",
    ]
    X6g_df = base.ensure_columns(df, v5p6_feats)
    X6g, _ = base.impute_features(X6g_df[v5p6_feats], train_mask)
    clf_over = base.train_lgbm_classifier(
        X6g[train_mask],
        overcast_label[train_mask],
        X6g[val_mask],
        overcast_label[val_mask],
        seed=args.seed,
    )
    p_over = clf_over.predict_proba(X6g)[:, 1]
    mask_over = overcast_label.astype(bool)
    model_over = base.train_lgbm_regressor(
        X6g[train_mask & mask_over],
        err[train_mask & mask_over],
        X6g[val_mask & mask_over],
        err[val_mask & mask_over],
        seed=args.seed,
    )
    e_over = model_over.predict(X6g)
    pred_v5p6 = pred_v5 + (p_over > 0.6) * np.clip(e_over, -3.0, 3.0)
    record("V5+6", pred_v5p6)

    # V5+7 analog/cluster residual correction
    cluster_feats = [
        "MRI_suppress",
        "MRI_late",
        "T06_adj",
        "night_drop_00_06",
        "std_last180",
        "feat_dd_models",
        "feat_q12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
    ]
    X7_df = base.ensure_columns(df, cluster_feats)
    X7, _ = base.impute_features(X7_df[cluster_feats], train_mask)
    X7_np = X7.to_numpy(dtype=float)
    mean7 = np.nanmean(X7_np[train_mask], axis=0)
    std7 = np.nanstd(X7_np[train_mask], axis=0) + 1e-6
    X7_std = (X7_np - mean7) / std7
    kmeans = KMeans(n_clusters=16, random_state=args.seed, n_init=10)
    kmeans.fit(X7_std[train_mask])
    cluster_id = kmeans.predict(X7_std)
    df["cluster_id"] = cluster_id
    mu = {}
    n = {}
    for c in range(16):
        mask_c = (cluster_id == c) & train_mask
        mu[c] = float(np.nanmean(err[mask_c])) if mask_c.any() else 0.0
        n[c] = int(mask_c.sum())
    w_c = np.array([n[c] / (n[c] + 50.0) for c in cluster_id])
    mu_c = np.array([mu[c] for c in cluster_id])
    pred_v5p7 = pred_v5 + w_c * mu_c
    record("V5+7", pred_v5p7)

    # V5+8 quantile spread shrinkage
    r10 = train_experts(expert_features_base + minute_all, objective="quantile", alpha=0.1) - base_vals
    r50 = train_experts(expert_features_base + minute_all, objective="quantile", alpha=0.5) - base_vals
    r90 = train_experts(expert_features_base + minute_all, objective="quantile", alpha=0.9) - base_vals
    spread = r90 - r10
    k_grid = [0.0, 0.2, 0.4, 0.6, 0.8]
    best_k = 0.0
    best_mae = 1e9
    for k in k_grid:
        w = np.exp(-k * spread)
        pred = base_vals + w * r50
        mae = _mae(y[val_mask], pred[val_mask])
        if mae < best_mae:
            best_mae = mae
            best_k = k
    w_final = np.exp(-best_k * spread)
    pred_v5p8 = base_vals + w_final * r50
    record("V5+8", pred_v5p8)

    # Build report
    report_lines = []
    report_lines.append("# V5+ Suite Results")
    report_lines.append("")
    report_lines.append(f"Feature store: {feature_store_path}")
    report_lines.append("")
    report_lines.append("| Variant | Train MAE | Val MAE | Test MAE | ΔTest vs V5 | Test Bias |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    v5_test = results["V5"]["test"]
    for name, metrics in results.items():
        delta = metrics["test"] - v5_test
        report_lines.append(
            f"| {name} | {metrics['train']:.4f} | {metrics['val']:.4f} | {metrics['test']:.4f} | {delta:+.4f} | {metrics['bias']:.4f} |"
        )
    report_lines.append("")
    report_lines.append(f"V5+8 best k on val: {best_k:.1f} (val MAE {best_mae:.4f})")

    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    preds_df = pd.DataFrame(
        {
            "target_date_local": df["target_date_local"],
            "y": y,
            "base": base_vals,
            **{k: v for k, v in preds.items()},
        }
    )
    preds_df.to_parquet(out_dir / "preds.parquet", index=False)

    # Save revision features for traceability
    rev_wide.to_parquet(out_dir / "mos_revision_features.parquet", index=False)

    print(f"Wrote report to {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
