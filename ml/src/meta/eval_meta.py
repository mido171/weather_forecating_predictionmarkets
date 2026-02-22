"""Evaluate meta stacking models on validation/test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import common
from weather_ml import config as config_module
from weather_ml import kalshi_tmax_train as kalshi_train
from weather_ml import metrics as metrics_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate meta models on val/test.")
    parser.add_argument("--train-end", required=True, help="Meta-train end date (YYYY-MM-DD).")
    parser.add_argument("--val", required=True, help="Validation range YYYY-MM-DD:YYYY-MM-DD.")
    parser.add_argument("--test", required=True, help="Test range YYYY-MM-DD:YYYY-MM-DD.")
    parser.add_argument("--grib-config", help="Path to EX210 config_resolved.yaml.")
    parser.add_argument("--mos-features", help="Path to MOS features.csv.")
    parser.add_argument("--mos-train-config", help="Path to MOS training config YAML.")
    parser.add_argument("--meta-model-dir", help="Directory containing meta models/metadata.")
    parser.add_argument("--oof-features", help="Optional OOF features CSV for rolling history.")
    parser.add_argument("--quantiles", nargs="*", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[30])
    parser.add_argument("--truth-lag", type=int, default=2)
    parser.add_argument("--output-root", help="Output root under artifacts.")
    return parser


def _parse_range(value: str) -> tuple[datetime, datetime]:
    if ":" not in value:
        raise ValueError("Range must be in START:END format.")
    start_s, end_s = value.split(":", 1)
    return pd.to_datetime(start_s).to_pydatetime(), pd.to_datetime(end_s).to_pydatetime()


def _predict_gribstream_full(
    df: pd.DataFrame,
    *,
    config,
    train_index: pd.Index,
    predict_index: pd.Index,
    truth_lag: int,
    quantiles: list[float],
) -> pd.DataFrame:
    model, feature_cols, _, sigma, features = common.train_gribstream_model(
        df, train_index=train_index, config=config, truth_lag=truth_lag
    )
    X_pred = features.loc[predict_index, feature_cols].to_numpy(dtype=float)
    mu = model.predict(X_pred)
    sigma_vec = np.full_like(mu, sigma, dtype=float)
    q_preds = common._normal_quantiles(mu, sigma_vec, quantiles)
    q_preds = common._enforce_monotonic_quantiles(q_preds)
    out = pd.DataFrame(
        {
            "target_date_local": df.loc[predict_index, "target_date_local"].to_numpy(),
            "asof_utc": df.loc[predict_index, "asof_utc"].to_numpy(),
            "station_id": df.loc[predict_index, "station_id"].to_numpy(),
            "y_true_f": df.loc[predict_index, "actual_tmax_f"].to_numpy(dtype=float),
            "mu_f": mu,
            "sigma_f": sigma_vec,
        }
    )
    for key, values in q_preds.items():
        out[key] = values
    return out


def _predict_mos_full(
    df: pd.DataFrame,
    *,
    train_index: pd.Index,
    predict_index: pd.Index,
    quantiles: list[float],
    lgbm_params: dict,
    quantile_params: dict,
    recency_lambda: float,
) -> pd.DataFrame:
    feature_cols = kalshi_train._select_feature_columns(df)
    baseline, _ = kalshi_train._baseline_series(df)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y_actual_tmax_f"].to_numpy(dtype=float)
    baseline_np = baseline.to_numpy(dtype=float)

    train_mask = df.index.isin(train_index)
    X_train = X[train_mask]
    y_train = y[train_mask]
    baseline_train = baseline_np[train_mask]
    valid_train = np.isfinite(y_train) & np.isfinite(baseline_train)
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    baseline_train = baseline_train[valid_train]

    weights = None
    if recency_lambda > 0:
        train_dates = pd.to_datetime(df.loc[train_mask, "target_date_local"])[valid_train]
        ages = (train_dates - train_dates.min()).dt.days
        weights = np.exp(-recency_lambda * ages.to_numpy(dtype=float))

    model = kalshi_train._fit_lgbm_residual_model(
        X_train,
        y_train,
        baseline_train,
        params=lgbm_params,
        sample_weight=weights,
    )

    pred_mask = df.index.isin(predict_index)
    X_pred = X[pred_mask]
    baseline_pred = baseline_np[pred_mask]
    mu = baseline_pred + model.predict(X_pred)
    mu = np.where(np.isfinite(baseline_pred), mu, np.nan)

    split_preds = kalshi_train._fit_quantile_models_multi(
        X_train=X_train,
        y_train=y_train,
        baseline_train=baseline_train,
        eval_splits={"pred": (X_pred, baseline_pred)},
        params=quantile_params,
        quantiles=quantiles,
        sample_weight=weights,
    )
    q_preds = split_preds.get("pred", {})
    q_preds = common._enforce_monotonic_quantiles(q_preds)
    sigma = None
    if common._quantile_key(quantiles[0]) in q_preds and common._quantile_key(quantiles[-1]) in q_preds:
        q10 = q_preds[common._quantile_key(quantiles[0])]
        q90 = q_preds[common._quantile_key(quantiles[-1])]
        sigma = (q90 - q10) / 2.563
    if sigma is None:
        resid = y_train - (baseline_train + model.predict(X_train))
        sigma = np.full_like(mu, float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0, dtype=float)
    else:
        sigma = np.maximum(sigma, 1e-6)

    out = pd.DataFrame(
        {
            "target_date_local": df.loc[predict_index, "target_date_local"].to_numpy(),
            "asof_utc": df.loc[predict_index, "asof_utc"].to_numpy(),
            "station_id": df.loc[predict_index, "station_id"].to_numpy(),
            "y_true_f": df.loc[predict_index, "y_actual_tmax_f"].to_numpy(dtype=float),
            "mu_f": mu,
            "sigma_f": sigma,
        }
    )
    for key, values in q_preds.items():
        out[key] = values
    return out


def _event_probs(
    preds: dict[str, np.ndarray],
    quantiles: list[float],
    event_specs: list[dict],
) -> dict[str, np.ndarray]:
    q_keys = [common._quantile_key(q) for q in sorted(quantiles)]
    q_vals = np.vstack([preds[k] for k in q_keys])
    probs = {}
    for spec in event_specs:
        name = spec.get("name") or kalshi_train._event_name(spec)
        probs[name] = kalshi_train._event_probability_from_quantiles(q_vals, sorted(quantiles), spec)
    return probs


def _metrics_bundle(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    quantiles: list[float],
    event_specs: list[dict],
    *,
    mu_override: np.ndarray | None = None,
) -> dict:
    point_pred = mu_override if mu_override is not None else preds[common._quantile_key(0.5)]
    point = metrics_module.regression_metrics(y_true, point_pred)
    quantile_metrics = kalshi_train._quantile_metrics(y_true, preds, quantiles=quantiles)
    event_metrics = kalshi_train._probabilistic_event_metrics(
        y_true, preds, quantiles=quantiles, event_specs=event_specs
    )
    reliab = {}
    probs = _event_probs(preds, quantiles, event_specs)
    for name, p in probs.items():
        y_event = metrics_module.event_indicator(y_true, next(spec for spec in event_specs if (spec.get("name") or kalshi_train._event_name(spec)) == name)).astype(int)
        reliab[name] = common.event_reliability(p, y_event)
    return {
        "point": point,
        "quantiles": quantile_metrics,
        "events": event_metrics,
        "reliability": reliab,
    }


def _build_compare_table(
    split_name: str,
    metrics_by_model: dict[str, dict],
    event_specs: list[dict],
) -> list[str]:
    lines = [f"## {split_name.title()}"]
    headers = ["Metric", "Base MOS", "Base Gribstream", "Meta (Quantile Stack)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    def fmt(value: float | None) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "na"
        return f"{value:.4f}"

    def row(label: str, key: str):
        mos = metrics_by_model["mos"]["point"].get(key)
        gs = metrics_by_model["gribstream"]["point"].get(key)
        meta = metrics_by_model["meta_q"]["point"].get(key)
        lines.append(f"| {label} | {fmt(mos)} | {fmt(gs)} | {fmt(meta)} |")

    row("MAE", "mae")
    row("RMSE", "rmse")
    row("Bias", "bias")

    def qrow(label: str, key: str):
        mos = metrics_by_model["mos"]["quantiles"].get(key)
        gs = metrics_by_model["gribstream"]["quantiles"].get(key)
        meta = metrics_by_model["meta_q"]["quantiles"].get(key)
        lines.append(f"| {label} | {fmt(mos)} | {fmt(gs)} | {fmt(meta)} |")

    qrow("CRPS approx", "crps_approx")
    for interval in ("p80", "p90"):
        mos_cov = metrics_by_model["mos"]["quantiles"]["coverage"].get(interval)
        gs_cov = metrics_by_model["gribstream"]["quantiles"]["coverage"].get(interval)
        meta_cov = metrics_by_model["meta_q"]["quantiles"]["coverage"].get(interval)
        lines.append(f"| {interval} coverage | {fmt(mos_cov)} | {fmt(gs_cov)} | {fmt(meta_cov)} |")
        mos_w = metrics_by_model["mos"]["quantiles"]["interval_width"].get(interval)
        gs_w = metrics_by_model["gribstream"]["quantiles"]["interval_width"].get(interval)
        meta_w = metrics_by_model["meta_q"]["quantiles"]["interval_width"].get(interval)
        lines.append(f"| {interval} width | {fmt(mos_w)} | {fmt(gs_w)} | {fmt(meta_w)} |")

    for spec in event_specs:
        name = spec.get("name") or kalshi_train._event_name(spec)
        mos_b = metrics_by_model["mos"]["events"]["brier"].get(name)
        gs_b = metrics_by_model["gribstream"]["events"]["brier"].get(name)
        meta_b = metrics_by_model["meta_q"]["events"]["brier"].get(name)
        lines.append(f"| Brier {name} | {fmt(mos_b)} | {fmt(gs_b)} | {fmt(meta_b)} |")
        mos_l = metrics_by_model["mos"]["events"]["log_loss"].get(name)
        gs_l = metrics_by_model["gribstream"]["events"]["log_loss"].get(name)
        meta_l = metrics_by_model["meta_q"]["events"]["log_loss"].get(name)
        lines.append(f"| LogLoss {name} | {fmt(mos_l)} | {fmt(gs_l)} | {fmt(meta_l)} |")
        mos_e = metrics_by_model["mos"]["reliability"].get(name, {}).get("ece")
        gs_e = metrics_by_model["gribstream"]["reliability"].get(name, {}).get("ece")
        meta_e = metrics_by_model["meta_q"]["reliability"].get(name, {}).get("ece")
        lines.append(f"| ECE {name} | {fmt(mos_e)} | {fmt(gs_e)} | {fmt(meta_e)} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = common.resolve_repo_root()

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
    meta_model_dir = Path(args.meta_model_dir) if args.meta_model_dir else None
    if meta_model_dir is None:
        # Fall back to latest meta_stack_eval dir
        root = repo_root / "artifacts" / "meta_stack_eval"
        if not root.exists():
            raise FileNotFoundError("meta_stack_eval directory not found.")
        candidates = sorted(root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("No meta_stack_eval runs found.")
        meta_model_dir = candidates[0]

    oof_path = Path(args.oof_features) if args.oof_features else meta_model_dir / "meta_features_train_oof.csv"
    use_oof = oof_path.exists()

    grib_cfg = config_module.load_config(grib_config_path)
    grib_cfg = config_module.resolve_paths(grib_cfg, repo_root=repo_root)
    mos_train_cfg = common.load_yaml(mos_train_cfg_path)

    grib_df = common.load_gribstream_df(Path(grib_cfg.data.csv_path))
    mos_df = common.load_mos_df(mos_features_path)

    train_end = common.parse_date(args.train_end)
    val_start, val_end = _parse_range(args.val)
    test_start, test_end = _parse_range(args.test)

    grib_train_idx = grib_df.index[grib_df["target_date_local"] <= train_end]
    mos_train_idx = mos_df.index[mos_df["target_date_local"] <= train_end]

    val_mask_grib = (grib_df["target_date_local"] >= val_start.date()) & (
        grib_df["target_date_local"] <= val_end.date()
    )
    test_mask_grib = (grib_df["target_date_local"] >= test_start.date()) & (
        grib_df["target_date_local"] <= test_end.date()
    )
    val_mask_mos = (mos_df["target_date_local"] >= val_start.date()) & (
        mos_df["target_date_local"] <= val_end.date()
    )
    test_mask_mos = (mos_df["target_date_local"] >= test_start.date()) & (
        mos_df["target_date_local"] <= test_end.date()
    )

    quantiles = [float(q) for q in args.quantiles]
    gs_val = _predict_gribstream_full(
        grib_df,
        config=grib_cfg,
        train_index=grib_train_idx,
        predict_index=grib_df.index[val_mask_grib],
        truth_lag=int(args.truth_lag),
        quantiles=quantiles,
    )
    gs_test = _predict_gribstream_full(
        grib_df,
        config=grib_cfg,
        train_index=grib_train_idx,
        predict_index=grib_df.index[test_mask_grib],
        truth_lag=int(args.truth_lag),
        quantiles=quantiles,
    )

    mos_val = _predict_mos_full(
        mos_df,
        train_index=mos_train_idx,
        predict_index=mos_df.index[val_mask_mos],
        quantiles=quantiles,
        lgbm_params=mos_train_cfg.get("lgbm_params", {}),
        quantile_params=mos_train_cfg.get("quantile_params", {}),
        recency_lambda=float(mos_train_cfg.get("recency_lambda", 0.0)),
    )
    mos_test = _predict_mos_full(
        mos_df,
        train_index=mos_train_idx,
        predict_index=mos_df.index[test_mask_mos],
        quantiles=quantiles,
        lgbm_params=mos_train_cfg.get("lgbm_params", {}),
        quantile_params=mos_train_cfg.get("quantile_params", {}),
        recency_lambda=float(mos_train_cfg.get("recency_lambda", 0.0)),
    )

    merged_val = common.merge_base_predictions(gs_val, mos_val)
    merged_test = common.merge_base_predictions(gs_test, mos_test)

    if use_oof:
        oof_df = pd.read_csv(oof_path)
        oof_df["target_date_local"] = pd.to_datetime(oof_df["target_date_local"]).dt.date
        oof_df["asof_utc"] = pd.to_datetime(oof_df["asof_utc"], utc=True)
        base_cols = [
            "target_date_local",
            "asof_utc",
            "mu_f_gs",
            "mu_f_mos",
            "sigma_f_gs",
            "sigma_f_mos",
            "q10_gs",
            "q50_gs",
            "q90_gs",
            "q10_mos",
            "q50_mos",
            "q90_mos",
            "y_true_f",
        ]
        oof_base = oof_df[base_cols].copy()
        oof_base = oof_base.rename(columns={"y_true_f": "y_true_f_gs"})
        combined = pd.concat(
            [
                oof_base,
                merged_val,
                merged_test,
            ],
            ignore_index=True,
        )
        combined = common.add_meta_features(
            combined,
            windows=[int(w) for w in args.rolling_windows],
            lag_days=int(args.truth_lag),
        )
        val_features = combined.loc[combined["target_date_local"].isin(merged_val["target_date_local"])]
        test_features = combined.loc[combined["target_date_local"].isin(merged_test["target_date_local"])]
    else:
        val_features = common.add_meta_features(
            merged_val,
            windows=[int(w) for w in args.rolling_windows],
            lag_days=int(args.truth_lag),
        )
        test_features = common.add_meta_features(
            merged_test,
            windows=[int(w) for w in args.rolling_windows],
            lag_days=int(args.truth_lag),
        )

    meta_meta = common.load_yaml(meta_model_dir / "meta_model_metadata.json")
    feature_cols = meta_meta.get("feature_columns", [])
    sigma_cols = meta_meta.get("sigma_feature_columns", [])
    medians = meta_meta.get("feature_medians", {})
    sigma_medians = meta_meta.get("sigma_feature_medians", {})

    def _impute(df: pd.DataFrame, cols: list[str], med: dict[str, float]) -> np.ndarray:
        values = df[cols].to_numpy(dtype=float)
        meds = np.array([med.get(col, 0.0) for col in cols], dtype=float)
        return np.where(np.isnan(values), meds, values)

    X_val = _impute(val_features, feature_cols, medians)
    X_test = _impute(test_features, feature_cols, medians)
    X_sigma_val = _impute(val_features, sigma_cols, sigma_medians)
    X_sigma_test = _impute(test_features, sigma_cols, sigma_medians)

    point_model = joblib.load(meta_model_dir / "meta_model_point.pkl")
    sigma_model = joblib.load(meta_model_dir / "meta_model_sigma.pkl")

    quantile_dir = meta_model_dir / "meta_models_quantiles"
    quantile_models = {}
    for q in quantiles:
        key = common._quantile_key(q)
        quantile_models[key] = joblib.load(quantile_dir / f"{key}.pkl")

    def _predict_quantile_stack(X: np.ndarray) -> dict[str, np.ndarray]:
        preds = {key: model.predict(X) for key, model in quantile_models.items()}
        return common._enforce_monotonic_quantiles(preds)

    pred_q_val = _predict_quantile_stack(X_val)
    pred_q_test = _predict_quantile_stack(X_test)

    mu_val = point_model.predict(X_val)
    mu_test = point_model.predict(X_test)
    sigma_val = np.expm1(sigma_model.predict(X_sigma_val))
    sigma_test = np.expm1(sigma_model.predict(X_sigma_test))
    sigma_val = np.maximum(sigma_val, 0.25)
    sigma_test = np.maximum(sigma_test, 0.25)
    pred_mu_sigma_val = common._normal_quantiles(mu_val, sigma_val, quantiles)
    pred_mu_sigma_test = common._normal_quantiles(mu_test, sigma_test, quantiles)

    event_specs = mos_train_cfg.get("probability_events", common.EVENT_SPECS_DEFAULT)

    def _bundle(
        df: pd.DataFrame,
        preds: dict[str, np.ndarray],
        *,
        mu_override: np.ndarray | None = None,
    ) -> dict:
        y_true = df["y_true_f"].to_numpy(dtype=float)
        return _metrics_bundle(y_true, preds, quantiles, event_specs, mu_override=mu_override)

    metrics_val = {
        "mos": _bundle(
            val_features,
            {
                common._quantile_key(0.1): val_features["q10_mos"].to_numpy(dtype=float),
                common._quantile_key(0.5): val_features["q50_mos"].to_numpy(dtype=float),
                common._quantile_key(0.9): val_features["q90_mos"].to_numpy(dtype=float),
            },
            mu_override=val_features["mu_mos"].to_numpy(dtype=float),
        ),
        "gribstream": _bundle(
            val_features,
            {
                common._quantile_key(0.1): val_features["q10_gs"].to_numpy(dtype=float),
                common._quantile_key(0.5): val_features["q50_gs"].to_numpy(dtype=float),
                common._quantile_key(0.9): val_features["q90_gs"].to_numpy(dtype=float),
            },
            mu_override=val_features["mu_gs"].to_numpy(dtype=float),
        ),
        "meta_q": _bundle(val_features, pred_q_val),
        "meta_mu_sigma": _bundle(val_features, pred_mu_sigma_val, mu_override=mu_val),
    }
    metrics_test = {
        "mos": _bundle(
            test_features,
            {
                common._quantile_key(0.1): test_features["q10_mos"].to_numpy(dtype=float),
                common._quantile_key(0.5): test_features["q50_mos"].to_numpy(dtype=float),
                common._quantile_key(0.9): test_features["q90_mos"].to_numpy(dtype=float),
            },
            mu_override=test_features["mu_mos"].to_numpy(dtype=float),
        ),
        "gribstream": _bundle(
            test_features,
            {
                common._quantile_key(0.1): test_features["q10_gs"].to_numpy(dtype=float),
                common._quantile_key(0.5): test_features["q50_gs"].to_numpy(dtype=float),
                common._quantile_key(0.9): test_features["q90_gs"].to_numpy(dtype=float),
            },
            mu_override=test_features["mu_gs"].to_numpy(dtype=float),
        ),
        "meta_q": _bundle(test_features, pred_q_test),
        "meta_mu_sigma": _bundle(test_features, pred_mu_sigma_test, mu_override=mu_test),
    }

    output_root = (
        Path(args.output_root)
        if args.output_root
        else repo_root / "artifacts" / "meta_stack_eval"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    common.write_json(
        run_dir / "metrics_meta.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "val": metrics_val,
            "test": metrics_test,
            "meta_model_dir": str(meta_model_dir),
        },
    )

    pred_val_df = pd.DataFrame(
        {
            "target_date_local": val_features["target_date_local"],
            "asof_utc": val_features["asof_utc"],
            "y_true_f": val_features["y_true_f"],
            "mu_meta": mu_val,
            "q10_meta": pred_q_val[common._quantile_key(0.1)],
            "q50_meta": pred_q_val[common._quantile_key(0.5)],
            "q90_meta": pred_q_val[common._quantile_key(0.9)],
        }
    )
    pred_test_df = pd.DataFrame(
        {
            "target_date_local": test_features["target_date_local"],
            "asof_utc": test_features["asof_utc"],
            "y_true_f": test_features["y_true_f"],
            "mu_meta": mu_test,
            "q10_meta": pred_q_test[common._quantile_key(0.1)],
            "q50_meta": pred_q_test[common._quantile_key(0.5)],
            "q90_meta": pred_q_test[common._quantile_key(0.9)],
        }
    )
    pred_val_df.to_csv(run_dir / "predictions_meta_val.csv", index=False)
    pred_test_df.to_csv(run_dir / "predictions_meta_test.csv", index=False)

    report_lines = ["# Meta Stack Comparison", ""]
    report_lines += _build_compare_table("validation", metrics_val, event_specs)
    report_lines.append("")
    report_lines += _build_compare_table("test", metrics_test, event_specs)
    common.write_markdown(run_dir / "compare_report.md", report_lines)

    print(f"Meta evaluation written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
