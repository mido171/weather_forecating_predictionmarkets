"""Godlike MOS experiment suite (G01-G15) for KMIA Tmax next-day."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from weather_ml import metrics


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_suite_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"station_id": "string"})
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])
    return df


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = np.nan
    if missing:
        LOGGER.warning("Added missing columns with NaN: %s", missing)
    return df


def split_by_date(
    df: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> dict:
    date_series = pd.to_datetime(df["target_date_local"])
    train_mask = (date_series >= train_start) & (date_series <= train_end)
    val_mask = (date_series >= val_start) & (date_series <= val_end)
    test_mask = (date_series >= test_start) & (date_series <= test_end)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Split masks are empty; adjust date ranges.")
    return {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_mask": train_mask.to_numpy(),
        "val_mask": val_mask.to_numpy(),
        "test_mask": test_mask.to_numpy(),
    }


def impute_features(features: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    cleaned = features.replace([np.inf, -np.inf], np.nan)
    train_means = cleaned.loc[train_mask].mean(axis=0, skipna=True)
    train_means = train_means.fillna(0.0)
    filled = cleaned.fillna(train_means)
    meta = {"method": "train_mean", "fill_values": train_means.to_dict()}
    return filled, meta


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {}
    return metrics.regression_metrics(y_true, y_pred)


def cols_by_prefix(df: pd.DataFrame, prefixes: list[str], *, exclude_prefixes: list[str] | None = None) -> list[str]:
    exclude_prefixes = exclude_prefixes or []
    cols: list[str] = []
    for col in df.columns:
        if any(col.startswith(ex) for ex in exclude_prefixes):
            continue
        if any(col.startswith(pref) for pref in prefixes):
            cols.append(col)
    return cols


def mos_cols_for_vars(
    df: pd.DataFrame,
    models: list[str],
    variables: list[str],
    stats: list[str],
) -> list[str]:
    cols = []
    for model in models:
        for var in variables:
            for stat in stats:
                col = f"mos_{model}_{var}_{stat}"
                if col in df.columns:
                    cols.append(col)
    return cols


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tmp_mean = df.get("mos_xmodel_blend_tmp_mean")
    tmp_max = df.get("mos_xmodel_blend_tmp_max")
    dpt_mean = df.get("mos_xmodel_blend_dpt_mean")
    wsp_mean = df.get("mos_xmodel_blend_wsp_mean")
    wdr_mean = df.get("mos_xmodel_blend_wdr_mean")

    if tmp_mean is not None and dpt_mean is not None:
        df["deriv_dd_mean"] = tmp_mean - dpt_mean
        df["deriv_dd_x_dpt"] = df["deriv_dd_mean"] * dpt_mean
    else:
        df["deriv_dd_mean"] = np.nan
        df["deriv_dd_x_dpt"] = np.nan

    if tmp_max is not None and dpt_mean is not None:
        df["deriv_dd_max"] = tmp_max - dpt_mean
    else:
        df["deriv_dd_max"] = np.nan

    if wdr_mean is not None:
        wdr = pd.to_numeric(wdr_mean, errors="coerce")
        wdr_deg = np.where(wdr <= 36.0, wdr * 10.0, wdr)
        radians = np.deg2rad(wdr_deg)
        df["deriv_wdr_sin"] = np.sin(radians)
        df["deriv_wdr_cos"] = np.cos(radians)
        df["deriv_onshore"] = ((wdr_deg >= 45.0) & (wdr_deg <= 160.0)).astype(float)
        df["deriv_offshore"] = (
            (wdr_deg >= 225.0) | (wdr_deg <= 45.0)
        ).astype(float)
    else:
        df["deriv_wdr_sin"] = np.nan
        df["deriv_wdr_cos"] = np.nan
        df["deriv_onshore"] = np.nan
        df["deriv_offshore"] = np.nan

    if wsp_mean is not None:
        df["deriv_u"] = wsp_mean * df.get("deriv_wdr_cos", np.nan)
        df["deriv_v"] = wsp_mean * df.get("deriv_wdr_sin", np.nan)
        df["deriv_onshore_wsp"] = wsp_mean * df.get("deriv_onshore", np.nan)
    else:
        df["deriv_u"] = np.nan
        df["deriv_v"] = np.nan
        df["deriv_onshore_wsp"] = np.nan

    cig = df.get("mos_xmodel_blend_cig_median")
    if cig is not None:
        df["deriv_cig_low"] = (cig <= 3).astype(float)
    else:
        df["deriv_cig_low"] = np.nan

    vis = df.get("mos_xmodel_blend_vis_mean")
    if vis is not None:
        df["deriv_vis_low"] = (vis <= 3).astype(float)
    else:
        df["deriv_vis_low"] = np.nan

    p12 = df.get("mos_xmodel_blend_p12_max")
    if p12 is not None:
        df["deriv_pop12_hi"] = (p12 >= 60).astype(float)
    else:
        df["deriv_pop12_hi"] = np.nan

    q12 = df.get("mos_xmodel_blend_q12_mean")
    if q12 is not None:
        df["deriv_q12_hi"] = (q12 >= 3).astype(float)
    else:
        df["deriv_q12_hi"] = np.nan

    t06 = df.get("mos_xmodel_blend_t06_mean")
    if t06 is not None:
        df["deriv_t06_hi"] = (t06 >= 50).astype(float)
    else:
        df["deriv_t06_hi"] = np.nan

    return df

def train_lgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    params: dict[str, Any] | None = None,
) -> Any:
    import lightgbm as lgb

    base_params = {
        "objective": "regression_l1",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 2000,
        "random_state": seed,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
        "verbose": -1,
    }
    if params:
        base_params.update(params)
    model = lgb.LGBMRegressor(**base_params)
    if len(y_val):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def train_lgbm_quantile(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    alpha: float,
) -> Any:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=1500,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    if len(y_val):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def train_xgb_quantile(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    alpha: float,
) -> Any:
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=800,
        tree_method="hist",
        random_state=seed,
        nthread=1,
        verbosity=0,
    )
    if len(y_val):
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def train_extratrees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features=0.6,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def predict_extratrees_quantiles(
    model: ExtraTreesRegressor,
    X: np.ndarray,
    quantiles: list[float],
) -> dict[float, np.ndarray]:
    preds = np.stack([tree.predict(X) for tree in model.estimators_], axis=1)
    out: dict[float, np.ndarray] = {}
    for q in quantiles:
        out[q] = np.quantile(preds, q, axis=1)
    return out


def save_predictions(path: Path, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    out = pd.DataFrame(
        {
            "target_date_local": df["target_date_local"].values,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    out.to_csv(path, index=False)

@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    feature_cols: list[str]
    metrics: dict[str, Any]
    extras: dict[str, Any]
    pred_train: np.ndarray | None = None
    pred_val: np.ndarray | None = None
    pred_test: np.ndarray | None = None


@dataclass
class ExperimentContext:
    df: pd.DataFrame
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    seed: int
    split_info: dict
    cache: dict[str, ExperimentResult]


def run_point_model(
    ctx: ExperimentContext,
    feature_cols: list[str],
    *,
    model_params: dict[str, Any] | None = None,
    residual_base: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    df = ctx.df
    y = df["y_actual_tmax_f"].to_numpy(dtype=float)
    features = df[feature_cols]
    features, impute_meta = impute_features(features, ctx.train_mask)
    X = features.to_numpy(dtype=float)
    X_train = X[ctx.train_mask]
    X_val = X[ctx.val_mask]
    X_test = X[ctx.test_mask]
    y_train = y[ctx.train_mask]
    y_val = y[ctx.val_mask]
    y_test = y[ctx.test_mask]

    if residual_base:
        base_series = pd.to_numeric(df[residual_base], errors="coerce")
        fallback = pd.to_numeric(df.get("mos_xmodel_blend_tmp_max"), errors="coerce")
        if fallback is not None:
            base_series = base_series.where(~base_series.isna(), fallback)
        base_mean = float(np.nanmean(base_series.loc[ctx.train_mask]))
        base_series = base_series.fillna(base_mean)
        base = base_series.to_numpy(dtype=float)
        base_train = base[ctx.train_mask]
        base_val = base[ctx.val_mask]
        base_test = base[ctx.test_mask]
        y_train_resid = y_train - base_train
        y_val_resid = y_val - base_val
        model = train_lgbm_regressor(X_train, y_train_resid, X_val, y_val_resid, seed=ctx.seed, params=model_params)
        pred_train = base_train + model.predict(X_train)
        pred_val = base_val + model.predict(X_val)
        pred_test = base_test + model.predict(X_test)
    else:
        model = train_lgbm_regressor(X_train, y_train, X_val, y_val, seed=ctx.seed, params=model_params)
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)

    meta = {"impute": impute_meta, "model_params": model_params or {}, "residual_base": residual_base}
    return pred_train, pred_val, pred_test, meta


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    models = ["gfs", "nam"]
    variables = [
        "cig",
        "dpt",
        "n_x",
        "p06",
        "p12",
        "q06",
        "q12",
        "t06",
        "t06_1",
        "t06_2",
        "t12",
        "t12_1",
        "t12_2",
        "tmp",
        "vis",
        "wdr",
        "wsp",
    ]
    mos_stats = ["min", "max", "mean", "median"]
    core_temp = mos_cols_for_vars(df, models, ["tmp", "n_x"], ["max", "mean"])
    all_mos = mos_cols_for_vars(df, models, variables, mos_stats)
    mos_shape = cols_by_prefix(df, ["mos_shape_"])
    mos_xmodel = cols_by_prefix(df, ["mos_xmodel_"])
    bias_cols = cols_by_prefix(df, ["bias_"])
    obs_cols = cols_by_prefix(df, ["obs_"])
    cal_cols = cols_by_prefix(df, ["cal_"])
    knn_cols = cols_by_prefix(df, ["knn_"], exclude_prefixes=["knn_v0_nn"])
    rev_cols = [c for c in df.columns if c.startswith("mos_rev_") or c.endswith("_b24") or c.endswith("_b48") or c.endswith("_update_count")]
    deriv_cols = cols_by_prefix(df, ["deriv_"])

    return {
        "core_temp": core_temp,
        "all_mos": all_mos,
        "mos_shape": mos_shape,
        "mos_xmodel": mos_xmodel,
        "bias": bias_cols,
        "obs": obs_cols,
        "cal": cal_cols,
        "knn": knn_cols,
        "rev": rev_cols,
        "deriv": deriv_cols,
    }


def compute_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gfs_mae = df.get("bias_gfs_mae_30")
    nam_mae = df.get("bias_nam_mae_30")
    if gfs_mae is None or nam_mae is None:
        df["skill_w_gfs"] = np.nan
        df["skill_w_nam"] = np.nan
        df["skill_tmp"] = np.nan
        df["skill_spread"] = np.nan
        df["skill_entropy"] = np.nan
        return df
    w_gfs = 1.0 / (gfs_mae + 0.25)
    w_nam = 1.0 / (nam_mae + 0.25)
    w_sum = w_gfs + w_nam
    w_gfs = w_gfs / w_sum
    w_nam = w_nam / w_sum
    df["skill_w_gfs"] = w_gfs
    df["skill_w_nam"] = w_nam
    gfs = pd.to_numeric(df.get("mos_gfs_n_x_max"), errors="coerce")
    nam = pd.to_numeric(df.get("mos_nam_n_x_max"), errors="coerce")
    df["skill_tmp"] = w_gfs * gfs + w_nam * nam
    df["skill_spread"] = (gfs - nam).abs()
    df["skill_entropy"] = -(w_gfs * np.log(w_gfs + 1e-9) + w_nam * np.log(w_nam + 1e-9))
    return df


def compute_suppression_label(df: pd.DataFrame) -> pd.Series:
    p12 = df.get("mos_xmodel_blend_p12_max")
    t06 = df.get("mos_xmodel_blend_t06_mean")
    cig = df.get("mos_xmodel_blend_cig_median")
    vis = df.get("mos_xmodel_blend_vis_mean")
    parts = []
    if p12 is not None:
        parts.append((p12 >= 60).astype(float))
    if t06 is not None:
        parts.append((t06 >= 50).astype(float))
    if cig is not None:
        parts.append((cig <= 3).astype(float))
    if vis is not None:
        parts.append((vis <= 3).astype(float))
    if not parts:
        return pd.Series(np.zeros(len(df)), index=df.index)
    score = np.sum(parts, axis=0)
    if not isinstance(score, pd.Series):
        score = pd.Series(score, index=df.index)
    return (score >= 2).astype(int)

def run_experiments(ctx: ExperimentContext, output_root: Path) -> list[ExperimentResult]:
    df = ctx.df
    sets = build_feature_sets(df)

    results: list[ExperimentResult] = []

    def record_result(exp_id: str, name: str, feature_cols: list[str], preds: tuple[np.ndarray, np.ndarray, np.ndarray], meta: dict, extra: dict | None = None):
        y = df["y_actual_tmax_f"].to_numpy(dtype=float)
        y_train = y[ctx.train_mask]
        y_val = y[ctx.val_mask]
        y_test = y[ctx.test_mask]
        pred_train, pred_val, pred_test = preds
        base_fallback = pd.to_numeric(df.get("base_tmax_blend"), errors="coerce")
        if base_fallback is None:
            base_fallback = pd.Series(np.nan, index=df.index)
        base_vals = base_fallback.to_numpy(dtype=float)
        train_mean = float(np.nanmean(y_train))

        def fill_nan(pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
            filled = pred.copy()
            if np.isnan(filled).any():
                fb = base_vals[mask]
                filled = np.where(np.isnan(filled), fb, filled)
                filled = np.where(np.isnan(filled), train_mean, filled)
            return filled

        pred_train = fill_nan(pred_train, ctx.train_mask)
        pred_val = fill_nan(pred_val, ctx.val_mask)
        pred_test = fill_nan(pred_test, ctx.test_mask)
        metrics_payload = {
            "train": regression_metrics(y_train, pred_train),
            "validation": regression_metrics(y_val, pred_val),
            "test": regression_metrics(y_test, pred_test),
        }
        extras = {"meta": meta}
        if extra:
            extras.update(extra)
        result = ExperimentResult(
            experiment_id=exp_id,
            name=name,
            feature_cols=feature_cols,
            metrics=metrics_payload,
            extras=extras,
            pred_train=pred_train,
            pred_val=pred_val,
            pred_test=pred_test,
        )
        results.append(result)

        run_dir = output_root / exp_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "experiment_meta.json", {"experiment_id": exp_id, "name": name, "features": feature_cols})
        write_json(run_dir / "metrics.json", {"metrics": metrics_payload, "extras": extras})
        save_predictions(run_dir / "predictions_test.csv", df.loc[ctx.test_mask], y_test, pred_test)
        (run_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

        ctx.cache[exp_id] = result

    # G01 baseline: core temp + calendar
    g01_features = sets["core_temp"] + sets["mos_xmodel"] + sets["cal"]
    g01_features = sorted(set(g01_features))
    pred_train, pred_val, pred_test, meta = run_point_model(ctx, g01_features)
    record_result("G01", "LGBM Core MOS baseline", g01_features, (pred_train, pred_val, pred_test), meta)

    # G02 residual model on broad MOS + derived
    g02_features = sorted(set(sets["all_mos"] + sets["mos_shape"] + sets["mos_xmodel"] + sets["deriv"] + sets["cal"]))
    pred_train, pred_val, pred_test, meta = run_point_model(ctx, g02_features, residual_base="base_tmax_blend")
    record_result("G02", "Residual MOS full features", g02_features, (pred_train, pred_val, pred_test), meta)

    # G03 rolling bias features
    g03_features = sorted(set(sets["bias"] + sets["mos_xmodel"] + sets["deriv"] + sets["cal"] + ["base_tmax_blend"]))
    pred_train, pred_val, pred_test, meta = run_point_model(ctx, g03_features)
    record_result("G03", "Bias-aware MOS model", g03_features, (pred_train, pred_val, pred_test), meta)

    # G04 skill-weighted ensemble features
    df_skill = compute_skill_features(df)
    ctx.df = df_skill
    skill_cols = ["skill_w_gfs", "skill_w_nam", "skill_tmp", "skill_spread", "skill_entropy"]
    g04_features = sorted(set(skill_cols + sets["mos_xmodel"] + sets["deriv"] + sets["cal"]))
    pred_train, pred_val, pred_test, meta = run_point_model(ctx, g04_features)
    record_result("G04", "Skill-weighted ensemble model", g04_features, (pred_train, pred_val, pred_test), meta)
    ctx.df = df

    # G05 regime-conditioned bias (gates as features)
    regime_cols = ["deriv_onshore", "deriv_offshore", "deriv_pop12_hi", "deriv_t06_hi", "deriv_cig_low", "deriv_vis_low"]
    g05_features = sorted(set(sets["bias"] + sets["mos_xmodel"] + sets["deriv"] + regime_cols + sets["cal"] + ["base_tmax_blend"]))
    pred_train, pred_val, pred_test, meta = run_point_model(ctx, g05_features)
    record_result("G05", "Regime-conditioned bias model", g05_features, (pred_train, pred_val, pred_test), meta)

    # G06 revision dynamics
    g06_features = sorted(set(sets["rev"] + sets["mos_xmodel"] + sets["deriv"] + sets["cal"] + ["base_tmax_blend"]))
    pred_train, pred_val, pred_test, meta = run_point_model(ctx, g06_features)
    record_result("G06", "Revision dynamics model", g06_features, (pred_train, pred_val, pred_test), meta)

    # G07 heteroscedastic shrinkage
    g07_features = sorted(set(sets["all_mos"] + sets["mos_xmodel"] + sets["deriv"] + sets["cal"]))
    pred_train, pred_val, pred_test, meta = run_point_model(ctx, g07_features, residual_base="base_tmax_blend")
    y = df["y_actual_tmax_f"].to_numpy(dtype=float)
    resid_train = np.abs(y[ctx.train_mask] - pred_train)
    resid_val = np.abs(y[ctx.val_mask] - pred_val)
    resid_test = np.abs(y[ctx.test_mask] - pred_test)
    sigma_features = sorted(set(sets["mos_xmodel"] + sets["deriv"] + sets["cal"]))
    X_sigma, _ = impute_features(df[sigma_features], ctx.train_mask)
    X_sigma = X_sigma.to_numpy(dtype=float)
    X_train = X_sigma[ctx.train_mask]
    X_val = X_sigma[ctx.val_mask]
    X_test = X_sigma[ctx.test_mask]
    sigma_model = train_lgbm_regressor(X_train, resid_train, X_val, resid_val, seed=ctx.seed, params={"objective": "regression_l2"})
    sigma_train = sigma_model.predict(X_train)
    sigma_val = sigma_model.predict(X_val)
    sigma_test = sigma_model.predict(X_test)
    sigma_ref = np.nanpercentile(sigma_train, 90)
    def shrink(pred, sigma, climo):
        alpha = 1.0 - np.clip(sigma / (sigma_ref + 1e-6), 0.0, 0.8)
        return alpha * pred + (1.0 - alpha) * climo
    climo = df.get("obs_climo_d_mean", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    climo_alt = df.get("obs_climo_mean", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    base_blend = df.get("base_tmax_blend", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    climo = np.where(np.isnan(climo), climo_alt, climo)
    climo = np.where(np.isnan(climo), base_blend, climo)
    climo = np.where(np.isnan(climo), np.nanmean(y[ctx.train_mask]), climo)
    pred_train_shrink = shrink(pred_train, sigma_train, climo[ctx.train_mask])
    pred_val_shrink = shrink(pred_val, sigma_val, climo[ctx.val_mask])
    pred_test_shrink = shrink(pred_test, sigma_test, climo[ctx.test_mask])
    record_result(
        "G07",
        "Heteroscedastic shrinkage model",
        g07_features,
        (pred_train_shrink, pred_val_shrink, pred_test_shrink),
        meta,
        extra={"sigma_ref_p90": float(sigma_ref)},
    )

    # G08 LGBM quantile suite (0.1/0.5/0.9)
    g08_features = sorted(set(sets["all_mos"] + sets["mos_xmodel"] + sets["deriv"] + sets["cal"] + sets["bias"]))
    features, impute_meta = impute_features(df[g08_features], ctx.train_mask)
    X = features.to_numpy(dtype=float)
    y = df["y_actual_tmax_f"].to_numpy(dtype=float)
    X_train = X[ctx.train_mask]
    X_val = X[ctx.val_mask]
    X_test = X[ctx.test_mask]
    y_train = y[ctx.train_mask]
    y_val = y[ctx.val_mask]
    y_test = y[ctx.test_mask]
    q10 = train_lgbm_quantile(X_train, y_train, X_val, y_val, seed=ctx.seed, alpha=0.1)
    q50 = train_lgbm_quantile(X_train, y_train, X_val, y_val, seed=ctx.seed, alpha=0.5)
    q90 = train_lgbm_quantile(X_train, y_train, X_val, y_val, seed=ctx.seed, alpha=0.9)
    pred_train = q50.predict(X_train)
    pred_val = q50.predict(X_val)
    pred_test = q50.predict(X_test)
    q10_test = q10.predict(X_test)
    q90_test = q90.predict(X_test)
    coverage_80 = float(np.mean((y_test >= q10_test) & (y_test <= q90_test)))
    meta = {"impute": impute_meta, "quantiles": [0.1, 0.5, 0.9]}
    record_result(
        "G08",
        "LGBM quantile suite",
        g08_features,
        (pred_train, pred_val, pred_test),
        meta,
        extra={"coverage_80_test": coverage_80},
    )

    # G09 ExtraTrees quantile forest
    g09_features = g08_features
    features, impute_meta = impute_features(df[g09_features], ctx.train_mask)
    X = features.to_numpy(dtype=float)
    X_train = X[ctx.train_mask]
    X_val = X[ctx.val_mask]
    X_test = X[ctx.test_mask]
    y = df["y_actual_tmax_f"].to_numpy(dtype=float)
    y_train = y[ctx.train_mask]
    y_val = y[ctx.val_mask]
    y_test = y[ctx.test_mask]
    et = train_extratrees(X_train, y_train, seed=ctx.seed)
    pred_train = et.predict(X_train)
    pred_val = et.predict(X_val)
    pred_test = et.predict(X_test)
    q = predict_extratrees_quantiles(et, X_test, [0.1, 0.9])
    coverage_80 = float(np.mean((y_test >= q[0.1]) & (y_test <= q[0.9])))
    record_result(
        "G09",
        "ExtraTrees quantile forest",
        g09_features,
        (pred_train, pred_val, pred_test),
        {"impute": impute_meta},
        extra={"coverage_80_test": coverage_80},
    )

    # G10 XGBoost quantile (median)
    g10_features = g08_features
    features, impute_meta = impute_features(df[g10_features], ctx.train_mask)
    X = features.to_numpy(dtype=float)
    X_train = X[ctx.train_mask]
    X_val = X[ctx.val_mask]
    X_test = X[ctx.test_mask]
    y = df["y_actual_tmax_f"].to_numpy(dtype=float)
    y_train = y[ctx.train_mask]
    y_val = y[ctx.val_mask]
    y_test = y[ctx.test_mask]
    xgb_q50 = train_xgb_quantile(X_train, y_train, X_val, y_val, seed=ctx.seed, alpha=0.5)
    pred_train = xgb_q50.predict(X_train)
    pred_val = xgb_q50.predict(X_val)
    pred_test = xgb_q50.predict(X_test)
    record_result("G10", "XGBoost quantile median", g10_features, (pred_train, pred_val, pred_test), {"impute": impute_meta})

    # G11 Conformalized intervals from G08
    g08 = ctx.cache.get("G08")
    if g08:
        q10_pred = q10.predict(X_test)
        q90_pred = q90.predict(X_test)
        # Use validation set for calibration.
        q10_val = q10.predict(X_val)
        q90_val = q90.predict(X_val)
        calib_scores = np.maximum(np.maximum(q10_val - y_val, y_val - q90_val), 0.0)
        s_star = float(np.quantile(calib_scores, 0.9)) if len(calib_scores) else 0.0
        lower = q10_pred - s_star
        upper = q90_pred + s_star
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
        record_result(
            "G11",
            "Conformalized quantile intervals",
            g08_features,
            (g08.pred_train, g08.pred_val, g08.pred_test),
            {"calib_s": s_star},
            extra={"coverage_80_test": coverage},
        )

    # G12 Stacking (G02/G03/G06/G08)
    base_ids = ["G02", "G03", "G06", "G08"]
    if all(eid in ctx.cache for eid in base_ids):
        train_preds = np.column_stack([ctx.cache[eid].pred_train for eid in base_ids])
        val_preds = np.column_stack([ctx.cache[eid].pred_val for eid in base_ids])
        test_preds = np.column_stack([ctx.cache[eid].pred_test for eid in base_ids])
        y_val = df.loc[ctx.val_mask, "y_actual_tmax_f"].to_numpy(dtype=float)
        y_test = df.loc[ctx.test_mask, "y_actual_tmax_f"].to_numpy(dtype=float)
        meta_model = Ridge(random_state=ctx.seed)
        meta_model.fit(val_preds, y_val)
        pred_train = meta_model.predict(train_preds)
        pred_val = meta_model.predict(val_preds)
        pred_test = meta_model.predict(test_preds)
        record_result(
            "G12",
            "Stacked meta model",
            base_ids,
            (pred_train, pred_val, pred_test),
            {"base_models": base_ids},
        )

    # G13 seasonal experts
    g13_features = g02_features
    season_models = {}
    season_preds = np.full(len(df), np.nan)
    for season_name, months in {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }.items():
        mask = df["target_date_local"].dt.month.isin(months).to_numpy()
        train_mask = ctx.train_mask & mask
        val_mask = ctx.val_mask & mask
        test_mask = ctx.test_mask & mask
        if not train_mask.any():
            continue
        features, _ = impute_features(df[g13_features], train_mask)
        X = features.to_numpy(dtype=float)
        X_train = X[train_mask]
        X_val = X[val_mask]
        y = df["y_actual_tmax_f"].to_numpy(dtype=float)
        y_train = y[train_mask]
        y_val = y[val_mask]
        model = train_lgbm_regressor(X_train, y_train, X_val, y_val, seed=ctx.seed)
        season_models[season_name] = model
        season_preds[mask] = model.predict(X[mask])
    pred_train = season_preds[ctx.train_mask]
    pred_val = season_preds[ctx.val_mask]
    pred_test = season_preds[ctx.test_mask]
    record_result("G13", "Seasonal expert models", g13_features, (pred_train, pred_val, pred_test), {"seasons": list(season_models.keys())})

    # G14 convective suppression MoE
    g14_features = g02_features
    suppression = compute_suppression_label(df)
    from lightgbm import LGBMClassifier
    features, impute_meta = impute_features(df[g14_features], ctx.train_mask)
    X = features.to_numpy(dtype=float)
    X_train = X[ctx.train_mask]
    X_val = X[ctx.val_mask]
    X_test = X[ctx.test_mask]
    y = df["y_actual_tmax_f"].to_numpy(dtype=float)
    y_train = y[ctx.train_mask]
    y_val = y[ctx.val_mask]
    y_test = y[ctx.test_mask]
    gate = np.asarray(suppression)
    gate_train = gate[ctx.train_mask]
    gate_val = gate[ctx.val_mask]
    gate_test = gate[ctx.test_mask]
    gate_model = LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=50,
        n_estimators=1000,
        random_state=ctx.seed,
        n_jobs=1,
        verbose=-1,
    )
    gate_model.fit(X_train, gate_train, eval_set=[(X_val, gate_val)], eval_metric="binary_logloss")
    p_gate_train = gate_model.predict_proba(X_train)[:, 1]
    p_gate_val = gate_model.predict_proba(X_val)[:, 1]
    p_gate_test = gate_model.predict_proba(X_test)[:, 1]

    def fit_expert(train_gate: np.ndarray, val_gate: np.ndarray, *, value: int) -> Any:
        train_mask = train_gate == value
        val_mask = val_gate == value
        if not train_mask.any():
            return None
        return train_lgbm_regressor(
            X_train[train_mask],
            y_train[train_mask],
            X_val[val_mask],
            y_val[val_mask],
            seed=ctx.seed,
        )

    expert_cool = fit_expert(gate_train, gate_val, value=0)
    expert_wet = fit_expert(gate_train, gate_val, value=1)
    pred_train = np.where(
        p_gate_train >= 0.5,
        expert_wet.predict(X_train) if expert_wet else np.nan,
        expert_cool.predict(X_train) if expert_cool else np.nan,
    )
    pred_val = np.where(
        p_gate_val >= 0.5,
        expert_wet.predict(X_val) if expert_wet else np.nan,
        expert_cool.predict(X_val) if expert_cool else np.nan,
    )
    pred_test = np.where(
        p_gate_test >= 0.5,
        expert_wet.predict(X_test) if expert_wet else np.nan,
        expert_cool.predict(X_test) if expert_cool else np.nan,
    )
    # Soft blend
    if expert_wet and expert_cool:
        pred_train = p_gate_train * expert_wet.predict(X_train) + (1 - p_gate_train) * expert_cool.predict(X_train)
        pred_val = p_gate_val * expert_wet.predict(X_val) + (1 - p_gate_val) * expert_cool.predict(X_val)
        pred_test = p_gate_test * expert_wet.predict(X_test) + (1 - p_gate_test) * expert_cool.predict(X_test)
    record_result(
        "G14",
        "Convective suppression MoE",
        g14_features,
        (pred_train, pred_val, pred_test),
        {"impute": impute_meta},
    )

    # G15 online EWMA bias correction
    base = ctx.cache.get("G02")
    if base:
        y = df["y_actual_tmax_f"].to_numpy(dtype=float)
        pred_full = np.full(len(df), np.nan)
        pred_full[ctx.train_mask] = base.pred_train
        pred_full[ctx.val_mask] = base.pred_val
        pred_full[ctx.test_mask] = base.pred_test
        errors = y - pred_full
        bias = np.zeros(len(df))
        lam = 2.0 / (14.0 + 1.0)
        for i in range(len(df)):
            if i == 0:
                bias[i] = 0.0
                continue
            prev = bias[i - 1]
            idx = i - 2
            if idx >= 0 and not np.isnan(errors[idx]):
                prev = (1.0 - lam) * prev + lam * errors[idx]
            bias[i] = prev
        pred_adj = pred_full + bias
        pred_train = pred_adj[ctx.train_mask]
        pred_val = pred_adj[ctx.val_mask]
        pred_test = pred_adj[ctx.test_mask]
        record_result(
            "G15",
            "Online EWMA bias correction",
            g02_features,
            (pred_train, pred_val, pred_test),
            {"lambda": lam},
        )

    return results


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run KMIA MOS godlike experiment suite (G01-G15)")
    parser.add_argument("--csv", required=True, help="Path to MOS features CSV.")
    parser.add_argument("--output-root", default="artifacts/MOS/experiments", help="Output root directory.")
    parser.add_argument("--suite-id", help="Optional suite id override.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-start", default="2002-01-22")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--val-start", default="2020-01-01")
    parser.add_argument("--val-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    args = parser.parse_args()

    df = load_csv(args.csv)
    df = add_derived_features(df)
    df = df[df["y_actual_tmax_f"].notna()].copy()

    split = split_by_date(
        df,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    ctx = ExperimentContext(
        df=df,
        train_mask=split.pop("train_mask"),
        val_mask=split.pop("val_mask"),
        test_mask=split.pop("test_mask"),
        seed=args.seed,
        split_info=split,
        cache={},
    )

    suite_id = args.suite_id or default_suite_id()
    output_root = Path(args.output_root) / suite_id
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "split_info.json", split)

    results = run_experiments(ctx, output_root)

    summary = {
        "suite_id": suite_id,
        "created_utc": utc_now_iso(),
        "csv_path": str(Path(args.csv).resolve()),
        "split": split,
        "experiments": [
            {
                "experiment_id": r.experiment_id,
                "name": r.name,
                "metrics": r.metrics,
                "extras": r.extras,
            }
            for r in results
        ],
    }
    write_json(output_root / "experiments_summary.json", summary)

    rows = []
    for r in results:
        test_mae = r.metrics.get("test", {}).get("mae")
        rows.append(
            {
                "experiment_id": r.experiment_id,
                "name": r.name,
                "test_mae": test_mae,
                "val_mae": r.metrics.get("validation", {}).get("mae"),
                "train_mae": r.metrics.get("train", {}).get("mae"),
            }
        )
    pd.DataFrame(rows).sort_values("test_mae").to_csv(output_root / "experiments_summary.csv", index=False)
    LOGGER.info("Suite complete. Output: %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
