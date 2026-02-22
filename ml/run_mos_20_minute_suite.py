"""Minute-enhanced MOS suite (E46-E65) for KMIA Tmax next-day."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import run_mos_45_suite as base

LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    features: list[str]
    metrics: dict[str, dict[str, float]]
    extras: dict[str, Any]


@dataclass
class SuiteContext:
    df: pd.DataFrame
    y: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    seed: int
    cache: dict[str, Any]


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return base.ensure_columns(df, columns)


def impute_features(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    return base.impute_features(df, train_mask)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return base.regression_metrics(y_true, y_pred)


def load_csv(path: str | Path) -> pd.DataFrame:
    return base.load_csv(path)


def split_by_date(
    df: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, Any]:
    return base.split_by_date(
        df,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )


def train_lgbm_huber(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    alpha: float = 1.0,
) -> lgb.LGBMRegressor:
    params = {
        "objective": "huber",
        "alpha": alpha,
        "learning_rate": 0.05,
        "n_estimators": 700,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**params)
    if len(y_val):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def _prepare_matrix(ctx: SuiteContext, features: list[str]) -> np.ndarray:
    df = ensure_columns(ctx.df, features)
    filled, _ = impute_features(df[features], ctx.train_mask)
    return filled.to_numpy(dtype=float)


def _gate_probability(
    ctx: SuiteContext,
    gate_features: list[str],
    gate_target: pd.Series,
) -> np.ndarray:
    df = ensure_columns(ctx.df, gate_features)
    filled, _ = impute_features(df[gate_features], ctx.train_mask)
    X = filled.to_numpy(dtype=float)
    y_gate = gate_target.fillna(0.0).astype(int).to_numpy()
    model = base.train_lgbm_classifier(
        X[ctx.train_mask],
        y_gate[ctx.train_mask],
        X[ctx.val_mask],
        y_gate[ctx.val_mask],
        seed=ctx.seed,
    )
    return model.predict_proba(X)[:, 1]


def _fit_residual_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    objective: str = "l1",
    alpha: float = 0.5,
) -> lgb.LGBMRegressor:
    if objective == "quantile":
        return base.train_lgbm_quantile(X_train, y_train, X_val, y_val, seed=seed, alpha=alpha)
    if objective == "huber":
        return train_lgbm_huber(X_train, y_train, X_val, y_val, seed=seed, alpha=alpha)
    return base.train_lgbm_regressor(X_train, y_train, X_val, y_val, seed=seed, params=None)


def _predict_or_zero(model: lgb.LGBMRegressor | None, X: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(X), dtype=float)
    return model.predict(X)


def _zscore(series: pd.Series, mean: float, std: float) -> pd.Series:
    return (series - mean) / (std + 1e-6)


def _zscore_low(series: pd.Series, mean: float, std: float) -> pd.Series:
    return (mean - series) / (std + 1e-6)


def _hourly_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [f"feat_iem_hour_{h:02d}_tminus1" for h in range(24)]
    hourly = ensure_columns(df, cols)[cols].to_numpy(dtype=float)
    return hourly


def _add_pca_features(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    hourly = _hourly_matrix(df)
    tmean = pd.to_numeric(df.get("feat_iem_tmean_tminus1"), errors="coerce").to_numpy(dtype=float)
    trange = pd.to_numeric(df.get("feat_iem_range_tminus1"), errors="coerce").to_numpy(dtype=float)
    norm = (hourly - tmean[:, None]) / (trange[:, None] + 1e-3)
    norm = np.where(np.isnan(norm), 0.0, norm)
    train_norm = norm[train_mask]
    if train_norm.shape[0] < 10:
        comps = np.full((len(df), 4), np.nan)
    else:
        pca = PCA(n_components=4)
        pca.fit(train_norm)
        comps = pca.transform(norm)
    for i in range(4):
        df[f"feat_iem_pca{i+1}_tminus1"] = comps[:, i]
    return df


def _add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    hourly = _hourly_matrix(df)
    row_mean = np.nanmean(hourly, axis=1)
    temps = np.where(np.isnan(hourly), row_mean[:, None], hourly)
    h = np.arange(24, dtype=float)
    cos1 = np.cos(2 * np.pi * h / 24.0)
    sin1 = np.sin(2 * np.pi * h / 24.0)
    cos2 = np.cos(4 * np.pi * h / 24.0)
    sin2 = np.sin(4 * np.pi * h / 24.0)
    a1 = temps @ cos1
    b1 = temps @ sin1
    a2 = temps @ cos2
    b2 = temps @ sin2
    amp1 = np.sqrt(a1 * a1 + b1 * b1)
    amp2 = np.sqrt(a2 * a2 + b2 * b2)
    phase1 = np.arctan2(b1, a1)
    phase2 = np.arctan2(b2, a2)
    trange = pd.to_numeric(df.get("feat_iem_range_tminus1"), errors="coerce").to_numpy(dtype=float)
    df["feat_iem_amp1_tminus1"] = amp1
    df["feat_iem_phase1_tminus1"] = phase1
    df["feat_iem_amp2_tminus1"] = amp2
    df["feat_iem_phase2_tminus1"] = phase2
    df["feat_iem_amp1_norm_tminus1"] = amp1 / (trange + 1e-3)
    return df


def _add_seabreeze_features(df: pd.DataFrame) -> pd.DataFrame:
    hourly = _hourly_matrix(df)
    row_mean = np.nanmean(hourly, axis=1)
    temps = np.where(np.isnan(hourly), row_mean[:, None], hourly)
    slopes = temps[:, 1:] - temps[:, :-1]
    inflect = np.full(len(df), np.nan)
    for h in range(11, 18):
        cond = (slopes[:, h] <= 0.5) & (slopes[:, h + 1] <= 0.5)
        inflect = np.where(np.isnan(inflect) & cond, float(h), inflect)
    df["feat_iem_sb_inflect_hour_tminus1"] = inflect
    df["feat_iem_sb_flatten_strength_tminus1"] = (
        pd.to_numeric(df.get("feat_iem_slope_12_15_tminus1"), errors="coerce")
        - pd.to_numeric(df.get("feat_iem_slope_15_18_tminus1"), errors="coerce")
    )
    window = temps[:, 15:20]
    df["feat_iem_sb_drop_after_15_tminus1"] = np.nanmax(window, axis=1) - np.nanmin(window, axis=1)
    return df


def add_minute_derived_features(
    df: pd.DataFrame,
    train_mask: np.ndarray,
) -> pd.DataFrame:
    df = df.copy()

    # T-2 memory from shifting T-1 features
    shift_cols = [
        "feat_iem_range_tminus1",
        "feat_iem_tmax_time_local_min_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
    ]
    for col in shift_cols:
        df[col.replace("_tminus1", "_tminus2")] = pd.to_numeric(df.get(col), errors="coerce").shift(1)
    df["feat_iem_range_delta_tminus1"] = (
        pd.to_numeric(df.get("feat_iem_range_tminus1"), errors="coerce")
        - pd.to_numeric(df.get("feat_iem_range_tminus2"), errors="coerce")
    )
    df["feat_iem_tmax_time_delta_tminus1"] = (
        pd.to_numeric(df.get("feat_iem_tmax_time_local_min_tminus1"), errors="coerce")
        - pd.to_numeric(df.get("feat_iem_tmax_time_local_min_tminus2"), errors="coerce")
    )

    df["feat_iem_temp_06z_adj"] = (
        pd.to_numeric(df.get("feat_iem_temp_06z"), errors="coerce")
        + pd.to_numeric(df.get("feat_iem_diff_ewma_30_tminus1"), errors="coerce")
    )
    df["feat_iem_climo_06z"] = pd.to_numeric(df.get("feat_iem_temp_06z"), errors="coerce") - pd.to_numeric(
        df.get("feat_iem_night_warm_anom"), errors="coerce"
    )

    train = df.loc[train_mask]

    q80_std = float(train["feat_iem_std_last180"].quantile(0.8))
    q20_cool = float(train["feat_iem_cool_00_06"].quantile(0.2))
    q80_night = float(train["feat_iem_night_warm_anom"].quantile(0.8))
    df["label_cloudy_night"] = (
        (df["feat_iem_std_last180"] > q80_std)
        | (df["feat_iem_cool_00_06"] < q20_cool)
        | (df["feat_iem_night_warm_anom"] > q80_night)
    ).astype(int)

    df["label_outflow"] = (
        (df["feat_iem_max_drop_30_tminus1"] >= 3.0) | (df["feat_iem_drop_cnt_30_tminus1"] >= 2.0)
    ).astype(int)

    df["label_late_surge"] = (df["feat_iem_tmax_time_local_min_tminus1"] > (17.5 * 60)).astype(int)

    q20_range = float(train["feat_iem_range_tminus1"].quantile(0.2))
    q80_plateau = float(train["feat_iem_plateau_mins_eps_tminus1"].quantile(0.8))
    df["label_suppressed"] = (
        (df["feat_iem_range_tminus1"] < q20_range)
        & (df["feat_iem_plateau_mins_eps_tminus1"] > q80_plateau)
    ).astype(int)

    range_mean = float(train["feat_iem_range_tminus1"].mean())
    range_std = float(train["feat_iem_range_tminus1"].std())
    plateau_mean = float(train["feat_iem_plateau_mins_eps_tminus1"].mean())
    plateau_std = float(train["feat_iem_plateau_mins_eps_tminus1"].std())
    drop_mean = float(train["feat_iem_drop_cnt_30_tminus1"].mean())
    drop_std = float(train["feat_iem_drop_cnt_30_tminus1"].std())
    slope_mean = float(train["feat_iem_slope_12_15_tminus1"].mean())
    slope_std = float(train["feat_iem_slope_12_15_tminus1"].std())

    df["feat_iem_supp_idx_tminus1"] = (
        _zscore_low(df["feat_iem_range_tminus1"], range_mean, range_std)
        + 0.8 * _zscore(df["feat_iem_plateau_mins_eps_tminus1"], plateau_mean, plateau_std)
        + 0.8 * _zscore(df["feat_iem_drop_cnt_30_tminus1"], drop_mean, drop_std)
        + 0.6 * _zscore_low(df["feat_iem_slope_12_15_tminus1"], slope_mean, slope_std)
    )

    df = _add_pca_features(df, train_mask)
    df = _add_fourier_features(df)
    df = _add_seabreeze_features(df)

    return df


def _get_base(ctx: SuiteContext, base_series: str) -> np.ndarray:
    base = pd.to_numeric(ctx.df.get(base_series), errors="coerce").to_numpy(dtype=float)
    base_mean = float(np.nanmean(ctx.y[ctx.train_mask]))
    return np.where(np.isnan(base), base_mean, base)


def _fit_expert(
    ctx: SuiteContext,
    X: np.ndarray,
    base: np.ndarray,
    mask: np.ndarray,
    *,
    objective: str = "l1",
    alpha: float = 0.5,
) -> lgb.LGBMRegressor | None:
    train_mask = ctx.train_mask & mask
    val_mask = ctx.val_mask & mask
    if not train_mask.any():
        return None
    return _fit_residual_model(
        X[train_mask],
        ctx.y[train_mask] - base[train_mask],
        X[val_mask],
        ctx.y[val_mask] - base[val_mask],
        seed=ctx.seed,
        objective=objective,
        alpha=alpha,
    )


def run_e37_moe(
    ctx: SuiteContext,
    *,
    gate_features: list[str],
    expert_features: list[str],
    base_series: str,
    gate_label: pd.Series,
    objective: str = "l1",
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_vals = _get_base(ctx, base_series)
    p_gate = _gate_probability(ctx, gate_features, gate_label)
    X_exp = _prepare_matrix(ctx, expert_features)
    expert_on = _fit_expert(ctx, X_exp, base_vals, gate_label == 1, objective=objective, alpha=alpha)
    expert_off = _fit_expert(ctx, X_exp, base_vals, gate_label == 0, objective=objective, alpha=alpha)
    resid_on = _predict_or_zero(expert_on, X_exp)
    resid_off = _predict_or_zero(expert_off, X_exp)
    pred_all = base_vals + p_gate * resid_on + (1 - p_gate) * resid_off
    return pred_all[ctx.train_mask], pred_all[ctx.val_mask], pred_all[ctx.test_mask], p_gate


def run_two_gate_moe(
    ctx: SuiteContext,
    *,
    gate1_features: list[str],
    gate1_label: pd.Series,
    gate2_features: list[str],
    gate2_label: pd.Series,
    expert_features: list[str],
    base_series: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_vals = _get_base(ctx, base_series)
    p_gate1 = _gate_probability(ctx, gate1_features, gate1_label)
    p_gate2 = _gate_probability(ctx, gate2_features, gate2_label)
    X_exp = _prepare_matrix(ctx, expert_features)

    def mask(a: int, b: int) -> np.ndarray:
        return (gate1_label == a) & (gate2_label == b)

    expert_11 = _fit_expert(ctx, X_exp, base_vals, mask(1, 1))
    expert_10 = _fit_expert(ctx, X_exp, base_vals, mask(1, 0))
    expert_01 = _fit_expert(ctx, X_exp, base_vals, mask(0, 1))
    expert_00 = _fit_expert(ctx, X_exp, base_vals, mask(0, 0))

    resid_11 = _predict_or_zero(expert_11, X_exp)
    resid_10 = _predict_or_zero(expert_10, X_exp)
    resid_01 = _predict_or_zero(expert_01, X_exp)
    resid_00 = _predict_or_zero(expert_00, X_exp)

    resid_mix = (
        p_gate1 * (p_gate2 * resid_11 + (1 - p_gate2) * resid_10)
        + (1 - p_gate1) * (p_gate2 * resid_01 + (1 - p_gate2) * resid_00)
    )
    pred_all = base_vals + resid_mix
    return pred_all[ctx.train_mask], pred_all[ctx.val_mask], pred_all[ctx.test_mask], p_gate1, p_gate2


def run_three_gate_moe(
    ctx: SuiteContext,
    *,
    gate_features: list[str],
    labels: pd.Series,
    expert_features: list[str],
    base_series: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_vals = _get_base(ctx, base_series)
    X_gate = _prepare_matrix(ctx, gate_features)
    probs = []
    for cls in [0, 1, 2]:
        target = (labels == cls).astype(int)
        model = base.train_lgbm_classifier(
            X_gate[ctx.train_mask],
            target.to_numpy()[ctx.train_mask],
            X_gate[ctx.val_mask],
            target.to_numpy()[ctx.val_mask],
            seed=ctx.seed,
        )
        probs.append(model.predict_proba(X_gate)[:, 1])
    p_raw = np.vstack(probs).T
    p_sum = p_raw.sum(axis=1, keepdims=True)
    p_sum = np.where(p_sum == 0, 1.0, p_sum)
    p = p_raw / p_sum

    X_exp = _prepare_matrix(ctx, expert_features)
    experts = []
    for cls in [0, 1, 2]:
        expert = _fit_expert(ctx, X_exp, base_vals, labels == cls)
        experts.append(expert)
    resid = np.vstack([_predict_or_zero(experts[i], X_exp) for i in range(3)]).T
    pred_all = base_vals + np.sum(p * resid, axis=1)
    return pred_all[ctx.train_mask], pred_all[ctx.val_mask], pred_all[ctx.test_mask], p


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MOS minute-enhanced experiments (E46-E65).")
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument("--suite-id", default=base.default_suite_id())
    parser.add_argument("--out-root", default="artifacts/MOS/experiments")
    parser.add_argument("--train-start", default="2002-01-22")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--val-start", default="2020-01-01")
    parser.add_argument("--val-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base.setup_logging()

    df_raw = load_csv(args.features)
    df_raw = df_raw.sort_values("asof_date_local").drop_duplicates(subset=["target_date_local"], keep="last")
    feature_store = base.build_feature_store(df_raw)
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
    train_mask = split.pop("train_mask")
    val_mask = split.pop("val_mask")
    test_mask = split.pop("test_mask")

    feature_store = add_minute_derived_features(feature_store, train_mask)

    ctx = SuiteContext(
        df=feature_store,
        y=feature_store["y_actual_tmax_f"].to_numpy(dtype=float),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        seed=args.seed,
        cache={},
    )

    output_root = Path(args.out_root) / args.suite_id
    output_root.mkdir(parents=True, exist_ok=True)

    feature_store_path = output_root / "feature_store.csv"
    feature_store.to_csv(feature_store_path, index=False)
    base.write_json(output_root / "split_info.json", split)

    results: list[ExperimentResult] = []

    def record(
        exp_id: str,
        name: str,
        features: list[str],
        preds: tuple[np.ndarray, np.ndarray, np.ndarray],
        extra: dict[str, Any] | None = None,
    ) -> None:
        pred_train, pred_val, pred_test = preds
        metrics_payload = {
            "train": regression_metrics(ctx.y[ctx.train_mask], pred_train),
            "validation": regression_metrics(ctx.y[ctx.val_mask], pred_val),
            "test": regression_metrics(ctx.y[ctx.test_mask], pred_test),
        }
        extras = dict(extra or {})
        results.append(
            ExperimentResult(
                experiment_id=exp_id,
                name=name,
                features=features,
                metrics=metrics_payload,
                extras=extras,
            )
        )
        exp_dir = output_root / exp_id
        exp_dir.mkdir(exist_ok=True, parents=True)
        base.save_predictions(
            exp_dir / "predictions_train.csv",
            feature_store.loc[ctx.train_mask],
            ctx.y[ctx.train_mask],
            pred_train,
        )
        base.save_predictions(
            exp_dir / "predictions_val.csv",
            feature_store.loc[ctx.val_mask],
            ctx.y[ctx.val_mask],
            pred_val,
        )
        base.save_predictions(
            exp_dir / "predictions_test.csv",
            feature_store.loc[ctx.test_mask],
            ctx.y[ctx.test_mask],
            pred_test,
        )

    DOY = ["cal_d_doy_sin", "cal_d_doy_cos"]
    GATE_FEATURES = ["feat_u", "feat_v", *DOY]
    BASE_SERIES = "feat_le_median_biascorr"
    gate_label_onshore = (feature_store["feat_onshore"] > 0.5).astype(int)

    base_features = [
        "feat_dd_models",
        "feat_tmp_range_mean_models",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
        *DOY,
    ]
    night_features = [
        "feat_iem_temp_00z",
        "feat_iem_temp_03z",
        "feat_iem_temp_06z",
        "feat_iem_cool_00_06",
        "feat_iem_slope_last180",
        "feat_iem_std_last180",
        "feat_iem_night_warm_anom",
    ]
    tminus1_shape = [
        "feat_iem_tmax_tminus1",
        "feat_iem_tmin_tminus1",
        "feat_iem_range_tminus1",
        "feat_iem_tmax_time_local_min_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_max_drop_30_tminus1",
        "feat_iem_slope_12_15_tminus1",
        "feat_iem_slope_15_18_tminus1",
        "feat_iem_slope_18_21_tminus1",
    ]

    # Experiments E46-E65
    features_e46 = base_features + night_features
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e46,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
    )
    record("E46", "E37 + Night airmass obs injection", features_e46, (pred_train, pred_val, pred_test))

    features_e47 = base_features + night_features + tminus1_shape
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e47,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
    )
    record("E47", "E37 + T-1 diurnal shape memory", features_e47, (pred_train, pred_val, pred_test))

    translator_features = [
        "feat_iem_diff_tminus1",
        "feat_iem_diff_ewma_7_tminus1",
        "feat_iem_diff_ewma_30_tminus1",
        "feat_iem_diff_vol_30_tminus1",
        "feat_iem_temp_06z_adj",
    ]
    features_e48 = base_features + translator_features
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e48,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
    )
    record("E48", "IEM->NWS translator features", features_e48, (pred_train, pred_val, pred_test))

    p_gate = _gate_probability(ctx, GATE_FEATURES, gate_label_onshore)
    feature_store["feat_iem_diff_state"] = p_gate * feature_store["feat_iem_diff_ewma_onshore"] + (1 - p_gate) * feature_store[
        "feat_iem_diff_ewma_offshore"
    ]
    feature_store["feat_iem_temp_06z_adj_state"] = feature_store["feat_iem_temp_06z"] + feature_store[
        "feat_iem_diff_state"
    ]
    features_e49 = base_features + ["feat_iem_diff_state", "feat_iem_temp_06z_adj_state"]
    base_vals = _get_base(ctx, BASE_SERIES)
    X_exp = _prepare_matrix(ctx, features_e49)
    expert_on = _fit_expert(ctx, X_exp, base_vals, gate_label_onshore == 1)
    expert_off = _fit_expert(ctx, X_exp, base_vals, gate_label_onshore == 0)
    resid_on = _predict_or_zero(expert_on, X_exp)
    resid_off = _predict_or_zero(expert_off, X_exp)
    pred_all = base_vals + p_gate * resid_on + (1 - p_gate) * resid_off
    record(
        "E49",
        "Regime-conditioned translator state",
        features_e49,
        (pred_all[ctx.train_mask], pred_all[ctx.val_mask], pred_all[ctx.test_mask]),
    )

    cloud_gate_features = [
        "feat_iem_temp_00z",
        "feat_iem_temp_03z",
        "feat_iem_temp_06z",
        "feat_iem_cool_00_06",
        "feat_iem_std_last180",
        "feat_iem_slope_last180",
        "feat_dd_models",
        "feat_p12_max",
        "feat_cig_min",
        *DOY,
    ]
    features_e50 = base_features + night_features + [
        "feat_iem_range_tminus1",
        "feat_iem_tmax_time_local_min_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
    ]
    pred_train, pred_val, pred_test, _, _ = run_two_gate_moe(
        ctx,
        gate1_features=GATE_FEATURES,
        gate1_label=gate_label_onshore,
        gate2_features=cloud_gate_features,
        gate2_label=feature_store["label_cloudy_night"],
        expert_features=features_e50,
        base_series=BASE_SERIES,
    )
    record("E50", "Hierarchical MoE (onshore x cloudy night)", features_e50, (pred_train, pred_val, pred_test))

    labels_e51 = np.where(
        feature_store["label_late_surge"] == 1,
        1,
        np.where(feature_store["label_suppressed"] == 1, 2, 0),
    )
    gate_features_e51 = [
        "feat_iem_range_tminus1",
        "feat_iem_tmax_time_local_min_tminus1",
        "feat_iem_slope_15_18_tminus1",
        "feat_iem_slope_18_21_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        *DOY,
    ]
    features_e51 = base_features + night_features + [
        "feat_iem_range_tminus1",
        "feat_iem_tmax_time_local_min_tminus1",
        "feat_iem_slope_15_18_tminus1",
        "feat_iem_slope_18_21_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
    ]
    pred_train, pred_val, pred_test, _ = run_three_gate_moe(
        ctx,
        gate_features=gate_features_e51,
        labels=pd.Series(labels_e51),
        expert_features=features_e51,
        base_series=BASE_SERIES,
    )
    record("E51", "Late-surge vs suppressed vs normal MoE", features_e51, (pred_train, pred_val, pred_test))

    outflow_gate_features = [
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_max_drop_30_tminus1",
        "feat_iem_range_tminus1",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_dd_models",
        *DOY,
    ]
    features_e52 = base_features + night_features + [
        "feat_iem_range_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_max_drop_30_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
    ]
    pred_train, pred_val, pred_test, _, _ = run_two_gate_moe(
        ctx,
        gate1_features=GATE_FEATURES,
        gate1_label=gate_label_onshore,
        gate2_features=outflow_gate_features,
        gate2_label=feature_store["label_outflow"],
        expert_features=features_e52,
        base_series=BASE_SERIES,
    )
    record("E52", "Outflow gate (onshore x outflow)", features_e52, (pred_train, pred_val, pred_test))

    features_e53 = [
        "feat_iem_supp_idx_tminus1",
        "feat_iem_range_tminus1",
        "feat_iem_temp_00z",
        "feat_iem_temp_03z",
        "feat_iem_temp_06z",
        "feat_iem_cool_00_06",
        "feat_iem_slope_last180",
        "feat_iem_std_last180",
        "feat_iem_night_warm_anom",
        "feat_u",
        "feat_v",
        "feat_dd_models",
        *DOY,
    ]
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e53,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
    )
    record("E53", "Suppression index replaces MOS proxies", features_e53, (pred_train, pred_val, pred_test))

    pca_features = [f"feat_iem_pca{i}_tminus1" for i in range(1, 5)]
    features_e54 = base_features + night_features + pca_features
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e54,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
    )
    record("E54", "Compact PCA curve embedding", features_e54, (pred_train, pred_val, pred_test))

    fourier_features = [
        "feat_iem_amp1_tminus1",
        "feat_iem_phase1_tminus1",
        "feat_iem_amp2_tminus1",
        "feat_iem_phase2_tminus1",
        "feat_iem_amp1_norm_tminus1",
    ]
    features_e55 = base_features + night_features + fourier_features
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e55,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
    )
    record("E55", "Fourier diurnal shape features", features_e55, (pred_train, pred_val, pred_test))

    # E56: second-stage minute-only corrector on top of E37
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e47,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
    )
    pred_all = np.full(len(ctx.y), np.nan)
    pred_all[ctx.train_mask] = pred_train
    pred_all[ctx.val_mask] = pred_val
    pred_all[ctx.test_mask] = pred_test
    residual = ctx.y - pred_all

    minute_only = night_features + [
        "feat_iem_range_tminus1",
        "feat_iem_tmax_time_local_min_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_diff_ewma_30_tminus1",
    ]
    X_minute = _prepare_matrix(ctx, minute_only)
    res_model = base.train_lgbm_regressor(
        X_minute[ctx.train_mask],
        residual[ctx.train_mask],
        X_minute[ctx.val_mask],
        residual[ctx.val_mask],
        seed=ctx.seed,
    )
    res_pred = res_model.predict(X_minute)
    pred_all_56 = pred_all + res_pred
    record(
        "E56",
        "Minute-only residual corrector",
        minute_only,
        (pred_all_56[ctx.train_mask], pred_all_56[ctx.val_mask], pred_all_56[ctx.test_mask]),
    )

    # E57: uncertainty-aware shrinkage between E37 and fallback
    climo_mean = pd.to_numeric(feature_store.get("feat_climo_mean_doy"), errors="coerce").to_numpy(dtype=float)
    temp_06z_adj = pd.to_numeric(feature_store.get("feat_iem_temp_06z_adj"), errors="coerce").to_numpy(dtype=float)
    climo_06z = pd.to_numeric(feature_store.get("feat_iem_climo_06z"), errors="coerce").to_numpy(dtype=float)
    fallback = climo_mean + 0.5 * (temp_06z_adj - climo_06z)
    fallback = np.where(np.isnan(fallback), np.nanmean(ctx.y[ctx.train_mask]), fallback)
    target_w = (np.abs(ctx.y - pred_all) <= np.abs(ctx.y - fallback)).astype(int)
    w_features = [
        "feat_iem_std_last180",
        "feat_iem_cool_00_06",
        "feat_iem_diff_vol_30_tminus1",
        "feat_p12_max",
        "feat_le_spread",
        *DOY,
    ]
    X_w = _prepare_matrix(ctx, w_features)
    w_model = base.train_lgbm_classifier(
        X_w[ctx.train_mask],
        target_w[ctx.train_mask],
        X_w[ctx.val_mask],
        target_w[ctx.val_mask],
        seed=ctx.seed,
    )
    w = w_model.predict_proba(X_w)[:, 1]
    pred_all_57 = w * pred_all + (1 - w) * fallback
    record(
        "E57",
        "Minute-informed shrinkage blend",
        w_features,
        (pred_all_57[ctx.train_mask], pred_all_57[ctx.val_mask], pred_all_57[ctx.test_mask]),
    )

    # E58: quantile-median experts
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e47,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
        objective="quantile",
        alpha=0.5,
    )
    record("E58", "Quantile-median experts (E37)", features_e47, (pred_train, pred_val, pred_test))

    # E59: Huber experts with volatility regimes
    q80_vol = float(feature_store["feat_iem_std_last180"].quantile(0.8))
    vol_high = (feature_store["feat_iem_std_last180"] > q80_vol).astype(int)
    p_gate = _gate_probability(ctx, GATE_FEATURES, gate_label_onshore)
    p_vol = vol_high.to_numpy(dtype=float)
    features_e59 = base_features + night_features + [
        "feat_iem_range_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_max_drop_30_tminus1",
    ]
    base_vals = _get_base(ctx, BASE_SERIES)
    X_exp = _prepare_matrix(ctx, features_e59)
    expert_on_high = _fit_expert(ctx, X_exp, base_vals, (gate_label_onshore == 1) & (vol_high == 1), objective="huber", alpha=2.0)
    expert_on_low = _fit_expert(ctx, X_exp, base_vals, (gate_label_onshore == 1) & (vol_high == 0), objective="huber", alpha=1.0)
    expert_off_high = _fit_expert(ctx, X_exp, base_vals, (gate_label_onshore == 0) & (vol_high == 1), objective="huber", alpha=2.0)
    expert_off_low = _fit_expert(ctx, X_exp, base_vals, (gate_label_onshore == 0) & (vol_high == 0), objective="huber", alpha=1.0)
    resid_on_high = _predict_or_zero(expert_on_high, X_exp)
    resid_on_low = _predict_or_zero(expert_on_low, X_exp)
    resid_off_high = _predict_or_zero(expert_off_high, X_exp)
    resid_off_low = _predict_or_zero(expert_off_low, X_exp)
    resid_mix = p_gate * (p_vol * resid_on_high + (1 - p_vol) * resid_on_low) + (1 - p_gate) * (
        p_vol * resid_off_high + (1 - p_vol) * resid_off_low
    )
    pred_all = base_vals + resid_mix
    record(
        "E59",
        "Huber experts with volatility regimes",
        features_e59,
        (pred_all[ctx.train_mask], pred_all[ctx.val_mask], pred_all[ctx.test_mask]),
    )

    # E60: clustered residual bias tables
    cluster_features = [
        "feat_iem_range_tminus1",
        "feat_iem_tmax_time_local_min_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_slope_12_15_tminus1",
        "feat_iem_slope_15_18_tminus1",
    ]
    X_cluster = _prepare_matrix(ctx, cluster_features)
    kmeans = KMeans(n_clusters=8, random_state=ctx.seed, n_init=10)
    kmeans.fit(X_cluster[ctx.train_mask])
    cluster_id = kmeans.predict(X_cluster)
    base_vals = _get_base(ctx, BASE_SERIES)
    residual = ctx.y - base_vals
    bias_on = np.zeros(8, dtype=float)
    bias_off = np.zeros(8, dtype=float)
    global_bias = float(np.nanmean(residual[ctx.train_mask]))
    for k in range(8):
        mask_train = (cluster_id == k) & ctx.train_mask
        on_mask = mask_train & (gate_label_onshore == 1)
        off_mask = mask_train & (gate_label_onshore == 0)
        bias_on[k] = float(np.nanmean(residual[on_mask])) if on_mask.any() else float(np.nanmean(residual[mask_train]))
        bias_off[k] = float(np.nanmean(residual[off_mask])) if off_mask.any() else float(np.nanmean(residual[mask_train]))
        if np.isnan(bias_on[k]):
            bias_on[k] = global_bias
        if np.isnan(bias_off[k]):
            bias_off[k] = global_bias
    cluster_bias = np.where(gate_label_onshore == 1, bias_on[cluster_id], bias_off[cluster_id])
    feature_store["feat_cluster_bias_lookup"] = cluster_bias
    feature_store["feat_base_cluster_bias"] = base_vals + cluster_bias
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=base_features,
        base_series="feat_base_cluster_bias",
        gate_label=gate_label_onshore,
    )
    record("E60", "Regime-clustered residual bias table", base_features, (pred_train, pred_val, pred_test))

    # E61: two-day minute memory
    tminus2_features = [
        "feat_iem_range_tminus2",
        "feat_iem_tmax_time_local_min_tminus2",
        "feat_iem_plateau_mins_eps_tminus2",
        "feat_iem_drop_cnt_30_tminus2",
        "feat_iem_range_delta_tminus1",
        "feat_iem_tmax_time_delta_tminus1",
    ]
    features_e61 = base_features + night_features + tminus1_shape + tminus2_features
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e61,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
    )
    record("E61", "Two-day minute memory", features_e61, (pred_train, pred_val, pred_test))

    features_e62 = features_e47 + [
        "feat_iem_sb_inflect_hour_tminus1",
        "feat_iem_sb_flatten_strength_tminus1",
        "feat_iem_sb_drop_after_15_tminus1",
    ]
    pred_train, pred_val, pred_test, _ = run_e37_moe(
        ctx,
        gate_features=GATE_FEATURES,
        expert_features=features_e62,
        base_series=BASE_SERIES,
        gate_label=gate_label_onshore,
    )
    record("E62", "Sea-breeze inflection features", features_e62, (pred_train, pred_val, pred_test))

    # E63: obs-anchored diurnal-range model
    anchor = pd.to_numeric(feature_store.get("feat_iem_temp_06z_adj"), errors="coerce").to_numpy(dtype=float)
    anchor = np.where(np.isnan(anchor), np.nanmean(ctx.y[ctx.train_mask]), anchor)
    gain_features = [
        "feat_dd_models",
        "feat_p12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
        *DOY,
        "feat_iem_cool_00_06",
        "feat_iem_std_last180",
        "feat_iem_range_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
    ]
    X_gain = _prepare_matrix(ctx, gain_features)
    gain_target = ctx.y - anchor
    gain_model = base.train_lgbm_regressor(
        X_gain[ctx.train_mask],
        gain_target[ctx.train_mask],
        X_gain[ctx.val_mask],
        gain_target[ctx.val_mask],
        seed=ctx.seed,
    )
    gain_pred = gain_model.predict(X_gain)
    base_obs = anchor + gain_pred
    base_vals = _get_base(ctx, BASE_SERIES)
    pred_all = 0.7 * base_vals + 0.3 * base_obs
    record(
        "E63",
        "Obs-anchored gain model blended with E37",
        gain_features,
        (pred_all[ctx.train_mask], pred_all[ctx.val_mask], pred_all[ctx.test_mask]),
    )

    # E64: directional residual MoE
    base_vals = _get_base(ctx, BASE_SERIES)
    residual = ctx.y - base_vals
    q70 = float(np.quantile(np.abs(residual[ctx.train_mask]), 0.7))
    labels_e64 = np.where(residual < -q70, 0, np.where(residual > q70, 2, 1))
    gate_features_e64 = [
        "feat_iem_temp_00z",
        "feat_iem_temp_03z",
        "feat_iem_temp_06z",
        "feat_iem_cool_00_06",
        "feat_iem_std_last180",
        "feat_iem_range_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_dd_models",
        "feat_u",
        "feat_v",
        *DOY,
    ]
    features_e64 = base_features + night_features + [
        "feat_iem_range_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
        "feat_iem_tmax_time_local_min_tminus1",
    ]
    pred_train, pred_val, pred_test, _ = run_three_gate_moe(
        ctx,
        gate_features=gate_features_e64,
        labels=pd.Series(labels_e64),
        expert_features=features_e64,
        base_series=BASE_SERIES,
    )
    record("E64", "Directional residual MoE", features_e64, (pred_train, pred_val, pred_test))

    # E65: hierarchical 5-expert MoE
    p_onshore = _gate_probability(ctx, GATE_FEATURES, gate_label_onshore)
    p_cloud = _gate_probability(ctx, cloud_gate_features, feature_store["label_cloudy_night"])
    late_gate_features = [
        "feat_iem_tmax_time_local_min_tminus1",
        "feat_iem_slope_15_18_tminus1",
        "feat_iem_slope_18_21_tminus1",
        "feat_iem_range_tminus1",
    ]
    p_late = _gate_probability(ctx, late_gate_features, feature_store["label_late_surge"])
    features_e65 = base_features + [
        "feat_iem_temp_06z",
        "feat_iem_cool_00_06",
        "feat_iem_std_last180",
        "feat_iem_range_tminus1",
        "feat_iem_tmax_time_local_min_tminus1",
        "feat_iem_plateau_mins_eps_tminus1",
        "feat_iem_drop_cnt_30_tminus1",
        "feat_iem_diff_ewma_30_tminus1",
    ]
    base_vals = _get_base(ctx, BASE_SERIES)
    X_exp = _prepare_matrix(ctx, features_e65)
    def mask(a: int, b: int) -> np.ndarray:
        return (gate_label_onshore == a) & (feature_store["label_cloudy_night"] == b)
    expert_11 = _fit_expert(ctx, X_exp, base_vals, mask(1, 1))
    expert_10 = _fit_expert(ctx, X_exp, base_vals, mask(1, 0))
    expert_01 = _fit_expert(ctx, X_exp, base_vals, mask(0, 1))
    expert_00 = _fit_expert(ctx, X_exp, base_vals, mask(0, 0))
    resid_11 = _predict_or_zero(expert_11, X_exp)
    resid_10 = _predict_or_zero(expert_10, X_exp)
    resid_01 = _predict_or_zero(expert_01, X_exp)
    resid_00 = _predict_or_zero(expert_00, X_exp)
    resid_mix = (
        p_onshore * (p_cloud * resid_11 + (1 - p_cloud) * resid_10)
        + (1 - p_onshore) * (p_cloud * resid_01 + (1 - p_cloud) * resid_00)
    )
    late_expert = _fit_expert(ctx, X_exp, base_vals, feature_store["label_late_surge"] == 1)
    resid_late = _predict_or_zero(late_expert, X_exp)
    resid_final = (1 - p_late) * resid_mix + p_late * resid_late
    pred_all = base_vals + resid_final
    record("E65", "Hierarchical 5-expert MoE", features_e65, (pred_all[ctx.train_mask], pred_all[ctx.val_mask], pred_all[ctx.test_mask]))

    summary = {
        "suite_id": args.suite_id,
        "created_utc": utc_now_iso(),
        "split": split,
        "feature_store_path": str(feature_store_path),
        "experiments": [
            {
                "experiment_id": r.experiment_id,
                "name": r.name,
                "features": r.features,
                "metrics": r.metrics,
                "extras": r.extras,
            }
            for r in results
        ],
    }
    summary_path = output_root / "experiments_summary.json"
    base.write_json(summary_path, summary)

    rows = []
    for r in results:
        row = {
            "experiment_id": r.experiment_id,
            "name": r.name,
            "train_mae": r.metrics.get("train", {}).get("mae"),
            "val_mae": r.metrics.get("validation", {}).get("mae"),
            "test_mae": r.metrics.get("test", {}).get("mae"),
        }
        rows.append(row)
    pd.DataFrame(rows).sort_values("test_mae").to_csv(output_root / "experiments_summary.csv", index=False)

    LOGGER.info("Wrote MOS minute suite to %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
