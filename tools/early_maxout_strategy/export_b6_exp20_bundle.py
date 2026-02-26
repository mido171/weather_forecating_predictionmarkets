from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


UTC = timezone.utc


def _utc_now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BetaCalibratorParams:
    epsilon: float
    coef_log_p: float
    coef_log1m_p: float
    intercept: float


@dataclass(frozen=True)
class POnshoreParams:
    feature_cols: List[str]
    imputer_medians: Dict[str, float]
    coef: List[float]
    intercept: float


@dataclass(frozen=True)
class BundleMeta:
    created_at_utc: str
    station_id: str
    model_name: str
    feature_store_path: str
    residual_model_path: str
    base_model_params: Dict[str, object]
    cp_penalty_quantile: float
    cp_penalty_value: float
    splits: Dict[str, object]
    validation: Dict[str, object]


def _clip_probs(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _beta_fit(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> BetaCalibratorParams:
    p = _clip_probs(p, eps)
    X = np.column_stack([np.log(p), np.log1p(-p)])
    lr = LogisticRegression(max_iter=200)
    lr.fit(X, y.astype(int))
    coef = lr.coef_.reshape(-1)
    intercept = float(lr.intercept_.reshape(-1)[0])
    return BetaCalibratorParams(
        epsilon=float(eps),
        coef_log_p=float(coef[0]),
        coef_log1m_p=float(coef[1]),
        intercept=float(intercept),
    )


def _beta_apply(params: BetaCalibratorParams, p: np.ndarray) -> np.ndarray:
    p = _clip_probs(np.asarray(p, dtype=float), params.epsilon)
    X = np.column_stack([np.log(p), np.log1p(-p)])
    z = X @ np.array([params.coef_log_p, params.coef_log1m_p], dtype=float) + params.intercept
    return 1.0 / (1.0 + np.exp(-z))


def _fit_p_onshore(df: pd.DataFrame, train_mask: np.ndarray) -> Tuple[pd.Series, POnshoreParams]:
    # Mirrors tools/early_maxout_strategy/run_hit1830_v6_suite.py::_add_p_onshore
    if "mos_wdr_mean_models" not in df.columns:
        s = pd.Series(np.nan, index=df.index)
        return s, POnshoreParams(feature_cols=[], imputer_medians={}, coef=[float("nan")] * 0, intercept=float("nan"))

    wdr = df["mos_wdr_mean_models"].to_numpy(dtype=float)
    labels = ((wdr >= 30) & (wdr <= 170)).astype(int)
    feat_cols = [c for c in ["mos_u_mean", "mos_v_mean", "mos_wsp_mean_models", "doy_sin", "doy_cos"] if c in df.columns]
    if not feat_cols:
        s = pd.Series(np.nan, index=df.index)
        return s, POnshoreParams(feature_cols=[], imputer_medians={}, coef=[float("nan")] * 0, intercept=float("nan"))

    X = df[feat_cols]
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(X)
    X_train = X_all[train_mask]
    y_train = labels[train_mask]

    med = {c: float(v) for c, v in zip(feat_cols, imputer.statistics_)}

    if np.unique(y_train).size < 2:
        p = float(np.mean(y_train)) if y_train.size else float("nan")
        return pd.Series(np.full(len(df), p, dtype=float), index=df.index), POnshoreParams(
            feature_cols=feat_cols, imputer_medians=med, coef=[], intercept=float(p)
        )

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    probs = lr.predict_proba(X_all)[:, 1]
    coef = lr.coef_.reshape(-1).astype(float).tolist()
    intercept = float(lr.intercept_.reshape(-1)[0])
    return pd.Series(probs, index=df.index), POnshoreParams(
        feature_cols=feat_cols, imputer_medians=med, coef=coef, intercept=intercept
    )


def _compute_cp_exists(df: pd.DataFrame, train_mask: np.ndarray, penalty_quantile: float) -> Tuple[pd.Series, float]:
    if "cp_improvement" not in df.columns:
        return pd.Series(np.nan, index=df.index), float("nan")
    improv = pd.to_numeric(df["cp_improvement"], errors="coerce").to_numpy(dtype=float)
    train_improv = improv[train_mask]
    train_improv = train_improv[np.isfinite(train_improv)]
    penalty = float(np.quantile(train_improv, penalty_quantile)) if train_improv.size else 0.0
    cp_exists = (improv > penalty).astype(float)
    return pd.Series(cp_exists, index=df.index), penalty


def _compute_tmp_bias_features(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    # Mirrors tools/early_maxout_strategy/run_hit1830_v6_suite.py::_compute_tmp_bias_features
    df = df.copy()
    df = df.sort_values("target_date_local")
    for model in ["gfs", "nam"]:
        col = f"mos_tmax_{model}"
        if col not in df.columns:
            continue
        err = (df[col].astype(float) - df["tmax_full"].astype(float)).shift(1)
        bias = err.ewm(alpha=alpha, adjust=False).mean()
        df[f"mos_tmax_bias_{model}_a{alpha:.3f}"] = bias
        df[f"mos_tmax_bc_{model}_a{alpha:.3f}"] = df[col].astype(float) - bias
    gfs_bc = df.get(f"mos_tmax_bc_gfs_a{alpha:.3f}")
    nam_bc = df.get(f"mos_tmax_bc_nam_a{alpha:.3f}")
    if gfs_bc is not None and nam_bc is not None:
        df[f"mos_tmax_mean_bc_a{alpha:.3f}"] = pd.concat([gfs_bc, nam_bc], axis=1).mean(axis=1)
    return df


def _train_base_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_list: List[str],
) -> lgb.Booster:
    # Mirrors the base model in B6_EXP20 (GAM proxy): shallow + monotone constraints.
    constraint_map = {
        "minutes_since_max": 1,
        "drop_from_max": 1,
        "plateau_frac_0p2_last120": 1,
        "heat_gap": -1,
        "heat_gap_norm": -1,
        "slope_60m": -1,
    }
    monotone_constraints = [constraint_map.get(c, -1 if c.startswith("gap_bc") else 0) for c in feature_list]
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 3,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbose": -1,
        "monotone_constraints": monotone_constraints,
    }
    train_set = lgb.Dataset(X_train, label=y_train.astype(int))
    val_set = lgb.Dataset(X_val, label=y_val.astype(int), reference=train_set)
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a deployable bundle for B6_EXP20_GAM_RESIDUAL (base+residual+cal).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: artifacts/model_bundles/hit1830_v6/B6_EXP20_GAM_RESIDUAL/<timestamp>)",
    )
    args = parser.parse_args()

    repo = _repo_root()
    station_id = "KMIA"
    model_name = "B6_EXP20_GAM_RESIDUAL"

    model_dir = repo / "artifacts" / "experiments" / station_id / "early_maxout_strategy" / "B6" / model_name
    feature_store = repo / "cache" / "hit1830_v6_features.parquet"
    residual_model_path = model_dir / "model.txt"
    preds_val_path = model_dir / "preds_val.parquet"
    preds_test_path = model_dir / "preds_test.parquet"
    feature_list_path = model_dir / "features.json"

    feature_list = json.loads(feature_list_path.read_text(encoding="utf-8"))

    # Guardrail: refuse to export if the model feature list contains obvious future/label columns.
    banned_feature_cols = {
        # Labels / future-only
        "y_hit_by_cutoff",
        "y_exceed_future",
        "exceed_time_min",
        # Full-day truth at cutoff (must never be used directly as a feature)
        "tmax_full",
        "tmin_full",
        "range_full",
        "minutes_since_tmax",
        "tmax_time_local_minute",
    }
    banned_in_features = sorted(set(feature_list).intersection(banned_feature_cols))
    if banned_in_features:
        raise SystemExit(f"Banned feature columns detected in features.json: {banned_in_features}")

    missing = {
        "cp_exists",
        "mos_tmax_bias_gfs_a0.020",
        "mos_tmax_bc_gfs_a0.020",
        "mos_tmax_bias_nam_a0.020",
        "mos_tmax_bc_nam_a0.020",
        "mos_tmax_mean_bc_a0.020",
        "mos_tmax_bias_gfs_a0.050",
        "mos_tmax_bc_gfs_a0.050",
        "mos_tmax_bias_nam_a0.050",
        "mos_tmax_bc_nam_a0.050",
        "mos_tmax_mean_bc_a0.050",
    }

    read_cols = [c for c in feature_list if c not in missing]
    read_cols += [
        "target_date_local",
        "y_hit_by_cutoff",
        "cp_improvement",
        "tmax_full",
        "mos_wdr_mean_models",
        "mos_u_mean",
        "mos_v_mean",
        "mos_wsp_mean_models",
        "doy_sin",
        "doy_cos",
    ]
    read_cols = sorted(set(read_cols))

    df = pd.read_parquet(feature_store, columns=read_cols)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date

    years = pd.to_datetime(df["target_date_local"]).dt.year.to_numpy()
    train_mask = years <= 2019
    val_mask = (years >= 2020) & (years <= 2022)
    test_mask = (years >= 2023) & (years <= 2025)

    y = df["y_hit_by_cutoff"].to_numpy(dtype=int)

    # p_onshore overwrite (matches training)
    p_onshore, p_onshore_params = _fit_p_onshore(df, train_mask)
    df["p_onshore"] = p_onshore

    # cp_exists + gate cp columns (matches training)
    cp_penalty_quantile = 0.6
    cp_exists, cp_penalty_value = _compute_cp_exists(df, train_mask, cp_penalty_quantile)
    df["cp_exists"] = cp_exists
    for col in ["cp_time_since", "cp_drop_magnitude", "cp_slope_before_v6", "cp_slope_after_v6"]:
        if col in df.columns:
            df.loc[df["cp_exists"] < 0.5, col] = np.nan

    # tmp bias features (matches training)
    df = _compute_tmp_bias_features(df, 0.02)
    df = _compute_tmp_bias_features(df, 0.05)

    X = df[feature_list]
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X[train_mask])
    X_val = imputer.transform(X[val_mask])
    X_test = imputer.transform(X[test_mask])

    base_model = _train_base_model(X_train, y[train_mask], X_val, y[val_mask], feature_list)
    residual_model = lgb.Booster(model_file=str(residual_model_path))

    base_raw_val = _logit(base_model.predict(X_val))
    base_raw_test = _logit(base_model.predict(X_test))
    resid_raw_val = residual_model.predict(X_val, raw_score=True)
    resid_raw_test = residual_model.predict(X_test, raw_score=True)

    p_val_raw = _sigmoid(base_raw_val + resid_raw_val)
    p_test_raw = _sigmoid(base_raw_test + resid_raw_test)

    preds_val = pd.read_parquet(preds_val_path)
    preds_test = pd.read_parquet(preds_test_path)
    preds_val["target_date_local"] = pd.to_datetime(preds_val["target_date_local"]).dt.date
    preds_test["target_date_local"] = pd.to_datetime(preds_test["target_date_local"]).dt.date

    dates_val = df.loc[val_mask, "target_date_local"].to_numpy()
    dates_test = df.loc[test_mask, "target_date_local"].to_numpy()
    p_val_raw_ref = preds_val.set_index("target_date_local").loc[dates_val]["p_raw"].to_numpy(dtype=float)
    p_test_raw_ref = preds_test.set_index("target_date_local").loc[dates_test]["p_raw"].to_numpy(dtype=float)

    # Fit beta calibrator on val (matches training)
    beta_params = _beta_fit(preds_val["y_true"].to_numpy(dtype=int), p_val_raw, eps=1e-6)
    p_val_cal = _beta_apply(beta_params, p_val_raw)
    p_test_cal = _beta_apply(beta_params, p_test_raw)

    p_val_cal_ref = preds_val.set_index("target_date_local").loc[dates_val]["p_cal"].to_numpy(dtype=float)
    p_test_cal_ref = preds_test.set_index("target_date_local").loc[dates_test]["p_cal"].to_numpy(dtype=float)

    def _err(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        e = np.abs(a - b)
        return {
            "max_abs": float(np.max(e)),
            "mean_abs": float(np.mean(e)),
            "p99_abs": float(np.quantile(e, 0.99)),
        }

    validation = {
        "p_val_raw": _err(p_val_raw, p_val_raw_ref),
        "p_test_raw": _err(p_test_raw, p_test_raw_ref),
        "p_val_cal": _err(p_val_cal, p_val_cal_ref),
        "p_test_cal": _err(p_test_cal, p_test_cal_ref),
    }

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        repo / "artifacts" / "model_bundles" / "hit1830_v6" / model_name / _utc_now_tag()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write bundle files
    base_model_path = out_dir / "base_model.txt"
    residual_out_path = out_dir / "residual_model.txt"
    features_out_path = out_dir / "features.json"

    base_model.save_model(str(base_model_path))
    shutil.copyfile(residual_model_path, residual_out_path)
    shutil.copyfile(feature_list_path, features_out_path)

    med = {name: float(v) for name, v in zip(feature_list, imputer.statistics_)}
    (out_dir / "imputer_medians.json").write_text(json.dumps(med, indent=2, sort_keys=True), encoding="utf-8")

    (out_dir / "beta_calibrator.json").write_text(json.dumps(asdict(beta_params), indent=2), encoding="utf-8")
    (out_dir / "p_onshore_lr.json").write_text(json.dumps(asdict(p_onshore_params), indent=2), encoding="utf-8")

    meta = BundleMeta(
        created_at_utc=datetime.now(UTC).isoformat(),
        station_id=station_id,
        model_name=model_name,
        feature_store_path=str(feature_store),
        residual_model_path=str(residual_model_path),
        base_model_params={
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 3,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "seed": 42,
            "early_stopping_rounds": 100,
        },
        cp_penalty_quantile=float(cp_penalty_quantile),
        cp_penalty_value=float(cp_penalty_value),
        splits={
            "train_years_le": 2019,
            "val_years": [2020, 2022],
            "test_years": [2023, 2025],
            "n_rows": int(len(df)),
            "n_train": int(np.sum(train_mask)),
            "n_val": int(np.sum(val_mask)),
            "n_test": int(np.sum(test_mask)),
        },
        validation=validation,
    )
    (out_dir / "bundle_meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    # Human-readable validation summary
    lines = []
    lines.append("# B6_EXP20 Bundle Export Validation\n")
    lines.append(f"- Created: `{meta.created_at_utc}`")
    lines.append(f"- Output dir: `{out_dir}`")
    lines.append(f"- Feature store: `{feature_store}`")
    lines.append(f"- Residual model: `{residual_model_path}`\n")
    lines.append("## Reproduction Check (vs saved preds_val/preds_test)\n")
    for k, v in validation.items():
        lines.append(f"- `{k}`: max_abs={v['max_abs']:.12g}, mean_abs={v['mean_abs']:.12g}, p99_abs={v['p99_abs']:.12g}")
    lines.append("")
    (out_dir / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote bundle:", out_dir)
    print("Validation:", json.dumps(validation, indent=2))

    # Fail closed if anything is materially different from the saved preds.
    # We allow tiny floating noise (~1e-16) from sklearn/lightgbm math.
    tol = 1e-12
    for k, v in validation.items():
        if v["max_abs"] > tol:
            raise SystemExit(f"Non-zero reproduction error for {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
