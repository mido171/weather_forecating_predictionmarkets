from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .train_quantiles import avg_pinball


@dataclass
class GateModelPack:
    model: lgb.LGBMRegressor
    feature_cols: list[str]
    medians: dict[str, float]


def blend_quantiles(ml_q: pd.DataFrame, knn_q: pd.DataFrame, alpha: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=ml_q.index)
    a = np.clip(alpha.astype(float), 0.0, 1.0)
    for c in ml_q.columns:
        out[c] = a * ml_q[c].to_numpy(dtype=float) + (1.0 - a) * knn_q[c].to_numpy(dtype=float)
    return out


def compute_alpha_oracle(
    y: np.ndarray,
    ml_q: pd.DataFrame,
    knn_q: pd.DataFrame,
    quantiles: list[float],
    alpha_grid: np.ndarray,
) -> np.ndarray:
    y = y.astype(float)
    alpha = np.zeros(len(y), dtype=float)
    weights = np.array([2.0 if abs(q - 0.5) < 1e-9 or abs(q - 0.9) < 1e-9 or abs(q - 0.1) < 1e-9 else 1.0 for q in quantiles], dtype=float)

    ml_arr = ml_q[[f"q_{q:.3f}" for q in quantiles]].to_numpy(dtype=float)
    knn_arr = knn_q[[f"q_{q:.3f}" for q in quantiles]].to_numpy(dtype=float)

    for i in range(len(y)):
        best_a = 0.5
        best_loss = np.inf
        for a in alpha_grid:
            pred = a * ml_arr[i] + (1.0 - a) * knn_arr[i]
            e = y[i] - pred
            losses = np.maximum(np.array(quantiles) * e, (np.array(quantiles) - 1.0) * e)
            loss = float(np.sum(losses * weights) / np.sum(weights))
            if loss < best_loss:
                best_loss = loss
                best_a = float(a)
        alpha[i] = best_a
    return alpha


def build_gate_features(base_rows: pd.DataFrame, ml_q: pd.DataFrame, knn_q: pd.DataFrame, knn_trust: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=base_rows.index)
    trust_cols = [
        "knn_dist_min",
        "knn_dist_p10",
        "knn_dist_mean",
        "knn_weighted_iqr",
        "knn_weighted_mad",
        "knn_effective_k",
        "knn_support_span",
        "knn_entropy",
    ]
    for c in trust_cols:
        out[c] = pd.to_numeric(knn_trust.get(c), errors="coerce")

    out["abs_q50_gap"] = (ml_q["q_0.500"] - knn_q["q_0.500"]).abs()
    out["abs_q90_gap"] = (ml_q["q_0.900"] - knn_q["q_0.900"]).abs()
    out["cutoff_minutes"] = pd.to_numeric(base_rows["cutoff_minutes"], errors="coerce")
    out["doy_sin"] = pd.to_numeric(base_rows["doy_sin"], errors="coerce")
    out["doy_cos"] = pd.to_numeric(base_rows["doy_cos"], errors="coerce")

    out["cloud_bin"] = (pd.to_numeric(base_rows.get("clds_norm"), errors="coerce") >= 0.6).astype(float)
    wdir = pd.to_numeric(base_rows.get("wdir"), errors="coerce")
    out["wind_e_se"] = ((wdir >= 70) & (wdir <= 160)).astype(float)
    out["wind_w_nw"] = ((wdir >= 260) & (wdir <= 340)).astype(float)
    out["precip_flag"] = pd.to_numeric(base_rows.get("precip_flag"), errors="coerce").fillna(0.0)
    return out


def train_gate_model(features: pd.DataFrame, alpha_oracle: np.ndarray, random_seed: int) -> GateModelPack:
    cols = list(features.columns)
    med = {c: float(pd.to_numeric(features[c], errors="coerce").median()) for c in cols}
    x = features.copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(med[c])

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=240,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=4.0,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(x, alpha_oracle)
    return GateModelPack(model=model, feature_cols=cols, medians=med)


def predict_gate_alpha(pack: GateModelPack, features: pd.DataFrame) -> np.ndarray:
    x = features[pack.feature_cols].copy()
    for c in pack.feature_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(pack.medians[c])
    alpha = pack.model.predict(x)
    return np.clip(alpha, 0.0, 1.0)


def gate_diagnostics(
    base_rows: pd.DataFrame,
    alpha_pred: np.ndarray,
    alpha_oracle: np.ndarray | None,
    ml_q: pd.DataFrame,
    knn_q: pd.DataFrame,
    blend_q: pd.DataFrame,
    quantiles: list[float],
) -> dict[str, Any]:
    diag: dict[str, Any] = {
        "alpha_mean": float(np.nanmean(alpha_pred)),
        "alpha_std": float(np.nanstd(alpha_pred)),
        "alpha_min": float(np.nanmin(alpha_pred)),
        "alpha_max": float(np.nanmax(alpha_pred)),
    }
    if alpha_oracle is not None and len(alpha_oracle) == len(alpha_pred):
        diag["gate_mae_vs_oracle"] = float(np.mean(np.abs(alpha_pred - alpha_oracle)))

    y = base_rows["y_tmax"].to_numpy(dtype=float)
    ml_loss = avg_pinball(y, ml_q, quantiles)
    knn_loss = avg_pinball(y, knn_q, quantiles)
    blend_loss = avg_pinball(y, blend_q, quantiles)
    diag["avg_pinball_ml"] = ml_loss
    diag["avg_pinball_knn"] = knn_loss
    diag["avg_pinball_blend"] = blend_loss
    diag["blend_beats_both"] = bool(blend_loss < min(ml_loss, knn_loss))
    return diag


def gate_feature_importance(pack: GateModelPack) -> pd.DataFrame:
    booster = pack.model.booster_
    gain = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    total = float(np.sum(gain)) if np.sum(gain) > 0 else 1.0
    return pd.DataFrame(
        {
            "feature": names,
            "gain": gain.astype(float),
            "gain_contribution": gain.astype(float) / total,
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)
