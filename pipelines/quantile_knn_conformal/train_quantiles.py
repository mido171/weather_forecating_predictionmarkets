from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class QuantileModelPack:
    models: dict[float, lgb.LGBMRegressor]
    feature_medians: dict[str, float]
    feature_cols: list[str]
    quantiles: list[float]
    params: dict[str, Any]


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def avg_pinball(y_true: np.ndarray, pred_df: pd.DataFrame, quantiles: list[float]) -> float:
    vals = []
    for q in quantiles:
        vals.append(pinball_loss(y_true, pred_df[f"q_{q:.3f}"].to_numpy(dtype=float), q))
    return float(np.mean(vals))


def repair_quantile_crossings(pred_df: pd.DataFrame, quantiles: list[float]) -> tuple[pd.DataFrame, int]:
    out = pred_df.copy()
    cols = [f"q_{q:.3f}" for q in quantiles]
    arr = out[cols].to_numpy(dtype=float)
    before = arr.copy()
    arr_sorted = np.sort(arr, axis=1)
    out[cols] = arr_sorted
    repaired_rows = int(np.sum(np.any(np.abs(before - arr_sorted) > 1e-12, axis=1)))
    return out, repaired_rows


def _decision_slice(df: pd.DataFrame, decision_stockholm_minutes: int = 1140) -> pd.DataFrame:
    cands = df[df["stockholm_minutes"] >= decision_stockholm_minutes].copy()
    if cands.empty:
        return cands
    idx = cands.sort_values(["target_date_local", "valid_time_utc"]).groupby("target_date_local")["valid_time_utc"].idxmin()
    return cands.loc[idx].copy()


def _fit_one_quantile(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    feature_cols: list[str],
    alpha: float,
    params: dict[str, Any],
    feature_medians: dict[str, float],
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=int(params.get("n_estimators", 300)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        num_leaves=int(params.get("num_leaves", 31)),
        min_child_samples=int(params.get("min_child_samples", 80)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=int(params.get("random_seed", 42)),
        n_jobs=int(params.get("n_jobs", -1)),
        verbosity=-1,
    )

    x_train = train_df[feature_cols].copy()
    for c, med in feature_medians.items():
        x_train[c] = x_train[c].fillna(med)
    y_train = train_df["y_tmax"].to_numpy(dtype=float)

    fit_kwargs: dict[str, Any] = {}
    if val_df is not None and not val_df.empty:
        x_val = val_df[feature_cols].copy()
        for c, med in feature_medians.items():
            x_val[c] = x_val[c].fillna(med)
        y_val = val_df["y_tmax"].to_numpy(dtype=float)
        fit_kwargs["eval_set"] = [(x_val, y_val)]
        fit_kwargs["eval_metric"] = "quantile"
        fit_kwargs["callbacks"] = [lgb.early_stopping(40, verbose=False)]

    model.fit(x_train, y_train, **fit_kwargs)
    return model


def tune_quantile_params(
    pre2022_df: pd.DataFrame,
    feature_cols: list[str],
    quantiles: list[float],
    inner_folds: list[tuple[str, str, str]],
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = [
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 120,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.5,
            "n_jobs": -1,
            "random_seed": random_seed,
        },
        {
            "n_estimators": 420,
            "learning_rate": 0.035,
            "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "n_jobs": -1,
            "random_seed": random_seed,
        },
    ]

    rows: list[dict[str, Any]] = []
    best_score = np.inf
    best_params = candidates[0]

    for cand_idx, cand in enumerate(candidates):
        fold_scores: list[float] = []
        fold_mae: list[float] = []
        fold_crossings: list[int] = []

        for fold_id, (train_end, val_start, val_end) in enumerate(inner_folds, start=1):
            tr = pre2022_df[pd.to_datetime(pre2022_df["target_date_local"]) <= pd.Timestamp(train_end)].copy()
            va = pre2022_df[(pd.to_datetime(pre2022_df["target_date_local"]) >= pd.Timestamp(val_start)) & (pd.to_datetime(pre2022_df["target_date_local"]) <= pd.Timestamp(val_end))].copy()

            tr_dec = _decision_slice(tr)
            va_dec = _decision_slice(va)
            if tr_dec.empty or va_dec.empty:
                continue

            feature_medians = {c: float(tr_dec[c].median()) for c in feature_cols}
            models: dict[float, lgb.LGBMRegressor] = {}
            pred = pd.DataFrame(index=va_dec.index)
            for q in quantiles:
                models[q] = _fit_one_quantile(tr_dec, va_dec, feature_cols, q, cand, feature_medians)
                xv = va_dec[feature_cols].copy()
                for c, med in feature_medians.items():
                    xv[c] = xv[c].fillna(med)
                pred[f"q_{q:.3f}"] = models[q].predict(xv)

            raw_crossings = int(np.sum(np.any(np.diff(pred[[f"q_{q:.3f}" for q in quantiles]].to_numpy(dtype=float), axis=1) < 0, axis=1)))
            pred_repaired, _ = repair_quantile_crossings(pred, quantiles)
            y = va_dec["y_tmax"].to_numpy(dtype=float)
            score = avg_pinball(y, pred_repaired, quantiles)
            mae = float(np.mean(np.abs(y - pred_repaired["q_0.500"].to_numpy(dtype=float))))

            fold_scores.append(score)
            fold_mae.append(mae)
            fold_crossings.append(raw_crossings)

            rows.append(
                {
                    "candidate_id": cand_idx,
                    "fold_id": fold_id,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "avg_pinball": score,
                    "mae_q50": mae,
                    "raw_crossing_rows": raw_crossings,
                }
            )

        if not fold_scores:
            continue

        mean_score = float(np.mean(fold_scores))
        mean_mae = float(np.mean(fold_mae))
        mean_cross = float(np.mean(fold_crossings))
        rows.append(
            {
                "candidate_id": cand_idx,
                "fold_id": "aggregate",
                "train_end": None,
                "val_start": None,
                "val_end": None,
                "avg_pinball": mean_score,
                "mae_q50": mean_mae,
                "raw_crossing_rows": mean_cross,
            }
        )

        if mean_score < best_score:
            best_score = mean_score
            best_params = cand

    return best_params, pd.DataFrame(rows)


def train_quantile_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    quantiles: list[float],
    params: dict[str, Any],
) -> QuantileModelPack:
    feature_medians = {c: float(train_df[c].median()) for c in feature_cols}
    models: dict[float, lgb.LGBMRegressor] = {}
    for q in quantiles:
        models[q] = _fit_one_quantile(train_df, None, feature_cols, q, params, feature_medians)
    return QuantileModelPack(
        models=models,
        feature_medians=feature_medians,
        feature_cols=feature_cols,
        quantiles=quantiles,
        params=params,
    )


def predict_quantile_models(pack: QuantileModelPack, df: pd.DataFrame) -> pd.DataFrame:
    x = df[pack.feature_cols].copy()
    for c, med in pack.feature_medians.items():
        x[c] = x[c].fillna(med)

    pred = pd.DataFrame(index=df.index)
    for q in pack.quantiles:
        pred[f"q_{q:.3f}"] = pack.models[q].predict(x)
    return pred


def feature_importance_table(pack: QuantileModelPack) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q, model in pack.models.items():
        imp = model.booster_.feature_importance(importance_type="gain")
        names = model.booster_.feature_name()
        total = float(np.sum(imp)) if np.sum(imp) > 0 else 1.0
        for n, v in zip(names, imp, strict=False):
            rows.append(
                {
                    "quantile": q,
                    "feature": n,
                    "gain": float(v),
                    "gain_contribution": float(v / total),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["quantile", "gain"], ascending=[True, False]).reset_index(drop=True)
