"""Tail-error specialist overlay for HKG Tmax residual correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax.features.pruned_feature_policy import EVALUATION_ONLY_COLUMNS
from hkg_tmax.modeling.residual_models import (
    fit_lgbm_classifier,
    fit_lgbm_residual,
    predict_proba_with_fitted,
    predict_with_fitted,
)
from hkg_tmax.modeling.selective_router import capped_correction


def build_tail_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["true_residual_c"] = pd.to_numeric(out["y_true_c"], errors="coerce") - pd.to_numeric(
        out["anchor_forecast_max_c"], errors="coerce"
    )
    out["abs_residual_c"] = out["true_residual_c"].abs()
    out["tail_150_label"] = (out["abs_residual_c"] >= 1.5).astype(int)
    out["tail_200_label"] = (out["abs_residual_c"] >= 2.0).astype(int)
    out["tail_positive_label"] = (out["true_residual_c"] > 0).astype(int)
    out["tail_sign_label"] = np.sign(out["true_residual_c"].fillna(0.0)).astype(int)
    out["tail_sample_weight"] = 1.0
    out.loc[out["abs_residual_c"] >= 1.5, "tail_sample_weight"] = 2.5
    out.loc[out["abs_residual_c"] >= 2.0, "tail_sample_weight"] = 4.0
    return out


def _validate_tail_features(feature_names: list[str]) -> None:
    forbidden = sorted(set(feature_names) & EVALUATION_ONLY_COLUMNS)
    if forbidden:
        raise ValueError(f"Tail model features include evaluation-only columns: {forbidden}")


def fit_tail_models(
    train: pd.DataFrame,
    feature_names: list[str],
    seed: int,
) -> dict[str, Any]:
    _validate_tail_features(feature_names)
    frame = build_tail_labels(train).dropna(subset=["y_true_c", "anchor_forecast_max_c"]).copy()
    features = [feature for feature in feature_names if feature in frame.columns]
    if not features:
        raise ValueError("No tail features are available for fit_tail_models")
    _, tail150 = fit_lgbm_classifier(frame, frame, features, "tail_150_label", seed, model_id="T1_tail150_classifier")
    _, tail200 = fit_lgbm_classifier(frame, frame, features, "tail_200_label", seed + 5, model_id="T2_tail200_classifier")
    _, tail_sign = fit_lgbm_classifier(frame, frame, features, "tail_positive_label", seed + 9, model_id="T3_tail_sign_classifier")
    weighted = frame.copy()
    weighted["residual_y_c"] = weighted["true_residual_c"] * weighted["tail_sample_weight"]
    _, tail_resid = fit_lgbm_residual(weighted, weighted, features, seed + 13)
    tail_resid.model_id = "T4_tail_residual_regressor"
    return {
        "features": features,
        "tail150_classifier": tail150,
        "tail200_classifier": tail200,
        "tail_sign_classifier": tail_sign,
        "tail_residual_regressor": tail_resid,
        "training_rows": int(len(frame)),
        "tail150_rate": float(frame["tail_150_label"].mean()) if len(frame) else 0.0,
        "tail200_rate": float(frame["tail_200_label"].mean()) if len(frame) else 0.0,
    }


def predict_tail_scores(frame: pd.DataFrame, tail_models: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    if not tail_models:
        out["tail150_probability"] = out.get("tail150_probability", 0.0)
        out["tail200_probability"] = out.get("tail200_probability", 0.0)
        out["tail_positive_probability"] = out.get("tail_positive_probability", 0.5)
        out["tail_residual_prediction_c"] = out.get("tail_residual_prediction_c", 0.0)
    else:
        out["tail150_probability"] = predict_proba_with_fitted(out, tail_models["tail150_classifier"])
        out["tail200_probability"] = predict_proba_with_fitted(out, tail_models["tail200_classifier"])
        out["tail_positive_probability"] = predict_proba_with_fitted(out, tail_models["tail_sign_classifier"])
        out["tail_residual_prediction_c"] = predict_with_fitted(out, tail_models["tail_residual_regressor"])
    out["tail_sign_probability"] = np.where(
        pd.to_numeric(out["tail_residual_prediction_c"], errors="coerce").fillna(0.0) >= 0,
        out["tail_positive_probability"],
        1.0 - out["tail_positive_probability"],
    )
    out["predicted_tail_benefit_c"] = (
        pd.to_numeric(out["tail150_probability"], errors="coerce").fillna(0.0)
        * pd.to_numeric(out["tail_sign_probability"], errors="coerce").fillna(0.0)
        * pd.to_numeric(out["tail_residual_prediction_c"], errors="coerce").abs().fillna(0.0)
    )
    return out


def apply_tail_overlay(
    valid: pd.DataFrame,
    base_router_predictions: pd.DataFrame,
    tail_models: dict[str, Any],
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    frame = base_router_predictions.copy()
    for column in valid.columns:
        if column not in frame.columns:
            frame[column] = valid[column].to_numpy()
    frame = predict_tail_scores(frame, tail_models)
    tail150_threshold = float(thresholds.get("tail150_probability", thresholds.get("tail_probability", 0.60)))
    sign_threshold = float(thresholds.get("tail_sign_probability", thresholds.get("sign_probability", 0.62)))
    min_abs_correction = float(thresholds.get("min_abs_tail_correction_c", 0.25))
    hard_abs_cap = float(thresholds.get("hard_abs_cap_c", 1.00))
    tail_pass = (
        pd.to_numeric(frame["tail150_probability"], errors="coerce").fillna(0.0) >= tail150_threshold
    )
    tail_pass &= pd.to_numeric(frame["tail_sign_probability"], errors="coerce").fillna(0.0) >= sign_threshold
    tail_pass &= pd.to_numeric(frame["tail_residual_prediction_c"], errors="coerce").abs().fillna(0.0) >= min_abs_correction
    router_benefit = pd.to_numeric(frame.get("router_expected_benefit_c", 0.0), errors="coerce").fillna(0.0)
    tail_pass &= pd.to_numeric(frame["predicted_tail_benefit_c"], errors="coerce").fillna(0.0) > router_benefit
    tail_correction = capped_correction(
        frame["tail_residual_prediction_c"],
        {
            "positive_cap_c": hard_abs_cap,
            "negative_cap_c": hard_abs_cap,
            "hard_abs_cap_c": hard_abs_cap,
        },
    )
    base_correction = pd.to_numeric(frame.get("residual_prediction_c", 0.0), errors="coerce").fillna(0.0).to_numpy()
    frame["tail_overlay_applied_flag"] = tail_pass.astype(int)
    frame["residual_prediction_c"] = np.where(tail_pass, tail_correction, base_correction)
    frame["prediction_c"] = pd.to_numeric(frame["anchor_forecast_max_c"], errors="coerce") + frame["residual_prediction_c"]
    frame["final_correction_source"] = np.where(
        tail_pass,
        "tail_overlay",
        np.where(pd.to_numeric(frame.get("router_applied_flag", 0), errors="coerce").fillna(0).astype(int).eq(1), "router", "zero"),
    )
    frame["model_id"] = "C3_tail_overlay_router"
    frame["model_family"] = "tail_overlay_router"
    if "y_true_c" in frame:
        raw_abs = (pd.to_numeric(frame["y_true_c"], errors="coerce") - pd.to_numeric(frame["anchor_forecast_max_c"], errors="coerce")).abs()
        model_abs = (pd.to_numeric(frame["y_true_c"], errors="coerce") - pd.to_numeric(frame["prediction_c"], errors="coerce")).abs()
        frame["abs_improvement_vs_raw_c"] = raw_abs - model_abs
        frame["helped_vs_raw_flag"] = (frame["abs_improvement_vs_raw_c"] > 1e-9).astype(int)
        frame["worsened_vs_raw_flag"] = (frame["abs_improvement_vs_raw_c"] < -1e-9).astype(int)
    return frame
