"""Selective correction router for HKG Tmax residual forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax.evaluation.ablation_runner import FoldSpec
from hkg_tmax.evaluation.metrics import score_arrays
from hkg_tmax.features.pruned_feature_policy import EVALUATION_ONLY_COLUMNS
from hkg_tmax.modeling.residual_models import (
    fit_catboost_residual,
    fit_lgbm_benefit_regressor,
    fit_lgbm_classifier,
    fit_lgbm_residual,
    fit_robust_linear_residual,
    predict_proba_with_fitted,
    predict_with_fitted,
)

CANDIDATE_RESID_COLUMNS = [
    "candidate_resid_lgbm_a3_c",
    "candidate_resid_lgbm_pruned_full_c",
    "candidate_resid_catboost_c",
    "candidate_resid_linear_c",
]


def make_inner_folds(train: pd.DataFrame) -> list[FoldSpec]:
    """Build chronological inner folds for router OOF candidate generation."""
    if train.empty or "target_date" not in train:
        return []
    frame = train.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["target_date"]).sort_values("target_date")
    if frame.empty:
        return []
    start = frame["target_date"].min()
    years = sorted(frame["target_date"].dt.year.unique().tolist())
    folds: list[FoldSpec] = []
    for year in years:
        if year - int(start.year) < 5:
            continue
        valid_start = pd.Timestamp(year=year, month=1, day=1)
        valid_end = pd.Timestamp(year=year, month=12, day=31)
        train_end = valid_start - pd.Timedelta(days=1)
        if not frame["target_date"].between(valid_start, valid_end).any():
            continue
        folds.append(
            FoldSpec(
                fold_id=f"inner_{year}",
                train_start=start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                stage="router_inner_oof",
            )
        )
    return folds


def add_candidate_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CANDIDATE_RESID_COLUMNS:
        if column not in out:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    if "candidate_resid_ensemble_c" not in out:
        out["candidate_resid_ensemble_c"] = out[CANDIDATE_RESID_COLUMNS].mean(axis=1)
    out["candidate_resid_ensemble_c"] = pd.to_numeric(out["candidate_resid_ensemble_c"], errors="coerce").fillna(0.0)
    out["candidate_abs_resid_ensemble_c"] = out["candidate_resid_ensemble_c"].abs()
    out["candidate_resid_std_c"] = out[CANDIDATE_RESID_COLUMNS].std(axis=1).fillna(0.0)
    signs = np.sign(out[CANDIDATE_RESID_COLUMNS].to_numpy(dtype=float))
    ensemble_sign = np.sign(out["candidate_resid_ensemble_c"].to_numpy(dtype=float)).reshape(-1, 1)
    out["candidate_resid_sign_agreement_count"] = ((signs == ensemble_sign) & (signs != 0)).sum(axis=1)
    out["candidate_positive_correction_flag"] = (out["candidate_resid_ensemble_c"] > 0).astype(int)
    out["candidate_negative_correction_flag"] = (out["candidate_resid_ensemble_c"] < 0).astype(int)
    out["candidate_correction_magnitude_bin"] = pd.cut(
        out["candidate_abs_resid_ensemble_c"],
        bins=[-np.inf, 0.10, 0.25, 0.50, 0.75, np.inf],
        labels=["le010", "010_025", "025_050", "050_075", "gt075"],
    ).astype(str)
    return out


def build_oof_candidate_corrections(
    train: pd.DataFrame,
    feature_names: list[str],
    seed: int,
) -> pd.DataFrame:
    """Build inner-fold OOF candidate residual corrections from training rows."""
    rows: list[pd.DataFrame] = []
    for fold in make_inner_folds(train):
        train_part = train[
            pd.to_datetime(train["target_date"]).between(fold.train_start, fold.train_end)
            & train["label_source"].eq("label_core")
        ].copy()
        valid_part = train[pd.to_datetime(train["target_date"]).between(fold.valid_start, fold.valid_end)].copy()
        if len(train_part) < 100 or valid_part.empty:
            continue
        base = valid_part.copy()
        pred_a3, _ = fit_lgbm_residual(train_part, valid_part, feature_names, seed)
        pred_full, _ = fit_lgbm_residual(train_part, valid_part, feature_names, seed + 11)
        pred_cat, _ = fit_catboost_residual(train_part, valid_part, feature_names, seed + 19)
        pred_linear, _ = fit_robust_linear_residual(train_part, valid_part, feature_names)
        base["candidate_resid_lgbm_a3_c"] = np.clip(pred_a3, -3.0, 3.0)
        base["candidate_resid_lgbm_pruned_full_c"] = np.clip(pred_full, -3.0, 3.0)
        base["candidate_resid_catboost_c"] = np.clip(pred_cat, -3.0, 3.0)
        base["candidate_resid_linear_c"] = np.clip(pred_linear, -3.0, 3.0)
        base["inner_fold_id"] = fold.fold_id
        base["inner_train_start"] = fold.train_start
        base["inner_train_end"] = fold.train_end
        rows.append(add_candidate_meta_features(base))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_router_labels(oof: pd.DataFrame) -> pd.DataFrame:
    out = add_candidate_meta_features(oof)
    out["true_residual_c"] = pd.to_numeric(out["y_true_c"], errors="coerce") - pd.to_numeric(
        out["anchor_forecast_max_c"], errors="coerce"
    )
    out["candidate_residual_c"] = pd.to_numeric(out["candidate_resid_ensemble_c"], errors="coerce").fillna(0.0)
    out["raw_abs_error_c"] = out["true_residual_c"].abs()
    out["candidate_abs_error_c"] = (out["true_residual_c"] - out["candidate_residual_c"]).abs()
    out["benefit_c"] = out["raw_abs_error_c"] - out["candidate_abs_error_c"]
    out["apply_label"] = (out["benefit_c"] > 0).astype(int)
    out["strong_apply_label"] = (out["benefit_c"] >= 0.10).astype(int)
    out["sign_label"] = np.sign(out["true_residual_c"].fillna(0.0)).astype(int)
    out["candidate_sign_correct"] = (
        np.sign(out["candidate_residual_c"].fillna(0.0)).astype(int).eq(out["sign_label"]) & out["sign_label"].ne(0)
    ).astype(int)
    return out


def _validate_router_features(router_features: list[str]) -> None:
    forbidden = sorted(set(router_features) & EVALUATION_ONLY_COLUMNS)
    if forbidden:
        raise ValueError(f"Router features include evaluation-only columns: {forbidden}")


def fit_router_models(
    oof: pd.DataFrame,
    router_features: list[str],
    seed: int,
) -> dict[str, Any]:
    _validate_router_features(router_features)
    frame = build_router_labels(oof).dropna(subset=["y_true_c", "anchor_forecast_max_c"]).copy()
    features = [feature for feature in router_features if feature in frame.columns]
    if not features:
        raise ValueError("No router features are available for fit_router_models")
    _, apply_model = fit_lgbm_classifier(
        frame,
        frame,
        features,
        "apply_label",
        seed,
        model_id="R1_apply_classifier",
    )
    _, sign_model = fit_lgbm_classifier(
        frame,
        frame,
        features,
        "candidate_sign_correct",
        seed + 3,
        model_id="R2_sign_classifier",
    )
    _, benefit_model = fit_lgbm_benefit_regressor(
        frame,
        frame,
        features,
        seed + 7,
        label_column="benefit_c",
        model_id="R3_expected_benefit_regressor",
    )
    return {
        "features": features,
        "apply_classifier": apply_model,
        "sign_classifier": sign_model,
        "benefit_regressor": benefit_model,
        "training_rows": int(len(frame)),
        "training_stage_counts": frame.get("stage", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
    }


def predict_router_scores(frame: pd.DataFrame, router_models: dict[str, Any]) -> pd.DataFrame:
    out = add_candidate_meta_features(frame)
    if not router_models:
        out["router_apply_probability"] = out.get("router_apply_probability", 0.0)
        out["router_sign_probability"] = out.get("router_sign_probability", 0.0)
        out["router_expected_benefit_c"] = out.get("router_expected_benefit_c", 0.0)
        return out
    features = router_models["features"]
    out["router_apply_probability"] = predict_proba_with_fitted(out, router_models["apply_classifier"])
    out["router_sign_probability"] = predict_proba_with_fitted(out, router_models["sign_classifier"])
    out["router_expected_benefit_c"] = predict_with_fitted(out, router_models["benefit_regressor"])
    return out


def _threshold_grid(config: dict[str, Any]) -> list[dict[str, float]]:
    router = config.get("router", config)
    rows: list[dict[str, float]] = []
    for threshold_benefit in router.get("threshold_benefit_grid", [0.0]):
        for threshold_apply in router.get("threshold_apply_grid", [0.52]):
            for threshold_sign in router.get("threshold_sign_grid", [0.56]):
                for positive_cap in router.get("positive_cap_grid_c", [0.50]):
                    for negative_cap in router.get("negative_cap_grid_c", [0.35]):
                        for hard_abs_cap in router.get("hard_abs_cap_grid_c", [0.75]):
                            rows.append(
                                {
                                    "threshold_benefit": float(threshold_benefit),
                                    "threshold_apply": float(threshold_apply),
                                    "threshold_sign": float(threshold_sign),
                                    "positive_cap_c": float(positive_cap),
                                    "negative_cap_c": float(negative_cap),
                                    "hard_abs_cap_c": float(hard_abs_cap),
                                }
                            )
    return rows


def capped_correction(values: pd.Series | np.ndarray, thresholds: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    positive_cap = float(thresholds.get("positive_cap_c", 0.50))
    negative_cap = float(thresholds.get("negative_cap_c", 0.35))
    hard_abs_cap = float(thresholds.get("hard_abs_cap_c", 0.75))
    arr = np.clip(arr, -negative_cap, positive_cap)
    arr = np.clip(arr, -hard_abs_cap, hard_abs_cap)
    return arr


def _apply_rule(frame: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    rule = (
        pd.to_numeric(out["router_expected_benefit_c"], errors="coerce").fillna(-999.0)
        >= float(thresholds.get("threshold_benefit", 0.0))
    )
    rule &= (
        pd.to_numeric(out["router_apply_probability"], errors="coerce").fillna(0.0)
        >= float(thresholds.get("threshold_apply", 0.52))
    )
    rule &= (
        pd.to_numeric(out["router_sign_probability"], errors="coerce").fillna(0.0)
        >= float(thresholds.get("threshold_sign", 0.56))
    )
    correction = capped_correction(out["candidate_resid_ensemble_c"], thresholds)
    out["router_applied_flag"] = rule.astype(int)
    out["residual_prediction_c"] = np.where(rule, correction, 0.0)
    out["prediction_c"] = pd.to_numeric(out["anchor_forecast_max_c"], errors="coerce") + out["residual_prediction_c"]
    return out


def select_router_thresholds(
    oof: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame = build_router_labels(oof)
    if "router_apply_probability" not in frame:
        frame["router_apply_probability"] = frame["apply_label"]
    if "router_sign_probability" not in frame:
        frame["router_sign_probability"] = frame["candidate_sign_correct"]
    if "router_expected_benefit_c" not in frame:
        frame["router_expected_benefit_c"] = frame["benefit_c"]
    if "stage" in frame:
        selection = frame[frame["stage"].eq("rolling_validation") | frame["stage"].eq("router_inner_oof")].copy()
        if selection.empty:
            selection = frame[~frame["stage"].isin(["presealed_holdout", "sealed_confirmation"])].copy()
    else:
        selection = frame.copy()
    guardrails = config.get("router", config).get("no_harm_guardrails", {})
    min_apply = float(guardrails.get("min_apply_rate", 0.0))
    max_apply = float(guardrails.get("max_apply_rate", 1.0))
    rows: list[dict[str, Any]] = []
    for idx, thresholds in enumerate(_threshold_grid(config), start=1):
        scored = _apply_rule(selection, thresholds)
        metrics = score_arrays(
            pd.to_numeric(scored["y_true_c"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(scored["prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        apply_rate = float(scored["router_applied_flag"].mean()) if len(scored) else 0.0
        applied = build_router_labels(scored[scored["router_applied_flag"].eq(1)])
        wrong_sign_rate = float(1.0 - applied["candidate_sign_correct"].mean()) if not applied.empty else 0.0
        rows.append(
            {
                "threshold_id": f"router_grid_{idx:04d}",
                **thresholds,
                **metrics,
                "apply_rate": apply_rate,
                "wrong_sign_apply_rate": wrong_sign_rate,
                "apply_rate_gate_pass": min_apply <= apply_rate <= max_apply,
            }
        )
    selection_frame = pd.DataFrame(rows)
    eligible = selection_frame[selection_frame["apply_rate_gate_pass"]].copy()
    if eligible.empty:
        eligible = selection_frame.copy()
    best = eligible.sort_values(["mae", "rmse", "wrong_sign_apply_rate"], na_position="last").iloc[0].to_dict()
    best["selection_rows"] = selection_frame.to_dict(orient="records")
    best["selection_stage"] = "rolling_validation_or_router_inner_oof_only"
    best["sealed_rows_used_for_selection"] = False
    return best


def apply_selective_router(
    valid: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    router_models: dict[str, Any],
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    frame = valid.copy()
    if candidate_predictions is not None and not candidate_predictions.empty:
        for column in candidate_predictions.columns:
            if column not in frame.columns:
                frame[column] = candidate_predictions[column].to_numpy()
    frame = predict_router_scores(frame, router_models)
    out = _apply_rule(frame, thresholds)
    out["router_selected_threshold_id"] = thresholds.get("threshold_id", "router_threshold")
    out["model_id"] = "C2_selective_router"
    out["model_family"] = "selective_router"
    if "y_true_c" in out:
        raw_abs = (pd.to_numeric(out["y_true_c"], errors="coerce") - pd.to_numeric(out["anchor_forecast_max_c"], errors="coerce")).abs()
        model_abs = (pd.to_numeric(out["y_true_c"], errors="coerce") - pd.to_numeric(out["prediction_c"], errors="coerce")).abs()
        out["abs_improvement_vs_raw_c"] = raw_abs - model_abs
        out["helped_vs_raw_flag"] = (out["abs_improvement_vs_raw_c"] > 1e-9).astype(int)
        out["worsened_vs_raw_flag"] = (out["abs_improvement_vs_raw_c"] < -1e-9).astype(int)
    return out
