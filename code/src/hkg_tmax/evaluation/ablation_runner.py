"""Walk-forward ablation and residual ensemble runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax.evaluation.metrics import score_arrays, score_frame
from hkg_tmax.modeling.baselines import (
    climatology_persistence_prediction,
    grouped_residual_prediction,
    raw_official_prediction,
)
from hkg_tmax.modeling.ensemble import apply_ensemble, fit_nonnegative_weights
from hkg_tmax.modeling.residual_models import (
    feature_importance_frame,
    fit_catboost_residual,
    fit_direct_lgbm,
    fit_huber_residual,
    fit_lgbm_residual,
)


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    stage: str


ROLLING_FOLDS = (
    FoldSpec("fold1_2011_2013", pd.Timestamp("2000-01-02"), pd.Timestamp("2010-12-31"), pd.Timestamp("2011-01-01"), pd.Timestamp("2013-12-31"), "rolling_validation"),
    FoldSpec("fold2_2014_2016", pd.Timestamp("2000-01-02"), pd.Timestamp("2013-12-31"), pd.Timestamp("2014-01-01"), pd.Timestamp("2016-12-31"), "rolling_validation"),
    FoldSpec("fold3_2017_2019", pd.Timestamp("2000-01-02"), pd.Timestamp("2016-12-31"), pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31"), "rolling_validation"),
    FoldSpec("fold4_2020_2021", pd.Timestamp("2000-01-02"), pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), "rolling_validation"),
)

HOLDOUT_FOLDS = (
    FoldSpec("presealed_holdout_2022_2023", pd.Timestamp("2000-01-02"), pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"), "presealed_holdout"),
    FoldSpec("sealed_confirmation_2024_plus", pd.Timestamp("2000-01-02"), pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01"), pd.Timestamp("2026-05-31"), "sealed_confirmation"),
)

ABLATION_FAMILIES = {
    "A2_revision_residual_lgbm": ["official_anchor", "calendar", "forecast_revision"],
    "A3_hko_hourly_state_residual_lgbm": [
        "official_anchor",
        "calendar",
        "forecast_revision",
        "hko_hourly_state",
    ],
    "A4_network_gradients_residual_lgbm": [
        "official_anchor",
        "calendar",
        "forecast_revision",
        "hko_hourly_state",
        "station_network",
    ],
    "A5_text_warning_residual_lgbm": [
        "official_anchor",
        "calendar",
        "forecast_revision",
        "hko_hourly_state",
        "station_network",
        "text_warning_regime",
    ],
    "A6_target_memory_residual_lgbm": [
        "official_anchor",
        "calendar",
        "forecast_revision",
        "hko_hourly_state",
        "station_network",
        "text_warning_regime",
        "target_history",
        "target_history_normalized",
    ],
}

NEXT_ROUND_MODEL_IDS = (
    "C0_current_A7_reproduction",
    "C1_pruned_residual_ensemble",
    "C2_selective_router",
    "C3_tail_overlay_router",
)


def features_for_families(families: dict[str, list[str]], names: list[str], frame: pd.DataFrame) -> list[str]:
    ordered: list[str] = []
    for name in names:
        for feature in families.get(name, []):
            if feature in frame.columns and feature not in ordered:
                ordered.append(feature)
    return ordered


def official_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["forecast_selector_status"].eq("selected")
        & frame["anchor_forecast_max_c"].notna()
        & frame["y_true_c"].notna()
    ].copy()


def run_ablation_pipeline(
    matrix: pd.DataFrame,
    families: dict[str, list[str]],
    *,
    cutoff_profiles: list[str],
    seed: int,
) -> dict[str, Any]:
    model_rows: list[pd.DataFrame] = []
    ensemble_residual_rows: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    model_selection: dict[str, Any] = {"cutoffs": {}}
    for cutoff in cutoff_profiles:
        print(f"[hkg_tmax_residual_ml] cutoff={cutoff} selecting official rows", flush=True)
        cutoff_frame = official_rows(matrix[matrix["cutoff_profile"].eq(cutoff)]).sort_values("target_date").reset_index(drop=True)
        if cutoff_frame.empty:
            continue
        cutoff_predictions: list[pd.DataFrame] = []
        cutoff_ensemble_parts: list[pd.DataFrame] = []
        for fold in (*ROLLING_FOLDS, *HOLDOUT_FOLDS):
            train = cutoff_frame[
                cutoff_frame["target_date"].between(fold.train_start, fold.train_end)
                & cutoff_frame["label_source"].eq("label_core")
            ].copy()
            valid = cutoff_frame[cutoff_frame["target_date"].between(fold.valid_start, fold.valid_end)].copy()
            if train.empty or valid.empty:
                continue
            print(
                f"[hkg_tmax_residual_ml] cutoff={cutoff} fold={fold.fold_id} train={len(train)} valid={len(valid)}",
                flush=True,
            )
            fold_predictions, residual_frame, importances = run_fold(train, valid, families, fold, seed)
            cutoff_predictions.append(fold_predictions)
            cutoff_ensemble_parts.append(residual_frame)
            importance_frames.extend(importances)
        predictions = pd.concat(cutoff_predictions, ignore_index=True) if cutoff_predictions else pd.DataFrame()
        residuals = pd.concat(cutoff_ensemble_parts, ignore_index=True) if cutoff_ensemble_parts else pd.DataFrame()
        if not predictions.empty:
            model_rows.append(predictions)
        if not residuals.empty:
            ensemble_model = fit_nonnegative_weights(
                residuals[residuals["stage"].eq("rolling_validation")],
                ["resid_M0_zero", "resid_M1_grouped", "resid_M2_lgbm", "resid_M3_catboost", "resid_M4_huber"],
            )
            residuals["prediction_c"] = apply_ensemble(residuals, ensemble_model)
            residuals["model_id"] = "A7_final_residual_ensemble"
            residuals["model_family"] = "final_residual_ensemble"
            residuals["residual_prediction_c"] = residuals["prediction_c"] - residuals["anchor_forecast_max_c"]
            model_rows.append(prediction_records(residuals, "A7_final_residual_ensemble", "final_residual_ensemble"))
            ensemble_residual_rows.append(residuals)
            model_selection["cutoffs"][cutoff] = {
                "ensemble": ensemble_model,
                "feature_set": "A6_target_memory_residual_lgbm",
            }
    all_predictions = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    ensemble_rows = pd.concat(ensemble_residual_rows, ignore_index=True) if ensemble_residual_rows else pd.DataFrame()
    importances = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    scoreboards = build_scoreboards(all_predictions)
    model_selection["promotion"] = promotion_decision(scoreboards["scoreboard"])
    return {
        "predictions": all_predictions,
        "ensemble_rows": ensemble_rows,
        "feature_importance": importances,
        "scoreboards": scoreboards,
        "model_selection": model_selection,
    }


def run_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    families: dict[str, list[str]],
    fold: FoldSpec,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    prediction_parts: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    base = valid.copy()
    base["fold_id"] = fold.fold_id
    base["stage"] = fold.stage
    raw = raw_official_prediction(base)
    prediction_parts.append(prediction_records_with_array(base, raw, "A0_raw_official", "baseline"))
    clim = climatology_persistence_prediction(base)
    prediction_parts.append(prediction_records_with_array(base, clim, "B0_climatology_persistence", "baseline"))
    grouped = grouped_residual_prediction(train, base)
    prediction_parts.append(prediction_records_with_array(base, grouped, "A1_grouped_residual", "grouped_residual"))
    residual_keep = [
        "target_date",
        "cutoff_profile",
        "split",
        "label_source",
        "fold_id",
        "stage",
        "y_true_c",
        "anchor_forecast_max_c",
        "official_max_bin",
        "official_range_bin",
        "issue_hour_bucket",
        "month",
        "season_bucket",
        "network_latest_temp_spread_c",
        "inland_nt_mean_minus_coastal_marine_mean_c",
        "fcst_flag_thunderstorm",
        "hourly_any_thunderstorm_warning_24h",
        "hourly_any_rainstorm_warning_24h",
    ]
    residual_frame = base[[col for col in residual_keep if col in base.columns]].copy()
    residual_frame["resid_M0_zero"] = 0.0
    residual_frame["resid_M1_grouped"] = grouped - raw
    for ablation, family_names in ABLATION_FAMILIES.items():
        feature_names = features_for_families(families, family_names, train)
        if not feature_names:
            continue
        print(
            f"[hkg_tmax_residual_ml] fold={fold.fold_id} model={ablation} features={len(feature_names)}",
            flush=True,
        )
        pred_resid, fitted = fit_lgbm_residual(train, base, feature_names, seed)
        pred = raw + np.clip(pred_resid, -3.0, 3.0)
        prediction_parts.append(prediction_records_with_array(base, pred, ablation, "ablation_lgbm"))
        if ablation == "A6_target_memory_residual_lgbm":
            residual_frame["resid_M2_lgbm"] = np.clip(pred_resid, -3.0, 3.0)
            importances.append(feature_importance_frame(fitted).assign(fold_id=fold.fold_id))
            print(f"[hkg_tmax_residual_ml] fold={fold.fold_id} model=M3_catboost_residual", flush=True)
            cat_resid, cat_model = fit_catboost_residual(train, base, feature_names, seed)
            residual_frame["resid_M3_catboost"] = np.clip(cat_resid, -3.0, 3.0)
            importances.append(feature_importance_frame(cat_model).assign(fold_id=fold.fold_id))
            print(f"[hkg_tmax_residual_ml] fold={fold.fold_id} model=M4_huber_residual", flush=True)
            huber_resid, huber_model = fit_huber_residual(train, base, feature_names)
            residual_frame["resid_M4_huber"] = np.clip(huber_resid, -3.0, 3.0)
            importances.append(feature_importance_frame(huber_model).assign(fold_id=fold.fold_id))
            print(f"[hkg_tmax_residual_ml] fold={fold.fold_id} model=A8_direct_lgbm_absolute", flush=True)
            direct_pred, direct_model = fit_direct_lgbm(train, base, feature_names, seed)
            prediction_parts.append(prediction_records_with_array(base, direct_pred, "A8_direct_lgbm_absolute", "direct_diagnostic"))
            importances.append(feature_importance_frame(direct_model).assign(fold_id=fold.fold_id))
    for col in ("resid_M2_lgbm", "resid_M3_catboost", "resid_M4_huber"):
        if col not in residual_frame:
            residual_frame[col] = 0.0
    return pd.concat(prediction_parts, ignore_index=True), residual_frame, importances


def prediction_records_with_array(
    base: pd.DataFrame,
    prediction: np.ndarray,
    model_id: str,
    model_family: str,
) -> pd.DataFrame:
    out = base.copy()
    out["prediction_c"] = prediction
    out["model_id"] = model_id
    out["model_family"] = model_family
    out["residual_prediction_c"] = out["prediction_c"] - out["anchor_forecast_max_c"]
    return prediction_records(out, model_id, model_family)


def prediction_records(frame: pd.DataFrame, model_id: str, model_family: str) -> pd.DataFrame:
    keep = [
        "target_date",
        "cutoff_profile",
        "split",
        "label_source",
        "fold_id",
        "stage",
        "y_true_c",
        "anchor_forecast_max_c",
        "prediction_c",
        "residual_prediction_c",
        "official_max_bin",
        "official_range_bin",
        "issue_hour_bucket",
        "month",
        "season_bucket",
        "network_latest_temp_spread_c",
        "inland_nt_mean_minus_coastal_marine_mean_c",
        "fcst_flag_thunderstorm",
        "hourly_any_thunderstorm_warning_24h",
        "hourly_any_rainstorm_warning_24h",
    ]
    out = frame[[col for col in keep if col in frame.columns]].copy()
    out["model_id"] = model_id
    out["model_family"] = model_family
    return out


def build_scoreboards(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if predictions.empty:
        empty = pd.DataFrame()
        return {
            "scoreboard": empty,
            "scoreboard_by_split": empty,
            "scoreboard_by_month": empty,
            "scoreboard_by_regime": empty,
            "ablation_scoreboard": empty,
            "cutoff_sensitivity_scoreboard": empty,
        }
    overall = score_frame(predictions, ["cutoff_profile", "model_id", "model_family"], scope="overall")
    by_split = score_frame(predictions, ["cutoff_profile", "model_id", "stage"], scope="by_split")
    by_month = score_frame(predictions, ["cutoff_profile", "model_id", "month"], scope="by_month")
    regime_parts: list[pd.DataFrame] = []
    for col in (
        "season_bucket",
        "official_max_bin",
        "official_range_bin",
        "issue_hour_bucket",
        "fcst_flag_thunderstorm",
        "hourly_any_thunderstorm_warning_24h",
        "hourly_any_rainstorm_warning_24h",
    ):
        if col in predictions:
            part = score_frame(predictions, ["cutoff_profile", "model_id", col], scope=f"by_{col}")
            regime_parts.append(part)
    by_regime = pd.concat(regime_parts, ignore_index=True) if regime_parts else pd.DataFrame()
    ablation = overall[overall["model_id"].astype(str).str.startswith(("A", "B0", "C"))].copy()
    cutoff_rows: list[dict[str, Any]] = []
    for cutoff, group in overall.groupby("cutoff_profile"):
        best = group.sort_values(["mae", "rmse"], na_position="last").iloc[0]
        cutoff_rows.append(
            {
                "cutoff_profile": cutoff,
                "best_model_id": best["model_id"],
                "best_mae": best["mae"],
                "best_rmse": best["rmse"],
                "best_p90_absolute_error": best["p90_absolute_error"],
                "best_n_scored": best["n_scored"],
            }
        )
    return {
        "scoreboard": overall.sort_values(["cutoff_profile", "mae"], na_position="last").reset_index(drop=True),
        "scoreboard_by_split": by_split.sort_values(["cutoff_profile", "stage", "mae"], na_position="last").reset_index(drop=True),
        "scoreboard_by_month": by_month.sort_values(["cutoff_profile", "model_id", "month"]).reset_index(drop=True),
        "scoreboard_by_regime": by_regime.reset_index(drop=True),
        "ablation_scoreboard": ablation.reset_index(drop=True),
        "cutoff_sensitivity_scoreboard": pd.DataFrame(cutoff_rows).sort_values("best_mae").reset_index(drop=True),
    }


def promotion_decision(scoreboard: pd.DataFrame) -> dict[str, Any]:
    if scoreboard.empty:
        return {"outcome": "blocked_no_scores"}
    primary = scoreboard[scoreboard["cutoff_profile"].eq("tminus1_2359")]
    raw = primary[primary["model_id"].eq("A0_raw_official")]
    final = primary[primary["model_id"].eq("A7_final_residual_ensemble")]
    if raw.empty or final.empty:
        return {"outcome": "blocked_missing_primary_raw_or_final"}
    raw_row = raw.iloc[0]
    final_row = final.iloc[0]
    improvement = float(raw_row["mae"] - final_row["mae"])
    rmse_delta = float(final_row["rmse"] - raw_row["rmse"])
    p90_delta = float(final_row["p90_absolute_error"] - raw_row["p90_absolute_error"])
    if float(final_row["mae"]) <= 0.67 and improvement >= 0.25 and rmse_delta <= 0 and p90_delta <= 0:
        outcome = "stretch_success"
    elif improvement >= 0.08 and rmse_delta <= 0.02 and p90_delta <= 0.02:
        outcome = "credible_major_improvement"
    else:
        outcome = "no_promote_cosmetic"
    return {
        "outcome": outcome,
        "primary_cutoff": "tminus1_2359",
        "raw_official_mae": float(raw_row["mae"]),
        "final_mae": float(final_row["mae"]),
        "mae_improvement": improvement,
        "rmse_delta": rmse_delta,
        "p90_delta": p90_delta,
        "n_scored": int(final_row["n_scored"]),
    }
