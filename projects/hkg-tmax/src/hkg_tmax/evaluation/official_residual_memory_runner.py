"""Governed point-forecast residual-memory experiment runner helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax.evaluation.ablation_runner import (
    HOLDOUT_FOLDS,
    ROLLING_FOLDS,
    FoldSpec,
    build_scoreboards,
    official_rows,
)
from hkg_tmax.evaluation.metrics import score_frame
from hkg_tmax.features.pruned_feature_policy import family_for_feature
from hkg_tmax.features.residual_memory_policy import assert_no_forbidden_residual_memory_predictors
from hkg_tmax.modeling.ensemble import apply_ensemble, fit_nonnegative_weights
from hkg_tmax.modeling.residual_models import (
    feature_importance_frame,
    fit_lgbm_residual,
    fit_robust_linear_residual,
)


D_MODEL_IDS = (
    "D0_A7_reproduction",
    "D1_official_residual_memory_shrinkage",
    "D2_A3_plus_residual_memory_lgbm",
    "D3_pruned_full_plus_residual_memory_lgbm",
    "D4_residual_memory_constrained_stack",
    "D5_conservative_A7_plus_memory_blend",
)

PREDICTION_KEY_COLUMNS = ["target_date", "cutoff_profile", "fold_id", "stage"]
MEMORY_CONTEXT_COLUMNS = [
    "residual_roll7_mean_lag2_c",
    "residual_roll30_mean_lag2_c",
    "residual_memory_count_roll7",
    "residual_memory_count_roll30",
    "residual_memory_max_source_date",
    "residual_memory_min_lag_days",
]


@dataclass(frozen=True)
class ResidualMemoryRunResult:
    predictions: pd.DataFrame
    candidate_rows: pd.DataFrame
    feature_importance: pd.DataFrame
    scoreboards: dict[str, pd.DataFrame]
    model_selection_log: dict[str, Any]
    ensemble_weights: dict[str, Any]
    promotion: dict[str, Any]
    row_identity_gate: dict[str, Any]


def a3_plus_memory_features(feature_names: list[str], memory_feature_names: list[str]) -> list[str]:
    keep_families = {"official_anchor", "calendar", "forecast_revision", "hko_hourly_state"}
    base = [feature for feature in feature_names if family_for_feature(feature) in keep_families]
    return [*base, *[feature for feature in memory_feature_names if feature not in base]]


def core_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
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
        "residual_roll7_mean_lag2_c",
        "residual_roll30_mean_lag2_c",
        "residual_memory_count_roll7",
        "residual_memory_count_roll30",
        "residual_memory_max_source_date",
        "residual_memory_min_lag_days",
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def d0_from_previous(previous_predictions: pd.DataFrame) -> pd.DataFrame:
    d0 = previous_predictions[previous_predictions["model_id"].eq("A7_final_residual_ensemble")].copy()
    d0["model_id"] = "D0_A7_reproduction"
    d0["model_family"] = "a7_reproduction_reference"
    return d0


def attach_memory_context_to_predictions(predictions: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    context_cols = [
        "target_date",
        "cutoff_profile",
        *[column for column in MEMORY_CONTEXT_COLUMNS if column in matrix.columns],
    ]
    context = official_rows(matrix)[context_cols].drop_duplicates(["target_date", "cutoff_profile"]).copy()
    out = predictions.copy()
    for column in MEMORY_CONTEXT_COLUMNS:
        if column in out.columns:
            out = out.drop(columns=[column])
    out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce").dt.normalize()
    context["target_date"] = pd.to_datetime(context["target_date"], errors="coerce").dt.normalize()
    return out.merge(context, on=["target_date", "cutoff_profile"], how="left", validate="many_to_one")


def prediction_records_with_residuals(
    base: pd.DataFrame,
    *,
    prediction: np.ndarray,
    model_id: str,
    model_family: str,
    residual_columns: list[str] | None = None,
) -> pd.DataFrame:
    out = core_prediction_columns(base)
    out["prediction_c"] = np.asarray(prediction, dtype=float)
    out["residual_prediction_c"] = out["prediction_c"] - pd.to_numeric(out["anchor_forecast_max_c"], errors="coerce")
    for column in residual_columns or []:
        if column in base:
            out[column] = base[column].to_numpy()
    out["model_id"] = model_id
    out["model_family"] = model_family
    return out


def residual_memory_signal(frame: pd.DataFrame) -> pd.Series:
    columns = [
        "residual_roll7_ewm_halflife3_lag2_c",
        "residual_roll14_mean_lag2_c",
        "residual_roll30_mean_lag2_c",
    ]
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return frame[available].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.0)


def select_d1_params(candidates: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    d1_config = config.get("residual_memory", {}).get("d1_shrinkage", {})
    shrink_grid = d1_config.get("shrink_grid", [0.10, 0.20, 0.35, 0.50])
    cap_grid = d1_config.get("cap_abs_grid_c", [0.20, 0.35, 0.50])
    selection = candidates[candidates["stage"].eq("rolling_validation")].copy()
    rows: list[dict[str, Any]] = []
    if selection.empty:
        return {"shrink": 0.0, "cap_abs_c": 0.0, "status": "fallback_no_rolling_rows"}, pd.DataFrame()
    y = pd.to_numeric(selection["y_true_c"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(selection["anchor_forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    signal = pd.to_numeric(selection["resid_D1_memory_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for shrink in shrink_grid:
        for cap in cap_grid:
            correction = np.clip(float(shrink) * signal, -float(cap), float(cap))
            pred = anchor + correction
            err = pred - y
            abs_err = np.abs(err)
            rows.append(
                {
                    "shrink": float(shrink),
                    "cap_abs_c": float(cap),
                    "mae": float(np.nanmean(abs_err)),
                    "rmse": float(np.sqrt(np.nanmean(err * err))),
                    "p90_absolute_error": float(np.nanquantile(abs_err, 0.90)),
                    "n_scored": int(np.isfinite(abs_err).sum()),
                }
            )
    selection_frame = pd.DataFrame(rows)
    best = selection_frame.sort_values(["mae", "rmse", "cap_abs_c", "shrink"], na_position="last").iloc[0].to_dict()
    best["status"] = "selected_on_rolling_validation"
    return best, selection_frame


def apply_d1_params(frame: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    signal = pd.to_numeric(frame["resid_D1_memory_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.clip(float(params.get("shrink", 0.0)) * signal, -float(params.get("cap_abs_c", 0.0)), float(params.get("cap_abs_c", 0.0)))


def _attach_a7_residual(candidates: pd.DataFrame, previous_predictions: pd.DataFrame) -> pd.DataFrame:
    a7 = previous_predictions[previous_predictions["model_id"].eq("A7_final_residual_ensemble")].copy()
    a7 = a7[[*PREDICTION_KEY_COLUMNS, "prediction_c", "residual_prediction_c"]].rename(
        columns={"prediction_c": "prediction_A7_c", "residual_prediction_c": "resid_D0_a7"}
    )
    out = candidates.merge(a7, on=PREDICTION_KEY_COLUMNS, how="left", validate="one_to_one")
    out["resid_D0_a7"] = pd.to_numeric(out["resid_D0_a7"], errors="coerce").fillna(0.0)
    return out


def _fit_fold_models(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    fold: FoldSpec,
    *,
    a3_memory_features: list[str],
    full_memory_features: list[str],
    seed: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    base = valid.copy()
    base["fold_id"] = fold.fold_id
    base["stage"] = fold.stage
    pred_a3, model_a3 = fit_lgbm_residual(train, valid, a3_memory_features, seed)
    pred_full, model_full = fit_lgbm_residual(train, valid, full_memory_features, seed + 31)
    pred_linear, model_linear = fit_robust_linear_residual(train, valid, full_memory_features)
    base["resid_D0_zero"] = 0.0
    base["resid_D1_memory_raw"] = residual_memory_signal(base)
    base["resid_D2_lgbm_a3_memory"] = np.clip(pred_a3, -3.0, 3.0)
    base["resid_D3_lgbm_pruned_memory"] = np.clip(pred_full, -3.0, 3.0)
    base["resid_D_linear_memory"] = np.clip(pred_linear, -3.0, 3.0)
    importances = [
        feature_importance_frame(model_a3).assign(fold_id=fold.fold_id, model_slot="D2_A3_plus_memory_LGBM"),
        feature_importance_frame(model_full).assign(fold_id=fold.fold_id, model_slot="D3_pruned_full_plus_memory_LGBM"),
        feature_importance_frame(model_linear).assign(fold_id=fold.fold_id, model_slot="D_linear_robust_memory"),
    ]
    return base, importances


def select_d5_params(
    candidates: pd.DataFrame,
    previous_predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    gate_config = config.get("acceptance_gates", {})
    d5_config = config.get("residual_memory", {}).get("d5_conservative_blend", {})
    scalar_grid = d5_config.get("scalar_grid", [0.70, 0.85, 1.00])
    cap_grid = d5_config.get("cap_abs_grid_c", [0.35, 0.50, 0.75])
    rolling = candidates[candidates["stage"].eq("rolling_validation")].copy()
    a7 = previous_predictions[
        previous_predictions["model_id"].eq("A7_final_residual_ensemble")
        & previous_predictions["stage"].eq("rolling_validation")
        & previous_predictions["cutoff_profile"].isin(rolling["cutoff_profile"].dropna().unique().tolist())
    ][[*PREDICTION_KEY_COLUMNS, "prediction_c"]].rename(columns={"prediction_c": "a7_prediction_c"})
    rolling = rolling.merge(a7, on=PREDICTION_KEY_COLUMNS, how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    y = pd.to_numeric(rolling["y_true_c"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(rolling["anchor_forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    d4 = pd.to_numeric(rolling["resid_D4_stack"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    a7_pred = pd.to_numeric(rolling["a7_prediction_c"], errors="coerce").to_numpy(dtype=float)
    a7_err = a7_pred - y
    a7_rmse = float(np.sqrt(np.nanmean(a7_err * a7_err)))
    a7_p90 = float(np.nanquantile(np.abs(a7_err), 0.90))
    for scalar in scalar_grid:
        for cap in cap_grid:
            resid = np.clip(float(scalar) * d4, -float(cap), float(cap))
            pred = anchor + resid
            err = pred - y
            abs_err = np.abs(err)
            rmse = float(np.sqrt(np.nanmean(err * err)))
            p90 = float(np.nanquantile(abs_err, 0.90))
            rows.append(
                {
                    "scalar": float(scalar),
                    "cap_abs_c": float(cap),
                    "mae": float(np.nanmean(abs_err)),
                    "rmse": rmse,
                    "p90_absolute_error": p90,
                    "rmse_delta_vs_a7": rmse - a7_rmse,
                    "p90_delta_vs_a7": p90 - a7_p90,
                    "passes_no_harm": bool(
                        (rmse - a7_rmse) <= float(gate_config.get("max_rmse_worse_vs_a7_c", 0.005))
                        and (p90 - a7_p90) <= float(gate_config.get("max_p90_worse_vs_a7_c", 0.010))
                    ),
                    "n_scored": int(np.isfinite(abs_err).sum()),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"scalar": 0.0, "cap_abs_c": 0.0, "status": "fallback_no_rolling_rows"}, frame
    eligible = frame[frame["passes_no_harm"]].copy()
    pool = eligible if not eligible.empty else frame
    best = pool.sort_values(["mae", "rmse", "cap_abs_c", "scalar"], na_position="last").iloc[0].to_dict()
    best["status"] = "selected_on_rolling_validation" if not eligible.empty else "selected_on_rolling_validation_no_noharm_candidate"
    return best, frame


def apply_d5_params(frame: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    d4 = pd.to_numeric(frame["resid_D4_stack"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.clip(float(params.get("scalar", 0.0)) * d4, -float(params.get("cap_abs_c", 0.0)), float(params.get("cap_abs_c", 0.0)))


def run_official_residual_memory_experiment(
    matrix: pd.DataFrame,
    previous_predictions: pd.DataFrame,
    *,
    cutoff_profiles: list[str],
    feature_names: list[str],
    memory_feature_names: list[str],
    config: dict[str, Any],
    seed: int,
) -> ResidualMemoryRunResult:
    full_memory_features = [*feature_names, *[feature for feature in memory_feature_names if feature not in feature_names]]
    a3_memory_features = a3_plus_memory_features(feature_names, memory_feature_names)
    assert_no_forbidden_residual_memory_predictors(full_memory_features)
    previous_with_context = attach_memory_context_to_predictions(previous_predictions, matrix)
    candidate_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = [d0_from_previous(previous_with_context)]
    importance_frames: list[pd.DataFrame] = []
    model_selection_log: dict[str, Any] = {
        "sealed_rows_used_for_model_selection": False,
        "folds_for_model_selection": [fold.fold_id for fold in ROLLING_FOLDS],
        "presealed_used_after_candidate_freeze": True,
        "sealed_confirmation_report_only": True,
        "cutoffs": {},
        "feature_sets": {
            "D2_A3_plus_residual_memory_lgbm": a3_memory_features,
            "D3_pruned_full_plus_residual_memory_lgbm": full_memory_features,
        },
    }
    ensemble_weights: dict[str, Any] = {}
    residual_cols = [
        "resid_D0_zero",
        "resid_D0_a7",
        "resid_D1_memory",
        "resid_D2_lgbm_a3_memory",
        "resid_D3_lgbm_pruned_memory",
        "resid_D_linear_memory",
        "resid_D4_stack",
        "resid_D5_conservative",
    ]
    for cutoff in cutoff_profiles:
        cutoff_frame = official_rows(matrix[matrix["cutoff_profile"].eq(cutoff)]).sort_values("target_date").reset_index(drop=True)
        if cutoff_frame.empty:
            continue
        cutoff_fold_parts: list[pd.DataFrame] = []
        cutoff_importances: list[pd.DataFrame] = []
        for fold in (*ROLLING_FOLDS, *HOLDOUT_FOLDS):
            train = cutoff_frame[
                cutoff_frame["target_date"].between(fold.train_start, fold.train_end)
                & cutoff_frame["label_source"].eq("label_core")
            ].copy()
            valid = cutoff_frame[cutoff_frame["target_date"].between(fold.valid_start, fold.valid_end)].copy()
            if train.empty or valid.empty:
                continue
            fold_rows, fold_importances = _fit_fold_models(
                train,
                valid,
                fold,
                a3_memory_features=a3_memory_features,
                full_memory_features=full_memory_features,
                seed=seed,
            )
            cutoff_fold_parts.append(fold_rows)
            cutoff_importances.extend([frame.assign(cutoff_profile=cutoff) for frame in fold_importances])
        if not cutoff_fold_parts:
            continue
        candidates = pd.concat(cutoff_fold_parts, ignore_index=True)
        candidates = _attach_a7_residual(candidates, previous_with_context)
        d1_params, d1_selection = select_d1_params(candidates, config)
        candidates["resid_D1_memory"] = apply_d1_params(candidates, d1_params)
        prediction_parts.append(
            prediction_records_with_residuals(
                candidates,
                prediction=pd.to_numeric(candidates["anchor_forecast_max_c"], errors="coerce").to_numpy(dtype=float)
                + candidates["resid_D1_memory"].to_numpy(dtype=float),
                model_id="D1_official_residual_memory_shrinkage",
                model_family="residual_memory_shrinkage",
                residual_columns=residual_cols,
            )
        )
        prediction_parts.append(
            prediction_records_with_residuals(
                candidates,
                prediction=pd.to_numeric(candidates["anchor_forecast_max_c"], errors="coerce").to_numpy(dtype=float)
                + candidates["resid_D2_lgbm_a3_memory"].to_numpy(dtype=float),
                model_id="D2_A3_plus_residual_memory_lgbm",
                model_family="a3_plus_residual_memory_lgbm",
                residual_columns=residual_cols,
            )
        )
        prediction_parts.append(
            prediction_records_with_residuals(
                candidates,
                prediction=pd.to_numeric(candidates["anchor_forecast_max_c"], errors="coerce").to_numpy(dtype=float)
                + candidates["resid_D3_lgbm_pruned_memory"].to_numpy(dtype=float),
                model_id="D3_pruned_full_plus_residual_memory_lgbm",
                model_family="pruned_full_plus_residual_memory_lgbm",
                residual_columns=residual_cols,
            )
        )
        weight_cols = [
            "resid_D0_zero",
            "resid_D0_a7",
            "resid_D1_memory",
            "resid_D2_lgbm_a3_memory",
            "resid_D3_lgbm_pruned_memory",
            "resid_D_linear_memory",
        ]
        d4_model = fit_nonnegative_weights(candidates[candidates["stage"].eq("rolling_validation")], weight_cols)
        candidates["prediction_D4_c"] = apply_ensemble(candidates, d4_model)
        candidates["resid_D4_stack"] = candidates["prediction_D4_c"] - pd.to_numeric(candidates["anchor_forecast_max_c"], errors="coerce")
        d5_params, d5_selection = select_d5_params(candidates, previous_with_context, config)
        candidates["resid_D5_conservative"] = apply_d5_params(candidates, d5_params)
        candidates["prediction_D5_c"] = pd.to_numeric(candidates["anchor_forecast_max_c"], errors="coerce") + candidates["resid_D5_conservative"]
        prediction_parts.append(
            prediction_records_with_residuals(
                candidates,
                prediction=candidates["prediction_D4_c"].to_numpy(dtype=float),
                model_id="D4_residual_memory_constrained_stack",
                model_family="residual_memory_constrained_stack",
                residual_columns=residual_cols,
            )
        )
        prediction_parts.append(
            prediction_records_with_residuals(
                candidates,
                prediction=candidates["prediction_D5_c"].to_numpy(dtype=float),
                model_id="D5_conservative_A7_plus_memory_blend",
                model_family="conservative_a7_plus_memory_blend",
                residual_columns=residual_cols,
            )
        )
        candidate_parts.append(candidates)
        importance_frames.extend(cutoff_importances)
        ensemble_weights[cutoff] = {"D4_stack": d4_model, "D5_conservative_blend": d5_params}
        model_selection_log["cutoffs"][cutoff] = {
            "D1_selected_params": d1_params,
            "D1_selection_rows": d1_selection.to_dict(orient="records"),
            "D4_stack": d4_model,
            "D5_selected_params": d5_params,
            "D5_selection_rows": d5_selection.to_dict(orient="records"),
            "training_governance": {
                "selection_stage": "rolling_validation_fold1_to_fold4",
                "presealed_stage": "post_freeze_validation_only",
                "sealed_stage": "report_only_no_tuning",
            },
        }
    candidates_all = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    new_predictions = pd.concat(prediction_parts, ignore_index=True, sort=False) if prediction_parts else pd.DataFrame()
    combined = pd.concat([previous_with_context, new_predictions], ignore_index=True, sort=False)
    combined = add_residual_memory_bins(combined)
    scoreboards = build_memory_scoreboards(combined)
    row_gate = row_identity_gate(combined, primary_cutoff=config.get("primary_cutoff_profile", "tminus1_2359"))
    promotion = evaluate_promotion_gates(scoreboards, config, row_gate)
    model_selection_log["promotion"] = promotion
    return ResidualMemoryRunResult(
        predictions=combined,
        candidate_rows=candidates_all,
        feature_importance=pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(),
        scoreboards=scoreboards,
        model_selection_log=model_selection_log,
        ensemble_weights=ensemble_weights,
        promotion=promotion,
        row_identity_gate=row_gate,
    )


def add_residual_memory_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    if "residual_roll7_mean_lag2_c" not in out:
        out["residual_memory_bin"] = "missing"
        return out
    value = pd.to_numeric(out["residual_roll7_mean_lag2_c"], errors="coerce")
    bins = [-np.inf, -0.50, -0.20, 0.20, 0.50, np.inf]
    labels = ["<=-0.50", "-0.50_to_-0.20", "-0.20_to_0.20", "0.20_to_0.50", ">=0.50"]
    out["residual_memory_bin"] = pd.cut(value, bins=bins, labels=labels).astype("object")
    out.loc[value.isna(), "residual_memory_bin"] = "missing"
    return out


def build_memory_scoreboards(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    predictions = add_residual_memory_bins(predictions)
    boards = build_scoreboards(predictions)
    boards["scoreboard_by_season"] = score_frame(predictions, ["cutoff_profile", "model_id", "season_bucket"], scope="by_season").sort_values(
        ["cutoff_profile", "season_bucket", "mae"], na_position="last"
    )
    boards["scoreboard_by_official_max_bin"] = score_frame(
        predictions, ["cutoff_profile", "model_id", "official_max_bin"], scope="by_official_max_bin"
    ).sort_values(["cutoff_profile", "official_max_bin", "mae"], na_position="last")
    boards["scoreboard_by_residual_memory_bin"] = score_frame(
        predictions, ["cutoff_profile", "model_id", "residual_memory_bin"], scope="by_residual_memory_bin"
    ).sort_values(["cutoff_profile", "residual_memory_bin", "mae"], na_position="last")
    d_models = set(D_MODEL_IDS) | {"A0_raw_official", "A7_final_residual_ensemble"}
    boards["ablation_scoreboard"] = boards["scoreboard"][boards["scoreboard"]["model_id"].isin(d_models)].copy()
    return boards


def row_identity_gate(predictions: pd.DataFrame, *, primary_cutoff: str) -> dict[str, Any]:
    required = ["A0_raw_official", "A7_final_residual_ensemble", "D5_conservative_A7_plus_memory_blend"]
    primary = predictions[predictions["cutoff_profile"].eq(primary_cutoff)].copy()
    key_cols = ["target_date", "cutoff_profile", "fold_id", "stage"]
    key_sets = {
        model_id: set(map(tuple, primary[primary["model_id"].eq(model_id)][key_cols].astype(str).to_numpy()))
        for model_id in required
    }
    reference = key_sets.get("A7_final_residual_ensemble", set())
    mismatches = {
        model_id: {
            "missing_vs_a7": len(reference - keys),
            "extra_vs_a7": len(keys - reference),
            "rows": len(keys),
        }
        for model_id, keys in key_sets.items()
    }
    status = "pass" if all(item["missing_vs_a7"] == 0 and item["extra_vs_a7"] == 0 for item in mismatches.values()) else "fail"
    return {"status": status, "primary_cutoff": primary_cutoff, "models": mismatches}


def _score_row(scoreboard: pd.DataFrame, *, cutoff: str, model_id: str) -> pd.Series | None:
    row = scoreboard[scoreboard["cutoff_profile"].eq(cutoff) & scoreboard["model_id"].eq(model_id)]
    if row.empty:
        return None
    return row.iloc[0]


def _stage_mae(scoreboard_by_split: pd.DataFrame, *, cutoff: str, model_id: str, stage: str) -> dict[str, float] | None:
    row = scoreboard_by_split[
        scoreboard_by_split["cutoff_profile"].eq(cutoff)
        & scoreboard_by_split["model_id"].eq(model_id)
        & scoreboard_by_split["stage"].eq(stage)
    ]
    if row.empty:
        return None
    first = row.iloc[0]
    return {
        "mae": float(first["mae"]),
        "rmse": float(first["rmse"]),
        "p90_absolute_error": float(first["p90_absolute_error"]),
        "n_scored": int(first["n_scored"]),
    }


def _development_score(scoreboard_by_split: pd.DataFrame, *, cutoff: str, model_id: str) -> dict[str, float] | None:
    rows = scoreboard_by_split[
        scoreboard_by_split["cutoff_profile"].eq(cutoff)
        & scoreboard_by_split["model_id"].eq(model_id)
        & scoreboard_by_split["stage"].isin(["rolling_validation", "presealed_holdout"])
    ]
    if rows.empty:
        return None
    weights = pd.to_numeric(rows["n_scored"], errors="coerce").to_numpy(dtype=float)
    if weights.sum() == 0:
        return None
    return {
        "mae": float(np.average(pd.to_numeric(rows["mae"], errors="coerce"), weights=weights)),
        "rmse": float(np.average(pd.to_numeric(rows["rmse"], errors="coerce"), weights=weights)),
        "p90_absolute_error": float(np.average(pd.to_numeric(rows["p90_absolute_error"], errors="coerce"), weights=weights)),
        "n_scored": int(weights.sum()),
    }


def slice_no_harm_gate(scoreboard: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    gates = config.get("acceptance_gates", {})
    primary = config.get("primary_cutoff_profile", "tminus1_2359")
    checks: list[dict[str, Any]] = []
    for scope, min_n, max_worse in [
        ("by_month", int(gates.get("min_month_n", 100)), float(gates.get("max_month_worse_vs_a7_c", 0.015))),
        ("by_season", int(gates.get("min_season_n", 1)), float(gates.get("max_season_worse_vs_a7_c", 0.010))),
        ("by_official_max_bin", int(gates.get("min_official_max_bin_n", 150)), float(gates.get("max_official_max_bin_worse_vs_a7_c", 0.020))),
    ]:
        rows = scoreboard[scoreboard["scope"].eq(scope) & scoreboard["cutoff_profile"].eq(primary)].copy()
        if rows.empty:
            continue
        group_cols_by_scope = {
            "by_month": ["month"],
            "by_season": ["season_bucket"],
            "by_official_max_bin": ["official_max_bin"],
        }
        group_cols = [col for col in group_cols_by_scope[scope] if col in rows.columns]
        for _, group in rows.groupby(group_cols, dropna=False):
            a7 = group[group["model_id"].eq("A7_final_residual_ensemble")]
            d5 = group[group["model_id"].eq("D5_conservative_A7_plus_memory_blend")]
            if a7.empty or d5.empty or int(d5.iloc[0]["n_scored"]) < min_n:
                continue
            delta = float(d5.iloc[0]["mae"] - a7.iloc[0]["mae"])
            checks.append(
                {
                    "scope": scope,
                    "slice": {col: str(group.iloc[0][col]) for col in group_cols},
                    "n_scored": int(d5.iloc[0]["n_scored"]),
                    "mae_delta_vs_a7": delta,
                    "status": "pass" if delta <= max_worse else "fail",
                    "max_allowed_worse_c": max_worse,
                }
            )
    return {
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "checks": checks,
        "fail_count": int(sum(check["status"] == "fail" for check in checks)),
    }


def evaluate_promotion_gates(scoreboards: dict[str, pd.DataFrame], config: dict[str, Any], row_gate: dict[str, Any]) -> dict[str, Any]:
    primary = config.get("primary_cutoff_profile", "tminus1_2359")
    gates = config.get("acceptance_gates", {})
    overall = scoreboards["scoreboard"]
    by_split = scoreboards["scoreboard_by_split"]
    raw = _score_row(overall, cutoff=primary, model_id="A0_raw_official")
    a7 = _score_row(overall, cutoff=primary, model_id="A7_final_residual_ensemble")
    d5 = _score_row(overall, cutoff=primary, model_id="D5_conservative_A7_plus_memory_blend")
    if raw is None or a7 is None or d5 is None:
        return {"decision": "no_promote", "reason": "missing primary score rows", "primary_cutoff": primary}
    development_raw = _development_score(by_split, cutoff=primary, model_id="A0_raw_official")
    development_a7 = _development_score(by_split, cutoff=primary, model_id="A7_final_residual_ensemble")
    development_d5 = _development_score(by_split, cutoff=primary, model_id="D5_conservative_A7_plus_memory_blend")
    presealed_raw = _stage_mae(by_split, cutoff=primary, model_id="A0_raw_official", stage="presealed_holdout")
    presealed_a7 = _stage_mae(by_split, cutoff=primary, model_id="A7_final_residual_ensemble", stage="presealed_holdout")
    presealed_d5 = _stage_mae(by_split, cutoff=primary, model_id="D5_conservative_A7_plus_memory_blend", stage="presealed_holdout")
    sealed_a7 = _stage_mae(by_split, cutoff=primary, model_id="A7_final_residual_ensemble", stage="sealed_confirmation")
    sealed_d5 = _stage_mae(by_split, cutoff=primary, model_id="D5_conservative_A7_plus_memory_blend", stage="sealed_confirmation")
    slice_gate = slice_no_harm_gate(
        pd.concat(
            [
                scoreboards.get("scoreboard_by_month", pd.DataFrame()),
                scoreboards.get("scoreboard_by_season", pd.DataFrame()),
                scoreboards.get("scoreboard_by_official_max_bin", pd.DataFrame()),
            ],
            ignore_index=True,
        ),
        config,
    )
    checks: list[dict[str, Any]] = []
    if development_raw and development_a7 and development_d5:
        checks.append(
            {
                "check_name": "development_d5_improves_a7_min_mae",
                "status": "pass"
                if development_a7["mae"] - development_d5["mae"] >= float(gates.get("min_dev_mae_gain_vs_a7_c", 0.012))
                else "fail",
                "value": development_a7["mae"] - development_d5["mae"],
                "threshold": float(gates.get("min_dev_mae_gain_vs_a7_c", 0.012)),
            }
        )
        checks.append(
            {
                "check_name": "development_d5_improves_raw_min_mae",
                "status": "pass"
                if development_raw["mae"] - development_d5["mae"] >= float(gates.get("min_dev_mae_gain_vs_raw_c", 0.045))
                else "fail",
                "value": development_raw["mae"] - development_d5["mae"],
                "threshold": float(gates.get("min_dev_mae_gain_vs_raw_c", 0.045)),
            }
        )
    if presealed_raw and presealed_a7 and presealed_d5:
        checks.append(
            {
                "check_name": "presealed_d5_improves_a7_min_mae",
                "status": "pass"
                if presealed_a7["mae"] - presealed_d5["mae"] >= float(gates.get("min_presealed_mae_gain_vs_a7_c", 0.008))
                else "fail",
                "value": presealed_a7["mae"] - presealed_d5["mae"],
                "threshold": float(gates.get("min_presealed_mae_gain_vs_a7_c", 0.008)),
            }
        )
        checks.append(
            {
                "check_name": "presealed_d5_improves_raw_min_mae",
                "status": "pass"
                if presealed_raw["mae"] - presealed_d5["mae"] >= float(gates.get("min_presealed_mae_gain_vs_raw_c", 0.040))
                else "fail",
                "value": presealed_raw["mae"] - presealed_d5["mae"],
                "threshold": float(gates.get("min_presealed_mae_gain_vs_raw_c", 0.040)),
            }
        )
        checks.append(
            {
                "check_name": "presealed_rmse_no_worse_vs_a7",
                "status": "pass"
                if presealed_d5["rmse"] - presealed_a7["rmse"] <= float(gates.get("max_rmse_worse_vs_a7_c", 0.005))
                else "fail",
                "value": presealed_d5["rmse"] - presealed_a7["rmse"],
                "threshold": float(gates.get("max_rmse_worse_vs_a7_c", 0.005)),
            }
        )
        checks.append(
            {
                "check_name": "presealed_p90_no_worse_vs_a7",
                "status": "pass"
                if presealed_d5["p90_absolute_error"] - presealed_a7["p90_absolute_error"]
                <= float(gates.get("max_p90_worse_vs_a7_c", 0.010))
                else "fail",
                "value": presealed_d5["p90_absolute_error"] - presealed_a7["p90_absolute_error"],
                "threshold": float(gates.get("max_p90_worse_vs_a7_c", 0.010)),
            }
        )
    if sealed_a7 and sealed_d5:
        checks.append(
            {
                "check_name": "sealed_report_only_no_large_reversal_vs_a7",
                "status": "pass"
                if sealed_d5["mae"] - sealed_a7["mae"] <= float(gates.get("max_sealed_reversal_vs_a7_c", 0.005))
                else "fail",
                "value": sealed_d5["mae"] - sealed_a7["mae"],
                "threshold": float(gates.get("max_sealed_reversal_vs_a7_c", 0.005)),
                "report_only": True,
            }
        )
    checks.append(
        {
            "check_name": "row_identity_gate",
            "status": row_gate.get("status", "fail"),
            "value": row_gate.get("status"),
            "threshold": "pass",
        }
    )
    checks.append(
        {
            "check_name": "slice_no_harm_gate",
            "status": slice_gate.get("status", "fail"),
            "value": slice_gate.get("fail_count", 0),
            "threshold": 0,
        }
    )
    decision = "promote_D5" if all(check["status"] == "pass" for check in checks) else "no_promote"
    reason = "all gates passed" if decision == "promote_D5" else "one or more predeclared gates failed"
    return {
        "decision": decision,
        "reason": reason,
        "primary_cutoff": primary,
        "overall_primary_mae": {
            "A0_raw_official": float(raw["mae"]),
            "A7_final_residual_ensemble": float(a7["mae"]),
            "D5_conservative_A7_plus_memory_blend": float(d5["mae"]),
            "D5_gain_vs_raw_c": float(raw["mae"] - d5["mae"]),
            "D5_gain_vs_A7_c": float(a7["mae"] - d5["mae"]),
        },
        "development_primary": {
            "raw": development_raw,
            "a7": development_a7,
            "d5": development_d5,
        },
        "presealed_primary": {
            "raw": presealed_raw,
            "a7": presealed_a7,
            "d5": presealed_d5,
        },
        "sealed_report_only_primary": {
            "a7": sealed_a7,
            "d5": sealed_d5,
        },
        "checks": checks,
        "slice_no_harm": slice_gate,
    }
