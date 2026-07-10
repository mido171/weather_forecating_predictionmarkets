from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import yaml

from hkg_tmax.data.anchor_provenance_audit import (
    build_anchor_provenance_audit,
    load_all_forecast_candidates,
    summarize_anchor_provenance,
)
from hkg_tmax.data.forecast_anchor import load_targets
from hkg_tmax.evaluation.ablation_runner import (
    HOLDOUT_FOLDS,
    ROLLING_FOLDS,
    build_scoreboards,
    official_rows,
    prediction_records_with_array,
)
from hkg_tmax.evaluation.no_harm_reporting import (
    apply_rate_by,
    benefit_deciles,
    help_worse_rows,
    no_harm_audit,
)
from hkg_tmax.evaluation.reporting import (
    artifact_manifest,
    feature_missingness_report,
    next_round_model_card,
    next_round_summary_payload,
    source_eligibility_audit,
    write_csv,
    write_json,
    write_parquet,
    write_text,
)
from hkg_tmax.features.leakage_guards import next_round_leakage_audit_payload
from hkg_tmax.features.pruned_feature_policy import (
    CANDIDATE_META_FEATURES,
    feature_policy_report,
    family_for_feature,
    router_feature_names,
    validate_pruned_features,
)
from hkg_tmax.modeling.ensemble import apply_ensemble, fit_nonnegative_weights
from hkg_tmax.modeling.residual_models import (
    feature_importance_frame,
    fit_catboost_residual,
    fit_lgbm_residual,
    fit_robust_linear_residual,
)
from hkg_tmax.modeling.selective_router import (
    add_candidate_meta_features,
    apply_selective_router,
    build_router_labels,
    fit_router_models,
    predict_router_scores,
    select_router_thresholds,
)
from hkg_tmax.modeling.tail_specialist import apply_tail_overlay, fit_tail_models, predict_tail_scores


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "hkg_tmax" / "residual_ml_next_round.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "hkg_tmax" / "0002_selective_no_harm_router_20260705" / "results"
DEFAULT_COMPAT_OUTPUT = REPO_ROOT / "experiments" / "hkg_tmax_residual_ml_next_round" / "results"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"[hkg_tmax_next_round] {utc_now()} {message}", flush=True)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_previous_artifacts(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrices = [
        pd.read_parquet(results_dir / "feature_matrix_trainval.parquet"),
        pd.read_parquet(results_dir / "feature_matrix_presealed_holdout.parquet"),
        pd.read_parquet(results_dir / "feature_matrix_sealed_confirmation.parquet"),
    ]
    matrix = pd.concat(matrices, ignore_index=True)
    matrix["target_date"] = pd.to_datetime(matrix["target_date"], errors="coerce").dt.normalize()
    predictions = pd.read_parquet(results_dir / "prediction_rows.parquet")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    lineage = pd.DataFrame(json.loads((results_dir / "feature_lineage.json").read_text(encoding="utf-8")))
    source_eligibility = pd.read_csv(results_dir / "source_eligibility_audit.csv")
    if "target_date" in source_eligibility:
        source_eligibility["target_date"] = pd.to_datetime(source_eligibility["target_date"], errors="coerce").dt.normalize()
    return matrix, predictions, lineage, source_eligibility


def a3_pruned_features(feature_names: list[str]) -> list[str]:
    keep_families = {"official_anchor", "calendar", "forecast_revision", "hko_hourly_state"}
    return [feature for feature in feature_names if family_for_feature(feature) in keep_families]


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
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def run_pruned_candidate_pass(
    matrix: pd.DataFrame,
    *,
    cutoff_profiles: list[str],
    feature_names: list[str],
    seed: int,
) -> dict[str, Any]:
    all_candidates: list[pd.DataFrame] = []
    all_c1_predictions: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    linear_diagnostics: list[dict[str, Any]] = []
    ensemble_weights: dict[str, Any] = {}
    a3_features = a3_pruned_features(feature_names)
    for cutoff in cutoff_profiles:
        cutoff_frame = official_rows(matrix[matrix["cutoff_profile"].eq(cutoff)]).sort_values("target_date").reset_index(drop=True)
        if cutoff_frame.empty:
            continue
        cutoff_parts: list[pd.DataFrame] = []
        for fold in (*ROLLING_FOLDS, *HOLDOUT_FOLDS):
            train = cutoff_frame[
                cutoff_frame["target_date"].between(fold.train_start, fold.train_end)
                & cutoff_frame["label_source"].eq("label_core")
            ].copy()
            valid = cutoff_frame[cutoff_frame["target_date"].between(fold.valid_start, fold.valid_end)].copy()
            if train.empty or valid.empty:
                continue
            log(f"fit pruned candidates cutoff={cutoff} fold={fold.fold_id} train={len(train)} valid={len(valid)}")
            base = valid.copy()
            base["fold_id"] = fold.fold_id
            base["stage"] = fold.stage
            pred_a3, model_a3 = fit_lgbm_residual(train, valid, a3_features, seed)
            pred_full, model_full = fit_lgbm_residual(train, valid, feature_names, seed + 11)
            pred_cat, model_cat = fit_catboost_residual(train, valid, feature_names, seed + 19)
            pred_linear, model_linear = fit_robust_linear_residual(train, valid, feature_names)
            base["candidate_resid_lgbm_a3_c"] = np.clip(pred_a3, -3.0, 3.0)
            base["candidate_resid_lgbm_pruned_full_c"] = np.clip(pred_full, -3.0, 3.0)
            base["candidate_resid_catboost_c"] = np.clip(pred_cat, -3.0, 3.0)
            base["candidate_resid_linear_c"] = np.clip(pred_linear, -3.0, 3.0)
            base["resid_C0_zero"] = 0.0
            base["resid_C1_lgbm_a3"] = base["candidate_resid_lgbm_a3_c"]
            base["resid_C1_lgbm_full"] = base["candidate_resid_lgbm_pruned_full_c"]
            base["resid_C1_catboost"] = base["candidate_resid_catboost_c"]
            base["resid_C1_linear"] = base["candidate_resid_linear_c"]
            cutoff_parts.append(add_candidate_meta_features(base))
            importance_frames.append(feature_importance_frame(model_a3).assign(fold_id=fold.fold_id, cutoff_profile=cutoff, model_slot="M2a_pruned_A3_LGBM_residual"))
            importance_frames.append(feature_importance_frame(model_full).assign(fold_id=fold.fold_id, cutoff_profile=cutoff, model_slot="M2b_pruned_full_LGBM_residual"))
            importance_frames.append(feature_importance_frame(model_cat).assign(fold_id=fold.fold_id, cutoff_profile=cutoff, model_slot="M3_pruned_CatBoost_residual"))
            importance_frames.append(feature_importance_frame(model_linear).assign(fold_id=fold.fold_id, cutoff_profile=cutoff, model_slot="M4_robust_linear_residual"))
            linear_diagnostics.append(
                {
                    "cutoff_profile": cutoff,
                    "fold_id": fold.fold_id,
                    "model_id": model_linear.model_id,
                    "status": model_linear.status,
                    "diagnostics": json.dumps(model_linear.diagnostics or {}, sort_keys=True),
                }
            )
        if not cutoff_parts:
            continue
        candidates = pd.concat(cutoff_parts, ignore_index=True)
        weight_cols = ["resid_C0_zero", "resid_C1_lgbm_a3", "resid_C1_lgbm_full", "resid_C1_catboost", "resid_C1_linear"]
        model = fit_nonnegative_weights(candidates[candidates["stage"].eq("rolling_validation")], weight_cols)
        candidates["prediction_c"] = apply_ensemble(candidates, model)
        candidates["candidate_resid_ensemble_c"] = candidates["prediction_c"] - candidates["anchor_forecast_max_c"]
        candidates = add_candidate_meta_features(candidates)
        ensemble_weights[cutoff] = model
        all_candidates.append(candidates)
        c1 = candidates.copy()
        c1["model_id"] = "C1_pruned_residual_ensemble"
        c1["model_family"] = "pruned_residual_ensemble"
        c1["residual_prediction_c"] = c1["prediction_c"] - c1["anchor_forecast_max_c"]
        all_c1_predictions.append(c1)
    return {
        "candidate_rows": pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame(),
        "c1_predictions": pd.concat(all_c1_predictions, ignore_index=True) if all_c1_predictions else pd.DataFrame(),
        "feature_importance": pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(),
        "linear_diagnostics": pd.DataFrame(linear_diagnostics),
        "ensemble_weights": ensemble_weights,
    }


def score_router_for_cutoff(candidates: pd.DataFrame, router_features: list[str], config: dict[str, Any], seed: int) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    scored_parts: list[pd.DataFrame] = []
    model_params: dict[str, Any] = {"fold_models": []}
    rolling = build_router_labels(candidates[candidates["stage"].eq("rolling_validation")]).copy()
    if rolling.empty:
        thresholds = select_router_thresholds(candidates, config)
        return apply_selective_router(candidates, pd.DataFrame(), {}, thresholds), thresholds, pd.DataFrame(thresholds.get("selection_rows", [])), model_params
    for fold_id, valid in rolling.groupby("fold_id", dropna=False):
        train = rolling[~rolling["fold_id"].eq(fold_id)].copy()
        if train.empty:
            train = rolling.copy()
        models = fit_router_models(train, router_features, seed)
        scored = predict_router_scores(valid, models)
        scored["router_training_fold_ids"] = "|".join(sorted(map(str, train["fold_id"].dropna().unique().tolist())))
        scored_parts.append(scored)
        model_params["fold_models"].append({"fold_id": str(fold_id), "training_rows": models["training_rows"], "features": models["features"]})
    final_train = rolling.copy()
    final_models = fit_router_models(final_train, router_features, seed + 101)
    for stage in ["presealed_holdout", "sealed_confirmation"]:
        valid = candidates[candidates["stage"].eq(stage)].copy()
        if valid.empty:
            continue
        scored = predict_router_scores(valid, final_models)
        scored["router_training_fold_ids"] = "|".join(sorted(map(str, final_train["fold_id"].dropna().unique().tolist())))
        scored_parts.append(scored)
    scored_all = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    thresholds = select_router_thresholds(scored_all[scored_all["stage"].eq("rolling_validation")], config)
    selection_frame = pd.DataFrame(thresholds.get("selection_rows", []))
    c2 = apply_selective_router(scored_all, pd.DataFrame(), {}, thresholds)
    model_params["final_training_rows"] = final_models["training_rows"]
    model_params["features"] = final_models["features"]
    return c2, thresholds, selection_frame, model_params


def select_tail_thresholds(scored_tail: pd.DataFrame, base_router: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    tail_config = config.get("tail_specialist", {})
    rolling_tail = scored_tail[scored_tail["stage"].eq("rolling_validation")].copy()
    rolling_base = base_router[base_router["stage"].eq("rolling_validation")].copy()
    rows: list[dict[str, Any]] = []
    idx = 1
    for tail_prob in tail_config.get("tail_probability_grid", [0.60]):
        for sign_prob in tail_config.get("sign_probability_grid", [0.62]):
            for min_corr in tail_config.get("min_abs_tail_correction_c", [0.25]):
                thresholds = {
                    "threshold_id": f"tail_grid_{idx:04d}",
                    "tail150_probability": float(tail_prob),
                    "tail_sign_probability": float(sign_prob),
                    "min_abs_tail_correction_c": float(min_corr),
                    "hard_abs_cap_c": float(tail_config.get("hard_abs_cap_c", 1.0)),
                }
                candidate = apply_tail_overlay(rolling_tail, rolling_base, {}, thresholds)
                metrics = build_scoreboards(candidate)["scoreboard"]
                row = metrics.iloc[0].to_dict() if not metrics.empty else {}
                rows.append({**thresholds, **row, "tail_apply_rate": float(candidate["tail_overlay_applied_flag"].mean()) if len(candidate) else 0.0})
                idx += 1
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"threshold_id": "tail_disabled", "tail150_probability": 1.0, "tail_sign_probability": 1.0, "min_abs_tail_correction_c": 999.0, "hard_abs_cap_c": 1.0}, frame
    best = frame.sort_values(["rmse", "mae"], na_position="last").iloc[0].to_dict()
    return best, frame


def score_tail_for_cutoff(candidates: pd.DataFrame, c2: pd.DataFrame, router_features: list[str], config: dict[str, Any], seed: int) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    scored_parts: list[pd.DataFrame] = []
    model_params: dict[str, Any] = {"fold_models": []}
    rolling = candidates[candidates["stage"].eq("rolling_validation")].copy()
    for fold_id, valid in rolling.groupby("fold_id", dropna=False):
        train = rolling[~rolling["fold_id"].eq(fold_id)].copy()
        if train.empty:
            train = rolling.copy()
        models = fit_tail_models(train, router_features, seed)
        scored = predict_tail_scores(valid, models)
        scored["tail_training_fold_ids"] = "|".join(sorted(map(str, train["fold_id"].dropna().unique().tolist())))
        scored_parts.append(scored)
        model_params["fold_models"].append({"fold_id": str(fold_id), "training_rows": models["training_rows"], "tail150_rate": models["tail150_rate"]})
    if not rolling.empty:
        final_models = fit_tail_models(rolling, router_features, seed + 211)
        for stage in ["presealed_holdout", "sealed_confirmation"]:
            valid = candidates[candidates["stage"].eq(stage)].copy()
            if valid.empty:
                continue
            scored = predict_tail_scores(valid, final_models)
            scored["tail_training_fold_ids"] = "|".join(sorted(map(str, rolling["fold_id"].dropna().unique().tolist())))
            scored_parts.append(scored)
        model_params["final_training_rows"] = final_models["training_rows"]
        model_params["features"] = final_models["features"]
    scored_tail = pd.concat(scored_parts, ignore_index=True) if scored_parts else candidates.copy()
    thresholds, selection_frame = select_tail_thresholds(scored_tail, c2, config)
    c3 = apply_tail_overlay(scored_tail, c2, {}, thresholds)
    return c3, thresholds, selection_frame, model_params


def c0_from_previous(predictions: pd.DataFrame) -> pd.DataFrame:
    c0 = predictions[predictions["model_id"].eq("A7_final_residual_ensemble")].copy()
    c0["model_id"] = "C0_current_A7_reproduction"
    c0["model_family"] = "current_a7_reproduction"
    return c0


def write_required_artifacts(
    *,
    output_dir: Path,
    compat_output_dir: Path | None,
    config: dict[str, Any],
    matrix: pd.DataFrame,
    lineage: pd.DataFrame,
    source_eligibility: pd.DataFrame,
    previous_predictions: pd.DataFrame,
    c1_result: dict[str, Any],
    c2: pd.DataFrame,
    c3: pd.DataFrame,
    router_threshold_frames: list[pd.DataFrame],
    router_thresholds: dict[str, Any],
    router_model_params: dict[str, Any],
    tail_threshold_frame: pd.DataFrame,
    tail_thresholds: dict[str, Any],
    tail_model_params: dict[str, Any],
    feature_names: list[str],
    feature_policy_payload: dict[str, Any],
    anchor_audit: pd.DataFrame,
    anchor_summary: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    c0 = c0_from_previous(previous_predictions)
    combined_predictions = pd.concat(
        [previous_predictions, c0, c1_result["c1_predictions"], c2, c3],
        ignore_index=True,
        sort=False,
    )
    scoreboards = build_scoreboards(combined_predictions)
    raw_primary = previous_predictions[previous_predictions["model_id"].eq("A0_raw_official")].copy()
    a7_primary = previous_predictions[previous_predictions["model_id"].eq("A7_final_residual_ensemble")].copy()
    nh_audit = no_harm_audit(c2, raw_predictions=raw_primary, current_a7_predictions=a7_primary, config=config)
    leakage = next_round_leakage_audit_payload(
        matrix,
        lineage,
        feature_names=feature_names + [feature for feature in CANDIDATE_META_FEATURES if feature in c2.columns],
        router_thresholds=router_thresholds,
        router_predictions=c2,
    )
    summary = next_round_summary_payload(
        generated_at_utc=utc_now(),
        config=config,
        feature_policy=feature_policy_payload,
        scoreboard=scoreboards["scoreboard"],
        no_harm_audit=nh_audit,
        leakage_audit=leakage,
        router_thresholds=router_thresholds,
        output_dir=output_dir,
    )
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "next_round_summary.json", summary)
    write_text(
        output_dir / "next_round_model_card.md",
        next_round_model_card(
            summary=summary,
            scoreboard=scoreboards["scoreboard"],
            no_harm_audit=nh_audit,
            leakage_audit=leakage,
            router_thresholds=router_thresholds,
            feature_count=len(feature_names),
        ),
    )
    for name, frame in scoreboards.items():
        write_csv(output_dir / f"{name}.csv", frame)
    official_proxy = scoreboards["scoreboard_by_regime"]
    if not official_proxy.empty:
        official_proxy = official_proxy[
            official_proxy["scope"].isin(["by_official_max_bin", "by_official_range_bin", "by_issue_hour_bucket"])
        ]
    write_csv(output_dir / "scoreboard_by_official_error_proxy.csv", official_proxy)
    threshold_frame = pd.concat(router_threshold_frames, ignore_index=True) if router_threshold_frames else pd.DataFrame()
    write_csv(output_dir / "router_threshold_selection.csv", threshold_frame)
    write_csv(output_dir / "router_oof_diagnostics.csv", c2)
    write_csv(output_dir / "router_apply_rate_by_split.csv", apply_rate_by(c2, ["cutoff_profile", "stage"]))
    write_csv(output_dir / "router_apply_rate_by_month.csv", apply_rate_by(c2, ["cutoff_profile", "month"]))
    write_csv(output_dir / "router_apply_rate_by_regime.csv", apply_rate_by(c2, ["cutoff_profile", "season_bucket"]))
    write_csv(output_dir / "router_benefit_deciles.csv", benefit_deciles(c2))
    write_csv(output_dir / "help_worse_rows.csv", help_worse_rows(c2))
    tail_scoreboard = build_scoreboards(c3)["scoreboard"] if not c3.empty else pd.DataFrame()
    write_json(
        output_dir / "tail_specialist_audit.json",
        {
            "thresholds": tail_thresholds,
            "model_params": tail_model_params,
            "selection_rows": tail_threshold_frame.to_dict(orient="records"),
        },
    )
    write_csv(output_dir / "tail_specialist_scoreboard.csv", tail_scoreboard)
    write_json(output_dir / "no_harm_audit.json", nh_audit)
    write_json(output_dir / "leakage_audit.json", leakage)
    write_json(
        output_dir / "row_count_audit.json",
        {
            "matrix_rows": int(len(matrix)),
            "prediction_rows": int(len(combined_predictions)),
            "c1_rows": int(len(c1_result["c1_predictions"])),
            "c2_rows": int(len(c2)),
            "c3_rows": int(len(c3)),
            "rows_by_cutoff": matrix.groupby("cutoff_profile").size().astype(int).to_dict(),
        },
    )
    write_json(output_dir / "feature_lineage.json", lineage.to_dict(orient="records"))
    write_json(
        output_dir / "feature_matrix_schema.json",
        [
            {
                "feature_name": feature,
                "family": family_for_feature(feature),
                "dtype": str(matrix[feature].dtype) if feature in matrix else "missing",
                "missing_pct": float(matrix[feature].isna().mean() * 100.0) if feature in matrix else None,
            }
            for feature in feature_names
        ],
    )
    write_csv(output_dir / "feature_missingness_report.csv", feature_missingness_report(matrix, feature_names))
    write_csv(output_dir / "feature_policy_report.csv", feature_policy_report(matrix, max_raw_features=config["feature_policy"]["max_raw_features"]))
    importance = c1_result["feature_importance"]
    write_csv(output_dir / "feature_importance_lgbm.csv", importance[importance["model_slot"].astype(str).str.contains("LGBM", na=False)] if not importance.empty else pd.DataFrame())
    write_csv(output_dir / "feature_importance_catboost.csv", importance[importance["model_slot"].astype(str).str.contains("CatBoost", na=False)] if not importance.empty else pd.DataFrame())
    write_csv(output_dir / "linear_model_diagnostics.csv", c1_result["linear_diagnostics"])
    write_json(output_dir / "ensemble_weights.json", c1_result["ensemble_weights"])
    write_json(output_dir / "router_model_params.json", {"thresholds": router_thresholds, "model_params": router_model_params})
    write_csv(output_dir / "anchor_provenance_audit.csv", anchor_audit)
    write_json(output_dir / "anchor_provenance_summary.json", anchor_summary)
    write_csv(output_dir / "source_eligibility_audit.csv", source_eligibility)
    write_parquet(output_dir / "prediction_rows.parquet", combined_predictions)
    write_csv(output_dir / "prediction_rows.csv", combined_predictions)
    write_csv(output_dir / "artifact_manifest.csv", artifact_manifest(output_dir))
    if compat_output_dir is not None:
        compat_output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_dir, compat_output_dir, dirs_exist_ok=True)
    return summary


def run(config_path: Path, output_dir: Path, compat_output_dir: Path | None, database_url: str) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("seed", 20260705))
    previous_dir = REPO_ROOT / config.get("input_artifacts", {}).get("previous_results_dir", "experiments/hkg_tmax_residual_ml_strategy/results")
    log(f"loading previous leakage-safe artifacts from {previous_dir}")
    matrix, previous_predictions, lineage, source_eligibility = read_previous_artifacts(previous_dir)
    cutoff_profiles = list(config.get("cutoff_profiles", ["tminus1_2359", "tminus1_2100", "tminus1_1800"]))
    policy = validate_pruned_features(
        matrix,
        max_raw_features=int(config.get("feature_policy", {}).get("max_raw_features", 90)),
    )
    feature_names = policy.feature_names
    log(f"selected pruned raw features={len(feature_names)} missing={len(policy.missing_features)}")
    c1_result = run_pruned_candidate_pass(matrix, cutoff_profiles=cutoff_profiles, feature_names=feature_names, seed=seed)
    candidate_rows = c1_result["candidate_rows"]
    router_all: list[pd.DataFrame] = []
    router_threshold_frames: list[pd.DataFrame] = []
    router_thresholds_by_cutoff: dict[str, Any] = {}
    router_params_by_cutoff: dict[str, Any] = {}
    tail_all: list[pd.DataFrame] = []
    tail_threshold_frames: list[pd.DataFrame] = []
    tail_thresholds_by_cutoff: dict[str, Any] = {}
    tail_params_by_cutoff: dict[str, Any] = {}
    for cutoff in cutoff_profiles:
        candidates = candidate_rows[candidate_rows["cutoff_profile"].eq(cutoff)].copy()
        if candidates.empty:
            continue
        r_features = router_feature_names(candidates, max_raw_features=int(config.get("feature_policy", {}).get("max_raw_features", 90)))
        log(f"fit router cutoff={cutoff} router_features={len(r_features)}")
        c2_cutoff, thresholds, threshold_frame, params = score_router_for_cutoff(candidates, r_features, config, seed)
        threshold_frame["cutoff_profile"] = cutoff
        router_all.append(c2_cutoff)
        router_threshold_frames.append(threshold_frame)
        router_thresholds_by_cutoff[cutoff] = {key: value for key, value in thresholds.items() if key != "selection_rows"}
        router_params_by_cutoff[cutoff] = params
        log(f"fit tail overlay cutoff={cutoff}")
        c3_cutoff, tail_thresholds, tail_threshold_frame, tail_params = score_tail_for_cutoff(candidates, c2_cutoff, r_features, config, seed)
        tail_threshold_frame["cutoff_profile"] = cutoff
        tail_all.append(c3_cutoff)
        tail_threshold_frames.append(tail_threshold_frame)
        tail_thresholds_by_cutoff[cutoff] = tail_thresholds
        tail_params_by_cutoff[cutoff] = tail_params
    c2 = pd.concat(router_all, ignore_index=True) if router_all else pd.DataFrame()
    c3 = pd.concat(tail_all, ignore_index=True) if tail_all else pd.DataFrame()
    log("running anchor provenance audit")
    date_params = {
        "start_date": config["dates"]["start_date"],
        "presealed_end_date": config["dates"]["presealed_end_date"],
        "sealed_start_date": config["dates"]["sealed_start_date"],
        "sealed_end_date": config["dates"]["sealed_end_date"],
    }
    try:
        with psycopg.connect(database_url) as connection:
            targets = load_targets(connection, date_params)
            forecasts_all = load_all_forecast_candidates(connection, date_params)
        anchor_audit = build_anchor_provenance_audit(
            targets,
            forecasts_all,
            config.get("early_cutoff_audit", {}).get("cutoff_profiles", []),
        )
        anchor_summary = summarize_anchor_provenance(anchor_audit)
    except Exception as exc:
        anchor_audit = pd.DataFrame(
            [
                {
                    "target_date": None,
                    "cutoff_profile": "audit_failed",
                    "strict_selected_anchor_status": "audit_failed",
                    "reason_no_anchor": str(exc),
                }
            ]
        )
        anchor_summary = {"status": "fail", "error": str(exc)}
    primary_thresholds = router_thresholds_by_cutoff.get(config.get("primary_cutoff_profile", "tminus1_2359"), {})
    primary_tail_thresholds = tail_thresholds_by_cutoff.get(config.get("primary_cutoff_profile", "tminus1_2359"), {})
    log("writing required artifacts")
    return write_required_artifacts(
        output_dir=output_dir,
        compat_output_dir=compat_output_dir,
        config=config,
        matrix=matrix,
        lineage=lineage,
        source_eligibility=source_eligibility,
        previous_predictions=previous_predictions,
        c1_result=c1_result,
        c2=c2,
        c3=c3,
        router_threshold_frames=router_threshold_frames,
        router_thresholds=primary_thresholds,
        router_model_params=router_params_by_cutoff,
        tail_threshold_frame=pd.concat(tail_threshold_frames, ignore_index=True) if tail_threshold_frames else pd.DataFrame(),
        tail_thresholds=primary_tail_thresholds,
        tail_model_params=tail_params_by_cutoff,
        feature_names=feature_names,
        feature_policy_payload=policy.to_record(),
        anchor_audit=anchor_audit,
        anchor_summary=anchor_summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG Tmax residual ML next-round selective router experiment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--compat-output-dir", default=str(DEFAULT_COMPAT_OUTPUT))
    parser.add_argument(
        "--database-url",
        default=os.environ.get("HKG_TMAX_DATABASE_URL") or os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL,
    )
    parser.add_argument("--no-compat-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        compat_output_dir=None if args.no_compat_copy else Path(args.compat_output_dir),
        database_url=args.database_url,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
