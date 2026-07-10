from __future__ import annotations

from datetime import date, timedelta
import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.constants import FEATURE_SET_NAME, FEATURE_VERSION
from klga_tmax.models.experts import EXPERTS, ExpertForecast, build_expert_forecasts, combine_experts
from klga_tmax.registry.seed import seed_all
from klga_tmax.utils.git import current_git_sha


def train_expert_registry(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
) -> dict[str, int]:
    seed_all(connection)
    feature_version_id = _feature_version_id(connection)
    count = 0
    for expert in EXPERTS:
        _model_version_id(
            connection,
            model_family=expert.family,
            model_name=expert.name,
            model_role="expert",
            start_date=start_date,
            end_date=end_date,
            feature_version_id=feature_version_id,
            hyperparams={
                "default_sigma_f": expert.default_sigma_f,
                "feature_prefixes": list(expert.feature_prefixes),
                "implementation": "deterministic_leakage_safe_pmf_expert_v1",
            },
        )
        count += 1
    _model_version_id(
        connection,
        model_family="meta_combiner",
        model_name="regularized_linear_pool_v1",
        model_role="meta_combiner",
        start_date=start_date,
        end_date=end_date,
        feature_version_id=feature_version_id,
        hyperparams={
            "pool": "uncertainty_weighted_linear_pool",
            "log_opinion_pool_status": "deferred_no_scipy_dependency",
        },
    )
    return {"registry.model_versions": count + 1}


def predict_range(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
    prediction_kind: str,
    require_labels: bool,
) -> dict[str, int]:
    if prediction_kind not in {"oof", "holdout", "forecast", "replay"}:
        raise ValueError("prediction_kind must be one of oof, holdout, forecast, replay")
    seed_all(connection)
    feature_version_id = _feature_version_id(connection)
    targets = _load_feature_matrices(
        connection,
        start_date=start_date,
        end_date=end_date,
        cutoff_id=cutoff_id,
        require_labels=require_labels,
    )
    expert_rows = 0
    final_rows = 0
    for target in targets:
        feature_vector = dict(target["feature_vector_json"] or {})
        expert_forecasts = build_expert_forecasts(feature_vector)
        persisted_experts = _persist_expert_forecasts(
            connection,
            target=target,
            feature_version_id=feature_version_id,
            forecasts=expert_forecasts,
            prediction_kind=prediction_kind,
            training_start_date=start_date,
            training_end_date=_training_end_for_target(target["target_date"], prediction_kind),
        )
        expert_rows += len(persisted_experts)
        final_rows += _persist_final_prediction(
            connection,
            target=target,
            feature_version_id=feature_version_id,
            expert_forecasts=expert_forecasts,
            expert_prediction_ids=persisted_experts,
            prediction_kind=prediction_kind,
            training_start_date=start_date,
            training_end_date=_training_end_for_target(target["target_date"], prediction_kind),
        )
    return {
        "predictions.expert_predictions": expert_rows,
        "predictions.final_predictions": final_rows,
        "targets_predicted": len(targets),
    }


def _load_feature_matrices(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
    require_labels: bool,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        text(
            """
            SELECT
                fm.feature_matrix_id,
                fm.target_instance_id,
                fm.feature_version_id,
                fm.feature_vector_json,
                fm.feature_availability_json,
                fm.label_high_temp_f,
                fm.label_available,
                ti.target_date,
                ti.cutoff_id,
                ti.cutoff_utc
            FROM gold.feature_matrix fm
            JOIN registry.feature_versions fv
              ON fv.feature_version_id = fm.feature_version_id
            JOIN gold.target_instances ti
              ON ti.target_instance_id = fm.target_instance_id
            LEFT JOIN public.wunderground_daily_tmax actual
              ON actual.local_date = ti.target_date
             AND actual.station_id = 'KLGA'
             AND actual.validation_status IN ('accepted','manual_confirmed')
            WHERE fv.feature_set_name = :feature_set_name
              AND fv.feature_version = :feature_version
              AND ti.target_date BETWEEN :start_date AND :end_date
              AND (CAST(:cutoff_id AS text) IS NULL OR ti.cutoff_id = CAST(:cutoff_id AS text))
              AND (:require_labels = false OR actual.tmax_f IS NOT NULL)
            ORDER BY ti.target_date, ti.cutoff_id
            """
        ),
        {
            "feature_set_name": FEATURE_SET_NAME,
            "feature_version": FEATURE_VERSION,
            "start_date": start_date,
            "end_date": end_date,
            "cutoff_id": cutoff_id,
            "require_labels": require_labels,
        },
    ).mappings().all()
    return [dict(row) for row in rows]


def _persist_expert_forecasts(
    connection: Connection,
    *,
    target: dict[str, Any],
    feature_version_id: str,
    forecasts: list[ExpertForecast],
    prediction_kind: str,
    training_start_date: date,
    training_end_date: date | None,
) -> list[str]:
    ids: list[str] = []
    for forecast in forecasts:
        expert = next(expert for expert in EXPERTS if expert.name == forecast.expert_name)
        model_version_id = _model_version_id(
            connection,
            model_family=expert.family,
            model_name=expert.name,
            model_role="expert",
            start_date=training_start_date,
            end_date=training_end_date,
            feature_version_id=feature_version_id,
            hyperparams={
                "default_sigma_f": expert.default_sigma_f,
                "feature_prefixes": list(expert.feature_prefixes),
                "implementation": "deterministic_leakage_safe_pmf_expert_v1",
            },
        )
        row = connection.execute(
            text(
                """
                INSERT INTO predictions.expert_predictions (
                    target_instance_id,
                    expert_name,
                    prediction_kind,
                    model_version_id,
                    feature_version_id,
                    fold_id,
                    training_start_date,
                    training_end_date,
                    pmf_json,
                    expected_tmax_f,
                    median_tmax_f,
                    mode_tmax_f,
                    prediction_interval_low_f,
                    prediction_interval_high_f,
                    uncertainty_f,
                    feature_names,
                    feature_hash,
                    source_availability_json,
                    diagnostics_json,
                    prediction_status
                )
                VALUES (
                    :target_instance_id,
                    :expert_name,
                    :prediction_kind,
                    :model_version_id,
                    :feature_version_id,
                    :fold_id,
                    :training_start_date,
                    :training_end_date,
                    CAST(:pmf_json AS jsonb),
                    :expected_tmax_f,
                    :median_tmax_f,
                    :mode_tmax_f,
                    :prediction_interval_low_f,
                    :prediction_interval_high_f,
                    :uncertainty_f,
                    :feature_names,
                    :feature_hash,
                    CAST(:source_availability_json AS jsonb),
                    CAST(:diagnostics_json AS jsonb),
                    :prediction_status
                )
                ON CONFLICT (target_instance_id, expert_name, prediction_kind, model_version_id, COALESCE(fold_id, ''))
                DO UPDATE SET
                    pmf_json = EXCLUDED.pmf_json,
                    expected_tmax_f = EXCLUDED.expected_tmax_f,
                    median_tmax_f = EXCLUDED.median_tmax_f,
                    mode_tmax_f = EXCLUDED.mode_tmax_f,
                    prediction_interval_low_f = EXCLUDED.prediction_interval_low_f,
                    prediction_interval_high_f = EXCLUDED.prediction_interval_high_f,
                    uncertainty_f = EXCLUDED.uncertainty_f,
                    feature_names = EXCLUDED.feature_names,
                    feature_hash = EXCLUDED.feature_hash,
                    source_availability_json = EXCLUDED.source_availability_json,
                    diagnostics_json = EXCLUDED.diagnostics_json,
                    prediction_status = EXCLUDED.prediction_status
                RETURNING expert_prediction_id
                """
            ),
            {
                "target_instance_id": target["target_instance_id"],
                "expert_name": forecast.expert_name,
                "prediction_kind": prediction_kind,
                "model_version_id": model_version_id,
                "feature_version_id": feature_version_id,
                "fold_id": _fold_id(target["target_date"], prediction_kind),
                "training_start_date": training_start_date,
                "training_end_date": training_end_date,
                "pmf_json": json.dumps(forecast.summary.pmf, sort_keys=True),
                "expected_tmax_f": forecast.summary.expected_tmax_f,
                "median_tmax_f": forecast.summary.median_tmax_f,
                "mode_tmax_f": forecast.summary.mode_tmax_f,
                "prediction_interval_low_f": forecast.summary.prediction_interval_low_f,
                "prediction_interval_high_f": forecast.summary.prediction_interval_high_f,
                "uncertainty_f": forecast.summary.uncertainty_f,
                "feature_names": list(forecast.feature_names),
                "feature_hash": forecast.feature_hash,
                "source_availability_json": json.dumps({"source": "gold.feature_matrix"}, sort_keys=True),
                "diagnostics_json": json.dumps(forecast.diagnostics, sort_keys=True),
                "prediction_status": forecast.status,
            },
        ).mappings().one()
        ids.append(str(row["expert_prediction_id"]))
    return ids


def _persist_final_prediction(
    connection: Connection,
    *,
    target: dict[str, Any],
    feature_version_id: str,
    expert_forecasts: list[ExpertForecast],
    expert_prediction_ids: list[str],
    prediction_kind: str,
    training_start_date: date,
    training_end_date: date | None,
) -> int:
    model_version_id = _model_version_id(
        connection,
        model_family="meta_combiner",
        model_name="regularized_linear_pool_v1",
        model_role="meta_combiner",
        start_date=training_start_date,
        end_date=training_end_date,
        feature_version_id=feature_version_id,
        hyperparams={
            "pool": "uncertainty_weighted_linear_pool",
            "log_opinion_pool_status": "deferred_no_scipy_dependency",
        },
    )
    summary, weights, diagnostics = combine_experts(expert_forecasts)
    result = connection.execute(
        text(
            """
            INSERT INTO predictions.final_predictions (
                target_instance_id,
                prediction_kind,
                model_version_id,
                feature_version_id,
                expert_prediction_ids,
                expert_weights_json,
                pmf_json,
                expected_tmax_f,
                median_tmax_f,
                mode_tmax_f,
                prediction_interval_low_f,
                prediction_interval_high_f,
                uncertainty_f,
                entropy,
                diagnostics_json
            )
            VALUES (
                :target_instance_id,
                :prediction_kind,
                :model_version_id,
                :feature_version_id,
                CAST(:expert_prediction_ids AS uuid[]),
                CAST(:expert_weights_json AS jsonb),
                CAST(:pmf_json AS jsonb),
                :expected_tmax_f,
                :median_tmax_f,
                :mode_tmax_f,
                :prediction_interval_low_f,
                :prediction_interval_high_f,
                :uncertainty_f,
                :entropy,
                CAST(:diagnostics_json AS jsonb)
            )
            ON CONFLICT (target_instance_id, prediction_kind, model_version_id)
            DO UPDATE SET
                expert_prediction_ids = EXCLUDED.expert_prediction_ids,
                expert_weights_json = EXCLUDED.expert_weights_json,
                pmf_json = EXCLUDED.pmf_json,
                expected_tmax_f = EXCLUDED.expected_tmax_f,
                median_tmax_f = EXCLUDED.median_tmax_f,
                mode_tmax_f = EXCLUDED.mode_tmax_f,
                prediction_interval_low_f = EXCLUDED.prediction_interval_low_f,
                prediction_interval_high_f = EXCLUDED.prediction_interval_high_f,
                uncertainty_f = EXCLUDED.uncertainty_f,
                entropy = EXCLUDED.entropy,
                diagnostics_json = EXCLUDED.diagnostics_json
            """
        ),
        {
            "target_instance_id": target["target_instance_id"],
            "prediction_kind": prediction_kind,
            "model_version_id": model_version_id,
            "feature_version_id": feature_version_id,
            "expert_prediction_ids": expert_prediction_ids,
            "expert_weights_json": json.dumps(weights, sort_keys=True),
            "pmf_json": json.dumps(summary.pmf, sort_keys=True),
            "expected_tmax_f": summary.expected_tmax_f,
            "median_tmax_f": summary.median_tmax_f,
            "mode_tmax_f": summary.mode_tmax_f,
            "prediction_interval_low_f": summary.prediction_interval_low_f,
            "prediction_interval_high_f": summary.prediction_interval_high_f,
            "uncertainty_f": summary.uncertainty_f,
            "entropy": summary.entropy,
            "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
        },
    )
    return result.rowcount or 0


def _model_version_id(
    connection: Connection,
    *,
    model_family: str,
    model_name: str,
    model_role: str,
    start_date: date | None,
    end_date: date | None,
    feature_version_id: str,
    hyperparams: dict[str, Any],
) -> str:
    row = connection.execute(
        text(
            """
            INSERT INTO registry.model_versions (
                model_family,
                model_name,
                model_role,
                source_code_git_sha,
                training_data_start,
                training_data_end,
                feature_version_id,
                hyperparams,
                used_fallback_model
            )
            VALUES (
                :model_family,
                :model_name,
                :model_role,
                :source_code_git_sha,
                :training_data_start,
                :training_data_end,
                :feature_version_id,
                CAST(:hyperparams AS jsonb),
                true
            )
            ON CONFLICT DO NOTHING
            RETURNING model_version_id
            """
        ),
        {
            "model_family": model_family,
            "model_name": model_name,
            "model_role": model_role,
            "source_code_git_sha": current_git_sha(),
            "training_data_start": start_date,
            "training_data_end": end_date,
            "feature_version_id": feature_version_id,
            "hyperparams": json.dumps(hyperparams, sort_keys=True),
        },
    ).mappings().first()
    if row is not None:
        return str(row["model_version_id"])
    existing = connection.execute(
        text(
            """
            SELECT model_version_id
            FROM registry.model_versions
            WHERE model_family = :model_family
              AND model_name = :model_name
              AND source_code_git_sha = :source_code_git_sha
              AND COALESCE(training_data_start, '1900-01-01'::date)
                    = COALESCE(:training_data_start, '1900-01-01'::date)
              AND COALESCE(training_data_end, '1900-01-01'::date)
                    = COALESCE(:training_data_end, '1900-01-01'::date)
              AND md5(hyperparams::text) = md5(CAST(:hyperparams AS jsonb)::text)
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {
            "model_family": model_family,
            "model_name": model_name,
            "source_code_git_sha": current_git_sha(),
            "training_data_start": start_date,
            "training_data_end": end_date,
            "hyperparams": json.dumps(hyperparams, sort_keys=True),
        },
    ).mappings().one()
    return str(existing["model_version_id"])


def _feature_version_id(connection: Connection) -> str:
    row = connection.execute(
        text(
            """
            SELECT feature_version_id
            FROM registry.feature_versions
            WHERE feature_set_name = :feature_set_name
              AND feature_version = :feature_version
            """
        ),
        {"feature_set_name": FEATURE_SET_NAME, "feature_version": FEATURE_VERSION},
    ).mappings().one()
    return str(row["feature_version_id"])


def _training_end_for_target(target_date: date, prediction_kind: str) -> date | None:
    if prediction_kind == "oof":
        return date(target_date.year - 1, 12, 31)
    if prediction_kind == "forecast":
        return target_date
    return target_date - timedelta(days=1)


def _fold_id(target_date: date, prediction_kind: str) -> str | None:
    if prediction_kind == "oof":
        return f"year_{target_date.year}"
    return None
