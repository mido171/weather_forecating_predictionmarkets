from __future__ import annotations

from datetime import date, datetime, timezone
import json
import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.constants import TEMP_GRID_F
from klga_tmax.models.pmf import validate_pmf
from klga_tmax.utils.git import current_git_sha


class ForecastEvaluationError(RuntimeError):
    pass


def evaluate_accuracy(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
    prediction_kind: str,
    run_name: str | None = None,
) -> dict[str, Any]:
    rows = _load_prediction_label_rows(
        connection,
        start_date=start_date,
        end_date=end_date,
        cutoff_id=cutoff_id,
        prediction_kind=prediction_kind,
    )
    if not rows:
        raise ForecastEvaluationError("no predictions with settled Wunderground labels were found")

    run_id_text = f"forecast_eval_{prediction_kind}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    evaluation_run_id = _start_evaluation_run(
        connection,
        run_id_text=run_id_text,
        run_name=run_name or run_id_text,
        prediction_kind=prediction_kind,
        start_date=start_date,
        end_date=end_date,
        cutoff_id=cutoff_id,
        first_row=rows[0],
    )
    daily_scores = [_score_row(row) for row in rows]
    _insert_daily_scores(connection, evaluation_run_id=evaluation_run_id, daily_scores=daily_scores)
    metrics = aggregate_scores(daily_scores)
    _finish_evaluation_run(connection, evaluation_run_id=evaluation_run_id, metrics=metrics)
    _insert_metric_rows(connection, evaluation_run_id=evaluation_run_id, daily_scores=daily_scores, metrics=metrics)
    return {
        "run_id": run_id_text,
        "reports.forecast_evaluation_runs": 1,
        "reports.forecast_evaluation_daily_scores": len(daily_scores),
        "reports.metrics": len(metrics),
        **metrics,
    }


def aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(scores)
    if count == 0:
        raise ValueError("cannot aggregate empty score set")
    mae = sum(score["absolute_error_f"] for score in scores) / count
    rmse = math.sqrt(sum(score["squared_error_f"] for score in scores) / count)
    bias = sum(score["signed_error_f"] for score in scores) / count
    exact = sum(1 for score in scores if round(score["expected_tmax_f"]) == score["settled_wu_tmax_f"]) / count
    within_1 = sum(1 for score in scores if score["within_1f"]) / count
    within_2 = sum(1 for score in scores if score["within_2f"]) / count
    log_score = sum(score["log_score"] for score in scores) / count
    crps = sum(score["crps_discrete"] for score in scores) / count
    interval_coverage = sum(1 for score in scores if score["prediction_interval_hit"]) / count
    return {
        "row_count": count,
        "mae_f": mae,
        "rmse_f": rmse,
        "bias_f": bias,
        "exact_degree_hit_rate": exact,
        "within_1f_hit_rate": within_1,
        "within_2f_hit_rate": within_2,
        "mean_log_score": log_score,
        "mean_discrete_crps": crps,
        "prediction_interval_coverage": interval_coverage,
    }


def _load_prediction_label_rows(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
    prediction_kind: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        text(
            """
            WITH latest_calibrated AS (
                SELECT DISTINCT ON (cp.final_prediction_id)
                    cp.final_prediction_id,
                    cp.calibrated_prediction_id,
                    cp.calibration_version_id,
                    cp.pmf_json,
                    cp.expected_tmax_f,
                    cp.median_tmax_f,
                    cp.mode_tmax_f,
                    cp.prediction_interval_low_f,
                    cp.prediction_interval_high_f,
                    cp.uncertainty_f
                FROM predictions.calibrated_predictions cp
                ORDER BY cp.final_prediction_id, cp.created_at DESC
            )
            SELECT
                ti.target_date,
                ti.cutoff_id,
                ti.cutoff_utc,
                fp.final_prediction_id,
                COALESCE(lc.calibrated_prediction_id, NULL) AS calibrated_prediction_id,
                COALESCE(lc.calibration_version_id, NULL) AS calibration_version_id,
                fp.model_version_id,
                fp.feature_version_id,
                COALESCE(lc.pmf_json, fp.pmf_json) AS pmf_json,
                COALESCE(lc.expected_tmax_f, fp.expected_tmax_f) AS expected_tmax_f,
                COALESCE(lc.median_tmax_f, fp.median_tmax_f) AS median_tmax_f,
                COALESCE(lc.mode_tmax_f, fp.mode_tmax_f) AS mode_tmax_f,
                COALESCE(lc.prediction_interval_low_f, fp.prediction_interval_low_f) AS prediction_interval_low_f,
                COALESCE(lc.prediction_interval_high_f, fp.prediction_interval_high_f) AS prediction_interval_high_f,
                actual.tmax_f AS settled_wu_tmax_f,
                NULL::uuid AS label_source_record_id,
                1 AS label_revision_number,
                actual.settlement_available_at_utc AS label_available_at_utc
            FROM predictions.final_predictions fp
            JOIN gold.target_instances ti
              ON ti.target_instance_id = fp.target_instance_id
            LEFT JOIN latest_calibrated lc
              ON lc.final_prediction_id = fp.final_prediction_id
            JOIN public.wunderground_daily_tmax actual
              ON actual.local_date = ti.target_date
             AND actual.station_id = 'KLGA'
             AND actual.validation_status IN ('accepted','manual_confirmed')
             AND actual.tmax_f IS NOT NULL
            WHERE ti.target_date BETWEEN :start_date AND :end_date
              AND (CAST(:cutoff_id AS text) IS NULL OR ti.cutoff_id = CAST(:cutoff_id AS text))
              AND fp.prediction_kind = :prediction_kind
            ORDER BY ti.target_date, ti.cutoff_id
            """
        ),
        {
            "start_date": start_date,
            "end_date": end_date,
            "cutoff_id": cutoff_id,
            "prediction_kind": prediction_kind,
        },
    ).mappings().all()
    return [dict(row) for row in rows]


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    observed = int(row["settled_wu_tmax_f"])
    if observed not in TEMP_GRID_F:
        raise ForecastEvaluationError(f"settled WU Tmax {observed} is outside TEMP_GRID_F")
    pmf = {str(key): float(value) for key, value in dict(row["pmf_json"]).items()}
    validate_pmf(pmf)
    expected = float(row["expected_tmax_f"])
    signed = expected - observed
    probability_at_observed = max(float(pmf[str(observed)]), 1e-12)
    return {
        "target_date": row["target_date"],
        "cutoff_id": row["cutoff_id"],
        "prediction_id": row["final_prediction_id"],
        "calibrated_prediction_id": row["calibrated_prediction_id"],
        "settled_wu_tmax_f": observed,
        "expected_tmax_f": expected,
        "median_tmax_f": int(row["median_tmax_f"]),
        "mode_tmax_f": int(row["mode_tmax_f"]),
        "absolute_error_f": abs(signed),
        "signed_error_f": signed,
        "squared_error_f": signed * signed,
        "pmf_probability_at_observed": probability_at_observed,
        "log_score": -math.log(probability_at_observed),
        "crps_discrete": _discrete_crps(pmf, observed),
        "within_1f": abs(signed) <= 1.0,
        "within_2f": abs(signed) <= 2.0,
        "prediction_interval_low_f": int(row["prediction_interval_low_f"]),
        "prediction_interval_high_f": int(row["prediction_interval_high_f"]),
        "prediction_interval_hit": int(row["prediction_interval_low_f"]) <= observed <= int(row["prediction_interval_high_f"]),
        "label_source_record_id": row["label_source_record_id"],
        "label_revision_number": int(row["label_revision_number"]),
        "label_available_at_utc": row["label_available_at_utc"],
        "leakage_checked": True,
        "diagnostics_json": {
            "calibrated": row["calibrated_prediction_id"] is not None,
            "label_source": "public.wunderground_daily_tmax",
        },
    }


def _discrete_crps(pmf: dict[str, float], observed: int) -> float:
    forecast_cdf = 0.0
    total = 0.0
    for temp in TEMP_GRID_F:
        forecast_cdf += float(pmf[str(temp)])
        observed_cdf = 1.0 if temp >= observed else 0.0
        total += (forecast_cdf - observed_cdf) ** 2
    return total / len(TEMP_GRID_F)


def _start_evaluation_run(
    connection: Connection,
    *,
    run_id_text: str,
    run_name: str,
    prediction_kind: str,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
    first_row: dict[str, Any],
) -> str:
    row = connection.execute(
        text(
            """
            INSERT INTO reports.forecast_evaluation_runs (
                run_id_text,
                run_name,
                prediction_kind,
                status,
                start_date,
                end_date,
                cutoff_id,
                model_version_id,
                calibration_version_id,
                feature_version_id,
                source_code_git_sha,
                config_json
            )
            VALUES (
                :run_id_text,
                :run_name,
                :prediction_kind,
                'started',
                :start_date,
                :end_date,
                :cutoff_id,
                :model_version_id,
                :calibration_version_id,
                :feature_version_id,
                :source_code_git_sha,
                CAST(:config_json AS jsonb)
            )
            RETURNING evaluation_run_id
            """
        ),
        {
            "run_id_text": run_id_text,
            "run_name": run_name,
            "prediction_kind": prediction_kind,
            "start_date": start_date,
            "end_date": end_date,
            "cutoff_id": cutoff_id,
            "model_version_id": first_row["model_version_id"],
            "calibration_version_id": first_row["calibration_version_id"],
            "feature_version_id": first_row["feature_version_id"],
            "source_code_git_sha": current_git_sha(),
            "config_json": json.dumps({"source_of_truth": "wunderground_klga_tmax"}, sort_keys=True),
        },
    ).mappings().one()
    return str(row["evaluation_run_id"])


def _insert_daily_scores(
    connection: Connection,
    *,
    evaluation_run_id: str,
    daily_scores: list[dict[str, Any]],
) -> None:
    rows = [
        {
            **score,
            "evaluation_run_id": evaluation_run_id,
            "diagnostics_json": json.dumps(score["diagnostics_json"], sort_keys=True, default=str),
        }
        for score in daily_scores
    ]
    connection.execute(
        text(
            """
            INSERT INTO reports.forecast_evaluation_daily_scores (
                evaluation_run_id,
                target_date,
                cutoff_id,
                prediction_id,
                calibrated_prediction_id,
                settled_wu_tmax_f,
                expected_tmax_f,
                median_tmax_f,
                mode_tmax_f,
                absolute_error_f,
                signed_error_f,
                squared_error_f,
                pmf_probability_at_observed,
                log_score,
                within_1f,
                within_2f,
                prediction_interval_low_f,
                prediction_interval_high_f,
                prediction_interval_hit,
                label_source_record_id,
                label_revision_number,
                label_available_at_utc,
                leakage_checked,
                diagnostics_json
            )
            VALUES (
                :evaluation_run_id,
                :target_date,
                :cutoff_id,
                :prediction_id,
                :calibrated_prediction_id,
                :settled_wu_tmax_f,
                :expected_tmax_f,
                :median_tmax_f,
                :mode_tmax_f,
                :absolute_error_f,
                :signed_error_f,
                :squared_error_f,
                :pmf_probability_at_observed,
                :log_score,
                :within_1f,
                :within_2f,
                :prediction_interval_low_f,
                :prediction_interval_high_f,
                :prediction_interval_hit,
                :label_source_record_id,
                :label_revision_number,
                :label_available_at_utc,
                :leakage_checked,
                CAST(:diagnostics_json AS jsonb)
            )
            ON CONFLICT (evaluation_run_id, target_date, cutoff_id, prediction_id)
            DO UPDATE SET
                settled_wu_tmax_f = EXCLUDED.settled_wu_tmax_f,
                expected_tmax_f = EXCLUDED.expected_tmax_f,
                median_tmax_f = EXCLUDED.median_tmax_f,
                mode_tmax_f = EXCLUDED.mode_tmax_f,
                absolute_error_f = EXCLUDED.absolute_error_f,
                signed_error_f = EXCLUDED.signed_error_f,
                squared_error_f = EXCLUDED.squared_error_f,
                pmf_probability_at_observed = EXCLUDED.pmf_probability_at_observed,
                log_score = EXCLUDED.log_score,
                within_1f = EXCLUDED.within_1f,
                within_2f = EXCLUDED.within_2f,
                prediction_interval_hit = EXCLUDED.prediction_interval_hit,
                diagnostics_json = EXCLUDED.diagnostics_json
            """
        ),
        rows,
    )


def _finish_evaluation_run(
    connection: Connection,
    *,
    evaluation_run_id: str,
    metrics: dict[str, Any],
) -> None:
    connection.execute(
        text(
            """
            UPDATE reports.forecast_evaluation_runs
            SET status = 'success',
                finished_at = now(),
                metrics_json = CAST(:metrics_json AS jsonb)
            WHERE evaluation_run_id = :evaluation_run_id
            """
        ),
        {"evaluation_run_id": evaluation_run_id, "metrics_json": json.dumps(metrics, sort_keys=True)},
    )


def _insert_metric_rows(
    connection: Connection,
    *,
    evaluation_run_id: str,
    daily_scores: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    run = connection.execute(
        text(
            """
            SELECT cutoff_id, model_version_id, feature_version_id
            FROM reports.forecast_evaluation_runs
            WHERE evaluation_run_id = :evaluation_run_id
            """
        ),
        {"evaluation_run_id": evaluation_run_id},
    ).mappings().one()
    metric_rows = [
        {
            "metric_group": "forecast_accuracy",
            "metric_name": name,
            "metric_value": float(value) if isinstance(value, (int, float)) else None,
            "metric_text": None,
            "metric_json": "{}",
            "target_date": None,
            "cutoff_id": run["cutoff_id"],
            "model_version_id": run["model_version_id"],
            "feature_version_id": run["feature_version_id"],
        }
        for name, value in metrics.items()
    ]
    connection.execute(
        text(
            """
            INSERT INTO reports.metrics (
                metric_group,
                metric_name,
                metric_value,
                metric_text,
                metric_json,
                target_date,
                cutoff_id,
                model_version_id,
                feature_version_id
            )
            VALUES (
                :metric_group,
                :metric_name,
                :metric_value,
                :metric_text,
                CAST(:metric_json AS jsonb),
                :target_date,
                :cutoff_id,
                :model_version_id,
                :feature_version_id
            )
            """
        ),
        metric_rows,
    )
