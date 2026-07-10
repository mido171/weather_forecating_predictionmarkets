from __future__ import annotations

from datetime import date
import json
import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.constants import TEMP_GRID_F
from klga_tmax.models.pmf import shift_pmf, summarize_pmf
from klga_tmax.utils.git import current_git_sha


class TargetGridError(ValueError):
    pass


def calibrate_predictions(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
    prediction_kind: str,
) -> dict[str, int | float]:
    rows = _load_training_rows(
        connection,
        start_date=start_date,
        end_date=end_date,
        cutoff_id=cutoff_id,
        prediction_kind=prediction_kind,
    )
    if not rows:
        raise RuntimeError("no settled final predictions available for calibration")
    errors = []
    for row in rows:
        observed = int(row["high_temp_f"])
        if observed not in TEMP_GRID_F:
            raise TargetGridError(f"settled WU Tmax {observed} is outside TEMP_GRID_F")
        errors.append(observed - float(row["expected_tmax_f"]))
    bias = sum(errors) / len(errors)
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    model_version_id = str(rows[0]["model_version_id"])
    calibration_version_id = _calibration_version_id(
        connection,
        calibration_name="whole_degree_bias_shift_v1",
        prediction_kind=prediction_kind,
        model_version_id=model_version_id,
        start_date=start_date,
        end_date=end_date,
        cutoff_id=cutoff_id,
        config={"bias_shift_f": bias},
        metrics={"training_rows": len(rows), "mae_f": mae, "rmse_f": rmse, "bias_f": bias},
    )
    applied = _apply_calibration(
        connection,
        start_date=start_date,
        end_date=end_date,
        cutoff_id=cutoff_id,
        prediction_kind=prediction_kind,
        calibration_version_id=calibration_version_id,
        bias=bias,
    )
    return {
        "predictions.calibration_versions": 1,
        "predictions.calibrated_predictions": applied,
        "training_rows": len(rows),
        "bias_f": bias,
        "training_mae_f": mae,
        "training_rmse_f": rmse,
    }


def _load_training_rows(
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
            SELECT
                fp.final_prediction_id,
                fp.model_version_id,
                fp.expected_tmax_f,
                fp.pmf_json,
                ti.target_date,
                ti.cutoff_id,
                actual.tmax_f AS high_temp_f
            FROM predictions.final_predictions fp
            JOIN gold.target_instances ti
              ON ti.target_instance_id = fp.target_instance_id
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


def _calibration_version_id(
    connection: Connection,
    *,
    calibration_name: str,
    prediction_kind: str,
    model_version_id: str,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    row = connection.execute(
        text(
            """
            INSERT INTO predictions.calibration_versions (
                calibration_name,
                prediction_kind,
                model_version_id,
                training_start_date,
                training_end_date,
                cutoff_id,
                method,
                config_json,
                metrics_json,
                artifact_hash
            )
            VALUES (
                :calibration_name,
                :prediction_kind,
                :model_version_id,
                :training_start_date,
                :training_end_date,
                :cutoff_id,
                'whole_degree_bias_shift',
                CAST(:config_json AS jsonb),
                CAST(:metrics_json AS jsonb),
                :artifact_hash
            )
            ON CONFLICT DO NOTHING
            RETURNING calibration_version_id
            """
        ),
        {
            "calibration_name": calibration_name,
            "prediction_kind": prediction_kind,
            "model_version_id": model_version_id,
            "training_start_date": start_date,
            "training_end_date": end_date,
            "cutoff_id": cutoff_id,
            "config_json": json.dumps(config, sort_keys=True),
            "metrics_json": json.dumps(metrics, sort_keys=True),
            "artifact_hash": current_git_sha(),
        },
    ).mappings().first()
    if row is not None:
        return str(row["calibration_version_id"])
    existing = connection.execute(
        text(
            """
            SELECT calibration_version_id
            FROM predictions.calibration_versions
            WHERE calibration_name = :calibration_name
              AND prediction_kind = :prediction_kind
              AND model_version_id = :model_version_id
              AND training_start_date = :training_start_date
              AND training_end_date = :training_end_date
              AND COALESCE(cutoff_id, '') = COALESCE(:cutoff_id, '')
              AND method = 'whole_degree_bias_shift'
              AND md5(config_json::text) = md5(CAST(:config_json AS jsonb)::text)
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {
            "calibration_name": calibration_name,
            "prediction_kind": prediction_kind,
            "model_version_id": model_version_id,
            "training_start_date": start_date,
            "training_end_date": end_date,
            "cutoff_id": cutoff_id,
            "config_json": json.dumps(config, sort_keys=True),
        },
    ).mappings().one()
    return str(existing["calibration_version_id"])


def _apply_calibration(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
    prediction_kind: str,
    calibration_version_id: str,
    bias: float,
) -> int:
    rows = connection.execute(
        text(
            """
            SELECT
                fp.final_prediction_id,
                fp.pmf_json,
                ti.target_date,
                ti.cutoff_id
            FROM predictions.final_predictions fp
            JOIN gold.target_instances ti
              ON ti.target_instance_id = fp.target_instance_id
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
    count = 0
    for row in rows:
        shifted = shift_pmf(dict(row["pmf_json"]), bias)
        summary = summarize_pmf(shifted)
        result = connection.execute(
            text(
                """
                INSERT INTO predictions.calibrated_predictions (
                    final_prediction_id,
                    calibration_version_id,
                    pmf_json,
                    expected_tmax_f,
                    median_tmax_f,
                    mode_tmax_f,
                    prediction_interval_low_f,
                    prediction_interval_high_f,
                    uncertainty_f,
                    diagnostics_json
                )
                VALUES (
                    :final_prediction_id,
                    :calibration_version_id,
                    CAST(:pmf_json AS jsonb),
                    :expected_tmax_f,
                    :median_tmax_f,
                    :mode_tmax_f,
                    :prediction_interval_low_f,
                    :prediction_interval_high_f,
                    :uncertainty_f,
                    CAST(:diagnostics_json AS jsonb)
                )
                ON CONFLICT (final_prediction_id, calibration_version_id)
                DO UPDATE SET
                    pmf_json = EXCLUDED.pmf_json,
                    expected_tmax_f = EXCLUDED.expected_tmax_f,
                    median_tmax_f = EXCLUDED.median_tmax_f,
                    mode_tmax_f = EXCLUDED.mode_tmax_f,
                    prediction_interval_low_f = EXCLUDED.prediction_interval_low_f,
                    prediction_interval_high_f = EXCLUDED.prediction_interval_high_f,
                    uncertainty_f = EXCLUDED.uncertainty_f,
                    diagnostics_json = EXCLUDED.diagnostics_json
                """
            ),
            {
                "final_prediction_id": row["final_prediction_id"],
                "calibration_version_id": calibration_version_id,
                "pmf_json": json.dumps(summary.pmf, sort_keys=True),
                "expected_tmax_f": summary.expected_tmax_f,
                "median_tmax_f": summary.median_tmax_f,
                "mode_tmax_f": summary.mode_tmax_f,
                "prediction_interval_low_f": summary.prediction_interval_low_f,
                "prediction_interval_high_f": summary.prediction_interval_high_f,
                "uncertainty_f": summary.uncertainty_f,
                "diagnostics_json": json.dumps({"bias_shift_f": bias}, sort_keys=True),
            },
        )
        count += result.rowcount or 0
    return count
