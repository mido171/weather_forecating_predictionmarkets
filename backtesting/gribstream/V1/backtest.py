from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from math import sqrt
from typing import Iterable, Mapping

from . import db
from .blend import PredictionRunResult, compute_daily_predictions
from .config import EVALUATION_END_DATE, EVALUATION_START_DATE, STATION, isoformat_utc, utc_now
from .model_catalog import MODEL_SPECS
from .weights import compute_daily_model_weights

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestRunResult:
    weight_rows: list[dict[str, object]]
    prediction_result: PredictionRunResult
    metrics_rows: list[dict[str, object]]
    coverage_rows: list[dict[str, object]]
    diagnostics: dict[str, int]


def _metric_row(
    *,
    metric_scope: str,
    metric_name: str,
    evaluation_start: date,
    evaluation_end: date,
    errors: list[float],
) -> dict[str, object]:
    abs_errors = [abs(error) for error in errors]
    n_days = len(errors)
    return {
        "metric_scope": metric_scope,
        "metric_name": metric_name,
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "n_days": n_days,
        "mae_f": sum(abs_errors) / n_days,
        "rmse_f": sqrt(sum(error * error for error in errors) / n_days),
        "bias_f": sum(errors) / n_days,
        "median_abs_error_f": statistics.median(abs_errors),
        "within_0_5f": sum(abs_error <= 0.5 for abs_error in abs_errors) / n_days,
        "within_1f": sum(abs_error <= 1.0 for abs_error in abs_errors) / n_days,
        "within_2f": sum(abs_error <= 2.0 for abs_error in abs_errors) / n_days,
        "created_at_utc": isoformat_utc(utc_now()),
    }


def _metric_row_from_pairs(
    *,
    metric_scope: str,
    metric_name: str,
    evaluation_start: date,
    evaluation_end: date,
    pairs: Iterable[tuple[float, float]],
) -> dict[str, object] | None:
    pair_list = list(pairs)
    if not pair_list:
        return None
    errors = [prediction - actual for prediction, actual in pair_list]
    return _metric_row(
        metric_scope=metric_scope,
        metric_name=metric_name,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        errors=errors,
    )


def _diagnostic_row(
    metric_name: str,
    evaluation_start: date,
    evaluation_end: date,
    count: int,
) -> dict[str, object]:
    return {
        "metric_scope": "diagnostic",
        "metric_name": metric_name,
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "n_days": count,
        "mae_f": None,
        "rmse_f": None,
        "bias_f": None,
        "median_abs_error_f": None,
        "within_0_5f": None,
        "within_1f": None,
        "within_2f": None,
        "created_at_utc": isoformat_utc(utc_now()),
    }


def compute_diagnostics(
    connection,
    *,
    evaluation_start: date = EVALUATION_START_DATE,
    evaluation_end: date = EVALUATION_END_DATE,
) -> dict[str, int]:
    diagnostics: dict[str, int] = {}
    blend_days = connection.execute(
        """
        SELECT model_code, COUNT(*) AS c
        FROM daily_model_weights
        WHERE station_id = ?
          AND settlement_date_local BETWEEN ? AND ?
          AND final_weight > 0
        GROUP BY model_code
        ORDER BY model_code
        """,
        (STATION.station_id, evaluation_start.isoformat(), evaluation_end.isoformat()),
    ).fetchall()
    for row in blend_days:
        diagnostics[f"days_in_blend::{row['model_code']}"] = int(row["c"])
    prediction_rows = connection.execute(
        """
        SELECT actual_tmax_f, family_capped_blend_f, nbm_only_f, hrrr_only_f, rap_only_f
        FROM daily_predictions
        WHERE station_id = ?
          AND settlement_date_local BETWEEN ? AND ?
        ORDER BY settlement_date_local
        """,
        (STATION.station_id, evaluation_start.isoformat(), evaluation_end.isoformat()),
    ).fetchall()
    beats_nbm = 0
    beats_best_noaa_short = 0
    for row in prediction_rows:
        actual_tmax_f = float(row["actual_tmax_f"])
        if row["family_capped_blend_f"] is None:
            continue
        family_error = abs(float(row["family_capped_blend_f"]) - actual_tmax_f)
        if row["nbm_only_f"] is not None and family_error < abs(float(row["nbm_only_f"]) - actual_tmax_f):
            beats_nbm += 1
        short_range_errors = [
            abs(float(row[field_name]) - actual_tmax_f)
            for field_name in ("hrrr_only_f", "rap_only_f")
            if row[field_name] is not None
        ]
        if short_range_errors and family_error < min(short_range_errors):
            beats_best_noaa_short += 1
    diagnostics["family_capped_beats_nbm"] = beats_nbm
    diagnostics["family_capped_beats_best_noaa_short"] = beats_best_noaa_short
    return diagnostics


def compute_metrics_summary(
    connection,
    *,
    evaluation_start: date = EVALUATION_START_DATE,
    evaluation_end: date = EVALUATION_END_DATE,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    metrics_rows: list[dict[str, object]] = []
    model_errors = connection.execute(
        """
        SELECT model_code, error_f
        FROM model_daily_errors
        WHERE station_id = ?
          AND settlement_date_local BETWEEN ? AND ?
        ORDER BY model_code, settlement_date_local
        """,
        (STATION.station_id, evaluation_start.isoformat(), evaluation_end.isoformat()),
    ).fetchall()
    grouped_model_errors: dict[str, list[float]] = defaultdict(list)
    for row in model_errors:
        grouped_model_errors[str(row["model_code"])].append(float(row["error_f"]))
    for model_code, errors in sorted(grouped_model_errors.items()):
        metrics_rows.append(
            _metric_row(
                metric_scope="model",
                metric_name=model_code,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                errors=errors,
            )
        )
    prediction_rows = connection.execute(
        """
        SELECT *
        FROM daily_predictions
        WHERE station_id = ?
          AND settlement_date_local BETWEEN ? AND ?
        ORDER BY settlement_date_local
        """,
        (STATION.station_id, evaluation_start.isoformat(), evaluation_end.isoformat()),
    ).fetchall()
    prediction_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in prediction_rows:
        actual_tmax_f = float(row["actual_tmax_f"])
        for field_name, scope in (
            ("equal_weight_blend_f", "blend"),
            ("inverse_rmse_blend_f", "blend"),
            ("family_capped_blend_f", "blend"),
            ("nbm_only_f", "baseline"),
            ("hrrr_only_f", "baseline"),
            ("rap_only_f", "baseline"),
            ("gfs_only_f", "baseline"),
        ):
            if row[field_name] is not None:
                prediction_pairs[f"{scope}::{field_name}"].append((float(row[field_name]), actual_tmax_f))
    for scoped_name, pairs in sorted(prediction_pairs.items()):
        scope, name = scoped_name.split("::", 1)
        metric_row = _metric_row_from_pairs(
            metric_scope=scope,
            metric_name=name.removesuffix("_f"),
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            pairs=pairs,
        )
        if metric_row is not None:
            metrics_rows.append(metric_row)
    family_rows = connection.execute(
        """
        SELECT w.settlement_date_local, w.family, w.bias_corrected_tmax_f, t.actual_tmax_f
        FROM daily_model_weights w
        JOIN nws_daily_settlements t
          ON t.station_id = w.station_id
         AND t.settlement_date_local = w.settlement_date_local
        WHERE w.station_id = ?
          AND w.settlement_date_local BETWEEN ? AND ?
          AND w.included_in_blend = 1
          AND w.bias_corrected_tmax_f IS NOT NULL
        ORDER BY w.settlement_date_local, w.family, w.model_code
        """,
        (STATION.station_id, evaluation_start.isoformat(), evaluation_end.isoformat()),
    ).fetchall()
    grouped_family_rows: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in family_rows:
        grouped_family_rows[(str(row["settlement_date_local"]), str(row["family"]))].append(row)
    family_predictions: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (_, family), rows in grouped_family_rows.items():
        actual_tmax_f = float(rows[0]["actual_tmax_f"])
        prediction_f = sum(float(row["bias_corrected_tmax_f"]) for row in rows) / float(len(rows))
        family_predictions[family].append((prediction_f, actual_tmax_f))
    for family, pairs in sorted(family_predictions.items()):
        metric_row = _metric_row_from_pairs(
            metric_scope="family",
            metric_name=family,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            pairs=pairs,
        )
        if metric_row is not None:
            metrics_rows.append(metric_row)
    diagnostics = compute_diagnostics(
        connection,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )
    for metric_name, count in sorted(diagnostics.items()):
        metrics_rows.append(_diagnostic_row(metric_name, evaluation_start, evaluation_end, count))
    connection.execute(
        "DELETE FROM metrics_summary WHERE evaluation_start = ? AND evaluation_end = ?",
        (evaluation_start.isoformat(), evaluation_end.isoformat()),
    )
    db.commit(connection)
    db.replace_metrics_summary(connection, metrics_rows)
    db.commit(connection)
    LOGGER.info(
        "Persisted metrics_summary rows=%d range=%s..%s",
        len(metrics_rows),
        evaluation_start,
        evaluation_end,
    )
    return metrics_rows, diagnostics


def compute_coverage_summary(
    connection,
    *,
    evaluation_start: date = EVALUATION_START_DATE,
    evaluation_end: date = EVALUATION_END_DATE,
) -> list[dict[str, object]]:
    successful_requests = connection.execute(
        """
        SELECT model_code,
               MIN(settlement_date_local) AS first_date_fetched,
               MAX(settlement_date_local) AS last_date_fetched,
               COUNT(DISTINCT settlement_date_local) AS fetched_day_count
        FROM gribstream_requests
        WHERE station_id = ?
          AND success = 1
        GROUP BY model_code
        """,
        (STATION.station_id,),
    ).fetchall()
    success_by_model = {str(row["model_code"]): row for row in successful_requests}
    scored_days = connection.execute(
        """
        SELECT model_code, COUNT(*) AS scored_day_count
        FROM model_daily_errors
        WHERE station_id = ?
          AND settlement_date_local BETWEEN ? AND ?
        GROUP BY model_code
        """,
        (STATION.station_id, evaluation_start.isoformat(), evaluation_end.isoformat()),
    ).fetchall()
    scored_by_model = {str(row["model_code"]): int(row["scored_day_count"]) for row in scored_days}
    rows_to_persist: list[dict[str, object]] = []
    created_at_utc = isoformat_utc(utc_now())
    for spec in MODEL_SPECS:
        success_row = success_by_model.get(spec.model_code)
        rows_to_persist.append(
            {
                "model_code": spec.model_code,
                "role": spec.role,
                "archive_start": spec.archive_start.isoformat(),
                "first_date_fetched": str(success_row["first_date_fetched"]) if success_row else None,
                "last_date_fetched": str(success_row["last_date_fetched"]) if success_row else None,
                "fetched_day_count": int(success_row["fetched_day_count"]) if success_row else 0,
                "scored_day_count": int(scored_by_model.get(spec.model_code, 0)),
                "notes": spec.notes,
                "created_at_utc": created_at_utc,
            }
        )
    connection.execute("DELETE FROM coverage_summary")
    db.commit(connection)
    db.replace_coverage_summary(connection, rows_to_persist)
    db.commit(connection)
    LOGGER.info("Persisted coverage_summary rows=%d", len(rows_to_persist))
    return rows_to_persist


def run_backtest(
    connection,
    *,
    evaluation_start: date = EVALUATION_START_DATE,
    evaluation_end: date = EVALUATION_END_DATE,
    include_live_only: bool = False,
) -> BacktestRunResult:
    weight_rows = compute_daily_model_weights(
        connection,
        start_date=evaluation_start,
        end_date=evaluation_end,
        include_live_only=include_live_only,
    )
    prediction_result = compute_daily_predictions(
        connection,
        start_date=evaluation_start,
        end_date=evaluation_end,
        require_truth=True,
    )
    metrics_rows, diagnostics = compute_metrics_summary(
        connection,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )
    coverage_rows = compute_coverage_summary(
        connection,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )
    return BacktestRunResult(
        weight_rows=weight_rows,
        prediction_result=prediction_result,
        metrics_rows=metrics_rows,
        coverage_rows=coverage_rows,
        diagnostics=diagnostics,
    )
