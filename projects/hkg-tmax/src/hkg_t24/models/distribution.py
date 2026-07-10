"""Jira003 distributional calibration and threshold probabilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import erf, pi, sqrt
from statistics import mean
from typing import Any

from lightgbm import LGBMRegressor

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.constants import DISTRIBUTION_THRESHOLDS_C
from hkg_t24.features.matrix_builder import FeatureMatrixRow
from hkg_t24.models.final_formula import SystemPrediction, with_distribution
from hkg_t24.validation.metrics import ForecastMetricSummary, clip, forecast_metrics, percentile

DISTRIBUTION_VERSION = "distribution_v1_20260627"


@dataclass(frozen=True)
class DistributionTrainingResult:
    distribution_status: str
    scoreboard: ForecastMetricSummary | None
    p50_scoreboard: ForecastMetricSummary | None
    updated_predictions: tuple[SystemPrediction, ...]
    monotonic_repair_count: int
    threshold_key_count: int


def threshold_probability_key(threshold_c: float) -> str:
    return f"prob_tmax_ge_{threshold_c:.1f}".replace(".", "_")


def threshold_probability_keys() -> tuple[str, ...]:
    return tuple(threshold_probability_key(threshold) for threshold in DISTRIBUTION_THRESHOLDS_C)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def gaussian_threshold_probabilities(point_c: float, expected_abs_error_c: float) -> dict[str, float]:
    sigma = max(expected_abs_error_c * sqrt(pi / 2.0), 0.60)
    return {
        threshold_probability_key(threshold): clip(
            1.0 - _normal_cdf((threshold - point_c) / sigma),
            0.001,
            0.999,
        )
        for threshold in DISTRIBUTION_THRESHOLDS_C
    }


def _feature_vector(prediction: SystemPrediction, row: FeatureMatrixRow) -> list[float]:
    spread = row.features.get("gefsens__center__tmax_spread_p90_p10_c")
    official = row.features.get("official__forecast_max_c")
    gfs = row.features.get("gfs__center__tmax_c")
    gefs = row.features.get("gefsens__center__tmax_p50_c")
    if gefs is None:
        gefs = row.features.get("gefsmean__center__tmax_c")
    numeric_spread = float(spread) if isinstance(spread, int | float) and not isinstance(spread, bool) else 0.0
    official_value = float(official) if isinstance(official, int | float) and not isinstance(official, bool) else 0.0
    gfs_value = float(gfs) if isinstance(gfs, int | float) and not isinstance(gfs, bool) else 0.0
    gefs_value = float(gefs) if isinstance(gefs, int | float) and not isinstance(gefs, bool) else 0.0
    expert_disagreement = max(official_value, gfs_value, gefs_value) - min(official_value, gfs_value, gefs_value)
    return [
        0.0 if prediction.final_pre_distribution_c is None else float(prediction.final_pre_distribution_c),
        float(row.target_date_hkt.month),
        numeric_spread,
        expert_disagreement,
        float(prediction.specialist_total_correction_c),
    ]


def _labels_by_date(rows: Sequence[FeatureMatrixRow]) -> dict[object, float]:
    return {row.target_date_hkt: float(row.target_tmax_c) for row in rows if row.target_tmax_c is not None}


def _residuals_by_month(
    predictions: Sequence[SystemPrediction],
    labels_by_date: Mapping[object, float],
) -> tuple[dict[int, list[float]], list[float]]:
    by_month: dict[int, list[float]] = {month: [] for month in range(1, 13)}
    global_residuals: list[float] = []
    for prediction in predictions:
        if prediction.final_pre_distribution_c is None:
            continue
        label = labels_by_date.get(prediction.target_date_hkt)
        if label is None:
            continue
        residual = label - prediction.final_pre_distribution_c
        by_month[prediction.target_date_hkt.month].append(residual)
        global_residuals.append(residual)
    return by_month, global_residuals


def _empirical_quantiles(
    prediction: SystemPrediction,
    residuals_by_month: Mapping[int, Sequence[float]],
    global_residuals: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    point = 0.0 if prediction.final_pre_distribution_c is None else float(prediction.final_pre_distribution_c)
    residuals = residuals_by_month.get(prediction.target_date_hkt.month, ())
    if len(residuals) < 100:
        residuals = global_residuals
    if not residuals:
        return point - 1.20, point - 0.60, point, point + 0.60, point + 1.20, 1.20
    q10 = point + percentile(list(residuals), 0.10)
    q25 = point + percentile(list(residuals), 0.25)
    q75 = point + percentile(list(residuals), 0.75)
    q90 = point + percentile(list(residuals), 0.90)
    q25 = min(q25, point)
    q10 = min(q10, q25)
    q75 = max(q75, point)
    q90 = max(q90, q75)
    expected_abs_error = max(mean(abs(value) for value in residuals), 0.20)
    return q10, q25, point, q75, q90, clip(expected_abs_error, 0.20, 3.00)


def _core_inputs_missing(row: FeatureMatrixRow) -> bool:
    official = row.features.get("official__forecast_max_c")
    gfs = row.features.get("gfs__center__tmax_c")
    gefs = row.features.get("gefsens__center__tmax_p50_c")
    if gefs is None:
        gefs = row.features.get("gefsmean__center__tmax_c")
    return official is None and gfs is None and gefs is None


def _confidence(
    expected_abs_error_c: float,
    prediction: SystemPrediction,
    row: FeatureMatrixRow,
) -> tuple[str, bool]:
    spread = row.features.get("gefsens__center__tmax_spread_p90_p10_c")
    numeric_spread = float(spread) if isinstance(spread, int | float) and not isinstance(spread, bool) else 999.0
    if expected_abs_error_c <= 0.55 and numeric_spread <= 0.80:
        state = "HIGH"
    elif expected_abs_error_c <= 0.85:
        state = "MEDIUM"
    else:
        state = "LOW"
    no_trade = (
        state == "LOW"
        or expected_abs_error_c > 1.00
        or _core_inputs_missing(row)
        or prediction.leakage_status != "passed"
    )
    return state, no_trade


def _try_lightgbm_distribution(
    predictions: Sequence[SystemPrediction],
    rows: Sequence[FeatureMatrixRow],
    labels_by_date: Mapping[object, float],
) -> tuple[bool, dict[float, LGBMRegressor], LGBMRegressor | None]:
    training = [
        (prediction, row, labels_by_date[prediction.target_date_hkt])
        for prediction, row in zip(predictions, rows, strict=True)
        if prediction.final_pre_distribution_c is not None and prediction.target_date_hkt in labels_by_date
    ]
    if len(training) < 120:
        return False, {}, None
    x = [_feature_vector(prediction, row) for prediction, row, _ in training]
    residuals: list[float] = []
    for prediction, _, label in training:
        if prediction.final_pre_distribution_c is None:
            continue
        residuals.append(label - float(prediction.final_pre_distribution_c))
    quantile_models: dict[float, LGBMRegressor] = {}
    try:
        for alpha in (0.10, 0.25, 0.50, 0.75, 0.90):
            model = LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=7,
                max_depth=3,
                min_child_samples=20,
                reg_lambda=1.0,
                random_state=20260626,
                verbosity=-1,
            )
            model.fit(x, residuals)
            quantile_models[alpha] = model
        error_model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=3,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=20260626,
            verbosity=-1,
        )
        error_model.fit(x, [abs(value) for value in residuals])
    except Exception:
        return False, {}, None
    return True, quantile_models, error_model


def train_distribution_from_inputs(
    predictions: Sequence[SystemPrediction],
    rows: Sequence[FeatureMatrixRow],
    *,
    force_empirical: bool = False,
) -> DistributionTrainingResult:
    if len(predictions) != len(rows):
        raise ValueError("System predictions and rows must be aligned")
    labels_by_date = _labels_by_date(rows)
    residuals_by_month, global_residuals = _residuals_by_month(predictions, labels_by_date)
    lightgbm_ok, quantile_models, error_model = (
        (False, {}, None) if force_empirical else _try_lightgbm_distribution(predictions, rows, labels_by_date)
    )
    updated: list[SystemPrediction] = []
    repair_count = 0
    status = "promoted" if lightgbm_ok and error_model is not None else "demoted_empirical_fallback"
    for prediction, row in zip(predictions, rows, strict=True):
        if prediction.final_pre_distribution_c is None:
            expected_error = 3.0
            updated.append(
                with_distribution(
                    prediction,
                    p10_c=None,
                    p25_c=None,
                    p50_c=None,
                    p75_c=None,
                    p90_c=None,
                    expected_abs_error_c=expected_error,
                    threshold_probabilities=gaussian_threshold_probabilities(0.0, expected_error),
                    confidence_state="LOW",
                    no_trade_flag=True,
                    distribution_status="failed_closed",
                    quantile_monotonic_repair=False,
                )
            )
            continue
        elif status == "promoted" and error_model is not None:
            vector = _feature_vector(prediction, row)
            residual_quantiles = [float(quantile_models[alpha].predict([vector])[0]) for alpha in (0.10, 0.25, 0.50, 0.75, 0.90)]
            quantiles = [float(prediction.final_pre_distribution_c) + value for value in residual_quantiles]
            repaired = sorted(quantiles)
            if repaired != quantiles:
                repair_count += 1
            q10, q25, q50, q75, q90 = repaired
            expected_error = clip(float(error_model.predict([vector])[0]), 0.20, 3.00)
            row_status = status
        else:
            q10, q25, q50, q75, q90, expected_error = _empirical_quantiles(
                prediction,
                residuals_by_month,
                global_residuals,
            )
            row_status = status
        confidence_state, no_trade_flag = _confidence(expected_error, prediction, row)
        updated.append(
            with_distribution(
                prediction,
                p10_c=q10,
                p25_c=q25,
                p50_c=q50,
                p75_c=q75,
                p90_c=q90,
                expected_abs_error_c=expected_error,
                threshold_probabilities=gaussian_threshold_probabilities(q50, expected_error),
                confidence_state=confidence_state,
                no_trade_flag=no_trade_flag,
                distribution_status=row_status,
                quantile_monotonic_repair=repair_count > 0,
            )
        )

    pairs_pre = [
        (float(prediction.final_pre_distribution_c), labels_by_date[prediction.target_date_hkt])
        for prediction in predictions
        if prediction.final_pre_distribution_c is not None and prediction.target_date_hkt in labels_by_date
    ]
    pairs_p50 = [
        (float(prediction.p50_c), labels_by_date[prediction.target_date_hkt])
        for prediction in updated
        if prediction.p50_c is not None and prediction.target_date_hkt in labels_by_date
    ]
    pre_metrics = forecast_metrics(pairs_pre) if pairs_pre else None
    p50_metrics = forecast_metrics(pairs_p50) if pairs_p50 else None
    if (
        status == "promoted"
        and pre_metrics is not None
        and p50_metrics is not None
        and p50_metrics.mae_c - pre_metrics.mae_c > 0.005
    ):
        fallback = train_distribution_from_inputs(
            predictions,
            rows,
            force_empirical=True,
        )
        return DistributionTrainingResult(
            distribution_status="demoted_empirical_fallback",
            scoreboard=fallback.scoreboard,
            p50_scoreboard=fallback.p50_scoreboard,
            updated_predictions=fallback.updated_predictions,
            monotonic_repair_count=fallback.monotonic_repair_count,
            threshold_key_count=fallback.threshold_key_count,
        )
    return DistributionTrainingResult(
        distribution_status=status,
        scoreboard=pre_metrics,
        p50_scoreboard=p50_metrics,
        updated_predictions=tuple(updated),
        monotonic_repair_count=repair_count,
        threshold_key_count=len(threshold_probability_keys()),
    )


def write_distribution_reports(writer: ReportWriter, result: DistributionTrainingResult) -> None:
    metrics = result.p50_scoreboard
    writer.write_csv(
        "distribution_scoreboard.csv",
        (
            "distribution_version",
            "distribution_status",
            "row_count",
            "mae_c",
            "rmse_c",
            "bias_c",
            "p90_abs_error_c",
            "monotonic_repair_count",
        ),
        [
            (
                DISTRIBUTION_VERSION,
                result.distribution_status,
                None if metrics is None else metrics.row_count,
                None if metrics is None else metrics.mae_c,
                None if metrics is None else metrics.rmse_c,
                None if metrics is None else metrics.bias_c,
                None if metrics is None else metrics.p90_abs_error_c,
                result.monotonic_repair_count,
            )
        ],
    )
    writer.write_csv(
        "distribution_calibration_report.csv",
        (
            "distribution_status",
            "threshold_key_count",
            "monotonic_repair_count",
            "p50_mae_c",
        ),
        [
            (
                result.distribution_status,
                result.threshold_key_count,
                result.monotonic_repair_count,
                None if metrics is None else metrics.mae_c,
            )
        ],
    )
    writer.write_csv(
        "threshold_probability_scoreboard.csv",
        ("key", "present"),
        [(key, True) for key in threshold_probability_keys()],
    )
    writer.write_csv(
        "prediction_interval_coverage_report.csv",
        ("distribution_status", "row_count", "p10_p90_rows"),
        [
            (
                result.distribution_status,
                len(result.updated_predictions),
                sum(1 for prediction in result.updated_predictions if prediction.p10_c is not None and prediction.p90_c is not None),
            )
        ],
    )
    sections = (
        ("Status", result.distribution_status),
        ("Threshold Keys", f"{result.threshold_key_count} keys from prob_tmax_ge_20_0 to prob_tmax_ge_40_0."),
        ("Monotonic Repairs", str(result.monotonic_repair_count)),
    )
    writer.write_root_report("distribution_calibration_report.md", "HKG-T24-003 Distribution Calibration Report", sections)
    writer.write_root_report(
        "calibration_report.md",
        "HKG-T24-003 Calibration Report Compatibility Copy",
        (("Compatibility Header", "Compatibility copy of reports/distribution_calibration_report.md."),) + sections,
    )


def persist_system_predictions(connection: Any, predictions: Sequence[SystemPrediction]) -> int:
    count = 0
    with connection.cursor() as cursor:
        for prediction in predictions:
            cursor.execute(
                """
                INSERT INTO model_oof.system_prediction (
                  target_date_hkt, cutoff_id, snapshot_id, system_version, router_selected,
                  router_selection_reason, base_forecast_c, specialist_total_correction_c,
                  final_pre_distribution_c, final_point_tmax_c, p10_c, p25_c, p50_c,
                  p75_c, p90_c, expected_abs_error_c, threshold_probabilities_jsonb,
                  confidence_state, no_trade_flag, distribution_status,
                  quantile_monotonic_repair, component_jsonb, leakage_status
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb,%s)
                ON CONFLICT (target_date_hkt, cutoff_id, system_version) DO UPDATE SET
                  snapshot_id = EXCLUDED.snapshot_id,
                  router_selected = EXCLUDED.router_selected,
                  router_selection_reason = EXCLUDED.router_selection_reason,
                  base_forecast_c = EXCLUDED.base_forecast_c,
                  specialist_total_correction_c = EXCLUDED.specialist_total_correction_c,
                  final_pre_distribution_c = EXCLUDED.final_pre_distribution_c,
                  final_point_tmax_c = EXCLUDED.final_point_tmax_c,
                  p10_c = EXCLUDED.p10_c,
                  p25_c = EXCLUDED.p25_c,
                  p50_c = EXCLUDED.p50_c,
                  p75_c = EXCLUDED.p75_c,
                  p90_c = EXCLUDED.p90_c,
                  expected_abs_error_c = EXCLUDED.expected_abs_error_c,
                  threshold_probabilities_jsonb = EXCLUDED.threshold_probabilities_jsonb,
                  confidence_state = EXCLUDED.confidence_state,
                  no_trade_flag = EXCLUDED.no_trade_flag,
                  distribution_status = EXCLUDED.distribution_status,
                  quantile_monotonic_repair = EXCLUDED.quantile_monotonic_repair,
                  component_jsonb = EXCLUDED.component_jsonb,
                  leakage_status = EXCLUDED.leakage_status,
                  created_at_utc = now()
                """,
                (
                    prediction.target_date_hkt,
                    prediction.cutoff_id,
                    prediction.snapshot_id,
                    prediction.system_version,
                    prediction.router_selected,
                    prediction.router_selection_reason,
                    prediction.base_forecast_c,
                    prediction.specialist_total_correction_c,
                    prediction.final_pre_distribution_c,
                    prediction.final_point_tmax_c,
                    prediction.p10_c,
                    prediction.p25_c,
                    prediction.p50_c,
                    prediction.p75_c,
                    prediction.p90_c,
                    prediction.expected_abs_error_c,
                    json.dumps(prediction.threshold_probabilities, sort_keys=True),
                    prediction.confidence_state,
                    prediction.no_trade_flag,
                    prediction.distribution_status,
                    prediction.quantile_monotonic_repair,
                    json.dumps(prediction.component_jsonb, sort_keys=True),
                    prediction.leakage_status,
                ),
            )
            count += 1
    return count
