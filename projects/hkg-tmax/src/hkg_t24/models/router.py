"""Jira003 expert-router training, replay, reporting, and persistence."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.constants import (
    CUTOFF_ID,
    EXPERT_STRICT_WEIGHT_CAPS,
    ROUTER_EXPERTS,
    ROUTER_SHORT_IDS,
    STRICT_SCHEMA_VERSION,
)
from hkg_t24.features.matrix_builder import FeatureMatrixRow, FeatureValue
from hkg_t24.models.db_expert_factory import (
    folds_for_scope,
    labels_from_rows,
    load_feature_matrix_rows,
)
from hkg_t24.models.expected_error import (
    ExpertExpectedErrorModel,
    context_feature_names,
    context_vector,
    fit_expected_error_model,
)
from hkg_t24.models.experts import ExpertPrediction, generate_expert_oof_predictions
from hkg_t24.models.static_weights import (
    apply_caps_and_masks,
    blend_static_dynamic_weights,
    dynamic_weights_from_expected_errors,
    optimize_static_weights,
    select_tau_lambda,
)
from hkg_t24.timeutils import snapshot_id
from hkg_t24.validation.metrics import ForecastMetricSummary, forecast_metrics

ROUTER_VERSION = "router_v1_20260627"


@dataclass(frozen=True)
class RouterPrediction:
    target_date_hkt: date
    cutoff_id: str
    snapshot_id: str
    router_id: str
    router_version: str
    router_scope: str
    fold_id: str
    base_forecast_c: float | None
    static_weights: dict[str, float]
    dynamic_weights: dict[str, float]
    final_weights: dict[str, float]
    expected_error_c_by_expert: dict[str, float]
    availability_mask: dict[str, bool]
    selected_tau: float
    selected_lambda: float
    promotion_status: str
    demotion_reason: str | None
    expert_mask: dict[str, bool]
    cap_trace: dict[str, object]
    leakage_status: str


@dataclass(frozen=True)
class RouterTrainingResult:
    router_id: str
    router_scope: str
    promotion_status: str
    demotion_reason: str | None
    selected_tau: float
    selected_lambda: float
    metrics: ForecastMetricSummary | None
    baseline_metrics: ForecastMetricSummary | None
    mae_delta_vs_baseline: float | None
    mae_delta_vs_r0: float | None
    p90_abs_error_delta: float | None
    predictions: tuple[RouterPrediction, ...]


def canonical_router_id(router: str) -> str:
    normalized = router.upper()
    if normalized in ROUTER_SHORT_IDS:
        return ROUTER_SHORT_IDS[normalized]
    if normalized in ROUTER_EXPERTS:
        return normalized
    raise ValueError(f"Unsupported router: {router}")


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _promotion_map(predictions: Sequence[ExpertPrediction]) -> dict[str, bool]:
    grouped: dict[str, list[ExpertPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.expert_id].append(prediction)
    return {
        expert_id: any(
            row.prediction_status == "active"
            and row.prediction_tmax_c is not None
            and row.router_weight_cap > 0.0
            and row.expert_scope == "strict"
            for row in rows
        )
        for expert_id, rows in grouped.items()
    }


def _strict_predictions_by_date(
    predictions: Sequence[ExpertPrediction],
) -> dict[date, dict[str, ExpertPrediction]]:
    grouped: dict[date, dict[str, ExpertPrediction]] = defaultdict(dict)
    for prediction in predictions:
        if prediction.expert_scope != "strict":
            continue
        if (
            prediction.train_end_date is not None
            and prediction.test_start_date is not None
            and prediction.train_end_date >= prediction.test_start_date
        ):
            raise ValueError("Router refuses non-OOF expert prediction chronology")
        if prediction.expert_id.startswith(("E6_", "E7_", "E8_", "E9_", "E10_", "E11_")):
            raise ValueError("Strict router refuses proxy/shadow expert predictions")
        grouped[prediction.target_date_hkt][prediction.expert_id] = prediction
    return dict(grouped)


def _expert_values_for_date(
    target_date_hkt: date,
    expert_ids: Sequence[str],
    grouped: Mapping[date, Mapping[str, ExpertPrediction]],
) -> tuple[dict[str, float], dict[str, bool], str]:
    values: dict[str, float] = {}
    availability: dict[str, bool] = {}
    fold_id = "unknown"
    for expert_id in expert_ids:
        prediction = grouped.get(target_date_hkt, {}).get(expert_id)
        available = (
            prediction is not None
            and prediction.prediction_status == "active"
            and prediction.prediction_tmax_c is not None
        )
        availability[expert_id] = available
        if available and prediction is not None and prediction.prediction_tmax_c is not None:
            values[expert_id] = float(prediction.prediction_tmax_c)
            fold_id = prediction.fold_id
    return values, availability, fold_id


def _common_prediction_matrix(
    *,
    rows: Sequence[FeatureMatrixRow],
    grouped_predictions: Mapping[date, Mapping[str, ExpertPrediction]],
    expert_ids: Sequence[str],
    promoted: Mapping[str, bool],
) -> tuple[list[dict[str, float]], list[float]]:
    matrix: list[dict[str, float]] = []
    labels: list[float] = []
    for row in rows:
        if row.target_tmax_c is None:
            continue
        values, availability, _ = _expert_values_for_date(row.target_date_hkt, expert_ids, grouped_predictions)
        if all(availability.get(expert_id, False) and promoted.get(expert_id, False) for expert_id in expert_ids):
            matrix.append(values)
            labels.append(float(row.target_tmax_c))
    return matrix, labels


def _date_metrics(
    predictions: Mapping[date, float],
    labels_by_date: Mapping[date, float],
    dates: Sequence[date],
) -> ForecastMetricSummary | None:
    pairs = [
        (predictions[target_date], labels_by_date[target_date])
        for target_date in dates
        if target_date in predictions and target_date in labels_by_date
    ]
    if not pairs:
        return None
    return forecast_metrics(pairs)


def _baseline_predictions(
    expert_id: str,
    grouped_predictions: Mapping[date, Mapping[str, ExpertPrediction]],
) -> dict[date, float]:
    output: dict[date, float] = {}
    for target_date_hkt, predictions in grouped_predictions.items():
        prediction = predictions.get(expert_id)
        if prediction is not None and prediction.prediction_status == "active" and prediction.prediction_tmax_c is not None:
            output[target_date_hkt] = float(prediction.prediction_tmax_c)
    return output


def _prediction_map(router_predictions: Sequence[RouterPrediction]) -> dict[date, float]:
    return {
        prediction.target_date_hkt: float(prediction.base_forecast_c)
        for prediction in router_predictions
        if prediction.base_forecast_c is not None
    }


def _select_static_result(
    *,
    expert_ids: Sequence[str],
    grouped_predictions: Mapping[date, Mapping[str, ExpertPrediction]],
    rows: Sequence[FeatureMatrixRow],
    promoted: Mapping[str, bool],
) -> tuple[dict[str, float], dict[str, object], str]:
    matrix, labels = _common_prediction_matrix(
        rows=rows,
        grouped_predictions=grouped_predictions,
        expert_ids=expert_ids,
        promoted=promoted,
    )
    caps = {expert_id: EXPERT_STRICT_WEIGHT_CAPS.get(expert_id, 0.0) for expert_id in expert_ids}
    result = optimize_static_weights(
        expert_ids=expert_ids,
        prediction_matrix=matrix,
        labels=labels,
        caps=caps,
    )
    return result.weights, result.cap_trace, result.status


def _expected_errors_for_row(
    *,
    row: FeatureMatrixRow,
    expert_ids: Sequence[str],
    predictions: Sequence[ExpertPrediction],
    grouped_predictions: Mapping[date, Mapping[str, ExpertPrediction]],
    feature_rows_by_date: Mapping[date, FeatureMatrixRow],
    labels_by_date: Mapping[date, float],
    feature_names: Sequence[str],
    model_cache: dict[tuple[str, date | None], ExpertExpectedErrorModel],
) -> dict[str, float]:
    result: dict[str, float] = {}
    prediction_by_expert = grouped_predictions.get(row.target_date_hkt, {})
    for expert_id in expert_ids:
        prediction = prediction_by_expert.get(expert_id)
        cutoff_before = prediction.test_start_date if prediction is not None else row.target_date_hkt
        cache_key = (expert_id, cutoff_before)
        if cache_key not in model_cache:
            model_cache[cache_key] = fit_expected_error_model(
                expert_id=expert_id,
                feature_rows_by_date=feature_rows_by_date,
                predictions=predictions,
                labels_by_date=labels_by_date,
                feature_names=feature_names,
                cutoff_before=cutoff_before,
            )
        model = model_cache[cache_key]
        result[expert_id] = model.predict(context_vector(row, feature_names))
    return result


def _router_required_promotion(router_id: str, promoted: Mapping[str, bool]) -> tuple[bool, str | None]:
    if router_id == "R0_OFFICIAL_LONG_HISTORY":
        if not promoted.get("E0_OFFICIAL_RAW_ANCHOR", False):
            return False, "STRICT_E0_OFFICIAL_ANCHOR_UNAVAILABLE"
        if not any(promoted.get(expert_id, False) for expert_id in ("E1_OFFICIAL_RESIDUAL", "E2_TARGET_MEMORY")):
            return False, "FEWER_THAN_TWO_PROMOTED_R0_EXPERTS"
        return True, None
    if router_id == "R1_CORE_GFS_GEFS":
        promoted_count = sum(1 for expert_id in ROUTER_EXPERTS[router_id] if promoted.get(expert_id, False))
        if promoted_count < 2:
            return False, "FEWER_THAN_TWO_PROMOTED_R1_EXPERTS"
        return True, None
    return False, "STRICT_PRE2024_SHADOW_ADAPTER_ZERO_IMPACT"


def train_router_from_inputs(
    *,
    rows: Sequence[FeatureMatrixRow],
    predictions: Sequence[ExpertPrediction],
    router_id: str,
    r0_baseline_predictions: Sequence[RouterPrediction] = (),
) -> RouterTrainingResult:
    """Train/replay one strict pre-2024 router from OOF expert predictions."""
    router_id = canonical_router_id(router_id)
    expert_ids = ROUTER_EXPERTS[router_id]
    grouped_predictions = _strict_predictions_by_date(predictions)
    labels_by_date = labels_from_rows(rows)
    feature_rows_by_date = {row.target_date_hkt: row for row in rows}
    feature_names = context_feature_names(rows)
    promoted = _promotion_map(predictions)
    meets_expert_gate, expert_gate_reason = _router_required_promotion(router_id, promoted)
    static_weights, static_cap_trace, static_status = _select_static_result(
        expert_ids=expert_ids,
        grouped_predictions=grouped_predictions,
        rows=rows,
        promoted=promoted,
    )

    training_matrix, training_labels = _common_prediction_matrix(
        rows=rows,
        grouped_predictions=grouped_predictions,
        expert_ids=expert_ids,
        promoted=promoted,
    )
    expected_training_rows = [
        {expert_id: 0.80 for expert_id in expert_ids}
        for _ in training_matrix
    ]
    selection = select_tau_lambda(
        labels=training_labels,
        predictions_by_row=training_matrix,
        expected_error_by_row=expected_training_rows,
        static_weights=static_weights,
        expert_ids=expert_ids,
    )

    router_predictions: list[RouterPrediction] = []
    expected_error_model_cache: dict[tuple[str, date | None], ExpertExpectedErrorModel] = {}
    for row in rows:
        values, availability, fold_id = _expert_values_for_date(
            row.target_date_hkt,
            expert_ids,
            grouped_predictions,
        )
        expected_errors = _expected_errors_for_row(
            row=row,
            expert_ids=expert_ids,
            predictions=predictions,
            grouped_predictions=grouped_predictions,
            feature_rows_by_date=feature_rows_by_date,
            labels_by_date=labels_by_date,
            feature_names=feature_names,
            model_cache=expected_error_model_cache,
        )
        dynamic_weights = dynamic_weights_from_expected_errors(
            expected_errors,
            expert_ids=expert_ids,
            tau=selection.tau,
        )
        blended = blend_static_dynamic_weights(
            static_weights,
            dynamic_weights,
            expert_ids=expert_ids,
            lambda_=selection.lambda_,
        )
        capped = apply_caps_and_masks(
            blended,
            expert_ids=expert_ids,
            caps={expert_id: EXPERT_STRICT_WEIGHT_CAPS.get(expert_id, 0.0) for expert_id in expert_ids},
            availability=availability,
            promoted=promoted,
        )
        if capped.status == "no_available_expert":
            base_forecast = None
        else:
            base_forecast = sum(capped.weights[expert_id] * values.get(expert_id, 0.0) for expert_id in expert_ids)
        router_predictions.append(
            RouterPrediction(
                target_date_hkt=row.target_date_hkt,
                cutoff_id=row.cutoff_id,
                snapshot_id=row.snapshot_id,
                router_id=router_id,
                router_version=ROUTER_VERSION,
                router_scope="strict-pre2024",
                fold_id=fold_id,
                base_forecast_c=base_forecast,
                static_weights=dict(static_weights),
                dynamic_weights=dynamic_weights,
                final_weights=dict(capped.weights),
                expected_error_c_by_expert=expected_errors,
                availability_mask=availability,
                selected_tau=selection.tau,
                selected_lambda=selection.lambda_,
                promotion_status="candidate",
                demotion_reason=None,
                expert_mask={expert_id: capped.weights.get(expert_id, 0.0) > 0.0 for expert_id in expert_ids},
                cap_trace={
                    "static": static_cap_trace,
                    "row": capped.cap_trace,
                    "static_status": static_status,
                    "expert_values": values,
                },
                leakage_status="passed",
            )
        )

    router_forecasts = _prediction_map(router_predictions)
    dates = sorted(router_forecasts)
    metrics = _date_metrics(router_forecasts, labels_by_date, dates)
    e0_forecasts = _baseline_predictions("E0_OFFICIAL_RAW_ANCHOR", grouped_predictions)
    e0_metrics = _date_metrics(e0_forecasts, labels_by_date, dates)
    r0_forecasts = _prediction_map(r0_baseline_predictions)
    r0_metrics = _date_metrics(r0_forecasts, labels_by_date, dates)

    status = "promoted"
    demotion_reason = expert_gate_reason
    baseline_metrics = e0_metrics
    mae_delta_vs_baseline: float | None = None
    mae_delta_vs_r0: float | None = None
    p90_abs_error_delta: float | None = None
    if metrics is None:
        status = "demoted"
        demotion_reason = "NO_ROUTER_FORECAST_ROWS"
    elif not meets_expert_gate:
        status = "demoted"
    elif router_id == "R1_CORE_GFS_GEFS":
        reference_metrics = [metric for metric in (e0_metrics, r0_metrics) if metric is not None]
        if not reference_metrics:
            status = "demoted"
            demotion_reason = "NO_R1_BASELINE_IDENTICAL_ROWS"
        else:
            baseline_metrics = min(reference_metrics, key=lambda item: item.mae_c)
            mae_delta_vs_baseline = metrics.mae_c - baseline_metrics.mae_c
            mae_delta_vs_r0 = None if r0_metrics is None else metrics.mae_c - r0_metrics.mae_c
            p90_abs_error_delta = metrics.p90_abs_error_c - baseline_metrics.p90_abs_error_c
            bias_gate = abs(metrics.bias_c) <= abs(baseline_metrics.bias_c) + 0.03
            if (
                baseline_metrics.mae_c - metrics.mae_c < 0.01
                or metrics.p90_abs_error_c - baseline_metrics.p90_abs_error_c > 0.02
                or not bias_gate
                or (r0_metrics is not None and metrics.mae_c >= r0_metrics.mae_c)
            ):
                status = "demoted"
                demotion_reason = "R1_PROMOTION_GATES_FAILED"
    elif e0_metrics is not None:
        baseline_metrics = e0_metrics
        mae_delta_vs_baseline = metrics.mae_c - e0_metrics.mae_c
        p90_abs_error_delta = metrics.p90_abs_error_c - e0_metrics.p90_abs_error_c

    finalized = tuple(
        RouterPrediction(
            target_date_hkt=prediction.target_date_hkt,
            cutoff_id=prediction.cutoff_id,
            snapshot_id=prediction.snapshot_id,
            router_id=prediction.router_id,
            router_version=prediction.router_version,
            router_scope=prediction.router_scope,
            fold_id=prediction.fold_id,
            base_forecast_c=prediction.base_forecast_c,
            static_weights=prediction.static_weights,
            dynamic_weights=prediction.dynamic_weights,
            final_weights=prediction.final_weights if status == "promoted" else {key: 0.0 for key in prediction.final_weights},
            expected_error_c_by_expert=prediction.expected_error_c_by_expert,
            availability_mask=prediction.availability_mask,
            selected_tau=prediction.selected_tau,
            selected_lambda=prediction.selected_lambda,
            promotion_status=status,
            demotion_reason=demotion_reason,
            expert_mask=prediction.expert_mask,
            cap_trace=prediction.cap_trace,
            leakage_status=prediction.leakage_status,
        )
        for prediction in router_predictions
    )
    return RouterTrainingResult(
        router_id=router_id,
        router_scope="strict-pre2024",
        promotion_status=status,
        demotion_reason=demotion_reason,
        selected_tau=selection.tau,
        selected_lambda=selection.lambda_,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        mae_delta_vs_baseline=mae_delta_vs_baseline,
        mae_delta_vs_r0=mae_delta_vs_r0,
        p90_abs_error_delta=p90_abs_error_delta,
        predictions=finalized,
    )


def train_router_suite_from_inputs(
    rows: Sequence[FeatureMatrixRow],
    predictions: Sequence[ExpertPrediction],
) -> tuple[RouterTrainingResult, RouterTrainingResult]:
    r0 = train_router_from_inputs(rows=rows, predictions=predictions, router_id="R0")
    r1 = train_router_from_inputs(
        rows=rows,
        predictions=predictions,
        router_id="R1",
        r0_baseline_predictions=r0.predictions,
    )
    return r0, r1


def write_router_reports(writer: ReportWriter, results: Sequence[RouterTrainingResult]) -> None:
    writer.write_csv(
        "router_scoreboard_strict.csv",
        (
            "router_version",
            "router_scope",
            "router_id",
            "promotion_status",
            "demotion_reason",
            "mae",
            "baseline_mae",
            "mae_delta_vs_baseline",
            "mae_delta_vs_r0",
            "p90_abs_error_delta",
            "row_count",
            "selected_tau",
            "selected_lambda",
        ),
        [
            (
                ROUTER_VERSION,
                result.router_scope,
                result.router_id,
                result.promotion_status,
                result.demotion_reason,
                None if result.metrics is None else result.metrics.mae_c,
                None if result.baseline_metrics is None else result.baseline_metrics.mae_c,
                result.mae_delta_vs_baseline,
                result.mae_delta_vs_r0,
                result.p90_abs_error_delta,
                0 if result.metrics is None else result.metrics.row_count,
                result.selected_tau,
                result.selected_lambda,
            )
            for result in results
        ],
    )
    writer.write_csv(
        "router_promotion_decisions.csv",
        (
            "router_id",
            "candidate_id",
            "evaluation_scope",
            "identical_row_n",
            "mae_candidate",
            "mae_baseline_e0",
            "mae_baseline_r0",
            "delta_vs_e0",
            "delta_vs_r0",
            "rmse_candidate",
            "bias_candidate",
            "p90_abs_error_candidate",
            "promotion_status",
            "demotion_reason",
            "created_at_utc",
        ),
        [
            (
                result.router_id,
                ROUTER_VERSION,
                result.router_scope,
                0 if result.metrics is None else result.metrics.row_count,
                None if result.metrics is None else result.metrics.mae_c,
                None if result.baseline_metrics is None else result.baseline_metrics.mae_c,
                None,
                result.mae_delta_vs_baseline,
                result.mae_delta_vs_r0,
                None if result.metrics is None else result.metrics.rmse_c,
                None if result.metrics is None else result.metrics.bias_c,
                None if result.metrics is None else result.metrics.p90_abs_error_c,
                result.promotion_status,
                result.demotion_reason,
                "generated_by_cli",
            )
            for result in results
        ],
    )
    writer.write_csv(
        "router_weight_diagnostics.csv",
        (
            "target_date_hkt",
            "router_id",
            "base_forecast_c",
            "promotion_status",
            "weights_json",
            "availability_json",
            "expected_error_json",
        ),
        [
            (
                prediction.target_date_hkt,
                prediction.router_id,
                prediction.base_forecast_c,
                prediction.promotion_status,
                json.dumps(prediction.final_weights, sort_keys=True),
                json.dumps(prediction.availability_mask, sort_keys=True),
                json.dumps(prediction.expected_error_c_by_expert, sort_keys=True),
            )
            for result in results
            for prediction in result.predictions
        ],
    )
    writer.write_root_report(
        "router_report.md",
        "HKG-T24-003 Router Report",
        (
            ("Status", "PASS"),
            (
                "Routers",
                "\n".join(
                    f"- `{result.router_id}`: {result.promotion_status}, rows="
                    f"{0 if result.metrics is None else result.metrics.row_count}, "
                    f"mae={None if result.metrics is None else round(result.metrics.mae_c, 6)}, "
                    f"reason={result.demotion_reason or 'none'}"
                    for result in results
                ),
            ),
        ),
    )


def persist_router_results(connection: Any, results: Sequence[RouterTrainingResult]) -> int:
    count = 0
    with connection.cursor() as cursor:
        for result in results:
            metrics = result.metrics
            baseline = result.baseline_metrics
            cursor.execute(
                """
                INSERT INTO model_router.router_scoreboard (
                  router_version, router_scope, promotion_status, promotion_gate_passed,
                  demotion_reason, included_experts_jsonb, excluded_experts_jsonb,
                  mae, baseline_mae, mae_delta_vs_baseline, mae_delta_vs_r0,
                  p90_abs_error_delta, row_count, first_date, last_date
                )
                VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (router_version, router_scope) DO UPDATE SET
                  promotion_status = EXCLUDED.promotion_status,
                  promotion_gate_passed = EXCLUDED.promotion_gate_passed,
                  demotion_reason = EXCLUDED.demotion_reason,
                  included_experts_jsonb = EXCLUDED.included_experts_jsonb,
                  excluded_experts_jsonb = EXCLUDED.excluded_experts_jsonb,
                  mae = EXCLUDED.mae,
                  baseline_mae = EXCLUDED.baseline_mae,
                  mae_delta_vs_baseline = EXCLUDED.mae_delta_vs_baseline,
                  mae_delta_vs_r0 = EXCLUDED.mae_delta_vs_r0,
                  p90_abs_error_delta = EXCLUDED.p90_abs_error_delta,
                  row_count = EXCLUDED.row_count,
                  first_date = EXCLUDED.first_date,
                  last_date = EXCLUDED.last_date,
                  created_at_utc = now()
                """,
                (
                    result.router_id,
                    result.router_scope,
                    result.promotion_status,
                    result.promotion_status == "promoted",
                    result.demotion_reason,
                    json.dumps(ROUTER_EXPERTS[result.router_id], sort_keys=True),
                    json.dumps([], sort_keys=True),
                    None if metrics is None else metrics.mae_c,
                    None if baseline is None else baseline.mae_c,
                    result.mae_delta_vs_baseline,
                    result.mae_delta_vs_r0,
                    result.p90_abs_error_delta,
                    0 if metrics is None else metrics.row_count,
                    None if not result.predictions else result.predictions[0].target_date_hkt,
                    None if not result.predictions else result.predictions[-1].target_date_hkt,
                ),
            )
            for prediction in result.predictions:
                cursor.execute(
                    """
                    INSERT INTO model_router.router_prediction (
                      target_date_hkt, cutoff_id, snapshot_id, router_version, router_scope,
                      fold_id, base_forecast_c, static_weights_jsonb, dynamic_weights_jsonb,
                      final_weights_jsonb, expected_error_jsonb, availability_mask_jsonb,
                      selected_tau, selected_lambda, promotion_status, demotion_reason,
                      expert_mask_jsonb, cap_trace_jsonb, leakage_status
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
                    ON CONFLICT (target_date_hkt, cutoff_id, router_version, router_scope) DO UPDATE SET
                      snapshot_id = EXCLUDED.snapshot_id,
                      fold_id = EXCLUDED.fold_id,
                      base_forecast_c = EXCLUDED.base_forecast_c,
                      static_weights_jsonb = EXCLUDED.static_weights_jsonb,
                      dynamic_weights_jsonb = EXCLUDED.dynamic_weights_jsonb,
                      final_weights_jsonb = EXCLUDED.final_weights_jsonb,
                      expected_error_jsonb = EXCLUDED.expected_error_jsonb,
                      availability_mask_jsonb = EXCLUDED.availability_mask_jsonb,
                      selected_tau = EXCLUDED.selected_tau,
                      selected_lambda = EXCLUDED.selected_lambda,
                      promotion_status = EXCLUDED.promotion_status,
                      demotion_reason = EXCLUDED.demotion_reason,
                      expert_mask_jsonb = EXCLUDED.expert_mask_jsonb,
                      cap_trace_jsonb = EXCLUDED.cap_trace_jsonb,
                      leakage_status = EXCLUDED.leakage_status,
                      created_at_utc = now()
                    """,
                    (
                        prediction.target_date_hkt,
                        prediction.cutoff_id,
                        prediction.snapshot_id,
                        prediction.router_id,
                        prediction.router_scope,
                        prediction.fold_id,
                        prediction.base_forecast_c,
                        json.dumps(prediction.static_weights, sort_keys=True),
                        json.dumps(prediction.dynamic_weights, sort_keys=True),
                        json.dumps(prediction.final_weights, sort_keys=True),
                        json.dumps(prediction.expected_error_c_by_expert, sort_keys=True),
                        json.dumps(prediction.availability_mask, sort_keys=True),
                        prediction.selected_tau,
                        prediction.selected_lambda,
                        prediction.promotion_status,
                        prediction.demotion_reason,
                        json.dumps(prediction.expert_mask, sort_keys=True),
                        json.dumps(prediction.cap_trace, sort_keys=True),
                        prediction.leakage_status,
                    ),
                )
                count += 1
    return count


def load_expert_predictions(
    connection: Any,
    *,
    start_date: date,
    end_date: date,
) -> list[ExpertPrediction]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT target_date_hkt, cutoff_id, snapshot_id, expert_id, expert_scope, fold_id,
                   prediction_tmax_c, prediction_residual_c, raw_anchor_tmax_c,
                   prediction_status, placeholder_reason, train_end_date, test_start_date,
                   router_weight_cap, feature_schema_version
            FROM model_oof.expert_prediction
            WHERE cutoff_id = 'H24N'
              AND target_date_hkt BETWEEN %s AND %s
            ORDER BY target_date_hkt, expert_id
            """,
            (start_date, end_date),
        )
        rows = cursor.fetchall()
    return [
        ExpertPrediction(
            target_date_hkt=row[0],
            cutoff_id=str(row[1]),
            snapshot_id=str(row[2]),
            expert_id=str(row[3]),
            expert_scope=str(row[4]),
            fold_id=str(row[5]),
            prediction_tmax_c=None if row[6] is None else float(row[6]),
            prediction_residual_c=None if row[7] is None else float(row[7]),
            raw_anchor_tmax_c=None if row[8] is None else float(row[8]),
            prediction_status=str(row[9]),
            placeholder_reason=None if row[10] is None else str(row[10]),
            train_end_date=row[11],
            test_start_date=row[12],
            router_weight_cap=float(row[13]),
            feature_schema_version=str(row[14]),
        )
        for row in rows
    ]


def synthetic_router_inputs(days: int = 180) -> tuple[list[FeatureMatrixRow], list[ExpertPrediction]]:
    start = date(2021, 1, 1)
    rows: list[FeatureMatrixRow] = []
    for offset in range(days):
        target = start + timedelta(days=offset)
        seasonal = (offset % 31) / 31.0
        target_tmax = 27.0 + 2.0 * seasonal + (0.25 if target.month in {6, 7, 8} else 0.0)
        official = target_tmax - 0.18
        gfs = target_tmax - 0.08
        gefs = target_tmax - 0.04
        features: dict[str, FeatureValue] = {
            "calendar__month_sin1": float(target.month),
            "calendar__is_mam": target.month in {3, 4, 5},
            "calendar__is_jja": target.month in {6, 7, 8},
            "official__forecast_max_c": official,
            "official__psr_numeric_proxy": 0.30,
            "target__lag2_tmax_c": target_tmax - 0.10,
            "target__roll7_mean_lag2_c": target_tmax - 0.12,
            "target__roll14_std_lag2_c": 0.45 + seasonal * 0.2,
            "target__slope7_minus_slope30_lag2_c_per_day": 0.03 - seasonal * 0.01,
            "target__lag2_minus_roll7_c": 0.10,
            "target__hot_spell_length_lag2_days": float(offset % 4),
            "gfs__center__tmax_c": gfs,
            "gfs__center__dewpoint_change_proxy_c": 0.15,
            "gfs__center__low_cloud_pct_mean": 30.0 + seasonal * 20.0,
            "gfs__center__shortwave_w_m2_mean": 520.0 - seasonal * 80.0,
            "gfs__center__precip_mm_sum": seasonal * 2.0,
            "gfs__center__wind_speed_10m_mean_mps": 3.0 + seasonal,
            "gfs__center__onshore_easterly_component_mps": 1.0 + seasonal,
            "gfs__center__temp_dewpoint_spread_mean_c": 4.0 + seasonal,
            "gfs__center__t850_c_mean": 17.0 + seasonal,
            "gfs__center__z500_m_mean": 5880.0 + seasonal * 10.0,
            "gfs__center__relative_humidity_700_pct_mean": 65.0 - seasonal * 10.0,
            "gfs__center__dewpoint_2m_c_mean": 22.0 + seasonal,
            "gfs__spatial__inland_nw_minus_center_tmax_c": 0.35,
            "gfs__spatial__inland_nw_minus_marine_s_tmax_c": 0.75,
            "gfs__spatial__center_minus_marine_s_tmax_c": 0.40,
            "gefsmean__center__tmax_c": gefs,
            "gefsmean__center__pwat_kg_m2_mean": 45.0,
            "gefsmean__center__onshore_east_component_mps_mean": 0.8,
            "gefsmean__center__wind_speed_10m_mps_mean": 3.1,
            "gefsens__center__tmax_p10_c": gefs - 0.4,
            "gefsens__center__tmax_p50_c": gefs,
            "gefsens__center__tmax_p90_c": gefs + 0.4,
            "gefsens__center__tmax_spread_p90_p10_c": 0.8,
            "online__official_raw__global__abs_error_h20_c": 0.35,
        }
        rows.append(
            FeatureMatrixRow(
                target_date_hkt=target,
                cutoff_id=CUTOFF_ID,
                snapshot_id=snapshot_id(target),
                feature_scope="strict",
                schema_version=STRICT_SCHEMA_VERSION,
                features=features,
                target_tmax_c=target_tmax,
            )
        )
    folds = folds_for_scope(start_date=rows[0].target_date_hkt, end_date=rows[-1].target_date_hkt, smoke=True)
    predictions = generate_expert_oof_predictions(rows, folds)
    prediction_dates = {prediction.target_date_hkt for prediction in predictions}
    return [row for row in rows if row.target_date_hkt in prediction_dates], predictions


def run_router_training(
    connection: Any | None,
    writer: ReportWriter,
    *,
    router: str,
    scope: str,
    start_date: date,
    end_date: date,
    smoke: bool,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if scope != "strict-pre2024":
        raise ValueError("Jira003 routers support --scope strict-pre2024")
    if smoke and connection is None:
        rows, predictions = synthetic_router_inputs()
    else:
        if connection is None:
            raise ValueError("connection is required for non-smoke router training")
        rows = load_feature_matrix_rows(
            connection,
            scope="strict",
            start_date=start_date,
            end_date=end_date,
        )
        predictions = load_expert_predictions(connection, start_date=start_date, end_date=end_date)
    if not rows:
        raise ValueError("No strict feature rows available for router training")
    if not predictions:
        raise ValueError("No OOF expert predictions available for router training")
    requested = canonical_router_id(router)
    results: tuple[RouterTrainingResult, ...]
    if requested == "R0_OFFICIAL_LONG_HISTORY":
        results = (train_router_from_inputs(rows=rows, predictions=predictions, router_id="R0"),)
    elif requested == "R1_CORE_GFS_GEFS":
        r0, r1 = train_router_suite_from_inputs(rows, predictions)
        results = (r0, r1)
    else:
        result = train_router_from_inputs(rows=rows, predictions=predictions, router_id=requested)
        results = (result,)
    write_router_reports(writer, results)
    if connection is not None:
        persist_router_results(connection, results)
    return "PASS", (), (f"router_predictions={sum(len(result.predictions) for result in results)}",)
