"""Final strict system formula for Jira003 point forecasts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date

from hkg_t24.features.matrix_builder import FeatureMatrixRow
from hkg_t24.models.experts import ExpertPrediction
from hkg_t24.models.router import RouterPrediction, RouterTrainingResult
from hkg_t24.models.specialists import SpecialistPrediction, SpecialistTrainingResult
from hkg_t24.validation.metrics import clip

SYSTEM_VERSION = "system_v1_20260627"


@dataclass(frozen=True)
class SystemPrediction:
    target_date_hkt: date
    cutoff_id: str
    snapshot_id: str
    system_version: str
    router_selected: str | None
    router_selection_reason: str
    base_forecast_c: float | None
    specialist_total_correction_c: float
    final_pre_distribution_c: float | None
    final_point_tmax_c: float | None
    p10_c: float | None
    p25_c: float | None
    p50_c: float | None
    p75_c: float | None
    p90_c: float | None
    expected_abs_error_c: float | None
    threshold_probabilities: dict[str, float]
    confidence_state: str
    no_trade_flag: bool
    distribution_status: str
    quantile_monotonic_repair: bool
    component_jsonb: dict[str, object]
    leakage_status: str


def with_distribution(
    prediction: SystemPrediction,
    *,
    p10_c: float | None,
    p25_c: float | None,
    p50_c: float | None,
    p75_c: float | None,
    p90_c: float | None,
    expected_abs_error_c: float,
    threshold_probabilities: dict[str, float],
    confidence_state: str,
    no_trade_flag: bool,
    distribution_status: str,
    quantile_monotonic_repair: bool,
) -> SystemPrediction:
    return replace(
        prediction,
        final_point_tmax_c=None if p50_c is None else round(p50_c, 1),
        p10_c=p10_c,
        p25_c=p25_c,
        p50_c=p50_c,
        p75_c=p75_c,
        p90_c=p90_c,
        expected_abs_error_c=expected_abs_error_c,
        threshold_probabilities=threshold_probabilities,
        confidence_state=confidence_state,
        no_trade_flag=no_trade_flag,
        distribution_status=distribution_status,
        quantile_monotonic_repair=quantile_monotonic_repair,
    )


def _expert_prediction_map(predictions: Sequence[ExpertPrediction]) -> dict[date, dict[str, ExpertPrediction]]:
    grouped: dict[date, dict[str, ExpertPrediction]] = defaultdict(dict)
    for prediction in predictions:
        grouped[prediction.target_date_hkt][prediction.expert_id] = prediction
    return dict(grouped)


def _router_map(results: Sequence[RouterTrainingResult]) -> dict[str, dict[date, RouterPrediction]]:
    output: dict[str, dict[date, RouterPrediction]] = {}
    for result in results:
        output[result.router_id] = {prediction.target_date_hkt: prediction for prediction in result.predictions}
    return output


def _specialist_by_date(
    results: Sequence[SpecialistTrainingResult],
) -> dict[date, list[SpecialistPrediction]]:
    grouped: dict[date, list[SpecialistPrediction]] = defaultdict(list)
    for result in results:
        for prediction in result.predictions:
            grouped[prediction.target_date_hkt].append(prediction)
    return dict(grouped)


def _active_expert_value(
    grouped: Mapping[date, Mapping[str, ExpertPrediction]],
    target_date_hkt: date,
    expert_id: str,
) -> float | None:
    prediction = grouped.get(target_date_hkt, {}).get(expert_id)
    if prediction is None or prediction.prediction_status != "active" or prediction.prediction_tmax_c is None:
        return None
    return float(prediction.prediction_tmax_c)


def _select_router(
    target_date_hkt: date,
    routers: Mapping[str, Mapping[date, RouterPrediction]],
) -> tuple[RouterPrediction | None, str]:
    r1 = routers.get("R1_CORE_GFS_GEFS", {}).get(target_date_hkt)
    if r1 is not None and r1.promotion_status == "promoted" and r1.base_forecast_c is not None:
        return r1, "R1_promoted_available"
    r0 = routers.get("R0_OFFICIAL_LONG_HISTORY", {}).get(target_date_hkt)
    if r0 is not None and r0.promotion_status == "promoted" and r0.base_forecast_c is not None:
        return r0, "R1_unavailable_or_demoted_use_R0"
    return None, "router_unavailable_use_fallback_expert"


def assemble_pre_distribution_predictions(
    *,
    rows: Sequence[FeatureMatrixRow],
    expert_predictions: Sequence[ExpertPrediction],
    router_results: Sequence[RouterTrainingResult],
    specialist_results: Sequence[SpecialistTrainingResult],
) -> tuple[SystemPrediction, ...]:
    grouped_experts = _expert_prediction_map(expert_predictions)
    routers = _router_map(router_results)
    specialists = _specialist_by_date(specialist_results)
    system_predictions: list[SystemPrediction] = []
    for row in rows:
        router_prediction, selection_reason = _select_router(row.target_date_hkt, routers)
        fallback_expert: str | None = None
        if router_prediction is not None:
            base = router_prediction.base_forecast_c
            router_selected = router_prediction.router_id
        else:
            e0 = _active_expert_value(grouped_experts, row.target_date_hkt, "E0_OFFICIAL_RAW_ANCHOR")
            e2 = _active_expert_value(grouped_experts, row.target_date_hkt, "E2_TARGET_MEMORY")
            if e0 is not None:
                base = e0
                fallback_expert = "E0_OFFICIAL_RAW_ANCHOR"
            elif e2 is not None:
                base = e2
                fallback_expert = "E2_TARGET_MEMORY"
            else:
                base = None
            router_selected = None

        active_specialists = [
            prediction
            for prediction in specialists.get(row.target_date_hkt, [])
            if prediction.activated and prediction.promotion_status == "promoted"
        ]
        specialist_total = clip(
            sum(prediction.applied_correction_c for prediction in active_specialists),
            -0.40,
            0.40,
        )
        if base is None:
            pre_distribution = None
            point = None
            leakage_status = "failed_closed"
            no_trade = True
            confidence = "LOW"
        else:
            pre_distribution = base + specialist_total
            official = row.features.get("official__forecast_max_c")
            official_value = float(official) if isinstance(official, int | float) and not isinstance(official, bool) else None
            if official_value is not None:
                pre_distribution = clip(pre_distribution, official_value - 1.20, official_value + 1.20)
            point = round(pre_distribution, 1)
            leakage_status = "passed"
            no_trade = False
            confidence = "MEDIUM"
        system_predictions.append(
            SystemPrediction(
                target_date_hkt=row.target_date_hkt,
                cutoff_id=row.cutoff_id,
                snapshot_id=row.snapshot_id,
                system_version=SYSTEM_VERSION,
                router_selected=router_selected,
                router_selection_reason=selection_reason,
                base_forecast_c=base,
                specialist_total_correction_c=specialist_total,
                final_pre_distribution_c=pre_distribution,
                final_point_tmax_c=point,
                p10_c=None,
                p25_c=None,
                p50_c=None,
                p75_c=None,
                p90_c=None,
                expected_abs_error_c=None,
                threshold_probabilities={},
                confidence_state=confidence,
                no_trade_flag=no_trade,
                distribution_status="not_trained",
                quantile_monotonic_repair=False,
                component_jsonb={
                    "router_selected": router_selected,
                    "router_selection_reason": selection_reason,
                    "fallback_expert": fallback_expert,
                    "active_specialists": [prediction.specialist_id for prediction in active_specialists],
                    "specialist_total_uncapped_c": sum(
                        prediction.applied_correction_c for prediction in active_specialists
                    ),
                    "official_clip_applied": base is not None
                    and row.features.get("official__forecast_max_c") is not None
                    and pre_distribution is not None
                    and abs((base + specialist_total) - pre_distribution) > 1e-9,
                },
                leakage_status=leakage_status,
            )
        )
    return tuple(system_predictions)
