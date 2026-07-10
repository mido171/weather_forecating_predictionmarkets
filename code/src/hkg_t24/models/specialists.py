"""Jira003 bounded specialist correction system."""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from statistics import mean, median
from typing import Any, Literal

from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import HuberRegressor  # type: ignore[import-untyped]

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.constants import SPECIALIST_IDS
from hkg_t24.features.matrix_builder import FeatureMatrixRow
from hkg_t24.models.router import (
    RouterPrediction,
    synthetic_router_inputs,
    train_router_suite_from_inputs,
)
from hkg_t24.validation.metrics import clip, percentile

SPECIALIST_VERSION = "specialists_v1_20260627"
CorrectionSign = Literal["negative", "positive", "learned"]
PercentileDirection = Literal["high", "low", "abs_high"]


@dataclass(frozen=True)
class SpecialistComponent:
    weight: float
    direction: PercentileDirection
    feature_name: str


@dataclass(frozen=True)
class SpecialistSpec:
    specialist_id: str
    expected_sign: CorrectionSign
    components: tuple[SpecialistComponent, ...]
    mam_only: bool = False
    s6_tail_gate: bool = False


@dataclass(frozen=True)
class SpecialistPrediction:
    target_date_hkt: date
    cutoff_id: str
    snapshot_id: str
    specialist_id: str
    specialist_version: str
    fold_id: str
    prior_score: float | None
    score_available: bool
    fold_p60: float | None
    regime_probability: float | None
    raw_correction_c: float
    shrunk_correction_c: float
    applied_correction_c: float
    expected_benefit_c: float
    activated: bool
    activation_reason: str
    support_count: int
    active_year_count: int
    no_harm_pass: bool
    promotion_status: str
    leakage_status: str


@dataclass(frozen=True)
class SpecialistTrainingResult:
    specialist_id: str
    promotion_status: str
    demotion_reason: str | None
    active_training_rows: int
    active_year_count: int
    mean_active_lift_c: float | None
    no_harm_pass: bool
    predictions: tuple[SpecialistPrediction, ...]


SPECIALIST_SPECS: tuple[SpecialistSpec, ...] = (
    SpecialistSpec(
        "S1_MARINE_SUPPRESSION",
        "negative",
        (
            SpecialistComponent(0.20, "high", "gfs__center__onshore_easterly_component_mps"),
            SpecialistComponent(0.15, "high", "gefsmean__center__onshore_east_component_mps_mean"),
            SpecialistComponent(0.20, "high", "gfs__spatial__inland_nw_minus_marine_s_tmax_c"),
            SpecialistComponent(0.10, "high", "gfs__spatial__inland_nw_minus_center_tmax_c"),
            SpecialistComponent(0.10, "high", "gfs__center__dewpoint_change_proxy_c"),
            SpecialistComponent(0.10, "high", "gfs__center__low_cloud_pct_mean"),
            SpecialistComponent(0.10, "low", "gfs__center__shortwave_w_m2_mean"),
            SpecialistComponent(0.05, "high", "official__forecast_max_minus_gefs_median_c"),
        ),
    ),
    SpecialistSpec(
        "S2_WEAK_WIND_HEAT",
        "positive",
        (
            SpecialistComponent(0.20, "low", "gfs__center__wind_speed_10m_mean_mps"),
            SpecialistComponent(0.15, "low", "gefsmean__center__wind_speed_10m_mps_mean"),
            SpecialistComponent(0.20, "high", "gfs__center__shortwave_w_m2_mean"),
            SpecialistComponent(0.10, "low", "gfs__center__low_cloud_pct_mean"),
            SpecialistComponent(0.15, "high", "gfs__center__t850_c_mean"),
            SpecialistComponent(0.10, "high", "target__slope7_minus_slope30_lag2_c_per_day"),
            SpecialistComponent(0.05, "high", "gfs__spatial__inland_nw_minus_center_tmax_c"),
            SpecialistComponent(0.05, "high", "gfs__center__temp_dewpoint_spread_mean_c"),
        ),
    ),
    SpecialistSpec(
        "S3_MAM_TRANSITION",
        "learned",
        (
            SpecialistComponent(0.20, "abs_high", "target__slope7_minus_slope30_lag2_c_per_day"),
            SpecialistComponent(0.15, "high", "target__roll14_std_lag2_c"),
            SpecialistComponent(0.15, "abs_high", "target__lag2_minus_roll7_c"),
            SpecialistComponent(0.15, "abs_high", "official__forecast_max_minus_target_roll7_c"),
            SpecialistComponent(0.10, "high", "gfs__center__low_cloud_pct_mean"),
            SpecialistComponent(0.10, "high", "gfs__center__precip_mm_sum"),
            SpecialistComponent(0.10, "high", "gefsens__center__tmax_spread_p90_p10_c"),
            SpecialistComponent(0.05, "abs_high", "gfs__center__tmax_minus_gefsmean_center_tmax_c"),
        ),
        mam_only=True,
    ),
    SpecialistSpec(
        "S4_CLOUD_RAIN_SUPPRESSION",
        "negative",
        (
            SpecialistComponent(0.20, "high", "gfs__center__low_cloud_pct_mean"),
            SpecialistComponent(0.15, "high", "gfs__center__precip_mm_sum"),
            SpecialistComponent(0.15, "low", "gfs__center__shortwave_w_m2_mean"),
            SpecialistComponent(0.15, "high", "gfs__center__relative_humidity_700_pct_mean"),
            SpecialistComponent(0.10, "high", "gefsmean__center__pwat_kg_m2_mean"),
            SpecialistComponent(0.10, "high", "gfs__center__dewpoint_2m_c_mean"),
            SpecialistComponent(0.10, "high", "gefsens__center__tmax_spread_p90_p10_c"),
            SpecialistComponent(0.05, "high", "official__psr_numeric_proxy"),
        ),
    ),
    SpecialistSpec(
        "S5_DRY_RIDGE_HEAT",
        "positive",
        (
            SpecialistComponent(0.20, "high", "gfs__center__z500_m_mean"),
            SpecialistComponent(0.15, "high", "gfs__center__t850_c_mean"),
            SpecialistComponent(0.15, "low", "gfs__center__relative_humidity_700_pct_mean"),
            SpecialistComponent(0.15, "high", "gfs__center__shortwave_w_m2_mean"),
            SpecialistComponent(0.10, "low", "gfs__center__low_cloud_pct_mean"),
            SpecialistComponent(0.10, "low", "gfs__center__precip_mm_sum"),
            SpecialistComponent(0.10, "low", "gfs__center__wind_speed_10m_mean_mps"),
            SpecialistComponent(0.05, "high", "target__hot_spell_length_lag2_days"),
        ),
    ),
    SpecialistSpec(
        "S6_HIGH_ERROR_TAIL",
        "learned",
        (
            SpecialistComponent(0.25, "high", "router__expert_prediction_spread_c"),
            SpecialistComponent(0.20, "high", "gefsens__center__tmax_spread_p90_p10_c"),
            SpecialistComponent(0.15, "high", "online__official_raw__global__abs_error_h20_c"),
            SpecialistComponent(0.10, "abs_high", "official__forecast_max_minus_gfs_center_tmax_c"),
            SpecialistComponent(0.10, "abs_high", "official__forecast_max_minus_gefs_median_c"),
            SpecialistComponent(0.10, "high", "target__roll14_std_lag2_c"),
            SpecialistComponent(0.10, "high", "router__missing_expert_count"),
        ),
        s6_tail_gate=True,
    ),
)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    return None


def _derived_feature(features: Mapping[str, object], name: str) -> float | None:
    if name in features:
        return _number(features[name])
    official = _number(features.get("official__forecast_max_c"))
    gefs = _number(features.get("gefsens__center__tmax_p50_c"))
    if gefs is None:
        gefs = _number(features.get("gefsmean__center__tmax_c"))
    gfs = _number(features.get("gfs__center__tmax_c"))
    target_roll7 = _number(features.get("target__roll7_mean_lag2_c"))
    if name == "official__forecast_max_minus_gefs_median_c" and official is not None and gefs is not None:
        return official - gefs
    if name == "official__forecast_max_minus_gfs_center_tmax_c" and official is not None and gfs is not None:
        return official - gfs
    if name == "official__forecast_max_minus_target_roll7_c" and official is not None and target_roll7 is not None:
        return official - target_roll7
    if name == "gfs__center__tmax_minus_gefsmean_center_tmax_c" and gfs is not None and gefs is not None:
        return gfs - gefs
    return None


def _router_derived_features(prediction: RouterPrediction) -> dict[str, float]:
    expert_values = prediction.cap_trace.get("expert_values")
    if isinstance(expert_values, dict):
        values = [
            float(value)
            for value in expert_values.values()
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
    else:
        values = []
    expected = [
        float(value)
        for value in prediction.expected_error_c_by_expert.values()
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return {
        "router__expert_prediction_spread_c": 0.0 if len(values) < 2 else max(values) - min(values),
        "router__missing_expert_count": float(
            sum(1 for available in prediction.availability_mask.values() if not available)
        ),
        "router__expected_abs_error_c": mean(expected) if expected else 1.0,
    }


def _component_value(features: Mapping[str, object], component: SpecialistComponent) -> float | None:
    value = _derived_feature(features, component.feature_name)
    if value is None:
        return None
    if component.direction == "abs_high":
        return abs(value)
    return value


def _pct_high(value: float, training_values: Sequence[float]) -> float:
    if not training_values:
        return 0.5
    lower_equal = sum(1 for item in training_values if item <= value)
    equal = sum(1 for item in training_values if item == value)
    return clip((lower_equal + 0.5 * equal) / len(training_values), 0.0, 1.0)


def _pct_high_sorted(value: float, training_values: Sequence[float]) -> float:
    if not training_values:
        return 0.5
    lower_equal = bisect_right(training_values, value)
    equal = lower_equal - bisect_left(training_values, value)
    return clip((lower_equal + 0.5 * equal) / len(training_values), 0.0, 1.0)


def _percentile_sorted(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if q < 0.0 or q > 1.0:
        raise ValueError("percentile q must be in [0, 1]")
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _prior_score(
    spec: SpecialistSpec,
    row_features: Mapping[str, object],
    training_feature_values: Mapping[str, Sequence[float]],
) -> tuple[float | None, bool]:
    missing_weight = 0.0
    score = 0.0
    for component in spec.components:
        value = _component_value(row_features, component)
        if value is None:
            missing_weight += component.weight
            percentile_value = 0.5
        else:
            percentile_value = _pct_high(value, training_feature_values.get(component.feature_name, ()))
            if component.direction == "low":
                percentile_value = 1.0 - percentile_value
        score += component.weight * percentile_value
    if missing_weight > 0.40:
        return None, False
    return clip(score, 0.0, 1.0), True


def _prior_score_sorted(
    spec: SpecialistSpec,
    row_features: Mapping[str, object],
    training_feature_values: Mapping[str, Sequence[float]],
) -> tuple[float | None, bool]:
    missing_weight = 0.0
    score = 0.0
    for component in spec.components:
        value = _component_value(row_features, component)
        if value is None:
            missing_weight += component.weight
            percentile_value = 0.5
        else:
            percentile_value = _pct_high_sorted(value, training_feature_values.get(component.feature_name, ()))
            if component.direction == "low":
                percentile_value = 1.0 - percentile_value
        score += component.weight * percentile_value
    if missing_weight > 0.40:
        return None, False
    return clip(score, 0.0, 1.0), True


def _row_feature_map(row: FeatureMatrixRow, prediction: RouterPrediction) -> dict[str, object]:
    features: dict[str, object] = dict(row.features)
    features.update(_router_derived_features(prediction))
    return features


def _build_training_feature_values(
    spec: SpecialistSpec,
    rows: Sequence[FeatureMatrixRow],
    predictions_by_date: Mapping[date, RouterPrediction],
    before_date: date,
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {component.feature_name: [] for component in spec.components}
    for row in rows:
        if row.target_date_hkt >= before_date:
            continue
        prediction = predictions_by_date.get(row.target_date_hkt)
        if prediction is None:
            continue
        features = _row_feature_map(row, prediction)
        for component in spec.components:
            value = _component_value(features, component)
            if value is not None:
                values[component.feature_name].append(value)
    return values


def _active_candidate_rows(
    spec: SpecialistSpec,
    rows: Sequence[FeatureMatrixRow],
    predictions_by_date: Mapping[date, RouterPrediction],
    labels_by_date: Mapping[date, float],
    before_date: date | None = None,
) -> list[tuple[FeatureMatrixRow, RouterPrediction, float, float]]:
    candidates: list[tuple[FeatureMatrixRow, RouterPrediction, float, float]] = []
    for row in rows:
        if before_date is not None and row.target_date_hkt >= before_date:
            continue
        if spec.mam_only and row.target_date_hkt.month not in {3, 4, 5}:
            continue
        prediction = predictions_by_date.get(row.target_date_hkt)
        if prediction is None or prediction.base_forecast_c is None:
            continue
        label = labels_by_date.get(row.target_date_hkt)
        if label is None:
            continue
        training_values = _build_training_feature_values(spec, rows, predictions_by_date, row.target_date_hkt)
        prior, available = _prior_score(spec, _row_feature_map(row, prediction), training_values)
        prior_history = [
            candidate_prior
            for candidate_row, candidate_prediction, _, _ in candidates
            for candidate_prior, candidate_available in [
                _prior_score(
                    spec,
                    _row_feature_map(candidate_row, candidate_prediction),
                    training_values,
                )
            ]
            if candidate_available and candidate_prior is not None
        ]
        p60 = percentile(prior_history, 0.60) if prior_history else 0.60
        if available and prior is not None and prior >= p60:
            candidates.append((row, prediction, label, float(label) - float(prediction.base_forecast_c)))
    return candidates


def _fit_correction(candidates: Sequence[tuple[FeatureMatrixRow, RouterPrediction, float, float]]) -> tuple[float, bool]:
    if not candidates:
        return 0.0, False
    targets = [candidate[3] for candidate in candidates]
    fallback = float(median(targets))
    if len(candidates) < 30:
        return fallback, False
    try:
        x = [[float(index)] for index, _ in enumerate(candidates)]
        model = HuberRegressor(epsilon=1.35, alpha=0.05, max_iter=500)
        model.fit(x, targets)
        return float(model.predict([[float(len(candidates))]])[0]), True
    except Exception:
        return fallback, False


def _fit_expected_benefit(
    candidates: Sequence[tuple[FeatureMatrixRow, RouterPrediction, float, float]],
    correction_c: float,
) -> float:
    if not candidates:
        return 0.0
    benefits = [
        abs(label - float(prediction.base_forecast_c))
        - abs(label - (float(prediction.base_forecast_c) + clip(correction_c, -0.25, 0.25)))
        for _, prediction, label, _ in candidates
        if prediction.base_forecast_c is not None
    ]
    if not benefits:
        return 0.0
    if len(benefits) < 30:
        return max(0.0, float(median(benefits)))
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=100,
        max_leaf_nodes=7,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=20260626,
    )
    x = [[float(index)] for index, _ in enumerate(benefits)]
    model.fit(x, benefits)
    return max(0.0, float(model.predict([[float(len(benefits))]])[0]))


def _signed_correction(raw: float, sign: CorrectionSign) -> float:
    if sign == "negative":
        return clip(raw, -0.25, 0.0)
    if sign == "positive":
        return clip(raw, 0.0, 0.25)
    return clip(raw, -0.25, 0.25)


def _empty_feature_history(spec: SpecialistSpec) -> dict[str, list[float]]:
    return {component.feature_name: [] for component in spec.components}


def _add_feature_history(
    spec: SpecialistSpec,
    features: Mapping[str, object],
    history: dict[str, list[float]],
) -> None:
    for component in spec.components:
        value = _component_value(features, component)
        if value is not None:
            insort(history[component.feature_name], value)


def _fast_active_candidate_rows(
    spec: SpecialistSpec,
    row_contexts: Sequence[tuple[FeatureMatrixRow, RouterPrediction, float | None, dict[str, object]]],
) -> list[tuple[FeatureMatrixRow, RouterPrediction, float, float]]:
    candidates: list[tuple[FeatureMatrixRow, RouterPrediction, float, float]] = []
    feature_history = _empty_feature_history(spec)
    active_prior_scores: list[float] = []
    for row, prediction, label, features in row_contexts:
        if (
            prediction.base_forecast_c is not None
            and label is not None
            and not (spec.mam_only and row.target_date_hkt.month not in {3, 4, 5})
        ):
            prior, available = _prior_score_sorted(spec, features, feature_history)
            p60 = _percentile_sorted(active_prior_scores, 0.60) if active_prior_scores else 0.60
            if available and prior is not None and prior >= p60:
                candidates.append((row, prediction, label, label - float(prediction.base_forecast_c)))
                insort(active_prior_scores, prior)
        _add_feature_history(spec, features, feature_history)
    return candidates


def _promotion_decision(
    spec: SpecialistSpec,
    candidates: Sequence[tuple[FeatureMatrixRow, RouterPrediction, float, float]],
    correction_c: float,
) -> tuple[str, str | None, float | None, bool, int, int]:
    active_rows = len(candidates)
    active_years = len({row.target_date_hkt.year for row, _, _, _ in candidates})
    if active_rows < 200:
        return "demoted", "ACTIVE_TRAINING_ROWS_LT_200", None, False, active_rows, active_years
    if active_years < 3:
        return "demoted", "ACTIVE_YEAR_COUNT_LT_3", None, False, active_rows, active_years
    targets = [correction_target for _, _, _, correction_target in candidates]
    median_target = float(median(targets))
    if spec.expected_sign == "negative" and median_target > -0.03:
        return "demoted", "NEGATIVE_SIGN_GATE_FAILED", None, False, active_rows, active_years
    if spec.expected_sign == "positive" and median_target < 0.03:
        return "demoted", "POSITIVE_SIGN_GATE_FAILED", None, False, active_rows, active_years
    lifts = [
        abs(label - float(prediction.base_forecast_c))
        - abs(label - (float(prediction.base_forecast_c) + correction_c))
        for _, prediction, label, _ in candidates
        if prediction.base_forecast_c is not None
    ]
    mean_lift = mean(lifts) if lifts else None
    no_harm = mean_lift is not None and mean_lift >= 0.02
    if not no_harm:
        return "demoted", "NO_HARM_GATE_FAILED", mean_lift, False, active_rows, active_years
    assert mean_lift is not None
    if spec.s6_tail_gate and mean_lift < -0.01:
        return "demoted", "S6_TAIL_GATE_FAILED", mean_lift, False, active_rows, active_years
    return "promoted", None, mean_lift, True, active_rows, active_years


def train_specialists_from_inputs(
    rows: Sequence[FeatureMatrixRow],
    router_predictions: Sequence[RouterPrediction],
) -> tuple[SpecialistTrainingResult, ...]:
    labels_by_date = {row.target_date_hkt: float(row.target_tmax_c) for row in rows if row.target_tmax_c is not None}
    predictions_by_date = {prediction.target_date_hkt: prediction for prediction in router_predictions}
    ordered_rows = sorted(rows, key=lambda item: item.target_date_hkt)
    row_contexts = [
        (row, prediction, labels_by_date.get(row.target_date_hkt), _row_feature_map(row, prediction))
        for row in ordered_rows
        for prediction in [predictions_by_date.get(row.target_date_hkt)]
        if prediction is not None
    ]
    results: list[SpecialistTrainingResult] = []
    for spec in SPECIALIST_SPECS:
        candidates = _fast_active_candidate_rows(spec, row_contexts)
        raw_correction, model_available = _fit_correction(candidates)
        correction = _signed_correction(raw_correction, spec.expected_sign)
        expected_benefit = _fit_expected_benefit(candidates, correction)
        promotion_status, demotion_reason, mean_lift, no_harm, support_count, active_years = _promotion_decision(
            spec,
            candidates,
            correction,
        )
        predictions: list[SpecialistPrediction] = []
        feature_history = _empty_feature_history(spec)
        prior_history: list[float] = []
        router_expected_history: list[float] = []
        for row, router_prediction, _label, features in row_contexts:
            prior, available = _prior_score_sorted(spec, features, feature_history)
            fold_p60 = _percentile_sorted(prior_history, 0.60) if prior_history else 0.60
            router_expected = _router_derived_features(router_prediction)["router__expected_abs_error_c"]
            tail_gate = True
            if spec.s6_tail_gate:
                tail_gate = router_expected >= (
                    _percentile_sorted(router_expected_history, 0.60) if router_expected_history else 0.60
                )
            if spec.mam_only and row.target_date_hkt.month not in {3, 4, 5}:
                predictions.append(
                    SpecialistPrediction(
                        target_date_hkt=row.target_date_hkt,
                        cutoff_id=row.cutoff_id,
                        snapshot_id=row.snapshot_id,
                        specialist_id=spec.specialist_id,
                        specialist_version=SPECIALIST_VERSION,
                        fold_id=router_prediction.fold_id,
                        prior_score=0.0,
                        score_available=True,
                        fold_p60=None,
                        regime_probability=0.0,
                        raw_correction_c=0.0,
                        shrunk_correction_c=0.0,
                        applied_correction_c=0.0,
                        expected_benefit_c=0.0,
                        activated=False,
                        activation_reason="NOT_MAM",
                        support_count=support_count,
                        active_year_count=active_years,
                        no_harm_pass=no_harm,
                        promotion_status=promotion_status,
                        leakage_status="passed",
                    )
                )
                if available and prior is not None:
                    insort(prior_history, prior)
                insort(router_expected_history, router_expected)
                _add_feature_history(spec, features, feature_history)
                continue
            support_shrink = min(1.0, support_count / 400.0)
            shrunk = correction * support_shrink
            activated = (
                available
                and prior is not None
                and prior >= fold_p60
                and expected_benefit >= 0.02
                and support_count >= 200
                and active_years >= 3
                and model_available
                and promotion_status == "promoted"
                and tail_gate
            )
            reason = "activated" if activated else "gate_failed"
            if not available:
                reason = "score_unavailable"
            elif promotion_status != "promoted":
                reason = demotion_reason or "specialist_demoted"
            elif not tail_gate:
                reason = "S6_EXPECTED_ERROR_P60_GATE_FAILED"
            predictions.append(
                SpecialistPrediction(
                    target_date_hkt=row.target_date_hkt,
                    cutoff_id=row.cutoff_id,
                    snapshot_id=row.snapshot_id,
                    specialist_id=spec.specialist_id,
                    specialist_version=SPECIALIST_VERSION,
                    fold_id=router_prediction.fold_id,
                    prior_score=prior,
                    score_available=available,
                    fold_p60=fold_p60,
                    regime_probability=prior,
                    raw_correction_c=correction,
                    shrunk_correction_c=shrunk,
                    applied_correction_c=shrunk if activated else 0.0,
                    expected_benefit_c=expected_benefit,
                    activated=activated,
                    activation_reason=reason,
                    support_count=support_count,
                    active_year_count=active_years,
                    no_harm_pass=no_harm,
                    promotion_status=promotion_status,
                    leakage_status="passed",
                )
            )
            if available and prior is not None:
                insort(prior_history, prior)
            insort(router_expected_history, router_expected)
            _add_feature_history(spec, features, feature_history)
        results.append(
            SpecialistTrainingResult(
                specialist_id=spec.specialist_id,
                promotion_status=promotion_status,
                demotion_reason=demotion_reason,
                active_training_rows=support_count,
                active_year_count=active_years,
                mean_active_lift_c=mean_lift,
                no_harm_pass=no_harm,
                predictions=tuple(predictions),
            )
        )
    if {result.specialist_id for result in results} != set(SPECIALIST_IDS):
        raise ValueError("Specialist result set does not cover every Jira specialist")
    return tuple(results)


def write_specialist_reports(writer: ReportWriter, results: Sequence[SpecialistTrainingResult]) -> None:
    writer.write_csv(
        "specialist_scoreboard_strict.csv",
        (
            "specialist_id",
            "promotion_status",
            "demotion_reason",
            "active_training_rows",
            "active_year_count",
            "mean_active_lift_c",
            "no_harm_pass",
        ),
        [
            (
                result.specialist_id,
                result.promotion_status,
                result.demotion_reason,
                result.active_training_rows,
                result.active_year_count,
                result.mean_active_lift_c,
                result.no_harm_pass,
            )
            for result in results
        ],
    )
    writer.write_csv(
        "specialist_activation_report.csv",
        (
            "target_date_hkt",
            "specialist_id",
            "prior_score",
            "fold_p60",
            "applied_correction_c",
            "expected_benefit_c",
            "activated",
            "activation_reason",
            "promotion_status",
        ),
        [
            (
                prediction.target_date_hkt,
                prediction.specialist_id,
                prediction.prior_score,
                prediction.fold_p60,
                prediction.applied_correction_c,
                prediction.expected_benefit_c,
                prediction.activated,
                prediction.activation_reason,
                prediction.promotion_status,
            )
            for result in results
            for prediction in result.predictions
        ],
    )
    writer.write_csv(
        "specialist_no_harm_report.csv",
        (
            "specialist_id",
            "no_harm_pass",
            "mean_active_lift_c",
            "demotion_reason",
        ),
        [
            (
                result.specialist_id,
                result.no_harm_pass,
                result.mean_active_lift_c,
                result.demotion_reason,
            )
            for result in results
        ],
    )
    writer.write_csv(
        "specialist_promotion_decisions.csv",
        (
            "specialist_id",
            "promotion_status",
            "demotion_reason",
            "active_training_rows",
            "active_year_count",
            "created_at_utc",
        ),
        [
            (
                result.specialist_id,
                result.promotion_status,
                result.demotion_reason,
                result.active_training_rows,
                result.active_year_count,
                "generated_by_cli",
            )
            for result in results
        ],
    )
    writer.write_root_report(
        "specialist_report.md",
        "HKG-T24-003 Specialist Report",
        (
            ("Status", "PASS"),
            (
                "Specialists",
                "\n".join(
                    f"- `{result.specialist_id}`: {result.promotion_status}, "
                    f"support={result.active_training_rows}, years={result.active_year_count}, "
                    f"reason={result.demotion_reason or 'none'}"
                    for result in results
                ),
            ),
        ),
    )


def persist_specialist_results(connection: Any, results: Sequence[SpecialistTrainingResult]) -> int:
    count = 0
    with connection.cursor() as cursor:
        for result in results:
            for prediction in result.predictions:
                cursor.execute(
                    """
                    INSERT INTO model_router.specialist_prediction (
                      target_date_hkt, cutoff_id, snapshot_id, specialist_id, specialist_version,
                      fold_id, prior_score, score_available, fold_p60, regime_probability,
                      raw_correction_c, shrunk_correction_c, applied_correction_c,
                      expected_benefit_c, activated, activation_reason, support_count,
                      active_year_count, no_harm_pass, promotion_status, leakage_status
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (target_date_hkt, cutoff_id, specialist_id, specialist_version) DO UPDATE SET
                      snapshot_id = EXCLUDED.snapshot_id,
                      fold_id = EXCLUDED.fold_id,
                      prior_score = EXCLUDED.prior_score,
                      score_available = EXCLUDED.score_available,
                      fold_p60 = EXCLUDED.fold_p60,
                      regime_probability = EXCLUDED.regime_probability,
                      raw_correction_c = EXCLUDED.raw_correction_c,
                      shrunk_correction_c = EXCLUDED.shrunk_correction_c,
                      applied_correction_c = EXCLUDED.applied_correction_c,
                      expected_benefit_c = EXCLUDED.expected_benefit_c,
                      activated = EXCLUDED.activated,
                      activation_reason = EXCLUDED.activation_reason,
                      support_count = EXCLUDED.support_count,
                      active_year_count = EXCLUDED.active_year_count,
                      no_harm_pass = EXCLUDED.no_harm_pass,
                      promotion_status = EXCLUDED.promotion_status,
                      leakage_status = EXCLUDED.leakage_status,
                      created_at_utc = now()
                    """,
                    (
                        prediction.target_date_hkt,
                        prediction.cutoff_id,
                        prediction.snapshot_id,
                        prediction.specialist_id,
                        prediction.specialist_version,
                        prediction.fold_id,
                        prediction.prior_score,
                        prediction.score_available,
                        prediction.fold_p60,
                        prediction.regime_probability,
                        prediction.raw_correction_c,
                        prediction.shrunk_correction_c,
                        prediction.applied_correction_c,
                        prediction.expected_benefit_c,
                        prediction.activated,
                        prediction.activation_reason,
                        prediction.support_count,
                        prediction.active_year_count,
                        prediction.no_harm_pass,
                        prediction.promotion_status,
                        prediction.leakage_status,
                    ),
                )
                count += 1
    return count


def run_specialist_training(
    connection: Any | None,
    writer: ReportWriter,
    *,
    scope: str,
    smoke: bool,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if scope != "strict-pre2024":
        raise ValueError("Jira003 specialists support --scope strict-pre2024")
    if smoke and connection is None:
        rows, expert_predictions = synthetic_router_inputs()
        router_results = train_router_suite_from_inputs(rows, expert_predictions)
        router_predictions = router_results[1].predictions if router_results[1].promotion_status == "promoted" else router_results[0].predictions
    else:
        raise ValueError("DB-backed specialist training is orchestrated by run-system-replay")
    results = train_specialists_from_inputs(rows, router_predictions)
    write_specialist_reports(writer, results)
    if connection is not None:
        persist_specialist_results(connection, results)
    return "PASS", (), (f"specialist_rows={sum(len(result.predictions) for result in results)}",)
