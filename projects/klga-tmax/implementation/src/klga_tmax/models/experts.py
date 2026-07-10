from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from klga_tmax.models.pmf import PmfSummary, gaussian_pmf, normalize_pmf, summarize_pmf


@dataclass(frozen=True)
class ExpertDefinition:
    name: str
    family: str
    feature_prefixes: tuple[str, ...]
    default_sigma_f: float
    fallback_status: str = "fallback"


@dataclass(frozen=True)
class ExpertForecast:
    expert_name: str
    summary: PmfSummary
    status: str
    feature_names: tuple[str, ...]
    feature_hash: str
    diagnostics: dict[str, Any]


EXPERTS: tuple[ExpertDefinition, ...] = (
    ExpertDefinition(
        "expert_a_long_history_climatology_mos_station",
        "long_history",
        ("climatology_", "wu_history_", "station_actuals_", "mos_"),
        4.0,
    ),
    ExpertDefinition(
        "expert_b_dynamic_bias_corrected_composite",
        "dynamic_composite",
        ("mos_guidance_tmax_", "gribstream_tmax_", "wu_history_"),
        3.5,
    ),
    ExpertDefinition(
        "expert_c_nbm_nbmqmd_specialist",
        "nbm",
        ("grib_nbm", "grib_nbmqmd", "grib_nbmqmd", "grib_blend"),
        3.2,
    ),
    ExpertDefinition(
        "expert_d_hrrr_rap_local_regime",
        "hrrr_rap",
        ("grib_hrrr", "grib_rap", "obs_", "risk_sea_breeze", "risk_backdoor"),
        3.8,
    ),
    ExpertDefinition(
        "expert_e_global_ensemble_distribution",
        "global_ensemble",
        ("grib_gfs", "grib_gefs", "grib_ifs", "grib_aifs", "ensemble_"),
        4.2,
    ),
    ExpertDefinition(
        "expert_f_ai_model_specialist",
        "ai_model",
        ("grib_aifsoper", "grib_aigfssfc", "model_disagreement_", "ensemble_spread"),
        4.5,
        "disabled_data_sufficiency",
    ),
    ExpertDefinition(
        "expert_g_current_state_observation_correction",
        "obs_correction",
        ("obs_", "mos_guidance_tmax_", "gribstream_tmax_"),
        3.7,
    ),
    ExpertDefinition(
        "expert_h_analog_residual",
        "analog_residual",
        ("climatology_", "wu_history_", "regime_", "risk_"),
        4.8,
    ),
    ExpertDefinition(
        "expert_i_settlement_source_reconciliation",
        "settlement_reconciliation",
        ("wu_history_", "station_actuals_", "obs_", "risk_marine_layer"),
        4.4,
    ),
)


def build_expert_forecasts(feature_vector: dict[str, Any]) -> list[ExpertForecast]:
    return [
        _forecast_for_definition(definition, feature_vector)
        for definition in EXPERTS
    ]


def combine_experts(experts: list[ExpertForecast]) -> tuple[PmfSummary, dict[str, float], dict[str, Any]]:
    ok_experts = [expert for expert in experts if expert.status in {"ok", "fallback", "disabled_data_sufficiency"}]
    if not ok_experts:
        raise ValueError("at least one expert forecast is required")
    raw_weights: dict[str, float] = {}
    for expert in ok_experts:
        weight = 1.0 / max(expert.summary.uncertainty_f, 0.75)
        if expert.status == "disabled_data_sufficiency":
            weight *= 0.35
        elif expert.status == "fallback":
            weight *= 0.65
        raw_weights[expert.expert_name] = weight
    total = sum(raw_weights.values())
    weights = {name: value / total for name, value in raw_weights.items()}
    combined = {temp: 0.0 for temp in ok_experts[0].summary.pmf}
    for expert in ok_experts:
        for temp, probability in expert.summary.pmf.items():
            combined[temp] += weights[expert.expert_name] * probability
    pmf = normalize_pmf(combined)
    diagnostics = {
        "combiner": "regularized_linear_pool",
        "log_opinion_pool_status": "not_used_no_scipy_dependency",
        "expert_count": len(ok_experts),
    }
    return summarize_pmf(pmf), weights, diagnostics


def _forecast_for_definition(
    definition: ExpertDefinition,
    feature_vector: dict[str, Any],
) -> ExpertForecast:
    selected = _select_feature_values(feature_vector, definition.feature_prefixes)
    source_values = [value for _, value in selected]
    fallback_features = _fallback_feature_values(feature_vector)
    status = "ok"
    if source_values:
        center = sum(source_values) / len(source_values)
    elif fallback_features:
        center = sum(value for _, value in fallback_features) / len(fallback_features)
        selected = fallback_features
        status = definition.fallback_status
    else:
        center = 75.0
        status = definition.fallback_status
    spread = _std(source_values) if len(source_values) >= 2 else None
    sigma = max(definition.default_sigma_f, (spread or 0.0) + 1.5)
    if definition.family == "obs_correction":
        center = _obs_corrected_center(feature_vector, center)
    if definition.family == "dynamic_composite":
        center = _dynamic_bias_corrected_center(feature_vector, center)
    pmf = gaussian_pmf(center, sigma)
    feature_names = tuple(name for name, _ in selected)
    feature_hash = hashlib.sha256(
        json.dumps(
            {name: feature_vector.get(name) for name in feature_names},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return ExpertForecast(
        expert_name=definition.name,
        summary=summarize_pmf(pmf),
        status=status,
        feature_names=feature_names,
        feature_hash=feature_hash,
        diagnostics={
            "family": definition.family,
            "center_f": center,
            "sigma_f": sigma,
            "source_feature_count": len(source_values),
            "fallback_used": status != "ok",
        },
    )


def _select_feature_values(
    feature_vector: dict[str, Any],
    prefixes: tuple[str, ...],
) -> list[tuple[str, float]]:
    selected: list[tuple[str, float]] = []
    for name, raw_value in feature_vector.items():
        lower = name.lower()
        if not any(lower.startswith(prefix) for prefix in prefixes):
            continue
        if not _looks_like_temperature_feature(lower):
            continue
        value = _as_float(raw_value)
        if value is not None and 35.0 <= value <= 125.0:
            selected.append((name, value))
    return selected


def _fallback_feature_values(feature_vector: dict[str, Any]) -> list[tuple[str, float]]:
    preferred = [
        "mos_guidance_tmax_mean_f",
        "gribstream_tmax_mean_f",
        "climatology_wu_tmax_mean_31d_f",
        "wu_history_tmax_mean_14d_f",
        "obs_klga_latest_temp_f",
    ]
    values: list[tuple[str, float]] = []
    for name in preferred:
        value = _as_float(feature_vector.get(name))
        if value is not None and 35.0 <= value <= 125.0:
            values.append((name, value))
    return values


def _looks_like_temperature_feature(name: str) -> bool:
    if name.endswith("_count"):
        return False
    return (
        "tmax" in name
        or "temp" in name
        or "tmp" in name
        or "climatology" in name
    )


def _obs_corrected_center(feature_vector: dict[str, Any], center: float) -> float:
    obs = _as_float(feature_vector.get("obs_klga_latest_temp_f"))
    if obs is None:
        return center
    return 0.85 * center + 0.15 * obs


def _dynamic_bias_corrected_center(feature_vector: dict[str, Any], center: float) -> float:
    lag2 = _as_float(feature_vector.get("wu_history_tmax_lag_2d_f"))
    mean14 = _as_float(feature_vector.get("wu_history_tmax_mean_14d_f"))
    if lag2 is None or mean14 is None:
        return center
    recent_anomaly = max(min(lag2 - mean14, 4.0), -4.0)
    return center + 0.20 * recent_anomaly


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
