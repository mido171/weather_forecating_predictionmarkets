"""Constrained static/dynamic expert weighting for Jira003 routers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite
from statistics import mean
from typing import Any

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from hkg_t24.constants import ROUTER_LAMBDA_GRID, ROUTER_TAU_GRID


@dataclass(frozen=True)
class WeightOptimizationResult:
    weights: dict[str, float]
    objective_mae_c: float | None
    status: str
    cap_trace: dict[str, object]


@dataclass(frozen=True)
class DynamicWeightSelection:
    tau: float
    lambda_: float
    mae_c: float


def normalize_key(raw: str) -> str:
    """Return a stable JSON/report key fragment."""
    return raw.replace(".", "_").replace("-", "_")


def _available_experts(
    expert_ids: Sequence[str],
    caps: Mapping[str, float],
    availability: Mapping[str, bool],
    promoted: Mapping[str, bool],
) -> list[str]:
    return [
        expert_id
        for expert_id in expert_ids
        if caps.get(expert_id, 0.0) > 0.0
        and availability.get(expert_id, False)
        and promoted.get(expert_id, False)
    ]


def apply_caps_and_masks(
    weights: Mapping[str, float],
    *,
    expert_ids: Sequence[str],
    caps: Mapping[str, float],
    availability: Mapping[str, bool],
    promoted: Mapping[str, bool],
    fallback_order: Sequence[str] = ("E0_OFFICIAL_RAW_ANCHOR", "E2_TARGET_MEMORY"),
) -> WeightOptimizationResult:
    """Mask unavailable/demoted/capped experts, then renormalize or fall back."""
    allowed = _available_experts(expert_ids, caps, availability, promoted)
    capped = {
        expert_id: min(max(float(weights.get(expert_id, 0.0)), 0.0), float(caps.get(expert_id, 0.0)))
        if expert_id in allowed
        else 0.0
        for expert_id in expert_ids
    }
    total = sum(capped.values())
    if total > 0.0:
        normalized = {expert_id: value / total for expert_id, value in capped.items()}
        return WeightOptimizationResult(
            weights=normalized,
            objective_mae_c=None,
            status="renormalized",
            cap_trace={"allowed_experts": allowed, "masked_weights": capped},
        )

    for expert_id in fallback_order:
        if expert_id in expert_ids and availability.get(expert_id, False) and promoted.get(expert_id, False):
            fallback = {candidate: 0.0 for candidate in expert_ids}
            fallback[expert_id] = 1.0
            return WeightOptimizationResult(
                weights=fallback,
                objective_mae_c=None,
                status=f"fallback_{expert_id}",
                cap_trace={"allowed_experts": allowed, "masked_weights": capped},
            )

    return WeightOptimizationResult(
        weights={expert_id: 0.0 for expert_id in expert_ids},
        objective_mae_c=None,
        status="no_available_expert",
        cap_trace={"allowed_experts": allowed, "masked_weights": capped},
    )


def optimize_static_weights(
    *,
    expert_ids: Sequence[str],
    prediction_matrix: Sequence[Mapping[str, float]],
    labels: Sequence[float],
    caps: Mapping[str, float],
) -> WeightOptimizationResult:
    """Minimize MAE subject to non-negative weights, sum=1, and expert caps."""
    if len(prediction_matrix) != len(labels):
        raise ValueError("prediction_matrix and labels length mismatch")
    allowed = [expert_id for expert_id in expert_ids if caps.get(expert_id, 0.0) > 0.0]
    if not prediction_matrix or not allowed:
        return WeightOptimizationResult(
            weights={expert_id: 0.0 for expert_id in expert_ids},
            objective_mae_c=None,
            status="no_training_rows",
            cap_trace={"allowed_experts": allowed},
        )

    complete_rows: list[list[float]] = []
    complete_labels: list[float] = []
    for row, label in zip(prediction_matrix, labels, strict=True):
        values: list[float] = []
        complete = True
        for expert_id in allowed:
            value = row.get(expert_id)
            if value is None or not isfinite(float(value)):
                complete = False
                break
            values.append(float(value))
        if complete:
            complete_rows.append(values)
            complete_labels.append(float(label))

    if not complete_rows:
        return WeightOptimizationResult(
            weights={expert_id: 0.0 for expert_id in expert_ids},
            objective_mae_c=None,
            status="no_common_training_rows",
            cap_trace={"allowed_experts": allowed},
        )

    x = np.asarray(complete_rows, dtype=float)
    y = np.asarray(complete_labels, dtype=float)
    cap_values = np.asarray([float(caps[expert_id]) for expert_id in allowed], dtype=float)
    initial = cap_values / float(cap_values.sum())

    def objective(candidate: Any) -> float:
        prediction = x @ candidate
        return float(np.mean(np.abs(prediction - y)))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, float(cap)) for cap in cap_values],
        constraints=({"type": "eq", "fun": lambda candidate: float(np.sum(candidate)) - 1.0},),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        weights = {expert_id: 0.0 for expert_id in expert_ids}
        best_index = int(np.argmin([objective(np.eye(len(allowed))[idx]) for idx in range(len(allowed))]))
        weights[allowed[best_index]] = 1.0
        return WeightOptimizationResult(
            weights=weights,
            objective_mae_c=objective(np.eye(len(allowed))[best_index]),
            status=f"slsqp_failed_{result.message}",
            cap_trace={"allowed_experts": allowed, "common_row_count": len(complete_rows)},
        )

    raw_weights = [max(0.0, min(float(value), float(cap))) for value, cap in zip(result.x, cap_values, strict=True)]
    total = sum(raw_weights)
    normalized = [value / total for value in raw_weights] if total > 0.0 else [0.0 for _ in raw_weights]
    weights = {expert_id: 0.0 for expert_id in expert_ids}
    weights.update({expert_id: weight for expert_id, weight in zip(allowed, normalized, strict=True)})
    return WeightOptimizationResult(
        weights=weights,
        objective_mae_c=objective(np.asarray(normalized, dtype=float)),
        status="optimized",
        cap_trace={"allowed_experts": allowed, "common_row_count": len(complete_rows)},
    )


def dynamic_weights_from_expected_errors(
    expected_errors: Mapping[str, float],
    *,
    expert_ids: Sequence[str],
    tau: float,
) -> dict[str, float]:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    raw = {
        expert_id: exp(-max(float(expected_errors.get(expert_id, 3.0)), 0.20) / tau)
        for expert_id in expert_ids
    }
    total = sum(raw.values())
    if total <= 0.0:
        return {expert_id: 0.0 for expert_id in expert_ids}
    return {expert_id: value / total for expert_id, value in raw.items()}


def blend_static_dynamic_weights(
    static_weights: Mapping[str, float],
    dynamic_weights: Mapping[str, float],
    *,
    expert_ids: Sequence[str],
    lambda_: float,
) -> dict[str, float]:
    if lambda_ < 0.0 or lambda_ > 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    raw = {
        expert_id: (1.0 - lambda_) * float(static_weights.get(expert_id, 0.0))
        + lambda_ * float(dynamic_weights.get(expert_id, 0.0))
        for expert_id in expert_ids
    }
    total = sum(raw.values())
    if total <= 0.0:
        return {expert_id: 0.0 for expert_id in expert_ids}
    return {expert_id: value / total for expert_id, value in raw.items()}


def select_tau_lambda(
    *,
    labels: Sequence[float],
    predictions_by_row: Sequence[Mapping[str, float]],
    expected_error_by_row: Sequence[Mapping[str, float]],
    static_weights: Mapping[str, float],
    expert_ids: Sequence[str],
) -> DynamicWeightSelection:
    """Select tau/lambda by MAE, then lower lambda, then higher tau."""
    if not labels:
        return DynamicWeightSelection(tau=0.50, lambda_=0.0, mae_c=999.0)
    candidates: list[DynamicWeightSelection] = []
    for lambda_ in ROUTER_LAMBDA_GRID:
        for tau in ROUTER_TAU_GRID:
            abs_errors: list[float] = []
            for label, predictions, expected_errors in zip(
                labels, predictions_by_row, expected_error_by_row, strict=True
            ):
                dynamic_weights = dynamic_weights_from_expected_errors(
                    expected_errors,
                    expert_ids=expert_ids,
                    tau=tau,
                )
                final_weights = blend_static_dynamic_weights(
                    static_weights,
                    dynamic_weights,
                    expert_ids=expert_ids,
                    lambda_=lambda_,
                )
                forecast = sum(final_weights[expert_id] * predictions[expert_id] for expert_id in expert_ids)
                abs_errors.append(abs(forecast - label))
            candidates.append(DynamicWeightSelection(tau=tau, lambda_=lambda_, mae_c=mean(abs_errors)))
    candidates.sort(key=lambda item: (round(item.mae_c, 12), item.lambda_, -item.tau))
    return candidates[0]
