"""Shared Jira003 forecast metric calculations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import mean, median


@dataclass(frozen=True)
class ForecastMetricSummary:
    row_count: int
    mae_c: float
    rmse_c: float
    bias_c: float
    median_abs_error_c: float
    p75_abs_error_c: float
    p90_abs_error_c: float
    p95_abs_error_c: float
    large_error_ge_1c_rate: float
    large_error_ge_2c_rate: float


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile with q in [0, 1]."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if q < 0.0 or q > 1.0:
        raise ValueError("percentile q must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def forecast_metrics(pairs: Iterable[tuple[float, float]]) -> ForecastMetricSummary:
    """Compute completion-spec point forecast metrics from (prediction, label) pairs."""
    errors = [prediction - label for prediction, label in pairs]
    if not errors:
        raise ValueError("forecast_metrics requires at least one prediction/label pair")
    abs_errors = [abs(value) for value in errors]
    return ForecastMetricSummary(
        row_count=len(errors),
        mae_c=mean(abs_errors),
        rmse_c=sqrt(mean(value * value for value in errors)),
        bias_c=mean(errors),
        median_abs_error_c=median(abs_errors),
        p75_abs_error_c=percentile(abs_errors, 0.75),
        p90_abs_error_c=percentile(abs_errors, 0.90),
        p95_abs_error_c=percentile(abs_errors, 0.95),
        large_error_ge_1c_rate=mean(1.0 if value >= 1.0 else 0.0 for value in abs_errors),
        large_error_ge_2c_rate=mean(1.0 if value >= 2.0 else 0.0 for value in abs_errors),
    )


def metric_delta(candidate: ForecastMetricSummary, baseline: ForecastMetricSummary) -> float:
    """Return candidate MAE minus baseline MAE on already identical rows."""
    return candidate.mae_c - baseline.mae_c
