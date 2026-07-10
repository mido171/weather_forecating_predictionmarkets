from __future__ import annotations

import math
from collections.abc import Sequence


class MetricError(ValueError):
    """Raised for malformed forecast/target arrays."""


def _paired(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> list[tuple[float, float]]:
    if len(actual) != len(predicted) or not actual:
        raise MetricError("actual and predicted must be non-empty and equal length")
    pairs = [(float(a), float(p)) for a, p in zip(actual, predicted, strict=True)]
    if not all(math.isfinite(a) and math.isfinite(p) for a, p in pairs):
        raise MetricError("metrics require finite values")
    return pairs


def bias(actual: Sequence[float], predicted: Sequence[float]) -> float:
    pairs = _paired(actual, predicted)
    return sum(p - a for a, p in pairs) / len(pairs)


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    pairs = _paired(actual, predicted)
    return sum(abs(p - a) for a, p in pairs) / len(pairs)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    pairs = _paired(actual, predicted)
    return math.sqrt(sum((p - a) ** 2 for a, p in pairs) / len(pairs))


def multiclass_log_loss(
    actual_indices: Sequence[int],
    probability_rows: Sequence[Sequence[float]],
    epsilon: float = 1e-15,
) -> float:
    if len(actual_indices) != len(probability_rows) or not actual_indices:
        raise MetricError("targets and probability rows must be non-empty and equal length")
    losses: list[float] = []
    expected_width: int | None = None
    for target, row in zip(actual_indices, probability_rows, strict=True):
        probabilities = [float(value) for value in row]
        if expected_width is None:
            expected_width = len(probabilities)
        if not probabilities or len(probabilities) != expected_width:
            raise MetricError("all probability rows must have the same non-zero width")
        if target < 0 or target >= len(probabilities):
            raise MetricError(f"target index {target} outside probability row")
        if any(not math.isfinite(p) or p < 0 for p in probabilities):
            raise MetricError("probabilities must be finite and non-negative")
        total = sum(probabilities)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise MetricError(f"probability row sums to {total}, not 1")
        losses.append(-math.log(max(epsilon, min(1.0, probabilities[target]))))
    return sum(losses) / len(losses)


def multiclass_brier(
    actual_indices: Sequence[int],
    probability_rows: Sequence[Sequence[float]],
) -> float:
    if len(actual_indices) != len(probability_rows) or not actual_indices:
        raise MetricError("targets and probability rows must be non-empty and equal length")
    scores: list[float] = []
    for target, row in zip(actual_indices, probability_rows, strict=True):
        probabilities = [float(value) for value in row]
        if target < 0 or target >= len(probabilities):
            raise MetricError(f"target index {target} outside probability row")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise MetricError("probability row must sum to 1")
        scores.append(
            sum(
                (probability - (1.0 if index == target else 0.0)) ** 2
                for index, probability in enumerate(probabilities)
            )
        )
    return sum(scores) / len(scores)


def crps_ensemble(observation: float, samples: Sequence[float]) -> float:
    """CRPS for an equally weighted empirical ensemble."""
    values = [float(value) for value in samples]
    if not values:
        raise MetricError("samples must be non-empty")
    if not math.isfinite(observation) or not all(math.isfinite(x) for x in values):
        raise MetricError("observation and samples must be finite")
    first = sum(abs(x - observation) for x in values) / len(values)
    pairwise = sum(abs(x - y) for x in values for y in values)
    second = pairwise / (2 * len(values) ** 2)
    return first - second
