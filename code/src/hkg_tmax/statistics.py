from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean


class StatisticalError(ValueError):
    """Raised for invalid statistical comparison inputs."""


@dataclass(frozen=True)
class BootstrapDifference:
    """Candidate-minus-baseline loss difference; negative means candidate improves."""

    observed_mean: float
    lower: float
    upper: float
    confidence_level: float
    repetitions: int
    block_length: int


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise StatisticalError("Cannot compute quantile of empty values")
    if probability <= 0:
        return float(sorted_values[0])
    if probability >= 1:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )


def paired_moving_block_bootstrap(
    candidate_losses: Sequence[float],
    baseline_losses: Sequence[float],
    *,
    block_length: int,
    repetitions: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20260618,
) -> BootstrapDifference:
    if len(candidate_losses) != len(baseline_losses) or not candidate_losses:
        raise StatisticalError("Loss arrays must be non-empty and equal length")
    if block_length < 1 or block_length > len(candidate_losses):
        raise StatisticalError("block_length must be in [1, sample_size]")
    if repetitions < 100:
        raise StatisticalError("Use at least 100 bootstrap repetitions")
    if not 0 < confidence_level < 1:
        raise StatisticalError("confidence_level must be between 0 and 1")

    differences = [
        float(candidate) - float(baseline)
        for candidate, baseline in zip(candidate_losses, baseline_losses, strict=True)
    ]
    if not all(math.isfinite(value) for value in differences):
        raise StatisticalError("Loss differences must be finite")

    sample_size = len(differences)
    starts = range(0, sample_size - block_length + 1)
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(repetitions):
        sampled: list[float] = []
        while len(sampled) < sample_size:
            start = rng.choice(starts)
            sampled.extend(differences[start : start + block_length])
        replicates.append(mean(sampled[:sample_size]))

    replicates.sort()
    alpha = 1 - confidence_level
    return BootstrapDifference(
        observed_mean=mean(differences),
        lower=_quantile(replicates, alpha / 2),
        upper=_quantile(replicates, 1 - alpha / 2),
        confidence_level=confidence_level,
        repetitions=repetitions,
        block_length=block_length,
    )
