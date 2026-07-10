from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .settlement import BucketSet, as_decimal


class DistributionError(ValueError):
    """Raised for malformed forecast distributions."""


@dataclass(frozen=True)
class DiscreteTemperatureDistribution:
    """Probability mass function over exact decimal temperatures."""

    masses: Mapping[Decimal, float]
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        normalized: dict[Decimal, float] = {}
        for raw_temperature, raw_probability in self.masses.items():
            temperature = as_decimal(raw_temperature)
            if temperature is None:
                raise DistributionError("Temperature support cannot contain null")
            probability = float(raw_probability)
            if not math.isfinite(probability) or probability < 0:
                raise DistributionError(
                    f"Invalid probability at {temperature}: {raw_probability!r}"
                )
            normalized[temperature] = normalized.get(temperature, 0.0) + probability
        if not normalized:
            raise DistributionError("Distribution must have non-empty support")
        total = sum(normalized.values())
        if not math.isclose(total, 1.0, rel_tol=self.tolerance, abs_tol=self.tolerance):
            raise DistributionError(f"Probability mass sums to {total}, not 1")
        object.__setattr__(self, "masses", normalized)

    @classmethod
    def from_mapping(
        cls,
        masses: Mapping[Any, float],
        tolerance: float = 1e-9,
    ) -> DiscreteTemperatureDistribution:
        converted: dict[Decimal, float] = {}
        for temperature, probability in masses.items():
            decimal_temperature = as_decimal(temperature)
            if decimal_temperature is None:
                raise DistributionError("Temperature support cannot contain null")
            converted[decimal_temperature] = (
                converted.get(decimal_temperature, 0.0) + float(probability)
            )
        return cls(converted, tolerance=tolerance)

    @property
    def mean(self) -> Decimal:
        return sum(
            (temperature * Decimal(str(probability)) for temperature, probability in self.masses.items()),
            start=Decimal("0"),
        )

    def probability_less_than(self, threshold: Decimal | str | float | int) -> float:
        boundary = as_decimal(threshold)
        assert boundary is not None
        return sum(
            probability
            for temperature, probability in self.masses.items()
            if temperature < boundary
        )

    def bucket_probabilities(self, bucket_set: BucketSet) -> dict[str, float]:
        probabilities = {bucket.label: 0.0 for bucket in bucket_set.buckets}
        for temperature, probability in self.masses.items():
            winner = bucket_set.winner(temperature)
            probabilities[winner.label] += probability
        total = sum(probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=self.tolerance, abs_tol=self.tolerance):
            raise DistributionError(f"Bucket probability mass sums to {total}, not 1")
        return probabilities
