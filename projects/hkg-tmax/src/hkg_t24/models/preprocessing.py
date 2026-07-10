"""Fold-local preprocessing for lightweight expert models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median

Numeric = float | int


@dataclass(frozen=True)
class FeaturePreprocessor:
    feature_names: tuple[str, ...]
    impute_values: dict[str, float]

    def transform_row(self, features: Mapping[str, object]) -> list[float]:
        values: list[float] = []
        for feature_name in self.feature_names:
            raw = features.get(feature_name)
            if isinstance(raw, bool):
                values.append(float(int(raw)))
            elif isinstance(raw, int | float):
                values.append(float(raw))
            else:
                values.append(self.impute_values[feature_name])
        return values


def fit_preprocessor(
    rows: Sequence[Mapping[str, object]],
    feature_names: Sequence[str],
) -> FeaturePreprocessor:
    """Fit median imputers from training rows only."""
    impute_values: dict[str, float] = {}
    for feature_name in feature_names:
        values = [
            float(value)
            for row in rows
            for value in [row.get(feature_name)]
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        impute_values[feature_name] = 0.0 if not values else float(median(values))
    return FeaturePreprocessor(feature_names=tuple(feature_names), impute_values=impute_values)
