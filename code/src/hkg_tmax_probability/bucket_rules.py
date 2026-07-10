"""Decimal-safe bucket rules for HKG one-decimal daily Tmax markets.

The public target is the one-decimal HKO Daily Extract maximum at HKG.  Bucket
assignment therefore must be done after one-decimal decimal normalization, not
with binary float rounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Iterable

import numpy as np

BUCKET_KEYS: tuple[str, ...] = (
    "24_or_below",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34_or_higher",
)

BUCKET_INDEX: dict[str, int] = {bucket: index for index, bucket in enumerate(BUCKET_KEYS)}
PROBABILITY_COLUMNS: tuple[str, ...] = tuple(f"p_{bucket}" for bucket in BUCKET_KEYS)


@dataclass(frozen=True)
class BucketInterval:
    key: str
    lower: Decimal | None
    upper: Decimal | None


BUCKET_INTERVALS: tuple[BucketInterval, ...] = (
    BucketInterval("24_or_below", None, Decimal("24.9")),
    BucketInterval("25", Decimal("25.0"), Decimal("25.9")),
    BucketInterval("26", Decimal("26.0"), Decimal("26.9")),
    BucketInterval("27", Decimal("27.0"), Decimal("27.9")),
    BucketInterval("28", Decimal("28.0"), Decimal("28.9")),
    BucketInterval("29", Decimal("29.0"), Decimal("29.9")),
    BucketInterval("30", Decimal("30.0"), Decimal("30.9")),
    BucketInterval("31", Decimal("31.0"), Decimal("31.9")),
    BucketInterval("32", Decimal("32.0"), Decimal("32.9")),
    BucketInterval("33", Decimal("33.0"), Decimal("33.9")),
    BucketInterval("34_or_higher", Decimal("34.0"), None),
)

ONE_DECIMAL = Decimal("0.1")


def decimal_1dp(value: object) -> Decimal:
    """Normalize a numeric value to the published one-decimal target precision."""
    try:
        return Decimal(str(value)).quantize(ONE_DECIMAL, rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert value to one-decimal Tmax: {value!r}") from exc


def bucket_key(value: object) -> str:
    """Return the bucket key for a one-decimal Tmax value."""
    tmax = decimal_1dp(value)
    if tmax <= Decimal("24.9"):
        return "24_or_below"
    if tmax >= Decimal("34.0"):
        return "34_or_higher"
    whole = int(tmax.to_integral_value(rounding=ROUND_HALF_UP))
    if Decimal(whole) > tmax:
        whole -= 1
    if whole < 25 or whole > 33:
        raise ValueError(f"Unexpected bucket after normalization: {value!r} -> {tmax}")
    return str(whole)


def bucket_index(value: object) -> int:
    return BUCKET_INDEX[bucket_key(value)]


def key_to_index(bucket: str) -> int:
    return BUCKET_INDEX[bucket]


def bucket_midpoints() -> np.ndarray:
    """Representative values used only for distribution diagnostics."""
    return np.array([24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5], dtype=float)


def bucket_boundaries_for_cdf() -> np.ndarray:
    """Upper CDF cut points for buckets over the continuous Tmax scale."""
    return np.array([24.95, 25.95, 26.95, 27.95, 28.95, 29.95, 30.95, 31.95, 32.95, 33.95], dtype=float)


def bucket_indices_from_values(values: Iterable[object]) -> np.ndarray:
    return np.array([bucket_index(value) for value in values], dtype=int)


def bucket_keys_from_values(values: Iterable[object]) -> list[str]:
    return [bucket_key(value) for value in values]


def probability_columns(prefix: str = "p_") -> list[str]:
    return [f"{prefix}{bucket}" for bucket in BUCKET_KEYS]


def assert_probability_mass(probs: np.ndarray, tolerance: float = 1e-8) -> None:
    matrix = np.asarray(probs, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(BUCKET_KEYS):
        raise ValueError(f"Expected (n, {len(BUCKET_KEYS)}) probability matrix, got {matrix.shape}")
    if np.any(matrix < -tolerance):
        raise ValueError("Probability matrix contains negative mass")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        raise ValueError(f"Probability rows do not sum to 1; max error={np.max(np.abs(row_sums - 1.0))}")


def normalize_probability_matrix(probs: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    matrix = np.asarray(probs, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(BUCKET_KEYS):
        raise ValueError(f"Expected (n, {len(BUCKET_KEYS)}) probability matrix, got {matrix.shape}")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.maximum(matrix, floor)
    return matrix / matrix.sum(axis=1, keepdims=True)


def bucket_probability_frame(probs: np.ndarray) -> dict[str, np.ndarray]:
    matrix = normalize_probability_matrix(probs)
    return {column: matrix[:, idx] for idx, column in enumerate(PROBABILITY_COLUMNS)}


def validate_bucket_rule_contract() -> dict[str, str]:
    """Return important boundary examples for audit artifacts and tests."""
    examples = ["24.9", "25.0", "25.9", "26.0", "31.9", "32.0", "33.9", "34.0"]
    return {example: bucket_key(example) for example in examples}
