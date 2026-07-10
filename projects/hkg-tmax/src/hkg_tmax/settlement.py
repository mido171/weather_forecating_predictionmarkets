from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .config import ConfigError, load_yaml
from .hashing import sha256_text


class SettlementError(ValueError):
    """Raised when bucket or settlement semantics are invalid."""


def as_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SettlementError(f"Invalid decimal boundary/value: {value!r}") from exc


@dataclass(frozen=True)
class Bucket:
    label: str
    lower_inclusive: Decimal | None
    upper_exclusive: Decimal | None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Bucket:
        label = value.get("label")
        if not isinstance(label, str) or not label:
            raise SettlementError("Bucket label must be a non-empty string")
        return cls(
            label=label,
            lower_inclusive=as_decimal(value.get("lower_inclusive")),
            upper_exclusive=as_decimal(value.get("upper_exclusive")),
        )

    def contains(self, temperature: Decimal | str | float | int) -> bool:
        candidate = as_decimal(temperature)
        assert candidate is not None
        if self.lower_inclusive is not None and candidate < self.lower_inclusive:
            return False
        return not (
            self.upper_exclusive is not None and candidate >= self.upper_exclusive
        )


@dataclass(frozen=True)
class BucketSet:
    buckets: tuple[Bucket, ...]
    require_full_coverage: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mappings(
        cls,
        values: Iterable[dict[str, Any]],
        require_full_coverage: bool = True,
    ) -> BucketSet:
        return cls(
            tuple(Bucket.from_mapping(value) for value in values),
            require_full_coverage=require_full_coverage,
        )

    def sorted(self) -> tuple[Bucket, ...]:
        return tuple(
            sorted(
                self.buckets,
                key=lambda bucket: (
                    bucket.lower_inclusive is not None,
                    bucket.lower_inclusive or Decimal("-Infinity"),
                ),
            )
        )

    def validate(self) -> None:
        if not self.buckets:
            raise SettlementError("At least one bucket is required")
        labels = [bucket.label for bucket in self.buckets]
        if len(labels) != len(set(labels)):
            raise SettlementError("Bucket labels must be unique")

        ordered = self.sorted()
        for bucket in ordered:
            if (
                bucket.lower_inclusive is not None
                and bucket.upper_exclusive is not None
                and bucket.lower_inclusive >= bucket.upper_exclusive
            ):
                raise SettlementError(f"Invalid empty/reversed bucket: {bucket.label}")

        if self.require_full_coverage:
            if ordered[0].lower_inclusive is not None:
                raise SettlementError("Full-coverage bucket set must have a lower tail")
            if ordered[-1].upper_exclusive is not None:
                raise SettlementError("Full-coverage bucket set must have an upper tail")

        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.upper_exclusive is None:
                raise SettlementError(
                    f"Unbounded upper bucket {previous.label!r} is not last"
                )
            if current.lower_inclusive is None:
                raise SettlementError(
                    f"Unbounded lower bucket {current.label!r} is not first"
                )
            if previous.upper_exclusive < current.lower_inclusive:
                raise SettlementError(
                    f"Gap between {previous.label!r} and {current.label!r}: "
                    f"{previous.upper_exclusive} to {current.lower_inclusive}"
                )
            if previous.upper_exclusive > current.lower_inclusive:
                raise SettlementError(
                    f"Overlap between {previous.label!r} and {current.label!r}"
                )

    def winner(self, temperature: Decimal | str | float | int) -> Bucket:
        matches = [bucket for bucket in self.buckets if bucket.contains(temperature)]
        if len(matches) != 1:
            raise SettlementError(
                f"Expected exactly one bucket for {temperature!r}; found {len(matches)}"
            )
        return matches[0]

    def probability_from_cdf(
        self,
        bucket: Bucket,
        cdf: Callable[[Decimal], float],
    ) -> float:
        lower_probability = 0.0 if bucket.lower_inclusive is None else cdf(
            bucket.lower_inclusive
        )
        upper_probability = 1.0 if bucket.upper_exclusive is None else cdf(
            bucket.upper_exclusive
        )
        value = upper_probability - lower_probability
        if value < -1e-12 or value > 1 + 1e-12:
            raise SettlementError(f"Invalid CDF-derived probability: {value}")
        return min(1.0, max(0.0, value))


def load_bucket_set(path: Path) -> BucketSet:
    data = load_yaml(path)
    values = data.get("buckets")
    if not isinstance(values, list):
        raise ConfigError(f"{path}: buckets must be a list")
    return BucketSet.from_mappings(values)


_WHITESPACE = re.compile(r"\s+")


def normalize_rules_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def rules_hash(text: str, normalized: bool = True) -> str:
    value = normalize_rules_text(text) if normalized else text
    return sha256_text(value)
