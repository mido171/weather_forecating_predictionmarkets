from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .timeutils import asof_eligible, require_aware


class AsOfError(ValueError):
    """Raised for invalid point-in-time records."""


@dataclass(frozen=True)
class TemporalRecord:
    entity_id: str
    variable: str
    valid_at: datetime
    available_at: datetime
    value: Any
    issued_at: datetime | None = None
    published_at: datetime | None = None
    retrieved_at: datetime | None = None
    source_id: str | None = None
    raw_sha256: str | None = None

    def __post_init__(self) -> None:
        require_aware(self.valid_at, "valid_at")
        require_aware(self.available_at, "available_at")
        for name in ("issued_at", "published_at", "retrieved_at"):
            value = getattr(self, name)
            if value is not None:
                require_aware(value, name)


def latest_available_by_key(
    records: Iterable[TemporalRecord],
    *,
    cutoff_at: datetime,
    key_fields: Sequence[str] = ("entity_id", "variable", "valid_at"),
) -> dict[tuple[Hashable, ...], TemporalRecord]:
    """Select the newest eligible vintage for each key, never crossing the cutoff."""
    require_aware(cutoff_at, "cutoff_at")
    selected: dict[tuple[Hashable, ...], TemporalRecord] = {}
    for record in records:
        if not asof_eligible(record.available_at, cutoff_at):
            continue
        try:
            key = tuple(getattr(record, field) for field in key_fields)
        except AttributeError as exc:
            raise AsOfError(f"Unknown key field in {key_fields}") from exc
        if not all(isinstance(value, Hashable) for value in key):
            raise AsOfError(f"Non-hashable key: {key!r}")
        prior = selected.get(key)
        if prior is None or record.available_at > prior.available_at or (
            record.available_at == prior.available_at
            and record.retrieved_at is not None
            and (prior.retrieved_at is None or record.retrieved_at > prior.retrieved_at)
        ):
            selected[key] = record
    return selected


def assert_no_target_columns(
    feature_rows: Iterable[Mapping[str, Any]],
    forbidden_names: Iterable[str] = (
        "target",
        "target_value",
        "tmax_final",
        "absolute_daily_max",
        "resolved_winner",
    ),
) -> None:
    forbidden = {name.lower() for name in forbidden_names}
    for index, row in enumerate(feature_rows):
        collisions = forbidden & {str(key).lower() for key in row}
        if collisions:
            raise AsOfError(
                f"Feature row {index} contains forbidden target-like columns: "
                f"{sorted(collisions)}"
            )
