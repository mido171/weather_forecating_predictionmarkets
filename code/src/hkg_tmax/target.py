from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from .hko import DailyExtractRow


class TargetError(ValueError):
    """Raised when target construction cannot be proven fail-closed."""


EXPECTED_DAILY_EXTRACT_FIELD = "Absolute Daily Max (deg. C)"
EXPECTED_STATION_CODE = "HKO"
EXPECTED_PRECISION_C = Decimal("0.1")


@dataclass(frozen=True)
class TargetObservation:
    source_id: str
    station_code: str
    local_date: date
    value_c: Decimal
    field_name: str
    precision_c: Decimal
    quality_state: str
    source_sha256: str | None
    retrieved_at: datetime | None


def require_daily_extract_target(
    rows: Sequence[DailyExtractRow],
    *,
    target_date: date,
    source_id: str,
    station_code: str = EXPECTED_STATION_CODE,
    field_name: str = EXPECTED_DAILY_EXTRACT_FIELD,
    required_precision_c: Decimal = EXPECTED_PRECISION_C,
    source_sha256: str | None = None,
    retrieved_at: datetime | None = None,
    source_error: str | None = None,
) -> TargetObservation:
    if source_error:
        raise TargetError(f"source failure: {source_error}")
    if not source_id:
        raise TargetError("missing source_id")
    if station_code != EXPECTED_STATION_CODE:
        raise TargetError(f"station mismatch: {station_code!r} != {EXPECTED_STATION_CODE!r}")
    if field_name != EXPECTED_DAILY_EXTRACT_FIELD:
        raise TargetError(f"missing expected field: {EXPECTED_DAILY_EXTRACT_FIELD}")

    matches = [row for row in rows if row.local_date == target_date]
    if len(matches) != 1:
        raise TargetError(
            f"ambiguous date match for {target_date.isoformat()}: {len(matches)} rows"
        )
    row = matches[0]
    if row.absolute_daily_max_c is None:
        raise TargetError(f"missing target value for {target_date.isoformat()}")
    if row.parse_issue:
        raise TargetError(
            f"target value is not complete for {target_date.isoformat()}: {row.parse_issue}"
        )
    precision = row.value_precision
    if precision is None or precision != required_precision_c:
        raise TargetError(
            "unsupported precision for "
            f"{target_date.isoformat()}: {precision} != {required_precision_c}"
        )

    return TargetObservation(
        source_id=source_id,
        station_code=station_code,
        local_date=row.local_date,
        value_c=row.absolute_daily_max_c,
        field_name=field_name,
        precision_c=precision,
        quality_state=row.completeness,
        source_sha256=source_sha256,
        retrieved_at=retrieved_at,
    )
