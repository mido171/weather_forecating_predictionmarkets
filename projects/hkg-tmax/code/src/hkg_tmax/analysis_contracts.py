from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from .timeutils import parse_iso_aware, require_aware

HKT = ZoneInfo("Asia/Hong_Kong")

OPERATIONAL_ROLES = {
    "OPERATIONAL_POINT_IN_TIME",
    "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
    "PROXY_WITH_LIMITATIONS",
    "STATIC_DETERMINISTIC",
}

REJECTED_OPERATIONAL_ROLES = {
    "TARGET_ONLY",
    "RETROSPECTIVE_MECHANISM_ONLY",
    "RETROSPECTIVE_ONLY",
    "PROSPECTIVE_ONLY_NOT_YET_BACKTESTABLE",
    "REJECTED",
}


@dataclass(frozen=True)
class EligibilityViolation:
    row_index: int
    reason: str
    field: str
    value: str


class PointInTimeEligibilityError(RuntimeError):
    """Raised when a candidate feature row violates the T-24 eligibility contract."""

    def __init__(self, violations: list[EligibilityViolation]) -> None:
        self.violations = violations
        detail = "; ".join(
            f"row {v.row_index}: {v.reason} ({v.field}={v.value})" for v in violations[:5]
        )
        if len(violations) > 5:
            detail += f"; plus {len(violations) - 5} more"
        super().__init__(detail)


def hko_tminus1_15_cutoff(target_date: datetime) -> datetime:
    """Return the 15:00 HKT cutoff on T-1 for a local target date/datetime."""

    local = require_aware(target_date, "target_date").astimezone(HKT)
    cutoff_date = local.date().toordinal() - 1
    cutoff_local_date = datetime.fromordinal(cutoff_date).date()
    return datetime.combine(cutoff_local_date, time(15, 0), tzinfo=HKT)


def validate_point_in_time_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    cutoff_hkt: datetime,
    allow_proxy: bool = True,
) -> None:
    """Reject rows unavailable at the forecast cutoff or operationally disallowed.

    Each row is expected to contain at least `available_at` and `role`. `available_at`
    may be an ISO string or a timezone-aware datetime. This validator is intentionally
    small and strict so tests can inject future data, reanalysis and target-derived
    fields and verify rejection before modelling code can consume them.
    """

    cutoff = require_aware(cutoff_hkt, "cutoff_hkt").astimezone(HKT)
    violations: list[EligibilityViolation] = []
    allowed_roles = set(OPERATIONAL_ROLES)
    if not allow_proxy:
        allowed_roles.discard("PROXY_WITH_LIMITATIONS")

    for index, row in enumerate(rows):
        role = str(row.get("role", ""))
        if role in REJECTED_OPERATIONAL_ROLES or role not in allowed_roles:
            violations.append(
                EligibilityViolation(index, "role is not operationally eligible", "role", role)
            )

        raw_available_at = row.get("available_at", "")
        try:
            if isinstance(raw_available_at, datetime):
                available_at = require_aware(raw_available_at, "available_at")
            else:
                available_at = parse_iso_aware(str(raw_available_at), "available_at")
        except Exception:
            violations.append(
                EligibilityViolation(
                    index, "available_at is missing or invalid", "available_at", str(raw_available_at)
                )
            )
            continue

        if available_at.astimezone(HKT) > cutoff:
            violations.append(
                EligibilityViolation(
                    index,
                    "available_at is after cutoff",
                    "available_at",
                    available_at.isoformat(),
                )
            )

        target_derived = str(row.get("target_derived", "false")).lower()
        if target_derived in {"1", "true", "yes"}:
            violations.append(
                EligibilityViolation(
                    index,
                    "target-derived feature is forbidden",
                    "target_derived",
                    str(row.get("target_derived", "")),
                )
            )

    if violations:
        raise PointInTimeEligibilityError(violations)

