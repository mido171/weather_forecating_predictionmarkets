from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

LOCKED_TEST_START = date(2025, 1, 1)


class LockedTestAccessError(RuntimeError):
    """Raised when ordinary research code attempts to access locked-test dates."""


@dataclass(frozen=True)
class LockedDateViolation:
    index: int
    target_date: date


def coerce_local_date(value: object) -> date:
    """Convert a common scalar date representation to a local calendar date."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("empty date value")
    return datetime.fromisoformat(text[:10]).date()


def locked_test_violations(dates: Iterable[object]) -> list[LockedDateViolation]:
    violations: list[LockedDateViolation] = []
    for index, value in enumerate(dates):
        day = coerce_local_date(value)
        if day >= LOCKED_TEST_START:
            violations.append(LockedDateViolation(index=index, target_date=day))
    return violations


def assert_no_locked_dates(dates: Iterable[object], *, context: str) -> None:
    """Fail closed if any target date belongs to the locked-test period."""

    violations = locked_test_violations(dates)
    if not violations:
        return
    examples = ", ".join(v.target_date.isoformat() for v in violations[:5])
    extra = "" if len(violations) <= 5 else f" plus {len(violations) - 5} more"
    raise LockedTestAccessError(
        f"{context} attempted to access locked-test target dates >= "
        f"{LOCKED_TEST_START.isoformat()}: {examples}{extra}"
    )


def locked_test_guard_markdown() -> str:
    return f"""# Locked-Test Guard

Ordinary HKG T-24 research commands must reject target dates greater than or
equal to `{LOCKED_TEST_START.isoformat()}`. Existing archived rows for 2025-2026
may remain on disk, but research code must not compute losses, select features,
tune models, or inspect failure cases on those rows.

The guard is implemented in `hkg_tmax.hkg_t24.guard` and covered by unit tests.
Any future explicit unlock must be audited separately and was not invoked for
this goal.
"""


def write_locked_test_guard_report(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(locked_test_guard_markdown(), encoding="utf-8")
    return path

