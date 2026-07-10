from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone


class TraceAvailabilityViolation(ValueError):
    pass


class LabelLeakageViolation(ValueError):
    pass


@dataclass(frozen=True)
class FeatureSourceTrace:
    source_name: str
    record_key: str
    effective_available_at_utc: datetime


def validate_feature_trace_for_cutoff(
    *,
    cutoff_utc: datetime,
    source_trace: list[FeatureSourceTrace],
) -> None:
    if cutoff_utc.tzinfo is None or cutoff_utc.utcoffset() is None:
        raise ValueError("cutoff_utc must be timezone-aware")
    cutoff = cutoff_utc.astimezone(timezone.utc)
    violating = [
        trace
        for trace in source_trace
        if trace.effective_available_at_utc.astimezone(timezone.utc) > cutoff
    ]
    if violating:
        latest = max(
            trace.effective_available_at_utc.astimezone(timezone.utc)
            for trace in violating
        )
        raise TraceAvailabilityViolation(
            f"feature source availability {latest.isoformat()} is after cutoff {cutoff.isoformat()}"
        )


def assert_daily_high_label_history_safe(
    *,
    target_date: date,
    label_dates_used: list[date],
    feature_name: str,
) -> None:
    max_allowed = target_date - timedelta(days=2)
    bad_dates = [label_date for label_date in label_dates_used if label_date > max_allowed]
    if bad_dates:
        raise LabelLeakageViolation(
            f"{feature_name} may use KLGA daily-high labels only through {max_allowed.isoformat()}; "
            f"rejected {sorted(bad_dates)}"
        )
