from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class AvailabilityInput:
    cutoff_utc: datetime
    our_ingested_at_utc: datetime | None = None
    provider_available_at_utc: datetime | None = None
    run_time_utc: datetime | None = None
    valid_time_utc: datetime | None = None
    conservative_lag: timedelta | None = None


def require_aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def effective_available_at_utc(value: AvailabilityInput) -> datetime:
    cutoff = require_aware_utc(value.cutoff_utc, "cutoff_utc")
    del cutoff

    if value.our_ingested_at_utc is not None:
        return require_aware_utc(value.our_ingested_at_utc, "our_ingested_at_utc")
    if value.provider_available_at_utc is not None:
        return require_aware_utc(value.provider_available_at_utc, "provider_available_at_utc")
    if value.conservative_lag is None:
        raise ValueError("conservative_lag is required when observed availability is absent")
    if value.run_time_utc is not None:
        return require_aware_utc(value.run_time_utc, "run_time_utc") + value.conservative_lag
    if value.valid_time_utc is not None:
        return require_aware_utc(value.valid_time_utc, "valid_time_utc") + value.conservative_lag
    raise ValueError("run_time_utc or valid_time_utc is required for conservative lag")


def is_cutoff_eligible(value: AvailabilityInput) -> bool:
    cutoff = require_aware_utc(value.cutoff_utc, "cutoff_utc")
    return effective_available_at_utc(value) <= cutoff
