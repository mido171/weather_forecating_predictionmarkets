from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class MaterializationContext:
    target_date: date
    cutoff_id: str
    cutoff_utc: datetime
    local_day_start_utc: datetime
    local_day_end_utc: datetime
    feature_version: str
    mode: str
