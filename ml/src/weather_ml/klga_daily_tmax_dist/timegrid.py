from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import (
    CUTOFF_END_HOUR,
    CUTOFF_END_MINUTE,
    CUTOFF_START_HOUR,
    CUTOFF_START_MINUTE,
    CUTOFF_STEP_MINUTES,
    NY_ZONE,
    UTC_ZONE,
)


@dataclass(frozen=True)
class CutoffPoint:
    target_date_local: date
    cutoff_local: datetime
    cutoff_utc: datetime
    midnight_utc: datetime
    cutoff_minutes: int
    n_expected_bins: int
    is_dst: int


def make_cutoffs_for_date(
    target_date_local: date,
    tz: ZoneInfo = NY_ZONE,
    *,
    start_hm: tuple[int, int] = (CUTOFF_START_HOUR, CUTOFF_START_MINUTE),
    end_hm: tuple[int, int] = (CUTOFF_END_HOUR, CUTOFF_END_MINUTE),
    step_minutes: int = CUTOFF_STEP_MINUTES,
) -> list[CutoffPoint]:
    start_hour, start_minute = start_hm
    end_hour, end_minute = end_hm
    midnight_local = datetime(
        target_date_local.year,
        target_date_local.month,
        target_date_local.day,
        0,
        0,
        tzinfo=tz,
    )
    current = datetime(
        target_date_local.year,
        target_date_local.month,
        target_date_local.day,
        start_hour,
        start_minute,
        tzinfo=tz,
    )
    end_local = datetime(
        target_date_local.year,
        target_date_local.month,
        target_date_local.day,
        end_hour,
        end_minute,
        tzinfo=tz,
    )
    out: list[CutoffPoint] = []
    while current <= end_local:
        elapsed_seconds = (current - midnight_local).total_seconds()
        expected_bins = int(np.floor(elapsed_seconds / 1800.0) + 1)
        cutoff_minutes = current.hour * 60 + current.minute
        out.append(
            CutoffPoint(
                target_date_local=target_date_local,
                cutoff_local=current,
                cutoff_utc=current.astimezone(UTC_ZONE),
                midnight_utc=midnight_local.astimezone(UTC_ZONE),
                cutoff_minutes=cutoff_minutes,
                n_expected_bins=expected_bins,
                is_dst=1 if current.utcoffset() == timedelta(hours=-4) else 0,
            )
        )
        current = current + timedelta(minutes=step_minutes)
    return out


def make_calendar_grid(
    dates: Iterable[date],
    tz: ZoneInfo = NY_ZONE,
) -> pd.DataFrame:
    rows: list[dict] = []
    for d in sorted(set(dates)):
        for p in make_cutoffs_for_date(d, tz=tz):
            rows.append(
                {
                    "target_date_local": p.target_date_local,
                    "cutoff_local": p.cutoff_local,
                    "cutoff_utc": p.cutoff_utc,
                    "midnight_utc": p.midnight_utc,
                    "cutoff_minutes": p.cutoff_minutes,
                    "n_expected_bins": p.n_expected_bins,
                    "is_dst": p.is_dst,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "target_date_local",
                "cutoff_local",
                "cutoff_utc",
                "midnight_utc",
                "cutoff_minutes",
                "n_expected_bins",
                "is_dst",
            ]
        )
    df = pd.DataFrame(rows)
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    df["midnight_utc"] = pd.to_datetime(df["midnight_utc"], utc=True)
    return df.sort_values(["target_date_local", "cutoff_minutes"]).reset_index(drop=True)

