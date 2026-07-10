from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time


def classify_peak_time(value: datetime | time) -> str:
    """Classify peak timing with fixed clock thresholds, not fitted labels."""

    clock = value.timetz() if isinstance(value, datetime) else value
    if clock < time(12, 0):
        return "early_before_1200"
    if clock < time(17, 0):
        return "normal_1200_1659"
    return "late_1700_or_after"


def count_peak_episodes(peak_times: Sequence[datetime], *, gap_minutes: float = 15.0) -> int:
    """Count separated peak episodes from ordered timestamps at the daily maximum."""

    if not peak_times:
        return 0
    ordered = sorted(peak_times)
    episodes = 1
    for previous, current in zip(ordered, ordered[1:], strict=False):
        gap = (current - previous).total_seconds() / 60.0
        if gap > gap_minutes:
            episodes += 1
    return episodes


def maximum_heating_in_window(
    times: Sequence[datetime],
    values: Sequence[float],
    *,
    end_time: datetime,
    window_minutes: float,
) -> float | None:
    """Return the largest positive temperature increase inside a backward window."""

    paired = [
        (timestamp, value)
        for timestamp, value in zip(times, values, strict=True)
        if 0 <= (end_time - timestamp).total_seconds() / 60.0 <= window_minutes
    ]
    if len(paired) < 2:
        return None
    best = 0.0
    min_so_far = paired[0][1]
    for _, value in paired[1:]:
        best = max(best, value - min_so_far)
        min_so_far = min(min_so_far, value)
    return best

