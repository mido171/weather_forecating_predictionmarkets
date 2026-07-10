"""Deterministic H24N calendar model features."""

from __future__ import annotations

import math
from datetime import date

from hkg_t24.constants import CALENDAR_MODEL_FEATURE_WHITELIST
from hkg_t24.timeutils import season_for_month


def calendar_feature_map(target_date_hkt: date) -> dict[str, float | int]:
    """Return the final Jira 002 calendar feature map for one target date."""
    month = target_date_hkt.month
    day_of_year = target_date_hkt.timetuple().tm_yday
    season = season_for_month(month)
    features: dict[str, float | int] = {
        "calendar__month_sin1": math.sin(2.0 * math.pi * month / 12.0),
        "calendar__month_cos1": math.cos(2.0 * math.pi * month / 12.0),
        "calendar__doy_sin1": math.sin(2.0 * math.pi * day_of_year / 365.2425),
        "calendar__doy_cos1": math.cos(2.0 * math.pi * day_of_year / 365.2425),
        "calendar__is_mam": int(season == "MAM"),
        "calendar__is_jja": int(season == "JJA"),
        "calendar__is_son": int(season == "SON"),
        "calendar__is_djf": int(season == "DJF"),
        "calendar__year_index": target_date_hkt.year - 2000,
    }
    missing = sorted(set(CALENDAR_MODEL_FEATURE_WHITELIST) - set(features))
    if missing:
        raise ValueError("Calendar feature builder missed required features: " + ", ".join(missing))
    return {name: features[name] for name in CALENDAR_MODEL_FEATURE_WHITELIST}
