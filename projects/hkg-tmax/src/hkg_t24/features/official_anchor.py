"""Official HKO forecast anchor and revision feature builders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from hkg_t24.constants import OFFICIAL_FEATURE_WHITELIST
from hkg_t24.features.official_text import official_text_flags, psr_numeric_proxy


@dataclass(frozen=True)
class OfficialForecastRow:
    """One HKO official forecast vintage row for a target date."""

    issue_at_utc: datetime
    forecast_min_c: float | None
    forecast_max_c: float | None
    forecast_text: str | None = None
    psr_value: object | None = None
    row_quality_status: str = "usable_local_minmax"


def eligible_official_rows(
    rows: Sequence[OfficialForecastRow],
    *,
    operational_freeze_utc: datetime,
) -> list[OfficialForecastRow]:
    """Return usable rows issued no later than the operational freeze."""
    usable_statuses = {"usable_local_minmax", "usable_local_tmax_only"}
    return sorted(
        [
            row
            for row in rows
            if row.issue_at_utc <= operational_freeze_utc
            and row.row_quality_status in usable_statuses
            and row.forecast_max_c is not None
        ],
        key=lambda row: row.issue_at_utc,
    )


def latest_eligible_official_row(
    rows: Sequence[OfficialForecastRow],
    *,
    operational_freeze_utc: datetime,
) -> OfficialForecastRow | None:
    """Return the latest usable official row visible before H24N freeze."""
    eligible = eligible_official_rows(rows, operational_freeze_utc=operational_freeze_utc)
    return eligible[-1] if eligible else None


def official_feature_map(
    rows: Sequence[OfficialForecastRow],
    *,
    operational_freeze_utc: datetime,
) -> dict[str, float | int | None]:
    """Build official anchor, revision, text, and PSR features for one date."""
    eligible = eligible_official_rows(rows, operational_freeze_utc=operational_freeze_utc)
    features: dict[str, float | int | None] = {name: None for name in OFFICIAL_FEATURE_WHITELIST}
    latest = eligible[-1] if eligible else None
    if latest is None:
        for flag_name in official_text_flags(None):
            features[flag_name] = 0
        return features

    forecast_min = latest.forecast_min_c
    forecast_max = latest.forecast_max_c
    features["official__forecast_min_c"] = forecast_min
    features["official__forecast_max_c"] = forecast_max
    if forecast_min is not None and forecast_max is not None:
        features["official__forecast_range_c"] = forecast_max - forecast_min
        features["official__forecast_midpoint_c"] = (forecast_max + forecast_min) / 2.0
    features["official__issue_hour_hkt"] = (latest.issue_at_utc.hour + 8) % 24
    features["official__hours_before_cutoff"] = (
        operational_freeze_utc - latest.issue_at_utc
    ).total_seconds() / 3600.0
    features["official__revision_count"] = len(eligible)

    first = eligible[0]
    if latest.forecast_min_c is not None and first.forecast_min_c is not None:
        features["official__revision_min_delta_c"] = latest.forecast_min_c - first.forecast_min_c
    if latest.forecast_max_c is not None and first.forecast_max_c is not None:
        features["official__revision_max_delta_c"] = latest.forecast_max_c - first.forecast_max_c
    latest_range = (
        None
        if latest.forecast_min_c is None or latest.forecast_max_c is None
        else latest.forecast_max_c - latest.forecast_min_c
    )
    first_range = (
        None
        if first.forecast_min_c is None or first.forecast_max_c is None
        else first.forecast_max_c - first.forecast_min_c
    )
    if latest_range is not None and first_range is not None:
        features["official__revision_range_delta_c"] = latest_range - first_range

    features.update(official_text_flags(latest.forecast_text))
    features["official__psr_numeric_proxy"] = psr_numeric_proxy(latest.psr_value, latest.forecast_text)
    return features
