"""Diagnostic proxy-only climate feature helpers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from statistics import mean


def lagged_climate_proxy_features(
    rows: Sequence[tuple[date, float | None]],
    *,
    target_date_hkt: date,
) -> dict[str, float | int | None]:
    """Build simple climate proxy features using only local_date <= T-2."""
    cutoff = target_date_hkt - timedelta(days=2)
    eligible = [float(value) for local_date, value in rows if value is not None and local_date <= cutoff]
    last7 = eligible[-7:]
    last30 = eligible[-30:]
    return {
        "climate__lagged__available_count": len(eligible),
        "climate__lagged__last_tmax_c": None if not eligible else eligible[-1],
        "climate__lagged__roll7_mean_c": None if len(last7) < 7 else mean(last7),
        "climate__lagged__roll30_mean_c": None if len(last30) < 30 else mean(last30),
    }


def diagnostic_report_feature_names() -> tuple[str, ...]:
    """Report-only proxy families that must not enter strict matrices."""
    return ("igra_diagnostic_report__status", "tc_diagnostic_report__status")
