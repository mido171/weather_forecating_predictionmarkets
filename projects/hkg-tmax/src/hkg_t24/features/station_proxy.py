"""Proxy-only station feature helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class StationMetadata:
    station_id: str
    station_role: str = "unknown_role"
    role_confidence: float = 0.0
    latitude: float | None = None
    longitude: float | None = None
    elevation_m: float | None = None


def station_proxy_features(
    station_tmax_c: Mapping[str, float],
    metadata: Sequence[StationMetadata] = (),
) -> dict[str, float | int | None]:
    """Build proxy-only station features and deliberately omit wind direction."""
    values = list(station_tmax_c.values())
    features: dict[str, float | int | None] = {
        "station__network__available_count": len(values),
        "station__network__tmax_mean_c": None if not values else mean(values),
        "station__network__tmax_max_c": None if not values else max(values),
        "station__network__tmax_min_c": None if not values else min(values),
        "station__network__tmax_range_c": None if not values else max(values) - min(values),
    }
    for item in metadata:
        prefix = f"station__{item.station_id}"
        features[f"{prefix}__role_confidence"] = item.role_confidence
        features[f"{prefix}__elevation_m"] = item.elevation_m
    forbidden = [name for name in features if "wind_direction" in name or name.endswith("__wdir_deg")]
    if forbidden:
        raise ValueError("Station proxy features must not include wind direction: " + ", ".join(forbidden))
    return features
