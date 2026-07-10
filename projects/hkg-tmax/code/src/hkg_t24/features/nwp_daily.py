"""NWP unit conversions and deterministic daily feature builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean

HKG_NWP_LOCATIONS = (
    "center",
    "inland_nw",
    "marine_s",
    "local_n",
    "local_s",
    "local_e",
    "local_w",
    "airport",
    "urban_ne",
    "kowloon",
    "nt_east",
    "lantau",
)


def kelvin_to_c(value: float) -> float:
    return value - 273.15


def pa_to_hpa(value: float) -> float:
    return value / 100.0


def kg_m2_precip_to_mm(value: float) -> float:
    return value


def meter_precip_to_mm(value: float) -> float:
    return value * 1000.0


def joule_m2_to_mj_m2(value: float) -> float:
    return value / 1_000_000.0


def threshold_feature_key(threshold_c: float) -> str:
    """Return the exact GEFS probability threshold key suffix."""
    return f"prob_ge_{threshold_c:.1f}".replace(".", "_")


def _difference(
    values: Mapping[str, float],
    left: str,
    right: str,
) -> float | None:
    left_value = values.get(left)
    right_value = values.get(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def build_location_tmax_features(
    prefix: str,
    location_tmax_c: Mapping[str, float],
) -> dict[str, float | None]:
    """Build `<prefix>__{location}__tmax_c` for all canonical locations."""
    return {
        f"{prefix}__{location}__tmax_c": location_tmax_c.get(location)
        for location in HKG_NWP_LOCATIONS
    }


def build_gfs_daily_features(
    *,
    location_tmax_c: Mapping[str, float],
    center_dewpoint_c: Sequence[float] = (),
    center_low_cloud_pct: Sequence[float] = (),
    center_shortwave_w_m2: Sequence[float] = (),
    center_precip_mm: Sequence[float] = (),
    center_wind_speed_10m_mps: Sequence[float] = (),
    center_easterly_component_mps: Sequence[float] = (),
    center_t850_c: Sequence[float] = (),
    center_z500_m: Sequence[float] = (),
    center_relative_humidity_700_pct: Sequence[float] = (),
    center_dewpoint_2m_c: Sequence[float] = (),
) -> dict[str, float | None]:
    """Build deterministic GFS strict features."""
    features = build_location_tmax_features("gfs", location_tmax_c)
    center_tmax = location_tmax_c.get("center")
    dew_first = center_dewpoint_c[0] if center_dewpoint_c else None
    dew_last = center_dewpoint_c[-1] if center_dewpoint_c else None
    features.update(
        {
            "gfs__center__dewpoint_change_proxy_c": None
            if dew_first is None or dew_last is None
            else dew_last - dew_first,
            "gfs__center__low_cloud_pct_mean": None
            if not center_low_cloud_pct
            else mean(center_low_cloud_pct),
            "gfs__center__shortwave_w_m2_mean": None
            if not center_shortwave_w_m2
            else mean(center_shortwave_w_m2),
            "gfs__center__precip_mm_sum": None if not center_precip_mm else sum(center_precip_mm),
            "gfs__center__wind_speed_10m_mean_mps": None
            if not center_wind_speed_10m_mps
            else mean(center_wind_speed_10m_mps),
            "gfs__center__onshore_easterly_component_mps": None
            if not center_easterly_component_mps
            else mean(center_easterly_component_mps),
            "gfs__center__temp_dewpoint_spread_mean_c": None
            if center_tmax is None or not center_dewpoint_c
            else center_tmax - mean(center_dewpoint_c),
            "gfs__center__t850_c_mean": None if not center_t850_c else mean(center_t850_c),
            "gfs__center__z500_m_mean": None if not center_z500_m else mean(center_z500_m),
            "gfs__center__relative_humidity_700_pct_mean": None
            if not center_relative_humidity_700_pct
            else mean(center_relative_humidity_700_pct),
            "gfs__center__dewpoint_2m_c_mean": None
            if not center_dewpoint_2m_c
            else mean(center_dewpoint_2m_c),
            "gfs__spatial__inland_nw_minus_center_tmax_c": _difference(
                location_tmax_c, "inland_nw", "center"
            ),
            "gfs__spatial__inland_nw_minus_marine_s_tmax_c": _difference(
                location_tmax_c, "inland_nw", "marine_s"
            ),
            "gfs__spatial__center_minus_marine_s_tmax_c": _difference(
                location_tmax_c, "center", "marine_s"
            ),
            "gfs__spatial__local_n_minus_local_s_tmax_c": _difference(
                location_tmax_c, "local_n", "local_s"
            ),
            "gfs__spatial__local_e_minus_local_w_tmax_c": _difference(
                location_tmax_c, "local_e", "local_w"
            ),
        }
    )
    return features


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def build_gefs_mean_features(
    *,
    location_tmax_c: Mapping[str, float],
    center_pwat_kg_m2: Sequence[float] = (),
    center_onshore_east_component_mps: Sequence[float] = (),
    center_wind_speed_10m_mps: Sequence[float] = (),
) -> dict[str, float | None]:
    """Build GEFS mean strict features."""
    features = build_location_tmax_features("gefsmean", location_tmax_c)
    features["gefsmean__center__pwat_kg_m2_mean"] = (
        None if not center_pwat_kg_m2 else mean(center_pwat_kg_m2)
    )
    features["gefsmean__center__onshore_east_component_mps_mean"] = (
        None if not center_onshore_east_component_mps else mean(center_onshore_east_component_mps)
    )
    features["gefsmean__center__wind_speed_10m_mps_mean"] = (
        None if not center_wind_speed_10m_mps else mean(center_wind_speed_10m_mps)
    )
    return features


def build_gefs_ensemble_features(center_member_tmax_c: Sequence[float]) -> dict[str, float | None]:
    """Build HKO-center GEFS ensemble strict features."""
    sorted_members = sorted(center_member_tmax_c)
    features: dict[str, float | None] = {
        "gefsens__center__tmax_p10_c": _quantile(sorted_members, 0.10),
        "gefsens__center__tmax_p25_c": _quantile(sorted_members, 0.25),
        "gefsens__center__tmax_p50_c": _quantile(sorted_members, 0.50),
        "gefsens__center__tmax_p75_c": _quantile(sorted_members, 0.75),
        "gefsens__center__tmax_p90_c": _quantile(sorted_members, 0.90),
    }
    p10 = features["gefsens__center__tmax_p10_c"]
    p90 = features["gefsens__center__tmax_p90_c"]
    features["gefsens__center__tmax_spread_p90_p10_c"] = (
        None if p10 is None or p90 is None else p90 - p10
    )
    for step in range(60, 81):
        threshold = step / 2.0
        key = f"gefsens__center__{threshold_feature_key(threshold)}"
        if not center_member_tmax_c:
            features[key] = None
        else:
            features[key] = sum(1 for value in center_member_tmax_c if value >= threshold) / len(center_member_tmax_c)
    return features


def shadow_center_feature_names() -> tuple[str, ...]:
    return (
        "ifsoper__center__tmax_c",
        "ifsens__center__tmax_c",
        "aifsoper__center__tmax_c",
        "aifsens__center__tmax_c",
        "aigfssfc__center__tmax_c",
        "graphcast__center__tmax_c",
        "fourcastnet__center__tmax_c",
        "cwawrf15__center__tmax_c",
        "arwf__center__tmax_c",
    )
