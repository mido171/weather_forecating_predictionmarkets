from __future__ import annotations

FEATURE_ALIASES = {
    "nbm_tmax_f": "grib_nbm_klga_core_tmax_hourly_f",
    "nbm_tmax": "grib_nbm_klga_core_tmax_hourly_f",
    "hrrr_tmax_f": "grib_hrrr_klga_core_tmax_hourly_f",
    "hrrr_tmax": "grib_hrrr_klga_core_tmax_hourly_f",
    "rap_tmax_f": "grib_rap_klga_core_tmax_hourly_f",
    "rap_tmax": "grib_rap_klga_core_tmax_hourly_f",
    "gfs_tmax_f": "grib_gfs_klga_core_tmax_hourly_f",
    "gfs_tmax": "grib_gfs_klga_core_tmax_hourly_f",
    "gefs_mean_tmax_f": "grib_gefsatmosmean_klga_core_tmax_hourly_f",
    "gefs_mean_tmax": "grib_gefsatmosmean_klga_core_tmax_hourly_f",
    "ifsoper_tmax_f": "grib_ifsoper_klga_core_tmax_hourly_f",
    "aifsoper_tmax_f": "grib_aifsoper_klga_core_tmax_hourly_f",
    "aigfssfc_tmax_f": "grib_aigfssfc_klga_core_tmax_hourly_f",
}


def resolve_feature_alias(feature_name: str) -> str:
    return FEATURE_ALIASES.get(feature_name, feature_name)
