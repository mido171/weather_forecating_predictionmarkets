# Normalized Public GFS/GEFS/Himawari Smoke Data

Generated: `2026-07-08T07:53:09.863891Z`

This folder converts the provider-native files under `raw/` into readable HKG-focused artifacts.

## Fastest files to open

| File | Use |
|---|---|
| `feature_snapshot.json` | Compact one-file summary for humans and code. |
| `model_target_station_features.csv` | One row per model source at the nearest grid point to the canonical HKO target station. |
| `model_source_comparison_features.csv` | Direct GEFS-control minus GFS deltas. |
| `gfs_hkg_bbox_grid_long.csv` / `gefs_hkg_bbox_grid_long.csv` | Long-form cropped HKG grid values. |
| `himawari_b13_header_summary.json` | Decoded Himawari Standard Data metadata for the B13 segment. |

## Nearest-HKO Model Features

| Source | issuedAt UTC | validAt UTC | tmax C | t2m C | dewpoint C | wind10 m/s | MSLP hPa |
|---|---:|---:|---:|---:|---:|---:|---:|
| gfs | 2026-07-08T00:00:00Z | 2026-07-09T00:00:00Z | 27.43 | 27.18 | 24.56 | 3.62 | 1006.02 |
| gefs_control | 2026-07-08T00:00:00Z | 2026-07-09T00:00:00Z | 28.24 | 28.07 | 25.09 | 3.73 | 1005.39 |

## Scope

- GFS/GEFS are decoded from GRIB2 with `cfgrib/eccodes`, cropped to `21.5-23.5N`, `113.0-115.5E`.
- The target station is `Hong Kong Observatory` from `config/stations_hko.yaml target_station` at `22.301944`, `114.174167`.
- Temperatures are normalized from Kelvin to Celsius; pressure from Pa to hPa; wind remains m/s.
- Himawari is decoded to Standard Data header metadata. Pixel-value calibration/remapping is explicitly marked as not completed here.
