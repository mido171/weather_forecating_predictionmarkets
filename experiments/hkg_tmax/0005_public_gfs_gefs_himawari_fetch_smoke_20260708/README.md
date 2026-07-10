# Public GFS/GEFS/Himawari Fetch Smoke

Generated: `2026-07-08T06:35:31.089440Z`

This folder proves direct public-provider fetchability for the latest accessible GFS, GEFS, and Himawari-9 payloads without using GribStream.

| Source | Status | issuedAt / observedAt UTC | validAt UTC | Bytes | Saved payload |
|---|---|---:|---:|---:|---|
| gfs | fetched | 2026-07-08T00:00:00Z | 2026-07-09T00:00:00Z | 4062413 | `raw\gfs\gfs_20260708_00z_f024_hkg_bbox.grib2` |
| gefs | fetched | 2026-07-08T00:00:00Z | 2026-07-09T00:00:00Z | 3515940 | `raw\gefs\gefs_control_20260708_00z_f024_hkg_bbox.grib2` |
| himawari9 | fetched | 2026-07-08T06:20:00Z | 2026-07-08T06:20:00Z | 1347225 | `raw\himawari\HS_H09_20260708_0620_B13_FLDK_R20_S0110.DAT.bz2` |

Raw provider payloads live under `raw/`. Machine-readable metadata lives in `artifacts/fetch_summary.json`.

## Normalized Outputs

Normalized, readable outputs were generated at `2026-07-08T07:53:09.863891Z` under `normalized/`.

- `normalized/README.md` is the human entrypoint.
- `normalized/model_target_station_features.csv` gives nearest-HKO model features.
- `normalized/model_source_comparison_features.csv` gives GEFS-control minus GFS deltas.
- `normalized/hkg_bbox_grid_long_all_sources.csv` gives all cropped model grid rows in long form.
- `normalized/himawari_b13_header_summary.json` decodes Himawari B13 metadata and records that pixel calibration is still separate work.
