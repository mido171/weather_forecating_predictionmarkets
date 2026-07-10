# 0008_last2_gfs_gefs_radar_structured_delivery_20260708

Two-day structured delivery for GFS, GEFS control, and radar data.

UTC window: `2026-07-06T00:00:00Z` to `2026-07-08T00:00:00Z` exclusive.

## Key Outputs

| Output | Rows | Meaning |
|---|---:|---|
| `normalized/model_fetch_manifest_last2.csv` | 272 | GFS/GEFS requested object manifest, status, URL, issue/valid/as-of clocks, raw hash/bytes from the 7-day run. |
| `normalized/model_idx_catalog_last2.csv` | 272 | Full NOMADS GRIB index-level catalog per cycle/lead, including available variables. |
| `normalized/model_station_features_last2.csv` | 272 | HKO point feature rows. |
| `normalized/model_bbox_features_last2.csv` | 3840 | HKG bounding-box summary feature rows. |
| `normalized/radar_envf_manifest_frames_last2.csv` | 240 | Historical ENVF-served HKO radar frame manifest. |
| `normalized/radar_envf_image_features_last2.csv` | 240 | Numeric image-derived radar color/rainfall proxies. |
| `normalized/attribute_catalog_last2.csv` | 163 | Column-by-column attribute catalog for every table above. |
| `normalized/source_issue_glue_last2.csv` | 512 | High-level glue rows suitable for a Postgres registry table. |
| `metadata/postgres_glue_schema.sql` | - | Proposed high-level Postgres schema. |

## Leakage / As-Of Clocks

GFS/GEFS rows retain `issued_at_utc`, `valid_at_utc`, and `availability_proxy_utc` from experiment 0007.

Radar rows are from HKUST ENVF historical display of HKO radar imagery. They have observed image times, not native HKO historical issue metadata, so the delivery marks them `not_native_exact_vintage` and uses `observed_at_utc + 30m` as a conservative availability proxy.

## Raw Retention

This folder intentionally keeps no raw payloads. Radar image bytes are fetched, decoded into numeric features, and discarded in memory. The script removes its staging directory at the end.
