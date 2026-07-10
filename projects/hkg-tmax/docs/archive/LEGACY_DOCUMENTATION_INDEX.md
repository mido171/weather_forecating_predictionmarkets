# HKG Tmax Dataset Documentation

Generated at: 2026-06-23T16:34:40Z

This folder documents the normalized HKG Tmax research dataset corpus from the 2026-06-23 audit snapshot and the database ingestion run. It is intentionally audit-backed: counts, layers, model roles, date ranges, quality issues, and attribute semantics are copied from the structured audit and ingestion artifacts.

## What is documented

| Document | Purpose |
| --- | --- |
| DATASET_CATALOG.md | One-row-per-dataset overview with use, value, data range, row counts, and recommended DB layer. |
| SOURCE_TABLE_INVENTORY.md | All 52 source files/tables with row counts, date ranges, DB actions, model status, hashes where available, and reconciliation status. |
| DATA_QUALITY_REGISTER.md | All 22 open audit quality issues and their required actions. |
| DATABASE_USAGE_AND_LAYER_GUIDE.md | How the dataset corpus is saved in the local research database and which layers are safe to query. |
| ATTRIBUTE_DICTIONARY_FULL.csv | Machine-readable full attribute dictionary for all 1,869 audited attributes. |
| datasets/*.md | Per-dataset documentation with full attribute tables and dataset-specific quality notes. |

## Corpus snapshot

| Metric | Value |
| --- | --- |
| datasets_profiled | 13 |
| files_profiled | 52 |
| row_tables_profiled | 51 |
| row_table_rows_total | 7219745 |
| attributes_profiled | 1869 |
| quality_issues | 22 |
| stations | 36 |
| audit_bundle_sha256 | bdbc1fce90c03ce74ee20b864691467bf0dd9a00be996a42119618ababb3fb27 |
| database_engine_used_for_ingestion | postgresql |
| database_ingestion_status | PASS |
| ingestion_batch_id | audit-ingest-bdbc1fce90c0-primary |

## Dataset documents

| dataset_id | dataset | documentation_file | recommended_layer | operational_value | diagnostic_value | future_potential |
| --- | --- | --- | --- | --- | --- | --- |
| 01_hko_daily_tmax_target | HKO Daily Tmax Target | datasets/01_hko_daily_tmax_target.md | label_core / raw_audit | 95 | 100 | 100 |
| 02_hko_daily_climate_all_elements | HKO Daily Climate All Elements | datasets/02_hko_daily_climate_all_elements.md | diagnostic_physics | 0 | 90 | 75 |
| 03_noaa_igra_upper_air_hkm00045004 | NOAA IGRA Upper Air Hkm00045004 | datasets/03_noaa_igra_upper_air_hkm00045004.md | diagnostic_physics | 0 | 95 | 85 |
| 04_noaa_isd_regional_surface | NOAA ISD Regional Surface | datasets/04_noaa_isd_regional_surface.md | diagnostic_station_network | 0 | 98 | 92 |
| 05_hko_historical_rss_forecasts | HKO Historical RSS Forecasts | datasets/05_hko_historical_rss_forecasts.md | operational_archive / anchor / research_supervised | 100 | 100 | 100 |
| 06_hko_tropical_cyclone_best_track | HKO Tropical Cyclone Best Track | datasets/06_hko_tropical_cyclone_best_track.md | diagnostic_regime_labels | 0 | 75 | 65 |
| 07_hko_radar_satellite_lightning_nowcast | HKO Radar Satellite Lightning Nowcast | datasets/07_hko_radar_satellite_lightning_nowcast.md | live_exact_vintage / live_object_catalog | 5 | 45 | 92 |
| 08_hko_marine_tide_coastal_waters | HKO Marine Tide Coastal Waters | datasets/08_hko_marine_tide_coastal_waters.md | live_exact_vintage | 5 | 40 | 60 |
| 09_hko_arwf_station_forecasts | HKO ARWF Station Forecasts | datasets/09_hko_arwf_station_forecasts.md | live_nwp_anchor | 5 | 35 | 98 |
| 10_ncep_operational_grib_inventory | NCEP Operational GRIB Inventory | datasets/10_ncep_operational_grib_inventory.md | nwp_object_catalog | 0 | 15 | 100 |
| 11_static_geospatial_inventory | Static Geospatial Inventory | datasets/11_static_geospatial_inventory.md | static_object_catalog / station_context | 20 | 70 | 85 |
| 12_hkg_t24_robust_experiment_outputs | HKG T24 Robust Experiment Outputs | datasets/12_hkg_t24_robust_experiment_outputs.md | research_artifacts / research_metrics / research_oof | 0 | 100 | 80 |
| [root] | [root] | datasets/root.md | catalog / acquisition_provenance / object_catalog | 0 | 45 | 50 |

## Source artifacts used

| Artifact | Path |
| --- | --- |
| Audit summary | data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/AUDIT_SUMMARY.json |
| Dataset decisions | data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_DATASET_DECISION_MATRIX.csv |
| Table decisions | data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_TABLE_DECISIONS_ALL_52.csv |
| Attribute decisions | data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_ATTRIBUTE_DECISIONS_ALL_1869.csv |
| Quality issues | data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_DATA_QUALITY_ISSUES.csv |
| DB ingestion summary | experiments/0206_audit_driven_database_ingestion/summary.json |
