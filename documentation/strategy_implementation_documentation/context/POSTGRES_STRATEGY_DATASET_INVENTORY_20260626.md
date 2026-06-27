# PostgreSQL Strategy Dataset Inventory - 2026-06-26

Generated at UTC: `2026-06-26T13:53:42Z`

This file inventories every strategy-relevant dataset currently visible in the local PostgreSQL database for the HKG T+24 Tmax implementation. It was produced after reading the strategy implementation docs in this folder and then querying PostgreSQL directly. Credentials are intentionally redacted; counts, spans, and table names come from the live database.

## Strategy docs read before DB profiling

| Document | Bytes | Last modified |
| --- | --- | --- |
| GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md | 21,565 | 2026-06-26T14:02:52 |
| GRIBSTREAM_LEAKAGE_SAFE_DB_RETRIEVAL_LEDGER_20260626.md | 28,191 | 2026-06-26T14:14:19 |
| HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md | 64,667 | 2026-06-26T14:29:38 |

## Database inspected

| Field | Value |
| --- | --- |
| database | hkg_tmax_research |
| host | 127.0.0.1 |
| port | 5432 |
| connection role used | postgres |
| inspected_at_utc | 2026-06-26 13:53:42.383713+00:00 |
| postgres_version | PostgreSQL 16.3, compiled by Visual C++ build 1938, 64-bit |
| non_system_objects | 94 |
| base_table_exact_rows_excluding_views | 8,123,580 |
| base_table_total_size_excluding_views | 5.39 GB |

Important interpretation rules:

- The original audit-driven PostgreSQL ingestion accounts for 13 cataloged datasets, 52 source files/tables, and 1,869 audited attributes.
- The tactical GribStream H24N backfill is an additional live PostgreSQL corpus under `nwp_tactical.*` and is included here because it is central to the current strategy docs.
- A table existing in PostgreSQL does not mean it is safe for model input. The strategy requires point-in-time eligibility, H24N cutoff discipline, strict/proxy/live scoreboards, and sealed 2024/2025 treatment.
- `feature_safe.hko_t24_official_anchor` is currently empty in this live DB state even though `public.hko_historical_forecasts_2000_2026` contains the usable official forecast archive. Treat the anchor view as not yet materialized/populated here.

## High-level source map

| Dataset family | Primary DB objects | Rows/objects | Date span | Strategy role |
| --- | --- | --- | --- | --- |
| HKO daily Tmax target history | `label_core.hko_daily_tmax`; `raw_audit.*`; `sealed_confirmation.hko_daily_tmax`; `feature_safe.hko_target_history_pre2024` | 48,577 | 1884-01-01..2023-12-31 | Canonical label and T-2-or-older target-memory source; HKO history starts in 1884 in the DB. |
| HKO official forecast archive | `public.hko_historical_forecasts_2000_2026`; empty `operational_anchor.*` / `feature_safe.hko_t24_official_anchor` | 324,179 | 2000-01-02..2026-06-21 | Primary official anchor/residual source; anchor extraction still needs materialized/populated safe view. |
| GribStream tactical H24N NWP | `nwp_tactical.raw_response_object`; `nwp_tactical.forecast_wide`; plans/stencil tables | 1,965,090 | 2020-10-03..2026-06-26 by target-date/model scope | Modern dynamic atmosphere; must join raw object scope and apply 6h H24N safety filter. |
| HKO daily climate elements | `diagnostic_physics.codex_audit_ds_02_*` | 556,399 | see variable table below | Diagnostic and lagged-cautious physical history; not same-day deployable without publication proof. |
| NOAA IGRA upper-air | two `diagnostic_physics.codex_audit_ds_03_*` tables | 565,921 | see object inventory | Physics teacher/proxy design; current quality and release timing block production use. |
| NOAA ISD regional surface | two `diagnostic_station_network.codex_audit_ds_04_*` tables | 4,346,780 | see station summaries | High-value station-network source, currently research/proxy until wind/parser/vintage issues are repaired. |
| HKO tropical cyclone best track | `diagnostic_regime_labels.codex_audit_ds_06_*` | 26,189 | 1985-01-06T06:00:00Z..2024-12-25T06:00:00Z | Retrospective diagnostic regime labels only, not live predictors. |
| HKO radar/satellite/lightning/nowcast | `live_exact_vintage.codex_audit_ds_07_*` | 143 | short live sample | Live exact-vintage cloud/rain/convection layer; shadow/prospective only now. |
| HKO marine/tide/coastal waters | `live_exact_vintage.codex_audit_ds_08_*` | 135 | short live sample | Marine context; prospective/diagnostic until long exact-vintage archive exists. |
| HKO ARWF station forecasts | `live_nwp_anchor.codex_audit_ds_09_*` | 530 | one/few-cycle live sample | Potential future local model anchor; currently shadow/prospective only. |
| NCEP operational GRIB inventory | `object_catalog.asset`; catalog metadata | 19 | object metadata only | Future NWP object inventory, not decoded model features yet. |
| Static geospatial/station context | `catalog.location*`; `catalog.station*`; `catalog.station_dim` | 532 | static/versioned | Safe deterministic context when versioned. |
| Research metrics and OOF outputs | `research_metrics.*`; `research_oof_predictions.*`; `research.*` | 231,936 | historical research artifacts | Evidence and institutional memory only; not canonical live features. |

## Date Coverage Matrices

Coverage percentages below are computed from the live PostgreSQL database.

Definitions:

- Daily calendar coverage = distinct dates present / inclusive calendar days between min and max date.
- GribStream target-day coverage = distinct `target_date_hkt` values present / inclusive calendar days in that model's target-date range.
- GribStream safe row % = rows passing the documented H24N rule `run_time_utc + 6 hours <= T-1 15:00 HKT` / full-scope rows.
- GribStream safe Tmax-candidate day % = target days with at least one safe non-null `COALESCE(interval_tmax_2m_k, temperature_2m_k)` row / distinct target days present.
- For multi-station or subdaily sources, day coverage means at least one row exists on that date; it does not prove every station/hour/member is complete.

### Core daily and anchor coverage

| Dataset/object | Min date | Max date | Calendar days | Distinct dates | Missing dates | Coverage % | Coverage meaning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `label_core.hko_daily_tmax` | 1884-01-01 | 2023-12-31 | 51,134 | 48,577 | 2,557 | 94.999% | Canonical pre-2024 HKO Tmax labels. |
| `feature_safe.hko_target_history_pre2024` | 1884-01-01 | 2023-12-31 | 51,134 | 48,577 | 2,557 | 94.999% | Feature-safe pre-2024 target-history view. |
| `sealed_confirmation.hko_daily_tmax` | 2024-01-01 | 2026-05-31 | 882 | 882 | 0 | 100.000% | Sealed 2024+ labels; not for development tuning. |
| `raw_audit.codex_audit_ds_01_*` valid local dates | 1884-01-01 | 2026-06-17 | 52,033 | 49,476 | 2,557 | 95.086% | Raw audited target table including 2024+ rows. |
| `public.hko_historical_forecasts_2000_2026` usable local min/max targets | 2000-01-02 | 2026-06-21 | 9,668 | 9,667 | 1 | 99.990% | All usable local min/max target dates. |
| `public.hko_historical_forecasts_2000_2026` usable local lead-1 targets | 2000-01-02 | 2026-06-21 | 9,668 | 9,665 | 3 | 99.969% | H24N-style lead-1 official target-date availability. |

### GribStream tactical H24N target-day and safe-candidate coverage

| Dataset | Target-date range HKT | Calendar days | Target days present | Missing target days | Target-day coverage % | Full-scope rows | Safe row % 6h | Safe Tmax-candidate rows | Safe Tmax-candidate days | Safe Tmax-candidate day % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `aifsenfo` | 2025-07-04..2026-06-23 | 355 | 355 | 0 | 100.000% | 72,270 | 100.000% | 72,270 | 355 | 100.000% |
| `aifsoper` | 2025-02-26..2026-06-23 | 483 | 483 | 0 | 100.000% | 28,884 | 79.975% | 23,100 | 482 | 99.793% |
| `aigefssfc` | 2025-06-03..2026-06-23 | 386 | 373 | 13 | 96.632% | 46,252 | 100.000% | 8,308 | 67 | 17.962% |
| `aigfspres` | 2026-04-22..2026-06-23 | 63 | 63 | 0 | 100.000% | 3,660 | 80.000% | 0 | 0 | 0.000% |
| `aigfssfc` | 2026-04-22..2026-06-23 | 63 | 63 | 0 | 100.000% | 3,660 | 80.000% | 2,928 | 61 | 96.825% |
| `cwawrf15` | 2026-06-23..2026-06-26 | 4 | 4 | 0 | 100.000% | 180 | 80.000% | 144 | 3 | 75.000% |
| `fourcastnetgfs` | 2024-05-03..2026-02-20 | 659 | 648 | 11 | 98.331% | 37,824 | 79.981% | 30,252 | 631 | 97.377% |
| `gefsatmos` | 2020-10-03..2026-06-23 | 2,090 | 2,085 | 5 | 99.761% | 516,891 | 100.000% | 516,891 | 2,085 | 100.000% |
| `gefsatmosmean` | 2020-10-03..2026-06-23 | 2,090 | 2,088 | 2 | 99.904% | 200,436 | 100.000% | 200,436 | 2,088 | 100.000% |
| `gfs` | 2021-03-23..2026-06-23 | 1,919 | 1,919 | 0 | 100.000% | 575,004 | 95.999% | 551,808 | 1,918 | 99.948% |
| `graphcast` | 2024-04-26..2026-05-06 | 741 | 741 | 0 | 100.000% | 44,220 | 80.000% | 35,376 | 737 | 99.460% |
| `ifsenfo` | 2024-03-03..2026-06-23 | 843 | 843 | 0 | 100.000% | 343,616 | 100.000% | 343,616 | 843 | 100.000% |
| `ifsoper` | 2024-02-29..2026-06-23 | 846 | 846 | 0 | 100.000% | 91,260 | 88.889% | 81,120 | 845 | 99.882% |

### HKO daily climate element coverage

| Variable | Date range | Calendar days | Distinct dates | Missing dates | Coverage % | Rows | Station/domain count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bright_sunshine_duration` | 1961-01-01..2026-05-31 | 23,892 | 23,892 | 0 | 100.000% | 23,892 | 1 |
| `cloud_to_cloud_lightning` | 2005-06-21..2026-05-31 | 7,650 | 7,650 | 0 | 100.000% | 7,650 | 1 |
| `cloud_to_ground_lightning` | 2005-06-21..2026-05-31 | 7,650 | 7,650 | 0 | 100.000% | 7,650 | 1 |
| `daily_maximum_temperature` | 1884-01-01..2026-05-31 | 52,016 | 49,459 | 2,557 | 95.084% | 49,459 | 1 |
| `daily_minimum_temperature` | 1884-01-01..2026-05-31 | 52,016 | 49,459 | 2,557 | 95.084% | 49,459 | 1 |
| `daily_rainfall` | 1884-03-01..2026-05-31 | 51,956 | 49,399 | 2,557 | 95.079% | 49,399 | 1 |
| `evaporation` | 1968-01-01..2026-05-31 | 21,336 | 21,336 | 0 | 100.000% | 21,336 | 1 |
| `global_solar_radiation` | 1978-01-01..2026-05-31 | 17,683 | 17,683 | 0 | 100.000% | 17,683 | 1 |
| `grass_minimum_temperature` | 1968-01-01..2026-05-31 | 21,336 | 21,336 | 0 | 100.000% | 21,336 | 1 |
| `mean_cloud_amount` | 1949-01-01..2026-05-31 | 28,275 | 28,275 | 0 | 100.000% | 28,275 | 1 |
| `mean_dew_point_temperature` | 1961-01-01..2026-05-31 | 23,892 | 23,892 | 0 | 100.000% | 23,892 | 1 |
| `mean_relative_humidity` | 1947-01-01..2026-05-31 | 29,006 | 29,006 | 0 | 100.000% | 29,006 | 1 |
| `mean_sea_level_pressure` | 1884-03-01..2026-05-31 | 51,956 | 49,399 | 2,557 | 95.079% | 49,399 | 1 |
| `mean_temperature` | 1884-03-01..2026-05-31 | 51,956 | 49,399 | 2,557 | 95.079% | 49,399 | 1 |
| `mean_wet_bulb_temperature` | 1947-01-01..2026-05-31 | 29,006 | 29,006 | 0 | 100.000% | 29,006 | 1 |
| `mean_wind_speed` | 1975-01-01..2026-05-31 | 18,779 | 18,779 | 0 | 100.000% | 18,779 | 1 |
| `prevailing_wind_direction` | 1975-01-01..2026-05-31 | 18,779 | 18,779 | 0 | 100.000% | 18,779 | 1 |
| `reduced_visibility_hours` | 1997-01-01..2026-05-31 | 10,743 | 10,743 | 0 | 100.000% | 10,743 | 1 |
| `sea_temperature` | 1990-01-01..2026-05-31 | 13,300 | 13,300 | 0 | 100.000% | 13,300 | 1 |
| `sea_temperature_am` | 1974-06-18..2026-05-31 | 18,976 | 18,976 | 0 | 100.000% | 18,976 | 1 |
| `sea_temperature_pm` | 1974-06-18..2026-05-31 | 18,976 | 18,976 | 0 | 100.000% | 18,976 | 1 |

### Diagnostic, live, and research-output date coverage

| Dataset/table family | Date range | Calendar days | Distinct dates | Missing dates | Day coverage % | Rows | Coverage meaning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IGRA key-level profile table | 1949-06-02..2026-06-17 | 28,140 | 27,519 | 621 | 97.793% | 88,407 | Days with at least one upper-air profile row. |
| IGRA raw/level table | 1949-06-02..2026-06-17 | 28,140 | 27,516 | 624 | 97.783% | 477,514 | Days with at least one upper-air level row. |
| NOAA ISD observation rows | 1945-11-30..2025-08-24 | 29,123 | 26,192 | 2,931 | 89.936% | 4,029,291 | Days with at least one regional station observation row. |
| NOAA ISD daily station summary | 1945-12-01..2025-08-25 | 29,123 | 26,194 | 2,929 | 89.943% | 317,489 | Days with at least one daily station-summary row. |
| TC best track | 1985-01-06..2024-12-25 | 14,599 | 5,382 | 9,217 | 36.866% | 26,189 | Days with at least one TC best-track record; sparse by event nature. |
| HKO ARWF station forecasts | 2026-06-19..2026-06-19 | 1 | 1 | 0 | 100.000% | 530 | One-day live/prospective sample, not a long archive. |
| Marine tide/coastal waters table 1 | 2026-06-19..2026-06-20 | 2 | 2 | 0 | 100.000% | 105 | Two-day live/prospective sample. |
| Marine tide/coastal waters table 2 | 2026-06-19..2026-06-20 | 2 | 2 | 0 | 100.000% | 30 | Two-day live/prospective sample. |
| Radar/satellite/lightning nowcast table 1 | 2026-06-19..2026-06-20 | 2 | 2 | 0 | 100.000% | 41 | Two-day live/prospective sample. |
| Radar/satellite/lightning nowcast table 2 | 2026-06-19..2026-06-20 | 2 | 2 | 0 | 100.000% | 102 | Two-day live/prospective sample. |
| Research OOF predictions `0ea616a5` | 1965-01-01..2023-12-31 | 21,549 | 18,624 | 2,925 | 86.426% | 55,872 | Historical OOF evidence, not a live feature source. |
| Research OOF predictions `38d58a4a` | 1965-01-01..2023-12-31 | 21,549 | 18,627 | 2,922 | 86.440% | 55,881 | Historical OOF evidence, not a live feature source. |
| Research OOF predictions `9be0ab50` | 1965-01-01..2023-12-31 | 21,549 | 18,624 | 2,925 | 86.426% | 55,872 | Historical OOF evidence, not a live feature source. |
| Research OOF predictions `ce701451` | 1965-01-01..2023-12-31 | 21,549 | 21,313 | 236 | 98.905% | 63,939 | Historical OOF evidence, not a live feature source. |

## Cataloged audit datasets

| dataset_id | DB inclusion | recommended layer | operational value | diagnostic value | future potential | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| [root] | YES-catalog/provenance only; archive bytes outside DB | catalog / acquisition_provenance / object_catalog | 0 | 45 | 50 | Manifests support reproducibility; ZIP bytes and local paths are not model inputs. |
| 01_hko_daily_tmax_target | YES-canonical labels plus audit raw | label_core / raw_audit | 95 | 100 | 100 | Use canonical labels; derive only prior-day/older target-memory features. Never use target T. |
| 02_hko_daily_climate_all_elements | YES-diagnostic schema | diagnostic_physics | 0 | 90 | 75 | Store all 21 variables, but do not promote as operational predictors until publication timing is proven. |
| 03_noaa_igra_upper_air_hkm00045004 | YES-raw quarantine + rebuilt clean diagnostic | diagnostic_physics | 0 | 95 | 85 | Very valuable mechanism data; current sentinels/scales and release timing block deployment. |
| 04_noaa_isd_regional_surface | YES-raw quarantine + rebuilt station schema | diagnostic_station_network | 0 | 98 | 92 | Highest-value non-forecast research source, but wind parser, coordinates and operational-vintage contract must be repaired. |
| 05_hko_historical_rss_forecasts | YES-highest priority canonical operational source | operational_archive / anchor / research_supervised | 100 | 100 | 100 | Primary official anchor and text/residual source. Quarantine invalid press rows and load Parquet once. |
| 06_hko_tropical_cyclone_best_track | YES-diagnostic only | diagnostic_regime_labels | 0 | 75 | 65 | Useful for TC mechanism labels and proxy discovery; retrospective best track is forbidden live. |
| 07_hko_radar_satellite_lightning_nowcast | YES-live exact-vintage collection/object catalog | live_exact_vintage / live_object_catalog | 5 | 45 | 92 | Potentially valuable cloud/rain/convection layer, but present history is only days and imagery is not feature-extracted. |
| 08_hko_marine_tide_coastal_waters | YES-live exact-vintage collection | live_exact_vintage | 5 | 40 | 60 | Secondary marine-regime source; current sample is tiny and tide likely indirect. |
| 09_hko_arwf_station_forecasts | YES-critical live collection | live_nwp_anchor | 5 | 35 | 98 | Potentially one of the best future independent anchors; only one cycle now, timestamps/station mapping need normalization. |
| 10_ncep_operational_grib_inventory | YES-object metadata only until decoded | nwp_object_catalog | 0 | 15 | 100 | Very high future potential; current table is only an incomplete inventory with empty cycle fields. |
| 11_static_geospatial_inventory | YES-object metadata, then derived feature dimension | static_object_catalog / station_context | 20 | 70 | 85 | Terrain/coast/exposure features are valuable; inventory rows themselves are not predictors. |
| 12_hkg_t24_robust_experiment_outputs | YES-research registry/artifact store only | research_artifacts / research_metrics / research_oof | 0 | 100 | 80 | Essential institutional memory and OOF evidence; never use as canonical live feature source. |

## Catalog attribute coverage by dataset

| dataset_id | audited attributes | operational-ish attributes | label/target-ish attributes |
| --- | --- | --- | --- |
| [root] | 19 | 0 | 0 |
| 01_hko_daily_tmax_target | 22 | 0 | 2 |
| 02_hko_daily_climate_all_elements | 17 | 0 | 0 |
| 03_noaa_igra_upper_air_hkm00045004 | 86 | 0 | 0 |
| 04_noaa_isd_regional_surface | 34 | 0 | 0 |
| 05_hko_historical_rss_forecasts | 306 | 0 | 10 |
| 06_hko_tropical_cyclone_best_track | 16 | 0 | 0 |
| 07_hko_radar_satellite_lightning_nowcast | 40 | 0 | 0 |
| 08_hko_marine_tide_coastal_waters | 18 | 0 | 0 |
| 09_hko_arwf_station_forecasts | 14 | 0 | 0 |
| 10_ncep_operational_grib_inventory | 12 | 0 | 0 |
| 11_static_geospatial_inventory | 11 | 0 | 0 |
| 12_hkg_t24_robust_experiment_outputs | 1,274 | 0 | 45 |

## HKO target labels and long history

| Object | Rows | Date range | Stations/source count | Tmax range C | Notes |
| --- | --- | --- | --- | --- | --- |
| label_core.hko_daily_tmax | 48,577 | 1884-01-01..2023-12-31 | 1 | 3.20..36.60 | Canonical pre-2024 labels. Primary label table and safe target-memory source with T-2-or-older lag discipline. |
| sealed_confirmation.hko_daily_tmax | 882 | 2024-01-01..2026-05-31 | 1 | 10.40..35.70 | 2024+ confirmation labels; must stay sealed from model selection/training until formal holdout opening. |
| raw_audit.codex_audit_ds_01_* | 49,628 | 1884-01-01..2026-06-17 | source_ids=2; availability_tiers=1 |  | Audit raw table backing target history. |
| feature_safe.hko_target_history_pre2024 | 48,577 | pre-2024 |  |  | Feature-safe view over pre-2024 target history. |

## HKO official forecast archive

| Metric | Value |
| --- | --- |
| public.hko_historical_forecasts_2000_2026 total rows | 324,179 |
| local product rows | 264,325 |
| usable_local_minmax rows | 115,795 |
| usable distinct target dates | 9,667 |
| usable issue UTC range | 2000-01-01 08:22:00..2026-06-20 15:45:00 |
| usable target-date range | 2000-01-02..2026-06-21 |
| usable local lead-1 rows | 88,504 |

Current anchor/view population:

| Object | Rows |
| --- | --- |
| operational_anchor.hko_t24_official_anchor_rows | 0 |
| feature_safe.hko_t24_official_anchor | 0 |
| feature_safe.hko_target_history_pre2024 | 48,577 |

Product/quality breakdown:

| product_type | row_quality_status | rows | target_date range |
| --- | --- | --- | --- |
| 5day | bulletin_only_multiday_product | 6,193 | None..None |
| 7day | bulletin_only_multiday_product | 23,223 | None..None |
| 9day | bulletin_only_multiday_product | 30,438 | None..None |
| local | usable_local_minmax | 115,795 | 2000-01-02..2026-06-21 |
| local | missing_target_date | 60,940 | None..None |
| local | usable_local_tmax_only | 58,199 | 2000-01-01..2026-06-20 |
| local | missing_forecast_max | 29,383 | 2000-01-01..2026-06-20 |
| local | invalid_target_lead | 8 | 1990-11-03..2005-02-25 |

## GribStream tactical H24N PostgreSQL corpus

Raw response scopes:

| source_scope | raw objects | raw row-count sum | raw bytes | retrieved UTC range |
| --- | --- | --- | --- | --- |
| batch_smoke_10w | 95 | 123,652 | 2.98 MB | 2026-06-25 07:03:05.980412..2026-06-25 07:33:36.296761 |
| first_week | 13 | 11,796 | 291.86 KB | 2026-06-25 05:26:03.954344..2026-06-25 05:28:56.366353 |
| full_tactical_backfill_ok_tmax | 1,163 | 1,964,157 | 53.87 MB | 2026-06-25 05:29:06.383119..2026-06-25 17:31:03.998675 |
| smoke | 14 | 1,728 | 49.66 KB | 2026-06-24 23:47:05.150292..2026-06-25 00:15:28.450302 |

Rows currently in `nwp_tactical.forecast_wide` by source scope:

| source_scope | dataset_code | rows |
| --- | --- | --- |
| batch_smoke_10w | gefsatmos | 933 |
| full_tactical_backfill_ok_tmax | aifsenfo | 72,270 |
| full_tactical_backfill_ok_tmax | aifsoper | 28,884 |
| full_tactical_backfill_ok_tmax | aigefssfc | 46,252 |
| full_tactical_backfill_ok_tmax | aigfspres | 3,660 |
| full_tactical_backfill_ok_tmax | aigfssfc | 3,660 |
| full_tactical_backfill_ok_tmax | cwawrf15 | 180 |
| full_tactical_backfill_ok_tmax | fourcastnetgfs | 37,824 |
| full_tactical_backfill_ok_tmax | gefsatmos | 516,891 |
| full_tactical_backfill_ok_tmax | gefsatmosmean | 200,436 |
| full_tactical_backfill_ok_tmax | gfs | 575,004 |
| full_tactical_backfill_ok_tmax | graphcast | 44,220 |
| full_tactical_backfill_ok_tmax | ifsenfo | 343,616 |
| full_tactical_backfill_ok_tmax | ifsoper | 91,260 |

Full tactical backfill model inventory from live DB (`object_uri LIKE '%full_tactical_backfill_ok_tmax%'`):

| dataset | rows | target HKT | run UTC | valid UTC | runs | target days | locations | members | lead hours | safe rows 6h | unsafe rows 6h | Tmax candidate rows/days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aifsenfo | 72,270 | 2025-07-04..2026-06-23 | 2025-07-02 18:00:00..2026-06-21 18:00:00 | 2025-07-03 18:00:00..2026-06-23 12:00:00 | 355 | 355 | 1 | 51 | 24.00..42.00 | 72,270 | 0 | 72,270/355 |
| aifsoper | 28,884 | 2025-02-26..2026-06-23 | 2025-02-25 18:00:00..2026-06-21 18:00:00 | 2025-02-26 12:00:00..2026-06-23 12:00:00 | 482 | 483 | 12 | 1 | 18.00..42.00 | 23,100 | 5,784 | 28,884/483 |
| aigefssfc | 46,252 | 2025-06-03..2026-06-23 | 2025-06-01 18:00:00..2026-06-21 18:00:00 | 2025-06-02 18:00:00..2026-06-23 12:00:00 | 373 | 373 | 1 | 31 | 24.00..42.00 | 46,252 | 0 | 8,308/67 |
| aigfspres | 3,660 | 2026-04-22..2026-06-23 | 2026-04-21 18:00:00..2026-06-21 18:00:00 | 2026-04-22 12:00:00..2026-06-23 12:00:00 | 61 | 63 | 12 | 1 | 18.00..42.00 | 2,928 | 732 | 0/0 |
| aigfssfc | 3,660 | 2026-04-22..2026-06-23 | 2026-04-21 18:00:00..2026-06-21 18:00:00 | 2026-04-22 12:00:00..2026-06-23 12:00:00 | 61 | 63 | 12 | 1 | 18.00..42.00 | 2,928 | 732 | 3,660/63 |
| cwawrf15 | 180 | 2026-06-23..2026-06-26 | 2026-06-22 18:00:00..2026-06-24 18:00:00 | 2026-06-23 12:00:00..2026-06-26 12:00:00 | 3 | 4 | 12 | 1 | 18.00..42.00 | 144 | 36 | 180/4 |
| fourcastnetgfs | 37,824 | 2024-05-03..2026-02-20 | 2024-05-02 18:00:00..2026-02-18 18:00:00 | 2024-05-03 12:00:00..2026-02-20 12:00:00 | 631 | 648 | 12 | 1 | 18.00..42.00 | 30,252 | 7,572 | 37,824/648 |
| gefsatmos | 516,891 | 2020-10-03..2026-06-23 | 2020-10-01 18:00:00..2026-06-21 18:00:00 | 2020-10-02 18:00:00..2026-06-23 15:00:00 | 2,085 | 2,085 | 1 | 31 | 24.00..45.00 | 516,891 | 0 | 516,891/2,085 |
| gefsatmosmean | 200,436 | 2020-10-03..2026-06-23 | 2020-10-01 18:00:00..2026-06-21 18:00:00 | 2020-10-02 18:00:00..2026-06-23 15:00:00 | 2,088 | 2,088 | 12 | 1 | 24.00..45.00 | 200,436 | 0 | 200,436/2,088 |
| gfs | 575,004 | 2021-03-23..2026-06-23 | 2021-03-23 00:00:00..2026-06-22 00:00:00 | 2021-03-23 15:00:00..2026-06-23 15:00:00 | 1,918 | 1,919 | 12 | 1 | 15.00..39.00 | 552,000 | 23,004 | 574,812/1,919 |
| graphcast | 44,220 | 2024-04-26..2026-05-06 | 2024-04-25 18:00:00..2026-05-04 18:00:00 | 2024-04-26 12:00:00..2026-05-06 12:00:00 | 737 | 741 | 12 | 1 | 18.00..42.00 | 35,376 | 8,844 | 44,220/741 |
| ifsenfo | 343,616 | 2024-03-03..2026-06-23 | 2024-03-01 18:00:00..2026-06-21 18:00:00 | 2024-03-02 18:00:00..2026-06-23 15:00:00 | 843 | 843 | 1 | 51 | 24.00..45.00 | 343,616 | 0 | 343,616/843 |
| ifsoper | 91,260 | 2024-02-29..2026-06-23 | 2024-02-28 18:00:00..2026-06-21 18:00:00 | 2024-02-29 15:00:00..2026-06-23 15:00:00 | 845 | 846 | 12 | 1 | 21.00..45.00 | 81,120 | 10,140 | 91,260/846 |

Weather-variable non-null counts by full-scope model:

| dataset | temperature 2m k | interval tmax 2m k | dewpoint 2m k | relative humidity 2m pct | u wind 10m mps | v wind 10m mps | mslp pa | low cloud pct | accumulated precip kg m2 | downward shortwave w m2 | net shortwave w m2 | total precip m | shortwave down j m2 | total column water vapour kg m2 | pwat kg m2 | temperature 925 k | temperature 850 k | relative humidity 700 pct | geopotential height 500 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aifsenfo | 72,270 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| aifsoper | 28,884 | 0 | 28,884 | 0 | 28,884 | 28,884 | 28,884 | 0 | 0 | 0 | 0 | 28,884 | 28,884 | 0 | 0 | 0 | 28,884 | 0 | 0 |
| aigefssfc | 8,308 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| aigfspres | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,660 | 0 | 3,660 |
| aigfssfc | 3,660 | 0 | 0 | 0 | 3,660 | 3,660 | 3,660 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cwawrf15 | 180 | 0 | 180 | 0 | 180 | 180 | 180 | 0 | 180 | 0 | 180 | 0 | 0 | 0 | 0 | 0 | 180 | 180 | 180 |
| fourcastnetgfs | 37,824 | 0 | 0 | 0 | 37,824 | 37,824 | 37,824 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 37,824 | 0 | 37,824 |
| gefsatmos | 0 | 516,891 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gefsatmosmean | 200,436 | 200,436 | 200,436 | 200,436 | 200,436 | 200,436 | 200,436 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200,436 | 0 | 0 | 0 | 0 |
| gfs | 574,812 | 574,812 | 574,812 | 0 | 574,812 | 574,812 | 574,980 | 574,812 | 574,812 | 574,812 | 0 | 0 | 0 | 0 | 0 | 574,812 | 574,812 | 574,812 | 574,812 |
| graphcast | 44,220 | 0 | 0 | 0 | 44,220 | 44,220 | 44,220 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 44,220 | 0 | 44,220 |
| ifsenfo | 343,616 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ifsoper | 91,260 | 0 | 90,612 | 0 | 91,260 | 91,260 | 91,260 | 0 | 0 | 0 | 0 | 91,260 | 90,612 | 91,260 | 0 | 91,260 | 91,260 | 91,260 | 91,260 |

Model plan:

| dataset_code | family | provider | role | status | notes |
| --- | --- | --- | --- | --- | --- |
| aifsenfo |  |  |  |  | AI ensemble challenger |
| aifsoper |  |  |  |  | Deterministic AI challenger |
| aigefssfc |  |  |  |  | NOAA AI ensemble surface challenger |
| aigfspres |  |  |  |  | Selected NOAA AI pressure-level context |
| aigfssfc |  |  |  |  | NOAA AI deterministic surface challenger |
| cwawrf15 |  |  |  |  | Regional East Asia expert; rolling last-three-day historical window only |
| fourcastnetgfs |  |  |  |  | Historical AI-versus-physics expert |
| gefsatmos |  |  |  |  | GEFS full-member HKO Tmax distribution |
| gefsatmosmean |  |  |  |  | Low-volume ensemble mean context |
| gfs |  |  |  |  | Main deterministic NWP trajectory and HKO MOS expert |
| graphcast |  |  |  |  | Historical AI-versus-physics expert |
| ifsenfo |  |  |  |  | Independent ECMWF ensemble distribution |
| ifsoper |  |  |  |  | Independent ECMWF deterministic expert |
| nbmoc |  |  |  |  | Tiny marine coverage probe only; no full backfill authorized |

Variable selectors planned per model:

| dataset_code | selector count | selectors |
| --- | --- | --- |

HKG location stencil used by tactical pull:

| location_code | location_name | lat | lon | role |
| --- | --- | --- | --- | --- |

Mandatory use rule: production/modeling feature extraction must join `forecast_wide` to `raw_response_object`, filter to `full_tactical_backfill_ok_tmax`, apply the 6-hour H24N safety predicate, and exclude `nbmoc`, `aigfspres`, and `aigefssfc` as daily Tmax sources under the current docs.

## HKO daily climate elements

| variable | unit | rows | local_date range | station/domain count | operational_allowed_rows |
| --- | --- | --- | --- | --- | --- |
| bright_sunshine_duration | hours | 23,892 | 1961-01-01..2026-05-31 | 1 | 0 |
| cloud_to_cloud_lightning | count | 7,650 | 2005-06-21..2026-05-31 | 1 | 0 |
| cloud_to_ground_lightning | count | 7,650 | 2005-06-21..2026-05-31 | 1 | 0 |
| daily_maximum_temperature | degC | 49,460 | 1884-01-01..2026-05-31 | 1 | 0 |
| daily_minimum_temperature | degC | 49,460 | 1884-01-01..2026-05-31 | 1 | 0 |
| daily_rainfall | mm | 49,400 | 1884-03-01..2026-05-31 | 1 | 0 |
| evaporation | mm | 21,336 | 1968-01-01..2026-05-31 | 1 | 0 |
| global_solar_radiation | MJ/m2 | 17,683 | 1978-01-01..2026-05-31 | 1 | 0 |
| grass_minimum_temperature | degC | 21,336 | 1968-01-01..2026-05-31 | 1 | 0 |
| mean_cloud_amount | percent | 28,275 | 1949-01-01..2026-05-31 | 1 | 0 |
| mean_dew_point_temperature | degC | 23,892 | 1961-01-01..2026-05-31 | 1 | 0 |
| mean_relative_humidity | percent | 29,006 | 1947-01-01..2026-05-31 | 1 | 0 |
| mean_sea_level_pressure | hPa | 49,400 | 1884-03-01..2026-05-31 | 1 | 0 |
| mean_temperature | degC | 49,400 | 1884-03-01..2026-05-31 | 1 | 0 |
| mean_wet_bulb_temperature | degC | 29,006 | 1947-01-01..2026-05-31 | 1 | 0 |
| mean_wind_speed | km/h | 18,779 | 1975-01-01..2026-05-31 | 1 | 0 |
| prevailing_wind_direction | degree_or_compass | 18,779 | 1975-01-01..2026-05-31 | 1 | 0 |
| reduced_visibility_hours | hours | 10,743 | 1997-01-01..2026-05-31 | 1 | 0 |
| sea_temperature | degC | 13,300 | 1990-01-01..2026-05-31 | 1 | 0 |
| sea_temperature_am | degC | 18,976 | 1974-06-18..2026-05-31 | 1 | 0 |
| sea_temperature_pm | degC | 18,976 | 1974-06-18..2026-05-31 | 1 | 0 |

## NOAA ISD regional station summaries

| object | rows | stations | observed UTC range | operational_allowed_rows |
| --- | --- | --- | --- | --- |
| diagnostic_station_network.codex_audit_ds_04_noaa_isd_regional_surface_noaa_isd_c_688279e2 | 4,029,291 | 36 | 1945-11-30T16:00:00+00:00..2025-08-24T21:30:00+00:00 | 0 |

## Tropical cyclone best-track diagnostic labels

| object | rows | time range | storms | source IDs | strategy status |
| --- | --- | --- | --- | --- | --- |
| diagnostic_regime_labels.codex_audit_ds_06_* | 26,189 | 1985-01-06T06:00:00Z..2024-12-25T06:00:00Z | 41 | 1 | Retrospective diagnostic labels only; denied as live model input unless operational publication timing is proven. |

## Live/prospective exact-vintage and ARWF tables

| object | rows | size | time ranges | top categories |
| --- | --- | --- | --- | --- |
| live_exact_vintage.catalog | 0 | 16.00 KB |  | dataset_id: ; source_id: ; eligibility_status: |
| live_exact_vintage.codex_audit_ds_07_hko_radar_satellite_lightning_nowcas_97d078a5 | 41 | 72.00 KB | ingested_at_utc=2026-06-23 15:02:41.414344+00..2026-06-23 15:02:41.414344+00 (non-null 41); raw_retrieved_at_utc=2026-06-19T04:50:52.482529Z..2026-06-20T07:56:22.144891Z (non-null 41) | source_id: hko_gridded_rainfall_nowcast=41; availability_tier: GOLD_EXACT_VINTAGE=41; operational_input_allowed: true=41 |
| live_exact_vintage.codex_audit_ds_07_hko_radar_satellite_lightning_nowcas_eb458062 | 102 | 88.00 KB | ingested_at_utc=2026-06-23 15:02:42.013512+00..2026-06-23 15:02:42.013512+00 (non-null 102); raw_retrieved_at_utc=2026-06-19T04:57:47.135172Z..2026-06-20T08:03:02.341258Z (non-null 102) | source_id: hko_lightning_counts_latest=102; availability_tier: GOLD_EXACT_VINTAGE=102; operational_input_allowed: true=102 |
| live_exact_vintage.codex_audit_ds_08_hko_marine_tide_coastal_waters_hko_l_be7bc027 | 105 | 88.00 KB | ingested_at_utc=2026-06-23 15:02:43.33808+00..2026-06-23 15:02:43.33808+00 (non-null 105); raw_retrieved_at_utc=2026-06-19T04:50:57.280298Z..2026-06-20T07:56:27.102158Z (non-null 105); observed_at_hkt=2026-06-19T12:45:00+08:00..2026-06-20T15:50:00+08:00 (non-null 105) | source_id: hko_latest_tidal_information=105; availability_tier: GOLD_EXACT_VINTAGE=105; operational_input_allowed: true=105 |
| live_exact_vintage.codex_audit_ds_08_hko_marine_tide_coastal_waters_hko_s_4bce52ff | 30 | 64.00 KB | ingested_at_utc=2026-06-23 15:02:43.981515+00..2026-06-23 15:02:43.981515+00 (non-null 30); raw_retrieved_at_utc=2026-06-19T04:50:55.669615Z..2026-06-20T07:56:24.707400Z (non-null 30) | source_id: hko_south_china_coastal_waters_bulletin=30; availability_tier: GOLD_EXACT_VINTAGE=30; operational_input_allowed: true=30 |
| live_nwp_anchor.codex_audit_ds_09_hko_arwf_station_forecasts_hko_arwf__e2cc97ac | 530 | 352.00 KB | ingested_at_utc=2026-06-23 15:02:44.773109+00..2026-06-23 15:02:44.773109+00 (non-null 530); raw_retrieved_at_utc=2026-06-19T06:12:34.954063Z..2026-06-19T06:14:15.039351Z (non-null 530); model_time=2026061812..2026061812 (non-null 530); last_modified=20260619131207..20260619134017 (non-null 530); forecast_date=20260619..20260628 (non-null 530) | source_id: hko_arwf_station_forecast=530; station_code: CCH=10; CWB=10; G119=10; G120=10; G121=10; G135=10; availability_tier: GOLD_EXACT_VINTAGE=530; operational_input_allowed: true=530 |

## Object/catalog assets

| asset_id | asset_role | media_type | asset_uri | bytes | content_sha256 | registered_at_utc |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | AUDIT_BUNDLE | application/zip | repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT_BUNDLE.zip | 144.44 KB | bdbc1fce90c03ce74ee20b864691467bf0dd9a00be996a42119618ababb3fb27 | 2026-06-23 15:27:32.075687+02:00 |
| 2 | AUDIT_CONTRACT_FILE | text/markdown | repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_DATASET_DB_AND_MODEL_VALUE_AUDIT.md | 895.53 KB | c5a99d1ac75cb2002d2f4a0320b0ffcedc3bf00c42a77c03e4ef08e3772191df | 2026-06-23 15:27:32.075687+02:00 |
| 3 | AUDIT_CONTRACT_FILE | text/csv | repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_DATASET_DECISION_MATRIX.csv | 2.86 KB | 23ec92db4e65f408d262181e572c565e20fdb771078edcdfa50daf0aaba456e9 | 2026-06-23 15:27:32.075687+02:00 |
| 4 | AUDIT_CONTRACT_FILE | text/csv | repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_TABLE_DECISIONS_ALL_52.csv | 15.80 KB | 2ff119344613cd3f088bbc749bf78e3b0c1972162233047ce1833cf794e331ed | 2026-06-23 15:27:32.075687+02:00 |
| 5 | AUDIT_CONTRACT_FILE | text/csv | repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_ATTRIBUTE_DECISIONS_ALL_1869.csv | 760.29 KB | bdd5904bac4af25d022596c52aad202b3c8063b0fdf5584013ab545c325719fa | 2026-06-23 15:27:32.075687+02:00 |
| 6 | AUDIT_CONTRACT_FILE | text/csv | repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_DATA_QUALITY_ISSUES.csv | 5.82 KB | 60916256dc73a40fbe639b24cd91ac210a0c21c09f96856dca4c91fb19275d9c | 2026-06-23 15:27:32.075687+02:00 |
| 7 | AUDIT_CONTRACT_FILE | text/csv | repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_ISD_STATION_DOSSIER_36.csv | 6.72 KB | 930023835b19cf595f5b56f8d52fa454ed70d5a71bd3d96b73a05acbdca3e0d5 | 2026-06-23 15:27:32.075687+02:00 |
| 8 | AUDIT_CONTRACT_FILE | application/sql | repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_DB_SCHEMA_BLUEPRINT.sql | 11.94 KB | 5129f3d2f9a378b0bc044101365c001c4aaaa6c440dbd5acd9acb9bcf9b4032f | 2026-06-23 15:27:32.075687+02:00 |
| 9 | AUDIT_CONTRACT_FILE | text/markdown | repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/README.md | 1.07 KB | a5703242f5066e5f899cb5626374f6f2da4db57ad92d5da5e842dcf2d691162e | 2026-06-23 15:27:32.075687+02:00 |
| 10 | REGISTER_OBJECT_METADATA | application/vnd.apache.parquet | repo://data/datasets/07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | 6.65 KB | ab499242c7810270481016ca532fa2811deea1e811840806332cb68f989cc71f | 2026-06-23 15:27:32.075687+02:00 |
| 11 | REGISTER_OBJECT_METADATA | application/vnd.apache.parquet | repo://data/datasets/07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | 690.68 KB | 3a4466b3f5d1c64cddbaba168bd045f84fe34acaf9ca60199a429fb045dc82c6 | 2026-06-23 15:27:32.075687+02:00 |
| 12 | REGISTER_OBJECT_METADATA_AND_REBUILD_INVENTORY | application/vnd.apache.parquet | repo://data/datasets/10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | 504.01 KB | 505602502b8901786a4daecbc9c5e38e68af18e54e5af422596598297817c3c0 | 2026-06-23 15:27:32.075687+02:00 |
| 13 | REGISTER_OBJECT_METADATA | application/vnd.apache.parquet | repo://data/datasets/11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | 20.33 KB | f970591d52de2e39ad17d7717978013447723c84e61b417f36da8583f13ad60a | 2026-06-23 15:27:32.075687+02:00 |
| 14 | REGISTER_RESEARCH_ARTIFACT | application/vnd.apache.parquet | repo://data/datasets/12_hkg_t24_robust_experiment_outputs/hkg_t24_exp0050_0099_feature_matrix.parquet | 46.32 MB | 77ba7af1227aa47cbc2d075c19ff0a8944f14bb34ef6f73870b5fe373a240480 | 2026-06-23 15:27:32.075687+02:00 |
| 15 | REGISTER_RESEARCH_ARTIFACT | application/vnd.apache.parquet | repo://data/datasets/12_hkg_t24_robust_experiment_outputs/hkg_t24_r14_feature_matrix.parquet | 4.23 MB | b4c776986d31bb640b262809044391b2e34029f0ca3be508603072538f3c3c8f | 2026-06-23 15:27:32.075687+02:00 |
| 16 | REGISTER_RESEARCH_ARTIFACT | application/vnd.apache.parquet | repo://data/datasets/12_hkg_t24_robust_experiment_outputs/hkg_t24_r15_feature_matrix.parquet | 4.08 MB | be985aa670f9485e379d40f113ded59193ef3ff4f4fb8aeed9c9e03d5758b05b | 2026-06-23 15:27:32.075687+02:00 |
| 17 | REGISTER_RESEARCH_ARTIFACT | application/vnd.apache.parquet | repo://data/datasets/12_hkg_t24_robust_experiment_outputs/hkg_t24_r16_feature_matrix.parquet | 4.16 MB | 1d05df97908aaf42ee14c76a53326956011f42b061d1f77131149b3f21bf09ba | 2026-06-23 15:27:32.075687+02:00 |
| 18 | REGISTER_RESEARCH_ARTIFACT | application/vnd.apache.parquet | repo://data/datasets/12_hkg_t24_robust_experiment_outputs/hkg_t24_r17_feature_matrix.parquet | 4.08 MB | be985aa670f9485e379d40f113ded59193ef3ff4f4fb8aeed9c9e03d5758b05b | 2026-06-23 15:27:32.075687+02:00 |
| 19 | REGISTER_OBJECT_ONLY | application/zip | repo://data/datasets/hko_forecast_rss_archives_20200601_20260619.zip | 181.94 MB | 0f6ee8910900e2c372038dc14b4606b6c9a12f934301bfe1b637a34b1fbe62d8 | 2026-06-23 15:27:32.075687+02:00 |

## Static station/geospatial dimension

| station_id | station_name | country | icao | lat | lon | elev_m | distance_to_hko_km | tier | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 450010-99999 | CHEUNG CHAU | CH | VHCH | 22.2 | 114.017 | 79.0 | 19.025077815957136 | B | Offshore southwest/island historical |
| 450030-99999 | TAI-O | CH |  | 22.25 | 113.85 | 76.0 | 33.08878278226951 | B | Western Lantau/coastal historical |
| 450040-99999 | KOWLOON | CH |  | 22.312 | 114.173 | 66.4 | 1.4701879397747686 | B | Very-near urban/Kowloon |
| 450050-99999 | HONG KONG OBSERVATORY | CH |  | 22.3 | 114.167 | 62.0 | 0.0 | A | Target/local urban reference |
| 450060-99999 | GREEN ISLAND | CH |  | 22.283 | 114.117 | 76.0 | 5.480559692932972 | D | West-harbour/island |
| 450070-99999 | HONG KONG INTL | HK | VHHH | 22.309 | 113.915 | 8.5 | 25.943884354778422 | A | Airport/open west coast |
| 450090-99999 | TAI MO SHAN | CH |  | 22.417 | 114.117 | 947.0 | 13.989027676872668 | C | High mountain |
| 450100-99999 | TATE'S CAIRN | CH |  | 22.367 | 114.217 | 576.0 | 9.052663835371938 | D | Elevated hill/synoptic |
| 450110-99999 | MACAU INTL | MC | VMMC | 22.15 | 113.592 | 6.1 | 61.49212169350101 | A | Macau airport southwest marine |
| 450200-99999 | MACAO/FORTALEZA DO MONTE | MC |  | 22.183 | 113.533 | 59.0 | 66.53655364742232 | D | Macau urban hill |
| 450320-99999 | TA KWU LING | CH |  | 22.533 | 114.15 | 13.0 | 25.96728286720018 | A | Far north inland |
| 450330-99999 | KAT O | CH |  | 22.533 | 114.3 | 10.0 | 29.294255504664683 | C | Northeast island/coastal historical |
| 450340-99999 | TAI MEI TUK | CH |  | 22.483 | 114.233 | 53.0 | 21.450217709579704 | A | Northeast coastal/reservoir |
| 450350-99999 | LAU FAU SHAN | CH |  | 22.467 | 113.983 | 35.0 | 26.509077362333528 | A | Northwest coast/estuary |
| 450390-99999 | SHA TIN | CH |  | 22.4 | 114.2 | 8.0 | 11.625869594463575 | A | Inland/valley New Territories |
| 450410-99999 | TSEUNG KAWN O | CH |  | 22.317 | 114.25 | 32.0 | 8.745151058922215 | D | East urban/coastal |
| 450440-99999 | CHEUNG CHAU | CH |  | 22.2 | 114.017 | 79.0 | 19.025077815957136 | A | Offshore southwest/island |
| 450450-99999 | WAGLAN ISLAND | CH |  | 22.183 | 114.3 | 60.0 | 18.884707396387636 | A | Offshore east/marine |
| 590750-99999 | YANGSHAN | CH |  | 24.483 | 112.633 | 64.0 | 288.8362099309494 | D | Yangshan far north historical |
| 590870-99999 | FOGANG | CH |  | 23.883 | 113.517 | 68.0 | 188.15823624403694 | A | Fogang north inland |
| 590960-99999 | LIANPING | CH |  | 24.367 | 114.483 | 214.0 | 232.09301269547203 | A | Lianping far north |
| 592710-99999 | HUAIJI | CH |  | 23.95 | 112.2 | 57.0 | 272.24387334293573 | B | Huaiji northwest historical |
| 592730-99999 | QINGCHENG | CH |  | 23.667 | 112.867 | 20.0 | 202.02350518892865 | D | Qingcheng northwest historical |
| 592780-99999 | GAOYAO | CH |  | 23.05 | 112.467 | 12.0 | 193.32984128514695 | A | Gaoyao west inland |
| 592800-99999 | QING YUAN | CH |  | 23.7 | 113.083 | 19.0 | 191.1640460791295 | C | Qingyuan north-west historical |
| 592870-99999 | BAIYUN INTL | CH | ZGGG | 23.392 | 113.299 | 15.2 | 150.51552993657924 | A | Guangzhou/Baiyun inland metropolis |
| 592930-99999 | HEYUAN | CH |  | 23.8 | 114.733 | 41.0 | 176.55924833292787 | A | Heyuan north-northeast inland |
| 592980-99999 | HUI-YANG | CH |  | 23.083 | 114.417 | 16.0 | 90.76439458444014 | B | Huizhou/Huiyang northeast historical |
| 593030-99999 | SONG-LIN-BA | CH |  | 23.967 | 115.967 | 152.0 | 261.2124555640404 | B | Songlinba far northeast historical |
| 593090-99999 | DAPING | CH |  | 23.183 | 115.85 | 97.0 | 198.5629245338472 | D | Daping east-northeast historical |
| 594780-99999 | TAI-SHAN | CH |  | 22.267 | 112.783 | 46.0 | 142.4475995044708 | B | Taishan west mainland historical |
| 594880-99999 | ZHUHAI SANZAO | CH |  | 22.017 | 113.383 | 3.0 | 86.65381078305995 | D | Zhuhai/Sanzao southwest |
| 594930-99999 | BAOAN INTL | CH | ZGSZ | 22.639 | 113.811 | 4.0 | 52.52640408904244 | A | Shenzhen/Baoan airport mainland |
| 595010-99999 | SHANWEI | CH |  | 22.783 | 115.367 | 5.0 | 134.43324977443731 | A | Shanwei east coast |
| 595050-99999 | APPROXIMATE LOCALE | CH |  | 22.5 | 115.5 | 0.0 | 138.83106030771398 | C | Approximate east-coast locale |
| 596730-99999 | SHANGCHUAN DAO | CH |  | 21.733 | 112.767 | 18.0 | 157.48976465338188 | A | Shangchuan Dao southwest marine |

## All source files/tables from catalog.source_file_registry

| dataset_id | source_file | type | source rows | attributes | data range | ingestion action | DB layer | model status | priority | status | original local path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [root] | hko_forecast_rss_archives_20200601_20260619.zip | zip | 0 | 0 | None..None | REGISTER_OBJECT_ONLY | object_catalog | NOT_DIRECT | MEDIUM | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\hko_forecast_rss_archives_20200601_20260619.zip |
| [root] | hko_forecast_rss_archives_20200601_20260619_manifest.csv | csv | 522 | 13 | None..None | LOAD_METADATA | acquisition_provenance | NOT_DIRECT | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\hko_forecast_rss_archives_20200601_20260619_manifest.csv |
| [root] | MANIFEST.csv | csv | 39 | 6 | None..None | LOAD_METADATA | catalog | NOT_DIRECT | MEDIUM | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\MANIFEST.csv |
| 01_hko_daily_tmax_target | 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet | parquet | 49,628 | 14 | 1884-01-01 00:00:00+00:00..2026-06-17 00:00:00+00:00 | LOAD_PROVENANCE | raw_audit | LABEL_AUDIT_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\01_hko_daily_tmax_target\hko_daily_extract_tmax_payload_rows.parquet |
| 01_hko_daily_tmax_target | 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet | parquet | 49,459 | 8 | 1884-01-01 00:00:00+00:00..2026-05-31 00:00:00+00:00 | LOAD_CANONICAL | label_core | LABEL_ONLY | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\01_hko_daily_tmax_target\hko_daily_tmax_target_labels.parquet |
| 02_hko_daily_climate_all_elements | 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet | parquet | 556,399 | 17 | 1884-01-01 00:00:00+00:00..2026-05-31 00:00:00+00:00 | LOAD_DIAGNOSTIC | diagnostic_physics | BLOCKED_UNTIL_PUBLICATION_PROOF | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\02_hko_daily_climate_all_elements\hko_daily_climate_elements.parquet |
| 03_noaa_igra_upper_air_hkm00045004 | 03_noaa_igra_upper_air_hkm00045004/noaa_igra_hkm00045004_key_pressure_levels.parquet | parquet | 477,514 | 28 | 1949-06-01 22:00:00+00:00..2026-06-17 18:00:00+00:00 | LOAD_RAW_QUARANTINE_AND_REBUILD_CLEAN | diagnostic_physics | DIAGNOSTIC_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\03_noaa_igra_upper_air_hkm00045004\noaa_igra_hkm00045004_key_pressure_levels.parquet |
| 03_noaa_igra_upper_air_hkm00045004 | 03_noaa_igra_upper_air_hkm00045004/noaa_igra_hkm00045004_sounding_features.parquet | parquet | 88,407 | 58 | 1949-06-01 22:00:00+00:00..2026-06-17 18:00:00+00:00 | RECOMPUTE_BEFORE_CANONICAL_LOAD | diagnostic_physics | DIAGNOSTIC_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\03_noaa_igra_upper_air_hkm00045004\noaa_igra_hkm00045004_sounding_features.parquet |
| 04_noaa_isd_regional_surface | 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet | parquet | 4,029,291 | 21 | 1945-11-30 16:00:00+00:00..2025-08-24 21:30:00+00:00 | LOAD_RAW_QUARANTINE_AND_REBUILD_CLEAN | diagnostic_station_network | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\04_noaa_isd_regional_surface\noaa_isd_core_observations.parquet |
| 04_noaa_isd_regional_surface | 04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet | parquet | 317,489 | 13 | 1945-12-01 00:00:00+00:00..2025-08-25 00:00:00+00:00 | REBUILD_AFTER_RAW_FIX | diagnostic_station_network | DIAGNOSTIC_ONLY_UNTIL_VINTAGE_PROOF | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\04_noaa_isd_regional_surface\noaa_isd_station_day_cutoff_summary.parquet |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_historical_rss_items.parquet | parquet | 349,206 | 17 | 2020-05-31 04:30:06+00:00..2026-06-18 15:08:00+00:00 | LOAD_CANONICAL | operational_archive_raw | EXACT_VINTAGE_CANDIDATE | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_historical_rss_items.parquet |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_historical_rss_temperature_forecasts.parquet | parquet | 66,612 | 21 | 2020-05-31 16:00:00+00:00..2026-06-27 00:00:00+00:00 | LOAD_CANONICAL | operational_anchor | PREDICTOR_NOW | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_historical_rss_temperature_forecasts.parquet |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_official_t15_scored_pre2024.csv | csv | 5,265 | 62 | 2000-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | SKIP_DUPLICATE_FORMAT | none | DUPLICATE | LOW | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_official_t15_scored_pre2024.csv |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_official_t15_scored_pre2024.parquet | parquet | 5,265 | 62 | 2000-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_supervised | TRAINING_TABLE_WITH_LABEL_FIREWALL | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_official_t15_scored_pre2024.parquet |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_press_archive_bulletins_offline.parquet | parquet | 137,291 | 28 | 1990-11-03 00:00:00+00:00..2011-09-06 01:45:00+00:00 | LOAD_WITH_QUARANTINE | operational_archive_raw | EXACT_VINTAGE_CANDIDATE | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_press_archive_bulletins_offline.parquet |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_press_archive_candidate_detail_coverage.csv | csv | 56 | 10 | None..None | LOAD_METADATA | acquisition_quality | NOT_DIRECT | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_press_archive_candidate_detail_coverage.csv |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_press_archive_forecast_days.parquet | parquet | 233,430 | 32 | 1990-11-03 00:00:00+00:00..2011-09-13 00:00:00+00:00 | LOAD_WITH_QUARANTINE | operational_archive_normalized | EXACT_VINTAGE_CANDIDATE | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_press_archive_forecast_days.parquet |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_press_archive_missing_detail_coverage.csv | csv | 37 | 10 | None..None | LOAD_METADATA | acquisition_quality | NOT_DIRECT | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_press_archive_missing_detail_coverage.csv |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_press_archive_parse_failures.csv | csv | 0 | 0 | None..None | CREATE_EMPTY_MONITOR_TABLE | quality_monitoring | NOT_DIRECT | MEDIUM | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_press_archive_parse_failures.csv |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_press_archive_temperature_forecast_days.csv | csv | 219,238 | 32 | 1999-12-31 17:46:00+00:00..2011-09-13 00:00:00+00:00 | SKIP_DUPLICATE_FORMAT | none | DUPLICATE | LOW | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_press_archive_temperature_forecast_days.csv |
| 05_hko_historical_rss_forecasts | 05_hko_historical_rss_forecasts/hko_press_archive_temperature_forecast_days.parquet | parquet | 219,238 | 32 | 1999-12-31 17:46:00+00:00..2011-09-13 00:00:00+00:00 | LOAD_WITH_QUARANTINE | operational_archive_normalized | EXACT_VINTAGE_CANDIDATE | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\05_hko_historical_rss_forecasts\hko_press_archive_temperature_forecast_days.parquet |
| 06_hko_tropical_cyclone_best_track | 06_hko_tropical_cyclone_best_track/hko_tropical_cyclone_best_track.parquet | parquet | 26,189 | 16 | 1985-01-06 06:00:00+00:00..2024-12-25 06:00:00+00:00 | LOAD_DIAGNOSTIC | diagnostic_regime_labels | RETROSPECTIVE_ONLY | MEDIUM | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\06_hko_tropical_cyclone_best_track\hko_tropical_cyclone_best_track.parquet |
| 07_hko_radar_satellite_lightning_nowcast | 07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet | parquet | 41 | 13 | 2026-06-19 04:36:00+00:00..2026-06-20 09:36:00+00:00 | LOAD_LIVE_COLLECTION | live_exact_vintage | FUTURE_AFTER_HISTORY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\07_hko_radar_satellite_lightning_nowcast\hko_gridded_rainfall_nowcast_summary.parquet |
| 07_hko_radar_satellite_lightning_nowcast | 07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet | parquet | 102 | 9 | None..None | LOAD_LIVE_COLLECTION | live_exact_vintage | FUTURE_AFTER_HISTORY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\07_hko_radar_satellite_lightning_nowcast\hko_lightning_counts_latest.parquet |
| 07_hko_radar_satellite_lightning_nowcast | 07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet | parquet | 80 | 8 | 2026-06-19 02:54:00+00:00..2026-06-19 04:48:00+00:00 | REGISTER_OBJECT_METADATA | live_object_catalog | FUTURE_AFTER_HISTORY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\07_hko_radar_satellite_lightning_nowcast\hko_radar_manifest_frames.parquet |
| 07_hko_radar_satellite_lightning_nowcast | 07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet | parquet | 4,589 | 10 | 2026-06-17 00:15:00+00:00..2026-06-19 23:30:00+00:00 | REGISTER_OBJECT_METADATA | live_object_catalog | FUTURE_AFTER_HISTORY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\07_hko_radar_satellite_lightning_nowcast\hko_satellite_image_inventory.parquet |
| 08_hko_marine_tide_coastal_waters | 08_hko_marine_tide_coastal_waters/hko_latest_tidal_information.parquet | parquet | 105 | 8 | 2026-06-19 04:45:00+00:00..2026-06-20 07:50:00+00:00 | LOAD_LIVE_COLLECTION | live_exact_vintage | FUTURE_AFTER_HISTORY | MEDIUM | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\08_hko_marine_tide_coastal_waters\hko_latest_tidal_information.parquet |
| 08_hko_marine_tide_coastal_waters | 08_hko_marine_tide_coastal_waters/hko_south_china_coastal_waters_bulletin.parquet | parquet | 30 | 10 | 2026-06-19 04:30:00+00:00..2026-06-20 04:30:00+00:00 | LOAD_LIVE_COLLECTION | live_exact_vintage | FUTURE_AFTER_HISTORY | MEDIUM | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\08_hko_marine_tide_coastal_waters\hko_south_china_coastal_waters_bulletin.parquet |
| 09_hko_arwf_station_forecasts | 09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet | parquet | 530 | 14 | 2026-06-19 00:00:00+00:00..2026-06-28 00:00:00+00:00 | LOAD_LIVE_COLLECTION | live_nwp_anchor | HIGH_POTENTIAL_AFTER_HISTORY | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\09_hko_arwf_station_forecasts\hko_arwf_station_daily_forecasts.parquet |
| 10_ncep_operational_grib_inventory | 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | parquet | 3,400 | 12 | None..None | REGISTER_OBJECT_METADATA_AND_REBUILD_INVENTORY | nwp_object_catalog | NOT_YET_DECODED | CRITICAL | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\10_ncep_operational_grib_inventory\ncep_operational_grib2_inventory.parquet |
| 11_static_geospatial_inventory | 11_static_geospatial_inventory/static_geospatial_package_inventory.parquet | parquet | 60 | 11 | None..None | REGISTER_OBJECT_METADATA | static_object_catalog | DERIVE_STATIC_FEATURES | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\11_static_geospatial_inventory\static_geospatial_package_inventory.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_exp0050_0099_feature_matrix.parquet | parquet | 48,577 | 566 | 1884-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | REGISTER_RESEARCH_ARTIFACT | research_artifacts | DO_NOT_USE_AS_CANONICAL_FEATURE_STORE | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_exp0050_0099_feature_matrix.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r14_feature_diagnostics.parquet | parquet | 44 | 8 | 1949-06-03 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r14_feature_diagnostics.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r14_feature_matrix.parquet | parquet | 26,632 | 120 | 1949-06-01 22:00:00+00:00..2023-12-31 00:00:00+00:00 | REGISTER_RESEARCH_ARTIFACT | research_artifacts | DO_NOT_USE_AS_CANONICAL_FEATURE_STORE | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r14_feature_matrix.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r14_fold_score_deltas.parquet | parquet | 36 | 16 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r14_fold_score_deltas.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r14_oof_predictions.parquet | parquet | 63,939 | 22 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_oof_predictions | OOF_STACKING_RESEARCH | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r14_oof_predictions.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r14_scoreboard.parquet | parquet | 3 | 11 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r14_scoreboard.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r15_feature_diagnostics.parquet | parquet | 68 | 8 | 1949-06-03 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r15_feature_diagnostics.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r15_feature_matrix.parquet | parquet | 23,943 | 120 | 1949-06-01 22:00:00+00:00..2023-12-31 00:00:00+00:00 | REGISTER_RESEARCH_ARTIFACT | research_artifacts | DO_NOT_USE_AS_CANONICAL_FEATURE_STORE | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r15_feature_matrix.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r15_fold_score_deltas.parquet | parquet | 36 | 16 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r15_fold_score_deltas.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r15_oof_predictions.parquet | parquet | 55,872 | 22 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_oof_predictions | OOF_STACKING_RESEARCH | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r15_oof_predictions.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r15_scoreboard.parquet | parquet | 3 | 11 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r15_scoreboard.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r16_feature_diagnostics.parquet | parquet | 29 | 8 | 1947-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r16_feature_diagnostics.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r16_feature_matrix.parquet | parquet | 25,202 | 120 | 1947-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | REGISTER_RESEARCH_ARTIFACT | research_artifacts | DO_NOT_USE_AS_CANONICAL_FEATURE_STORE | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r16_feature_matrix.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r16_fold_score_deltas.parquet | parquet | 36 | 16 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r16_fold_score_deltas.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r16_oof_predictions.parquet | parquet | 55,881 | 22 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_oof_predictions | OOF_STACKING_RESEARCH | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r16_oof_predictions.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r16_scoreboard.parquet | parquet | 3 | 11 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r16_scoreboard.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r17_feature_diagnostics.parquet | parquet | 75 | 8 | 1949-06-03 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r17_feature_diagnostics.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r17_feature_matrix.parquet | parquet | 23,943 | 120 | 1949-06-01 22:00:00+00:00..2023-12-31 00:00:00+00:00 | REGISTER_RESEARCH_ARTIFACT | research_artifacts | DO_NOT_USE_AS_CANONICAL_FEATURE_STORE | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r17_feature_matrix.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r17_fold_score_deltas.parquet | parquet | 36 | 16 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r17_fold_score_deltas.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r17_oof_predictions.parquet | parquet | 55,872 | 22 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_oof_predictions | OOF_STACKING_RESEARCH | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r17_oof_predictions.parquet |
| 12_hkg_t24_robust_experiment_outputs | 12_hkg_t24_robust_experiment_outputs/hkg_t24_r17_scoreboard.parquet | parquet | 3 | 11 | 1965-01-01 00:00:00+00:00..2023-12-31 00:00:00+00:00 | LOAD_RESEARCH_ONLY | research_metrics | EVIDENCE_ONLY | HIGH | PASS | C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\datasets\12_hkg_t24_robust_experiment_outputs\hkg_t24_r17_scoreboard.parquet |

## Governance table-load contract

| none |
| --- |

## Open quality issues from governance.quality_issue

| quality_issue_id | severity | dataset_id | source_table | attributes | evidence | required_action | current_status | remediation_implementation_path | validation_evidence_uri | resolution_timestamp | resolution_commit | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QI-016 | MEDIUM | 07_hko_radar_satellite_lightning_nowcast | hko_lightning_counts_latest.parquet | lightning_count | All 102 values are zero. | Retain feed but current sample has no information gain; continue collection and monitor variance. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-017 | MEDIUM | 08_hko_marine_tide_coastal_waters | hko_latest_tidal_information.parquet | height_m | 14.3% of tide heights are missing. | Retain with station-specific availability flags; do not impute indiscriminately. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-018 | MEDIUM | 09_hko_arwf_station_forecasts | hko_arwf_station_daily_forecasts.parquet | forecast_min/max_temperature_c | 28.3% missing. | Determine whether missingness is station capability, lead, or parser behavior; model only eligible station/lead cells. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-019 | MEDIUM | 11_static_geospatial_inventory | land-use assets | 2018-2024 land-use rasters | Using recent land-use maps for early historical rows creates temporal context leakage/misrepresentation. | Use terrain/coastline as static; make land use date-effective or reserve it for modern-era models. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-020 | MEDIUM | 05_hko_historical_rss_forecasts | duplicate CSV/Parquet outputs | all columns | Two major normalized tables exist in both CSV and Parquet. | Load Parquet only; register CSV as export artifact. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-021 | MEDIUM | 05_hko_historical_rss_forecasts | bulletins_offline | snapshot_at_hkt; stale_hours; retrieval_id; attempted_at_utc | These columns are entirely null. | Omit current physical columns or populate properly in future; do not pretend they add information. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-022 | MEDIUM |  | multiple | all weather attributes | Current history is only one to three days/one ARWF cycle. | Store and collect, but do not estimate model skill or feature importance from the current sample. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. Audit dataset scope was '07/08/09 live feeds', not a concrete dataset id. |
| QI-006 | HIGH | 04_noaa_isd_regional_surface | noaa_isd_core_observations.parquet | latitude; longitude | Profile includes latitude 0 and longitude -114.283/144.2 despite a regional Hong Kong/South China network. | Use date-effective NOAA station history metadata, not row-level coordinates, and quarantine impossible station metadata. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-007 | HIGH | 04_noaa_isd_regional_surface | noaa_isd_station_day_cutoff_summary.parquet | daily_air_temperature_min_c; daily_air_temperature_max_c | Column names do not prove values were truncated at 15:00 HKT. | Reject as predictor until aggregation code proves no post-cutoff observations entered. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-008 | HIGH | 03_noaa_igra_upper_air_hkm00045004 | both tables | source_id | Period-of-record and year-to-date sources coexist and can overlap. | Deduplicate by station, valid time, pressure level and source priority before feature generation. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-009 | HIGH | 02_hko_daily_climate_all_elements | hko_daily_climate_elements.parquet | value; parse_issue | 6,916 'Trace' values are currently nonnumeric; 7,389 are missing; 5 invalid dates exist. | Preserve trace_flag separately and apply variable-specific trace policy; quarantine invalid dates. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-010 | HIGH | 01_hko_daily_tmax_target | two tables | local_date; target values | Payload table has 49,628 rows versus 49,459 canonical labels and includes monthly/yearly sources plus one parse failure. | Use labels table as canonical. Deduplicate/reconcile payload dates and source overlap only for audit. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-011 | HIGH | 05_hko_historical_rss_forecasts | archive coverage tables | raw_detail_coverage_pct | Coverage is incomplete and highly era/product dependent. | Store coverage metadata and include source-era/frame controls in every evaluation; do not treat missing archive days as random. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-012 | HIGH | 07_hko_radar_satellite_lightning_nowcast | hko_gridded_rainfall_nowcast_summary.parquet | rainfall_max_mm | Maximum is 544.88 mm while sample covers only 41 snapshots. | Verify accumulation interval, grid fill values and units before use. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-013 | HIGH | 07_hko_radar_satellite_lightning_nowcast | hko_satellite_image_inventory.parquet | image_time_hkt | 132 of 4,589 non-null image times fail datetime parsing. | Quarantine malformed entries and distinguish page/JS manifests from actual images. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-014 | HIGH | 09_hko_arwf_station_forecasts | hko_arwf_station_daily_forecasts.parquet | model_time; last_modified | Model time is numeric and constant; last_modified is string; only one cycle exists. | Parse cycle/issue/valid timestamps, calculate lead hours, and collect many cycles before scoring. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-015 | HIGH | 12_hkg_t24_robust_experiment_outputs | feature matrices | IGRA/UA/ISD/daily feature families | Large portions are timestamp-blocked and some depend on corrupted raw fields. | Keep as research artifacts only; recompute any promoted feature from repaired canonical sources. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-001 | CRITICAL | 03_noaa_igra_upper_air_hkm00045004 | both tables | Multiple meteorological fields | IGRA raw/derived values contain sentinel-like -888.8 and -8888 values; relative humidity reaches 1000. | Do not load derived feature table as clean. Map documented source missing codes to NULL, apply correct scale, rerun all sounding features, and compare row counts/ranges. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-002 | CRITICAL | 04_noaa_isd_regional_surface | both tables | wind_direction_deg; wind_direction_deg_latest_before_1500 | Both fields are constant at 20 degrees across millions/hundreds of thousands of rows. | Treat current wind direction and all dependent u/v/directional features as invalid. Fix extractor and rebuild downstream matrices. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-003 | CRITICAL | 10_ncep_operational_grib_inventory | ncep_operational_grib2_inventory.parquet | cycle_date; cycle_hour_utc | Both cycle fields are empty for all 3,400 rows. | Rebuild inventory from filename and GRIB metadata with cycle_time_utc and valid_time_utc before any modelling. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-004 | CRITICAL | 05_hko_historical_rss_forecasts | hko_press_archive_forecast_days.parquet | forecast_max_c; target_issue_lead_days | Forecast maximum reaches 310 degreesC and lead days range from -4382 to 370. | Quarantine invalid rows; use only scoreable_row_valid, target_date_plausible and explicit T+24 lead checks. Repair parser. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. |
| QI-005 | CRITICAL |  | multiple | *_hkt versus *_utc | Profile min/max values are frequently identical for HKT and UTC columns. | Audit actual timezone semantics. Store one canonical timestamptz UTC plus explicitly derived HKT; never infer availability from a mislabeled naive timestamp. | OPEN |  |  |  |  | Loaded from audit contract; not resolved without validation evidence. Audit dataset scope was 'All timestamped datasets', not a concrete dataset id. |

## PostgreSQL object inventory

Every non-system table/view/partitioned table visible in the live DB is listed below. Exact rows are `count(*)` results; for views this means the view was counted at generation time.

| object | kind | exact rows | estimated rows | size | time/date ranges | top category values |
| --- | --- | --- | --- | --- | --- | --- |
| acquisition_provenance.codex_audit_root_hko_forecast_rss_archives_20200601_20_5fc53457 | table | 522 | 522 | 640.00 KB | ingested_at_utc=2026-06-23 15:03:30.455761+00..2026-06-23 15:03:30.455761+00 (non-null 522); timestamp=20200601..20260619 (non-null 516); retrieved_at_utc=2026-06-20 14:23:43.96598+00..2026-06-20 14:31:20.142217+00 (non-null 522) | feed: rss_local_forecast_en=92; rss_local_forecast_sc=92; rss_local_forecast_tc=92; rss_9day_forecast_en=82; rss_9day_forecast_sc=82; rss_9day_forecast_tc=82; kind: archive=516; listing=6 |
| catalog.attribute_contract | table | 1,869 | 1,869 | 1.64 MB |  | dataset_id: 12_hkg_t24_robust_experiment_outputs=1,274; 05_hko_historical_rss_forecasts=306; 03_noaa_igra_upper_air_hkm00045004=86; 07_hko_radar_satellite_lightning_nowcast=40; 04_noaa_isd_regional_surface=34; 01_hko_daily_tmax_target=22 |
| catalog.audit_snapshot | table | 1 | 1 | 136.00 KB | generated_at_utc=2026-06-23 11:53:01.458398+00..2026-06-23 11:53:01.458398+00 (non-null 1); extracted_at_utc=2026-06-23 13:24:25.476879+00..2026-06-23 13:24:25.476879+00 (non-null 1) |  |
| catalog.catalog_snapshot | table | 1 | 1 | 128.00 KB | retrieved_at_utc=2026-06-24 15:15:49+00..2026-06-24 15:15:49+00 (non-null 1) |  |
| catalog.codex_audit_root_manifest | table | 39 | 39 | 64.00 KB | ingested_at_utc=2026-06-23 15:03:31.275284+00..2026-06-23 15:03:31.275284+00 (non-null 39) |  |
| catalog.dataset_registry | table | 13 | 13 | 32.00 KB | loaded_at_utc=2026-06-23 13:24:25.980917+00..2026-06-23 13:24:25.980917+00 (non-null 13) | dataset_id: [root]=1; 01_hko_daily_tmax_target=1; 02_hko_daily_climate_all_elements=1; 03_noaa_igra_upper_air_hkm00045004=1; 04_noaa_isd_regional_surface=1; 05_hko_historical_rss_forecasts=1 |
| catalog.location | table | 132 | 132 | 160.00 KB | valid_from=1945-11-30..2026-06-19 (non-null 89); valid_to=1956-03-31..2025-08-24 (non-null 36); created_at_utc=2026-06-24 13:35:07.389533+00..2026-06-24 13:35:07.389533+00 (non-null 132) |  |
| catalog.location_group | table | 8 | 8 | 32.00 KB | created_at_utc=2026-06-24 13:35:07.389533+00..2026-06-24 13:35:07.389533+00 (non-null 8) |  |
| catalog.location_group_member | table | 177 | 177 | 96.00 KB | created_at_utc=2026-06-24 13:35:07.389533+00..2026-06-24 13:35:07.389533+00 (non-null 177) |  |
| catalog.model_registry | table | 56 | 56 | 120.00 KB | archive_start=2018-10-31..2026-04-16 (non-null 28); retrieved_at_utc=2026-06-24 15:15:49+00..2026-06-24 15:15:49+00 (non-null 56) | model_code: aifsenfo=1; aifsoper=1; aigefspres=1; aigefssfc=1; aigfspres=1; aigfssfc=1 |
| catalog.profile_snapshot | table | 0 | 0 | 24.00 KB |  |  |
| catalog.selector_snapshot | table | 1,288 | 1,288 | 960.00 KB | retrieved_at_utc=2026-06-24 15:15:49+00..2026-06-24 15:15:49+00 (non-null 1,288) | model_code: aifsenfo=56; aifsoper=56; aigefspres=56; aigefssfc=56; aigfspres=56; aigfssfc=56 |
| catalog.source_file_registry | table | 52 | 52 | 200.00 KB | data_min=1884-01-01 00:00:00+00..2026-06-19 04:45:00+00 (non-null 43); data_max=2011-09-06 01:45:00+00..2026-06-28 00:00:00+00 (non-null 43); metadata_min=2026-06-18 22:51:28.58811+00..2026-06-21 17:52:29.418284+00 (non-null 26); metadata_max=2026-06-18 22:51:53.53612+00..2026-06-21 17:52:29.418284+00 (non-null 26); created_at_utc=2026-06-23 13:24:27.186844+00..2026-06-23 13:24:27.186844+00 (non-null 52) | dataset_id: 12_hkg_t24_robust_experiment_outputs=21; 05_hko_historical_rss_forecasts=11; 07_hko_radar_satellite_lightning_nowcast=4; [root]=3; 01_hko_daily_tmax_target=2; 03_noaa_igra_upper_air_hkm00045004=2 |
| catalog.source_license | table | 1 | 1 | 32.00 KB | terms_last_updated=2026-05-21..2026-05-21 (non-null 1); retrieved_at_utc=2026-06-24 15:15:49+00..2026-06-24 15:15:49+00 (non-null 1) |  |
| catalog.station | table | 179 | 179 | 240.00 KB | valid_from=1945-11-30..2026-06-19 (non-null 89); valid_to=1956-03-31..2025-08-24 (non-null 36); created_at_utc=2026-06-24 13:35:07.389533+00..2026-06-24 13:35:07.389533+00 (non-null 179) | station_id: 1=1; 10=1; 100=1; 101=1; 102=1; 103=1; station_code: ARWF:CCH=1; ARWF:CWB=1; ARWF:G119=1; ARWF:G120=1; ARWF:G121=1; ARWF:G135=1 |
| catalog.station_dim | table | 36 | 36 | 80.00 KB | valid_from=1945-11-30..2012-08-23 (non-null 36); valid_to=1956-03-31..2025-08-24 (non-null 36) | station_id: 450010-99999=1; 450030-99999=1; 450040-99999=1; 450050-99999=1; 450060-99999=1; 450070-99999=1 |
| catalog.station_metadata_history | table | 0 | 0 | 16.00 KB |  | station_id: |
| catalog.variable | table | 19 | 19 | 48.00 KB | created_at_utc=2026-06-24 17:59:41.447155+00..2026-06-24 21:56:14.330959+00 (non-null 19) |  |
| catalog.variable_selector_snapshot | table | 0 | 0 | 56.00 KB |  | model_id: |
| catalog.weather_model | table | 0 | 0 | 48.00 KB |  | model_id: ; model_code: |
| catalog.source_registry | view | 52 | -1 | 0 B | data_min=1884-01-01 00:00:00+00..2026-06-19 04:45:00+00 (non-null 43); data_max=2011-09-06 01:45:00+00..2026-06-28 00:00:00+00 (non-null 43); metadata_min=2026-06-18 22:51:28.58811+00..2026-06-21 17:52:29.418284+00 (non-null 26); metadata_max=2026-06-18 22:51:53.53612+00..2026-06-21 17:52:29.418284+00 (non-null 26); created_at_utc=2026-06-23 13:24:27.186844+00..2026-06-23 13:24:27.186844+00 (non-null 52) | dataset_id: 12_hkg_t24_robust_experiment_outputs=21; 05_hko_historical_rss_forecasts=11; 07_hko_radar_satellite_lightning_nowcast=4; [root]=3; 01_hko_daily_tmax_target=2; 03_noaa_igra_upper_air_hkm00045004=2; source_id: 01_hko_daily_tmax_target/hko_daily_extract_tmax_payload_rows.parquet=1; 01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet=1; 02_hko_daily_climate_all_elements/hko_daily_climate_elements.parquet=1; 03_noaa_igra_upper_air_hkm00045004/noaa_igra_hkm00045004_key_pressure_levels.parquet=1; 03_noaa_igra_upper_air_hkm00045004/noaa_igra_hkm00045004_sounding_features.parquet=1; 04_noaa_isd_regional_surface/noaa_isd_core_observations.parquet=1 |
| diagnostic_physics.codex_audit_ds_02_hko_daily_climate_all_elements_hko_d_f7bb0017 | table | 556,399 | 556,385 | 288.99 MB | ingested_at_utc=2026-06-23 14:52:33.693072+00..2026-06-23 14:52:47.090122+00 (non-null 556,399); raw_retrieved_at_utc=2026-06-18T22:51:28.588110Z..2026-06-18T22:52:35.306583Z (non-null 556,399); local_date=1884-01-01..2026-05-31 (non-null 556,394); year=1884..2026 (non-null 556,399); month=1..12 (non-null 556,399); day=1..31 (non-null 556,399) | source_id: hko_daily_climate_maximum_temperature_all=49,460; hko_daily_climate_minimum_temperature_all=49,460; hko_daily_climate_mean_temperature_all=49,400; hko_daily_climate_mslp_all=49,400; hko_daily_climate_rainfall_all=49,400; hko_daily_climate_relative_humidity_all=29,006; station_or_domain: Hong Kong Observatory=378,635; King's Park=62,911; Waglan Island=50,858; North Point=37,952; Hong Kong Territory=15,300; Hong Kong International Airport=10,743; variable: daily_maximum_temperature=49,460; daily_minimum_temperature=49,460; daily_rainfall=49,400; mean_sea_level_pressure=49,400; mean_temperature=49,400; mean_relative_humidity=29,006; availability_tier: MECHANISM_ONLY=506,939; TARGET_ONLY=49,460 |
| diagnostic_physics.codex_audit_ds_03_noaa_igra_upper_air_hkm00045004_noaa_a791906b | table | 88,407 | 88,407 | 62.32 MB | ingested_at_utc=2026-06-23 14:53:51.371137+00..2026-06-23 14:53:51.371137+00 (non-null 88,407); raw_retrieved_at_utc=2026-06-19T04:57:24.874712Z..2026-06-19T04:57:27.098275Z (non-null 88,407); valid_at_utc=1949-06-01T22:00:00Z..2026-06-17T18:00:00Z (non-null 88,407); valid_at_hkt=1949-06-02T07:00:00+09:00..2026-06-18T02:00:00+08:00 (non-null 88,407) | source_id: noaa_igra_hkm00045004_period_of_record=86,305; noaa_igra_hkm00045004_year_to_date=2,102; station_id: HKM00045004=88,407; availability_tier: PROXY_WITH_LIMITATIONS=88,407; operational_input_allowed: false=88,407 |
| diagnostic_physics.codex_audit_ds_03_noaa_igra_upper_air_hkm00045004_noaa_f6023703 | table | 477,514 | 477,505 | 281.12 MB | ingested_at_utc=2026-06-23 14:53:11.801822+00..2026-06-23 14:53:26.242194+00 (non-null 477,514); raw_retrieved_at_utc=2026-06-19T04:57:24.874712Z..2026-06-19T04:57:27.098275Z (non-null 477,514); valid_at_utc=1949-06-01T22:00:00Z..2026-06-17T18:00:00Z (non-null 477,514); valid_at_hkt=1949-06-02T07:00:00+09:00..2026-06-18T02:00:00+08:00 (non-null 477,514) | source_id: noaa_igra_hkm00045004_period_of_record=466,719; noaa_igra_hkm00045004_year_to_date=10,795; station_id: HKM00045004=477,514; availability_tier: PROXY_WITH_LIMITATIONS=477,514; operational_input_allowed: false=477,514 |
| diagnostic_regime_labels.codex_audit_ds_06_hko_tropical_cyclone_best_track_hko__9d8b03ac | table | 26,189 | 26,189 | 13.91 MB | ingested_at_utc=2026-06-23 15:02:38.706342+00..2026-06-23 15:02:38.706342+00 (non-null 26,189); raw_retrieved_at_utc=2026-06-19T04:51:07.648318Z..2026-06-19T04:52:38.219460Z (non-null 26,189); valid_at_utc=1985-01-06T06:00:00Z..2024-12-25T06:00:00Z (non-null 26,189); valid_at_hkt=1985-01-06T14:00:00+08:00..2024-12-25T14:00:00+08:00 (non-null 26,189) | source_id: hko_tropical_cyclone_best_track=26,189; availability_tier: MECHANISM_ONLY=26,189; operational_input_allowed: false=26,189 |
| diagnostic_station_network.codex_audit_ds_04_noaa_isd_regional_surface_noaa_isd_c_688279e2 | table | 4,029,291 | 4,029,604 | 2.16 GB | ingested_at_utc=2026-06-23 14:54:03.561695+00..2026-06-23 14:56:27.259487+00 (non-null 4,029,291); observed_at_utc=1945-11-30T16:00:00+00:00..2025-08-24T21:30:00+00:00 (non-null 4,029,291); observed_at_hkt=1945-12-01T00:00:00+08:00..2025-08-25T05:30:00+08:00 (non-null 4,029,291); raw_retrieved_at_utc=2026-06-19T06:23:47.650000Z..2026-06-19T06:45:31.127456Z (non-null 4,029,291) | station_id: 450070-99999=743,894; 592870-99999=596,924; 450110-99999=556,678; 594930-99999=347,257; 595010-99999=173,965; 592930-99999=173,836; source_id: noaa_isd_nearby_station_year=4,029,291; availability_tier: PROXY_WITH_LIMITATIONS=4,029,291; operational_input_allowed: false=4,029,291 |
| diagnostic_station_network.codex_audit_ds_04_noaa_isd_regional_surface_noaa_isd_s_80f559e2 | table | 317,489 | 317,489 | 108.79 MB | ingested_at_utc=2026-06-23 14:59:28.579155+00..2026-06-23 14:59:33.602262+00 (non-null 317,489); local_date=1945-12-01..2025-08-25 (non-null 317,489) | station_id: 592870-99999=22,728; 450110-99999=22,628; 592930-99999=22,256; 595010-99999=22,251; 450070-99999=21,404; 594930-99999=21,385; availability_tier: PROXY_WITH_LIMITATIONS=317,489; operational_input_allowed: false=317,489 |
| feature_safe.hko_t24_official_anchor | view | 0 | -1 | 0 B |  | source_product: ; quality_status: ; eligibility_status: |
| feature_safe.hko_target_history_pre2024 | view | 48,577 | -1 | 0 B | local_date=1884-01-01..2023-12-31 (non-null 48,577) | quality_status: VALID=48,577 |
| feature_safe.live_exact_vintage_catalog | view | 0 | -1 | 0 B |  | dataset_id: ; source_id: ; eligibility_status: |
| feature_store.feature_definition | table | 0 | 0 | 24.00 KB |  |  |
| feature_store.feature_value | table | 0 | 0 | 16.00 KB |  |  |
| feature_store.target_snapshot_manifest | table | 0 | 0 | 24.00 KB |  |  |
| feature_store.h24n_tactical_nwp_feature_source | view | 1,965,090 | -1 | 0 B | target_date_hkt=2020-10-03..2026-06-26 (non-null 1,965,090); run_time_utc=2020-10-01 18:00:00+00..2026-06-24 18:00:00+00 (non-null 1,965,090); valid_time_utc=2020-10-02 18:00:00+00..2026-06-26 12:00:00+00 (non-null 1,965,090); created_at_utc=2026-06-25 05:26:04.179242+00..2026-06-25 17:30:57.709773+00 (non-null 1,965,090) | dataset_code: gfs=575,004; gefsatmos=517,824; ifsenfo=343,616; gefsatmosmean=200,436; ifsoper=91,260; aifsenfo=72,270; quality_status: raw_valid=1,965,090 |
| governance.availability_contract | table | 0 | 0 | 24.00 KB |  | dataset_id: ; source_id: |
| governance.availability_grade | table | 5 | 5 | 48.00 KB |  |  |
| governance.feature_eligibility | table | 0 | 0 | 24.00 KB |  | dataset_id: ; eligibility_status: |
| governance.gribstream_usage_constraint | table | 1 | 1 | 32.00 KB | created_at_utc=2026-06-24 13:35:06.880847+00..2026-06-24 13:35:06.880847+00 (non-null 1) |  |
| governance.leakage_test_result | table | 0 | 0 | 16.00 KB |  |  |
| governance.operational_contract | table | 1 | 1 | 32.00 KB | development_end_date=2023-12-31..2023-12-31 (non-null 1); created_at_utc=2026-06-24 10:27:46.165307+00..2026-06-24 10:27:46.165307+00 (non-null 1) |  |
| governance.parser_version | table | 0 | 0 | 16.00 KB |  |  |
| governance.quality_issue | table | 22 | 22 | 96.00 KB |  | dataset_id: 05_hko_historical_rss_forecasts=4; 04_noaa_isd_regional_surface=3; 07_hko_radar_satellite_lightning_nowcast=3; 03_noaa_igra_upper_air_hkm00045004=2; 09_hko_arwf_station_forecasts=2; (null)=2 |
| governance.quarantine_reason | table | 6 | 6 | 32.00 KB |  |  |
| governance.schema_version | table | 7 | 7 | 32.00 KB | applied_at_utc=2026-06-23 13:24:24.459262+00..2026-06-24 23:43:10.374476+00 (non-null 7) |  |
| governance.sealed_period | table | 1 | 1 | 32.00 KB | local_date_start=2024-01-01..2024-01-01 (non-null 1); created_at_utc=2026-06-24 10:27:46.165307+00..2026-06-24 10:27:46.165307+00 (non-null 1) |  |
| governance.table_load_contract | table | 52 | 52 | 112.00 KB | data_min=1884-01-01 00:00:00+00..2026-06-19 04:45:00+00 (non-null 43); data_max=2011-09-06 01:45:00+00..2026-06-28 00:00:00+00 (non-null 43) | dataset_id: 12_hkg_t24_robust_experiment_outputs=21; 05_hko_historical_rss_forecasts=11; 07_hko_radar_satellite_lightning_nowcast=4; [root]=3; 01_hko_daily_tmax_target=2; 03_noaa_igra_upper_air_hkm00045004=2 |
| governance.attribute_contract | view | 1,869 | -1 | 0 B |  | dataset_id: 12_hkg_t24_robust_experiment_outputs=1,274; 05_hko_historical_rss_forecasts=306; 03_noaa_igra_upper_air_hkm00045004=86; 07_hko_radar_satellite_lightning_nowcast=40; 04_noaa_isd_regional_surface=34; 01_hko_daily_tmax_target=22 |
| ingestion.batch | table | 1 | 1 | 32.00 KB | started_at_utc=2026-06-23 14:52:20.824969+00..2026-06-23 14:52:20.824969+00 (non-null 1); finished_at_utc=2026-06-23 15:03:34.863712+00..2026-06-23 15:03:34.863712+00 (non-null 1) |  |
| ingestion.file_result | table | 52 | 52 | 112.00 KB | started_at_utc=2026-06-23 13:31:19.134067+00..2026-06-23 14:14:11.988879+00 (non-null 52); finished_at_utc=2026-06-23 14:52:29.187014+00..2026-06-23 15:03:31.658228+00 (non-null 52) |  |
| ingestion.reconciliation | table | 52 | 52 | 168.00 KB |  | dataset_id: 12_hkg_t24_robust_experiment_outputs=21; 05_hko_historical_rss_forecasts=11; 07_hko_radar_satellite_lightning_nowcast=4; [root]=3; 01_hko_daily_tmax_target=2; 03_noaa_igra_upper_air_hkm00045004=2 |
| ingestion.row_rejection | table | 0 | 0 | 16.00 KB |  | dataset_id: |
| label_core.hko_daily_tmax | table | 48,577 | 48,577 | 12.72 MB | local_date=1884-01-01..2023-12-31 (non-null 48,577); retrieved_at_utc=2026-06-18 22:51:53.53612+00..2026-06-18 22:51:53.53612+00 (non-null 48,577) | quality_status: VALID=48,577 |
| live.issued_forecast | table | 0 | 0 | 24.00 KB |  |  |
| live_exact_vintage.catalog | table | 0 | 0 | 16.00 KB |  | dataset_id: ; source_id: ; eligibility_status: |
| live_exact_vintage.codex_audit_ds_07_hko_radar_satellite_lightning_nowcas_97d078a5 | table | 41 | 41 | 72.00 KB | ingested_at_utc=2026-06-23 15:02:41.414344+00..2026-06-23 15:02:41.414344+00 (non-null 41); raw_retrieved_at_utc=2026-06-19T04:50:52.482529Z..2026-06-20T07:56:22.144891Z (non-null 41) | source_id: hko_gridded_rainfall_nowcast=41; availability_tier: GOLD_EXACT_VINTAGE=41; operational_input_allowed: true=41 |
| live_exact_vintage.codex_audit_ds_07_hko_radar_satellite_lightning_nowcas_eb458062 | table | 102 | 102 | 88.00 KB | ingested_at_utc=2026-06-23 15:02:42.013512+00..2026-06-23 15:02:42.013512+00 (non-null 102); raw_retrieved_at_utc=2026-06-19T04:57:47.135172Z..2026-06-20T08:03:02.341258Z (non-null 102) | source_id: hko_lightning_counts_latest=102; availability_tier: GOLD_EXACT_VINTAGE=102; operational_input_allowed: true=102 |
| live_exact_vintage.codex_audit_ds_08_hko_marine_tide_coastal_waters_hko_l_be7bc027 | table | 105 | 105 | 88.00 KB | ingested_at_utc=2026-06-23 15:02:43.33808+00..2026-06-23 15:02:43.33808+00 (non-null 105); raw_retrieved_at_utc=2026-06-19T04:50:57.280298Z..2026-06-20T07:56:27.102158Z (non-null 105); observed_at_hkt=2026-06-19T12:45:00+08:00..2026-06-20T15:50:00+08:00 (non-null 105) | source_id: hko_latest_tidal_information=105; availability_tier: GOLD_EXACT_VINTAGE=105; operational_input_allowed: true=105 |
| live_exact_vintage.codex_audit_ds_08_hko_marine_tide_coastal_waters_hko_s_4bce52ff | table | 30 | 30 | 64.00 KB | ingested_at_utc=2026-06-23 15:02:43.981515+00..2026-06-23 15:02:43.981515+00 (non-null 30); raw_retrieved_at_utc=2026-06-19T04:50:55.669615Z..2026-06-20T07:56:24.707400Z (non-null 30) | source_id: hko_south_china_coastal_waters_bulletin=30; availability_tier: GOLD_EXACT_VINTAGE=30; operational_input_allowed: true=30 |
| live_nwp_anchor.codex_audit_ds_09_hko_arwf_station_forecasts_hko_arwf__e2cc97ac | table | 530 | 530 | 352.00 KB | ingested_at_utc=2026-06-23 15:02:44.773109+00..2026-06-23 15:02:44.773109+00 (non-null 530); raw_retrieved_at_utc=2026-06-19T06:12:34.954063Z..2026-06-19T06:14:15.039351Z (non-null 530); model_time=2026061812..2026061812 (non-null 530); last_modified=20260619131207..20260619134017 (non-null 530); forecast_date=20260619..20260628 (non-null 530) | source_id: hko_arwf_station_forecast=530; station_code: CCH=10; CWB=10; G119=10; G120=10; G121=10; G135=10; availability_tier: GOLD_EXACT_VINTAGE=530; operational_input_allowed: true=530 |
| nwp_core.point_value | partitioned table | 0 | 0 | 0 B |  | quality_status: |
| nwp_core.model_run | table | 0 | 0 | 72.00 KB |  | model_id: |
| nwp_core.point_value_default | table | 0 | 0 | 327.39 MB |  | quality_status: |
| nwp_tactical.acquisition_chunk | table | 1,190 | 1,190 | 2.84 MB | created_at_utc=2026-06-24 23:47:05.150292+00..2026-06-25 17:31:03.998675+00 (non-null 1,190); started_at_utc=2026-06-24 23:47:05.150292+00..2026-06-25 17:31:03.998675+00 (non-null 1,190); completed_at_utc=2026-06-25 00:13:27.951745+00..2026-06-25 17:31:15.99885+00 (non-null 1,190) | dataset_code: gefsatmos=420; ifsenfo=171; gfs=140; aigefssfc=80; aifsenfo=73; gefsatmosmean=71 |
| nwp_tactical.forecast_wide | table | 1,965,090 | 1,965,813 | 1.43 GB | target_date_hkt=2020-10-03..2026-06-26 (non-null 1,965,090); run_time_utc=2020-10-01 18:00:00+00..2026-06-24 18:00:00+00 (non-null 1,965,090); valid_time_utc=2020-10-02 18:00:00+00..2026-06-26 12:00:00+00 (non-null 1,965,090); created_at_utc=2026-06-25 05:26:04.179242+00..2026-06-25 17:30:57.709773+00 (non-null 1,965,090) | dataset_code: gfs=575,004; gefsatmos=517,824; ifsenfo=343,616; gefsatmosmean=200,436; ifsoper=91,260; aifsenfo=72,270; quality_status: raw_valid=1,965,090 |
| nwp_tactical.location_stencil | table | 12 | 12 | 32.00 KB | created_at_utc=2026-06-24 23:43:10.374476+00..2026-06-24 23:43:10.374476+00 (non-null 12) |  |
| nwp_tactical.model_plan | table | 14 | 14 | 32.00 KB | archive_run_start_utc=2020-10-01 18:00:00+00..2026-04-16 18:00:00+00 (non-null 12); archive_run_end_utc=2026-03-01 12:00:00+00..2026-06-22 00:00:00+00 (non-null 12); target_date_start=2020-10-03..2024-03-03 (non-null 5); target_date_end=2026-06-23..2026-06-23 (non-null 5); updated_at_utc=2026-06-25 13:14:47.139774+00..2026-06-25 13:14:47.139774+00 (non-null 14) | dataset_code: aifsenfo=1; aifsoper=1; aigefssfc=1; aigfspres=1; aigfssfc=1; cwawrf15=1 |
| nwp_tactical.raw_response_object | table | 1,285 | 1,285 | 1.08 MB | retrieved_at_utc=2026-06-24 23:47:05.150292+00..2026-06-25 17:31:03.998675+00 (non-null 1,285) |  |
| nwp_tactical.validation_issue | table | 0 | 0 | 16.00 KB |  | dataset_code: |
| nwp_tactical.variable_plan | table | 78 | 78 | 72.00 KB |  | dataset_code: gfs=13; ifsoper=12; cwawrf15=10; aifsoper=8; gefsatmosmean=8; fourcastnetgfs=6 |
| object_catalog.asset | table | 19 | 19 | 120.00 KB | registered_at_utc=2026-06-23 13:27:32.075687+00..2026-06-23 13:27:32.075687+00 (non-null 19) | dataset_id: (null)=9; 12_hkg_t24_robust_experiment_outputs=5; 07_hko_radar_satellite_lightning_nowcast=2; [root]=1; 10_ncep_operational_grib_inventory=1; 11_static_geospatial_inventory=1 |
| operational_anchor.hko_t24_official_anchor_rows | table | 0 | 0 | 32.00 KB |  | source_product: ; quality_status: ; eligibility_status: |
| public.hko_historical_forecasts_2000_2026 | table | 324,179 | 324,179 | 606.78 MB | index_date=2000-01-01..2026-06-20 (non-null 324,179); issue_at_hkt=2000-01-01 01:46:00..2026-06-20 23:45:00 (non-null 324,132); issue_at_utc=1999-12-31 17:46:00+00..2026-06-20 15:45:00+00 (non-null 324,132); target_date=1990-11-03..2026-06-21 (non-null 203,385); source_archive_mtime_utc=2026-06-23 17:56:45.90709+00..2026-06-23 17:56:45.90709+00 (non-null 324,179); ingested_at_utc=2026-06-24 07:30:32+00..2026-06-24 07:30:32+00 (non-null 324,179) | product_type: local=264,325; 9day=30,438; 7day=23,223; 5day=6,193; row_quality_status: usable_local_minmax=115,795; missing_target_date=60,940; bulletin_only_multiday_product=59,854; usable_local_tmax_only=58,199; missing_forecast_max=29,383; invalid_target_lead=8 |
| quarantine.rejected_payload | table | 0 | 0 | 16.00 KB |  |  |
| raw_audit.acquisition_request | table | 0 | 0 | 104.00 KB |  | model_code: |
| raw_audit.codex_audit_ds_01_hko_daily_tmax_target_hko_daily_extr_23cb54da | table | 49,628 | 49,628 | 24.67 MB | ingested_at_utc=2026-06-23 14:52:25.869901+00..2026-06-23 14:52:25.869901+00 (non-null 49,628); raw_retrieved_at_utc=2026-06-18T22:52:48.528263Z..2026-06-18T22:58:13.112613Z (non-null 49,628); local_date=1884-01-01..2026-06-17 (non-null 49,627); year=1884..2026 (non-null 49,628); month=1..12 (non-null 49,627); day=1..31 (non-null 49,627) | source_id: hko_daily_extract_year=49,460; hko_daily_extract_month=168; availability_tier: TARGET_ONLY=49,627; (null)=1; operational_input_allowed: false=49,627; (null)=1 |
| raw_audit.response_object | table | 0 | 0 | 96.00 KB |  |  |
| research.expert_oof_prediction | table | 0 | 0 | 16.00 KB |  |  |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_184a0162 | table | 36 | 36 | 64.00 KB | ingested_at_utc=2026-06-23 15:02:48.185824+00..2026-06-23 15:02:48.185824+00 (non-null 36); first_date=1965-01-01..2020-01-01 (non-null 36); last_date=1969-12-31..2023-12-31 (non-null 36) | model_id: r14_lag_calendar_baseline=12; r14_stability_only=12; r14_upper_air_core=12 |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_2bee51a4 | table | 3 | 3 | 32.00 KB | ingested_at_utc=2026-06-23 15:03:08.403214+00..2026-06-23 15:03:08.403214+00 (non-null 3); first_date=1965-01-01..1965-01-01 (non-null 3); last_date=2023-12-31..2023-12-31 (non-null 3) | model_id: r15_coupling_terms=1; r15_lag_calendar_baseline=1; r15_upper_air_plus_isd=1 |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_2ed3981b | table | 68 | 68 | 72.00 KB | ingested_at_utc=2026-06-23 15:02:58.750474+00..2026-06-23 15:02:58.750474+00 (non-null 68); first_date=1949-06-03..1976-05-10 (non-null 68); last_date=2023-12-31..2023-12-31 (non-null 68) |  |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_38e9e3a0 | table | 44 | 44 | 64.00 KB | ingested_at_utc=2026-06-23 15:02:46.939058+00..2026-06-23 15:02:46.939058+00 (non-null 44); first_date=1949-06-03..1976-05-10 (non-null 44); last_date=2023-12-31..2023-12-31 (non-null 44) |  |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_4df989ab | table | 3 | 3 | 32.00 KB | ingested_at_utc=2026-06-23 15:03:18.832909+00..2026-06-23 15:03:18.832909+00 (non-null 3); first_date=1965-01-01..1965-01-01 (non-null 3); last_date=2023-12-31..2023-12-31 (non-null 3) | model_id: r16_isd_regional_aggregate=1; r16_isd_station_panel=1; r16_lag_calendar_baseline=1 |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_52683771 | table | 36 | 36 | 64.00 KB | ingested_at_utc=2026-06-23 15:03:21.381989+00..2026-06-23 15:03:21.381989+00 (non-null 36); first_date=1965-01-01..2020-01-01 (non-null 36); last_date=1965-01-02..2023-12-31 (non-null 36) | model_id: r17_combined_long_history_core=12; r17_era_transfer_terms=12; r17_lag_calendar_baseline=12 |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_75354da8 | table | 3 | 3 | 32.00 KB | ingested_at_utc=2026-06-23 15:03:29.442415+00..2026-06-23 15:03:29.442415+00 (non-null 3); first_date=1965-01-01..1965-01-01 (non-null 3); last_date=2023-12-31..2023-12-31 (non-null 3) | model_id: r17_combined_long_history_core=1; r17_era_transfer_terms=1; r17_lag_calendar_baseline=1 |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_8894ab84 | table | 75 | 75 | 72.00 KB | ingested_at_utc=2026-06-23 15:03:19.600568+00..2026-06-23 15:03:19.600568+00 (non-null 75); first_date=1949-06-03..1976-05-10 (non-null 75); last_date=2023-12-31..2023-12-31 (non-null 75) |  |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_9c7c6556 | table | 36 | 36 | 64.00 KB | ingested_at_utc=2026-06-23 15:03:10.318443+00..2026-06-23 15:03:10.318443+00 (non-null 36); first_date=1965-01-01..2020-01-01 (non-null 36); last_date=1965-01-02..2023-12-31 (non-null 36) | model_id: r16_isd_regional_aggregate=12; r16_isd_station_panel=12; r16_lag_calendar_baseline=12 |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_b10ef890 | table | 3 | 3 | 32.00 KB | ingested_at_utc=2026-06-23 15:02:57.645984+00..2026-06-23 15:02:57.645984+00 (non-null 3); first_date=1965-01-01..1965-01-01 (non-null 3); last_date=2023-12-31..2023-12-31 (non-null 3) | model_id: r14_lag_calendar_baseline=1; r14_stability_only=1; r14_upper_air_core=1 |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_b6697d99 | table | 29 | 29 | 32.00 KB | ingested_at_utc=2026-06-23 15:03:09.247929+00..2026-06-23 15:03:09.247929+00 (non-null 29); first_date=1947-01-01..1947-01-01 (non-null 29); last_date=2023-12-31..2023-12-31 (non-null 29) |  |
| research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_cfa16ade | table | 36 | 36 | 64.00 KB | ingested_at_utc=2026-06-23 15:03:00.169661+00..2026-06-23 15:03:00.169661+00 (non-null 36); first_date=1965-01-01..2020-01-01 (non-null 36); last_date=1965-01-02..2023-12-31 (non-null 36) | model_id: r15_coupling_terms=12; r15_lag_calendar_baseline=12; r15_upper_air_plus_isd=12 |
| research_oof_predictions.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_0ea616a5 | table | 55,872 | 55,872 | 23.60 MB | ingested_at_utc=2026-06-23 15:03:03.010551+00..2026-06-23 15:03:03.010551+00 (non-null 55,872); target_date=1965-01-01 00:00:00+00..2023-12-31 00:00:00+00 (non-null 55,872); year=1965..2023 (non-null 55,872); month=1..12 (non-null 55,872); training_start=1949-06-03 00:00:00+00..1949-06-03 00:00:00+00 (non-null 55,872); training_end=1964-12-31 00:00:00+00..2019-12-31 00:00:00+00 (non-null 55,872) | research_id: HKG-T24-R15=55,872; model_id: r15_coupling_terms=18,624; r15_lag_calendar_baseline=18,624; r15_upper_air_plus_isd=18,624; model_family: ridge_calendar_lag_baseline=18,624; ridge_surface_upper_air=18,624; ridge_surface_upper_air_coupling=18,624 |
| research_oof_predictions.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_38d58a4a | table | 55,881 | 55,881 | 23.60 MB | ingested_at_utc=2026-06-23 15:03:13.830038+00..2026-06-23 15:03:13.830038+00 (non-null 55,881); target_date=1965-01-01 00:00:00+00..2023-12-31 00:00:00+00 (non-null 55,881); year=1965..2023 (non-null 55,881); month=1..12 (non-null 55,881); training_start=1947-01-01 00:00:00+00..1947-01-01 00:00:00+00 (non-null 55,881); training_end=1964-12-31 00:00:00+00..2019-12-31 00:00:00+00 (non-null 55,881) | research_id: HKG-T24-R16=55,881; model_id: r16_isd_regional_aggregate=18,627; r16_isd_station_panel=18,627; r16_lag_calendar_baseline=18,627; model_family: ridge_calendar_lag_baseline=18,627; ridge_isd_regional_aggregate=18,627; ridge_isd_station_panel=18,627 |
| research_oof_predictions.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_9be0ab50 | table | 55,872 | 55,872 | 23.60 MB | ingested_at_utc=2026-06-23 15:03:24.528368+00..2026-06-23 15:03:24.528368+00 (non-null 55,872); target_date=1965-01-01 00:00:00+00..2023-12-31 00:00:00+00 (non-null 55,872); year=1965..2023 (non-null 55,872); month=1..12 (non-null 55,872); training_start=1949-06-03 00:00:00+00..1949-06-03 00:00:00+00 (non-null 55,872); training_end=1964-12-31 00:00:00+00..2019-12-31 00:00:00+00 (non-null 55,872) | research_id: HKG-T24-R17=55,872; model_id: r17_combined_long_history_core=18,624; r17_era_transfer_terms=18,624; r17_lag_calendar_baseline=18,624; model_family: ridge_calendar_lag_baseline=18,624; ridge_combined_long_history_core=18,624; ridge_era_transfer_terms=18,624 |
| research_oof_predictions.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_ce701451 | table | 63,939 | 63,939 | 27.34 MB | ingested_at_utc=2026-06-23 15:02:51.804363+00..2026-06-23 15:02:51.804363+00 (non-null 63,939); target_date=1965-01-01 00:00:00+00..2023-12-31 00:00:00+00 (non-null 63,939); year=1965..2023 (non-null 63,939); month=1..12 (non-null 63,939); training_start=1949-06-03 00:00:00+00..1949-06-03 00:00:00+00 (non-null 63,939); training_end=1964-12-31 00:00:00+00..2019-12-31 00:00:00+00 (non-null 63,939) | research_id: HKG-T24-R14=63,939; model_id: r14_lag_calendar_baseline=21,313; r14_stability_only=21,313; r14_upper_air_core=21,313; model_family: ridge_calendar_lag_baseline=21,313; ridge_upper_air_core=21,313; ridge_upper_air_stability_ablation=21,313 |
| sealed_confirmation.hko_daily_tmax | table | 882 | 882 | 392.00 KB | local_date=2024-01-01..2026-05-31 (non-null 882); retrieved_at_utc=2026-06-18 22:51:53.53612+00..2026-06-18 22:51:53.53612+00 (non-null 882) | quality_status: VALID=882 |

## PostgreSQL column inventory

This appendix lists the columns for every non-system PostgreSQL object in this database. It is intentionally exhaustive so a future implementer can see the fields available without opening psql.

### `acquisition_provenance.codex_audit_root_hko_forecast_rss_archives_20200601_20_5fc53457`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | feed | text |
| 7 | kind | text |
| 8 | timestamp | double precision |
| 9 | url | text |
| 10 | status_code | bigint |
| 11 | content_sha256 | text |
| 12 | content_length | bigint |
| 13 | path | text |
| 14 | retrieved_at_utc | timestamp with time zone |
| 15 | data_gov_filename | text |
| 16 | data_gov_period | text |
| 17 | data_gov_expected_size | double precision |
| 18 | skipped_existing | boolean |

### `catalog.attribute_contract`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | attribute_contract_id | bigint |
| 2 | dataset_id | text |
| 3 | source_file | text |
| 4 | file_type | text |
| 5 | attribute_name | text |
| 6 | source_dtype | text |
| 7 | semantic_class | text |
| 8 | row_count | bigint |
| 9 | non_null_count | bigint |
| 10 | null_count | bigint |
| 11 | null_pct | double precision |
| 12 | storage_decision | text |
| 13 | db_layer | text |
| 14 | model_role | text |
| 15 | operational_status | text |
| 16 | quality_action | text |
| 17 | usefulness_score | smallint |
| 18 | rationale | text |
| 19 | profile_min | text |
| 20 | profile_max | text |
| 21 | audit_snapshot_id | text |
| 22 | contract_version | text |
| 23 | reconciliation_status | text |
| 24 | physical_destination | text |

### `catalog.audit_snapshot`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | audit_snapshot_id | text |
| 2 | bundle_sha256 | character |
| 3 | bundle_bytes | bigint |
| 4 | original_local_path | text |
| 5 | repository_uri | text |
| 6 | extracted_uri | text |
| 7 | generated_at_utc | timestamp with time zone |
| 8 | extracted_at_utc | timestamp with time zone |
| 9 | git_commit_before | text |
| 10 | git_commit_after | text |
| 11 | manifest | jsonb |

### `catalog.catalog_snapshot`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | catalog_snapshot_id | text |
| 2 | provider | text |
| 3 | source_url | text |
| 4 | retrieved_at_utc | timestamp with time zone |
| 5 | status_code | integer |
| 6 | content_sha256 | character |
| 7 | content_bytes | bigint |
| 8 | content_json | jsonb |
| 9 | notes | text |

### `catalog.codex_audit_root_manifest`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | dataset_folder | text |
| 7 | dataset_title | text |
| 8 | file_name | text |
| 9 | organized_path | text |
| 10 | bytes | bigint |
| 11 | storage | text |

### `catalog.dataset_registry`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | dataset_id | text |
| 2 | db_inclusion | text |
| 3 | recommended_layer | text |
| 4 | current_operational_value | smallint |
| 5 | diagnostic_research_value | smallint |
| 6 | future_potential | smallint |
| 7 | verdict | text |
| 8 | audit_snapshot_id | text |
| 9 | contract_version | text |
| 10 | loaded_at_utc | timestamp with time zone |

### `catalog.location`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | location_id | bigint |
| 2 | location_code | text |
| 3 | name | text |
| 4 | latitude | double precision |
| 5 | longitude | double precision |
| 6 | elevation_m | double precision |
| 7 | location_role | text |
| 8 | valid_from | date |
| 9 | valid_to | date |
| 10 | metadata_source | text |
| 11 | metadata_sha256 | character |
| 12 | created_at_utc | timestamp with time zone |

### `catalog.location_group`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | group_code | text |
| 2 | group_name | text |
| 3 | group_type | text |
| 4 | description | text |
| 5 | created_at_utc | timestamp with time zone |

### `catalog.location_group_member`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | group_code | text |
| 2 | location_id | bigint |
| 3 | member_role | text |
| 4 | valid_from | date |
| 5 | valid_to | date |
| 6 | created_at_utc | timestamp with time zone |

### `catalog.model_registry`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | model_code | text |
| 2 | provider | text |
| 3 | model_name | text |
| 4 | domain | text |
| 5 | disposition | text |
| 6 | archive_or_window | text |
| 7 | archive_start | date |
| 8 | model_type | text |
| 9 | native_resolution | text |
| 10 | update_cadence | text |
| 11 | lead_time | text |
| 12 | page_url | text |
| 13 | catalog_snapshot_id | text |
| 14 | selector_count | integer |
| 15 | coverage_status | text |
| 16 | final_status | text |
| 17 | retrieved_at_utc | timestamp with time zone |
| 18 | notes | text |

### `catalog.profile_snapshot`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | profile_snapshot_id | bigint |
| 2 | profile_name | text |
| 3 | generated_at_utc | timestamp with time zone |
| 4 | source_root | text |
| 5 | content_sha256 | character |
| 6 | object_uri | text |
| 7 | dataset_count | integer |
| 8 | file_count | integer |
| 9 | row_table_count | integer |
| 10 | row_count | bigint |
| 11 | attribute_count | integer |
| 12 | audit_snapshot_id | text |
| 13 | loaded_at_utc | timestamp with time zone |

### `catalog.selector_snapshot`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | selector_snapshot_id | bigint |
| 2 | model_code | text |
| 3 | semantic_variable | text |
| 4 | semantic_family | text |
| 5 | semantic_priority | text |
| 6 | requested_levels | text |
| 7 | native_name | text |
| 8 | native_level | text |
| 9 | native_info | text |
| 10 | exact_selector | jsonb |
| 11 | selector_status | text |
| 12 | blocker | text |
| 13 | source_sha256 | character |
| 14 | retrieved_at_utc | timestamp with time zone |

### `catalog.source_file_registry`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | source_file_id | bigint |
| 2 | dataset_id | text |
| 3 | source_file | text |
| 4 | repository_uri | text |
| 5 | original_local_path | text |
| 6 | file_type | text |
| 7 | physical_sha256 | character |
| 8 | byte_size | bigint |
| 9 | source_row_count | bigint |
| 10 | attribute_count | integer |
| 11 | data_min | timestamp with time zone |
| 12 | data_max | timestamp with time zone |
| 13 | metadata_min | timestamp with time zone |
| 14 | metadata_max | timestamp with time zone |
| 15 | ingestion_action | text |
| 16 | target_database_layer | text |
| 17 | model_status | text |
| 18 | priority | text |
| 19 | ingestion_version | text |
| 20 | audit_snapshot_id | text |
| 21 | status | text |
| 22 | created_at_utc | timestamp with time zone |

### `catalog.source_license`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | source_code | text |
| 2 | source_name | text |
| 3 | provider | text |
| 4 | terms_url | text |
| 5 | terms_last_updated | date |
| 6 | licence_status | text |
| 7 | commercial_or_bulk_status | text |
| 8 | asof_availability_status | text |
| 9 | quota_status | text |
| 10 | notes | text |
| 11 | retrieved_at_utc | timestamp with time zone |

### `catalog.station`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | station_id | bigint |
| 2 | station_code | text |
| 3 | station_name | text |
| 4 | network | text |
| 5 | icao | text |
| 6 | country_code | text |
| 7 | location_id | bigint |
| 8 | station_role | text |
| 9 | target_station | boolean |
| 10 | valid_from | date |
| 11 | valid_to | date |
| 12 | metadata_status | text |
| 13 | source_uri | text |
| 14 | source_sha256 | character |
| 15 | created_at_utc | timestamp with time zone |

### `catalog.station_dim`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | station_sk | bigint |
| 2 | station_id | text |
| 3 | station_name | text |
| 4 | country_code | text |
| 5 | icao | text |
| 6 | valid_from | date |
| 7 | valid_to | date |
| 8 | latitude | double precision |
| 9 | longitude | double precision |
| 10 | elevation_m | double precision |
| 11 | distance_to_hko_km | double precision |
| 12 | bearing_from_hko_deg | double precision |
| 13 | tier | text |
| 14 | meteorological_role | text |
| 15 | research_note | text |
| 16 | dossier_version | text |
| 17 | audit_snapshot_id | text |

### `catalog.station_metadata_history`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | station_metadata_history_id | bigint |
| 2 | station_id | bigint |
| 3 | field_name | text |
| 4 | old_value | text |
| 5 | new_value | text |
| 6 | valid_from | date |
| 7 | valid_to | date |
| 8 | evidence_uri | text |
| 9 | metadata_sha256 | character |
| 10 | created_at_utc | timestamp with time zone |

### `catalog.variable`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | variable_id | bigint |
| 2 | semantic_variable | text |
| 3 | semantic_family | text |
| 4 | canonical_unit | text |
| 5 | value_role | text |
| 6 | created_at_utc | timestamp with time zone |

### `catalog.variable_selector_snapshot`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | selector_id | bigint |
| 2 | model_id | bigint |
| 3 | variable_id | bigint |
| 4 | semantic_variable | text |
| 5 | native_name | text |
| 6 | native_level | text |
| 7 | native_info | text |
| 8 | native_unit | text |
| 9 | introduced_at | date |
| 10 | retired_at | date |
| 11 | retrieved_at_utc | timestamp with time zone |
| 12 | source_sha256 | character |

### `catalog.weather_model`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | model_id | bigint |
| 2 | provider | text |
| 3 | model_code | text |
| 4 | domain | text |
| 5 | model_type | text |
| 6 | native_resolution | text |
| 7 | archive_start | date |
| 8 | archive_end | date |
| 9 | disposition | text |
| 10 | catalog_snapshot_sha256 | character |
| 11 | created_at_utc | timestamp with time zone |

### `catalog.source_registry`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | source_registry_id | bigint |
| 2 | dataset_id | text |
| 3 | source_id | text |
| 4 | source_file | text |
| 5 | repository_uri | text |
| 6 | original_local_path | text |
| 7 | file_type | text |
| 8 | physical_sha256 | character |
| 9 | byte_size | bigint |
| 10 | source_row_count | bigint |
| 11 | attribute_count | integer |
| 12 | data_min | timestamp with time zone |
| 13 | data_max | timestamp with time zone |
| 14 | metadata_min | timestamp with time zone |
| 15 | metadata_max | timestamp with time zone |
| 16 | disposition | text |
| 17 | db_layer | text |
| 18 | model_status | text |
| 19 | priority | text |
| 20 | ingestion_version | text |
| 21 | audit_snapshot_id | text |
| 22 | status | text |
| 23 | created_at_utc | timestamp with time zone |

### `diagnostic_physics.codex_audit_ds_02_hko_daily_climate_all_elements_hko_d_f7bb0017`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | station_or_domain | text |
| 10 | variable | text |
| 11 | unit | text |
| 12 | local_date | text |
| 13 | year | bigint |
| 14 | month | bigint |
| 15 | day | bigint |
| 16 | value | double precision |
| 17 | value_precision | double precision |
| 18 | completeness | text |
| 19 | parse_issue | text |
| 20 | availability_tier | text |
| 21 | operational_input_allowed | boolean |
| 22 | source_time_policy | text |

### `diagnostic_physics.codex_audit_ds_03_noaa_igra_upper_air_hkm00045004_noaa_a791906b`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | station_id | text |
| 10 | valid_at_utc | text |
| 11 | valid_at_hkt | text |
| 12 | nominal_hour_utc | bigint |
| 13 | latitude | double precision |
| 14 | longitude | double precision |
| 15 | availability_tier | text |
| 16 | operational_input_allowed | boolean |
| 17 | release_latency_proven | boolean |
| 18 | source_time_policy | text |
| 19 | key_level_count | bigint |
| 20 | temperature_c_1000hpa | double precision |
| 21 | relative_humidity_pct_1000hpa | double precision |
| 22 | dewpoint_depression_c_1000hpa | double precision |
| 23 | geopotential_height_m_1000hpa | double precision |
| 24 | wind_direction_deg_1000hpa | double precision |
| 25 | wind_speed_mps_1000hpa | double precision |
| 26 | temperature_c_850hpa | double precision |
| 27 | relative_humidity_pct_850hpa | double precision |
| 28 | dewpoint_depression_c_850hpa | double precision |
| 29 | geopotential_height_m_850hpa | double precision |
| 30 | wind_direction_deg_850hpa | double precision |
| 31 | wind_speed_mps_850hpa | double precision |
| 32 | temperature_c_700hpa | double precision |
| 33 | relative_humidity_pct_700hpa | double precision |
| 34 | dewpoint_depression_c_700hpa | double precision |
| 35 | geopotential_height_m_700hpa | double precision |
| 36 | wind_direction_deg_700hpa | double precision |
| 37 | wind_speed_mps_700hpa | double precision |
| 38 | temperature_c_500hpa | double precision |
| 39 | relative_humidity_pct_500hpa | double precision |
| 40 | dewpoint_depression_c_500hpa | double precision |
| 41 | geopotential_height_m_500hpa | double precision |
| 42 | wind_direction_deg_500hpa | double precision |
| 43 | wind_speed_mps_500hpa | double precision |
| 44 | temp_850_minus_500_c | double precision |
| 45 | temperature_c_300hpa | double precision |
| 46 | relative_humidity_pct_300hpa | double precision |
| 47 | dewpoint_depression_c_300hpa | double precision |
| 48 | geopotential_height_m_300hpa | double precision |
| 49 | wind_direction_deg_300hpa | double precision |
| 50 | wind_speed_mps_300hpa | double precision |
| 51 | temperature_c_200hpa | double precision |
| 52 | relative_humidity_pct_200hpa | double precision |
| 53 | dewpoint_depression_c_200hpa | double precision |
| 54 | geopotential_height_m_200hpa | double precision |
| 55 | wind_direction_deg_200hpa | double precision |
| 56 | wind_speed_mps_200hpa | double precision |
| 57 | temperature_c_925hpa | double precision |
| 58 | relative_humidity_pct_925hpa | double precision |
| 59 | dewpoint_depression_c_925hpa | double precision |
| 60 | geopotential_height_m_925hpa | double precision |
| 61 | wind_direction_deg_925hpa | double precision |
| 62 | wind_speed_mps_925hpa | double precision |
| 63 | temp_925_minus_850_c | double precision |

### `diagnostic_physics.codex_audit_ds_03_noaa_igra_upper_air_hkm00045004_noaa_f6023703`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | station_id | text |
| 10 | valid_at_utc | text |
| 11 | valid_at_hkt | text |
| 12 | nominal_hour_utc | bigint |
| 13 | latitude | double precision |
| 14 | longitude | double precision |
| 15 | availability_tier | text |
| 16 | operational_input_allowed | boolean |
| 17 | release_latency_proven | boolean |
| 18 | source_time_policy | text |
| 19 | level_type | bigint |
| 20 | elapsed_time_minutes | double precision |
| 21 | pressure_hpa | double precision |
| 22 | pressure_flag | text |
| 23 | geopotential_height_m | double precision |
| 24 | geopotential_flag | text |
| 25 | temperature_c | double precision |
| 26 | temperature_flag | text |
| 27 | relative_humidity_pct | double precision |
| 28 | relative_humidity_flag | text |
| 29 | dewpoint_depression_c | double precision |
| 30 | dewpoint_depression_flag | text |
| 31 | wind_direction_deg | double precision |
| 32 | wind_speed_mps | double precision |
| 33 | pressure_level_tag | text |

### `diagnostic_regime_labels.codex_audit_ds_06_hko_tropical_cyclone_best_track_hko__9d8b03ac`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | cyclone_name | text |
| 10 | valid_at_utc | text |
| 11 | valid_at_hkt | text |
| 12 | intensity | text |
| 13 | latitude | double precision |
| 14 | longitude | double precision |
| 15 | minimum_central_pressure_hpa | double precision |
| 16 | maximum_surface_wind_kt | double precision |
| 17 | jma_code | text |
| 18 | hko_code | text |
| 19 | availability_tier | text |
| 20 | operational_input_allowed | boolean |
| 21 | source_time_policy | text |

### `diagnostic_station_network.codex_audit_ds_04_noaa_isd_regional_surface_noaa_isd_c_688279e2`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | station_id | text |
| 7 | observed_at_utc | text |
| 8 | observed_at_hkt | text |
| 9 | report_type | text |
| 10 | latitude | double precision |
| 11 | longitude | double precision |
| 12 | elevation_m | double precision |
| 13 | wind_direction_deg | bigint |
| 14 | wind_speed_mps | double precision |
| 15 | air_temperature_c | double precision |
| 16 | dew_point_c | double precision |
| 17 | sea_level_pressure_hpa | double precision |
| 18 | temperature_quality_code | text |
| 19 | dew_point_quality_code | text |
| 20 | sea_level_pressure_quality_code | text |
| 21 | source_id | text |
| 22 | content_sha256 | text |
| 23 | raw_retrieved_at_utc | text |
| 24 | availability_tier | text |
| 25 | operational_input_allowed | boolean |
| 26 | source_time_policy | text |

### `diagnostic_station_network.codex_audit_ds_04_noaa_isd_regional_surface_noaa_isd_s_80f559e2`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | station_id | text |
| 7 | local_date | text |
| 8 | obs_count | bigint |
| 9 | latest_before_1500_hkt | text |
| 10 | air_temperature_c_latest_before_1500 | double precision |
| 11 | dew_point_c_latest_before_1500 | double precision |
| 12 | sea_level_pressure_hpa_latest_before_1500 | double precision |
| 13 | wind_direction_deg_latest_before_1500 | double precision |
| 14 | wind_speed_mps_latest_before_1500 | double precision |
| 15 | daily_air_temperature_min_c | double precision |
| 16 | daily_air_temperature_max_c | double precision |
| 17 | availability_tier | text |
| 18 | operational_input_allowed | boolean |

### `feature_safe.hko_t24_official_anchor`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | target_date | date |
| 2 | cutoff_utc | timestamp with time zone |
| 3 | official_tmin_c | double precision |
| 4 | official_tmax_c | double precision |
| 5 | forecast_range_c | double precision |
| 6 | source_era | text |
| 7 | source_product | text |
| 8 | issue_time_utc | timestamp with time zone |
| 9 | published_at_utc | timestamp with time zone |
| 10 | available_at_utc | timestamp with time zone |
| 11 | selected_source_row_id | text |
| 12 | selection_rule_version | text |
| 13 | quality_status | text |
| 14 | eligibility_status | text |

### `feature_safe.hko_target_history_pre2024`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | local_date | date |
| 2 | target_tmax_c | numeric |
| 3 | target_station | text |
| 4 | target_source_id | text |
| 5 | quality_status | text |

### `feature_safe.live_exact_vintage_catalog`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | live_exact_vintage_id | bigint |
| 2 | dataset_id | text |
| 3 | source_id | text |
| 4 | valid_time_utc | timestamp with time zone |
| 5 | issue_time_utc | timestamp with time zone |
| 6 | available_at_utc | timestamp with time zone |
| 7 | retrieved_at_utc | timestamp with time zone |
| 8 | eligibility_status | text |
| 9 | metadata | jsonb |

### `feature_store.feature_definition`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | feature_id | bigint |
| 2 | feature_name | text |
| 3 | feature_family | text |
| 4 | source_contract | text |
| 5 | eligibility_grade | text |
| 6 | value_unit | text |
| 7 | created_at_utc | timestamp with time zone |

### `feature_store.feature_value`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | snapshot_id | uuid |
| 2 | feature_id | bigint |
| 3 | location_id | bigint |
| 4 | value | double precision |
| 5 | value_json | jsonb |
| 6 | created_at_utc | timestamp with time zone |

### `feature_store.target_snapshot_manifest`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | snapshot_id | uuid |
| 2 | target_date | date |
| 3 | cutoff_utc | timestamp with time zone |
| 4 | cutoff_contract_version | text |
| 5 | builder_version | text |
| 6 | source_manifest_sha256 | character |
| 7 | feature_manifest_sha256 | character |
| 8 | created_at_utc | timestamp with time zone |

### `feature_store.h24n_tactical_nwp_feature_source`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | dataset_code | text |
| 2 | acquisition_version | text |
| 3 | target_date_hkt | date |
| 4 | cutoff_id | text |
| 5 | run_time_utc | timestamp with time zone |
| 6 | valid_time_utc | timestamp with time zone |
| 7 | lead_hours | numeric |
| 8 | location_code | text |
| 9 | requested_latitude | double precision |
| 10 | requested_longitude | double precision |
| 11 | returned_latitude | double precision |
| 12 | returned_longitude | double precision |
| 13 | returned_grid_distance_km | double precision |
| 14 | member_number | integer |
| 15 | temperature_2m_k | double precision |
| 16 | interval_tmax_2m_k | double precision |
| 17 | dewpoint_2m_k | double precision |
| 18 | relative_humidity_2m_pct | double precision |
| 19 | u_wind_10m_mps | double precision |
| 20 | v_wind_10m_mps | double precision |
| 21 | mslp_pa | double precision |
| 22 | low_cloud_pct | double precision |
| 23 | accumulated_precip_kg_m2 | double precision |
| 24 | downward_shortwave_w_m2 | double precision |
| 25 | net_shortwave_w_m2 | double precision |
| 26 | total_precip_m | double precision |
| 27 | shortwave_down_j_m2 | double precision |
| 28 | total_column_water_vapour_kg_m2 | double precision |
| 29 | pwat_kg_m2 | double precision |
| 30 | temperature_925_k | double precision |
| 31 | temperature_850_k | double precision |
| 32 | relative_humidity_700_pct | double precision |
| 33 | geopotential_height_500_m | double precision |
| 34 | raw_values_jsonb | jsonb |
| 35 | source_response_object_id | bigint |
| 36 | quality_status | text |
| 37 | created_at_utc | timestamp with time zone |

### `governance.availability_contract`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | availability_contract_id | bigint |
| 2 | dataset_id | text |
| 3 | source_id | text |
| 4 | contract_version | text |
| 5 | decision_cutoff_rule | text |
| 6 | valid_time_rule | text |
| 7 | issue_time_rule | text |
| 8 | available_at_rule | text |
| 9 | conservative_latency | interval |
| 10 | operationally_eligible | boolean |
| 11 | evidence_uri | text |
| 12 | approved_by | text |
| 13 | approved_at_utc | timestamp with time zone |

### `governance.availability_grade`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | grade_code | text |
| 2 | grade_rank | smallint |
| 3 | strict_eligible | boolean |
| 4 | description | text |

### `governance.feature_eligibility`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | feature_eligibility_id | bigint |
| 2 | dataset_id | text |
| 3 | source_file | text |
| 4 | attribute_name | text |
| 5 | eligibility_status | text |
| 6 | live_inference_allowed | boolean |
| 7 | reason | text |
| 8 | contract_version | text |

### `governance.gribstream_usage_constraint`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | constraint_id | text |
| 2 | source_code | text |
| 3 | constraint_kind | text |
| 4 | constraint_status | text |
| 5 | evidence_uri | text |
| 6 | operational_effect | text |
| 7 | created_at_utc | timestamp with time zone |

### `governance.leakage_test_result`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | leakage_test_result_id | bigint |
| 2 | test_name | text |
| 3 | status | text |
| 4 | evidence_uri | text |
| 5 | details | jsonb |
| 6 | executed_at_utc | timestamp with time zone |

### `governance.operational_contract`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | contract_version | text |
| 2 | target_station | text |
| 3 | target_variable | text |
| 4 | timezone_name | text |
| 5 | decision_cutoff_local_time | time without time zone |
| 6 | decision_day_offset | integer |
| 7 | cutoff_function | regproc |
| 8 | development_end_date | date |
| 9 | locked_validation_year | integer |
| 10 | final_historical_test_year | integer |
| 11 | post_final_holdout_year | integer |
| 12 | status | text |
| 13 | created_at_utc | timestamp with time zone |
| 14 | notes | text |

### `governance.parser_version`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | parser_version_id | text |
| 2 | parser_name | text |
| 3 | code_uri | text |
| 4 | code_sha256 | character |
| 5 | created_at_utc | timestamp with time zone |

### `governance.quality_issue`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | quality_issue_id | text |
| 2 | severity | text |
| 3 | dataset_id | text |
| 4 | source_table | text |
| 5 | attributes | text |
| 6 | evidence | text |
| 7 | required_action | text |
| 8 | current_status | text |
| 9 | remediation_implementation_path | text |
| 10 | validation_evidence_uri | text |
| 11 | resolution_timestamp | timestamp with time zone |
| 12 | resolution_commit | text |
| 13 | notes | text |

### `governance.quarantine_reason`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | reason_code | text |
| 2 | severity | text |
| 3 | description | text |

### `governance.schema_version`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | migration_version | text |
| 2 | applied_at_utc | timestamp with time zone |
| 3 | description | text |
| 4 | checksum_sha256 | text |

### `governance.sealed_period`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | sealed_period_id | text |
| 2 | local_date_start | date |
| 3 | local_date_end | date |
| 4 | label_schema | text |
| 5 | label_table | text |
| 6 | read_role | text |
| 7 | status | text |
| 8 | access_policy | text |
| 9 | created_at_utc | timestamp with time zone |

### `governance.table_load_contract`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | dataset_id | text |
| 2 | source_file | text |
| 3 | file_type | text |
| 4 | row_count | bigint |
| 5 | byte_size | bigint |
| 6 | attribute_count | integer |
| 7 | data_min | timestamp with time zone |
| 8 | data_max | timestamp with time zone |
| 9 | db_action | text |
| 10 | db_layer | text |
| 11 | model_status | text |
| 12 | priority | text |
| 13 | notes | text |
| 14 | audit_snapshot_id | text |
| 15 | contract_version | text |

### `governance.attribute_contract`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | attribute_contract_id | bigint |
| 2 | dataset_id | text |
| 3 | source_file | text |
| 4 | file_type | text |
| 5 | attribute | text |
| 6 | attribute_name | text |
| 7 | source_dtype | text |
| 8 | semantic_class | text |
| 9 | row_count | bigint |
| 10 | non_null_count | bigint |
| 11 | null_count | bigint |
| 12 | null_pct | double precision |
| 13 | storage_decision | text |
| 14 | db_layer | text |
| 15 | model_role | text |
| 16 | operational_status | text |
| 17 | quality_action | text |
| 18 | usefulness_score | smallint |
| 19 | rationale | text |
| 20 | profile_min | text |
| 21 | profile_max | text |
| 22 | audit_snapshot_id | text |
| 23 | contract_version | text |
| 24 | reconciliation_status | text |
| 25 | physical_destination | text |

### `ingestion.batch`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | batch_id | text |
| 2 | started_at_utc | timestamp with time zone |
| 3 | finished_at_utc | timestamp with time zone |
| 4 | status | text |
| 5 | code_commit | text |
| 6 | audit_snapshot_hash | character |
| 7 | dataset_root_uri | text |
| 8 | cutoff_rule_version | text |
| 9 | database_target_redacted | text |
| 10 | loader_version | text |
| 11 | command_line | text |
| 12 | host_metadata | jsonb |
| 13 | files_succeeded | integer |
| 14 | files_failed | integer |
| 15 | files_skipped | integer |
| 16 | files_resumed | integer |

### `ingestion.file_result`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | file_result_id | bigint |
| 2 | batch_id | text |
| 3 | source_file_id | bigint |
| 4 | source_file | text |
| 5 | expected_hash | character |
| 6 | observed_hash | character |
| 7 | expected_row_count | bigint |
| 8 | observed_row_count | bigint |
| 9 | expected_schema | jsonb |
| 10 | observed_schema | jsonb |
| 11 | load_action | text |
| 12 | started_at_utc | timestamp with time zone |
| 13 | finished_at_utc | timestamp with time zone |
| 14 | rows_staged | bigint |
| 15 | rows_inserted | bigint |
| 16 | rows_updated_versioned | bigint |
| 17 | rows_quarantined | bigint |
| 18 | rows_skipped_as_duplicate | bigint |
| 19 | status | text |
| 20 | error_text | text |
| 21 | reconciliation_artifact_uri | text |

### `ingestion.reconciliation`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | reconciliation_id | bigint |
| 2 | batch_id | text |
| 3 | reconciliation_scope | text |
| 4 | dataset_id | text |
| 5 | source_file | text |
| 6 | attribute_name | text |
| 7 | expected_disposition | text |
| 8 | actual_disposition | text |
| 9 | physical_destination | text |
| 10 | count_hash_evidence | jsonb |
| 11 | status | text |
| 12 | exception_explanation | text |

### `ingestion.row_rejection`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | rejection_id | bigint |
| 2 | batch_id | text |
| 3 | source_file_id | bigint |
| 4 | source_row_number | bigint |
| 5 | dataset_id | text |
| 6 | target_table | text |
| 7 | reason_code | text |
| 8 | reason_detail | text |
| 9 | raw_row_payload | jsonb |
| 10 | raw_content_hash | character |
| 11 | detected_at_utc | timestamp with time zone |
| 12 | repair_status | text |
| 13 | repair_lineage | text |

### `label_core.hko_daily_tmax`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | local_date | date |
| 2 | target_tmax_c | numeric |
| 3 | target_station | text |
| 4 | target_source_id | text |
| 5 | content_sha256 | character |
| 6 | retrieved_at_utc | timestamp with time zone |
| 7 | quality_status | text |
| 8 | source_file_id | bigint |
| 9 | ingestion_batch_id | text |

### `live.issued_forecast`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | issued_forecast_id | uuid |
| 2 | target_date | date |
| 3 | cutoff_utc | timestamp with time zone |
| 4 | issued_at_utc | timestamp with time zone |
| 5 | final_point_tmax_c | double precision |
| 6 | p10_c | double precision |
| 7 | p50_c | double precision |
| 8 | p90_c | double precision |
| 9 | system_version | text |
| 10 | snapshot_id | uuid |
| 11 | decision_log_json | jsonb |

### `live_exact_vintage.catalog`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | live_exact_vintage_id | bigint |
| 2 | dataset_id | text |
| 3 | source_file_id | bigint |
| 4 | source_id | text |
| 5 | valid_time_utc | timestamp with time zone |
| 6 | issue_time_utc | timestamp with time zone |
| 7 | available_at_utc | timestamp with time zone |
| 8 | retrieved_at_utc | timestamp with time zone |
| 9 | eligibility_status | text |
| 10 | metadata | jsonb |

### `live_exact_vintage.codex_audit_ds_07_hko_radar_satellite_lightning_nowcas_97d078a5`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | issue_time_hkt | text |
| 10 | ending_time_hkt | text |
| 11 | grid_cell_count | bigint |
| 12 | rainfall_mean_mm | double precision |
| 13 | rainfall_max_mm | double precision |
| 14 | rainfall_p95_mm | double precision |
| 15 | rain_area_fraction_gt_0mm | double precision |
| 16 | rain_area_fraction_ge_1mm | double precision |
| 17 | availability_tier | text |
| 18 | operational_input_allowed | boolean |

### `live_exact_vintage.codex_audit_ds_07_hko_radar_satellite_lightning_nowcas_eb458062`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | period | text |
| 10 | lightning_type | text |
| 11 | region | text |
| 12 | lightning_count | bigint |
| 13 | availability_tier | text |
| 14 | operational_input_allowed | boolean |

### `live_exact_vintage.codex_audit_ds_08_hko_marine_tide_coastal_waters_hko_l_be7bc027`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | tide_station | text |
| 10 | observed_at_hkt | text |
| 11 | height_m | double precision |
| 12 | availability_tier | text |
| 13 | operational_input_allowed | boolean |

### `live_exact_vintage.codex_audit_ds_08_hko_marine_tide_coastal_waters_hko_s_4bce52ff`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | update_time_hkt | text |
| 10 | location_name | text |
| 11 | wind_info | text |
| 12 | weather_description | text |
| 13 | sea_situation | text |
| 14 | availability_tier | text |
| 15 | operational_input_allowed | boolean |

### `live_nwp_anchor.codex_audit_ds_09_hko_arwf_station_forecasts_hko_arwf__e2cc97ac`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | station_code | text |
| 10 | latitude | double precision |
| 11 | longitude | double precision |
| 12 | model_time | bigint |
| 13 | last_modified | text |
| 14 | forecast_date | text |
| 15 | forecast_max_temperature_c | double precision |
| 16 | forecast_min_temperature_c | double precision |
| 17 | availability_tier | text |
| 18 | operational_input_allowed | boolean |
| 19 | source_time_policy | text |

### `nwp_core.point_value`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | model_run_id | bigint |
| 2 | valid_time_utc | timestamp with time zone |
| 3 | lead_minutes | integer |
| 4 | location_id | bigint |
| 5 | selector_id | bigint |
| 6 | member_number | integer |
| 7 | value | double precision |
| 8 | response_object_id | bigint |
| 9 | quality_status | text |
| 10 | created_at_utc | timestamp with time zone |

### `nwp_core.model_run`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | model_run_id | bigint |
| 2 | model_id | bigint |
| 3 | run_time_utc | timestamp with time zone |
| 4 | first_seen_at_utc | timestamp with time zone |
| 5 | documented_release_at_utc | timestamp with time zone |
| 6 | availability_grade | text |
| 7 | availability_contract_version | text |
| 8 | model_version | text |
| 9 | created_at_utc | timestamp with time zone |

### `nwp_core.point_value_default`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | model_run_id | bigint |
| 2 | valid_time_utc | timestamp with time zone |
| 3 | lead_minutes | integer |
| 4 | location_id | bigint |
| 5 | selector_id | bigint |
| 6 | member_number | integer |
| 7 | value | double precision |
| 8 | response_object_id | bigint |
| 9 | quality_status | text |
| 10 | created_at_utc | timestamp with time zone |

### `nwp_tactical.acquisition_chunk`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | chunk_id | text |
| 2 | acquisition_version | text |
| 3 | dataset_code | text |
| 4 | endpoint | text |
| 5 | time_selector | text |
| 6 | run_times_utc | ARRAY |
| 7 | min_lead_hours | integer |
| 8 | max_lead_hours | integer |
| 9 | location_policy | text |
| 10 | variable_bundle_id | text |
| 11 | member_policy | text |
| 12 | members | ARRAY |
| 13 | expected_rows | integer |
| 14 | expected_credits | integer |
| 15 | request_json | jsonb |
| 16 | request_sha256 | character |
| 17 | status | text |
| 18 | raw_object_uri | text |
| 19 | response_sha256 | character |
| 20 | http_status | integer |
| 21 | row_count | integer |
| 22 | error_class | text |
| 23 | error_message | text |
| 24 | created_at_utc | timestamp with time zone |
| 25 | started_at_utc | timestamp with time zone |
| 26 | completed_at_utc | timestamp with time zone |

### `nwp_tactical.forecast_wide`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | dataset_code | text |
| 2 | acquisition_version | text |
| 3 | target_date_hkt | date |
| 4 | cutoff_id | text |
| 5 | run_time_utc | timestamp with time zone |
| 6 | valid_time_utc | timestamp with time zone |
| 7 | lead_hours | numeric |
| 8 | location_code | text |
| 9 | requested_latitude | double precision |
| 10 | requested_longitude | double precision |
| 11 | returned_latitude | double precision |
| 12 | returned_longitude | double precision |
| 13 | returned_grid_distance_km | double precision |
| 14 | member_number | integer |
| 15 | temperature_2m_k | double precision |
| 16 | interval_tmax_2m_k | double precision |
| 17 | dewpoint_2m_k | double precision |
| 18 | relative_humidity_2m_pct | double precision |
| 19 | u_wind_10m_mps | double precision |
| 20 | v_wind_10m_mps | double precision |
| 21 | mslp_pa | double precision |
| 22 | low_cloud_pct | double precision |
| 23 | accumulated_precip_kg_m2 | double precision |
| 24 | downward_shortwave_w_m2 | double precision |
| 25 | net_shortwave_w_m2 | double precision |
| 26 | total_precip_m | double precision |
| 27 | shortwave_down_j_m2 | double precision |
| 28 | total_column_water_vapour_kg_m2 | double precision |
| 29 | pwat_kg_m2 | double precision |
| 30 | temperature_925_k | double precision |
| 31 | temperature_850_k | double precision |
| 32 | relative_humidity_700_pct | double precision |
| 33 | geopotential_height_500_m | double precision |
| 34 | raw_values_jsonb | jsonb |
| 35 | source_response_object_id | bigint |
| 36 | quality_status | text |
| 37 | created_at_utc | timestamp with time zone |

### `nwp_tactical.location_stencil`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | location_code | text |
| 2 | cutoff_id | text |
| 3 | location_role | text |
| 4 | latitude | double precision |
| 5 | longitude | double precision |
| 6 | lat_offset_deg | double precision |
| 7 | lon_offset_deg | double precision |
| 8 | description | text |
| 9 | created_at_utc | timestamp with time zone |

### `nwp_tactical.model_plan`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | dataset_code | text |
| 2 | priority | text |
| 3 | stage | text |
| 4 | endpoint | text |
| 5 | archive_run_start_utc | timestamp with time zone |
| 6 | archive_run_end_utc | timestamp with time zone |
| 7 | target_date_start | date |
| 8 | target_date_end | date |
| 9 | exact_cycle_template | text |
| 10 | min_lead_hours | integer |
| 11 | max_lead_hours | integer |
| 12 | expected_native_step_hours | integer |
| 13 | expected_valid_steps_per_run | integer |
| 14 | location_policy | text |
| 15 | member_policy | text |
| 16 | availability_grade | text |
| 17 | promotion_status | text |
| 18 | expected_wide_rows | bigint |
| 19 | approximate_credits | bigint |
| 20 | notes | text |
| 21 | updated_at_utc | timestamp with time zone |

### `nwp_tactical.raw_response_object`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | response_object_id | bigint |
| 2 | chunk_id | text |
| 3 | object_uri | text |
| 4 | byte_size | bigint |
| 5 | sha256 | character |
| 6 | content_type | text |
| 7 | retrieved_at_utc | timestamp with time zone |
| 8 | row_count | integer |

### `nwp_tactical.validation_issue`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | validation_issue_id | bigint |
| 2 | chunk_id | text |
| 3 | dataset_code | text |
| 4 | issue_class | text |
| 5 | issue_severity | text |
| 6 | evidence_json | jsonb |
| 7 | created_at_utc | timestamp with time zone |

### `nwp_tactical.variable_plan`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | dataset_code | text |
| 2 | alias | text |
| 3 | native_name | text |
| 4 | native_level | text |
| 5 | native_info | text |
| 6 | required | boolean |
| 7 | variable_role | text |
| 8 | canonical_unit | text |
| 9 | notes | text |

### `object_catalog.asset`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | asset_id | bigint |
| 2 | asset_uri | text |
| 3 | original_local_path | text |
| 4 | content_sha256 | character |
| 5 | byte_size | bigint |
| 6 | media_type | text |
| 7 | dataset_id | text |
| 8 | source_file_id | bigint |
| 9 | asset_role | text |
| 10 | extraction_status | text |
| 11 | metadata | jsonb |
| 12 | registered_at_utc | timestamp with time zone |

### `operational_anchor.hko_t24_official_anchor_rows`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | anchor_row_id | bigint |
| 2 | target_date | date |
| 3 | cutoff_utc | timestamp with time zone |
| 4 | forecast_min_c | double precision |
| 5 | forecast_max_c | double precision |
| 6 | forecast_range_c | double precision |
| 7 | source_era | text |
| 8 | source_product | text |
| 9 | issue_time_utc | timestamp with time zone |
| 10 | published_at_utc | timestamp with time zone |
| 11 | available_at_utc | timestamp with time zone |
| 12 | selected_source_row_id | text |
| 13 | selection_rule_version | text |
| 14 | quality_status | text |
| 15 | eligibility_status | text |
| 16 | source_file_id | bigint |
| 17 | ingestion_batch_id | text |

### `public.hko_historical_forecasts_2000_2026`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | bulletin_id | text |
| 2 | source | text |
| 3 | source_url | text |
| 4 | product_type | text |
| 5 | title | text |
| 6 | index_date | date |
| 7 | snapshot_at_hkt | timestamp without time zone |
| 8 | snapshot_at_utc | timestamp with time zone |
| 9 | issue_at_hkt | timestamp without time zone |
| 10 | issue_at_utc | timestamp with time zone |
| 11 | issue_parse_method | text |
| 12 | target_date | date |
| 13 | target_issue_lead_days | integer |
| 14 | target_date_confidence | text |
| 15 | forecast_min_c | double precision |
| 16 | forecast_max_c | double precision |
| 17 | forecast_range_c | double precision |
| 18 | forecast_midpoint_c | double precision |
| 19 | has_target_date | boolean |
| 20 | has_forecast_min | boolean |
| 21 | has_forecast_max | boolean |
| 22 | has_forecast_minmax | boolean |
| 23 | temperature_valid | boolean |
| 24 | usable_local_tmax_forecast | boolean |
| 25 | row_quality_status | text |
| 26 | temperature_text | text |
| 27 | stale_snapshot_flag | boolean |
| 28 | stale_hours | double precision |
| 29 | parse_status | text |
| 30 | parse_notes | text |
| 31 | full_text | text |
| 32 | raw_sha256 | text |
| 33 | raw_path | text |
| 34 | source_archive_path | text |
| 35 | source_archive_mtime_utc | timestamp with time zone |
| 36 | ingested_at_utc | timestamp with time zone |

### `quarantine.rejected_payload`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | rejected_payload_id | bigint |
| 2 | request_id | uuid |
| 3 | response_object_id | bigint |
| 4 | rejection_class | text |
| 5 | rejection_reason | text |
| 6 | evidence_json | jsonb |
| 7 | created_at_utc | timestamp with time zone |

### `raw_audit.acquisition_request`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | request_id | uuid |
| 2 | provider | text |
| 3 | model_code | text |
| 4 | endpoint | text |
| 5 | canonical_request_json | jsonb |
| 6 | request_sha256 | character |
| 7 | status | text |
| 8 | attempt_count | integer |
| 9 | started_at_utc | timestamp with time zone |
| 10 | completed_at_utc | timestamp with time zone |
| 11 | error_class | text |
| 12 | error_message | text |
| 13 | created_at_utc | timestamp with time zone |

### `raw_audit.codex_audit_ds_01_hko_daily_tmax_target_hko_daily_extr_23cb54da`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | source_id | text |
| 7 | content_sha256 | text |
| 8 | raw_retrieved_at_utc | text |
| 9 | local_date | text |
| 10 | year | bigint |
| 11 | month | double precision |
| 12 | day | double precision |
| 13 | absolute_daily_max_c | double precision |
| 14 | value_precision | double precision |
| 15 | completeness | text |
| 16 | parse_issue | text |
| 17 | availability_tier | text |
| 18 | operational_input_allowed | boolean |
| 19 | source_time_policy | text |

### `raw_audit.response_object`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | response_object_id | bigint |
| 2 | request_id | uuid |
| 3 | object_uri | text |
| 4 | byte_size | bigint |
| 5 | sha256 | character |
| 6 | content_type | text |
| 7 | retrieved_at_utc | timestamp with time zone |
| 8 | first_seen_at_utc | timestamp with time zone |
| 9 | row_count | bigint |

### `research.expert_oof_prediction`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | target_date | date |
| 2 | frame_id | text |
| 3 | fold_id | text |
| 4 | expert_id | text |
| 5 | point_forecast_c | double precision |
| 6 | predicted_abs_error_c | double precision |
| 7 | model_artifact_sha256 | character |
| 8 | snapshot_id | uuid |
| 9 | created_at_utc | timestamp with time zone |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_184a0162`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | fold_id | text |
| 7 | model_id | text |
| 8 | n | bigint |
| 9 | first_date | text |
| 10 | last_date | text |
| 11 | mae | double precision |
| 12 | rmse | double precision |
| 13 | median_abs_error | double precision |
| 14 | bias | double precision |
| 15 | crps_normal | double precision |
| 16 | coverage_80 | double precision |
| 17 | coverage_90 | double precision |
| 18 | baseline_mae | double precision |
| 19 | baseline_crps | double precision |
| 20 | mae_improvement_vs_baseline | double precision |
| 21 | crps_improvement_vs_baseline | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_2bee51a4`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | model_id | text |
| 7 | n | bigint |
| 8 | first_date | text |
| 9 | last_date | text |
| 10 | mae | double precision |
| 11 | rmse | double precision |
| 12 | median_abs_error | double precision |
| 13 | bias | double precision |
| 14 | crps_normal | double precision |
| 15 | coverage_80 | double precision |
| 16 | coverage_90 | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_2ed3981b`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | feature | text |
| 7 | n | bigint |
| 8 | first_date | text |
| 9 | last_date | text |
| 10 | pearson_with_target | double precision |
| 11 | spearman_with_target | double precision |
| 12 | feature_mean | double precision |
| 13 | feature_std | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_38e9e3a0`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | feature | text |
| 7 | n | bigint |
| 8 | first_date | text |
| 9 | last_date | text |
| 10 | pearson_with_target | double precision |
| 11 | spearman_with_target | double precision |
| 12 | feature_mean | double precision |
| 13 | feature_std | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_4df989ab`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | model_id | text |
| 7 | n | bigint |
| 8 | first_date | text |
| 9 | last_date | text |
| 10 | mae | double precision |
| 11 | rmse | double precision |
| 12 | median_abs_error | double precision |
| 13 | bias | double precision |
| 14 | crps_normal | double precision |
| 15 | coverage_80 | double precision |
| 16 | coverage_90 | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_52683771`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | fold_id | text |
| 7 | model_id | text |
| 8 | n | bigint |
| 9 | first_date | text |
| 10 | last_date | text |
| 11 | mae | double precision |
| 12 | rmse | double precision |
| 13 | median_abs_error | double precision |
| 14 | bias | double precision |
| 15 | crps_normal | double precision |
| 16 | coverage_80 | double precision |
| 17 | coverage_90 | double precision |
| 18 | baseline_mae | double precision |
| 19 | baseline_crps | double precision |
| 20 | mae_improvement_vs_baseline | double precision |
| 21 | crps_improvement_vs_baseline | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_75354da8`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | model_id | text |
| 7 | n | bigint |
| 8 | first_date | text |
| 9 | last_date | text |
| 10 | mae | double precision |
| 11 | rmse | double precision |
| 12 | median_abs_error | double precision |
| 13 | bias | double precision |
| 14 | crps_normal | double precision |
| 15 | coverage_80 | double precision |
| 16 | coverage_90 | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_8894ab84`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | feature | text |
| 7 | n | bigint |
| 8 | first_date | text |
| 9 | last_date | text |
| 10 | pearson_with_target | double precision |
| 11 | spearman_with_target | double precision |
| 12 | feature_mean | double precision |
| 13 | feature_std | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_9c7c6556`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | fold_id | text |
| 7 | model_id | text |
| 8 | n | bigint |
| 9 | first_date | text |
| 10 | last_date | text |
| 11 | mae | double precision |
| 12 | rmse | double precision |
| 13 | median_abs_error | double precision |
| 14 | bias | double precision |
| 15 | crps_normal | double precision |
| 16 | coverage_80 | double precision |
| 17 | coverage_90 | double precision |
| 18 | baseline_mae | double precision |
| 19 | baseline_crps | double precision |
| 20 | mae_improvement_vs_baseline | double precision |
| 21 | crps_improvement_vs_baseline | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_b10ef890`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | model_id | text |
| 7 | n | bigint |
| 8 | first_date | text |
| 9 | last_date | text |
| 10 | mae | double precision |
| 11 | rmse | double precision |
| 12 | median_abs_error | double precision |
| 13 | bias | double precision |
| 14 | crps_normal | double precision |
| 15 | coverage_80 | double precision |
| 16 | coverage_90 | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_b6697d99`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | feature | text |
| 7 | n | bigint |
| 8 | first_date | text |
| 9 | last_date | text |
| 10 | pearson_with_target | double precision |
| 11 | spearman_with_target | double precision |
| 12 | feature_mean | double precision |
| 13 | feature_std | double precision |

### `research_metrics.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_cfa16ade`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | fold_id | text |
| 7 | model_id | text |
| 8 | n | bigint |
| 9 | first_date | text |
| 10 | last_date | text |
| 11 | mae | double precision |
| 12 | rmse | double precision |
| 13 | median_abs_error | double precision |
| 14 | bias | double precision |
| 15 | crps_normal | double precision |
| 16 | coverage_80 | double precision |
| 17 | coverage_90 | double precision |
| 18 | baseline_mae | double precision |
| 19 | baseline_crps | double precision |
| 20 | mae_improvement_vs_baseline | double precision |
| 21 | crps_improvement_vs_baseline | double precision |

### `research_oof_predictions.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_0ea616a5`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | target_date | timestamp with time zone |
| 7 | target_tmax_c | double precision |
| 8 | year | bigint |
| 9 | month | bigint |
| 10 | research_id | text |
| 11 | fold_id | text |
| 12 | model_id | text |
| 13 | model_family | text |
| 14 | is_control | boolean |
| 15 | training_start | timestamp with time zone |
| 16 | training_end | timestamp with time zone |
| 17 | training_rows | bigint |
| 18 | feature_count | bigint |
| 19 | point_forecast | double precision |
| 20 | distribution_sigma_c | double precision |
| 21 | q05 | double precision |
| 22 | q10 | double precision |
| 23 | q25 | double precision |
| 24 | q50 | double precision |
| 25 | q75 | double precision |
| 26 | q90 | double precision |
| 27 | q95 | double precision |

### `research_oof_predictions.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_38d58a4a`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | target_date | timestamp with time zone |
| 7 | target_tmax_c | double precision |
| 8 | year | bigint |
| 9 | month | bigint |
| 10 | research_id | text |
| 11 | fold_id | text |
| 12 | model_id | text |
| 13 | model_family | text |
| 14 | is_control | boolean |
| 15 | training_start | timestamp with time zone |
| 16 | training_end | timestamp with time zone |
| 17 | training_rows | bigint |
| 18 | feature_count | bigint |
| 19 | point_forecast | double precision |
| 20 | distribution_sigma_c | double precision |
| 21 | q05 | double precision |
| 22 | q10 | double precision |
| 23 | q25 | double precision |
| 24 | q50 | double precision |
| 25 | q75 | double precision |
| 26 | q90 | double precision |
| 27 | q95 | double precision |

### `research_oof_predictions.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_9be0ab50`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | target_date | timestamp with time zone |
| 7 | target_tmax_c | double precision |
| 8 | year | bigint |
| 9 | month | bigint |
| 10 | research_id | text |
| 11 | fold_id | text |
| 12 | model_id | text |
| 13 | model_family | text |
| 14 | is_control | boolean |
| 15 | training_start | timestamp with time zone |
| 16 | training_end | timestamp with time zone |
| 17 | training_rows | bigint |
| 18 | feature_count | bigint |
| 19 | point_forecast | double precision |
| 20 | distribution_sigma_c | double precision |
| 21 | q05 | double precision |
| 22 | q10 | double precision |
| 23 | q25 | double precision |
| 24 | q50 | double precision |
| 25 | q75 | double precision |
| 26 | q90 | double precision |
| 27 | q95 | double precision |

### `research_oof_predictions.codex_audit_ds_12_hkg_t24_robust_experiment_outputs_hk_ce701451`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | ingest_source_file | text |
| 2 | ingest_source_file_id | bigint |
| 3 | ingest_source_row_number | bigint |
| 4 | ingested_at_utc | timestamp with time zone |
| 5 | ingestion_batch_id | text |
| 6 | target_date | timestamp with time zone |
| 7 | target_tmax_c | double precision |
| 8 | year | bigint |
| 9 | month | bigint |
| 10 | research_id | text |
| 11 | fold_id | text |
| 12 | model_id | text |
| 13 | model_family | text |
| 14 | is_control | boolean |
| 15 | training_start | timestamp with time zone |
| 16 | training_end | timestamp with time zone |
| 17 | training_rows | bigint |
| 18 | feature_count | bigint |
| 19 | point_forecast | double precision |
| 20 | distribution_sigma_c | double precision |
| 21 | q05 | double precision |
| 22 | q10 | double precision |
| 23 | q25 | double precision |
| 24 | q50 | double precision |
| 25 | q75 | double precision |
| 26 | q90 | double precision |
| 27 | q95 | double precision |

### `sealed_confirmation.hko_daily_tmax`

| ordinal | column | data_type |
| --- | --- | --- |
| 1 | local_date | date |
| 2 | target_tmax_c | numeric |
| 3 | target_station | text |
| 4 | target_source_id | text |
| 5 | content_sha256 | character |
| 6 | retrieved_at_utc | timestamp with time zone |
| 7 | quality_status | text |
| 8 | source_file_id | bigint |
| 9 | ingestion_batch_id | text |

## Bottom-line implementation notes

- HKO labels from 1884 through 2023 are present in `label_core.hko_daily_tmax`; 2024+ labels are separated in `sealed_confirmation.hko_daily_tmax`.
- The official forecast archive is present in `public.hko_historical_forecasts_2000_2026`, with 115,795 usable local min/max rows and 9,667 usable target dates, but the current feature-safe official anchor view is empty and must be populated or replaced by a tested extraction query before modeling.
- The tactical GribStream backfill is present in `nwp_tactical.forecast_wide` with the known full-run plus 933 older smoke rows; model code must filter by raw-object source scope.
- Diagnostic tables such as HKO daily climate, ISD, IGRA, and TC best track are queryable and valuable, but their strategy status remains diagnostic/proxy unless exact availability and quality gates are cleared.
- Live exact-vintage radar/marine/ARWF sources exist as small samples and should be treated as prospective/shadow sources until enough history accumulates.
- Research metrics and OOF tables are evidence stores; do not use old feature matrices or OOF predictions as canonical production input unless their generation contract is revalidated.
