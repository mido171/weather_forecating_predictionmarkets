# Data Quality Register

All audit issues remain visible in this file. A dataset being saved to the database does not make it automatically safe for operational features.

## Issue counts

| severity | issue_count |
| --- | --- |
| CRITICAL | 5 |
| HIGH | 10 |
| MEDIUM | 7 |

## Open issues

| severity | dataset_id | source_table | attributes | evidence | required_action |
| --- | --- | --- | --- | --- | --- |
| CRITICAL | 03_noaa_igra_upper_air_hkm00045004 | both tables | Multiple meteorological fields | IGRA raw/derived values contain sentinel-like -888.8 and -8888 values; relative humidity reaches 1000. | Do not load derived feature table as clean. Map documented source missing codes to NULL, apply correct scale, rerun all sounding features, and compare row counts/ranges. |
| CRITICAL | 04_noaa_isd_regional_surface | both tables | wind_direction_deg; wind_direction_deg_latest_before_1500 | Both fields are constant at 20 degrees across millions/hundreds of thousands of rows. | Treat current wind direction and all dependent u/v/directional features as invalid. Fix extractor and rebuild downstream matrices. |
| CRITICAL | 10_ncep_operational_grib_inventory | ncep_operational_grib2_inventory.parquet | cycle_date; cycle_hour_utc | Both cycle fields are empty for all 3,400 rows. | Rebuild inventory from filename and GRIB metadata with cycle_time_utc and valid_time_utc before any modelling. |
| CRITICAL | 05_hko_historical_rss_forecasts | hko_press_archive_forecast_days.parquet | forecast_max_c; target_issue_lead_days | Forecast maximum reaches 310 degreesC and lead days range from -4382 to 370. | Quarantine invalid rows; use only scoreable_row_valid, target_date_plausible and explicit T+24 lead checks. Repair parser. |
| CRITICAL | All timestamped datasets | multiple | *_hkt versus *_utc | Profile min/max values are frequently identical for HKT and UTC columns. | Audit actual timezone semantics. Store one canonical timestamptz UTC plus explicitly derived HKT; never infer availability from a mislabeled naive timestamp. |
| HIGH | 04_noaa_isd_regional_surface | noaa_isd_core_observations.parquet | latitude; longitude | Profile includes latitude 0 and longitude -114.283/144.2 despite a regional Hong Kong/South China network. | Use date-effective NOAA station history metadata, not row-level coordinates, and quarantine impossible station metadata. |
| HIGH | 04_noaa_isd_regional_surface | noaa_isd_station_day_cutoff_summary.parquet | daily_air_temperature_min_c; daily_air_temperature_max_c | Column names do not prove values were truncated at 15:00 HKT. | Reject as predictor until aggregation code proves no post-cutoff observations entered. |
| HIGH | 03_noaa_igra_upper_air_hkm00045004 | both tables | source_id | Period-of-record and year-to-date sources coexist and can overlap. | Deduplicate by station, valid time, pressure level and source priority before feature generation. |
| HIGH | 02_hko_daily_climate_all_elements | hko_daily_climate_elements.parquet | value; parse_issue | 6,916 'Trace' values are currently nonnumeric; 7,389 are missing; 5 invalid dates exist. | Preserve trace_flag separately and apply variable-specific trace policy; quarantine invalid dates. |
| HIGH | 01_hko_daily_tmax_target | two tables | local_date; target values | Payload table has 49,628 rows versus 49,459 canonical labels and includes monthly/yearly sources plus one parse failure. | Use labels table as canonical. Deduplicate/reconcile payload dates and source overlap only for audit. |
| HIGH | 05_hko_historical_rss_forecasts | archive coverage tables | raw_detail_coverage_pct | Coverage is incomplete and highly era/product dependent. | Store coverage metadata and include source-era/frame controls in every evaluation; do not treat missing archive days as random. |
| HIGH | 07_hko_radar_satellite_lightning_nowcast | hko_gridded_rainfall_nowcast_summary.parquet | rainfall_max_mm | Maximum is 544.88 mm while sample covers only 41 snapshots. | Verify accumulation interval, grid fill values and units before use. |
| HIGH | 07_hko_radar_satellite_lightning_nowcast | hko_satellite_image_inventory.parquet | image_time_hkt | 132 of 4,589 non-null image times fail datetime parsing. | Quarantine malformed entries and distinguish page/JS manifests from actual images. |
| HIGH | 09_hko_arwf_station_forecasts | hko_arwf_station_daily_forecasts.parquet | model_time; last_modified | Model time is numeric and constant; last_modified is string; only one cycle exists. | Parse cycle/issue/valid timestamps, calculate lead hours, and collect many cycles before scoring. |
| HIGH | 12_hkg_t24_robust_experiment_outputs | feature matrices | IGRA/UA/ISD/daily feature families | Large portions are timestamp-blocked and some depend on corrupted raw fields. | Keep as research artifacts only; recompute any promoted feature from repaired canonical sources. |
| MEDIUM | 07_hko_radar_satellite_lightning_nowcast | hko_lightning_counts_latest.parquet | lightning_count | All 102 values are zero. | Retain feed but current sample has no information gain; continue collection and monitor variance. |
| MEDIUM | 08_hko_marine_tide_coastal_waters | hko_latest_tidal_information.parquet | height_m | 14.3% of tide heights are missing. | Retain with station-specific availability flags; do not impute indiscriminately. |
| MEDIUM | 09_hko_arwf_station_forecasts | hko_arwf_station_daily_forecasts.parquet | forecast_min/max_temperature_c | 28.3% missing. | Determine whether missingness is station capability, lead, or parser behavior; model only eligible station/lead cells. |
| MEDIUM | 11_static_geospatial_inventory | land-use assets | 2018-2024 land-use rasters | Using recent land-use maps for early historical rows creates temporal context leakage/misrepresentation. | Use terrain/coastline as static; make land use date-effective or reserve it for modern-era models. |
| MEDIUM | 05_hko_historical_rss_forecasts | duplicate CSV/Parquet outputs | all columns | Two major normalized tables exist in both CSV and Parquet. | Load Parquet only; register CSV as export artifact. |
| MEDIUM | 05_hko_historical_rss_forecasts | bulletins_offline | snapshot_at_hkt; stale_hours; retrieval_id; attempted_at_utc | These columns are entirely null. | Omit current physical columns or populate properly in future; do not pretend they add information. |
| MEDIUM | 07/08/09 live feeds | multiple | all weather attributes | Current history is only one to three days/one ARWF cycle. | Store and collect, but do not estimate model skill or feature importance from the current sample. |

## Global or multi-dataset issues

No global issue is recorded.
