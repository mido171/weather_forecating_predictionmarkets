# HKG T24 Non-Minute Source Normalization

Generated: `2026-06-20T11:18:13.321141Z`

This pass parsed and normalized raw source families without running predictive experiments, validation scoring, Polymarket logic, backtesting, or ML.

- data root: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data`
- user exclusion honored: short HKO minute/snapshot historical archives starting around 2020/2021 were skipped.
- raw archive objects were not modified.
- operational use still requires each table's `operational_input_allowed` and availability fields to pass the as-of contract.

## Normalized Tables

| Table | Rows | Range | Status | Path | Notes |
|---|---:|---|---|---|---|
| hko_daily_climate_elements | 556,399 | 1884-01-01 to 2026-05-31 | parsed | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\hko_daily_climate_elements.parquet` | Long-history finalized daily HKO climate rows. Target-day daily elements are not operational predictors. |
| hko_daily_extract_tmax_payload_rows | 49,628 | 1884-01-01 to 2026-06-17 | parsed | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\hko_daily_extract_tmax_payload_rows.parquet` | Annual/monthly Daily Extract payload rows normalized as target-side publication evidence. |
| noaa_igra_hkm00045004_key_pressure_levels | 477,514 | 1949-06-01T22:00:00Z to 2026-06-17T18:00:00Z | parsed_key_levels | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\noaa_igra_hkm00045004_key_pressure_levels.parquet` | Key pressure levels only; full level dump omitted to keep derived tables tractable. |
| noaa_igra_hkm00045004_sounding_features | 88,407 | 1949-06-01T22:00:00Z to 2026-06-17T18:00:00Z | parsed_proxy_features | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\silver\source_normalized_non_minute\noaa_igra_hkm00045004_sounding_features.parquet` | Upper-air profile features parsed, but operational release latency remains unproven and fails closed. |
| noaa_isd_core_observations | 4,029,291 | 1945-12-01 to 2025-08-25 | parsed_streaming | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\noaa_isd_core_observations.parquet` | Core hourly observations parsed from NOAA ISD station-year gzip files. |
| noaa_isd_station_day_cutoff_summary | 317,489 | 1945-12-01 to 2025-08-25 | parsed_proxy_features | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\silver\source_normalized_non_minute\noaa_isd_station_day_cutoff_summary.parquet` | Daily station summaries include latest observation before 15:00 HKT, but archive is proxy-limited. |
| hko_historical_rss_items | 349,206 | 2020-06-01T00:03:00+08:00 to 2026-06-18T23:08:00+08:00 | parsed_vintages | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\hko_historical_rss_items.parquet` | All downloaded HKO historical RSS XML archive entries normalized as forecast/warning vintages. |
| hko_historical_rss_temperature_forecasts | 66,612 | 2020-06-01 to 2026-06-27 | parsed_forecast_temperatures | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\silver\source_normalized_non_minute\hko_historical_rss_temperature_forecasts.parquet` | English local and 9-day RSS forecast temperature ranges extracted for MOS-style later experiments. |
| hko_tropical_cyclone_best_track | 26,189 | 1985-01-06T06:00:00Z to 2024-12-25T06:00:00Z | parsed_retrospective | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\hko_tropical_cyclone_best_track.parquet` | Best-track rows are parsed for mechanism/regime labels only, not operational inputs. |
| hko_arwf_station_daily_forecasts | 530 | 20260619 to 20260628 | parsed_live_vintages | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\silver\source_normalized_non_minute\hko_arwf_station_daily_forecasts.parquet` | ARWF station forecast JSON snapshots normalized. |
| hko_gridded_rainfall_nowcast_summary | 41 | 2026-06-19T12:36:00+08:00 to 2026-06-20T15:36:00+08:00 | parsed_summary | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\silver\source_normalized_non_minute\hko_gridded_rainfall_nowcast_summary.parquet` | Nowcast grid summarized by vintage; individual grid cells are not expanded. |
| hko_lightning_counts_latest | 102 | 202606191100-202606191159 to 202606201400-202606201459 | parsed_live_vintages | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\hko_lightning_counts_latest.parquet` | Live lightning count snapshots normalized. |
| hko_latest_tidal_information | 105 | 2026-06-19T12:45:00+08:00 to 2026-06-20T15:50:00+08:00 | parsed_live_vintages | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\hko_latest_tidal_information.parquet` | Latest tidal rows normalized. |
| hko_south_china_coastal_waters_bulletin | 30 | 2026-06-19T12:30:00+08:00 to 2026-06-20T12:30:00+08:00 | parsed_live_vintages | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\hko_south_china_coastal_waters_bulletin.parquet` | Marine bulletin area forecasts normalized. |
| hko_radar_manifest_frames | 80 | 2026-06-19T10:54:00+08:00 to 2026-06-19T12:48:00+08:00 | parsed_metadata | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\hko_radar_manifest_frames.parquet` | Radar manifest frame times normalized; image pixels are not decoded. |
| hko_satellite_image_inventory | 4,589 | 2026-06-17T08:15:00+08:00 to 2026-06-20T07:30:00+08:00 | metadata_only | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\hko_satellite_image_inventory.parquet` | Satellite raw imagery normalized to metadata inventory; pixels not decoded in this pass. |
| ncep_operational_grib2_inventory | 3,400 | to | metadata_only | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\ncep_operational_grib2_inventory.parquet` | NCEP GFS/GEFS GRIB2 subset inventory normalized; fields not decoded in this pass. |
| static_geospatial_package_inventory | 60 | to | metadata_only | `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\bronze\source_normalized_non_minute\static_geospatial_package_inventory.parquet` | Static geospatial raw packages inventoried; terrain/coast/LUHK station buffers still require geospatial decoding. |

## Explicitly Skipped Short Minute/Snapshot Archives

| Source | Success rows | Reason |
|---|---:|---|
| datagov_hko_historical_latest_10min_wind_archive | 78 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_15min_uvindex_archive | 90 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_1min_humidity_archive | 90 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_1min_humidity_listing | 1 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_1min_pressure_archive | 78 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_1min_pressure_listing | 1 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_1min_solar_archive | 78 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_1min_solar_listing | 1 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_1min_temperature_archive | 90 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_1min_temperature_listing | 1 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |
| datagov_hko_historical_latest_since_midnight_maxmin_archive | 90 | short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass |

## Important Fail-Closed Notes

- IGRA upper-air profiles are parsed, but release latency is not proven, so those rows are proxy-limited until the R14 availability contract is completed.
- NOAA ISD station observations are parsed from the quality-controlled archive, so they are useful for long-history proxy/mechanism work but not exact real-time vintages by default.
- HKO RSS forecast archives are exact DATA.GOV.HK historical archive entries and are the strongest current official forecast-vintage source.
- Satellite, radar-image, static geospatial, and GRIB2 products are normalized to metadata inventories here; pixel/raster/GRIB field decoding is a separate parser step.
