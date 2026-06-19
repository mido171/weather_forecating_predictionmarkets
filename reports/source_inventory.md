# Source Inventory

Generated from `config/data_sources.yaml`. Endpoint implementation and source-contract status must still be verified individually.

| ID | Provider | Priority | Point-in-time role | Research role | Cadence | Access |
|---|---|---|---|---|---|---|
| hko_daily_extract | Hong Kong Observatory | P0 | TARGET_ONLY | canonical_target_candidate | daily | http_html |
| hko_daily_extract_catalog | Hong Kong Observatory | P0 | METADATA | daily_extract_backing_coverage_index | provider_updated | http_json_text |
| hko_daily_extract_month | Hong Kong Observatory | P0 | TARGET_ONLY | canonical_target_candidate_backing_payload | daily_until_month_finalized | http_json_text_template |
| hko_daily_extract_year | Hong Kong Observatory | P1 | PROXY_WITH_LIMITATIONS | canonical_target_candidate_backing_payload | annual_or_provider_rollup | http_json_text_template |
| hko_clmmaxt_hko | Hong Kong Observatory | P0 | PROXY_WITH_LIMITATIONS | candidate_historical_target_label | monthly_update_of_daily_history | http_csv |
| hko_daily_climate_download | Hong Kong Observatory | P1 | PROXY_WITH_LIMITATIONS | historical_daily_elements | monthly_or_element_specific | interactive_download_and_api_discovery |
| hko_open_data_catalog | Hong Kong Observatory | P0 | METADATA | authoritative_dataset_catalog | provider_updated | html |
| hko_api_documentation | Hong Kong Observatory | P0 | METADATA | authoritative_api_and_station_metadata | provider_updated | linked_pdf_from_catalog |
| hko_latest_1min_temperature | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | live_station_network_temperature | every_10_minutes | http_csv |
| hko_since_midnight_maxmin | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | live_maximum_and_minimum_since_midnight | every_10_minutes | http_csv |
| hko_latest_relative_humidity | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | live_station_network_humidity | every_10_minutes | http_csv |
| hko_latest_pressure | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | live_station_network_pressure | every_10_minutes | http_csv |
| hko_latest_wind | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | live_station_network_wind | every_10_minutes | http_csv |
| hko_latest_solar_radiation | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | live_solar_radiation | every_10_minutes | http_csv |
| hko_latest_uv_index | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | live_uv_index | every_15_minutes | http_csv |
| hko_automatic_rainfall | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | live_station_network_rainfall | every_15_minutes | http_json |
| hko_latest_visibility | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | live_station_network_visibility | every_10_minutes | http_csv |
| hko_current_weather_report | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | current_weather_report_and_regional_observations | provider_updated | http_json |
| hko_gridded_rainfall_nowcast | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | issued_rainfall_nowcast | every_12_minutes | http_csv |
| hko_radar | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | radar_images_and_derived_features | approximately_every_6_minutes | official_image_pages_and_endpoint_discovery |
| hko_radar_image_manifest | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | current_radar_image_manifest | approximately_every_6_minutes | http_json |
| hko_lightning | Hong Kong Observatory | P2 | OPERATIONAL_POINT_IN_TIME | lightning_observations | page_approximately_5_minutes_counts_15_minutes | official_page_and_endpoint_discovery |
| hko_lightning_counts_latest | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | lightning_counts_latest | every_15_minutes | http_csv |
| hko_satellite | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | satellite_imagery | product_specific | official_page_and_endpoint_discovery |
| hko_local_weather_forecast | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | official_forecast_benchmark | hourly_and_as_needed | weather_api |
| hko_nine_day_forecast | Hong Kong Observatory | P0 | OPERATIONAL_POINT_IN_TIME | official_forecast_benchmark | at_least_twice_daily_and_as_needed | weather_api |
| hko_weather_warning_summary | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | official_warning_vintage_summary | warning_update | http_json |
| hko_weather_warning_information | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | official_warning_vintage_detail | warning_update | http_json |
| hko_special_weather_tips | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | official_special_weather_tips | provider_updated | http_json |
| hko_automatic_regional_forecast | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | station_specific_automatic_forecast | product_specific | official_gis_service_discovery |
| hko_upper_air | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | king_s_park_soundings | typically_twice_daily | open_data_catalog_or_wmo_mirror |
| hko_station_metadata | Hong Kong Observatory | P0 | METADATA | station_identity_and_history | provider_updated | html |
| hko_historical_archive_api | DATA.GOV.HK | P0 | POTENTIAL_POINT_IN_TIME_ARCHIVE | historical_version_discovery | daily | api_specification |
| hko_south_china_coastal_waters_bulletin | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | south_china_coastal_waters_bulletin | provider_updated | http_json |
| hko_latest_tidal_information | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | latest_tidal_information | provider_updated | http_csv |
| ecmwf_open_ifs_aifs | ECMWF | P0 | OPERATIONAL_POINT_IN_TIME | operational_nwp_and_ai_forecasts | cycle_and_product_specific | open_forecast_data |
| noaa_gfs | NOAA NCEP | P0 | OPERATIONAL_POINT_IN_TIME | operational_global_deterministic_forecast | four_cycles_daily | nomads_or_cloud_archive |
| noaa_gefs | NOAA NCEP | P0 | OPERATIONAL_POINT_IN_TIME | operational_global_ensemble_forecast | cycle_specific | nomads_or_cloud_archive |
| dwd_icon | Deutscher Wetterdienst | P1 | OPERATIONAL_POINT_IN_TIME | operational_global_deterministic_forecast | cycle_specific | open_data |
| dwd_icon_eps | Deutscher Wetterdienst | P1 | OPERATIONAL_POINT_IN_TIME | operational_global_ensemble_forecast | cycle_specific | open_data |
| copernicus_era5 | Copernicus Climate Change Service | P2 | RETROSPECTIVE_ONLY | retrospective_synoptic_and_mechanism_analysis | hourly | cds_api |
| copernicus_era5_land | Copernicus Climate Change Service | P3 | RETROSPECTIVE_ONLY | retrospective_land_surface_analysis | hourly | cds_api |
| noaa_isd | NOAA NCEI | P2 | PROXY_WITH_LIMITATIONS | regional_synoptic_station_history | hourly_subhourly_station_dependent | public_archive |
| hko_tropical_cyclone_realtime | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | point_in_time_tropical_cyclone_advisories_and_tracks | advisory_specific | official_api_or_page_discovery |
| hko_tropical_cyclone_track_realtime | Hong Kong Observatory | P1 | OPERATIONAL_POINT_IN_TIME | realtime_tropical_cyclone_track_list | advisory_specific | http_xml |
| hko_tropical_cyclone_best_track | Hong Kong Observatory | P2 | RETROSPECTIVE_ONLY | retrospective_regime_label | annual_or_event_update | official_download |
| hong_kong_air_quality | Hong Kong Environmental Protection Department | P3 | PROXY_WITH_LIMITATIONS | exploratory_aerosol_haze_proxy | hourly_or_subhourly | open_data_discovery |
| terrain_land_cover_coastline | multiple_official_or_open_geospatial_sources | P2 | STATIC_METADATA | static_station_and_flow_context | static_versioned | source_selection_required |
| polymarket_gamma_event_by_slug | Polymarket | P0 | MARKET_ONLY | event_market_rules_metadata | poll_on_change | http_json_template |
| polymarket_clob_book | Polymarket | P0 | MARKET_ONLY | executable_order_book_snapshot | on_demand_or_websocket | http_json_template |
| polymarket_market_websocket | Polymarket | P0 | MARKET_ONLY | live_order_book_deltas_and_trades | event_driven | websocket |
| polymarket_price_history | Polymarket | P1 | MARKET_ONLY | historical_token_prices | on_demand | http_json_template |
| polymarket_fee_parameters | Polymarket | P0 | MARKET_ONLY | cost_model | market_and_policy_specific | official_documentation_and_market_endpoint |

## Counts by point-in-time role

- **MARKET_ONLY:** 5
- **METADATA:** 4
- **OPERATIONAL_POINT_IN_TIME:** 32
- **POTENTIAL_POINT_IN_TIME_ARCHIVE:** 1
- **PROXY_WITH_LIMITATIONS:** 5
- **RETROSPECTIVE_ONLY:** 3
- **STATIC_METADATA:** 1
- **TARGET_ONLY:** 2

## Required next action

For each implemented source, create a source contract under `docs/source_contracts/` and verify its official endpoint, timestamps, cadence, revision policy, terms, and tests.
