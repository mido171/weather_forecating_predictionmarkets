# Data Inventory

Polymarket is explicitly excluded from the current acquisition goal.

| Source | Family | Priority | Status | Point-in-time class | Last success | Unique hashes | Blocker |
|---|---|---|---|---|---|---:|---|
| hko_clmmaxt_hko | A_hko_target_labels_daily_climate | P0_CRITICAL | IMPLEMENTED_INITIAL_FETCH | PROXY_WITH_LIMITATIONS | 2026-06-18T22:30:24.926164Z | 1 |  |
| hko_daily_extract_operational | A_hko_target_labels_daily_climate | P0_CRITICAL | SUPERSEDED_POLLING_FAMILY_CLOSED | TARGET_ONLY |  | 0 | template adapter must bind yyyymm from operational date |
| hko_daily_climate_elements | A_hko_target_labels_daily_climate | P0_CRITICAL | DISCOVERY_REQUIRED | PROXY_WITH_LIMITATIONS |  | 0 | official element identifiers must be discovered from HKO documentation, not guessed |
| hko_station_metadata | B_hko_station_metadata_history | P0_CRITICAL | IMPLEMENTED_INITIAL_FETCH | METADATA | 2026-06-18T22:30:37.384385Z | 1 |  |
| hko_open_data_catalog | B_hko_station_metadata_history | P0_CRITICAL | IMPLEMENTED_INITIAL_FETCH | METADATA | 2026-06-18T22:30:35.418177Z | 1 |  |
| hko_latest_1min_temperature | C_high_frequency_hko_regional_observations | P0_CRITICAL | IMPLEMENTED_INITIAL_FETCH | OPERATIONAL_POINT_IN_TIME | 2026-06-18T22:30:27.743512Z | 1 | historical minute backfill unknown |
| hko_since_midnight_maxmin | C_high_frequency_hko_regional_observations | P0_CRITICAL | IMPLEMENTED_INITIAL_FETCH | OPERATIONAL_POINT_IN_TIME | 2026-06-18T22:30:29.600202Z | 1 | historical intraday max/min backfill unknown |
| hko_live_humidity_pressure_wind_rain_radiation | C_high_frequency_hko_regional_observations | P0_CRITICAL | DISCOVERY_REQUIRED | OPERATIONAL_POINT_IN_TIME |  | 0 | exact official resource URLs and schemas must be resolved |
| hko_local_weather_forecast | D_official_hko_forecast_vintages | P0_CRITICAL | IMPLEMENTED_INITIAL_FETCH | OPERATIONAL_POINT_IN_TIME | 2026-06-18T22:30:31.317950Z | 1 |  |
| hko_nine_day_forecast | D_official_hko_forecast_vintages | P0_CRITICAL | IMPLEMENTED_INITIAL_FETCH | OPERATIONAL_POINT_IN_TIME | 2026-06-18T22:30:33.279570Z | 1 |  |
| operational_nwp_gfs_gefs_ecmwf_icon | E_operational_numerical_ai_forecast_archives | P0_CRITICAL | NOT_STARTED | OPERATIONAL_POINT_IN_TIME |  | 0 | server-side subset definitions and byte budget must be finalized before large backfill |
| king_park_upper_air_and_regional_soundings | F_upper_air_vertical_profile_observations | P0_CRITICAL | DISCOVERY_REQUIRED | OPERATIONAL_POINT_IN_TIME |  | 0 | exact HKO or IGRA station identifiers and access URLs must be confirmed |
| radar_nowcast_lightning | G_radar_rainfall_nowcasts_lightning | P1_HIGH_VALUE | DISCOVERY_REQUIRED | OPERATIONAL_POINT_IN_TIME |  | 0 | product endpoints, color scales and georeferencing must be documented |
| satellite_cloud_aerosol | H_satellite_cloud_aerosol_observations | P1_HIGH_VALUE | NOT_STARTED | OPERATIONAL_POINT_IN_TIME |  | 0 | lawful provider and product subset must be selected |
| tropical_cyclone_monsoon_synoptic | I_tropical_cyclone_monsoon_synoptic_information | P1_HIGH_VALUE | DISCOVERY_REQUIRED | MIXED_POINT_IN_TIME_AND_RETROSPECTIVE_ONLY |  | 0 | exact operational track/advisory endpoints must be confirmed |
| marine_ocean_surface_state | J_marine_ocean_surface_state | P1_HIGH_VALUE | NOT_STARTED | PROXY_WITH_LIMITATIONS |  | 0 | source selection and issue-time contracts required |
| reanalysis_mechanism_datasets | K_reanalysis_retrospective_mechanism_datasets | P1_HIGH_VALUE | CREDENTIAL_BLOCKED | RETROSPECTIVE_ONLY |  | 0 | CDS credentials and exact request subsets required |
| static_geospatial_deterministic_context | L_static_geospatial_deterministic_context | P1_HIGH_VALUE | NOT_STARTED | STATIC_METADATA |  | 0 | source selection and licensing review required |
| frontier_context_inventory | M_frontier_context | P2_FRONTIER | DEFERRED_WITH_REASON | MIXED |  | 0 | P2 must not delay incomplete P0 acquisition |
