# HKG T24 Source Coverage

Generated: `2026-06-20T10:26:21.027270Z`

- source contracts: `61`
- sources with retrieval rows: `242`
- availability tiers: `{'FORBIDDEN': 5, 'GOLD_EXACT_VINTAGE': 35, 'MECHANISM_ONLY': 5, 'SILVER_OPERATIONAL_REPLAY': 14, 'TARGET_ONLY': 2}`

| Source | Tier | Operational allowed | Success rows | Unique hashes | First success | Last success |
|---|---|---:|---:|---:|---|---|
| hko_daily_extract | TARGET_ONLY | False | 1 | 1 | 2026-06-19T05:19:22.930380Z | 2026-06-19T05:19:22.930380Z |
| hko_daily_extract_catalog | SILVER_OPERATIONAL_REPLAY | True | 1 | 1 | 2026-06-18T22:52:43.946761Z | 2026-06-18T22:52:43.946761Z |
| hko_daily_extract_month | TARGET_ONLY | False | 6 | 6 | 2026-06-18T22:57:51.519431Z | 2026-06-18T22:58:13.112613Z |
| hko_daily_extract_year | SILVER_OPERATIONAL_REPLAY | True | 143 | 137 | 2026-06-18T22:52:48.528263Z | 2026-06-18T22:57:44.448217Z |
| hko_clmmaxt_hko | SILVER_OPERATIONAL_REPLAY | True | 1 | 1 | 2026-06-18T22:30:24.926164Z | 2026-06-18T22:30:24.926164Z |
| hko_daily_climate_download | SILVER_OPERATIONAL_REPLAY | True | 0 | 0 |  |  |
| hko_open_data_catalog | SILVER_OPERATIONAL_REPLAY | True | 2 | 1 | 2026-06-18T22:30:35.418177Z | 2026-06-19T22:33:24.592245Z |
| hko_api_documentation | SILVER_OPERATIONAL_REPLAY | True | 0 | 0 |  |  |
| hko_latest_1min_temperature | GOLD_EXACT_VINTAGE | True | 42 | 42 | 2026-06-18T22:30:27.743512Z | 2026-06-20T07:55:52.478734Z |
| hko_since_midnight_maxmin | GOLD_EXACT_VINTAGE | True | 42 | 42 | 2026-06-18T22:30:29.600202Z | 2026-06-20T07:55:54.935847Z |
| hko_latest_relative_humidity | GOLD_EXACT_VINTAGE | True | 42 | 42 | 2026-06-19T04:50:28.022710Z | 2026-06-20T07:55:57.323003Z |
| hko_latest_pressure | GOLD_EXACT_VINTAGE | True | 42 | 42 | 2026-06-19T04:50:30.389936Z | 2026-06-20T07:55:59.904180Z |
| hko_latest_wind | GOLD_EXACT_VINTAGE | True | 42 | 42 | 2026-06-19T04:50:32.384622Z | 2026-06-20T07:56:02.259971Z |
| hko_latest_solar_radiation | GOLD_EXACT_VINTAGE | True | 42 | 42 | 2026-06-19T04:50:34.780147Z | 2026-06-20T07:56:04.478708Z |
| hko_latest_uv_index | GOLD_EXACT_VINTAGE | True | 40 | 25 | 2026-06-19T04:50:36.769986Z | 2026-06-20T08:02:55.937925Z |
| hko_automatic_rainfall | GOLD_EXACT_VINTAGE | True | 40 | 40 | 2026-06-19T04:50:38.863655Z | 2026-06-20T08:02:58.047493Z |
| hko_latest_visibility | GOLD_EXACT_VINTAGE | True | 42 | 42 | 2026-06-19T04:50:41.645100Z | 2026-06-20T07:56:06.967020Z |
| hko_current_weather_report | GOLD_EXACT_VINTAGE | True | 42 | 17 | 2026-06-19T04:50:43.774030Z | 2026-06-20T07:56:09.139314Z |
| hko_gridded_rainfall_nowcast | GOLD_EXACT_VINTAGE | True | 41 | 41 | 2026-06-19T04:50:52.482529Z | 2026-06-20T07:56:22.144891Z |
| hko_radar | GOLD_EXACT_VINTAGE | True | 0 | 0 |  |  |
| hko_radar_image_manifest | GOLD_EXACT_VINTAGE | True | 1 | 1 | 2026-06-19T04:57:42.275051Z | 2026-06-19T04:57:42.275051Z |
| hko_lightning | GOLD_EXACT_VINTAGE | True | 0 | 0 |  |  |
| hko_lightning_counts_latest | GOLD_EXACT_VINTAGE | True | 40 | 17 | 2026-06-19T04:57:47.135172Z | 2026-06-20T08:03:02.341258Z |
| hko_satellite | GOLD_EXACT_VINTAGE | True | 0 | 0 |  |  |
| hko_local_weather_forecast | GOLD_EXACT_VINTAGE | True | 15 | 15 | 2026-06-18T22:30:31.317950Z | 2026-06-20T07:56:11.758536Z |
| hko_nine_day_forecast | GOLD_EXACT_VINTAGE | True | 15 | 7 | 2026-06-18T22:30:33.279570Z | 2026-06-20T07:56:13.900867Z |
| hko_weather_warning_summary | GOLD_EXACT_VINTAGE | True | 42 | 1 | 2026-06-19T04:50:45.813149Z | 2026-06-20T07:56:16.322038Z |
| hko_weather_warning_information | GOLD_EXACT_VINTAGE | True | 42 | 1 | 2026-06-19T04:50:47.744730Z | 2026-06-20T07:56:18.412698Z |
| hko_special_weather_tips | GOLD_EXACT_VINTAGE | True | 40 | 2 | 2026-06-19T04:50:49.000512Z | 2026-06-20T08:03:00.165607Z |
| hko_automatic_regional_forecast | GOLD_EXACT_VINTAGE | True | 0 | 0 |  |  |
| hko_upper_air | GOLD_EXACT_VINTAGE | True | 0 | 0 |  |  |
| hko_station_metadata | SILVER_OPERATIONAL_REPLAY | True | 2 | 1 | 2026-06-18T22:30:37.384385Z | 2026-06-19T22:33:22.804768Z |
| hko_historical_archive_api | SILVER_OPERATIONAL_REPLAY | True | 1 | 1 | 2026-06-19T17:47:45.599124Z | 2026-06-19T17:47:45.599124Z |
| hko_south_china_coastal_waters_bulletin | GOLD_EXACT_VINTAGE | True | 15 | 6 | 2026-06-19T04:50:55.669615Z | 2026-06-20T07:56:24.707400Z |
| hko_latest_tidal_information | GOLD_EXACT_VINTAGE | True | 15 | 15 | 2026-06-19T04:50:57.280298Z | 2026-06-20T07:56:27.102158Z |
| ecmwf_open_ifs_aifs | GOLD_EXACT_VINTAGE | True | 5 | 2 | 2026-06-19T17:47:38.415088Z | 2026-06-20T07:17:05.106848Z |
| noaa_gfs | GOLD_EXACT_VINTAGE | True | 5 | 5 | 2026-06-19T17:47:39.726831Z | 2026-06-20T07:17:07.645817Z |
| noaa_gefs | GOLD_EXACT_VINTAGE | True | 5 | 5 | 2026-06-19T17:47:41.211998Z | 2026-06-20T07:17:09.863526Z |
| dwd_icon | GOLD_EXACT_VINTAGE | True | 5 | 1 | 2026-06-19T17:47:42.390015Z | 2026-06-20T07:17:11.973382Z |
| dwd_icon_eps | GOLD_EXACT_VINTAGE | True | 5 | 1 | 2026-06-19T17:47:43.595225Z | 2026-06-20T07:17:14.015449Z |
| copernicus_era5 | MECHANISM_ONLY | False | 0 | 0 |  |  |
| copernicus_era5_land | MECHANISM_ONLY | False | 0 | 0 |  |  |
| noaa_isd | SILVER_OPERATIONAL_REPLAY | True | 0 | 0 |  |  |
| hko_tropical_cyclone_realtime | GOLD_EXACT_VINTAGE | True | 0 | 0 |  |  |
| hko_tropical_cyclone_track_realtime | GOLD_EXACT_VINTAGE | True | 40 | 3 | 2026-06-19T04:50:54.082962Z | 2026-06-20T08:03:04.726780Z |
| hko_tropical_cyclone_best_track | MECHANISM_ONLY | False | 40 | 40 | 2026-06-19T04:51:07.648318Z | 2026-06-19T04:52:38.219460Z |
| hong_kong_air_quality | SILVER_OPERATIONAL_REPLAY | True | 0 | 0 |  |  |
| terrain_land_cover_coastline | SILVER_OPERATIONAL_REPLAY | True | 0 | 0 | None | None |
| noaa_ghcnh_hko | SILVER_OPERATIONAL_REPLAY | True | 0 | 0 |  |  |
| noaa_oisst_v21 | SILVER_OPERATIONAL_REPLAY | True | 0 | 0 |  |  |
| himawari_ahi_noaa_aws | GOLD_EXACT_VINTAGE | True | 0 | 0 |  |  |
| gpm_imerg_final | MECHANISM_ONLY | False | 0 | 0 |  |  |
| gpm_imerg_early_late | GOLD_EXACT_VINTAGE | True | 0 | 0 |  |  |
| cams_eac4_reanalysis | MECHANISM_ONLY | False | 0 | 0 |  |  |
| cams_global_composition_forecasts | GOLD_EXACT_VINTAGE | True | 0 | 0 |  |  |
| hong_kong_epd_air_quality_observations | SILVER_OPERATIONAL_REPLAY | True | 0 | 0 |  |  |
