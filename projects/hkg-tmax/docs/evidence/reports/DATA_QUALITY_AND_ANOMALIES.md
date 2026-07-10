# Data Quality And Anomalies

## Daily Climate Missingness Top 25

| source_id | variable | rows | missing_values |
| --- | --- | --- | --- |
| hko_daily_climate_rainfall_all | daily_rainfall | 49400 | 6917 |
| hko_daily_climate_sea_temp_waglan_all | sea_temperature | 13300 | 5502 |
| hko_daily_climate_prevailing_wind_direction_all | prevailing_wind_direction | 18779 | 695 |
| hko_daily_climate_mean_wind_speed_all | mean_wind_speed | 18779 | 480 |
| hko_daily_climate_global_solar_radiation_all | global_solar_radiation | 17683 | 334 |
| hko_daily_climate_grass_min_temperature_all | grass_minimum_temperature | 21336 | 109 |
| hko_daily_climate_evaporation_all | evaporation | 21336 | 99 |
| hko_daily_climate_reduced_visibility_hka_all | reduced_visibility_hours | 10743 | 44 |
| hko_daily_climate_mean_temperature_all | mean_temperature | 49400 | 32 |
| hko_daily_climate_mslp_all | mean_sea_level_pressure | 49400 | 32 |
| hko_daily_climate_sea_temp_np_am_all | sea_temperature_am | 18976 | 32 |
| hko_daily_climate_sea_temp_np_pm_all | sea_temperature_pm | 18976 | 32 |
| hko_daily_climate_maximum_temperature_all | daily_maximum_temperature | 49460 | 1 |
| hko_daily_climate_minimum_temperature_all | daily_minimum_temperature | 49460 | 1 |
| hko_daily_climate_bright_sunshine_all | bright_sunshine_duration | 23892 | 0 |
| hko_daily_climate_cloud_amount_all | mean_cloud_amount | 28275 | 0 |
| hko_daily_climate_dew_point_all | mean_dew_point_temperature | 23892 | 0 |
| hko_daily_climate_lightning_cloud_all | cloud_to_cloud_lightning | 7650 | 0 |
| hko_daily_climate_lightning_ground_all | cloud_to_ground_lightning | 7650 | 0 |
| hko_daily_climate_relative_humidity_all | mean_relative_humidity | 29006 | 0 |
| hko_daily_climate_wet_bulb_all | mean_wet_bulb_temperature | 29006 | 0 |

## High-Frequency Parsed Value Ranges

| source_id | variable | rows | min_value | max_value |
| --- | --- | --- | --- | --- |
| datagov_hko_historical_latest_10min_wind_archive | max_wind_gust_kmh | 727886 | 0.0 | 231.0 |
| datagov_hko_historical_latest_10min_wind_archive | mean_wind_speed_kmh | 732979 | 0.0 | 126.0 |
| datagov_hko_historical_latest_1min_humidity_archive | relative_humidity_pct | 63287 | 17.0 | 98.0 |
| datagov_hko_historical_latest_1min_pressure_archive | msl_pressure_hpa | 56502 | 981.1 | 1032.7 |
| datagov_hko_historical_latest_1min_solar_archive | diffuse_solar_wm2 | 57946 | 0.0 | 1103.0 |
| datagov_hko_historical_latest_1min_solar_archive | direct_solar_wm2 | 57951 | 0.0 | 1023.0 |
| datagov_hko_historical_latest_1min_solar_archive | global_solar_wm2 | 57919 | 0.0 | 1380.0 |
| datagov_hko_historical_latest_1min_temperature_archive | air_temperature_c | 63324 | 6.7 | 35.7 |
| datagov_hko_historical_latest_since_midnight_maxmin_archive | temperature_since_midnight_max_c | 6657 | 9.2 | 35.3 |
| datagov_hko_historical_latest_since_midnight_maxmin_archive | temperature_since_midnight_min_c | 63290 | 7.5 | 31.0 |

## T-24 Feature Missingness

| field | missing_rows |
| --- | --- |
| local_date | 0 |
| hko_temp_at_tminus1_1500_c | 47527 |
| hko_rh_at_tminus1_1500_pct | 47528 |
| hko_mslp_at_tminus1_1500_hpa | 47851 |

## Current QC Interpretation

- HKO daily Tmax target has the longest official record and is fit for target anatomy.
- High-frequency HKO station features are the strongest operationally aligned source but only cover 2020/2021 onward in the public archive acquired so far.
- Same-day daily climate values are not usable as T-24 predictors; they are retained for mechanism analysis only.
