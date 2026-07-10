| source_id | location | rows | first_date | last_date | null_or_unusable_percent | source_role |
| --- | --- | --- | --- | --- | --- | --- |
| target_history | feature_safe.hko_target_history_pre2024 | 8765 | 2000-01-02 | 2023-12-31 | 0.00000 | official target labels for daily absolute maximum temperature at HKO/HKG station |
| lead1_hko_forecast_archive | public.hko_historical_forecasts_2000_2026 | 80089 | 2000-01-02 | 2023-12-31 | 0.00000 | HKO local lead-1 forecast min/max archive used as official anchor |
| hko_daily_climate | diagnostic_physics.codex_audit_ds_02_hko_daily_climate_all_elements_hko_d_f7bb0017 | 187025 | 1999-01-01 | 2023-12-31 | 3.08568 | daily HKO climate state, lagged to T-2 in production features |
| gpt_pro_strategy_spec | C:\Users\ahmad\.codex\attachments\2f15d411-f901-46b6-9fb4-5bae7b3c26ef\pasted-text.txt | 1722 |  |  | 0.00000 | implementation specification read before coding |

| variable | rows | first_date | last_date | null_value_percent | operational_input_allowed_percent | availability_tiers |
| --- | --- | --- | --- | --- | --- | --- |
| bright_sunshine_duration | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| cloud_to_cloud_lightning | 6768 | 2005-06-21 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| cloud_to_ground_lightning | 6768 | 2005-06-21 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| daily_maximum_temperature | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | TARGET_ONLY |
| daily_minimum_temperature | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| daily_rainfall | 9131 | 1999-01-01 | 2023-12-31 | 20.39207 | 0.00000 | MECHANISM_ONLY |
| evaporation | 9131 | 1999-01-01 | 2023-12-31 | 0.76662 | 0.00000 | MECHANISM_ONLY |
| global_solar_radiation | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| grass_minimum_temperature | 9131 | 1999-01-01 | 2023-12-31 | 0.77757 | 0.00000 | MECHANISM_ONLY |
| mean_cloud_amount | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| mean_dew_point_temperature | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| mean_relative_humidity | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| mean_sea_level_pressure | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| mean_temperature | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| mean_wet_bulb_temperature | 9131 | 1999-01-01 | 2023-12-31 | 0.00000 | 0.00000 | MECHANISM_ONLY |
| mean_wind_speed | 9131 | 1999-01-01 | 2023-12-31 | 0.85423 | 0.00000 | MECHANISM_ONLY |
| prevailing_wind_direction | 9131 | 1999-01-01 | 2023-12-31 | 1.33611 | 0.00000 | MECHANISM_ONLY |
| reduced_visibility_hours | 9131 | 1999-01-01 | 2023-12-31 | 0.03286 | 0.00000 | MECHANISM_ONLY |
| sea_temperature | 9131 | 1999-01-01 | 2023-12-31 | 38.59380 | 0.00000 | MECHANISM_ONLY |
| sea_temperature_am | 9131 | 1999-01-01 | 2023-12-31 | 0.21903 | 0.00000 | MECHANISM_ONLY |
| sea_temperature_pm | 9131 | 1999-01-01 | 2023-12-31 | 0.22999 | 0.00000 | MECHANISM_ONLY |
