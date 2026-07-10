# Coverage Matrix

| source_id | dataset | raw_status | success_rows | parsed_rows | date_or_observed_range | point_in_time_role |
| --- | --- | --- | --- | --- | --- | --- |
| hko_daily_climate_maximum_temperature_all | HKO official daily Tmax target | downloaded | 1 | 49460 | 1884-01-01 to NaT | TARGET_ONLY |
| hko_daily_climate_mean_temperature_all | HKO daily mean temperature | downloaded | 1 | 49400 | 1884-03-01 to NaT | RETROSPECTIVE_MECHANISM_ONLY |
| hko_daily_climate_rainfall_all | HKO daily rainfall | downloaded | 1 | 49400 | 1884-03-01 to NaT | RETROSPECTIVE_MECHANISM_ONLY |
| hko_daily_climate_cloud_amount_all | HKO daily cloud amount | downloaded | 1 | 28275 | 1949-01-01 to 2026-05-31 | RETROSPECTIVE_MECHANISM_ONLY |
| datagov_hko_historical_latest_1min_temperature_archive | HKO historical high-frequency temperature | downloaded | 90 | 21 | 2026-06-18T08:30:00+08:00 to 2026-06-18T15:30:00+08:00 | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| datagov_hko_historical_latest_1min_humidity_archive | HKO historical high-frequency humidity | downloaded | 90 | 21 | 2026-06-18T08:30:00+08:00 to 2026-06-18T15:30:00+08:00 | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| datagov_hko_historical_latest_1min_pressure_archive | HKO historical high-frequency pressure | downloaded | 78 | 21 | 2026-06-18T08:30:00+08:00 to 2026-06-18T15:30:00+08:00 | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| datagov_hko_historical_latest_since_midnight_maxmin_archive | HKO historical since-midnight max/min | downloaded | 90 | 21 | 2026-06-18T08:30:00+08:00 to 2026-06-18T15:30:00+08:00 | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| datagov_hko_historical_latest_10min_wind_archive | HKO historical 10-minute wind network | downloaded | 78 | 537 | 2026-06-18T08:30:00+08:00 to 2026-06-18T15:30:00+08:00 | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| datagov_hko_historical_latest_1min_solar_archive | HKO historical solar support stations | downloaded | 78 | 63 | 2026-06-18T08:30:00+08:00 to 2026-06-18T15:30:00+08:00 | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| derived:hko_daily_tmax | Silver official HKO Tmax target table | derived |  | 49459 | 1884-01-01 to 2026-05-31 | TARGET_ONLY |
| derived:t24_cutoff_feature_candidates | Leakage-screened T-24 candidate feature table | derived |  |  |  | OPERATIONAL_WITH_CONSERVATIVE_LATENCY plus documented proxies |
| derived:hko_station_temperature_cutoff_summary | All-station 15:00 cutoff temperature summaries from HKO minute archive | derived |  | 75166 | 2020-06-30 to 2026-06-18 | EDA and station-network context |
