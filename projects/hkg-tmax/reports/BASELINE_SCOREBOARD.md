# Baseline Scoreboard

Champion baseline: `station_state_analogue` selected by `lowest validation_2024 MAE among predeclared scored baselines with n >= 300; RMSE as tie-breaker`.

| model_id | split | status | n | mae | rmse | median_abs_error | bias | crps_normal | pinball_mean | coverage_80 | width_80 | coverage_90 | width_90 | mae_ci95_low | mae_ci95_high | method_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| station_state_analogue | development | scored | 729 | 1.4875 | 1.9272 | 1.2160 | 0.1509 | 1.0651 | 0.4344 | 0.8272 | 4.9277 | 0.9012 | 6.3247 | 1.3574 | 1.6213 | scored |
| cutoff_station_temperature_persistence | development | scored | 729 | 1.7350 | 2.2168 | 1.4000 | -0.8941 | 1.2354 | 0.5024 | 0.7984 | 5.2028 | 0.8834 | 6.6777 | 1.6052 | 1.8893 | scored |
| transparent_equal_weight_blend | development | scored | 729 | 1.7695 | 2.2515 | 1.4889 | -0.1338 | 1.2517 | 0.5078 | 0.8176 | 5.7646 | 0.9136 | 7.3988 | 1.6144 | 1.9185 | scored |
| multi_day_thermal_memory | development | scored | 729 | 2.1463 | 2.7521 | 1.7330 | -0.0190 | 1.5344 | 0.6236 | 0.8134 | 7.0585 | 0.9081 | 9.0595 | 1.9175 | 2.3620 | scored_proxy_publication_lag_needed |
| recent10y_climatology | development | scored | 729 | 2.1564 | 2.6944 | 1.8783 | -0.3242 | 1.5121 | 0.6089 | 0.8176 | 6.8606 | 0.9095 | 8.8054 | 1.9610 | 2.3364 | scored |
| seasonal_anomaly_persistence | development | scored | 729 | 2.1632 | 2.8475 | 1.6419 | -0.0100 | 1.5683 | 0.6424 | 0.8189 | 7.3033 | 0.9040 | 9.3737 | 1.9337 | 2.3872 | scored_proxy_publication_lag_needed |
| last_final_tmax_persistence | development | scored | 729 | 2.1667 | 2.8509 | 1.6000 | -0.0103 | 1.5699 | 0.6426 | 0.8134 | 7.3120 | 0.8999 | 9.3849 | 1.9381 | 2.3919 | scored_proxy_publication_lag_needed |
| trend_adjusted_climatology | development | scored | 729 | 2.2492 | 2.7688 | 1.9802 | -0.6570 | 1.5634 | 0.6265 | 0.8093 | 6.8988 | 0.9012 | 8.8545 | 2.0568 | 2.4296 | scored |
| day_of_year_climatology | development | scored | 729 | 2.6030 | 3.1510 | 2.2972 | -1.5680 | 1.8105 | 0.7231 | 0.7202 | 7.0100 | 0.8615 | 8.9973 | 2.3605 | 2.8694 | scored |
| raw_hko_official_forecast | development | pending_source_parser | 0 |  |  |  |  |  |  |  |  |  |  |  |  | HKO forecast vintages are acquired but not yet parsed into last-eligible Tmax vintage rows. |
| bias_corrected_hko_official_forecast | development | pending_source_parser | 0 |  |  |  |  |  |  |  |  |  |  |  |  | Requires raw HKO official forecast baseline first. |
| raw_deterministic_nwp | development | pending_historical_vintages | 0 |  |  |  |  |  |  |  |  |  |  |  |  | Current GFS subsets exist, but historical point-in-time model cycles are not yet backtestable. |
| raw_ensemble_mean_distribution | development | pending_historical_vintages | 0 |  |  |  |  |  |  |  |  |  |  |  |  | GEFS historical/live cycle contract is not complete. |
| simple_mos_correction | development | pending_historical_vintages | 0 |  |  |  |  |  |  |  |  |  |  |  |  | MOS correction requires historical forecast-vs-target pairs. |
| station_state_analogue | locked_test | scored | 516 | 1.5668 | 1.9776 | 1.3440 | 0.0551 | 1.1094 | 0.4486 | 0.7810 | 4.9277 | 0.8798 | 6.3247 | 1.4498 | 1.7249 | scored |
| transparent_equal_weight_blend | locked_test | scored | 516 | 1.8369 | 2.3306 | 1.5938 | -0.2406 | 1.2942 | 0.5233 | 0.8178 | 5.7646 | 0.9050 | 7.3988 | 1.6976 | 2.0567 | scored |
| cutoff_station_temperature_persistence | locked_test | scored | 516 | 1.9436 | 2.4446 | 1.6000 | -0.8742 | 1.3793 | 0.5628 | 0.7403 | 5.2028 | 0.8314 | 6.6777 | 1.8251 | 2.1049 | scored |
| recent10y_climatology | locked_test | scored | 516 | 1.9870 | 2.5074 | 1.7117 | -0.3570 | 1.4009 | 0.5672 | 0.8450 | 6.8606 | 0.9283 | 8.8054 | 1.8125 | 2.2206 | scored |
| multi_day_thermal_memory | locked_test | scored | 516 | 2.1344 | 2.8013 | 1.7800 | -0.0913 | 1.5348 | 0.6271 | 0.8256 | 7.0585 | 0.9031 | 9.0595 | 1.9342 | 2.4100 | scored_proxy_publication_lag_needed |
| trend_adjusted_climatology | locked_test | scored | 516 | 2.1846 | 2.6752 | 1.8642 | -0.9254 | 1.5110 | 0.6059 | 0.8081 | 6.8988 | 0.9167 | 8.8545 | 1.9726 | 2.4337 | scored |
| seasonal_anomaly_persistence | locked_test | scored | 516 | 2.3574 | 3.0798 | 1.8580 | -0.0009 | 1.7015 | 0.6957 | 0.7907 | 7.3033 | 0.8721 | 9.3737 | 2.1560 | 2.6782 | scored_proxy_publication_lag_needed |
| last_final_tmax_persistence | locked_test | scored | 516 | 2.3603 | 3.0840 | 1.8000 | -0.0397 | 1.7041 | 0.6970 | 0.7868 | 7.3120 | 0.8682 | 9.3849 | 2.1599 | 2.6815 | scored_proxy_publication_lag_needed |
| day_of_year_climatology | locked_test | scored | 516 | 2.7244 | 3.2684 | 2.4388 | -1.9991 | 1.8899 | 0.7524 | 0.6938 | 7.0100 | 0.8488 | 8.9973 | 2.4166 | 3.0765 | scored |
| raw_hko_official_forecast | locked_test | pending_source_parser | 0 |  |  |  |  |  |  |  |  |  |  |  |  | HKO forecast vintages are acquired but not yet parsed into last-eligible Tmax vintage rows. |
| bias_corrected_hko_official_forecast | locked_test | pending_source_parser | 0 |  |  |  |  |  |  |  |  |  |  |  |  | Requires raw HKO official forecast baseline first. |
| raw_deterministic_nwp | locked_test | pending_historical_vintages | 0 |  |  |  |  |  |  |  |  |  |  |  |  | Current GFS subsets exist, but historical point-in-time model cycles are not yet backtestable. |
| raw_ensemble_mean_distribution | locked_test | pending_historical_vintages | 0 |  |  |  |  |  |  |  |  |  |  |  |  | GEFS historical/live cycle contract is not complete. |
| simple_mos_correction | locked_test | pending_historical_vintages | 0 |  |  |  |  |  |  |  |  |  |  |  |  | MOS correction requires historical forecast-vs-target pairs. |
| station_state_analogue | validation_2024 | scored | 364 | 1.5032 | 1.8974 | 1.2980 | 0.0185 | 1.0654 | 0.4297 | 0.8187 | 4.9277 | 0.9093 | 6.3247 | 1.3576 | 1.6672 | scored |
| transparent_equal_weight_blend | validation_2024 | scored | 364 | 1.6792 | 2.1621 | 1.3475 | -0.2340 | 1.1975 | 0.4900 | 0.8324 | 5.7646 | 0.9121 | 7.3988 | 1.4938 | 1.8824 | scored |
| cutoff_station_temperature_persistence | validation_2024 | scored | 364 | 1.7835 | 2.2441 | 1.5000 | -0.8379 | 1.2688 | 0.5113 | 0.7473 | 5.2028 | 0.8599 | 6.6777 | 1.6123 | 1.9592 | scored |
| multi_day_thermal_memory | validation_2024 | scored | 364 | 1.9878 | 2.6149 | 1.5780 | 0.0174 | 1.4385 | 0.5903 | 0.8571 | 7.0585 | 0.9231 | 9.0595 | 1.7377 | 2.2833 | scored_proxy_publication_lag_needed |
| recent10y_climatology | validation_2024 | scored | 364 | 2.0933 | 2.6321 | 1.7663 | -0.5104 | 1.4786 | 0.5982 | 0.8049 | 6.8606 | 0.9066 | 8.8054 | 1.8764 | 2.3589 | scored |
| last_final_tmax_persistence | validation_2024 | scored | 364 | 2.1170 | 2.7471 | 1.7000 | 0.0005 | 1.5287 | 0.6225 | 0.8242 | 7.3120 | 0.9066 | 9.3849 | 1.8325 | 2.4237 | scored_proxy_publication_lag_needed |
| seasonal_anomaly_persistence | validation_2024 | scored | 364 | 2.1195 | 2.7421 | 1.7050 | 0.0012 | 1.5273 | 0.6221 | 0.8352 | 7.3033 | 0.9176 | 9.3737 | 1.8396 | 2.4235 | scored_proxy_publication_lag_needed |
| trend_adjusted_climatology | validation_2024 | scored | 364 | 2.2391 | 2.7717 | 1.8483 | -0.9730 | 1.5668 | 0.6313 | 0.7995 | 6.8988 | 0.8791 | 8.8545 | 1.9803 | 2.5329 | scored |
| day_of_year_climatology | validation_2024 | scored | 364 | 2.6981 | 3.2870 | 2.3815 | -1.9216 | 1.8835 | 0.7581 | 0.7473 | 7.0100 | 0.8407 | 8.9973 | 2.3320 | 3.1326 | scored |
| raw_hko_official_forecast | validation_2024 | pending_source_parser | 0 |  |  |  |  |  |  |  |  |  |  |  |  | HKO forecast vintages are acquired but not yet parsed into last-eligible Tmax vintage rows. |
| bias_corrected_hko_official_forecast | validation_2024 | pending_source_parser | 0 |  |  |  |  |  |  |  |  |  |  |  |  | Requires raw HKO official forecast baseline first. |
| raw_deterministic_nwp | validation_2024 | pending_historical_vintages | 0 |  |  |  |  |  |  |  |  |  |  |  |  | Current GFS subsets exist, but historical point-in-time model cycles are not yet backtestable. |
| raw_ensemble_mean_distribution | validation_2024 | pending_historical_vintages | 0 |  |  |  |  |  |  |  |  |  |  |  |  | GEFS historical/live cycle contract is not complete. |
| simple_mos_correction | validation_2024 | pending_historical_vintages | 0 |  |  |  |  |  |  |  |  |  |  |  |  | MOS correction requires historical forecast-vs-target pairs. |

## Leakage Statement

- Common sample begins on 2021-07-01 because pressure/modern high-frequency archives are not complete before then.
- Target labels are used only as labels or as lagged values through T-2.
- HKO T-1 15:00 station features are selected by `available_at <= cutoff` under a +20 minute conservative latency assumption.
- HKO official forecast and NWP baselines are not scored until historical vintages are parsed.
