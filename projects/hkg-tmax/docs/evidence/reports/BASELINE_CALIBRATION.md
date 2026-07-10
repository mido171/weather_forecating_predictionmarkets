# Baseline Calibration

All scored baselines output a normal residual distribution with sigma calibrated on the development split. This is intentionally simple and will be challenged by conformal/distributional experiments later.

| model_id | split | n | crps_normal | pinball_mean | coverage_80 | width_80 | coverage_90 | width_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| station_state_analogue | development | 729 | 1.0651 | 0.4344 | 0.8272 | 4.9277 | 0.9012 | 6.3247 |
| cutoff_station_temperature_persistence | development | 729 | 1.2354 | 0.5024 | 0.7984 | 5.2028 | 0.8834 | 6.6777 |
| transparent_equal_weight_blend | development | 729 | 1.2517 | 0.5078 | 0.8176 | 5.7646 | 0.9136 | 7.3988 |
| multi_day_thermal_memory | development | 729 | 1.5344 | 0.6236 | 0.8134 | 7.0585 | 0.9081 | 9.0595 |
| recent10y_climatology | development | 729 | 1.5121 | 0.6089 | 0.8176 | 6.8606 | 0.9095 | 8.8054 |
| seasonal_anomaly_persistence | development | 729 | 1.5683 | 0.6424 | 0.8189 | 7.3033 | 0.9040 | 9.3737 |
| last_final_tmax_persistence | development | 729 | 1.5699 | 0.6426 | 0.8134 | 7.3120 | 0.8999 | 9.3849 |
| trend_adjusted_climatology | development | 729 | 1.5634 | 0.6265 | 0.8093 | 6.8988 | 0.9012 | 8.8545 |
| day_of_year_climatology | development | 729 | 1.8105 | 0.7231 | 0.7202 | 7.0100 | 0.8615 | 8.9973 |
| raw_hko_official_forecast | development | 0 |  |  |  |  |  |  |
| bias_corrected_hko_official_forecast | development | 0 |  |  |  |  |  |  |
| raw_deterministic_nwp | development | 0 |  |  |  |  |  |  |
| raw_ensemble_mean_distribution | development | 0 |  |  |  |  |  |  |
| simple_mos_correction | development | 0 |  |  |  |  |  |  |
| station_state_analogue | locked_test | 516 | 1.1094 | 0.4486 | 0.7810 | 4.9277 | 0.8798 | 6.3247 |
| transparent_equal_weight_blend | locked_test | 516 | 1.2942 | 0.5233 | 0.8178 | 5.7646 | 0.9050 | 7.3988 |
| cutoff_station_temperature_persistence | locked_test | 516 | 1.3793 | 0.5628 | 0.7403 | 5.2028 | 0.8314 | 6.6777 |
| recent10y_climatology | locked_test | 516 | 1.4009 | 0.5672 | 0.8450 | 6.8606 | 0.9283 | 8.8054 |
| multi_day_thermal_memory | locked_test | 516 | 1.5348 | 0.6271 | 0.8256 | 7.0585 | 0.9031 | 9.0595 |
| trend_adjusted_climatology | locked_test | 516 | 1.5110 | 0.6059 | 0.8081 | 6.8988 | 0.9167 | 8.8545 |
| seasonal_anomaly_persistence | locked_test | 516 | 1.7015 | 0.6957 | 0.7907 | 7.3033 | 0.8721 | 9.3737 |
| last_final_tmax_persistence | locked_test | 516 | 1.7041 | 0.6970 | 0.7868 | 7.3120 | 0.8682 | 9.3849 |
| day_of_year_climatology | locked_test | 516 | 1.8899 | 0.7524 | 0.6938 | 7.0100 | 0.8488 | 8.9973 |
| raw_hko_official_forecast | locked_test | 0 |  |  |  |  |  |  |
| bias_corrected_hko_official_forecast | locked_test | 0 |  |  |  |  |  |  |
| raw_deterministic_nwp | locked_test | 0 |  |  |  |  |  |  |
| raw_ensemble_mean_distribution | locked_test | 0 |  |  |  |  |  |  |
| simple_mos_correction | locked_test | 0 |  |  |  |  |  |  |
| station_state_analogue | validation_2024 | 364 | 1.0654 | 0.4297 | 0.8187 | 4.9277 | 0.9093 | 6.3247 |
| transparent_equal_weight_blend | validation_2024 | 364 | 1.1975 | 0.4900 | 0.8324 | 5.7646 | 0.9121 | 7.3988 |
| cutoff_station_temperature_persistence | validation_2024 | 364 | 1.2688 | 0.5113 | 0.7473 | 5.2028 | 0.8599 | 6.6777 |
| multi_day_thermal_memory | validation_2024 | 364 | 1.4385 | 0.5903 | 0.8571 | 7.0585 | 0.9231 | 9.0595 |
| recent10y_climatology | validation_2024 | 364 | 1.4786 | 0.5982 | 0.8049 | 6.8606 | 0.9066 | 8.8054 |
| last_final_tmax_persistence | validation_2024 | 364 | 1.5287 | 0.6225 | 0.8242 | 7.3120 | 0.9066 | 9.3849 |
| seasonal_anomaly_persistence | validation_2024 | 364 | 1.5273 | 0.6221 | 0.8352 | 7.3033 | 0.9176 | 9.3737 |
| trend_adjusted_climatology | validation_2024 | 364 | 1.5668 | 0.6313 | 0.7995 | 6.8988 | 0.8791 | 8.8545 |
| day_of_year_climatology | validation_2024 | 364 | 1.8835 | 0.7581 | 0.7473 | 7.0100 | 0.8407 | 8.9973 |
| raw_hko_official_forecast | validation_2024 | 0 |  |  |  |  |  |  |
| bias_corrected_hko_official_forecast | validation_2024 | 0 |  |  |  |  |  |  |
| raw_deterministic_nwp | validation_2024 | 0 |  |  |  |  |  |  |
| raw_ensemble_mean_distribution | validation_2024 | 0 |  |  |  |  |  |  |
| simple_mos_correction | validation_2024 | 0 |  |  |  |  |  |  |
