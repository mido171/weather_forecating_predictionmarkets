# R01 Baseline Reproduction Audit

This report recomputes supplied baseline metrics from row-level predictions for development and validation only. Locked-test rows from 2025-01-01 onward were not scored.

## Result

- Champion validation row: `station_state_analogue`.
- Reproduction status: `PASS`.
- Scoreboard CSV: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\reports\hkg_t24\baseline_reproduction_r01_scoreboard.csv`.
- Missing-date CSV: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\reports\hkg_t24\baseline_reproduction_r01_missing_dates.csv`.
- Locked-test metadata rows counted but not scored: `4644`.

## Validation Scoreboard

| model_id | n | first_date | last_date | mae | rmse | median_abs_error | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| station_state_analogue | 364 | 2024-01-01 | 2024-12-31 | 1.503176 | 1.897374 | 1.298000 | 0.018473 | 1.065403 | 0.818681 | 0.909341 |
| transparent_equal_weight_blend | 364 | 2024-01-01 | 2024-12-31 | 1.679223 | 2.162135 | 1.347455 | -0.233997 | 1.197493 | 0.832418 | 0.912088 |
| cutoff_station_temperature_persistence | 364 | 2024-01-01 | 2024-12-31 | 1.783516 | 2.244150 | 1.500000 | -0.837912 | 1.268804 | 0.747253 | 0.859890 |
| multi_day_thermal_memory | 364 | 2024-01-01 | 2024-12-31 | 1.987849 | 2.614863 | 1.578000 | 0.017376 | 1.438533 | 0.857143 | 0.923077 |
| recent10y_climatology | 364 | 2024-01-01 | 2024-12-31 | 2.093277 | 2.632064 | 1.766265 | -0.510366 | 1.478601 | 0.804945 | 0.906593 |
| last_final_tmax_persistence | 364 | 2024-01-01 | 2024-12-31 | 2.117033 | 2.747146 | 1.700000 | 0.000549 | 1.528732 | 0.824176 | 0.906593 |
| seasonal_anomaly_persistence | 364 | 2024-01-01 | 2024-12-31 | 2.119535 | 2.742123 | 1.704992 | 0.001179 | 1.527300 | 0.835165 | 0.917582 |
| trend_adjusted_climatology | 364 | 2024-01-01 | 2024-12-31 | 2.239114 | 2.771721 | 1.848284 | -0.973014 | 1.566827 | 0.799451 | 0.879121 |
| day_of_year_climatology | 364 | 2024-01-01 | 2024-12-31 | 2.698125 | 3.287024 | 2.381508 | -1.921616 | 1.883533 | 0.747253 | 0.840659 |


## Discrepancies

The row-level archive confirms predictions start on `2021-12-30 00:00:00`, not `2021-07-01`.

The five named missing dates are absent because the feature candidate table has no HKO cutoff temperature at T-1 15:00 for those target dates. The July-December 2021 gap is not explained by the current feature table, which now contains cutoff temperature for that interval. It aligns with first pressure-feature availability and is therefore recorded as a baseline archive/code-version discrepancy until the exact historical generator is recovered.
