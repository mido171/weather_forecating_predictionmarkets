# 0004 Station-hour residual information atlas

Status: `information_gain_positive_no_promote`. Significance: 82/100.

## Question and as-of contract

Test whether Info.gov HKO and neighboring-station observations available before
T-1 23:59 HKT explain the official Tmax forecast residual. Forecast issue,
observation, and dispatch timestamps all had to be at or before cutoff; only
the preceding 24-hour observation window was eligible. No 2024+ confirmation
row was used.

## Method and result

The atlas analyzed 8,762 target-frame rows, 228,391 joined hourly rows,
4,620,370 station-long rows, 1,083 features, and 27 stations. It measured
correlation, temporal stability, quantile spread, and guarded single-feature
walk-forward actionability.

Best guarded feature:
`hko__latest_temp_minus_official_min_c`.

| Metric | Value |
|---|---:|
| Candidate MAE | 0.919812 |
| Official MAE | 0.927334 |
| Bias-only MAE | 0.922780 |
| Delta vs bias-only | -0.002969 C |
| Folds beating bias-only | 3 of 4 |

The strongest families represented forecast contradiction and retained
late-window heat. The incremental gain after bias correction was too small for
promotion.

## Decision

Information gain was real and supports controlled specialist work; no champion
changed.

## Reproduce and evidence

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_0004_station_hour_residual_information_atlas.py
```

Set `HKG_TMAX_DATABASE_URL` first. `results/metrics.json` is the compact
machine record; `DATA_MANIFEST.yaml`, `RUN_CONFIG.yaml`, and `STATUS.yaml`
preserve the run contract.
