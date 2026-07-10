# HKG Tmax Probability Distribution Methods V2 Model Card

Supreme method after V2 gates: `B4_hierarchical_residual_pmf`.

Scope: weather probability distribution only. No market prices, EV, order books, Kelly sizing, PnL, market-implied blending, or trade recommendations are used or emitted.

Target: HKO Daily Extract one-decimal HKG daily maximum temperature bucket.
Forecast surface: strict HKO Info.gov local forecast rows selected at the configured pre-target cutoffs.
Primary cutoff: T-1 23:59 HKT. Sensitivity cutoffs: T-1 18:00 and T-1 21:00 HKT.

Promotion rule: challengers must beat B4 by at least 1.5% RPS on folds 1-4 and 1.0% RPS on the 2022-2023 presealed holdout, while not worsening NLL by more than 0.005 or Brier by more than 0.002.

Champion normalized RPS: 0.041524
Champion NLL: 1.037181
Champion Brier: 0.045921
Champion ECE: 0.019859
Champion gates: `reference`

Leakage audit: `pass` with total violations `0`.
Row-identity gate: `pass` with violations `0`.
Label first-publication audit: `ok`, bucket changes `0`.

## Top Leaderboard Rows

| rank | method | family | rps | relative_rps_gain_vs_b4 | fold14_relative_rps_gain_vs_b4 | presealed_relative_rps_gain_vs_b4 | nll | brier | gates | champion_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B5_kernel_analog_pmf | analog_pmf | 0.041287 | 0.005686 | 0.005332 | 0.007601 | 1.075467 | 0.045778 | fail:fold14_rps_gain,presealed_rps_gain,nll | False |
| 2 | H1_b4_challenger_linear_pool | hybrid_pool | 0.041470 | 0.001290 | 0.000641 | 0.004802 | 1.034210 | 0.045862 | fail:fold14_rps_gain,presealed_rps_gain | False |
| 3 | S1_conservative_simplex_stack | stack | 0.041472 | 0.001232 | 0.000961 | 0.002699 | 1.032837 | 0.045867 | fail:fold14_rps_gain,presealed_rps_gain | False |
| 4 | T1_time_decay_b4 | time_decay_residual_pmf | 0.041486 | 0.000915 | -0.000188 | 0.006882 | 1.031941 | 0.045941 | fail:fold14_rps_gain,presealed_rps_gain | False |
| 5 | K2_B4_monotone_cdf_projected | calibration | 0.041524 | 0.000000 | 0.000000 | 0.000000 | 1.037181 | 0.045921 | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 6 | B4_hierarchical_residual_pmf | residual_pmf | 0.041524 | 0.000000 | 0.000000 | 0.000000 | 1.037181 | 0.045921 | reference | True |
| 7 | K0_B4_identity | calibration | 0.041524 | 0.000000 | 0.000000 | 0.000000 | 1.037181 | 0.045921 | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 8 | K1_B4_power_calibrated | calibration | 0.041531 | -0.000173 | -0.000147 | -0.000311 | 1.036386 | 0.045904 | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 9 | B3_forecast_level_residual_pmf | residual_pmf | 0.041616 | -0.002229 | -0.003200 | 0.003027 | 1.038655 | 0.045978 | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 10 | E2_student_t_emos | emos | 0.041658 | -0.003229 | -0.004081 | 0.001380 | 1.047414 | 0.046140 | fail:fold14_rps_gain,presealed_rps_gain,nll,overall_rps_not_better | False |
| 11 | B2_month_residual_pmf | residual_pmf | 0.041666 | -0.003440 | -0.005615 | 0.008332 | 1.041036 | 0.046030 | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 12 | B1_global_residual_pmf | residual_pmf | 0.041700 | -0.004248 | -0.006562 | 0.008271 | 1.041529 | 0.046147 | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |

## Methods Benchmarked

- V1 baselines and champion family: B0-B6, P1/P2, C1/C2, K0-K2, S1.
- V2 challengers: E1 normal EMOS, E2 Student-t EMOS, E3 two-piece normal EMOS, G1 tree location-scale, Q1 quantile CDF gradient boosting, Q2 threshold CDF gradient boosting, T1 time-decay B4, H1 conservative B4-plus-challenger pool.