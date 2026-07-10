# Verified Current State

Generated: `2026-06-20T08:32:30.245773+00:00`

Repository HEAD: `14bc7f0d0af818f013566b2a745e32784b817657`

Dirty worktree entries at audit start: `509`. These include pre-existing user changes and deleted root experiment folders; this audit did not revert them.

## Archive

- Archive path: `analysis\hkg_tmax_t24.rar`
- Archive SHA256: `cc5ec36773255f96d427fee2fc9a83c7f355a7469322c67c075116cbb3893634`
- Archive files: `30`
- Archive directories: `18`

## Core Data Facts

- Target table rows: `49459`, range `1884-01-01 00:00:00` to `2026-05-31 00:00:00`.
- Selected high-frequency table rows: `1887741`, range `2020-06-30 09:00:00+08:00` to `2026-06-18 15:30:00+08:00`.
- Feature candidate table rows: `49459`, range `1884-01-01 00:00:00` to `2026-05-31 00:00:00`.
- Archived baseline prediction rows: `14481` total, `4644` locked rows counted as metadata only.

## Baseline Reproduction

R01 recomputed metrics only for target dates before `2025-01-01`. Locked-test losses were not computed.

Champion validation reproduction status: `PASS`.

| model_id | split | n | first_date | last_date | mae | rmse | median_abs_error | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| station_state_analogue | validation_2024 | 364 | 2024-01-01 | 2024-12-31 | 1.503176 | 1.897374 | 1.298000 | 0.018473 | 1.065403 | 0.818681 | 0.909341 |
| transparent_equal_weight_blend | validation_2024 | 364 | 2024-01-01 | 2024-12-31 | 1.679223 | 2.162135 | 1.347455 | -0.233997 | 1.197493 | 0.832418 | 0.912088 |
| cutoff_station_temperature_persistence | validation_2024 | 364 | 2024-01-01 | 2024-12-31 | 1.783516 | 2.244150 | 1.500000 | -0.837912 | 1.268804 | 0.747253 | 0.859890 |
| multi_day_thermal_memory | validation_2024 | 364 | 2024-01-01 | 2024-12-31 | 1.987849 | 2.614863 | 1.578000 | 0.017376 | 1.438533 | 0.857143 | 0.923077 |
| recent10y_climatology | validation_2024 | 364 | 2024-01-01 | 2024-12-31 | 2.093277 | 2.632064 | 1.766265 | -0.510366 | 1.478601 | 0.804945 | 0.906593 |
| last_final_tmax_persistence | validation_2024 | 364 | 2024-01-01 | 2024-12-31 | 2.117033 | 2.747146 | 1.700000 | 0.000549 | 1.528732 | 0.824176 | 0.906593 |
| seasonal_anomaly_persistence | validation_2024 | 364 | 2024-01-01 | 2024-12-31 | 2.119535 | 2.742123 | 1.704992 | 0.001179 | 1.527300 | 0.835165 | 0.917582 |
| trend_adjusted_climatology | validation_2024 | 364 | 2024-01-01 | 2024-12-31 | 2.239114 | 2.771721 | 1.848284 | -0.973014 | 1.566827 | 0.799451 | 0.879121 |
| day_of_year_climatology | validation_2024 | 364 | 2024-01-01 | 2024-12-31 | 2.698125 | 3.287024 | 2.381508 | -1.921616 | 1.883533 | 0.747253 | 0.840659 |


## Date Discrepancies

- Archived first prediction date: `2021-12-30 00:00:00`.
- Declared EXP-0002 common-sample start in its date-range document: `2021-07-01`.
- Missing development/validation target dates from declared common sample: `187`.

Known missing dates requested by the goal:

| target_date | reason |
| --- | --- |
| 2022-09-24 | missing_required_hko_cutoff_temperature_at_T_minus_1_1500 |
| 2022-09-25 | missing_required_hko_cutoff_temperature_at_T_minus_1_1500 |
| 2022-09-26 | missing_required_hko_cutoff_temperature_at_T_minus_1_1500 |
| 2024-12-19 | missing_required_hko_cutoff_temperature_at_T_minus_1_1500 |
| 2024-12-20 | missing_required_hko_cutoff_temperature_at_T_minus_1_1500 |


The five named missing dates are explained by absent HKO cutoff-temperature features. The larger July-December 2021 discrepancy remains a reproduction blocker because the current generator code does not contain the old effective start gate, while the archived rows begin exactly at the first 15:00 pressure-feature date.

## Four-Year OOF Feasibility

- Strict user requirement: `at least four years of out-of-fold test data for all experiments`.
- Long-history target/daily-climate families: `FEASIBLE for target-only and daily-climate families with 1884-2026 coverage, subject to as-of publication constraints`.
- Modern high-frequency development-only sample: `BLOCKED under strict rule: HKO high-frequency before validation has less than four full years before 2024`.
- Modern archived baseline sample: `BLOCKED under strict rule: archived EXP-0002 scored development starts 2021-12-30 and has about two years before validation`.
- Required handling: `Do not run/promote modern high-frequency R02-R29 experiments as satisfying the strict four-year OOF requirement unless a revised predeclared split is approved or enough prospective data accrues.`.

## Locked-Test Guard

Status: active in `hkg_tmax.hkg_t24.guard`; ordinary research access rejects dates `>= 2025-01-01`. No unlock was invoked by this audit.
