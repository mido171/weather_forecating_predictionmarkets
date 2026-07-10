# HKG T24 Point-In-Time Eligibility

Generated: `2026-06-20T10:26:20.551301Z`

- Cutoff: T-1 15:00:00 Asia/Hong_Kong.
- Governing timestamp: `available_at`.
- Locked-test ordinary access denied from: `2025-01-01`.
- feature entries: `13`
- registered in existing eligibility registry: `6`
- not operationally allowed by tier: `8`

## Four-Year OOF Gate

- strict requirement: `at least four years of OOF test data for all experiments`
- long-history status: `PASS` - target/daily climate development history before validation 2024: 140.00 years available
- modern high-frequency status: `BLOCKED` - modern HKO high-frequency development history before validation 2024: 3.50 years available, requires at least 4.0 years

| Feature | Tier | Registered | Available-at column | Non-null rows | Range | Notes |
|---|---|---:|---:|---:|---|---|
| target_tmax_c | TARGET_ONLY | True | False | 49459 | 1884-01-01 to 2026-05-31 | Target label only. Never a predictor. |
| hko_temp_at_tminus1_1500_c | SILVER_OPERATIONAL_REPLAY | True | True | 1932 | 2020-07-01 to 2026-05-31 | Use latest observation available by cutoff. |
| hko_rh_at_tminus1_1500_pct | SILVER_OPERATIONAL_REPLAY | True | True | 1931 | 2020-07-01 to 2026-05-31 | Moisture state at cutoff. |
| hko_mslp_at_tminus1_1500_hpa | SILVER_OPERATIONAL_REPLAY | True | True | 1608 | 2021-12-30 to 2026-05-31 | Synoptic pressure state at cutoff. |
| hko_tminus1_max_so_far_1500_c | SILVER_OPERATIONAL_REPLAY | True | True | 318 | 2020-07-01 to 2021-05-14 | Only valid for T-1, not target day T. |
| hko_tminus1_min_so_far_1500_c | FORBIDDEN | False | True | 1920 | 2020-07-01 to 2026-05-31 |  |
| hko_temp_tminus1_1200_c | FORBIDDEN | False | True | 1930 | 2020-07-01 to 2026-05-31 |  |
| hko_temp_tminus1_0900_c | FORBIDDEN | False | True | 1933 | 2020-07-01 to 2026-05-31 |  |
| hko_mslp_tminus1_1200_hpa | FORBIDDEN | False | True | 1606 | 2021-12-30 to 2026-05-31 |  |
| hko_temp_3h_change_to_cutoff_c | FORBIDDEN | False | False | 1930 | 2020-07-01 to 2026-05-31 |  |
| hko_temp_6h_change_to_cutoff_c | FORBIDDEN | False | False | 1931 | 2020-07-01 to 2026-05-31 |  |
| hko_mslp_3h_change_to_cutoff_hpa | FORBIDDEN | False | False | 1606 | 2021-12-30 to 2026-05-31 |  |
| hko_tminus2_official_tmax_c | SILVER_OPERATIONAL_REPLAY | True | False | 49455 | 1884-01-03 to 2026-05-31 | Useful benchmark feature, but publication timing must be proven before production. |
