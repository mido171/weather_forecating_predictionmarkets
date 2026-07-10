# R14 Onward Robust Long-History Continuation

Generated: `2026-06-20T12:33:11.538705Z`

This continuation obeys the user's robust-only constraint: scored experiments must use at least 39 years of parsed history and at least four to five years of OOF data. Short-history RSS, radar, satellite, lightning, nowcast, and live-only feeds are not scored here.

| research_id | status | history_years | oof_period | baseline_mae | best_nonbaseline | best_mae | mae_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HKG-T24-R14 | scored | 74.579 | 1965-01-01 to 2023-12-31 | 1.7854 | r14_upper_air_core | 1.4898 | 0.2956 |
| HKG-T24-R15 | scored | 74.579 | 1965-01-01 to 2023-12-31 | 1.7807 | r15_coupling_terms | 1.2432 | 0.5374 |
| HKG-T24-R16 | scored | 76.999 | 1965-01-01 to 2023-12-31 | 1.7803 | r16_isd_regional_aggregate | 1.2585 | 0.5218 |
| HKG-T24-R17 | scored | 74.579 | 1965-01-01 to 2023-12-31 | 1.7807 | r17_era_transfer_terms | 1.2324 | 0.5483 |
| HKG-T24-R18 | not scored | 2020-2026 only | fails >=39-year robust-history gate |  | RSS official forecasts short history |  |  |
| HKG-T24-R19-R30 | not scored | mixed or dependency-limited | blocked until robust long-history source families are accepted/hardened |  | not eligible under current robust-only instruction |  |  |

Validation 2024 and locked-test dates were not accessed. The scored rows remain proxy-limited, not production-eligible, until exact operational vintage/release semantics are proven.
