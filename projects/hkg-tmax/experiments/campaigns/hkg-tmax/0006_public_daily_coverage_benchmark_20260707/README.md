# Public Daily Coverage Benchmark

Generated: `2026-07-08T08:36:14.244429Z`

Target UTC day: `2026-07-07`

This benchmark fetched and normalized one complete practical daily coverage set:

- GFS: `00/06/12/18Z`, f024, selected HKG weather feature pack.
- GEFS control: `00/06/12/18Z`, f024, selected HKG weather feature pack.
- Himawari-9: B13 HKG segment `S0510`, every 10 minutes, 144 scans.

## Result

| Metric | Value |
|---|---:|
| total download + normalize seconds | 578.23 |
| model items ok | 8 / 8 |
| Himawari items ok | 141 / 144 |
| total downloaded MB | 520.70 |
| model normalized station rows | 8 |
| Himawari normalized scan rows | 141 |

See `normalized/daily_coverage_benchmark_summary.json` for timings, bytes, and attribute counts.


## GEFS Repair

GEFS was rerun after removing `MSLET`, which caused NOMADS filter HTTP 500. The table above uses the repaired equivalent timing.
