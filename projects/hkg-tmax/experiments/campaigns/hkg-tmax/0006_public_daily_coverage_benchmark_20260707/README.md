# 0006 Public daily coverage benchmark

Status: `complete`. Target UTC day: 2026-07-07.

## Question and result

Measure one practical day of GFS, GEFS control, and 10-minute Himawari B13
coverage.

| Metric | Result |
|---|---:|
| Download + normalize time | 578.23 s |
| Model items | 8/8 |
| Himawari scans | 141/144 |
| Downloaded volume | 520.70 MB |
| Model station rows | 8 |
| Himawari scan rows | 141 |

GEFS was repaired by removing `MSLET` after NOMADS returned HTTP 500.

## Decision and limitations

One-day feasibility passed. Three Himawari scans were unavailable; this run did
not prove a long-history backfill.

## Reproduce and evidence

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_public_daily_coverage_gfs_gefs_himawari.py --help
```

`normalized/daily_coverage_benchmark_summary.json` is the machine summary.
CSV manifests and `STATUS.yaml` preserve coverage and timestamps.
