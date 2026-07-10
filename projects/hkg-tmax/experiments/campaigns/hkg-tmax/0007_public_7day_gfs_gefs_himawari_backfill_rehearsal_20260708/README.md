# 0007 Seven-day public-weather backfill rehearsal

Status: `complete_with_provider_gaps`. Window: 2026-07-01 through 2026-07-07 UTC.

## Result

| Metric | Value |
|---|---:|
| Requested fetches | 1,960 |
| Successful fetches | 1,671 |
| Invalid/error fetches | 289 |
| Model station rows | 680 |
| Model bbox-variable rows | 9,856 |
| Himawari scan rows | 991 |
| Raw volume during run | 9.626 GB |
| Compact normalized volume | 9.00 MB |

GFS and Himawari were broadly fetchable. Older GEFS control requests had live
NOMADS availability gaps. Model decoding required process isolation for
cfgrib/eccodes; threaded decoding was unstable. Himawari normalization
parallelized cleanly.

## As-of contract

Model rows retained issue, valid, and conservative availability timestamps.
Himawari retained observation, file-creation, and conservative availability
timestamps.

## Decision and limitations

The rehearsal proved the workflow and exposed provider/decoder constraints; it
did not establish complete GEFS history. Experiment 0012 is the later accepted
DB-backed pipeline evidence.

## Reproduce and evidence

```powershell
.\.venv\Scripts\python.exe scripts\run_public_gfs_gefs_himawari_7day_backfill_rehearsal.py --help
```

Compact JSON/CSV manifests under `normalized/` preserve the recorded run.
References to non-retained raw files in historical prose are provenance only.
