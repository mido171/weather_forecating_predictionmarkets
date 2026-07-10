# 0011 Public-weather speed optimization

Status: `completed`. Significance: 88/100.

## Question

Find faster, low-disk public GFS/GEFS/Himawari settings without unbounded CPU,
worker counts, or retained raw payloads.

## Results

| Trial | Result |
|---|---:|
| Model fetch, 8 workers / 4 ranges | 442.1 s |
| Safe adjacent-message coalescing (`gap=0`) | 323.0 s |
| 1 MB coalescing | 256.3 s, but extra-message risk |
| Himawari 4 workers | 212.2 s, 142/144 |
| Himawari 8 workers | 114.2 s, 142/144 |
| Model sample fetch+normalize, 2 processes | 97.0 s, 12/12 |

Safe `gap=0` reduced range requests from 2,140 to 1,004 with unchanged bytes.
Increasing model workers to 16 made performance worse. `wgrib2` was absent and
CPU telemetry was unavailable because `psutil` was not installed.

## Decision

Adopted direction: adjacent-only range coalescing, bounded fetch workers,
bounded normalization processes, and immediate cleanup. Nonzero-gap
coalescing remains experimental until strict variable/level filtering proves
safe.

## Reproduce and evidence

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_public_weather_speed_optimization.py --help
```

`results/trial_summary.csv`, trial JSON, `results/metrics.json`,
`RUN_CONFIG.yaml`, and `STATUS.yaml` preserve the benchmark. Current resource
policy limits workers to two unless the user explicitly authorizes more.
