# 0011 Public Weather Speed Optimization

This experiment benchmarks faster, low-disk public weather acquisition for HKG Tmax inputs.

Scope:
- GFS and GEFS control GRIB feature payloads through S3 `.idx` plus byte-range extraction.
- Himawari-9 B13/S0510 HSD fetch and HKO-window normalization.
- Worker-count tests that improve throughput without trying to saturate the CPU.

Primary harness:

```powershell
python .\scripts\benchmark_public_weather_speed_optimization.py --date 2026-06-21
```

Raw payload policy: all fetched raw payloads are transient and deleted inside each benchmark worker after fetch-only or fetch-normalize completion. Trial staging folders are removed at the end of each trial.
