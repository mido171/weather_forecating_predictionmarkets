# Reproduce

From the repository root:

```powershell
python .\scripts\benchmark_public_weather_speed_optimization.py --date 2026-06-21
```

Run a smaller smoke:

```powershell
python .\scripts\benchmark_public_weather_speed_optimization.py --date 2026-06-21 --max-tasks 8 --trials wgrib2_probe,model_fetch_s3_w8_r4,himawari_fetch_normalize_w4
```

Run only full-day Himawari worker comparison:

```powershell
python .\scripts\benchmark_public_weather_speed_optimization.py --date 2026-06-21 --trials himawari_fetch_normalize_w4,himawari_fetch_normalize_w8
```

Run only model coalescing comparisons:

```powershell
python .\scripts\benchmark_public_weather_speed_optimization.py --date 2026-06-21 --trials model_fetch_s3_w8_c0,model_fetch_s3_w8_c1m
```

Generated compact per-trial outputs live under `r/` to avoid Windows path-length failures.
