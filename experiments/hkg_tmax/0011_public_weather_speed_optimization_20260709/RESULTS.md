# Results

Benchmark date: `2026-06-21`.

Prior rehearsal reference: GFS+GEFS+Himawari serial day path averaged about `2,728s` (`45.5 min`) before these speed tests.

## Trial Summary

| Trial | What Changed | Wall Time | OK | Bytes | Notes |
|---|---:|---:|---:|---:|---|
| `wgrib2_probe` | local binary check | `0.1s` | n/a | n/a | `wgrib2` not on PATH |
| `model_fetch_s3_w8_r4` | current S3 per-message range fetch, 8 object workers, 4 range workers | `442.1s` | `136/136` | `1.367 GB` | best non-coalesced model setting |
| `model_fetch_s3_w8_c0` | adjacent selected-message coalescing, same bytes | `323.0s` | `136/136` | `1.367 GB` | safe low-hanging production candidate |
| `model_fetch_s3_w8_c1m` | coalesce gaps up to 1 MB | `256.3s` | `136/136` | `1.446 GB` | fastest, but may include extra GRIB messages |
| `model_fetch_s3_w16_r4` | 16 object workers, 4 range workers | `469.7s` | `136/136` | `1.367 GB` | slower than 8 workers |
| `model_fetch_s3_w16_r8` | 16 object workers, 8 range workers | `557.3s` | `136/136` | `1.367 GB` | worse; over-parallelized S3 ranges |
| `himawari_fetch_normalize_w4` | Himawari fetch+decode, 4 workers | `212.2s` | `142/144` | `456.9 MB` | two source-side 404 scans |
| `himawari_fetch_normalize_w8` | Himawari fetch+decode, 8 workers | `114.2s` | `142/144` | `456.9 MB` | best Himawari setting tested |
| `model_fetch_normalize_sample_w2` | 12-task model fetch+cfgrib normalize sample, 2 process workers | `97.0s` | `12/12` | `111.9 MB` | cfgrib worked in process workers |

## Range Coalescing

The model-fetch request bottleneck is real:

| Strategy | Range GETs/day | Bytes/day | Request Multiplier | Byte Multiplier |
|---|---:|---:|---:|---:|
| Current per-message ranges | `2,140` | `1.367 GB` | `1.00x` | `1.00x` |
| Coalesce adjacent selected messages (`gap=0`) | `1,004` | `1.367 GB` | `0.47x` | `1.00x` |
| Coalesce gaps up to `1 MB` | `868` | `1.446 GB` | `0.41x` | `1.06x` |

The safe production candidate is `gap=0`: same selected messages, same bytes, fewer HTTP requests. The `1 MB` version is faster, but it may include extra complete GRIB messages and therefore needs explicit downstream variable/level filtering before production.

## Normalization

The 12-task model sample with 2 process workers:
- Mean fetch time per object: `7.36s`.
- Mean cfgrib normalization time per object: `6.57s`.
- All 12 normalized successfully.

Projected across 136 model objects, model normalization alone is roughly `447s` with 2 process workers if the sample is representative. With a bounded fetch-normalize pipeline, full model processing likely lands around `8-10 min/day` unless we replace or accelerate cfgrib normalization.

## Disk And Cleanup

Raw payloads were transient only. The final audit shows:
- `_s0011` staging root absent.
- `0` raw `.grib2`, `.bz2`, `.DAT`, or `.idx` payloads under this experiment folder.

Measured max staging samples stayed below `46 MB`; in-flight instantaneous usage can be slightly higher between 1-second samples, but the architecture is not accumulating day-scale raw payloads.

## Caveats

- CPU telemetry was unavailable because `psutil` is not installed in the active venv.
- The first non-coalesced model trial originally had a stale sampled `staging_end_bytes` value before the long-path cleanup patch. The aggregate CSV is corrected to `0`; the original stale sample is retained in `r/m8r4/summary.json` as `original_sampled_staging_end_bytes`.
