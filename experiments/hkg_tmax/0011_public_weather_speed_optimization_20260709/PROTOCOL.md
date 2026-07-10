# Protocol

Benchmark date: `2026-06-21` by default.

Default trials:
- `wgrib2_probe`: check whether local `wgrib2` is available.
- `model_fetch_s3_w8_r4`: GFS+GEFS full-day S3 byte-range fetch, 8 object workers, 4 range workers per object.
- `model_fetch_s3_w8_c0`: same worker settings, but coalesce adjacent selected GRIB messages into one byte-range request when the gap is exactly 0 bytes.
- `model_fetch_s3_w16_r4`: same with 16 object workers and 4 range workers.
- `model_fetch_s3_w16_r8`: same with 16 object workers and 8 range workers.
- `himawari_fetch_normalize_w4`: full-day Himawari B13/S0510 fetch+normalize with 4 workers.
- `himawari_fetch_normalize_w8`: same with 8 workers.
- `model_fetch_normalize_sample_w2`: small GFS+GEFS cfgrib fetch+normalize sample with 2 process workers.

Additional non-default trial:
- `model_fetch_s3_w8_c1m`: coalesce selected GRIB ranges across gaps up to 1 MB. This is a speed probe, not a drop-in production candidate unless downstream normalization filters strictly, because it may include extra complete GRIB messages between selected messages.

Recorded metrics:
- Wall time, task count, success/failure count.
- Bytes fetched and MB/s wall throughput.
- Per-task p50/p90 fetch and normalization latency.
- Mean/max CPU percent when `psutil` is installed.
- Max and final staging bytes.
- First failure examples.

The harness deletes raw payloads inside each worker and removes trial staging after each trial.
