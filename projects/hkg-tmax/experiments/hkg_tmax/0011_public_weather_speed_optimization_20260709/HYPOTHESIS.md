# Hypothesis

The current lean DB backfill is disk-safe but slow because it processes each source object serially within a day. Throughput should improve materially by:

1. Keeping S3 `.idx` plus byte-range GRIB extraction, avoiding full GRIB downloads.
2. Running bounded per-object model fetches in parallel while keeping range workers capped.
3. Running Himawari fetch plus decode in a bounded thread pool because the workload mixes network I/O and NumPy-heavy decoding.
4. Keeping model normalization conservative in process workers because cfgrib/eccodes can be unstable under aggressive threaded use.

Success means a credible path from roughly 45 minutes per GFS+GEFS+Himawari day toward the 3-5 minute target, without large raw staging or excessive CPU use.
