# Hypothesis

Point-in-time GFS, GEFS control, and Himawari B13/S0510 features can be acquired through the full
Postgres-backed path with conservative availability timestamps, complete successful-issue
features, idempotent natural keys, and zero final raw staging.

The optimized implementation should materially reduce wall-clock throughput relative to serial
day processing while keeping worker counts and transient disk bounded. The experiment is an
acquisition/persistence validation; it makes no claim that these features improve Tmax MAE.
