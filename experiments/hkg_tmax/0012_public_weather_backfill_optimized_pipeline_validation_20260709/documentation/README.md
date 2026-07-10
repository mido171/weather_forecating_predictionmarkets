# Experiment 0012 Documentation

This directory is the maintainer and operator handoff for experiment
`0012_public_weather_backfill_optimized_pipeline_validation_20260709`.

The experiment established that the optimized GFS, GEFS control, and Himawari B13/S0510
pipeline can fetch historical source issues, normalize HKO station and surrounding-area
features, persist point-in-time metadata and values in Postgres, and remove transient raw
payloads without leaving staging residue.

## Read Order

1. [PUBLIC_WEATHER_BACKFILL_IMPLEMENTATION_AND_VALIDATION.md](PUBLIC_WEATHER_BACKFILL_IMPLEMENTATION_AND_VALIDATION.md)
   explains the implementation, control flow, database contract, validation sequence,
   failures, recovery behavior, and operating commands.
2. [LIVE_POSTGRES_MEASUREMENT_SNAPSHOT_20260710.md](LIVE_POSTGRES_MEASUREMENT_SNAPSHOT_20260710.md)
   records the live row counts, relation sizes, indexes, row widths, and PostgreSQL settings
   used for the capacity calculation.
3. [POSTGRES_STORAGE_CAPACITY_ESTIMATE_2017_TO_2026.md](POSTGRES_STORAGE_CAPACITY_ESTIMATE_2017_TO_2026.md)
   projects the complete `2017-01-01` through `2026-07-10` retained database footprint and
   gives minimum and recommended free-space targets.

## Headline Results

| Question | Answer |
| --- | --- |
| Final experiment state | `accepted_with_notes` |
| Significance score | `89/100` |
| Fresh one-day runtime, p50 | `676.0 s` (`11.3 min`) |
| Fresh one-day runtime, p90 | `714.1 s` (`11.9 min`) |
| Measured two-worker throughput | `6.9 min per day-equivalent` |
| Full 2017-present runtime estimate | About `16.6 days` continuously; budget `18-21 days` |
| Peak aggregate raw staging | `241,375,243 bytes` (`230.2 MiB`) |
| Final raw staging | `0 bytes` |
| Projected retained Postgres footprint | `121.4 GB` decimal (`113.0 GiB`) |
| Conservative retained-size range | `100-150 GB` decimal |
| Recommended free space before launch | `180-200 GiB`; `230-250 GiB` if same-disk rebuilds/backups are required |

## Scope

The capacity estimate covers the three sources validated by experiment 0012:

- GFS deterministic, cycles `00/06/12/18`, leads `0..48h` every `3h`.
- GEFS control, cycles `00/06/12/18`, leads `0..48h` every `3h`.
- Himawari B13, segment `S0510`, ten-minute scans.

Radar, full GRIB grids, full satellite rasters, retained raw payloads, Parquet mirrors, database
backups, and unrelated schemas are excluded unless a document explicitly says otherwise.

## Authoritative Evidence

The concise experiment outcome remains in the parent [RESULTS.md](../RESULTS.md). The corrected
29-day run is under `../full_robustness_jun10_jul8_rerun4/`, and the successful cleanup retry is
under `../targeted_gfs_retry_after_full_rerun4/`.
