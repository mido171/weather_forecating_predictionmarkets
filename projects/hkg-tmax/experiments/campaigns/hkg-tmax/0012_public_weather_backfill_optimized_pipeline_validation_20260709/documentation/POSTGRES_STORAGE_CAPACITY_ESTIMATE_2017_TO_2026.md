# Postgres Capacity Estimate for the 2017-2026 Public Weather Backfill

## Executive Answer

For the current normalized schema and the experiment 0012 source scope, the estimated retained
Postgres footprint from `2017-01-01` through `2026-07-10` is:

> **121.4 GB decimal, equivalent to 113.0 GiB.**

The estimate includes table heaps, current indexes, tuple/page overhead reflected by the live
database, source-issue metadata, one ingest event per attempted issue, and one ingest-run record
per day shard. It does not include raw GRIB/Himawari files because those are deleted after each
successful DB commit.

The correct capacity decision is larger than the 121.4 GB retained footprint:

| Capacity level | Free space before starting | Meaning |
| --- | ---: | --- |
| Bare operational minimum | `150 GiB` | Expected retained data plus normal WAL/autovacuum margin, with little room for other growth |
| Recommended | `180-200 GiB` | Reasonable protection against estimate error, reruns, WAL variation, and continued daily ingestion |
| Maintenance-safe on the same disk | `230-250 GiB` | Adds room for a large index rebuild, local backup, or temporary second copy of a major relation |

The host had `226.97 GiB` free when measured. The expected net increase from the existing
`weather_backfill` footprint is approximately `112 GiB`, leaving roughly `115 GiB` if nothing
else on the volume grows. The backfill fits, but same-disk maintenance and unrelated machine
growth need active monitoring.

## Scope and Date Contract

Included sources:

- GFS deterministic: four cycles/day, 17 leads/cycle, 68 source issues/day.
- GEFS control: four cycles/day, 17 leads/cycle, 68 source issues/day.
- Himawari B13/S0510: ten-minute scans, 144 expected source issues/day.

Included persisted objects:

- `weather_backfill.source_issue`
- `weather_backfill.station_feature`
- `weather_backfill.area_feature`
- `weather_backfill.ingest_event`
- `weather_backfill.ingest_run`
- Existing indexes on those tables

Excluded objects:

- Radar data, because radar was outside the accepted experiment 0012 robustness run.
- Full GRIB grids and full satellite rasters.
- Raw transient payloads after commit.
- Optional Parquet mirrors and normalized artifacts, which are currently unused by this path.
- PostgreSQL base backups, logical dumps, replicas, and archived WAL.
- Other database schemas in `hkg_tmax_research`.

The main projection is inclusive from `2017-01-01` through the current local date
`2026-07-10`, or `3,478` calendar days. A second check through the last fully validated date,
`2026-07-08`, covers `3,476` days and yields essentially the same result: `121.3 GB` decimal.

## Why This Estimate Is Evidence-Based

The estimate uses three measured inputs from the live Postgres database:

1. Actual feature counts from the accepted 29-day experiment window.
2. Actual `pg_total_relation_size` divided by exact live row counts for each target table.
3. Actual source-specific ingest-event widths from experiment 0012.

This approach carries forward the physical cost of the current indexes and page layout. It is
more realistic than summing SQL column data types, which would omit variable-length text,
JSONB, tuple headers, alignment, page slack, and B-tree storage.

The detailed source measurements are preserved in
[LIVE_POSTGRES_MEASUREMENT_SNAPSHOT_20260710.md](LIVE_POSTGRES_MEASUREMENT_SNAPSHOT_20260710.md).

## Measured Daily Population

The 29-day window produced the following mean daily inventory:

| Object | Mean rows/day | Derivation |
| --- | ---: | --- |
| Source issues | 280.00 | Fixed inventory: 68 GFS + 68 GEFS + 144 Himawari |
| Successful source issues | 277.76 | 8,055 successful issues / 29 days |
| Station features | 15,815.03 | 458,636 rows / 29 days |
| Area features | 28,546.90 | 827,860 rows / 29 days |
| Ingest events | 280.00 | One terminal processing/error event per attempted source issue in a clean run |
| Ingest runs | 1.00 | One day-shard run per calendar day |

The capacity model intentionally keeps all 280 `source_issue` rows/day, including missing
Himawari objects, because the pipeline records expected-but-unavailable source issues with their
404 status and leakage timestamp.

## Measured Physical Cost Per Row

| Table | Total bytes/live row | What is included |
| --- | ---: | --- |
| `source_issue` | 1,979.89 | Heap, three indexes, page overhead, and metadata JSONB |
| `station_feature` | 658.67 | Heap, primary key, lookup index, and repeated feature context |
| `area_feature` | 826.65 | Heap, primary key, lookup index, and repeated variable context |
| `ingest_event` | 887.76 baseline | Heap and primary key; adjusted by source-specific event payload width below |
| `ingest_run` | 10,678.86 | Heap, primary key, and TOASTed config/summary JSONB |

The event estimate is source-weighted because model events carry more fetch/range metadata than
Himawari events. The logical event widths measured in experiment 0012 were approximately
2,008.92 bytes for GFS success, 1,410.65 bytes for GEFS success, and 446.16 bytes for Himawari
success. Applying the live event table's logical-to-physical factor gives approximately 1.097 GB
for one event per issue over the complete period.

## Projection Formula

For station and area features:

```text
projected_rows = measured_29_day_rows * (3,478 / 29)
projected_bytes = projected_rows * measured_physical_bytes_per_live_row
```

For source issues and ingest events:

```text
projected_source_issues = 280 issues/day * 3,478 days = 973,840 issues
projected_ingest_events = 1 event/issue * 973,840 issues = 973,840 events
```

For run records:

```text
projected_ingest_runs = 1 run/day * 3,478 days = 3,478 runs
```

## Projected Rows

| Table | Projected rows through 2026-07-10 |
| --- | ---: |
| `source_issue` | 973,840 |
| `station_feature` | 55,004,690 |
| `area_feature` | 99,286,106 |
| `ingest_event` | 973,840 |
| `ingest_run` | 3,478 |
| **Station + area feature rows** | **154,290,796** |

The two feature tables dominate both row count and storage. Metadata tables are operationally
important but represent only about 2.5% of the retained footprint.

## Projected Retained Storage by Table

| Table | Projected bytes | Decimal GB | GiB | Share |
| --- | ---: | ---: | ---: | ---: |
| `area_feature` | 82,075,224,807 | 82.075 | 76.439 | 67.6% |
| `station_feature` | 36,229,681,470 | 36.230 | 33.742 | 29.9% |
| `source_issue` | 1,928,096,467 | 1.928 | 1.796 | 1.6% |
| `ingest_event` | 1,097,392,668 | 1.097 | 1.022 | 0.9% |
| `ingest_run` | 37,141,065 | 0.037 | 0.035 | Less than 0.1% |
| **Total** | **121,367,536,477** | **121.368** | **113.032** | **100%** |

The projected average retained growth is `34.896 MB` decimal per historical day. After the
backfill, continued ingestion at the same feature shape adds approximately `12.74 GB` decimal
per non-leap year.

## Projected Feature Storage by Source

This table allocates feature rows to their originating source and applies the measured physical
cost of the target feature table.

| Source | Station rows | Area rows | Projected feature storage |
| --- | ---: | ---: | ---: |
| GFS | 4,187,512 | 48,288,552 | 42.676 GB |
| GEFS control | 3,978,712 | 45,574,153 | 40.295 GB |
| Himawari B13/S0510 | 46,838,466 | 5,423,401 | 35.334 GB |
| **Total** | **55,004,690** | **99,286,106** | **118.305 GB** |

GFS and GEFS use most `area_feature` storage because each model issue produces statistics for
many variable/level combinations. Himawari uses most `station_feature` storage because every
successful ten-minute scan writes approximately 95 scalar HKO/window attributes.

## Heap Versus Index Storage

Applying the current index share of each relation gives this approximate split:

| Table | Heap/auxiliary GB | Index GB | Total GB |
| --- | ---: | ---: | ---: |
| `area_feature` | 58.776 | 23.299 | 82.075 |
| `station_feature` | 23.946 | 12.284 | 36.230 |
| `source_issue` | 1.664 | 0.264 | 1.928 |
| `ingest_event` | 1.068 | 0.029 | 1.097 |
| `ingest_run` | 0.036 | 0.001 | 0.037 |
| **Total** | **85.490** | **35.877** | **121.368** |

The projected feature indexes alone are approximately 35.6 GB decimal. Query plans, cache hit
rates, and maintenance time therefore need measurement after loading representative historical
years; row count alone is not enough to declare the final database fast.

## Uncertainty and Sensitivity

### Himawari source availability

The 29-day sample averaged 141.76 successful scans/day rather than 144 because 65 source objects
returned 404. If every historical Himawari scan were available, the full projection would add
approximately:

- 740,574 station rows.
- 85,751 area rows.
- 0.559 GB decimal.

Himawari missingness is therefore not the main sizing uncertainty.

### Historical model-era differences

Older GFS/GEFS archives can expose different message inventories, naming, or availability. The
normalizer only persists selected messages that decode successfully. Historical years may
therefore produce a different number of model features per issue than the 2026 sample. This can
move storage in either direction.

### Repeated runs

Feature and source-issue natural keys are upserted, so rerunning completed dates does not create
duplicate feature rows. `ingest_event` and `ingest_run` are append-oriented audit tables. A full
rerun can therefore add roughly another 1.1 GB of events plus about 37 MB of run summaries even
when feature rows are unchanged. Production reruns should use `--skip-existing-complete` and
target only failed issue keys.

### Bloat and page fill

The live ratios were measured after autovacuum and with no material dead-tuple estimate in the
two feature tables. Future updates, changing JSON context widths, different B-tree fill, and
long-running transactions can increase physical storage. Applying a 20% planning band to the
121.4 GB point estimate yields approximately 97-146 GB. The documented conservative range is
therefore rounded to **100-150 GB decimal**.

## Retained DB Space Versus Transient Space

These are separate budgets:

| Space type | Measured/estimated amount | Persistence |
| --- | ---: | --- |
| Final Postgres relations | 121.4 GB decimal | Retained |
| Net new `weather_backfill` growth from current state | About 120.3 GB decimal, or 112.0 GiB | Retained |
| Peak raw staging in the 29-day validation | 230.2 MiB | Deleted after DB commits |
| Final raw staging | 0 bytes | None |
| WAL | Configured around a 1 GB target but may temporarily exceed it | Recycled because archive mode is off |
| Backups/index rebuild copies | Not included | Operator-dependent |

Raw GRIB/Himawari download volume is large over the life of the backfill, but it does not add to
the retained disk estimate because every payload is deleted after its source issue and features
commit, or after failure recording when failed raw retention is disabled.

## Capacity Recommendation for This Machine

At measurement time:

- `C:` total: `1,429.65 GiB`.
- `C:` used: `1,202.68 GiB`.
- `C:` free: `226.97 GiB`.
- Current `weather_backfill`: `0.989 GiB`.
- Projected final `weather_backfill`: `113.032 GiB`.
- Projected net increase: approximately `112.04 GiB`.
- Simplified post-load free space if nothing else changes: approximately `114.9 GiB`.

The machine has enough measured capacity for the expected backfill. The operator should still
stop or pause if free disk approaches the script's configured safety threshold because browser
caches, other databases, logs, Windows updates, and unrelated workloads share the same volume.

For routine operation, retain at least `50 GiB` free after loading. For a concurrent rebuild of
the projected `area_feature` table or its 23.3 GB index set, substantially more temporary space
is prudent. A same-disk base backup or database clone should not be started unless its complete
space requirement has been measured first.

## Database Performance Implication

The current design projects approximately 154.3 million feature rows in two unpartitioned
tables. PostgreSQL 16 can store this volume, but good performance depends on query shape:

- Station queries should constrain `station_id`, `feature_name`, and `available_at_utc` so they
  can use `ix_weather_station_feature_lookup`.
- Area queries should constrain `area_key`, `variable_name`, `statistic`, and
  `available_at_utc` so they can use `ix_weather_area_feature_lookup`.
- Leakage-safe training queries must join `source_issue` and apply the cutoff to
  `available_at_utc`; broad full-history scans should be measured with `EXPLAIN (ANALYZE,
  BUFFERS)` after representative years are loaded.
- Autovacuum progress, index growth, cache hit rate, and long-running transactions should be
  monitored during the load.

Partitioning is not required by experiment 0012 and was not implemented. Before loading the
full decade, monthly time partitioning or a more compact wide-feature representation may be
worth a separate benchmark if operators require frequent date-range deletion, fast cold-year
maintenance, or materially lower disk use. Any such redesign changes the current sizing model
and must be validated before replacing this estimate.

## Acceptance Decision

Use the following numbers for planning the current schema:

- **Expected retained footprint:** `121.4 GB` decimal (`113.0 GiB`).
- **Conservative retained range:** `100-150 GB` decimal.
- **Expected historical load rate:** `34.9 MB/day` retained.
- **Recommended free capacity before launch:** `180-200 GiB`.
- **Estimated free capacity currently available:** `226.97 GiB`.
- **Expected free capacity after load, assuming no other growth:** about `115 GiB`.

This estimate is strong enough for capacity planning because it is derived from the live schema,
live indexes, exact feature counts, and physical relation sizes. It is not a contractual maximum;
historical message shape, repeated audit runs, table bloat, backups, or schema changes can exceed
the point estimate.
