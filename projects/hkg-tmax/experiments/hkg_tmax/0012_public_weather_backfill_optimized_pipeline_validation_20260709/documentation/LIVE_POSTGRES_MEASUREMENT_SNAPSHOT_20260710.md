# Live Postgres Measurement Snapshot, 2026-07-10

## Purpose

This snapshot records the live measurements used by the experiment 0012 capacity model. It is
an observation of the local `hkg_tmax_research` database, not a generic PostgreSQL sizing rule.
The database was queried read-only at `2026-07-09T22:44:26Z`, which was `2026-07-10` local time
in Europe/Stockholm.

Credentials are intentionally omitted. The measurement connected through the existing local
`HKG_TMAX_DATABASE_URL` contract and queried schema `weather_backfill`.

## PostgreSQL Environment

| Setting | Measured value |
| --- | ---: |
| PostgreSQL version | `16.3` |
| Block size | `8192` bytes |
| `max_wal_size` | `1GB` |
| `min_wal_size` | `80MB` |
| `wal_compression` | `off` |
| `full_page_writes` | `on` |
| `checkpoint_timeout` | `5min` |
| `archive_mode` | `off` |
| `autovacuum` | `on` |
| Entire database size | `10,707,407,331 bytes` (`10211 MB` as reported by PostgreSQL) |
| `weather_backfill` table footprint | `1,061,871,616 bytes` (`0.989 GiB`) |

`max_wal_size` is a checkpoint target rather than an absolute upper bound. The storage
projection therefore does not assume that WAL can never exceed 1 GB during sustained writes.

## Live Relation Sizes

`total_bytes` is `pg_total_relation_size`, so it includes the table heap, indexes, and auxiliary
storage owned by the relation.

| Table | Exact rows | Heap bytes | Index bytes | Total bytes | Total bytes/live row |
| --- | ---: | ---: | ---: | ---: | ---: |
| `source_issue` | 9,562 | 16,302,080 | 2,588,672 | 18,931,712 | 1,979.89 |
| `station_feature` | 481,708 | 209,616,896 | 107,577,344 | 317,284,352 | 658.67 |
| `area_feature` | 858,142 | 507,838,464 | 201,375,744 | 709,386,240 | 826.65 |
| `ingest_event` | 16,942 | 14,598,144 | 401,408 | 15,040,512 | 887.76 |
| `ingest_run` | 112 | 270,336 | 40,960 | 1,196,032 | 10,678.86 |
| `normalized_artifact` | 0 | 0 | 8,192 | 16,384 | Not applicable |
| `artifact_column` | 0 | 0 | 8,192 | 16,384 | Not applicable |

The `ingest_run` total includes `851,968` bytes of TOAST storage because its JSON summaries are
larger than the inline tuple threshold. The feature and source-issue tables had no material
TOAST heap at measurement time.

## Live Index Inventory

| Table | Index | Bytes | Indexed columns |
| --- | --- | ---: | --- |
| `source_issue` | `source_issue_pkey` | 876,544 | `issue_key` unique |
| `source_issue` | `ix_weather_source_issue_available` | 630,784 | `available_at_utc, source, product` |
| `source_issue` | `ix_weather_source_issue_time` | 1,081,344 | `source, product, availability_proxy_utc, valid_at_utc` |
| `station_feature` | `station_feature_pkey` | 68,419,584 | `issue_key, station_id, feature_name` unique |
| `station_feature` | `ix_weather_station_feature_lookup` | 39,157,760 | `station_id, feature_name, available_at_utc` |
| `area_feature` | `area_feature_pkey` | 175,546,368 | `issue_key, area_key, variable_name, statistic` unique |
| `area_feature` | `ix_weather_area_feature_lookup` | 25,829,376 | `area_key, variable_name, statistic, available_at_utc` |
| `ingest_event` | `ingest_event_pkey` | 401,408 | `event_id` unique |
| `ingest_run` | `ingest_run_pkey` | 40,960 | `run_id` unique |

At the measured scale, indexes account for approximately `33.9%` of
`station_feature`, `28.4%` of `area_feature`, and `13.7%` of `source_issue` total storage.

## Experiment 0012 Population

The accepted validation window is `2026-06-10` through `2026-07-08`, inclusive, or 29 days.

| Source | Source issues | Successful issues | Station rows | Area rows |
| --- | ---: | ---: | ---: | ---: |
| GFS | 1,972 | 1,972 | 34,916 | 402,636 |
| GEFS control | 1,972 | 1,972 | 33,175 | 380,003 |
| Himawari B13/S0510 | 4,176 | 4,111 | 390,545 | 45,221 |
| **Total** | **8,120** | **8,055** | **458,636** | **827,860** |

All 65 unsuccessful issues were Himawari source-side 404 responses. The database still retains
one `source_issue` row for each missing scan, preserving the expected inventory and failure
reason without creating feature rows for nonexistent source objects.

## Daily Distribution Across the 29-Day Window

| Metric | Minimum | Median | Maximum | Mean |
| --- | ---: | ---: | ---: | ---: |
| Expected source issues/day | 280 | 280 | 280 | 280.00 |
| Successful source issues/day | 276 | 278 | 278 | 277.76 |
| Station feature rows/day | 15,648 | 15,838 | 15,838 | 15,815.03 |
| Area feature rows/day | 28,526 | 28,550 | 28,550 | 28,546.90 |

## Logical Row Widths

`pg_column_size(row)` measures the logical tuple payload. The capacity model uses physical
bytes/live row from the relation table above because the requested answer must include indexes
and PostgreSQL page overhead. Logical widths are retained here to show that the physical ratios
are consistent with the actual feature shapes.

| Table/source | Rows sampled | Average logical bytes | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| `source_issue` / GFS | 1,972 | 1,658.54 | 992 | 1,984 |
| `source_issue` / GEFS control | 1,972 | 1,697.91 | 1,048 | 1,760 |
| `source_issue` / Himawari successful | 4,111 | 841.73 | 696 | 864 |
| `source_issue` / Himawari 404 | 65 | 876.68 | 872 | 880 |
| `station_feature` / GFS | 34,916 | 413.95 | 376 | 424 |
| `station_feature` / GEFS control | 33,175 | 440.84 | 392 | 456 |
| `station_feature` / Himawari | 390,545 | 411.22 | 368 | 432 |
| `area_feature` / GFS | 402,636 | 576.24 | 504 | 616 |
| `area_feature` / GEFS control | 380,003 | 595.14 | 520 | 632 |
| `area_feature` / Himawari | 45,221 | 445.13 | 408 | 456 |

## Table Health at Measurement Time

Autovacuum had run on all populated high-volume tables. Estimated dead tuples were zero for
`station_feature`, `area_feature`, and `ingest_event`; `source_issue` had 120 estimated dead
tuples and `ingest_run` had 15. This means the measured bytes/live-row ratios are not dominated
by a large current dead-tuple backlog, although future reruns and updates can still create
bloat.

## Host Disk Snapshot

The Windows `C:` volume had `226.97 GiB` free at measurement time. The storage estimate projects
approximately `112 GiB` of additional retained `weather_backfill` growth beyond the schema's
current `0.989 GiB`. If no other large consumers grow, the host would retain roughly `115 GiB`
free after the historical load. That is enough for the expected tables and ordinary WAL churn,
but it is not a generous margin for same-disk copies, concurrent index rebuilds, or large local
backups.

## Reproduction Queries

The core size measurement can be reproduced with read-only SQL:

```sql
SELECT
    c.relname,
    pg_relation_size(c.oid) AS heap_bytes,
    pg_indexes_size(c.oid) AS index_bytes,
    pg_total_relation_size(c.oid) AS total_bytes
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'weather_backfill'
  AND c.relkind = 'r'
ORDER BY c.relname;
```

Exact row counts and logical widths use:

```sql
SELECT count(*) AS rows, avg(pg_column_size(t.*)) AS avg_logical_row_bytes
FROM weather_backfill.station_feature AS t;
```

Run the same statement for `source_issue`, `area_feature`, `ingest_event`, and `ingest_run`.
