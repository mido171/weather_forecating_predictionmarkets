# HKG Tmax Data Acquisition Status

## Scope

Weather-data acquisition only for official HKO Headquarters (`HKO`) daily maximum
air temperature forecasting. Polymarket work is excluded.

## Current State

- current commit before this reset milestone: `efafc9dc90881f76c57651c1d307949ff152270d`
- worktree state: acquisition reset in progress
- data root: `C:\hkg_tmax_data`
- repository data path length: 146 characters
- acquisition data-root path length: 16 characters
- long paths enabled: false on this machine
- filesystem: NTFS
- free disk at data-root initialization: about 315 GB

## Completed This Work Unit

- stopped the Daily Extract polling loop; no active polling process was found;
- closed EXP-0032 as `SUPERSEDED` and preserved its unchanged-payload evidence;
- added a content-addressed acquisition storage layer;
- added append-only retrieval, file-manifest, and lineage ledgers;
- added `config/data_source_catalog.yaml`;
- added `config/collector_schedules.yaml`;
- added Windows collector management scripts;
- added the acquisition scope file and polling-loop postmortem;
- initialized `C:\hkg_tmax_data`;
- generated `metadata/source_catalog.parquet`;
- ran an initial non-market HKO acquisition batch.
- built source-native bronze Parquet datasets for five acquired HKO sources.

## Sources Completed

Initial content-addressed raw retrieval succeeded for:

| Source | Bytes | Content hash prefix |
|---|---:|---|
| `hko_clmmaxt_hko` | 838832 | `5a0a646b4d125e40` |
| `hko_latest_1min_temperature` | 1256 | `be489e7eed8928cd` |
| `hko_since_midnight_maxmin` | 1526 | `3397e65ae661fdea` |
| `hko_local_weather_forecast` | 718 | `59c71377cf53c464` |
| `hko_nine_day_forecast` | 4287 | `4ec7863a04016c74` |
| `hko_open_data_catalog` | 63849 | `63108144f079be3e` |
| `hko_station_metadata` | 66374 | `9d54c0d4ff6df23e` |

## Sources Actively Downloading

None.

## Live Collectors Installed/Running

Not installed yet. Scripts exist; Task Scheduler registration requires explicit
operator execution of `scripts/install_windows_collectors.ps1`.

## Coverage and Bytes

- retrieval ledger: `C:\hkg_tmax_data\manifests\retrieval_ledger.parquet`
- file manifest: `C:\hkg_tmax_data\manifests\file_manifest.parquet`
- dataset lineage: `C:\hkg_tmax_data\manifests\dataset_lineage.parquet`
- successful retrieval attempts in first batch: 7
- failed retrieval attempts in first batch: 0
- unique raw content hashes in first batch: 7
- total bytes in first batch: 976842

## Bronze Rebuilds

| Source | Bronze rows |
|---|---:|
| `hko_clmmaxt_hko` | 49460 |
| `hko_latest_1min_temperature` | 39 |
| `hko_since_midnight_maxmin` | 39 |
| `hko_local_weather_forecast` | 1 |
| `hko_nine_day_forecast` | 9 |

## Completion Percentages

Based on source contracts, not file counts:

- catalog entries: 19 total
- priority split: 12 P0, 6 P1, 1 P2
- status split: 7 `IMPLEMENTED_INITIAL_FETCH`, 1 `SUPERSEDED_POLLING_FAMILY_CLOSED`,
  5 `DISCOVERY_REQUIRED`, 4 `NOT_STARTED`, 1 `CREDENTIAL_BLOCKED`,
  1 `DEFERRED_WITH_REASON`
- P0: 7/12 entries have initial retrieval or closed operational policy; parsers,
  complete backfills, and durable collector installation remain incomplete
- P1: 0/6 acquired; all require discovery, credentials, source selection, or
  implementation
- P2: deferred by policy until P0/P1 acquisition is mature

## Blockers Requiring User Action

- CDS/ERA5 credentials are required before reanalysis backfill.
- Some ECMWF products may require provider-specific setup.
- Large NWP backfills require a finalized subset and storage-volume budget.
- Historical minute/sub-hourly HKO data availability or paid custom-data process
  must be confirmed.

## QC Failures and Repairs

- No acquisition retrieval failures in the first batch.
- Bronze parsing succeeded for the five sources listed above.
- Deeper station-code normalization, unit validation, impossible-value checks,
  and silver/gold outputs remain pending.
- EXP-0032 was not a QC failure; it was superseded as process waste.

## Next Ten Tasks

1. Parse `CLMMAXT` into a bronze daily HKO table with quality flags.
2. Parse HKO latest one-minute temperature feed into every station row.
3. Parse HKO since-midnight max/min feed into every station row.
4. Parse HKO local forecast and nine-day forecast issue times into forecast vintages.
5. Resolve official DATA.GOV.HK URLs for humidity, wind, rainfall, radiation, and pressure.
6. Implement the template-bound once-daily Daily Extract collector without rapid polling.
7. Add focused parser tests for the three HKO live feeds acquired in this batch.
8. Generate station-code coverage tables from parsed live feeds and station metadata.
9. Implement NOAA GFS subset discovery and byte-budget report before any large NWP backfill.
10. Install or manually run Windows collectors after parser-level smoke tests pass.
