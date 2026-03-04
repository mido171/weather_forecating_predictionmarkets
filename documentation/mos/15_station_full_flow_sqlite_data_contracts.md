# 15 - Station Full Flow SQLite Data Contracts

## Purpose
Define canonical station data contracts for:
1. NWS truth (`D:\Ahmed\data\sqlite\NWS\<STATION>\...`)
2. MOS guidance (`D:\Ahmed\data\sqlite\MOS\<STATION>\...`)
3. Compatibility exports consumed by training/live/backtest scripts.

## Authoritative Source Policy (Explicit)
1. Main MOS/NWS data reads must come from SQLite under `D:\Ahmed\data\sqlite`.
2. SQLite tables are canonical source-of-truth for station truth/guidance history.
3. Compatibility CSV exports are non-canonical derivatives generated from SQLite.

## NWS SQLite Contract
Root:
`D:\Ahmed\data\sqlite\NWS\<STATION>\`

DB:
`<STATION>_nws_truth_<startYear>_<endYear>.sqlite`

### Tables
1. `nws_raw_snapshots`
2. `nws_truth_canonical`
3. `nws_truth_enriched`
4. `nws_qa_reports`
5. `nws_run_meta`

### Key Constraint
`nws_truth_canonical` enforces uniqueness by `(station_id, target_date_local)`.

### Canonical Columns (`nws_truth_canonical`)
1. `station_id` (`TEXT`)
2. `station_usw` (`TEXT`)
3. `target_date_local` (`TEXT`, ISO date)
4. `tmax_f` (`INTEGER`)
5. `truth_source` (`TEXT`)
6. `source_record_id` (`TEXT`)
7. `retrieved_at_utc` (`TEXT`, ISO timestamp)

## MOS SQLite Contract
Root:
`D:\Ahmed\data\sqlite\MOS\<STATION>\`

DB:
`<STATION>_mos_<startYear>_<endYear>.sqlite`

### Tables
1. `mos_raw_payloads`
2. `mos_hourly_values`
3. `mos_download_manifest`
4. `mos_run_meta`

### Raw Payload Metadata (`mos_raw_payloads`)
Primary key: `(station_id, model, year)`.

### Hourly Guidance (`mos_hourly_values`)
Core columns:
1. `station_id`
2. `model`
3. `year`
4. `runtime_utc`
5. `forecast_time_utc`
6. `retrieved_at_utc`
7. `response_sha256`
8. MOS values and raw fields (`tmp`, `dpt`, `sky`, `wdr`, `wsp`, etc.)

### Coverage Requirement
Requested windows must include non-zero rows for `GFS` and `NAM`.

## Compatibility Export Contracts
### Truth CSV
Path:
`D:\Ahmed\data\kalshi\training_data\02_truth\<STATION>_settled_tmax_2002_2026.csv`

Columns:
1. `station_id`
2. `date`
3. `settled_tmax`

Constraints:
1. One row per `station_id + date`.
2. `settled_tmax` is integer Fahrenheit.
3. Export is derived from `nws_truth_canonical` (SQLite), not independently sourced.

### MOS Archive CSV.GZ
Path:
`D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\<STATION>_mos_archive_2002_2026.csv.gz`

Columns:
1. Same canonical hourly contract as `mos_hourly_values`.

Constraints:
1. Non-empty.
2. Contains both `GFS` and `NAM`.
3. Export is derived from `mos_hourly_values` (SQLite), not independently sourced.

## Transformation Chain
1. Raw NWS snapshots -> `nws_truth_canonical`/`nws_truth_enriched`.
2. Raw MOS payloads + yearly archives -> `mos_hourly_values`.
3. SQLite canonical tables -> compatibility exports used by train/live/backtest.

## Audit Fields
Common audit expectations:
1. Response hashes where available (`response_sha256` / raw hash files).
2. Retrieval timestamps (`retrieved_at_utc`).
3. Run metadata snapshots in `*_run_meta`.

## Reproducibility
Per run, manifest includes:
1. Inputs and resolved station metadata.
2. Command history.
3. Canonical DB paths.
4. Export paths + hashes.
5. Bundle/backtest outputs.
