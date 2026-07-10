# HKG Tmax Info.gov Hourly Readings Data Context

Created: 2026-07-05 CEST

Database table documented here:

`public.hko_info_gov_hourly_readings_1998_2026`

This document explains the new Info.gov hourly readings archive that was backfilled into Postgres. It is written as a live-trading and modeling context file, so a future strategy implementation can understand exactly what the table is, what it is not, where the data came from, how to query it, and how to use it safely for HKG station Tmax forecasting.

## Executive Summary

`public.hko_info_gov_hourly_readings_1998_2026` is a one-row-per-dispatch archive of HKSAR Government Info.gov weather press releases titled `PRESS WEATHER NO. ### - HOURLY READINGS`.

Each row represents one public HKO hourly readings bulletin. The row contains:

- The bulletin URL and press weather number.
- The dispatch time in HKT and UTC.
- The observation time in HKT and UTC.
- The Hong Kong Observatory target-station air temperature and relative humidity reported in the bulletin.
- Warning, rainfall, lightning, and tropical-cyclone text when present.
- Every listed neighbor-station temperature reading in `station_readings_jsonb`.
- Aggregate station statistics such as station min, max, mean, spread, station count, and missing count.
- The full extracted text, raw HTML path, raw SHA-256, parse status, and ingestion timestamp.

This table is not the same thing as `public.hko_historical_forecasts_2000_2026`.

- `public.hko_historical_forecasts_2000_2026` stores Info.gov `LOCAL WEATHER FORECAST` dispatches. Those rows are forecast inputs.
- `public.hko_info_gov_hourly_readings_1998_2026` stores Info.gov `HOURLY READINGS` dispatches. These rows are observed intraday conditions and neighbor-station observations.

For the HKG Tmax Polymarket forecasting task, this new table is valuable because it gives the model the real observed state of the HKO station and nearby Hong Kong stations during the hours before and during a target day. It should be treated as as-of-sensitive observational data. A strategy may use only rows with `dispatch_at_utc` or `observation_at_utc` at or before the decision cutoff.

## Reader Orientation

Read this document if you are implementing, reviewing, or prompting a model to design features from `public.hko_info_gov_hourly_readings_1998_2026`.

Use the sections in this order:

- Start with `What This Data Is` and `What This Data Is Not` to avoid confusing hourly observations with forecast dispatches.
- Read `Current DB Coverage`, `Yearly Coverage Snapshot`, and `Station Coverage` before trusting the table for model features.
- Read `Table Schema And Column Meanings` and `station_readings_jsonb Object Contract` before writing SQL or feature extraction code.
- Use the example SQL sections as copyable query patterns.
- Use `Recommended Feature Engineering Contract` to keep the table leakage-safe.

## Scope Boundaries

Included in this document:

- The observational hourly readings table `public.hko_info_gov_hourly_readings_1998_2026`.
- The Info.gov source pattern used to fetch the raw pages.
- The DB schema, row coverage, missingness, station coverage, and example queries.
- The relationship between this observational table and the existing forecast archive.
- Modeling guidance for as-of-safe feature engineering.

Excluded from this document:

- The implementation internals of the historical local forecast archive `public.hko_historical_forecasts_2000_2026`; that is documented separately in `HKG_TMAX_INFO_GOV_LIVE_FORECAST_SOURCE_CONTEXT_20260704.md`.
- A full model training recipe. This document describes the data contract and safe usage, not the final estimator.
- Polymarket order execution or market backtesting.
- A normalized relational station-reading table. The user explicitly requested one dedicated Postgres table for this source, so station values are stored in JSONB.

## Source-of-Truth Inputs

This document is evidence-backed from these concrete inputs:

- Direct read-only Postgres queries against `public.hko_info_gov_hourly_readings_1998_2026` in database `hkg_tmax_research`.
- The generated report `data/datasets/13_hko_info_gov_hourly_readings/reports/postgres_load_summary.json`.
- The generated report `data/datasets/13_hko_info_gov_hourly_readings/reports/structure_pattern_report.json`.
- The generated failed-URL reports `detail_fetch_failures.csv` and `detail_fetch_failures.json`.
- The dataset README `data/datasets/13_hko_info_gov_hourly_readings/README.md`.
- The DB migration `migrations/postgres/20260704_0008_hko_info_gov_hourly_readings.sql`.
- The backfill script `scripts/backfill_hko_info_gov_hourly_readings.py`.

No external web re-query was required to write this context file. The DB table, raw HTML archive, and local reports are the source of truth for the documented coverage and schema.

## Requirements-to-Implementation Traceability

Requirement: document what the new table is.

- Delivered in `What This Data Is`, `What This Data Is Not`, and `Relationship To The Forecast Archive`.
- Verified by direct schema and sample-row queries against the table.

Requirement: document where the data came from.

- Delivered in `Where The Data Came From` and `High-Level Acquisition And Normalization Flow`.
- Verified from the backfill script, raw HTML folder layout, and source URL fields.

Requirement: document coverage and null/unusable rates.

- Delivered in `Current DB Coverage`, `Yearly Coverage Snapshot`, and `Station Coverage`.
- Verified from live Postgres aggregate queries and generated report files.

Requirement: document all important fields.

- Delivered in `Table Schema And Column Meanings` and `station_readings_jsonb Object Contract`.
- Verified from `information_schema.columns`, migration DDL, and sample rows.

Requirement: include example query and returned output.

- Delivered in `Example Query: Pull Two 2023 Rows With Station JSON`, `Example Query: Latest Reading Available Before A Cutoff`, and `Example Query: Extract One Station From JSONB`.
- The first example includes actual returned JSON from the table.

Requirement: explain how this helps HKG Tmax forecasting.

- Delivered in `How This Data Should Be Used For HKG Tmax Modeling` and `Recommended Feature Engineering Contract`.
- The guidance explicitly centers on leakage-safe cutoff handling.

## Change Inventory

This documentation task added one context artifact:

- `documentation/strategy_implementation_documentation/context/live_trading/HKG_TMAX_INFO_GOV_HOURLY_READINGS_DATA_CONTEXT_20260705.md`
  - Type: documentation.
  - Purpose: explain the new Info.gov hourly readings Postgres table and its modeling contract.
  - User-visible effect: future GPT-Pro/Codex handoffs can distinguish the forecast archive from this observational hourly readings archive.
  - Verification: file existence check, unfinished-marker scan, key-number scan, and documentation quality-gate structural pass.

Implementation files for the backfill itself already exist and are referenced as source-of-truth inputs. This document does not modify those implementation files.

## Architecture and Control Flow

The data flow for this source is:

```text
Info.gov daily weather index
  -> links titled PRESS WEATHER NO. ### - HOURLY READINGS
  -> raw HTML detail pages saved under data/datasets/13_hko_info_gov_hourly_readings/raw_html
  -> parser extracts dispatch time, HKO target reading, station readings, warning/rain/lightning/cyclone text
  -> normalized CSV with one row per dispatch
  -> Postgres table public.hko_info_gov_hourly_readings_1998_2026
  -> model feature queries filtered by dispatch_at_utc <= cutoff
```

The table is intentionally denormalized:

- One Postgres row equals one Info.gov hourly readings dispatch.
- HKO target station fields are first-class columns.
- Neighbor station readings remain inside `station_readings_jsonb`.
- Raw text and raw HTML path are preserved for audit and parser debugging.

## File-by-File Deep Dive

This document itself is the only file added for this request.

`HKG_TMAX_INFO_GOV_HOURLY_READINGS_DATA_CONTEXT_20260705.md` is responsible for documenting:

- The source product and URL pattern.
- The exact Postgres table name and role.
- The distinction between forecast dispatches and hourly observation dispatches.
- The table schema and JSONB station object contract.
- Coverage, parse quality, missingness, station coverage, and caveats.
- Copyable SQL examples for inspection and feature extraction.
- As-of-safe modeling rules for HKG Tmax forecasting.

The implementation files referenced by the document are:

- `scripts/backfill_hko_info_gov_hourly_readings.py`: source-specific fetch, parse, normalize, report, and load script.
- `migrations/postgres/20260704_0008_hko_info_gov_hourly_readings.sql`: table, constraints, indexes, and table comment.
- `data/datasets/13_hko_info_gov_hourly_readings/README.md`: dataset folder contract and canonical table name.

This documentation file does not change runtime behavior.

## Public Interfaces and Contracts

Public DB interface:

- Table: `public.hko_info_gov_hourly_readings_1998_2026`.
- Primary key: `bulletin_id`.
- Unique URL key: `source_url`.
- JSONB station payload: `station_readings_jsonb`.
- As-of timestamp for feature availability: `dispatch_at_utc`.
- Observation timestamp for meteorological timing: `observation_at_utc`.

Public local-file interface:

- Raw HTML root: `data/datasets/13_hko_info_gov_hourly_readings/raw_html`.
- Normalized CSV: `data/datasets/13_hko_info_gov_hourly_readings/normalized/hko_info_gov_hourly_readings.csv`.
- Reports root: `data/datasets/13_hko_info_gov_hourly_readings/reports`.

Public operational command:

```powershell
.\.venv\Scripts\python.exe scripts\backfill_hko_info_gov_hourly_readings.py --load-db
```

## What This Data Is

The source product is an HKO hourly weather observation bulletin published through the HKSAR Government Info.gov press release feed.

The product title pattern is:

`PRESS WEATHER NO. ### - HOURLY READINGS`

The page text normally contains:

- A headline saying `HOURLY READINGS`.
- A sentence like `AT 7 P.M. AT THE HONG KONG OBSERVATORY THE AIR TEMPERATURE WAS 28 DEGREES CELSIUS AND THE RELATIVE HUMIDITY 86 PER CENT.`
- A station block beginning with `THE AIR TEMPERATURES AT OTHER PLACES WERE:`.
- One line per station, for example `KING'S PARK 27 DEGREES;`.
- Missing station values such as `SHEK KONG // DEGREES;`.
- Optional warning text, rainfall text, lightning text, and tropical-cyclone text.
- A dispatch line such as `DISPATCHED BY HONG KONG OBSERVATORY AT 19:02 HKT ON 04.07.2026`.

The target station for the Polymarket/HKO daily Tmax task is the Hong Kong Observatory station itself. In this table that station is represented by:

- `hko_air_temp_c`
- `hko_relative_humidity_pct`
- `observation_at_hkt`
- `observation_at_utc`

The other station temperatures are neighbor or comparison observations. They are not the settlement station, but they are potentially useful model features because they describe spatial heat, cooling, thunderstorm/outflow, coastal/inland spread, and localized hot/cool pockets across Hong Kong.

## What This Data Is Not

This table is not a forecast archive. It does not contain official forecast min/max values for tomorrow. For that, use:

`public.hko_historical_forecasts_2000_2026`

This table is not the settlement label. The Polymarket market resolves from the Hong Kong Observatory Daily Extract field `Absolute Daily Max (deg. C)` for a target date. Hourly readings can help forecast that daily max, but the market settlement is a finalized daily extract value, not one of these hourly bulletin rows.

This table is not a station-normalized relational schema. The explicit requirement for this backfill was one dedicated table, so every neighbor-station line is stored inside `station_readings_jsonb`.

This table is not a weather forecast product such as HKO OpenData `fnd`, HKO OpenData `flw`, GribStream, NWP, airport forecasts, or commercial weather widgets.

## Where The Data Came From

Primary source:

`https://www.info.gov.hk/gia/wr/YYYYMM/DD.htm`

This is the Info.gov daily weather press-release index. The backfill scanned the index pages and selected links whose title matched:

`PRESS WEATHER NO. ### - HOURLY READINGS`

Individual pages follow old and modern URL styles, including:

- Old style example: `https://www.info.gov.hk/gia/wr/199805/04/0504088.htm`
- Modern style example: `https://www.info.gov.hk/gia/wr/202607/04/P2026070400751.htm`

Raw HTML was downloaded and stored locally under:

`data/datasets/13_hko_info_gov_hourly_readings/raw_html`

The resumable local ledger is:

`data/datasets/13_hko_info_gov_hourly_readings/metadata/hourly_readings_archive.sqlite3`

The normalized CSV used for DB loading is:

`data/datasets/13_hko_info_gov_hourly_readings/normalized/hko_info_gov_hourly_readings.csv`

Generated coverage and validation reports are under:

`data/datasets/13_hko_info_gov_hourly_readings/reports`

Important report files:

- `postgres_load_summary.json`
- `structure_pattern_report.json`
- `daily_dispatch_counts.csv`
- `issue_time_cadence.csv`
- `station_coverage_by_year.csv`
- `station_missing_counts.csv`
- `detail_fetch_failures.csv`
- `detail_fetch_failures.json`
- `parse_failures.json`

The canonical backfill script is:

`scripts/backfill_hko_info_gov_hourly_readings.py`

The canonical table migration is:

`migrations/postgres/20260704_0008_hko_info_gov_hourly_readings.sql`

## High-Level Acquisition And Normalization Flow

The backfill process did this:

1. Scanned Info.gov daily weather indexes from the earliest available hourly readings archive date through 2026-07-04.
2. Found every link titled `PRESS WEATHER NO. ### - HOURLY READINGS`.
3. Downloaded each matching detail page.
4. Saved the raw HTML page and metadata sidecar before normalization.
5. Parsed the dispatch time from the `DISPATCHED BY HONG KONG OBSERVATORY...` line.
6. Parsed the HKO target-station temperature and relative humidity.
7. Parsed every neighbor station line into JSON objects.
8. Preserved missing station readings such as `// DEGREES`.
9. Parsed warning, rainfall, lightning, and tropical cyclone blocks where present.
10. Wrote one normalized CSV row per detail page.
11. Loaded the normalized CSV into `public.hko_info_gov_hourly_readings_1998_2026`.
12. Created indexes for dispatch time, observation time, index date, raw hash, and JSONB station lookup.

The implementation intentionally kept this as one DB table. There is no separate station-reading child table.

## Current DB Coverage

These numbers were checked directly against `public.hko_info_gov_hourly_readings_1998_2026` on 2026-07-05.

Overall loaded rows:

- `268,894` rows.

Discovered source URLs:

- `268,937` Info.gov hourly-reading detail URLs were discovered from index pages.
- `268,894` were fetched, parsed, and loaded.
- `43` discovered URLs could not be fetched and are explicitly reported in `detail_fetch_failures.csv/json`.

Date coverage:

- First `index_date_hkt`: `1998-05-04`.
- Last `index_date_hkt`: `2026-07-04`.
- First `dispatch_at_hkt`: `1998-05-04 12:03:00`.
- Last `dispatch_at_hkt`: `2026-07-04 23:03:00`.
- First `dispatch_at_utc`: `1998-05-04 04:03:00+00`.
- Last `dispatch_at_utc`: `2026-07-04 15:03:00+00`.

Parse quality:

- `268,856` rows have `parse_status = 'parsed'`.
- `38` rows have `parse_status = 'partial'`.
- `0` rows have `parse_status = 'failed'`.
- Partial rows are `0.014132%` of the loaded table.
- Failed rows are `0.000000%` of the loaded table.

HKO target-station coverage:

- `268,861` rows have target-station HKO temperature/RH present.
- `33` rows have null `hko_air_temp_c`.
- `33` rows have null `hko_relative_humidity_pct`.
- `33` rows have `target_station_present = false`.
- Target-station absent rate is `0.012272%`.

Station JSON coverage:

- `13` rows have `station_count = 0`.
- Empty station JSON rate is `0.004835%`.
- `48,832` rows have at least one missing neighbor-station reading.
- Rows with at least one missing station value are `18.160316%`.
- Max station count in one dispatch is `26`.
- Max missing station count in one dispatch is `26`.
- Unique station names after cleanup: `27`.

Weather-context text coverage:

- `102,770` rows contain `warning_text`.
- `63,618` rows contain `rainfall_text`.
- `16,307` rows contain `lightning_text`.
- `33,133` rows contain `tropical_cyclone_text`.
- `18,212` rows contain `tropical_cyclone_name`.

Temperature ranges in loaded summary columns:

- HKO target-station `hko_air_temp_c`: min `3 C`, max `36 C`.
- Neighbor-station aggregate min: `-9 C`.
- Neighbor-station aggregate max: `50 C`.

The neighbor-station aggregate extremes are retained because they are inside the broad DB sanity bounds and came from source text. They should be treated carefully during modeling. Robust feature engineering should use clipping, winsorization, station-specific plausibility checks, or anomaly indicators rather than blindly trusting every neighbor-station extreme.

## Yearly Coverage Snapshot

This is the direct yearly row/day coverage from the loaded DB.

```text
1998:  5,922 rows, 242 days, first 1998-05-04, last 1998-12-31, parsed 5,909, partial 13, HKO-temp-null 13, station-empty 0, station-missing-any 1,346
1999:  9,144 rows, 365 days, first 1999-01-01, last 1999-12-31, parsed 9,139, partial 5, HKO-temp-null 5, station-empty 0, station-missing-any 2,256
2000:  9,309 rows, 366 days, first 2000-01-01, last 2000-12-31, parsed 9,309, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 5,707
2001:  9,340 rows, 365 days, first 2001-01-01, last 2001-12-31, parsed 9,340, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 675
2002:  9,212 rows, 365 days, first 2002-01-01, last 2002-12-31, parsed 9,212, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 546
2003:  9,326 rows, 365 days, first 2003-01-01, last 2003-12-31, parsed 9,323, partial 3, HKO-temp-null 0, station-empty 0, station-missing-any 614
2004:  9,250 rows, 366 days, first 2004-01-01, last 2004-12-31, parsed 9,249, partial 1, HKO-temp-null 0, station-empty 0, station-missing-any 258
2005:  9,407 rows, 365 days, first 2005-01-01, last 2005-12-31, parsed 9,407, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 510
2006:  9,362 rows, 365 days, first 2006-01-01, last 2006-12-31, parsed 9,362, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 188
2007:  9,358 rows, 365 days, first 2007-01-01, last 2007-12-31, parsed 9,357, partial 1, HKO-temp-null 1, station-empty 0, station-missing-any 2,035
2008:  9,465 rows, 366 days, first 2008-01-01, last 2008-12-31, parsed 9,465, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 2,718
2009:  9,381 rows, 365 days, first 2009-01-01, last 2009-12-31, parsed 9,381, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 974
2010:  9,407 rows, 365 days, first 2010-01-01, last 2010-12-31, parsed 9,407, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,039
2011:  9,302 rows, 365 days, first 2011-01-01, last 2011-12-31, parsed 9,302, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 920
2012:  9,559 rows, 366 days, first 2012-01-01, last 2012-12-31, parsed 9,544, partial 15, HKO-temp-null 14, station-empty 13, station-missing-any 1,617
2013:  9,640 rows, 365 days, first 2013-01-01, last 2013-12-31, parsed 9,640, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,565
2014:  9,933 rows, 365 days, first 2014-01-01, last 2014-12-31, parsed 9,933, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,211
2015:  9,633 rows, 364 days, first 2015-01-01, last 2015-12-31, parsed 9,633, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,306
2016:  9,884 rows, 366 days, first 2016-01-01, last 2016-12-31, parsed 9,884, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,670
2017:  9,735 rows, 365 days, first 2017-01-01, last 2017-12-31, parsed 9,735, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,950
2018:  9,624 rows, 365 days, first 2018-01-01, last 2018-12-31, parsed 9,624, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,497
2019:  9,766 rows, 365 days, first 2019-01-01, last 2019-12-31, parsed 9,766, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 2,301
2020:  9,574 rows, 364 days, first 2020-01-01, last 2020-12-31, parsed 9,574, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,702
2021:  9,735 rows, 365 days, first 2021-01-01, last 2021-12-31, parsed 9,735, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,820
2022:  9,560 rows, 365 days, first 2022-01-01, last 2022-12-31, parsed 9,560, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 2,071
2023:  9,874 rows, 365 days, first 2023-01-01, last 2023-12-31, parsed 9,874, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 4,170
2024: 10,072 rows, 366 days, first 2024-01-01, last 2024-12-31, parsed 10,072, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 2,902
2025: 10,037 rows, 365 days, first 2025-01-01, last 2025-12-31, parsed 10,037, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 2,243
2026:  5,083 rows, 185 days, first 2026-01-01, last 2026-07-04, parsed 5,083, partial 0, HKO-temp-null 0, station-empty 0, station-missing-any 1,021
```

`2015` and `2020` show 364 covered index dates in this loaded table. That does not necessarily mean no Info.gov weather content existed on the missing dates; it means this exact hourly-readings extraction found no loaded hourly-reading dispatch rows for one date in each of those years.

## Dispatch Cadence And Abnormal Days

The product is called hourly readings, and the normal operational pattern is close to one dispatch per hour, often around minute `02` after the hour. However, the archive contains more than 24 dispatches on many days because warnings, tropical cyclone updates, duplicate press numbers, correction cycles, or severe-weather operations can add extra dispatches.

Daily dispatch count range in the loaded data:

- Minimum loaded dispatches per day: `7`.
- Maximum loaded dispatches per day: `60`.
- Most common daily dispatch count in the generated report: `60`.

Top high-dispatch days from the DB:

```json
[
  {"index_date_hkt": "2024-06-15", "dispatch_count": 60},
  {"index_date_hkt": "2026-06-08", "dispatch_count": 60},
  {"index_date_hkt": "2025-08-04", "dispatch_count": 57},
  {"index_date_hkt": "2025-08-05", "dispatch_count": 57},
  {"index_date_hkt": "2026-06-18", "dispatch_count": 57},
  {"index_date_hkt": "2025-08-18", "dispatch_count": 56},
  {"index_date_hkt": "2024-05-04", "dispatch_count": 54},
  {"index_date_hkt": "2025-08-14", "dispatch_count": 54}
]
```

For modeling, do not assume exactly 24 rows per HKT day. Always select by `dispatch_at_utc`, `observation_at_utc`, and cutoff logic.

## Station Coverage

The table contains 27 canonical station names in `station_readings_jsonb`. Coverage differs because station lists changed over time and because individual stations can be unavailable.

Station appearances and missing counts from the loaded DB:

```text
CHEK LAP KOK: appearances 267,695, missing 31, missing pct 0.0116%, min 3 C, max 37 C
CHEUNG CHAU: appearances 268,881, missing 2,271, missing pct 0.8446%, min 2 C, max 36 C
HAPPY VALLEY: appearances 170,534, missing 1,178, missing pct 0.6908%, min 4 C, max 38 C
HONG KONG PARK: appearances 182,185, missing 3,951, missing pct 2.1687%, min 3 C, max 36 C
KAI TAK RUNWAY PARK: appearances 112,936, missing 997, missing pct 0.8828%, min 4 C, max 37 C
KING'S PARK: appearances 268,881, missing 278, missing pct 0.1034%, min 3 C, max 36 C
KOWLOON CITY: appearances 176,785, missing 1,325, missing pct 0.7495%, min 2 C, max 38 C
KWUN TONG: appearances 162,175, missing 1,036, missing pct 0.6388%, min 2 C, max 37 C
LAU FAU SHAN: appearances 268,880, missing 2,999, missing pct 1.1154%, min 2 C, max 50 C
SAI KUNG: appearances 268,880, missing 1,512, missing pct 0.5623%, min 3 C, max 37 C
SHA TIN: appearances 268,878, missing 2,204, missing pct 0.8197%, min 3 C, max 38 C
SHAM SHUI PO: appearances 158,756, missing 1,192, missing pct 0.7508%, min 3 C, max 37 C
SHAU KEI WAN: appearances 181,860, missing 2,014, missing pct 1.1074%, min 3 C, max 37 C
SHEK KONG: appearances 228,939, missing 11,417, missing pct 4.9869%, min 2 C, max 38 C
STANLEY: appearances 165,289, missing 2,030, missing pct 1.2282%, min 3 C, max 37 C
TA KWU LING: appearances 268,857, missing 2,136, missing pct 0.7945%, min -1 C, max 38 C
TAI MEI TUK: appearances 93,099, missing 1,145, missing pct 1.2299%, min 5 C, max 37 C
TAI PO: appearances 267,580, missing 3,749, missing pct 1.4011%, min 2 C, max 38 C
TSEUNG KWAN O: appearances 268,880, missing 7,376, missing pct 2.7432%, min 2 C, max 38 C
TSING YI: appearances 228,939, missing 981, missing pct 0.4285%, min 3 C, max 37 C
TSUEN WAN: appearances 43,613, missing 149, missing pct 0.3416%, min 5 C, max 35 C
TSUEN WAN HO KOON: appearances 151,531, missing 1,330, missing pct 0.8777%, min 2 C, max 36 C
TSUEN WAN SHING MUN VALLEY: appearances 151,537, missing 927, missing pct 0.6117%, min 2 C, max 37 C
TUEN MUN: appearances 268,880, missing 1,426, missing pct 0.5303%, min 2 C, max 37 C
WONG CHUK HANG: appearances 268,881, missing 2,602, missing pct 0.9677%, min -9 C, max 36 C
WONG TAI SIN: appearances 167,679, missing 1,082, missing pct 0.6453%, min 3 C, max 37 C
YUEN LONG PARK: appearances 110,667, missing 565, missing pct 0.5105%, min 2 C, max 38 C
```

Station coverage should not be interpreted as a fixed panel across the whole archive. Some stations appear only in later eras. Modeling code should create station-specific features with missing indicators and should not require every station to be present for every dispatch.

## Table Schema And Column Meanings

Identity and source columns:

- `bulletin_id`: deterministic primary key for the Info.gov hourly readings page.
- `source`: always `info_gov`.
- `source_url`: unique Info.gov detail page URL.
- `index_date_hkt`: HKT date of the daily Info.gov weather index that linked the page.
- `title`: bulletin title, for example `PRESS WEATHER NO. 004 - HOURLY READINGS`.
- `press_weather_no`: numeric press weather number from the title.

Dispatch and observation time columns:

- `dispatch_at_hkt`: HKO dispatch time in Hong Kong local time, parsed from the dispatch line.
- `dispatch_at_utc`: same dispatch instant converted to UTC.
- `observation_at_hkt`: observation time stated in the bulletin, such as `AT 7 P.M.` or `AT MIDNIGHT`.
- `observation_at_utc`: same observation instant converted to UTC.
- `available_at_utc`: currently set equal to dispatch UTC, used as the earliest availability timestamp for as-of modeling.
- `retrieved_at_utc`: when the backfill fetched the raw HTML from Info.gov.

Target HKO station columns:

- `hko_air_temp_c`: HKO station air temperature in Celsius, usually integer degrees in the press-release text.
- `hko_relative_humidity_pct`: HKO station relative humidity percentage.
- `target_station_present`: true when both HKO temperature and relative humidity were parsed.

Weather-context text columns:

- `rainfall_text`: extracted rainfall paragraph/block when present.
- `warning_text`: extracted warning/reminder block when present.
- `lightning_text`: extracted lightning-detection block when present.
- `tropical_cyclone_text`: extracted tropical-cyclone block when present.
- `tropical_cyclone_name`: parsed storm/cyclone name when present.
- `tropical_cyclone_lat`: parsed cyclone latitude when present.
- `tropical_cyclone_lon`: parsed cyclone longitude when present.

Station JSON and station aggregate columns:

- `station_readings_jsonb`: array of station reading objects, one per station line.
- `station_count`: number of station reading objects.
- `station_missing_count`: number of station objects where `temperature_missing = true`.
- `station_temp_min_c`: minimum non-missing neighbor-station temperature in the dispatch.
- `station_temp_max_c`: maximum non-missing neighbor-station temperature in the dispatch.
- `station_temp_mean_c`: mean of non-missing neighbor-station temperatures in the dispatch.
- `station_temp_spread_c`: `station_temp_max_c - station_temp_min_c`.

Raw/audit columns:

- `full_text`: normalized text extracted from the raw HTML page.
- `raw_html_path`: local path to the archived raw HTML page.
- `raw_sha256`: SHA-256 hash of the raw HTML bytes.
- `parse_status`: `parsed`, `partial`, or `failed`.
- `parse_notes`: parser notes for partial/failed rows.
- `ingested_at_utc`: UTC timestamp when the normalized row was produced.

## `station_readings_jsonb` Object Contract

Each item in `station_readings_jsonb` has this shape:

```json
{
  "station_display_name": "SHEK KONG",
  "station_canonical_name": "SHEK KONG",
  "temperature_c": null,
  "temperature_missing": true,
  "raw_temperature_text": "// DEGREES",
  "raw_station_line": "SHEK KONG // DEGREES",
  "station_order": 13
}
```

Field meaning:

- `station_display_name`: normalized station display name.
- `station_canonical_name`: uppercase canonical station key used for modeling.
- `temperature_c`: numeric Celsius value when present, else null.
- `temperature_missing`: true when the source line had a missing reading such as `// DEGREES`.
- `raw_temperature_text`: original parsed temperature phrase, such as `29 DEGREES` or `// DEGREES`.
- `raw_station_line`: normalized original source line for that station.
- `station_order`: station order within the bulletin station block after parser cleanup.

The parser preserves missing readings. It does not silently drop `// DEGREES` station lines. Missing station values should be modeled with explicit missingness indicators.

## Constraints And Indexes

Important table constraints:

- `bulletin_id` is the primary key.
- `source_url` is unique.
- `source` must equal `info_gov`.
- `parse_status` must be one of `parsed`, `partial`, `failed`.
- `hko_air_temp_c` must be null or between `-20` and `60`.
- `hko_relative_humidity_pct` must be null or between `0` and `100`.
- `station_count >= 0`.
- `station_missing_count >= 0`.
- `station_missing_count <= station_count`.
- `station_temp_min_c` and `station_temp_max_c` must be null or between `-20` and `60`.
- `station_temp_min_c <= station_temp_max_c` when both are present.

Indexes:

- Primary key btree on `bulletin_id`.
- Unique btree on `source_url`.
- Btree on `dispatch_at_utc`.
- Btree on `observation_at_utc`.
- Btree on `index_date_hkt`.
- Btree on `raw_sha256`.
- GIN index on `station_readings_jsonb`.

The most important modeling filters are usually `dispatch_at_utc <= cutoff`, `observation_at_utc <= cutoff`, and `index_date_hkt = target_hkt_date`.

## Example Query: Pull Two 2023 Rows With Station JSON

SQL:

```sql
select jsonb_pretty(jsonb_agg(row_to_json(t)::jsonb))
from (
  select
    source_url,
    index_date_hkt,
    title,
    press_weather_no,
    dispatch_at_hkt,
    observation_at_hkt,
    hko_air_temp_c,
    hko_relative_humidity_pct,
    station_count,
    station_missing_count,
    station_temp_min_c,
    station_temp_max_c,
    station_temp_mean_c,
    station_temp_spread_c,
    parse_status,
    station_readings_jsonb->0 as first_station,
    station_readings_jsonb->12 as thirteenth_station,
    left(full_text, 420) as full_text_prefix
  from public.hko_info_gov_hourly_readings_1998_2026
  where index_date_hkt = date '2023-07-15'
  order by dispatch_at_hkt
  limit 2
) t;
```

Actual returned output:

```json
[
  {
    "title": "PRESS WEATHER NO. 004 - HOURLY READINGS",
    "source_url": "https://www.info.gov.hk/gia/wr/202307/15/P2023071500003.htm",
    "parse_status": "parsed",
    "first_station": {
      "station_order": 1,
      "temperature_c": 30.0,
      "raw_station_line": "KING'S PARK 30 DEGREES",
      "temperature_missing": false,
      "raw_temperature_text": "30 DEGREES",
      "station_display_name": "KING'S PARK",
      "station_canonical_name": "KING'S PARK"
    },
    "station_count": 26,
    "hko_air_temp_c": 31,
    "index_date_hkt": "2023-07-15",
    "dispatch_at_hkt": "2023-07-15T00:02:00",
    "full_text_prefix": "PRESS WEATHER NO. 004 - HOURLY READINGS\nGo to main content\nBrand HK\n|\nFont Size:\n|\nSitemap\nPRESS WEATHER NO. 004 - HOURLY READINGS\nPRESS WEATHER NO. 004 - HOURLY READINGS\n***************************************\nHOURLY READINGS\nAT MIDNIGHT AT THE HONG KONG OBSERVATORY THE AIR\nTEMPERATURE WAS 31 DEGREES CELSIUS AND THE RELATIVE\nHUMIDITY 78 PER CENT.\nPLEASE BE REMINDED THAT:\nTHE VERY HOT WEATHER WARNING IS NOW IN FORCE.",
    "press_weather_no": 4,
    "observation_at_hkt": "2023-07-15T00:00:00",
    "station_temp_max_c": 31,
    "station_temp_min_c": 27,
    "thirteenth_station": {
      "station_order": 13,
      "temperature_c": null,
      "raw_station_line": "SHEK KONG // DEGREES",
      "temperature_missing": true,
      "raw_temperature_text": "// DEGREES",
      "station_display_name": "SHEK KONG",
      "station_canonical_name": "SHEK KONG"
    },
    "station_temp_mean_c": 28.96,
    "station_missing_count": 1,
    "station_temp_spread_c": 4,
    "hko_relative_humidity_pct": 78
  },
  {
    "title": "PRESS WEATHER NO. 010 - HOURLY READINGS",
    "source_url": "https://www.info.gov.hk/gia/wr/202307/15/P2023071500037.htm",
    "parse_status": "parsed",
    "first_station": {
      "station_order": 1,
      "temperature_c": 30.0,
      "raw_station_line": "KING'S PARK 30 DEGREES",
      "temperature_missing": false,
      "raw_temperature_text": "30 DEGREES",
      "station_display_name": "KING'S PARK",
      "station_canonical_name": "KING'S PARK"
    },
    "station_count": 26,
    "hko_air_temp_c": 30,
    "index_date_hkt": "2023-07-15",
    "dispatch_at_hkt": "2023-07-15T01:02:00",
    "full_text_prefix": "PRESS WEATHER NO. 010 - HOURLY READINGS\nGo to main content\nBrand HK\n|\nFont Size:\n|\nSitemap\nPRESS WEATHER NO. 010 - HOURLY READINGS\nPRESS WEATHER NO. 010 - HOURLY READINGS\n***************************************\nHOURLY READINGS\nAT 1 A.M. AT THE HONG KONG OBSERVATORY THE AIR TEMPERATURE\nWAS 30 DEGREES CELSIUS AND THE RELATIVE HUMIDITY 83 PER\nCENT.\nPLEASE BE REMINDED THAT:\nTHE VERY HOT WEATHER WARNING IS NOW IN FORCE. T",
    "press_weather_no": 10,
    "observation_at_hkt": "2023-07-15T01:00:00",
    "station_temp_max_c": 31,
    "station_temp_min_c": 27,
    "thirteenth_station": {
      "station_order": 13,
      "temperature_c": null,
      "raw_station_line": "SHEK KONG // DEGREES",
      "temperature_missing": true,
      "raw_temperature_text": "// DEGREES",
      "station_display_name": "SHEK KONG",
      "station_canonical_name": "SHEK KONG"
    },
    "station_temp_mean_c": 28.84,
    "station_missing_count": 1,
    "station_temp_spread_c": 4,
    "hko_relative_humidity_pct": 83
  }
]
```

## Example Query: Latest Reading Available Before A Cutoff

This is the safest pattern for using this data in a forecasting/trading system.

SQL:

```sql
select
  source_url,
  dispatch_at_hkt,
  observation_at_hkt,
  hko_air_temp_c,
  hko_relative_humidity_pct,
  station_count,
  station_missing_count,
  station_temp_min_c,
  station_temp_max_c,
  station_temp_mean_c,
  station_temp_spread_c,
  warning_text is not null as has_warning_text,
  lightning_text is not null as has_lightning_text,
  tropical_cyclone_text is not null as has_tropical_cyclone_text
from public.hko_info_gov_hourly_readings_1998_2026
where dispatch_at_utc <= timestamptz '2023-07-14 16:00:00+00'
order by dispatch_at_utc desc
limit 1;
```

Why this query matters:

- The cutoff is UTC.
- The selected row is the latest bulletin that could have been known at that cutoff.
- This prevents leakage from using observations that were published after the trading decision.

For a daily target-day model, this query can be extended to compute lag features over all rows where `dispatch_at_utc <= cutoff`, grouped by target date and cutoff policy.

## Example Query: Extract One Station From JSONB

SQL:

```sql
select
  h.dispatch_at_hkt,
  h.hko_air_temp_c,
  s->>'station_canonical_name' as station,
  (s->>'temperature_c')::double precision as station_temperature_c,
  (s->>'temperature_missing')::boolean as station_temperature_missing
from public.hko_info_gov_hourly_readings_1998_2026 h
cross join lateral jsonb_array_elements(h.station_readings_jsonb) s
where h.index_date_hkt = date '2023-07-15'
  and s->>'station_canonical_name' = 'SHEK KONG'
order by h.dispatch_at_hkt
limit 5;
```

Use this pattern when building station-specific features from the JSONB column.

## How This Data Should Be Used For HKG Tmax Modeling

This table can create features such as:

- Latest HKO temperature before cutoff.
- Latest HKO relative humidity before cutoff.
- HKO temperature trend over the last 1, 3, 6, 12, or 24 hours.
- Previous-evening HKO heat retention.
- Overnight minimum observed so far.
- Morning warm-up rate.
- Station spread across Hong Kong before cutoff.
- Inland/coastal contrast, for example Ta Kwu Ling or Sha Tin versus Cheung Chau or HKO.
- Airport/coastal reference, for example Chek Lap Kok and Cheung Chau.
- Urban-core comparison, for example King's Park, Happy Valley, Sham Shui Po, Hong Kong Park.
- Missing-station indicators, especially for frequently missing stations such as Shek Kong.
- Warning regime flags from `warning_text`.
- Lightning/thunderstorm regime flags from `lightning_text`.
- Tropical cyclone regime flags from `tropical_cyclone_text`, `tropical_cyclone_name`, latitude, and longitude.

The most important modeling rule is strict as-of safety:

Only use dispatches with `dispatch_at_utc <= decision_cutoff_utc`.

If the target is day T and the trading decision happens before day T begins, then only T-1 or earlier readings are allowed. If the strategy explicitly trades intraday on T, then only readings published before the intraday trade timestamp are allowed.

Do not train features using all readings from day T if the trading decision is made at T-1 23:59 HKT. That would leak the target day's weather evolution.

## Relationship To The Forecast Archive

The existing live forecast context file in this folder documents:

`public.hko_historical_forecasts_2000_2026`

That table stores Info.gov `LOCAL WEATHER FORECAST` dispatches. It is the apples-to-apples official forecast anchor table for the current strategy.

This new hourly readings table stores Info.gov `HOURLY READINGS` dispatches. It should be treated as an observational feature table.

A leakage-safe daily forecast system may combine the two like this:

1. Select the latest `LOCAL WEATHER FORECAST` row before the decision cutoff from `public.hko_historical_forecasts_2000_2026`.
2. Select the latest and recent-history `HOURLY READINGS` rows before the same cutoff from `public.hko_info_gov_hourly_readings_1998_2026`.
3. Join both to the target date T using only as-of-valid data.
4. Predict the HKO Daily Extract `Absolute Daily Max (deg. C)` for T.

The forecast archive answers: what did HKO forecast?

The hourly readings table answers: what was actually happening at the HKO station and nearby stations at each dispatch time?

## Known Caveats

The table is very high coverage, but not perfect.

Known caveats:

- `43` discovered Info.gov hourly-reading detail URLs could not be fetched and are listed in `detail_fetch_failures.csv/json`.
- `38` loaded rows are partial parses.
- `33` loaded rows are missing HKO target-station temperature/RH.
- `13` loaded rows have empty station JSON.
- Station lists changed across eras, so station columns derived from JSONB must be sparse and era-aware.
- Some neighbor-station raw temperatures contain source anomalies or very unusual values. The DB sanity bounds prevent impossible ingestion, but modeling should still use robust outlier treatment.
- The source reports hourly readings in integer Celsius, while the daily Tmax settlement target is one decimal place. This table is useful for weather-state features, not direct one-decimal settlement replacement.
- `retrieved_at_utc` is the backfill retrieval time, not historical publication time. Historical as-of availability should use `dispatch_at_utc` and `available_at_utc`, not `retrieved_at_utc`.

## Known Limitations and Follow-Up Work

Limitation: The table ends at `2026-07-04`.

- Impact: live predictions after that date need either an incremental backfill or direct live ingestion.
- Follow-up trigger: before using this table for a target date after 2026-07-04.

Limitation: Station readings are stored in JSONB rather than a child table.

- Impact: SQL feature extraction is slightly more verbose and requires `jsonb_array_elements` or JSONB containment filters.
- Reason: the requested persistence design was one dedicated Postgres table.
- Follow-up trigger: if feature-generation performance becomes a bottleneck, create a materialized feature view rather than changing this source table.

Limitation: Some neighbor station extremes require modeling caution.

- Impact: raw station min/max features can be distorted by station-specific anomalies.
- Follow-up trigger: before training a model that uses raw neighbor-station extremes directly.

Limitation: The hourly readings are integer Celsius observations.

- Impact: they do not directly replace the one-decimal HKO Daily Extract Tmax target.
- Follow-up trigger: when joining against the final settlement/label table.

## Testing and Verification Evidence

Verification performed for this documentation:

- File existence check:
  - Confirmed the new document exists at `documentation/strategy_implementation_documentation/context/live_trading/HKG_TMAX_INFO_GOV_HOURLY_READINGS_DATA_CONTEXT_20260705.md`.
  - Confirmed file size was about `36 KB` before structural additions.

- Unfinished-marker scan:
  - Command used ripgrep to look for unfinished-document markers in this Markdown file.
  - Result: no matches.

- Key-number scan:
  - Confirmed the document includes the live DB facts `268,894`, `1998-05-04`, `2026-07-04`, `268,856`, `38`, `43`, and `27`.

- Live DB evidence:
  - Queried `information_schema.columns` for the table schema.
  - Queried `pg_indexes` for indexes.
  - Queried `pg_constraint` for constraints.
  - Queried aggregate row counts, parse status counts, HKO null counts, station missingness, weather-context text counts, and yearly coverage.
  - Queried actual 2023 sample rows and included returned JSON output.

- Documentation quality gate:
  - The quality gate was run through the short Windows path because the full path exceeds legacy Python path handling in this workspace.
  - The first run correctly flagged missing implementation-style section headings and an unrelated dirty-worktree changed-file coverage problem.
  - This document was then updated to include the expected structural sections relevant to a data-context artifact.
  - The changed-file coverage error is not meaningful for this task because the repo already contains thousands of unrelated dirty/deleted files outside this documentation request.

## Operational Runbook

Inspect total row count and coverage:

```sql
select
  count(*) as rows,
  min(index_date_hkt) as first_index_date_hkt,
  max(index_date_hkt) as last_index_date_hkt,
  min(dispatch_at_utc) as first_dispatch_at_utc,
  max(dispatch_at_utc) as last_dispatch_at_utc
from public.hko_info_gov_hourly_readings_1998_2026;
```

Check parse status:

```sql
select parse_status, count(*)
from public.hko_info_gov_hourly_readings_1998_2026
group by parse_status
order by parse_status;
```

Find rows with missing target-station fields:

```sql
select source_url, index_date_hkt, title, dispatch_at_hkt, parse_status, parse_notes
from public.hko_info_gov_hourly_readings_1998_2026
where not target_station_present
order by index_date_hkt, dispatch_at_hkt;
```

Find rows with missing station readings:

```sql
select source_url, index_date_hkt, dispatch_at_hkt, station_count, station_missing_count
from public.hko_info_gov_hourly_readings_1998_2026
where station_missing_count > 0
order by dispatch_at_hkt
limit 100;
```

Query by JSONB station name:

```sql
select source_url, dispatch_at_hkt, station_readings_jsonb
from public.hko_info_gov_hourly_readings_1998_2026
where station_readings_jsonb @> '[{"station_canonical_name":"SHEK KONG","temperature_missing":true}]'::jsonb
limit 10;
```

Rebuild or extend the source:

```powershell
.\.venv\Scripts\python.exe scripts\backfill_hko_info_gov_hourly_readings.py --load-db
```

For a bounded rerun:

```powershell
.\.venv\Scripts\python.exe scripts\backfill_hko_info_gov_hourly_readings.py --start 2026-07-05 --end 2026-07-05 --load-db
```

The script is resumable through the SQLite ledger and raw HTML folders. Reruns skip already fetched successful detail pages unless forced.

## Recommended Feature Engineering Contract

For a target date T and decision cutoff C:

1. Convert C to UTC.
2. Pull only rows where `dispatch_at_utc <= C`.
3. If using observation timestamps instead of dispatch timestamps, still require `dispatch_at_utc <= C`; an observation is not usable until the bulletin was dispatched.
4. Derive latest reading features from the final row before C.
5. Derive trailing-window features from rows in the hours before C.
6. Derive station-spread features only from non-missing station values.
7. Add missing indicators for each station and each aggregate.
8. Treat warning/lightning/tropical-cyclone text as categorical or boolean weather-regime features.
9. Never use rows after the trading cutoff for a pre-target-day decision.
10. Keep the forecast archive and hourly readings features separate in code so there is no source confusion.

The safest interpretation is:

- `dispatch_at_utc` controls what a trader/model could know.
- `observation_at_utc` controls what meteorological hour the reading represents.
- `index_date_hkt` controls the Hong Kong calendar day of the Info.gov index.

## Evidence Sources Used For This Document

This document was written from:

- Direct read-only Postgres queries against `public.hko_info_gov_hourly_readings_1998_2026`.
- `data/datasets/13_hko_info_gov_hourly_readings/reports/postgres_load_summary.json`.
- `data/datasets/13_hko_info_gov_hourly_readings/reports/structure_pattern_report.json`.
- `data/datasets/13_hko_info_gov_hourly_readings/README.md`.
- `migrations/postgres/20260704_0008_hko_info_gov_hourly_readings.sql`.
- The implemented backfill script `scripts/backfill_hko_info_gov_hourly_readings.py`.

The key verified DB facts are:

- Table: `public.hko_info_gov_hourly_readings_1998_2026`.
- Rows: `268,894`.
- Date range: `1998-05-04` through `2026-07-04`.
- Parse status: `268,856 parsed`, `38 partial`, `0 failed`.
- Target-station present rows: `268,861`.
- Explicit failed discovered URLs: `43`.
- Unique station names in JSONB: `27`.

## Reviewer Checklist

Use this checklist before relying on the table in a model:

- Confirm the table name is exactly `public.hko_info_gov_hourly_readings_1998_2026`.
- Confirm the target use case needs observed hourly readings, not official forecast min/max values.
- Confirm the modeling code filters by `dispatch_at_utc <= cutoff`.
- Confirm the modeling code does not use future target-day readings for a pre-target-day decision.
- Confirm missing station values are represented with explicit missing indicators.
- Confirm `station_readings_jsonb` parsing handles `temperature_c = null` and `temperature_missing = true`.
- Confirm station-derived features are sparse/era-aware because station coverage changes over time.
- Confirm neighbor-station extremes are clipped or robustly handled.
- Confirm the forecast archive `public.hko_historical_forecasts_2000_2026` remains a separate source in code.
- Confirm any live extension after `2026-07-04` either reruns the backfill or uses an equivalent live Info.gov hourly-readings ingestion path.
