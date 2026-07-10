# HKG Tmax Data Acquisition Scope

## Current Goal

Build the complete, trustworthy, point-in-time data foundation for forecasting
official daily maximum air temperature at Hong Kong Observatory Headquarters
station `HKO` on local civil date `T` in `Asia/Hong_Kong`.

This goal covers data acquisition, archiving, normalization, provenance,
coverage, and quality control only.

## Explicit Exclusions

- Polymarket APIs, metadata, rules, prices, order books, trades, liquidity,
  fees, buckets, resolutions, profitability, and backtesting.
- Additional settlement-parity work.
- Rapid Daily Extract polling loops.
- One experiment folder per polling window or routine download.
- Forecast model training, tuning, scoring, feature selection, or comparison.
- Locked-test evaluation.

Existing Polymarket files and history are preserved but not used or modified by
this acquisition goal.

## Target Definition

The meteorological target for acquisition is:

```text
Official daily maximum air temperature
Station: Hong Kong Observatory Headquarters
Station code: HKO
Location: Tsim Sha Tsui
Date basis: local civil date T in Asia/Hong_Kong
```

## Daily Extract Policy

Daily Extract collection is operational monitoring, not research experimentation:

- at most one successful fetch per `Asia/Hong_Kong` local day;
- scheduled at 09:00 local time;
- one retry six hours later only after a failed request;
- unchanged content hashes are deduplicated;
- every retrieval attempt is appended to the acquisition ledger;
- routine unchanged retrievals do not create experiments, full test runs, or
  commits.

## Data Root

The repository path is long on Windows and long paths are not enabled on this
machine. Large acquisition data therefore belongs under the configurable
`HKG_TMAX_DATA_ROOT`, with `C:\hkg_tmax_data` as the default local path in
`.env.example`.

Bulk raw, bronze, silver, gold, logs, state, and manifests are not committed to
Git. Code, configuration, source contracts, reports, schemas, and small
machine-readable catalogs are committed.
