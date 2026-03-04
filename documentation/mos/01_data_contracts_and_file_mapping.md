# 01 - Data Contracts and File Mapping

This document is the schema and mapping contract for MOS backtesting inputs/outputs.

Canonical-source note:

1. For station-onboarding and canonical MOS/NWS history, `D:\Ahmed\data\sqlite` is the primary read source.
2. CSV inputs referenced below are compatibility artifacts derived from SQLite for current downstream script interfaces.

## 1) Prediction Input Contracts

### 1.1 Required Prediction Columns

All prediction parquet inputs used by MOS backtests must include:

- `target_date_local`
- `y_tmax`
- `q_0.05`, `q_0.10`, `q_0.25`, `q_0.50`, `q_0.75`, `q_0.90`, `q_0.95`

### 1.2 Normalization Rules

- `target_date_local` is normalized to date (no time component).
- duplicate `target_date_local` rows are resolved by keeping the last row.
- dev and test files are concatenated before date lookup.

### 1.3 Common Prediction Paths

Single-station baseline:

- `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_00\dev_predictions.parquet`
- `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_00\test_predictions.parquet`

Co-joined blend_12:

- KNYC:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\dev_predictions.parquet`
  - `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\test_predictions.parquet`
- KMIA:
  - `D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\dev_predictions.parquet`
  - `D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\test_predictions.parquet`
- KMDW:
  - `D:\Ahmed\data\kalshi\Experiments\MOS_KMDW\03_blends\blend_12\dev_predictions.parquet`
  - `D:\Ahmed\data\kalshi\Experiments\MOS_KMDW\03_blends\blend_12\test_predictions.parquet`

### 1.4 Generic Station Mapping Inputs (Co-Joined Script)

`backtesting/mos_blend12_knyc_kmia_cojoined_audit.py` now supports station-agnostic maps via:

- `--pred-dev-by-station-json`
- `--pred-test-by-station-json`
- `--truth-csv-by-station-json`
- `--kalshi-root-by-station-json`
- `--file-prefix-by-station-json`

Values may be:
1. inline JSON object text, or
2. path to a JSON file containing `{ "STATION": "path" }`.

## 2) Market Input Contracts

### 2.1 Day File Naming

- KNYC: `KNYC_YYYYMMDD.csv`
- KMIA: `KMIA_YYYYMMDD.csv`
- KMDW: `KMDW_YYYYMMDD.csv`

### 2.2 Required Market Columns

- mandatory: `timestamp` (parsed as UTC)
- all remaining columns are treated as candidate bucket labels

### 2.3 Market Roots in Current Runs

Single-station historical root:

- `D:\Ahmed\data\kalshi\kalshi_history`

Co-joined dedicated roots:

- KNYC:
  - `D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2024_10_01_to_2025_12_31`
- KMIA:
  - `D:\Ahmed\data\kalshi\kalshi_history\kxhighmia_2024_10_01_to_2025_12_31`
- KMDW:
  - `D:\Ahmed\data\kalshi\kalshi_history\kxhighchi_2024_10_01_to_2026_03_03`

## 3) Date Mapping Rules

For prediction target date `T`:

- only day file `*_YYYYMMDD.csv` with `YYYYMMDD == T` is valid.
- no `T-1` filename shift is allowed.

`T-1` is used only for gate time computation, never for selecting day file date.

## 4) Duplicate-Date Resolution

When scanning broad roots containing multiple folders with same day file:

- resolve deterministically by configured folder precedence.
- each `target_date_local` must resolve to exactly one selected market file.

Co-joined dedicated roots (`kxhighny_2024_10_01_to_2025_12_31` and `kxhighmia_...`) avoid most duplicate collisions by design.

## 5) Timestamp and Entry Semantics

All market timestamps are UTC instants.

Gate cutoff:

- `gate_cutoff_utc = (T - 1 day) + entry_hour_z:entry_minute_z`

Optional open delay:

- `effective_cutoff_utc = max(gate_cutoff_utc, market_open_utc + delay_minutes)`

Entry row:

- first market row with `timestamp >= effective_cutoff_utc`

If no open-delay is configured, `effective_cutoff_utc == gate_cutoff_utc`.

## 6) Bucket Label Parsing Contract

Supported forms:

- `X to Y`
- `X-Y`
- `X or below` / `X or less`
- `X or above` / `X or higher`

Critical parsing rule:

- extract unsigned integers from label text.

This prevents interpreting range hyphen as negative sign.

Canonical labels emitted by backtest outputs:

- `81F or below`
- `45F to 46F`
- `98F or above`

## 7) Price Normalization Contract

Raw value normalization for each bucket column value:

1. missing -> invalid candidate
2. `< 0` -> invalid candidate
3. `> 1` -> divide by `100` (percentage scale fallback)
4. clip to `[0, 1]`

Side mapping:

- YES side price = normalized bucket column value
- NO side price = `1 - YES price`

## 8) Truth/Settlement Contract

`y_tmax` from prediction rows is used as settlement truth for win/loss adjudication:

- YES wins if `y_tmax` is inside bucket
- NO wins if `y_tmax` is outside bucket

## 9) Output Table Column Contract

Co-joined side-aware-with-balance table is expected to contain:

- `Entry time (Stockholm)`
- `Station`
- `Bucket`
- `Side`
- `Market win % (side)`
- `Model win %`
- `EV`
- `Amount invested ($)`
- `Profit made ($)`
- `Result`
- `Balance after trade ($)`
- `Market open (UTC)`
- `Gate cutoff (UTC)`
- `Effective cutoff (UTC)`
- `Market file`
