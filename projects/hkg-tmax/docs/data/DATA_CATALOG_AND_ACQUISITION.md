# Data Catalog and Acquisition Strategy

## Principle

“Download everything” means **enumerate, prioritize, lawfully acquire, version, and understand everything potentially useful**. It does not mean scrape blindly, violate terms, or mix retrospective data into operational forecasts.

The authoritative source registry is `config/data_sources.yaml`.

## Data layers

### Raw

Exact payload as received plus sidecar:

```text
source_id
request URL and parameters
retrieved_at
HTTP status
headers
content SHA-256
content length
adapter version
```

Raw is immutable.

### Bronze

Source-native parsed rows with minimal normalization. Preserve original fields and units.

### Silver

Quality-controlled, unit-normalized, timestamp-resolved, station-identified records.

### Gold

Forecast examples, targets, features, predictions, and evaluation tables tied to exact versions.

## Storage partitioning

Recommended:

```text
data/raw/<source_id>/YYYY/MM/DD/<retrievedUTC>__<sha12>.<ext>
data/bronze/<dataset_version>/...
data/silver/<dataset_version>/...
data/gold/<dataset_version>/...
```

For NWP:

```text
model/init_time/member/variable/level/forecast_hour/file
```

For images:

```text
product/scan_time/retrieved_time/hash.ext
```

## Acquisition priority

### P0 — Start immediately

- Daily Extract target polling;
- CLMMAXT history;
- live 1-minute temperature;
- max/min since midnight;
- HKO official forecasts;
- station metadata;
- Polymarket event rules;
- Polymarket books/trades;
- ECMWF/GFS/GEFS cycles if accessible;
- exact source-version discovery.

These are perishable and often lack complete public vintage archives.

### P1 — High expected value

- humidity, pressure, wind, rain, radiation;
- regional station network;
- radar, satellite, lightning;
- automatic regional forecasts;
- upper-air soundings;
- ICON/ICON-EPS;
- tropical-cyclone advisories/tracks;
- historical hourly observations.

### P2/P3 — Mechanism and incremental research

- reanalysis;
- air-quality/aerosol proxies;
- land cover, terrain, coastline;
- marine/tide/SST;
- specialized satellite products;
- paid historical observation requests after value assessment.

## Date ranges

### Target history

- ingest all CLMMAXT dates available;
- retain 1884+ for climatology and structural-break research;
- prioritize modern periods for current-regime prediction;
- explicitly model/document the 1940–1946 gap;
- do not assume equal relevance across 140 years.

### Daily covariates

Acquire maximum feasible overlap. Record element-specific starts and missing periods.

### Sub-daily observations

Acquire all official historical data available. If only latest is public, begin self-archival immediately and evaluate paid official history.

### NWP vintages

Acquire every historical operational cycle available from official archives, but distinguish:

- true archived operational files;
- retrospective reruns;
- interpolated convenience datasets.

Begin prospective archival now because rolling open archives expire.

### Market data

Backfill metadata and price history, then prospectively archive full books/deltas. Historical price series is not historical depth.

## Rate and integrity controls

- identify with a research user agent where appropriate;
- apply source-specific minimum intervals;
- exponential retry with cap;
- do not retry permanent 4xx blindly;
- verify content length and format;
- detect provider error pages returned with HTTP 200;
- save partial/incomplete state separately;
- hash before parse;
- checkpoint backfills;
- write query manifests;
- never silently skip failed dates.

## Source contracts

Each adapter gets `docs/source_contracts/<source_id>.md` with:

- purpose;
- official evidence;
- endpoint/query;
- response schema;
- units;
- timestamps;
- cadence;
- historical coverage;
- latency;
- revisions;
- quality flags;
- rate/terms;
- operational role;
- tests;
- known limitations.

## Coverage report

Generate matrices by:

- date × source;
- station × element;
- model × cycle × member × variable;
- date × horizon;
- date × target parity;
- market event × order-book completeness.

Coverage is itself a potential source of bias. Every model comparison should report common-sample and all-available-sample results.

## Image data

Radar/satellite ingestion must preserve:

- original image;
- product name;
- scan time;
- publication/retrieval time;
- projection/geolocation;
- palette/units;
- missing sectors;
- processing code/hash.

Derived image features are versioned separately. Never train on an animation or composite that includes post-cutoff frames.

## Historical archive API

DATA.GOV.HK’s historical-data API may expose file versions for eligible datasets. Codex must test coverage source by source; do not assume every HKO live feed has historical versions. Archive all API responses and use Hong Kong time semantics documented by the provider.

## Paid data decision

For paid official hourly/sub-hourly history:

1. enumerate variables, stations, dates, cadence, and quote;
2. estimate expected research value;
3. run a power/coverage analysis;
4. document license and storage constraints;
5. purchase only when it fills a material validation gap.

## Data minimization for reliability

Store broadly, but do not force every source into a model. Data is eligible only after:

- provenance pass;
- as-of pass;
- quality audit;
- plausible mechanism;
- incremental evidence.
