# WFPM-103 — Ingest FAWN mesonet data (daily summaries + near‑real‑time API) into DB

**Type:** Story  
**Priority:** P1  
**Epic:** WFPM-EPIC-EXTDATA  

## Goal

Ingest Florida Automated Weather Network (FAWN) observations for a small set of South Florida stations and persist them in DB so they can be used as **antecedent wetness + boundary layer state** features for KMIA next‑day Tmax.

We will store:

1) **Daily summaries** (multi‑year history, stable, small files) → best for rainfall/soil temperature context.
2) **Near‑real‑time 15‑minute observations (last96)** → best for operational/live “as‑of” updates.

## Why FAWN matters (signal)

FAWN adds signals that airports often do not provide reliably:

- **Soil temperature (`tsoil`)**: proxy for land surface heat storage; affects next‑day sensible heat flux and Tmax.
- **Recent rainfall (`rain`)**: wet ground suppresses Tmax via latent heat (especially important in summer convection cycles).
- **Radiant flux density (`rfd`)**: cloudiness proxy; morning cloud cover suppresses Tmax.
- **Wind @ 10m (`ws`, `wdir`)**: inland/canopy wind helps detect mixing regime.

## Data source (free, public)

FAWN official API documentation (columns, units, and endpoint patterns are defined here):  
`https://fawn.ifas.ufl.edu/controller.php/today/`

That page lists:

- `today/obs` and `today/last96` endpoints  
- Column definitions and units (e.g., `t2m` °C, `ws` km/hr, `rain` inches, etc.)

FAWN station ID list (needed to build URLs):  
`https://fawn.ifas.ufl.edu/controller.php/today/` → “StationID” link

FAWN historical downloads (daily summaries) are available under the public “fawnpub” directory; examples referenced in UF/IFAS publications include:  
`https://fawn.ifas.ufl.edu/data/fawnpub/daily_summaries/BY_YEAR/2020_daily.zip`  
`https://fawn.ifas.ufl.edu/data/fawnpub/daily_summaries/BY_YEAR/2021_daily.csv`  
(Other years follow the same BY_YEAR pattern.)

## Stations (v1)

Use these station IDs (from FAWN StationID list):

- **420** — Ft. Lauderdale (coastal urban)
- **440** — Homestead (south / inland edge)
- **410** — Belle Glade (interior Everglades signal)
- **425** — Wellington (Palm Beach inland)

These give us coastal vs interior gradients and wetness regime shifts that matter for Miami Tmax.

## Endpoints (exact)

### A) Near‑real‑time, last 96 obs (15‑minute)

JSON:
- `https://fawn.ifas.ufl.edu/controller.php/today/last96/{id};json`

CSV:
- `https://fawn.ifas.ufl.edu/controller.php/today/last96/{id};csv`

Example for Ft Lauderdale (420):
- `.../today/last96/420;json`

Per FAWN docs: observations are generally 15‑minute averages/sums from 5‑second sampling.

### B) Daily summaries (historical backfill)

Base path:
- `https://fawn.ifas.ufl.edu/data/fawnpub/daily_summaries/BY_YEAR/`

File name patterns (implement both; the server uses a mix of csv and zip historically):
- `{YYYY}_daily.csv`
- `{YYYY}_daily.zip`

Inside `.zip`, assume there is at least one `.csv` file; pick the largest `.csv` by bytes if more than one.

## Parsing rules

### Near‑real‑time API (`last96`)

FAWN “today” docs define the key fields and units (you must parse at least these):

- `StationID` (int)
- `endTime` (ISO8601; treat as UTC if it ends with `Z`)
- `rfd` (W/m^2)
- `tsoil` (°C)
- `t2m` (°C)
- `t60cm` (°C)
- `t10m` (°C)
- `rh` (%)
- `ws` (km/hr)
- `wsmax` (km/hr)
- `wdir` (degrees from N)
- `rain` (inches)
- `dp` (°C)

Implementation detail:
- Parse JSON to records; convert numeric fields to float; missing should become NULL.
- Convert `ws` and `wsmax` to m/s in derived columns: `ws_ms = ws_kmh / 3.6`.

### Daily summaries

The daily file is expected to have at minimum:
- a station identifier column
- a date column (local or UTC; treat carefully)
- daily aggregates (tmax/tmin/tavg, rain totals, maybe soil temp summaries)

**Important requirement:** Implement this parser defensively:
- Read the header row dynamically.
- Normalize column names to snake_case.
- Identify “date” column by best match among: `date`, `day`, `datetime`, `ymd`.
- Identify station column by best match among: `stationid`, `station_id`, `stid`.

If the daily file lacks a clear date/station column, fail fast with an actionable error showing the first 5 header columns.

## Database schema

### Table 1 — 15‑minute observations (near‑real‑time, last96)

```sql
CREATE TABLE IF NOT EXISTS fawn_obs_15min (
  station_id        INTEGER NOT NULL,
  end_time_utc      TIMESTAMP NOT NULL,

  rfd_wm2           REAL,
  tsoil_c           REAL,
  t2m_c             REAL,
  t60cm_c           REAL,
  t10m_c            REAL,
  rh_pct            REAL,
  ws_kmh            REAL,
  wsmax_kmh         REAL,
  wdir_deg          REAL,
  rain_in           REAL,
  dp_c              REAL,

  -- derived
  ws_ms             REAL,
  wsmax_ms          REAL,
  rain_mm           REAL,

  source            TEXT NOT NULL DEFAULT 'FAWN_TODAY_LAST96',
  source_url        TEXT,
  retrieved_at_utc  TIMESTAMP NOT NULL,
  raw_payload_hash_ref TEXT,

  PRIMARY KEY (station_id, end_time_utc)
);
CREATE INDEX IF NOT EXISTS idx_fawn_obs15_time ON fawn_obs_15min(end_time_utc);
```

Derived conversions:
- `ws_ms = ws_kmh / 3.6`
- `wsmax_ms = wsmax_kmh / 3.6`
- `rain_mm = rain_in * 25.4`

### Table 2 — Daily summaries (historical)

Store daily summaries in a wide but schema‑stable way. Because the daily CSV schema may evolve, store:
- a small set of canonical columns
- plus a JSON blob of “extra columns” to avoid losing information

```sql
CREATE TABLE IF NOT EXISTS fawn_daily_summary (
  station_id         INTEGER NOT NULL,
  date_local         DATE NOT NULL,  -- interpret in America/New_York unless file states otherwise

  tmax_c             REAL,
  tmin_c             REAL,
  tmean_c            REAL,
  rain_in            REAL,
  rain_mm            REAL,
  tsoil_mean_c       REAL,
  rfd_mean_wm2       REAL,

  extra_json         TEXT,           -- JSON string of remaining columns

  source             TEXT NOT NULL DEFAULT 'FAWN_DAILY_SUMMARY',
  source_url          TEXT,
  retrieved_at_utc    TIMESTAMP NOT NULL,
  raw_payload_hash_ref TEXT,

  PRIMARY KEY (station_id, date_local)
);
CREATE INDEX IF NOT EXISTS idx_fawn_daily_date ON fawn_daily_summary(date_local);
```

## Ingestion implementation

### CLI

`python -m weather_ml.ingest.fawn --stations 410,420,425,440 --start 2007-01-01 --end 2026-02-18 --db <dsn>`

Subcommands/modes:
- `--mode daily_backfill` → downloads BY_YEAR files and loads daily summaries
- `--mode last96_poll` → polls last96 for each station and upserts 15‑min obs

### Daily backfill algorithm

For each year in range:
1) Try download `{YYYY}_daily.zip` first; if 404, try `{YYYY}_daily.csv`.
2) If zip:
   - download bytes
   - open zip in memory
   - choose the largest `.csv`
3) Parse CSV:
   - normalize column names
   - map into canonical columns + extra_json
4) Upsert.

### last96 polling algorithm

Run every hour (or every 15 minutes if desired):
1) For each station:
   - download `{id};json`
   - parse records
   - upsert
2) This is idempotent and can run frequently.

## Validation & QA

- `end_time_utc` should align to 15‑minute boundaries (minute in {00,15,30,45}); warn if not.
- Rain should be >= 0.
- Temperature sanity bounds (°C) for Florida:
  - t2m_c in [-5, 45]
  - tsoil_c in [0, 45]
- Idempotency tests:
  - running the same last96 payload twice yields 0 duplicates.
- Daily summary integrity:
  - No duplicate (station, date) keys.

## Acceptance criteria

- DB tables exist and contain:
  - last96 15‑minute obs for at least station 420 and 440 for the last 24 hours
  - daily summaries successfully loaded for at least years 2020–2025 (more if available)
- Parsers are robust to schema evolution (unknown columns preserved in `extra_json`)
- Unit tests exist for:
  - JSON parse + upsert
  - zip extraction + csv parse
  - column normalization and mapping

## Notes for downstream “as‑of” usage

Because daily summary is *end-of-day*, the feature builder must only use daily rows that are certainly complete at decision time:
- For target day T (decision T‑1 12Z), use daily summaries up through **T‑2**.
Near‑real‑time last96 can be used up to `asof_utc − 30 minutes` for operational forecasts.
