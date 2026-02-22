# WFPM-101 — Ingest ASOS/METAR “airport ring” observations (IEM `asos.py`) into DB

**Type:** Story  
**Priority:** P0 (blocking all downstream feature work)  
**Epic:** WFPM-EPIC-EXTDATA  

## Goal (what this enables)

Persist high-frequency **surface truth** around KMIA (temperature, dewpoint, RH, wind, pressure, visibility, hourly precip) so we can build leakage‑free “as‑of T‑1 12Z” state features (sea‑breeze signal, boundary‑layer moisture, morning trend) for next‑day Tmax forecasting.

## Why it matters (signal)

Airport METAR/ASOS is the cleanest, most reliable near‑real‑time measurement set we can use for:
- **Morning trend** (how fast the boundary layer is warming/drying)
- **Sea‑breeze regime detection** (coastal vs inland gradients, wind direction shifts)
- **Convective suppression proxy** (dewpoint + wind + pressure tendencies)

## Data source (100% free)

Iowa Environmental Mesonet (IEM) ASOS/METAR request CGI.

- Help / parameter doc: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=`  
- Variable units overview: `https://mesonet.agron.iastate.edu/request/download.phtml` (describes `p01i`, `alti`, `mslp`, `vsby`)  

### Stations to ingest (initial v1 set)

Use exactly these station IDs (ICAO):

- **KMIA** (target anchor)
- **KFLL** (coastal north)
- **KOPF** (nearby north / urban)
- **KTMB** (southwest / inland edge)
- **KHST** (south / inland)
- **KPBI** (north gradient)
- (Optional later) **KEYW** (marine air mass proxy)

Stations are encoded in the request as repeated `station=` parameters.

### Columns to request (v1)

Request a **minimal but high-signal** subset (all are available options in `asos.py`):

- `tmpf` (F)
- `dwpf` (F)
- `relh` (%)
- `drct` (deg from N)
- `sknt` (knots)
- `gust` (knots)
- `p01i` (1h precip, inches)
- `alti` (altimeter, inches)
- `mslp` (sea level pressure, millibar)
- `vsby` (miles)
- `skyc1..skyc4`, `skyl1..skyl4` (cloud cover + base; optional but useful for Tmax cap)
- `wxcodes` (optional; rain/thunder flags)

**Important:** The IEM output uses `"M"` for missing values. Treat as null.

## Request format (exact URL template)

Endpoint:

`https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`

Use these parameters:

- `station=KMIA&station=KFLL&...`
- `data=tmpf&data=dwpf&data=relh&...` (repeat per field)
- `year1=YYYY&month1=MM&day1=DD`
- `year2=YYYY&month2=MM&day2=DD`
- `tz=UTC`  (do not rely on defaults)
- `format=onlycomma`
- `missing=M` (or default; we still coerce `"M"` → NULL)
- `latlon=yes` and `elev=yes` (adds station metadata; safe and helpful)
- `report_type=3` (routine METARs; optional but recommended to reduce noisy SPECI bursts)
- `direct=yes` (forces file download; when implementing HTTP requests, it’s fine to omit and just read response)

**Chunking rule (do this):** request at **monthly granularity** per station set to avoid large responses/timeouts.

Example (Jan 2025):

`...&year1=2025&month1=1&day1=1&year2=2025&month2=2&day2=1&tz=UTC&format=onlycomma&...`

## Parsing rules (must be exact)

1) Parse CSV header row; expected columns include:
   - `station` (string)
   - `valid` (UTC timestamp)
   - requested data columns
   - optional `lat`, `lon`, `elevation_m`
2) Convert `valid` to `valid_time_utc` (timezone-aware UTC).
3) Replace `"M"` (and empty strings) with NULL.
4) Convert types:
   - `tmpf`, `dwpf`, `relh`, `drct`, `sknt`, `gust`, `p01i`, `alti`, `mslp`, `vsby` → float (nullable)
5) Create derived unit conversions **at ingest time** (store both raw and metric):
   - `sknt` knots → `wind_ms = sknt * 0.514444`
   - `gust` knots → `gust_ms = gust * 0.514444`
   - `p01i` inches → `p01_mm = p01i * 25.4`
   - `alti` inches Hg → `altimeter_hpa = alti * 33.8638866667`
   - `tmpf`, `dwpf` F → `tmpc`, `dwpc` in °C
6) Generate a **row hash** for raw provenance:
   - `raw_payload_hash_ref` (sha256 of the raw CSV bytes OR sha256 of the parsed row JSON string; follow project conventions)

## Database schema (DDL)

Create a new table (or adapt to your DB migration style). Use UTC for time columns.

```sql
CREATE TABLE IF NOT EXISTS iem_asos_obs (
  station_id           TEXT NOT NULL,
  valid_time_utc       TIMESTAMP NOT NULL,
  -- raw units (IEM native)
  tmpf                REAL,
  dwpf                REAL,
  relh                REAL,
  drct_deg            REAL,
  sknt                REAL,
  gust                REAL,
  p01i                REAL,
  alti_inhg           REAL,
  mslp_mb             REAL,
  vsby_mi             REAL,
  skyc1               TEXT,
  skyc2               TEXT,
  skyc3               TEXT,
  skyc4               TEXT,
  skyl1_ft            REAL,
  skyl2_ft            REAL,
  skyl3_ft            REAL,
  skyl4_ft            REAL,
  wxcodes             TEXT,

  -- derived metric units (recommended for downstream)
  tmpc                REAL,
  dwpc                REAL,
  wind_ms             REAL,
  gust_ms             REAL,
  p01_mm              REAL,
  altimeter_hpa       REAL,

  -- provenance
  source              TEXT NOT NULL DEFAULT 'IEM_ASOS',
  source_url          TEXT,
  retrieved_at_utc    TIMESTAMP NOT NULL,
  raw_payload_hash_ref TEXT,

  PRIMARY KEY (station_id, valid_time_utc)
);
CREATE INDEX IF NOT EXISTS idx_iem_asos_obs_time ON iem_asos_obs(valid_time_utc);
```

## Ingestion implementation (Codex must implement exactly)

### New CLI entrypoint

Create a CLI command (pattern: existing ingestion CLIs) named:

`python -m weather_ml.ingest.iem_asos_ring --start YYYY-MM-DD --end YYYY-MM-DD --stations KMIA,KFLL,... --db <dsn>`

Required flags:
- `--start` and `--end` (inclusive start, exclusive end; document clearly)
- `--stations` comma list
- `--chunk` default `month`
- `--max-retries` default 5
- `--sleep-seconds` default 1.0 (polite throttling)
- `--db` DSN

### Algorithm

For each month chunk in [start, end):
1) Build URL with `year1/month1/day1` and `year2/month2/day2` boundaries.
2) HTTP GET with timeout 60s; retry with exponential backoff on non-200/timeout.
3) If response is empty (no data), continue.
4) Parse CSV to rows; upsert into DB.
5) Log counts: requested rows, inserted, updated, skipped.

### Backfill plan (must be runnable)

Backfill: **2007‑01‑01 → today** for the station list above.

- Use monthly chunks.
- Store progress checkpoints (e.g., a `ingest_state` table with last completed month per source/station set) so the job can resume.

### Daily incremental plan

A daily job runs at ~01:00Z:
- Fetch yesterday UTC day plus today so far (2-day window) for safety.
- Upsert (idempotent).
- This ensures late-reporting stations don’t create holes.

## Validation & QA (acceptance criteria)

1) **Row count sanity**: for KMIA, expect multiple observations per hour (METAR) — at least 12 per day typical; flag days with < 8 as potential outage.
2) **Range checks** (non-fatal warnings):
   - tmpf in [-20, 120]
   - dwpf <= tmpf (allow small tolerance)
   - drct in [0, 360]
   - sknt >= 0, gust >= 0
   - mslp in [900, 1100] mb
3) **Time correctness**: all `valid_time_utc` are timezone-aware UTC; no naive timestamps.
4) **Idempotency**: rerunning the same month produces 0 net new rows and no duplicates.
5) **Unit conversion tests**: verify a known value converts correctly (e.g., 10 knots = 5.14444 m/s).

## Definition of done

- Table exists and is populated for 2007‑present for at least KMIA + KFLL.
- CLI backfill completes without manual edits.
- Unit tests exist for parser + upsert + conversions.
- A short README is added describing how to run and where data live.
