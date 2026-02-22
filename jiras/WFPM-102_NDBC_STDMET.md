# WFPM-102 — Ingest coastal marine observations (NDBC/NOS) for South Florida into DB

**Type:** Story  
**Priority:** P0  
**Epic:** WFPM-EPIC-EXTDATA  

## Goal

Persist near‑shore marine observations that control the **sea‑breeze boundary condition** for KMIA Tmax. We will ingest two “high‑leverage” coastal stations:

- **VAKF1** — Virginia Key, FL (very near Miami coast)
- **PEGF1** — Port Everglades, FL (north coastal reference)

These stations are hosted on the NOAA NDBC site and provide wind and pressure at minimum; many stations also provide air temperature and (sometimes) water temperature.

## Why this matters

Miami Tmax busts are often caused by sea‑breeze timing/strength errors. The **marine wind direction/speed + pressure + air temperature** upwind of KMIA provides a strong, low‑noise indicator of whether the sea breeze will arrive early (cooler Tmax) or late/weak (hotter Tmax).

## Data source (free, public)

NOAA National Data Buoy Center (NDBC):

- 5‑day rolling “standard meteorological” files format: https://www.ndbc.noaa.gov/faq/5day.shtml  
- File unit header behavior and examples (important for parsing): https://www.ndbc.noaa.gov/faq/mods.shtml  
- Measurement descriptions / units reference: https://www.ndbc.noaa.gov/faq/measdes.shtml  
- 5‑day file directory listing (proves URL pattern exists): https://www.ndbc.noaa.gov/data/5day2/  
- Historical archive directory listing (proves naming pattern exists): https://www.ndbc.noaa.gov/data/historical/stdmet/  
- Station pages (metadata + coords):
  - VAKF1: https://www.ndbc.noaa.gov/station_page.php?station=vakf1
  - PEGF1: https://www.ndbc.noaa.gov/station_page.php?station=pegf1

## Endpoints (exact)

### A) Near‑real‑time (for daily updates)

Standard met 5‑day file:

- `https://www.ndbc.noaa.gov/data/5day2/VAKF1_5day.txt`
- `https://www.ndbc.noaa.gov/data/5day2/PEGF1_5day.txt`

Per NDBC docs, these files contain a header and the most recent observations. Header example includes a second units line in modern format (see NDBC “mods” examples).

### B) Historical backfill (for model training)

Historical “standard meteorological” yearly gz files:

- Directory: `https://www.ndbc.noaa.gov/data/historical/stdmet/`
- Naming pattern: `{station_lower}h{YYYY}.txt.gz`

Examples of naming in the directory listing include `spgf1h1985.txt.gz`, so for our stations:

- `vakf1h2007.txt.gz`, `vakf1h2008.txt.gz`, …  
- `pegf1h2007.txt.gz`, `pegf1h2008.txt.gz`, …

**Important:** some stations may not exist for all years; the ingestion must tolerate 404.

## Parsing rules (must be robust)

NDBC text formats vary slightly across time. Implement a parser that:

1) Reads text (decompressed if `.gz`).
2) Skips blank lines.
3) Detects header lines:
   - Lines beginning with `#` are headers/metadata.
   - In modern format, there are often **two header lines**:
     - First: column names (e.g., `#YY MM DD hh mm WDIR WSPD ...`)
     - Second: units (e.g., `#yr mo dy hr mn degT m/s ...`)
   - In older format (pre‑2007 or some archives), only one header may exist.
4) The first non-`#` line begins data.

### Standard meteorological fields (canonical set)

From NDBC 5‑day format doc, header can include:

`YYYY MM DD hh WD WSPD GST WVHT DPD APD MWD BARO ATMP WTMP DEWP VIS PTDY TIDE`

From NDBC “mods” examples, many files use:

`#YY  MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS PTDY TIDE`

Your parser must map both naming styles to canonical internal names.

### Missing values

NDBC uses `MM` (and sometimes `99`, `999`, `9999`) to indicate missing. Treat these as NULL.

## Normalization decisions (do these)

- Store `valid_time_utc` from the `YY/MM/DD/hh/mm` columns (UTC).
- Convert numeric fields to float.
- Keep the original NDBC units in “raw_*” columns and also create a few derived metric fields:
  - `wspd_ms` is already m/s in most NDBC files; keep as-is.
  - `atmp_c`, `dewp_c`, `wtmp_c` are °C; keep.
  - `pres_hpa` is hPa; keep.

## Database schema (DDL)

```sql
CREATE TABLE IF NOT EXISTS ndbc_stdmet_obs (
  station_id         TEXT NOT NULL,
  valid_time_utc     TIMESTAMP NOT NULL,

  -- winds
  wdir_deg           REAL,   -- WDIR/WDIR/WD
  wspd_ms            REAL,   -- WSPD
  gust_ms            REAL,   -- GST

  -- thermodynamics
  atmp_c             REAL,   -- ATMP
  dewp_c             REAL,   -- DEWP
  wtmp_c             REAL,   -- WTMP (may be missing for some stations)

  -- pressure and tendency
  pres_hpa           REAL,   -- PRES or BARO
  ptdy_hpa           REAL,   -- PTDY (pressure tendency)

  -- sea state (optional; many NOS stations do not have)
  wvht_m             REAL,   -- WVHT
  dpd_s              REAL,   -- DPD
  apd_s              REAL,   -- APD
  mwd_deg            REAL,   -- MWD

  -- misc
  vis_mi             REAL,   -- VIS
  tide_ft            REAL,   -- TIDE (often missing)

  -- provenance
  source             TEXT NOT NULL DEFAULT 'NDBC_STDMET',
  source_url         TEXT,
  retrieved_at_utc   TIMESTAMP NOT NULL,
  raw_payload_hash_ref TEXT,

  PRIMARY KEY (station_id, valid_time_utc)
);
CREATE INDEX IF NOT EXISTS idx_ndbc_stdmet_time ON ndbc_stdmet_obs(valid_time_utc);
```

## Ingestion implementation

### New CLI

`python -m weather_ml.ingest.ndbc_stdmet --stations VAKF1,PEGF1 --start 2007-01-01 --end 2026-02-18 --db <dsn>`

Modes:
- `--mode backfill` downloads yearly `.txt.gz` files for each station/year.
- `--mode daily` downloads `*_5day.txt` and upserts whatever is new.

### Backfill algorithm (yearly)

For year in [start_year .. end_year]:
- For station in stations:
  1) Build URL: `.../historical/stdmet/{station_lower}h{year}.txt.gz`
  2) HTTP GET; if 404, record “missing year” and continue.
  3) Decompress gzip.
  4) Parse all rows; upsert.

**Chunk size:** yearly is correct; files are manageable and align with NDBC archival structure.

### Daily incremental algorithm

- Fetch `.../data/5day2/{STATION}_5day.txt` once per station.
- Parse all rows; upsert.
- This job can run multiple times per day; it is idempotent.

## Validation & QA

1) `valid_time_utc` must be strictly increasing within a file after parsing (warn if not).
2) Basic physical bounds:
   - wspd_ms >= 0
   - wdir_deg in [0, 360]
   - pres_hpa in [900, 1100]
3) Cross-check station count:
   - After backfill, ensure each station has at least 1 year with data.
4) Idempotency:
   - Running daily ingest twice yields 0 net new rows the second time.

## Acceptance criteria

- DB table exists and contains backfilled records for VAKF1 and PEGF1 for all available years 2007–present.
- Daily ingestion updates the last 5 days correctly.
- Parser correctly handles both 1‑header and 2‑header formats and `MM` missing values.
- Unit tests cover:
  - gzip parsing
  - header mapping
  - missing values conversion

## Notes for downstream “as‑of” usage

For day T features (asof = T‑1 12Z), the feature builder must only use rows with `valid_time_utc <= asof_utc − 30 minutes` and should compute:
- last observed marine wind vector
- 6h/12h mean wind and pressure trend
- marine vs airport wind direction difference (sea breeze cue)
