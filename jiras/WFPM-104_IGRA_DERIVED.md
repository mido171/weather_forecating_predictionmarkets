# WFPM-104 — Ingest IGRA v2.2 sounding‑derived parameters (Miami: USM00072202) into DB

**Type:** Story  
**Priority:** P1  
**Epic:** WFPM-EPIC-EXTDATA  

## Goal

Ingest NOAA/NCEI IGRA v2.2 **derived sounding parameters** for the Miami radiosonde station and persist them in DB so we can build daily stability/moisture features that materially impact KMIA next‑day Tmax distribution (cloudiness/convection cap).

## Why this matters

Even with strong NWP, Miami Tmax depends on whether convection initiates early. Sounding‑derived parameters such as:

- **PW (precipitable water)**
- **CAPE / CIN**
- **Lifted Index / Showalter / K‑index / Total Totals**
- **Inversion strength / mixed‑layer height / LCL/LFC heights**

are *direct state variables* for convection timing and thus Tmax suppression or enhancement.

## Data source (free, public)

NOAA NCEI IGRA v2.2, derived parameter files:

- IGRA readme (directory structure, “derived-por”): https://www.ncei.noaa.gov/pub/data/igra/igra2-readme.txt  
- Derived file format (fixed‑width columns, units, missing codes, update frequency): https://www.ncei.noaa.gov/pub/data/igra/derived/igra2-derived-format.txt  
- IGRA station list (Miami ID): https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt  

**Miami station identifier:** `USM00072202` (lat ~25.75, lon ~-80.38 in station list).

## Download location (exact)

Use the web-accessible mirror (works in browsers and scripts):

Base:
`https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/derived-por/`

File:
`USM00072202-drvd.txt.zip`

Full URL:
`https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/derived-por/USM00072202-drvd.txt.zip`

## File structure & parsing (must match docs exactly)

Per the IGRA derived format doc:

- Each file is a series of **multi-line sounding records**.
- Each record begins with a **header line** that starts with `#` and contains summary parameters.
- After the header line, there are `NUMLEV` **data lines**, one per pressure level.

**Missing values are coded as `-99999`.** Convert to NULL.

### A) Header record format (fixed-width columns)

From the format doc:

- `YEAR` columns 14–17
- `MONTH` 19–20
- `DAY` 22–23
- `HOUR` 25–26
- `NUMLEV` 32–36
- `PW` 38–43 (mm*100)
- `INVPRESS` 44–49 (Pa or mb*100)
- `INVHGT` 50–55 (m AGL)
- `INVTEMPDIF` 56–61 (K*10)
- `MIXPRESS` 62–67
- `MIXHGT` 68–73
- `FRZPRESS` 74–79
- `FRZHGT` 80–85
- `LCLPRESS` 86–91
- `LCLHGT` 92–97
- `LFCPRESS` 98–103
- `LFCHGT` 104–109
- `LNBPRESS` 110–115
- `LNBHGT` 116–121
- `LI` 122–127 (°C, stored as integer)
- `SI` 128–133
- `KI` 134–139
- `TTI` 140–145
- `CAPE` 146–151 (J/kg)
- `CIN` 152–157 (J/kg)

### B) Data record format (pressure-level lines)

From the format doc:

- `PRESS` 1–7 (Pa or mb*100)
- `REPGPH` 9–15 (m)
- `CALCGPH` 17–23 (m)
- `TEMP` 25–31 (K*10)
- `TEMPGRAD` 33–39 ((K/km)*10)
- `PTEMP` 41–47 (K*10)
- `PTEMPGRAD` 49–55
- `VTEMP` 57–63 (K*10)
- `VPTEMP` 65–71 (K*10)
- `VAPPRESS` 73–79 (mb*1000)
- `SATVAP` 81–87 (mb*1000)
- `REPRH` 89–95 (%*10)
- `CALCRH` 97–103 (%*10)
- `RHGRAD` 105–111 ((%/km)*10)
- `UWND` 113–119 ((m/s)*10)
- `UWDGRAD` 121–127
- `VWND` 129–135 ((m/s)*10)
- `VWNDGRAD` 137–143
- `N` 145–151 (count/flag; keep raw int)

## Normalization (do this)

Store both raw integer fields and convenient floats:

- Pressure fields: `press_mb = press_raw / 100.0`
- Temperatures: `temp_c = temp_raw / 10.0 - 273.15`
- RH: `rh_pct = rh_raw / 10.0`
- Winds: `u_ms = uwnd_raw / 10.0`, `v_ms = vwnd_raw / 10.0`
- PW: `pw_mm = pw_raw / 100.0`
- Inversion temp diff: `invtempdiff_k = invtempdiff_raw / 10.0`

**Do not drop the raw integer columns.** If NOAA changes scaling, we still have the original.

## Database schema

### Table 1 — Sounding header summary parameters

```sql
CREATE TABLE IF NOT EXISTS igra2_derived_header (
  station_id          TEXT NOT NULL,
  sounding_time_utc   TIMESTAMP NOT NULL,  -- YEAR/MONTH/DAY/HOUR

  numlev              INTEGER,

  -- key summary signals (raw ints)
  pw_raw              INTEGER,
  invpress_raw        INTEGER,
  invhgt_m            INTEGER,
  invtempdiff_raw     INTEGER,
  mixpress_raw        INTEGER,
  mixhgt_m            INTEGER,
  frzpress_raw        INTEGER,
  frzhgt_m            INTEGER,
  lclpress_raw        INTEGER,
  lclhgt_m            INTEGER,
  lfcpress_raw        INTEGER,
  lfchgt_m            INTEGER,
  lnbpress_raw        INTEGER,
  lnbhgt_m            INTEGER,
  li_raw              INTEGER,
  si_raw              INTEGER,
  ki_raw              INTEGER,
  tti_raw             INTEGER,
  cape_raw            INTEGER,
  cin_raw             INTEGER,

  -- scaled floats (nullable)
  pw_mm               REAL,
  invpress_mb         REAL,
  invtempdiff_k       REAL,
  mixpress_mb         REAL,
  frzpress_mb         REAL,
  lclpress_mb         REAL,
  lfcpress_mb         REAL,
  lnbpress_mb         REAL,

  source              TEXT NOT NULL DEFAULT 'IGRA2_DERIVED',
  source_url          TEXT,
  retrieved_at_utc    TIMESTAMP NOT NULL,
  raw_payload_hash_ref TEXT,

  PRIMARY KEY (station_id, sounding_time_utc)
);
CREATE INDEX IF NOT EXISTS idx_igra2_hdr_time ON igra2_derived_header(sounding_time_utc);
```

### Table 2 — Pressure-level records (optional but recommended)

This table is “long” but only for one station; it is manageable and enables richer features later.

```sql
CREATE TABLE IF NOT EXISTS igra2_derived_level (
  station_id          TEXT NOT NULL,
  sounding_time_utc   TIMESTAMP NOT NULL,
  press_raw           INTEGER NOT NULL,   -- mb*100
  press_mb            REAL,

  repgph_m            INTEGER,
  calcgph_m           INTEGER,

  temp_raw            INTEGER,
  temp_c              REAL,
  tempgrad_raw        INTEGER,

  ptemp_raw           INTEGER,
  ptemp_c             REAL,
  ptempgrad_raw       INTEGER,

  vtemp_raw           INTEGER,
  vtemp_c             REAL,
  vptemp_raw          INTEGER,
  vptemp_c            REAL,

  vap_press_raw       INTEGER,
  sat_vap_raw         INTEGER,

  reprh_raw           INTEGER,
  reprh_pct           REAL,
  calcrh_raw          INTEGER,
  calcrh_pct          REAL,
  rhgrad_raw          INTEGER,

  uwnd_raw            INTEGER,
  u_ms                REAL,
  vwnd_raw            INTEGER,
  v_ms                REAL,

  uwdgrad_raw         INTEGER,
  vwdgrad_raw         INTEGER,

  n_raw               INTEGER,

  source              TEXT NOT NULL DEFAULT 'IGRA2_DERIVED',
  retrieved_at_utc    TIMESTAMP NOT NULL,
  raw_payload_hash_ref TEXT,

  PRIMARY KEY (station_id, sounding_time_utc, press_raw)
);
CREATE INDEX IF NOT EXISTS idx_igra2_lvl_time ON igra2_derived_level(sounding_time_utc);
```

## Ingestion implementation

### CLI

`python -m weather_ml.ingest.igra2_derived --station USM00072202 --db <dsn>`

Modes:
- `--mode refresh_full` (download ZIP and fully upsert; station file is one ZIP, so a “full refresh” is acceptable)
- `--mode incremental` (optional: if you later store last seen sounding_time_utc; but not required for v1)

### Algorithm (refresh_full)

1) Download the ZIP file (timeout 120s, retry 5x).
2) Compute `raw_payload_hash_ref = sha256(zip_bytes)`.
3) Open ZIP in memory; read the `*-drvd.txt` file.
4) Stream lines:
   - When a line starts with `#`, parse header fields; compute `sounding_time_utc`.
   - Read next `NUMLEV` lines; parse each as a level record.
5) Upsert header and level records.
6) Record `retrieved_at_utc` once per run; store `source_url`.

## Validation & QA

- The number of parsed level lines must equal `NUMLEV` for every header (fail if not).
- Sounding timestamps must be unique per station in header table.
- Missing values `-99999` become NULL; verify with unit tests.
- Spot-check conversions:
  - TEMP raw `3000` → `26.85°C`
  - RH raw `750` → `75.0%`
  - UWND raw `50` → `5.0 m/s`

## Acceptance criteria

- DB has header + level tables populated for USM00072202.
- Parser passes unit tests for at least 3 synthetic sample soundings.
- Re-running the ingest is idempotent (0 duplicates, updates allowed).

## Notes for downstream “as‑of” usage

The IGRA derived parameters are “updated once a day in the early morning Eastern Time” (per format doc). Therefore, feature generation must enforce a conservative latency:
- For target day T (decision time T‑1 12Z), use the most recent sounding with `sounding_time_utc <= asof_utc − 12h`.  
- If missing, fall back to `<= asof_utc − 36h`.
