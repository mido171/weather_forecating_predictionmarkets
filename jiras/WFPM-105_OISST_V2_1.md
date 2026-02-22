# WFPM-105 — Ingest OISST v2.1 daily SST/anomaly near South Florida into DB

**Type:** Story  
**Priority:** P2  
**Epic:** WFPM-EPIC-EXTDATA  

## Goal

Ingest NOAA OISST v2.1 daily sea surface temperature (SST) and anomaly for a small ocean box near Miami and persist it as a compact daily time series in DB. This provides a stable boundary-condition feature that improves regime awareness (warm/cool ocean phases that modulate sea‑breeze cooling and humidity).

## Why SST matters for KMIA Tmax

SST is slow-moving but impactful:
- Warmer SST reduces coastal cooling potential (warmer marine air advects inland).
- SST anomaly shifts humidity and stability, affecting cloudiness and convective timing.
- SST acts like a “hidden state” that NWP can misrepresent locally; adding it reduces systematic seasonal bias.

## Data source (free, public)

NOAA/NCEI THREDDS catalog for OISST Base netCDF files:

- Catalog root: https://www.ncei.noaa.gov/thredds/catalog/OisstBase/NetCDF/V2.1/AVHRR/catalog.html
- Monthly subfolders contain daily files named like `oisst-avhrr-v02r01.YYYYMMDD.nc` (visible in catalog listings).

Operational latency: OISST is produced with ~24 hour latency (as stated in the PO.DAAC release notes), so we must apply a lag in “as‑of” feature usage.

## What exactly to ingest (keep it small!)

We will NOT store the global grid. We store only a small ocean box summary per day.

### Spatial box (v1)

Use this region (covers waters east/southeast of Miami and is robust to grid alignment):

- Latitude: 24.5°N to 26.5°N
- Longitude: -81.0°W to -79.0°W

**Important:** OISST uses longitude in **degrees_east (0–360)** in netCDF.
Convert:
- -81 → 279
- -79 → 281

So use:
- lon_east: 279.0 to 281.0

### Variables

From OISST netCDF:
- `sst` (°C, scale_factor 0.01, _FillValue -999)
- `anom` (°C, scale_factor 0.01, _FillValue -999)

Optionally store:
- `ice` and `err` if present (not required for Tmax, but harmless).

## Endpoints and file naming

Monthly directory pattern (from THREDDS catalog):

`https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/V2.1/AVHRR/{YYYYMM}/oisst-avhrr-v02r01.{YYYYMMDD}.nc`

Example (Jan 1, 2025):
- Folder: `202501`
- File: `oisst-avhrr-v02r01.20250101.nc`

So full fileServer URL is:

`https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/V2.1/AVHRR/202501/oisst-avhrr-v02r01.20250101.nc`

## Parsing & normalization rules

1) Download netCDF bytes (timeout 120s, retry 5x).
2) Open with `xarray` (engine `netcdf4` or `h5netcdf`).
3) Select the spatial box:
   - lat slice [24.5, 26.5]
   - lon slice [279.0, 281.0]
4) Decode scale_factor automatically via xarray (it will if `decode_cf=True`).
5) Convert fill values to NaN (xarray should handle; if not, replace -999 with NaN before stats).
6) Compute per-day summary stats:
   - `sst_mean_c`, `sst_std_c`, `sst_min_c`, `sst_max_c`
   - `anom_mean_c`, `anom_std_c`
   - `n_valid` (count of non-null cells)

## Database schema

```sql
CREATE TABLE IF NOT EXISTS oisst_box_daily (
  date_utc         DATE NOT NULL,      -- the SST day (dataset time coordinate)
  box_name         TEXT NOT NULL,       -- e.g., 'MIA_E_24.5_26.5_279_281'

  sst_mean_c       REAL,
  sst_std_c        REAL,
  sst_min_c        REAL,
  sst_max_c        REAL,

  anom_mean_c      REAL,
  anom_std_c       REAL,

  n_valid          INTEGER,

  source           TEXT NOT NULL DEFAULT 'OISST_V2_1',
  source_url       TEXT,
  retrieved_at_utc TIMESTAMP NOT NULL,
  raw_payload_hash_ref TEXT,

  PRIMARY KEY (date_utc, box_name)
);
CREATE INDEX IF NOT EXISTS idx_oisst_date ON oisst_box_daily(date_utc);
```

## Ingestion implementation

### CLI

`python -m weather_ml.ingest.oisst_box --start 2016-01-01 --end 2026-02-18 --db <dsn>`

Flags:
- `--box` (defaults to the Miami box above)
- `--parallel` (optional; default 4 workers)
- `--max-retries` 5
- `--sleep-seconds` 0.25 (polite)

### Algorithm

For each date in [start, end):
1) Compute `YYYYMM` and `YYYYMMDD`.
2) Build fileServer URL.
3) Download netCDF; if 404, skip (some days may be absent).
4) Compute box stats; upsert one row.

### Daily incremental update

Run daily at ~02:00Z:
- Fetch the last 7 days (to capture late-available or corrected files), upsert.

## Validation & QA

- `n_valid > 0` for most days; warn if 0.
- SST should be physically plausible:
  - `sst_mean_c` in [10, 40] for South Florida
- Idempotency: re-run yields same stats and no duplicates.

## Acceptance criteria

- Backfill from 2016‑01‑01 (or configured start) succeeds and produces a continuous daily DB table for most days.
- Daily incremental job is idempotent and keeps table current.
- Unit tests cover:
  - date → URL mapping
  - lon conversion to degrees_east
  - fill value handling

## Notes for downstream “as‑of” usage

Because OISST has ~24h latency and may have short-term revisions, enforce a lag in feature construction:
- For target day T (decision time T‑1 12Z), use SST through **T‑2** (i.e., `date_utc <= T−2`).
- Prefer rolling means (7‑day, 14‑day) ending at T‑2 to reduce any revision sensitivity.
