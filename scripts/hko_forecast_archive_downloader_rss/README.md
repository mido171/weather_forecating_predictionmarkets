# HKO Historical Forecast Archive Downloader

This project downloads and normalizes historical **issued Hong Kong Observatory forecast bulletins**.

It covers four complementary sources:

1. **Official HKSAR Press Weather archive** — primary source.
   - Daily index pattern: `https://www.info.gov.hk/gia/wr/YYYYMM/DD.htm`
   - Downloads Local Weather Forecast and 5-/7-/9-Day Weather Forecast bulletins.
2. **HKUST ENVF mirror of historical HKO webpages** — gap-filling and independent verification.
   - Research use only unless separate permission is obtained.
   - Default mode samples snapshots around the 15:00 HKT T-24 cutoff instead of hammering every historical hour.
3. **Internet Archive Wayback Machine** — gap-filling for retired HKO forecast URLs.
4. **DATA.GOV.HK historical archive** — official RSS overlap from the dates actually available there.

The downloader is resumable, immutable, hash-deduplicated, rate-limited, and writes:
- raw HTML/XML;
- HTTP metadata sidecars;
- a SQLite retrieval and parsing ledger;
- normalized bulletin records;
- normalized daily forecast rows;
- coverage and failure reports.

## Important usage rule

The HKUST archive states that its information is for **education and research only** and that commercial or publication use requires approval. Use HKUST data for the Lund University research project, but do not silently make it a production trading dependency without permission.

## Recommended Windows installation

Open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install.ps1
```

The default data location is:

```text
C:\hkg_tmax_data\historical_hko_forecasts
```

You can override it with `--data-root`.

## Recommended acquisition sequence

### Step 1 — Inventory official HKSAR weather index pages

This fetches only one index page per day and discovers all relevant forecast links:

```powershell
.\run_01_official_index.ps1
```

Default range:

```text
2000-01-01 through 2020-05-31
```

The command is resumable. Re-running it skips successful URLs.

### Step 2 — Download official forecast bulletin pages

```powershell
.\run_02_official_bulletins.ps1
```

This downloads only:

- Local Weather Forecast
- 5-Day Weather Forecast
- 7-Day Weather Forecast
- 9-Day Weather Forecast

It parses issue times, forecast text, target dates, minimum/maximum temperatures, wind, weather descriptions, and relative-humidity ranges where present.

### Step 3 — Download HKUST T-24 cutoff snapshots

```powershell
.\run_03_hkust_cutoff.ps1
```

The default samples `06:00` and `07:00 UTC` for every day, corresponding to approximately `14:00` and `15:00 HKT`. This is the highest-value archive for reconstructing what was visible near the operational T-24 cutoff, while avoiding a 150,000-request hourly crawl.

Default range:

```text
2002-08-01 through 2020-05-31
```

Every HKUST page is parsed using its **embedded HKO bulletin issue timestamp**, not merely the snapshot filename. Stale bundled pages are flagged.

### Step 4 — Query and download Wayback captures

```powershell
.\run_04_wayback.ps1
```

This queries the CDX API for known retired HKO Local Weather Forecast and 5-/7-/9-Day forecast paths, deduplicates captures by digest, and downloads raw archived HTML.

### Step 5 — Download DATA.GOV.HK official RSS overlap

```powershell
.\run_05_data_gov_overlap.ps1
```

This uses the official historical-archive API for:

- `LocalWeatherForecast.xml`
- `SeveralDaysWeatherForecast.xml`

It does not invent coverage before the official archive begins.

### Step 6 — Build normalized exports and reports

```powershell
.\run_06_export_audit.ps1
```

Outputs include:

```text
bronze/forecast_bulletins.jsonl
bronze/forecast_days.jsonl
bronze/forecast_bulletins.csv
bronze/forecast_days.csv
reports/coverage_by_source_product_year.csv
reports/coverage_by_issue_date.csv
reports/failed_requests.csv
reports/parse_failures.csv
reports/stale_hkust_snapshots.csv
reports/candidate_link_counts.csv
metadata/archive.sqlite3
```

## One-command research backfill

After confirming the polite request intervals in the PowerShell files:

```powershell
.\run_all.ps1
```

The official bulletin stage can take many hours or days because the archive may contain many Local Weather Forecast revisions each day. It is safe to stop and restart.

## Direct CLI examples

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Inventory official archive:

```powershell
python .\hko_archive.py official-index `
  --start 2000-01-01 `
  --end 2020-05-31 `
  --data-root C:\hkg_tmax_data\historical_hko_forecasts `
  --delay-seconds 1.0
```

Download official forecast pages:

```powershell
python .\hko_archive.py official-details `
  --data-root C:\hkg_tmax_data\historical_hko_forecasts `
  --types local,5day,7day,9day `
  --delay-seconds 1.25
```

Download HKUST snapshots at 14:00 and 15:00 HKT:

```powershell
python .\hko_archive.py hkust `
  --start 2002-08-01 `
  --end 2020-05-31 `
  --hours-utc 6,7 `
  --data-root C:\hkg_tmax_data\historical_hko_forecasts `
  --delay-seconds 1.5 `
  --acknowledge-research-only
```

Full hourly HKUST crawl is deliberately not the default. It would be very large and should only be run after receiving permission:

```powershell
python .\hko_archive.py hkust `
  --start 2002-08-01 `
  --end 2020-05-31 `
  --hours-utc 0-23 `
  --delay-seconds 2.0 `
  --acknowledge-research-only `
  --acknowledge-large-crawl
```

## Targeting T-24 forecasts

For target day `T`, the operational cutoff is:

```text
15:00:00 Asia/Hong_Kong on T-1
```

Do not select a forecast because its file date looks correct. Select the latest bulletin satisfying:

```text
issue_at_hkt <= cutoff_hkt
```

and whose normalized valid date includes `T`.

The exports retain:

- `issue_at_hkt`
- `snapshot_at_hkt`
- `target_date`
- `target_date_confidence`
- `stale_snapshot_flag`
- `source`
- `source_url`
- `raw_sha256`

## Safety and integrity

- Raw files are never modified in place.
- Every payload is SHA-256 hashed.
- Every fetch has an HTTP metadata sidecar.
- The SQLite ledger makes all stages resumable.
- 404s are recorded and not retried indefinitely.
- 429 and 5xx responses use exponential backoff.
- Identical payloads are stored only once per source.
- Parsing failures remain visible in reports.
- The downloader never treats observed climate data as historical forecast data.
