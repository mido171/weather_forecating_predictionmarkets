# HKG Tmax Data Acquisition Status

## Scope

Weather-data acquisition only for official HKO Headquarters (`HKO`) daily maximum
air temperature forecasting. Polymarket work is excluded. No predictive
modelling, machine learning, feature selection, or backtesting has been started.

## Current State

- current committed baseline: `14bc7f0`
- worktree state: acquisition reset implementation, reports, and documentation in progress
- data root: `C:\hkg_tmax_data`
- current environment state: outbound HTTP and `C:\hkg_tmax_data` writes worked during this reset
- acquisition data-root path length: 16 characters
- filesystem: NTFS
- Windows collector task: installed and enabled; current Task Scheduler state `Ready`
- raw archive model: content-addressed raw objects, HTTP metadata sidecars, append-only retrieval ledger, file manifest, and dataset lineage

## Completed This Work Unit

- read and followed the uploaded HKG Tmax remaining acquisition goal with Polymarket/model/backtest/ML excluded;
- verified `.env` already exists from `.env.example`;
- expanded weather-only live collection schedules for HKO observations, forecasts, warnings, ARWF/current, radar/lightning/nowcast, satellite/current images, TC, marine, Daily Extract refresh, upper-air refresh, and NCEP current subsets;
- patched Windows Task Scheduler scripts and installed the `HKG-Tmax-Collector` task;
- downloaded/verified every currently runnable public weather family in scope without blind re-downloads, using `--skip-existing-successes` where appropriate;
- resumed HKO current satellite acquisition and archived all currently resolvable preflighted H8/FY4B/GK2B/MODIS products from the live batch;
- reran HKO radar/lightning current acquisition after a stale scheduler partial state; manual rerun ended with `requested=86`, `succeeded=80`, `skipped=6`, `failed=0`;
- downloaded NOAA NCEP NOMADS GFS/GEFS Hong Kong regional server-side subsets for the current operational cycle already in policy scope;
- collected direct HKO mutable feeds for temperature, max/min, humidity, pressure, wind, solar radiation, UV, rainfall, visibility, current weather, forecasts, warnings, tips, nowcast, lightning, TC, marine, and tide;
- generated deterministic acquisition-readiness static context outputs: station registry, station distance/bearing matrix, and 2026 solar geometry table;
- added historical/live pair contracts and gridded acquisition policy configs, including credential and byte-budget gates;
- generated acquisition reports, coverage reports, blocker reports, health reports, source inventory, machine catalogs, and historical/live pair metadata;
- rebuilt representative bronze tables for acquired HKO climate/current/forecast families for acquisition QA only;
- reran the raw archive audit after the latest downloads.

## Ledger Coverage

- retrieval ledger: `C:\hkg_tmax_data\manifests\retrieval_ledger.csv`
- retrieval attempts: 8,432
- successful retrieval attempts: 8,431
- failed retrieval attempts: 1
- logical source IDs observed in the ledger: 235
- successful raw bytes archived: 4,399,970,389
- successful unique content hashes: 8,018
- file manifest rows: 8,018
- dataset lineage rows: 8,431

## Source Families Downloaded

| Family | Status | Coverage |
|---|---|---|
| A HKO target labels/daily climate | downloaded | Daily Extract HTML shell; `hko.xml`; Daily Extract annual payloads 1884-2026; Daily Extract monthly payloads 202601-202606; 21 full-history HKO D1 daily climate element payloads |
| B station/catalog metadata | downloaded initial | HKO station page, open-data catalog, API documentation, and ARWF AWS/RMN station config scripts |
| C high-frequency HKO regional observations | partial with historical backfill | DATA.GOV.HK historical ZIP archives for temperature/max-min/humidity/UV from 20200601-20260618 and pressure/wind/solar from 20210601-20260618; rainfall/visibility/RHR older histories not found |
| D official HKO forecasts/warnings | partial with historical RSS | latest HKO JSON feeds, DATA.GOV.HK historical RSS forecast/warning archives, and ARWF current station/grid payloads |
| E operational NWP/AI forecasts | partial current NCEP | NCEP GFS/GEFS Hong Kong regional GRIB2 subsets for 20260619; ECMWF/DWD/full-history model bulk remains byte-budget/source-policy blocked |
| F upper-air | downloaded | NOAA IGRA2 HKM00045004 period-of-record and year-to-date archives plus station/product docs |
| G radar/rainfall nowcasts/lightning | downloaded initial | HKO radar page, manifests/current radar frames, lightning pages/counts, gridded rainfall nowcast, ARWF rainfall/GeoJSON nowcast tarballs, and radar KML overlays |
| H satellite/cloud/aerosol | partial current archived | HKO satellite page/manifests, frontend path-rule scripts, resolvable MODIS true-colour/SST images, and current H8/FY4B/GK2B frames; archive-scale historical Himawari remains byte-budgeted |
| I tropical cyclone/monsoon/synoptic | partial with regional surface archive | realtime HKO TC track list, HKO best-track CSVs 1985-2024, and NOAA ISD nearby station-year files across 36 station histories back to 1945/1946 where available |
| J marine/ocean | downloaded initial | South China coastal waters bulletin, latest tide feed, and HKO daily sea-temperature climate records |
| K reanalysis | credential/byte-budget blocked | ERA5/ERA5-Land require CDS credentials and explicit retrospective subset/release-lag policy |
| L static geospatial | partial derived context | 60 official static-context raw objects plus derived station registry, distance/bearing, and solar geometry; terrain/coastline/LUHK station parsers remain |
| M frontier context | deferred | P2 inventory work must not delay incomplete P0/P1 acquisition |

## QC Failures and Repairs

- `hko_satellite_modis_aod_image` failed once with HTTP 404 for a filename listed in the HKO MODIS AOD manifest; the downloader now preserves manifest evidence and skips non-2xx image URLs after bounded preflight.
- A timed-out satellite run left overlapping Python processes and a stale manifest lock; acquisition processes were stopped, the lock was removed only after verifying no acquisition writer was active, and the batch was rerun cleanly.
- Windows scheduled-task install initially failed on trigger/repetition settings; scripts were patched and the task installed.
- Health reporting initially showed stale batch counts after manual reruns; the health report now lets newer append-only ledger successes supersede stale scheduler state.
- Raw archive audit status: `PASS`; 8,431 successful ledger rows, 8,018 unique hashes, all audited object hashes/lengths matched, all successful rows had HTTP metadata sidecars, and file-manifest/dataset-lineage coverage was complete.

## Reports Generated

- `reports/source_inventory.md`
- `reports/acquisition_progress_snapshot.md`
- `reports/raw_archive_audit.md`
- `reports/source_family_coverage.md`
- `reports/station_weather_coverage.md`
- `reports/historical_live_pairing.md`
- `reports/nwp_cycle_coverage.md`
- `reports/satellite_coverage.md`
- `reports/gridded_precipitation_coverage.md`
- `reports/source_blockers.md`
- `reports/acquisition_remaining_blockers.md`
- `reports/official_request_gaps.md`
- `reports/live_collector_health.md`
- `reports/static_context_derived.md`

## Blockers Requiring User Action Or Separate Approval

- Approve byte budget/source policy before expanding full historical/continuous GFS, GEFS, ECMWF, DWD ICON/ICON-EPS, AI forecasts, or other global/regional model archives.
- Provide CDS credentials and approve retrospective-only subsets before ERA5/ERA5-Land acquisition.
- Provide Earthdata/NASA access and approve regional subsets before GPM IMERG historical/live acquisition.
- Provide CAMS ADS credentials/terms and approve subsets before CAMS aerosol/haze acquisition.
- Approve OISST/SST product, domain, cadence, date range, and byte budget before bulk ocean/SST acquisition.
- Approve historical Himawari/archive-scale satellite crop/downsampling and byte budget before bulk acquisition.
- Treat HKO rainfall/visibility/RHR older histories and historical JSON forecast versions as official-request gaps unless another lawful archive is found.
- Resolve official HK EPD air-quality endpoints and station metadata before acquisition.
- Parse terrain/coastline/LUHK static source packages into station context; full PlanD 3D tile payloads need separate byte-budget/source-use approval.

## Next Ten Tasks

1. Parse Daily Extract, HKO D1 daily climate, and DATA.GOV.HK historical live-feed ZIPs into source-native bronze tables.
2. Parse live HKO observations and ARWF latest AWS/RMN/lightning files into station-row bronze tables with source/retrieval timestamps.
3. Parse HKO JSON latest forecasts plus DATA.GOV.HK RSS and ARWF forecast archives into vintage-safe bronze tables.
4. Parse NCEP GFS/GEFS GRIB2 subsets into cycle/member/lead/variable coverage matrices.
5. Parse NOAA ISD annual gzip files and NOAA IGRA HKM00045004 upper-air archives into bronze observation tables.
6. Monitor the enabled Windows collector through `reports/live_collector_health.md`.
7. Implement OISST, Himawari archive, GPM IMERG, ERA5/ERA5-Land, CAMS, ECMWF/DWD dry-run request builders with byte estimates and credential gates.
8. Generate the official HKO request package for unavailable pre-2020/sub-daily HKO station histories and unavailable historical JSON forecast/weather feeds.
9. Parse archived static geospatial sources into terrain, coastline/land-water, and LUHK station-context metadata.
10. Keep G1/model work gated until raw acquisition, contracts, parser validation, and coverage reports are current.
