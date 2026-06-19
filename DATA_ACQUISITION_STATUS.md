# HKG Tmax Data Acquisition Status

## Scope

Weather-data acquisition only for official HKO Headquarters (`HKO`) daily maximum
air temperature forecasting. Polymarket work is excluded. No predictive modelling,
machine learning, or backtesting has been started.

## Current State

- current commit before this acquisition expansion: `a4bd6a2`
- worktree state: acquisition expansion and documentation in progress
- data root: `C:\hkg_tmax_data`
- repository data path length: 146 characters
- acquisition data-root path length: 16 characters
- long paths enabled: false on this machine
- filesystem: NTFS

## Completed This Work Unit

- kept the immutable content-addressed raw archive and append-only ledgers;
- added an HKO/IGRA backfill command for source-family acquisition batches;
- added source-specific raw extension and metadata support for acquisition writes;
- downloaded the HKO Daily Extract HTML shell and full backing coverage currently exposed by HKO;
- downloaded all resolved HKO daily climate D1 full-history payloads;
- downloaded resolved HKO live observation, forecast, warning, nowcast, marine, and TC feeds;
- downloaded NOAA IGRA2 King's Park/Kowloon upper-air archives and documentation;
- downloaded HKO current radar manifest/frames and lightning feeds/pages;
- downloaded HKO satellite manifests and resolvable MODIS true-colour/SST images;
- documented large/credential/source-selection blockers instead of silently skipping them.

## Ledger Coverage

- retrieval ledger: `C:\hkg_tmax_data\manifests\retrieval_ledger.csv`
- retrieval attempts: 443
- successful retrieval attempts: 442
- failed retrieval attempts: 1
- logical source IDs observed in the ledger: 72
- successful raw bytes archived: 103,660,702
- successful unique content hashes: 378

## Source Families Downloaded

| Family | Status | Coverage |
|---|---|---|
| A HKO target labels/daily climate | downloaded | Daily Extract HTML shell; `hko.xml`; Daily Extract annual payloads 1884-2026; Daily Extract monthly payloads 202601-202606; 21 full-history HKO D1 daily climate element payloads |
| B station/catalog metadata | downloaded initial | HKO station page, open-data catalog, API documentation |
| C high-frequency HKO regional observations | downloaded initial | latest temperature, since-midnight max/min, humidity, pressure, wind, solar, UV, rainfall, visibility, current weather report |
| D official HKO forecasts/warnings | downloaded initial | local forecast, nine-day forecast, warning summary, warning detail, special weather tips |
| E operational NWP/AI forecasts | blocked by byte budget | GFS/GEFS/ECMWF/ICON GRIB cycles require subset, lead/member policy, and storage approval before download |
| F upper-air | downloaded | NOAA IGRA2 HKM00045004 period-of-record and year-to-date archives plus station/product docs |
| G radar/rainfall nowcasts/lightning | downloaded initial | HKO radar page, current radar manifest, 80 radar frames, lightning pages/count feed, gridded rainfall nowcast |
| H satellite/cloud/aerosol | partial with blocker | HKO satellite page/manifests and resolvable MODIS true-colour/SST images; one HKO MODIS AOD manifest image returned 404 |
| I tropical cyclone/monsoon/synoptic | downloaded initial | realtime HKO TC track list and HKO best-track CSVs 1985-2024; best-track is retrospective only |
| J marine/ocean | downloaded initial | South China coastal waters bulletin, latest tide feed, and HKO daily sea-temperature climate records |
| K reanalysis | credential/byte-budget blocked | ERA5/ERA5-Land require CDS credentials and explicit retrospective subset plan |
| L static geospatial | source-selection blocked | terrain/coastline/land-cover source, license, version, and raster budget still need selection |
| M frontier context | deferred | explicitly not allowed to delay P0/P1 acquisition |

## QC Failures and Repairs

- `hko_satellite_modis_aod_image` failed once with HTTP 404 for a filename listed in the HKO MODIS AOD manifest.
- The downloader was patched to preflight manifest-derived satellite image URLs and skip non-2xx image URLs while still archiving the manifest evidence.
- This is documented as a provider/path-resolution blocker, not counted as completed AOD image coverage.
- Content-addressed storage still deduplicates identical payloads and appends retrieval-ledger rows without overwriting raw objects.

## Live Collectors Installed/Running

Not installed yet. `config/collector_schedules.yaml` now includes the newly
resolved HKO feeds, but Windows Task Scheduler registration still requires
operator execution of `scripts/install_windows_collectors.ps1`.

## Blockers Requiring User Action

- Approve an explicit byte budget and subset policy before any operational NWP bulk GRIB download.
- Provide CDS credentials and approve a retrospective-only subset before ERA5/ERA5-Land acquisition.
- Decide whether gridded SST/ocean products such as NOAA OISST/OSTIA are worth a byte-budgeted acquisition plan.
- Select official geospatial datasets and license/version contracts for terrain/coastline/land-cover.
- Decide how aggressively to pursue unresolved HKO satellite AOD/Himawari image path issues.

## Next Ten Tasks

1. Parse Daily Extract and HKO D1 daily climate payloads into source-native bronze tables.
2. Parse live HKO observations into station-row bronze tables with retrieval and source timestamps.
3. Parse official forecast, warning, nowcast, TC, and marine feeds into vintage-safe bronze tables.
4. Parse NOAA IGRA HKM00045004 upper-air archives into bronze sounding profiles.
5. Add focused source-contract docs for each newly acquired feed family.
6. Run one prospective `acquisition run-due` smoke test after parser contracts are ready.
7. Create a byte-budgeted GFS/GEFS/ECMWF/ICON acquisition plan before any NWP download.
8. Investigate DATA.GOV.HK historical file-version APIs for latest-only HKO feeds.
9. Resolve HKO satellite AOD/Himawari image URL rules or choose an alternate lawful satellite archive.
10. Keep G1/model work gated until raw acquisition, contracts, and bronze validation are complete.
