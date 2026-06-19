# HKG Tmax Data Acquisition Status

## Scope

Weather-data acquisition only for official HKO Headquarters (`HKO`) daily maximum
air temperature forecasting. Polymarket work is excluded. No predictive modelling,
machine learning, feature selection, or backtesting has been started.

## Current State

- current commit before this static-context expansion: `276c87f`
- worktree state: static-context acquisition expansion and documentation in progress
- data root: `C:\hkg_tmax_data`
- repository data path length: 146 characters
- acquisition data-root path length: 16 characters
- long paths enabled: false on this machine
- filesystem: NTFS
- accidental repo-local content-addressed acquisition root was imported into `C:\hkg_tmax_data` and removed from `data/raw/objects`, `data/manifests`, and `data/metadata`

## Completed This Work Unit

- added DATA.GOV.HK historical archive discovery/download support for HKO live feed ZIP archives;
- downloaded DATA.GOV.HK historical archives for 7 high-frequency HKO regional feeds through `20260618`;
- added and ran an HKO ARWF portal batch, including station config, parser/config scripts, latest observations, nowcast tarballs, radar KML overlays, animation CSVs, and 53 valid station/grid forecast payloads;
- downloaded NOAA ISD metadata/docs and 951 annual station gzip files for all station-history rows in the Hong Kong/Pearl River Delta bounding box with available annual files;
- added and ran a NOAA NCEP NOMADS GFS/GEFS server-side regional subset batch for the latest complete `20260619 00Z` cycle through `f120`;
- downloaded DATA.GOV.HK historical RSS forecast/warning archives for current weather, local forecast, 9-day forecast, warning bulletin, and warning summary in English, Traditional Chinese, and Simplified Chinese;
- added a verified `static-context-current` batch and downloaded 60 official LandsD/CSDI/PlanD/DATA.GOV.HK static-context raw objects totaling 215,229,122 bytes;
- downloaded official terrain/elevation context, topographic GML and GeoTIFF products, iGeoCommunity packages, i-Series revision metadata, LUHK 10 m land-utilization raster packages for 2018-2024, LUHK 2022-2024 statistics, and PlanD 3D model tile indexes;
- added `truststore`-backed TLS verification for Python HTTP clients so official Hong Kong Map Service endpoints validate against the Windows trust store without disabling TLS verification;
- preserved immutable raw bytes, hashes, HTTP metadata sidecars, append-only ledger rows, and file/dataset manifests in `C:\hkg_tmax_data`;
- kept full global/full-history NWP, reanalysis, gridded SST, unresolved satellite AOD/Himawari, derived static matrices, and full 3D model tile payloads classified as blockers where evidence requires user approval, credentials, source-use decisions, or byte budgets.

## Ledger Coverage

- retrieval ledger: `C:\hkg_tmax_data\manifests\retrieval_ledger.csv`
- retrieval attempts: 4,141
- successful retrieval attempts: 4,140
- failed retrieval attempts: 1
- logical source IDs observed in the ledger: 216
- successful raw bytes archived: 2,853,645,750
- successful unique content hashes: 4,075

## Source Families Downloaded

| Family | Status | Coverage |
|---|---|---|
| A HKO target labels/daily climate | downloaded | Daily Extract HTML shell; `hko.xml`; Daily Extract annual payloads 1884-2026; Daily Extract monthly payloads 202601-202606; 21 full-history HKO D1 daily climate element payloads |
| B station/catalog metadata | downloaded initial | HKO station page, open-data catalog, API documentation; ARWF AWS/RMN station config scripts |
| C high-frequency HKO regional observations | partial with historical backfill | latest temperature, since-midnight max/min, humidity, pressure, wind, solar, UV, rainfall, visibility, current weather report; DATA.GOV.HK historical ZIP archives for 7 live feeds through 20260618 |
| D official HKO forecasts/warnings | partial with historical backfill | latest HKO JSON forecasts/warnings; DATA.GOV.HK historical RSS forecast/warning archives; ARWF station/grid forecast current payloads |
| E operational NWP/AI forecasts | partial with blocker | NCEP GFS/GEFS Hong Kong regional server-side GRIB2 subset for 20260619 00Z through f120; ECMWF/DWD/full-history model bulk remains byte-budget/source-policy blocked |
| F upper-air | downloaded | NOAA IGRA2 HKM00045004 period-of-record and year-to-date archives plus station/product docs |
| G radar/rainfall nowcasts/lightning | downloaded initial | HKO radar page, current radar manifest, 80 current radar frames, lightning pages/count feed, gridded rainfall nowcast, ARWF rainfall/GeoJSON nowcast tarballs and radar KML overlays |
| H satellite/cloud/aerosol | partial with blocker | HKO satellite page/manifests and resolvable MODIS true-colour/SST images; one HKO MODIS AOD manifest image returned 404 |
| I tropical cyclone/monsoon/synoptic | partial with regional surface archive | realtime HKO TC track list and HKO best-track CSVs 1985-2024; NOAA ISD nearby surface station annual archives; best-track is retrospective only |
| J marine/ocean | downloaded initial | South China coastal waters bulletin, latest tide feed, and HKO daily sea-temperature climate records |
| K reanalysis | credential/byte-budget blocked | ERA5/ERA5-Land require CDS credentials and explicit retrospective subset plan |
| L static geospatial | partial with verified official downloads | 60 official static-context objects: LandsD 5 m DTM direct ASC zip and CSDI GeoTIFF package/docs; LandsD topographic GML/GeoTIFF packages; iGeoCom CSV/GeoJSON packages; i-Series revision CSVs; PlanD/CSDI LUHK 10 m raster packages for 2018-2024 and LUHK statistics; PlanD 3D model tile index CSVs |
| M frontier context | deferred | P2 inventory work is not allowed to delay incomplete P0/P1 acquisition |

## QC Failures and Repairs

- `hko_satellite_modis_aod_image` failed once with HTTP 404 for a filename listed in the HKO MODIS AOD manifest.
- The satellite downloader preflights manifest-derived image URLs and skips non-2xx image URLs while preserving manifest evidence.
- Direct Python acquisition commands initially wrote NCEP/RSS batches to repo-local `data/`; those 2,000 successful ledger rows and raw objects were imported into `C:\hkg_tmax_data` with hash checks, then the accidental repo-local content-addressed store was removed.
- Python/certifi TLS validation failed for `open.hkmapservice.gov.hk` while Windows trusted the endpoint; the HTTP client now uses `truststore` when available, preserving TLS verification and enabling official iGeoCom downloads.
- `LUHK2022_English_description.csv` did not resolve during preflight and was not included; the 2022 LUHK English statistics CSV itself downloaded successfully, while 2023 and 2024 description CSVs were archived.
- Content-addressed storage deduplicates identical payloads and appends retrieval-ledger rows without overwriting raw objects.
- Full raw archive integrity audit passed after the static-context batch: 4,140 success rows, 4,075 unique hashes, all content hashes/lengths matched, all sidecars had HTTP metadata, and file-manifest/dataset-lineage coverage was complete.

## Live Collectors Installed/Running

Not installed yet. `config/collector_schedules.yaml` includes resolved HKO feeds,
but Windows Task Scheduler registration still requires operator execution of
`scripts/install_windows_collectors.ps1`. NCEP, ARWF, DATA.GOV.HK historical
refresh, and NOAA ISD periodic archive refresh schedules still need collector
entries before this acquisition program can be called durable.

## Blockers Requiring User Action

- Approve an explicit byte budget and source policy for full historical/continuous GFS, GEFS, ECMWF, DWD ICON/ICON-EPS, AI forecast, and other operational model archives beyond the downloaded latest NCEP regional subset.
- Provide CDS credentials and approve a retrospective-only subset before ERA5/ERA5-Land acquisition.
- Decide whether gridded SST/ocean products such as NOAA OISST/OSTIA are worth a byte-budgeted acquisition plan.
- Generate deterministic station distance/bearing, solar geometry, terrain/slope/aspect, coastline/land-water, and station-context metadata from the archived static sources.
- Decide whether full PlanD 3D photo-realistic model tile payloads are needed; only tile indexes were downloaded because bulk tiles are byte-budget/source-use dependent.
- Decide how aggressively to pursue unresolved HKO satellite AOD/Himawari image path issues or approve an alternate lawful satellite archive.

## Next Ten Tasks

1. Parse Daily Extract, HKO D1 daily climate, and DATA.GOV.HK historical live-feed ZIPs into source-native bronze tables.
2. Parse live HKO observations and ARWF latest AWS/RMN/lightning files into station-row bronze tables with source and retrieval timestamps.
3. Parse HKO JSON latest forecasts plus DATA.GOV.HK RSS and ARWF forecast archives into vintage-safe bronze tables.
4. Parse NCEP GFS/GEFS GRIB2 subsets into cycle/member/lead/variable coverage matrices.
5. Parse NOAA ISD annual gzip files and NOAA IGRA HKM00045004 upper-air archives into bronze observation tables.
6. Install/update Windows collectors for HKO live feeds, ARWF, RSS/forecast feeds, NCEP rolling subsets, and health checks.
7. Create a byte-budgeted acquisition plan for ECMWF open data, DWD ICON/ICON-EPS, and longer historical GFS/GEFS archives.
8. Parse archived static geospatial sources and generate deterministic solar geometry, station distance/bearing, terrain, coastline/land-water, and land-utilization station-context metadata.
9. Resolve HKO satellite AOD/Himawari image URL rules or choose an alternate lawful official satellite source.
10. Keep G1/model work gated until raw acquisition, contracts, parser validation, and coverage reports are current.
