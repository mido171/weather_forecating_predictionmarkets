# HKG Tmax Data Inventory And Gaps Handoff

Generated for GPT-Pro orchestration on 2026-06-20.

Scope: weather-data acquisition for HKO/HKG Tmax only. No Polymarket, no
modelling, no ML, no backtesting.

## Snapshot

- Repository: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex`
- Canonical data root: `C:\hkg_tmax_data`
- Latest refreshed raw archive audit: `PASS`
- Retrieval ledger rows: `10,046`
- Successful retrieval rows: `10,044`
- Failed retrieval rows: `2`
- Unique successful content hashes: `9,526`
- File manifest rows: `9,526`
- Dataset lineage rows: `10,044`
- Audited content objects: `9,526`
- Audit errors: `0`

## Important Interpretation Rules

- `Downloaded` means raw immutable payloads were archived with hashes, HTTP
  metadata sidecars, file manifest rows, and dataset lineage.
- Most sources are not yet parsed into final bronze/silver tables. Several
  exact per-variable date ranges still need parser verification.
- Date ranges below are actual archived source coverage where known. For live
  feeds, the range is the acquired current/prospective archive window, not
  historical backfill.
- HKO daily climate payloads are full-history raw files. Some elements naturally
  start later than `1884`; exact per-element start/end must be confirmed by
  parsing the raw files.

## Acquired Data Summary

| Data family | Downloaded range | Current status | Notes |
|---|---|---|---|
| HKO target labels / Daily Extract / daily climate | `1884` to `2026-06` | downloaded raw | Includes Daily Extract annual/monthly payloads, CLMMAXT HKO, and HKO daily single-element climate payloads. |
| HKO station/catalog metadata | current metadata snapshots, fetched 2026-06-18/2026-06-20 | downloaded initial | HKO station metadata, HKO open-data catalog, API docs. Station history reconciliation still pending. |
| HKO high-frequency station/weather archives | feed-specific, `2020-06-30` or `2021-06-28/29` to `2026-06-18` observations | partial historical backfill | DATA.GOV.HK historical ZIP archives downloaded where public archive exists. |
| HKO latest/current observations | current/prospective snapshots from 2026-06-19 onward | live/current only | Rainfall, visibility, current weather report, temperature, max/min, humidity, pressure, wind, solar, UV. Older rainfall/visibility/RHR history not found. |
| HKO RSS forecast/warning archives | `2020-06-01` to `2026-06-18`; 9-day RSS from `2021-04-01` to `2026-06-18` | partial historical forecast backfill | RSS archives downloaded in EN/TC/SC. Historical JSON API versions not found. |
| HKO latest JSON forecasts/warnings | current/prospective snapshots from 2026-06-18/19 onward | live/current only | Local forecast, 9-day forecast, warning summary/info, special weather tips. |
| HKO ARWF station/grid forecasts and nowcasts | current/prospective snapshots from 2026-06-19 onward | live/current only | 53 valid station/grid forecast payloads plus portal/config/nowcast/radar KML assets. Older ARWF vintages not found. |
| NCEP GFS/GEFS Hong Kong regional subsets | `2026-06-19` operational cycle only, leads through `f120` | partial current NWP | Current GFS/GEFS subset downloaded. Full historical/continuous NWP not downloaded. |
| NOAA IGRA upper-air HKM00045004 | `1949` to `2026` | downloaded raw | Period-of-record and year-to-date archive downloaded. Parsing still pending. |
| HKO radar / rainfall nowcast / lightning | `2026-06-19 10:54` to `2026-06-20 06:36` | current/prospective only | Current radar frames, radar manifest, lightning pages/count feed, gridded nowcast. Historical imagery not found. |
| HKO satellite / cloud / aerosol current products | `2025-06-20 14:40` to `2026-06-19 22:30` | partial current archive | HKO current/rolling products and resolvable MODIS/H8/FY4B/GK2B frames. Historical Himawari-scale archive not downloaded. |
| NOAA ISD surrounding stations | station-dependent, overall `1945` to `2025` | downloaded raw | 951 annual station-year gzip files across 36 nearby/regional station histories. 2026 annual files not exposed in tested archive. |
| HKO tropical cyclone best tracks | `1985` to `2024` | downloaded raw | Retrospective best tracks only. Realtime TC snapshot collected prospectively. |
| Marine / tide / HKO sea-temperature files | current marine/tide snapshots from 2026-06-19 onward; HKO daily sea-temperature raw files from daily climate source | partial | Gridded SST/OISST not downloaded. Exact HKO sea-temp date span pending parser. |
| Static geospatial context | static/versioned; LUHK `2018` to `2024`; solar geometry `2026` | partial derived context | Raw DTM/topographic/LUHK/iGeoCom/PlanD index files downloaded; station registry, distance/bearing, and solar geometry generated. Terrain/coastline/LUHK station parsers pending. |

## HKO Daily Single-Element Climate Data Downloaded

These correspond to the HKO "Daily Data for Single Element" page in the
screenshot. Raw full-history payloads were downloaded for every listed element
below. Family coverage is `1884` to `2026-06`, but exact per-element start/end
must still be parsed.

| Station/domain | Elements downloaded | Raw source IDs |
|---|---|---|
| Hong Kong Observatory | Mean Temp, Max Temp, Min Temp, Pressure/MSLP, Dew Point Temp, Wet-Bulb Temp, Relative Humidity, Amount of Cloud, Rainfall, Grass Min Temp | `hko_daily_climate_mean_temperature_all`, `hko_daily_climate_maximum_temperature_all`, `hko_daily_climate_minimum_temperature_all`, `hko_daily_climate_mslp_all`, `hko_daily_climate_dew_point_all`, `hko_daily_climate_wet_bulb_all`, `hko_daily_climate_relative_humidity_all`, `hko_daily_climate_cloud_amount_all`, `hko_daily_climate_rainfall_all`, `hko_daily_climate_grass_min_temperature_all` |
| King's Park | Bright Sunshine, Global Solar Radiation, Evaporation | `hko_daily_climate_bright_sunshine_all`, `hko_daily_climate_global_solar_radiation_all`, `hko_daily_climate_evaporation_all` |
| Waglan Island | Prevailing Wind Direction, Wind Speed, Sea Temp | `hko_daily_climate_prevailing_wind_direction_all`, `hko_daily_climate_mean_wind_speed_all`, `hko_daily_climate_sea_temp_waglan_all` |
| North Point | Sea Temp a.m., Sea Temp p.m. | `hko_daily_climate_sea_temp_np_am_all`, `hko_daily_climate_sea_temp_np_pm_all` |
| Hong Kong Territory | Cloud-to-Ground Lightning, Cloud-to-Cloud Lightning | `hko_daily_climate_lightning_ground_all`, `hko_daily_climate_lightning_cloud_all` |
| Hong Kong International Airport | Reduced Visibility Hours | `hko_daily_climate_reduced_visibility_hka_all` |

## HKO High-Frequency Historical Station Archives

These are the main sub-daily station/weather archives downloaded from
DATA.GOV.HK. They do not go back before 2020/2021 because those are the oldest
public historical archive versions found for these feed families.

| Feed | Archive date range | Observation date/time range | Rows | Stations |
|---|---:|---:|---:|---:|
| 1-min temperature | `2020-06-01` to `2026-06-18` | `2020-06-30 09:00` to `2026-06-18 23:50` | 16,859,348 | 39 |
| Since-midnight max/min | `2020-06-01` to `2026-06-18` | `2020-06-30 09:00` to `2026-06-18 23:50` | 16,882,046 | 39 |
| 1-min humidity | `2020-06-01` to `2026-06-18` | `2020-06-30 09:00` to `2026-06-18 23:50` | 10,666,499 | 28 |
| 15-min UV index | `2020-06-01` to `2026-06-18` | `2020-06-30 10:15` to `2026-06-18 18:15` | 101,423 | not station-row based |
| 1-min pressure | `2021-06-01` to `2026-06-18` | `2021-06-29 10:10` to `2026-06-18 23:50` | 4,640,858 | 12 |
| 1-min solar | `2021-06-01` to `2026-06-18` | `2021-06-28 00:00` to `2026-06-18 23:50` | 825,420 | 2 |
| 10-min wind | `2021-06-01` to `2026-06-18` | `2021-06-29 10:10` to `2026-06-18 23:49` | 11,605,053 | 31 |

## NOAA ISD Surrounding Surface Stations

Downloaded 951 annual station-year files across 36 nearby/regional station
histories. Overall range is `1945` to `2025`.

| Station | Downloaded range | Files |
|---|---:|---:|
| 450010-99999 | 1973 to 1997 | 8 |
| 450030-99999 | 1977 to 2002 | 2 |
| 450040-99999 | 1979 to 1997 | 8 |
| 450050-99999 | 1946 to 2018 | 43 |
| 450060-99999 | 2001 to 2001 | 1 |
| 450070-99999 | 1948 to 2025 | 62 |
| 450090-99999 | 1947 to 1956 | 10 |
| 450100-99999 | 2001 to 2001 | 1 |
| 450110-99999 | 1951 to 2025 | 64 |
| 450200-99999 | 2012 to 2012 | 1 |
| 450320-99999 | 1992 to 2025 | 27 |
| 450330-99999 | 1992 to 1999 | 3 |
| 450340-99999 | 1992 to 2023 | 4 |
| 450350-99999 | 2004 to 2025 | 22 |
| 450390-99999 | 2004 to 2025 | 22 |
| 450410-99999 | 2004 to 2004 | 1 |
| 450440-99999 | 2002 to 2025 | 24 |
| 450450-99999 | 2004 to 2025 | 13 |
| 590750-99999 | 1973 to 1974 | 2 |
| 590870-99999 | 1957 to 2025 | 61 |
| 590960-99999 | 1957 to 2025 | 61 |
| 592710-99999 | 1957 to 1997 | 32 |
| 592730-99999 | 1974 to 1974 | 1 |
| 592780-99999 | 1957 to 2025 | 61 |
| 592800-99999 | 1999 to 1999 | 1 |
| 592870-99999 | 1945 to 2025 | 65 |
| 592930-99999 | 1956 to 2025 | 62 |
| 592980-99999 | 1957 to 1997 | 32 |
| 593030-99999 | 1957 to 1997 | 32 |
| 593090-99999 | 1974 to 1975 | 2 |
| 594780-99999 | 1956 to 1997 | 33 |
| 594880-99999 | 1974 to 1974 | 1 |
| 594930-99999 | 1957 to 2025 | 61 |
| 595010-99999 | 1956 to 2025 | 62 |
| 595050-99999 | 1983 to 2001 | 7 |
| 596730-99999 | 1959 to 2025 | 59 |

## Current/Live Families Already Collecting Prospectively

The Windows collector task exists and should be kept enabled after handoff.
These families are currently collected as changed/current payloads, not as full
historical archives:

- HKO latest regional observations.
- HKO latest forecasts/warnings/tips.
- HKO ARWF current portal payloads.
- HKO radar/lightning/nowcast current frames.
- HKO satellite current/rolling-window frames.
- HKO TC realtime track snapshot.
- HKO marine/tide feeds.
- NCEP GFS/GEFS current subsets.
- Daily Extract refresh.
- Upper-air refresh.

## Missing Or Incomplete Data

| Missing/incomplete item | Desired range | Current blocker/status | Required next action |
|---|---|---|---|
| HKO pre-2020/2021 sub-daily station observations | as far back as official records allow | public DATA.GOV.HK feed archives found only from 2020/2021 for downloaded high-frequency families | prepare official HKO request package; search official rescue/publication sources |
| HKO historical rainfall feed versions | full history if available | direct historical archive versions not found | official request or alternative public archive discovery |
| HKO historical visibility feed versions | full history if available | direct historical archive versions not found | official request or alternative public archive discovery |
| HKO direct RHR/current-weather JSON history | full history if available | direct historical archive versions not found | official request or alternative public archive discovery |
| HKO historical JSON forecasts (`flw`, `fnd`, `warnsum`, `warningInfo`, `swt`) | full issue/vintage history if available | RSS archives downloaded, but JSON version history not found | official request or alternative public archive discovery |
| HKO ARWF historical station/grid forecast vintages | full issue/vintage history if available | only current portal payloads found | official request or alternate archive discovery |
| Historical radar imagery / lightning / nowcast | as far back as lawful archive allows | public historical imagery archive not found | source discovery or official request |
| Historical Himawari/HKO satellite archive | ideally July 2015 to present for Himawari-8/9 | byte-budget and crop/downsampling policy required | implement object inventory dry-run, estimate bytes, approve subset before bulk download |
| Full historical GFS/GEFS operational archives | target-dependent multi-year period | large GRIB archive; byte-budget/source policy required | define exact variables, pressure levels, members, leads, cycles, domain, cadence, and byte budget |
| GEFSv12 reforecasts / TIGGE / other ensemble archives | target-dependent multi-year period | source policy and byte budget required | dry-run inventory and approval before download |
| ECMWF IFS/ENS/AIFS | target-dependent multi-year period | access/source policy and byte budget required | decide allowed products, terms, domain, variables, cycles, leads |
| DWD ICON/ICON-EPS | target-dependent multi-year period | large files and no confirmed server-side HK spatial subset | source policy, subset plan, byte estimate |
| ERA5 / ERA5-Land | retrospective mechanism data; date range TBD | CDS credentials and retrospective release-lag policy required | configure credentials; write dry-run request; approve point-in-time use policy |
| NOAA OISST / gridded SST | OISST v2.1 is available from 1981-present, but not downloaded | byte-budget/product/domain approval required | implement regional subset dry-run and byte estimate |
| GPM IMERG Final / Early / Late precipitation | Final historical plus Early/Late live if approved | Earthdata/NASA access and subset implementation required | configure credentials; dry-run regional requests; approve terms |
| CAMS aerosol/composition | historical and forecast subsets if approved | ADS credentials/terms and byte budget required | configure credentials; dry-run request; approve subset |
| Hong Kong EPD air quality | historical/live station data if official endpoint exists | official endpoint discovery still required | resolve endpoint, station metadata, and HKO-neighbor mapping |
| NOAA ISD 2026 annual station files | 2026 when published | 2026 annual files not exposed in tested NOAA annual archive | periodic refresh later in 2026/2027 |
| Terrain/slope/aspect station context | static/versioned | raw DTM/topographic data downloaded but not parsed | parse rasters/vectors into station context |
| Coastline/land-water station context | static/versioned | raw geospatial packages downloaded but not parsed | derive distance/water/coastline features with lineage |
| LUHK land-utilization station context | 2018-2024 downloaded | raw LUHK rasters downloaded but not parsed to station context | parse station buffers/context by LUHK class |
| Full PlanD 3D model tile payloads | static/versioned | only tile indexes downloaded; bulk tiles need approval | approve byte budget/source-use if actually needed |

## Parsing And Readiness Gaps

These are not missing downloads, but they are missing before modelling can be
trusted:

- Parse HKO Daily Extract and daily single-element climate files into bronze.
- Parse DATA.GOV.HK high-frequency ZIPs into station-variable-time bronze.
- Parse HKO latest observation feeds into station-row bronze with `retrieved_at`.
- Parse RSS forecast/warning archives into immutable issue/vintage tables.
- Parse ARWF station/grid forecasts and nowcasts.
- Parse NCEP GFS/GEFS GRIB2 subsets into cycle/member/lead/variable coverage.
- Parse NOAA ISD annual gzip records and IGRA upper-air soundings.
- Parse radar/satellite manifests and image timestamps/georeferencing metadata.
- Parse static terrain/coastline/LUHK context into deterministic station tables.
- Produce exact per-element start/end ranges from parsed HKO daily climate files.

## Recommended Next Decisions For GPT-Pro

1. First priority: parse existing raw data and generate exact per-source and
   per-variable coverage tables. This will clarify what is truly model-ready.
2. In parallel: prepare the official HKO request package for older sub-daily
   HKO station data and unavailable historical JSON/feed versions.
3. Decide byte budgets and lawful source policies for:
   - historical Himawari/archive satellite,
   - full historical NWP,
   - OISST/SST,
   - GPM IMERG.
4. Decide whether to provide credentials for:
   - CDS/ERA5,
   - NASA Earthdata/GPM,
   - CAMS ADS.
5. Do not start modelling until source-native bronze parsing, coverage reports,
   and point-in-time eligibility are complete.

