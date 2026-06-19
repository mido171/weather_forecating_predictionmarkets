# Acquisition Remaining Blockers

Polymarket is excluded. These are weather-data acquisition blockers only.

| Family | Current status | Why it is not downloaded now | Required next decision |
|---|---|---|---|
| Operational NWP/AI forecasts | `BYTE_BUDGET_REQUIRED` | GFS, GEFS, ECMWF, ICON, and AI forecast files are large GRIB cycle/member/lead archives. Fetching all cycles blindly would be high-volume and poorly specified. | Approve provider order, bounding box/domain, variables, pressure levels, forecast leads, cycles, ensemble members, date range, and byte budget. |
| Reanalysis | `CREDENTIAL_BLOCKED` | ERA5/ERA5-Land need CDS credentials and are retrospective-only large products. | Provide credentials and approve retrospective-only request subsets plus release-lag policy. |
| Gridded ocean/SST | `BYTE_BUDGET_REQUIRED` | NOAA OISST/OSTIA-style products are large gridded archives and not needed for raw HKO P0 completion without a subset plan. | Decide whether to include, then approve product, region, cadence, date range, and byte budget. |
| Static geospatial context | `SOURCE_SELECTION_REQUIRED` | Terrain, coastline, land-cover, and urban morphology need source/license/version selection; raster sizes may be non-trivial. | Select official/open products and allowed versions before download. |
| Satellite AOD/Himawari frames | `PARTIAL_IMPLEMENTED_WITH_BLOCKER` | HKO manifests were archived, but one MODIS AOD manifest image returned HTTP 404 and tested Himawari image path variants did not resolve. | Resolve HKO path rules or select another lawful official satellite archive. |
| Historical sub-hourly HKO latest feeds | `LATEST_ONLY_UNLESS_VERSIONED` | Public endpoints are latest/current feeds. Historical backfill is only lawful if DATA.GOV.HK historical versions or HKO archive access expose prior payloads. | Investigate historical API versions or accept prospective-only collection. |

No blocked family should be pulled into modelling or validation as if it were complete.
