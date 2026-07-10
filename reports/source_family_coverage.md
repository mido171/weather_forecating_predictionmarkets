# Source Family Coverage

Generated offline from the canonical raw retrieval ledger. This report is
acquisition evidence only; it does not perform modelling or mutate the raw archive.

- data root: `C:\hkg_tmax_data`

| Family | Status | Success rows | Source IDs | Bytes | Date/token range | Remaining acquisition work |
|---|---|---:|---:|---:|---|---|
| A target labels / daily climate | `DOWNLOADED` | 170 | 23 | 13,594,253 | 1884 to 202606 | Parse to bronze target/daily climate tables; no additional raw download known. |
| B station/catalog metadata | `DOWNLOADED_INITIAL` | 5 | 3 | 841,835 |  | Parse station metadata and reconcile station aliases/history. |
| C high-frequency HKO observations | `PARTIAL_WITH_HISTORICAL_BACKFILL` | 779 | 24 | 1,853,497,726 | 20200601 to 20260618 | Rainfall, visibility, and direct RHR JSON older histories were not found; collect prospectively or find another official archive. |
| D forecasts / warnings / ARWF | `PARTIAL_WITH_HISTORICAL_RSS` | 1,465 | 64 | 544,895,045 | 20200601 to 20260618 | Historical JSON forecast versions still need another official archive; install prospective collectors. |
| E operational NWP | `PARTIAL_CURRENT_NCEP` | 2,042 | 4 | 18,967,512 | 20260619 to 20260619 | Full historical/continuous GFS/GEFS/ECMWF/DWD/AI archives need approved byte-budgeted subsets. |
| F upper-air | `DOWNLOADED` | 6 | 6 | 57,674,159 |  | Parse IGRA period-of-record and year-to-date archives. |
| G radar / rainfall nowcast / lightning | `DOWNLOADED_INITIAL` | 527 | 8 | 92,701,325 | 202606191054 to 202606200636 | Historical imagery/backfill limited unless another official archive is found; collect prospectively. |
| H satellite / cloud / aerosol | `PARTIAL_CURRENT_ARCHIVED` | 3,947 | 27 | 1,722,097,511 | 202506201440 to 202606192230 | Keep current HKO satellite collectors running; historical Himawari/archive-scale satellite acquisition remains byte-budgeted. |
| I tropical cyclone / regional surface | `PARTIAL_WITH_SURFACE_ARCHIVE` | 1,013 | 8 | 131,681,208 | 1985 to 2024 | Operational advisory vintages before acquisition remain unavailable unless another archive is found. |
| J marine / ocean | `DOWNLOADED_INITIAL` | 17 | 5 | 885,140 |  | Gridded SST/OISST/OSTIA needs product choice and byte budget. |
| L static geospatial context | `PARTIAL_DERIVED_CONTEXT` | 60 | 60 | 215,229,122 |  | Station registry, distance/bearing, and solar geometry are derived; terrain/coastline/LUHK context still needs source-specific parsers. |

## Unclassified Successful Source IDs

- `datagov_historical_api_documentation`
- `datagov_historical_rss_api_documentation`
- `dwd_icon`
- `dwd_icon_eps`
- `ecmwf_open_ifs_aifs`
- `hko_historical_archive_api`
- `noaa_gefs`
- `noaa_gfs`
