# Acquisition Progress Snapshot

Generated offline from the canonical raw ledger. This report is read-only with
respect to `C:\hkg_tmax_data`.

## Ledger Totals

- retrieval attempts: `10,046`
- successful retrievals: `10,044`
- failed retrievals: `2`
- logical source IDs: `242`
- unique successful content hashes: `9,526`
- successful archived bytes: `4,652,546,574`

## HKO Current Satellite Progress

Latest archived HKO current-satellite manifests list `3,552` frames. Of those manifest-listed filenames, `2,759` are already archived and `793` are not yet present in the ledger.

Note: these manifests are rolling operational windows and can list candidate filenames that are no longer provider-resolvable. The `satellite-current` batch is authoritative for live collection because it preflights current URLs and archives only resolvable non-2xx-skipped frames without overwriting immutable raw objects.

| Product | Manifest-listed | Downloaded matching manifest | Downloaded total | Manifest-listed missing | First missing | Last missing |
|---|---:|---:|---:|---:|---|---|
| H8 infrared | 849 | 849 | 1,089 | 0 |  |  |
| FY4B deep convection | 191 | 191 | 242 | 0 |  |  |
| H8 deep convection | 849 | 759 | 996 | 90 | h8_dcred_x2M_20260617091000.png | h8_dcred_x8M_20260617140000.png |
| FY4B infrared | 191 | 171 | 222 | 20 | fy4b_ir_WA_20260617090000.jpg | fy4b_ir_WA_20260617134500.jpg |
| H8 true colour | 447 | 415 | 454 | 32 | h8_tc_x2M_20260617091000.jpg | h8_tc_x8M_20260617101000.jpg |
| FY4B true colour | 125 | 105 | 125 | 20 | fy4b_nc_WA_20260617090000.jpg | fy4b_nc_WA_20260617134500.jpg |
| H8 all-day visible | 849 | 252 | 331 | 597 | h8_advis_x2M_20260617090000.png | h8_advis_x8M_20260617140000.png |
| GK2B aerosol optical depth | 51 | 17 | 17 | 34 | gk2b_aod_x2M_20260617081500.png | gk2b_aod_x4M_20260619081500.png |

Current satellite acquisition state:

- `2,759/3,552` manifest-listed filenames are archived in the current ledger snapshot.
- `793` manifest-listed filenames are not in the ledger snapshot; treat them as rolling-window/persistent-miss candidates until the live preflight proves they are still resolvable.

## HKO High-Frequency Historical Feeds

| Feed archive source | Archive files | Start | End |
|---|---:|---:|---:|
| datagov_hko_historical_latest_10min_wind_archive | 78 | 20210601 | 20260618 |
| datagov_hko_historical_latest_15min_uvindex_archive | 90 | 20200601 | 20260618 |
| datagov_hko_historical_latest_1min_humidity_archive | 90 | 20200601 | 20260618 |
| datagov_hko_historical_latest_1min_pressure_archive | 78 | 20210601 | 20260618 |
| datagov_hko_historical_latest_1min_solar_archive | 78 | 20210601 | 20260618 |
| datagov_hko_historical_latest_1min_temperature_archive | 90 | 20200601 | 20260618 |
| datagov_hko_historical_latest_since_midnight_maxmin_archive | 90 | 20200601 | 20260618 |

## Nearby / Surrounding Station Coverage

NOAA ISD nearby station-year files: `951` across `36` station histories.

| Station | Start | End | Files |
|---|---:|---:|---:|
| 450010-99999 | 1973 | 1997 | 8 |
| 450030-99999 | 1977 | 2002 | 2 |
| 450040-99999 | 1979 | 1997 | 8 |
| 450050-99999 | 1946 | 2018 | 43 |
| 450060-99999 | 2001 | 2001 | 1 |
| 450070-99999 | 1948 | 2025 | 62 |
| 450090-99999 | 1947 | 1956 | 10 |
| 450100-99999 | 2001 | 2001 | 1 |
| 450110-99999 | 1951 | 2025 | 64 |
| 450200-99999 | 2012 | 2012 | 1 |
| 450320-99999 | 1992 | 2025 | 27 |
| 450330-99999 | 1992 | 1999 | 3 |
| 450340-99999 | 1992 | 2023 | 4 |
| 450350-99999 | 2004 | 2025 | 22 |
| 450390-99999 | 2004 | 2025 | 22 |
| 450410-99999 | 2004 | 2004 | 1 |
| 450440-99999 | 2002 | 2025 | 24 |
| 450450-99999 | 2004 | 2025 | 13 |
| 590750-99999 | 1973 | 1974 | 2 |
| 590870-99999 | 1957 | 2025 | 61 |
| 590960-99999 | 1957 | 2025 | 61 |
| 592710-99999 | 1957 | 1997 | 32 |
| 592730-99999 | 1974 | 1974 | 1 |
| 592780-99999 | 1957 | 2025 | 61 |
| 592800-99999 | 1999 | 1999 | 1 |
| 592870-99999 | 1945 | 2025 | 65 |
| 592930-99999 | 1956 | 2025 | 62 |
| 592980-99999 | 1957 | 1997 | 32 |
| 593030-99999 | 1957 | 1997 | 32 |
| 593090-99999 | 1974 | 1975 | 2 |
| 594780-99999 | 1956 | 1997 | 33 |
| 594880-99999 | 1974 | 1974 | 1 |
| 594930-99999 | 1957 | 2025 | 61 |
| 595010-99999 | 1956 | 2025 | 62 |
| 595050-99999 | 1983 | 2001 | 7 |
| 596730-99999 | 1959 | 2025 | 59 |

## Other Current Coverage Counts

- HKO ARWF station/grid forecast payloads: `53`
- HKO radar current frames: `488`
- NCEP GFS regional subset files: `60`
- NCEP GEFS regional subset files: `1,980`
- HKO tropical cyclone best-track annual CSVs: `40`

## Immediate Remaining Fetch Queue

1. Keep the scheduled live collector enabled and monitor changed-payload health.
2. Rerun the live satellite batch when a fresh current-window audit is required:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax acquisition hko-backfill --batch satellite-current --continue-on-error --delay-seconds 0 --skip-existing-successes
```

3. Investigate persistent manifest-listed satellite misses only if the live preflight still returns resolvable 2xx URLs.
4. Continue the remaining credential-gated or byte-budgeted historical families through the gridded acquisition policy.
