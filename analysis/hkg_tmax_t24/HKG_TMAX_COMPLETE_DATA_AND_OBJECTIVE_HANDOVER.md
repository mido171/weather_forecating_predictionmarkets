# HKG Tmax Complete Data And Objective Handover

Generated: 2026-06-20

Repository:
`C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex`

Canonical local data root:
`C:\hkg_tmax_data`

## 1. What We Are Doing

We are building a leakage-safe weather forecasting system for the official Hong
Kong Observatory Headquarters daily maximum air temperature.

The scientific target is:

| Item | Value |
|---|---|
| Target station | Hong Kong Observatory Headquarters |
| Common station name in feeds | `HK Observatory` / `Hong Kong Observatory` |
| WMO station family | HKO / 45005 family |
| Target variable | Official daily maximum air temperature |
| Unit | degrees Celsius, usually 0.1 C precision |
| Target day | Hong Kong local calendar day T |
| Primary forecast cutoff | 15:00 HKT on day T-1 |
| Forecast product desired | Point forecast plus calibrated predictive distribution |

The key operational question is:

```text
Given only information actually available by 15:00 HKT on T-1,
what will the official HKO daily maximum temperature be on day T?
```

This is not a Polymarket handover. Market pricing, trading, PnL, backtesting,
execution simulation, and settlement bucket optimization are intentionally out of
scope for the current work. The current goal is meteorological prediction of HKO
Tmax only.

The work has three strict constraints:

1. Point-in-time safety: no feature can use information published or observable
   after the forecast cutoff.
2. Provenance: every raw payload is archived immutably with content hash,
   retrieval ledger row, sidecar metadata, and lineage.
3. Reproducibility: parsing, coverage, feature construction, baselines, and
   experiments must be documented and rerunnable.

## 2. Current Readiness Status

The raw archive audit passed.

| Item | Current value |
|---|---:|
| Retrieval ledger rows | 11,025 |
| Successful retrieval rows | 11,023 |
| Failed retrieval rows | 2 |
| Unique successful content hashes | 10,483 |
| Audit errors | 0 |

The parsed Phase A/B evidence currently includes:

| Parsed artifact | Rows | Range |
|---|---:|---|
| HKO daily climate bronze | 556,399 | element-specific, see below |
| HKO official Tmax target silver | 49,459 | 1884-01-01 to 2026-05-31 |
| Selected HKO high-frequency observations | 1,887,741 | 2020-06-30 09:00 HKT to 2026-06-18 15:30 HKT |
| HKO station-network cutoff summaries | 75,166 | 2020-06-30 to 2026-06-18 |
| T-24 candidate feature table | 49,459 | target calendar 1884-01-01 to 2026-05-31 |
| Rows with HKO cutoff temperature feature | 1,932 | mainly 2020-07-01 to 2026-05-31 |

The project is partially model-ready for HKO high-frequency station-state
analysis and first baselines. It is not fully production-ready because several
families remain raw-only, current-only, credential-gated, historically
unavailable, or missing detailed publication-lag proof.

## 3. Point-In-Time And Leakage Rules

The universal eligibility rule is:

```text
available_at <= 15:00:00 Asia/Hong_Kong on T-1
```

For high-frequency HKO station observations, the current conservative replay
assumption is:

```text
available_at = observed_at + 20 minutes
```

This matters. A station row observed at exactly 15:00 HKT is not treated as
available at 15:00. It is treated as available at 15:20 and is therefore not
eligible for a forecast issued at 15:00. Candidate features are selected by
`available_at <= cutoff`, not merely by `observed_at <= cutoff`.

Same-day official daily climate values for day T are not operational predictors.
They are target labels or retrospective mechanism data. Reanalysis, final best
tracks, finalized gridded precipitation, finalized SST analyses, and current-only
NWP cycles are not allowed as historical operational features unless a defensible
real-time release simulation exists.

## 4. Core Parsed Tables And Attributes

### 4.1 HKO Official Tmax Target Table

Path:
`C:\hkg_tmax_data\silver\targets\hko_daily_tmax.parquet`

Purpose:
This is the official target label table for the prediction problem.

Date range:
`1884-01-01` to `2026-05-31`

Rows:
`49,459`

Attributes:

| Column | Meaning |
|---|---|
| `local_date` | Hong Kong local calendar day T |
| `target_tmax_c` | Official HKO daily maximum temperature in C |
| `content_sha256` | Hash of the raw source object used to derive the row |
| `raw_retrieved_at` | Retrieval timestamp of the raw source object |
| `target_station` | `Hong Kong Observatory` |
| `target_source_id` | `hko_daily_climate_maximum_temperature_all` |
| `target_role` | `TARGET_ONLY` |

Station/domain:
Hong Kong Observatory.

Operational role:
Target only. Never use `target_tmax_c` as a same-day feature.

### 4.2 HKO Daily Climate Bronze Table

Path:
`C:\hkg_tmax_data\bronze\analysis_phase_a\hko_daily_climate_elements.parquet`

Purpose:
Source-native daily single-element HKO climate rows, parsed with units,
station/domain, completeness flags, source hash, and role classification.

Common attributes:

| Column | Meaning |
|---|---|
| `source_id` | Canonical raw source ID |
| `content_sha256` | Raw object hash |
| `retrieved_at` | Raw retrieval timestamp |
| `station_or_domain` | HKO station or domain label |
| `variable` | Canonical variable name |
| `unit` | Unit |
| `role` | Operational eligibility role |
| `local_date` | Hong Kong local date |
| `year`, `month`, `day` | Source-native date fields |
| `value` | Parsed numeric value when available |
| `value_precision` | Numeric precision inferred from source text |
| `completeness` | Source completeness flag when present |
| `parse_issue` | Parser issue marker, for example missing/non-numeric |

Element-specific coverage:

| Station/domain | Variable | Unit | Role | Start | End | Rows | Non-missing |
|---|---|---|---|---|---|---:|---:|
| Hong Kong Observatory | daily_maximum_temperature | C | TARGET_ONLY | 1884-01-01 | 2026-05-31 | 49,459 | 49,459 |
| Hong Kong Observatory | daily_minimum_temperature | C | RETROSPECTIVE_MECHANISM_ONLY | 1884-01-01 | 2026-05-31 | 49,459 | 49,459 |
| Hong Kong Observatory | mean_temperature | C | RETROSPECTIVE_MECHANISM_ONLY | 1884-03-01 | 2026-05-31 | 49,399 | 49,368 |
| Hong Kong Observatory | mean_sea_level_pressure | hPa | RETROSPECTIVE_MECHANISM_ONLY | 1884-03-01 | 2026-05-31 | 49,399 | 49,368 |
| Hong Kong Observatory | daily_rainfall | mm | RETROSPECTIVE_MECHANISM_ONLY | 1884-03-01 | 2026-05-31 | 49,399 | 42,483 |
| Hong Kong Observatory | mean_relative_humidity | percent | RETROSPECTIVE_MECHANISM_ONLY | 1947-01-01 | 2026-05-31 | 29,006 | 29,006 |
| Hong Kong Observatory | mean_wet_bulb_temperature | C | RETROSPECTIVE_MECHANISM_ONLY | 1947-01-01 | 2026-05-31 | 29,006 | 29,006 |
| Hong Kong Observatory | mean_cloud_amount | percent | RETROSPECTIVE_MECHANISM_ONLY | 1949-01-01 | 2026-05-31 | 28,275 | 28,275 |
| Hong Kong Observatory | mean_dew_point_temperature | C | RETROSPECTIVE_MECHANISM_ONLY | 1961-01-01 | 2026-05-31 | 23,892 | 23,892 |
| Hong Kong Observatory | grass_minimum_temperature | C | RETROSPECTIVE_MECHANISM_ONLY | 1968-01-01 | 2026-05-31 | 21,336 | 21,227 |
| King's Park | bright_sunshine_duration | hours | RETROSPECTIVE_MECHANISM_ONLY | 1961-01-01 | 2026-05-31 | 23,892 | 23,892 |
| King's Park | evaporation | mm | RETROSPECTIVE_MECHANISM_ONLY | 1968-01-01 | 2026-05-31 | 21,336 | 21,237 |
| King's Park | global_solar_radiation | MJ/m2 | RETROSPECTIVE_MECHANISM_ONLY | 1978-01-01 | 2026-05-31 | 17,683 | 17,349 |
| Waglan Island | prevailing_wind_direction | degree_or_compass | RETROSPECTIVE_MECHANISM_ONLY | 1975-01-01 | 2026-05-31 | 18,779 | 18,084 |
| Waglan Island | mean_wind_speed | km/h | RETROSPECTIVE_MECHANISM_ONLY | 1975-01-01 | 2026-05-31 | 18,779 | 18,299 |
| Waglan Island | sea_temperature | C | RETROSPECTIVE_MECHANISM_ONLY | 1990-01-01 | 2026-05-31 | 13,300 | 7,798 |
| North Point | sea_temperature_am | C | RETROSPECTIVE_MECHANISM_ONLY | 1974-06-18 | 2026-05-31 | 18,976 | 18,944 |
| North Point | sea_temperature_pm | C | RETROSPECTIVE_MECHANISM_ONLY | 1974-06-18 | 2026-05-31 | 18,976 | 18,944 |
| Hong Kong Territory | cloud_to_ground_lightning | count | RETROSPECTIVE_MECHANISM_ONLY | 2005-06-21 | 2026-05-31 | 7,650 | 7,650 |
| Hong Kong Territory | cloud_to_cloud_lightning | count | RETROSPECTIVE_MECHANISM_ONLY | 2005-06-21 | 2026-05-31 | 7,650 | 7,650 |
| Hong Kong International Airport | reduced_visibility_hours | hours | RETROSPECTIVE_MECHANISM_ONLY | 1997-01-01 | 2026-05-31 | 10,743 | 10,699 |

Stations/domains covered:
Hong Kong Observatory, King's Park, Waglan Island, North Point, Hong Kong
Territory, Hong Kong International Airport.

### 4.3 HKO High-Frequency Historical Station Archives

Purpose:
Sub-daily station/weather observations from DATA.GOV.HK historical ZIP archives.
These are the strongest currently acquired operational-style station features.
They do not go before 2020/2021 in the public archive currently found.

Common raw attributes:

| Attribute | Meaning |
|---|---|
| `Date time` | Observation timestamp in HKT source format, usually `YYYYMMDDHHMM` |
| `Automatic Weather Station` | Station name |
| Value columns | Feed-specific meteorological measurements |

Parsed selected-observation attributes:

| Column | Meaning |
|---|---|
| `source_id` | Raw archive source ID |
| `family` | Feed family |
| `content_sha256` | Raw ZIP hash |
| `retrieved_at` | Retrieval timestamp |
| `station` | Station name |
| `observed_at_hkt` | Observation timestamp in Hong Kong time |
| `local_date` | Local date |
| `variable` | Canonical variable |
| `unit` | Unit |
| `value` | Numeric value |
| `role` | `OPERATIONAL_WITH_CONSERVATIVE_LATENCY` |
| `availability_assumption` | Current replay latency rule |
| `available_at_hkt` | `observed_at_hkt + 20 minutes` |

Feed coverage:

| Feed | Archive range | Observation range | Rows | Stations | Main attributes |
|---|---|---|---:|---:|---|
| 1-min temperature | 2020-06-01 to 2026-06-18 | 2020-06-30 09:00 to 2026-06-18 23:50 | 16,859,348 | 39 | air temperature, C |
| Since-midnight max/min | 2020-06-01 to 2026-06-18 | 2020-06-30 09:00 to 2026-06-18 23:50 | 16,882,046 | 39 | maximum temperature since midnight, minimum temperature since midnight, C |
| 1-min humidity | 2020-06-01 to 2026-06-18 | 2020-06-30 09:00 to 2026-06-18 23:50 | 10,666,499 | 28 | relative humidity, percent |
| 15-min UV index | 2020-06-01 to 2026-06-18 | 2020-06-30 10:15 to 2026-06-18 18:15 | 101,423 | not station-row based | UV index |
| 1-min pressure | 2021-06-01 to 2026-06-18 | 2021-06-29 10:10 to 2026-06-18 23:50 | 4,640,858 | 12 | mean sea level pressure, hPa |
| 1-min solar | 2021-06-01 to 2026-06-18 | 2021-06-28 00:00 to 2026-06-18 23:50 | 825,420 | 2 | global solar radiation, direct solar radiation, diffuse radiation, W/m2 |
| 10-min wind | 2021-06-01 to 2026-06-18 | 2021-06-29 10:10 to 2026-06-18 23:49 | 11,605,053 | 31 | mean wind direction, mean wind speed, maximum gust |

#### 1-min Temperature And Since-Midnight Max/Min Stations

These two feeds cover the same 39 station names:

- Chek Lap Kok
- Cheung Chau
- Clear Water Bay
- HK Observatory
- HK Park
- Happy Valley
- Kai Tak Runway Park
- Kau Sai Chau
- King's Park
- Kowloon City
- Kwun Tong
- Lau Fau Shan
- Ngong Ping
- Pak Tam Chung
- Peng Chau
- Sai Kung
- Sha Tin
- Sham Shui Po
- Shau Kei Wan
- Shek Kong
- Sheung Shui
- Stanley
- Ta Kwu Ling
- Tai Lung
- Tai Mei Tuk
- Tai Mo Shan
- Tai Po
- Tate's Cairn
- The Peak
- Tseung Kwan O
- Tsing Yi
- Tsuen Wan Ho Koon
- Tsuen Wan Shing Mun Valley
- Tuen Mun
- Waglan Island
- Wetland Park
- Wong Chuk Hang
- Wong Tai Sin
- Yuen Long Park

#### 1-min Humidity Stations

- Chek Lap Kok
- Cheung Chau
- Clear Water Bay
- HK Observatory
- HK Park
- Kai Tak Runway Park
- Kau Sai Chau
- King's Park
- Kowloon City
- Lau Fau Shan
- Pak Tam Chung
- Peng Chau
- Sai Kung
- Sha Tin
- Shau Kei Wan
- Shek Kong
- Sheung Shui
- Ta Kwu Ling
- Tai Lung
- Tai Po
- Tseung Kwan O
- Tsing Yi
- Tsuen Wan Ho Koon
- Tsuen Wan Shing Mun Valley
- Tuen Mun
- Waglan Island
- Wetland Park
- Wong Chuk Hang

#### 1-min Pressure Stations

- Chek Lap Kok
- Cheung Chau
- HK Observatory
- Lau Fau Shan
- Peng Chau
- Sha Tin
- Shek Kong
- Sheung Shui
- Ta Kwu Ling
- Tai Po
- Waglan Island
- Wetland Park

#### 1-min Solar Stations

- Kau Sai Chau
- King's Park

#### 10-min Wind Stations

- Central Pier
- Chek Lap Kok
- Cheung Chau
- Cheung Chau Beach
- Green Island
- Hong Kong Sea School
- Kai Tak
- King's Park
- Lamma Island
- Lau Fau Shan
- Ngong Ping
- North Point
- Peng Chau
- Sai Kung
- Sha Chau
- Sha Tin
- Shek Kong
- Stanley
- Star Ferry
- Ta Kwu Ling
- Tai Mei Tuk
- Tai Po Kau
- Tap Mun
- Tate's Cairn
- Tseung Kwan O
- Tsing Yi
- Tuen Mun
- Waglan Island
- Wetland Park
- Wong Chuk Han
- Wong Chuk Hang

UV index is not station-row based.

### 4.4 T-24 Candidate Feature Table

Path:
`C:\hkg_tmax_data\silver\features\t24_cutoff_feature_candidates.parquet`

Purpose:
Candidate feature table for target day T. It joins target labels with features
that can be reconstructed for the T-1 15:00 HKT cutoff.

Important date facts:

- Full target calendar rows: `1884-01-01` to `2026-05-31`.
- Rows with HKO cutoff temperature feature: `1,932`.
- Modern common scoring sample used in EXP-0002: `2021-07-01` to `2026-05-31`.

Attributes:

| Column | Meaning |
|---|---|
| `local_date` | Target day T |
| `target_tmax_c` | Official HKO daily max temperature label |
| `cutoff_hkt` | Forecast cutoff, 15:00 HKT on T-1 |
| `hko_temp_at_tminus1_1500_c` | Latest HKO station temperature available by cutoff |
| `hko_temp_at_tminus1_1500_c_observed_at_hkt` | Source observation time for that temperature |
| `hko_temp_at_tminus1_1500_c_available_at_hkt` | Availability time after latency assumption |
| `hko_rh_at_tminus1_1500_pct` | Latest HKO relative humidity available by cutoff |
| `hko_rh_at_tminus1_1500_pct_observed_at_hkt` | RH observation time |
| `hko_rh_at_tminus1_1500_pct_available_at_hkt` | RH availability time |
| `hko_mslp_at_tminus1_1500_hpa` | Latest HKO MSL pressure available by cutoff |
| `hko_mslp_at_tminus1_1500_hpa_observed_at_hkt` | Pressure observation time |
| `hko_mslp_at_tminus1_1500_hpa_available_at_hkt` | Pressure availability time |
| `hko_tminus1_max_so_far_1500_c` | HKO maximum temperature since midnight on T-1, available by cutoff |
| `hko_tminus1_min_so_far_1500_c` | HKO minimum temperature since midnight on T-1, available by cutoff |
| `hko_temp_tminus1_1200_c` | HKO temperature around T-1 12:00, availability-checked |
| `hko_temp_tminus1_0900_c` | HKO temperature around T-1 09:00, availability-checked |
| `hko_mslp_tminus1_1200_hpa` | HKO pressure around T-1 12:00, availability-checked |
| `hko_temp_3h_change_to_cutoff_c` | Difference between cutoff temperature and 12:00 temperature |
| `hko_temp_6h_change_to_cutoff_c` | Difference between cutoff temperature and 09:00 temperature |
| `hko_mslp_3h_change_to_cutoff_hpa` | Difference between cutoff pressure and 12:00 pressure |
| `hko_tminus2_official_tmax_c` | T-2 official HKO Tmax, proxy feature pending publication-lag proof |
| `split_role` | Development, validation, or locked-test label |

Stations:
Feature table currently uses HKO station features for the main target-station
state. Station-network summaries are separate.

### 4.5 Station-Network Cutoff Summary

Path:
`C:\hkg_tmax_data\silver\observations\hko_station_temperature_cutoff_summary.parquet`

Purpose:
Daily station-network summary of station temperature near the T-1 15:00 cutoff.
This is used for station-network EDA and later candidate spatial features.

Date range:
`2020-06-30` to `2026-06-18`

Rows:
`75,166`

Attributes:

| Column | Meaning |
|---|---|
| `station` | HKO station name |
| `local_date` | Local date |
| `cutoff_window_obs_count` | Count of station observations in the cutoff window used for that day |
| `latest_before_1500_at_hkt` | Latest observation timestamp before or at the cutoff observation window |
| `cutoff_temperature_c` | Temperature at the selected cutoff-window observation |

Station set:
Same source station universe as HKO high-frequency 1-min temperature, with
availability depending on actual station coverage and valid values.

## 5. Raw-Only Or Partially Parsed Dataset Families

The following families are acquired in the raw archive, but many still require
source-specific parsers before they are trusted model inputs.

### 5.1 NOAA ISD Nearby / Regional Surface Stations

Purpose:
Regional/surrounding surface observations around Hong Kong and nearby mainland
stations. These can provide long-history context, regional gradients, wind,
temperature, pressure, visibility, dew point, cloud, and precipitation signals
depending on station and year.

Current status:
Raw station-year gzip files acquired. Not yet parsed into canonical silver.

Overall range:
`1945` to `2025`

Files:
`951` annual station-year files across `36` station histories.

Typical raw ISD attributes to parse:

| Attribute family | Meaning |
|---|---|
| Station identifiers | USAF/WBAN or station ID |
| Timestamp | Observation date/time, usually UTC |
| Location metadata | latitude, longitude, elevation, station metadata where available |
| Wind | direction, speed, gust, quality flags |
| Visibility | visibility distance and quality flags |
| Temperature | air temperature and quality flags |
| Dew point | dew point temperature and quality flags |
| Pressure | sea-level pressure or station pressure depending record |
| Cloud / ceiling | ceiling height, sky cover fields where available |
| Present weather | weather codes where available |
| Precipitation | accumulation fields where available |
| Quality flags | ISD source quality/control flags |

Station IDs and ranges:

| Station ID | Start | End | Files |
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

### 5.2 IGRA Upper-Air

Source family:
IGRA upper-air station `HKM00045004`, King's Park / Hong Kong upper-air family.

Range:
`1949` to `2026`

Status:
Raw period-of-record and year-to-date archives acquired. Canonical parser still
pending.

Attributes to parse:

| Attribute | Meaning |
|---|---|
| Sounding date/time | Launch/report time, must be converted to point-in-time availability |
| Pressure level | Mandatory/significant pressure levels |
| Geopotential height | Height of pressure level |
| Temperature | Upper-air temperature |
| Dew point / dew point depression / humidity | Moisture profile depending IGRA field |
| Wind direction | Direction by level |
| Wind speed | Speed by level |
| Quality flags | IGRA quality flags |

Forecast relevance:
Can derive 1000/975/950/925/900/850 hPa temperature, lapse rate, inversion,
mixing proxies, moisture aloft, and transport winds, but only if the sounding was
available before the cutoff.

### 5.3 HKO Forecasts, Warnings, RSS, And ARWF

Ranges:

| Family | Range | Status |
|---|---|---|
| HKO RSS forecast/warning archives | 2020-06-01 to 2026-06-18 | raw downloaded |
| HKO 9-day RSS forecast archives | 2021-04-01 to 2026-06-18 | raw downloaded |
| Latest JSON forecasts/warnings | live/current snapshots from 2026-06-18/19 onward | prospective raw |
| HKO ARWF station/grid forecasts and nowcasts | live/current snapshots from 2026-06-19 onward | prospective raw |

Attributes to parse:

| Attribute | Meaning |
|---|---|
| Source ID / product | Forecast, warning, RSS, ARWF, nowcast product family |
| Issue time | Provider issue/publication time when present |
| Retrieved time | Local archive retrieval timestamp |
| Valid time / forecast period | Forecast validity window |
| Forecast text | Human-readable weather forecast text |
| Forecast Tmax/Tmin | Forecast temperatures where available |
| Weather icons/codes | Structured weather state where available |
| Wind forecast | Wind direction/speed/force where available |
| Rain/cloud descriptors | Text or structured rain/cloud fields |
| Warning/tip state | Active warning or special weather tip information |
| Station/grid identifiers | ARWF station or grid IDs where present |

Station coverage:
RSS and JSON forecast products are generally HKO territory/product-level rather
than direct station observations. ARWF has station/grid products, but station/grid
coverage still needs parser-level confirmation.

### 5.4 NCEP GFS/GEFS Current Subsets

Range:
`2026-06-19` operational cycle only, leads through `f120`.

Status:
Current/prospective NWP raw subsets acquired. Not historical/backtestable yet.

Attributes to parse:

| Attribute | Meaning |
|---|---|
| Model | GFS or GEFS |
| Cycle initialization time | Model run cycle |
| Forecast lead | Forecast hour, through f120 in current subset |
| Valid time | Cycle plus lead |
| Variable | GRIB variable name |
| Level | Surface, pressure level, or other model level |
| Member | Ensemble member for GEFS |
| Grid/domain | Hong Kong regional subset |
| Value array | Gridded field values |

Operational role:
`PROSPECTIVE_ONLY_NOT_YET_BACKTESTABLE` until historical cycle archives exist.

Station coverage:
Not station-specific. Gridded regional data must be extracted to HKO/station
locations or station neighborhoods later.

### 5.5 Radar, Rainfall Nowcast, Lightning

Range:
`2026-06-19 10:54` to `2026-06-20 06:36`

Status:
Current/prospective raw archive. Historical imagery/backfill not found.

Attributes:

| Product | Main raw attributes |
|---|---|
| Radar frames | Product timestamp, image/frame URL, raw image bytes, product family |
| Rainfall nowcast | Product timestamp, grid/image/KML metadata, nowcast validity |
| Lightning pages/counts | Product timestamp, count/category/domain fields where present |

Station coverage:
Not station-specific. These are spatial/domain products. They need geospatial
parsing before station-local features can be derived.

### 5.6 Satellite / Cloud / Aerosol Current Products

Range:
`2025-06-20 14:40` to `2026-06-19 22:30`

Status:
Partial current/rolling archive. Historical Himawari-scale archive not downloaded
yet because it requires byte-budgeted subset policy.

Attributes:

| Attribute | Meaning |
|---|---|
| Product family | HKO current satellite/cloud/aerosol product type |
| Frame timestamp | Image/product valid time |
| Retrieval timestamp | Local archived retrieval time |
| Raw image/object bytes | Immutable raw payload |
| Image metadata | URL/path/hash/sidecar; georeferencing still parser-dependent |

Station coverage:
Not station-specific until cropped or sampled around stations.

### 5.7 Tropical Cyclone Data

Range:
`1985` to `2024` for HKO tropical cyclone best tracks.

Status:
Raw retrospective best tracks downloaded. Realtime TC snapshots are collected
prospectively.

Attributes to parse:

| Attribute | Meaning |
|---|---|
| Storm ID/name | Tropical cyclone identifier |
| Timestamp | Track valid time |
| Position | Latitude/longitude |
| Intensity | Wind/pressure/intensity fields where present |
| Track status | Best-track retrospective status versus realtime advisory |

Operational role:
Best tracks are retrospective mechanism data only. They cannot be used as
operational historical features unless replaced by advisory vintages available at
the cutoff.

### 5.8 Marine, Tide, And Sea Temperature

Available:

| Dataset | Range | Status |
|---|---|---|
| HKO daily Waglan sea temperature | 1990-01-01 to 2026-05-31 | parsed daily climate |
| HKO North Point sea temp AM/PM | 1974-06-18 to 2026-05-31 | parsed daily climate |
| Current marine/tide feeds | from 2026-06-19 onward | raw/current |

Attributes:

| Attribute | Meaning |
|---|---|
| Sea temperature | Daily sea temperature at Waglan/North Point |
| Tide information | Current tide station/product fields, parser pending |
| Coastal waters bulletin | Marine forecast text, wind/sea descriptors, parser pending |

Station coverage:
Waglan Island, North Point, and current tide/marine station/product coverage to
be confirmed by parsers.

### 5.9 Static And Geospatial Context

Status:
Raw static packages downloaded; some deterministic derived context created.

Known ranges:

| Dataset | Range/status |
|---|---|
| LUHK land utilization | 2018 to 2024 raw files |
| Solar geometry | 2026 derived |
| Station registry / distance / bearing | derived static context |
| Terrain/topographic/coastline/geospatial packages | raw static files, parser pending |

Attributes:

| Attribute | Meaning |
|---|---|
| Station ID/name | HKO or regional station identifier |
| Latitude/longitude | Station coordinates when available |
| Distance/bearing | Station-neighbor geometry |
| Solar geometry | Deterministic sun/solar-position values |
| Terrain/elevation/slope/aspect | Pending derived station context |
| Coastline/water distance | Pending derived station context |
| Land-use buffer fractions | Pending derived station context from LUHK |

Station coverage:
Station registry and station-neighbor context apply to HKO/HKG station lists once
station aliases and metadata are reconciled.

## 6. Current Baseline Experiment Context

Experiment folder:
`analysis\hkg_tmax_t24\experiments\EXP-0002-baseline-suite`

The first baseline suite froze this modern split:

| Split | Range | Scored rows |
|---|---|---:|
| Development | 2021-07-01 to 2023-12-31 | 729 |
| Validation 2024 | 2024-01-01 to 2024-12-31 | 364 |
| Locked test | 2025-01-01 to 2026-05-31 | 516 |

Champion baseline:
`station_state_analogue`

Validation 2024:
MAE `1.5032 C`, RMSE `1.8974 C`, median absolute error `1.2980 C`.

Locked test:
MAE `1.5668 C`, RMSE `1.9776 C`.

This is not a final model. It is the frozen comparison baseline for the next
predeclared meteorological experiments.

## 7. What Is Missing Or Not Yet Model-Ready

Important gaps:

- HKO pre-2020/2021 sub-daily station observations are not in the public
  historical feed archive found so far.
- HKO historical rainfall feed versions, visibility feed versions, and direct
  RHR/current-weather JSON history are not found as public historical archives.
- HKO historical JSON forecast versions are not found, though RSS forecast and
  warning archives are downloaded.
- ARWF historical station/grid forecast vintages are not found.
- Historical radar/lightning/nowcast archives are not found.
- Full historical NWP/ensemble cycle archives are not downloaded.
- ERA5, OISST, GPM IMERG, CAMS, and large satellite archives need credentials,
  byte budgets, or source policy before bulk acquisition.
- NOAA ISD, IGRA, forecasts, ARWF, NWP, radar, satellite, TC, marine/tide, and
  static geospatial files still need canonical parsers before model use.

## 8. Recommended Next Task

The next high-value task is to predeclare a new experiment before looking at new
locked-test decisions:

```text
EXP-0003 station thermal-memory challenger
```

The experiment should test only cutoff-safe features against the frozen
`station_state_analogue` champion:

- HKO temperature at cutoff.
- Morning-to-cutoff temperature slope.
- T-1 since-midnight max/min available by cutoff.
- HKO humidity at cutoff.
- HKO pressure level and pressure tendency.
- T-2 and multi-day thermal memory with publication-lag caveats.
- Station-network cutoff contrasts such as HKO versus King's Park, Waglan,
  Chek Lap Kok, Lau Fau Shan, Sha Tin, Ta Kwu Ling, and other nearby stations.

Do not start Polymarket work. Do not start uncontrolled ML. Do not use locked
test as a feature-selection surface.
