# 02 — IEM MOS Station Guidance Acquisition

## 1. Purpose

IEM MOS is a mandatory source because it provides long-history, station-specific forecast guidance. It is especially valuable for KLGA daily-high markets because MOS is already point-calibrated guidance derived from NWS model output and statistical equations.

The IEM MOS archive must be used as the long-history forecast backbone of the system.

Official source pages:

```text
MOS archive landing page: https://mesonet.agron.iastate.edu/mos/
MOS download interface:   https://mesonet.agron.iastate.edu/mos/fe.phtml
AFOS retrieve API help:   https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?help=
```

IEM states that it maintains an interactive MOS archive for research and that the archive goes back to June 2000 for some products. IEM also states that raw text identifiers generally take the form `<model><3-char station>`, e.g. `LAVDSM`; for KLGA this means products like `MAVLGA`, `METLGA`, `MEXLGA`, `LAVLGA`, `NBSLGA`, and `NBELGA`.

## 2. Required products

Codex must fetch and parse the following MOS products for every station in the station registry that has a MOS station id. KLGA/LGA is mandatory; nearby core stations are mandatory; regional stations are strongly required unless IEM lacks the station/product combination.

| Product family | PIL prefix | Example for KLGA | Model/system | Status/use |
|---|---|---|---|---|
| GFS MOS short-range | `MAV` | `MAVLGA` | GFS MOS | Mandatory long-history source. |
| NAM MOS short-range | `MET` | `METLGA` | NAM MOS | Mandatory where available since 2008. |
| GFS MOS extended | `MEX` | `MEXLGA` | GFS extended MOS | Mandatory where available since 2020. |
| GFS LAMP | `LAV` | `LAVLGA` | LAMP-style guidance | Mandatory for short-range/late cutoff features. |
| National Blend MOS short-range | `NBS` | `NBSLGA` | NBM/MOS text product | Mandatory where available. |
| National Blend ensemble/extended | `NBE` | `NBELGA` | NBM ensemble/extended text product | Mandatory where available. |

Historical availability varies by product. Codex must store per-product `first_available_run_time_utc` and `last_available_run_time_utc` after backfill.

## 3. Required station list

Use the canonical station registry. MOS products use three-character station suffixes.

Mandatory target/core:

```text
LGA, NYC, JFK, EWR, TEB, HPN, ISP, FRG, BDR
```

Mandatory regional if available:

```text
SWF, POU, MMU, CDW, PHL, BOS, DCA, BWI, ALB, ABE
```

For each station, expected PILs are:

```python
for prefix in ["MAV", "MET", "MEX", "LAV", "NBS", "NBE"]:
    pil = f"{prefix}{mos_station_id}"
```

If a PIL is not available for a station/product, write a `source_gap` row:

```text
source_name = iem_mos
source_product = prefix
station_id = canonical station
missing_reason = provider_no_product_for_station
```

Do not fail the whole ingestion because one regional station lacks a product.

## 4. Required variables to parse

MOS text/table products differ by model. Codex must parse all rows present, but these variables are required when present:

| MOS variable | Meaning | Canonical variable name | Required use |
|---|---|---|---|
| `N/X`, `X/N`, `n_x`, `MAX`, `MIN` | Max/min temperature forecast, °F | `mos_max_min_temp_f` | Primary source for target-day Tmax. |
| `TMP` | Temperature, °F | `mos_tmp_f` | Hourly/period temperature curve. |
| `DPT` | Dew point, °F | `mos_dpt_f` | Humidity/air-mass feature. |
| `WDR` | Wind direction | `mos_wind_dir` | Wind regime/sea-breeze feature. |
| `WSP` | Wind speed | `mos_wind_speed` | Mixing/marine influence. |
| `GST` | Gust | `mos_wind_gust` | Boundary-layer/mixing proxy. |
| `CLD`, `SKY` | Cloud category/sky cover | `mos_cloud` / `mos_sky_pct` | Cloud bust feature. |
| `P06`, `P12`, `P24` | Probability of precipitation | `mos_pop_6h_12h_24h` | Precip/cloud risk. |
| `Q06`, `Q12`, `Q24` | QPF category/amount | `mos_qpf_6h_12h_24h` | Precip/cloud risk. |
| `T03`, `T06`, `T12`, `T24` | Thunderstorm probability | `mos_tstm_prob` | Convective bust risk. |
| `SOL` | Solar radiation if present | `mos_solar` | Radiation/high-temp support. |
| `MHT` | Mixing height if present | `mos_mixing_height` | Boundary-layer feature. |
| `LCB` | Lowest cloud base if present | `mos_low_cloud_base` | Cloud regime feature. |
| `TXN`, `XND`, `TSD` | MOS uncertainty/probability fields when present | `mos_uncertainty_field` | Probabilistic calibration. |

The parser must preserve the raw token name and raw value even if a canonical mapping is not known.

## 5. Acquisition methods

Codex must implement two retrieval modes.

### 5.1 Preferred raw-text retrieval through AFOS API

Use the IEM AFOS retrieve API for raw text products.

Endpoint:

```text
GET https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py
```

Required query parameters:

```text
pil={PIL}                    # e.g. MAVLGA
sdate={YYYY-MM-DDTHH:MM:SSZ}
edate={YYYY-MM-DDTHH:MM:SSZ}
fmt=text
limit=9999
order=asc
```

Example for KLGA GFS MOS short-range raw text:

```text
https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=MAVLGA&sdate=2024-06-01T00:00:00Z&edate=2024-06-02T00:00:00Z&fmt=text&limit=9999&order=asc
```

Bronze storage must persist the returned raw text exactly.

### 5.2 Secondary table/CSV retrieval

Use the MOS download/table interface when it can provide easier structured output for variable-by-run analysis. Codex must still keep raw responses. If table output and raw AFOS output disagree, raw AFOS text is the source of record and the disagreement must be logged.

## 6. Required run cycles

Fetch every available cycle for each product, not only cycles near the desired cutoff. This allows run-to-run trend features and cycle-specific bias estimation.

Canonical cycles by product:

```text
MAV: all available cycles, normally 00/06/12/18 UTC
MET: all available cycles, normally 00/06/12/18 UTC
MEX: all available cycles available from IEM
LAV: 00/06/12/18 UTC where archived by IEM
NBS/NBE: use archived cycle schedule by date; IEM documents changes around 2020 and 2026
```

Codex must parse the actual issue/run timestamp from the product text. Do not infer cycle solely from request time.

## 7. Required historical range

Backfill by product from the earliest IEM availability:

```text
AVN legacy: 2000-06-01 through 2003-12-16 if useful for pre-GFS continuity, optional after initial system.
GFS MAV: 2003-12-16 through realtime, mandatory.
NAM MET: 2008-12-09 through realtime, mandatory.
GFS LAMP LAV: 2020-07-12 through realtime, mandatory.
GFS Extended MEX: 2020-07-12 through realtime, mandatory.
NBS/NBE: all available periods, mandatory.
```

Initial production backfill priority:

```text
1. KLGA all mandatory products, all available dates.
2. Nearby core stations all mandatory products, all available dates.
3. Regional context stations all mandatory products, all available dates.
```

## 8. MOS period-to-market-day mapping

This is critical and must be handled explicitly.

For each MOS value, Codex must compute:

```text
run_time_utc
forecast_period_start_utc
forecast_period_end_utc
forecast_valid_time_utc        # if point forecast
period_type                    # point, 3h, 6h, 12h, 24h, max_min_period
mos_projection_hour
```

For target date `T`:

```text
market_local_start = T 00:00:00 America/New_York
market_local_end   = T 23:59:59 America/New_York
```

A MOS max/min forecast is linked to target date `T` only if its forecast period overlaps the market local day and its label convention is understood. Codex must store:

```text
mos_target_mapping_method
mos_period_overlap_fraction
mos_period_start_local
mos_period_end_local
```

If a MOS max/min period spans 12Z-to-12Z or another non-calendar period, do not pretend it equals the Wunderground local-day high. Instead, create features such as:

```text
mos_max_temp_for_overlapping_period
mos_max_temp_period_overlap_fraction
mos_max_temp_period_midpoint_local_hour
```

The model will learn the mapping bias.

## 9. Silver table schema

```text
CREATE TABLE iem_mos_values (
    source_name TEXT NOT NULL DEFAULT 'iem_mos',
    source_product TEXT NOT NULL,                 -- MAV, MET, MEX, LAV, NBS, NBE
    pil TEXT NOT NULL,                            -- e.g. MAVLGA
    station_id TEXT NOT NULL,                     -- canonical, e.g. KLGA
    mos_station_id TEXT NOT NULL,                 -- e.g. LGA
    run_time_utc TIMESTAMP NOT NULL,
    issue_time_utc TIMESTAMP,
    parsed_cycle_utc INTEGER,
    forecast_valid_time_utc TIMESTAMP,
    forecast_period_start_utc TIMESTAMP,
    forecast_period_end_utc TIMESTAMP,
    forecast_hour DOUBLE PRECISION,
    period_type TEXT NOT NULL,
    raw_variable_name TEXT NOT NULL,
    canonical_variable_name TEXT,
    unit_original TEXT,
    value_original TEXT,
    unit_canonical TEXT,
    value_canonical DOUBLE PRECISION,
    row_position INTEGER,
    token_position INTEGER,
    raw_product_text_hash TEXT NOT NULL,
    provider_available_at_utc TIMESTAMP,
    our_ingested_at_utc TIMESTAMP NOT NULL,
    availability_method TEXT NOT NULL,
    source_request_id TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    quality_flag TEXT NOT NULL DEFAULT 'ok',
    quality_note TEXT
);
```

## 10. Availability rules

Preferred:

```text
provider_available_at_utc = parsed product issue time + 15 minutes
availability_method = parsed_product_issue_time
```

Fallback if issue time cannot be parsed:

```text
provider_available_at_utc = inferred_run_time_utc + 2 hours
availability_method = conservative_lag_rule
```

Live ingestion must store actual `our_first_seen_at_utc` when the product is detected. Once live logs exist, actual ingestion time overrides the historical conservative fallback.

## 11. Gold features from MOS

For each target date and cutoff, build features separately by product, station, cycle, and lead group.

### 11.1 Target station direct features

```text
mos_{product}_KLGA_latest_eligible_max_temp_f
mos_{product}_KLGA_latest_eligible_min_temp_f
mos_{product}_KLGA_latest_tmp_peak_window_max_f
mos_{product}_KLGA_latest_tmp_at_16z_17z_18z_19z_20z_21z_22z_23z
mos_{product}_KLGA_latest_dpt_peak_window_mean_f
mos_{product}_KLGA_latest_wind_dir_peak_window_circular_mean
mos_{product}_KLGA_latest_wind_speed_peak_window_mean
mos_{product}_KLGA_latest_cloud_risk
mos_{product}_KLGA_latest_pop_max
mos_{product}_KLGA_latest_qpf_max
mos_{product}_KLGA_latest_tstm_prob_max
```

### 11.2 Run-trend features

```text
mos_{product}_KLGA_max_temp_latest_minus_prev_cycle
mos_{product}_KLGA_tmp_peak_latest_minus_prev_cycle
mos_{product}_KLGA_latest_minus_24h_prior_run
mos_{product}_KLGA_run_to_run_std_last_4_cycles
```

### 11.3 Cross-station features

```text
mos_{product}_KEWR_minus_KLGA_max_temp
mos_{product}_KJFK_minus_KLGA_max_temp
mos_{product}_KNYC_minus_KLGA_max_temp
mos_{product}_inland_mean_minus_KLGA_max_temp
mos_{product}_coastal_mean_minus_KLGA_max_temp
mos_{product}_southwest_upstream_mean_minus_KLGA_max_temp
mos_{product}_backdoor_front_mean_minus_KLGA_max_temp
```

### 11.4 Product disagreement features

```text
mos_MAV_minus_MET_KLGA_max_temp
mos_MAV_minus_NBS_KLGA_max_temp
mos_NBE_minus_NBS_KLGA_max_temp
mos_LAV_short_range_minus_NBM_or_MAV_if_same_valid_period
mos_product_spread_KLGA_max_temp = std(latest eligible product max forecasts)
```

## 12. Parser requirements

Codex must implement MOS parsing as a resilient parser, not a brittle one-off split.

Mandatory parser behavior:

```text
1. Split raw AFOS response into individual products using WMO/AWIPS headers and timestamp blocks.
2. Identify station section matching the MOS station id, e.g. LGA.
3. Parse run/issue time from product header.
4. Parse the forecast-hour/date/time header row.
5. Parse every variable row into individual token values.
6. Convert missing markers such as M, X, -, blank into null with quality flag.
7. Preserve raw variable names and raw text positions.
8. Emit a warning, not silent failure, when a row has unexpected token count.
9. Unit-test against at least one sample product for each product family.
```

If a parser cannot map exact valid times for a product, it must still store raw values with `period_type = unknown` and `quality_flag = needs_parser_mapping`. Such values must not enter gold features until mapping is fixed.

## 13. Rate limiting and retries

IEM services must be used politely.

```text
Minimum delay between IEM requests from same IP: 1.1 seconds.
Maximum retry attempts: 5.
Retry backoff: exponential starting at 2 seconds, jittered.
HTTP 503: retry.
HTTP 422: split request into smaller date ranges.
HTTP 404/no product: record source_gap, do not retry endlessly.
```

## 14. Acceptance tests

```text
[ ] For KLGA, MAVLGA, METLGA, MEXLGA, LAVLGA, NBSLGA, NBELGA are attempted over their full available periods.
[ ] At least one product per mandatory family is parsed into normalized rows when provider data exists.
[ ] MOS run_time_utc and provider_available_at_utc are populated for every row entering gold features.
[ ] MOS max/min period mapping is explicit; no calendar-day assumption is hidden.
[ ] Latest eligible MOS features differ by cutoff when newer products become available.
[ ] MOS product disagreement and run-trend features are generated.
[ ] Every source gap is recorded with station, product, date range, and reason.
```
