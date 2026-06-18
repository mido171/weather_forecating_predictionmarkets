# As-of Contract

## Forecast cutoff

- timezone: Asia/Hong_Kong
- horizon ID: not selected in G0
- local expression: not applicable
- UTC expression: not applicable
- grace/latency rule: no forecast rows are built; raw source availability is
  represented by archive `retrieved_at`

## Feature eligibility

Every feature row must satisfy:

```text
available_at <= forecast_cutoff
```

## Source-specific timing

| Source ID | Valid/issue timestamp | Availability evidence | Conservative latency | Revision behavior | Eligible? |
|---|---|---|---:|---|---|
| hko_daily_extract | provider page publication not yet parsed | archive `retrieved_at` and HTTP metadata | not modelled | target-only page may revise; G1 must audit first publication | raw archive only |
| hko_clmmaxt_hko | historical local date rows after parsing | archive `retrieved_at` and HTTP metadata | not modelled | proxy with limitations; not canonical until G1 | raw archive only |
| hko_open_data_catalog | provider catalog page time not parsed | archive `retrieved_at` and HTTP metadata | not modelled | metadata page may change | raw archive only |
| hko_latest_1min_temperature | payload-specific source timestamp not parsed in G0 | archive `retrieved_at` and HTTP metadata | not modelled | latest-only provisional observations | raw archive only |
| hko_since_midnight_maxmin | payload-specific source timestamp not parsed in G0 | archive `retrieved_at` and HTTP metadata | not modelled | latest-only provisional observations | raw archive only |
| hko_local_weather_forecast | HKO issue/update time not parsed in G0 | archive `retrieved_at` and HTTP metadata | not modelled | each issue is a separate vintage | raw archive only |
| hko_nine_day_forecast | HKO issue/update time not parsed in G0 | archive `retrieved_at` and HTTP metadata | not modelled | each issue is a separate vintage | raw archive only |

## Explicitly forbidden data

G0 forbids all predictive feature construction, target labels, target-day
realized observations as features, future forecast cycles, reanalysis, final
tropical-cyclone best tracks, corrected historical files as operational
vintages, and any locked-test outcome inspection.

## Preprocessing timing

No fitting, imputation, scaling, feature selection, calibration, or regime
classification is performed.

## Automated checks

- `.venv\Scripts\python.exe -m hkg_tmax validate all`
- `.venv\Scripts\python.exe -m pytest`
- `.venv\Scripts\python.exe -m pytest tests/test_fetch.py tests/test_hko.py`
- archive verification script recomputing hashes from sidecars and checking HTTP
  metadata for all `bootstrap_now` retrievals

## Residual uncertainty

G0 records only retrieval-time availability. Provider-declared issue,
publication, and first-appearance semantics are intentionally left unresolved
for G1/G2 and must not be inferred from this smoke test.
