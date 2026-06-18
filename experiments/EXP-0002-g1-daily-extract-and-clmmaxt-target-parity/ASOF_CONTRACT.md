# As-of Contract

## Forecast cutoff

- timezone: Asia/Hong_Kong
- horizon ID: not selected; G2 remains open
- local expression: not applicable to G1 target truth
- UTC expression: not applicable
- grace/latency rule: no feature rows are generated; source availability is
  represented by archived `retrieved_at` and, when extractable, provider
  publication/update fields

## Feature eligibility

Every feature row must satisfy:

```text
available_at <= forecast_cutoff
```

## Source-specific timing

| Source ID | Valid/issue timestamp | Availability evidence | Conservative latency | Revision behavior | Eligible? |
|---|---|---|---:|---|---|
| hko_daily_extract | Hong Kong local calendar date in Daily Extract page | raw archived page/payload, retrieval time, provider fields if present | first-publication timing must be observed; no default latency | later values may revise; first payload remains canonical candidate | target evidence only |
| hko_clmmaxt_hko | local date row in HKO climate CSV | raw archived CSV and retrieval time | latest history is not proof of historical availability | proxy/finalized history; revisions possible | parity proxy only |
| hko_open_data_catalog | provider page/document timestamps where present | raw archived page/PDF and retrieval time | not used as forecast feature | provider docs may change | source-contract evidence only |
| hko_station_metadata | station identity and metadata effective dates where present | raw archived page/payload and retrieval time | not used as forecast feature | provider metadata may change | station-contract evidence only |

## Explicitly forbidden data

Forbidden in G1:

- any predictive features or model fitting;
- any Polymarket backtesting, price history, books, trades, liquidity,
  execution, or market replay artifacts;
- any claim that latest CLMMAXT was available at historical forecast cutoffs;
- any inference that integer outcome labels imply nearest-integer rounding;
- any current/latest Daily Extract page treated as historical first publication
  without archived first-publication timing;
- reanalysis, final best tracks, future model cycles, and target-day realized
  observations as features.

## Preprocessing timing

No fitting, imputation, scaling, feature selection, calibration, or regime
classification is performed. Parsing and equality checks are deterministic.

## Automated checks

- `.venv\Scripts\python.exe -m hkg_tmax validate all`
- `.venv\Scripts\python.exe -m pytest`
- fail-closed target adapter tests for unknown rules, missing field,
  ambiguous date, unsupported precision, source failure, bucket overlaps, and
  bucket gaps
- raw archive sidecar/hash verification for every G1 source input

## Residual uncertainty

Historical first-publication Daily Extract payloads may not be recoverable for
all resolved market dates. Such dates cannot prove first-publication parity and
must be labelled `MISSING_FIRST_PUBLICATION` or otherwise quarantined. Latest
Daily Extract versus CLMMAXT equality can support proxy confidence but cannot
alone prove first-publication semantics.
