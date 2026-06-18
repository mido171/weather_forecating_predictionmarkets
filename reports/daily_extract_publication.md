# Daily Extract Publication Ledger

Generated for HKO Daily Extract `2026-06`.

## Gate Status

**G1 remains blocked.** This run proves polling/ledger mechanics and archive-first-observed evidence. It does not by itself prove provider first publication.

## Latest Poll

- catalog hash: `f80772b68545c56e6842c34998696fd11b7b9a80c0088bb1f6e4da65102616eb`
- catalog retrieved_at: `2026-06-18T20:33:49.134961Z`
- monthly source: `hko_daily_extract_202606`
- monthly hash: `c50910ab74e2ba8bff1f661fb1ae663d15b128dae0dfb4ed97c0e40c97bcbefc`
- monthly retrieved_at: `2026-06-18T20:33:50.378189Z`

## Ledger Summary

- row count: `17`
- evidence counts: `{'ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST': 17}`
- revision count: `0`
- provider first publication proven: `False`
- poll snapshot count: `6`
- fetch attempts per request: `3`
- retry sleep seconds: `2.0`
- active polling start: `2026-06-18T17:48:59.956593Z`
- watched candidate dates: `['2026-06-18']`
- watched candidate dates present: `[]`
- watched candidate dates missing: `['2026-06-18']`

## Artifacts

- ledger CSV: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\gold\target_publication\daily_extract_first_seen.csv`
- metrics JSON: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\experiments\EXP-0024-g1-daily-extract-active-first-publication-watch-2026-06-18-ninet\results\metrics.json`

## Limitations

- Rows already visible before active polling are only first observed by this archive.
- Provider first-publication candidate status requires active absent-before-present evidence.
- No predictive modelling or market backtesting was run.
