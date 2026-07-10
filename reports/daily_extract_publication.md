# Daily Extract Publication Ledger

Generated for HKO Daily Extract `2026-07`.

## Gate Status

**G1 remains blocked.** This run proves polling/ledger mechanics and archive-first-observed evidence. It does not by itself prove provider first publication.

## Latest Poll

- catalog hash: `9e0a32fb9e1aeb0fed55f1d8be897b8a25589e6656aa6264923eab0b5c0dfe9d`
- catalog retrieved_at: `2026-07-05T17:37:32.994597Z`
- monthly source: `hko_daily_extract_202607`
- monthly hash: `00b950c055bb80985e892edb94550a08e619733f8bbea4960773f4a8ca2c42e9`
- monthly retrieved_at: `2026-07-05T17:37:33.681712Z`

## Ledger Summary

- row count: `2`
- evidence counts: `{'ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST': 2}`
- revision count: `0`
- provider first publication proven: `False`
- poll snapshot count: `1`
- fetch attempts per request: `3`
- retry sleep seconds: `2.0`
- active polling start: `2026-07-05T17:37:31.911151Z`
- watched candidate dates: `['2026-07-06']`
- watched candidate dates present: `[]`
- watched candidate dates missing: `['2026-07-06']`

## Artifacts

- ledger CSV: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\_pipeline_internal\gold\target_publication\daily_extract_first_seen.csv`
- metrics JSON: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\reports\generated\daily_extract_publication_metrics.json`

## Limitations

- Rows already visible before active polling are only first observed by this archive.
- Provider first-publication candidate status requires active absent-before-present evidence.
- No predictive modelling or market backtesting was run.
