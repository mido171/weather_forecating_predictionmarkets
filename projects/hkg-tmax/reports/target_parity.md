# Target Parity Report

Generated for HKO Daily Extract 2026-05 against CLMMAXT HKO.

## Gate status

**G1 is not passed.** This table proves latest-payload equality for the archived May 2026 sample only. It does not prove first-publication parity.

## Sources

- Daily Extract: `hko_daily_extract_202605`
  - retrieved_at: `2026-06-18T17:12:45.346959Z`
  - sha256: `a97230cd78e0a11c4455c23288c96542ec3c13584071619750900a385146dc95`
  - path: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\raw\hko_daily_extract_202605\2026\06\18\20260618T171245.346959Z__a97230cd78e0a11c.xml`
  - final_url: `https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_202605.xml`
- CLMMAXT HKO: `hko_clmmaxt_hko`
  - retrieved_at: `2026-06-18T16:05:52.249124Z`
  - sha256: `5a0a646b4d125e40c25871abbccd5cd24e4f552063a547803ebac820166be4c9`
  - path: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\raw\hko_clmmaxt_hko\2026\06\18\20260618T160552.249124Z__5a0a646b4d125e40.csv`
  - final_url: `https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO`

## Results

- parity rows: `31`
- compared rows: `31`
- latest-payload matches: `31`
- latest-payload mismatches: `0`
- latest-payload match rate: `1.0`

## Limitations

- `daily_extract_first_value` is intentionally blank because first-publication capture has not yet been observed for these historical dates.
- Polymarket backtesting, price history, order books, trades, liquidity, execution, and market replay are deferred by user instruction.
- CLMMAXT remains a proxy until first-publication Daily Extract parity is proven.

## Artifacts

- parity CSV: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\data\gold\target_parity\target_parity.csv`
- metrics JSON: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\experiments\EXP-0002-g1-daily-extract-and-clmmaxt-target-parity\results\metrics.json`
