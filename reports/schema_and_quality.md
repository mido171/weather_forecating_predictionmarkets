# Schema and Quality

No model features are built. Current QC is limited to retrieval integrity, source-native bronze parsing, content length, hashes, and schema/version metadata.

- bronze dataset metadata files: `5`
- bronze sources: `hko_clmmaxt_hko, hko_latest_1min_temperature, hko_local_weather_forecast, hko_nine_day_forecast, hko_since_midnight_maxmin`

| Check | Status |
|---|---|
| content-addressed raw dedupe | implemented |
| retrieval ledger per attempt | implemented |
| first HKO bronze rebuilds | implemented for acquired CLMMAXT, live temperature, since-midnight max/min, local forecast, and nine-day forecast |
| station-level schema validation | pending deeper source-specific adapters |
| silver/gold rebuilds | pending after source-specific bronze QA |
