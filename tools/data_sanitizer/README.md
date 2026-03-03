# Data Sanitizer

`data_sanitizer` cleans raw 30-minute observation CSVs without mutating source files.

## CLI

```bash
python -m tools.data_sanitizer.data_sanitizer \
  --input data/raw/observations_30m_required_columns.csv \
  --output data/clean/observations_30m_required_columns.sanitized.csv.gz \
  --report-out data/clean/observations_30m_required_columns.sanitization_report.json \
  --samples-out data/clean/observations_30m_required_columns.sanitization_samples.csv \
  --station-universe data/raw/station_universe.csv \
  --schema-profile training_data_complete_schema_profile.json \
  --config tools/data_sanitizer/default_rules.yaml \
  --chunksize 250000 \
  --compression gzip \
  --emit-flags false \
  --drop-invalid-timestamps true \
  --dedupe-policy best_non_null \
  --strict-columns true
```

## Outputs

- `<output>.csv.gz`: sanitized observations.
- `*.sanitization_report.json`: machine-readable rule counts and station stats.
- `*.sanitization_samples.csv`: triggered-row samples with `triggered_rules`.
- `data_sanitizer_manifest.jsonl`: append-only run log with hashes.

## In-memory Invocation

For model training loaders, call:

- `tools.data_sanitizer.data_sanitizer.sanitize_observations_dataframe(...)`

This applies sanitation during CSV read, leaving original CSV files untouched.
