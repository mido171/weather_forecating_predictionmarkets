# System Architecture

## Logical flow

```text
official sources / model files / market feeds
                  │
                  ▼
        immutable raw snapshot store
                  │
                  ▼
      source-specific bronze parsers
                  │
                  ▼
 quality + station + timestamp resolution
                  │
                  ▼
       point-in-time silver datasets
                  │
                  ▼
     forecast-example builder by cutoff
                  │
                  ▼
 baselines / experiments / champion models
                  │
                  ▼
 continuous distribution + bucket adapter
                  │
                  ▼
 meteorological evaluation / market replay
                  │
                  ▼
  live shadow, monitoring, risk/no-trade gate
```

## Core identifiers

Use stable identifiers:

- `source_id`
- `station_id` plus metadata version
- `model_id`
- `model_cycle`
- `member_id`
- `target_local_date`
- `horizon_id`
- `rules_sha256`
- `dataset_version`
- `feature_set_version`
- `model_version`
- `prediction_id`
- `experiment_id`

## Raw sidecar schema

```json
{
  "source_id": "...",
  "requested_url": "...",
  "request_method": "GET",
  "request_headers_redacted": {},
  "retrieved_at": "...Z",
  "http_status": 200,
  "response_headers": {},
  "content_sha256": "...",
  "content_length": 123,
  "adapter_version": "..."
}
```

## Point-in-time record schema

```text
entity_id
variable
value
unit
valid_at
issued_at
published_at
available_at
retrieved_at
source_id
raw_sha256
parser_version
quality_state
```

## Forecast example schema

```text
target_local_date
horizon_id
cutoff_at
target_value
target_first_available_at
rules_hash
feature_manifest_hash
max_feature_available_at
model_version
prediction_created_at
```

Target value is separated during prediction generation and inaccessible to feature code.

## Technology choices

Bootstrap uses:

- Python 3.11+;
- YAML configuration;
- HTTPX;
- raw files plus sidecars;
- Parquet/Zarr/NetCDF for later analytical data;
- pytest;
- Git;
- Codex AGENTS, skills, and subagents.

For scale, add a catalog/database such as DuckDB/PostgreSQL and object storage, but preserve the same contracts.

## Isolation

Recommended processes:

- source pollers;
- NWP backfill/archive workers;
- market WebSocket recorder;
- transformation jobs;
- forecast runner;
- scorer;
- dashboard/alerts.

A source failure must not corrupt unrelated archives.

## Determinism

- UTC internally, HKT only for local target/calendar semantics;
- timezone-aware datetimes;
- sorted input manifests;
- stable hashes;
- fixed seeds;
- explicit dependency environment;
- no mutable “latest” paths as experiment inputs.

## Security

- secrets in environment or secret store;
- redact auth headers;
- least-privilege API keys;
- read-only review agents;
- no trading credentials in research jobs;
- checksums and backups.
