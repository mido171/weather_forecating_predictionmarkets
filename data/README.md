# Data Directory

Raw and derived data are intentionally absent from this bootstrap ZIP. Third-party data may have licensing and redistribution constraints, and current data should be fetched directly from authoritative providers.

## Layers

- `raw/` — exact immutable payloads plus sidecars;
- `bronze/` — source-native parsed data;
- `silver/` — quality-controlled, normalized, point-in-time data;
- `gold/` — targets, forecast examples, predictions, evaluation tables;
- `cache/` — disposable download/compute cache.

Do not commit bulk data to Git. Preserve it in durable storage and commit manifests/hashes.

## Raw invariant

Never edit a raw payload. If a source changes, archive a new payload.

## Derived invariant

Every derived dataset must identify:

- raw input hashes;
- parser/transform version;
- config hash;
- schema;
- build time;
- row count;
- quality summary.
