# Database Schema and Lineage Design

## Storage principle

Use PostgreSQL for catalog, run metadata, selected point values, derived features, scores, and live forecasts. Store large raw API responses and gridded patches as immutable compressed NDJSON/Parquet objects with database manifests and checksums. Do not force every gridded value into one enormous unpartitioned SQL table.

## Schemas

```text
catalog       source, model, selector, location, station, object registry
governance    cutoff, availability contracts, sealed periods, eligibility audits
raw_audit     immutable source rows and request/response manifests
nwp_core      model runs, point values, trajectory values, ensemble values
feature_store feature definitions, snapshot manifests, feature matrices
research      OOF predictions, scores, router/specialist diagnostics
live          issued forecasts, expert outputs, weights, state updates
quarantine    invalid rows, parser failures, impossible values, blocked requests
```

## Core keys

- Model run natural key: `(provider, model_code, run_time_utc, model_version)`.
- Forecast value natural key: `(model_run_id, valid_time_utc, location_id, variable_id, member_number)`.
- Official HKO vintage natural key: preserve the existing primary key plus issue time, target date, and product/parser identity.
- Target snapshot natural key: `(target_date, cutoff_contract_version, snapshot_builder_version)`.

## Canonical time columns

Use `timestamptz` UTC for all timestamps. Store a local date only when it is a true Hong Kong calendar date. Never store an HKT wall time in a UTC-labelled column. Derived HKT views must use `AT TIME ZONE 'Asia/Hong_Kong'`.

## Partitioning

Partition high-volume NWP tables by model and run month. Index:

```text
(model_code, run_time_utc)
(model_run_id, valid_time_utc)
(location_id, valid_time_utc)
(variable_id, valid_time_utc)
(target_date, cutoff_utc)
```

For ensembles, include member in the unique index.

## Required lineage

Every derived feature and prediction must point to:

- source row/object hashes;
- request and response IDs;
- selector snapshot;
- code commit;
- config hash;
- cutoff contract version;
- feature definition version;
- model artifact hash;
- training-data manifest hash.

## Access control

- development role cannot read sealed 2024+ labels;
- live inference role cannot read target labels, residuals, research scores, diagnostic-only sources, or quarantine;
- diagnostic role cannot publish production features;
- only governance/admin can promote an availability contract.

The reference SQL file `schemas/REFERENCE_POSTGRES_SCHEMA.sql` provides concrete DDL concepts. Codex must adapt it to existing repository migration conventions rather than blindly applying a parallel schema.
