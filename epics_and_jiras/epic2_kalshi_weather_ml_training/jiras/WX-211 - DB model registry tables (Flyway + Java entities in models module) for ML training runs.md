# WX-211 — DB model registry tables (Flyway + Java entities in models module) for ML training runs

## Objective
Add DB support to track training runs and locate artifacts.

## Tables (minimum)
### ml_training_run
- run_id (PK, string)
- created_at_utc
- code_version (git hash)
- config_hash
- dataset_id
- stations
- asof_policy_id
- status (SUCCESS/FAILED)
- notes

### ml_model_artifact
- run_id (FK)
- station_id
- artifact_type (MEAN_MODEL, SIGMA_MODEL, CALIBRATOR, METRICS, REPORT)
- artifact_path (filesystem path or s3 url)
- sha256
- created_at_utc
- UNIQUE(run_id, station_id, artifact_type)

## Requirements
- Flyway migration lives in Java `models` module (per repo rule).
- JPA entities/enums live in `models` module.

## Acceptance Criteria
- [ ] Schema migration applies cleanly.
- [ ] JPA entities compile.
- [ ] Java module tests pass and root `mvn clean install` passes.

