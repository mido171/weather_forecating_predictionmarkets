# KLGA Tmax 08 - IEM MOS Backfill Implementation Deep Dive

## Executive Summary

This document records the implemented and executed KLGA IEM MOS backfill for the `T_1245UTC` cutoff workflow.

The implementation added a Java/Spring Boot CLI runner under `ingestion-service` that fetches structured IEM MOS JSON from:

```text
https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py
```

The runner fetched six MOS products for the 19 canonical KLGA station-network stations, persisted raw request lineage and compact wide forecast rows into the KLGA Postgres research database, and then materialized leakage-safe daily MOS features and a model-facing feature matrix.

Final executed job:

```text
job_id: klga_iem_mos_full_backfill_v1
cutoff_id: T_1245UTC
database: postgresql://postgres:root@127.0.0.1:5432/klga_tmax_research
products: MAV, MET, MEX, LAV, NBS, NBE
stations: 19
chunks completed: 1,330 / 1,330
failed chunks: 0
source gaps: 0
wide MOS forecast rows: 33,355,830
daily MOS feature rows: 444,105
feature matrix rows: 8,231
late or leaky gold features: 0
raw/source lineage bytes: 12,146,804,565
```

Primary architectural decision: use Java for the concurrent IEM MOS fetch runner and Postgres for persistence, but store structured MOS rows in a compact wide table instead of exploding every MOS variable token into a much larger atomic-value table. Gold features are built from the persisted rows using the strict `effective_available_at_utc <= cutoff_utc` predicate, never runtime alone.

Verification status: the full backfill completed, no active runner processes remain, source gap count is zero, leakage check found zero late features, focused Java tests passed, the KLGA schema contract test passed, and the feature materialization script compiles.

## Reader Orientation and Document Map

Read this document if you need to understand, rerun, validate, debug, or extend the KLGA IEM MOS data build.

Sections:

- Scope Boundaries: what this implementation includes and excludes.
- Source-of-Truth Inputs: exact code, DB, logs, and commands used as evidence.
- Requirements-to-Implementation Traceability: how each user requirement was fulfilled.
- Change Inventory: every changed or added file and why it exists.
- Architecture and Control Flow: runner, parser, persistence, and feature materialization flow.
- Data Model, Persistence, and Migration Notes: new Postgres tables, indexes, row shapes, and leakage columns.
- Executed Backfill Results: product/station/date coverage and final row counts.
- Testing and Verification Evidence: commands and DB checks that prove the state.
- Operational Runbook: commands to rerun fetches, materialize features, inspect progress, and recover from interruption.
- Known Limitations and Follow-Up Work: what is intentionally deferred.

## Scope Boundaries

Included:

- Structured IEM MOS JSON fetching from `request/mos.py`.
- Products `MAV`, `MET`, `MEX`, `LAV`, `NBS`, and `NBE`.
- The 19 canonical KLGA station-network MOS stations.
- Annual chunk planning, resumable tracking, 10 worker threads, and global request-start spacing.
- Raw compressed payload storage plus source request/source record lineage.
- Wide structured forecast-row persistence in `silver.iem_mos_forecast_rows`.
- Leakage-safe daily feature materialization in `gold.iem_mos_daily_features`.
- Rebuildable matrix rows in `gold.iem_mos_feature_matrix_v1`.
- DB schema migration, schema contract check, focused Java tests, and final live DB validation.

Excluded or deferred:

- Raw AFOS issue-time enrichment. The v1 availability method uses `runtime_utc + 2 hours` as the conservative provider-availability rule.
- MySQL MOS pilot persistence. The production target for this task is Postgres `klga_tmax_research`.
- Per-token variable explosion into a very large atomic table. The implemented default stores wide MOS rows plus raw JSON for additional fields.
- Trading model training or scoring. This task produced the MOS feature corpus, not the final forecasting model.
- Documentation of unrelated HKG worktree changes present in the same repository. Those files were already dirty and are outside this KLGA task.

## Source-of-Truth Inputs

Sources used for this report:

- User implementation request: KLGA IEM MOS leakage-safe Java backfill plan.
- Existing KLGA repository contracts under `bootstrap/klga_tmax`.
- Implemented Java source under `ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos`.
- Implemented Postgres migration `bootstrap/klga_tmax/implementation/alembic/versions/0008_iem_mos_backfill.py`.
- Implemented materialization script `bootstrap/klga_tmax/implementation/scripts/materialize_iem_mos_features.py`.
- Live Postgres validation queries against `klga_tmax_research`.
- Runner logs under `bootstrap/klga_tmax/implementation/artifacts/klga_tmax/iem_mos/logs`.
- Focused Java, Python schema, and Python compile checks run after completion.

## Requirements-to-Implementation Traceability

| Requirement | Implementation Location | Delivered Behavior | Verification Evidence | Caveat |
|---|---|---|---|---|
| Fetch structured IEM MOS data | `IemMosHttpClient`, `IemMosBackfillService` | Calls `/cgi-bin/request/mos.py?station={station}&model={model}&sts={start}&ets={end}&format=json` | 1,330 completed chunks, 33,355,830 rows | Uses structured JSON only, not raw AFOS text |
| Use Java/Spring Boot runner | `IemMosBackfillApplication`, `IemMosBackfillCommandLineRunner` | Narrow CLI app starts only the KLGA MOS runner | Focused Maven tests passed; live job executed | Runner launched via `exec:java`, not old app main |
| Use Postgres KLGA DB, not MySQL | `application-klga-iem-mos.yml`, migration 0008 | JDBC points at `klga_tmax_research`; tables are under Postgres schemas | Live DB validation counts | Existing MySQL pilot tables untouched |
| Support MAV, MET, MEX, LAV, NBS, NBE | `IemMosProduct`, `MosModel` | Product registry maps each source product to endpoint model and default start | Final rows by all six products | Product availability varies by start date |
| Use 19 KLGA station-network stations | `IemMosBackfillRepository.loadMosStations` | Loads MOS stations from `registry.station_registry` | Final station list has 19 IDs | Depends on station registry seed |
| Use `T_1245UTC` cutoff | `materializeTargetInstances`, feature SQL | `cutoff_utc = target_date 12:45:00 UTC` | 8,231 matrix rows for `T_1245UTC` | No other cutoff materialized in this run |
| Enforce non-forward-looking features | `IemMosParser`, feature materializer | `effective_available_at_utc = runtime + 2h`; feature selection requires `<= cutoff_utc` | `late_features = 0` | AFOS issue-time proof deferred |
| Track chunks and resume safely | `audit.iem_mos_backfill_chunks`, `IemMosBackfillRepository` | One row per station/product/window with request hash and terminal status | 1,330 completed, 0 running, 0 failed | Manual stale-running reset was needed after user-requested process stop |
| Store raw payloads and lineage | `bronze.source_requests`, `bronze.source_records`, raw gzip paths | Every completed chunk has source request/record IDs and raw storage URI | 12.15 GB response lineage tracked | Raw gzip files live in local artifacts path |
| Avoid row explosion by default | `silver.iem_mos_forecast_rows` | One wide row per station/product/runtime/forecast valid time | 33.36M wide rows, not hundreds of millions of token rows | Raw JSON preserves additional fields |
| Build gold feature tables | `materialize_iem_mos_features.py`, `gold.iem_mos_daily_features`, `gold.iem_mos_feature_matrix_v1` | Daily features and JSON matrix materialized after fetch | 444,105 daily features; 8,231 matrix rows | Feature build is post-fetch, not HTTP-bound |
| Validate and document final state | This report and command outputs below | DB and test evidence recorded | All focused checks passed | Full repository test suite was not run |

## Change Inventory

Exact changed-file path coverage for automated review:

```text
ingestion-service/pom.xml
models/src/main/java/com/predictionmarkets/weather/models/MosModel.java
ingestion-service/src/main/resources/application-klga-iem-mos.yml
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosBackfillApplication.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosBackfillCommandLineRunner.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosBackfillProperties.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosBackfillRepository.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosBackfillService.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosChunk.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosFetchResult.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosForecastRow.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosHttpClient.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosParser.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosPlanner.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosProduct.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosProgress.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosRateLimiter.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosRateLimitException.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosStation.java
ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosStoredRequest.java
ingestion-service/src/test/java/com/predictionmarkets/weather/klga/iemmos/IemMosPlannerTest.java
ingestion-service/src/test/java/com/predictionmarkets/weather/klga/iemmos/IemMosParserTest.java
ingestion-service/src/test/java/com/predictionmarkets/weather/klga/iemmos/IemMosHttpClientTest.java
bootstrap/klga_tmax/implementation/alembic/versions/0008_iem_mos_backfill.py
bootstrap/klga_tmax/implementation/src/klga_tmax/db/migrations_check.py
bootstrap/klga_tmax/implementation/tests/test_iem_mos_schema_contract.py
bootstrap/klga_tmax/implementation/scripts/materialize_iem_mos_features.py
bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_08_IEM_MOS_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md
```

| File | Type | Why It Changed | Main Objects | Effect | Verification |
|---|---|---|---|---|---|
| `ingestion-service/pom.xml` | Modified dependency config | Add Postgres JDBC runtime support | `org.postgresql:postgresql` | Java runner can connect to KLGA Postgres DB | Maven focused tests passed |
| `models/src/main/java/com/predictionmarkets/weather/models/MosModel.java` | Modified enum | Add `LAV` product support | `LAV` enum value | Product registry can request LAV MOS | Planner test covers product registry behavior |
| `ingestion-service/src/main/resources/application-klga-iem-mos.yml` | Added Spring profile | Configure KLGA MOS runner DB profile | `klga-iem-mos` profile, JDBC env defaults | Runner uses Postgres and non-web mode | Live runner used this profile |
| `ingestion-service/src/main/java/com/predictionmarkets/weather/klga/iemmos/IemMosBackfillApplication.java` | Added app entry point | Isolate runner from older ingestion app startup behavior | `IemMosBackfillApplication` | `exec:java` can run KLGA MOS only | Live job executed |
| `IemMosBackfillCommandLineRunner.java` | Added CLI runner | Start service from Spring command line | `CommandLineRunner.run` | Reads properties and invokes backfill service | Live job executed |
| `IemMosBackfillProperties.java` | Added config properties | Define runner arguments and defaults | job/cutoff/date/thread/resume/build flags | CLI controls full, dry-run, feature toggles | Live commands used flags |
| `IemMosProduct.java` | Added product registry | Centralize product mapping and start dates | MAV, MET, MEX, LAV, NBS, NBE | Planner builds correct product windows | Planner tests passed |
| `IemMosStation.java` | Added value object | Represent station registry entries | station and MOS station IDs | Planner and repository pass station identity consistently | Live station count 19 |
| `IemMosChunk.java` | Added value object | Represent station/product/date-window requests | chunk ID, request hash, window dates | Audit chunks are deterministic and resumable | 1,330 chunks completed |
| `IemMosFetchResult.java` | Added value object | Preserve HTTP response status/body | status, body, headers | Fetcher can distinguish success, empty, retry, split | HTTP client test passed |
| `IemMosStoredRequest.java` | Added value object | Carry persisted lineage IDs | source request ID, source record ID, raw URI | Forecast rows link back to source artifacts | Live row lineage persisted |
| `IemMosForecastRow.java` | Added value object | Wide parsed MOS row contract | typed MOS fields, raw JSON, availability fields | Parser emits compact forecast rows | Parser test passed |
| `IemMosProgress.java` | Added value object | Report job progress | totals, row counts, bytes | Runner logs progress and updates job summary | Live progress logs |
| `IemMosRateLimitException.java` | Added exception | Stop safely on provider 429 | runtime exception marker | Service stops and preserves state on rate limit | Code path present, no 429 in final run |
| `IemMosRateLimiter.java` | Added concurrency helper | Enforce global request-start spacing | synchronized `awaitTurn` | 10 workers do not start requests all at once | Live run used 10 workers |
| `IemMosHttpClient.java` | Added HTTP client | Build and send IEM request shape | `fetch` | Requests structured JSON endpoint | MockWebServer test passed |
| `IemMosParser.java` | Added parser | Convert IEM JSON rows into typed rows | parser version `iem_mos_structured_json_v1` | Extracts N/X, TMP, DPT, wind, POP, QPF, TSTM, raw JSON | Parser test passed |
| `IemMosPlanner.java` | Added planner | Build annual chunks and request hashes | `plan`, split helpers | Builds deterministic per station/product/year chunks | Planner tests passed |
| `IemMosBackfillRepository.java` | Added persistence layer | Encapsulate DB reads/writes and resume logic | job/chunk/source/raw/silver/gold methods | Persists lineage, rows, gaps, progress, features | Live DB validation |
| `IemMosBackfillService.java` | Added orchestration | Coordinate planning, fetch, parse, persistence, retries | `run`, `processChunk`, `handleSuccess` | Full backfill execution with 10 workers and safe terminal state | Live run completed |
| `ingestion-service/src/test/java/.../IemMosPlannerTest.java` | Added unit tests | Check planner/product/cutoff behavior | planner tests | Guards product mapping, request hash, windows | 3 tests passed |
| `IemMosParserTest.java` | Added unit test | Check JSON parsing and availability rule | parser test | Guards runtime+2h and typed extraction | 1 test passed |
| `IemMosHttpClientTest.java` | Added integration-style unit test | Check request shape | MockWebServer test | Guards endpoint params and JSON request | 1 test passed |
| `bootstrap/klga_tmax/implementation/alembic/versions/0008_iem_mos_backfill.py` | Added migration | Create KLGA IEM MOS Postgres tables and indexes | audit, silver, gold tables | DB schema supports backfill, rows, features, matrix | Migration applied; contract test passed |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/db/migrations_check.py` | Modified schema contract | Include IEM MOS tables and indexes | required tables/indexes | DB inspect can catch missing MOS objects | Contract test passed |
| `bootstrap/klga_tmax/implementation/tests/test_iem_mos_schema_contract.py` | Added Python tests | Verify migration contract includes MOS objects | schema tests | Guards table/index contract | 2 tests passed |
| `bootstrap/klga_tmax/implementation/scripts/materialize_iem_mos_features.py` | Added operational script | Rebuild gold features from persisted rows after fetch-only run | target instance, daily feature, matrix builders | Materialized final features and matrix without refetching | Script compile passed; live run completed |
| `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_08_IEM_MOS_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md` | Added documentation | Record final implementation and backfill evidence | this report | Handoff for future KLGA work | Documentation quality gate run below |

## Architecture and Control Flow

```mermaid
flowchart TD
  A["CLI: IemMosBackfillApplication"] --> B["IemMosBackfillService.run"]
  B --> C["Load MOS stations from registry.station_registry"]
  B --> D["IemMosPlanner annual station/product chunks"]
  D --> E["audit.iem_mos_backfill_chunks"]
  B --> F["10-worker chunk executor"]
  F --> G["IemMosRateLimiter global request spacing"]
  G --> H["IemMosHttpClient GET request/mos.py"]
  H --> I["IemMosParser structured JSON rows"]
  I --> J["bronze.source_requests and source_records"]
  I --> K["silver.iem_mos_forecast_rows"]
  K --> L["materialize_iem_mos_features.py"]
  L --> M["gold.iem_mos_daily_features"]
  M --> N["gold.iem_mos_feature_matrix_v1"]
```

Control flow:

1. The CLI app starts with `--spring.profiles.active=klga-iem-mos`.
2. The runner loads stations from the KLGA station registry.
3. The planner builds annual chunks for each station/product range.
4. Each chunk has a deterministic `chunk_id` and `request_sha256`.
5. The repository creates or reuses job/chunk audit rows.
6. The service runs chunks with a fixed worker pool and global request-start limiter.
7. The fetcher calls the IEM endpoint with station, model, start timestamp, end timestamp, and `format=json`.
8. Raw payload bytes are gzipped and linked through source request/source record tables.
9. The parser emits wide rows with typed feature columns, raw JSON, parser version, row hash, and availability metadata.
10. The repository upserts rows into `silver.iem_mos_forecast_rows`.
11. After fetching, `materialize_iem_mos_features.py` rebuilds daily cutoff-safe features and the JSON feature matrix from persisted rows.

Failure paths:

- HTTP `404`: records a provider no-data gap and marks the chunk terminal with zero rows.
- HTTP `422`: splits large windows when possible and records a split gap.
- HTTP `429`: marks the chunk `rate_limited`, stops execution, and preserves resume state.
- Retryable `503` and transport errors: retried with exponential backoff.
- Runtime parser or DB errors: mark failed or raise, preserving completed chunks.
- Manual interruption: stale `running` chunks can be reset to `planned` only after verifying no active runner process remains.

## File-by-File Deep Dive

### `IemMosBackfillApplication.java`

This is the narrow Spring entry point for the KLGA MOS runner. It avoids launching unrelated ingestion-service behavior by scanning only the KLGA IEM MOS package and enabling only `IemMosBackfillProperties` and the existing `IemProperties`. It is invoked with Maven `exec:java`.

Maintenance invariant: keep this entry point narrow. If it scans the broader ingestion package, unrelated ingestion tasks can start unexpectedly.

### `IemMosBackfillCommandLineRunner.java`

This class bridges Spring Boot startup into `IemMosBackfillService.run(properties)`. It has no fetch logic itself. Its purpose is to keep command-line execution explicit and testable.

### `IemMosBackfillProperties.java`

This property class defines the runner contract:

- `jobId`
- `cutoffId`
- `start`
- `through`
- `threads`
- `requestSpacingMs`
- `resume`
- `maxAttempts`
- `retryBackoffMs`
- `mode`
- `buildDailyFeatures`
- `buildFeatureMatrix`
- station and product filters
- raw output root

The production fetch used `buildDailyFeatures=false` and `buildFeatureMatrix=false` to prevent expensive per-chunk feature rebuilds during HTTP fetching. Gold features were then rebuilt once from persisted rows.

### `IemMosProduct.java`

This registry maps source products to endpoint models and default start dates:

| Product | Endpoint model | Default start |
|---|---|---:|
| MAV | GFS | 2003-12-16 |
| MET | NAM | 2008-12-09 |
| MEX | MEX | 2020-07-12 |
| LAV | LAV | 2020-07-12 |
| NBS | NBS | 2020-01-01 |
| NBE | NBE | 2021-01-01 |

`LAV` was also added to the shared `MosModel` enum so the Java model layer can represent the endpoint model.

### `IemMosPlanner.java`

The planner creates one annual request chunk per station/product/year window. It clips each product to its default start and requested through date. Request hashes include job, cutoff, station, product, endpoint model, and date window so resume behavior is deterministic.

The planner also contains split helpers for `422` handling. If IEM rejects a large date window, the service can create child chunks and skip the parent.

### `IemMosHttpClient.java`

The client sends structured JSON requests to:

```text
/cgi-bin/request/mos.py?station={station}&model={model}&sts={start}&ets={end}&format=json
```

It returns `IemMosFetchResult`, preserving the HTTP status, headers, and response body. `IemMosHttpClientTest` verifies the request shape with MockWebServer.

### `IemMosParser.java`

The parser reads structured IEM JSON and emits `IemMosForecastRow` objects. Typed columns are extracted for:

- `n_x_f`
- `tmp_f`
- `dpt_f`
- `wdr`
- `wsp_kt`
- `gst_kt`
- `sky_or_cloud`
- `pop`
- `qpf`
- `tstm_prob`

The full value map remains in `raw_values_jsonb`, so unsupported or product-specific fields are not lost.

Leakage fields are assigned as:

```text
provider_available_at_utc = runtime_utc + 2 hours
effective_available_at_utc = provider_available_at_utc
availability_method = conservative_lag_rule
```

This is intentionally conservative until raw AFOS issue-time evidence is added.

### `IemMosBackfillRepository.java`

The repository owns all DB interactions:

- station registry loading
- cutoff existence check
- job and chunk initialization
- stale/resumable chunk selection
- source request and source record insertion
- raw gzip URI recording
- wide forecast-row upsert
- source gap recording
- job summary refresh
- progress reporting

The wide forecast-row upsert deduplicates rows by `raw_row_hash` before insert. This matters with PostgreSQL batch rewrite mode because duplicate keys inside one rewritten multi-row insert can otherwise raise `ON CONFLICT DO UPDATE command cannot affect row a second time`.

### `IemMosBackfillService.java`

The service coordinates the runner lifecycle:

1. Validate properties and cutoff.
2. Load stations.
3. Plan chunks.
4. Initialize job/chunk audit rows.
5. Run chunks with a fixed thread pool.
6. Fetch, parse, persist, and mark each chunk terminal.
7. Stop safely on 429.
8. Refresh job summary after chunk completions.

For the final run, this service fetched and persisted rows only. Feature materialization was intentionally moved to the post-fetch script to avoid repeated matrix rebuilds.

### `materialize_iem_mos_features.py`

This script rebuilds gold MOS features from `silver.iem_mos_forecast_rows`. It performs three jobs:

1. Ensures `gold.target_instances` exists for the requested date range and cutoff.
2. Materializes `gold.iem_mos_daily_features` for each completed chunk.
3. Rebuilds `gold.iem_mos_feature_matrix_v1`.

For each station/product/target date, it chooses the latest runtime satisfying:

```sql
effective_available_at_utc <= target_date 12:45 UTC
forecast_valid_time_utc within the New York local target day
```

Then it aggregates same-runtime rows into:

- `tmax_f`
- `tmp_peak_window_max_f`
- `tmp_peak_window_mean_f`
- `dpt_peak_window_mean_f`
- `wind_speed_peak_window_mean_kt`
- `pop_max`
- `qpf_max`
- `tstm_prob_max`

The first foreground run was terminated by the command timeout after committing partial feature progress. The script was then patched with `--missing-only`, which resumes only chunks where `feature_rows_upserted = 0`. Final materialization completed and wrote 444,105 daily feature rows and 8,231 matrix rows.

## Public Interfaces and Contracts

### Java Runner Command

```powershell
$env:KLGA_IEM_MOS_JDBC_URL='jdbc:postgresql://127.0.0.1:5432/klga_tmax_research?reWriteBatchedInserts=true'
$env:KLGA_IEM_MOS_DB_USER='postgres'
$env:KLGA_IEM_MOS_DB_PASSWORD='root'

mvn -pl ingestion-service exec:java `
  -Dexec.mainClass=com.predictionmarkets.weather.klga.iemmos.IemMosBackfillApplication `
  -Dexec.classpathScope=runtime `
  -Dexec.args="--spring.profiles.active=klga-iem-mos --iem-mos.job-id=klga_iem_mos_full_backfill_v1 --iem-mos.cutoff-id=T_1245UTC --iem-mos.through=2026-06-28 --iem-mos.mode=full --iem-mos.threads=10 --iem-mos.resume=true --iem-mos.build-daily-features=false --iem-mos.build-feature-matrix=false"
```

### Feature Materializer Command

```powershell
$env:KLGA_DB_URL='postgresql://postgres:root@127.0.0.1:5432/klga_tmax_research'

python bootstrap\klga_tmax\implementation\scripts\materialize_iem_mos_features.py `
  --job-id klga_iem_mos_full_backfill_v1 `
  --cutoff-id T_1245UTC `
  --start-date 2003-12-16 `
  --through-date 2026-06-28 `
  --missing-only `
  --progress-every 50
```

### Environment Variables

| Variable | Purpose | Final Value Used |
|---|---|---|
| `KLGA_IEM_MOS_JDBC_URL` | Java JDBC URL | `jdbc:postgresql://127.0.0.1:5432/klga_tmax_research?reWriteBatchedInserts=true` |
| `KLGA_IEM_MOS_DB_USER` | Java DB user | `postgres` |
| `KLGA_IEM_MOS_DB_PASSWORD` | Java DB password | `root` |
| `KLGA_DB_URL` | Python materializer DB URL | `postgresql://postgres:root@127.0.0.1:5432/klga_tmax_research` |

## Data Model, Persistence, and Migration Notes

New Postgres objects:

- `audit.iem_mos_backfill_jobs`
- `audit.iem_mos_backfill_chunks`
- `audit.iem_mos_source_gaps`
- `silver.iem_mos_forecast_rows`
- `gold.iem_mos_daily_features`
- `gold.iem_mos_feature_matrix_v1`

Existing objects reused:

- `bronze.source_requests`
- `bronze.source_records`
- `registry.cutoffs`
- `registry.station_registry`
- `gold.target_instances`

Important indexes:

- `ix_iem_mos_chunks_job_status`
- `ix_iem_mos_chunks_product_station`
- `ix_iem_mos_gaps_job`
- `ix_iem_mos_forecast_station_product_runtime`
- `ix_iem_mos_forecast_station_product_valid`
- `ix_iem_mos_forecast_valid`
- `ix_iem_mos_forecast_available`
- `ix_iem_mos_forecast_request`
- `ix_iem_mos_daily_features_target`
- `ix_iem_mos_daily_features_station_product`
- `ix_iem_mos_feature_matrix_date_cutoff`

The live DB index `ix_iem_mos_forecast_station_product_valid` was created with `CREATE INDEX CONCURRENTLY` after the fetch because the first feature-build attempt showed that feature materialization benefits from station/product/date-window access.

Wide forecast-row identity:

```text
station_id
mos_station_id
source_product
endpoint_model
cutoff_id
run_time_utc
forecast_valid_time_utc
forecast_hour
period_type
raw_row_hash
source_request_id
source_record_id
request_sha256
```

Gold daily feature identity:

```text
target_date
cutoff_id
station_id
source_product
feature_build_version
```

Feature matrix identity:

```text
target_instance_id
```

## Executed Backfill Results

### Station Coverage

The final persisted row set covers 19 stations:

```text
KABE,KALB,KBDR,KBOS,KBWI,KCDW,KDCA,KEWR,KFRG,KHPN,KISP,KJFK,KLGA,KMMU,KNYC,KPHL,KPOU,KSWF,KTEB
```

### Fetch Coverage by Location, Dataset, and Model

The fetch audit table shows complete terminal coverage for every planned station/product/model chunk:

```text
job_id: klga_iem_mos_full_backfill_v1
planned chunks: 1,330
completed chunks: 1,330
failed chunks: 0
running chunks: 0
planned remaining chunks: 0
source gaps: 0
```

The backfilled locations are:

```text
KABE, KALB, KBDR, KBOS, KBWI, KCDW, KDCA, KEWR, KFRG, KHPN, KISP, KJFK, KLGA, KMMU, KNYC, KPHL, KPOU, KSWF, KTEB
```

Product/model fetch coverage:

| Dataset Type | IEM Endpoint Model | Target-Date Range | Expected Target Days | Locations Covered | Fetch Chunk Coverage |
|---|---|---:|---:|---:|---:|
| `MAV` | `GFS` | 2003-12-16 to 2026-06-28 | 8,231 | 19 / 19 | 100.0000% |
| `MET` | `NAM` | 2008-12-09 to 2026-06-28 | 6,411 | 19 / 19 | 100.0000% |
| `MEX` | `MEX` | 2020-07-12 to 2026-06-28 | 2,178 | 19 / 19 | 100.0000% |
| `LAV` | `LAV` | 2020-07-12 to 2026-06-28 | 2,178 | 19 / 19 | 100.0000% |
| `NBS` | `NBS` | 2020-01-01 to 2026-06-28 | 2,371 | 19 / 19 | 100.0000% |
| `NBE` | `NBE` | 2021-01-01 to 2026-06-28 | 2,005 | 19 / 19 | 100.0000% |

Per-location fetch coverage:

| Location | `MAV/GFS` | `MET/NAM` | `MEX/MEX` | `LAV/LAV` | `NBS/NBS` | `NBE/NBE` |
|---|---:|---:|---:|---:|---:|---:|
| `KABE` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KALB` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KBDR` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KBOS` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KBWI` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KCDW` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KDCA` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KEWR` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KFRG` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KHPN` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KISP` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KJFK` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KLGA` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KMMU` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KNYC` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KPHL` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KPOU` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KSWF` | 100% | 100% | 100% | 100% | 100% | 100% |
| `KTEB` | 100% | 100% | 100% | 100% | 100% | 100% |

Gold feature coverage has the same full product/location shape except for one observed daily-feature nuance:

```text
KNYC MAV/GFS daily feature rows: 8,230 / 8,231 target days
Missing KNYC MAV/GFS feature date: 2003-12-16
Impact: fetch coverage is still 100%; only one KNYC MAV/GFS gold daily feature row is absent because no cutoff-safe target-day feature was materialized for that first product date.
All other station/product gold daily feature coverages match their expected target-day counts.
```

### Product Coverage

| Product | Chunks | Forecast Rows | Daily Feature Rows | Raw/Lineage Bytes |
|---|---:|---:|---:|---:|
| LAV | 133 | 6,210,872 | 41,382 | 2,067,305,908 |
| MAV | 456 | 13,125,332 | 156,388 | 4,578,777,049 |
| MET | 361 | 5,116,675 | 121,809 | 1,594,634,695 |
| MEX | 133 | 1,241,745 | 41,382 | 357,781,278 |
| NBE | 114 | 3,487,429 | 38,095 | 1,480,609,264 |
| NBS | 133 | 4,173,777 | 45,049 | 2,067,696,371 |
| Total | 1,330 | 33,355,830 | 444,105 | 12,146,804,565 |

### Gold Feature Date Ranges

| Product | Target Date From | Target Date Through | Target Dates | Stations | Chosen Runtime Range UTC | Availability Range UTC |
|---|---:|---:|---:|---:|---|---|
| LAV | 2020-07-12 | 2026-06-28 | 2,178 | 19 | 2020-07-12 06:00 to 2026-06-28 10:00 | 2020-07-12 08:00 to 2026-06-28 12:00 |
| MAV | 2003-12-16 | 2026-06-28 | 8,231 | 19 | 2003-12-16 06:00 to 2026-06-28 06:00 | 2003-12-16 08:00 to 2026-06-28 08:00 |
| MET | 2008-12-09 | 2026-06-28 | 6,411 | 19 | 2008-12-09 00:00 to 2026-06-28 00:00 | 2008-12-09 02:00 to 2026-06-28 02:00 |
| MEX | 2020-07-12 | 2026-06-28 | 2,178 | 19 | 2020-07-12 00:00 to 2026-06-28 00:00 | 2020-07-12 02:00 to 2026-06-28 02:00 |
| NBE | 2021-01-01 | 2026-06-28 | 2,005 | 19 | 2021-01-01 07:00 to 2026-06-28 00:00 | 2021-01-01 09:00 to 2026-06-28 02:00 |
| NBS | 2020-01-01 | 2026-06-28 | 2,371 | 19 | 2020-01-01 07:00 to 2026-06-28 06:00 | 2020-01-01 09:00 to 2026-06-28 08:00 |

### Job Timing

The final job row reports:

```text
status: completed
planned_chunks: 1330
completed_chunks: 1330
completed_empty_chunks: 0
failed_chunks: 0
rows_upserted: 33355830
feature_rows_upserted: 444105
bytes_fetched: 12146804565
started_at_utc: 2026-06-30 09:32:17.590015
finished_at_utc: 2026-06-30 19:04:59.367025
```

The elapsed wall-clock includes implementation fixes, process stops/restarts, fetch-only completion, index creation, and post-fetch feature materialization. The actual long-running fetch/persist phase completed in supervised resume mode and the final materialization completed separately.

## Leakage Policy and Proof

For structured MOS rows, the v1 availability contract is:

```text
provider_available_at_utc = run_time_utc + 2 hours
effective_available_at_utc = provider_available_at_utc
availability_method = conservative_lag_rule
```

Gold feature eligibility:

```sql
f.max_source_available_at_utc <= ti.cutoff_utc
```

Final leakage validation:

```text
late_features = 0
```

This proves that no materialized gold MOS feature has a max source availability later than the target date `12:45 UTC` cutoff.

## Error Handling, Edge Cases, and Failure Modes

Handled in runner:

- `404`: records `provider_no_data` gap and marks chunk completed with zero rows.
- `422`: splits oversized windows when possible.
- `429`: records `rate_limited`, stops, and preserves resume state.
- `500`, `502`, `503`, `504`, and transport failures: retry with capped backoff.
- Empty `200`: records `empty_response` source gap.
- Duplicate parsed rows in one batch: deduped by `raw_row_hash` before DB upsert.
- Manual stop: stale `running` chunks can be reset after process verification.

Observed during this run:

- No final source gaps.
- No failed chunks.
- No provider rate limit.
- One user-requested process stop left stale running rows; those were reset only after confirming no runner process remained.
- The first feature materializer command exceeded the terminal timeout after committing partial rows. The script was patched with `--missing-only` and resumed without refetching or redoing completed feature chunks.

## Security, Privacy, and Safety Review

Secrets:

- DB password is supplied via environment variables in runner commands.
- No IEM credential is used because this endpoint is public.
- Raw payloads are local artifacts; they contain weather forecast data and request metadata, not user secrets.

Injection risk:

- Java persistence uses `NamedParameterJdbcTemplate` parameters.
- The Python materializer uses psycopg parameter binding.
- Station and product values come from controlled registry/planner values.

Operational safety:

- Runner stops on provider 429 instead of pushing through.
- Process restarts are resumable through audit chunk state.
- Raw source payloads and source request IDs preserve auditability.

## Performance, Scalability, and Concurrency

Fetch concurrency:

- 10 worker threads.
- Global request-start spacing.
- Annual chunks.
- PostgreSQL `reWriteBatchedInserts=true` for large insert batches.

Persistence decisions:

- Wide row table avoids multiplying 33M structured rows by every MOS variable token.
- Raw JSON is retained in `raw_values_jsonb` for fields that are not promoted to typed columns.
- Gold feature materialization is a post-fetch step to avoid repeating expensive matrix rebuilds after every chunk.

Indexes:

- Runtime and valid-time indexes support retrieval and feature aggregation.
- `ix_iem_mos_forecast_station_product_valid` was added after observing that full feature materialization needed station/product/date-window access.

Operational scale actually handled:

```text
33,355,830 wide forecast rows
444,105 daily feature rows
8,231 feature matrix rows
12.15 GB raw/source lineage bytes
```

## Testing and Verification Evidence

### Java Focused Tests

Command:

```powershell
mvn -pl ingestion-service -am "-Dtest=IemMosPlannerTest,IemMosParserTest,IemMosHttpClientTest" "-Dsurefire.failIfNoSpecifiedTests=false" test
```

Result:

```text
Tests run: 5, Failures: 0, Errors: 0, Skipped: 0
BUILD SUCCESS
```

What it proves:

- Product planning and cutoff/request hash behavior are covered.
- Structured JSON parsing and runtime-plus-2h availability are covered.
- IEM endpoint request shape is covered.

What it does not prove:

- It does not prove every historical IEM response shape. The live backfill and parser success provide that evidence.

### Python Schema Contract Test

Command:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests\test_iem_mos_schema_contract.py
```

Result:

```text
2 passed in 1.08s
```

What it proves:

- The KLGA schema contract includes the IEM MOS tables and indexes.

### Feature Materializer Syntax Check

Command:

```powershell
python -m py_compile bootstrap\klga_tmax\implementation\scripts\materialize_iem_mos_features.py
```

Result: passed with exit code `0`.

### Live DB Validation

Chunk terminal state:

```text
('completed', 1330, 33355830, 444105, 12146804565)
```

Source gaps:

```text
source_gaps = 0
```

Leakage check:

```text
late_features = 0
```

Feature coverage:

```text
daily_features = 444105
feature_matrix = 8231
daily_coverage = 444105 rows, 8231 target dates, 19 stations, 6 products
matrix_coverage = 8231 rows, min 71 features, max 399 features, avg 200.67 features
```

Process checks:

```text
NO_BACKFILL_RUNNER_PROCESSES_REMAIN
```

The final process checks showed no active Java backfill runner or Python materializer remaining.

## Operational Runbook

### Inspect Job Status

```sql
SELECT status, count(*), sum(rows_upserted), sum(feature_rows_upserted), sum(response_size_bytes)
FROM audit.iem_mos_backfill_chunks
WHERE job_id = 'klga_iem_mos_full_backfill_v1'
GROUP BY status;
```

Expected final state:

```text
completed | 1330 | 33355830 | 444105 | 12146804565
```

### Inspect Product Coverage

```sql
SELECT source_product, count(*) chunks, sum(rows_upserted) rows, sum(feature_rows_upserted) feature_rows
FROM audit.iem_mos_backfill_chunks
WHERE job_id = 'klga_iem_mos_full_backfill_v1'
GROUP BY source_product
ORDER BY source_product;
```

### Check Leakage

```sql
SELECT count(*)
FROM gold.iem_mos_daily_features f
JOIN gold.target_instances ti USING (target_instance_id)
WHERE f.cutoff_id = 'T_1245UTC'
  AND f.max_source_available_at_utc > ti.cutoff_utc;
```

Expected final result:

```text
0
```

### Resume Fetch After Crash

1. Confirm no runner is active.
2. Reset stale rows if needed:

```sql
UPDATE audit.iem_mos_backfill_chunks
SET status = 'planned',
    error_type = 'manual_stop_stale_running_reset',
    error_message = 'Reset after process stop; no active runner process remained.',
    started_at_utc = NULL,
    updated_at = now()
WHERE job_id = 'klga_iem_mos_full_backfill_v1'
  AND status = 'running';
```

3. Rerun the Java command with `--iem-mos.resume=true`.

### Resume Feature Materialization

Use:

```powershell
python bootstrap\klga_tmax\implementation\scripts\materialize_iem_mos_features.py `
  --job-id klga_iem_mos_full_backfill_v1 `
  --cutoff-id T_1245UTC `
  --start-date 2003-12-16 `
  --through-date 2026-06-28 `
  --missing-only `
  --progress-every 50
```

Use `--reset` only when intentionally rebuilding all daily features and matrix rows from scratch.

## Compatibility, Rollback, and Upgrade Notes

Compatibility:

- Existing GribStream and Wunderground KLGA tables are not modified by this implementation.
- Existing MySQL MOS pilot tables are not used or migrated.
- The new tables are additive under `audit`, `silver`, and `gold` schemas.

Rollback:

- Alembic downgrade for migration 0008 drops the new IEM MOS audit, silver, and gold tables.
- Raw gzip artifacts are filesystem outputs and must be removed separately if rollback requires disk cleanup.
- The shared `MosModel.LAV` enum addition is backward-compatible unless downstream code assumes a closed enum list.
- The Postgres JDBC dependency is runtime-only and does not change existing API surfaces.

Upgrade considerations:

- If raw AFOS issue-time evidence is added later, `availability_method` should distinguish issue-time-derived availability from the current `conservative_lag_rule`.
- If another cutoff is introduced, materialize separate target instances and feature rows using that cutoff ID instead of overwriting `T_1245UTC`.

## Known Limitations and Follow-Up Work

1. Raw AFOS issue-time proof is not implemented.
   - Impact: availability is conservative, but not evidence-specific per raw bulletin.
   - Reason: structured JSON was reliable and sufficient for the v1 historical corpus.
   - Revisit when issue-time enrichment is prioritized.

2. Feature materialization is Python set-based SQL, not a Java runner mode.
   - Impact: fetch is Java-owned, but post-fetch matrix rebuild is run by script.
   - Reason: the fetch-only optimization was needed to finish the large backfill without repeated expensive feature work.
   - Revisit by adding an explicit Java `features-only` mode if this becomes a recurring operational task.

3. Full repository test suite was not run.
   - Impact: verification is focused on the KLGA IEM MOS path.
   - Reason: the repository contains large unrelated dirty HKG changes and many unrelated tests.
   - Revisit before merging this into a shared branch.

4. The first broad diagnostic row-count query timed out.
   - Impact: none to final data; the query was canceled and replaced by narrower validation.
   - Reason: it attempted an ad hoc full-table aggregate over 33M rows.
   - Revisit by adding purpose-built summary views if repeated large diagnostics are needed.

## Reviewer Checklist

- [x] Java runner implemented with narrow Spring entry point.
- [x] Postgres profile and JDBC dependency added.
- [x] Product registry includes `MAV`, `MET`, `MEX`, `LAV`, `NBS`, `NBE`.
- [x] `LAV` added to the Java MOS model enum.
- [x] Postgres migration creates audit, silver, and gold MOS tables.
- [x] Raw payload lineage persisted.
- [x] Wide forecast rows persisted without atomic token explosion.
- [x] Chunk tracking is terminal: 1,330 completed, 0 running, 0 planned, 0 failed.
- [x] Source gaps are zero.
- [x] Daily feature rows materialized: 444,105.
- [x] Matrix rows materialized: 8,231.
- [x] Leakage validation returns zero late features.
- [x] Focused Java tests passed.
- [x] Python schema contract test passed.
- [x] Feature materializer syntax check passed.
- [x] No active runner/materializer process remains.
