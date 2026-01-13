# Runbook (Epic #1 Ingestion)

## 1) Prerequisites
- JDK 18
- Maven 3.9+
- MySQL 8+
- Network access to Kalshi + IEM endpoints (see `docs/architecture.md`)

## 2) Build the repo
From repo root:
```bash
mvn clean install
```

## 3) Configure MySQL
Edit `ingestion-service/src/main/resources/application-mysql.yml`, or set
environment variables:
- `SPRING_DATASOURCE_URL`
- `SPRING_DATASOURCE_USERNAME`
- `SPRING_DATASOURCE_PASSWORD`

Run the ingestion service with the MySQL profile:
```bash
--spring.profiles.active=mysql
```

Flyway migrations run automatically on startup.

## 4) Seed the as-of policy (required)
Create at least one policy used by MOS materialization:
```sql
INSERT INTO asof_policy (name, asof_local_time, enabled)
VALUES ('default-23', '23:00:00', TRUE);
```

Capture the policy id:
```sql
SELECT id FROM asof_policy WHERE name = 'default-23';
```

## 5) Run a 30-day backfill for one station
Example series ticker: `KXHIGHMIA`
Example date range: `2024-07-01` to `2024-07-30` (station-local dates)

Optional: seed station mapping via Kalshi series sync:
```bash
mvn -pl ingestion-service spring-boot:run -Dspring-boot.run.arguments="--spring.profiles.active=mysql --backfill.enabled=true --backfill.job=kalshi_series_sync --backfill.series-ticker=KXHIGHMIA"
```

CLI daily truth (year-sliced ingestion):
```bash
mvn -pl ingestion-service spring-boot:run -Dspring-boot.run.arguments="--spring.profiles.active=mysql --backfill.enabled=true --backfill.job=cli_ingest_year --backfill.series-ticker=KXHIGHMIA --backfill.date-start-local=2024-07-01 --backfill.date-end-local=2024-07-30"
```

MOS runs (UTC window ingestion, 1-day windows):
```bash
mvn -pl ingestion-service spring-boot:run -Dspring-boot.run.arguments="--spring.profiles.active=mysql --backfill.enabled=true --backfill.job=mos_ingest_window --backfill.series-ticker=KXHIGHMIA --backfill.date-start-local=2024-07-01 --backfill.date-end-local=2024-07-30 --backfill.models=GFS,MEX,NAM,NBS,NBE --backfill.mos-window-days=1"
```

MOS as-of feature materialization:
```bash
mvn -pl ingestion-service spring-boot:run -Dspring-boot.run.arguments="--spring.profiles.active=mysql --backfill.enabled=true --backfill.job=mos_asof_materialize_range --backfill.series-ticker=KXHIGHMIA --backfill.date-start-local=2024-07-01 --backfill.date-end-local=2024-07-30 --backfill.asof-policy-id=1 --backfill.models=GFS,MEX,NAM,NBS,NBE"
```

Notes:
- For `kalshi_series_sync` you can pass a comma-separated list of series tickers.
- For all other backfill jobs, only one series ticker is supported.
- All jobs are restartable; checkpoints are stored in `ingest_checkpoint`.

## 6) Run the full ingestion pipeline (all configured series)
Use `com.predictionmarkets.weather.executors.FullIngestionExecutor` as the entry point.
Pipeline settings live in `ingestion-service/src/main/resources/application.yml` under `pipeline.*`.
Key settings:
- `pipeline.series-tickers` (stations to ingest)
- `pipeline.date-start-local` / `pipeline.date-end-local`
- `pipeline.models`
- `pipeline.mos-window-days`
- `pipeline.thread-count` (parallel station workers)

## 7) Run the audit report
```bash
mvn -pl ingestion-service spring-boot:run -Dspring-boot.run.arguments="--spring.profiles.active=mysql --audit.enabled=true --audit.series-ticker=KXHIGHMIA --audit.date-start-local=2024-07-01 --audit.date-end-local=2024-07-30 --audit.asof-policy-id=1 --audit.output-dir=reports"
```

Artifacts:
- Markdown report in `reports/`
- JSON report in `reports/`

## 8) Gribstream CSV transfer (export/import)
Export `gribstream_daily_feature` to a CSV file:
```bash
mvn -pl ingestion-service spring-boot:run -Dspring-boot.run.mainClass=com.predictionmarkets.weather.executors.GribstreamDailyFeatureCsvExportExecutor -Dspring-boot.run.arguments="--spring.profiles.active=mysql --gribstream.transfer.export.output-path=gribstream_daily_feature.csv"
```

Import a CSV file into `gribstream_daily_feature` (idempotent upsert):
```bash
mvn -pl ingestion-service spring-boot:run -Dspring-boot.run.mainClass=com.predictionmarkets.weather.executors.GribstreamDailyFeatureCsvImportExecutor -Dspring-boot.run.arguments="--spring.profiles.active=mysql --gribstream.transfer.import.input-path=gribstream_daily_feature.csv"
```

Config keys (see `ingestion-service/src/main/resources/application.yml`):
- `gribstream.transfer.export.output-path`
- `gribstream.transfer.export.page-size`
- `gribstream.transfer.export.include-header`
- `gribstream.transfer.import.input-path`
- `gribstream.transfer.import.batch-size`
- `gribstream.transfer.import.has-header`

## 9) Common failure modes
- Station mapping mismatch (needs `station_override`)
- IEM endpoint 503 rate limiting (retry/backoff)
- Missing `asof_policy` row for materialization
- MOS model availability gaps for older periods

## 10) Audit SQL helpers
Example queries for no-leakage and completeness live under `docs/sql/`.
