# agents.md — Implementation Handoff Rules (for Codex / agent tooling)

This repository is a weather forecasting + prediction markets playground with a strict backbone: data ingestion must be reproducible, auditable, restartable, and safe against leakage. This file is the single handoff doc for how code is organized, how to run ingestion/reporting tasks, and which invariants must never be violated.

The repo currently contains two kinds of systems:
1) Java/Spring ingestion + MySQL persistence + simple CLI runners (Epic #1)
2) Python training/evaluation/backtesting artifacts (Epic #2/#3 work-in-progress), which are treated as sources to ingest and catalog rather than as the canonical system of record.

The newest addition in Epic #1 is an "experiment catalog" in MySQL (`model_experiment`) that ingests training reports/metrics from `artifacts/` and `ml/`, exposes full CRUD, generates a per-experiment high-level description (50–80 words), and can export an LLM-friendly text report of every experiment record (including metadata and metrics).

## Scope for this repo (important)
This repository will implement:
- Epic #1: Java ingestion + MySQL persistence for MOS + CLI
Later epics will add:
- Epic #2: Python ML training
- Epic #3: Python backtesting vs Kalshi prices

## Non-negotiable rules
1) All JPA entities, enums, and shared DB models MUST live in module: `models`
2) All timestamps are stored in UTC, except:
   - `target_date_local` which is a DATE
   - `station_zoneid` stored as string
3) No forward-looking leakage:
   - MOS `runtime_utc` used for as-of features must be <= `asof_utc`
4) Idempotency:
   - Every ingestion job can be killed and restarted without manual cleanup
5) Auditing:
   - store raw payload hashes and retrieved timestamps
6) Build:
   - `mvn clean install` from repo root must pass at every commit

## Repository Layout (high-level)
The repo is a multi-module Maven build. Important folders:
- `models/`: JPA entities + Flyway migrations + shared DB model code (must stay here per rule #1).
- `ingestion-service/`: Spring Boot service that runs ingestion tasks, exposes CRUD APIs, and runs CLI-style command line runners.
- `common/`: shared Java utilities (hashing, etc.).
- `artifacts/`: experiment outputs, evaluation runs, plots, and sweep directories produced by Python workflows.
- `ml/`: Python training pipelines and their artifacts (also ingested into `model_experiment`).
- `tools/`: scripts and local utilities (including the live predictor runner in `tools/live/`).

This repo uses a "DB is the system of record" approach: once ingested, the DB row is canonical; raw JSON is stored as text blobs for auditability; downstream tools should use the DB export/report instead of scraping ad-hoc files.

## Database: MySQL + Flyway
The ingestion service expects a local MySQL instance. The default profile is `mysql` (see `ingestion-service/src/main/resources/application-mysql.yml`). Flyway is configured to load migrations from:
- `ingestion-service/src/main/resources/db/migration`
- `models/src/main/resources/db/migration`

All schema changes should be done via versioned SQL migrations, not Hibernate auto-DDL. If you add tables/columns used by JPA entities, put the migration in `models/src/main/resources/db/migration` so it stays coupled to the shared DB model module.

Guidelines:
- Migrations are append-only (`V13__...sql`, `V14__...sql`, etc.). Never edit historical migrations in a shared environment.
- Keep UTC semantics: use `TIMESTAMP`/`DATETIME` columns that store UTC values, and name them with `_utc` suffix. Avoid local time unless explicitly allowed.
- Idempotency in ingestion is primarily achieved through unique keys and upserts; migrations must support these constraints (unique indexes, foreign keys as needed).

## Experiment Catalog (model_experiment)
The `model_experiment` table is a catalog of model runs/experiments ingested from file artifacts. It exists so that:
- you can query experiments by station/model family,
- sort by test MAE (or any metric),
- keep raw metadata and metric JSON blobs for auditability,
- attach a high-quality, high-level natural language description of what the experiment is trying to do (not its numeric results),
- export a single structured text report suitable for an LLM planning conversation.

Key implementation points:
- JPA entity: `models/src/main/java/com/predictionmarkets/weather/models/ModelExperiment.java`
- Migration: `models/src/main/resources/db/migration/V12__model_experiment.sql`
- Repository: `ingestion-service/src/main/java/com/predictionmarkets/weather/repository/ModelExperimentRepository.java`
- CRUD controller: `ingestion-service/src/main/java/com/predictionmarkets/weather/experiments/ModelExperimentController.java`

### Identity, auditing, and idempotency
Every `ModelExperiment` row has a stable identity and audit fields:
- `experiment_key` is unique and is used for idempotent upserts. It is constructed from the repo-relative source path plus a suffix (e.g., `::item_0` for array entries or `::name` for named configs).
- `raw_payload_hash` is a SHA-256 hash of the ingested metadata JSON so you can detect changes.
- `retrieved_at_utc` records the ingestion timestamp (when the file was read).
- `created_at_utc` and `updated_at_utc` support audit history for row creation/updates.

Ingest jobs must be safe to re-run:
- If an experiment file is re-ingested, it updates the existing row for that `experiment_key` rather than creating a duplicate.
- If the process is killed mid-run, the next run re-processes candidates and converges to a correct state.

## Experiment Ingestion (scans artifacts/ and ml/)
Experiment ingestion walks one or more directories, finds candidate JSON files, parses them, and upserts one or more `ModelExperiment` rows per file.

Implementation:
- Ingest service: `ingestion-service/src/main/java/com/predictionmarkets/weather/experiments/ModelExperimentIngestService.java`
- Properties: `ingestion-service/src/main/java/com/predictionmarkets/weather/experiments/ModelExperimentIngestProperties.java`

Candidate file patterns (current behavior):
- `metrics*.json`
- `report*.json`
- `results.json` (often an array of configurations; each array item becomes a row)
- `summary.json` (best-only shortcut in some sweeps)

Station filtering:
- The ingest scan supports `station-filter` so you can focus on a station such as `KMIA`.
- Station detection is done from the file path, raw JSON content, and selected "extra files" that live next to the metrics/report file.

Metadata schema and extras:
- Ingest stores `metadata_json` as a structured object with `schema_version` and `primary_path` plus:
  - a `raw.primary` node containing the parsed JSON that triggered ingestion,
  - optional `raw.extra` node (e.g., `summary.json` when ingesting `results.json`),
  - optional `extras` map containing adjacent files like `experiment_meta.json`, `feature_list*.json`, `config_resolved.yaml`, and other run context.
- The goal is "everything needed to understand the run later" without losing provenance.

### Anti-leakage note
The experiment catalog is mostly historical evaluation data; it does not directly feed the MOS as-of features. However, the repo-wide leakage rule still applies to anything that builds predictive features for as-of inference. Do not create feature pipelines that can see future truth values relative to the as-of timestamp.

## Per-Experiment Description Generation (50–80 words)
Each experiment row has a `description_text` field. This is not a template and not a metrics summary. It is intended to be a short, high-level narrative of:
- what the experiment is attempting to validate,
- what lever it is changing (feature set, guidance sources, analog/kNN layer, sweep vs baseline, etc.),
- what family of runs it belongs to (e.g., a sweep family vs a targeted augmentation run),
- and how to interpret its intent at a glance.

Rules:
- Length is enforced in a range: 50–80 words.
- Do not mention numerical metrics (MAE/RMSE/etc.). Those are stored in dedicated columns and exported in the report.
- Descriptions must be individually derived from the experiment's metadata/feature signals so they do not collapse into near-identical filler text.

Implementation:
- Generator + snapshot/apply pipeline:
  - `ingestion-service/src/main/java/com/predictionmarkets/weather/experiments/ModelExperimentDescriptionService.java`
- CLI runner (writes snapshot first, then updates DB):
  - `ingestion-service/src/main/java/com/predictionmarkets/weather/experiments/ModelExperimentDescribeCommandLineRunner.java`

Snapshot-first workflow (important):
1) Generate descriptions for all rows and write them to an NDJSON "snapshot" file.
2) Optionally review/edit the snapshot externally.
3) Apply the snapshot back into the DB (update `description_text` per id).

This approach gives you a stable artifact you can diff/review and ensures DB updates are deliberate rather than "mystery changed" by a background job.

Snapshot outputs are written under repo-root `artifacts/experiment_descriptions/` with names like:
- `model_experiment_descriptions_YYYYMMDDTHHMMSSZ.ndjson`
- `model_experiment_descriptions_YYYYMMDDTHHMMSSZ.md`

The Markdown version is a human-readable review file; the NDJSON version is the exact "apply input".

## Aggregated Experiment Report (LLM-friendly)
The report executor exports a single text file containing every `ModelExperiment` row, item-by-item, with all scalar fields and full JSON blobs. This is intended as the primary input to an LLM (e.g., GPT-Pro) to plan future experiments.

Implementation:
- Report service: `ingestion-service/src/main/java/com/predictionmarkets/weather/experiments/ModelExperimentReportService.java`
- CLI runner: `ingestion-service/src/main/java/com/predictionmarkets/weather/experiments/ModelExperimentReportCommandLineRunner.java`
- Properties: `ingestion-service/src/main/java/com/predictionmarkets/weather/experiments/ModelExperimentReportProperties.java`

Report file characteristics:
- Output is stable, parseable text with explicit record boundaries:
  - header: `MODEL_EXPERIMENT_REPORT`
  - record start: `EXPERIMENT_START`
  - record end: `EXPERIMENT_END`
- Each experiment record includes:
  - identity and provenance (id/key/source_path/hashes/timestamps),
  - scalar metrics for train/validation/test (mae, rmse, bias, etc.) when present,
  - the 50–80 word `description_text`,
  - raw metric JSON blobs,
  - pretty-printed `metadata_json` block.
- Rows are sorted by `test_mae` (ascending, nulls last) and then `experiment_key` to make "best-first" scanning easy.

Report outputs are written under repo-root `artifacts/experiment_reports/` with names like:
- `model-experiments-KMIA-YYYYMMDDTHHMMSSZ.txt`

Important practical note: this file can be large. For LLM usage you may need to:
- chunk it (split by `EXPERIMENT_START` boundaries),
- or run with a limit (e.g., top 200 by test MAE) for iterative strategy discussions.

## Full CRUD for model_experiment
The ingestion-service exposes CRUD endpoints for the experiment catalog. Controller:
- `ingestion-service/src/main/java/com/predictionmarkets/weather/experiments/ModelExperimentController.java`

Endpoints (current behavior):
- `POST /api/model-experiments` creates/updates by `experimentKey` (idempotent key-based write path).
- `GET /api/model-experiments/{id}` fetches a row by id.
- `GET /api/model-experiments?stationId=...&modelFamily=...&limit=...` lists rows with optional filters.
- `GET /api/model-experiments/by-key?experimentKey=...` fetches a row by key.
- `PUT /api/model-experiments/{id}` updates a row by id.
- `DELETE /api/model-experiments/{id}` deletes a row by id.

In CLI mode (`--spring.main.web-application-type=none`) the web server is disabled and these endpoints are not served. Run the service normally if you want API access.

## How to Run (common flows)
All commands are intended to be run from the repo root. Use the `ingestion-service` module via Maven so it picks up the correct classpath and Flyway configuration.

1) Build everything (must pass):
- `mvn clean install`

2) Ingest experiments from `artifacts/` and `ml/` into MySQL:
- `mvn -pl ingestion-service spring-boot:run -D"spring-boot.run.arguments=--spring.main.web-application-type=none --experiments.ingest.enabled=true"`

3) Generate + apply descriptions (snapshot-first):
- `mvn -pl ingestion-service spring-boot:run -D"spring-boot.run.arguments=--spring.main.web-application-type=none --experiments.describe.enabled=true"`

If you want to only write the snapshot and not update the DB:
- set `experiments.describe.apply-to-database: false` in `ingestion-service/src/main/resources/application.yml`

4) Generate the aggregated report text file for LLM usage:
- `mvn -pl ingestion-service spring-boot:run -D"spring-boot.run.arguments=--spring.main.web-application-type=none --experiments.report.enabled=true"`

5) Generate report for a smaller subset (example):
- Set `experiments.report.limit: 200` in `ingestion-service/src/main/resources/application.yml` to export the top 200 by test MAE.

## TFS2 Sweep v2 (DB-backed) + MOS Alignment Fix (added 2026-02-16)
This section documents the new DB-backed sweep pipeline and the MOS alignment fixes that made MOS features usable for KMIA. This is critical context because MOS data *exists* in `mos_daily_value` but was previously filtered out by overly strict runtime/retrieved guards.

### Where the code lives
Core sweep + experiments:
- `ml/src/weather_ml/tfs2/` (config, data load, experiments, sweep runner)
- Entry point: `ml/run_tfs2_sweep.py`

MOS pipeline used by TFS2:
- `ml/src/weather_ml/mos_mos_features.py`
- `ml/src/weather_ml/mos_config.py`
- `ml/src/weather_ml/tfs2/data.py`

Experiment DB persistence:
- `ml/src/weather_ml/experiment_results_db.py`
- `ml/src/weather_ml/tfs2/db.py`

### Forecast timing invariant
The TFS2 pipeline assumes *every* forecast is made at **T‑1 12:00Z** for target day **T**:
- Gribstream selection uses `asof_utc == (target_date_local at 12Z) - 1 day`
- MOS selection uses `asof_utc <= cutoff` where cutoff is the same **T‑1 12:00Z**
- Error-based features use `TRUTH_LAG_DAYS = 2` (so all error history is ≤ D‑2)

### MOS availability fix (root cause + resolution)
**Root cause:** MOS rows existed, but selection logic filtered *everything* because `retrieved_at_utc` was always later than the as‑of cutoff. This is a common pattern when ingestion time is later than the as‑of time, even though the forecast is valid at as‑of.

**Fixes applied:**
1) **MOS eligibility now uses `asof_utc` as the primary gate.**
   - `select_latest_mos()` now filters on `asof_utc <= asof_utc_cutoff`.
   - `runtime_utc` is only a *tie‑breaker* for “latest”, not a hard filter.

2) **Disable retrieved_at guard for TFS2** (prevents false empty MOS):
   - In `ml/src/weather_ml/tfs2/data.py`, the MOS config now sets:
     - `include_retrieved_at_guard=False`

3) **Case‑insensitive filtering in SQL** for model/variable:
   - `fetch_mos_rows()` now filters `UPPER(model)` and `LOWER(variable_code)` so case mismatches do not drop rows.

4) **Detailed MOS diagnostics logging**:
   - Counts at each selection stage (total, eligible_asof, eligible_runtime, post‑guard).
   - Distinct model and variable lists.
   - When empty, a “MOS_DEBUG” summary is logged with available models/variables/asof hours.

### MOS selection rule (current, correct behavior)
**Rule:** For each target day, MOS rows are eligible if:
- `asof_utc <= target_date_local @ 12Z − 1 day`
- (retrieved_at guard is disabled for TFS2 to avoid false negatives)
- model and variable are within the configured sets

This makes MOS match the forecast’s as‑of time while staying leakage‑safe.

### How to validate MOS is now flowing
Quick sanity check from Python:
```
python - <<'PY'
from weather_ml.tfs2 import data, config
engine = data.create_engine_from_url(None)
bundle = data.build_dataset(engine, 'KMIA', config.DEFAULT_SPLIT.train_start, config.DEFAULT_SPLIT.test_end)
mos_cols = [c for c in bundle.df.columns if c.startswith('mos_')]
print('mos_cols', len(mos_cols))
print(bundle.df[mos_cols].notna().sum().head(10))
PY
```
If MOS is working, `mos_cols` should be **> 0** and non‑null counts should be non‑zero.

### TFS2 sweep runner behavior (important)
File: `ml/src/weather_ml/tfs2/sweep.py`

Key behaviors:
- **Extremely detailed logging** (dataset build, MOS counts, baselines, per‑experiment timing, heartbeats).
- **Heartbeat messages** for long experiments (default every 60s).
- **Skip slow experiments by default** to keep runtime reasonable:
  - `CatBoost-MAEWithBustMOS`
  - `XGB-QuantileTrioMeanReconstruction`
  - `CorrectedForecastLibrary-StackRidge`
  - `LocalLinearAnalogCalibration-LLR`
- Use `--include-slow` to run *everything*.

CLI examples:
```
# Fast run (skip slow)
python ml/run_tfs2_sweep.py --station KMIA

# Full run (include slow)
python ml/run_tfs2_sweep.py --station KMIA --include-slow
```

Outputs:
- NDJSON: `artifacts/time_feature_sweep_v2/<SWEEP_ID>/tfs2_results.ndjson`
- Summary JSON: `artifacts/time_feature_sweep_v2/<SWEEP_ID>/time_feature_sweep_v2.json`
- Log file: `artifacts/time_feature_sweep_v2/<SWEEP_ID>/tfs2_sweep.log`

### DB persistence (both experiment DB + model_experiment)
The sweep persists to:
1) **experiment_results_db** (separate DB):
   - `experiment_sweeps`, `experiment_variants`, `experiment_metrics`
   - Default DB: `weather_predictionmarkets_experiments`
   - `experiment_results_db.persist_sweep(...)` is called inside the sweep runner
2) **model_experiment** (core catalog):
   - Written via `ml/src/weather_ml/tfs2/db.py`

Environment variables:
- `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_HOST`, `MYSQL_PORT` used by default.

### Known notes from latest run (MOS fixed)
Sweep example:
- `artifacts/time_feature_sweep_v2/20260216T083249Z/`

MOS is now loaded:
- Dataset columns increased to include ~350 MOS features.

Baseline A/B changed due to MOS presence (though bust classifier still collapsed to a single class in this run).

### Validity note (important)
- **Any sweep run before the MOS fix is NOT valid for MOS‑related conclusions.** Those runs had `DATA_FETCH_MOS_EMPTY` and effectively used no MOS features.
- The **first valid MOS sweep** is:
  - `artifacts/time_feature_sweep_v2/20260216T083249Z/`

### Best results (latest valid sweep)
Baseline B test MAE: **1.1307**

Top performers by test MAE:
1. **KNNResidualCorrection‑ForecastSpace** — **1.0609** (Δ vs Baseline B: **−0.0699**)
2. **PCA2+MOSRegimeGMMProbFeatures** — **1.0942** (Δ −0.0365)
3. **MOS‑WindVectorSeaBreezeGating** — **1.0947** (Δ −0.0360)
4. **SeasonalAdaptiveHalfLifeBias** — **1.0959** (Δ −0.0349)
5. **ErrorCorrelationDiversityWeights** — **1.0975** (Δ −0.0332)

These results are from `time_feature_sweep_v2.json` at:
- `artifacts/time_feature_sweep_v2/20260216T083249Z/time_feature_sweep_v2.json`

### Why bust classifier can still collapse
The bust classifier uses **training residual labels**. If the labeler sees only one class (e.g., due to how residual bins were derived), it falls back to neutral probabilities. This is logged:
```
Bust classifier labels collapsed to a single class; using neutral probabilities.
```
If you want to fix this next, we can:
- Adjust label thresholds (e.g., quantiles computed on train only)
- Add a minimum class count rule and auto‑widen bins

### Summary of code changes (for traceability)
- `ml/src/weather_ml/mos_mos_features.py`:
  - Added MOS debug logging
  - Case‑insensitive filters
  - `asof_utc` eligibility (primary), `runtime_utc` tie‑breaker
- `ml/src/weather_ml/tfs2/data.py`:
  - Disabled retrieved‑at guard for TFS2 MOS
  - Added dataset build logging
- `ml/src/weather_ml/tfs2/sweep.py`:
  - Added heartbeat + skip‑slow mechanism + persistent logging
- `ml/src/weather_ml/tfs2/experiments.py`:
  - Additional context logs and minor safety improvements

## Configuration knobs to know about
The CLI runners are controlled by `experiments.*` config in `ingestion-service/src/main/resources/application.yml`:
- `experiments.ingest.enabled`: turn ingestion on/off.
- `experiments.ingest.scan-roots`: where to look for candidate files (default includes `artifacts` and `ml`).
- `experiments.ingest.station-filter`: station allowlist (default `KMIA`).
- `experiments.report.enabled`: turn report generation on/off.
- `experiments.report.output-dir`: where to write reports.
- `experiments.report.station-filter`: which stations to include in the report.
- `experiments.report.limit`: export top N (after sorting by test MAE).
- `experiments.describe.enabled`: turn description generation on/off.
- `experiments.describe.output-dir`: where to write description snapshots.
- `experiments.describe.apply-to-database`: preview-only vs apply changes to MySQL.
- `experiments.describe.limit`: cap number of rows described/updated.

## Security + hygiene
- Do not hardcode tokens/credentials in new code or docs. Prefer environment variables or a local override YAML (e.g., `application-local.yml`) that is gitignored.
- Experiment metadata JSON can include file paths and dataset identifiers. Treat exported reports as potentially sensitive if you share them.

## Development guidance (how to extend this safely)
When adding new experiment artifact formats:
- Add a narrow, deterministic parser in `ModelExperimentIngestService`.
- Preserve provenance: store the raw parsed JSON in `metadata_json.raw.primary`, and add any adjacent context files into `metadata_json.extras`.
- Maintain idempotency via stable `experiment_key` construction.

When adding new DB fields:
- Add columns via Flyway in `models/src/main/resources/db/migration`.
- Update `ModelExperiment` in `models/` only (rule #1).
- Re-run `mvn clean install` and validate a local migration to confirm schema compatibility.

When editing description generation:
- Treat it as a data product: update logic, generate a snapshot, inspect the Markdown review file, then apply.
- Keep descriptions focused on intent and levers; never summarize numeric results in free text.

## Operational Sanity Checks (recommended before/after runs)
These quick checks help confirm that ingestion/report/description jobs did what you think they did. The idea is to make failures obvious (wrong station, partial update, duplicate keys, broken word range, etc.) without manually eyeballing thousands of rows.

DB checks (MySQL CLI examples):
- Total records:
  - `SELECT COUNT(*) FROM model_experiment;`
- Ensure no duplicate keys (should be zero rows):
  - `SELECT experiment_key, COUNT(*) c FROM model_experiment GROUP BY experiment_key HAVING c > 1;`
- Ensure descriptions exist and are not empty:
  - `SELECT COUNT(*) FROM model_experiment WHERE description_text IS NULL OR TRIM(description_text)='';`
- Ensure descriptions do not contain metric terms you don't want in free text:
  - `SELECT COUNT(*) FROM model_experiment WHERE LOWER(description_text) LIKE '%mae%' OR LOWER(description_text) LIKE '%rmse%';`
- Ensure word-range compliance using a simple space-count approximation:
  - `SELECT COUNT(*) FROM model_experiment WHERE (1 + LENGTH(TRIM(description_text)) - LENGTH(REPLACE(TRIM(description_text),' ',''))) < 50 OR (1 + LENGTH(TRIM(description_text)) - LENGTH(REPLACE(TRIM(description_text),' ',''))) > 80;`
- Spot-check a handful of descriptions (best/worst/most recent):
  - `SELECT id, experiment_key, test_mae, LEFT(description_text, 240) FROM model_experiment ORDER BY test_mae ASC LIMIT 10;`
  - `SELECT id, experiment_key, test_mae, LEFT(description_text, 240) FROM model_experiment ORDER BY test_mae DESC LIMIT 10;`

File checks:
- The describe runner always writes a snapshot first. Confirm the newest files exist:
  - `artifacts/experiment_descriptions/model_experiment_descriptions_*.ndjson`
  - `artifacts/experiment_descriptions/model_experiment_descriptions_*.md`
- The report runner writes exactly one aggregated file per invocation:
  - `artifacts/experiment_reports/model-experiments-*.txt`

If you care about "exactly what changed" for descriptions, compare the DB row `updated_at_utc` before/after a run, or diff the Markdown snapshot outputs between runs.

## Using The Aggregated Report With GPT-Pro (practical workflow)
The report file is intentionally verbose and self-contained, but LLM context windows are finite. A reliable workflow is:
1) Start with the top slice:
   - Set `experiments.report.limit` to something like 100–300 to get a best-first subset.
2) Ask for strategy:
   - Provide the first chunk (e.g., first 30–50 experiments) and ask GPT-Pro to propose 5–10 new experiments, with rationales tied to patterns in those records.
3) Iterate with coverage:
   - Feed additional chunks representing different run families (sweeps, baselines, MOS augmentation, kNN analog runs, etc.).
4) Only then scale up:
   - Export the full report when you need broad coverage, and have GPT-Pro produce a "family map" or run taxonomy rather than reading every record linearly.

Prompting tip:
- Ask GPT-Pro to treat `EXPERIMENT_START/END` blocks as atomic records and to never infer fields that are not present.
- Ask it to separate "observed patterns" from "proposed next experiments" to avoid hallucinated claims.

## Troubleshooting (common failure modes)
MySQL connection issues:
- Confirm MySQL is running and reachable on `localhost:3306`.
- Check `ingestion-service/src/main/resources/application-mysql.yml` for username/password/database name.
- If you use a different DB or credentials, override via Spring profiles or a local YAML override rather than editing committed config.

Flyway migration issues:
- If Flyway reports a checksum mismatch, it usually means a historical migration was edited. In a shared repo, the fix is almost always "create a new migration that corrects the schema" rather than rewriting old SQL.

Report/describe output not appearing where expected:
- Outputs resolve relative to the repo root. Ensure you are running from the repo (the code searches upward for `pom.xml`, `models/`, and `ingestion-service/`).
- If you run the jar from a different working directory, pass absolute paths for `experiments.report.output-dir` / `experiments.describe.output-dir`.

Very large reports:
- Use `experiments.report.limit` to reduce size.
- Remember that `metadata_json` can include embedded extras; the report prints it pretty-printed, which is LLM-friendly but increases file size.

Partial updates:
- Ingest/report/describe jobs are restartable; re-run the command. Idempotency is handled via `experiment_key` upserts and snapshot application by id.

## NOAA Backfill Ingestion Notes (2026-02-17)
We added NOAA sanity/backfill scripts under `ingestion-service/noaa_sanity/` to get daily Tmax forecasts for KMIA using as-of runs, because Gribstream data does not extend back to 2017.

What we confirmed as working (NOAA public sources):
1. **HRRR** via NOAA AWS `noaa-hrrr-bdp-pds` + Herbie/xarray (point extraction) works for historical dates we tested.
2. **GEFS (mean/control)** via NOAA AWS `noaa-gefs-pds` + Herbie/xarray works, including legacy 2017 layout (member discovery from S3 prefixes).

What we could NOT confirm (as of 2026-02-17):
1. **RAP 2017 via NCEI THREDDS**: The day catalogs returned HTTP 404 even after adding the nested-month layout (`.../YYYYMM/YYYYMM/YYYYMMDD/catalog.xml`). The script reports `not_found_or_download_failed` for 2017-05-05.
2. **NBM 2017 via AWS NODD**: Listing common prefixes in `s3://noaa-nbm-grib2-pds` for 2017-05-04 12Z returned zero keys (likely no public backfill under standard layout).

Key scripts and how to run:
1. RAP THREDDS attempt (fails for 2017 as of now):
   `python ingestion-service/noaa_sanity/fetch_rap_tmax_ncei_thredds.py --date 2017-05-05 --station KMIA --lat 25.7959 --lon -80.2870 --tz America/New_York --out rap_kmia_20170505.csv --log-level DEBUG --prefer-ncss --dump-values`
2. NBM AWS presence check:
   `python ingestion-service/noaa_sanity/check_nbm_aws_2017.py --date 2017-05-04 --cycle 12`
3. HRRR + GEFS daily Tmax (NOAA AWS + Herbie):
   `python ingestion-service/noaa_sanity/fetch_noaa_tmax_hrrr_gefs_mean.py --date YYYY-MM-DD --station KMIA --lat 25.7959 --lon -80.2870 --tz America/New_York --out noaa_tmax_kmia_YYYYMMDD.csv --max-workers 1 --dump-values`

Interpretation:
1. **GEFS/HRRR are confirmed available** for 2017 backfill via NOAA AWS.
2. **RAP is not confirmed** for 2017 via NCEI THREDDS because catalog discovery failed (404 for day catalogs).
3. **NBM is not confirmed** for 2017 via AWS NODD (no keys under standard prefixes).
