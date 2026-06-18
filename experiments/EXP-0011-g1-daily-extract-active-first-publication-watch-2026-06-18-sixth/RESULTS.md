# Results

## Run Integrity

- command: `.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 6 --interval-seconds 30 --fetch-attempts 3 --retry-sleep-seconds 2 --active-polling-start-at 2026-06-18T17:48:59.956593Z --watch-candidate-date 2026-06-18 --metrics experiments\EXP-0011-g1-daily-extract-active-first-publication-watch-2026-06-18-sixth\results\metrics.json`
- active polling start: `2026-06-18T17:48:59.956593Z`
- polling iterations completed: 6
- poll snapshot count: 6
- fetch attempts per request: 3
- retry sleep seconds: 2
- leakage validator: target-publication evidence only; no forecast features or model fitting

## Primary Result

| Metric | Result |
|---|---:|
| Ledger rows | 17 |
| Watched date present | 0 |
| Watched date missing | 1 |
| Provider-first candidates | 0 |
| Revisions observed | 0 |
| Evidence class count: `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST` | 17 |

The watched `2026-06-18` row remained absent through the final monthly snapshot
at `2026-06-18T18:45:51.464137Z`.

## Archive Verification

- unique raw snapshots checked: 12
- metadata sidecars checked: 12
- metadata schema: top-level `content_path`, `content_sha256`, `retrieved_at`, `source_id`, `storage_schema_version`; nested `requested_url`, `final_url`, `http_status`, `request_headers`, `response_headers`
- HTTP status: 200 for all checked sidecars
- raw paths were unique, so no in-place overwrite was observed
- sidecar content hashes matched poll metrics

## Artifacts

- `experiments/EXP-0011-g1-daily-extract-active-first-publication-watch-2026-06-18-sixth/results/metrics.json`
- `reports/daily_extract_publication.md`
- `data/gold/target_publication/daily_extract_first_seen.csv` (ignored generated artifact)

## Gate Checks

- `pytest`: PASS (`59 passed`)
- `hkg_tmax validate all`: PASS with expected G1/G2 gating warnings
- `ruff`: PASS
- `mypy`: PASS
