# Results

## Failure Diagnosed

The first continuation run aborted before metrics were written because HKO
disconnected during a monthly payload request:

`FetchError: Request failed for hko_daily_extract_202606 ... Server disconnected without sending a response.`

This was not treated as a successful poll. `FetchPolicy` was extended with
explicit bounded retry settings, and retry behavior was covered by tests before
rerunning.

The aborted attempt did leave immutable raw evidence:

- catalog `2026-06-18T17:56:05.701956Z`, hash `f80772b68545c56e6842c34998696fd11b7b9a80c0088bb1f6e4da65102616eb`
- monthly `2026-06-18T17:56:07.780924Z`, hash `c50910ab74e2ba8bff1f661fb1ae663d15b128dae0dfb4ed97c0e40c97bcbefc`
- catalog-only `2026-06-18T17:56:44.354141Z`, hash `f80772b68545c56e6842c34998696fd11b7b9a80c0088bb1f6e4da65102616eb`

## Accepted Rerun

- command: `.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 6 --interval-seconds 30 --fetch-attempts 3 --retry-sleep-seconds 2 --active-polling-start-at 2026-06-18T17:48:59.956593Z --watch-candidate-date 2026-06-18 --metrics experiments\EXP-0006-g1-daily-extract-active-first-publication-watch-2026-06-18-conti\results\metrics.json`
- active polling start: `2026-06-18T17:48:59.956593Z`
- polling iterations completed: 6
- poll snapshot count: 6
- fetch attempts per request: 3
- retry sleep seconds: 2

| Metric | Result |
|---|---:|
| Ledger rows | 17 |
| Watched date present | 0 |
| Watched date missing | 1 |
| Provider-first candidates | 0 |
| Revisions observed | 0 |
| Evidence class count: `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST` | 17 |

The watched `2026-06-18` row remained absent through the final accepted monthly
snapshot at `2026-06-18T18:01:26.513653Z`.

## Guardrail Result

The accepted metrics now include per-iteration raw snapshot metadata under
`poll_snapshots`. This makes the active polling cadence auditable from the
experiment artifact. The full gate suite passed after documentation updates:
`pytest`, `hkg_tmax validate all`, `ruff`, and `mypy`.

## Artifacts

- `experiments/EXP-0006-g1-daily-extract-active-first-publication-watch-2026-06-18-conti/results/metrics.json`
- `reports/daily_extract_publication.md`
- `data/gold/target_publication/daily_extract_first_seen.csv` (ignored generated artifact)
