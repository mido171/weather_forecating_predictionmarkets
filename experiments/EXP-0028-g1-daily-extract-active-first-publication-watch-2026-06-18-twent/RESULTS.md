# Results

Run completed.

## Run integrity

- command: declared command in `RUN_CONFIG.yaml`
- start/end: `2026-06-18T21:07:18Z` to `2026-06-18T21:10:15Z`
- code commit: `cb08a2e73397b35a335162ae4d0140059de62152`
- dirty state: EXP-0028 predeclaration plus reservation files
- data manifest hash: not separately hashed
- rows: 17
- failed rows: 0
- leakage validator: PASS target-only no-model checkpoint
- reproducibility precheck: PASS pending final gate suite

## Primary result

| Metric | Value |
|---|---:|
| poll iterations completed | 6 |
| poll snapshot count | 6 |
| row count | 17 |
| provider first-publication candidates | 0 |
| revision count | 0 |
| watched date present | 0 |
| watched date missing | 1 |

`2026-06-18` remained absent through the final monthly retrieval at
`2026-06-18T21:10:15.704745Z`.

## Guardrails

- raw snapshot count: 12
- metadata sidecar count: 12
- unique raw paths: true
- sidecar hash matches raw bytes: true
- HTTP status 200: true
- request and response metadata present: true
- watched date present: false
- watched date missing: true
- problems: none

## Full artifact list

- `results/metrics.json`
- `reports/daily_extract_publication.md`
- raw snapshots under `data/raw/hko_daily_extract_catalog/`
- raw snapshots under `data/raw/hko_daily_extract_202606/`
