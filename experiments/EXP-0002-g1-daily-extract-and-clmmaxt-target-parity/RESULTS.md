# Results

## Run Integrity

- command: `.\.venv\Scripts\python.exe scripts\build_target_parity.py --year 2026 --month 5 --daily-source-id hko_daily_extract_202605`
- start/end: 2026-06-18 local run, deterministic after raw snapshots
- code commit: `bcbccd5a6f18d573bfa7a8a8308d3893be123d9d` plus dirty EXP-0002/parser changes
- dirty state: true, documented in this experiment
- data manifest hash: pending final manifest hash
- rows: 31
- failed rows: 0 latest-payload mismatches; 31 rows still lack first-publication proof
- leakage validator: no forecast features or model fitting performed
- reproducibility precheck: focused parser/adapter tests passed

## Primary Result

| Metric | Baseline | Candidate | Absolute delta | Relative delta | 95% CI |
|---|---:|---:|---:|---:|---|
| Latest Daily Extract vs latest CLMMAXT HKO exact-match rate, May 2026 | - | 1.000 (31/31) | - | - | n/a |
| First-publication parity proven | - | false | - | - | n/a |

## Guardrails

No predictive model, baseline, locked test, or market backtest was run. The
result is a target-data-system smoke/parity slice only.

## Year-By-Year

Only May 2026 was evaluated in this checkpoint.

## Regime Breakdown

Not applicable.

## Calibration

Not applicable.

## Boundary And Tail Days

No bucket/tail market mapping evaluated. Polymarket backtesting is deferred by
user instruction.

## Ablation

Not applicable.

## Sensitivity

Not applicable.

## Negative Controls

Fail-closed parser/adapter tests cover missing source, missing field, ambiguous
date, unsupported precision, station mismatch, missing value, source failure,
incomplete Daily Extract markers, and bilingual CLMMAXT headers.

## Worst Cases And Failure Taxonomy

- `MISSING_FIRST_PUBLICATION`: all 31 May 2026 rows, because the first public
  Daily Extract payloads were not captured at publication time.
- `CLMMAXT_DIFFERENCE`: 0 rows in this latest-payload sample.

## Missingness/Common Sample

Daily Extract 2026-05 and CLMMAXT HKO have a 31-row common sample. All compared
rows have complete numeric values.

## Compute And Operational Cost

Local parsing only after raw HTTP snapshots. No model training.

## Unexpected Findings

- The Daily Extract HTML page is a JavaScript shell; target values are in
  root-relative JSON-text backing payloads under `/cis/dailyExtract/`.
- The backing payloads use `.xml` URLs/content type but contain JSON text.
- The existing CLMMAXT parser needed bilingual-header and footer-row support to
  parse real HKO CSVs safely.

## Full Artifact List

- `data/gold/target_parity/target_parity.csv` (ignored generated artifact)
- `reports/target_parity.md`
- `experiments/EXP-0002-g1-daily-extract-and-clmmaxt-target-parity/results/metrics.json`
- `docs/source_contracts/hko_daily_extract.md`
- `docs/source_contracts/hko_clmmaxt_hko.md`
- `docs/source_contracts/hko_station_metadata.md`
