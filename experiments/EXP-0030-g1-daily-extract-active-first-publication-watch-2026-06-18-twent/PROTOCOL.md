# Predeclared Protocol

Complete before new provider polling.

## Target and horizon

- target version: HKO Daily Extract first-publication candidate
- rules/target adapter version: current G1 Daily Extract parser and polling ledger
- horizon: not applicable
- exact cutoff: not applicable
- prediction unit: not applicable

## Sample

- target local date: `2026-06-18`
- timezone: `Asia/Hong_Kong`
- active polling start: `2026-06-18T17:48:59.956593Z`
- prior absence checkpoint: `2026-06-18T21:21:19.382593Z`
- inclusion: HKO Daily Extract catalog and June 2026 monthly payload snapshots
- exclusion: forecast features, model outputs, market prices, locked test outcomes
- expected row count: latest monthly payload had 17 rows before this run

## Baseline

- champion/baseline version: none
- frozen prediction artifact: none
- reason: predictive modelling is blocked until G1

## Candidate

- feature/formula/model: none
- transformations: parse archived Daily Extract rows after raw-byte archival
- allowed hyperparameters: none
- selection procedure: none
- seeds: none
- compute budget: six iterations, 30 seconds apart, three fetch attempts per
  request, two seconds between retries

## Metrics

- primary: watched date status, provider first-publication candidate count
- guardrails: raw snapshot count, sidecar count, hash match, HTTP metadata,
  unique raw paths
- calibration: not applicable
- subgroup: not applicable
- operational: bounded polling completes without source/archive failure

## Uncertainty

- method: direct source observation only
- block length: not applicable
- repetitions: six polling iterations
- confidence level: not applicable

## Multiplicity

- experiment family: G1 Daily Extract active first-publication polling
- number of variants: one declared watch window
- correction/confirmation approach: no model selection or statistical promotion

## Acceptance

Accept as a G1 checkpoint if the poll completes and verifies immutable raw
snapshots, sidecars, sidecar hash matches, HTTP metadata, and no overwritten raw
paths. G1 itself remains blocked unless broader target parity criteria pass.

## Rejection

Reject or block if source retrieval fails after retries, raw archival is
incomplete, parser/date/field semantics are ambiguous, metrics are missing, or
post-run gates fail.

## Locked-test decision

This experiment is not authorized to open the locked test.

## Planned command

```powershell
.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 6 --interval-seconds 30 --fetch-attempts 3 --retry-sleep-seconds 2 --active-polling-start-at 2026-06-18T17:48:59.956593Z --watch-candidate-date 2026-06-18 --metrics experiments\EXP-0030-g1-daily-extract-active-first-publication-watch-2026-06-18-twent\results\metrics.json
```
