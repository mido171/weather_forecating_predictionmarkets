# Changelog

## G1 target-station checkpoint — 2026-06-18

- Added HKO Daily Extract backing-payload parsing for `Absolute Daily Max (deg. C)`.
- Fixed CLMMAXT parsing for real HKO bilingual CSV headers and footer rows.
- Added fail-closed target adapter tests for missing source/field/value, ambiguous
  date, unsupported precision, station mismatch, and source failure.
- Generated a May 2026 latest-payload parity sample: 31/31 Daily Extract rows
  matched CLMMAXT HKO.
- Marked EXP-0002 `BLOCKED` pending first-publication Daily Extract evidence;
  no predictive modelling or Polymarket backtesting was run.
- Added source contracts for Daily Extract, CLMMAXT HKO, and HKO station metadata.
- Added EXP-0003 Daily Extract polling infrastructure and a first-observation
  ledger for June 2026; it is accepted as infrastructure, while G1 remains
  blocked pending provider first-publication evidence.
- Added EXP-0004 bounded Daily Extract polling with explicit metrics output and
  watched-date provider-first candidate gating; G1 remains blocked until actual
  provider first-publication evidence is observed.

## G0 smoke test — 2026-06-18

- Archived every `bootstrap_now` source twice and documented EXP-0001.
- Generated `reports/source_inventory.md` and refreshed `MANIFEST.json`.
- Added fetch failure-path tests for HTTP errors and empty payloads.
- Added malformed HKO climate CSV parser coverage.
- Made experiment creation tests independent of the live registry counter.
- Fixed `scripts/bootstrap.ps1` so native command failures stop the bootstrap.
- Added compatibility docs for requested `HKG_TMAX_FIRST_GOALS.md`,
  `docs/00_PROJECT_OVERVIEW.md`, and `docs/06_LEAKAGE_CONTROL.md` paths.

## 0.1.0 — 2026-06-18

- Created complete Codex research bootstrap.
- Added target-parity and rules-change gates.
- Added point-in-time timestamp model and leakage controls.
- Added immutable raw archive and source catalog scaffolding.
- Added experiment ledger, milestone renderer, specialist agents, and skills.
- Added initial goal program from target verification through production eligibility.
- Added tests for bucket mapping, as-of availability, immutability, and config integrity.
