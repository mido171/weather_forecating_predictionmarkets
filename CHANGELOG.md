# Changelog

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
