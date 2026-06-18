# EXP-0002 — G1 Daily Extract and CLMMAXT target parity

**Status:** PREDECLARED  
**Created:** 2026-06-18T17:00:23.286516Z  
**Goal dependency:** G1  
**Owner:** Codex

## Plain-language question

For the Hong Kong Observatory target station, can the Daily Extract field
`Absolute Daily Max (deg. C)` be parsed fail-closed and reconciled with
`CLMMAXT station=HKO` well enough to use CLMMAXT as the historical target-label
proxy for the official Tmax system?

## Why this could improve Tmax

It does not improve forecast skill directly. It prevents the highest-risk
failure mode: training or scoring against a target that differs from the
contract-authoritative settlement field.

## What changed from prior work

EXP-0001 proved the repository and immutable archive path. EXP-0002 now focuses
on the target-station data system: HKO Daily Extract semantics, CLMMAXT station
history, parser safety, and target parity artifacts. Polymarket backtesting,
price history, order books, trades, execution, and market replay are explicitly
deferred until the user requests them later.

## Current conclusion

Predeclared. No parity conclusion has been inspected or accepted yet.

## Navigation

- hypothesis: `HYPOTHESIS.md`
- protocol: `PROTOCOL.md`
- as-of contract: `ASOF_CONTRACT.md`
- data: `DATA_MANIFEST.yaml`
- configuration: `RUN_CONFIG.yaml`
- results: `RESULTS.md`
- conclusion: `CONCLUSION.md`
- reproduction: `REPRODUCE.md`
- gates: `STATUS.yaml`
