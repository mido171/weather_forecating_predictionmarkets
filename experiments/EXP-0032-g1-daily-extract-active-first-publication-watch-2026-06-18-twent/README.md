# EXP-0032 - Superseded Daily Extract polling continuation

**Status:** SUPERSEDED
**Created:** 2026-06-18T21:44:55.561273Z
**Superseded by:** HKG Tmax Forecast Data Acquisition Reset
**Owner:** Codex

## Plain-language question

This folder was started by the interrupted Daily Extract polling loop as a
twenty-seventh continuation watch for `2026-06-18`.

## Why it is closed

The user reset the goal on 2026-06-19. The new goal explicitly forbids further
rapid Daily Extract polling windows and forbids creating experiment folders,
test runs, commits, or documentation bundles merely because an unchanged
provider payload was fetched again.

## Preserved evidence

Before the reset arrived, the poll had already completed six iterations. The
metrics artifact is preserved as evidence of that interrupted work. It showed
the same catalog and monthly payload hashes already seen in EXP-0005 through
EXP-0031, with `2026-06-18` still absent through
`2026-06-18T21:49:47.131681Z`.

## Current conclusion

This experiment is not accepted and is not a G1 checkpoint. The repeated
polling family is closed and replaced by operational, ledger-only Daily Extract
collection governed by `config/collector_schedules.yaml`.

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
