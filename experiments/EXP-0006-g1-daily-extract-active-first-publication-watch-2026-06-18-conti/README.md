# EXP-0006 - G1 Daily Extract active first-publication watch 2026-06-18 continuation

**Status:** ACCEPTED
**Created:** 2026-06-18T17:54:21.368079Z
**Goal dependency:** G1
**Owner:** Codex

## Plain-language question

Can continued active polling capture the HKO Daily Extract `2026-06-18` row
appearing after the absent snapshots already archived in EXP-0005?

## Why this could improve Tmax

This is target-system evidence only. Capturing the first publication mechanics
is required before any historical label proxy is treated as canonical training
truth.

## What changed from prior work

EXP-0005 proved the watched row was absent across four active snapshots. EXP-0006
continues that watch and improves metrics so every poll iteration records its
catalog and monthly raw snapshot hash/path, not only the final snapshot.

## Current conclusion

Accepted as G1 infrastructure. The first continuation attempt exposed a
transient HKO monthly-payload disconnect, so fetch retries were added and
tested. The retry-backed continuation poll completed six iterations; the
watched `2026-06-18` row remained absent, leaving G1 blocked.

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
