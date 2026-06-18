# EXP-0007 - G1 Daily Extract active first-publication watch 2026-06-18 second continuation

**Status:** ACCEPTED
**Created:** 2026-06-18T18:05:04.880604Z
**Goal dependency:** G1
**Owner:** Codex

## Plain-language question

Can a second continuation poll capture the HKO Daily Extract `2026-06-18` row
appearing after the absent active snapshots from EXP-0005 and EXP-0006?

## Why this could improve Tmax

This is settlement-truth infrastructure only. G1 needs a documented
first-publication capture path before CLMMAXT or any latest target payload can
be treated as canonical training truth.

## What changed from prior work

EXP-0006 added per-iteration poll snapshot metrics and bounded fetch retries.
EXP-0007 repeats the active watch with the same inherited active start to extend
the absence/presence evidence window.

## Current conclusion

Accepted as G1 infrastructure. Six additional active poll iterations completed
successfully, but the watched `2026-06-18` row remained absent. G1 remains
blocked pending actual provider first-publication evidence.

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
