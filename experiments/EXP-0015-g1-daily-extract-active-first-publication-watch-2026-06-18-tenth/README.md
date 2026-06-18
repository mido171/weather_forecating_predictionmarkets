# EXP-0015 - G1 Daily Extract active first-publication watch 2026-06-18 tenth continuation

**Status:** ACCEPTED
**Created:** 2026-06-18T19:13:58.220762Z
**Goal dependency:** G1 settlement truth and Daily Extract first-publication evidence
**Owner:** Codex

## Plain-language question

Does the HKO June 2026 Daily Extract backing payload publish the
`2026-06-18` row during this tenth bounded continuation watch?

## Why this could improve Tmax

This does not improve a forecast model. It improves the target-truth foundation
by continuing the active absent-before-present archive needed to identify the
first public Daily Extract value for the HKO daily maximum temperature target.

## What changed from prior work

EXP-0014 completed the ninth continuation and found that `2026-06-18` remained
absent through `2026-06-18T19:10:57.032373Z`. EXP-0015 extends the same live
watch with the same bounded, retry-backed polling protocol.

## Current conclusion

The tenth continuation poll completed six iterations. The June 2026 HKO Daily
Extract monthly payload still contained 17 rows, so `2026-06-18` remained
absent through `2026-06-18T19:18:33.818526Z`. G1 remains blocked pending an
actual provider-first publication candidate.

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
