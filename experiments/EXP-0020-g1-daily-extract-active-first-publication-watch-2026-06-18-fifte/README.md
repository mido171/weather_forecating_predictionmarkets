# EXP-0020 - G1 Daily Extract active first-publication watch 2026-06-18 fifteenth continuation

**Status:** ACCEPTED
**Created:** 2026-06-18T19:54:04.320375Z
**Goal dependency:** G1 settlement truth and Daily Extract first-publication evidence
**Owner:** Codex

## Plain-language question

Does the HKO June 2026 Daily Extract backing payload publish the
`2026-06-18` row during this fifteenth bounded continuation watch?

## Why this could improve Tmax

This does not improve a forecast model. It improves the target-truth foundation
by continuing the active absent-before-present archive needed to identify the
first public Daily Extract value for the HKO daily maximum temperature target.

## What changed from prior work

EXP-0019 completed the fourteenth continuation and found that `2026-06-18`
remained absent through `2026-06-18T19:50:50.551470Z`. EXP-0020 extends the
same live watch with the same bounded, retry-backed polling protocol.

## Current conclusion

The fifteenth continuation poll completed six iterations. The June 2026 HKO
Daily Extract monthly payload still contained 17 rows, so `2026-06-18` remained
absent through `2026-06-18T19:58:46.373498Z`. G1 remains blocked pending an
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
