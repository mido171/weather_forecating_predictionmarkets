# EXP-0031 - G1 Daily Extract active first-publication watch 2026-06-18 twenty-sixth continuation

**Status:** ACCEPTED
**Created:** 2026-06-18T21:33:53.549435Z
**Goal dependency:** G1 settlement truth / Daily Extract first-publication evidence
**Owner:** Codex

## Plain-language question

Does the twenty-sixth continuation of the active HKO Daily Extract watch capture
a public absent-to-present transition for the `2026-06-18` HKO daily maximum
temperature row?

## Why this could improve Tmax

This cannot improve forecasts yet. It improves the target system by extending
direct first-publication evidence for the official Daily Extract settlement
candidate. That timing must be proven before CLMMAXT can be trusted as a
historical proxy for modelling labels.

## What changed from prior work

EXP-0030 kept `2026-06-18` absent through
`2026-06-18T21:29:42.956957Z`. EXP-0031 continued the same active watch with
the same bounded retry-backed polling protocol and a new immutable metrics
artifact.

## Current conclusion

Accepted as a G1 checkpoint only. The twenty-sixth continuation poll completed
six iterations; `2026-06-18` remained absent through
`2026-06-18T21:39:42.317252Z`, so G1 remains blocked.

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
