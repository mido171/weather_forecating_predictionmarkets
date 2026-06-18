# EXP-0026 — G1 Daily Extract active first-publication watch 2026-06-18 twenty-first continuation

**Status:** ACCEPTED
**Created:** 2026-06-18T20:45:56.643080Z
**Goal dependency:** G1 settlement truth / Daily Extract first-publication evidence
**Owner:** Codex

## Plain-language question

Does the twenty-first continuation of the active HKO Daily Extract watch capture
a public absent-to-present transition for the `2026-06-18` HKO daily maximum
temperature row?

## Why this could improve Tmax

It cannot improve a forecast yet. It can improve the target system by proving
when the official Daily Extract settlement candidate first becomes available,
which is required before any leakage-safe modelling or horizon selection.

## What changed from prior work

EXP-0025 kept `2026-06-18` absent through
`2026-06-18T20:42:02.053116Z`. EXP-0026 continues the same active watch with
the same bounded retry-backed polling protocol and a new metrics artifact.

## Current conclusion

Accepted as a G1 checkpoint only. The twenty-first continuation poll completed
six iterations; `2026-06-18` remained absent through
`2026-06-18T20:50:08.714392Z`, so G1 remains blocked.

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
