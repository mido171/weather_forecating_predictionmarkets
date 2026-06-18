# EXP-0005 - G1 Daily Extract active first-publication watch 2026-06-18

**Status:** ACCEPTED
**Created:** 2026-06-18T17:44:06.256854Z
**Goal dependency:** G1
**Owner:** Codex

## Plain-language question

Can the HKO Daily Extract archive prove that the `2026-06-18` HKO daily
absolute maximum row appears during an active polling window, rather than only
being first seen by this repository after the fact?

## Why this could improve Tmax

This is target-system evidence, not a forecast. G1 needs proof that the
settlement target can be captured at first publication before any historical
label proxy is trusted for model training.

## What changed from prior work

EXP-0004 added bounded polling and watched-date gates. EXP-0005 strengthens the
candidate rule: a provider-first candidate requires an archived active-polling
absence before the first archived presence.

## Current conclusion

Accepted as G1 infrastructure. The stricter active absent-before-present
candidate rule passed tests, and the bounded active watch completed, but
`2026-06-18` was still missing from the HKO Daily Extract monthly payload. G1
remains blocked pending actual first-publication evidence.

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
