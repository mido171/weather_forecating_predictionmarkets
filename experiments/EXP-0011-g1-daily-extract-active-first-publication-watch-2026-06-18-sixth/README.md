# EXP-0011 - G1 Daily Extract active first-publication watch 2026-06-18 sixth continuation

**Status:** ACCEPTED
**Created:** 2026-06-18T18:41:32.205965Z
**Goal dependency:** G1
**Owner:** Codex

## Plain-language question

Can a sixth continuation poll capture the HKO Daily Extract `2026-06-18` row
appearing after the absent active snapshots from EXP-0005 through EXP-0010?

## Why this could improve Tmax

This is settlement-truth infrastructure only. G1 needs a documented
first-publication capture path before CLMMAXT or any latest target payload can
be treated as canonical training truth.

## What changed from prior work

EXP-0010 completed six retry-backed poll iterations and found `2026-06-18`
still absent through `2026-06-18T18:38:03.144324Z`. EXP-0011 extends the same
active watch with the inherited active start.

## Current conclusion

Accepted as G1 infrastructure. Six additional active poll iterations completed
successfully, but the watched `2026-06-18` row remained absent through
`2026-06-18T18:45:51.464137Z`. G1 remains blocked pending actual provider
first-publication evidence.

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
