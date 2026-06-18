# EXP-0003 - G1 Daily Extract first-publication polling

**Status:** ACCEPTED  
**Created:** 2026-06-18T17:27:20.928035Z  
**Goal dependency:** G1  
**Owner:** Codex

## Plain-language question

Can the repository repeatedly archive HKO Daily Extract backing payloads for the
current target month, parse target-station daily maxima, and produce a
conservative first-observation ledger that distinguishes "first seen by our
archive" from true provider first publication?

## Why this could improve Tmax

It does not improve forecast skill directly. It addresses the G1 blocker from
EXP-0002: latest Daily Extract and CLMMAXT matched on May 2026, but
first-publication Daily Extract evidence was missing. A polling ledger is the
necessary infrastructure for proving or falsifying first-publication parity.

## What changed from prior work

EXP-0002 implemented HKO parsers, source contracts, and a latest-payload parity
slice. EXP-0003 adds the live/archive polling mechanism needed to accumulate
first-observation evidence. Polymarket backtesting, prices, books, trades,
liquidity, execution, and market replay remain deferred.

## Current conclusion

Accepted as polling infrastructure. G1 remains blocked pending provider
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
