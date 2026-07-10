# Decision Log

## D-001 — Focus station

**Decision:** Begin with the contract-authoritative Hong Kong Observatory target rather than a generalized all-city production model.

**Reason:** Deep station/source specialization reduces semantic and physical heterogeneity. Software remains modular for later expansion.

## D-002 — Historical label status

**Decision:** Treat `CLMMAXT station=HKO` as a candidate proxy until parity with first-published Daily Extract is proven.

**Reason:** Source-product/revision differences can invalidate otherwise excellent modelling.

## D-003 — Primary horizon

**Decision:** Do not hard-code T-24 yet. Evaluate H39, H27, H24N, and H15, then freeze one in G2.

**Reason:** “24 hours before settlement” is ambiguous; forecast information and market liquidity arrive on distinct schedules.

## D-004 — ML sequencing

**Decision:** ML blocked until G1–G7.

**Reason:** Understanding, provenance, baselines, and classical signal must precede high-capacity fitting.

## D-005 — Reanalysis

**Decision:** Reanalysis is retrospective by default.

**Reason:** It incorporates information unavailable at forecast cutoff.

## D-006 — Profit claims

**Decision:** No profitability claim without executable prices, fees, slippage, latency, and fills.

**Reason:** meteorological edge and tradeable edge are different.

## D-007 — Experiment immutability

**Decision:** Never overwrite an experiment.

**Reason:** preserves research history, prevents hindsight rewriting, and enables context recovery.
