# HKG Tmax Milestones

**Last generated:** 2026-06-18T16:10:09Z  
**Primary horizon:** not yet selected  
**Canonical target parity:** not yet proven  
**Production status:** disabled

## Current champion

No champion model is eligible yet. Any forecast generated before G1–G5 pass is exploratory only.

## Accepted milestone findings

No accepted findings yet.

## Baseline scoreboard

| Model | Horizon | Sample | MAE °C | RMSE °C | CRPS | Multiclass log loss | Calibration | Status |
|---|---|---:|---:|---:|---:|---:|---|---|
| Seasonal climatology | TBD | — | — | — | — | — | — | Not run |
| Persistence/anomaly | TBD | — | — | — | — | — | — | Not run |
| HKO official forecast | TBD | — | — | — | — | — | — | Not archived |
| Raw NWP consensus | TBD | — | — | — | — | — | — | Not archived |
| Bias-corrected NWP | TBD | — | — | — | — | — | — | Not run |

## Required gates

- [x] G0 environment and archival smoke test — EXP-0001 accepted
- [ ] G1 contract target and Daily Extract parity
- [ ] G2 primary horizon selected and frozen
- [ ] G3 source inventory and historical acquisition
- [ ] G4 data quality and station-history audit
- [ ] G5 strong baselines
- [ ] G6 classical mechanism experiments
- [ ] G7 expert probabilistic stack
- [ ] G8 ML eligibility gate
- [ ] G9 executable market evaluation
- [ ] G10 production/shadow gate

## Recently rejected or null hypotheses

None yet. Rejected work must remain visible to prevent rediscovery.

## Live risks and blockers

1. `CLMMAXT station=HKO` has not yet been proven identical to the first-published Daily Extract settlement value.
2. Authentic archived forecast vintages are incomplete until acquisition is executed.
3. The preferred “24 hours early” cutoff is not yet aligned to actual market opening and information arrival.
4. Historical Polymarket book depth may be unavailable before our own archive starts.
