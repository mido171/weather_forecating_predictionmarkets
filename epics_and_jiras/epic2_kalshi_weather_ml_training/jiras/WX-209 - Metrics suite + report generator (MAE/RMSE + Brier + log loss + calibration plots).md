# WX-209 — Metrics suite + report generator (MAE/RMSE + Brier + log loss + calibration plots)

## Objective
Produce a comprehensive metrics report for trading relevance.

## Required metrics (per station and overall)
Point:
- MAE
- RMSE

Probabilistic:
- log loss over integer temperature classes
- Brier score per bin-event

Calibration:
- calibration curves (reliability diagrams) for top bins and/or all bins

## Implementation requirements
- Save:
  - metrics.json
  - report.md
  - plots/*.png
- Report must include baselines from WX-204 for comparison.

## Acceptance Criteria
- [ ] Report is generated for a 1-year test period.
- [ ] JSON contains per-station and aggregate metrics.
- [ ] Calibration plots are present.
- [ ] Metrics code is unit-tested.

