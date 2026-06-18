---
name: audit-leakage
description: Audit an HKG Tmax dataset, feature set, or experiment for point-in-time leakage and split contamination. Use before accepting any claimed accuracy gain.
---

1. Identify forecast target date and exact cutoff.
2. For every input column trace:
   `source → raw payload → source timestamp → available_at → transformation → feature row`.
3. Verify `available_at <= cutoff` row by row.
4. Look for:
   - future model cycles;
   - target-day finalized observations;
   - latest-only forecasts backfilled historically;
   - revised climate values presented as first publication;
   - ERA5/final best tracks without lag;
   - global scaling, imputation, selection, or calibration;
   - random folds;
   - target encoding across time;
   - duplicated event dates across folds;
   - labels embedded in filenames, missingness, or source choice.
5. Run negative controls and deliberately inject one invalid timestamp to ensure validators fail.
6. Write findings with severity, evidence, impact, and required remediation.
7. PASS only if all critical/high findings are closed.
