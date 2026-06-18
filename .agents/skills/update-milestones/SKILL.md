---
name: update-milestones
description: Promote a verified experiment into MILESTONES.md with exact metrics and gate status. Use only after leakage and reproducibility PASS.
---

1. Verify `STATUS.yaml` says `ACCEPTED`.
2. Verify leakage and reproducibility gates are PASS.
3. Verify locked OOS metrics, uncertainty, sample size, and baseline are present.
4. Add one concise milestone row and one detailed finding section.
5. State absolute and relative deltas, confidence interval, regimes, limitations, and experiment ID.
6. Update champion only when the predeclared primary metric and all guardrails pass.
7. Never remove rejected findings or rewrite historical performance.
8. Regenerate with `python -m hkg_tmax milestones render`.
