---
name: run-experiment
description: Create, predeclare, execute, document, and close one immutable HKG Tmax research experiment. Use for every new predictive or validation hypothesis; do not use for untracked ad hoc analysis.
---

1. Read `AGENTS.md`, `FIRST_GOALS.md`, `MILESTONES.md`, `EXPERIMENT_INDEX.md`, and related prior experiments.
2. Reserve an ID:
   ```bash
   python -m hkg_tmax experiments create --title "<title>"
   ```
3. Before viewing holdout results, complete:
   - `HYPOTHESIS.md`;
   - `PROTOCOL.md`;
   - `ASOF_CONTRACT.md`;
   - `DATA_MANIFEST.yaml`;
   - `RUN_CONFIG.yaml`.
4. State:
   - mechanism;
   - expected direction and regimes;
   - baseline;
   - sample/split;
   - primary and secondary metrics;
   - multiplicity family;
   - acceptance and falsification criteria.
5. Run `make validate` and save output in `logs/`.
6. Execute deterministic code and save row-level predictions.
7. Populate all metrics and diagnostics, including negative and null findings.
8. Request independent leakage and reproducibility review.
9. Complete `CONCLUSION.md` and `STATUS.yaml`.
10. Run:
    ```bash
    python -m hkg_tmax experiments index
    python -m hkg_tmax milestones render
    ```
11. Never overwrite a prior experiment. A changed hypothesis, data version, split, feature, or model creates a new ID.
