---
name: run-experiment
description: Create, predeclare, execute, document, and close one immutable HKG Tmax research experiment. Use for every new predictive or validation hypothesis; do not use for untracked ad hoc analysis.
---

1. Read `../../AGENTS.md`, `AGENTS.md`, `START_HERE.md`, `docs/status/CURRENT_STATE.md`, `EXPERIMENT_INDEX.md`, and related prior experiment conclusions.
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
5. Run `make validate` and save output under `${HKG_TMAX_RUN_ROOT}/logs/<experiment-id>/`.
6. Execute deterministic code with one worker by default and save row-level predictions under the external run root.
7. Populate all metrics and diagnostics, including negative and null findings.
8. Request independent leakage and reproducibility review.
9. Complete `CONCLUSION.md` and `STATUS.yaml`.
10. Run:
    ```bash
    python -m hkg_tmax experiments index
    python -m hkg_tmax milestones render
    ```
11. Never overwrite a prior experiment. A changed hypothesis, data version, split, feature, or model creates a new ID.
