# Rejection Protocol

Reject before scoring when a deployable experiment cannot satisfy the immutable contract. Create the normal folder and write `REJECTION.md` with:

- rejected hypothesis;
- exact failed gate and rejection code;
- evidence inspected;
- affected source, feature, rows, and dates;
- why lag, proxy, or alternate timestamp does or does not solve it;
- whether a diagnostic-only branch remains useful;
- exact artifact or proof that would unlock promotion;
- whether the Director should pursue a proxy-conversion lane.

Common codes include `CUTOFF_NOT_PRECISELY_DEFINED`, `NO_AVAILABLE_AT_PROOF`, `POST_CUTOFF_AGGREGATION`, `TARGET_T_INCLUDED`, `CONFIRMATION_CONTAMINATION`, `FRAME_NOT_REPRODUCIBLE`, `BASELINE_NOT_COMPARABLE`, `INSUFFICIENT_SUPPORT`, `DUPLICATE_EXPERIMENT_WITHOUT_NOVELTY`, `DATA_UNIT_OR_TIMEZONE_UNRESOLVED`, and `SPECIFICATION_UNDERDETERMINED`.

Rejection is valuable evidence and must enter the research registry.
