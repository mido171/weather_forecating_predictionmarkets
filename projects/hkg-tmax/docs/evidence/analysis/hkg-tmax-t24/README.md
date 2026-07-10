# HKG Tmax T-24 Analysis Workspace

This workspace records leakage-safe analysis for the HKO/HKG daily Tmax
forecasting objective. It is separate from historical experiment folders because
the current worktree already contains unrelated experiment-folder deletions.

## Structure

- `phase_a_data_readiness/` - parsing, source contracts, coverage and QC.
- `phase_b_eda/` - exploratory analysis reports and finding folders.
- `phase_c_evaluation_design/` - frozen split and point-in-time evaluation notes.
- `baselines/` - baseline design and future baseline outputs.
- `findings/` - dedicated folders for durable hypotheses or discovered signals.

## Current Gate

Phase A must remain the active gate until source-native bronze/silver tables,
timestamp contracts, coverage matrices and leakage tests are in place. Any EDA
claim must state the sample, period, point-in-time status and leakage review.

