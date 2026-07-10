# HKG Tmax Probability Distribution Methods V2 Conclusion

`B4_hierarchical_residual_pmf` remains the supreme method after V2.

Raw RPS ranking favored `B5_kernel_analog_pmf`, but it failed the promotion gates because the fold 1-4 and presealed gains were below the required thresholds and NLL was worse than B4 by more than the configured allowance.

The new true EMOS methods did not beat B4:

- `E2_student_t_emos` ranked above `E1_normal_emos`, but still had worse overall RPS and failed the NLL gate.
- `E1_normal_emos` was materially worse than B4 by RPS and NLL.
- `E3_two_piece_normal_emos` was worse on RPS, NLL, and Brier.

The other challengers also failed promotion:

- `T1_time_decay_b4` was close to B4 but did not clear the fold 1-4 or presealed gain thresholds.
- `H1_b4_challenger_linear_pool` improved raw RPS slightly but not enough to clear promotion.
- `G1_gamlss_tree_location_scale`, `Q1_quantile_cdf_gb`, and `Q2_threshold_cdf_gb` underperformed B4 materially.

Audits passed:

- Leakage audit: pass, zero violations.
- Row identity gate: pass, zero violations.
- Live probability output no-trading audit: pass.

The detailed evidence is in `results/supreme_method_summary.md` and `results/scoreboard.csv`.
