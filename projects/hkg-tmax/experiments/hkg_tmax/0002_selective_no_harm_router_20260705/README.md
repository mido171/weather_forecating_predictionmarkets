# 0002 Selective No-Harm Router

Status: in progress.

This experiment implements the GPT-Pro next-round recommendation:

1. Prune the raw feature set to a compact policy with a maximum of 90 raw features.
2. Reproduce the current A0-A8 ladder for comparison.
3. Score C1 pruned residual ensemble.
4. Score C2 selective correction router.
5. Score C3 tail specialist overlay.
6. Audit early forecast-anchor provenance for 15:00, 16:30, 18:00, 21:00, and 23:59 HKT.
7. Emit leakage, no-harm, row-count, feature-lineage, prediction-row, and model-card artifacts.

Canonical code and config:

- Config: `configs/hkg_tmax/residual_ml_next_round.yaml`
- Runner: `scripts/run_hkg_tmax_residual_ml_next_round.py`
- Feature policy: `code/src/hkg_tmax/features/pruned_feature_policy.py`
- Router: `code/src/hkg_tmax/modeling/selective_router.py`
- Tail overlay: `code/src/hkg_tmax/modeling/tail_specialist.py`
- No-harm reports: `code/src/hkg_tmax/evaluation/no_harm_reporting.py`
- Anchor audit: `code/src/hkg_tmax/data/anchor_provenance_audit.py`

Primary output folder:

`experiments/hkg_tmax/0002_selective_no_harm_router_20260705/results/`

Compatibility output folder requested by the GPT-Pro memo:

`experiments/hkg_tmax_residual_ml_next_round/results/`
