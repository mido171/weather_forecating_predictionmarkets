# HKG Tmax Probability Distribution Methods V2

This experiment tests whether true EMOS-style or other distribution engines beat the current probability champion, `B4_hierarchical_residual_pmf`.

Scope is weather probability only. The benchmark uses no market prices, no EV, no order books, no Kelly sizing, no PnL, no market-implied blending, and no trade recommendations.

## Implementation

- Config: `configs/hkg_tmax/probability_distribution_methods_v2.yaml`
- Runner: `scripts/run_hkg_tmax_probability_distribution_methods_v2.py`
- New V2 methods: `code/src/hkg_tmax_probability/distribution_methods_v2.py`
- V2 champion gates: `code/src/hkg_tmax_probability/leaderboard_v2.py`
- Tests: `code/tests/test_hkg_tmax_probability_distribution_methods_v2.py`

## Data Surface

The experiment reuses the V1 modeling table build from PostgreSQL `hkg_tmax_research`.

- Forecast rows: strict HKO Info.gov local forecast rows.
- Primary cutoff: `T-1 23:59 HKT`.
- Cutoff sensitivity: `T-1 18:00 HKT`, `T-1 21:00 HKT`, `T-1 23:59 HKT`.
- Target labels: canonical HKO Daily Extract one-decimal HKG daily Tmax, with sealed confirmation rows used only by the sealed confirmation split.
- Bucket contract: `<=24.9`, `25.0..25.9`, ..., `33.0..33.9`, `>=34.0`; `31.9` remains bucket `31`, `32.0` enters bucket `32`.

## Result

Final artifacts are under `results/`.

Primary files to read first:

1. `results/supreme_method_summary.md`
2. `results/scoreboard.csv`
3. `results/final_probability_model_card.md`
4. `results/leakage_audit.json`
5. `results/row_identity_gate.json`

Outcome: `B4_hierarchical_residual_pmf` remains the supreme method because no challenger clears the predeclared promotion gates.
