# Results

Experiment: `hkg_tmax_0003_official_residual_memory_20260706`

Scope: point forecast only. No probability, market, EV, sizing, PnL, or trading data was used.

Primary cutoff: `T-1 23:59 HKT`.

Primary result: `D5_conservative_A7_plus_memory_blend` did not promote.

## Primary Leaderboard

| Rank | Model | MAE | RMSE | p90 abs error | Rows |
|---:|---|---:|---:|---:|---:|
| 1 | `D4_residual_memory_constrained_stack` | 0.898479 | 1.153563 | 1.904831 | 5629 |
| 2 | `D5_conservative_A7_plus_memory_blend` | 0.898594 | 1.154004 | 1.905844 | 5629 |
| 3 | `A7_final_residual_ensemble` | 0.898665 | 1.154088 | 1.904225 | 5629 |
| 4 | `D0_A7_reproduction` | 0.898665 | 1.154088 | 1.904225 | 5629 |
| 5 | `A6_target_memory_residual_lgbm` | 0.900630 | 1.158318 | 1.906340 | 5629 |
| 6 | `D3_pruned_full_plus_residual_memory_lgbm` | 0.905652 | 1.163090 | 1.908600 | 5629 |
| 7 | `D2_A3_plus_residual_memory_lgbm` | 0.908000 | 1.167791 | 1.934933 | 5629 |
| 8 | `D1_official_residual_memory_shrinkage` | 0.930236 | 1.193903 | 1.975343 | 5629 |
| 9 | `A0_raw_official` | 0.930858 | 1.195757 | 2.000000 | 5629 |

## Gate Outcome

- Leakage audit: pass.
- Residual-memory publication safety audit: pass.
- Row identity gate: pass.
- Slice no-harm gate: pass.
- Development gain versus A7: fail, D5 was worse by 0.000075 C on the development frame.
- Development gain versus raw official: fail, D5 gained 0.032203 C, below the 0.045 C gate.
- Presealed gain versus A7: fail, D5 was worse by 0.000867 C.
- Presealed gain versus raw official: fail, D5 gained 0.029142 C, below the 0.040 C gate.
- Presealed p90 no-worse versus A7: fail by 0.000986 C above the allowed guardrail.
- Sealed report-only reversal check: pass, D5 was 0.000862 C better than A7 on sealed confirmation.

## Key Files

- `results/scoreboard.csv`
- `results/scoreboard_by_split.csv`
- `results/scoreboard_by_residual_memory_bin.csv`
- `results/model_card.md`
- `results/summary.json`
- `results/model_selection_log.json`
- `results/leakage_audit.json`
- `results/residual_memory_publication_safety_audit.json`
- `results/row_identity_gate.json`
- `results/prediction_rows.parquet`

