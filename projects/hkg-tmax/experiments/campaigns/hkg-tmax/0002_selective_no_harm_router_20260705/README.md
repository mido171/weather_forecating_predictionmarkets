# 0002 Selective no-harm router

Status: `complete_no_promote`.

## Hypothesis

A compact feature set plus selective abstention and a tail specialist might
apply the A7 residual correction only when expected benefit is strong, reducing
harm relative to all-row correction.

## Contract and method

- Historical primary cutoff: T-1 23:59 HKT.
- Sealed rows were excluded from feature, model, and threshold selection.
- Raw official-error slices and helped/worsened labels were evaluation-only.
- The raw policy selected 64 features (maximum 90).
- C1 was a pruned residual ensemble, C2 a selective router, and C3 a tail
  overlay.

## Primary result

| Model | MAE | Interpretation |
|---|---:|---|
| A0 raw official | 0.930858 | baseline |
| A7 prior residual ensemble | 0.898665 | reference winner |
| C1 pruned ensemble | 0.901052 | worse than A7 |
| C3 tail overlay | 0.902893 | worse than A7 |
| C2 selective router | 0.902930 | worse than A7 |

Leakage and no-harm audits passed, and 34/34 required artifacts were produced.
The tail overlay had effectively zero final apply rate.

The early-anchor audit found no viable 15:00 history; 16:30 coverage was
79.04%, below the 80% modeling gate. 18:00, 21:00, and 23:59 cleared that
historical coverage check.

## Decision

No promotion: C2 improved over raw official but lost to A7 by 0.004265 C MAE.
This remains point-forecast research only.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_residual_ml_next_round.py --config config\experiments\hkg_tmax\residual_ml_next_round.yaml --output-dir experiments\campaigns\hkg-tmax\0002_selective_no_harm_router_20260705\results --no-compat-copy
```

Set `HKG_TMAX_DATABASE_URL` instead of placing a DSN in the command.

## Evidence map

- `results/summary.json` and `scoreboard.csv`.
- `results/no_harm_audit.json` and `leakage_audit.json`.
- `results/router_threshold_selection.csv` and router/tail diagnostics.
- `results/anchor_provenance_summary.json`.
- `inputs/gpt_pro_next_round_memo_20260705.txt`: frozen source memo.
