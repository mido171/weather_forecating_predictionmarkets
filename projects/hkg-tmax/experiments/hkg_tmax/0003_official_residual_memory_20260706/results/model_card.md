# HKG Tmax Official Residual Memory Model Card

Experiment: `hkg_tmax_0003_official_residual_memory_20260706`.

Primary question: whether lag-safe official forecast residual memory can improve the existing A7 residual-ML research candidate enough to justify promotion over the raw official forecast.

Scope: point forecasting only. No probability buckets, Polymarket prices, EV, sizing, PnL, or trading features were used.

Primary target: HKO Daily Extract Absolute Daily Max in deg. C.

Primary anchor: latest strict Info.gov local forecast maximum before `T-1 23:59 HKT`.

Residual definition: `actual_tmax_c - selected_official_forecast_max_c`.

Residual-memory rule: for target date `T`, the newest allowed residual source date is `T-2`. Lag-1 residuals are disabled.

## Decision

Decision: `no_promote`.

Reason: `one or more predeclared gates failed`.

Leakage audit: `pass`.

Row-identity gate: `pass`.

Publication-safety audit: `pass`.

## Primary Score Rows

```csv
cutoff_profile,model_id,model_family,n_scored,mae,rmse,median_absolute_error,bias,p80_absolute_error,p90_absolute_error,p95_absolute_error,max_absolute_error,mean_prediction,mean_actual,mean_anchor_forecast,scope
tminus1_2359,D4_residual_memory_constrained_stack,residual_memory_constrained_stack,5629,0.8984787669449551,1.1535632828444047,0.7328045583448493,8.174675406389646e-05,1.458775705717812,1.904830694847434,2.3113254874138125,7.765872394008893,26.603030760788528,26.602949014034465,26.48001421211583,overall
tminus1_2359,D5_conservative_A7_plus_memory_blend,conservative_a7_plus_memory_blend,5629,0.8985938845238038,1.154004058945679,0.7329720294638342,-0.0006288534334980485,1.458775705717812,1.9058438177155756,2.3113254874138125,7.85,26.602320160600968,26.602949014034465,26.48001421211583,overall
tminus1_2359,A7_final_residual_ensemble,final_residual_ensemble,5629,0.8986654294414583,1.154087731949867,0.7312516838902781,-0.00048642663670142263,1.4512439685960263,1.9042251422766299,2.315997566243707,7.769879133313763,26.602462587397763,26.602949014034465,26.48001421211583,overall
tminus1_2359,D0_A7_reproduction,a7_reproduction_reference,5629,0.8986654294414583,1.154087731949867,0.7312516838902781,-0.00048642663670142263,1.4512439685960263,1.9042251422766299,2.315997566243707,7.769879133313763,26.602462587397763,26.602949014034465,26.48001421211583,overall
tminus1_2359,D3_pruned_full_plus_residual_memory_lgbm,pruned_full_plus_residual_memory_lgbm,5629,0.9056522671181472,1.1630900136545959,0.7382082022636602,0.008524726663575932,1.456140390483914,1.9086002209020008,2.3167541547878727,8.309032433536702,26.611473740698038,26.602949014034465,26.48001421211583,overall
tminus1_2359,D2_A3_plus_residual_memory_lgbm,a3_plus_residual_memory_lgbm,5629,0.9080003192028057,1.1677914196214745,0.7358954938502578,0.009603955239251139,1.4651631489211696,1.9349326849448507,2.3365587421678726,8.33355436070451,26.61255296927371,26.602949014034465,26.48001421211583,overall
tminus1_2359,D1_official_residual_memory_shrinkage,residual_memory_shrinkage,5629,0.9302363980312539,1.193903457402032,0.7441724784827244,-0.09816563234706824,1.490012221767254,1.9753432702711335,2.396692168697477,8.676413671335707,26.504783381687393,26.602949014034465,26.48001421211583,overall
tminus1_2359,A0_raw_official,baseline,5629,0.9308580564931607,1.1957569920989344,0.7000000000000028,-0.1229348019186356,1.5,2.0,2.3999999999999986,8.6,26.48001421211583,26.602949014034465,26.48001421211583,overall
```

## Interpretation

`D5_conservative_A7_plus_memory_blend` is the only promotable candidate. It can promote only if it clears the predeclared development, presealed, no-harm, row-identity, leakage, and sealed report-only reversal gates. If the gate result is `no_promote`, deployment remains raw official forecast while A7 remains a research reference.
