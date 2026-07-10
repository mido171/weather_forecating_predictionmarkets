# HKG Tmax Residual ML Next Round Model Card

Primary cutoff: T-1 23:59 HKT.
Primary target: HKO Daily Extract Absolute Daily Max (deg. C).
Primary baseline: strict Info.gov local lead-1 official forecast max.
Primary next-round question: should residual correction be applied selectively instead of on every row?

## Decision

Outcome: `no_promote`.

Reason: `router edge too small: c2_vs_raw=0.027927927881359227, c2_vs_a7=-0.004264699170343111`.

Leakage audit status: `pass`.
No-harm audit status: `pass`.
Selected raw feature count: `64`.

## Selected Router Threshold

Threshold id: `router_grid_0027`.

- Expected-benefit threshold: `0.0`
- Apply-probability threshold: `0.52`
- Sign-probability threshold: `0.56`
- Positive cap: `0.75` C
- Negative cap: `0.5` C
- Hard absolute cap: `0.75` C

## Primary Score Rows

```csv
cutoff_profile,model_id,model_family,n_scored,mae,rmse,median_absolute_error,bias,p80_absolute_error,p90_absolute_error,p95_absolute_error,max_absolute_error,mean_prediction,mean_actual,mean_anchor_forecast,scope
tminus1_2359,A7_final_residual_ensemble,final_residual_ensemble,5629,0.8986654294414583,1.154087731949867,0.7312516838902781,-0.00048642663670142263,1.4512439685960263,1.9042251422766299,2.315997566243707,7.769879133313763,26.602462587397763,26.602949014034465,26.48001421211583,overall
tminus1_2359,C1_pruned_residual_ensemble,pruned_residual_ensemble,5629,0.9010521325648249,1.1581620523969156,0.7250446205216505,0.01169254578054068,1.4444218539026787,1.932528637291614,2.3420040623154748,8.04772391350125,26.614641559815006,26.602949014034465,26.48001421211583,overall
tminus1_2359,C3_tail_overlay_router,tail_overlay_router,5629,0.902893193428153,1.1609888523282839,0.7048252380142657,-0.024123447883001393,1.4417590421930488,1.9279779142744717,2.3090866017147813,8.1,26.578825566151465,26.602949014034465,26.48001421211583,overall
tminus1_2359,C2_selective_router,selective_router,5629,0.9029301286118014,1.161023902131915,0.7048252380142657,-0.024056210459342132,1.4417590421930488,1.9279779142744717,2.3090866017147813,8.1,26.57889280357512,26.602949014034465,26.48001421211583,overall
tminus1_2359,A0_raw_official,baseline,5629,0.9308580564931607,1.1957569920989344,0.7000000000000028,-0.1229348019186356,1.5,2.0,2.3999999999999986,8.6,26.48001421211583,26.602949014034465,26.48001421211583,overall
```

## Notes

Sealed confirmation rows were not used for threshold selection, model selection, feature selection, calibration, or hyperparameter tuning. Raw official-error slices and helped/worsened labels are evaluation-only columns and are not model features.
