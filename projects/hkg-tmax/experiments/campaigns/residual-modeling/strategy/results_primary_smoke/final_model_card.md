# HKG Tmax Residual ML Strategy Model Card

Primary benchmark cutoff: T-1 23:59 HKT.
Primary baseline: latest eligible Info.gov LOCAL WEATHER FORECAST max before cutoff.
Primary target: HKO Daily Extract Absolute Daily Max (deg. C).
Primary model target: residual versus official forecast max.
Primary metric: MAE, with RMSE and p90 absolute error as guardrails.
Sealed confirmation rows were not used for model selection.
No post-cutoff forecast or hourly observation was used.
Raw Daily Extract payload rows were not used as predictors.

## Decision

Outcome: `no_promote_cosmetic`.

Primary MAE improvement versus raw official at T-1 23:59 HKT: `0.03219262705170234` C.

If the MAE improvement is below `0.035 C`, this run is classified as `no_promote_cosmetic` and does not claim a meaningful ML edge.

## Implementation Summary

- Feature count in schema: `323`.
- Leakage audit status: `pass`.
- CatBoost status: `fit`.
- Sealed mode reported here: `sealed_blind_mode`; online sealed lag replay was not enabled because row-level Daily Extract publication availability for lagged sealed labels was not proven in the supplied docs.

## Primary Score Rows

```csv
cutoff_profile,model_id,model_family,n_scored,mae,rmse,median_absolute_error,bias,p80_absolute_error,p90_absolute_error,p95_absolute_error,max_absolute_error,mean_prediction,mean_actual,mean_anchor_forecast,scope
tminus1_2359,A7_final_residual_ensemble,final_residual_ensemble,5629,0.8986654294414583,1.154087731949867,0.7312516838902781,-0.00048642663670142263,1.4512439685960263,1.9042251422766299,2.315997566243707,7.769879133313763,26.602462587397763,26.602949014034465,26.48001421211583,overall
tminus1_2359,A0_raw_official,baseline,5629,0.9308580564931607,1.1957569920989344,0.7000000000000028,-0.1229348019186356,1.5,2.0,2.3999999999999986,8.6,26.48001421211583,26.602949014034465,26.48001421211583,overall
```

## Notes

The final residual ensemble includes an explicit zero-correction option and a validation-fitted shrinkage scalar. If no non-zero correction passes the no-harm guardrails, the ensemble shrinks back to the raw official forecast.
