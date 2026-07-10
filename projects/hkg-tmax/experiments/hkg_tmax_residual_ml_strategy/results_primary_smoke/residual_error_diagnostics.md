# Residual Error Diagnostics

## Promotion Result

`{
  "outcome": "no_promote_cosmetic",
  "primary_cutoff": "tminus1_2359",
  "raw_official_mae": 0.9308580564931607,
  "final_mae": 0.8986654294414583,
  "mae_improvement": 0.03219262705170234,
  "rmse_delta": -0.041669260149067355,
  "p90_delta": -0.09577485772337013,
  "n_scored": 5629
}`

## Primary Comparison

```csv
model_id,stage,rows
A0_raw_official,presealed_holdout,730
A0_raw_official,rolling_validation,4017
A0_raw_official,sealed_confirmation,882
A7_final_residual_ensemble,presealed_holdout,730
A7_final_residual_ensemble,rolling_validation,4017
A7_final_residual_ensemble,sealed_confirmation,882
```

The raw official top-error decile is used only for post-hoc diagnostics and is not included as a feature.
