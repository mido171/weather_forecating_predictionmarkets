# HKG Tmax Residual ML Strategy GPT-Pro Handoff

Generated for the GPT-Pro orchestration conversation after the residual-ML implementation and full experiment run on 2026-07-05.

## Executive Summary

This package contains the complete evidence bundle for the HKG Tmax residual-ML strategy implementation. The experiment implemented a calibrated residual ensemble around the latest eligible Info.gov `LOCAL WEATHER FORECAST` Tmax anchor. The model target was:

```text
prediction_c = latest eligible official forecast max before cutoff + residual_hat
```

The implementation enforced the source contracts from the live-trading documentation:

- Primary target: HKO Daily Extract `Absolute Daily Max (deg. C)`.
- Primary baseline: latest eligible Info.gov local forecast max before cutoff.
- Primary cutoff: T-1 23:59 HKT, with secondary cutoffs T-1 15:00, 18:00, and 21:00 HKT.
- Target-history predictors use a lag2 floor. No lag1 target history was used.
- Sealed 2024+ confirmation rows were evaluated after model selection and were not used for model selection.
- Raw Daily Extract payload rows were not used as predictors.
- No post-cutoff forecast or hourly observation was used.

The full run completed successfully. The leakage audit passed with zero violations.

## Primary Result

Primary cutoff: `tminus1_2359`.

| Model | MAE | RMSE | p90 absolute error | Bias | Rows |
|---|---:|---:|---:|---:|---:|
| A0 raw official | 0.930858 | 1.195757 | 2.000000 | -0.122935 | 5,629 |
| A7 final residual ensemble | 0.898665 | 1.154088 | 1.904225 | -0.000486 | 5,629 |

Primary MAE improvement:

```text
0.9308580564931607 - 0.8986654294414583 = 0.03219262705170234 C
```

The result is classified as:

```text
no_promote_cosmetic
```

Reason: the MAE improvement is below the predeclared `0.035 C` meaningful-edge threshold.

## What Was Implemented

The pipeline added four new package areas under `code/src/hkg_tmax`:

- `data`: target loading, strict Info.gov forecast anchor selection, hourly readings features, and lag2+ target history features.
- `features`: feature registry, lineage, station groups, text flags, and leakage checks.
- `modeling`: raw official baseline, grouped residual baseline, LightGBM residual model, CatBoost residual model, Huber residual diagnostic, direct LightGBM diagnostic, and constrained residual ensemble.
- `evaluation`: rolling folds, holdout/sealed scoring, metrics, scoreboards, artifact writing, model card, and diagnostics.

The main command is:

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_residual_ml_strategy.py --config configs\hkg_tmax\residual_ml_strategy.yaml --output-dir experiments\hkg_tmax_residual_ml_strategy\results
```

The focused test command is:

```powershell
.\.venv\Scripts\python.exe -m pytest code\tests\test_hkg_tmax_residual_ml_strategy.py
```

The final verification commands run by Codex:

```powershell
.\.venv\Scripts\python.exe -m compileall code\src\hkg_tmax scripts\run_hkg_tmax_residual_ml_strategy.py
.\.venv\Scripts\python.exe -m pytest code\tests\test_hkg_tmax_residual_ml_strategy.py
```

Both passed.

## Full Run Counts

From `experiments/hkg_tmax_residual_ml_strategy/results/summary.json`:

```text
target_rows: 9647
forecast_rows: 88277
hourly_rows: 252644
target_history_rows: 48577
feature_rows: 38588
feature_count: 323
prediction_rows: 157670
leakage_status: pass
```

From `row_count_audit.json`:

```text
target_rows_by_source:
  label_core: 8765
  sealed_confirmation: 882

joined_rows_by_cutoff:
  tminus1_1500: 9647
  tminus1_1800: 9647
  tminus1_2100: 9647
  tminus1_2359: 9647

anchor_rows_by_cutoff:
  tminus1_1800: 8335
  tminus1_2100: 8428
  tminus1_2359: 9644
```

T-1 15:00 produced no strict selected-anchor scored folds because the strict Info.gov local forecast source did not provide eligible selected anchors at that cutoff under the implemented filter.

## Cutoff Sensitivity

From `cutoff_sensitivity_scoreboard.csv`:

| Cutoff | Best model | Best MAE | Best RMSE | Best p90 absolute error | Rows |
|---|---|---:|---:|---:|---:|
| tminus1_2359 | A7 final residual ensemble | 0.898665 | 1.154088 | 1.904225 | 5,629 |
| tminus1_2100 | A7 final residual ensemble | 0.921281 | 1.177615 | 1.937429 | 5,088 |
| tminus1_1800 | A3 HKO hourly-state residual LGBM | 0.934988 | 1.192917 | 1.968950 | 5,050 |

## Model Ladder Interpretation

Primary cutoff overall ranking:

| Model | MAE | Interpretation |
|---|---:|---|
| A7 final residual ensemble | 0.898665 | Best overall. Calibration/ensemble gain over A6 is small but positive. |
| A6 target-memory residual LGBM | 0.900630 | Best single full-family LGBM. |
| A3 HKO hourly-state residual LGBM | 0.900754 | Most of the practical ML gain appears by the hourly-state stage. |
| A4 network-gradients residual LGBM | 0.901236 | Station-network gradients did not add much after A3. |
| A5 text-warning residual LGBM | 0.901910 | Text/warning features did not add much after A3/A4. |
| A2 revision residual LGBM | 0.914367 | Forecast revision alone helped, but not enough. |
| A1 grouped residual | 0.924794 | Small grouped calibration improvement. |
| A0 raw official | 0.930858 | Strong baseline. |
| A8 direct LGBM absolute | 0.949068 | Direct absolute prediction underperformed residual modeling. |
| B0 climatology persistence | 1.905100 | Not competitive. |

The heavy feature/model stack did not unlock a large edge because the task is residual correction against an already strong official forecast. The model mainly removed official-forecast bias and made modest residual corrections. The residual signal was positive but weak.

## Feature Strength Snapshot

The feature-importance artifacts are:

- `results/feature_importance_lgbm.csv`
- `results/feature_importance_catboost.csv`
- `results/linear_coefficients.csv`
- `results/feature_lineage.json`

The strongest mean absolute LGBM features observed during the explanatory check were:

```text
official_max_c
network_mean_trend_6h_c
network_spread_mean_6h_c
official_midpoint_c
trend_years_since_2000
official_max_minus_doy_clim30_c
rev_path_min_c
hko_temp_mean_24h_minus_doy_clim_c
target_lag2_minus_doy30_clim_c
official_max_minus_month_clim10_c
nt_heat_ceiling_index_c
urban_core_mean_minus_coastal_marine_mean_c
target_roll30_anomaly_lag2_c
target_lag60_tmax_c
hko_temp_trend_12h_c
```

This ranking is a model-importance signal, not causal proof. GPT-Pro should combine importance with ablation results before assigning value to any feature family.

## Result Interpretation

The final residual ensemble improved MAE by only `0.032 C`, but it also improved RMSE and tail error:

```text
RMSE delta: -0.041669260149067355
p90 absolute error delta: -0.09577485772337013
raw bias: -0.1229348019186356
final bias: -0.00048642663670142263
```

The model corrected bias almost completely. The model did not discover a large new weather edge.

The learned residual correction was intentionally small:

```text
median absolute correction: about 0.17 C
p90 absolute correction: about 0.43 C
correlation between true residual and final correction: about 0.248
```

This means the model had some signal, but not enough reliable signal to justify large corrections.

## Important Caveats

- CatBoost was installed and used. Dependency files were updated with `catboost>=1.2,<2`.
- The runner used 120 tree iterations for practical runtime. The model family and validation design match the spec, but this is not an exhaustive hyperparameter sweep.
- Huber emitted sklearn convergence warnings. Huber was retained as a diagnostic model and ensemble candidate; the full run completed and wrote coefficients/predictions.
- The experiment is research only. It is not a production trading release.
- The repository had substantial unrelated dirty state before and during this work. This handoff package is built from an explicit allowlist to avoid unrelated changes.

## Files To Read First

1. `handoff_docs/GPT_PRO_NEXT_ROUND_PROMPT.md`
2. `handoff_docs/README_HANDOFF.md`
3. `source_docs/attached_spec/hkg_tmax_ml_strategy_codex_implementation_20260705.md`
4. `source_docs/live_trading/HKG_TMAX_INFO_GOV_LIVE_FORECAST_SOURCE_CONTEXT_20260704.md`
5. `results/final_model_card.md`
6. `results/summary.json`
7. `results/scoreboard.csv`
8. `results/ablation_scoreboard.csv`
9. `results/feature_importance_lgbm.csv`
10. `results/leakage_audit.json`

## What GPT-Pro Should Decide Next

GPT-Pro should not ask Codex to repeat this exact experiment with only broader model classes. The evidence suggests the next gain must come from sharper residual slices, stronger independent pre-cutoff information, or a no-harm selective correction policy rather than another generic all-row residual model.

The next round should produce a precise implementation spec for one or more of:

- Conditional abstention/router where correction is applied only when expected residual edge is high.
- Tail-error specialists for the top raw-official error regimes.
- Residual sign classifier with asymmetric correction caps.
- Feature-family interaction mining focused on A3-to-A7 failure cases.
- More faithful lead-time availability audit for earlier cutoffs, especially T-1 15:00.
- Additional independent source acquisition if and only if it is timestamp-proven before cutoff.

