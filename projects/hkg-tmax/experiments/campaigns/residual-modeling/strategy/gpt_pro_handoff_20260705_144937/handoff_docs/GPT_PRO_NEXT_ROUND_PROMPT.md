# GPT-Pro Deep-Dive Prompt: HKG Tmax Residual ML Strategy Next Round

You are the GPT-Pro orchestration and research director for the HKG Tmax daily maximum temperature prediction program.

You have been given a zip file containing the complete Codex implementation and experiment evidence for the HKG Tmax residual-ML strategy run completed on 2026-07-05. Your job is not to give a shallow summary. Your job is to deeply audit the evidence, explain why the result was small, and produce the full next-round experimental specification for Codex to implement.

## Non-Negotiable Instructions

Read the package before giving conclusions.

You must inspect, at minimum:

1. `handoff_docs/README_HANDOFF.md`
2. `source_docs/attached_spec/hkg_tmax_ml_strategy_codex_implementation_20260705.md`
3. All three files under `source_docs/live_trading/`
4. `results/final_model_card.md`
5. `results/summary.json`
6. `results/row_count_audit.json`
7. `results/leakage_audit.json`
8. `results/scoreboard.csv`
9. `results/scoreboard_by_split.csv`
10. `results/scoreboard_by_month.csv`
11. `results/scoreboard_by_regime.csv`
12. `results/ablation_scoreboard.csv`
13. `results/cutoff_sensitivity_scoreboard.csv`
14. `results/model_selection_log.json`
15. `results/ensemble_weights.json`
16. `results/feature_importance_lgbm.csv`
17. `results/feature_importance_catboost.csv`
18. `results/linear_coefficients.csv`
19. `results/feature_lineage.json`
20. `results/source_eligibility_audit.csv`
21. `results/prediction_rows.parquet` or `results/prediction_rows.csv`
22. The implementation files under `implementation/code/src/hkg_tmax/`
23. `implementation/scripts/run_hkg_tmax_residual_ml_strategy.py`
24. `implementation/configs/hkg_tmax/residual_ml_strategy.yaml`
25. `implementation/code/tests/test_hkg_tmax_residual_ml_strategy.py`
26. `evidence/git_status_short.txt`
27. `evidence/git_diff_stat.txt`
28. `evidence/scoped_git_diff.patch`
29. `run_logs/full.stdout.log`
30. `run_logs/full.stderr.log`

If any required file is missing from the package, say exactly which file is missing and how that limits your analysis.

## Core Facts From The Run

The implemented model target was residual correction against the latest eligible Info.gov local official forecast max:

```text
prediction_c = latest eligible official forecast max before cutoff + residual_hat
```

Primary cutoff:

```text
tminus1_2359
```

Primary target:

```text
HKO Daily Extract Absolute Daily Max (deg. C)
```

Primary result:

```text
A0 raw official MAE: 0.9308580564931607
A7 final residual ensemble MAE: 0.8986654294414583
MAE improvement: 0.03219262705170234 C
Decision: no_promote_cosmetic
```

The result is below the predeclared `0.035 C` meaningful-edge threshold.

The leakage audit passed:

```text
status: pass
total_violations: 0
```

The full run had:

```text
feature_rows: 38588
feature_count: 323
prediction_rows: 157670
target_rows: 9647
forecast_rows: 88277
hourly_rows: 252644
target_history_rows: 48577
```

## Your Required Analysis

Produce a rigorous research memo with these sections.

### 1. Evidence Audit

Confirm what was implemented and what evidence proves it.

You must explicitly verify:

- The target definition.
- The forecast anchor definition.
- The cutoff profiles.
- The strict Info.gov local forecast filter.
- The lag2 target-history floor.
- The sealed-confirmation handling.
- The no-raw-Daily-Extract-payload predictor rule.
- The no-post-cutoff forecast/hourly observation rule.
- The model families actually run.
- The result artifacts actually produced.

### 2. Why The Gain Was Small

Do a deep explanation of why the residual-ML stack only improved MAE by about `0.032 C`.

Do not stop at "official forecast is strong." Break it down:

- How strong the raw official anchor already was.
- How much of the gain came from bias correction.
- How much signal appeared in forecast revision features.
- How much signal appeared in HKO hourly features.
- Whether station-network gradients added incremental value.
- Whether text/warning/cyclone features added incremental value.
- Whether lag2+ target memory added incremental value.
- Whether the ensemble added incremental value beyond the best single model.
- Whether direct absolute prediction underperformed residual prediction and why.
- Whether sealed confirmation behavior matches rolling/holdout behavior.
- Whether improvements were robust or concentrated in a few slices.

### 3. Feature-Family Autopsy

Analyze feature importance together with ablations.

Do not treat feature importance as causal proof.

You must answer:

- Which individual features look repeatedly important?
- Which feature families actually moved MAE?
- Which feature families looked redundant after A3?
- Which features may be proxies for the official forecast itself?
- Which features might be overfit or unstable?
- Which features should be preserved in the next round?
- Which features should be challenged or dropped in a controlled ablation?

### 4. Error Slice Autopsy

Use `prediction_rows` plus scoreboards to identify where the final model helped and where it failed.

Analyze at least:

- Stage: rolling validation, presealed holdout, sealed confirmation.
- Month and season.
- Official max bins.
- Official range bins.
- Issue-hour buckets.
- Thunderstorm/rainstorm/warning regimes.
- High raw-official-error rows.
- Rows where final ensemble made error worse.
- Rows where final ensemble made error much better.
- Tail-error behavior at p80, p90, p95, max absolute error.

### 5. Cutoff Autopsy

Analyze T-1 18:00, 21:00, and 23:59.

Explain:

- Why T-1 23:59 won.
- Whether T-1 21:00 is viable.
- Why T-1 18:00 was weaker.
- Why T-1 15:00 produced no scored strict selected-anchor model.
- Whether the 15:00 issue is a real data availability limitation, a strict-filter artifact, or a selector implementation issue.
- What exact experiment should test earlier-cutoff viability next.

### 6. Model Autopsy

Analyze:

- A0 raw official.
- A1 grouped residual.
- A2 revision residual LGBM.
- A3 hourly-state residual LGBM.
- A4 station-network residual LGBM.
- A5 text-warning residual LGBM.
- A6 target-memory residual LGBM.
- A7 final residual ensemble.
- A8 direct absolute LGBM.
- B0 climatology/persistence.
- CatBoost residual role.
- Huber residual role and convergence warnings.

Explain whether the next experiment should keep the same model families, simplify them, or route into specialists.

### 7. Next-Round Direction Decision

Make a hard recommendation. Choose a small number of next experiments, not a vague brainstorm.

The next-round direction should be one of these, or a better alternative justified by evidence:

- Selective correction/abstention router.
- Tail-error specialist model.
- Residual sign classifier plus asymmetric cap.
- Monthly/regime-specific specialist correction.
- Feature-family pruning and robust stacking.
- Forecast-anchor provenance and earlier-cutoff repair.
- External timestamp-proven data acquisition.
- Online residual memory, but only if strictly lag-safe.

For each recommended direction, specify:

- Hypothesis.
- Why the current evidence supports it.
- Exact training data.
- Exact validation folds.
- Exact holdout/sealed policy.
- Exact features.
- Exact models.
- Exact metrics.
- Exact promotion gates.
- Exact no-harm gates.
- Exact artifacts to produce.
- Exact failure criteria.
- Exact implementation files Codex should add or modify.

### 8. Codex-Ready Implementation Spec

Produce a precise implementation plan for Codex for the next round.

The spec must be executable by Codex without guessing.

Include:

- File paths.
- Function/module responsibilities.
- Data contracts.
- CLI command.
- Config keys.
- Artifact list.
- Scoreboard schemas.
- Tests to add.
- Leakage audit requirements.
- Runtime constraints.
- Expected final output.

### 9. Do Not Overclaim

Explicitly state:

- The current run is not a trading-grade breakthrough.
- The current run is a clean leakage-safe calibration improvement.
- The next round must prove incremental edge under no-harm constraints.
- No sealed row may be used for selection.
- No post-cutoff source may be used.
- No raw Daily Extract payload row may become a predictor.

## Output Format Required

Return a long, structured research memo with:

1. Executive conclusion.
2. Evidence table.
3. Result diagnosis.
4. Feature-family diagnosis.
5. Error-slice diagnosis.
6. Model diagnosis.
7. Cutoff diagnosis.
8. Final next-round recommendation.
9. Codex-ready implementation spec.
10. Checklist for Codex verification.

Be direct. If the data says the next round should stop using broad generic ML and move to selective no-harm correction, say that. If the data says the next round must acquire better timestamp-proven independent data before another ML pass, say that. If the data says there is likely no exploitable edge in this public source set, say that and explain the evidence.

Do not produce vague ideas. Produce the next concrete experiment specification.

