# HKG Tmax Probability Distribution Engine V1 Results And Implementation Handoff

Date written: 2026-07-05  
Repository root: `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex`  
Experiment: `hkg_tmax_probability_buckets_v1`  
Primary result folder: `experiments/hkg_tmax_probability_buckets_v1/results/`

## Executive Summary

This document records the first full implementation and benchmark of the HKG Tmax probability distribution engine. The engine takes the official HKO Info.gov local forecast maximum/minimum point forecast, selects the latest eligible forecast available before a fixed cutoff, and converts that point forecast into an 11-bucket probability distribution for the HKG one-decimal daily Tmax target.

The current champion is:

```text
B4_hierarchical_residual_pmf
```

This champion is not a point forecast model. It is a probability distribution model. It uses the official forecast max as the distribution center and then applies a historical residual probability mass function conditioned by target month and official forecast level. In plain terms: if HKO says the max should be `32C`, the engine does not simply output `32C`; it asks, historically, when HKO forecasted about `32C` in this month, what one-decimal settled Tmax residuals actually occurred, and then converts that empirical residual distribution into bucket probabilities.

The primary leaderboard was evaluated on `4,747` rows across Fold 1-4 plus the presealed 2022-2023 holdout, using the primary cutoff `T-1 23:59 HKT`. The sealed 2024-2026 confirmation split was run after the development/presealed benchmark and was not used for tuning or champion promotion.

Important final result:

```text
Champion by configured acceptance rule: B4_hierarchical_residual_pmf
Primary normalized RPS: 0.041524
NLL: 1.037181
Brier: 0.045921
ECE: 0.019859
Leakage audit: pass, 0 violations
Row identity gate: pass, 0 violations
Label-publication audit: ok, 0 bucket changes
Live inference no-trading audit: pass
```

Two methods superficially looked better by raw RPS but were not promoted:

1. `B5_kernel_analog_pmf` had the lowest RPS, `0.041287`, but failed the NLL no-worse gate versus B4. It was sharper and sometimes more accurate by RPS, but less robust under log-loss.
2. `S1_conservative_simplex_stack` passed NLL/Brier gates and had RPS `0.041472`, but its relative RPS gain versus B4 was only `0.123%`, below the configured promotion threshold. The acceptance gate required a materially larger gain to justify replacing the simpler B4 engine.

The final practical conclusion is:

```text
Use B4_hierarchical_residual_pmf as the current reference probability distribution engine.
Do not promote B5 or S1 yet.
Do not treat MOS/Student-t MOS as superior from this run.
Do not claim a true EMOS benchmark was completed; only MOS-style Normal and Student-t residual distribution models were run.
```

## Reader Orientation And Document Map

This document is meant for future Codex/GPT-Pro orchestration, strategy implementation review, and live-trading preparation work. It is intentionally detailed so a future maintainer can understand what was tested, what won, what did not win, and why.

Read in this order:

1. `Scope Boundaries` to avoid confusing this with trading or EV logic.
2. `Probability Bucket Contract` for the exact target buckets and rounding rules.
3. `Data Sources And Row Construction` for what data was used.
4. `Validation Design` for exact train/test dates and cutoffs.
5. `Methods Benchmarked` for what was compared, including the MOS/EMOS clarification.
6. `Primary Leaderboard` for the final result.
7. `Champion Mechanics` for how B4 actually creates a probability distribution.
8. `Metric Definitions` for how to read the numbers.
9. `Audits And Acceptance Gates` for leakage, label publication, row identity, and no-trading evidence.
10. `Artifacts And Code Map` for where the implementation and generated files live.
11. `Operational Runbook` for how to reproduce or inspect the engine.
12. `Known Limitations And Next Work` for what remains unresolved.

## Scope Boundaries

### Included

This implementation and benchmark included:

- Weather-probability-only distribution generation.
- Official HKO Info.gov local forecast selection.
- Canonical HKO Daily Extract one-decimal target labels.
- Sealed confirmation labels for 2024-2026 after model selection.
- Decimal-safe bucket assignment.
- Multiple probability distribution approaches:
  - climatology baseline,
  - residual PMF baselines,
  - hierarchical residual PMF,
  - kernel analog PMF,
  - diagnostic month climatology,
  - Normal MOS,
  - Student-t MOS,
  - multinomial logistic classifier,
  - ordinal CDF logistic classifier,
  - B4 calibration variants,
  - conservative simplex stack.
- Proper scoring:
  - normalized RPS,
  - NLL,
  - multiclass Brier,
  - CRPS proxy,
  - ECE,
  - MCE,
  - entropy.
- Leakage audit.
- Row-identity gate.
- Label-publication audit.
- Live inference example with a no-trading-field audit.
- Reproducibility manifest.

### Excluded

This implementation deliberately excluded:

- Polymarket prices.
- Expected value.
- Order books.
- Bid/ask spreads.
- Kelly sizing.
- PnL.
- Market-implied blending.
- Trade recommendations.
- Any decision rule that says to buy or sell a bucket.

The result is a probability distribution engine only. Trading is a later stage.

### Important Clarification About The Official Forecast

The official HKO forecast is not a probability distribution. It is a point/range forecast. The probability engine uses that forecast as an anchor and supplies the missing distribution around it.

The clean mental model is:

```text
Official forecast -> location / center
Probability engine -> uncertainty shape / tail probabilities / bucket probabilities
```

For example, if the selected official forecast max is `32C`, B4 does not simply predict bucket `32`. It builds a historical residual distribution for comparable month/forecast-level cases and maps those possible residual outcomes into the 11 bucket probabilities.

## Probability Bucket Contract

The target is the HKO Daily Extract one-decimal daily maximum temperature at HKG. Bucket assignment must be done after one-decimal Decimal-safe normalization. The exact bucket keys are:

```text
24_or_below
25
26
27
28
29
30
31
32
33
34_or_higher
```

Exact rounding/bucket rules:

```text
24.9 or lower -> 24_or_below
25.0..25.9    -> 25
26.0..26.9    -> 26
27.0..27.9    -> 27
28.0..28.9    -> 28
29.0..29.9    -> 29
30.0..30.9    -> 30
31.0..31.9    -> 31
32.0..32.9    -> 32
33.0..33.9    -> 33
>=34.0        -> 34_or_higher
```

Boundary examples verified in the row-count audit:

```text
24.9 -> 24_or_below
25.0 -> 25
31.9 -> 31
32.0 -> 32
34.0 -> 34_or_higher
```

The `31.9` rule is especially important: `31.9` stays bucket `31`; it does not round into bucket `32`.

## Requirements-to-Implementation Traceability

| Requirement | Implementation Location | Behavior Delivered | Verification Evidence | Caveat |
|---|---|---|---|---|
| Build a weather-probability-only system, not a trading system. | `configs/hkg_tmax/probability_bucket_v1.yaml`, `live_inference.py`, `leakage_audit.py` | Config excludes market prices, EV, order books, Kelly, PnL, market-implied blending, and trade recommendations. Live output emits only bucket probabilities. | `live_inference_no_trading_audit.json` status `pass`. | Trading integration remains a separate future stage. |
| Use official HKO Info.gov local forecasts as the point-forecast anchor. | `data_build.py`, `forecast_selection.py` | Loads strict `source='info_gov'`, `product_type='local'`, `row_quality_status='usable_local_minmax'`, lead-1 rows. | `source_eligibility_audit.csv`, `row_count_audit.json`. | Future HKO schema drift would need parser/source audit updates. |
| Select the latest eligible forecast before cutoff. | `forecast_selection.py` | Builds target-date cutoff timestamps and selects the last deterministic eligible row where `issue_at_utc` is at or before `cutoff_at_utc`. | `leakage_audit.json` reports `post_cutoff_forecast_rows: 0`; unit test covers deterministic tie-break. | The chosen cutoff is only as good as archived issue timestamps. |
| Use exact bucket rounding rules. | `bucket_rules.py` | Uses Decimal one-decimal normalization and exact bucket boundaries. | Unit test covers `24.9`, `25.0`, `25.9`, `31.9`, `32.0`, `33.9`, `34.0`; `row_count_audit.json` records examples. | Bucket rules must be rechecked if market settlement language changes. |
| Compare all requested method families. | `models.py` and method wrapper modules | Implements B0-B6, P1-P2, C1-C2, K0-K2, and S1. | `scoreboard.csv`, `scoreboard_by_split.csv`, `model_selection_log.json`. | A true dedicated EMOS model is not separately implemented. |
| Use temporal governance without sealed tuning. | `validation.py`, runner | Fold 1-4 and presealed are used for selection; sealed 2024-2026 is scored after freeze. | `scoreboard_by_split.csv`; `leakage_audit.json` has `sealed_rows_tuning_allowed: false`. | Sealed results can inform next-round design, not this champion selection. |
| Produce full result artifacts. | `reporting.py`, runner | Writes scoreboards, predictions, PMFs, diagnostics, audits, stack weights, model card, manifest. | `experiments/hkg_tmax_probability_buckets_v1/results/` contains 41 artifacts. | Generated artifacts are local experiment outputs, not installed package data. |
| Choose champion objectively by gates. | `reporting.py`, config gates | Sorts by RPS but applies NLL/Brier and simplicity promotion gates. | `scoreboard.csv` marks B4 champion; B5 `fail:nll`; S1 pass but below promotion threshold. | Promotion thresholds are configurable and should stay predeclared. |
| Provide reproducible command and tests. | runner, test file, experiment README | Adds command-line benchmark and focused pytest coverage. | `pytest ... -q` returned `10 passed`; full runner completed successfully. | Full benchmark needs local PostgreSQL `hkg_tmax_research`. |

## Change Inventory

This inventory covers the files added or modified for the probability distribution engine work. It intentionally does not cover unrelated pre-existing dirty-worktree files outside this scope.

| Path | Type | Why It Changed | Main Objects | Effect | Verification |
|---|---|---|---|---|---|
| `code/src/hkg_tmax_probability/__init__.py` | added package init | Expose package constants and bucket helpers. | `BUCKET_KEYS`, `bucket_index`, `bucket_key` | Makes probability package importable. | Imported by tests and runner. |
| `code/src/hkg_tmax_probability/bucket_rules.py` | added implementation | Define exact Decimal-safe bucket contract. | `BUCKET_KEYS`, `bucket_key`, `bucket_index`, `normalize_probability_matrix` | Prevents float/rounding mistakes like `31.9` becoming `32`. | Boundary unit tests passed. |
| `code/src/hkg_tmax_probability/scoring.py` | added implementation | Score probability matrices. | `ranked_probability_score`, `multiclass_log_loss`, `multiclass_brier`, `calibration_errors` | Supplies primary and secondary metrics. | RPS ordering unit test; full scoreboards generated. |
| `code/src/hkg_tmax_probability/forecast_selection.py` | added implementation | Build cutoff timestamps and select latest eligible forecasts. | `CutoffProfile`, `target_cutoff_utc`, `select_latest_eligible_forecasts`, `build_revision_features` | Enforces point-in-time official forecast anchor selection. | No post-cutoff audit and tie-break unit test passed. |
| `code/src/hkg_tmax_probability/data_build.py` | added implementation | Load PostgreSQL sources and build modeling table. | `load_targets`, `load_strict_info_gov_forecasts`, `build_modeling_table` | Creates canonical rows for all cutoffs and splits. | `row_count_audit.json` and Parquet outputs generated. |
| `code/src/hkg_tmax_probability/label_publication_audit.py` | added implementation | Compare canonical labels against first raw Daily Extract publication. | `run_label_publication_audit`, `apply_first_publication_labels` | Detects whether first-publication bucket labels differ. | Audit found 49,627 raw rows and 0 bucket changes; unit test covers bucket-change application. |
| `code/src/hkg_tmax_probability/leakage_audit.py` | added implementation | Enforce leakage/no-trading checks. | `audit_modeling_table`, `audit_live_output` | Blocks post-cutoff, duplicate, forbidden predictor, sealed, and trading-field violations. | `leakage_audit.json` and live audit passed; unit tests passed. |
| `code/src/hkg_tmax_probability/models.py` | added implementation | Implement all benchmark model families. | B0-B6, P1-P2, C1-C2, K0-K2, S1 functions | Produces bucket probability matrices for every compared method. | Full benchmark generated per-method predictions and scoreboard. |
| `code/src/hkg_tmax_probability/residual_pmf.py` | added wrapper | Expose residual PMF helpers by family. | `grouped_residual_pmf_predict` | Keeps method-family imports stable. | Imported package successfully. |
| `code/src/hkg_tmax_probability/hierarchical_shrinkage.py` | added wrapper | Expose B4 helpers by family. | `hierarchical_month_forecast_pmf_predict`, `select_b4_alphas` | Keeps B4 family addressable. | Runner used B4 successfully. |
| `code/src/hkg_tmax_probability/kernel_analog.py` | added wrapper | Expose B5 helper by family. | `kernel_analog_residual_pmf_predict` | Keeps B5 family addressable. | B5 scored in benchmark. |
| `code/src/hkg_tmax_probability/mos_distributions.py` | added wrapper | Expose MOS helper by family. | `mos_predict` | Keeps P1/P2 family addressable. | P1/P2 scored in benchmark. |
| `code/src/hkg_tmax_probability/direct_classifiers.py` | added wrapper | Expose classifier helpers by family. | `multinomial_predict`, `ordinal_cdf_predict` | Keeps C1/C2 family addressable. | C1/C2 scored in benchmark. |
| `code/src/hkg_tmax_probability/cdf_calibration.py` | added wrapper | Expose calibration helpers. | `monotone_cdf_projection`, `power_calibration` | Supports K0/K1/K2 calibration layer tests. | Monotone projection unit test passed. |
| `code/src/hkg_tmax_probability/probability_stacking.py` | added wrapper | Expose stack helpers. | `fit_stack_weights`, `optimize_stack_weights` | Supports S1 conservative stack. | Stack weight unit test passed; `stack_weights.csv` generated. |
| `code/src/hkg_tmax_probability/reporting.py` | added implementation | Write scoreboards, diagnostics, champion flag, model card, manifest. | `score_predictions`, `add_leaderboard_rank_and_gates`, `write_diagnostics`, `write_model_card` | Applies B4 simplicity gate and writes final result artifacts. | `scoreboard.csv` marks B4 champion; model card generated. |
| `code/src/hkg_tmax_probability/live_inference.py` | added implementation | Write live probability-only example. | `write_live_inference_example` | Produces example bucket distribution with no trading fields. | `live_inference_no_trading_audit.json` passed. |
| `code/src/hkg_tmax_probability/validation.py` | added implementation | Build temporal split windows. | `SplitWindow`, `split_windows_from_config`, `train_validation_frames` | Prevents sealed rows from training sealed validation. | Unit test covers sealed exclusion. |
| `configs/hkg_tmax/probability_bucket_v1.yaml` | added config | Predeclare buckets, cutoffs, folds, models, metrics, gates, and exclusions. | All V1 experiment settings. | Makes run reproducible and governed. | Runner consumed config successfully. |
| `scripts/run_hkg_tmax_probability_bucket_v1.py` | added script | Execute full benchmark and artifact generation. | CLI `--config`, `--output-dir`, `--database-url` | One-command reproduction of V1 experiment. | Full benchmark completed successfully. |
| `code/tests/test_hkg_tmax_probability_bucket_v1.py` | added test | Cover V1 hard contracts. | 10 focused tests. | Prevents bucket, leakage, selection, PMF, stack, and live-output regressions. | `10 passed`. |
| `experiments/hkg_tmax_probability_buckets_v1/README.md` | added docs | Make experiment folder self-describing. | Reproduce command and result summary. | Faster future navigation. | Manual readback. |
| `experiments/hkg_tmax_probability_buckets_v1/results/*` | generated results | Store benchmark outputs and audits. | 41 artifacts. | Evidence for champion and diagnostics. | Manifest and file inventory generated. |
| `CHANGELOG.md` | modified docs | Record probability engine implementation. | New 2026-07-05 section. | Repo-level history updated. | Manual readback. |
| `EXPERIMENT_INDEX.md` | modified docs | Register HKG-PROB-V1. | New index row. | Future experiment lookup updated. | Manual readback. |
| `docs/PROJECT_STRUCTURE_AND_CODE_MAP.md` | modified docs | Map package, config, script, test, and evidence folder. | Code map entries. | Future agents know where to inspect. | Manual readback. |
| `documentation/strategy_implementation_documentation/context/live_trading/HKG_TMAX_PROBABILITY_DISTRIBUTION_ENGINE_RESULTS_20260705.md` | added docs | Durable live-trading context handoff for this engine. | This document. | Captures final result, decisions, risks, and next steps. | Text scan and scoped quality gate run. |

## Architecture and Control Flow

The V1 system is a local batch benchmark plus a probability-only inference example. Its data/control flow is:

```text
PostgreSQL hkg_tmax_research
  -> load canonical/sealed labels
  -> load strict Info.gov local forecast rows
  -> build target-date x cutoff modeling table
  -> select latest eligible pre-cutoff forecast anchor
  -> compute residuals and revision features
  -> split into chronological validation windows
  -> train each probability method on past rows only
  -> predict 11 bucket probabilities for validation rows
  -> score by RPS/NLL/Brier/CRPS/ECE/MCE/entropy
  -> apply leakage, row-identity, and acceptance gates
  -> write scoreboards, diagnostics, model card, live example, manifest
```

Key side effects:

- Reads local PostgreSQL.
- Writes local Parquet/CSV/JSON/Markdown artifacts under `experiments/hkg_tmax_probability_buckets_v1/results/`.
- Does not call external APIs.
- Does not emit or consume market/trading inputs.

Primary failure paths:

- PostgreSQL unavailable or schema changed.
- No eligible forecast rows before cutoff.
- Probability matrix has negative values or rows that cannot normalize.
- A method produces non-identical row sets.
- Leakage audit finds post-cutoff or forbidden target predictors.
- Acceptance gates reject a lower-RPS complex method.

## File-by-File Deep Dive

### `bucket_rules.py`

Responsibility:

Defines the exact market-style HKG Tmax bucket rule and probability matrix normalization behavior.

Important objects:

- `BUCKET_KEYS`: canonical 11 bucket labels.
- `PROBABILITY_COLUMNS`: `p_` prefixed probability column names.
- `decimal_1dp`: converts values to one decimal using Decimal half-up rounding.
- `bucket_key`: maps a one-decimal value into the exact bucket.
- `normalize_probability_matrix`: floors and row-normalizes probability arrays.

Maintenance notes:

- Preserve Decimal handling. Do not replace this with plain `round(float_value, 1)` logic.
- Preserve the `31.9 -> 31` invariant.
- If settlement rules change, update this file and the tests first.

### `forecast_selection.py`

Responsibility:

Implements cutoff timestamps and deterministic latest-eligible official forecast selection.

Important objects:

- `CutoffProfile`: stores cutoff name, HKT time, and primary flag.
- `target_cutoff_utc`: converts target date and HKT cutoff into UTC.
- `select_latest_eligible_forecasts`: filters forecasts by issue time and selects the deterministic latest row.
- `build_revision_features`: computes revision-path counts, widths, slopes, and direction.

Maintenance notes:

- Deterministic sorting is essential. Forecast archives can have multiple rows with the same issue time.
- Do not loosen the rule that `issue_at_utc` must be at or before `cutoff_at_utc`.
- Keep revision features pre-cutoff only.

### `data_build.py`

Responsibility:

Loads target and forecast data from PostgreSQL, builds the modeling table, assigns split labels, computes residuals, and writes table/audit artifacts.

Important objects:

- `load_targets`: unions canonical core labels and sealed confirmation labels.
- `load_strict_info_gov_forecasts`: loads strict official forecast rows.
- `build_modeling_table`: joins labels to selected forecasts and revision features.
- `write_modeling_artifacts`: writes modeling table, selected forecasts, eligible revisions, schema, and row-count audits.

Maintenance notes:

- PostgreSQL is the source of truth for this lane.
- Do not silently switch to MySQL or local CSVs for benchmark truth.
- Preserve `target_table` because sealed-row governance depends on it.

### `models.py`

Responsibility:

Contains the probability method implementations.

Important model paths:

- Residual PMF conversion: `residual_pmf_to_bucket_probs`.
- B1/B2/B3: `grouped_residual_pmf_predict`.
- B4: `hierarchical_month_forecast_pmf_predict` and `select_b4_alphas`.
- B5: `kernel_analog_residual_pmf_predict`.
- P1/P2: `mos_predict`.
- C1: `multinomial_predict`.
- C2: `ordinal_cdf_predict`.
- K1/K2: `power_calibration`, `monotone_cdf_projection`, `cdf_to_bucket_probs`.
- S1: `fit_stack_weights`, `optimize_stack_weights`, `predict_all_methods`.

Maintenance notes:

- B4 must remain simple and auditable because it is the champion.
- B5 should not be promoted unless NLL/calibration are repaired.
- S1 should stay conservative unless a future base method adds strong independent signal.
- Do not describe P1/P2 as full EMOS.

### `scoring.py`

Responsibility:

Scores probability distributions and calibration diagnostics.

Important objects:

- `ranked_probability_score`: normalized ordered bucket score.
- `multiclass_log_loss`: NLL.
- `multiclass_brier`: multiclass Brier.
- `crps_bucket_proxy`: midpoint-based CRPS proxy.
- `calibration_errors`: ECE/MCE and reliability bins.
- `summarize_scores`: one-call score summary.

Maintenance notes:

- RPS is the primary metric and must remain normalized consistently across runs.
- NLL is a gate, not just a secondary display metric.

### `reporting.py`

Responsibility:

Turns predictions into scoreboards, diagnostics, model cards, bootstrap deltas, and reproducibility metadata.

Important objects:

- `probability_predictions_frame`: builds per-row prediction artifact.
- `score_predictions`: aggregates method metrics.
- `add_leaderboard_rank_and_gates`: applies NLL/Brier gates and the B4 simplicity rule.
- `grouped_scoreboard`: creates slice scoreboards.
- `write_diagnostics`: reliability, PIT, Brier, interval, and sharpness diagnostics.
- `bootstrap_deltas`: bootstrap RPS deltas versus B4.
- `write_model_card`: final champion summary.
- `write_manifest`: artifact hashes/runtime metadata.

Maintenance notes:

- The champion is not simply the row with lowest RPS. The simplicity rule is intentional.
- If acceptance gates change, update the config and explain why before rerunning.

### `label_publication_audit.py`

Responsibility:

Checks whether first-published raw Daily Extract labels would change bucket outcomes versus canonical labels.

Important objects:

- `run_label_publication_audit`: reads raw audit rows and compares first-publication bucket to canonical bucket.
- `apply_first_publication_labels`: can create first-publication label columns for alternative scoring.

Maintenance notes:

- Keep this audit even when current bucket changes are zero.
- If future bucket changes appear, require both canonical and first-publication scoreboards.

### `leakage_audit.py`

Responsibility:

Detects invalid predictors, post-cutoff forecasts, duplicate target/cutoff rows, sealed misuse, and forbidden trading fields.

Important objects:

- `audit_modeling_table`.
- `audit_live_output`.

Maintenance notes:

- Add new forbidden predictor fragments if future labels/raw-target columns enter the modeling table.
- Keep trading fields forbidden until a separate trading-stage spec is approved.

### `live_inference.py`

Responsibility:

Writes an example probability-only inference payload and audits it for forbidden trading fields.

Important object:

- `write_live_inference_example`.

Maintenance notes:

- This file is not a trading adapter.
- It should emit weather probabilities only.

### `validation.py`

Responsibility:

Builds split windows from config and slices train/validation frames with sealed governance.

Important objects:

- `SplitWindow`.
- `split_windows_from_config`.
- `train_validation_frames`.

Maintenance notes:

- Preserve sealed exclusion for training.
- Do not tune on `sealed_confirmation`.

### `scripts/run_hkg_tmax_probability_bucket_v1.py`

Responsibility:

Main reproducible entry point for V1.

Important flow:

1. Parse CLI args.
2. Load config.
3. Build modeling table.
4. Write audits.
5. Run all primary validation windows.
6. Run cutoff sensitivity.
7. Aggregate scoreboards.
8. Write diagnostics and model card.
9. Write live inference example.
10. Write manifest.

Maintenance notes:

- Keep the default output path stable.
- Keep `--database-url` override available for controlled reruns.

## Public Interfaces and Contracts

### CLI

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_probability_bucket_v1.py --config configs\hkg_tmax\probability_bucket_v1.yaml --output-dir experiments\hkg_tmax_probability_buckets_v1\results
```

Optional override:

```powershell
--database-url postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research
```

### Config Contract

Required config sections:

```text
experiment_id
database_url
target
cutoffs
forecast_filter
temporal_governance
metrics
bootstrap
models
calibration
stacking
acceptance_gates
scope_exclusions
```

### Output Contract

The primary downstream contract is:

```text
bucket_probabilities.parquet
```

It contains one row per target/cutoff/method with columns:

```text
target_date
cutoff_profile
validation_split
method
p_24_or_below
p_25
p_26
p_27
p_28
p_29
p_30
p_31
p_32
p_33
p_34_or_higher
```

The live inference example output contract is:

```json
{
  "status": "ok",
  "target_date": "YYYY-MM-DD",
  "cutoff_profile": "t_minus_1_2359_hkt",
  "forecast_max_c": 32.0,
  "forecast_min_c": 27.0,
  "method": "B4_hierarchical_residual_pmf",
  "bucket_probabilities": {
    "24_or_below": 0.0,
    "25": 0.0,
    "26": 0.0,
    "27": 0.0,
    "28": 0.0,
    "29": 0.0,
    "30": 0.0,
    "31": 0.0,
    "32": 0.0,
    "33": 0.0,
    "34_or_higher": 0.0
  },
  "scope": "weather_probability_only"
}
```

Forbidden output contract:

```text
No market prices.
No EV.
No Kelly.
No PnL.
No order book.
No trade recommendation.
```

## Source-Of-Truth Inputs

The implementation and this document use these source-of-truth inputs:

### Configuration

```text
configs/hkg_tmax/probability_bucket_v1.yaml
```

This config defines:

- bucket keys,
- Decimal-safe bucket rules,
- strict forecast filters,
- cutoff profiles,
- validation windows,
- score metrics,
- bootstrap settings,
- model grids/defaults,
- calibration variants,
- stacking regularization,
- acceptance gates,
- explicit trading-scope exclusions.

### PostgreSQL Tables

Database:

```text
postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research
```

Tables used:

```text
public.hko_historical_forecasts_2000_2026
label_core.hko_daily_tmax
sealed_confirmation.hko_daily_tmax
raw_audit.codex_audit_ds_01_hko_daily_tmax_target_hko_daily_extr_23cb54da
```

Strict official forecast filter:

```text
source = 'info_gov'
product_type = 'local'
row_quality_status = 'usable_local_minmax'
target_issue_lead_days = 1
target_date is not null
issue_at_utc is not null
forecast_min_c is not null
forecast_max_c is not null
```

### Generated Evidence Files

Primary evidence folder:

```text
experiments/hkg_tmax_probability_buckets_v1/results/
```

Most important evidence files:

```text
scoreboard.csv
scoreboard_by_split.csv
scoreboard_by_cutoff.csv
scoreboard_july.csv
modeling_table.parquet
selected_forecast_rows.parquet
eligible_forecast_revision_rows.parquet
per_fold_predictions.parquet
bucket_probabilities.parquet
one_decimal_pmfs.parquet
leakage_audit.json
label_publication_audit.json
row_identity_gate.json
live_inference_no_trading_audit.json
model_selection_log.json
stack_weights.csv
proper_score_deltas_bootstrap.csv
final_probability_model_card.md
reproducibility_manifest.json
```

## Data Sources And Row Construction

### Loaded Rows

From `row_count_audit.json`:

```text
targets_loaded: 9,648
strict_forecasts_loaded: 88,504
eligible_revision_rows: 165,380
selected_forecast_rows: 26,407
modeling_rows: 26,407
primary_modeling_rows: 9,644
```

The `modeling_rows` count is larger than `primary_modeling_rows` because the modeling table includes multiple cutoff profiles:

```text
T-1 18:00 HKT
T-1 21:00 HKT
T-1 23:59 HKT
```

The primary leaderboard uses only:

```text
T-1 23:59 HKT
```

### Forecast Selection

For each target date and cutoff profile, the selection logic:

1. Loads all strict eligible Info.gov local forecast rows.
2. Filters to forecasts with `issue_at_utc` at or before `cutoff_at_utc`.
3. Sorts deterministically by target date, cutoff profile, issue time, snapshot time, ingest/archive timestamps, raw hash, and bulletin id where available.
4. Selects the last eligible row as the anchor forecast.
5. Preserves all prior eligible revision rows for revision-path features and auditability.

### Features Used By The Probability Models

The core residual PMF methods use:

```text
target_month
official_max_round
forecast_max_c
forecast_max_tenths
residual_tenths
```

Other methods use more forecast/revision features:

```text
forecast_min_c
forecast_max_c
forecast_range_c
forecast_midpoint_c
issue_hour_hkt
revision_count
forecast_max_revision_c
forecast_max_path_width_c
forecast_max_std_path
forecast_max_path_slope_c_per_revision
target_dayofyear
season
revision_direction
```

Target labels, raw audit target fields, and post-cutoff forecast rows are not used as predictors. This is enforced by `leakage_audit.json` and the unit tests.

## Validation Design

### Primary Development And Presealed Evaluation

The primary leaderboard combines Fold 1-4 plus presealed 2022-2023:

| Split | Training End | Validation Start | Validation End | Primary Rows |
|---|---:|---:|---:|---:|
| Fold 1 | 2010-12-31 | 2011-01-01 | 2013-12-31 | 1,096 |
| Fold 2 | 2013-12-31 | 2014-01-01 | 2016-12-31 | 1,095 |
| Fold 3 | 2016-12-31 | 2017-01-01 | 2019-12-31 | 1,095 |
| Fold 4 | 2019-12-31 | 2020-01-01 | 2021-12-31 | 731 |
| Presealed | 2021-12-31 | 2022-01-01 | 2023-12-31 | 730 |

Total primary leaderboard rows:

```text
4,747
```

### Sealed Confirmation

The sealed confirmation split was also run:

| Split | Training End | Validation Start | Validation End | Rows |
|---|---:|---:|---:|---:|
| Sealed confirmation | 2023-12-31 | 2024-01-01 | 2026-05-31 | 882 |

Sealed data was not allowed to tune model selection. It is used only as a post-freeze confirmation view.

### Cutoff Sensitivity

Three cutoffs were tested:

```text
T-1 18:00 HKT
T-1 21:00 HKT
T-1 23:59 HKT
```

The primary cutoff is:

```text
T-1 23:59 HKT
```

In Stockholm summer time, that is approximately:

```text
T-1 17:59 CEST
```

Cutoff sensitivity for B4:

| Cutoff | Rows | RPS | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| T-1 18:00 HKT | 5,050 | 0.044613 | 1.099855 | 0.048521 | 0.014054 |
| T-1 21:00 HKT | 5,088 | 0.044301 | 1.094410 | 0.048338 | 0.015359 |
| T-1 23:59 HKT | 5,629 | 0.041782 | 1.039917 | 0.046063 | 0.020421 |

The later cutoff is materially better by RPS/NLL/Brier because it can use later official forecast revisions before the target day.

## Methods Benchmarked

### Baselines And Residual PMFs

| Method | Family | Meaning |
|---|---|---|
| `B0_climatology` | baseline | Historical bucket frequencies only. Does not use the official forecast anchor. |
| `B1_global_residual_pmf` | residual PMF | Official forecast max plus one global empirical residual distribution. |
| `B2_month_residual_pmf` | residual PMF | Official forecast max plus month-conditioned residual distribution. |
| `B3_forecast_level_residual_pmf` | residual PMF | Official forecast max plus forecast-level-conditioned residual distribution. |
| `B4_hierarchical_residual_pmf` | residual PMF | Official forecast max plus month x forecast-level residual distribution with shrinkage. |
| `B5_kernel_analog_pmf` | analog PMF | Analog-weighted residual distribution using forecast max/range/revision/month distances. |
| `B6_month_climatology_diagnostic` | diagnostic | Month-only bucket climatology diagnostic. |

### MOS-Style Distribution Models

| Method | Family | Meaning |
|---|---|---|
| `P1_normal_mos` | MOS | Ridge residual mean model plus Normal residual distribution, converted to bucket probabilities. |
| `P2_student_t_mos` | MOS | Ridge residual mean model plus Student-t residual distribution, converted to bucket probabilities. |

These are MOS-style distribution models. They are not a full dedicated EMOS implementation.

### Direct Classifiers

| Method | Family | Meaning |
|---|---|---|
| `C1_multinomial_ridge` | direct classifier | Multiclass logistic bucket classifier. |
| `C2_ordinal_cdf_logistic` | direct classifier | Ordered-threshold logistic CDF model with monotone projection. |

### Calibration Layers

| Method | Family | Meaning |
|---|---|---|
| `K0_B4_identity` | calibration | B4 unchanged. |
| `K1_B4_power_calibrated` | calibration | B4 probabilities adjusted with fixed power gamma. |
| `K2_B4_monotone_cdf_projected` | calibration | B4 CDF projected to monotone CDF. B4 was already monotone, so this equals B4 in this run. |

### Stack

| Method | Family | Meaning |
|---|---|---|
| `S1_conservative_simplex_stack` | stack | Non-negative simplex blend, heavily regularized toward B4. |

Average S1 weights across splits:

| Base Method | Average Weight |
|---|---:|
| `B4_hierarchical_residual_pmf` | 0.850000 |
| `B1_global_residual_pmf` | 0.034587 |
| `P1_normal_mos` | 0.031271 |
| `B3_forecast_level_residual_pmf` | 0.029244 |
| `C1_multinomial_ridge` | 0.029196 |
| `B2_month_residual_pmf` | 0.024238 |
| `B0_climatology` | 0.001465 |

This confirms that even the stack mostly wanted B4.

## EMOS Clarification

A true dedicated EMOS implementation was not run as a separately labeled model. The table does not contain a method named `EMOS` because the implemented distribution methods were:

```text
P1_normal_mos
P2_student_t_mos
```

These are MOS-style residual distribution models. They use a residual-mean model and a parametric residual family, then integrate the resulting continuous distribution over the bucket boundaries.

The distinction matters:

- MOS-style models in this run were relatively simple.
- A full EMOS/GAMLSS-style model would generally model distribution parameters more explicitly, potentially including mean, scale, shape, and calibration terms conditioned on predictors.
- Therefore this run should not be cited as proof that EMOS cannot work.
- This run only shows that the implemented Normal MOS and Student-t MOS variants did not beat B4 and failed the NLL gate.

MOS results:

| Method | RPS | NLL | Brier | Gate |
|---|---:|---:|---:|---|
| `P1_normal_mos` | 0.041777 | 1.042185 | 0.046279 | fail:nll |
| `P2_student_t_mos` | 0.041868 | 1.047208 | 0.046332 | fail:nll |

Conclusion:

```text
Do not claim that EMOS was fully tested.
Do claim that the first MOS-style parametric residual distribution attempts did not beat B4.
```

## Champion Mechanics: How B4 Generates Probabilities

`B4_hierarchical_residual_pmf` works as follows.

For each historical training row:

```text
residual_c = target_tmax_c - forecast_max_c
residual_tenths = round(residual_c * 10)
forecast_max_tenths = round(forecast_max_c * 10)
official_max_round = round(forecast_max_c)
target_month = month(target_date)
```

It builds empirical residual distributions over a residual support grid:

```text
-12.0C to +12.0C in 0.1C residual increments
```

The hierarchy is:

```text
global residual PMF
  -> month residual PMF
      -> month x official_max_round residual PMF
```

The cell distribution is shrunk toward the month distribution, and the month distribution is shrunk toward the global distribution. This prevents sparse month/forecast-level cells from becoming too noisy.

For a new forecast row:

1. Identify its `target_month`.
2. Identify its rounded official forecast max, `official_max_round`.
3. Retrieve the corresponding month x forecast-level residual PMF when available.
4. Fall back to the month PMF or global PMF if the cell is missing.
5. Add every possible residual value to the official forecast max.
6. Convert each possible resulting one-decimal Tmax into the exact bucket rule.
7. Sum residual probabilities by bucket.
8. Normalize the final 11-bucket probability vector.

This is why B4 is a probability engine. The official forecast only supplies the center; B4 supplies the uncertainty distribution.

## Primary Leaderboard

Primary leaderboard split:

```text
fold1_4_plus_presealed_primary
```

Rows:

```text
4,747
```

Sorted by normalized RPS ascending:

| Rank | Method | Family | RPS | Delta vs B4 | Relative Gain vs B4 | NLL | Brier | ECE | Gate | Champion |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `B5_kernel_analog_pmf` | analog PMF | 0.041287 | -0.000236 | 0.5686% | 1.075467 | 0.045778 | 0.014419 | fail:nll | no |
| 2 | `S1_conservative_simplex_stack` | stack | 0.041472 | -0.000051 | 0.1232% | 1.032837 | 0.045867 | 0.018679 | pass | no |
| 3 | `K0_B4_identity` | calibration | 0.041524 | 0.000000 | 0.0000% | 1.037181 | 0.045921 | 0.019859 | pass | no |
| 4 | `B4_hierarchical_residual_pmf` | residual PMF | 0.041524 | 0.000000 | 0.0000% | 1.037181 | 0.045921 | 0.019859 | pass | yes |
| 5 | `K2_B4_monotone_cdf_projected` | calibration | 0.041524 | 0.000000 | 0.0000% | 1.037181 | 0.045921 | 0.019859 | pass | no |
| 6 | `K1_B4_power_calibrated` | calibration | 0.041531 | 0.000007 | -0.0173% | 1.036386 | 0.045904 | 0.014911 | pass | no |
| 7 | `B3_forecast_level_residual_pmf` | residual PMF | 0.041616 | 0.000093 | -0.2229% | 1.038655 | 0.045978 | 0.010817 | pass | no |
| 8 | `B2_month_residual_pmf` | residual PMF | 0.041666 | 0.000143 | -0.3440% | 1.041036 | 0.046030 | 0.014703 | pass | no |
| 9 | `B1_global_residual_pmf` | residual PMF | 0.041700 | 0.000176 | -0.4248% | 1.041529 | 0.046147 | 0.020253 | pass | no |
| 10 | `P1_normal_mos` | MOS | 0.041777 | 0.000254 | -0.6107% | 1.042185 | 0.046279 | 0.026100 | fail:nll | no |
| 11 | `P2_student_t_mos` | MOS | 0.041868 | 0.000345 | -0.8301% | 1.047208 | 0.046332 | 0.031952 | fail:nll | no |
| 12 | `C1_multinomial_ridge` | direct classifier | 0.042644 | 0.001121 | -2.6987% | 1.056927 | 0.046787 | 0.013813 | fail:nll | no |
| 13 | `C2_ordinal_cdf_logistic` | direct classifier | 0.043047 | 0.001524 | -3.6694% | 1.064943 | 0.046910 | 0.017663 | fail:nll | no |
| 14 | `B6_month_climatology_diagnostic` | diagnostic | 0.083575 | 0.042051 | -101.2706% | 1.531137 | 0.057426 | 0.051469 | fail:nll,brier | no |
| 15 | `B0_climatology` | baseline | 0.193317 | 0.151794 | -365.5608% | 2.115090 | 0.074987 | 0.032828 | fail:nll,brier | no |

## Why B4 Won Despite Ranking Fourth By Raw RPS

The scoreboard is sorted by raw normalized RPS. But champion selection also applies acceptance gates.

Acceptance gates from config:

```text
leakage_total_violations: 0
identical_row_set_per_leaderboard: true
complex_vs_b4_fold14_min_rps_gain: 0.015
complex_vs_b4_presealed_min_rps_gain: 0.010
nll_worse_than_b4_max: 0.005
brier_worse_than_b4_max: 0.002
key_calibration_bin_min_n: 100
key_calibration_abs_gap_max: 0.12
```

Decision logic:

1. B5 had the best RPS but failed NLL:

```text
B4 NLL: 1.037181
B5 NLL: 1.075467
Allowed no-worse margin: 0.005
B5 NLL excess over B4: 0.038286
Decision: fail:nll
```

2. S1 passed gates but did not clear promotion threshold:

```text
B4 RPS: 0.041524
S1 RPS: 0.041472
Absolute gain: 0.000051
Relative gain: 0.1232%
Required improvement threshold: at least 1.0% presealed and 1.5% fold selection gate
Decision: not enough gain to replace simpler B4
```

3. K0 and K2 equal B4 because they are identity/equivalent transformations in this run.

4. K1 slightly worsened RPS.

Therefore:

```text
Champion: B4_hierarchical_residual_pmf
Reason: best simple model that passed all gates; complex challengers did not justify promotion.
```

## Split-Level B4 Performance

| Split | Rows | RPS | NLL | Brier | CRPS | ECE | MCE | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fold 1 | 1,096 | 0.041357 | 1.018151 | 0.044524 | 0.413567 | 0.031314 | 0.209517 | 0.982237 |
| Fold 2 | 1,095 | 0.040075 | 0.990485 | 0.043992 | 0.400752 | 0.025934 | 0.117687 | 0.974442 |
| Fold 3 | 1,095 | 0.041195 | 1.046228 | 0.046509 | 0.411951 | 0.029475 | 0.078714 | 1.034267 |
| Fold 4 | 731 | 0.043839 | 1.112744 | 0.049042 | 0.438387 | 0.035648 | 0.118807 | 1.056255 |
| Presealed 2022-2023 | 730 | 0.042121 | 1.046562 | 0.046906 | 0.421209 | 0.032575 | 0.154770 | 1.029523 |
| Sealed 2024-2026-05 | 882 | 0.043174 | 1.054641 | 0.046825 | 0.431741 | 0.031253 | 0.112686 | 1.028066 |

Interpretation:

- Fold 2 was easiest for B4 by RPS.
- Fold 4 and sealed confirmation were harder.
- Sealed confirmation did not collapse; B4 remained in the same broad quality range.
- The sealed RPS `0.043174` is worse than the primary aggregate `0.041524`, but not an out-of-family failure.

## July Slice

The July-only scoreboard matters because the current operational use case often focuses on summer/high-temperature markets.

Rows:

```text
465 July rows in the primary-scoreboard slice
```

Top July methods:

| Method | RPS | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| `S1_conservative_simplex_stack` | 0.057678 | 1.422616 | 0.063542 | 0.032826 |
| `K1_B4_power_calibrated` | 0.057689 | 1.422736 | 0.063531 | 0.041144 |
| `B4_hierarchical_residual_pmf` | 0.057741 | 1.426918 | 0.063605 | 0.037342 |
| `K0_B4_identity` | 0.057741 | 1.426918 | 0.063605 | 0.037342 |
| `K2_B4_monotone_cdf_projected` | 0.057741 | 1.426918 | 0.063605 | 0.037342 |
| `B5_kernel_analog_pmf` | 0.057898 | 1.514639 | 0.063486 | 0.041693 |
| `B1_global_residual_pmf` | 0.058136 | 1.452134 | 0.064773 | 0.026294 |

Interpretation:

- July is harder than the full-year average. Full primary B4 RPS is `0.041524`; July B4 RPS is `0.057741`.
- S1 is marginally better than B4 in July, but the gain remains tiny.
- B5 is not attractive in July because its NLL is much worse.
- Future work should specifically revisit July/high-heat calibration and tails because this is likely one of the most operationally important regimes.

## Metric Definitions And How To Interpret Them

### Normalized RPS

Normalized Ranked Probability Score is the primary metric.

It measures how close the predicted cumulative bucket distribution is to the actual ordered bucket outcome. Lower is better. Because the buckets are ordered temperatures, RPS is more appropriate than plain multiclass accuracy: being one bucket off is not as bad as being five buckets off.

Interpretation:

```text
0.000 = perfect
lower = better
```

In this run:

```text
B0 climatology: 0.193317
B1 global residual PMF: 0.041700
B4 champion: 0.041524
```

The huge improvement from B0 to B1 shows that anchoring the distribution around the official forecast is essential. The small improvement from B1 to B4 shows that the better residual-conditioning layer adds value, but not a massive breakthrough.

### NLL

Negative log loss punishes confident wrong probabilities. Lower is better.

This is why B5 did not win. It had better RPS but materially worse NLL than B4:

```text
B4 NLL: 1.037181
B5 NLL: 1.075467
```

The configured allowed NLL deterioration versus B4 was only `0.005`. B5 was worse by about `0.038286`.

### Brier

Multiclass Brier score measures squared probability error across all buckets. Lower is better.

In this run:

```text
B4 Brier: 0.045921
S1 Brier: 0.045867
B5 Brier: 0.045778
```

B5 and S1 were slightly better on Brier, but B5 failed NLL and S1 lacked enough RPS gain.

### CRPS Proxy

CRPS proxy evaluates distribution quality on a bucket-midpoint scale. Lower is better.

In this run:

```text
B4 CRPS: 0.415236
S1 CRPS: 0.414724
B5 CRPS: 0.412875
```

This agrees with the RPS ordering, but the champion gate rejected B5 on NLL and S1 on insufficient gain.

### ECE And MCE

ECE is expected calibration error. MCE is maximum calibration error. Lower is better.

In this run:

```text
B4 ECE: 0.019859
B4 MCE: 0.040285
```

That is a decent calibration result. It does not mean perfect calibration, but it does not show obvious broad probability miscalibration.

### Entropy

Entropy measures how spread out the distribution is. Lower means sharper; higher means more diffuse. Entropy is not automatically better or worse on its own. A low-entropy model can be excellent if calibrated, or dangerous if overconfident.

In this run:

```text
B4 entropy: 1.011110
B5 entropy: 0.995639
S1 entropy: 1.029417
```

B5 is sharper than B4, which partly explains its better RPS but worse NLL. It likely over-concentrates probability in ways that hurt when it is wrong.

## Baselines And Objective Quality

There are three important baselines to understand.

### B0: Climatology-Only Baseline

`B0_climatology` ignores the official forecast and predicts historical bucket frequencies.

Result:

```text
RPS: 0.193317
NLL: 2.115090
Brier: 0.074987
```

This is a weak baseline. B4 is dramatically better.

### B1: Global Residual PMF Baseline

`B1_global_residual_pmf` uses the official forecast max as the center and applies a single global historical residual distribution.

Result:

```text
RPS: 0.041700
NLL: 1.041529
Brier: 0.046147
```

This is the real practical baseline because it already converts the official point forecast into a distribution.

### B4: Hierarchical Residual PMF Champion

`B4_hierarchical_residual_pmf` conditions the residual distribution by month and forecast level with shrinkage.

Result:

```text
RPS: 0.041524
NLL: 1.037181
Brier: 0.045921
```

Compared with B1:

```text
RPS improvement: about 0.42%
```

Compared with B0:

```text
RPS improvement: about 78.5%
```

The fair rating for B4 is around:

```text
78 / 100
```

Why not higher:

- It is stable, leakage-safe, interpretable, and calibrated decently.
- It crushes climatology.
- It turns a point forecast into a legitimate probability distribution.
- But its improvement over the simpler global residual distribution is modest.
- MOS-style models did not help.
- The stack only improved RPS by `0.123%`, which is too small to justify complexity.
- A true dedicated EMOS/GAMLSS/quantile/CDF-calibrated experiment has not yet been completed.

So B4 should be treated as a strong current reference engine, not as final proof that the distribution-generation problem is solved.

## Audits And Acceptance Gates

### Leakage Audit

File:

```text
experiments/hkg_tmax_probability_buckets_v1/results/leakage_audit.json
```

Result:

```json
{
  "status": "pass",
  "total_violations": 0,
  "violations": {
    "post_cutoff_forecast_rows": 0,
    "duplicate_target_cutoff_rows": 0,
    "forbidden_predictor_columns": [],
    "sealed_rows_before_2024": 0
  }
}
```

What this proves:

- No selected forecast rows had `issue_at_utc` after the cutoff.
- No duplicate target-date/cutoff rows entered the modeling table.
- No target/raw-audit fields were used as predictors.
- Sealed confirmation rows were not incorrectly included before 2024.

### Row Identity Gate

File:

```text
experiments/hkg_tmax_probability_buckets_v1/results/row_identity_gate.json
```

Result:

```text
status: pass
violations: 0
```

What this proves:

- Methods in a split were scored on identical row identities.
- The leaderboard is not comparing one method on an easier subset against another method on a harder subset.

### Label Publication Audit

File:

```text
experiments/hkg_tmax_probability_buckets_v1/results/label_publication_audit.json
```

Result:

```json
{
  "status": "ok",
  "raw_rows": 49627,
  "first_publication_rows": 9644,
  "bucket_changes": 0,
  "scoreboard_required": false
}
```

What this proves:

- The raw Daily Extract audit table was checked.
- First-publication rows were available for the primary canonical target dates.
- No first-publication bucket changes existed versus canonical labels.
- Therefore a separate first-publication scoreboard was not required in this run.

### Live Inference No-Trading Audit

File:

```text
experiments/hkg_tmax_probability_buckets_v1/results/live_inference_no_trading_audit.json
```

Result:

```json
{
  "status": "pass",
  "forbidden_fields": [],
  "probability_only": true
}
```

What this proves:

- The live inference example emits bucket probabilities only.
- It does not emit market prices, EV, Kelly, PnL, order-book fields, or trade recommendations.

## Live Inference Example

File:

```text
experiments/hkg_tmax_probability_buckets_v1/results/live_inference_example_output.json
```

Example target:

```text
target_date: 2026-05-31
cutoff_profile: t_minus_1_2359_hkt
forecast_max_c: 32.0
forecast_min_c: 27.0
method: B4_hierarchical_residual_pmf
```

Bucket probabilities:

| Bucket | Probability |
|---|---:|
| `24_or_below` | 0.000000001 |
| `25` | 0.000000001 |
| `26` | 0.000001207 |
| `27` | 0.000002414 |
| `28` | 0.014908934 |
| `29` | 0.040662827 |
| `30` | 0.139142953 |
| `31` | 0.300687470 |
| `32` | 0.376080652 |
| `33` | 0.079054756 |
| `34_or_higher` | 0.049458785 |

This example illustrates exactly what the engine is meant to do: it starts from an official max forecast of `32C`, then outputs a full bucket distribution, with nonzero mass on neighboring and tail buckets.

## Implementation And Artifact Map

### Main Package

```text
code/src/hkg_tmax_probability/
```

Important modules:

| File | Responsibility |
|---|---|
| `bucket_rules.py` | Decimal-safe bucket boundaries and probability column constants. |
| `forecast_selection.py` | Cutoff construction, latest eligible forecast selection, revision features. |
| `data_build.py` | PostgreSQL loading, modeling table construction, row-count artifacts. |
| `label_publication_audit.py` | First-publication audit against raw Daily Extract audit table. |
| `leakage_audit.py` | Cutoff, duplicate-row, forbidden-predictor, sealed, and no-trading audits. |
| `models.py` | B0-B6, P1-P2, C1-C2, K0-K2, and S1 probability methods. |
| `scoring.py` | RPS, NLL, Brier, CRPS proxy, ECE/MCE, entropy, calibration diagnostics. |
| `reporting.py` | Scoreboards, grouped diagnostics, bootstrap deltas, model card, manifest. |
| `live_inference.py` | Probability-only live inference example writer. |
| `validation.py` | Temporal split construction and train/validation slicing. |

Thin wrapper modules:

```text
residual_pmf.py
hierarchical_shrinkage.py
kernel_analog.py
mos_distributions.py
direct_classifiers.py
cdf_calibration.py
probability_stacking.py
```

These make method-family imports explicit and keep package boundaries clear.

### Config

```text
configs/hkg_tmax/probability_bucket_v1.yaml
```

Defines:

- exact buckets,
- cutoffs,
- strict forecast source filter,
- temporal governance,
- metrics,
- bootstrap settings,
- model grids/defaults,
- calibration settings,
- stack regularization,
- acceptance gates,
- trading-scope exclusions.

### Runner

```text
scripts/run_hkg_tmax_probability_bucket_v1.py
```

Responsibilities:

1. Load config.
2. Build modeling table from PostgreSQL.
3. Write table and row-count artifacts.
4. Run leakage and label-publication audits.
5. Run all validation windows.
6. Run cutoff sensitivity.
7. Score every method.
8. Apply gates and simplicity rule.
9. Write scoreboards and diagnostics.
10. Write model card, live inference example, and reproducibility manifest.

### Tests

```text
code/tests/test_hkg_tmax_probability_bucket_v1.py
```

Covers:

- Decimal bucket boundaries.
- No post-cutoff rows.
- Deterministic forecast tie-breaks.
- Forbidden target/raw-audit predictors.
- No sealed tuning.
- PMF row sums.
- RPS ordering behavior.
- Monotone CDF calibration.
- Stack weight constraints.
- Label-publication bucket-change application.
- No trading fields in live output.

### Result Folder

```text
experiments/hkg_tmax_probability_buckets_v1/results/
```

Generated file inventory:

| Artifact | Purpose |
|---|---|
| `scoreboard.csv` | Primary leaderboard sorted by normalized RPS with gates and champion flag. |
| `scoreboard_by_split.csv` | Per-split scores. |
| `scoreboard_by_month.csv` | Month-level scores. |
| `scoreboard_july.csv` | July-specific scoreboard. |
| `scoreboard_by_season.csv` | Seasonal scores. |
| `scoreboard_by_official_max_bin.csv` | Scores by official max bucket/bin. |
| `scoreboard_by_issue_hour.csv` | Scores by issue hour. |
| `scoreboard_by_revision_direction.csv` | Scores by forecast revision direction. |
| `scoreboard_by_cutoff.csv` | Cutoff sensitivity scores. |
| `calibration_layer_scoreboard.csv` | K0/K1/K2 calibration comparison. |
| `proper_score_deltas_bootstrap.csv` | Bootstrap deltas versus B4. |
| `modeling_table.parquet` | Full modeling table. |
| `modeling_table_schema.json` | Modeling table schema. |
| `modeling_table_with_first_publication_labels.parquet` | Modeling table plus first-publication label fields. |
| `selected_forecast_rows.parquet` | Selected forecast rows by target/cutoff. |
| `eligible_forecast_revision_rows.parquet` | All eligible pre-cutoff revisions. |
| `source_eligibility_audit.csv` | Source/filter row audit. |
| `row_count_audit.json` | Row counts and bucket boundary examples. |
| `row_counts_by_split_cutoff.csv` | Split/cutoff row counts. |
| `row_counts_by_month_cutoff.csv` | Month/cutoff row counts. |
| `leakage_audit.json` | Leakage audit. |
| `label_publication_audit.csv` | Row-level first-publication audit. |
| `label_publication_audit.json` | First-publication audit summary. |
| `row_identity_gate.json` | Identical-row-set gate. |
| `per_fold_predictions.parquet` | Per-method prediction rows. |
| `bucket_probabilities.parquet` | Wide bucket probabilities. |
| `one_decimal_pmfs.parquet` | Long-form bucket probability mass output. |
| `distribution_params.parquet` | Method/split parameter details. |
| `reliability_bins.csv` | Reliability/calibration bins. |
| `one_vs_rest_brier_by_bucket.csv` | Bucket-level Brier diagnostics. |
| `pit_values.csv` | PIT diagnostic values. |
| `interval_coverage.csv` | 90% interval coverage and width. |
| `sharpness_diagnostics.csv` | Entropy and max-probability diagnostics. |
| `stack_weights.csv` | S1 stack weights by split. |
| `model_selection_log.json` | Method settings and selection logs. |
| `final_probability_model_card.md` | Champion model card. |
| `live_inference_example_input.json` | Expected live inference input shape. |
| `live_inference_example_output.json` | Example weather-only probability output. |
| `live_inference_no_trading_audit.json` | Audit proving no trading fields in live output. |
| `reproducibility_manifest.json` | Artifact hashes and runtime info. |

## Testing And Verification Evidence

### Unit Tests

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest code\tests\test_hkg_tmax_probability_bucket_v1.py -q
```

Result:

```text
10 passed
```

What this proves:

- The hard rules and invariants work on synthetic controlled cases.
- Bucket boundaries are correct.
- Selection respects cutoffs.
- Tie-breaks are deterministic.
- PMFs sum to one.
- CDF calibration is monotone.
- Stack weights obey constraints.
- Live output remains probability-only.

What this does not prove:

- It does not prove full live DB correctness by itself.
- It does not prove future HKO schema drift will be handled.
- It does not prove a true EMOS model has been tested.

### Full Benchmark

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_probability_bucket_v1.py --config configs\hkg_tmax\probability_bucket_v1.yaml --output-dir experiments\hkg_tmax_probability_buckets_v1\results
```

Result:

```text
completed successfully
```

What this proves:

- PostgreSQL source tables were reachable.
- Modeling table was built.
- All benchmark methods ran.
- Scoreboards and diagnostics were generated.
- Audits passed.
- Reproducibility manifest was written.

Warnings observed:

- `pandas` warned that direct `psycopg` DBAPI connections are not its preferred SQLAlchemy path.
- scikit-learn emitted future deprecation warnings around `penalty='l2'`.

These warnings did not block the benchmark, but they should be cleaned up in a maintenance pass.

## Operational Runbook

### Reproduce Full V1

From repo root:

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_probability_bucket_v1.py --config configs\hkg_tmax\probability_bucket_v1.yaml --output-dir experiments\hkg_tmax_probability_buckets_v1\results
```

Expected output:

```text
HKG Tmax probability bucket V1 complete
Output: ...\experiments\hkg_tmax_probability_buckets_v1\results
```

### Inspect Champion

```powershell
Import-Csv experiments\hkg_tmax_probability_buckets_v1\results\scoreboard.csv |
  Where-Object {$_.champion_flag -eq 'True'} |
  Format-List
```

Expected champion:

```text
B4_hierarchical_residual_pmf
```

### Inspect Audits

```powershell
Get-Content experiments\hkg_tmax_probability_buckets_v1\results\leakage_audit.json
Get-Content experiments\hkg_tmax_probability_buckets_v1\results\label_publication_audit.json
Get-Content experiments\hkg_tmax_probability_buckets_v1\results\row_identity_gate.json
Get-Content experiments\hkg_tmax_probability_buckets_v1\results\live_inference_no_trading_audit.json
```

Expected:

```text
leakage_audit.status = pass
leakage_audit.total_violations = 0
label_publication_audit.status = ok
label_publication_audit.bucket_changes = 0
row_identity_gate.status = pass
live_inference_no_trading_audit.status = pass
```

### Inspect July

```powershell
Import-Csv experiments\hkg_tmax_probability_buckets_v1\results\scoreboard_july.csv |
  Sort-Object {[double]$_.rps} |
  Select-Object -First 10
```

Expected:

- S1 and B4 variants are very close.
- July is materially harder than full-year average.

## Safety, Security, And Trading Boundary

This engine reads local PostgreSQL data and writes local artifacts. It does not use secrets beyond the configured local database URL in the config. It does not call external APIs during the benchmark. It does not place orders, fetch order books, compute EV, compute Kelly, or create trade recommendations.

The live output audit explicitly rejects forbidden trading fields:

```text
market_price
market_probability
edge
ev
kelly
pnl
order_book
bid
ask
yes_price
no_price
trade
```

Current live output passed:

```text
forbidden_fields: []
probability_only: true
```

## Performance And Scalability Notes

The benchmark completed successfully but is not instant. The heaviest parts are:

- building selected forecast/revision tables across multiple cutoffs,
- B5 kernel analog prediction,
- repeated logistic/MOS models across splits and cutoff sensitivity,
- writing large Parquet/CSV diagnostics.

The result folder contains large artifacts:

```text
eligible_forecast_revision_rows.parquet: about 22.6 MB
one_decimal_pmfs.parquet: about 17.1 MB
pit_values.csv: about 15.6 MB
per_fold_predictions.parquet: about 12.7 MB
bucket_probabilities.parquet: about 10.9 MB
```

The B5 kernel analog model is a performance and robustness concern. It is more complex, slower, and failed NLL despite better RPS. It should not be promoted without further calibration and runtime review.

## Known Limitations and Follow-Up Work

### True EMOS Not Yet Benchmarked

Impact:

The current run cannot settle whether a proper EMOS/GAMLSS-style model would beat B4.

Reason:

Only `P1_normal_mos` and `P2_student_t_mos` were implemented as MOS-style residual distribution models.

Revisit trigger:

Run a dedicated EMOS/GAMLSS/quantile/CDF-calibrated experiment with predeclared temporal governance and no sealed tuning.

### July/High-Heat Calibration Is Harder

Impact:

July B4 RPS is `0.057741`, worse than full primary B4 RPS `0.041524`.

Reason:

Summer/high-heat regimes have tighter bucket competition and tail risk around `32`, `33`, and `34_or_higher`.

Revisit trigger:

Build a July/high-heat-specific calibration diagnostic and test whether no-harm seasonal calibration can improve tails without harming the full-year scoreboard.

### B5 Has Accuracy Potential But Calibration Risk

Impact:

B5 had the best RPS but failed NLL. It may be too sharp or unstable.

Reason:

Kernel analog weighting can over-concentrate probability on analog residuals that look good under RPS but are penalized by log loss when wrong.

Revisit trigger:

Test explicit B5 calibration, probability flooring, tail smoothing, and NLL-optimized bandwidth selection.

### Stack Gain Is Too Small

Impact:

S1 passed gates and slightly improved RPS, but the gain was not large enough to justify complexity.

Reason:

The fitted stack mostly assigned weight to B4, averaging `0.85`, and only small weights to other methods.

Revisit trigger:

Only revisit stack promotion if a new base method adds independent signal and produces at least the configured promotion improvement.

### Direct Classifiers Underperformed

Impact:

`C1_multinomial_ridge` and `C2_ordinal_cdf_logistic` were worse than residual PMF methods and failed NLL.

Reason:

The official forecast residual structure is probably easier to model as a residual distribution than as direct multiclass bucket classification with the current features.

Revisit trigger:

Try direct classifiers again only with stronger feature engineering, better regularization, or richer probabilistic calibration.

## Recommended Next Round

The next round should not blindly add complexity. It should target the specific failure modes from this run.

Recommended experiments:

1. Dedicated EMOS/GAMLSS benchmark:
   - model mean and scale separately,
   - compare Normal, Student-t, skewed distributions if feasible,
   - score by RPS/NLL/Brier/ECE,
   - use the same temporal governance.

2. B5 calibration repair:
   - keep analog structure,
   - test shrinkage to B4,
   - test log-loss-safe probability floors,
   - test temperature scaling/power calibration,
   - require NLL no-worse gate.

3. July/high-heat specialist calibration:
   - focus on July, JJA, forecast max bins `31`, `32`, `33`, `34_or_higher`,
   - require no-harm on full-year and presealed splits,
   - inspect reliability bins for high-temperature buckets.

4. First-publication robustness:
   - keep the first-publication audit in every future run,
   - if future bucket changes appear, produce canonical and first-publication scoreboards.

5. Operational live inference packaging:
   - package B4 model state into a stable artifact,
   - add a CLI to produce probabilities for a specified target date,
   - keep output probability-only until a separate trading-stage spec is approved.

## Reviewer Checklist

Before using this engine as the current reference distribution layer, verify:

```text
[x] Bucket rules are Decimal-safe and include 31.9 -> 31.
[x] Primary cutoff is T-1 23:59 HKT.
[x] Sensitivity cutoffs 18:00, 21:00, 23:59 HKT were run.
[x] Development folds and presealed holdout were evaluated.
[x] Sealed 2024-2026 confirmation was run but not used for tuning.
[x] B4 is marked champion under the simplicity gate.
[x] B5 is not promoted because it failed NLL.
[x] S1 is not promoted because its gain was below threshold.
[x] MOS-style Normal/Student-t models were run and lost.
[x] True dedicated EMOS was not claimed as completed.
[x] Leakage audit passed with 0 violations.
[x] Row identity gate passed with 0 violations.
[x] Label-publication audit found 0 bucket changes.
[x] Live inference no-trading audit passed.
[x] Result artifacts and reproducibility manifest exist.
```

## Bottom Line

The current best probability distribution engine is:

```text
B4_hierarchical_residual_pmf
```

It should be treated as the current reference/champion system for converting the latest eligible official HKO point forecast into full HKG Tmax bucket probabilities.

It is strong because it is leakage-safe, calibrated decently, simple, interpretable, and much better than climatology. It is not final because the improvement over a simpler forecast-anchored residual PMF is modest, July remains harder, and a true dedicated EMOS/GAMLSS-style probability experiment has not yet been completed.
