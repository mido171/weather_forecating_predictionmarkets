# HKG-T24-004 — Validation, Sealed Holdout, Live Inference, Post-Settlement Scoring, and Final Reporting

## Repository Implementation Location

In this repository, the contract path `src/hkg_t24` resolves to `code/src/hkg_t24`.

All implementation code for this Jira must live under `code/src/hkg_t24/`.

Supporting files must use:

```text
code/tests/hkg_t24/          tests
config/hkg_t24/              configuration
sql/hkg_t24/                 reviewed SQL/query assets
migrations/postgres/         durable PostgreSQL migrations
schemas/hkg_t24/             machine-readable schemas
reports/hkg_t24/             report indexes and non-canonical reports
artifacts/hkg_t24/           artifact indexes and small durable metadata
```

Do not put implementation logic in this Jira folder, root files, reports, notebooks, or ad hoc scripts. Scripts may call the package, but the package owns the implementation logic.

## Objective

Implement the complete end-to-end orchestration and validation layer. This ticket runs the full pre-2024 pipeline, executes negative controls, freezes the pre-2024 candidate, implements guarded sealed validation commands, implements 2024 adapter training only after 2024 strict pass, implements 2025 final-test guardrails, implements 2026 prospective/live replay, implements live prediction and post-settlement scoring, writes final reports/artifacts, and verifies the entire system reaches `READY_FOR_SEALED_VALIDATION` without using sealed labels by default.

## Full Detailed Scope

Implement canonical commands:

```bash
python -m hkg_t24.cli phase0-preflight
python -m hkg_t24.cli build-source-registry
python -m hkg_t24.cli build-h24n-snapshots
python -m hkg_t24.cli build-features --scope strict
python -m hkg_t24.cli build-features --scope proxy
python -m hkg_t24.cli build-features --scope live_shadow
python -m hkg_t24.cli train-experts --scope strict-pre2024
python -m hkg_t24.cli generate-oof --scope strict-pre2024
python -m hkg_t24.cli train-router --router R0
python -m hkg_t24.cli train-router --router R1
python -m hkg_t24.cli train-specialists --scope strict-pre2024
python -m hkg_t24.cli train-distribution --scope strict-pre2024
python -m hkg_t24.cli run-system-replay --scope strict-pre2024
python -m hkg_t24.cli run-negative-controls --scope strict-pre2024
python -m hkg_t24.cli freeze-candidate --stage pre2024
python -m hkg_t24.cli sealed-score --year 2024
python -m hkg_t24.cli train-adapters --through-year 2024
python -m hkg_t24.cli freeze-candidate --stage refit_through_2024
python -m hkg_t24.cli sealed-score --year 2025
python -m hkg_t24.cli live-predict --target-date YYYY-MM-DD --cutoff-id H24N
python -m hkg_t24.cli score-live --target-date YYYY-MM-DD
```

Compatibility aliases may exist, but CI and documentation must use only the canonical commands.

## Explicit Out of Scope

This ticket does not redesign features, models, router thresholds, specialist thresholds, distribution thresholds, or source status.

This ticket does not automatically open 2024 or 2025 labels during normal full implementation.

This ticket does not implement a scheduler. Live inference is a command only.

This ticket does not implement bankroll, order execution, or Polymarket trading automation.

## Required Implementation Steps

1. Implement modules:

```text
src/hkg_t24/validation/scoreboard.py
src/hkg_t24/validation/negative_controls.py
src/hkg_t24/validation/leakage_tests.py
src/hkg_t24/validation/sealed.py
src/hkg_t24/validation/reports.py
src/hkg_t24/artifacts/freeze.py
src/hkg_t24/live/inference.py
src/hkg_t24/live/post_settlement.py
src/hkg_t24/live/replay.py
src/hkg_t24/orchestration/run_full.py
```

2. Implement command:

```bash
python -m hkg_t24.cli run-full-pre2024
```

It must execute all pre-2024 stages in order and stop at first failure.

3. Implement `run-negative-controls --scope strict-pre2024`.

4. Implement mandatory negative controls:

```text
shuffled target control
lag-shifted NWP control
post-cutoff injection test
outcome-derived feature scan
future-normalization scan
same-row residual flag scan
GribStream source-scope contamination check
H24N NWP safety check
sealed-year target access check
```

5. Implement metric formulas:

```text
error = prediction - actual
absolute_error = abs(error)
MAE = mean(abs(error))
RMSE = sqrt(mean(error^2))
Bias = mean(error)
MedianAE = median(abs(error))
P75AE = percentile75(abs(error))
P90AE = percentile90(abs(error))
P95AE = percentile95(abs(error))
LargeError1Rate = mean(abs(error) >= 1.0)
LargeError2Rate = mean(abs(error) >= 2.0)
```

6. Implement required score slices:

```text
full period
year
month
season
MAM
JJA
source availability group
official max bucket <20, 20-25, 25-30, 30-33, >=33
GEFS spread tertile
marine specialist active/inactive
weak-wind specialist active/inactive
cloud/rain specialist active/inactive
high-error-tail specialist active/inactive
```

7. Implement identical-row comparison enforcement.

8. Implement `freeze-candidate --stage pre2024`.

9. The pre-2024 frozen manifest must include:

```text
git commit
source registry hash
feature schema hash
feature list
training row IDs hash
model artifacts
router artifacts
specialist artifacts
distribution artifacts
negative-control report hash
scoreboard hash
random seed
source row counts
configuration hash
```

10. Implement `sealed-score --year 2024`.

It must:

```text
require explicit sealed release authorization if configured
score frozen pre-2024 strict candidate
score frozen baselines
score shadow experts diagnostically
write sealed reports
not train adapters
not modify frozen candidate
```

11. Implement 2024 pass condition:

```text
strict candidate MAE <= E0 MAE - 0.01°C on identical 2024 rows
P90 absolute error worsening <= 0.03°C vs E0
negative controls already passed
no sealed-access violation
```

12. If 2024 fails, stop and do not allow adapter training or 2025 opening.

13. Implement `train-adapters --through-year 2024`.

It can run only after:

```text
sealed_2024_report exists
pre2024 strict candidate was scored
strict candidate was not modified
2024 pass condition passed
```

14. Adapter training candidates:

```text
ifsoper
ifsenfo
aifsoper
aifsenfo
graphcast
fourcastnet
```

15. Adapter entry condition for refit-through-2024 candidate:

```text
>=250 settled labelled rows
adapter improves 2024 shadow MAE vs pre-2024 strict candidate by >=0.015°C
adapter does not worsen P90 AE by >0.020°C
adapter negative controls pass
adapter max router cap <=0.10
```

16. ARWF and CWA adapters are not allowed unless prospective live history gate is met.

17. Implement `freeze-candidate --stage refit_through_2024`.

18. Implement `sealed-score --year 2025`.

It must score only. After 2025 is seen, it must forbid tuning of:

```text
features
thresholds
hyperparameters
router caps
specialist thresholds
adapter gates
calibration parameters
```

19. Implement 2026 prospective/live replay rules:

```text
predictions made before settlement -> prospective performance
predictions not made before settlement -> historical replay only
```

20. Implement `live-predict --target-date YYYY-MM-DD --cutoff-id H24N`.

It must refuse to create live prediction after formal cutoff unless replay mode is explicitly used.

21. Implement `score-live --target-date YYYY-MM-DD`.

It may join target label only after settlement label exists.

22. Implement online residual-state update after settlement.

23. Implement final reporting.

24. Implement full tests and real DB smoke.

## Required Database Schemas / Tables / Views / Materializations

Use existing tables from prior Jiras and ensure final population of:

```text
model_validation.scoreboard
model_validation.negative_control_result
model_validation.leakage_audit_event
model_live.prediction
model_live.live_prediction_component
model_eval.system_prediction_component
model_oof.system_prediction
```

Create sealed tracking table if absent:

```sql
CREATE TABLE IF NOT EXISTS model_validation.sealed_run (
  sealed_run_id text PRIMARY KEY,
  year integer NOT NULL,
  stage text NOT NULL,
  candidate_manifest_uri text NOT NULL,
  opened_at_utc timestamptz NOT NULL DEFAULT now(),
  opened_by text NOT NULL,
  sealed_release_token_hash text NOT NULL,
  status text NOT NULL,
  report_uri text NOT NULL,
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (year, stage, candidate_manifest_uri)
);
```

## Required CLI Commands / Scripts / Modules

Commands:

```bash
python -m hkg_t24.cli run-full-pre2024
python -m hkg_t24.cli run-negative-controls --scope strict-pre2024
python -m hkg_t24.cli freeze-candidate --stage pre2024
python -m hkg_t24.cli sealed-score --year 2024
python -m hkg_t24.cli train-adapters --through-year 2024
python -m hkg_t24.cli freeze-candidate --stage refit_through_2024
python -m hkg_t24.cli sealed-score --year 2025
python -m hkg_t24.cli live-predict --target-date YYYY-MM-DD --cutoff-id H24N
python -m hkg_t24.cli score-live --target-date YYYY-MM-DD
```

Modules:

```text
src/hkg_t24/validation/scoreboard.py
src/hkg_t24/validation/negative_controls.py
src/hkg_t24/validation/leakage_tests.py
src/hkg_t24/validation/sealed.py
src/hkg_t24/validation/reports.py
src/hkg_t24/artifacts/freeze.py
src/hkg_t24/live/inference.py
src/hkg_t24/live/post_settlement.py
src/hkg_t24/live/replay.py
src/hkg_t24/orchestration/run_full.py
```

## Required Feature / Model / Artifact Outputs

Source/data readiness:

```text
reports/source_inventory_report.md
reports/source_registry.csv
reports/schema_migration_source_registry.md
reports/schema_migration_feature_matrix.md
reports/gribstream_source_scope_audit.csv
reports/gribstream_source_scope_audit.md
reports/leakage_audit_report.md
reports/snapshot_coverage_report.csv
reports/snapshot_coverage_report.md
reports/live_shadow_availability_report.csv
reports/live_shadow_availability_report.md
```

Feature documentation:

```text
reports/feature_dictionary_strict.csv
reports/feature_dictionary_proxy.csv
reports/feature_dictionary_shadow.csv
reports/feature_dictionary.md
reports/feature_availability_matrix.csv
reports/feature_availability_matrix.md
reports/feature_null_rate_report.csv
reports/feature_schema_validation_report.md
```

OOF and expert outputs:

```text
reports/oof_integrity_report.md
reports/expert_scoreboard_strict.csv
reports/expert_scoreboard_proxy.csv
reports/expert_scoreboard_shadow.csv
reports/expert_fold_metrics.csv
reports/expert_promotion_decisions.csv
```

Router/specialist outputs:

```text
reports/router_scoreboard.csv
reports/router_weight_diagnostics.csv
reports/router_promotion_decisions.csv
reports/specialist_scoreboard.csv
reports/specialist_activation_report.csv
reports/specialist_no_harm_report.csv
reports/specialist_promotion_decisions.csv
```

Distribution outputs:

```text
reports/distribution_scoreboard.csv
reports/distribution_calibration_report.csv
reports/distribution_calibration_report.md
reports/calibration_report.md
reports/threshold_probability_scoreboard.csv
reports/prediction_interval_coverage_report.csv
```

Full-system outputs:

```text
reports/system_scoreboard_strict.csv
reports/system_scoreboard_proxy.csv
reports/system_ablation_matrix.csv
reports/negative_control_report.md
reports/frozen_candidate_manifest_pre2024.json
reports/frozen_candidate_manifest_refit_through_2024.json
reports/final_candidate_manifest.json
reports/ready_for_sealed_validation.md
```

Sealed validation outputs as command outputs, not default-run artifacts:

```text
reports/sealed_2024_scoreboard.csv
reports/sealed_2024_report.md
reports/sealed_2024_shadow_expert_scoreboard.csv
reports/sealed_2025_scoreboard.csv
reports/sealed_2025_report.md
reports/2026_prospective_replay_scoreboard.csv
reports/2026_prospective_replay_report.md
```

## Required Provenance / Audit / Logging Behavior

Every live prediction must write:

```text
model_live.prediction
model_live.live_prediction_component
audit_sha256
source_availability_jsonb
component_jsonb
```

Every sealed run must write:

```text
model_validation.sealed_run
sealed scoreboard
sealed report
sealed shadow expert scoreboard where applicable
```

Every negative control must write:

```text
model_validation.negative_control_result
reports/negative_control_report.md
```

Every leakage test must write:

```text
model_validation.leakage_audit_event
reports/leakage_audit_report.md
```

## Required Fail-Closed / Error Behavior

Fail closed when:

```text
run-full-pre2024 detects any upstream phase failure
negative controls fail
leakage audit has ERROR events
freeze requested before required reports exist
freeze manifest hash cannot be computed
sealed-score requested before pre2024 candidate is frozen
sealed-score lacks required sealed opening authorization when configured
2024 fails but train-adapters is invoked
2025 opened before 2024 pass and refit-through-2024 freeze
live-predict invoked after cutoff without replay mode
score-live invoked before settlement label exists
live prediction would use post-cutoff data
```

Report-only, not failure:

```text
shadow expert unavailable
ARWF absent
CWA absent
proxy features unavailable
```

## Leakage-Free / Non-Forward-Looking Requirements

This Jira must guarantee:

- no target-day observations in validation, live, or replay inference;
- no post-cutoff data in live-predict;
- no future labels in pre-2024 development;
- no target-derived or outcome-derived features;
- no sealed-year labels for development/tuning before explicit sealed command;
- no global normalization fit using future rows;
- no train/test contamination in OOF, router, specialist, distribution, adapters, or sealed scoring;
- no same-row residual leakage;
- no GribStream rows outside `full_tactical_backfill_ok_tmax` and H24N-safe filter;
- no blocked/proxy/shadow sources in strict scoreboards;
- fold-local preprocessing only;
- fold-local model fitting only;
- full provenance evidence for every prediction and score.

Sealed handling:

```text
2024 labels may be opened only after pre2024 freeze.
2024 shadow scoring does not modify frozen candidate.
2024 adapter training may occur only after strict 2024 pass.
2025 is final test and cannot trigger tuning.
2026 prospective rows count as live only when prediction preceded settlement.
```

## Dependencies on Earlier Jiras

Depends on HKG-T24-001, HKG-T24-002, and HKG-T24-003.

This ticket cannot run end-to-end until source foundation, feature matrices, expert OOF predictions, routers, specialists, and distributional system exist.

## Acceptance Criteria

1. `run-full-pre2024` executes all required pre-2024 commands in canonical order.
2. Pipeline stops at first failure.
3. Strict and proxy system scoreboards are produced.
4. Negative-control report exists and passes.
5. Leakage audit report exists with zero strict `ERROR` events.
6. Frozen pre-2024 manifest exists and has complete hashes.
7. `reports/final_candidate_manifest.json` exists.
8. `reports/ready_for_sealed_validation.md` exists with status `READY_FOR_SEALED_VALIDATION`.
9. `sealed-score --year 2024` refuses to run without authorization when authorization is configured.
10. `sealed-score --year 2024` scores strict frozen candidate and shadow experts without training adapters.
11. `train-adapters --through-year 2024` refuses unless 2024 strict pass exists.
12. `sealed-score --year 2025` refuses unless refit-through-2024 candidate exists.
13. `live-predict` writes prediction and component rows before cutoff.
14. `live-predict` refuses after cutoff unless replay mode.
15. `score-live` updates online states only after settlement label exists.
16. No report uses non-identical-row comparisons without marking them as such.
17. Final reports list strict/proxy/shadow/live scopes separately.

## Extensive Test Scenarios

Unit tests:

```text
tests/unit/test_metrics.py
tests/unit/test_sealed_command_guards.py
tests/unit/test_live_cutoff_guard.py
tests/unit/test_manifest_hashing.py
tests/unit/test_threshold_probability_keys.py
```

Integration tests:

```text
tests/integration/test_negative_controls_synthetic.py
tests/integration/test_sealed_guard.py
tests/integration/test_freeze_manifest_realdb.py
tests/integration/test_live_prediction_replay_synthetic.py
tests/integration/test_post_settlement_update_synthetic.py
tests/integration/test_realdb_smoke_h24n.py
```

End-to-end smoke:

```text
tests/smoke/test_full_pipeline_smoke.py
```

## Required Smoke Tests

Run:

```bash
python -m hkg_t24.cli run-full-pre2024 --smoke --from-date 2021-04-14 --to-date 2021-05-31
```

Expected minimum counts:

```text
snapshots >= 45
official anchors >= 45
target labels >= 45
gfs feature rows >= 40
gefs feature rows >= 40
feature matrix rows >= 40
E0/E2/E4/E5 predictions >= 30
router predictions >= 30
system predictions >= 30
negative controls complete in reduced mode
frozen smoke candidate manifest written
zero sealed label reads
```

## Required Integration Tests

Integration tests must prove:

- `run-full-pre2024` stops on intentional upstream failure;
- negative controls fail suspicious leaky models;
- freeze manifest includes all required hashes;
- sealed-score rejects missing authorization when configured;
- sealed-score 2024 cannot train adapters;
- adapter training requires 2024 pass;
- sealed-score 2025 cannot run before refit-through-2024 freeze;
- live-predict writes `model_live.prediction` and component rows;
- score-live refuses before label exists;
- score-live updates online residual states only after settlement.

## Leakage and Temporal Integrity Tests

Required tests:

```text
Shuffled target control: shuffled labels must not beat official raw by more than 0.02°C.
Lag-shifted NWP control: shifted NWP system must not improve official raw by more than 0.02°C.
Post-cutoff injection test: future target column is rejected before training.
Outcome-derived feature scan: forbidden feature patterns are absent from input matrices.
Future-normalization scan: preprocessor_fit_end_date < first_test_date for every fold.
Same-row residual scan: residual/error/overforecast/underforecast fields absent from features.
GribStream scope contamination check: strict NWP rows all use full_tactical_backfill_ok_tmax.
H24N NWP safety check: strict NWP rows all satisfy run_time + 6h <= formal cutoff.
Sealed-year target access check: no 2024+ labels read before sealed command.
Live cutoff test: live-predict cannot use data after operational freeze.
```

## Required Negative-Control Tests Where Relevant

Run all negative controls at full pre-2024 scope and produce:

```text
reports/negative_control_report.md
model_validation.negative_control_result
```

Pass conditions:

```text
shuffled target MAE does not pass promotion
lag-shifted NWP does not pass promotion
post-cutoff injection is rejected
forbidden outcome feature scan passes
future-normalization scan passes
same-row residual scan passes
source-scope contamination check passes
sealed label access check passes
```

## Required Final Artifacts / Reports

```text
reports/negative_control_report.md
reports/leakage_audit_report.md
reports/system_scoreboard_strict.csv
reports/system_scoreboard_proxy.csv
reports/system_ablation_matrix.csv
reports/frozen_candidate_manifest_pre2024.json
reports/frozen_candidate_manifest_refit_through_2024.json
reports/final_candidate_manifest.json
reports/ready_for_sealed_validation.md
reports/sealed_2024_scoreboard.csv
reports/sealed_2024_report.md
reports/sealed_2024_shadow_expert_scoreboard.csv
reports/sealed_2025_scoreboard.csv
reports/sealed_2025_report.md
reports/2026_prospective_replay_scoreboard.csv
reports/2026_prospective_replay_report.md
```

Sealed outputs are required as command outputs, not as default-run artifacts. Default first implementation must not open sealed labels.

## Definition of Done

This Jira is done when the full pre-2024 system runs end to end, all negative controls and leakage tests pass, the frozen pre-2024 candidate manifest is produced, sealed commands are implemented and guarded, live/replay commands are implemented and tested, final reports exist, and the system status is `READY_FOR_SEALED_VALIDATION`.

## Cross-Ticket Implementation Order

Codex must execute the four Jiras sequentially:

```text
1. HKG-T24-001
   Build the data contract, final schemas, source registry, H24N snapshots, GribStream safe-row foundation, and feature-store foundation.

2. HKG-T24-002
   Build all strict/proxy/shadow features, online residual states, feature matrices, feature dictionaries, expert models, shadow placeholders, and OOF predictions.

3. HKG-T24-003
   Train routers, specialists, distributional layer, final formula, and system replay scoreboards.

4. HKG-T24-004
   Run full validation, negative controls, freeze candidate, implement sealed guards, live/replay inference, post-settlement scoring, final reporting, and ready-for-sealed-validation status.
```

## Final Explicit Statement

YES — these 4 Jira tickets fully cover the entire uploaded implementation contract with no known omissions and with explicit leakage-free, non-forward-looking implementation guarantees.
