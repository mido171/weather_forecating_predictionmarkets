# Jira 004 Contract Coverage Report Template

Output path:

```text
reports/jira_004_contract_coverage.md
```

## Read Confirmation

Confirm that these were read in full:

```text
HKG-T24-004_Validation_Sealed_Holdout_Live_Inference_Reporting.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC_FINAL_CLARIFICATIONS.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_FINAL_CONSISTENCY_PATCH.md
```

## Command Coverage

| Command | Contract source | Implemented? | Test/artifact | Status |
|---|---|---|---|---|
| run-full-pre2024 |  |  |  |  |
| run-negative-controls --scope strict-pre2024 |  |  |  |  |
| freeze-candidate --stage pre2024 |  |  |  |  |
| sealed-score --year 2024 |  |  |  |  |
| train-adapters --through-year 2024 |  |  |  |  |
| freeze-candidate --stage refit_through_2024 |  |  |  |  |
| sealed-score --year 2025 |  |  |  |  |
| live-predict --target-date YYYY-MM-DD --cutoff-id H24N |  |  |  |  |
| score-live --target-date YYYY-MM-DD |  |  |  |  |

## Validation Coverage

| Validation/guard | Implemented in | Test/artifact | Status |
|---|---|---|---|
| Shuffled target control |  |  |  |
| Lag-shifted NWP control |  |  |  |
| Post-cutoff injection |  |  |  |
| Outcome-derived feature scan |  |  |  |
| Future-normalization scan |  |  |  |
| Same-row residual scan |  |  |  |
| GribStream scope contamination check |  |  |  |
| H24N NWP safety check |  |  |  |
| Sealed-year target access check |  |  |  |

## Artifact Coverage

| Required final artifact | Produced? | Hash/provenance recorded? | Status |
|---|---|---|---|
|  |  |  |  |

## Sealed And Live Integrity

| Requirement | Test/artifact | Status |
|---|---|---|
| 2024 opens only after pre-2024 freeze |  |  |
| 2024 scoring does not train adapters |  |  |
| Adapter training requires 2024 pass |  |  |
| 2025 is report-only final test |  |  |
| Live prediction refuses after cutoff |  |  |
| Score-live waits for settlement label |  |  |
| 2026 prospective/replay separation |  |  |

## Final Statement

State whether Jira 004 fully satisfies its contract scope with no known omissions.
