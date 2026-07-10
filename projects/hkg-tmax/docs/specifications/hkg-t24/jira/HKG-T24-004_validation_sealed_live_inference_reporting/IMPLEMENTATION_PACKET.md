# HKG-T24-004 Implementation Packet

> **Relocation note (2026-07-10):** Preserve the historical path text below,
> but apply the current path mapping in [the Jira index](../README.md).

## Read-First Requirement

Before implementing this Jira, Codex must fully read:

```text
HKG-T24-004_Validation_Sealed_Holdout_Live_Inference_Reporting.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC_FINAL_CLARIFICATIONS.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_FINAL_CONSISTENCY_PATCH.md
```

Reading only `HKG-T24-004_Validation_Sealed_Holdout_Live_Inference_Reporting.md` is not enough.

## Repository Implementation Location

In this repository, the contract path `src/hkg_t24` resolves to `code/src/hkg_t24`.

All implementation code for this Jira must live under `code/src/hkg_t24/`. Supporting tests, config, SQL, schemas, reports, and durable artifact metadata must use `code/tests/hkg_t24/`, `config/hkg_t24/`, `sql/hkg_t24/`, `schemas/hkg_t24/`, `reports/hkg_t24/`, and `artifacts/hkg_t24/`.

Do not put implementation logic in this Jira folder, root files, reports, notebooks, or ad hoc scripts. Scripts may call the package, but the package owns the implementation logic.

## Binding Precedence

```text
1. Final consistency patch
2. Final clarifications addendum
3. Completion specification
4. Original blueprint
5. This Jira packet
```

Sealed validation, freeze, live inference, and final artifact behavior must follow the final consistency patch exactly.

## Especially Binding Contract Sections

The full docs are binding, but these sections are especially relevant:

```text
Blueprint:
- 15. Validation and scoreboards
- 16. Implementation sequence, Phases 7-9
- 19. Promotion ladder
- 21. What success looks like
- 22. Daily operational checklist
- 23. Final implementation command summary

Completion Spec:
- 18. Metrics and scoreboards
- 19. Negative controls and leakage tests
- 20. Sealed validation protocol
- 21. Model promotion ladder
- 23. Live inference
- 24. Required tests
- 25. Required final artifacts
- 26. Coding priorities and constraints
- 27. Final implementation summary

Final Clarifications:
- 12. Sealed validation and shadow sources
- 13. Test expectations
- 14. Final Codex Implementation Readiness

Final Consistency Patch:
- 12. Sealed validation command naming
- 13. Freeze command naming
- 14. Final report and artifact naming consistency
- 19. Router command and artifact consistency
- 20. Distributional edge cases and threshold output keys
- 21. Sealed validation and shadow-source protocol
- 22. Final commands affected by this patch
```

## High-Risk Details To Carry From The Contract

This Jira must explicitly preserve these details from the binding docs:

```text
run-full-pre2024 must execute the canonical final-patch command sequence and stop at first failure.
Freeze commands must use freeze-candidate --stage pre2024 and freeze-candidate --stage refit_through_2024.
Sealed commands must use sealed-score --year 2024 and sealed-score --year 2025.
Compatibility aliases may exist, but CI/docs must use canonical commands only.
2024 scoring must score the frozen pre-2024 strict candidate before any adapter training.
2024 shadow scoring must not alter the frozen strict candidate.
train-adapters --through-year 2024 may run only after the 2024 strict pass condition.
IFS/AI adapter entry must satisfy the exact 250 labelled rows, MAE improvement, P90, negative-control, and cap conditions.
2025 is final test only; after seeing 2025, no tuning of features, thresholds, hyperparameters, caps, specialist thresholds, adapter gates, or calibration is allowed.
2026 prospective rows count as live only when the prediction existed before settlement.
Live-predict must refuse after cutoff unless explicit replay mode is used.
score-live must update online states only after settlement label exists.
The full final artifact list from the final consistency patch must be produced under canonical names.
Synthetic fixture generator and real DB smoke behavior from final clarifications must be implemented.
Real DB tests must skip with SKIPPED_REAL_DB_NO_DATABASE_URL when HKG_TMAX_DATABASE_URL is absent.
```

## Required Completion Evidence

At the end of this Jira, produce:

```text
reports/jira_004_contract_coverage.md
```

Use `CONTRACT_COVERAGE_TEMPLATE.md` as the required structure.

The report must prove:

```text
all final commands exist and use canonical names;
negative controls and leakage tests pass;
freeze manifests contain required hashes;
sealed guards prevent contamination;
live/replay and post-settlement commands are point-in-time safe;
all final artifacts and reports exist under canonical names;
READY_FOR_SEALED_VALIDATION status is justified.
```

## Done Condition

This Jira is complete only when the full pre-2024 pipeline can run, freeze, validate, guard sealed years, support live/replay prediction, score settlement safely, and map all final artifacts in `reports/jira_004_contract_coverage.md`.
