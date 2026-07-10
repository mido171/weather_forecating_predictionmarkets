# HKG-T24-001 Implementation Packet

> **Relocation note (2026-07-10):** Preserve the historical path text below,
> but apply the current path mapping in [the Jira index](../README.md).

## Read-First Requirement

Before implementing this Jira, Codex must fully read:

```text
HKG-T24-001_Data_Contract_Snapshot_Feature_Store_Foundation.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC_FINAL_CLARIFICATIONS.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_FINAL_CONSISTENCY_PATCH.md
```

Reading only `HKG-T24-001_Data_Contract_Snapshot_Feature_Store_Foundation.md` is not enough.

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

If `HKG-T24-001_Data_Contract_Snapshot_Feature_Store_Foundation.md` omits a detail that exists in the binding docs, the binding docs still apply. If any contradiction remains, fail closed and write the contradiction into the Jira coverage report.

## Especially Binding Contract Sections

The full docs are binding, but these sections are especially relevant:

```text
Blueprint:
- 1. Non-negotiable forecast contract
- 3. Current data inventory and how each dataset is treated
- 4. Canonical H24N snapshot
- 16. Implementation sequence, Phases 0-1
- 17. Database design
- 18. Hard denylist

Completion Spec:
- 0. Global directives
- 1. Definition of full implementation complete
- 2. Exact implementation order, Phases 0-3
- 3. Repository and code layout
- 4. Existing source-of-truth tables and schema contracts
- 5. New database schemas and tables
- 6. H24N snapshot builder
- 7. Official HKO forecast anchor
- 9. GribStream/NWP dataset treatment
- 19. Negative controls and leakage tests
- 24. Required tests
- 26. Coding priorities and constraints

Final Clarifications:
- 5. Schema consistency fixes and final DDL changes
- 6. Canonical feature matrix schema
- 7. Static geospatial and station dossier behavior
- 13. Test expectations

Final Consistency Patch:
- 0. Binding precedence and implementation rule
- 2. Feature matrix table naming conflict
- 3. Database environment variable conflict
- 4. Undefined live component table
- 5. Calendar fields
- 6. model_core.source_registry final schema and required rows
- 7. GribStream source names and feature prefixes
- 9. Canonical date field naming
- 10. LightGBM requirement
- 12. Sealed validation command naming
- 13. Freeze command naming
- 14. Final report and artifact naming consistency
- 15. Final consistency for strict/proxy/shadow feature dictionaries
- 16. Source feature prefixes and exact strict/proxy/shadow handling
- 22. Final commands affected by this patch
```

## High-Risk Details To Carry From The Contract

This Jira must explicitly preserve these details from the binding docs:

```text
HKG_TMAX_DATABASE_URL wins over HKG_TMAX_DB_DSN.
Missing DSN error text must be exact.
target_date_hkt is canonical in new model schemas.
model_features.feature_matrix is the only physical final feature matrix table.
snapshot_feature_matrix_strict and snapshot_feature_matrix_proxy are compatibility views only.
model_live.prediction, model_live.live_prediction_component, and model_eval.system_prediction_component use final patch DDL.
model_core.source_registry must use the final dedicated boolean status columns.
Every source_registry required row from the final patch must be populated exactly.
GribStream safe-row access must join raw_response_object and filter full_tactical_backfill_ok_tmax.
H24N safe predicate must enforce run_time_utc + 6 hours <= formal_cutoff_utc.
Blocked/support-only sources must never become strict Tmax sources.
The 6-hour GribStream buffer is a conservative project guardrail, not a provider guarantee.
model_features.feature_availability_matrix remains required unless the final patch explicitly replaces it.
Source/data reports must use the final artifact names from the final consistency patch.
```

## Required Completion Evidence

At the end of this Jira, produce:

```text
reports/jira_001_contract_coverage.md
```

Use `CONTRACT_COVERAGE_TEMPLATE.md` as the required structure.

The report must prove:

```text
every required schema/table/view exists or is intentionally superseded by the final patch;
all source contracts were checked against the real DB;
every H24N cutoff calculation is point-in-time correct;
every strict GribStream row passed source-scope and temporal filters;
all blocked/proxy/shadow sources are excluded from strict scope;
all foundation artifacts and reports exist under canonical names.
```

## Done Condition

This Jira is complete only when Codex can show a passing preflight/source/snapshot foundation and a completed `reports/jira_001_contract_coverage.md` with no unresolved omissions.
