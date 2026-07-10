# HKG-T24-002 Implementation Packet

> **Relocation note (2026-07-10):** Preserve the historical path text below,
> but apply the current path mapping in [the Jira index](../README.md).

## Read-First Requirement

Before implementing this Jira, Codex must fully read:

```text
HKG-T24-002_Feature_Engineering_Expert_Model_Factory.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC_FINAL_CLARIFICATIONS.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_FINAL_CONSISTENCY_PATCH.md
```

Reading only `HKG-T24-002_Feature_Engineering_Expert_Model_Factory.md` is not enough.

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

If `HKG-T24-002_Feature_Engineering_Expert_Model_Factory.md` summarizes a feature family but the contract gives exact formulas, the exact contract formulas are binding.

## Especially Binding Contract Sections

The full docs are binding, but these sections are especially relevant:

```text
Blueprint:
- 5. Feature families
- 6. Expert models
- 7. Out-of-fold expert generation
- 10. Concrete decision questions and how training answers them
- 11. Training date ranges and partitions
- 12. Model classes and training discipline
- 16. Implementation sequence, Phases 2-3

Completion Spec:
- 8. Target label and target-memory features
- 9. GribStream/NWP dataset treatment
- 10. NWP feature formulas
- 11. Combining datasets with different availability ranges
- 12. Station and proxy data policy
- 13. Expert models
- 14. OOF generation mechanics
- 24. Required tests
- 25. Required final artifacts
- 26. Coding priorities and constraints

Final Clarifications:
- 1. Online residual-memory features
- 3. Shadow expert prediction behavior
- 4. Inner validation and model selection
- 6. Canonical feature matrix schema
- 7. Static geospatial and station dossier behavior
- 8. Diagnostic proxy behavior
- 9. Official residual model and fallback behavior
- 13. Test expectations

Final Consistency Patch:
- 1. Target-memory lag naming contradiction
- 8. Features referenced by specialists/router
- 10. LightGBM requirement
- 11. ARWF and CWA live-shadow behavior
- 14. Final report and artifact naming consistency
- 15. Final consistency for strict/proxy/shadow feature dictionaries
- 16. Source feature prefixes and exact strict/proxy/shadow handling
- 17. Final handling of feature names in specialists and routers
- 18. Final rules for source absence, demotion, and fallback
```

## High-Risk Details To Carry From The Contract

This Jira must explicitly preserve these details from the binding docs:

```text
Target-memory finalized label features must use lag2 or older; no finalized target lag1 feature is allowed.
Every target-memory feature except target__year_index must have an is_missing companion.
target__clim30_* and target__warming_trend_10y_c_per_year must use the causal formulas from the final patch.
Online residual-memory must expose every exact online__{source_key}__{state_scope} feature from the clarifications, including volatility, state_available, capped correction, expected_abs_error, and streak fields.
Online state for target date T must use only dates before T.
NWP features must include the exact spatial aggregate features, deterministic 12-location formulas, non-temperature formulas, precipitation reset handling, DSWRF/SSRD radiation handling, and unit-semantic gates from the completion spec.
GEFS/IFS ensemble handling must include expected member counts, member0 handling for ifsenfo, quantiles, spread, std, and probability features exactly as specified.
Station proxy features must use only allowed fields, must exclude wind direction, must implement deterministic station groups, and must enforce minimum station support.
Diagnostic climate features must use the exact allowed variables, T-2 safe lag rule, long-to-wide formulas, and Trace handling.
IGRA and TC must produce diagnostic reports only and must not enter strict or proxy model matrices.
Every fitted expert/router model must write artifacts/models/{expert_or_router_id}/{fold_id}/model_selection.json using the exact schema from the clarifications.
LightGBM is mandatory in the first full implementation; no HistGradientBoostingRegressor fallback is allowed for required LightGBM models.
LightGBM early stopping/default settings and the completion-spec hyperparameter grids must be implemented exactly where applicable.
Synthetic fixture generator tests/fixtures/synthetic_h24n/create_synthetic_fixture.py and its exact expected properties must be implemented.
```

## Required Completion Evidence

At the end of this Jira, produce:

```text
reports/jira_002_contract_coverage.md
```

Use `CONTRACT_COVERAGE_TEMPLATE.md` as the required structure.

The report must prove:

```text
all strict/proxy/shadow feature dictionaries match the final contract;
every required feature formula is implemented or explicitly superseded;
every expert output is OOF, placeholder, or direct-shadow exactly as required;
fold-local preprocessing and model selection are proven;
all strict features are leakage-free and non-forward-looking;
no proxy/shadow/blocked/support-only source enters strict matrices.
```

## Done Condition

This Jira is complete only when feature matrices, feature dictionaries, online residual states, expert outputs, OOF reports, model-selection artifacts, and leakage tests are complete and mapped in `reports/jira_002_contract_coverage.md`.
