# HKG-T24-003 Implementation Packet

> **Relocation note (2026-07-10):** Preserve the historical path text below,
> but apply the current path mapping in [the Jira index](../README.md).

## Read-First Requirement

Before implementing this Jira, Codex must fully read:

```text
HKG-T24-003_Router_Specialists_Final_Forecast_Distribution.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC_FINAL_CLARIFICATIONS.md
binding_contract_docs/HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_FINAL_CONSISTENCY_PATCH.md
```

Reading only `HKG-T24-003_Router_Specialists_Final_Forecast_Distribution.md` is not enough.

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

Router, specialist, and distribution behavior must follow the exact final-patch fallback and demotion rules.

## Especially Binding Contract Sections

The full docs are binding, but these sections are especially relevant:

```text
Blueprint:
- 8. Router design
- 9. Specialist system
- 13. Final system forecast formula
- 14. Distributional and trading layer
- 15. Validation and scoreboards
- 19. Promotion ladder
- 20. Expected first system variants

Completion Spec:
- 15. Router specification
- 16. Specialists
- 17. Distributional layer
- 18. Metrics and scoreboards
- 21. Model promotion ladder
- 22. Final system formula
- 24. Required tests
- 25. Required final artifacts

Final Clarifications:
- 2. Specialist prior scores
- 4. Inner validation and model selection
- 9. Official residual model and fallback behavior
- 10. Router edge cases
- 11. Distributional layer edge cases
- 13. Test expectations

Final Consistency Patch:
- 8. Features referenced by specialists/router
- 17. Final handling of feature names in specialists and routers
- 18. Final rules for source absence, demotion, and fallback
- 19. Router command and artifact consistency
- 20. Distributional edge cases and threshold output keys
- 21. Sealed validation and shadow-source protocol
```

## High-Risk Details To Carry From The Contract

This Jira must explicitly preserve these details from the binding docs:

```text
Routers must train on OOF expert predictions only.
R1 must beat both E0_OFFICIAL_RAW_ANCHOR and R0_STATIC_BLEND on identical rows or be demoted.
If E4 or E5 demotes, R1 may still run only under the final-patch promoted-expert count rule.
If E0 is unavailable, strict fallback attempts E2 target-memory-only, then no strict forecast.
Static blend after demotion uses only promoted experts with nonzero caps.
Router demotion artifacts must use the exact reports/router_promotion_decisions.csv schema from the final patch.
Every specialist prior score must use the exact weighted formulas from the clarifications.
Specialist percentile ranks, missing-component rule, activation gates, no-harm gates, support shrinkage, and caps must match the clarifications.
S6 high-error-tail stricter no-harm rule must be implemented.
Distribution fallback must use empirical residual quantiles only from OOF residuals of the same frozen candidate.
Probability keys must be exactly prob_tmax_ge_20_0 through prob_tmax_ge_40_0 inclusive by 0.5, exactly 41 keys.
Gaussian fallback probability formula and clamps must match the final patch.
reports/calibration_report.md is a compatibility copy of reports/distribution_calibration_report.md and must contain the required compatibility header.
```

## Required Completion Evidence

At the end of this Jira, produce:

```text
reports/jira_003_contract_coverage.md
```

Use `CONTRACT_COVERAGE_TEMPLATE.md` as the required structure.

The report must prove:

```text
router inputs are OOF-only and cutoff-safe;
all router caps, masks, fallbacks, and demotion artifacts match the final patch;
every specialist formula and no-harm gate is implemented exactly;
the final system formula and distribution outputs are complete;
all probabilities, quantiles, confidence states, and no-trade flags are emitted;
strict/proxy/shadow outputs remain separated.
```

## Done Condition

This Jira is complete only when routers, specialists, final forecast, distributional outputs, reports, and all related leakage tests are complete and mapped in `reports/jira_003_contract_coverage.md`.
