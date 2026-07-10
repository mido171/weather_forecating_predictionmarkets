# HKG T24 Jira Implementation Packets

> **Relocation note (2026-07-10):** The four packets intentionally retain their
> original path vocabulary as frozen implementation-contract provenance. In the
> current project, map `code/src/hkg_t24` to `src/hkg_t24`, `code/tests/hkg_t24`
> to `tests/hkg_t24`, `config/hkg_t24` to the governed subfolders under
> `config`, `sql/hkg_t24` to `db/sql/hkg_t24`, `schemas/hkg_t24` to
> `db/schemas/hkg_t24`, and `migrations/postgres` to `db/migrations/postgres`.
> Generated reports and artifacts resolve through `HKG_TMAX_RUN_ROOT`; the old
> path text in descendant packets must not be treated as current routing.

This folder contains the four dense implementation Jiras plus one self-contained packet folder per Jira.

Use the packet folders for implementation work:

```text
HKG-T24-001_data_contract_snapshot_feature_store_foundation
HKG-T24-002_feature_engineering_expert_model_factory
HKG-T24-003_router_specialists_final_forecast_distribution
HKG-T24-004_validation_sealed_live_inference_reporting
```

Each packet contains:

```text
<JIRA_FILE>.md
IMPLEMENTATION_PACKET.md
CONTRACT_COVERAGE_TEMPLATE.md
binding_contract_docs/
```

The four Jira markdown files now live inside their matching packet folders under their original filenames. The packet folders are the implementation-ready handoff units.

## Implementation Location

This repository uses `code/src` as its Python package root. Therefore, the strategy contract path `src/hkg_t24` resolves in this checkout to:

```text
code/src/hkg_t24/
```

All implementation code for the four Jiras must live under `code/src/hkg_t24/`.

Supporting files must use these dedicated locations:

```text
code/tests/hkg_t24/          tests
config/hkg_t24/              configuration
sql/hkg_t24/                 reviewed SQL/query assets
migrations/postgres/         durable PostgreSQL migrations
schemas/hkg_t24/             machine-readable schemas
reports/hkg_t24/             report indexes and non-canonical reports
artifacts/hkg_t24/           artifact indexes and small durable metadata
```

Do not put implementation logic in the Jira folders, root files, reports, notebooks, or ad hoc scripts. Scripts may call the package, but the package owns the implementation logic.

## Binding Rule

For every Jira packet, Codex must read the Jira markdown file in that packet and all files under `binding_contract_docs/` fully before implementing.

The Jira is not a replacement for the strategy contract. The Jira narrows the work package, while the copied contract docs remain binding.

Precedence:

```text
1. HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_FINAL_CONSISTENCY_PATCH.md
2. HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC_FINAL_CLARIFICATIONS.md
3. HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC.md
4. HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md
5. <JIRA_FILE>.md
```

If the Jira omits a required contract detail, the contract still applies. If the Jira conflicts with the contract, the final consistency patch wins. Codex must not silently resolve contradictions; it must fail closed and report the conflict.

## Completion Rule

Each Jira must produce its own coverage report:

```text
reports/jira_001_contract_coverage.md
reports/jira_002_contract_coverage.md
reports/jira_003_contract_coverage.md
reports/jira_004_contract_coverage.md
```

The report must prove that every relevant contract section has an implementation target, test, artifact, and leakage or temporal-integrity check.
